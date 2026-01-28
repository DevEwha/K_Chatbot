"""
Split KV Caching for Progressive LLM Serving
progressive_serve/split_kv_cache.py

핵심 아이디어:
- KV Cache를 Base (불변)와 Delta (가변)로 분리
- Stage 전환 시 Base는 재사용, Delta만 재계산
- 레이어별 선택적 재사용으로 Prefill 시간 75-90% 감소
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Set
import torch


@dataclass
class SplitKVCache:
    """
    단일 레이어를 위한 Split KV Cache
    
    구조:
        K_final = K_base + K_delta
        V_final = V_base + V_delta
        
    - K_base, V_base: Base 모델의 KV (stage 전환 시에도 유지)
    - K_delta, V_delta: LoRA 어댑터의 delta (stage 변경 시 재계산)
    """
    
    # 영구 컴포넌트 (stage 전환 시에도 유지)
    k_base: torch.Tensor  # [batch, seq_len, num_heads, head_dim] or vLLM format
    v_base: torch.Tensor
    
    # 임시 컴포넌트 (stage 변경 시 재계산)
    k_delta: Optional[torch.Tensor] = None
    v_delta: Optional[torch.Tensor] = None
    
    # 메타데이터
    layer_idx: int = 0
    seq_len: int = 0
    
    # 입력 hidden states의 해시 (디버깅용)
    input_hash: Optional[int] = None
    
    def get_final_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Base와 Delta를 결합하여 최종 KV 반환"""
        if self.k_delta is None:
            return self.k_base, self.v_base
        return (self.k_base + self.k_delta, 
                self.v_base + self.v_delta)
    
    def update_delta(self, k_delta: torch.Tensor, v_delta: torch.Tensor):
        """Delta 업데이트 (어댑터 교체 시)"""
        self.k_delta = k_delta
        self.v_delta = v_delta
    
    def clear_delta(self):
        """Delta 제거"""
        self.k_delta = None
        self.v_delta = None
    
    def memory_usage_mb(self) -> Tuple[float, float]:
        """메모리 사용량 계산 (MB)"""
        base_mem = (self.k_base.numel() + self.v_base.numel()) * self.k_base.element_size() / 1024 / 1024
        delta_mem = 0.0
        if self.k_delta is not None:
            delta_mem = (self.k_delta.numel() + self.v_delta.numel()) * self.k_delta.element_size() / 1024 / 1024
        return base_mem, delta_mem
    
    def to(self, device: torch.device) -> 'SplitKVCache':
        """디바이스 이동"""
        self.k_base = self.k_base.to(device)
        self.v_base = self.v_base.to(device)
        if self.k_delta is not None:
            self.k_delta = self.k_delta.to(device)
            self.v_delta = self.v_delta.to(device)
        return self


class SplitCacheManager:
    """
    전체 레이어의 Split KV Cache 관리
    
    주요 기능:
    1. Cache 저장 및 조회
    2. Delta 무효화 (어댑터 교체 시)
    3. 레이어 범위 무효화 (입력 변경 시)
    4. Stage 전환 분석 및 자동 관리
    """
    
    def __init__(self, num_layers: int):
        """
        Args:
            num_layers: 모델의 총 레이어 수 (e.g., 32 for Llama-7B)
        """
        self.caches: Dict[int, SplitKVCache] = {}
        self.num_layers = num_layers
        
        # Stage 설정 (Progressive Serving 구조)
        # 실제 사용 시 prune_log.json에서 읽어옴
        self.stage_configs: Dict[int, Dict] = {}
        self.current_stage: int = 1
        
        # 통계
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'delta_recomputes': 0,
        }
    
    def set_stage_configs(
        self, 
        stage_configs: Dict[int, Dict],
        current_stage: int = 1
    ):
        """
        Stage 설정 저장
        
        Args:
            stage_configs: Stage별 레이어 구성
                예시:
                {
                    1: {'active_layers': [(0, 20), (29, 31)]},
                    2: {'active_layers': [(0, 20), (21, 24), (29, 31)]},
                    3: {'active_layers': [(0, 31)]},
                }
            current_stage: 현재 stage
        """
        self.stage_configs = stage_configs
        self.current_stage = current_stage
    
    # ============================================================
    # Cache 저장 및 조회
    # ============================================================
    
    def has_cache(self, layer_idx: int) -> bool:
        """해당 레이어의 cache 존재 여부"""
        return layer_idx in self.caches
    
    def get_cache(self, layer_idx: int) -> Optional[SplitKVCache]:
        """해당 레이어의 cache 반환"""
        return self.caches.get(layer_idx, None)
    
    def set_cache(
        self, 
        layer_idx: int, 
        k_base: torch.Tensor, 
        v_base: torch.Tensor,
        k_delta: Optional[torch.Tensor] = None,
        v_delta: Optional[torch.Tensor] = None,
        seq_len: int = 0,
        input_hash: Optional[int] = None,
    ):
        """
        Cache 저장
        
        Args:
            layer_idx: 레이어 인덱스
            k_base, v_base: Base KV (detach된 텐서)
            k_delta, v_delta: Delta KV (optional, detach된 텐서)
            seq_len: 시퀀스 길이
            input_hash: 입력 hidden states 해시 (디버깅용)
        """
        self.caches[layer_idx] = SplitKVCache(
            k_base=k_base.detach() if k_base.requires_grad else k_base,
            v_base=v_base.detach() if v_base.requires_grad else v_base,
            k_delta=k_delta.detach() if k_delta is not None and k_delta.requires_grad else k_delta,
            v_delta=v_delta.detach() if v_delta is not None and v_delta.requires_grad else v_delta,
            layer_idx=layer_idx,
            seq_len=seq_len,
            input_hash=input_hash,
        )
    
    def append_to_cache(
        self,
        layer_idx: int,
        k_base_new: torch.Tensor,
        v_base_new: torch.Tensor,
        k_delta_new: Optional[torch.Tensor] = None,
        v_delta_new: Optional[torch.Tensor] = None,
    ):
        """
        기존 cache에 새 토큰 추가 (decode phase)
        
        Args:
            layer_idx: 레이어 인덱스
            k_base_new, v_base_new: 새 토큰의 Base KV
            k_delta_new, v_delta_new: 새 토큰의 Delta KV (optional)
        """
        if layer_idx not in self.caches:
            raise ValueError(f"No cache for layer {layer_idx}")
        
        cache = self.caches[layer_idx]
        
        # Base 연결
        cache.k_base = torch.cat([cache.k_base, k_base_new.detach()], dim=1)
        cache.v_base = torch.cat([cache.v_base, v_base_new.detach()], dim=1)
        
        # Delta 연결 (있는 경우)
        if cache.k_delta is not None and k_delta_new is not None:
            cache.k_delta = torch.cat([cache.k_delta, k_delta_new.detach()], dim=1)
            cache.v_delta = torch.cat([cache.v_delta, v_delta_new.detach()], dim=1)
        
        cache.seq_len += k_base_new.size(1)
    
    # ============================================================
    # 무효화 메서드
    # ============================================================
    
    def invalidate_deltas(self):
        """
        모든 delta cache 제거 (어댑터 교체 시)
        
        이유: LoRA 어댑터가 바뀌면 Delta가 달라짐
        """
        print("🧹 Delta cache 무효화 중...")
        count = 0
        for cache in self.caches.values():
            if cache.k_delta is not None:
                cache.clear_delta()
                count += 1
        print(f"  ✅ {count}개 레이어의 delta cache 삭제")
        self.stats['delta_recomputes'] += count
    
    def invalidate_layer_range(self, start: int, end: int):
        """
        입력이 변경된 레이어의 cache 전체 제거
        
        Args:
            start: 시작 레이어 인덱스 (inclusive)
            end: 끝 레이어 인덱스 (inclusive)
            
        이유: 중간에 레이어가 추가되면 이후 레이어들의 입력이 변경됨
        """
        print(f"🧹 Layer {start}-{end} cache 무효화 중...")
        count = 0
        for layer_idx in range(start, end + 1):
            if layer_idx in self.caches:
                del self.caches[layer_idx]
                count += 1
        print(f"  ✅ {count}개 레이어의 cache 삭제")
    
    def invalidate_all(self):
        """모든 cache 제거"""
        print("🧹 전체 cache 무효화...")
        count = len(self.caches)
        self.caches.clear()
        print(f"  ✅ {count}개 레이어의 cache 삭제")
    
    # ============================================================
    # Stage 전환 분석
    # ============================================================
    
    def analyze_stage_transition(
        self, 
        current_stage: int, 
        next_stage: int
    ) -> Dict:
        """
        Stage 전환 시 레이어 변화 분석
        
        Args:
            current_stage: 현재 stage
            next_stage: 다음 stage
            
        Returns:
            {
                'reusable': [(start, end), ...],      # 재사용 가능 범위
                'new': [(start, end), ...],           # 새로 추가된 범위
                'invalidated': [(start, end), ...],   # 무효화 필요 범위
            }
        """
        if not self.stage_configs:
            print("⚠️  Stage configs not set. Using default analysis.")
            return self._analyze_default(current_stage, next_stage)
        
        current_config = self.stage_configs.get(current_stage, {})
        next_config = self.stage_configs.get(next_stage, {})
        
        current_ranges = current_config.get('active_layers', [])
        next_ranges = next_config.get('active_layers', [])
        
        return self._analyze_layer_changes(current_ranges, next_ranges)
    
    def _analyze_layer_changes(
        self, 
        current_ranges: List[Tuple[int, int]], 
        next_ranges: List[Tuple[int, int]]
    ) -> Dict:
        """
        레이어 변화 분석 (내부 메서드)
        
        핵심 로직:
        - 새 레이어가 추가되는 위치를 찾음
        - 그 위치 이전의 레이어들은 입력이 동일하므로 재사용 가능
        - 그 위치 이후의 레이어들은 입력이 변경되므로 무효화 필요
        
        예시 (Stage 1 → 2):
        - Stage 1: Layer 0-20, 29-31
        - Stage 2: Layer 0-20, 21-24, 29-31
        - 새 레이어 21-24가 Layer 20 뒤에 추가됨
        - Layer 0-20: 입력 동일 → 재사용 가능
        - Layer 29-31: 입력 변경 (hidden_20 → hidden_24) → 무효화
        """
        current_set = self._ranges_to_set(current_ranges)
        next_set = self._ranges_to_set(next_ranges)
        
        # 1. 새로 추가된 레이어 찾기
        new_layers = sorted(next_set - current_set)
        new = self._set_to_ranges(new_layers)
        
        # 2. 새 레이어가 추가되는 첫 번째 위치 찾기
        # 이 위치 이전의 레이어들은 재사용 가능
        if new_layers:
            first_new_layer = min(new_layers)
        else:
            # 새 레이어가 없으면 모든 기존 레이어 재사용 가능
            first_new_layer = float('inf')
        
        # 3. 재사용 가능한 레이어: 첫 번째 새 레이어 이전의 모든 기존 레이어
        reusable_layers = [
            l for l in current_set 
            if l < first_new_layer and l in next_set
        ]
        reusable = self._set_to_ranges(reusable_layers)
        
        # 4. 무효화 필요한 레이어: 첫 번째 새 레이어 이후의 기존 레이어
        # (다음 stage에도 존재하지만 입력이 변경됨)
        invalidated_layers = [
            l for l in current_set 
            if l >= first_new_layer and l in next_set
        ]
        invalidated = self._set_to_ranges(invalidated_layers)
        
        return {
            'reusable': reusable,
            'new': new,
            'invalidated': invalidated,
        }
    
    def _analyze_default(self, current_stage: int, next_stage: int) -> Dict:
        """
        기본 분석 (Progressive Serving 기준)
        
        Stage 1: Layer 0-20, 29-31
        Stage 2: Layer 0-20, 21-24, 29-31
        Stage 3: Layer 0-31
        """
        if current_stage == 1 and next_stage == 2:
            return {
                'reusable': [(0, 20)],
                'new': [(21, 24)],
                'invalidated': [(29, 31)],
            }
        elif current_stage == 2 and next_stage == 3:
            return {
                'reusable': [(0, 20), (21, 24)],
                'new': [(25, 28)],
                'invalidated': [(29, 31)],
            }
        else:
            return {
                'reusable': [],
                'new': [],
                'invalidated': [],
            }
    
    # ============================================================
    # Stage 전환 실행
    # ============================================================
    
    def handle_stage_transition(
        self, 
        current_stage: int, 
        next_stage: int,
        verbose: bool = True
    ) -> Dict:
        """
        Stage 전환 시 cache 관리
        
        Args:
            current_stage: 현재 stage
            next_stage: 다음 stage
            verbose: 상세 출력 여부
            
        Returns:
            분석 결과
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔄 Stage 전환: {current_stage} → {next_stage}")
            print(f"{'='*60}\n")
        
        # 1. 레이어 구조 분석
        analysis = self.analyze_stage_transition(current_stage, next_stage)
        
        if verbose:
            print(f"📋 레이어 구조 분석:")
            if analysis['reusable']:
                print(f"  ✅ 재사용 가능: {self._format_ranges(analysis['reusable'])}")
            if analysis['invalidated']:
                print(f"  ❌ 무효화 필요: {self._format_ranges(analysis['invalidated'])}")
            if analysis['new']:
                print(f"  🆕 새로 추가: {self._format_ranges(analysis['new'])}")
            print()
        
        # 2. Delta cache 무효화 (어댑터 교체)
        self.invalidate_deltas()
        
        # 3. 영향받은 레이어의 Base cache도 무효화 (입력 변경)
        for start, end in analysis['invalidated']:
            self.invalidate_layer_range(start, end)
        
        # 4. Stage 업데이트
        self.current_stage = next_stage
        
        if verbose:
            stats = self.get_memory_stats()
            print(f"\n💾 Cache 메모리 통계:")
            print(f"  캐시된 레이어 수: {stats['num_layers']}")
            print(f"  Base cache: {stats['base_MB']:.2f} MB (유지)")
            print(f"  Delta cache: {stats['delta_MB']:.2f} MB")
            print(f"  총 메모리: {stats['total_MB']:.2f} MB")
            print(f"\n✅ Stage {next_stage} 전환 준비 완료!")
            print(f"{'='*60}\n")
        
        return analysis
    
    # ============================================================
    # 유틸리티 메서드
    # ============================================================
    
    def _ranges_to_set(self, ranges: List[Tuple[int, int]]) -> Set[int]:
        """Range list를 set으로 변환"""
        result = set()
        for start, end in ranges:
            result.update(range(start, end + 1))
        return result
    
    def _set_to_ranges(self, layer_set: Set[int]) -> List[Tuple[int, int]]:
        """Set을 연속된 range list로 변환"""
        if not layer_set:
            return []
        
        sorted_layers = sorted(layer_set)
        ranges = []
        start = sorted_layers[0]
        prev = start
        
        for curr in sorted_layers[1:]:
            if curr != prev + 1:
                ranges.append((start, prev))
                start = curr
            prev = curr
        
        ranges.append((start, prev))
        return ranges
    
    def _format_ranges(self, ranges: List[Tuple[int, int]]) -> str:
        """Range를 읽기 쉬운 문자열로 변환"""
        if not ranges:
            return "없음"
        parts = []
        for start, end in ranges:
            if start == end:
                parts.append(f"Layer {start}")
            else:
                parts.append(f"Layer {start}-{end}")
        return ", ".join(parts)
    
    # ============================================================
    # 통계 및 디버깅
    # ============================================================
    
    def get_memory_stats(self) -> Dict:
        """메모리 사용 통계"""
        total_base = 0.0
        total_delta = 0.0
        
        for cache in self.caches.values():
            base, delta = cache.memory_usage_mb()
            total_base += base
            total_delta += delta
        
        return {
            'base_MB': total_base,
            'delta_MB': total_delta,
            'total_MB': total_base + total_delta,
            'overhead_pct': (total_delta / total_base * 100) if total_base > 0 else 0,
            'num_layers': len(self.caches),
        }
    
    def print_status(self):
        """현재 cache 상태 출력"""
        print(f"\n{'='*60}")
        print(f"Split Cache Manager Status - Stage {self.current_stage}")
        print(f"{'='*60}")
        
        if not self.caches:
            print("  (No cached layers)")
        else:
            # Layer 그룹별로 표시
            groups = {
                '0-20': range(0, 21),
                '21-24': range(21, 25),
                '25-28': range(25, 29),
                '29-31': range(29, 32)
            }
            
            for group_name, layer_range in groups.items():
                cached_in_group = [l for l in layer_range if l in self.caches]
                if cached_in_group:
                    print(f"\nLayer {group_name}:")
                    for layer_idx in cached_in_group:
                        cache = self.caches[layer_idx]
                        base_mb, delta_mb = cache.memory_usage_mb()
                        
                        status = "✅ Base+Delta" if cache.k_delta is not None else "⚠️  Base only"
                        print(f"  L{layer_idx:2d}: {status} | Base: {base_mb:.1f}MB | Delta: {delta_mb:.1f}MB | Seq: {cache.seq_len}")
        
        stats = self.get_memory_stats()
        print(f"\n총 메모리: {stats['total_MB']:.1f}MB across {stats['num_layers']} layers")
        print(f"통계: hits={self.stats['cache_hits']}, misses={self.stats['cache_misses']}, delta_recomputes={self.stats['delta_recomputes']}")
        print(f"{'='*60}\n")
    
    def get_cached_layers(self) -> List[int]:
        """캐시된 레이어 인덱스 목록"""
        return sorted(self.caches.keys())
    
    def is_layer_reusable(self, layer_idx: int, next_stage: int) -> bool:
        """
        특정 레이어가 다음 stage에서 재사용 가능한지 확인
        
        Args:
            layer_idx: 레이어 인덱스
            next_stage: 다음 stage
            
        Returns:
            재사용 가능 여부
        """
        if layer_idx not in self.caches:
            return False
        
        analysis = self.analyze_stage_transition(self.current_stage, next_stage)
        
        for start, end in analysis['reusable']:
            if start <= layer_idx <= end:
                return True
        
        return False


# ============================================================
# 테스트 코드
# ============================================================

if __name__ == "__main__":
    print("Split KV Cache Manager Test")
    print("="*60)
    
    # Manager 생성
    manager = SplitCacheManager(num_layers=32)
    
    # Stage 설정 (Progressive Serving 구조)
    stage_configs = {
        1: {'active_layers': [(0, 20), (29, 31)]},
        2: {'active_layers': [(0, 20), (21, 24), (29, 31)]},
        3: {'active_layers': [(0, 31)]},
    }
    manager.set_stage_configs(stage_configs, current_stage=1)
    
    # 더미 cache 생성
    print("\n1. 더미 cache 생성 (Stage 1)")
    for layer_idx in [0, 5, 10, 15, 20, 29, 30, 31]:
        k_base = torch.randn(1, 100, 32, 128)  # [batch, seq, heads, dim]
        v_base = torch.randn(1, 100, 32, 128)
        manager.set_cache(layer_idx, k_base, v_base, seq_len=100)
    
    manager.print_status()
    
    # Stage 1 → 2 전환 분석
    print("\n2. Stage 1 → 2 전환")
    analysis = manager.handle_stage_transition(1, 2)
    
    manager.print_status()
    
    # Stage 2 → 3 전환 분석
    print("\n3. Stage 2 → 3 전환")
    analysis = manager.handle_stage_transition(2, 3)
    
    manager.print_status()
    
    print("\n✅ 테스트 완료!")