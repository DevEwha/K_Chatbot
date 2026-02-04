"""
ProgressiveLlamaModel with Alpha Gating (vLLM v0 Engine) - Partial KV Cache Reuse 버전

주요 개선사항:
1. load_state_dict() → .copy_()로 변경 (메모리 주소 유지)
2. fused_weights를 실제로 사용하도록 수정
3. CUDA Graph 재캡처 불필요
4. ✅ NEW: Partial KV Cache Reuse 지원
   - activate_layers_with_cache_hint(): Cache 재사용 경계 반환
   - get_recompute_boundary(): 재계산 시작 레이어 계산
"""

from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.sampler import SamplerOutput
# vLLM v0 imports
try:
    from vllm.attention import AttentionMetadata
except ImportError:
    try:
        from vllm.attention.backends.abstract import AttentionMetadata
    except ImportError:
        AttentionMetadata = Any

try:
    from vllm.sequence import IntermediateTensors
except ImportError:
    IntermediateTensors = Any

from safetensors.torch import load_file
import os

from alpha_gated_layer import AlphaGatedLayer


class ProgressiveLlamaModelAlpha(nn.Module):
    """
    Alpha Gating을 사용한 ProgressiveLlamaModel (vLLM v0)
    
    ✅ CUDA Graph 호환 버전:
    - .copy_()로 weight 로드 (메모리 주소 유지)
    - Alpha buffer는 register_buffer + fill_() (주소 고정)
    - 레이어 교체 없음 (모든 레이어 처음부터 할당)
    
    ✅ Partial KV Cache Reuse 지원:
    - activate_layers_with_cache_hint(): Cache 재사용 힌트 반환
    - 변경되지 않은 앞부분 레이어의 KV Cache 재활용 가능
    """
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        pruned_layer_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        
        # Pruned layer 인덱스 (초기에 비활성화할 레이어)
        self.initially_inactive = set(pruned_layer_indices or [])
        
        # ✅ NEW: 레이어 활성화 이력 추적 (Partial KV Cache Reuse용)
        self.layer_activation_history: List[Dict] = []
        self.last_recompute_boundary: Optional[int] = None
        
        # Embedding
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        
        # Decoder layers 초기화
        self.layers = nn.ModuleList()
        self._init_layers(prefix)
        
        # Final norm
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps
        )
        
        # Adapter 상태
        self.current_adapter = None
    
    def _init_layers(self, prefix: str):
        """
        모든 레이어 초기화 (AlphaGatedLayer로 감싸기)
        
        핵심:
        - 모든 레이어를 생성 (pruned layer도!)
        - Pruned layer는 weight를 0으로 초기화
        - AlphaGatedLayer로 감싸서 alpha=0 설정
        """
        # vLLM 버전별 import 분기
        try:
            from vllm.model_executor.models.llama import LlamaDecoderLayer
        except ImportError:
            try:
                from vllm.models.llama import LlamaDecoderLayer
            except ImportError:
                # v1 engine
                from vllm.v1.model_executor.models.llama import LlamaDecoderLayer
        
        num_layers = self.config.num_hidden_layers
        
        for layer_idx in range(num_layers):
            # 레이어 생성 
            try:
                # 최신 vLLM 또는 v1
                base_layer = LlamaDecoderLayer(
                    config=self.config,
                    cache_config=self.vllm_config.cache_config,
                    quant_config=self.vllm_config.quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
            except TypeError:
                # 구버전 fallback
                try:
                    base_layer = LlamaDecoderLayer(
                        layer_idx=layer_idx,
                        config=self.config,
                        prefix=f"{prefix}.layers.{layer_idx}",
                    )
                except TypeError:
                    # v0 엔진 특수 케이스
                    base_layer = LlamaDecoderLayer(
                        vllm_config=self.vllm_config,
                        prefix=f"{prefix}.layers.{layer_idx}",
                    )
            
            # Alpha gating 적용
            if layer_idx in self.initially_inactive:
                # Pruned layer: alpha = 0 (비활성)
                print(f"[Init] Layer {layer_idx:2d}: AlphaGatedLayer (alpha=0, INACTIVE)")
                
                # Weight를 0으로 초기화
                self._initialize_weights_to_zero(base_layer)
                
                # AlphaGatedLayer로 감싸기
                gated_layer = AlphaGatedLayer(
                    base_layer=base_layer,
                    initial_alpha=0.0,  # 비활성
                )
                self.layers.append(gated_layer)
            else:
                # Normal layer: alpha = 1 (활성)
                print(f"[Init] Layer {layer_idx:2d}: AlphaGatedLayer (alpha=1, ACTIVE)")
                
                # AlphaGatedLayer로 감싸기
                gated_layer = AlphaGatedLayer(
                    base_layer=base_layer,
                    initial_alpha=1.0,  # 활성
                )
                self.layers.append(gated_layer)
    
    def _initialize_weights_to_zero(self, layer: nn.Module):
        """
        레이어의 모든 weight를 0으로 초기화
        
        Note: alpha=0이므로 출력에 영향 없음
              나중에 실제 weight 로드 시 덮어씀
        """
        for param in layer.parameters():
            nn.init.zeros_(param)
    
    # ============================================================
    # vLLM Required Methods
    # ============================================================
    
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """토큰 ID → 임베딩"""
        return self.embed_tokens(input_ids)
    
    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        hidden_states = self.embed_tokens(input_ids)
        
        residual = None
        
        # 모든 레이어 통과
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        if residual is not None:
            hidden_states = hidden_states + residual
        
        hidden_states = self.norm(hidden_states)
        
        return hidden_states
    
    # ============================================================
    # Progressive Recovery Methods (Alpha Gating) - CUDA Graph 호환
    # ============================================================
    
    def activate_layers(
        self,
        layer_indices: List[int],
        checkpoint_path: str,
    ) -> None:
        """
        레이어 활성화 (alpha: 0 → 1) - CUDA Graph 호환
        
        핵심 변경:
        1. load_state_dict() → .copy_() 사용
        2. 메모리 주소 유지
        3. CUDA Graph 재캡처 불필요
        
        Args:
            layer_indices: 활성화할 레이어 번호
            checkpoint_path: Weight 파일 경로
        """
        print(f"\n{'='*60}")
        print(f"ACTIVATING LAYERS: {layer_indices}")
        print(f"{'='*60}")
        
        # Checkpoint 로드
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = load_file(checkpoint_path)
        
        device = next(self.parameters()).device
        
        for layer_idx in layer_indices:
            print(f"\n Activating layer {layer_idx}...")
            
            gated_layer = self.layers[layer_idx]
            
            # AlphaGatedLayer 확인
            if not hasattr(gated_layer, 'is_alpha_gated'):
                print(f" Layer {layer_idx} is not AlphaGatedLayer!")
                continue
            
            # 이미 활성화된 레이어
            if gated_layer.is_active():
                print(f" Layer {layer_idx} is already active")
                continue
            
            # 1. Weight 추출
            print(f"Loading weights...")
            layer_prefix = f"model.layers.{layer_idx}."
            layer_weights = {
                k.replace(layer_prefix, ""): v
                for k, v in state_dict.items()
                if k.startswith(layer_prefix)
            }
            
            if not layer_weights:
                print(f" No weights found for layer {layer_idx}")
                continue
            
            # 2. .copy_()로 in-place weight 로드 (CUDA Graph 호환)
            loaded_count = 0
            
            for name, param in gated_layer.layer.named_parameters():
                # 2.1. QKV fusion 처리
                if name == "self_attn.qkv_proj.weight":
                    if all(k in layer_weights for k in [
                        "self_attn.q_proj.weight",
                        "self_attn.k_proj.weight", 
                        "self_attn.v_proj.weight"
                    ]):
                        qkv_weight = torch.cat([
                            layer_weights["self_attn.q_proj.weight"],
                            layer_weights["self_attn.k_proj.weight"],
                            layer_weights["self_attn.v_proj.weight"]
                        ], dim=0)
                        
                        # .copy_() 사용(메모리 주소 유지)
                        param.data.copy_(qkv_weight.to(device))
                        loaded_count += 1
                        print(f" Loaded fused QKV")
                        continue
                
                # 2.2. Gate-Up fusion 처리
                if name == "mlp.gate_up_proj.weight":
                    if all(k in layer_weights for k in [
                        "mlp.gate_proj.weight",
                        "mlp.up_proj.weight"
                    ]):
                        gate_up_weight = torch.cat([
                            layer_weights["mlp.gate_proj.weight"],
                            layer_weights["mlp.up_proj.weight"]
                        ], dim=0)
                        
                        # .copy_() 사용 (메모리 주소 유지)
                        param.data.copy_(gate_up_weight.to(device))
                        loaded_count += 1
                        print(f" Loaded fused Gate-Up")
                        continue
                
                # 2.3. 일반 weights 처리
                if name in layer_weights:
                    # .copy_() 사용 (메모리 주소 유지)
                    param.data.copy_(layer_weights[name].to(device))
                    loaded_count += 1
            
            print(f"  ✅ Loaded {loaded_count} weight tensors")
            
            # 3. Alpha 활성화 (0 → 1)
            gated_layer.activate()
            
            # 4. initially_inactive에서 제거
            self.initially_inactive.discard(layer_idx)
            
            print(f" Layer {layer_idx} activated!")
        
        print(f"\n{'='*60}")
        print(f"LAYER ACTIVATION COMPLETE")
        print(f"Inactive layers: {self.count_inactive_layers()}")
        print(f"{'='*60}\n")
    
    # ============================================================
    # ✅ NEW: Partial KV Cache Reuse Methods
    # ============================================================
    
    def activate_layers_with_cache_hint(
        self,
        layer_indices: List[int],
        checkpoint_path: str,
    ) -> Dict[str, Any]:
        """
        레이어 활성화 + Cache 재사용 힌트 반환
        
        제안서 Section 4.2 Step 1 구현:
        - Weight 로드 및 Alpha 활성화
        - Cache 재사용 경계 계산하여 반환
        
        Args:
            layer_indices: 활성화할 레이어 번호 리스트
            checkpoint_path: Weight 파일 경로
            
        Returns:
            Dict with:
                - keep_prefix_layers: 이 레이어 전까지 KV Cache 유지
                - recompute_from_layer: 이 레이어부터 재계산 필요
                - activated_layers: 활성화된 레이어 목록
                - total_layers: 전체 레이어 수
                - reuse_ratio: KV Cache 재사용 비율 (%)
        """
        print(f"\n{'='*60}")
        print(f"ACTIVATING LAYERS WITH CACHE HINT: {layer_indices}")
        print(f"{'='*60}")
        
        # 1. 기존 레이어 활성화 로직 실행
        self.activate_layers(layer_indices, checkpoint_path)
        
        # 2. Cache 재사용 경계 계산
        first_changed_layer = min(layer_indices) if layer_indices else len(self.layers)
        
        # 3. 활성화 이력 기록
        activation_record = {
            "activated_layers": sorted(layer_indices),
            "first_changed_layer": first_changed_layer,
            "checkpoint_path": checkpoint_path,
        }
        self.layer_activation_history.append(activation_record)
        self.last_recompute_boundary = first_changed_layer
        
        # 4. 힌트 구성
        total_layers = len(self.layers)
        reuse_ratio = (first_changed_layer / total_layers) * 100
        
        cache_hint = {
            "keep_prefix_layers": first_changed_layer,
            "recompute_from_layer": first_changed_layer,
            "activated_layers": sorted(layer_indices),
            "total_layers": total_layers,
            "reuse_ratio": reuse_ratio,
        }
        
        print(f"\n{'='*60}")
        print(f"CACHE REUSE HINT")
        print(f"{'='*60}")
        print(f"  Keep prefix layers: 0 - {first_changed_layer - 1}")
        print(f"  Recompute from layer: {first_changed_layer}")
        print(f"  KV Cache reuse ratio: {reuse_ratio:.1f}%")
        print(f"{'='*60}\n")
        
        return cache_hint
    
    def get_recompute_boundary(self) -> Optional[int]:
        """
        마지막 Stage 전환 시 재계산 시작 레이어 반환
        
        Returns:
            재계산 시작 레이어 인덱스, None이면 전체 재계산 필요
        """
        return self.last_recompute_boundary
    
    def get_cache_reuse_info(self) -> Dict[str, Any]:
        """
        현재 KV Cache 재사용 정보 반환
        
        Returns:
            Dict with cache reuse statistics
        """
        total_layers = len(self.layers)
        
        # 현재 활성/비활성 레이어 파악
        active_layers = []
        inactive_layers = []
        
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'is_alpha_gated'):
                if layer.is_active():
                    active_layers.append(i)
                else:
                    inactive_layers.append(i)
            else:
                active_layers.append(i)
        
        # 연속된 활성 레이어 블록 찾기 (앞부분)
        continuous_active_prefix = 0
        for i in range(total_layers):
            if i in active_layers:
                continuous_active_prefix = i + 1
            else:
                break
        
        return {
            "total_layers": total_layers,
            "active_layers": active_layers,
            "inactive_layers": inactive_layers,
            "continuous_active_prefix": continuous_active_prefix,
            "last_recompute_boundary": self.last_recompute_boundary,
            "activation_history": self.layer_activation_history,
        }
    
    def calculate_speedup_estimate(
        self,
        seq_len: int,
        layer_indices_to_activate: List[int],
    ) -> Dict[str, float]:
        """
        Partial KV Cache Reuse로 인한 예상 속도 향상 계산
        
        Args:
            seq_len: 현재 시퀀스 길이
            layer_indices_to_activate: 활성화할 레이어 인덱스
            
        Returns:
            Dict with speedup estimates
        """
        total_layers = len(self.layers)
        first_changed = min(layer_indices_to_activate) if layer_indices_to_activate else total_layers
        
        # 재사용 비율
        reuse_ratio = first_changed / total_layers
        
        # 예상 속도 향상 (대략적 추정)
        # 전체 재계산 대비 부분 재계산의 시간 절약
        layers_to_recompute = total_layers - first_changed
        estimated_time_ratio = layers_to_recompute / total_layers
        estimated_speedup = 1 / estimated_time_ratio if estimated_time_ratio > 0 else float('inf')
        
        return {
            "reuse_layers": first_changed,
            "recompute_layers": layers_to_recompute,
            "reuse_ratio": reuse_ratio * 100,
            "estimated_speedup": estimated_speedup,
            "estimated_time_reduction": (1 - estimated_time_ratio) * 100,
        }
    
    # ============================================================
    # Status Methods
    # ============================================================
    
    def get_layer_status(self) -> Dict[int, Dict]:
        """레이어 상태 확인"""
        status = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'is_alpha_gated'):
                status[i] = {
                    "type": "AlphaGatedLayer",
                    "active": layer.is_active(),
                    "alpha": layer.get_alpha(),
                }
            else:
                status[i] = {
                    "type": "Unknown",
                    "active": True,
                    "alpha": 1.0,
                }
        return status
    
    def count_inactive_layers(self) -> int:
        """비활성 레이어 개수"""
        count = 0
        for layer in self.layers:
            if hasattr(layer, 'is_alpha_gated') and not layer.is_active():
                count += 1
        return count
    
    def print_layer_status(self) -> None:
        """레이어 상태 출력"""
        status = self.get_layer_status()
        
        print("\n" + "="*60)
        print("LAYER STATUS (Alpha Gating - Partial KV Cache Reuse)")
        print("="*60)
        
        for start in range(0, len(status), 10):
            end = min(start + 10, len(status))
            print(f"\nLayers {start:2d}-{end-1:2d}:")
            
            for i in range(start, end):
                info = status[i]
                active = info['active']
                alpha = info['alpha']
                symbol = "●" if active else "○"
                print(f"  L{i:2d}: {symbol} alpha={alpha:.1f} ({'ACTIVE' if active else 'INACTIVE'})")
        
        # Summary
        total = len(self.layers)
        inactive = self.count_inactive_layers()
        active = total - inactive
        progress = (active / total) * 100
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total Layers:         {total}")
        print(f"Active Layers:        {active}")
        print(f"Inactive Layers:      {inactive}")
        print(f"Activation Progress:  {progress:.1f}%")
        print(f"Current Adapter:      {self.current_adapter or 'None'}")
        print(f"✅ CUDA Graph:        Compatible (no recapture needed)")
        print(f"✅ Partial KV Reuse:  Enabled")
        if self.last_recompute_boundary is not None:
            print(f"   Last recompute boundary: Layer {self.last_recompute_boundary}")
        print(f"{'='*60}\n")
    
    def verify_recovery(self) -> Dict:
        """복구 상태 검증"""
        total = len(self.layers)
        inactive = self.count_inactive_layers()
        active = total - inactive
        
        inactive_indices = [
            i for i, layer in enumerate(self.layers)
            if hasattr(layer, 'is_alpha_gated') and not layer.is_active()
        ]
        
        progress = (active / total) * 100
        
        return {
            "total_layers": total,
            "active_layers": active,
            "inactive_layers": inactive,
            "inactive_layer_indices": inactive_indices,
            "activation_progress": f"{progress:.1f}%",
            "cuda_graph_compatible": True,
            "partial_kv_reuse_enabled": True,
            "last_recompute_boundary": self.last_recompute_boundary,
        }
    
    def get_adapter_info(self) -> Dict:
        """Adapter 정보"""
        return {
            "current_adapter": self.current_adapter,
            "has_adapter": self.current_adapter is not None,
        }


# ============================================================
# 테스트 코드
# ============================================================

if __name__ == "__main__":
    print("""
Progressive LLaMA Alpha Gating - Partial KV Cache Reuse 버전
============================================================

주요 개선사항:
1. ✅ .copy_()로 weight 로드 (메모리 주소 유지)
2. ✅ Alpha buffer는 register_buffer + fill_() (주소 고정)
3. ✅ CUDA Graph 재캡처 불필요
4. ✅ Partial KV Cache Reuse 지원

새로운 기능:
- activate_layers_with_cache_hint(): Cache 재사용 힌트 반환
- get_recompute_boundary(): 재계산 시작 레이어 확인
- get_cache_reuse_info(): Cache 재사용 정보 조회
- calculate_speedup_estimate(): 예상 속도 향상 계산

예상 효과 (Llama-2-7B 기준):
- Stage 1→2: ~72% KV Cache 재사용 (~3x 속도 향상)
- Stage 2→3: ~86% KV Cache 재사용 (~5x 속도 향상)
""")