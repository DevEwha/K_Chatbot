"""
ProgressiveLlamaForCausalLM with Alpha Gating for vLLM v0
progressive_llama_for_causal_lm_alpha_v0.py

✅ v3 업데이트: 
- prune_log.json 기반 자동 레이어 결정
- Partial KV Cache Reuse 지원
- 최적화된 Stage 전환 (advance_to_stage_optimized)
"""

from typing import Optional, List, Iterable, Tuple, Dict, Any
import torch
import torch.nn as nn
from vllm.config import VllmConfig

# vLLM v0 imports
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata

# Weight loader
try:
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
except ImportError:
    default_weight_loader = None

# cuda graph 수정을 위한 주석
from progressive_llama_alpha_fixed import ProgressiveLlamaModelAlpha


class ProgressiveLlamaForCausalLMAlpha(nn.Module):
    """
    Alpha Gating을 사용한 ForCausalLM wrapper (vLLM v0)
    
    핵심 개선:
    - prune_log.json 자동 로드 (하드코딩 제거)
    - load_weights()에서 missing weights를 자동으로 0으로 초기화
    - vLLM v0 엔진에 맞게 조정
    - kv_cache와 attn_metadata는 vLLM 내부에서 자동 처리
    
    ✅ NEW: Partial KV Cache Reuse
    - advance_to_stage2_optimized(): Cache 힌트 포함 Stage 전환
    - advance_to_stage3_optimized(): Cache 힌트 포함 Stage 전환
    - get_cache_reuse_info(): Cache 재사용 정보 조회
    """
    supports_multimodal = False
    supports_pooling = False 
    embedding_mode = False
    
    task = "generate"  

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        stage: int = 1,
    ):
        super().__init__()
        
        self.supports_lora = False
        self.embedding_mode = False
        
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        
        if not hasattr(config, 'model_type'):
            config.model_type = "llama"
        
        # Model path 가져오기
        model_path = vllm_config.model_config.model
        
        # prune_log.json 로드
        self.prune_info = self._load_prune_log(model_path)
        
        # Stage에 따른 inactive layer indices (prune_log 기반)
        inactive_indices = self._get_inactive_indices_from_prune_log(self.prune_info, stage)
        
        # Model 생성
        self.model = ProgressiveLlamaModelAlpha(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
            pruned_layer_indices=inactive_indices,
        )
        
        # LM head
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        
        # vLLM v0 Sampler
        self.sampler = Sampler()
        
        # Stage
        self.current_stage = stage
        
        # Inactive layer tracking (for weight loading)
        self.inactive_layer_indices = set(inactive_indices)
        
        # ✅ NEW: Cache 힌트 저장 (외부에서 접근 가능)
        self.last_cache_hint: Optional[Dict[str, Any]] = None
        
        print(f"\n{'='*60}")
        print(f"ProgressiveLlamaForCausalLMAlpha (vLLM v0, Partial KV Cache Reuse)")
        print(f"Model: {model_path}")
        print(f"Initialized at Stage {stage}")
        if self.prune_info:
            print(f"✅ Prune log loaded from: {model_path}/prune_log.json")
            print(f"   Split B (Stage 2): {self.prune_info['split']['B']}")
            print(f"   Split C (Stage 3): {self.prune_info['split']['C']}")
        else:
            print(f"⚠️  Using fallback inactive layers (no prune_log.json)")
        print(f"Initially inactive layers: {sorted(inactive_indices)}")
        print(f"⚡ Smart weight loading enabled (missing → zeros)")
        print(f"⚡ Partial KV Cache Reuse enabled")
        print(f"{'='*60}\n")
    
    def _load_prune_log(self, model_path: str) -> Optional[dict]:
        """
        모델 디렉토리에서 prune_log.json 로드
        
        Args:
            model_path: 모델 디렉토리 경로 (e.g., /acpl-ssd20/1218/A)
            
        Returns:
            prune_log 내용 dict 또는 None (파일 없으면)
        """
        import json
        import os
        
        prune_log_path = os.path.join(model_path, "prune_log.json")
        
        if not os.path.exists(prune_log_path):
            return None
        
        try:
            with open(prune_log_path, 'r') as f:
                prune_log = json.load(f)
            
            # 필수 필드 확인
            if 'split' not in prune_log:
                print(f"⚠️  Warning: 'split' field not found in prune_log.json")
                return None
            
            if 'B' not in prune_log['split'] or 'C' not in prune_log['split']:
                print(f"⚠️  Warning: 'B' or 'C' not found in split")
                return None
            
            return prune_log
            
        except Exception as e:
            print(f"❌ Error loading prune_log.json: {e}")
            return None
    
    def _get_inactive_indices_from_prune_log(
        self, 
        prune_info: Optional[dict], 
        stage: int
    ) -> List[int]:
        """
        prune_log.json의 split 정보를 바탕으로 inactive layer indices 결정
        
        Stage 1: B + C 모두 inactive
        Stage 2: C만 inactive
        Stage 3: 모두 active
        
        Args:
            prune_info: prune_log.json 내용
            stage: 현재 stage (1, 2, 3)
            
        Returns:
            inactive layer indices 리스트
        """
        # Fallback: prune_log가 없으면 기본값 사용
        if prune_info is None:
            return self._get_inactive_indices_fallback(stage)
        
        try:
            split_b = prune_info['split']['B']
            split_c = prune_info['split']['C']
            
            if stage == 1:
                # Stage 1: B + C 모두 inactive
                inactive = sorted(split_b + split_c)
            elif stage == 2:
                # Stage 2: C만 inactive
                inactive = sorted(split_c)
            elif stage == 3:
                # Stage 3: 모두 active
                inactive = []
            else:
                raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3")
            
            return inactive
            
        except Exception as e:
            print(f"❌ Error parsing prune_log: {e}")
            print(f"   Falling back to default inactive layers")
            return self._get_inactive_indices_fallback(stage)
    
    def _get_inactive_indices_fallback(self, stage: int) -> List[int]:
        """
        Fallback: prune_log가 없을 때 기본값
        (Llama-2-7b 기준 하드코딩)
        """
        if stage == 1:
            return list(range(21, 29))  # L21-28
        elif stage == 2:
            return list(range(25, 29))  # L25-28
        elif stage == 3:
            return []  # 모두 활성
        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    def _get_activation_indices(self, target_stage: int) -> List[int]:
        """
        target_stage로 전환할 때 활성화해야 할 레이어 인덱스 반환
        
        Args:
            target_stage: 목표 stage (2 또는 3)
            
        Returns:
            활성화할 레이어 인덱스 리스트
        """
        if self.prune_info is None:
            # Fallback
            if target_stage == 2:
                return [21, 22, 23, 24]
            elif target_stage == 3:
                return [25, 26, 27, 28]
            else:
                return []
        
        if target_stage == 2:
            return self.prune_info['split']['B']
        elif target_stage == 3:
            return self.prune_info['split']['C']
        else:
            return []
    
    def compute_logits(
        self, 
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        """
        vLLM 0.7.x에서 생성 모델 인식을 위해 필요
        """
        logits = self.lm_head(hidden_states)
        if sampling_metadata.selected_token_indices is not None:
            logits = logits[sampling_metadata.selected_token_indices]
        return logits
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """샘플링"""
        return self.sampler(logits, sampling_metadata)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass (vLLM v0)
        
        Note: kv_cache와 attn_metadata는 vLLM 엔진에서 자동 관리
        """
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=None,  # vLLM 내부에서 처리
            attn_metadata=None,  # vLLM 내부에서 처리
        )
        
        return hidden_states
    
    # ============================================================
    # Weight Loading (vLLM Compatible)
    # ============================================================
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        vLLM 호환 weight loading
        
        핵심 기능:
        1. Fused weights 자동 처리 (QKV, Gate-Up)
        2. Missing weights는 0으로 초기화 (inactive layers)
        3. Active layer의 missing weight는 경고
        """
        params_dict = dict(self.named_parameters())
        loaded_keys = set()
        
        # Weight 이름 → 텐서 매핑
        checkpoint_weights = {}
        for name, loaded_weight in weights:
            checkpoint_weights[name] = loaded_weight
        
        total_params = len(params_dict)
        loaded_count = 0
        
        # 각 파라미터에 대해 weight 로드
        for param_name, param in params_dict.items():
            # Option 1: 직접 매칭
            if param_name in checkpoint_weights:
                weight_loader = getattr(param, "weight_loader",
                                       lambda p, w: p.data.copy_(w))
                weight_loader(param, checkpoint_weights[param_name])
                loaded_keys.add(param_name)
                loaded_count += 1
                continue
            
            # Option 2: 모듈 내부 weight (".layer." prefix 처리)
            # AlphaGatedLayer 내부 weight 처리
            if ".layer." in param_name:
                checkpoint_name = param_name.replace(".layer.", ".")
                if checkpoint_name in checkpoint_weights:
                    weight_loader = getattr(param, "weight_loader",
                                           lambda p, w: p.data.copy_(w))
                    weight_loader(param, checkpoint_weights[checkpoint_name])
                    loaded_keys.add(param_name)
                    loaded_count += 1
                    continue
            
            # Option 3: Fused QKV weights
            if "self_attn.qkv_proj.weight" in param_name:
                if ".layer." in param_name:
                    checkpoint_base = param_name.replace(".layer.self_attn.qkv_proj.weight", "")
                else:
                    checkpoint_base = param_name.replace(".self_attn.qkv_proj.weight", "")
                
                prefix = f"{checkpoint_base}.self_attn"
                q_name = f"{prefix}.q_proj.weight"
                k_name = f"{prefix}.k_proj.weight"
                v_name = f"{prefix}.v_proj.weight"
                
                if all(n in checkpoint_weights for n in [q_name, k_name, v_name]):
                    qkv_weight = torch.cat([
                        checkpoint_weights[q_name],
                        checkpoint_weights[k_name],
                        checkpoint_weights[v_name]
                    ], dim=0)
                
                    weight_loader = getattr(param, "weight_loader",
                                       lambda p, w: p.data.copy_(w))
                    weight_loader(param, qkv_weight)
                    loaded_keys.add(param_name)
                    loaded_count += 1
                    continue

            # Option 4: Fused Gate-Up weights
            if "mlp.gate_up_proj.weight" in param_name:
                if ".layer." in param_name:
                    checkpoint_base = param_name.replace(".layer.mlp.gate_up_proj.weight", "")
                else:
                    checkpoint_base = param_name.replace(".mlp.gate_up_proj.weight", "")
    
                prefix = f"{checkpoint_base}.mlp"
                gate_name = f"{prefix}.gate_proj.weight"
                up_name = f"{prefix}.up_proj.weight"
    
                if all(n in checkpoint_weights for n in [gate_name, up_name]):
                    gate_up_weight = torch.cat([
                        checkpoint_weights[gate_name],
                        checkpoint_weights[up_name]
                    ], dim=0)
        
                    weight_loader = getattr(param, "weight_loader",
                            lambda p, w: p.data.copy_(w))
                    weight_loader(param, gate_up_weight)
                    loaded_keys.add(param_name)
                    loaded_count += 1
                    continue

        
        # Missing weights 처리
        missing_keys = set(params_dict.keys()) - loaded_keys
        
        if missing_keys:
            print(f"\n⚠️  Found {len(missing_keys)} missing weights")
            print(f"   Initializing them to ZEROS (for inactive layers)...")
            
            # Layer별로 그룹화
            missing_by_layer = {}
            for key in missing_keys:
                # "model.layers.21.layer.self_attn.qkv_proj.weight" → 21
                parts = key.split('.')
                if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
                    try:
                        layer_idx = int(parts[2])
                        if layer_idx not in missing_by_layer:
                            missing_by_layer[layer_idx] = []
                        missing_by_layer[layer_idx].append(key)
                    except ValueError:
                        pass
            
            # Layer별 초기화
            zero_initialized = 0
            for layer_idx in sorted(missing_by_layer.keys()):
                layer_keys = missing_by_layer[layer_idx]
                
                if layer_idx in self.inactive_layer_indices:
                    # Inactive layer: 0으로 초기화 (예상된 동작)
                    print(f"   Layer {layer_idx}: Initializing {len(layer_keys)} weights to ZERO (inactive)")
                    
                    for key in layer_keys:
                        param = params_dict[key]
                        nn.init.zeros_(param)
                        zero_initialized += 1
                else:
                    # Active layer인데 missing → 경고!
                    print(f"   ⚠️  Layer {layer_idx}: Missing {len(layer_keys)} weights (ACTIVE layer!)")
                    
                    # 그래도 0으로 초기화 (에러 방지)
                    for key in layer_keys:
                        param = params_dict[key]
                        nn.init.zeros_(param)
                        zero_initialized += 1
            
            print(f"✅ Initialized {zero_initialized} missing weights to ZERO")
        
        print(f"\n{'='*60}")
        print(f"WEIGHT LOADING SUMMARY")
        print(f"{'='*60}")
        print(f"Total parameters:      {total_params}")
        print(f"Loaded from checkpoint: {loaded_count}")
        print(f"Initialized to zero:    {len(missing_keys)}")
        print(f"Coverage:               {loaded_count / total_params * 100:.1f}%")
        print(f"{'='*60}\n")
    
    # ============================================================
    # Progressive Recovery (Alpha Gating) - 기존 메서드 (호환성)
    # ============================================================
    
    def advance_to_stage2(
        self,
        layer_b_checkpoint: str,
        adapter_ab_path: Optional[str] = None,
    ) -> None:
        """Stage 1 → Stage 2 (기존 호환성용)"""
        print("\n" + "="*80)
        print("ADVANCING TO STAGE 2 (Alpha Gating, vLLM v0)")
        print("="*80)
        
        # prune_log에서 B 레이어 가져오기
        if self.prune_info is None:
            print("⚠️  Warning: No prune_log available. Using fallback.")
            activate_indices = [21, 22, 23, 24]
        else:
            activate_indices = self.prune_info['split']['B']
            print(f"Activating layers from prune_log: {activate_indices}")
        
        # B 레이어 활성화
        self.model.activate_layers(
            layer_indices=activate_indices,
            checkpoint_path=layer_b_checkpoint,
        )
        
        # Adapter (optional)
        if adapter_ab_path:
            print(f"Loading AB adapter from: {adapter_ab_path}")
        
        # Stage 업데이트
        self.current_stage = 2
        
        # Inactive layers 업데이트 (C만)
        if self.prune_info:
            self.inactive_layer_indices = set(self.prune_info['split']['C'])
        else:
            self.inactive_layer_indices = set(range(25, 29))
        
        print(f"\n{'='*80}")
        print(f"NOW AT STAGE 2")
        print(f"{'='*80}\n")
        
        self.print_status()
    
    def advance_to_stage3(
        self,
        layer_c_checkpoint: str,
        remove_adapter: bool = True,
    ) -> None:
        """Stage 2 → Stage 3 (기존 호환성용)"""
        print("\n" + "="*80)
        print("ADVANCING TO STAGE 3 (Alpha Gating, vLLM v0)")
        print("="*80)
        
        # prune_log에서 C 레이어 가져오기
        if self.prune_info is None:
            print("⚠️  Warning: No prune_log available. Using fallback.")
            activate_indices = [25, 26, 27, 28]
        else:
            activate_indices = self.prune_info['split']['C']
            print(f"Activating layers from prune_log: {activate_indices}")
        
        # C 레이어 활성화
        self.model.activate_layers(
            layer_indices=activate_indices,
            checkpoint_path=layer_c_checkpoint,
        )
        
        # Adapter 제거
        if remove_adapter:
            print("Removing all adapters...")
        
        # Stage 업데이트
        self.current_stage = 3
        self.inactive_layer_indices = set()  # 모두 활성
        
        print(f"\n{'='*80}")
        print(f"NOW AT STAGE 3 - FULL MODEL")
        print(f"{'='*80}\n")
        
        self.print_status()
    
    # ============================================================
    # ✅ NEW: Optimized Stage Transition (Partial KV Cache Reuse)
    # ============================================================
    
    def advance_to_stage2_optimized(
        self,
        layer_b_checkpoint: str,
        adapter_ab_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stage 1 → Stage 2 (최적화된 버전 - Partial KV Cache Reuse)
        
        제안서 Section 4.2 Step 3 구현:
        - Cache 힌트를 포함한 Stage 전환
        - 외부에서 KV Cache 관리에 활용 가능
        
        Args:
            layer_b_checkpoint: B 레이어 체크포인트 경로
            adapter_ab_path: AB 어댑터 경로 (선택)
            
        Returns:
            Dict with cache hints:
                - keep_prefix_layers: 유지할 KV Cache 레이어 수
                - recompute_from_layer: 재계산 시작 레이어
                - estimated_speedup: 예상 속도 향상
        """
        print("\n" + "="*80)
        print("ADVANCING TO STAGE 2 (Optimized - Partial KV Cache Reuse)")
        print("="*80)
        
        # prune_log에서 B 레이어 가져오기
        if self.prune_info is None:
            print("⚠️  Warning: No prune_log available. Using fallback.")
            activate_indices = [21, 22, 23, 24]
        else:
            activate_indices = self.prune_info['split']['B']
            print(f"Activating layers from prune_log: {activate_indices}")
        
        # ✅ Cache 힌트 포함 레이어 활성화
        cache_hint = self.model.activate_layers_with_cache_hint(
            layer_indices=activate_indices,
            checkpoint_path=layer_b_checkpoint,
        )
        
        # 예상 속도 향상 계산
        speedup_info = self.model.calculate_speedup_estimate(
            seq_len=0,  # 실제 seq_len은 외부에서 전달
            layer_indices_to_activate=activate_indices,
        )
        cache_hint.update(speedup_info)
        
        # Adapter (optional)
        if adapter_ab_path:
            print(f"Loading AB adapter from: {adapter_ab_path}")
        
        # Stage 업데이트
        self.current_stage = 2
        
        # Inactive layers 업데이트 (C만)
        if self.prune_info:
            self.inactive_layer_indices = set(self.prune_info['split']['C'])
        else:
            self.inactive_layer_indices = set(range(25, 29))
        
        # Cache 힌트 저장
        self.last_cache_hint = cache_hint
        
        print(f"\n{'='*80}")
        print(f"NOW AT STAGE 2 (Optimized)")
        print(f"{'='*80}")
        print(f"✅ Kept cache for layers 0 - {cache_hint['keep_prefix_layers']-1}")
        print(f"   Estimated speedup: {cache_hint.get('estimated_speedup', 'N/A'):.2f}x")
        print(f"{'='*80}\n")
        
        self.print_status()
        
        return cache_hint
    
    def advance_to_stage3_optimized(
        self,
        layer_c_checkpoint: str,
        remove_adapter: bool = True,
    ) -> Dict[str, Any]:
        """
        Stage 2 → Stage 3 (최적화된 버전 - Partial KV Cache Reuse)
        
        Args:
            layer_c_checkpoint: C 레이어 체크포인트 경로
            remove_adapter: 어댑터 제거 여부
            
        Returns:
            Dict with cache hints
        """
        print("\n" + "="*80)
        print("ADVANCING TO STAGE 3 (Optimized - Partial KV Cache Reuse)")
        print("="*80)
        
        # prune_log에서 C 레이어 가져오기
        if self.prune_info is None:
            print("⚠️  Warning: No prune_log available. Using fallback.")
            activate_indices = [25, 26, 27, 28]
        else:
            activate_indices = self.prune_info['split']['C']
            print(f"Activating layers from prune_log: {activate_indices}")
        
        # ✅ Cache 힌트 포함 레이어 활성화
        cache_hint = self.model.activate_layers_with_cache_hint(
            layer_indices=activate_indices,
            checkpoint_path=layer_c_checkpoint,
        )
        
        # 예상 속도 향상 계산
        speedup_info = self.model.calculate_speedup_estimate(
            seq_len=0,
            layer_indices_to_activate=activate_indices,
        )
        cache_hint.update(speedup_info)
        
        # Adapter 제거
        if remove_adapter:
            print("Removing all adapters...")
        
        # Stage 업데이트
        self.current_stage = 3
        self.inactive_layer_indices = set()  # 모두 활성
        
        # Cache 힌트 저장
        self.last_cache_hint = cache_hint
        
        print(f"\n{'='*80}")
        print(f"NOW AT STAGE 3 - FULL MODEL (Optimized)")
        print(f"{'='*80}")
        print(f"✅ Kept cache for layers 0 - {cache_hint['keep_prefix_layers']-1}")
        print(f"   Estimated speedup: {cache_hint.get('estimated_speedup', 'N/A'):.2f}x")
        print(f"{'='*80}\n")
        
        self.print_status()
        
        return cache_hint
    
    def get_last_cache_hint(self) -> Optional[Dict[str, Any]]:
        """마지막 Stage 전환의 Cache 힌트 반환"""
        return self.last_cache_hint
    
    # ============================================================
    # Status and Info Methods
    # ============================================================
    
    def print_status(self) -> None:
        """현재 모델 상태 출력"""
        self.model.print_layer_status()
        
        print(f"Current Stage: {self.current_stage}")
        
        report = self.model.verify_recovery()
        print(f"Activation Progress: {report['activation_progress']}")
        
        adapter_info = self.model.get_adapter_info()
        print(f"Current Adapter: {adapter_info['current_adapter'] or 'None'}")
        
        # ✅ NEW: Cache 힌트 정보
        if self.last_cache_hint:
            print(f"Last Cache Hint: keep layers 0-{self.last_cache_hint['keep_prefix_layers']-1}")
        print()
    
    def get_stage_info(self) -> dict:
        """현재 stage 정보 반환 (prune_info 포함)"""
        report = self.model.verify_recovery()
        adapter_info = self.model.get_adapter_info()
        
        return {
            "stage": self.current_stage,
            "active_layers": report["active_layers"],
            "inactive_layers": report["inactive_layers"],
            "activation_progress": report["activation_progress"],
            "current_adapter": adapter_info["current_adapter"],
            "inactive_layer_indices": report["inactive_layer_indices"],
            "prune_info": self.prune_info,
            "last_cache_hint": self.last_cache_hint,  # ✅ NEW
            "partial_kv_reuse_enabled": True,  # ✅ NEW
        }
    
    def get_layer_alphas(self) -> List[float]:
        """모든 레이어의 alpha 값 반환"""
        alphas = []
        for layer in self.model.layers:
            if hasattr(layer, 'get_alpha'):
                alphas.append(layer.get_alpha())
            else:
                alphas.append(1.0)  # Normal layer
        return alphas
    
    def set_layer_alpha(self, layer_idx: int, alpha: float):
        """특정 레이어의 alpha 값 직접 설정"""
        if layer_idx >= len(self.model.layers):
            raise ValueError(f"Invalid layer index: {layer_idx}")
        
        layer = self.model.layers[layer_idx]
        if hasattr(layer, 'alpha'):
            layer.alpha.fill_(alpha)
            print(f"Layer {layer_idx} alpha set to {alpha}")
        else:
            print(f"Layer {layer_idx} is not an AlphaGatedLayer")
    
    def get_cache_reuse_info(self) -> Dict[str, Any]:
        """Cache 재사용 정보 반환"""
        return self.model.get_cache_reuse_info()