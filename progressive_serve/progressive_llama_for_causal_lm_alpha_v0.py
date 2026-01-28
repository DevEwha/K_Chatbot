"""
ProgressiveLlamaForCausalLM with Alpha Gating and Split KV Caching for vLLM v0
progressive_serve/progressive_llama_for_causal_lm_alpha_v0.py

핵심 기능:
1. prune_log.json 기반 자동 레이어 결정
2. Alpha Gating으로 동적 레이어 활성화
3. Split KV Caching으로 Stage 전환 최적화
   - Base cache 재사용 (75-78% 레이어)
   - Prefill 시간 4.5배 감소
"""

from typing import Optional, List, Iterable, Tuple, Dict
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

from progressive_llama_alpha import ProgressiveLlamaModelAlpha


class ProgressiveLlamaForCausalLMAlpha(nn.Module):
    """
    Alpha Gating + Split KV Caching을 사용한 ForCausalLM wrapper (vLLM v0)
    
    핵심 기능:
    1. prune_log.json 자동 로드 (하드코딩 제거)
    2. Missing weights 자동 0 초기화
    3. Split KV Cache로 Stage 전환 최적화
    4. vLLM v0 엔진 호환
    
    Split KV Cache 최적화:
    - Stage 1→2: Layer 0-20 재사용 (75%), Layer 29-31 재계산 (11%)
    - Stage 2→3: Layer 0-24 재사용 (78%), Layer 29-31 재계산 (9%)
    - 예상 Prefill 시간 감소: 4.5배
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
        
        # Split KV Cache를 위한 Stage 설정
        self.model.set_stage_configs_from_prune_info(
            self.prune_info,
            num_layers=config.num_hidden_layers
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
        
        print(f"\n{'='*70}")
        print(f"ProgressiveLlamaForCausalLMAlpha (vLLM v0)")
        print(f"{'='*70}")
        print(f"Model: {model_path}")
        print(f"Initial Stage: {stage}")
        
        if self.prune_info:
            print(f"✅ Prune log loaded")
            print(f"   Split B: {self.prune_info['split']['B']}")
            print(f"   Split C: {self.prune_info['split']['C']}")
        else:
            print(f"⚠️  Using fallback (no prune_log.json)")
        
        print(f"Initially inactive: {sorted(inactive_indices)}")
        print(f"")
        print(f"✨ Features:")
        print(f"   - Alpha Gating (CUDA Graph compatible)")
        print(f"   - Split KV Caching (4.5x faster stage transition)")
        print(f"   - Smart weight loading")
        print(f"{'='*70}\n")
    
    def _load_prune_log(self, model_path: str) -> Optional[dict]:
        """모델 디렉토리에서 prune_log.json 로드"""
        import json
        import os
        
        prune_log_path = os.path.join(model_path, "prune_log.json")
        
        if not os.path.exists(prune_log_path):
            return None
        
        try:
            with open(prune_log_path, 'r') as f:
                prune_log = json.load(f)
            
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
        """
        if prune_info is None:
            return self._get_inactive_indices_fallback(stage)
        
        try:
            split_b = prune_info['split']['B']
            split_c = prune_info['split']['C']
            
            if stage == 1:
                inactive = sorted(split_b + split_c)
            elif stage == 2:
                inactive = sorted(split_c)
            elif stage == 3:
                inactive = []
            else:
                raise ValueError(f"Invalid stage: {stage}")
            
            return inactive
            
        except Exception as e:
            print(f"❌ Error parsing prune_log: {e}")
            return self._get_inactive_indices_fallback(stage)
    
    def _get_inactive_indices_fallback(self, stage: int) -> List[int]:
        """Fallback: prune_log가 없을 때 기본값"""
        if stage == 1:
            return list(range(21, 29))
        elif stage == 2:
            return list(range(25, 29))
        elif stage == 3:
            return []
        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    # ============================================================
    # vLLM v0 Required Methods
    # ============================================================
    
    def compute_logits(
        self, 
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        """vLLM 0.7.x에서 생성 모델 인식을 위해 필요"""
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
        **kwargs,
    ) -> torch.Tensor:
        """vLLM v0 forward"""
        kv_caches = kwargs.get('kv_caches', None)
        attn_metadata = kwargs.get('attn_metadata', None)
        
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        
        return hidden_states
    
    # ============================================================
    # Weight Loading
    # ============================================================
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Weights 로드 with Smart Missing Weight Handling + Fused Weights"""
        params_dict = dict(self.named_parameters())
        total_params = len(params_dict)
        print(f"Total model parameters: {total_params}")
        print(f"Inactive layers: {sorted(self.inactive_layer_indices)}")
        
        # Checkpoint weights를 dict에 저장
        checkpoint_weights = {}
        for name, weight in weights:
            checkpoint_weights[name] = weight
        print(f"Checkpoint contains {len(checkpoint_weights)} weights")
        
        print(f"\n{'='*60}")
        print(f"LOADING WEIGHTS (Smart Mode, vLLM v0)")
        print(f"{'='*60}")
        
        loaded_keys = set()
        loaded_count = 0
        
        for param_name, param in params_dict.items():
            # Option 1: 직접 매칭
            if param_name in checkpoint_weights:
                weight_loader = getattr(param, "weight_loader", 
                                   lambda p, w: p.data.copy_(w))
                weight_loader(param, checkpoint_weights[param_name])
                loaded_keys.add(param_name)
                loaded_count += 1
                continue
            
            # Option 2: .layer. 제거 후 매칭
            if ".layer." in param_name:
                original_name = param_name.replace(".layer.", ".")
                
                if "qkv_proj" in param_name or "gate_up_proj" in param_name:
                    pass  # Fusion으로 처리
                elif original_name in checkpoint_weights:
                    weight_loader = getattr(param, "weight_loader",
                                       lambda p, w: p.data.copy_(w))
                    weight_loader(param, checkpoint_weights[original_name])
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
                    print(f"   Layer {layer_idx}: {len(layer_keys)} weights → ZERO (inactive)")
                    
                    for key in layer_keys:
                        param = params_dict[key]
                        nn.init.zeros_(param)
                        zero_initialized += 1
                else:
                    print(f"   ⚠️  Layer {layer_idx}: {len(layer_keys)} weights missing (ACTIVE!)")
                    
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
    # Progressive Recovery with Split KV Cache
    # ============================================================
    
    def advance_to_stage2(
        self,
        layer_b_checkpoint: str,
        adapter_ab_path: Optional[str] = None,
    ) -> Dict:
        """
        Stage 1 → Stage 2 전환 (Split KV Cache 최적화)
        
        Args:
            layer_b_checkpoint: B 레이어 weight 파일 경로
            adapter_ab_path: AB 어댑터 경로 (optional)
            
        Returns:
            전환 분석 결과
        """
        print("\n" + "="*80)
        print("ADVANCING TO STAGE 2 (Alpha Gating + Split KV Cache)")
        print("="*80)
        
        # prune_log에서 B 레이어 가져오기
        if self.prune_info is None:
            print("⚠️  Warning: No prune_log available. Using fallback.")
            activate_indices = [21, 22, 23, 24]
        else:
            activate_indices = self.prune_info['split']['B']
            print(f"Activating layers from prune_log: {activate_indices}")
        
        # Split Cache 최적화와 함께 Stage 전환
        analysis = self.model.handle_stage_transition(
            from_stage=1,
            to_stage=2,
            layer_checkpoint_path=layer_b_checkpoint,
            adapter_path=adapter_ab_path,
        )
        
        # Stage 업데이트
        self.current_stage = 2
        
        # Inactive layers 업데이트 (C만)
        if self.prune_info:
            self.inactive_layer_indices = set(self.prune_info['split']['C'])
        else:
            self.inactive_layer_indices = set(range(25, 29))
        
        print(f"\n{'='*80}")
        print(f"NOW AT STAGE 2")
        print(f"  - Reused layers: {self._format_ranges(analysis['reusable'])}")
        print(f"  - New layers: {self._format_ranges(analysis['new'])}")
        print(f"  - Recomputed layers: {self._format_ranges(analysis['invalidated'])}")
        print(f"{'='*80}\n")
        
        self.print_status()
        
        return analysis
    
    def advance_to_stage3(
        self,
        layer_c_checkpoint: str,
        remove_adapter: bool = True,
    ) -> Dict:
        """
        Stage 2 → Stage 3 전환 (Split KV Cache 최적화)
        
        Args:
            layer_c_checkpoint: C 레이어 weight 파일 경로
            remove_adapter: 어댑터 제거 여부
            
        Returns:
            전환 분석 결과
        """
        print("\n" + "="*80)
        print("ADVANCING TO STAGE 3 (Alpha Gating + Split KV Cache)")
        print("="*80)
        
        # prune_log에서 C 레이어 가져오기
        if self.prune_info is None:
            print("⚠️  Warning: No prune_log available. Using fallback.")
            activate_indices = [25, 26, 27, 28]
        else:
            activate_indices = self.prune_info['split']['C']
            print(f"Activating layers from prune_log: {activate_indices}")
        
        # Split Cache 최적화와 함께 Stage 전환
        analysis = self.model.handle_stage_transition(
            from_stage=2,
            to_stage=3,
            layer_checkpoint_path=layer_c_checkpoint,
            adapter_path=None if remove_adapter else self.model.current_adapter,
        )
        
        # Adapter 제거
        if remove_adapter:
            print("Removing all adapters...")
            self.model.current_adapter = None
        
        # Stage 업데이트
        self.current_stage = 3
        self.inactive_layer_indices = set()  # 모두 활성
        
        print(f"\n{'='*80}")
        print(f"NOW AT STAGE 3 - FULL MODEL")
        print(f"  - Reused layers: {self._format_ranges(analysis['reusable'])}")
        print(f"  - New layers: {self._format_ranges(analysis['new'])}")
        print(f"  - Recomputed layers: {self._format_ranges(analysis['invalidated'])}")
        print(f"{'='*80}\n")
        
        self.print_status()
        
        return analysis
    
    def _format_ranges(self, ranges: List[Tuple[int, int]]) -> str:
        """Range를 읽기 쉬운 문자열로 변환"""
        if not ranges:
            return "없음"
        parts = []
        for start, end in ranges:
            if start == end:
                parts.append(f"L{start}")
            else:
                parts.append(f"L{start}-{end}")
        return ", ".join(parts)
    
    # ============================================================
    # Status and Info Methods
    # ============================================================
    
    def print_status(self) -> None:
        """현재 모델 상태 출력"""
        self.model.print_layer_status()
        
        print(f"Current Stage: {self.current_stage}")
        
        report = self.model.verify_recovery()
        print(f"Activation Progress: {report['activation_progress']}")
        
        # Split Cache 상태
        if 'split_cache' in report:
            cache_info = report['split_cache']
            print(f"Split Cache: {cache_info['cached_layers']} layers, {cache_info['total_cache_mb']:.1f}MB")
        
        adapter_info = self.model.get_adapter_info()
        print(f"Current Adapter: {adapter_info['current_adapter'] or 'None'}")
        print()
    
    def print_cache_status(self) -> None:
        """Split Cache 상태 출력"""
        self.model.print_cache_status()
    
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
            "split_cache": report.get("split_cache", {}),
        }
    
    def get_layer_alphas(self) -> List[float]:
        """모든 레이어의 alpha 값 반환"""
        alphas = []
        for layer in self.model.layers:
            if hasattr(layer, 'get_alpha'):
                alphas.append(layer.get_alpha())
            else:
                alphas.append(1.0)
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
    
    def clear_split_cache(self):
        """Split Cache 전체 삭제"""
        self.model.clear_split_cache()
    
    def get_split_cache_stats(self) -> Dict:
        """Split Cache 통계 반환"""
        return self.model.split_cache_manager.get_memory_stats()