"""
ProgressiveLlamaForCausalLM with Alpha Gating for vLLM v0
progressive_llama_for_causal_lm_alpha_v0.py

✅ v2 업데이트: prune_log.json 기반 자동 레이어 결정
"""

from typing import Optional, List, Iterable, Tuple
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
    """
    supports_multimodal=False
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
        
        self.supports_lora=False
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
        
        print(f"\n{'='*60}")
        print(f"ProgressiveLlamaForCausalLMAlpha (vLLM v0, Alpha Gating)")
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
        **kwargs,
    ) -> torch.Tensor:
        """
        vLLM v0 forward
        
        Note: kv_caches와 attn_metadata를 kwargs에서 받지만
              실제로는 vLLM 내부에서 자동 처리됨
        """
        # kwargs에서 가져오기 (vLLM v0가 전달)
        kv_caches = kwargs.get('kv_caches', None)
        attn_metadata = kwargs.get('attn_metadata', None)
        
        # Model forward
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
        )
        
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Weights 로드 with Smart Missing Weight Handling + Fused Weights
        
        핵심: Checkpoint에 없는 weight는 자동으로 0으로 초기화
        
        Args:
            weights: Iterator of (name, tensor) pairs
        """
        # 모든 모델 파라미터
        params_dict = dict(self.named_parameters())
        total_params = len(params_dict)
        print(f"Total model parameters: {total_params}")
        print(f"Inactive layers: {sorted(self.inactive_layer_indices)}")

        for i, key in enumerate(list(params_dict.keys())[:10]):
            print(f"  {i+1}. {key}")
        print()
        
        # 1단계: Checkpoint weights를 dict에 저장
        checkpoint_weights = {}
        for name, weight in weights:
            checkpoint_weights[name] = weight
        print(f"Checkpoint contains {len(checkpoint_weights)} weights\n")
        
        print(f"\n{'='*80}")
        print("DEBUG: Checking checkpoint contents")
        print(f"{'='*80}")

        gate_up_keys = [k for k in checkpoint_weights.keys() 
                        if 'gate_proj' in k or 'up_proj' in k]
        print(f"Gate/Up keys in checkpoint: {len(gate_up_keys)}")

        if gate_up_keys:
            print("First 5 gate/up keys:")
            for k in gate_up_keys[:5]:
                print(f"  - {k}")
        else:
            print("⚠️  NO gate_proj or up_proj in Stage 1 checkpoint!")
            print("   This is expected for pruned model.")

        print(f"{'='*80}\n")
        print(f"\n{'='*60}")
        print(f"LOADING WEIGHTS (Smart Mode, vLLM v0)")
        print(f"{'='*60}")
        
        loaded_keys = set()
        loaded_count = 0
        
        # 2단계: 모델 파라미터 순회하면서 로딩
        for param_name, param in params_dict.items():
        
            # Option 1: 직접 매칭 (embed_tokens, norm 등)
            if param_name in checkpoint_weights:
                weight_loader = getattr(param, "weight_loader", 
                                   lambda p, w: p.data.copy_(w))
                weight_loader(param, checkpoint_weights[param_name])
                loaded_keys.add(param_name)
                loaded_count += 1
                continue
                
            # Option 2: .layer. 제거 후 매칭 (먼저 실행!)
            if ".layer." in param_name:
                original_name = param_name.replace(".layer.", ".")
        
                # Fusion 대상은 건너뛰기
                if "qkv_proj" in param_name or "gate_up_proj" in param_name:
                    pass  # 나중에 fusion으로 처리
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
                    # QKV fusion
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
    # Progressive Recovery (Alpha Gating)
    # ============================================================
    
    def advance_to_stage2(
        self,
        layer_b_checkpoint: str,
        adapter_ab_path: Optional[str] = None,
    ) -> None:
        """Stage 1 → Stage 2 (prune_log 기반)"""
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
        """Stage 2 → Stage 3 (prune_log 기반)"""
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
            "prune_info": self.prune_info,  # ← prune_log.json 정보 포함
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
