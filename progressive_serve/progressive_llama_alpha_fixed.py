"""
ProgressiveLlamaModel with Alpha Gating (vLLM v0 Engine) - CUDA Graph 호환 버전

1. load_state_dict() → .copy_()로 변경 (메모리 주소 유지)
2. fused_weights를 실제로 사용하도록 수정
3. CUDA Graph 재캡처 불필요
"""

from typing import Optional, List, Dict, Any
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
        레이어 활성화 (alpha: 0 → 1) -  CUDA Graph 호환
        
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
        print("LAYER STATUS (Alpha Gating - CUDA Graph Compatible)")
        print("="*60)
        
        for start in range(0, len(status), 10):
            end = min(start + 10, len(status))
            print(f"\nLayers {start:2d}-{end-1:2d}:")
            
            for i in range(start, end):
                info = status[i]
                active = info['active']
                alpha = info['alpha']
                symbol = "o" if active else "x"
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
        print(f" CUDA Graph:        Compatible (no recapture needed)")
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
Progressive LLaMA Alpha Gating - CUDA Graph 호환 버전
====================================================

주요 개선사항:
1. ✅ .copy_()로 weight 로드 (메모리 주소 유지)
2. ✅ Alpha buffer는 register_buffer + fill_() (주소 고정)
3. ✅ CUDA Graph 재캡처 불필요

테스트:
    import torch
    from vllm.config import VllmConfig
    from progressive_llama_alpha_fixed import ProgressiveLlamaModelAlpha
    
    # 초기화
    model = ProgressiveLlamaModelAlpha(...)
    model.cuda()
    model.eval()
    
    # CUDA Graph 캡처
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out1 = model.forward(...)
    
    # Stage 전환
    model.activate_layers([21, 22, 23, 24], "stage2.safetensors")
    
    # Graph 재실행 (재캡처 없이!)
    g.replay()  # ✅ 성공!
""")