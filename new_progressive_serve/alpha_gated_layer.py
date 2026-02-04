"""
Alpha Gating Layer for ProgressiveServe (vLLM v0.7.4 Compatible)
alpha_gated_layer_fixed.py

CUDA Graph 호환 동적 레이어 활성화
✅ 수정사항:
1. kv_cache 파라미터 추가
2. attn_metadata 파라미터 추가
3. vLLM 0.7.4 호환 forward 시그니처
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Any

# NVTX는 선택적
try:
    import torch.cuda.nvtx as nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False
    class nvtx:
        @staticmethod
        def range_push(name): pass
        @staticmethod
        def range_pop(): pass


class AlphaGatedLayer(nn.Module):
    """
    Alpha Gating을 사용한 동적 레이어 활성화 (vLLM v0.7.4)
    
    핵심 아이디어:
        y = x + alpha * F(x)
        
    - alpha = 0: 레이어 비활성 (Pass through)
    - alpha = 1: 레이어 활성 (Normal operation)
    
    장점:
    
    1. CUDA Graph 호환: 커널 개수 항상 동일
    2. 동적 활성화: alpha만 변경하면 됨
    3. Weight 0 초기화: 추론에 영향 없음
    
    vLLM v0.7.4 호환:
    - forward 시그니처: (positions, hidden_states, kv_cache, attn_metadata, residual)
    - kv_cache가 반드시 전달되어야 함
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        initial_alpha: float = 0.0,
    ):
        """
        Args:
            base_layer: 실제 LlamaDecoderLayer
            initial_alpha: 초기 alpha 값 (0.0 = 비활성)
        """
        super().__init__()
        
        # 실제 레이어 (항상 존재)
        self.layer = base_layer
        
        # Alpha gate (learnable parameter는 아님)
        self.register_buffer('alpha', torch.tensor(initial_alpha))
        
        # 활성화 상태 플래그
        self._is_active = initial_alpha > 0.5
    
    def forward(
        self, 
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Any,  # vLLM KV Cache object
        attn_metadata: Any,  # vLLM AttentionMetadata
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with alpha gating
        
        Args:
            positions: Position tensor
            hidden_states: Input hidden states
            kv_cache: vLLM KV cache for this layer
            attn_metadata: vLLM attention metadata
            residual: Residual tensor from previous layer
            
        Returns:
            Tuple of (hidden_states, residual)
        """
        if HAS_NVTX:
            nvtx.range_push("AlphaGatedLayer")
        
        # base layer forward (kv_cache와 attn_metadata 전달!)
        if HAS_NVTX:
            nvtx.range_push("BaseLayer")
        
        # vLLM LlamaDecoderLayer의 forward 호출
        # 시그니처: forward(positions, hidden_states, kv_cache, attn_metadata, residual)
        delta, updated_residual = self.layer(
            positions, 
            hidden_states, 
            kv_cache, 
            attn_metadata,
            residual
        )
        
        if HAS_NVTX:
            nvtx.range_pop()
        
        # alpha gating 적용
        if HAS_NVTX:
            nvtx.range_push("AlphaMultiply")
        
        gated_delta = self.alpha * delta
        
        if HAS_NVTX:
            nvtx.range_pop()
        
        if HAS_NVTX:
            nvtx.range_pop()  # AlphaGatedLayer
        
        # vLLM 표준: (delta, residual) 반환 (합치지 않음!)
        return gated_delta, updated_residual
    
    def activate(self):
        """레이어 활성화 (alpha = 1)"""
        self.alpha.fill_(1.0)
        self._is_active = True
        print(f"  ✅ Layer activated (alpha = 1.0)")
    
    def deactivate(self):
        """레이어 비활성화 (alpha = 0)"""
        self.alpha.fill_(0.0)
        self._is_active = False
        print(f"  ⊗ Layer deactivated (alpha = 0.0)")
    
    def is_active(self) -> bool:
        """활성화 여부 확인"""
        return self._is_active
    
    def get_alpha(self) -> float:
        """현재 alpha 값"""
        return self.alpha.item()
    
    @property
    def is_alpha_gated(self) -> bool:
        """AlphaGatedLayer 식별용"""
        return True
