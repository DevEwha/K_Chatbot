"""
Alpha Gating Layer for ProgressiveServe (vLLM v0 Compatible)
progressive_serve/alpha_gated_layer.py

CUDA Graph 호환 동적 레이어 활성화 + Split KV Cache 지원
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    import torch.cuda.nvtx as nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False
    class DummyNVTX:
        @staticmethod
        def range_push(msg): pass
        @staticmethod
        def range_pop(): pass
    nvtx = DummyNVTX()


class AlphaGatedLayer(nn.Module):
    """
    Alpha Gating을 사용한 동적 레이어 활성화 (vLLM v0)
    
    핵심 아이디어:
        y = x + alpha * F(x)
        
    - alpha = 0: 레이어 비활성 (Pass through)
    - alpha = 1: 레이어 활성 (Normal operation)
    
    장점:
    1. CUDA Graph 호환: 커널 개수 항상 동일
    2. 동적 활성화: alpha만 변경하면 됨
    3. Weight 0 초기화: 추론에 영향 없음
    
    vLLM v0 호환:
    - forward 시그니처: (positions, hidden_states, residual)
    - kv_cache와 attn_metadata는 vLLM 내부에서 처리
    
    Split KV Cache 지원:
    - split_cache_manager를 통한 Base/Delta 분리 캐싱
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
        
        # 레이어 인덱스 (나중에 설정됨)
        self._layer_idx: Optional[int] = None
        
        # Split KV Cache Manager (외부에서 설정)
        self._split_cache_manager = None
    
    def set_layer_idx(self, idx: int):
        """레이어 인덱스 설정"""
        self._layer_idx = idx
    
    def set_split_cache_manager(self, manager):
        """Split Cache Manager 설정"""
        self._split_cache_manager = manager
    
    @property
    def layer_idx(self) -> Optional[int]:
        """레이어 인덱스"""
        return self._layer_idx
    
    def forward(
        self, 
        positions: torch.Tensor, 
        hidden_states: torch.Tensor, 
        residual: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with Alpha Gating
        
        Args:
            positions: Position IDs
            hidden_states: Input hidden states
            residual: Residual tensor (from previous layer)
            
        Returns:
            (gated_delta, updated_residual)
        """
        # base layer가 residual 관리 (위임)
        nvtx.range_push("BaseLayer")
        delta, updated_residual = self.layer(positions, hidden_states, residual)
        nvtx.range_pop()
        
        # alpha gating 적용
        nvtx.range_push("AlphaMultiply")
        gated_delta = self.alpha * delta
        nvtx.range_pop()
        
        # vLLM 표준: (delta, residual) 반환 (합치지 않음!)
        return gated_delta, updated_residual
    
    def forward_with_split_cache(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        use_cache: bool = True,
        recompute_delta_only: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Split KV Cache를 활용한 Forward pass
        
        Args:
            positions: Position IDs
            hidden_states: Input hidden states
            residual: Residual tensor
            use_cache: Cache 사용 여부
            recompute_delta_only: Delta만 재계산 (Base 재사용)
            
        Returns:
            (gated_delta, updated_residual)
            
        Note:
            이 메서드는 Split KV Cache가 Attention 레벨에서 통합될 때 사용됩니다.
            현재는 기본 forward와 동일하게 동작합니다.
        """
        # 현재는 기본 forward로 위임
        # 추후 Attention 레벨 통합 시 확장
        return self.forward(positions, hidden_states, residual)
    
    def activate(self):
        """레이어 활성화 (alpha = 1)"""
        self.alpha.fill_(1.0)
        self._is_active = True
        layer_str = f"Layer {self._layer_idx}" if self._layer_idx is not None else "Layer"
        print(f"✅ 💛 {layer_str} activated (alpha = 1.0)")
    
    def deactivate(self):
        """레이어 비활성화 (alpha = 0)"""
        self.alpha.fill_(0.0)
        self._is_active = False
        layer_str = f"Layer {self._layer_idx}" if self._layer_idx is not None else "Layer"
        print(f"⊗ {layer_str} deactivated (alpha = 0.0)")
    
    def is_active(self) -> bool:
        """활성화 여부 확인"""
        return self._is_active
    
    def get_alpha(self) -> float:
        """현재 alpha 값"""
        return self.alpha.item()
    
    def set_alpha(self, value: float):
        """alpha 값 직접 설정"""
        self.alpha.fill_(value)
        self._is_active = value > 0.5
    
    @property
    def is_alpha_gated(self) -> bool:
        """AlphaGatedLayer 식별용"""
        return True
    
    def __repr__(self) -> str:
        layer_str = f"layer_idx={self._layer_idx}, " if self._layer_idx is not None else ""
        return f"AlphaGatedLayer({layer_str}alpha={self.get_alpha():.2f}, active={self._is_active})"