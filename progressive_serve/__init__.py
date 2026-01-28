"""
ProgressiveServe: Alpha Gating + Split KV Caching for Progressive LLM Serving
progressive_serve/__init__.py

핵심 기능:
1. Alpha Gating: 동적 레이어 활성화 (CUDA Graph 호환)
2. Split KV Caching: Stage 전환 시 Prefill 최적화 (4.5배 빠름)

사용법:
    from progressive_serve import (
        ProgressiveLlamaForCausalLMAlpha,
        ProgressiveLlamaModelAlpha,
        AlphaGatedLayer,
        SplitCacheManager,
    )
    
    # 모델 초기화
    model = ProgressiveLlamaForCausalLMAlpha(vllm_config, stage=1)
    
    # Stage 전환 (Split Cache 최적화 적용)
    model.advance_to_stage2("layer_b.safetensors")
    model.advance_to_stage3("layer_c.safetensors")
    
    # 상태 확인
    model.print_status()
    model.print_cache_status()
"""

from alpha_gated_layer import AlphaGatedLayer
from split_kv_cache import SplitKVCache, SplitCacheManager
from progressive_llama_alpha import ProgressiveLlamaModelAlpha
from progressive_llama_for_causal_lm_alpha_v0 import ProgressiveLlamaForCausalLMAlpha

__all__ = [
    "AlphaGatedLayer",
    "SplitKVCache",
    "SplitCacheManager",
    "ProgressiveLlamaModelAlpha",
    "ProgressiveLlamaForCausalLMAlpha",
]

__version__ = "2.0.0"  # Split KV Cache 추가