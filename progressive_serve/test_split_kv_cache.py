#!/usr/bin/env python3
"""
Split KV Cache 기능 테스트
progressive_serve/test_split_kv_cache.py

테스트 항목:
1. SplitCacheManager 기본 기능
2. Stage 전환 분석
3. Cache 무효화 로직
4. 메모리 통계
"""

import sys
import torch

# 경로 추가 (필요시)
sys.path.insert(0, '/home/claude')

from split_kv_cache import SplitKVCache, SplitCacheManager


def test_split_kv_cache_dataclass():
    """SplitKVCache 데이터 구조 테스트"""
    print("\n" + "="*60)
    print("TEST 1: SplitKVCache 데이터 구조")
    print("="*60)
    
    # 더미 데이터 생성
    batch_size, seq_len, num_heads, head_dim = 1, 100, 32, 128
    k_base = torch.randn(batch_size, seq_len, num_heads, head_dim)
    v_base = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k_delta = torch.randn(batch_size, seq_len, num_heads, head_dim) * 0.1
    v_delta = torch.randn(batch_size, seq_len, num_heads, head_dim) * 0.1
    
    # Cache 생성
    cache = SplitKVCache(
        k_base=k_base,
        v_base=v_base,
        k_delta=k_delta,
        v_delta=v_delta,
        layer_idx=10,
        seq_len=seq_len,
    )
    
    # 테스트 1: 최종 KV 계산
    k_final, v_final = cache.get_final_kv()
    expected_k = k_base + k_delta
    expected_v = v_base + v_delta
    
    assert torch.allclose(k_final, expected_k), "K_final 계산 오류"
    assert torch.allclose(v_final, expected_v), "V_final 계산 오류"
    print("✅ get_final_kv() 정확")
    
    # 테스트 2: 메모리 사용량
    base_mb, delta_mb = cache.memory_usage_mb()
    print(f"✅ 메모리 사용량: Base={base_mb:.2f}MB, Delta={delta_mb:.2f}MB")
    
    # 테스트 3: Delta 삭제
    cache.clear_delta()
    assert cache.k_delta is None, "Delta 삭제 오류"
    
    k_final_no_delta, _ = cache.get_final_kv()
    assert torch.allclose(k_final_no_delta, k_base), "Delta 없을 때 계산 오류"
    print("✅ clear_delta() 정상 동작")
    
    print("\n✅ TEST 1 PASSED")


def test_cache_manager_basic():
    """CacheManager 기본 기능 테스트"""
    print("\n" + "="*60)
    print("TEST 2: SplitCacheManager 기본 기능")
    print("="*60)
    
    manager = SplitCacheManager(num_layers=32)
    
    # 테스트 1: Cache 저장
    for layer_idx in [0, 5, 10, 15, 20]:
        k = torch.randn(1, 100, 32, 128)
        v = torch.randn(1, 100, 32, 128)
        manager.set_cache(layer_idx, k, v, seq_len=100)
    
    assert len(manager.caches) == 5, "Cache 저장 오류"
    print(f"✅ 5개 레이어 cache 저장 완료")
    
    # 테스트 2: Cache 조회
    assert manager.has_cache(10), "has_cache() 오류"
    assert not manager.has_cache(25), "has_cache() 오류 (없는 레이어)"
    
    cache = manager.get_cache(10)
    assert cache is not None, "get_cache() 오류"
    assert cache.layer_idx == 10, "layer_idx 오류"
    print(f"✅ get_cache() 정상 동작")
    
    # 테스트 3: 메모리 통계
    stats = manager.get_memory_stats()
    assert stats['num_layers'] == 5, "레이어 수 오류"
    assert stats['base_MB'] > 0, "메모리 계산 오류"
    print(f"✅ 메모리 통계: {stats['num_layers']} layers, {stats['total_MB']:.2f}MB")
    
    # 테스트 4: 부분 무효화
    manager.invalidate_layer_range(10, 15)
    assert not manager.has_cache(10), "무효화 오류"
    assert manager.has_cache(0), "잘못된 무효화"
    print(f"✅ invalidate_layer_range() 정상 동작")
    
    print("\n✅ TEST 2 PASSED")


def test_stage_transition_analysis():
    """Stage 전환 분석 테스트"""
    print("\n" + "="*60)
    print("TEST 3: Stage 전환 분석")
    print("="*60)
    
    manager = SplitCacheManager(num_layers=32)
    
    # Stage 설정 (Progressive Serving 구조)
    stage_configs = {
        1: {'active_layers': [(0, 20), (29, 31)]},      # Layer 0-20, 29-31
        2: {'active_layers': [(0, 20), (21, 24), (29, 31)]},  # + Layer 21-24
        3: {'active_layers': [(0, 31)]},  # Full model
    }
    manager.set_stage_configs(stage_configs, current_stage=1)
    
    # 테스트 1: Stage 1 → 2 분석
    print("\n📊 Stage 1 → 2 분석:")
    analysis_1_2 = manager.analyze_stage_transition(1, 2)
    
    print(f"  재사용 가능: {analysis_1_2['reusable']}")
    print(f"  새로 추가: {analysis_1_2['new']}")
    print(f"  무효화 필요: {analysis_1_2['invalidated']}")
    
    # 검증
    # 재사용 가능한 레이어에 0-20이 포함되어야 함
    reusable_set = manager._ranges_to_set(analysis_1_2['reusable'])
    assert all(l in reusable_set for l in range(0, 21)), "Layer 0-20 재사용 가능해야 함"
    assert (21, 24) in analysis_1_2['new'], "Layer 21-24 새로 추가됨"
    
    # 무효화 필요한 레이어에 29-31이 포함되어야 함
    invalidated_set = manager._ranges_to_set(analysis_1_2['invalidated'])
    assert all(l in invalidated_set for l in range(29, 32)), "Layer 29-31 무효화 필요"
    print("✅ Stage 1→2 분석 정확")
    
    # 테스트 2: Stage 2 → 3 분석
    print("\n📊 Stage 2 → 3 분석:")
    analysis_2_3 = manager.analyze_stage_transition(2, 3)
    
    print(f"  재사용 가능: {analysis_2_3['reusable']}")
    print(f"  새로 추가: {analysis_2_3['new']}")
    print(f"  무효화 필요: {analysis_2_3['invalidated']}")
    
    # 검증
    reusable_set_2 = manager._ranges_to_set(analysis_2_3['reusable'])
    assert all(l in reusable_set_2 for l in range(0, 21)), "Layer 0-20 재사용 가능해야 함"
    assert all(l in reusable_set_2 for l in range(21, 25)), "Layer 21-24 재사용 가능해야 함"
    assert (25, 28) in analysis_2_3['new'], "Layer 25-28 새로 추가됨"
    
    invalidated_set_2 = manager._ranges_to_set(analysis_2_3['invalidated'])
    assert all(l in invalidated_set_2 for l in range(29, 32)), "Layer 29-31 무효화 필요"
    print("✅ Stage 2→3 분석 정확")
    
    print("\n✅ TEST 3 PASSED")


def test_stage_transition_execution():
    """Stage 전환 실행 테스트"""
    print("\n" + "="*60)
    print("TEST 4: Stage 전환 실행")
    print("="*60)
    
    manager = SplitCacheManager(num_layers=32)
    
    # Stage 설정
    stage_configs = {
        1: {'active_layers': [(0, 20), (29, 31)]},
        2: {'active_layers': [(0, 20), (21, 24), (29, 31)]},
        3: {'active_layers': [(0, 31)]},
    }
    manager.set_stage_configs(stage_configs, current_stage=1)
    
    # Stage 1에서 Cache 생성
    print("\n1️⃣ Stage 1 Cache 생성:")
    for layer_idx in [0, 5, 10, 15, 20, 29, 30, 31]:
        k = torch.randn(1, 100, 32, 128)
        v = torch.randn(1, 100, 32, 128)
        k_delta = torch.randn(1, 100, 32, 128) * 0.1
        v_delta = torch.randn(1, 100, 32, 128) * 0.1
        manager.set_cache(layer_idx, k, v, k_delta, v_delta, seq_len=100)
    
    print(f"   캐시된 레이어: {sorted(manager.caches.keys())}")
    
    # Stage 1 → 2 전환
    print("\n2️⃣ Stage 1 → 2 전환:")
    analysis = manager.handle_stage_transition(1, 2, verbose=True)
    
    # 검증: Layer 0-20 유지, Layer 29-31 삭제, Delta 모두 삭제
    assert manager.has_cache(0), "Layer 0 유지되어야 함"
    assert manager.has_cache(20), "Layer 20 유지되어야 함"
    assert not manager.has_cache(29), "Layer 29 무효화되어야 함"
    assert not manager.has_cache(30), "Layer 30 무효화되어야 함"
    
    # Delta 확인 (모두 None이어야 함)
    for layer_idx, cache in manager.caches.items():
        assert cache.k_delta is None, f"Layer {layer_idx} delta 삭제되어야 함"
    
    print(f"   캐시된 레이어: {sorted(manager.caches.keys())}")
    print("✅ Stage 1→2 전환 정상")
    
    # Stage 2 → 3 전환
    print("\n3️⃣ Stage 2 → 3 전환:")
    
    # 추가 cache (Stage 2에서 새로 계산된 것으로 가정)
    for layer_idx in [21, 22, 23, 24, 29, 30, 31]:
        k = torch.randn(1, 100, 32, 128)
        v = torch.randn(1, 100, 32, 128)
        manager.set_cache(layer_idx, k, v, seq_len=100)
    
    analysis = manager.handle_stage_transition(2, 3, verbose=True)
    
    # 검증: Layer 0-24 유지, Layer 29-31 삭제
    assert manager.has_cache(0), "Layer 0 유지되어야 함"
    assert manager.has_cache(20), "Layer 20 유지되어야 함"
    assert manager.has_cache(24), "Layer 24 유지되어야 함"
    assert not manager.has_cache(29), "Layer 29 무효화되어야 함"
    
    print(f"   캐시된 레이어: {sorted(manager.caches.keys())}")
    print("✅ Stage 2→3 전환 정상")
    
    print("\n✅ TEST 4 PASSED")


def test_performance_estimate():
    """성능 향상 추정 테스트"""
    print("\n" + "="*60)
    print("TEST 5: 성능 향상 추정")
    print("="*60)
    
    # 설정
    seq_len = 1000
    hidden_dim = 4096
    num_layers = 32
    lora_rank = 8
    
    # 레이어당 KV 계산 비용 (FLOPs)
    flops_per_layer = 2 * seq_len * hidden_dim * hidden_dim  # K + V
    
    # 기존 방식: 전체 재계산
    baseline_flops = num_layers * flops_per_layer
    
    # Stage 1 → 2 (Split Cache)
    # 재사용: Layer 0-20 (21개) - Delta만 재계산
    # 신규: Layer 21-24 (4개) - 전체 계산
    # 재계산: Layer 29-31 (3개) - 전체 계산
    
    delta_flops_per_layer = 2 * seq_len * lora_rank * hidden_dim * 2  # LoRA delta
    
    reused_layers = 21
    new_layers = 4
    recompute_layers = 3
    
    split_cache_flops = (
        reused_layers * delta_flops_per_layer +  # Delta만
        (new_layers + recompute_layers) * flops_per_layer  # 전체
    )
    
    speedup = baseline_flops / split_cache_flops
    
    print(f"📊 Stage 1 → 2 분석:")
    print(f"   기존 방식: {baseline_flops / 1e9:.2f}B FLOPs")
    print(f"   Split Cache: {split_cache_flops / 1e9:.2f}B FLOPs")
    print(f"   속도 향상: {speedup:.2f}x")
    
    # Stage 2 → 3
    reused_layers_2 = 25  # Layer 0-24
    new_layers_2 = 4  # Layer 25-28
    recompute_layers_2 = 3  # Layer 29-31
    
    split_cache_flops_2 = (
        reused_layers_2 * delta_flops_per_layer +
        (new_layers_2 + recompute_layers_2) * flops_per_layer
    )
    
    speedup_2 = baseline_flops / split_cache_flops_2
    
    print(f"\n📊 Stage 2 → 3 분석:")
    print(f"   기존 방식: {baseline_flops / 1e9:.2f}B FLOPs")
    print(f"   Split Cache: {split_cache_flops_2 / 1e9:.2f}B FLOPs")
    print(f"   속도 향상: {speedup_2:.2f}x")
    
    # 검증
    assert speedup > 3, f"Stage 1→2 속도 향상이 3배 이상이어야 함 (실제: {speedup:.2f}x)"
    assert speedup_2 > 3, f"Stage 2→3 속도 향상이 3배 이상이어야 함 (실제: {speedup_2:.2f}x)"
    
    print(f"\n✅ 예상 속도 향상: 4배 이상")
    print("\n✅ TEST 5 PASSED")


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "="*70)
    print("Split KV Cache 기능 테스트")
    print("="*70)
    
    tests = [
        ("SplitKVCache 데이터 구조", test_split_kv_cache_dataclass),
        ("CacheManager 기본 기능", test_cache_manager_basic),
        ("Stage 전환 분석", test_stage_transition_analysis),
        ("Stage 전환 실행", test_stage_transition_execution),
        ("성능 향상 추정", test_performance_estimate),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ {name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ {name} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"테스트 결과: {passed}/{len(tests)} PASSED")
    if failed > 0:
        print(f"⚠️  {failed}개 테스트 실패")
    else:
        print("🎉 모든 테스트 통과!")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)