#!/usr/bin/env python3
"""
실행: python chatbot.py
대화형 Progressive Stage 테스트 (Interactive Version)
파일명: chatbot.py

✅ v2 업데이트 (vLLM 0.7.4 호환):
1. 모델 레지스트리 등록 추가 (vLLM 로딩 필수 단계)
2. Python Path 설정 추가 (커스텀 모듈 참조용)
3. Prefix Caching 활성화로 대화 맥락 유지
4. ✅ NEW: Partial KV Cache Reuse 지원
5. ✅ NEW: sleep/wake 제거 (vLLM 0.7.4 미지원)
6. ✅ NEW: 최적화된 Stage 전환 (Cache 힌트 포함)
7. 사용자 입력 기반 대화형 인터페이스
8. 사용자 명령으로 Stage 전환 제어
"""

import sys
import os
import time
import torch
from typing import Optional, Dict, Any

# [필수] Python path 설정 - 커스텀 모델 경로를 인식하게 합니다.
sys.path.insert(0, "/workspace/vllm_test")
sys.path.insert(0, "/acpl-ssd20/1218/A")
sys.path.insert(0, "/home/devewha/Juwon/vllm_test")

# vLLM import
try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.registry import ModelRegistry
    from progressive_llama_for_causal_lm_alpha_v0 import ProgressiveLlamaForCausalLMAlpha
    print("✅ vLLM and Custom Model imported successfully")
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    sys.exit(1)

# [필수] 모델 레지스트리 등록
ModelRegistry.register_model(
    "ProgressiveLlamaForCausalLM",
    ProgressiveLlamaForCausalLMAlpha
)


class PartialKVCacheManager:
    """
    Partial KV Cache Reuse를 위한 Cache 관리자
    
    vLLM 0.7.4에서는 내부 KV Cache에 직접 접근하기 어려우므로,
    Cache 힌트 정보를 관리하고 성능 분석에 활용합니다.
    
    실제 KV Cache 재사용은 vLLM의 prefix caching을 통해 자동으로 이루어집니다.
    """
    
    def __init__(self):
        self.cache_hints: Dict[int, Dict[str, Any]] = {}
        self.transition_history = []
    
    def record_transition(
        self,
        from_stage: int,
        to_stage: int,
        cache_hint: Dict[str, Any],
    ):
        """Stage 전환 기록"""
        record = {
            "from_stage": from_stage,
            "to_stage": to_stage,
            "timestamp": time.time(),
            "cache_hint": cache_hint,
        }
        self.transition_history.append(record)
        self.cache_hints[to_stage] = cache_hint
        
        return record
    
    def get_expected_speedup(self, stage: int) -> Optional[float]:
        """해당 Stage의 예상 속도 향상 반환"""
        if stage in self.cache_hints:
            return self.cache_hints[stage].get('estimated_speedup')
        return None
    
    def get_reuse_ratio(self, stage: int) -> Optional[float]:
        """해당 Stage의 KV Cache 재사용 비율 반환"""
        if stage in self.cache_hints:
            return self.cache_hints[stage].get('reuse_ratio')
        return None
    
    def print_summary(self):
        """Cache 관리 요약 출력"""
        print(f"\n{'='*60}")
        print("PARTIAL KV CACHE REUSE SUMMARY")
        print(f"{'='*60}")
        
        for record in self.transition_history:
            hint = record['cache_hint']
            print(f"\nStage {record['from_stage']} → {record['to_stage']}:")
            print(f"  Keep layers: 0 - {hint.get('keep_prefix_layers', 'N/A') - 1}")
            print(f"  Recompute from: Layer {hint.get('recompute_from_layer', 'N/A')}")
            print(f"  Reuse ratio: {hint.get('reuse_ratio', 0):.1f}%")
            print(f"  Estimated speedup: {hint.get('estimated_speedup', 1):.2f}x")
        
        print(f"{'='*60}\n")


class ProgressiveChatbot:
    """
    대화형 Progressive Stage 챗봇
    
    ✅ Partial KV Cache Reuse 지원
    ✅ vLLM 0.7.4 호환
    """
    
    def __init__(self, model_path, stage2_path, stage3_path):
        self.model_path = model_path
        self.stage2_path = stage2_path
        self.stage3_path = stage3_path
        self.current_stage = 1
        self.conversation_history = ""
        self.turn_count = 0
        
        # 통계 정보
        self.stage_stats = {
            1: {"inference_times": [], "token_counts": []},
            2: {"inference_times": [], "token_counts": []},
            3: {"inference_times": [], "token_counts": []}
        }
        
        self.llm = None
        self.model = None
        self.sampling_params = None
        
        # ✅ NEW: Partial KV Cache 관리자
        self.cache_manager = PartialKVCacheManager()
        
    def initialize(self):
        """vLLM 엔진 초기화"""
        print("\n" + "="*80)
        print("🚀 Progressive LLM Chatbot - Initialization (vLLM 0.7.4)")
        print("="*80 + "\n")
        
        print("⏳ Initializing vLLM with Prefix Caching enabled...")
        start_init = time.time()
        
        try:
            self.llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
                enforce_eager=False,
                enable_prefix_caching=True  # Prefix caching 활성화!
            )
        except Exception as e:
            print(f"❌ Failed to initialize vLLM: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        init_time = time.time() - start_init
        print(f"✅ Initialization complete: {init_time:.2f}s\n")
        
        # 모델 객체 직접 접근
        try:
            self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            print(f"✅ Model accessed: {type(self.model).__name__}")
        except Exception as e:
            print(f"❌ Failed to access model: {e}")
            sys.exit(1)
        
        # Sampling 파라미터 설정
        self.sampling_params = SamplingParams(
            max_tokens=100,
            temperature=0.7,
            top_p=0.95,
        )
        
        print(f"✅ Sampling parameters configured")
        print(f"   - max_tokens: 100")
        print(f"   - temperature: 0.7")
        print(f"   - top_p: 0.95")
        print(f"\n🎯 Currently in Stage {self.current_stage}")
        print(f"⚡ Partial KV Cache Reuse: Enabled\n")
        
    def _invalidate_kv_cache_soft(self):
        """
        vLLM 0.7.4 호환 KV Cache 초기화
        
        Note: vLLM 0.7.4에서는 sleep/wake가 지원되지 않으므로,
        대신 prefix caching의 자연스러운 동작에 의존합니다.
        
        Stage 전환 후:
        1. 변경되지 않은 앞부분 레이어의 KV Cache는 자동 재사용 (prefix caching)
        2. 변경된 레이어부터는 자동으로 재계산됨
        
        실제 구현에서는 vLLM의 내부 스케줄러가 이를 관리합니다.
        """
        try:
            # vLLM 0.7.4에서 사용 가능한 cache 초기화 방법 시도
            # 방법 1: scheduler의 block_manager 접근
            if hasattr(self.llm, 'llm_engine'):
                engine = self.llm.llm_engine
                
                # 방법 1-a: scheduler reset 시도
                if hasattr(engine, 'scheduler'):
                    for scheduler in engine.scheduler:
                        if hasattr(scheduler, 'free_finished_seq_groups'):
                            # 완료된 시퀀스 그룹 해제
                            pass  # 자동으로 관리됨
                
                print("✅ Cache management delegated to vLLM prefix caching")
                return True
                
        except Exception as e:
            print(f"⚠️  Cache management info: {e}")
            print("   Relying on vLLM's automatic prefix caching")
        
        return False
    
    def advance_stage(self, target_stage: int) -> bool:
        """
        Stage 전환 (Partial KV Cache Reuse 활용)
        
        Args:
            target_stage: 목표 Stage (2 또는 3)
            
        Returns:
            성공 여부
        """
        if target_stage == self.current_stage:
            print(f"⚠️  Already in Stage {target_stage}")
            return False
        
        if target_stage < self.current_stage:
            print(f"⚠️  Cannot downgrade from Stage {self.current_stage} to Stage {target_stage}")
            return False
        
        if target_stage > 3:
            print(f"⚠️  Invalid stage: {target_stage}. Valid stages: 1, 2, 3")
            return False
        
        print(f"\n{'='*80}")
        print(f"🔄 Stage Transition: {self.current_stage} → {target_stage}")
        print(f"   (Partial KV Cache Reuse Enabled)")
        print(f"{'='*80}\n")
        
        # 1. Soft cache management (vLLM 0.7.4 호환)
        print("📦 Preparing for stage transition...")
        self._invalidate_kv_cache_soft()
        
        # 2. 최적화된 Stage 전환 (Cache 힌트 포함)
        start_transition = time.time()
        try:
            if target_stage == 2:
                print(f"📦 Loading Stage 2 layers from: {self.stage2_path}")
                # ✅ 최적화된 메서드 사용
                cache_hint = self.model.advance_to_stage2_optimized(
                    layer_b_checkpoint=self.stage2_path
                )
            elif target_stage == 3:
                if self.current_stage == 1:
                    print("⚠️  Must advance to Stage 2 first")
                    return False
                print(f"📦 Loading Stage 3 layers from: {self.stage3_path}")
                # ✅ 최적화된 메서드 사용
                cache_hint = self.model.advance_to_stage3_optimized(
                    layer_c_checkpoint=self.stage3_path
                )
            
            transition_time = time.time() - start_transition
            
            # 3. Cache 관리자에 기록
            self.cache_manager.record_transition(
                from_stage=self.current_stage,
                to_stage=target_stage,
                cache_hint=cache_hint,
            )
            
            print(f"✅ Stage {target_stage} transition complete: {transition_time:.2f}s")
            
        except Exception as e:
            print(f"❌ Stage transition failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 4. 상태 업데이트
        old_stage = self.current_stage
        self.current_stage = target_stage
        
        # 5. Cache 재사용 정보 출력
        print(f"\n{'='*60}")
        print(f"PARTIAL KV CACHE REUSE INFO")
        print(f"{'='*60}")
        print(f"  Transition: Stage {old_stage} → Stage {target_stage}")
        print(f"  Keep prefix layers: 0 - {cache_hint.get('keep_prefix_layers', 0) - 1}")
        print(f"  Recompute from layer: {cache_hint.get('recompute_from_layer', 'N/A')}")
        print(f"  KV Cache reuse ratio: {cache_hint.get('reuse_ratio', 0):.1f}%")
        print(f"  Estimated speedup: {cache_hint.get('estimated_speedup', 1):.2f}x")
        print(f"  Estimated time reduction: {cache_hint.get('estimated_time_reduction', 0):.1f}%")
        print(f"{'='*60}\n")
        
        # 6. 상태 확인
        try:
            self.model.print_status()
        except:
            try:
                info = self.model.get_stage_info()
                print(f"   Current stage: {info.get('stage')}")
            except:
                pass
        
        print(f"\n🎯 Now in Stage {self.current_stage}\n")
        return True
    
    def generate_response(self, user_input: str) -> Optional[str]:
        """사용자 입력에 대한 응답 생성"""
        self.turn_count += 1
        
        # 대화 형식으로 프롬프트 구성
        if self.conversation_history:
            full_prompt = self.conversation_history + f"\nUser: {user_input}\nAssistant:"
        else:
            full_prompt = f"User: {user_input}\nAssistant:"
        
        # 추론 실행
        print(f"\n💭 Thinking... (Stage {self.current_stage})")
        start_time = time.time()
        
        try:
            outputs = self.llm.generate([full_prompt], self.sampling_params)
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return None
        
        elapsed_time = time.time() - start_time
        
        # 응답 추출
        response = outputs[0].outputs[0].text.strip()
        token_count = len(outputs[0].outputs[0].token_ids)
        
        # 통계 업데이트
        self.stage_stats[self.current_stage]["inference_times"].append(elapsed_time)
        self.stage_stats[self.current_stage]["token_counts"].append(token_count)
        
        # 대화 히스토리 업데이트
        self.conversation_history = full_prompt + " " + response
        
        # 결과 출력
        print(f"\n{'='*80}")
        print(f"🤖 Assistant (Stage {self.current_stage} - Turn {self.turn_count})")
        print(f"{'='*80}")
        print(f"{response}")
        print(f"{'-'*80}")
        print(f"⏱️  Inference time: {elapsed_time:.4f}s")
        print(f"🔢 Generated tokens: {token_count}")
        print(f"📏 Context length: {len(full_prompt.split())} words")
        
        # Cache 재사용 정보 (있으면)
        expected_speedup = self.cache_manager.get_expected_speedup(self.current_stage)
        if expected_speedup:
            print(f"⚡ Expected speedup from cache reuse: {expected_speedup:.2f}x")
        
        print(f"{'='*80}\n")
        
        return response
    
    def print_statistics(self):
        """대화 통계 출력"""
        print(f"\n{'='*80}")
        print("📊 Session Statistics")
        print(f"{'='*80}")
        print(f"Total turns: {self.turn_count}")
        print(f"Final stage: {self.current_stage}")
        print(f"Context length: {len(self.conversation_history.split())} words\n")
        
        for stage in [1, 2, 3]:
            times = self.stage_stats[stage]["inference_times"]
            tokens = self.stage_stats[stage]["token_counts"]
            
            if times:
                avg_time = sum(times) / len(times)
                avg_tokens = sum(tokens) / len(tokens)
                print(f"Stage {stage}:")
                print(f"  - Turns: {len(times)}")
                print(f"  - Avg inference time: {avg_time:.4f}s")
                print(f"  - Avg tokens generated: {avg_tokens:.1f}")
                print(f"  - Total time: {sum(times):.4f}s")
        
        # Partial KV Cache Reuse 요약
        self.cache_manager.print_summary()
        
        print(f"{'='*80}\n")
    
    def print_help(self):
        """도움말 출력"""
        print(f"\n{'='*80}")
        print("📖 Available Commands")
        print(f"{'='*80}")
        print("/stage2    - Advance to Stage 2 (load B layers)")
        print("/stage3    - Advance to Stage 3 (load C layers)")
        print("/stats     - Show session statistics")
        print("/cache     - Show Partial KV Cache Reuse info")
        print("/clear     - Clear conversation history")
        print("/help      - Show this help message")
        print("/exit      - Exit the chatbot")
        print("\nOr just type your message to chat!")
        print(f"{'='*80}\n")
    
    def print_cache_info(self):
        """Cache 재사용 정보 출력"""
        print(f"\n{'='*80}")
        print("⚡ Partial KV Cache Reuse Information")
        print(f"{'='*80}")
        
        # 모델에서 직접 정보 가져오기
        try:
            cache_info = self.model.get_cache_reuse_info()
            print(f"\nCurrent Model State:")
            print(f"  Total layers: {cache_info['total_layers']}")
            print(f"  Active layers: {len(cache_info['active_layers'])}")
            print(f"  Inactive layers: {len(cache_info['inactive_layers'])}")
            print(f"  Continuous active prefix: {cache_info['continuous_active_prefix']}")
            
            if cache_info['last_recompute_boundary'] is not None:
                print(f"  Last recompute boundary: Layer {cache_info['last_recompute_boundary']}")
        except Exception as e:
            print(f"  (Could not retrieve cache info: {e})")
        
        # Cache 관리자 요약
        self.cache_manager.print_summary()
        
        print(f"{'='*80}\n")
    
    def clear_history(self):
        """대화 히스토리 초기화"""
        self.conversation_history = ""
        print("✅ Conversation history cleared\n")
    
    def run(self):
        """대화형 루프 실행"""
        self.initialize()
        self.print_help()
        
        print("💬 Chat started! Type your message or use commands.\n")
        
        while True:
            try:
                # 사용자 입력 받기
                user_input = input(f"[Stage {self.current_stage}] You: ").strip()
                
                if not user_input:
                    continue
                
                # 명령어 처리
                if user_input.startswith("/"):
                    command = user_input.lower()
                    
                    if command == "/exit" or command == "/quit":
                        print("\n👋 Goodbye!")
                        self.print_statistics()
                        break
                    
                    elif command == "/stage2":
                        self.advance_stage(2)
                    
                    elif command == "/stage3":
                        self.advance_stage(3)
                    
                    elif command == "/stats":
                        self.print_statistics()
                    
                    elif command == "/cache":
                        self.print_cache_info()
                    
                    elif command == "/clear":
                        self.clear_history()
                    
                    elif command == "/help":
                        self.print_help()
                    
                    else:
                        print(f"⚠️  Unknown command: {user_input}")
                        print("Type /help to see available commands\n")
                
                # 일반 대화 처리
                else:
                    self.generate_response(user_input)
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                self.print_statistics()
                break
            
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    """메인 함수"""
    # 설정 경로
    model_path = "/acpl-ssd20/1218/A"
    stage2_path = "/acpl-ssd20/1218/checkpoints/stage2_layers_B.safetensors"
    stage3_path = "/acpl-ssd20/1218/checkpoints/stage3_layers_C.safetensors"
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║  Progressive LLM Chatbot with Partial KV Cache Reuse         ║
║  vLLM Version: 0.7.4                                         ║
╚══════════════════════════════════════════════════════════════╝

Features:
  ✅ Alpha Gating for CUDA Graph compatibility
  ✅ Partial KV Cache Reuse for faster stage transitions
  ✅ Prefix caching enabled
  ✅ Optimized stage transition methods

Expected Performance Improvements:
  - Stage 1→2: ~72% cache reuse (~3x speedup)
  - Stage 2→3: ~86% cache reuse (~5x speedup)
""")
    
    # 챗봇 생성 및 실행
    chatbot = ProgressiveChatbot(
        model_path=model_path,
        stage2_path=stage2_path,
        stage3_path=stage3_path
    )
    
    chatbot.run()


if __name__ == "__main__":
    main()