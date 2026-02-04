#!/usr/bin/env python3
"""
실행: python chatbot.py
대화형 Progressive Stage 테스트 (Interactive Version)
파일명: chatbot.py

수정 사항:
1. 모델 레지스트리 등록 추가 (vLLM 로딩 필수 단계)
2. Python Path 설정 추가 (커스텀 모듈 참조용)
3. Prefix Caching 활성화로 대화 맥락 유지
4. Stage 전환 시 KV Cache 초기화
5. 사용자 입력 기반 대화형 인터페이스
6. 사용자 명령으로 Stage 전환 제어
"""

import sys
import os
import time
import torch

# [필수] Python path 설정 - 커스텀 모델 경로를 인식하게 합니다.
sys.path.insert(0, "/workspace/vllm_test")
sys.path.insert(0, "/acpl-ssd30/7b_results/pruning/A")
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


class ProgressiveChatbot:
    """대화형 Progressive Stage 챗봇"""
    
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
        
    def initialize(self):
        """vLLM 엔진 초기화"""
        print("\n" + "="*80)
        print("🚀 Progressive LLM Chatbot - Initialization")
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
        print(f"\n🎯 Currently in Stage {self.current_stage}\n")
        
    def advance_stage(self, target_stage):
        """Stage 전환 (KV Cache 초기화 포함)"""
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
        print(f"{'='*80}\n")
        
        # 1. KV Cache 초기화
        print("🧹 Clearing KV Cache...")
        try:
            self.llm.sleep(level=1)
            print("✅ KV Cache cleared")
        except Exception as e:
            print(f"⚠️  Cache clear warning: {e}")
        
        # 2. Stage 전환
        start_transition = time.time()
        try:
            if target_stage == 2:
                print(f"📦 Loading Stage 2 layers from: {self.stage2_path}")
                self.model.advance_to_stage2(layer_b_checkpoint=self.stage2_path)
            elif target_stage == 3:
                if self.current_stage == 1:
                    print("⚠️  Must advance to Stage 2 first")
                    return False
                print(f"📦 Loading Stage 3 layers from: {self.stage3_path}")
                self.model.advance_to_stage3(layer_c_checkpoint=self.stage3_path)
            
            transition_time = time.time() - start_transition
            print(f"✅ Stage {target_stage} transition complete: {transition_time:.2f}s")
        except Exception as e:
            print(f"❌ Stage transition failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 3. 엔진 재활성화
        try:
            self.llm.wake_up()
            print("✅ Engine reactivated")
        except Exception as e:
            print(f"⚠️  Wake up warning: {e}")
        
        self.current_stage = target_stage
        
        # 4. 상태 확인
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
    
    def generate_response(self, user_input):
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
        
        print(f"{'='*80}\n")
    
    def print_help(self):
        """도움말 출력"""
        print(f"\n{'='*80}")
        print("📖 Available Commands")
        print(f"{'='*80}")
        print("/stage2    - Advance to Stage 2 (load B layers)")
        print("/stage3    - Advance to Stage 3 (load C layers)")
        print("/stats     - Show session statistics")
        print("/clear     - Clear conversation history")
        print("/help      - Show this help message")
        print("/exit      - Exit the chatbot")
        print("\nOr just type your message to chat!")
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
    # 설정 경로
    model_path = "/acpl-ssd30/7b_results/pruning/A"
    stage2_path = "/acpl-ssd30/7b_results/pruning/checkpoints/stage2_layers_B.safetensors"
    stage3_path = "/acpl-ssd30/7b_results/pruning/checkpoints/stage3_layers_C.safetensors"
    
    # 챗봇 생성 및 실행
    chatbot = ProgressiveChatbot(
        model_path=model_path,
        stage2_path=stage2_path,
        stage3_path=stage3_path
    )
    
    chatbot.run()


if __name__ == "__main__":
    main()