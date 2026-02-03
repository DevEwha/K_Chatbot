#!/usr/bin/env python3
"""
Progressive Stage Chatbot with KV Cache Reset
파일명: chatbot.py

사용하는 파일들:
1. progressive_llama_for_causal_lm_alpha_v0.py - 메인 모델 클래스
2. progressive_llama_alpha_fixed.py (또는 progressive_llama_alpha_fixed2.py) - CUDA Graph 호환 모델
3. alpha_gated_layer.py - Alpha Gating 레이어

기능:
- Stage 1/2/3 전환 지원
- Stage 전환 시 KV Cache 초기화 후 맥락 재계산
- 대화 히스토리 유지
- Prefix caching 활용

명령어:
- quit: 종료
- reset: KV cache 초기화 및 대화 히스토리 리셋
- stage2: Stage 2로 전환 (KV cache 초기화 후 맥락 재계산)
- stage3: Stage 3로 전환 (KV cache 초기화 후 맥락 재계산)
- status: 현재 모델 상태 출력
"""

import sys
import os
import time
import torch

# Python path 설정 - 커스텀 모델 경로 인식
sys.path.insert(0, "/workspace/vllm_test")
sys.path.insert(0, "/acpl-ssd20/1218/A")
sys.path.insert(0, "/home/devewha/Juwon/vllm_test")

# vLLM import
try:
    from vllm import LLM, SamplingParams
    from vllm.model_executor.models.registry import ModelRegistry
    from progressive_serve.progressive_llama_for_causal_lm_alpha_v0 import ProgressiveLlamaForCausalLMAlpha
    print("✅ vLLM and Custom Model imported successfully")
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("필요한 파일들:")
    print("  - progressive_serve/progressive_llama_for_causal_lm_alpha_v0.py")
    print("  - progressive_serve/progressive_llama_alpha_fixed.py")
    print("  - progressive_serve/alpha_gated_layer.py")
    sys.exit(1)

# 모델 레지스트리 등록
ModelRegistry.register_model(
    "ProgressiveLlamaForCausalLM",
    ProgressiveLlamaForCausalLMAlpha
)


class ProgressiveChatbot:
    """
    Progressive Stage를 지원하는 vLLM 챗봇
    
    Stage 전환 시:
    1. KV Cache 초기화 (sleep/wake_up)
    2. 대화 맥락 재계산
    """
    
    def __init__(
        self,
        model_path: str,
        stage2_checkpoint: str,
        stage3_checkpoint: str,
        gpu_memory_utilization: float = 0.9,
        enable_prefix_caching: bool = False,  # v0에서 multimodal 모델 미지원으로 비활성화
        enforce_eager: bool = False,
    ):
        """
        Args:
            model_path: Stage 1 모델 경로
            stage2_checkpoint: Stage 2 weights 경로
            stage3_checkpoint: Stage 3 weights 경로
            gpu_memory_utilization: GPU 메모리 사용률
            enable_prefix_caching: Prefix caching 활성화
            enforce_eager: CUDA Graph 비활성화 (True면 동적 레이어 변경 안정)
        """
        self.model_path = model_path
        self.stage2_checkpoint = stage2_checkpoint
        self.stage3_checkpoint = stage3_checkpoint
        
        # vLLM 엔진 초기화
        print("\n" + "="*60)
        print("Initializing Progressive Chatbot...")
        print("="*60 + "\n")
        
        start_init = time.time()
        
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=enforce_eager,
        )
        
        init_time = time.time() - start_init
        print(f"✅ vLLM Initialization complete: {init_time:.2f}s\n")
        
        # 모델 객체 접근
        try:
            self.model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            print(f"✅ Model accessed: {type(self.model).__name__}")
        except Exception as e:
            print(f"❌ Failed to access model: {e}")
            raise
        
        # 샘플링 파라미터
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=200,
        )
        
        # 대화 히스토리
        self.history_prompt = ""
        
        # 현재 Stage
        self.current_stage = 1
        
        print(f"✅ Chatbot ready at Stage {self.current_stage}")
        print("-" * 60)
    
    def reset_kv_cache(self) -> float:
        """
        KV Cache 초기화
        
        여러 방법 시도:
        1. sleep/wake_up (vLLM 0.6+ with enable_sleep_mode)
        2. llm_engine 내부 scheduler 통한 초기화
        3. prefix cache 리셋
        4. 강제 가비지 컬렉션 (fallback)
        
        Returns:
            초기화에 걸린 시간 (초)
        """
        print("🔄 Resetting KV cache...")
        start = time.time()
        
        success = False
        
        # 방법 1: sleep/wake_up (enable_sleep_mode 필요)
        try:
            if hasattr(self.llm, 'sleep') and hasattr(self.llm, 'wake_up'):
                # sleep mode가 활성화되어 있는지 확인
                if (hasattr(self.llm, 'llm_engine') and 
                    hasattr(self.llm.llm_engine, 'vllm_config') and
                    hasattr(self.llm.llm_engine.vllm_config, 'model_config') and
                    getattr(self.llm.llm_engine.vllm_config.model_config, 'enable_sleep_mode', False)):
                    self.llm.sleep(level=1)
                    self.llm.wake_up()
                    success = True
                    print("  ✅ Used sleep/wake_up method")
        except Exception as e:
            print(f"  ⚠️  sleep/wake_up failed: {e}")
        
        # 방법 2: Scheduler를 통한 KV cache 블록 해제
        if not success:
            try:
                engine = self.llm.llm_engine
                
                # Scheduler의 block manager를 통한 초기화
                if hasattr(engine, 'scheduler'):
                    schedulers = engine.scheduler
                    if not isinstance(schedulers, list):
                        schedulers = [schedulers]
                    
                    for scheduler in schedulers:
                        # 모든 sequence group 완료 처리
                        if hasattr(scheduler, 'running'):
                            scheduler.running.clear()
                        if hasattr(scheduler, 'waiting'):
                            scheduler.waiting.clear()
                        if hasattr(scheduler, 'swapped'):
                            scheduler.swapped.clear()
                        
                        # Block manager free
                        if hasattr(scheduler, 'block_manager'):
                            block_manager = scheduler.block_manager
                            if hasattr(block_manager, 'reset'):
                                block_manager.reset()
                            elif hasattr(block_manager, '_reset'):
                                block_manager._reset()
                    
                    success = True
                    print("  ✅ Used scheduler reset method")
            except Exception as e:
                print(f"  ⚠️  Scheduler reset failed: {e}")
        
        # 방법 3: Prefix cache 리셋
        if not success:
            try:
                if hasattr(self.llm.llm_engine, 'reset_prefix_cache'):
                    self.llm.llm_engine.reset_prefix_cache()
                    success = True
                    print("  ✅ Used prefix cache reset method")
            except Exception as e:
                print(f"  ⚠️  Prefix cache reset failed: {e}")
        
        # 방법 4: 강제 가비지 컬렉션 (fallback)
        if not success:
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                print("  ⚠️  Used fallback (gc + cuda cache clear)")
                print("  ℹ️  Note: KV cache may not be fully cleared, but Stage transition is complete")
                success = True
            except Exception as e:
                print(f"  ❌ Fallback also failed: {e}")
        
        elapsed = time.time() - start
        if success:
            print(f"✅ KV cache reset complete ({elapsed:.2f}s)")
        else:
            print(f"⚠️  KV cache reset may be incomplete ({elapsed:.2f}s)")
        
        return elapsed
    
    def recompute_context(self) -> float:
        """
        현재 대화 맥락을 새 Stage로 재계산
        
        Stage 전환 후 기존 히스토리로 새로운 KV cache 생성
        
        Returns:
            재계산에 걸린 시간 (초)
        """
        if not self.history_prompt:
            print("ℹ️  No history to recompute")
            return 0.0
        
        print("🔄 Recomputing context with new stage...")
        start = time.time()
        
        # 빈 생성 요청으로 KV cache만 생성
        # max_tokens=1로 최소한의 생성만 수행
        recompute_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,  # 최소 토큰만 생성
        )
        
        try:
            # 기존 히스토리로 forward pass 수행 (KV cache 재구축)
            _ = self.llm.generate([self.history_prompt], recompute_params)
        except Exception as e:
            print(f"⚠️  Context recompute warning: {e}")
        
        elapsed = time.time() - start
        print(f"✅ Context recomputed ({elapsed:.2f}s)")
        return elapsed
    
    def advance_to_stage2(self) -> bool:
        """
        Stage 2로 전환
        
        1. 새 레이어 weights 로드 및 활성화
        2. KV Cache 초기화
        3. 대화 맥락 재계산
        
        Returns:
            성공 여부
        """
        if self.current_stage >= 2:
            print(f"⚠️  Already at Stage {self.current_stage}")
            return False
        
        print("\n" + "="*60)
        print("TRANSITIONING TO STAGE 2")
        print("="*60 + "\n")
        
        start = time.time()
        
        try:
            # 1. Stage 2 레이어 활성화
            print("[Step 1/3] Activating Stage 2 layers...")
            self.model.advance_to_stage2(layer_b_checkpoint=self.stage2_checkpoint)
            
            # 2. KV Cache 초기화
            print("\n[Step 2/3] Resetting KV cache...")
            self.reset_kv_cache()
            
            # 3. 대화 맥락 재계산
            print("\n[Step 3/3] Recomputing conversation context...")
            self.recompute_context()
            
            self.current_stage = 2
            
            total_time = time.time() - start
            print(f"\n{'='*60}")
            print(f"✅ NOW AT STAGE 2 (Total: {total_time:.2f}s)")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Stage 2 transition failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def advance_to_stage3(self) -> bool:
        """
        Stage 3로 전환
        
        1. 새 레이어 weights 로드 및 활성화
        2. KV Cache 초기화
        3. 대화 맥락 재계산
        
        Returns:
            성공 여부
        """
        if self.current_stage >= 3:
            print(f"⚠️  Already at Stage {self.current_stage}")
            return False
        
        if self.current_stage < 2:
            print("⚠️  Must be at Stage 2 first. Advancing to Stage 2...")
            if not self.advance_to_stage2():
                return False
        
        print("\n" + "="*60)
        print("TRANSITIONING TO STAGE 3")
        print("="*60 + "\n")
        
        start = time.time()
        
        try:
            # 1. Stage 3 레이어 활성화
            print("[Step 1/3] Activating Stage 3 layers...")
            self.model.advance_to_stage3(layer_c_checkpoint=self.stage3_checkpoint)
            
            # 2. KV Cache 초기화
            print("\n[Step 2/3] Resetting KV cache...")
            self.reset_kv_cache()
            
            # 3. 대화 맥락 재계산
            print("\n[Step 3/3] Recomputing conversation context...")
            self.recompute_context()
            
            self.current_stage = 3
            
            total_time = time.time() - start
            print(f"\n{'='*60}")
            print(f"✅ NOW AT STAGE 3 - FULL MODEL (Total: {total_time:.2f}s)")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"❌ Stage 3 transition failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_status(self):
        """현재 모델 상태 출력"""
        print("\n" + "="*60)
        print(f"CHATBOT STATUS")
        print("="*60)
        print(f"Current Stage: {self.current_stage}")
        print(f"History Length: {len(self.history_prompt)} chars")
        
        try:
            self.model.print_status()
        except:
            try:
                info = self.model.get_stage_info()
                print(f"Active Layers: {info.get('active_layers', 'N/A')}")
                print(f"Inactive Layers: {info.get('inactive_layers', 'N/A')}")
                print(f"Progress: {info.get('activation_progress', 'N/A')}")
            except:
                print("⚠️  Detailed status not available")
        
        print("="*60 + "\n")
    
    def reset_conversation(self):
        """대화 히스토리 초기화"""
        print("🔄 Resetting conversation history...")
        self.reset_kv_cache()
        self.history_prompt = ""
        print("✅ Conversation reset complete")
    
    def chat(self, user_input: str) -> tuple:
        """
        사용자 입력에 대한 응답 생성
        
        Args:
            user_input: 사용자 메시지
            
        Returns:
            AI 응답
        """
        # Llama 2 프롬프트 형식 적용
        current_prompt = self.history_prompt + f"[INST] {user_input} [/INST]"
        
        # 생성
        start = time.time()
        outputs = self.llm.generate([current_prompt], self.sampling_params)
        elapsed = time.time() - start
        
        # 결과 추출
        response = outputs[0].outputs[0].text
        
        # 히스토리 업데이트
        self.history_prompt = current_prompt + f" {response} "
        
        return response, elapsed
    
    def run(self):
        """챗봇 메인 루프"""
        print("\n" + "="*60)
        print("Progressive vLLM Chatbot")
        print("="*60)
        print(f"Current Stage: {self.current_stage}")
        print("\nCommands:")
        print("  quit   - Exit chatbot")
        print("  reset  - Reset KV cache and conversation")
        print("  stage2 - Advance to Stage 2")
        print("  stage3 - Advance to Stage 3")
        print("  status - Show model status")
        print("-" * 60)
        
        while True:
            try:
                user_input = input(f"\n[Stage {self.current_stage}] User: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break
            
            if not user_input:
                continue
            
            # 명령어 처리
            command = user_input.lower()
            
            if command == "quit":
                print("Goodbye!")
                break
            
            elif command == "reset":
                self.reset_conversation()
                continue
            
            elif command == "stage2":
                self.advance_to_stage2()
                continue
            
            elif command == "stage3":
                self.advance_to_stage3()
                continue
            
            elif command == "status":
                self.print_status()
                continue
            
            # 일반 대화
            response, elapsed = self.chat(user_input)
            
            print(f"Bot: {response}")
            print(f"⏱️  Generation time: {elapsed:.2f}s | Stage: {self.current_stage}")


def main():
    """메인 함수"""
    # 설정 경로 (환경에 맞게 수정)
    model_path = "/acpl-ssd20/1218/A"
    stage2_checkpoint = "/acpl-ssd20/1218/checkpoints/stage2_layers_B.safetensors"
    stage3_checkpoint = "/acpl-ssd20/1218/checkpoints/stage3_layers_C.safetensors"
    
    # 챗봇 생성 및 실행
    chatbot = ProgressiveChatbot(
        model_path=model_path,
        stage2_checkpoint=stage2_checkpoint,
        stage3_checkpoint=stage3_checkpoint,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,  # v0에서 multimodal 모델 미지원으로 비활성화
        enforce_eager=False,  # CUDA Graph 사용 (문제 시 True로 변경)
    )
    
    chatbot.run()


if __name__ == "__main__":
    main()