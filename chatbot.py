from vllm import LLM, SamplingParams

# 1. vLLM 엔진 초기화 (여기서 모델을 로드하고 GPU 메모리를 예약합니다)
# gpu_memory_utilization: GPU 메모리를 얼마나 쓸지 설정 (기본값 0.9)
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf", gpu_memory_utilization=0.9)

# 2. 텍스트 생성 옵션 설정 (온도, 최대 길이 등)
sampling_params = SamplingParams(temperature=0.7, max_tokens=200)

# 3. 대화 문맥을 저장할 변수
history_prompt = ""

print("vLLM 챗봇이 준비되었습니다. (종료: 'quit')")
print("-" * 30)

while True:
    user_input = input("\nUser: ")
    if user_input.lower() == "quit":
        break

    # 4. Llama 2 프롬프트 형식 적용
    # 이전 대화(history_prompt) 뒤에 새 질문을 붙입니다.
    current_prompt = history_prompt + f"[INST] {user_input} [/INST]"

    # 5. vLLM 생성 (리스트 형태로 입력을 받습니다)
    # vLLM은 내부적으로 앞부분(history_prompt)이 이전과 같다는 것을 감지하고
    # 자동으로 KV Cache를 재사용하여 계산 속도를 높입니다.
    outputs = llm.generate([current_prompt], sampling_params)

    # 6. 결과 추출
    # generated_text에는 입력 프롬프트는 제외하고 '새로 생성된 답변'만 들어있습니다.
    response = outputs[0].outputs[0].text
    
    print(f"Bot: {response}")

    # 7. 문맥 업데이트 (답변도 기억에 추가)
    history_prompt = current_prompt + f" {response} "