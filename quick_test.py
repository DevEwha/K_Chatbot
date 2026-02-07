from vllm import LLM, SamplingParams

print("Initializing LLM...")
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    gpu_memory_utilization=0.5,
    enable_prefix_caching=True,
    prefix_cache_num_layers=15,
)

sampling_params = SamplingParams(temperature=0, max_tokens=20)

print("\n" + "="*60)
print("Test 1: First request (no cache)")
print("="*60)
outputs = llm.generate(["[INST] What is 2+2? [/INST]"], sampling_params)
print(f"Response: {outputs[0].outputs[0].text}")

print("\n" + "="*60)
print("Test 2: Second request (expect cache hit + hidden state caching)")
print("="*60)
outputs = llm.generate(["[INST] What is 2+2? [/INST] 4 [INST] What is 3+3? [/INST]"], sampling_params)
print(f"Response: {outputs[0].outputs[0].text}")

print("\n" + "="*60)
print("Test complete!")
print("="*60)
