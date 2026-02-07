from vllm import LLM, SamplingParams
import time

# vLLM 엔진 초기화 with partial prefix caching
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    gpu_memory_utilization=0.5,
    enable_prefix_caching=True,
    prefix_cache_num_layers=15,  # Split at layer 15
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

print("\n" + "=" * 80)
print("Testing Partial Prefix Caching with Detailed Logging")
print("=" * 80)

# Test 1: First request (no cache)
print("\n[Test 1] First request - No cache hit expected")
print("-" * 80)
history = "[INST] Hello [/INST]"
start = time.time()
outputs = llm.generate([history], sampling_params)
response1 = outputs[0].outputs[0].text
print(f"Response: {response1[:100]}...")
print(f"Time: {time.time() - start:.2f}s")

# Test 2: Second request with same prefix (should hit cache)
print("\n[Test 2] Second request - Cache hit expected for prefix")
print("-" * 80)
history = "[INST] Hello [/INST] " + response1 + " [INST] Tell me more [/INST]"
start = time.time()
outputs = llm.generate([history], sampling_params)
response2 = outputs[0].outputs[0].text
print(f"Response: {response2[:100]}...")
print(f"Time: {time.time() - start:.2f}s")

# Test 3: Third request with longer prefix
print("\n[Test 3] Third request - Longer prefix cache hit expected")
print("-" * 80)
history = (
    "[INST] Hello [/INST] " + response1 +
    " [INST] Tell me more [/INST] " + response2 +
    " [INST] Continue [/INST]"
)
start = time.time()
outputs = llm.generate([history], sampling_params)
response3 = outputs[0].outputs[0].text
print(f"Response: {response3[:100]}...")
print(f"Time: {time.time() - start:.2f}s")

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)
print("\nKey logs to look for:")
print("  🔀 Partial prefix caching enabled (during initialization)")
print("  💾 Prefix cache hit (when prefix is reused)")
print("  🚀 Using split execution (when cache hit > 0)")
print("  🔍 Split execution details (token counts)")
print("  💿 Cached hidden states (layer 14 outputs)")
print("=" * 80)
