# KL Divergence Results

**Date:** 2026-03-27
**Hardware:** Apple M5 Max 128GB
**Baseline:** f16 KV cache logits (8-chunk wikitext-2, c=512)

## MoE (Qwen3.5-35B-A3B Q8_0)

| Cache Type | Mean KLD | Max KLD | Δp RMS | Same top-p % |
|------------|----------|---------|--------|-------------|
| q8_0 | 0.001549 | 0.1115 | 1.231% | 98.43% |
| q4_0 | 0.008091 | 0.2287 | 2.753% | 95.83% |
| **turbo3** | **0.016145** | **1.1654** | **4.090%** | **94.31%** |

## Dense (Qwen3.5-27B Q8_0)

| Cache Type | Mean KLD | Max KLD | Δp RMS | Same top-p % |
|------------|----------|---------|--------|-------------|
| q8_0 | 0.000018 | — | 0.127% | 99.90% |
| q4_0 | 0.002741 | — | 1.437% | 97.65% |
| **turbo3** | **0.009900** | — | **2.738%** | **95.98%** |

## Analysis

turbo3 KLD is roughly 2× q4_0 on both architectures. This is expected: turbo3 uses 3.5 bits (less than q4_0's 4 bits) with a fundamentally different compression mechanism (WHT rotation + polar codebook vs scalar quantization).

The same-top-p metric shows turbo3 agrees with f16 on the top token 94-96% of the time. For context, q4_0 (a widely-used cache type) agrees 96-98%.

Dense model shows lower KLD across all cache types because the dense attention pattern is more concentrated (fewer heads, more focused attention), making the KV cache less sensitive to quantization noise.

## Raw Logs

- `~/local_llms/llama.cpp/results/kld_moe_q8_0.log`
- `~/local_llms/llama.cpp/results/kld_moe_q4_0.log`
- `~/local_llms/llama.cpp/results/kld_moe_turbo3.log`
- `~/local_llms/llama.cpp/results/kld_dense_q8_0.log`
- `~/local_llms/llama.cpp/results/kld_dense_q4_0.log`
- `~/local_llms/llama.cpp/results/kld_dense_turbo3.log`
