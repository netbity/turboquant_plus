# Experiment: Decode Speed Parity

Branch: `experiment/decode-speed-parity`

## Problem
turbo3 decode is 0.84x of q8_0 (65.4 vs 78.3 tok/s at 8K context on M5 Max). External testers report 0.83x (Mario, M1 Max 32K).

## Root Cause Isolation

| Config | Decode tok/s | vs q8_0 |
|--------|-------------|---------|
| turbo3 (full dequant) | 65.4 | 0.84x |
| turbo3 (NO centroid LUT — speed ceiling) | **78.1** | **1.00x** |
| q8_0 | 78.3 | baseline |

**The entire 16% decode gap comes from the centroid table lookup.** Without the LUT (returning a constant), turbo3 matches q8_0 exactly.

82M constant memory accesses per decode token (8K context, 10 attn layers, 256-dim heads).

## Approaches Tested

### Register-Based Centroid Select
Split 8-entry LUT into two 4-entry float4 registers (c_lo for hi1=0, c_hi for hi1=1). Use ternary select instead of array index.

**Result:** 63.1 tok/s — SLOWER than the 65.4 baseline. Variable indexing into float4 registers is equally expensive on Metal as constant memory LUT. The GPU treats both as data-dependent access with potential pipeline stalls.

### Linear Approximation
Fit centroids to `c ≈ a * idx + b`. Max error 27% — unacceptable for quality.

## Remaining Approaches (Not Yet Tested)

### Store Centroid Values Directly
During quantize, store fp16 centroid values instead of 3-bit indices. Dequant becomes simple fp16 → fp32 + norm multiply (no LUT).

**Trade-off:** Block size increases. 32 × fp16 = 64 bytes vs 14 bytes currently. Compression drops from 4.6x to ~2.0x — equivalent to q8_0. This defeats the purpose.

### Fused Compressed Attention
Compute Q·K directly on quantized indices. Precompute `Q·centroid[c]` for 8 centroids, then for each K element just look up `q_dot_centroid[idx] * norm`.

**Trade-off:** Requires custom flash attention kernel variant. Complex but would eliminate dequant entirely for Q·K path. V path still needs dequant.

### Half-Precision Centroid LUT
Store the 8 centroids as half instead of float. Reduces constant memory bandwidth by 2x.

**Not yet tested.** May help on older hardware (M1) where constant memory bandwidth is more constrained.

## Key Finding
The 16% decode gap is a **fundamental cost of data-dependent indexing** in the flash attention hot path. It's not a bug — it's the compute cost of decompressing 3-bit indices at 82M accesses per decode token. The only way to close it is to avoid the indexing entirely (fused attention) or accept less compression (store values directly).

For M1 testers: the 0.83x decode ratio is the expected behavior at the current compression level. It's consistent across M1 Max and M5 Max.
