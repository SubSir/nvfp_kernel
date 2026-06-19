# nvfp_kernel — residual (two-level) NVFP4 GEMM

NVFP4 (W4A4) kernels extracted from vLLM, extended with a **fused residual
quantization** path that trades spare compute for activation precision in the
memory-bound LLM decode regime.

## Idea

LLM decode is memory-bound: the weight `W` is streamed once and dominates HBM
traffic. The matmul therefore has spare compute. We spend it on the *activation*
to recover precision lost by 4-bit quantization:

1. Quantize `x` to NVFP4 → `Q(x)` (level 1), keeping its global scale.
2. Dequantize and form the residual `r = x - dequant(Q(x))`.
3. Quantize `r` to a **second** NVFP4 level `Q(r)`, sharing the *same* global
   scale as `x`.
4. Stack along the token dimension: `A = [Q(x); Q(r)]` of shape `(2M, K)`, run a
   single `(2M × N)` FP4 GEMM against the single-level weight, then sum the two
   output halves:

   ```
   out = Q(x) @ W.T + Q(r) @ W.T  ≈  x @ W.T
   ```

Because the two levels share one global scale, the whole thing is **one** FP4
GEMM with a single `alpha` — no extra weight traffic. The activation goes from
~fp4 to ~fp7/8 effective precision.

Steps 1–3 are **fused into one CUDA kernel** (`cvt_fp16_to_fp4_residual`): it
reads `x` once, emits level-1 codes, decodes them in-register to compute the
residual, and emits level-2 codes into the stacked output — all in a single
pass.

## Results (B200 / sm100, MiniMax-M3 shapes)

MiniMax-M3 text config: hidden 6144, q=8192 (64×128) + kv=512 (4×128), MoE
expert intermediate 3072.

**Accuracy** (rel error vs fp32 reference):

| metric | single-level fp4 | residual fp4 | gain |
|---|---|---|---|
| activation reconstruction `‖x − x̂‖` | 0.095 | 0.0087 | **~11×** |
| end-to-end GEMM `‖out − ref‖` | 0.135 | 0.096 | **1.41× (√2)** |

The end-to-end gain is `√2`, not 11×, because residual removes the *activation*
quant error, leaving the single-level *weight* fp4 error as the floor:
`single ≈ √(εA² + εW²)`, `residual ≈ εW`.

**Speed** — pure FP4 GEMM at `M` vs `2M` rows (activations pre-quantized), i.e.
the cost of doubling tokens:

| M (decode) | `gemm(2M) / gemm(M)` |
|---|---|
| 1 – 64 | **≈ 1.0× (free)** |
| 256 | ≈ 1.1 – 1.4× |
| 1024 | ≈ 1.4 – 1.8× (→ compute-bound 2×) |

So in the decode regime (`M ≲ 64`) the second NVFP4 level is **free at the GEMM
level**; the full residual path costs only ~1.12× over single-level fp4 (the
extra residual-quant + output-add).

### FP4 vs bf16 — the crossover

At small decode shapes the FP4 GEMM sits on a ~0.05 ms host launch/overhead floor,
so **measured per-call in isolation** FP4 only beats cuBLAS bf16 once the matmul
is big enough that bf16 exceeds that floor:

| shape | single FP4 ≥ bf16 from (no graph) |
|---|---|
| qkv_proj / o_proj | M ≈ 2048 |
| moe_gate_up | M ≈ 4096 |
| moe_down | M ≈ 8192 |

That floor is launch overhead, not the GEMM. **With CUDA graphs** (how serving
actually runs — launch amortized) the picture flips: FP4 reads 4× fewer weight
bytes than bf16, so it wins in the memory-bound decode regime:

| shape | FP4 vs bf16 @ M=1 | residual vs bf16 @ M=1 |
|---|---|---|
| qkv_proj (6144×9216) | 1.6× faster | 1.3× faster |
| o_proj (8192×6144) | ~par | ~par |
| moe_gate_up / moe_down | ~par (FP4 ahead by M≈16–64) | ~0.8–0.9× |

So for the big projections, residual FP4 is **faster than bf16 while also more
accurate than single-level FP4** — the doubled tokens stay under bf16 latency
because the weight (4× smaller) dominates traffic. See
`tests/bench_crossover.py` (per-call) and `tests/bench_cudagraph.py` (graphed).

## API

```python
from nvfp.fp4_gemm import fp4_gemm, residual_fp4_gemm

out = fp4_gemm(a, b)            # single-level W4A4:  a @ b.T
out = residual_fp4_gemm(a, b)   # two-level activation, single weight level
# a: (M, K) activation, b: (N, K) weight  ->  out: (M, N)
```

Lower-level: `nvfp.ops.scaled_fp4_quant_residual(x, global_scale)` returns the
stacked `(2M, K/2)` FP4 tensor + swizzled e4m3 scales.

## Build

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# B200 / GB100:
NVFP_CUDA_ARCH=100a CUTLASS_DIR=/path/to/cutlass python setup.py install
# RTX 50xx / GB202 (default):
NVFP_CUDA_ARCH=120a CUTLASS_DIR=/path/to/cutlass python setup.py install
```

`setup.py` compiles only the GEMM kernel matching the target arch
(`nvfp4_scaled_mm_sm100_kernels.cu` or `..._sm120_kernels.cu`).

## Run on Modal (B200)

```bash
modal run modal_app.py::baseline    # build sanity + single-level fp4 gemm
modal run modal_app.py::accuracy    # residual vs single-level accuracy
modal run modal_app.py::speed       # MiniMax-M3 shape speed sweep
```

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SubSir/nvfp_kernel)
