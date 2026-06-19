"""High-level NVFP4 GEMM helpers built on the real CUDA kernels.

Conventions
-----------
* Activation ``a`` has shape ``(M, K)``; weight ``b`` has shape ``(N, K)``.
* The GEMM computes ``out[m, n] = sum_k a[m, k] * b[n, k]`` i.e. ``a @ b.T``.
* NVFP4: 16-element blocks share an fp8(e4m3) scale; a single fp32 ``global
  scale`` per tensor keeps those block scales inside the e4m3 range. The GEMM
  folds the two global scales back in through ``alpha = 1/(a_gs * b_gs)``.
"""

import torch

from .ops import (
    scaled_fp4_quant,
    scaled_fp4_quant_residual,
    cutlass_scaled_fp4_mm,
)

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
# global_scale maps amax -> e4m3 max so block scales use the full e4m3 range.
ALPHA_SCALE = FLOAT4_E2M1_MAX * FLOAT8_E4M3_MAX  # 2688.0


def global_scale(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().max().to(torch.float32)
    return (ALPHA_SCALE / amax).to(torch.float32)


def quantize_fp4(x: torch.Tensor, gs: torch.Tensor | None = None):
    """Return (packed_fp4_uint8 (M, K/2), swizzled e4m3 scales, fp32 global scale)."""
    if gs is None:
        gs = global_scale(x)
    q, sf = scaled_fp4_quant(x, gs)
    return q, sf, gs


def fp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    a_gs: torch.Tensor | None = None,
    b_gs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-level W4A4 GEMM: quantize a and b to NVFP4 and matmul."""
    a_q, a_sf, a_gs = quantize_fp4(a, a_gs)
    b_q, b_sf, b_gs = quantize_fp4(b, b_gs)
    alpha = (1.0 / (a_gs * b_gs)).to(torch.float32)
    return cutlass_scaled_fp4_mm(a_q, b_q, a_sf, b_sf, alpha, out_dtype)


def residual_fp4_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    a_gs: torch.Tensor | None = None,
    b_gs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Residual W4A4 GEMM: doubles activation tokens to recover ~2 NVFP4 levels.

    Quantizes the activation ``a`` to two stacked NVFP4 levels (value + residual)
    sharing one global scale, runs a single (2M x N) FP4 GEMM against the
    single-level weight, then sums the two halves of the output:

        out = NVFP4(a) @ W.T + NVFP4(a - dequant(NVFP4(a))) @ W.T

    Memory traffic for ``W`` is unchanged (it is streamed once), so in the
    memory-bound decode regime the extra GEMM rows are ~free while activation
    precision improves from ~fp4 to ~fp7/8.
    """
    M = a.shape[0]
    # Stacked two-level activation: (2M, K/2) fp4 + (2M-rounded) swizzled scales.
    if a_gs is None:
        a_gs = global_scale(a)
    a_q, a_sf = scaled_fp4_quant_residual(a, a_gs)

    b_q, b_sf, b_gs = quantize_fp4(b, b_gs)
    alpha = (1.0 / (a_gs * b_gs)).to(torch.float32)

    out2 = cutlass_scaled_fp4_mm(a_q, b_q, a_sf, b_sf, alpha, out_dtype)  # (2M, N)
    return out2[:M] + out2[M:]
