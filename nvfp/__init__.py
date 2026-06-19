from .ops import scaled_fp4_quant, cutlass_scaled_fp4_mm, reciprocal_approximate_ftz_tensor
from .fp4_gemm import fp4_gemm, quantize_fp4, global_scale

__all__ = [
    "scaled_fp4_quant",
    "cutlass_scaled_fp4_mm",
    "reciprocal_approximate_ftz_tensor",
    "fp4_gemm",
    "quantize_fp4",
    "global_scale",
]

# pseudo_quant pulls a torch internal that needs the optional `expecttest`
# package; keep it optional so the real-kernel path works without it.
try:
    from .pseudo_quant import nvfp4_pseudo_quantize, simple_fp4_pseudo_quantize

    __all__ += ["nvfp4_pseudo_quantize", "simple_fp4_pseudo_quantize"]
except Exception:  # pragma: no cover
    pass
