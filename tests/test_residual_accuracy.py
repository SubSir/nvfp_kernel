"""Accuracy: residual (two-level) NVFP4 vs single-level NVFP4.

Two independent checks:
  (1) Reconstruction: dequantize the fused stacked tensor and compare
      ||x - x_approx|| against the single-level ||x - dequant(NVFP4(x))||.
  (2) End-to-end GEMM: residual_fp4_gemm vs fp4_gemm against an fp32 reference.
"""

import sys

import torch

from nvfp.fp4_gemm import fp4_gemm, residual_fp4_gemm, global_scale, quantize_fp4
from nvfp.ops import scaled_fp4_quant_residual

_E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device="cuda"
)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """(R, K/2) uint8 -> (R, K) fp32 (e2m1 decoded, low nibble first)."""
    R, half = packed.shape
    p = packed.view(-1)
    lo = p & 0x0F
    hi = (p >> 4) & 0x0F

    def dec(n):
        sign = (n & 0x8) != 0
        mag = _E2M1_LUT[(n & 0x7).long()]
        return torch.where(sign, -mag, mag)

    out = torch.stack((dec(lo), dec(hi)), dim=-1).reshape(R, half * 2)
    return out


def unswizzle_128_4(sf_swizzled: torch.Tensor, R: int, k_blocks: int) -> torch.Tensor:
    """Invert the 128x4 swizzle and crop to (R, k_blocks)."""
    mn_pad, k_pad = sf_swizzled.shape
    m_tiles = mn_pad // 128
    k_tiles = k_pad // 4
    tmp = sf_swizzled.reshape(m_tiles, k_tiles, 32, 4, 4)
    lin = tmp.transpose(1, 3).reshape(mn_pad, k_pad)
    return lin[:R, :k_blocks]


def dequant_fp4(packed, sf_swizzled, gs, R, K, block=16):
    vals = unpack_fp4(packed)  # (R, K)
    sf = unswizzle_128_4(sf_swizzled.float(), R, K // block)  # (R, K/16)
    total = sf / gs  # stored sf = block_scale * gs  =>  value scale = sf/gs
    return (vals.reshape(R, K // block, block) * total.unsqueeze(-1)).reshape(R, K)


def rel_err(x, ref):
    return (x.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    ok = True

    print("=== (1) activation reconstruction error (lower = better) ===")
    print(f"{'M':>6} {'K':>6} {'single':>10} {'residual':>10} {'improve':>9}")
    print("-" * 46)
    for (M, K) in [(16, 6144), (64, 6144), (256, 8192), (128, 3072)]:
        x = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.5
        gs = global_scale(x)

        # single-level reconstruction
        q1, sf1 = quantize_fp4(x, gs)[:2]
        recon1 = dequant_fp4(q1, sf1, gs, M, K)

        # two-level (fused kernel) reconstruction
        qr, sfr = scaled_fp4_quant_residual(x, gs)
        deq = dequant_fp4(qr, sfr, gs, 2 * M, K)
        recon2 = deq[:M] + deq[M:]

        e1 = rel_err(recon1, x).item()
        e2 = rel_err(recon2, x).item()
        print(f"{M:>6} {K:>6} {e1:>10.4f} {e2:>10.4f} {e1 / e2:>8.2f}x")
        if not (e2 < e1 * 0.6):  # residual should clearly beat single level
            ok = False

    print("\n=== (2) end-to-end GEMM rel error vs fp32 reference ===")
    print(f"{'M':>6} {'K':>6} {'N':>6} {'single':>10} {'residual':>10} {'improve':>9}")
    print("-" * 56)
    for (M, K, N) in [(1, 6144, 9216), (16, 8192, 6144), (64, 6144, 6144), (256, 3072, 6144)]:
        a = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.5
        b = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.5
        ref = a.float() @ b.float().T

        out_s = fp4_gemm(a, b, out_dtype=torch.bfloat16)
        out_r = residual_fp4_gemm(a, b, out_dtype=torch.bfloat16)
        e1 = rel_err(out_s, ref).item()
        e2 = rel_err(out_r, ref).item()
        print(f"{M:>6} {K:>6} {N:>6} {e1:>10.4f} {e2:>10.4f} {e1 / e2:>8.2f}x")
        # Residual removes the activation quant error, leaving the single-level
        # WEIGHT fp4 error as the floor: single ~= sqrt(eA^2 + eW^2), residual
        # ~= eW, so the expected end-to-end gain is ~sqrt(2) ~= 1.41x.
        if not (e2 < e1 * 0.85):
            ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
