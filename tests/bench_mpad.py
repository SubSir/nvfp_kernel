"""Why residual looks "free" at small M -- and where the doubling cost really
appears. Two quantization effects stack on top of the memory-bound argument:

  1. M-padding: the FP4 GEMM pads M to a 128-row CTA tile, so for M<=64 the
     residual's 2M rows still occupy ONE 128-row tile -> identical compute.
  2. SM-wave quantization: the grid is ceil(rows/128) x ceil(N/128) CTAs; while
     the doubled grid still fits one wave (<= #SMs), wall-clock is unchanged.

Only once the doubled grid spills past a wave does 2M/M head toward 2x.
GEMM-only (activations pre-quantized) to isolate tensor-core work.

Dispatch note: rows<=256 use the 1-SM small config (128x128x256); rows>256 use
the 2-SM default config (256x256x256) -- printed per row.
"""

import math

import torch

from nvfp.fp4_gemm import global_scale
from nvfp.ops import scaled_fp4_quant, cutlass_scaled_fp4_mm

SHAPES = [("qkv_proj", 6144, 9216), ("o_proj", 8192, 6144)]
MS = [16, 32, 48, 64, 72, 96, 128, 192, 256, 384, 512]


def bench(fn, inner=50, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(iters):
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / inner)
    return best


def cfg(rows):
    mp2 = max(16, 1 << (rows - 1).bit_length()) if rows > 1 else 16
    return ("small", 128) if mp2 <= 256 else ("default", 256)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"device SMs = {nsm}")

    for (name, K, N) in SHAPES:
        w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.5
        w_gs = global_scale(w)
        w_q, w_sf = scaled_fp4_quant(w, w_gs)
        alpha = (1.0 / (w_gs * w_gs)).to(torch.float32)
        ntiles_n = math.ceil(N / 128)

        print(f"\n## {name}  (K={K}, N={N})  N-tiles={ntiles_n}  GEMM-only")
        print(f"{'M':>5} {'2M':>5} | {'ctaM':>5} {'wvM':>4} {'cfgM':>7} | "
              f"{'cta2M':>6} {'wv2M':>5} {'cfg2M':>7} | "
              f"{'gemm(M)':>9} {'gemm(2M)':>9} {'2M/M':>7}")
        print("-" * 92)
        for M in MS:
            aq_m, asf_m = scaled_fp4_quant(
                torch.randn(M, K, dtype=torch.bfloat16, device=dev), w_gs)
            aq_2m, asf_2m = scaled_fp4_quant(
                torch.randn(2 * M, K, dtype=torch.bfloat16, device=dev), w_gs)

            def gm():
                return cutlass_scaled_fp4_mm(aq_m, w_q, asf_m, w_sf, alpha, torch.bfloat16)

            def g2m():
                return cutlass_scaled_fp4_mm(aq_2m, w_q, asf_2m, w_sf, alpha, torch.bfloat16)

            tm, t2m = bench(gm), bench(g2m)
            cm, tile_m = cfg(M)
            c2, tile_2 = cfg(2 * M)
            cta_m = math.ceil(M / tile_m) * ntiles_n
            cta_2m = math.ceil(2 * M / tile_2) * ntiles_n
            wv_m = math.ceil(cta_m / nsm)
            wv_2m = math.ceil(cta_2m / nsm)
            print(f"{M:>5} {2*M:>5} | {cta_m:>5} {wv_m:>4} {cm:>7} | "
                  f"{cta_2m:>6} {wv_2m:>5} {c2:>7} | "
                  f"{tm:>9.5f} {t2m:>9.5f} {t2m/tm:>6.2f}x")


if __name__ == "__main__":
    main()
