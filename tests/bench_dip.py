"""Is the M=2048 throughput dip real or noise? Report min/mean/median/std (not
just min) over many iters, on a fine M grid, with CTA/wave accounting.

FP4 GEMM only (activations pre-quantized). Default config for rows>256 is
MmaTile 256x256x256, ClusterShape 2x1x1 -> 2 CTAs per (256-M x 256-N) tile.
"""

import math
import statistics

import torch

from nvfp.fp4_gemm import global_scale
from nvfp.ops import scaled_fp4_quant, cutlass_scaled_fp4_mm

K, N = 6144, 9216
MS = [1024, 1280, 1536, 1792, 2048, 2560, 3072, 4096, 6144, 8192]
FP4_TFLOPS = 9000.0


def stats(fn, inner=20, iters=80, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record()
        for _ in range(inner):
            fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / inner)
    return min(ts), statistics.mean(ts), statistics.median(ts), statistics.pstdev(ts)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"device SMs = {nsm}")
    w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.5
    w_gs = global_scale(w)
    w_q, w_sf = scaled_fp4_quant(w, w_gs)
    alpha = (1.0 / (w_gs * w_gs)).to(torch.float32)
    n_tiles = math.ceil(N / 256)  # default config N-tile = 256

    print(f"\n## qkv_proj K={K} N={N}  (TFLOPS from MIN time)")
    print(f"{'M':>6} {'Mtiles':>6} {'CTAs':>5} {'waves':>5} {'tail%':>6} | "
          f"{'min':>7} {'mean':>7} {'med':>7} {'std':>6} | {'TFLOP_min':>9} {'%pk':>5} {'spread%':>7}")
    print("-" * 96)
    for M in MS:
        aq, asf = scaled_fp4_quant(torch.randn(M, K, dtype=torch.bfloat16, device=dev), w_gs)
        tmin, tmean, tmed, tstd = stats(
            lambda: cutlass_scaled_fp4_mm(aq, w_q, asf, w_sf, alpha, torch.bfloat16))
        m_tiles = math.ceil(M / 256)
        ctas = 2 * m_tiles * n_tiles            # 2 CTAs per tile (cluster 2x1x1)
        waves = math.ceil(ctas / nsm)
        tail = ctas - (waves - 1) * nsm
        tailpct = 100 * tail / nsm
        tf = 2.0 * M * K * N / (tmin * 1e-3) / 1e12
        spread = 100 * (tmean - tmin) / tmin
        print(f"{M:>6} {m_tiles:>6} {ctas:>5} {waves:>5} {tailpct:>5.0f}% | "
              f"{tmin:>7.4f} {tmean:>7.4f} {tmed:>7.4f} {tstd:>6.4f} | "
              f"{tf:>9.0f} {100*tf/FP4_TFLOPS:>4.0f}% {spread:>6.1f}%")


if __name__ == "__main__":
    main()
