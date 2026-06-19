"""Does the FP4 GEMM stay stuck at ~50% of peak, or climb with problem size?
If it climbs past 50% at large M, the ~50% at M=2048 is a size/wave effect, not
a 'half the chip is idle' cap. GEMM-only, activations pre-quantized.

B200: FP4 dense peak 9 PFLOPS, 148 SMs (both dies), HBM3e 8 TB/s.
"""

import torch

from nvfp.fp4_gemm import global_scale
from nvfp.ops import scaled_fp4_quant, cutlass_scaled_fp4_mm

FP4_TFLOPS = 9000.0
SHAPES = [("qkv_proj", 6144, 9216), ("square8k", 8192, 8192)]
MS = [2048, 4096, 8192, 16384, 32768]


def bench(fn, inner=20, iters=20, warmup=5):
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


def main():
    torch.manual_seed(0)
    dev = "cuda"
    nsm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"device SMs = {nsm}  (both dies)")
    for (name, K, N) in SHAPES:
        w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.5
        w_gs = global_scale(w)
        w_q, w_sf = scaled_fp4_quant(w, w_gs)
        alpha = (1.0 / (w_gs * w_gs)).to(torch.float32)
        print(f"\n## {name}  (K={K}, N={N})")
        print(f"{'M':>7} {'gemm(ms)':>9} {'TFLOPS':>8} {'%peak':>6}")
        print("-" * 34)
        for M in MS:
            try:
                a = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.5
                aq, asf = scaled_fp4_quant(a, w_gs)

                def gm():
                    return cutlass_scaled_fp4_mm(aq, w_q, asf, w_sf, alpha, torch.bfloat16)

                t = bench(gm)
                tf = 2.0 * M * K * N / (t * 1e-3) / 1e12
                print(f"{M:>7} {t:>9.4f} {tf:>8.0f} {100*tf/FP4_TFLOPS:>5.0f}%")
            except RuntimeError as ex:
                print(f"{M:>7}  OOM/err: {ex}")
                break


if __name__ == "__main__":
    main()
