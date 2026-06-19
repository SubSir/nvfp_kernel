"""HONEST eager comparison (NO CUDA graph): bf16 vs single-W4A4 vs residual-W4A4.

This is how the public API actually runs: per-launch overhead included, dynamic
activation quant, the residual output add as a real op. Weight pre-quantized
(offline). This is the regime where residual's extra kernel launches matter, and
where it can lose to bf16 even though single W4A4 wins.
"""

import torch

from nvfp.fp4_gemm import global_scale
from nvfp.ops import scaled_fp4_quant, scaled_fp4_quant_residual, cutlass_scaled_fp4_mm

K, N = 6144, 9216
MS = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def bench(fn, inner=50, iters=30, warmup=20):
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
    w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.5
    w_gs = global_scale(w)
    w_q, w_sf = scaled_fp4_quant(w, w_gs)

    print("EAGER (no CUDA graph)  qkv_proj K=6144 N=9216")
    print(f"{'M':>6} {'bf16':>9} {'single':>9} {'resid':>9} "
          f"{'single/bf16':>11} {'resid/bf16':>11}")
    print("-" * 62)
    for M in MS:
        a = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.5

        def run_bf16():
            return torch.matmul(a, w.t())

        def run_fp4():
            a_gs = global_scale(a)
            aq, asf = scaled_fp4_quant(a, a_gs)
            al = (1.0 / (a_gs * w_gs)).to(torch.float32)
            return cutlass_scaled_fp4_mm(aq, w_q, asf, w_sf, al, torch.bfloat16)

        def run_resid():
            a_gs = global_scale(a)
            aqr, asfr = scaled_fp4_quant_residual(a, a_gs)
            al = (1.0 / (a_gs * w_gs)).to(torch.float32)
            o2 = cutlass_scaled_fp4_mm(aqr, w_q, asfr, w_sf, al, torch.bfloat16)
            return o2[:M] + o2[M:]

        tb = bench(run_bf16)
        tf = bench(run_fp4)
        tr = bench(run_resid)
        # speedup vs bf16 (>1 = faster than bf16)
        print(f"{M:>6} {tb:>9.4f} {tf:>9.4f} {tr:>9.4f} "
              f"{tb/tf:>10.2f}x {tb/tr:>10.2f}x")


if __name__ == "__main__":
    main()
