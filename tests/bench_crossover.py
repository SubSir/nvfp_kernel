"""Where does FP4 (and residual FP4) beat cuBLAS bf16?

At decode sizes the FP4 GEMM sits on a fixed launch/overhead floor (~0.076 ms)
while bf16 scales with M*K*N, so FP4 only wins once the matmul is big enough.
Sweep M for each MiniMax-M3 weight shape (weight pre-quantized offline) and mark
the crossover, plus a square M=N=K sweep for reference.
"""

import torch

from nvfp.fp4_gemm import global_scale
from nvfp.ops import scaled_fp4_quant, scaled_fp4_quant_residual, cutlass_scaled_fp4_mm

SHAPES = [
    ("qkv_proj", 6144, 9216),
    ("o_proj", 8192, 6144),
    ("moe_gate_up", 6144, 6144),
    ("moe_down", 3072, 6144),
]
MS = [1, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def bench(fn, inner=10, iters=20, warmup=5):
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


def sweep(name, K, N, dev):
    w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.5
    w_gs = global_scale(w)
    w_q, w_sf = scaled_fp4_quant(w, w_gs)

    print(f"\n## {name}  (K={K}, N={N})")
    print(f"{'M':>6} {'bf16':>9} {'fp4':>9} {'resid':>9} "
          f"{'fp4/bf16':>9} {'res/bf16':>9}  winners(<=bf16)")
    print("-" * 78)
    for M in MS:
        a = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.5

        def run_bf16():
            return torch.matmul(a, w.t())

        def run_fp4():
            a_gs = global_scale(a)
            a_q, a_sf = scaled_fp4_quant(a, a_gs)
            al = (1.0 / (a_gs * w_gs)).to(torch.float32)
            return cutlass_scaled_fp4_mm(a_q, w_q, a_sf, w_sf, al, torch.bfloat16)

        def run_resid():
            a_gs = global_scale(a)
            a_q, a_sf = scaled_fp4_quant_residual(a, a_gs)
            al = (1.0 / (a_gs * w_gs)).to(torch.float32)
            out2 = cutlass_scaled_fp4_mm(a_q, w_q, a_sf, w_sf, al, torch.bfloat16)
            return out2[:M] + out2[M:]

        tb = bench(run_bf16)
        tf = bench(run_fp4)
        tr = bench(run_resid)
        win = []
        if tf <= tb:
            win.append("fp4")
        if tr <= tb:
            win.append("resid")
        print(f"{M:>6} {tb:>9.4f} {tf:>9.4f} {tr:>9.4f} "
              f"{tf / tb:>8.2f}x {tr / tb:>8.2f}x  {','.join(win) or '-'}")


def square(dev):
    print("\n## square  (M = N = K)")
    print(f"{'S':>6} {'bf16':>9} {'fp4':>9} {'resid':>9} "
          f"{'fp4/bf16':>9} {'res/bf16':>9}")
    print("-" * 60)
    for S in [512, 1024, 2048, 4096, 8192]:
        a = torch.randn(S, S, dtype=torch.bfloat16, device=dev) * 0.5
        w = torch.randn(S, S, dtype=torch.bfloat16, device=dev) * 0.5
        w_gs = global_scale(w)
        w_q, w_sf = scaled_fp4_quant(w, w_gs)

        def run_bf16():
            return torch.matmul(a, w.t())

        def run_fp4():
            a_gs = global_scale(a)
            a_q, a_sf = scaled_fp4_quant(a, a_gs)
            al = (1.0 / (a_gs * w_gs)).to(torch.float32)
            return cutlass_scaled_fp4_mm(a_q, w_q, a_sf, w_sf, al, torch.bfloat16)

        def run_resid():
            a_gs = global_scale(a)
            a_q, a_sf = scaled_fp4_quant_residual(a, a_gs)
            al = (1.0 / (a_gs * w_gs)).to(torch.float32)
            out2 = cutlass_scaled_fp4_mm(a_q, w_q, a_sf, w_sf, al, torch.bfloat16)
            return out2[:S] + out2[S:]

        tb, tf, tr = bench(run_bf16), bench(run_fp4), bench(run_resid)
        print(f"{S:>6} {tb:>9.4f} {tf:>9.4f} {tr:>9.4f} "
              f"{tf / tb:>8.2f}x {tr / tb:>8.2f}x")


def main():
    torch.manual_seed(0)
    dev = "cuda"
    for (name, K, N) in SHAPES:
        sweep(name, K, N, dev)
    square(dev)


if __name__ == "__main__":
    main()
