"""Decode latency with CUDA graphs (amortized launch overhead).

The per-call FP4 floor (~0.05 ms) seen in the plain benchmarks is host launch +
workspace setup, not the GEMM. Real serving replays a captured graph, so capture
each path and time replay. In the memory-bound decode regime FP4 reads ~4x fewer
weight bytes than bf16, so it should win at small M once launch is amortized.
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
MS = [1, 4, 16, 64, 256]


def graph_bench(fn, inner=50, iters=50, warmup=5):
    # Warm up on a side stream, then capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(iters):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / inner)
    return best


def main():
    torch.manual_seed(0)
    dev = "cuda"

    for (name, K, N) in SHAPES:
        w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.5
        w_gs = global_scale(w)
        w_q, w_sf = scaled_fp4_quant(w, w_gs)

        print(f"\n## {name}  (K={K}, N={N})  [CUDA graph replay]")
        print(f"{'M':>5} {'bf16':>9} {'fp4':>9} {'resid':>9} "
              f"{'bf16/fp4':>9} {'bf16/res':>9}")
        print("-" * 60)
        for M in MS:
            # Static input buffers (filled once; graph replays over them).
            a = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.5
            a_gs = global_scale(a)
            al = (1.0 / (a_gs * w_gs)).to(torch.float32)
            aq, asf = scaled_fp4_quant(a, a_gs)
            aqr, asfr = scaled_fp4_quant_residual(a, a_gs)

            def run_bf16():
                return torch.matmul(a, w.t())

            def run_fp4():
                return cutlass_scaled_fp4_mm(aq, w_q, asf, w_sf, al, torch.bfloat16)

            def run_resid():
                out2 = cutlass_scaled_fp4_mm(aqr, w_q, asfr, w_sf, al, torch.bfloat16)
                return out2[:M] + out2[M:]

            try:
                tb = graph_bench(run_bf16)
                tf = graph_bench(run_fp4)
                tr = graph_bench(run_resid)
                print(f"{M:>5} {tb:>9.5f} {tf:>9.5f} {tr:>9.5f} "
                      f"{tb / tf:>8.2f}x {tb / tr:>8.2f}x")
            except Exception as ex:  # pragma: no cover
                print(f"{M:>5}  graph capture failed: {type(ex).__name__}: {ex}")
                return


if __name__ == "__main__":
    main()
