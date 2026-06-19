"""Throughput vs M for bf16 / single-FP4 / residual-FP4, CUDA-graph timed
(launch amortized, as in serving). Emits parseable DATA lines; plotted locally.

Shape: MiniMax-M3 qkv_proj (K=6144, N=9216). Weight pre-quantized (offline).
"""

import statistics

import torch

from nvfp.fp4_gemm import global_scale
from nvfp.ops import scaled_fp4_quant, scaled_fp4_quant_residual, cutlass_scaled_fp4_mm

K, N = 6144, 9216
MS = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def graph_bench(fn, inner=30, iters=60, warmup=10):
    """Return (mean_ms, std_ms) over `iters` windows, each averaging `inner`
    graph replays."""
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
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) / inner)
    return statistics.mean(ts), statistics.pstdev(ts)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.5
    w_gs = global_scale(w)
    w_q, w_sf = scaled_fp4_quant(w, w_gs)

    print(f"# shape qkv_proj K={K} N={N}  (mean +/- std over 60 windows)")
    print("# DATA,M,bf16_mean,bf16_std,fp4_mean,fp4_std,resid_mean,resid_std  (ms)")
    for M in MS:
        a = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.5
        a_gs = global_scale(a)
        al = (1.0 / (a_gs * w_gs)).to(torch.float32)
        aq, asf = scaled_fp4_quant(a, a_gs)
        aqr, asfr = scaled_fp4_quant_residual(a, a_gs)

        tb, sb = graph_bench(lambda: torch.matmul(a, w.t()))
        tf, sf = graph_bench(lambda: cutlass_scaled_fp4_mm(aq, w_q, asf, w_sf, al, torch.bfloat16))

        def resid():
            o2 = cutlass_scaled_fp4_mm(aqr, w_q, asfr, w_sf, al, torch.bfloat16)
            return o2[:M] + o2[M:]

        tr, sr = graph_bench(resid)
        print(f"DATA,{M},{tb:.6f},{sb:.6f},{tf:.6f},{sf:.6f},{tr:.6f},{sr:.6f}", flush=True)


if __name__ == "__main__":
    main()
