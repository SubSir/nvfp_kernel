"""Time breakdown + roofline: where does the residual path spend time, and why
does the GEMM double at large M?

Components (CUDA-graph timed): activation quant (single vs residual), the FP4
GEMM at M vs 2M rows, and the output split-add. For the GEMM we also report the
achieved tensor throughput (TFLOPS) and the achieved HBM bandwidth under an
"ideal traffic" model (weight read once + activation + output), each as a % of
the B200 roofline -- so you can see the memory-bound -> compute-bound crossover.

B200 (GB100) datasheet peaks: HBM3e 8.0 TB/s, FP4 dense tensor 9 PFLOPS
(18 PFLOPS with 2:4 sparsity -- we run dense, so 9000 TFLOPS is the ceiling).
"""

import torch

from nvfp.fp4_gemm import global_scale
from nvfp.ops import scaled_fp4_quant, scaled_fp4_quant_residual, cutlass_scaled_fp4_mm

SHAPES = [("qkv_proj", 6144, 9216), ("o_proj", 8192, 6144)]
MS = [128, 256, 512, 1024, 2048]

HBM_GBPS = 8000.0       # B200 HBM3e: 8.0 TB/s (datasheet)
FP4_TFLOPS = 9000.0     # B200 FP4 dense tensor peak: 9 PFLOPS (18 w/ sparsity)


def graph_bench(fn, inner=50, iters=50, warmup=5):
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
    import statistics
    return statistics.mean(ts)


def gemm_roofline(rows, K, N, t_ms):
    flops = 2.0 * rows * K * N
    tflops = flops / (t_ms * 1e-3) / 1e12
    # ideal traffic: W fp4 (once) + A fp4 + D bf16  (+ small scales ignored)
    bytes_ = (N * K / 2) + (rows * K / 2) + (rows * N * 2)
    gbps = bytes_ / (t_ms * 1e-3) / 1e9
    return tflops, 100 * tflops / FP4_TFLOPS, gbps, 100 * gbps / HBM_GBPS


def main():
    torch.manual_seed(0)
    dev = "cuda"
    for (name, K, N) in SHAPES:
        w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.5
        w_gs = global_scale(w)
        w_q, w_sf = scaled_fp4_quant(w, w_gs)

        print(f"\n## {name}  (K={K}, N={N})")
        print(f"{'M':>5} | {'q1':>7} {'qres':>7} {'gemmM':>7} {'gem2M':>7} {'add':>7} "
              f"| {'single':>8} {'resid':>8} {'r/s':>5} "
              f"| {'gemmM:TF/%pk/BW/%pk':>26} | {'gem2M:TF/%pk/BW/%pk':>26}")
        print("-" * 132)
        for M in MS:
            a = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.5
            a_gs = global_scale(a)
            al = (1.0 / (a_gs * w_gs)).to(torch.float32)
            aq, asf = scaled_fp4_quant(a, a_gs)
            aqr, asfr = scaled_fp4_quant_residual(a, a_gs)
            out2 = torch.randn(2 * M, N, dtype=torch.bfloat16, device=dev)

            t_q1 = graph_bench(lambda: scaled_fp4_quant(a, a_gs))
            t_qr = graph_bench(lambda: scaled_fp4_quant_residual(a, a_gs))
            t_gm = graph_bench(lambda: cutlass_scaled_fp4_mm(aq, w_q, asf, w_sf, al, torch.bfloat16))
            t_g2 = graph_bench(lambda: cutlass_scaled_fp4_mm(aqr, w_q, asfr, w_sf, al, torch.bfloat16))
            t_add = graph_bench(lambda: out2[:M] + out2[M:])

            single = t_q1 + t_gm
            resid = t_qr + t_g2 + t_add

            tf_m, tfp_m, bw_m, bwp_m = gemm_roofline(M, K, N, t_gm)
            tf_2, tfp_2, bw_2, bwp_2 = gemm_roofline(2 * M, K, N, t_g2)

            print(f"{M:>5} | {t_q1:>7.4f} {t_qr:>7.4f} {t_gm:>7.4f} {t_g2:>7.4f} {t_add:>7.4f} "
                  f"| {single:>8.4f} {resid:>8.4f} {resid/single:>4.2f}x "
                  f"| {tf_m:>6.0f} {tfp_m:>3.0f}% {bw_m:>6.0f} {bwp_m:>3.0f}% "
                  f"| {tf_2:>6.0f} {tfp_2:>3.0f}% {bw_2:>6.0f} {bwp_2:>3.0f}%")


if __name__ == "__main__":
    main()
