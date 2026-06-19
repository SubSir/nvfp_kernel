"""Speed: residual vs single-level NVFP4 vs bf16 on MiniMax-M3 GEMM shapes.

The core claim: in memory-bound decode the weight is streamed once and dominates
traffic, so doubling the activation rows (residual) costs little until the GEMM
turns compute-bound. We measure it two ways:

  [GEMM-only]  pure FP4 GEMM at M rows vs 2M rows, activations PRE-quantized
               -> isolates "cost of doubling tokens" (ratio ->1 memory-bound,
               ->2 compute-bound).
  [full path]  activation quant (+residual) + GEMM (+add), weight pre-quantized
               offline -> end-to-end decode latency; also vs cuBLAS bf16.

MiniMax-M3 (text): hidden=6144, 64x128 q heads (q=8192) + 4 kv heads (kv=512),
MoE expert intermediate=3072 (128 experts, top-4).
"""

import torch

from nvfp.fp4_gemm import global_scale
from nvfp.ops import scaled_fp4_quant, scaled_fp4_quant_residual, cutlass_scaled_fp4_mm

# (name, K=in_features, N=out_features)
SHAPES = [
    ("qkv_proj", 6144, 9216),
    ("o_proj", 8192, 6144),
    ("moe_gate_up", 6144, 6144),
    ("moe_down", 3072, 6144),
]
MS = [1, 4, 16, 64, 256, 1024]


def bench(fn, inner=25, iters=40, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    best = float("inf")
    for _ in range(iters):
        start.record()
        for _ in range(inner):
            fn()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / inner)
    return best  # ms


def main():
    torch.manual_seed(0)
    dev = "cuda"

    for (name, K, N) in SHAPES:
        w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.5
        w_gs = global_scale(w)
        w_q, w_sf = scaled_fp4_quant(w, w_gs)  # offline
        alpha = (1.0 / (w_gs * w_gs)).to(torch.float32)

        print(f"\n## {name}  (K={K}, N={N})")
        print(f"{'M':>5} | {'gemm(M)':>9} {'gemm(2M)':>9} {'2M/M':>6} | "
              f"{'bf16':>8} {'fp4':>8} {'resid':>8} {'res/fp4':>8} {'bf/res':>7}")
        print("-" * 86)

        for M in MS:
            a = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.5
            # Pre-quantized activations for the GEMM-only measurement.
            aq_m, asf_m = scaled_fp4_quant(torch.randn(M, K, dtype=torch.bfloat16, device=dev), w_gs)
            aq_2m, asf_2m = scaled_fp4_quant(torch.randn(2 * M, K, dtype=torch.bfloat16, device=dev), w_gs)

            def gemm_m():
                return cutlass_scaled_fp4_mm(aq_m, w_q, asf_m, w_sf, alpha, torch.bfloat16)

            def gemm_2m():
                return cutlass_scaled_fp4_mm(aq_2m, w_q, asf_2m, w_sf, alpha, torch.bfloat16)

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

            tg_m = bench(gemm_m)
            tg_2m = bench(gemm_2m)
            t_bf16 = bench(run_bf16)
            t_fp4 = bench(run_fp4)
            t_res = bench(run_resid)
            print(f"{M:>5} | {tg_m:>9.4f} {tg_2m:>9.4f} {tg_2m / tg_m:>5.2f}x | "
                  f"{t_bf16:>8.4f} {t_fp4:>8.4f} {t_res:>8.4f} "
                  f"{t_res / t_fp4:>7.2f}x {t_bf16 / t_res:>6.2f}x")


if __name__ == "__main__":
    main()
