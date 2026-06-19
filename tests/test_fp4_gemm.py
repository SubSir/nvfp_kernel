"""Baseline sanity: single-level NVFP4 GEMM on B200 (sm100) vs bf16 reference."""

import sys

import torch

from nvfp.fp4_gemm import fp4_gemm


def rel_err(out, ref):
    return (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    # (M, K, N) -- include the small-M decode regime and a couple of M3 shapes.
    shapes = [
        (1, 6144, 9216),
        (4, 6144, 9216),
        (16, 8192, 6144),
        (64, 6144, 6144),
        (256, 3072, 6144),
        (1024, 4096, 4096),
    ]
    ok = True
    print(f"{'M':>6} {'K':>6} {'N':>6} {'rel_err':>10}")
    print("-" * 34)
    for (M, K, N) in shapes:
        a = torch.randn(M, K, dtype=torch.bfloat16, device=dev) * 0.5
        b = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.5
        ref = (a.float() @ b.float().T).bfloat16()
        out = fp4_gemm(a, b, out_dtype=torch.bfloat16)
        e = rel_err(out, ref).item()
        # W4A4 has large but bounded quantization error; ~0.1-0.25 typical.
        flag = "OK" if e < 0.35 else "HIGH"
        if e >= 0.35:
            ok = False
        print(f"{M:>6} {K:>6} {N:>6} {e:>10.4f}  {flag}")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
