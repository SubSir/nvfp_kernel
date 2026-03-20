"""
Quick alignment test between torch Stage3 implementation and Triton Stage3 kernel.
"""
import torch

from emulation.core import HardwareCore
from emulation.rounding import RoundStrategy

try:
    from emulation.triton_stage3 import triton_stage3_reduce4, _TRITON_AVAILABLE
except Exception:
    triton_stage3_reduce4 = None
    _TRITON_AVAILABLE = False


def _run_one_case(m: int, n: int, g4: int, w_stage3: int = 25):
    grouped = torch.randn((m, n, g4, 4), device="cuda", dtype=torch.float32)

    v_list = [grouped[..., i] for i in range(4)]

    max_val = torch.max(grouped.abs(), dim=-1)[0]
    _, max_exp = torch.frexp(max_val)

    triton_out_f64 = triton_stage3_reduce4(grouped.contiguous(), max_exp.contiguous(), w_stage3)

    if g4 == 1:
        torch_out = HardwareCore.hardware_reduction_4to1(
            v_list,
            W=w_stage3,
            output_fp32=True,
            rounding=RoundStrategy.RZ,
        )
        triton_out = HardwareCore.to_float32_with_rounding(triton_out_f64, RoundStrategy.RZ)
    else:
        torch_out = HardwareCore.hardware_reduction_4to1(
            v_list,
            W=w_stage3,
            output_fp32=False,
            rounding=RoundStrategy.RZ,
        )
        triton_out = triton_out_f64

    abs_diff = (torch_out - triton_out).abs()
    max_diff = abs_diff.max().item()
    exact_match = bool((abs_diff == 0).all().item())

    return {
        "shape": (m, n, g4, 4),
        "max_diff": max_diff,
        "exact_match": exact_match,
    }


def main():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return 0
    if not _TRITON_AVAILABLE or triton_stage3_reduce4 is None:
        print("[SKIP] Triton not available")
        return 0

    torch.manual_seed(20260320)
    cases = [
        (32, 64, 1),
        (32, 64, 2),
        (64, 128, 4),
    ]

    failed = False
    for m, n, g4 in cases:
        ret = _run_one_case(m, n, g4)
        print(
            f"shape={ret['shape']} max_diff={ret['max_diff']:.6e} exact={ret['exact_match']}"
        )
        if not ret["exact_match"]:
            failed = True

    if failed:
        print("[FAIL] torch/triton stage3 not aligned")
        return 1

    print("[PASS] torch/triton stage3 aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
