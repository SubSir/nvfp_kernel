"""
Quick alignment test between torch Stage4 implementation and Triton Stage4 kernel.
"""
import torch

from emulation.core import HardwareCore
from emulation.rounding import RoundStrategy

try:
    from emulation.triton_stage4 import triton_stage4_add_wbits, _TRITON_AVAILABLE
except Exception:
    triton_stage4_add_wbits = None
    _TRITON_AVAILABLE = False


def _run_one_case(shape, w_stage4: int = 25):
    acc_fp32 = torch.randn(shape, device="cuda", dtype=torch.float32)
    new_val_wbits = torch.randn(shape, device="cuda", dtype=torch.float64)

    torch_out = HardwareCore.hardware_add_wbits(
        acc_fp32,
        new_val_wbits,
        W=w_stage4,
        rounding=RoundStrategy.RZ,
        use_triton=False,
    )

    triton_out = HardwareCore.hardware_add_wbits(
        acc_fp32,
        new_val_wbits,
        W=w_stage4,
        rounding=RoundStrategy.RZ,
        use_triton=True,
    )

    abs_diff = (torch_out - triton_out).abs()
    max_diff = abs_diff.max().item()
    exact_match = bool((abs_diff == 0).all().item())

    return {
        "shape": tuple(shape),
        "max_diff": max_diff,
        "exact_match": exact_match,
    }


def main():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available")
        return 0
    if not _TRITON_AVAILABLE or triton_stage4_add_wbits is None:
        print("[SKIP] Triton not available")
        return 0

    torch.manual_seed(20260320)
    cases = [
        (32, 64),
        (64, 128),
        (8, 16, 32),
    ]

    failed = False
    for shape in cases:
        ret = _run_one_case(shape)
        print(
            f"shape={ret['shape']} max_diff={ret['max_diff']:.6e} exact={ret['exact_match']}"
        )
        if not ret["exact_match"]:
            failed = True

    if failed:
        print("[FAIL] torch/triton stage4 not aligned")
        return 1

    print("[PASS] torch/triton stage4 aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
