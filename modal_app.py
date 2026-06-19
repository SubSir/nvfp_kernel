"""Modal app to build the nvfp CUDA extension and run tests/benchmarks on a B200.

Usage:
    modal run modal_app.py::baseline          # build sanity + baseline fp4 gemm
    modal run modal_app.py::accuracy          # residual vs single-level fp4 accuracy
    modal run modal_app.py::speed             # MiniMax-M3 shape speed sweep
    modal run modal_app.py::shell             # interactive-ish: run an arbitrary script
                                              #   modal run modal_app.py::shell --script tests/test_fp4_gemm.py

The CUTLASS build is baked into an image layer keyed on kernel/ + setup.py +
binding.cpp, so editing Python (nvfp/*.py, tests/*.py) does NOT trigger a
recompile -- those are mounted at runtime.
"""

import modal

CUTLASS_TAG = "v3.9.2"
GPU = "B200"  # GB100 / sm_100a

cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "torch==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("numpy", "pandas", "tabulate")
    .run_commands(
        f"git clone --depth 1 --branch {CUTLASS_TAG} "
        "https://github.com/NVIDIA/cutlass.git /root/cutlass"
    )
    .env({"CUTLASS_DIR": "/root/cutlass", "NVFP_CUDA_ARCH": "100a"})
    # --- build-relevant sources only (copy => part of the cached build layer) ---
    .add_local_dir("kernel", "/root/build_src/kernel", copy=True)
    .add_local_file("setup.py", "/root/build_src/setup.py", copy=True)
    .add_local_file("binding.cpp", "/root/build_src/binding.cpp", copy=True)
    .run_commands(
        # The standalone python's sysconfig links extensions with clang++, which
        # isn't installed; force g++ for compile+link.
        "bash -lc 'set -o pipefail; cd /root/build_src && "
        "CC=g++ CXX=g++ LDSHARED=\"g++ -shared\" "
        "python setup.py install 2>&1 | tail -150'"
    )
    # --- full repo at runtime (no copy => does NOT invalidate the build) ---
    .add_local_dir(
        ".",
        "/root/nvfp_kernel",
        ignore=[
            "**/.git/**",
            "**/__pycache__/**",
            "**/build/**",
            "**/*.egg-info/**",
        ],
    )
)

app = modal.App("nvfp-residual", image=cuda_image)


def _run(script: str):
    import os
    import subprocess

    print(f"===== running {script} on {GPU} =====", flush=True)
    env = {**os.environ, "PYTHONPATH": "/root/nvfp_kernel"}
    subprocess.run(["python", script], cwd="/root/nvfp_kernel", check=True, env=env)


@app.function(gpu=GPU, timeout=1800)
def baseline():
    import subprocess

    subprocess.run(
        ["python", "-c", "import torch, scaled_fp4_ops; "
         "print('torch', torch.__version__, 'device', torch.cuda.get_device_name())"],
        check=True,
    )
    _run("tests/test_fp4_gemm.py")


@app.function(gpu=GPU, timeout=1800)
def accuracy():
    _run("tests/test_residual_accuracy.py")


@app.function(gpu=GPU, timeout=3600)
def speed():
    _run("tests/bench_minimax_m3.py")


@app.function(gpu=GPU, timeout=3600)
def shell(script: str = "tests/test_fp4_gemm.py"):
    _run(script)
