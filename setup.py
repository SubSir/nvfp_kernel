from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, glob

home = os.path.expanduser("~")
current_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Architecture / CUTLASS configuration (overridable via environment variables)
#   NVFP_CUDA_ARCH : "120a" (RTX 50xx, default) or "100a" (B200 / GB100)
#   CUTLASS_DIR    : path to a CUTLASS source checkout (>=3.8 for sm100 nvfp4)
# ---------------------------------------------------------------------------
arch = os.environ.get("NVFP_CUDA_ARCH", "120a")  # e.g. "120a" or "100a"
cutlass_dir = os.environ.get("CUTLASS_DIR", os.path.join(home, "cutlass"))

if arch.startswith("100"):
    enable_macro = "ENABLE_NVFP4_SM100"
    excluded_mm = "nvfp4_scaled_mm_sm120_kernels.cu"
elif arch.startswith("120"):
    enable_macro = "ENABLE_NVFP4_SM120"
    excluded_mm = "nvfp4_scaled_mm_sm100_kernels.cu"
else:
    raise ValueError(f"Unsupported NVFP_CUDA_ARCH={arch!r}; expected '100a' or '120a'.")

# Only compile the GEMM kernel that matches the target arch (the CUTLASS MMA
# atoms for sm120 do not instantiate on sm100 and vice versa).
sources = ["binding.cpp"]
for cu in sorted(glob.glob("kernel/*.cu")):
    if os.path.basename(cu) == excluded_mm:
        continue
    sources.append(cu)

print(f"[setup] arch=sm_{arch}  enable={enable_macro}  cutlass={cutlass_dir}")
print(f"[setup] sources={sources}")

setup(
    name="nvfp",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="scaled_fp4_ops",
            sources=sources,
            include_dirs=[
                os.path.join(current_dir, "kernel"),
                os.path.join(cutlass_dir, "include"),
                os.path.join(cutlass_dir, "tools/util/include"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0",
                        f"-D{enable_macro}=1"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    f"-gencode=arch=compute_{arch},code=sm_{arch}",
                    f"-D{enable_macro}=1",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=2.8.0",
    ],
    python_requires=">=3.8",
)
