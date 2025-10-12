from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os, glob

home = os.path.expanduser("~")

setup(
    name="scaled_fp4_ops",
    ext_modules=[
        CUDAExtension(
            name="scaled_fp4_ops",
            sources=["binding.cpp", *glob.glob("kernel/*.cu")],
            include_dirs=["kernel/", os.path.join(home, "cutlass", "include")],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_100,code=sm_120",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
