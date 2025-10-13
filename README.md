# nvfp_kernel
It's the nvfp kernel extracted from vLLM ([GitHub repository](https://github.com/vllm-project/vllm)).

## Setup
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python setup.py build_ext
```

## Correctness Test
```bash
python example.py
```
There's still a small gap between the pseudo quantization and the real quantization.