import torch
import ops
import pseudo_quant

# Input tensors
a = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")

# Create tensor b with the last dimension having first 32 elements as 1 and last 32 as 0
b = torch.randn(64, 32, dtype=torch.bfloat16, device="cuda")


# Quantize weight to NVFP4
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
b_amax = torch.abs(b).max().to(torch.float32)
b_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / b_amax

# Quantize weight
b_fp4, scale_b_fp4 = ops.scaled_fp4_quant(b, b_global_scale)

# Quantize activation
a_amax = torch.abs(a).max().to(torch.float32)
a_global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / a_amax
a_fp4, scale_a_fp4 = ops.scaled_fp4_quant(a, a_global_scale)

# Compute GEMM with quantized tensors
alpha = 1.0 / (a_global_scale * b_global_scale)
output = ops.cutlass_scaled_fp4_mm(
    a_fp4, b_fp4, scale_a_fp4, scale_b_fp4, alpha, torch.bfloat16
)

print(output)

print(pseudo_quant.nvfp4_pseudo_quant(a) @ pseudo_quant.nvfp4_pseudo_quant(b).T)