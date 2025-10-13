import torch

def float_value(n_bit, signed=True):
    B = n_bit - 1 if signed else n_bit

    # mapping, total_bit: exponent_bit
    exp_field_map = {3: 2, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4}
    bias_field_map = {3: 0, 4: 0, 5: 0, 6: 4, 7: 4, 8: 8}
    if n_bit in exp_field_map:
        exp_field = exp_field_map[n_bit]
        bias = bias_field_map[n_bit]
    else:
        raise ValueError("Not support this bit width")
    exp_bit = exp_field

    man_bit = B - exp_bit
    values = []
    min_to_zero = True
    subnormal = True
    for i in range(2 ** exp_bit):
        for j in range(2 ** man_bit):
            if min_to_zero:
                values.append(0.)
                values.append(-0.)
                min_to_zero = False
            else:
                if subnormal:
                    values.append((2 ** (i - bias)) * (j * 2 ** (-man_bit)))
                else:
                    values.append((2 ** (i - 1 - bias)) * (1 + j * 2 ** (-man_bit)))

                if signed:
                    if subnormal:
                        values.append(-(2 ** (i - bias)) * (j * 2 ** (-man_bit)))
                    else:
                        values.append(-(2 ** (i - 1 - bias)) * (1 + j * 2 ** (-man_bit)))
        subnormal = False

    return torch.tensor(values)
def nvfp4_pseudo_quant(tensor_value: torch.Tensor):  
    tensor_value = tensor_value.float()
    org_shape = tensor_value.shape  
    quant_grid = float_value(4).to(tensor_value.device).float()  
    q_group_size = 16  
    assert org_shape[-1] % q_group_size == 0  
      
    # Reshape to process blocks  
    tensor_value = tensor_value.reshape(-1, q_group_size)  
    max_val = tensor_value.abs().amax(dim=1, keepdim=True)  
  
    max_quant_val = max(quant_grid)  
    assert torch.isinf(max_val).sum() == 0  
      
    # Compute scales and apply global scaling factor  
    scales = max_val / max_quant_val  
    
    # Use a global scaling factor to scale the scales
    global_scale = scales.max() / 448.0
    scales = scales / global_scale
    scales = scales.clamp(max=448.0)  

    scales_quantized = scales.to(torch.float8_e4m3fn).to(tensor_value.dtype) * global_scale

    zero_scale_mask = (scales_quantized.squeeze(-1) == 0)

    scales_for_div = scales_quantized.clone()
    scales_for_div[zero_scale_mask] = 1.0  

    labels = ((tensor_value / scales_for_div).unsqueeze(-1) - quant_grid).abs().argmin(dim=-1)  
    tensor_deq = quant_grid[labels] * scales_quantized

    tensor_deq[zero_scale_mask] = 0.0

    return tensor_deq.reshape(org_shape).float()