"""
Core emulation classes: HardwareCore and MMAEngine
Fixed rounding strategy implementation
"""
import time
from collections import defaultdict

import torch
import nvfp.pseudo_quant as pseudo_quant

from .rounding import RoundStrategy, RoundingRegistry

try:
    from .triton_stage3 import triton_stage3_reduce4
    _HAS_TRITON_STAGE3 = True
except Exception:
    triton_stage3_reduce4 = None
    _HAS_TRITON_STAGE3 = False


def _now_cuda_ms():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() * 1000.0


def _profile_begin(timing_dict, key):
    timing_dict[key] += -_now_cuda_ms()


def _profile_end(timing_dict, key):
    timing_dict[key] += _now_cuda_ms()


class HardwareCore:
    """负责硬件规约逻辑，支持可配置的 rounding strategy"""
    
    @staticmethod
    def to_float32_with_rounding(tensor_f64, rounding=RoundStrategy.RZ):
        """
        Convert float64 to float32 with specified rounding strategy.
        This is where rounding strategy is applied in hardware.
        """
        if rounding == RoundStrategy.RZ:
            # Round Toward Zero: truncate extra precision
            f32 = tensor_f64.to(torch.float32)
            mask = f32.abs() > tensor_f64.abs()
            res = f32.clone()
            if mask.any():
                res[mask] = torch.nextafter(f32[mask], torch.zeros_like(f32[mask]))
            return res
        elif rounding == RoundStrategy.RNE:
            # Round to Nearest Even: use PyTorch's round (banker's rounding) then cast
            # First convert to float32 (which rounds to nearest by default in PyTorch)
            return tensor_f64.to(torch.float32)
        else:
            # Other strategies should be handled by RoundingRegistry
            raise NotImplementedError(f"Rounding {rounding} not implemented in to_float32_with_rounding")

    @staticmethod
    def hardware_add_wbits(acc_fp32, new_val_wbits, W=25, rounding=RoundStrategy.RZ):
        """
        Stage 4: FP32 accumulator + W bits new_val -> FP32 output.
        
        Hardware behavior:
        1. Align new_val to acc's exponent (truncate/RZ - fixed hardware behavior)
        2. Add in fixed-point domain
        3. Convert back with specified rounding strategy
        """
        # Step 1: Align to acc's exponent
        _, acc_exp = torch.frexp(acc_fp32.abs())
        scale = 2.0**(W - acc_exp)
        
        # acc is already FP32, convert to fixed-point without truncation
        acc_aligned = acc_fp32.double() * scale
        
        # new_val is W-bits, truncate to align with acc's exponent (fixed hardware behavior)
        new_val_aligned = torch.trunc(new_val_wbits.double() * scale)
        
        # Step 2: Accumulate in fixed-point
        sum_fixed = acc_aligned + new_val_aligned
        
        # Step 3: Convert back to float with specified rounding
        sum_f64 = sum_fixed / scale
        return HardwareCore.to_float32_with_rounding(sum_f64, rounding)

    @staticmethod
    def hardware_reduction_4to1(v_list, W=25, output_fp32=True, rounding=RoundStrategy.RZ, profile_timing=None):
        """
        Stage 3: 4-to-1 reduction.
        
        Hardware behavior:
        1. Find max value across all 4 elements
        2. Align all to max exponent and truncate (fixed RZ behavior)
        3. Sum in fixed-point domain
        4. If output_fp32: convert with specified rounding
           Else: keep as W-bit precision (float64 intermediate)
        """
        if profile_timing is not None:
            _profile_begin(profile_timing, "stage3.stack")
        stacked = torch.stack(v_list, dim=0)
        if profile_timing is not None:
            _profile_end(profile_timing, "stage3.stack")

        if profile_timing is not None:
            _profile_begin(profile_timing, "stage3.max_abs_frexp")
        max_val = torch.max(stacked.abs(), dim=0)[0]
        _, max_exp = torch.frexp(max_val)
        if profile_timing is not None:
            _profile_end(profile_timing, "stage3.max_abs_frexp")

        if profile_timing is not None:
            _profile_begin(profile_timing, "stage3.scale_compute")
        scale = 2.0**(W - max_exp)
        if profile_timing is not None:
            _profile_end(profile_timing, "stage3.scale_compute")

        # Accumulate in fixed-point (truncate each element - fixed hardware RZ)
        if profile_timing is not None:
            _profile_begin(profile_timing, "stage3.reduction_total")
        v_aligned_sum = torch.zeros_like(v_list[0], dtype=torch.float64)
        for i, v in enumerate(v_list):
            if profile_timing is not None:
                _profile_begin(profile_timing, f"stage3.loop[{i}].double_mul")
            mul_val = v.double() * scale
            if profile_timing is not None:
                _profile_end(profile_timing, f"stage3.loop[{i}].double_mul")

            if profile_timing is not None:
                _profile_begin(profile_timing, f"stage3.loop[{i}].trunc")
            truncated = torch.trunc(mul_val)  # RZ truncation
            if profile_timing is not None:
                _profile_end(profile_timing, f"stage3.loop[{i}].trunc")

            if profile_timing is not None:
                _profile_begin(profile_timing, f"stage3.loop[{i}].accumulate")
            v_aligned_sum += truncated
            if profile_timing is not None:
                _profile_end(profile_timing, f"stage3.loop[{i}].accumulate")

        if profile_timing is not None:
            _profile_end(profile_timing, "stage3.reduction_total")

        if profile_timing is not None:
            _profile_begin(profile_timing, "stage3.divide")
        summed_f64 = v_aligned_sum / scale
        if profile_timing is not None:
            _profile_end(profile_timing, "stage3.divide")

        if output_fp32:
            # Apply specified rounding when casting to FP32
            if profile_timing is not None:
                _profile_begin(profile_timing, "stage3.cast_to_fp32")
            out = HardwareCore.to_float32_with_rounding(summed_f64, rounding)
            if profile_timing is not None:
                _profile_end(profile_timing, "stage3.cast_to_fp32")
            return out
        else:
            # Keep as W-bit precision (intermediate for Stage 4)
            return summed_f64


class MMAEngine:
    """整合 MMA 仿真流程"""

    _last_profile_stats = {}
    _cumulative_profile_stats = defaultdict(float)
    _profile_call_count = 0

    @classmethod
    def reset_profile_stats(cls):
        cls._last_profile_stats = {}
        cls._cumulative_profile_stats = defaultdict(float)
        cls._profile_call_count = 0

    @classmethod
    def get_last_profile_stats(cls):
        return dict(cls._last_profile_stats)

    @classmethod
    def get_cumulative_profile_stats(cls):
        stats = dict(cls._cumulative_profile_stats)
        stats["profile_call_count"] = cls._profile_call_count
        return stats
    
    @staticmethod
    def stage1_inner_mma_fp16(val_a, val_b):
        M, K = val_a.shape
        N, _ = val_b.shape
        K_groups = K // 16
        a_grouped = val_a.view(M, K_groups, 16).to(torch.float16)
        b_grouped = val_b.view(N, K_groups, 16).to(torch.float16)
        partial_sum1 = torch.einsum('mgk,ngk->mgn', a_grouped, b_grouped).to(torch.float16)
        return partial_sum1.permute(0, 2, 1)

    @staticmethod
    def emulation_scaled_fp4_mm(
        a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, 
        W_stage3=25, W_stage4=25,
        stage3_rounding=RoundStrategy.RZ,
        stage4_rounding=RoundStrategy.RZ,
        m_chunk_size=128,
        enable_profile=False,
        use_triton_stage3=False,
    ):
        """NVFP4 MMA Accuracy Emulation with M-dimension chunking to avoid OOM"""
        from .utils import NVFP4Utils

        timing = defaultdict(float) if enable_profile else None
        if timing is not None:
            _profile_begin(timing, "emulation.total")

        assert K % 16 == 0, "K must be multiple of 16 (group size)"
        assert (K // 16) % 4 == 0, "G must be multiple of 4 (mma.k64 blocks)"

        G = K // 16
        num_4_groups = G // 4
        
        # Pre-process B (weight) - shared across all chunks
        if timing is not None:
            _profile_begin(timing, "preprocess_b.total")
            _profile_begin(timing, "preprocess_b.unpack")
        val_b_fp16 = NVFP4Utils.unpack_nvfp4_to_fp16(b_fp4, (N, K))
        if timing is not None:
            _profile_end(timing, "preprocess_b.unpack")
            _profile_begin(timing, "preprocess_b.deswizzle")
        s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, G).to(torch.float32)
        if timing is not None:
            _profile_end(timing, "preprocess_b.deswizzle")
            _profile_end(timing, "preprocess_b.total")

        # Pre-process A scales - will slice per chunk
        if timing is not None:
            _profile_begin(timing, "preprocess_a.total")
            _profile_begin(timing, "preprocess_a.deswizzle")
        s_a_all = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, G).to(torch.float32)
        if timing is not None:
            _profile_end(timing, "preprocess_a.deswizzle")
            _profile_begin(timing, "preprocess_a.unpack")
        val_a_fp16_all = NVFP4Utils.unpack_nvfp4_to_fp16(a_fp4, (M, K))
        if timing is not None:
            _profile_end(timing, "preprocess_a.unpack")
            _profile_end(timing, "preprocess_a.total")
        
        # M-dimension chunking to avoid OOM
        summed_results_list = []
        
        for m_start in range(0, M, m_chunk_size):
            if timing is not None:
                _profile_begin(timing, "chunk.total")
            m_end = min(m_start + m_chunk_size, M)
            curr_chunk_size = m_end - m_start

            # Slice A for current chunk
            if timing is not None:
                _profile_begin(timing, "chunk.slice")
            val_a_chunk = val_a_fp16_all[m_start:m_end, :]
            s_a_chunk = s_a_all[m_start:m_end, :]
            if timing is not None:
                _profile_end(timing, "chunk.slice")

            # STEP 1: Intra-Group Partial Sum (Lossless FP16)
            if timing is not None:
                _profile_begin(timing, "step1.total")
                _profile_begin(timing, "step1.inner_mma")
            ps1_chunk = MMAEngine.stage1_inner_mma_fp16(val_a_chunk, val_b_fp16)
            if timing is not None:
                _profile_end(timing, "step1.inner_mma")
                _profile_end(timing, "step1.total")

            # STEP 2: Apply Shared Scales (Lossless F32)
            if timing is not None:
                _profile_begin(timing, "step2.total")
                _profile_begin(timing, "step2.compute_combined_scales")
            combined_scales_chunk = s_a_chunk.unsqueeze(1) * s_b.unsqueeze(0)
            if timing is not None:
                _profile_end(timing, "step2.compute_combined_scales")
                _profile_begin(timing, "step2.apply_scales")
            scaled_partials_chunk = ps1_chunk.float() * combined_scales_chunk
            if timing is not None:
                _profile_end(timing, "step2.apply_scales")
                _profile_end(timing, "step2.total")

            # STEP 3: Group -> MMA-K-Block Reduction
            if timing is not None:
                _profile_begin(timing, "step3.total")
                _profile_begin(timing, "step3.reshape")
            scaled_partials_4grouped = scaled_partials_chunk.view(curr_chunk_size, N, num_4_groups, 4)
            v_list = [scaled_partials_4grouped[..., i] for i in range(4)]
            if timing is not None:
                _profile_end(timing, "step3.reshape")
                _profile_begin(timing, "step3.hardware_reduction")

            use_triton_path = (
                use_triton_stage3
                and _HAS_TRITON_STAGE3
                and scaled_partials_4grouped.is_cuda
                and stage3_rounding == RoundStrategy.RZ
            )
            if use_triton_path and triton_stage3_reduce4 is not None:
                if timing is not None:
                    _profile_begin(timing, "stage3.max_abs_frexp")
                max_val = torch.max(scaled_partials_4grouped.abs(), dim=-1)[0]
                _, max_exp = torch.frexp(max_val)
                if timing is not None:
                    _profile_end(timing, "stage3.max_abs_frexp")

                if timing is not None:
                    _profile_begin(timing, "stage3.triton_reduce4")
                summed_groups = triton_stage3_reduce4(
                    scaled_partials_4grouped.contiguous(), max_exp.contiguous(), W_stage3
                )
                if num_4_groups == 1:
                    summed_groups = HardwareCore.to_float32_with_rounding(summed_groups, stage3_rounding)
                if timing is not None:
                    _profile_end(timing, "stage3.triton_reduce4")
            else:
                if num_4_groups == 1:
                    summed_groups = HardwareCore.hardware_reduction_4to1(
                        v_list, W=W_stage3, output_fp32=True, rounding=stage3_rounding, profile_timing=timing
                    )
                else:
                    summed_groups = HardwareCore.hardware_reduction_4to1(
                        v_list, W=W_stage3, output_fp32=False, rounding=stage3_rounding, profile_timing=timing
                    )
            if timing is not None:
                _profile_end(timing, "step3.hardware_reduction")
                _profile_end(timing, "step3.total")

            # STEP 4: Inter-Block Accumulation
            if timing is not None:
                _profile_begin(timing, "step4.total")
            if num_4_groups == 1:
                if timing is not None:
                    _profile_begin(timing, "step4.squeeze")
                summed_chunk = summed_groups.squeeze(-1)
                if timing is not None:
                    _profile_end(timing, "step4.squeeze")
            else:
                if timing is not None:
                    _profile_begin(timing, "step4.init_cast")
                acc = HardwareCore.to_float32_with_rounding(summed_groups[..., 0], stage4_rounding)
                if timing is not None:
                    _profile_end(timing, "step4.init_cast")
                for i in range(1, num_4_groups):
                    if timing is not None:
                        _profile_begin(timing, "step4.hardware_add_wbits")
                    acc = HardwareCore.hardware_add_wbits(
                        acc, summed_groups[..., i], W=W_stage4, rounding=stage4_rounding
                    )
                    if timing is not None:
                        _profile_end(timing, "step4.hardware_add_wbits")
                summed_chunk = acc
            if timing is not None:
                _profile_end(timing, "step4.total")

            if timing is not None:
                _profile_begin(timing, "chunk.append")
            summed_results_list.append(summed_chunk)
            if timing is not None:
                _profile_end(timing, "chunk.append")
            
            # Explicit cleanup to free memory
            del ps1_chunk, combined_scales_chunk, scaled_partials_chunk, scaled_partials_4grouped, v_list
            del summed_groups
            if num_4_groups > 1:
                del acc
            if timing is not None:
                _profile_end(timing, "chunk.total")
        
        # Concatenate all chunks
        if timing is not None:
            _profile_begin(timing, "post.concat")
        summed_result = torch.cat(summed_results_list, dim=0)
        if timing is not None:
            _profile_end(timing, "post.concat")

        # Cleanup pre-processed tensors
        if timing is not None:
            _profile_begin(timing, "post.cleanup")
        del val_a_fp16_all, val_b_fp16, s_a_all, s_b
        if timing is not None:
            _profile_end(timing, "post.cleanup")

        # STEP 5: Final Scaling & Precision Cast
        if timing is not None:
            _profile_begin(timing, "step5.total")
            _profile_begin(timing, "step5.alpha_item")
        alpha_val = alpha_tensor.item()
        if timing is not None:
            _profile_end(timing, "step5.alpha_item")
            _profile_begin(timing, "step5.mul_cast")
        output = (summed_result * alpha_val).to(torch.float16)
        if timing is not None:
            _profile_end(timing, "step5.mul_cast")
            _profile_end(timing, "step5.total")
            _profile_end(timing, "emulation.total")
            stats = dict(timing)
            MMAEngine._last_profile_stats = stats
            MMAEngine._profile_call_count += 1
            for k, v in stats.items():
                MMAEngine._cumulative_profile_stats[k] += v
        return output

    @staticmethod
    def emulation_scaled_fp4_mm_debug(
        a_fp4, b_fp4, scale_a, scale_b, alpha_tensor, M, N, K, 
        W_stage3=25, W_stage4=25,
        stage3_rounding=RoundStrategy.RZ,
        stage4_rounding=RoundStrategy.RZ
    ):
        """Debug version"""
        from .utils import NVFP4Utils
        
        assert K % 16 == 0, "K must be multiple of 16 (group size)"
        assert (K // 16) % 4 == 0, "G must be multiple of 4 (mma.k64 blocks)"

        # STEP 1-2
        val_a_fp16 = NVFP4Utils.unpack_nvfp4_to_fp16(a_fp4, (M, K))
        val_b_fp16 = NVFP4Utils.unpack_nvfp4_to_fp16(b_fp4, (N, K))
        ps1 = MMAEngine.stage1_inner_mma_fp16(val_a_fp16, val_b_fp16)
        
        G = K // 16
        s_a = pseudo_quant.swizzled_to_linear_128_4(scale_a, M, G).to(torch.float32)
        s_b = pseudo_quant.swizzled_to_linear_128_4(scale_b, N, G).to(torch.float32)
        combined_scales = s_a.unsqueeze(1) * s_b.unsqueeze(0)
        scaled_partials = ps1.float() * combined_scales

        # STEP 3: Stage 3 with W_stage3
        num_4_groups = G // 4
        scaled_partials_4grouped = scaled_partials.view(M, N, num_4_groups, 4)
        v_list = [scaled_partials_4grouped[..., i] for i in range(4)]
        
        if num_4_groups == 1:
            summed_groups = HardwareCore.hardware_reduction_4to1(
                v_list, W=W_stage3, output_fp32=True, rounding=stage3_rounding
            )
        else:
            summed_groups = HardwareCore.hardware_reduction_4to1(
                v_list, W=W_stage3, output_fp32=False, rounding=stage3_rounding
            )
        
        debug_info = {
            "num_blocks": num_4_groups,
            "block_sums": [summed_groups[..., i] for i in range(num_4_groups)],
            "W_stage3": W_stage3,
            "W_stage4": W_stage4,
            "stage3_rounding": stage3_rounding.value,
            "stage4_rounding": stage4_rounding.value,
        }
        
        # STEP 4
        if num_4_groups == 1:
            summed_result = summed_groups.squeeze(-1)
            stage4_sum_raw = None
        else:
            acc = HardwareCore.to_float32_with_rounding(summed_groups[..., 0], stage4_rounding)
            for i in range(1, num_4_groups):
                acc = HardwareCore.hardware_add_wbits(
                    acc, summed_groups[..., i], W=W_stage4, rounding=stage4_rounding
                )
            summed_result = acc
            stage4_sum_raw = summed_result.clone()
        
        debug_info["stage4_sum_raw"] = stage4_sum_raw
        
        # STEP 5
        alpha_val = alpha_tensor.item()
        final_result = (summed_result * alpha_val).to(torch.float16)
        debug_info["alpha"] = alpha_val
        
        return final_result, debug_info
