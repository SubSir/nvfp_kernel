#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

#include "launch_bounds_utils.h"

#define ELTS_PER_THREAD 8
constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;

namespace vllm {

// Convert PyTorch cpp type to CUDA type
template <typename T>
struct CUDATypeConverter {
  using Type = T;
};

template <>
struct CUDATypeConverter<at::Half> {
  using Type = half;
};

template <>
struct CUDATypeConverter<at::BFloat16> {
  using Type = __nv_bfloat16;
};

// Get type2 from type or vice versa (applied to half and bfloat16)
template <typename T>
struct TypeConverter {
  using Type = half2;
};  // keep for generality

template <>
struct TypeConverter<half2> {
  using Type = half;
};

template <>
struct TypeConverter<half> {
  using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

// Define a 16 bytes packed data type.
template <class Type>
struct PackedVec {
  typename TypeConverter<Type>::Type elts[4];
};

template <>
struct PackedVec<__nv_fp8_e4m3> {
  __nv_fp8x2_e4m3 elts[8];
};

// Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
        "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
  return val;
}

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
}

template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(int rowIdx, int colIdx,
                                                       int numCols,
                                                       SFType* SFout) {
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 ||
                CVT_FP4_NUM_THREADS_PER_SF == 2);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0) {
    // SF vector index (16 elements share one SF in the K dimension).
    int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
    int32_t mIdx = rowIdx;

    // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
    // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]

    int32_t mTileIdx = mIdx / (32 * 4);
    // SF vector size 16.
    int factor = CVT_FP4_SF_VEC_SIZE * 4;
    int32_t numKTiles = (numCols + factor - 1) / factor;
    int64_t mTileStride = numKTiles * 32 * 4 * 4;

    int32_t kTileIdx = (kIdx / 4);
    int64_t kTileStride = 32 * 4 * 4;

    // M tile layout [32, 4] is column-major.
    int32_t outerMIdx = (mIdx % 32);
    int64_t outerMStride = 4 * 4;

    int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
    int64_t innerMStride = 4;

    int32_t innerKIdx = (kIdx % 4);
    int64_t innerKStride = 1;

    // Compute the global offset.
    int64_t SFOffset = mTileIdx * mTileStride + kTileIdx * kTileStride +
                       outerMIdx * outerMStride + innerMIdx * innerMStride +
                       innerKIdx * innerKStride;

    return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
  }
  return nullptr;
}

// Fast reciprocal.
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Quantizes the provided PackedVec into the uint32_t output
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal,
                                         uint8_t* SFout) {
  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(vec.elts[0]);

// Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  // Get the absolute maximum among all 16 values (two threads).
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  // Get the final absolute maximum values.
  float vecMax;
  if constexpr (std::is_same_v<Type, __half>) {
    vecMax = __half2float(__hmax(localMax.x, localMax.y));
  } else {
    static_assert(std::is_same_v<Type, __nv_bfloat16>, "Unsupported type");
    vecMax = __bfloat162float(__hmax(localMax.x, localMax.y));
  }

  // Get the SF (max value of the vector / max value of e2m1).
  // maximum value of e2m1 = 6.0.
  // TODO: use half as compute data type.
  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  // 8 bits representation of the SF.
  uint8_t fp8SFVal;
  // Write the SF to global memory (STG.8).
  if constexpr (UE8M0_SF) {
    // Extract the 8 exponent bits from float32.
    // float 32bits = 1 sign bit + 8 exponent bits + 23 mantissa bits.
    uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
    fp8SFVal = tmp & 0xff;
    // Convert back to fp32.
    reinterpret_cast<uint32_t&>(SFValue) = tmp << 23;
  } else {
    // Here SFValue is always positive, so E4M3 is the same as UE4M3.
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
    // Convert back to fp32.
    SFValue = float(tmp);
  }
  // Get the output scale.
  // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) *
  //                       reciprocal(SFScaleVal))
  float outputScale =
      SFValue != 0 ? reciprocal_approximate_ftz(
                         SFValue * reciprocal_approximate_ftz(SFScaleVal))
                   : 0.0f;

  if (SFout) {
    // Write the SF to global memory (STG.8).
    *SFout = fp8SFVal;
  }

  // Convert the input to float.
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  // Write the e2m1 values to global memory.
  return e2m1Vec;
}

// Decode two packed e2m1 codes (one byte: low nibble -> .x, high nibble -> .y)
// back to fp32, using the Blackwell hardware converter (cvt.rn.f16x2.e2m1x2).
// Branch-free and exact -- the decoded values match the e2m1 grid bit-for-bit,
// so the residual matches what the downstream GEMM consumes.
inline __device__ float2 e2m1x2_to_float2(uint32_t byte) {
  __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(
      static_cast<__nv_fp4x2_storage_t>(byte & 0xFFu), __NV_E2M1);
  return __half22float2(*reinterpret_cast<__half2*>(&raw));
}

// Same as cvt_warp_fp16_to_fp4, but additionally returns the quantization
// residual (original - dequant(quantized)) in `residual` (in the input dtype),
// so a second NVFP4 level can be applied. `residual` may be null. The residual
// is computed by decoding the exact e2m1 codes this function emits, so it
// matches bit-for-bit what the downstream GEMM consumes.
template <class Type, bool UE8M0_SF = false>
__device__ uint32_t cvt_warp_fp16_to_fp4_residual(PackedVec<Type>& vec,
                                                  float SFScaleVal,
                                                  uint8_t* SFout,
                                                  PackedVec<Type>* residual) {
  auto localMax = __habs2(vec.elts[0]);
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }
  localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
  float vecMax;
  if constexpr (std::is_same_v<Type, __half>) {
    vecMax = __half2float(__hmax(localMax.x, localMax.y));
  } else {
    static_assert(std::is_same_v<Type, __nv_bfloat16>, "Unsupported type");
    vecMax = __bfloat162float(__hmax(localMax.x, localMax.y));
  }

  float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  uint8_t fp8SFVal;
  if constexpr (UE8M0_SF) {
    uint32_t tmp = reinterpret_cast<uint32_t&>(SFValue) >> 23;
    fp8SFVal = tmp & 0xff;
    reinterpret_cast<uint32_t&>(SFValue) = tmp << 23;
  } else {
    __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
    reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
    SFValue = float(tmp);
  }
  // outputScale maps original units -> e2m1 grid; totalScale is its inverse,
  // i.e. e2m1 grid value -> original units (== stored_block_scale / globalScale).
  float outputScale =
      SFValue != 0 ? reciprocal_approximate_ftz(
                         SFValue * reciprocal_approximate_ftz(SFScaleVal))
                   : 0.0f;
  float totalScale = SFValue * reciprocal_approximate_ftz(SFScaleVal);

  if (SFout) {
    *SFout = fp8SFVal;
  }

  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];
#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    if constexpr (std::is_same_v<Type, half>) {
      fp2Vals[i] = __half22float2(vec.elts[i]);
    } else {
      fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
    }
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }
  uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);

  if (residual != nullptr) {
#pragma unroll
    for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
      // byte i of e2m1Vec packs element 2i (low nibble) and 2i+1 (high nibble).
      // q.x = dequant(elt 2i), q.y = dequant(elt 2i+1).
      float2 q = e2m1x2_to_float2(e2m1Vec >> (8 * i));
      float2 o;
      if constexpr (std::is_same_v<Type, half>) {
        o = __half22float2(vec.elts[i]);
      } else {
        o = __bfloat1622float2(vec.elts[i]);
      }
      float reslo = o.x - q.x * totalScale;
      float reshi = o.y - q.y * totalScale;
      if constexpr (std::is_same_v<Type, half>) {
        residual->elts[i] = __floats2half2_rn(reslo, reshi);
      } else {
        residual->elts[i] = __floats2bfloat162_rn(reslo, reshi);
      }
    }
  }
  return e2m1Vec;
}

// Fused two-level (residual) NVFP4 quantization.
// Reads x once (numRows x numCols) and writes a stacked (2*numRows x numCols)
// FP4 tensor: rows [0, numRows)        = NVFP4(x)
//             rows [numRows, 2*numRows) = NVFP4(x - dequant(NVFP4(x)))
// Both levels share the SAME global scale (SFScale), so the downstream GEMM can
// treat the result as one (2*numRows x numCols) matrix with a single alpha.
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4_residual(int32_t numRows, int32_t numCols, Type const* in,
                             float const* SFScale, uint32_t* out,
                             uint32_t* SFout) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
  int const colsPacked = numCols / CVT_FP4_ELTS_PER_THREAD;

  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < colsPacked; colIdx += blockDim.x) {
      int64_t inOffset = static_cast<int64_t>(rowIdx) * colsPacked + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];

      // Level 1: quantize x into row `rowIdx`, capturing the residual.
      auto sf_x = cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                                     CVT_FP4_NUM_THREADS_PER_SF>(
          rowIdx, colIdx, numCols, SFout);
      PackedVec residual;
      out[inOffset] = cvt_warp_fp16_to_fp4_residual<Type, UE8M0_SF>(
          in_vec, SFScaleVal, sf_x, &residual);

      // Level 2: quantize the residual into row `numRows + rowIdx`.
      int const rRow = numRows + rowIdx;
      int64_t const rOffset = static_cast<int64_t>(rRow) * colsPacked + colIdx;
      auto sf_r = cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                                     CVT_FP4_NUM_THREADS_PER_SF>(
          rRow, colIdx, numCols, SFout);
      out[rOffset] = cvt_warp_fp16_to_fp4_residual<Type, UE8M0_SF>(
          residual, SFScaleVal, sf_r, nullptr);
    }
  }
}

// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                    float const* SFScale, uint32_t* out, uint32_t* SFout) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

  // Input tensor row/col loops.
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD;
         colIdx += blockDim.x) {
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      // Get the output tensor offset.
      // Same as inOffset because 8 elements are packed into one uint32_t.
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, numCols, SFout);

      out_pos =
          cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
    }
  }
}
}  // namespace vllm
