// Minimal shim replacing vLLM-internal headers (cutlass_extensions/common.hpp,
// core/math.hpp) that the sm100 nvfp4 GEMM kernel was ported from. Provides only
// the two symbols that kernel needs: CUTLASS_CHECK and next_pow_2.
#pragma once

#include <climits>
#include <torch/all.h>
#include "cutlass/cutlass.h"

#ifndef CUTLASS_CHECK
#define CUTLASS_CHECK(status)                       \
  {                                                 \
    cutlass::Status error = status;                 \
    TORCH_CHECK(error == cutlass::Status::kSuccess, \
                cutlassGetStatusString(error));     \
  }
#endif

inline constexpr uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}
