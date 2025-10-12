// binding.cpp
#include <torch/extension.h>

#include "nvfp4_quant.h"
#include "nvfp4_scaled_mm_entry.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scaled_fp4_quant_sm1xxa", &scaled_fp4_quant_sm1xxa);
  m.def("cutlass_scaled_fp4_mm", &cutlass_scaled_fp4_mm);
}
