#pragma once

#include <cuda_runtime.h>

void compute_ndvi_on_gpu(const float* d_red_data,
    const float* d_nir_data,
    float* d_ndvi_data,
    int total_pixels);