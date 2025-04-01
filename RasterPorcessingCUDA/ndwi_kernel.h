#pragma once
#include <cuda_runtime.h>

void compute_ndwi_on_gpu(const float* d_green_data,
    const float* d_nir_data,
    float* d_ndwi_data,
    int total_pixels);

