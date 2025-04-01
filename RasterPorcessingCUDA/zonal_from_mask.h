#pragma once
#include <cuda_runtime.h>

void compute_zonal_from_mask_gpu(
    const float* d_ndvi,
    const float* d_mask,
    int total_pixels,
    float& mean_in,
    float& mean_out);