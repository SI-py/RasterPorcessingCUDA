#pragma once
#include <cuda_runtime.h>

void compute_sobel_on_gpu(const float* d_input,
    float* d_output,
    int width,
    int height);