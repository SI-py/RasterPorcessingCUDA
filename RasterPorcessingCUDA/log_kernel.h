#pragma once
#include <cuda_runtime.h>

void log_transform_on_gpu(const float* d_in, float* d_out, int total_pixels);
