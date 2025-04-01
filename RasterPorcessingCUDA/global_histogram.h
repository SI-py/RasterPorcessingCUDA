#pragma once
#include <cuda_runtime.h>

void compute_histogram_on_gpu(const float* d_input, int* d_hist, int n, int num_bins, float min_val, float max_val);
