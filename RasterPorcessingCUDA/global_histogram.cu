#include "global_histogram.h"
#include <cuda_runtime.h>

__global__ void histogram_kernel(const float* input, int* hist, int n, int num_bins, float min_val, float max_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float value = input[idx];
        float norm = (value - min_val) / (max_val - min_val);
        int bin = (int)(norm * num_bins);
        if (bin < 0) bin = 0;
        if (bin >= num_bins) bin = num_bins - 1;
        atomicAdd(&hist[bin], 1);
    }
}

void compute_histogram_on_gpu(const float* d_input, int* d_hist, int n, int num_bins, float min_val, float max_val)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    histogram_kernel << <blocks, threads >> > (d_input, d_hist, n, num_bins, min_val, max_val);
    cudaDeviceSynchronize();
}
