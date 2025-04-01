#include "threshold_kernel.h"
#include <cuda_runtime.h>

__global__ void threshold_kernel(const float* d_in, float* d_out, int total_pixels, float threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        float val = d_in[idx];
        d_out[idx] = (val > threshold) ? 1.0f : 0.0f;
    }
}

void apply_threshold_on_gpu(const float* d_in, float* d_out, int total_pixels, float threshold)
{
    int threads_per_block = 256;
    int blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

    threshold_kernel << <blocks, threads_per_block >> > (d_in, d_out, total_pixels, threshold);
    cudaDeviceSynchronize();
}
