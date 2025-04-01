#include "log_kernel.h"
#include <cuda_runtime.h>
#include <math.h>

__global__ void log_kernel(const float* d_in, float* d_out, int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        float val = d_in[idx];
        if (val < 0.0f) val = 0.0f;
        d_out[idx] = logf(1.0f + val);
    }
}

void log_transform_on_gpu(const float* d_in, float* d_out, int total_pixels)
{
    int threads_per_block = 256;
    int blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

    log_kernel << <blocks, threads_per_block >> > (d_in, d_out, total_pixels);
    cudaDeviceSynchronize();
}
