#include "ndwi_kernel.h"
#include <cuda_runtime.h>

__global__ void ndwi_kernel(const float* d_green, const float* d_nir,
    float* d_ndwi, int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        float g = d_green[idx];
        float n = d_nir[idx];
        float sum = g + n;
        if (sum == 0.0f) {
            d_ndwi[idx] = 0.0f;
        }
        else {
            d_ndwi[idx] = (g - n) / sum;
        }
    }
}

void compute_ndwi_on_gpu(const float* d_green_data,
    const float* d_nir_data,
    float* d_ndwi_data,
    int total_pixels)
{
    int threads_per_block = 256;
    int blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

    ndwi_kernel << <blocks, threads_per_block >> > (d_green_data, d_nir_data, d_ndwi_data, total_pixels);
    cudaDeviceSynchronize();
}
