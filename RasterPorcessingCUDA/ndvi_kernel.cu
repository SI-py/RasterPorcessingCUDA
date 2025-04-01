#include "ndvi_kernel.h"
#include <cuda_runtime.h>
#include <iostream>

// ядро, выполн€ющее NDVI = (NIR - RED) / (NIR + RED)
__global__ void compute_ndvi_kernel(const float* d_red_data,
    const float* d_nir_data,
    float* d_ndvi_data,
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_pixels) {
        float red_value = d_red_data[idx];
        float nir_value = d_nir_data[idx];
        float sum = red_value + nir_value;

        if (sum == 0.0f) {
            d_ndvi_data[idx] = 0.0f;
        }
        else {
            d_ndvi_data[idx] = (nir_value - red_value) / sum;
        }
    }
}

// ќбЄртка дл€ запуска €дра
void compute_ndvi_on_gpu(const float* d_red_data,
    const float* d_nir_data,
    float* d_ndvi_data,
    int total_pixels)
{
    int threads_per_block = 256;
    int blocks = (total_pixels + threads_per_block - 1) / threads_per_block;

    compute_ndvi_kernel << <blocks, threads_per_block >> > (d_red_data,
        d_nir_data,
        d_ndvi_data,
        total_pixels);

    cudaDeviceSynchronize();
}