#include "zonal_from_mask.h"
#include <cuda_runtime.h>
#include <cmath>

__global__ void mask_zonal_kernel(
    const float* d_ndvi,
    const float* d_mask,
    int total_pixels,
    float* d_sum_in,
    int* d_count_in,
    float* d_sum_out,
    int* d_count_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels)
        return;

    float val = d_ndvi[idx];
    if (isnan(val))
        return;

    if (d_mask[idx] > 0.5f) {
        atomicAdd(d_sum_in, val);
        atomicAdd(d_count_in, 1);
    }
    else {
        atomicAdd(d_sum_out, val);
        atomicAdd(d_count_out, 1);
    }
}

void compute_zonal_from_mask_gpu(
    const float* d_ndvi,
    const float* d_mask,
    int total_pixels,
    float& mean_in,
    float& mean_out)
{
    float* d_sum_in = nullptr;
    float* d_sum_out = nullptr;
    int* d_count_in = nullptr;
    int* d_count_out = nullptr;

    cudaMalloc(&d_sum_in, sizeof(float));
    cudaMalloc(&d_sum_out, sizeof(float));
    cudaMalloc(&d_count_in, sizeof(int));
    cudaMalloc(&d_count_out, sizeof(int));

    cudaMemset(d_sum_in, 0, sizeof(float));
    cudaMemset(d_sum_out, 0, sizeof(float));
    cudaMemset(d_count_in, 0, sizeof(int));
    cudaMemset(d_count_out, 0, sizeof(int));

    int blockSize = 256;
    int gridSize = (total_pixels + blockSize - 1) / blockSize;
    mask_zonal_kernel << <gridSize, blockSize >> > (
        d_ndvi, d_mask, total_pixels,
        d_sum_in, d_count_in, d_sum_out, d_count_out
        );

    cudaDeviceSynchronize();

    float sum_in = 0.0f, sum_out = 0.0f;
    int count_in = 0, count_out = 0;
    cudaMemcpy(&sum_in, d_sum_in, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_out, d_sum_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count_in, d_count_in, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&count_out, d_count_out, sizeof(int), cudaMemcpyDeviceToHost);

    mean_in = (count_in > 0) ? sum_in / count_in : NAN;
    mean_out = (count_out > 0) ? sum_out / count_out : NAN;

    cudaFree(d_sum_in);
    cudaFree(d_sum_out);
    cudaFree(d_count_in);
    cudaFree(d_count_out);
}
