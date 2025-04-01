#include "sobel_kernel.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#define BLOCK_SIZE 16

__global__ void sobel_kernel(const float* input, float* output, int width, int height)
{
    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int y = blockIdx.y * BLOCK_SIZE + threadIdx.y;

    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;

    if (x < width && y < height) {
        tile[ty][tx] = input[y * width + x];
    }
    else {
        tile[ty][tx] = 0.0f;
    }

    if (threadIdx.x == 0 && x > 0) {
        tile[ty][0] = input[y * width + (x - 1)];
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && x < width - 1) {
        tile[ty][tx + 1] = input[y * width + (x + 1)];
    }
    if (threadIdx.y == 0 && y > 0) {
        tile[0][tx] = input[(y - 1) * width + x];
    }
    if (threadIdx.y == BLOCK_SIZE - 1 && y < height - 1) {
        tile[ty + 1][tx] = input[(y + 1) * width + x];
    }

    if (threadIdx.x == 0 && threadIdx.y == 0 && x > 0 && y > 0) {
        tile[0][0] = input[(y - 1) * width + (x - 1)];
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == 0 && x < width - 1 && y > 0) {
        tile[0][tx + 1] = input[(y - 1) * width + (x + 1)];
    }
    if (threadIdx.x == 0 && threadIdx.y == BLOCK_SIZE - 1 && x > 0 && y < height - 1) {
        tile[ty + 1][0] = input[(y + 1) * width + (x - 1)];
    }
    if (threadIdx.x == BLOCK_SIZE - 1 && threadIdx.y == BLOCK_SIZE - 1 && x < width - 1 && y < height - 1) {
        tile[ty + 1][tx + 1] = input[(y + 1) * width + (x + 1)];
    }

    __syncthreads();

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float gx = -tile[ty - 1][tx - 1] + tile[ty - 1][tx + 1]
            - 2.0f * tile[ty][tx - 1] + 2.0f * tile[ty][tx + 1]
            - tile[ty + 1][tx - 1] + tile[ty + 1][tx + 1];
        float gy = tile[ty - 1][tx - 1] + 2.0f * tile[ty - 1][tx] + tile[ty - 1][tx + 1]
            - tile[ty + 1][tx - 1] - 2.0f * tile[ty + 1][tx] - tile[ty + 1][tx + 1];
        float magnitude = sqrtf(gx * gx + gy * gy);
        output[y * width + x] = magnitude;
    }
    else if (x < width && y < height) {
        output[y * width + x] = input[y * width + x];
    }
}

void compute_sobel_on_gpu(const float* d_input, float* d_output, int width, int height)
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    sobel_kernel << <grid, block >> > (d_input, d_output, width, height);
    cudaDeviceSynchronize();
}
