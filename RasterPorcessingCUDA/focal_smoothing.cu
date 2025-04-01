#include "focal_smoothing.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ========================== Наивный ==============================

__global__ void smoothing_naive_kernel(const float* input, float* output,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < (width - 1) &&
        y >= 1 && y < (height - 1))
    {
        float sum = 0.0f;
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int xx = x + dx;
                int yy = y + dy;
                sum += input[yy * width + xx];
            }
        }
        output[y * width + x] = sum / 9.0f;
    }
    else
    {
        if (x < width && y < height)
        {
            output[y * width + x] = input[y * width + x];
        }
    }
}

void smoothing_naive_gpu(const float* d_input,
    float* d_output,
    int width,
    int height)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    smoothing_naive_kernel << <grid, block >> > (d_input, d_output, width, height);
    cudaDeviceSynchronize();
}


// ========================== Shared Memory ========================
__global__ void smoothing_shared_kernel(const float* input, float* output,
    int width, int height)
{
    const int TILE_W = blockDim.x;
    const int TILE_H = blockDim.y;
    __shared__ float tile[(16 + 2)][(16 + 2)];

    int x = blockIdx.x * TILE_W + threadIdx.x;
    int y = blockIdx.y * TILE_H + threadIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (x < width && y < height)
    {
        tile[ty + 1][tx + 1] = input[y * width + x];
    }
    else
    {
        tile[ty + 1][tx + 1] = 0.0f;
    }

    if (tx == 0 && x > 0 && y < height)
    {
        tile[ty + 1][0] = input[y * width + (x - 1)];
    }
    if (tx == TILE_W - 1 && (x < width - 1) && y < height)
    {
        tile[ty + 1][TILE_W + 1] = input[y * width + (x + 1)];
    }
    if (ty == 0 && y > 0 && x < width)
    {
        tile[0][tx + 1] = input[(y - 1) * width + x];
    }
    if (ty == TILE_H - 1 && (y < height - 1) && x < width)
    {
        tile[TILE_H + 1][tx + 1] = input[(y + 1) * width + x];
    }


    __syncthreads();

    if (x > 0 && x < (width - 1) &&
        y > 0 && y < (height - 1))
    {
        float sum = 0.0f;
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                sum += tile[(ty + 1) + dy][(tx + 1) + dx];
            }
        }
        output[y * width + x] = sum / 9.0f;
    }
    else if (x < width && y < height)
    {
        output[y * width + x] = input[y * width + x];
    }
}

void smoothing_shared_gpu(const float* d_input,
    float* d_output,
    int width,
    int height)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    smoothing_shared_kernel << <grid, block >> > (d_input, d_output, width, height);
    cudaDeviceSynchronize();
}
