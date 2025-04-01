#pragma once


static const int WINDOW_RADIUS = 1;

void smoothing_naive_gpu(const float* d_input,
    float* d_output,
    int width,
    int height);

void smoothing_shared_gpu(const float* d_input,
    float* d_output,
    int width,
    int height);
