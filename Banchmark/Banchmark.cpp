#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>  
#include <chrono>  
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <gdal_priv.h>
#include "sentinel_utils.h"  
#include "ndvi_kernel.h"         // GPU‑функция для NDVI
#include "ndwi_kernel.h"         // GPU‑функция для NDWI
#include "threshold_kernel.h"    // GPU‑функция для пороговой маски
#include "log_kernel.h"          // GPU‑функция для логарифмического преобразования
#include "focal_smoothing.h"     // GPU‑функции для сглаживания
#include "zonal_from_mask.h"     // GPU‑функция для зональных операций
#include "sobel_kernel.h"        // GPU‑функция для оператора Собеля
#include "global_histogram.h"    // GPU‑функция для гистограммы
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static CPLErr write_float_tiff(const std::string& output_path,
    const float* data,
    int width,
    int height,
    double* geo_transform,
    const char* projection)
{
    return write_ndvi_result(output_path, data, width, height, geo_transform, projection);
}


void compute_ndvi_on_cpu(const float* red, const float* nir, float* result, int total_pixels) {
    for (int i = 0; i < total_pixels; i++) {
        float sum = red[i] + nir[i];
        result[i] = (fabs(sum) < 1e-6f) ? NAN : (nir[i] - red[i]) / sum;
    }
}

void compute_ndwi_on_cpu(const float* green, const float* nir, float* result, int total_pixels) {
    for (int i = 0; i < total_pixels; i++) {
        float sum = green[i] + nir[i];
        result[i] = (fabs(sum) < 1e-6f) ? NAN : (green[i] - nir[i]) / sum;
    }
}

void apply_threshold_on_cpu(const float* input, float* result, int total_pixels, float threshold_value) {
    for (int i = 0; i < total_pixels; i++) {
        result[i] = (input[i] > threshold_value) ? input[i] : 0.0f;
    }
}

void log_transform_on_cpu(const float* input, float* result, int total_pixels) {
    for (int i = 0; i < total_pixels; i++) {
        result[i] = std::log(input[i] + 1e-6f);
    }
}

void smoothing_naive_cpu(const float* input, float* result, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            int count = 0;
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    int xx = x + i;
                    int yy = y + j;
                    if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                        sum += input[yy * width + xx];
                        count++;
                    }
                }
            }
            result[y * width + x] = sum / count;
        }
    }
}

void smoothing_shared_cpu(const float* input, float* result, int width, int height) {
    smoothing_naive_cpu(input, result, width, height);
}

void compute_sobel_on_cpu(const float* input, float* result, int width, int height) {
    std::fill(result, result + width * height, 0.0f);
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float Gx = -input[(y - 1) * width + (x - 1)] + input[(y - 1) * width + (x + 1)]
                - 2 * input[y * width + (x - 1)] + 2 * input[y * width + (x + 1)]
                - input[(y + 1) * width + (x - 1)] + input[(y + 1) * width + (x + 1)];
            float Gy = -input[(y - 1) * width + (x - 1)] - 2 * input[(y - 1) * width + x] - input[(y - 1) * width + (x + 1)]
                + input[(y + 1) * width + (x - 1)] + 2 * input[(y + 1) * width + x] + input[(y + 1) * width + (x + 1)];
            result[y * width + x] = std::sqrt(Gx * Gx + Gy * Gy);
        }
    }
}

void compute_histogram_on_cpu(const float* input, int total_pixels, int num_bins,
    float min_val, float max_val, int* histogram) {
    std::fill(histogram, histogram + num_bins, 0);
    float binSize = (max_val - min_val) / num_bins;
    for (int i = 0; i < total_pixels; i++) {
        int bin = static_cast<int>((input[i] - min_val) / binSize);
        if (bin < 0) bin = 0;
        if (bin >= num_bins) bin = num_bins - 1;
        histogram[bin]++;
    }
}

void compute_zonal_from_mask_cpu(const float* ndvi, const float* mask, int total_pixels,
    float& mean_in, float& mean_out) {
    double sum_in = 0.0, sum_out = 0.0;
    int count_in = 0, count_out = 0;
    for (int i = 0; i < total_pixels; i++) {
        float val = ndvi[i];
        if (std::isnan(val)) continue;
        if (mask[i] > 0.5f) {
            sum_in += val;
            count_in++;
        }
        else {
            sum_out += val;
            count_out++;
        }
    }
    mean_in = (count_in > 0) ? (float)(sum_in / count_in) : NAN;
    mean_out = (count_out > 0) ? (float)(sum_out / count_out) : NAN;
}

int main(int argc, char* argv[])
{
    SetConsoleOutputCP(CP_UTF8);

    if (argc < 2) {
        std::cerr << "Использование: " << argv[0]
            << " <input_tif> [--op=threshold/ndwi/log/ndvi/smooth_naive/smooth_shared/sobel/histogram/ndvi_sobel_zonal]"
            << std::endl;
        return 1;
    }

    std::string input_path = argv[1];

    std::string operation = "ndvi";
    for (int i = 2; i < argc; i++) {
        if (strncmp(argv[i], "--op=", 5) == 0) {
            operation = std::string(argv[i] + 5);
        }
    }
    std::cout << "[INFO] operation = " << operation << "\n";

    GDALAllRegister();
    GDALDataset* input_dataset = (GDALDataset*)GDALOpen(input_path.c_str(), GA_ReadOnly);
    if (!input_dataset) {
        std::cerr << "Ошибка: не удалось открыть " << input_path << "\n";
        return 1;
    }

    int width = input_dataset->GetRasterXSize();
    int height = input_dataset->GetRasterYSize();
    int total_pixels = width * height;
    double geo_transform[6];
    input_dataset->GetGeoTransform(geo_transform);
    const char* projection_ref = input_dataset->GetProjectionRef();

    bool need_two_channels = (operation == "ndvi" || operation == "ndvi_sobel_zonal" || operation == "ndwi");

    std::vector<float> channel1(total_pixels, 0.0f);
    std::vector<float> channel2(total_pixels, 0.0f);
    std::vector<float> result_cpu(total_pixels, 0.0f);

    if (need_two_channels) {
        if (operation == "ndvi" || operation == "ndvi_sobel_zonal") {
            int red_band = 3, nir_band = 4;
            float* temp = nullptr;
            read_sentinel_band(input_dataset, red_band, temp, width, height);
            std::copy(temp, temp + total_pixels, channel1.begin());
            delete[] temp;
            float* temp2 = nullptr;
            read_sentinel_band(input_dataset, nir_band, temp2, width, height);
            std::copy(temp2, temp2 + total_pixels, channel2.begin());
            delete[] temp2;
        }
        else if (operation == "ndwi") {
            int green_band = 2, nir_band = 4;
            float* temp = nullptr;
            read_sentinel_band(input_dataset, green_band, temp, width, height);
            std::copy(temp, temp + total_pixels, channel1.begin());
            delete[] temp;
            float* temp2 = nullptr;
            read_sentinel_band(input_dataset, nir_band, temp2, width, height);
            std::copy(temp2, temp2 + total_pixels, channel2.begin());
            delete[] temp2;
        }
    }
    else {
        int single_band = 1;
        float* temp = nullptr;
        read_sentinel_band(input_dataset, single_band, temp, width, height);
        std::copy(temp, temp + total_pixels, channel1.begin());
        delete[] temp;
    }
    GDALClose(input_dataset);

    const int iterations = 100000;

    // ===== CPU BENCHMARK =====
    std::vector<float> cpu_result(total_pixels, 0.0f);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        if (operation == "ndvi") {
            compute_ndvi_on_cpu(channel1.data(), channel2.data(), cpu_result.data(), total_pixels);
        }
        else if (operation == "ndwi") {
            compute_ndwi_on_cpu(channel1.data(), channel2.data(), cpu_result.data(), total_pixels);
        }
        else if (operation == "threshold") {
            float threshold_value = 0.3f;
            apply_threshold_on_cpu(channel1.data(), cpu_result.data(), total_pixels, threshold_value);
        }
        else if (operation == "log") {
            log_transform_on_cpu(channel1.data(), cpu_result.data(), total_pixels);
        }
        else if (operation == "smooth_naive") {
            smoothing_naive_cpu(channel1.data(), cpu_result.data(), width, height);
        }
        else if (operation == "smooth_shared") {
            smoothing_shared_cpu(channel1.data(), cpu_result.data(), width, height);
        }
        else if (operation == "sobel") {
            compute_sobel_on_cpu(channel1.data(), cpu_result.data(), width, height);
        }
        else if (operation == "histogram") {
            float min_val = *std::min_element(channel1.begin(), channel1.end());
            float max_val = *std::max_element(channel1.begin(), channel1.end());
            int num_bins = 100;
            std::vector<int> histogram(num_bins, 0);
            compute_histogram_on_cpu(channel1.data(), total_pixels, num_bins, min_val, max_val, histogram.data());
        }
        else if (operation == "ndvi_sobel_zonal") {
            std::vector<float> ndvi(total_pixels, 0.0f);
            compute_ndvi_on_cpu(channel1.data(), channel2.data(), ndvi.data(), total_pixels);
            std::vector<float> sobel_output(total_pixels, 0.0f);
            compute_sobel_on_cpu(channel1.data(), sobel_output.data(), width, height);
            std::vector<float> mask(total_pixels, 0.0f);
            float sobel_threshold = 0.2f;
            apply_threshold_on_cpu(sobel_output.data(), mask.data(), total_pixels, sobel_threshold);
            float mean_in = NAN, mean_out = NAN;
            compute_zonal_from_mask_cpu(ndvi.data(), mask.data(), total_pixels, mean_in, mean_out);
            if (iter == 0) {
                std::cout << "CPU: Средний NDVI внутри (по Sobel): " << mean_in << "\n";
                std::cout << "CPU: Средний NDVI вне: " << mean_out << "\n";
            }
            cpu_result = ndvi;
        }
        else {
            std::cerr << "[WARN] Неизвестная операция. Вызываем NDVI по умолчанию.\n";
            compute_ndvi_on_cpu(channel1.data(), channel2.data(), cpu_result.data(), total_pixels);
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
    std::cout << "[BENCHMARK] CPU (" << operation << ") время за " << iterations
        << " итераций: " << cpu_duration << " мс. (Среднее: "
        << (cpu_duration / (float)iterations) << " мс)" << std::endl;

    // ===== GPU BENCHMARK =====
    float* d_data1 = nullptr, * d_data2 = nullptr, * d_result = nullptr;
    cudaMalloc(&d_data1, total_pixels * sizeof(float));
    cudaMalloc(&d_result, total_pixels * sizeof(float));
    if (need_two_channels) {
        cudaMalloc(&d_data2, total_pixels * sizeof(float));
    }

    cudaMemcpy(d_data1, channel1.data(), total_pixels * sizeof(float), cudaMemcpyHostToDevice);
    if (need_two_channels) {
        cudaMemcpy(d_data2, channel2.data(), total_pixels * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start, 0);
    for (int iter = 0; iter < iterations; iter++) {
        if (operation == "ndvi") {
            compute_ndvi_on_gpu(d_data1, d_data2, d_result, total_pixels);
        }
        else if (operation == "ndwi") {
            compute_ndwi_on_gpu(d_data1, d_data2, d_result, total_pixels);
        }
        else if (operation == "threshold") {
            float threshold_value = 0.3f;
            apply_threshold_on_gpu(d_data1, d_result, total_pixels, threshold_value);
        }
        else if (operation == "log") {
            log_transform_on_gpu(d_data1, d_result, total_pixels);
        }
        else if (operation == "smooth_naive") {
            smoothing_naive_gpu(d_data1, d_result, width, height);
        }
        else if (operation == "smooth_shared") {
            smoothing_shared_gpu(d_data1, d_result, width, height);
        }
        else if (operation == "sobel") {
            compute_sobel_on_gpu(d_data1, d_result, width, height);
        }
        else if (operation == "histogram") {
            int num_bins = 100;
            float* temp_host = new float[total_pixels];
            cudaMemcpy(temp_host, d_data1, total_pixels * sizeof(float), cudaMemcpyDeviceToHost);
            float min_val = temp_host[0], max_val = temp_host[0];
            for (int i = 1; i < total_pixels; i++) {
                if (temp_host[i] < min_val) min_val = temp_host[i];
                if (temp_host[i] > max_val) max_val = temp_host[i];
            }
            delete[] temp_host;
            int* d_hist = nullptr;
            cudaMalloc(&d_hist, num_bins * sizeof(int));
            cudaMemset(d_hist, 0, num_bins * sizeof(int));
            compute_histogram_on_gpu(d_data1, d_hist, total_pixels, num_bins, min_val, max_val);
            cudaFree(d_hist);
            cudaMemcpy(d_result, d_data1, total_pixels * sizeof(float), cudaMemcpyDeviceToDevice);
        }
        else if (operation == "ndvi_sobel_zonal") {
            compute_ndvi_on_gpu(d_data1, d_data2, d_result, total_pixels);
            float* d_sobel = nullptr;
            float* d_mask = nullptr;
            cudaMalloc(&d_sobel, total_pixels * sizeof(float));
            cudaMalloc(&d_mask, total_pixels * sizeof(float));
            compute_sobel_on_gpu(d_data1, d_sobel, width, height);
            float sobel_threshold = 0.2f;
            apply_threshold_on_gpu(d_sobel, d_mask, total_pixels, sobel_threshold);
            float mean_in, mean_out;
            compute_zonal_from_mask_gpu(d_result, d_mask, total_pixels, mean_in, mean_out);
            if (iter == 0) {
                std::cout << "GPU: Средний NDVI внутри (по Sobel): " << mean_in << "\n";
                std::cout << "GPU: Средний NDVI вне: " << mean_out << "\n";
            }
            cudaFree(d_sobel);
            cudaFree(d_mask);
        }
        else {
            compute_ndvi_on_gpu(d_data1, d_data2, d_result, total_pixels);
        }
    }
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    float gpu_time = 0.0f;
    cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
    std::cout << "[BENCHMARK] GPU (" << operation << ") время за " << iterations
        << " итераций: " << gpu_time << " мс. (Среднее: "
        << (gpu_time / (float)iterations) << " мс)" << std::endl;

    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    cudaFree(d_data1);
    if (need_two_channels) cudaFree(d_data2);
    cudaFree(d_result);

    return 0;
}
