#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>
#include <gdal_priv.h>
#include "sentinel_utils.h"
#include "ndvi_kernel.h"
#include "ndwi_kernel.h"
#include "threshold_kernel.h"
#include "log_kernel.h"
#include "focal_smoothing.h"
#include "zonal_from_mask.h" 
#include "sobel_kernel.h"
#include "global_histogram.h"
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

int main(int argc, char* argv[])
{
    SetConsoleOutputCP(CP_UTF8);

    if (argc < 3) {
        std::cerr << "Использование: " << argv[0]
            << " <input_tif> <output_tif> [--op=threshold/ndwi/log/ndvi/smooth_naive/smooth_shared/sobel/histogram/ndvi_sobel_zonal]"
            << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

    // По умолчанию используется NDVI
    std::string operation = "ndvi";
    for (int i = 3; i < argc; i++) {
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

    float* result_host = new float[total_pixels];

    float* d_data1 = nullptr, * d_data2 = nullptr, * d_result = nullptr;
    cudaMalloc(&d_data1, total_pixels * sizeof(float));
    cudaMalloc(&d_result, total_pixels * sizeof(float));

    bool need_two_channels = (operation == "ndvi" || operation == "ndwi" || operation == "ndvi_sobel_zonal");
    if (need_two_channels) {
        cudaMalloc(&d_data2, total_pixels * sizeof(float));
    }

    // Считываем данные. Предполагаем, что итоговый TIFF имеет каналы в следующем порядке:
    // {B02, B03, B04, B08, B11, B12}
    float* host_data1 = nullptr;
    if (need_two_channels) {
        if (operation == "ndvi" || operation == "ndvi_sobel_zonal") {
            int red_band = 3, nir_band = 4;
            read_sentinel_band(input_dataset, red_band, host_data1, width, height);
            float* host_data2 = nullptr;
            read_sentinel_band(input_dataset, nir_band, host_data2, width, height);
            cudaMemcpy(d_data1, host_data1, total_pixels * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_data2, host_data2, total_pixels * sizeof(float), cudaMemcpyHostToDevice);
            delete[] host_data1;
            delete[] host_data2;
        }
        else if (operation == "ndwi") {
            int green_band = 2, nir_band = 4;
            read_sentinel_band(input_dataset, green_band, host_data1, width, height);
            float* host_data2 = nullptr;
            read_sentinel_band(input_dataset, nir_band, host_data2, width, height);
            cudaMemcpy(d_data1, host_data1, total_pixels * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_data2, host_data2, total_pixels * sizeof(float), cudaMemcpyHostToDevice);
            delete[] host_data1;
            delete[] host_data2;
        }
    }
    else {
        int single_band = 1;
        read_sentinel_band(input_dataset, single_band, host_data1, width, height);
        cudaMemcpy(d_data1, host_data1, total_pixels * sizeof(float), cudaMemcpyHostToDevice);
        delete[] host_data1;
    }

    auto overall_start = std::chrono::high_resolution_clock::now();

    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    if (operation == "ndvi") {
        cudaEventRecord(gpu_start, 0);
        compute_ndvi_on_gpu(d_data1, d_data2, d_result, total_pixels);
        cudaEventRecord(gpu_stop, 0);
    }
    else if (operation == "ndwi") {
        cudaEventRecord(gpu_start, 0);
        compute_ndwi_on_gpu(d_data1, d_data2, d_result, total_pixels);
        cudaEventRecord(gpu_stop, 0);
    }
    else if (operation == "threshold") {
        float threshold_value = 0.3f;
        cudaEventRecord(gpu_start, 0);
        apply_threshold_on_gpu(d_data1, d_result, total_pixels, threshold_value);
        cudaEventRecord(gpu_stop, 0);
    }
    else if (operation == "log") {
        cudaEventRecord(gpu_start, 0);
        log_transform_on_gpu(d_data1, d_result, total_pixels);
        cudaEventRecord(gpu_stop, 0);
    }
    else if (operation == "smooth_naive") {
        cudaEventRecord(gpu_start, 0);
        smoothing_naive_gpu(d_data1, d_result, width, height);
        cudaEventRecord(gpu_stop, 0);
    }
    else if (operation == "smooth_shared") {
        cudaEventRecord(gpu_start, 0);
        smoothing_shared_gpu(d_data1, d_result, width, height);
        cudaEventRecord(gpu_stop, 0);
    }
    else if (operation == "sobel") {
        cudaEventRecord(gpu_start, 0);
        compute_sobel_on_gpu(d_data1, d_result, width, height); 
        cudaEventRecord(gpu_stop, 0);
    }
    else if (operation == "histogram") {
        float* temp_host = new float[total_pixels];
        cudaMemcpy(temp_host, d_data1, total_pixels * sizeof(float), cudaMemcpyDeviceToHost);
        float min_val = temp_host[0], max_val = temp_host[0];
        for (int i = 1; i < total_pixels; ++i) {
            if (temp_host[i] < min_val) min_val = temp_host[i];
            if (temp_host[i] > max_val) max_val = temp_host[i];
        }
        std::cout << "Диапазон значений: min = " << min_val << ", max = " << max_val << "\n";
        delete[] temp_host;

        int num_bins = 100;
        int* d_hist = nullptr;
        cudaMalloc(&d_hist, num_bins * sizeof(int));
        cudaMemset(d_hist, 0, num_bins * sizeof(int));

        cudaEventRecord(gpu_start, 0);
        compute_histogram_on_gpu(d_data1, d_hist, total_pixels, num_bins, min_val, max_val);
        cudaEventRecord(gpu_stop, 0);

        int* hist_host = new int[num_bins];
        cudaMemcpy(hist_host, d_hist, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

        std::cout << "Гистограмма (центральное значение: количество пикселей):\n";
        int max_count = 0;
        for (int i = 0; i < num_bins; i++) {
            if (hist_host[i] > max_count)
                max_count = hist_host[i];
        }
        int max_bar_length = 50;
        float scale = (max_count > 0) ? (float)max_bar_length / max_count : 1.0f;
        for (int i = 0; i < num_bins; i++) {
            float bin_center = min_val + (max_val - min_val) * (i + 0.5f) / num_bins;
            int bar_length = (int)(hist_host[i] * scale);
            std::cout << bin_center << ": ";
            for (int j = 0; j < bar_length; j++) {
                std::cout << "#";
            }
            std::cout << " (" << hist_host[i] << ")\n";
        }
        cudaFree(d_hist);
        delete[] hist_host;
        cudaMemcpy(d_result, d_data1, total_pixels * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    else {
        std::cerr << "[WARN] Неизвестная операция. По умолчанию NDVI\n";
        cudaEventRecord(gpu_start, 0);
        compute_ndvi_on_gpu(d_data1, d_data2, d_result, total_pixels);
        cudaEventRecord(gpu_stop, 0);
    }

    cudaEventSynchronize(gpu_stop);
    float elapsed_gpu = 0.0f;
    cudaEventElapsedTime(&elapsed_gpu, gpu_start, gpu_stop);
    std::cout << "[TIMER] Время выполнения GPU операции: " << elapsed_gpu << " мс" << std::endl;

    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    auto overall_end = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end - overall_start).count();
    std::cout << "[TIMER] Общее время выполнения операции: " << overall_duration << " мс" << std::endl;

    cudaMemcpy(result_host, d_result, total_pixels * sizeof(float), cudaMemcpyDeviceToHost);

    CPLErr err_write = write_float_tiff(output_path, result_host, width, height, geo_transform, projection_ref);
    if (err_write != CE_None) {
        std::cerr << "[ERROR] Не удалось записать результат\n";
    }
    else {
        std::cout << "[OK] Результат сохранен в " << output_path << "\n";
    }

    GDALClose(input_dataset);
    delete[] result_host;
    cudaFree(d_data1);
    if (need_two_channels) cudaFree(d_data2);
    cudaFree(d_result);

    return 0;
}
