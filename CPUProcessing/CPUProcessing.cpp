#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <windows.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>  
#include <chrono> 
#include <gdal_priv.h>
#include "sentinel_utils.h" 
#include <nlohmann/json.hpp>
#include <cmath>
#include <algorithm>

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

    if (argc < 3) {
        std::cerr << "Использование: " << argv[0]
            << " <input_tif> <output_tif> [--op=threshold/ndwi/log/ndvi/smooth_naive/smooth_shared/sobel/histogram/ndvi_sobel_zonal]"
            << std::endl;
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = argv[2];

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

    auto overall_start = std::chrono::high_resolution_clock::now();

    if (operation == "ndvi") {
        compute_ndvi_on_cpu(channel1.data(), channel2.data(), result_cpu.data(), total_pixels);
    }
    else if (operation == "ndwi") {
        compute_ndwi_on_cpu(channel1.data(), channel2.data(), result_cpu.data(), total_pixels);
    }
    else if (operation == "threshold") {
        float threshold_value = 0.3f;
        apply_threshold_on_cpu(channel1.data(), result_cpu.data(), total_pixels, threshold_value);
    }
    else if (operation == "log") {
        log_transform_on_cpu(channel1.data(), result_cpu.data(), total_pixels);
    }
    else if (operation == "smooth_naive") {
        smoothing_naive_cpu(channel1.data(), result_cpu.data(), width, height);
    }
    else if (operation == "smooth_shared") {
        smoothing_shared_cpu(channel1.data(), result_cpu.data(), width, height);
    }
    else if (operation == "sobel") {
        compute_sobel_on_cpu(channel1.data(), result_cpu.data(), width, height);
    }
    else if (operation == "histogram") {
        float min_val = *std::min_element(channel1.begin(), channel1.end());
        float max_val = *std::max_element(channel1.begin(), channel1.end());
        std::cout << "Диапазон значений: min = " << min_val << ", max = " << max_val << "\n";
        int num_bins = 100;
        std::vector<int> histogram(num_bins, 0);
        compute_histogram_on_cpu(channel1.data(), total_pixels, num_bins, min_val, max_val, histogram.data());
        int max_count = *std::max_element(histogram.begin(), histogram.end());
        int max_bar_length = 50;
        float scale = (max_count > 0) ? (float)max_bar_length / max_count : 1.0f;
        for (int i = 0; i < num_bins; i++) {
            float bin_center = min_val + (max_val - min_val) * (i + 0.5f) / num_bins;
            int bar_length = (int)(histogram[i] * scale);
            std::cout << bin_center << ": ";
            for (int j = 0; j < bar_length; j++) {
                std::cout << "#";
            }
            std::cout << " (" << histogram[i] << ")\n";
        }
        result_cpu = channel1;
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
        std::cout << "Средний NDVI внутри объектов (по Sobel): " << mean_in << "\n";
        std::cout << "Средний NDVI вне объектов: " << mean_out << "\n";
        result_cpu = ndvi;
    }
    else {
        std::cerr << "[WARN] Неизвестная операция. По умолчанию NDVI\n";
        compute_ndvi_on_cpu(channel1.data(), channel2.data(), result_cpu.data(), total_pixels);
    }

    auto overall_end = std::chrono::high_resolution_clock::now();
    auto overall_duration = std::chrono::duration_cast<std::chrono::milliseconds>(overall_end - overall_start).count();
    std::cout << "[TIMER] Общее время выполнения CPU операции: " << overall_duration << " мс" << std::endl;

    CPLErr err_write = write_float_tiff(output_path, result_cpu.data(), width, height, geo_transform, projection_ref);
    if (err_write != CE_None) {
        std::cerr << "[ERROR] Не удалось записать результат\n";
    }
    else {
        std::cout << "[OK] Результат сохранен в " << output_path << "\n";
    }

    return 0;
}
