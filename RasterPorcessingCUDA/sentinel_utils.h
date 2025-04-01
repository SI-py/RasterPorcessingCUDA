#pragma once

#include <string>
#include <gdal_priv.h>

CPLErr read_sentinel_band(GDALDataset* dataset,
    int band_number,
    float*& band_data,
    int& raster_width,
    int& raster_height);

CPLErr write_ndvi_result(const std::string& output_path,
    const float* ndvi_data,
    int raster_width,
    int raster_height,
    double* geo_transform,
    const char* projection);
