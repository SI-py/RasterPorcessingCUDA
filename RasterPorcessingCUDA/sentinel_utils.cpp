#include "sentinel_utils.h"
#include <iostream>

CPLErr read_sentinel_band(GDALDataset* dataset,
    int band_number,
    float*& band_data,
    int& raster_width,
    int& raster_height)
{
    raster_width = dataset->GetRasterXSize();
    raster_height = dataset->GetRasterYSize();
    int total_pixels = raster_width * raster_height;

    band_data = new float[total_pixels];

    GDALRasterBand* band = dataset->GetRasterBand(band_number);
    if (!band) {
        std::cerr << "Ошибка: не удалось получить канал " << band_number << std::endl;
        return CE_Failure;
    }

    CPLErr err = band->RasterIO(GF_Read,
        0, 0,
        raster_width, raster_height,
        band_data,
        raster_width, raster_height,
        GDT_Float32,
        0, 0);
    return err;
}

CPLErr write_ndvi_result(const std::string& output_path,
    const float* ndvi_data,
    int raster_width,
    int raster_height,
    double* geo_transform,
    const char* projection)
{
    GDALDriver* geotiff_driver = (GDALDriver*)GDALGetDriverByName("GTiff");
    if (!geotiff_driver) {
        std::cerr << "Ошибка: драйвер GTiff не найден" << std::endl;
        return CE_Failure;
    }

    GDALDataset* output_dataset = geotiff_driver->Create(
        output_path.c_str(),
        raster_width,
        raster_height,
        1,
        GDT_Float32,
        nullptr
    );
    if (!output_dataset) {
        std::cerr << "Ошибка: не удалось создать выходной файл " << output_path << std::endl;
        return CE_Failure;
    }

    if (geo_transform) {
        output_dataset->SetGeoTransform(geo_transform);
    }

    if (projection) {
        output_dataset->SetProjection(projection);
    }

    GDALRasterBand* ndvi_band = output_dataset->GetRasterBand(1);
    CPLErr err = ndvi_band->RasterIO(GF_Write,
        0, 0,
        raster_width, raster_height,
        (void*)ndvi_data,
        raster_width, raster_height,
        GDT_Float32,
        0, 0);

    GDALClose(output_dataset);
    return err;
}