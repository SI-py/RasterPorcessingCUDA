#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX

#include <windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>
#include <iomanip>
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <gdal_priv.h>
#include <cpl_conv.h>

struct Region {
    std::string name;
    double minX;
    double minY;
    double maxX;
    double maxY;
};

std::vector<Region> REGIONS = {
    {"Paris(small)",     2.2,   48.7,  2.5,   49.0},
    {"Moscow(medium)",   37.3,  55.65, 38.0,   56.0},
    {"Dubai(large)", 55.1,  25.0,  55.8,   25.5}
};

using json = nlohmann::json;

size_t write_callback_str(void* ptr, size_t size, size_t nmemb, void* userdata)
{
    std::string* out = reinterpret_cast<std::string*>(userdata);
    size_t total = size * nmemb;
    out->append((char*)ptr, total);
    return total;
}

static const std::string CLIENT_ID = "8d27d571-77ef-4fe7-8678-83f535e5851d";
static const std::string CLIENT_SECRET = "ksuVT5xvr3xfFYCDfzqcWrYsKeGPxsEw";

std::string get_access_token()
{
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[ERROR] curl init failed\n";
        return "";
    }
    std::stringstream ss;
    ss << "grant_type=client_credentials"
        << "&client_id=" << CLIENT_ID
        << "&client_secret=" << CLIENT_SECRET;

    std::string postData = ss.str();
    std::string response;

    curl_easy_setopt(curl, CURLOPT_URL, "https://services.sentinel-hub.com/oauth/token");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)postData.size());

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/x-www-form-urlencoded");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback_str);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);

    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "[ERROR] get_access_token: " << curl_easy_strerror(res) << "\n";
    }
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (response.empty()) {
        std::cerr << "[ERROR] Пустой ответ при получении токена\n";
        return "";
    }

    try {
        auto j = json::parse(response);
        if (j.contains("access_token")) {
            return j["access_token"].get<std::string>();
        }
        else {
            std::cerr << "[ERROR] Нет access_token:\n" << response << "\n";
        }
    }
    catch (std::exception& e) {
        std::cerr << "[ERROR] JSON parse: " << e.what() << "\n"
            << "Raw: " << response << "\n";
    }
    return "";
}

std::string add_days(const std::string& dateStr, int days)
{
    std::tm tm_date = {};
    std::istringstream iss(dateStr);
    iss >> std::get_time(&tm_date, "%Y-%m-%d");
    if (iss.fail()) {
        return dateStr;
    }
    tm_date.tm_hour = 12;
    time_t timeVal = std::mktime(&tm_date);
    timeVal += 86400LL * days;

    std::tm* new_tm = std::localtime(&timeVal);
    std::ostringstream oss;
    oss << std::put_time(new_tm, "%Y-%m-%d");
    return oss.str();
}

// ======== расширение date range (±2 дня) ========
void expand_date_range(const std::string& baseDate, std::string& fromDate, std::string& toDate)
{
    fromDate = add_days(baseDate, -2);
    toDate = add_days(baseDate, +2);
}

bool download_single_band(const std::string& bandName,
    const std::string& fromDate,
    const std::string& toDate,
    const Region& region,
    const std::string& token)
{
    std::ostringstream evalScript;
    evalScript << "//VERSION=3\\n"
        << "function setup() {\\n"
        << "  return {\\n"
        << "    input: [\\\"" << bandName << "\\\"],\\n"
        << "    output: { bands: 1 }\\n"
        << "  };\\n"
        << "}\\n"
        << "function evaluatePixel(sample) {\\n"
        << "  return [sample." << bandName << "];\\n"
        << "}";

    std::stringstream body;
    body << R"({
  "input": {
    "bounds": {
      "bbox": [)"
        << region.minX << "," << region.minY << "," << region.maxX << "," << region.maxY << R"(],
      "properties": {
        "crs": "http://www.opengis.net/def/crs/EPSG/0/4326"
      }
    },
    "data": [{
      "type": "S2L2A",
      "dataFilter": {
        "timeRange": {
          "from": ")"
        << fromDate << R"(T00:00:00Z",
          "to": ")"
        << toDate << R"(T23:59:59Z"
        },
        "mosaickingOrder": "leastCC"
      }
    }]
  },
  "output": {
    "responses": [{
      "identifier": "default",
      "format": {
        "type": "image/tiff"
      }
    }]
  },
  "evalscript": ")"
        << evalScript.str() <<
        R"("
})";

    std::string postData = body.str();

    std::string outFile = bandName + "_" + fromDate + "_" + toDate + ".tif";

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "[ERROR] curl init fail in download_single_band\n";
        return false;
    }

    FILE* fp = fopen(outFile.c_str(), "wb");
    if (!fp) {
        std::cerr << "[ERROR] Не могу открыть " << outFile << "\n";
        curl_easy_cleanup(curl);
        return false;
    }

    struct curl_slist* headers = nullptr;
    std::string authH = "Authorization: Bearer " + token;
    headers = curl_slist_append(headers, authH.c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, "https://services.sentinel-hub.com/api/v1/process");
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)postData.size());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, nullptr);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);

    CURLcode res = curl_easy_perform(curl);
    fclose(fp);

    if (res != CURLE_OK) {
        std::cerr << "[ERROR] download_single_band(" << bandName << "): "
            << curl_easy_strerror(res) << "\n";
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        remove(outFile.c_str());
        return false;
    }

    char* ctype = nullptr;
    curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ctype);
    if (!ctype || std::string(ctype).find("image/tiff") == std::string::npos) {
        std::cerr << "[INFO] Канал " << bandName
            << " => не TIFF ("
            << (ctype ? ctype : "NULL") << ") => пропускаем\n";
        remove(outFile.c_str());
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return false;
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    std::cout << "[OK] Скачан канал " << bandName
        << " => " << outFile << "\n";
    return true;
}

bool merge_bands(const std::vector<std::string>& inputs, const std::string& outPath)
{
    GDALAllRegister();
    GDALDriver* driver = (GDALDriver*)GDALGetDriverByName("GTiff");
    if (!driver) {
        std::cerr << "[ERROR] GTiff driver not found\n";
        return false;
    }

    GDALDataset* dsFirst = (GDALDataset*)GDALOpen(inputs[0].c_str(), GA_ReadOnly);
    if (!dsFirst) {
        std::cerr << "[ERROR] Не могу открыть " << inputs[0] << "\n";
        return false;
    }
    int width = dsFirst->GetRasterXSize();
    int height = dsFirst->GetRasterYSize();
    double geoTransform[6] = { 0 };
    dsFirst->GetGeoTransform(geoTransform);
    const char* projRef = dsFirst->GetProjectionRef();

    int bandCount = (int)inputs.size();
    GDALDataset* dsOut = driver->Create(outPath.c_str(), width, height, bandCount, GDT_Float32, nullptr);
    if (!dsOut) {
        std::cerr << "[ERROR] Не могу создать " << outPath << "\n";
        GDALClose(dsFirst);
        return false;
    }

    dsOut->SetGeoTransform(geoTransform);
    if (projRef) dsOut->SetProjection(projRef);

    for (int i = 0; i < bandCount; i++) {
        GDALDataset* src = (GDALDataset*)GDALOpen(inputs[i].c_str(), GA_ReadOnly);
        if (!src) {
            std::cerr << "[ERROR] Не могу открыть " << inputs[i] << "\n";
            GDALClose(dsOut);
            GDALClose(dsFirst);
            return false;
        }
        GDALRasterBand* sBand = src->GetRasterBand(1);
        GDALRasterBand* dBand = dsOut->GetRasterBand(i + 1);

        std::vector<float> buffer(width * height);
        CPLErr err = sBand->RasterIO(GF_Read, 0, 0, width, height,
            buffer.data(), width, height,
            GDT_Float32, 0, 0);
        if (err != CE_None) {
            std::cerr << "[ERROR] read error from " << inputs[i] << "\n";
            GDALClose(src);
            GDALClose(dsOut);
            GDALClose(dsFirst);
            return false;
        }
        err = dBand->RasterIO(GF_Write, 0, 0, width, height,
            buffer.data(), width, height,
            GDT_Float32, 0, 0);
        if (err != CE_None) {
            std::cerr << "[ERROR] write error to " << outPath << "\n";
            GDALClose(src);
            GDALClose(dsOut);
            GDALClose(dsFirst);
            return false;
        }
        GDALClose(src);
    }
    GDALClose(dsOut);
    GDALClose(dsFirst);

    std::cout << "[OK] Сформирован " << outPath
        << " из " << bandCount << " каналов\n";
    return true;
}

int main()
{
    SetConsoleOutputCP(CP_UTF8);

    std::cout << "Выберите регион:\n";
    for (size_t i = 0; i < REGIONS.size(); ++i) {
        std::cout << (i + 1) << ") " << REGIONS[i].name << "\n";
    }
    std::cout << "Введите номер региона [1]: ";
    int regionChoice = 1;
    {
        std::string temp;
        std::getline(std::cin, temp);
        if (!temp.empty()) {
            regionChoice = std::stoi(temp);
        }
        if (regionChoice < 1 || regionChoice >(int)REGIONS.size()) {
            regionChoice = 1;
        }
    }
    Region selectedRegion = REGIONS[regionChoice - 1];
    std::cout << "[INFO] Выбран регион: " << selectedRegion.name << "\n\n";

    std::cout << "Введите дату (YYYY-MM-DD), напр. 2023-08-01: ";
    std::string dateStr;
    std::getline(std::cin, dateStr);
    if (dateStr.empty()) {
        dateStr = "2023-08-01";
    }

    std::string fromDate, toDate;
    expand_date_range(dateStr, fromDate, toDate);

    std::cout << "[INFO] Получаем OAuth-токен...\n";
    std::string token = get_access_token();
    if (token.empty()) {
        std::cerr << "[ERROR] Токен не получен\n";
        return 1;
    }
    std::cout << "[OK] Токен есть\n";

    std::vector<std::string> bands = { "B02","B03","B04","B08","B11","B12" };
    std::vector<std::string> downloadedPaths;

    for (auto& band : bands) {
        bool ok = download_single_band(band, fromDate, toDate, selectedRegion, token);
        if (!ok) {
            std::cerr << "[INFO] Канал " << band << " пропускаем (нет данных)\n";
        }
        else {
            std::string outFile = band + "_" + fromDate + "_" + toDate + ".tif";
            downloadedPaths.push_back(outFile);
        }
    }

    if (downloadedPaths.empty()) {
        std::cerr << "[INFO] Ни одного канала не удалось скачать\n";
        return 1;
    }

    std::string outMerged = "S2_" + selectedRegion.name + "_" + dateStr + ".tif";
    std::cout << "\nПробуем объединить " << downloadedPaths.size() << " канал(ов)\n";
    bool mergedOk = merge_bands(downloadedPaths, outMerged);
    if (!mergedOk) {
        std::cerr << "[ERROR] Не удалось объединить\n";
        return 1;
    }

    std::cout << "\n[FINAL] Файл " << outMerged << " готов!\n";
    return 0;
}
