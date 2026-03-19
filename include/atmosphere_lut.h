#ifndef ATMOSPHERE_LUT_H_
#define ATMOSPHERE_LUT_H_

#include "cuda_buffer.h"
#include "cuda_texture.h"
#include "launch_params.h"

#include <vector>
#include <cstdint>
#include <filesystem>

struct PreviewMetadata {
    float bottomRadius_m        = 0.0f;
    float topRadius_m           = 0.0f;
    float observerAltitude_m    = 0.0f;
    float muSMin                = -0.2f;
    float miePhaseFunctionG     = 0.8f;

    uint32_t skyNu  = 0;
    uint32_t skyMu  = 0;
    uint32_t skyMuS = 0;

    uint32_t irradianceR = 0;
    uint32_t irradianceMuS = 0;

    std::vector<float> wavelengthsNm;
    
};


enum class TableType : uint32_t {
  Transmittance2D = 0,    // 2D の透過率用 LUT (半径方向r と 方向余弦 mu を想定)
  SkyRayleighSingle = 1,  // Rayleigh 単一散乱の位相関数を含めない LUT
  SkyMieSingle = 2,       // Mie 単一散乱の位相関数を含めない LUT
  SkyMultiple = 3,        // 多重散乱の LUT
  SunTransmittance = 4,   // 固定観測点から見た太陽方向の透過率 LUT
  DirectIrradiance = 5    // 
};

enum class StorageFormat : uint32_t {
  FP32 = 0
};

struct TableHeader {
  uint32_t magic = 0x4F4D5441u;   // 'ATMO'
  uint32_t version = 1u;          // ファイルフォーマットのバージョン
  TableType tableType = TableType::SkyRayleighSingle;      // ファイルの中身
  StorageFormat storageFormat = StorageFormat::FP32;
  
  // Dimension semantics depend on table_type.
  // Transmittance2D                        : dim0 = n_mu, dim1 = n_r
  // SkyRayleighSingle/MieSingle /Multiple  : dim0 = n_nu, dim1 = n_mu, dim2 = n_mu_s, dim3 = n_lambda
  // SunTransmittance                       : dim0 = n_mu_s, dim1 = n_lambda
  uint32_t dim0 = 0u;
  uint32_t dim1 = 0u;
  uint32_t dim2 = 0u;
  uint32_t dim3 = 0u;
};

// Precomputed LUT
struct FinalTable {
    TableHeader header{};
    std::vector<float> values;  // データ部分
};

struct FinalLUTs {
    FinalTable sunTransmittance;    // mu_s, lambda, 1, 1
    FinalTable directIrradiance;    // r, mu_s, lambda, 1
    FinalTable skyRayleighSingle;   // nu, mu, mu_s, lambda
    FinalTable skyMieSingle;        // nu, mu, mu_s, lambda
    FinalTable skyMultiple;         // nu, mu, mu_s, lambda
};

class AtmosphericLUTs {
public:
    AtmosphericLUTs() = default;
    ~AtmosphericLUTs() = default;

    // コピーの禁止
    AtmosphericLUTs(const AtmosphericLUTs&) = delete;
    AtmosphericLUTs& operator=(const AtmosphericLUTs&) = delete;

    // move
    AtmosphericLUTs(AtmosphericLUTs&&) noexcept = default;
    AtmosphericLUTs& operator=(AtmosphericLUTs&&) noexcept = default;

    void free();
    void freeDevice();

    void setDirectory(const std::filesystem::path& dir);
    bool load(std::string& outError);

    bool isReady() const { return m_ready; }

    const AtmosphereDeviceData& deviceData() const { return m_deviceData; }
    const PreviewMetadata& metaData() const { return m_metaData; }

private:
    bool loadFromFile(std::string& outError);
    void uploadFromHost();
    void validateInputs() const;

    void uploadSunTransmittance(const FinalTable& table);
    void uploadDirectIrradiance(const FinalTable& table);
    void uploadSkyTable(const FinalTable& table, std::vector<CUDATexture3D<float>>& outTextures, CUDABuffer& outHandleBuffer);
    void uploadWavelengths(const std::vector<float>& wavelengthsNm);

    std::filesystem::path   m_directory;
    PreviewMetadata         m_metaData      {};
    AtmosphereDeviceData    m_deviceData    {};
    FinalLUTs               m_hostLUTs      {};
    bool                    m_ready = false;


    CUDATexture2D<float>    m_sunTransmittance;
    CUDATexture3D<float>    m_directIrradiance;

    std::vector<CUDATexture3D<float>> m_skyRayleighDirect;
    std::vector<CUDATexture3D<float>> m_skyMieDirect;
    std::vector<CUDATexture3D<float>> m_skyMultiple;

    CUDABuffer m_skyRayleighHandleBuffer;
    CUDABuffer m_skyMieHandleBuffer;
    CUDABuffer m_skyMultipleHandleBuffer;
    CUDABuffer m_wavelengthBuffer;
};


#endif