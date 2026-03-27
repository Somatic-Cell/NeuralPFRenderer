#include "atmosphere_lut.h"

#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <initializer_list>
#include <limits>
#include <fstream>
#include <string>
#include <iomanip>


inline void validateTableSize(const FinalTable& t, const char* name)
{
    const size_t expected = 
        static_cast<size_t>(t.header.dim0) *
        static_cast<size_t>(t.header.dim1) *
        static_cast<size_t>(t.header.dim2) *
        static_cast<size_t>(t.header.dim3);

    if(expected != t.values.size()){
        std::ostringstream oss;
        oss << name << "size mismatch: expected=" << expected
            << " actual=" << t.values.size();
        throw std::runtime_error(oss.str());
    }
}

inline size_t idx4(uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3, 
                   uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3)
{
    return ((((size_t)i0 * d1 + i1) * d2 + i2) * d3 + i3);
}

inline size_t idx3(uint32_t d0, uint32_t d1, uint32_t d2, 
                   uint32_t i0, uint32_t i1, uint32_t i2)
{
    return (((size_t)i0 * d1 + i1) * d2 + i2);
}

inline size_t idxXYZ(uint32_t xDim, uint32_t yDim, uint32_t zDim,
                   uint32_t x, uint32_t y, uint32_t z)
{
    return (static_cast<size_t>(z) * yDim + y) * xDim + x;
}

constexpr std::uint32_t kAtmoMagic = 0x4F4D5441u; // 'ATMO'
constexpr std::uint32_t kSupportedVersion = 1u;

bool readTextFile(const std::filesystem::path& path, std::string& outText, std::string& outError)
{
    std::ifstream ifs(path, std::ios::binary);
    if(!ifs) {
        outError = "Failed to open file: " + path.string();
        return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    outText = oss.str();
    return true;
}

std::size_t skipWs(const std::string& s, std::size_t pos)
{
    while(pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos])) != 0) {
        ++pos;
    }
    return pos;
}

bool extractObjectText(const std::string& text, const std::string& key, std::string& outObject)
{
    const std::string needle = "\"" + key + "\"";
    const std::size_t keyPos = text.find(needle);
    if(keyPos == std::string::npos) {
        return false;
    }

    std::size_t pos = text.find(':', keyPos + needle.size());
    if(pos == std::string::npos) {
        return false;
    }
    pos = skipWs(text, pos + 1);
    if(pos >= text.size() || text[pos] != '{') {
        return false;
    }

    int depth = 0;
    const std::size_t begin = pos;
    for(; pos < text.size(); ++pos) {
        if(text[pos] == '{') {
            ++depth;
        }
        else if(text[pos] == '}') {
            --depth;
            if(depth == 0) {
                outObject.assign(text.begin() + static_cast<std::ptrdiff_t>(begin),
                                 text.begin() + static_cast<std::ptrdiff_t>(pos + 1));
                return true;
            }
        }
    }
    return false;
}

bool extractArrayText(const std::string& text, const std::string& key, std::string& outArray)
{
    const std::string needle = "\"" + key + "\"";
    const std::size_t keyPos = text.find(needle);
    if(keyPos == std::string::npos) {
        return false;
    }

    std::size_t pos = text.find(':', keyPos + needle.size());
    if(pos == std::string::npos) {
        return false;
    }
    pos = skipWs(text, pos + 1);
    if(pos >= text.size() || text[pos] != '[') {
        return false;
    }

    int depth = 0;
    const std::size_t begin = pos;
    for(; pos < text.size(); ++pos) {
        if(text[pos] == '[') {
            ++depth;
        }
        else if(text[pos] == ']') {
            --depth;
            if(depth == 0) {
                outArray.assign(text.begin() + static_cast<std::ptrdiff_t>(begin),
                                text.begin() + static_cast<std::ptrdiff_t>(pos + 1));
                return true;
            }
        }
    }
    return false;
}

bool parseLeadingFloat(const std::string& text, std::size_t pos, float& outValue)
{
    pos = skipWs(text, pos);
    if(pos >= text.size()) {
        return false;
    }

    const char* begin = text.c_str() + pos;
    char* end = nullptr;
    const float value = std::strtof(begin, &end);
    if(end == begin) {
        return false;
    }
    if(value == HUGE_VALF || value == -HUGE_VALF || !std::isfinite(value)) {
        return false;
    }
    outValue = value;
    return true;
}

bool extractNumberByKeys(const std::string& text,
                         const std::initializer_list<const char*>& keys,
                         float& outValue)
{
    for(const char* key : keys) {
        const std::string needle = std::string("\"") + key + "\"";
        const std::size_t keyPos = text.find(needle);
        if(keyPos == std::string::npos) {
            continue;
        }
        const std::size_t colonPos = text.find(':', keyPos + needle.size());
        if(colonPos == std::string::npos) {
            continue;
        }
        if(parseLeadingFloat(text, colonPos + 1, outValue)) {
            return true;
        }
    }
    return false;
}

bool extractNumberByKeys(const std::string& text,
                         const std::initializer_list<const char*>& keys,
                         std::uint32_t& outValue)
{
    float tmp = 0.0f;
    if(!extractNumberByKeys(text, keys, tmp)) {
        return false;
    }
    if(tmp < 0.0f || tmp > static_cast<float>(std::numeric_limits<std::uint32_t>::max())) {
        return false;
    }
    outValue = static_cast<std::uint32_t>(tmp);
    return true;
}

bool parseFloatArray(const std::string& arrayText, std::vector<float>& outValues)
{
    outValues.clear();
    if(arrayText.size() < 2 || arrayText.front() != '[' || arrayText.back() != ']') {
        return false;
    }

    std::size_t pos = 1;
    while(pos + 1 < arrayText.size()) {
        pos = skipWs(arrayText, pos);
        if(pos >= arrayText.size() - 1) {
            break;
        }
        if(arrayText[pos] == ',') {
            ++pos;
            continue;
        }

        float value = 0.0f;
        if(!parseLeadingFloat(arrayText, pos, value)) {
            return false;
        }

        const char* begin = arrayText.c_str() + pos;
        char* end = nullptr;
        std::strtof(begin, &end);
        pos += static_cast<std::size_t>(end - begin);
        outValues.push_back(value);
    }
    return !outValues.empty();
}

std::uint64_t computeElementCount(const TableHeader& header)
{
    return static_cast<std::uint64_t>(header.dim0) *
           static_cast<std::uint64_t>(header.dim1) *
           static_cast<std::uint64_t>(header.dim2) *
           static_cast<std::uint64_t>(header.dim3);
}

bool readTableWithHeader(const std::filesystem::path& path,
                         TableType expectedType,
                         FinalTable& outTable,
                         std::string& outError)
{
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if(!ifs) {
        outError = "Failed to open LUT file: " + path.string();
        return false;
    }

    const std::streamsize fileSize = ifs.tellg();
    if(fileSize < static_cast<std::streamsize>(sizeof(TableHeader))) {
        outError = "LUT file is too small to contain TableHeader: " + path.string();
        return false;
    }

    ifs.seekg(0, std::ios::beg);
    if(!ifs.read(reinterpret_cast<char*>(&outTable.header), sizeof(outTable.header))) {
        outError = "Failed to read TableHeader: " + path.string();
        return false;
    }

    if(outTable.header.magic != kAtmoMagic) {
        outError = "Invalid table magic in: " + path.string();
        return false;
    }
    if(outTable.header.version != kSupportedVersion) {
        outError = "Unsupported table version in: " + path.string();
        return false;
    }
    if(outTable.header.tableType != expectedType) {
        outError = "Unexpected table type in: " + path.string();
        return false;
    }
    if(outTable.header.storageFormat != StorageFormat::FP32) {
        outError = "Unsupported storage format in: " + path.string();
        return false;
    }

    const std::uint64_t count = computeElementCount(outTable.header);
    if(count == 0) {
        outError = "Table has zero-sized dimension(s): " + path.string();
        return false;
    }

    const std::uint64_t expectedPayloadBytes = count * static_cast<std::uint64_t>(sizeof(float));
    const std::uint64_t actualPayloadBytes =
        static_cast<std::uint64_t>(fileSize) - static_cast<std::uint64_t>(sizeof(TableHeader));

    if(actualPayloadBytes != expectedPayloadBytes) {
        std::ostringstream oss;
        oss << "Payload size mismatch in " << path.string()
            << ": header expects " << expectedPayloadBytes
            << " bytes, but file contains " << actualPayloadBytes << " bytes.";
        outError = oss.str();
        return false;
    }

    outTable.values.resize(static_cast<std::size_t>(count));
    if(!ifs.read(reinterpret_cast<char*>(outTable.values.data()),
                 static_cast<std::streamsize>(expectedPayloadBytes))) {
        outError = "Failed to read LUT payload: " + path.string();
        return false;
    }

    return true;
}

bool validateSkyHeaderCompatibility(const TableHeader& ref,
                                    const TableHeader& other,
                                    const char* otherName,
                                    std::string& outError)
{
    if(ref.dim0 != other.dim0 || ref.dim1 != other.dim1 || ref.dim2 != other.dim2 || ref.dim3 != other.dim3) {
        std::ostringstream oss;
        oss << "Sky LUT dimension mismatch between sky_rayleigh_single.bin and "
            << otherName << ": ("
            << ref.dim0 << ", " << ref.dim1 << ", " << ref.dim2 << ", " << ref.dim3 << ") vs ("
            << other.dim0 << ", " << other.dim1 << ", " << other.dim2 << ", " << other.dim3 << ").";
        outError = oss.str();
        return false;
    }
    return true;
}

bool validateOrAssignDim(const char* label,
                         std::uint32_t currentValue,
                         std::uint32_t headerValue,
                         std::uint32_t& outValue,
                         std::string& outError)
{
    if(currentValue != 0 && currentValue != headerValue) {
        std::ostringstream oss;
        oss << "Dimension mismatch for " << label << ": metadata/CLI says "
            << currentValue << ", but LUT header says " << headerValue << ".";
        outError = oss.str();
        return false;
    }
    outValue = headerValue;
    return true;
}


bool loadMetadataJson(const std::filesystem::path& path,
                      PreviewMetadata& outMeta,
                      std::string& outError)
{
    std::string text;
    if(!readTextFile(path, text, outError)) {
        return false;
    }

    std::string planetText;
    if(extractObjectText(text, "planet", planetText)) {
        extractNumberByKeys(planetText, {"bottom_radius_m", "bottomRadiusM"}, outMeta.bottomRadius_m);
        extractNumberByKeys(planetText, {"top_radius_m", "topRadiusM"}, outMeta.topRadius_m);
        extractNumberByKeys(planetText, {"observer_altitude_m", "observerAltitudeM"}, outMeta.observerAltitude_m);
        extractNumberByKeys(planetText, {"mu_s_min", "muSMin"}, outMeta.muSMin);
    }

    std::string mieText;
    if(extractObjectText(text, "mie_phase_function", mieText)) {
        extractNumberByKeys(mieText, {"g", "miePhaseFunctionG"}, outMeta.miePhaseFunctionG);
    }
    else {
        extractNumberByKeys(text, {"mie_phase_function_g", "miePhaseFunctionG"}, outMeta.miePhaseFunctionG);
    }

    std::string lutText;
    if(extractObjectText(text, "lut", lutText)) {
        extractNumberByKeys(lutText, {"sky_mu", "n_mu", "skyMu"}, outMeta.skyMu);
        extractNumberByKeys(lutText, {"sky_mu_s", "n_mu_s", "skyMuS"}, outMeta.skyMuS);
        extractNumberByKeys(lutText, {"sky_nu", "n_nu", "skyNu"}, outMeta.skyNu);

        extractNumberByKeys(lutText, {"irradiance_r", "n_irradiance_r", "irradianceR"}, outMeta.irradianceR);
        extractNumberByKeys(lutText, {"irradiance_mu_s", "n_irradiance_mu_s", "irradianceMuS"}, outMeta.irradianceMuS);
    }

    std::string wavelengthsText;
    if(!extractArrayText(text, "wavelengths_nm", wavelengthsText)) {
        outError = "metadata.json does not contain wavelengths_nm array.";
        return false;
    }
    if(!parseFloatArray(wavelengthsText, outMeta.wavelengthsNm)) {
        outError = "Failed to parse wavelengths_nm array in metadata.json.";
        return false;
    }

    return true;
}

bool loadFinalLuts(const std::filesystem::path& outDir,
                   PreviewMetadata& inOutMeta,
                   FinalLUTs& outLuts,
                   std::string& outError)
{
    if(inOutMeta.wavelengthsNm.empty()) {
        outError = "wavelengths_nm is empty. metadata.json is required for spectral preview.";
        return false;
    }

    if(!readTableWithHeader(outDir / "transmittance_internal.bin",
                            TableType::SunTransmittance,
                            outLuts.sunTransmittance,
                            outError)) {
        return false;
    }
    if(!readTableWithHeader(outDir / "sky_rayleigh_single.bin",
                            TableType::SkyRayleighSingle,
                            outLuts.skyRayleighSingle,
                            outError)) {
        return false;
    }
    if(!readTableWithHeader(outDir / "sky_mie_single.bin",
                            TableType::SkyMieSingle,
                            outLuts.skyMieSingle,
                            outError)) {
        return false;
    }
    if(!readTableWithHeader(outDir / "sky_multiple.bin",
                            TableType::SkyMultiple,
                            outLuts.skyMultiple,
                            outError)) {
        return false;
    }
    if(!readTableWithHeader(outDir / "direct_irradiance.bin",
                            TableType::DirectIrradiance,
                            outLuts.directIrradiance,
                            outError)) {
        return false;
    }

    const TableHeader& skyHdr = outLuts.skyRayleighSingle.header;
    if(!validateSkyHeaderCompatibility(skyHdr, outLuts.skyMieSingle.header, "sky_mie_single.bin", outError)) {
        return false;
    }
    if(!validateSkyHeaderCompatibility(skyHdr, outLuts.skyMultiple.header, "sky_multiple.bin", outError)) {
        return false;
    }

    if(skyHdr.dim0 == 0 || skyHdr.dim1 == 0 || skyHdr.dim2 == 0 || skyHdr.dim3 == 0) {
        outError = "Sky LUT header contains zero dimension(s).";
        return false;
    }

    std::uint32_t resolvedSkyNu = 0;
    std::uint32_t resolvedSkyMu = 0;
    std::uint32_t resolvedSkyMuS = 0;

    if(!validateOrAssignDim("skyNu", inOutMeta.skyNu, skyHdr.dim0, resolvedSkyNu, outError)) {
        return false;
    }
    if(!validateOrAssignDim("skyMu", inOutMeta.skyMu, skyHdr.dim1, resolvedSkyMu, outError)) {
        return false;
    }
    if(!validateOrAssignDim("skyMuS", inOutMeta.skyMuS, skyHdr.dim2, resolvedSkyMuS, outError)) {
        return false;
    }

    const std::uint32_t lambdaCount = skyHdr.dim3;
    if(inOutMeta.wavelengthsNm.size() != static_cast<std::size_t>(lambdaCount)) {
        std::ostringstream oss;
        oss << "wavelength count mismatch: metadata.json has "
            << inOutMeta.wavelengthsNm.size() << " wavelengths, but sky LUT header says "
            << lambdaCount << ".";
        outError = oss.str();
        return false;
    }

    const TableHeader& sunHdr = outLuts.sunTransmittance.header;
    if(sunHdr.dim0 != resolvedSkyMuS) {
        std::ostringstream oss;
        oss << "sun_transmittance.bin dim0 (mu_s count) is " << sunHdr.dim0
            << ", but sky LUT dim2 (mu_s count) is " << resolvedSkyMuS << ".";
        outError = oss.str();
        return false;
    }
    if(sunHdr.dim1 != lambdaCount) {
        std::ostringstream oss;
        oss << "sun_transmittance.bin dim1 (lambda count) is " << sunHdr.dim1
            << ", but sky LUT dim3 (lambda count) is " << lambdaCount << ".";
        outError = oss.str();
        return false;
    }

    const TableHeader& irrHdr = outLuts.directIrradiance.header;
    if(irrHdr.dim0 == 0 || irrHdr.dim1 == 0 || irrHdr.dim2 == 0 || irrHdr.dim3 == 0) {
        outError = "direct_irradiance.bin header contains zero dimension(s).";
        return false;
    }

    std::uint32_t resolvedIrradianceR = 0;
    std::uint32_t resolvedIrradianceMuS = 0;

    if(!validateOrAssignDim("irradianceR", inOutMeta.irradianceR, irrHdr.dim0, resolvedIrradianceR, outError)) {
        return false;
    }
    if(!validateOrAssignDim("irradianceMuS", inOutMeta.irradianceMuS, irrHdr.dim1, resolvedIrradianceMuS, outError)) {
        return false;
    }

    if(irrHdr.dim2 != lambdaCount) {
        std::ostringstream oss;
        oss << "direct_irradiance.bin dim2 (lambda count) is " << irrHdr.dim2
            << ", but sky LUT dim3 (lambda count) is " << lambdaCount << ".";
        outError = oss.str();
        return false;
    }

    if(irrHdr.dim3 != 1u) {
        std::ostringstream oss;
        oss << "direct_irradiance.bin dim3 is " << irrHdr.dim3
            << ", expected 1.";
        outError = oss.str();
        return false;
    }

    inOutMeta.skyNu = resolvedSkyNu;
    inOutMeta.skyMu = resolvedSkyMu;
    inOutMeta.skyMuS = resolvedSkyMuS;
    inOutMeta.irradianceR = resolvedIrradianceR;
    inOutMeta.irradianceMuS = resolvedIrradianceMuS;
    return true;
}


void AtmosphericLUTs::freeDevice()
{
    m_sunTransmittance.free();
    m_directIrradiance.free();

    m_skyRayleighDirect.clear();
    m_skyMieDirect.clear();
    m_skyMultiple.clear();

    m_skyRayleighHandleBuffer.free();
    m_skyMieHandleBuffer.free();
    m_skyMultipleHandleBuffer.free();
    m_wavelengthBuffer.free();

    m_deviceData = {};
    m_ready = false;
}
void AtmosphericLUTs::free()
{
    freeDevice();
    m_metaData = {};
    m_hostLUTs = {};
}

bool AtmosphericLUTs::loadFromFile(std::string& outError)
{
    if(!loadMetadataJson(m_directory / "metadata.json", m_metaData, outError)) {
        return false;
    }

    if(!loadFinalLuts(m_directory, m_metaData, m_hostLUTs, outError)) {
        return false;
    }

    return true;
}

void AtmosphericLUTs::validateInputs() const
{
    if(m_metaData.wavelengthsNm.empty()){
        throw std::runtime_error("PreviewMetadata.wavelengthNm is empty.");
    }

    validateTableSize(m_hostLUTs.sunTransmittance,    "sunTransmittance");
    validateTableSize(m_hostLUTs.directIrradiance,    "directIrradiance");
    validateTableSize(m_hostLUTs.skyRayleighSingle,   "skyRayleighSingle");
    validateTableSize(m_hostLUTs.skyMieSingle,        "skyMieSingle");
    validateTableSize(m_hostLUTs.skyMultiple,         "skyMultiple");

    const uint32_t lambdaCount = static_cast<uint32_t>(m_metaData.wavelengthsNm.size());

    if(m_hostLUTs.sunTransmittance.header.dim1 != lambdaCount){
        throw std::runtime_error("sunTransmittance lambda dimension mismatch.");
    }
    if(m_hostLUTs.directIrradiance.header.dim2 != lambdaCount){
        throw std::runtime_error("directIrradiance lambda dimension mismatch.");
    }
    if(m_hostLUTs.skyRayleighSingle.header.dim3 != lambdaCount || 
    m_hostLUTs.skyMieSingle.header.dim3 != lambdaCount || 
    m_hostLUTs.skyMultiple.header.dim3 != lambdaCount){
        throw std::runtime_error("sky LUT lambda dimension mismatch.");
    }
}


void AtmosphericLUTs::uploadSunTransmittance(const FinalTable& table)
{
    const uint32_t muSDim    = table.header.dim0;
    const uint32_t lambdaDim = table.header.dim1;

    std::vector<float> packed(static_cast<size_t>(muSDim) * lambdaDim);

    for(uint32_t iMuS = 0; iMuS < muSDim; ++iMuS){
        for(uint32_t iL = 0; iL < lambdaDim; ++iL){
            // file: table.values[iMuS * lambdaDim + iL]
            // tex : packed[iL * muSDim + iMuS]   (x=muS, y=lambda)
            packed[static_cast<size_t>(iL) * muSDim + iMuS] =
                table.values[static_cast<size_t>(iMuS) * lambdaDim + iL];
        }
    }

    m_sunTransmittance.createFromHost(packed.data(), muSDim, lambdaDim);
    m_deviceData.sunTransmittanceTex = m_sunTransmittance.texture();
}

void AtmosphericLUTs::uploadDirectIrradiance(const FinalTable& table){
    uint32_t rDim = table.header.dim0;
    uint32_t muSDim = table.header.dim1;
    uint32_t lambdaDim = table.header.dim2;

    std::vector<float> packed(static_cast<size_t>(muSDim) * rDim * lambdaDim);

    for(uint32_t iR = 0; iR < rDim; ++iR){
        for(uint32_t iMuS = 0; iMuS < muSDim; ++iMuS){
            for(uint32_t iL = 0; iL < lambdaDim; ++iL){
                const float v = table.values[idx4(table.header.dim0, table.header.dim1, table.header.dim2, table.header.dim3, iR, iMuS, iL, 0)];
                packed[idxXYZ(muSDim, rDim, lambdaDim, iMuS, iR, iL)] = v;
            }
        }
    }

    m_directIrradiance.createFromHost(
        packed.data(),
        muSDim,
        rDim,
        lambdaDim
    );

    m_deviceData.directIrradianceTex = m_directIrradiance.texture();
}

void AtmosphericLUTs::uploadSkyTable(const FinalTable& table, std::vector<CUDATexture3D<float>>& outTextures, CUDABuffer& outHandleBuffer)
{
    uint32_t nuDim = table.header.dim0;
    uint32_t muDim = table.header.dim1;
    uint32_t muSDim = table.header.dim2;
    uint32_t lambdaDim = table.header.dim3;

    outTextures.clear();
    outTextures.resize(lambdaDim);

    std::vector<float> packed(static_cast<size_t>(nuDim) * muDim * muSDim);

    for(uint32_t iL = 0; iL < lambdaDim; ++iL){
        for(uint32_t iNu = 0; iNu < nuDim; ++iNu){
            for(uint32_t iMu = 0; iMu < muDim; ++iMu){
                for(uint32_t iMuS = 0; iMuS < muSDim; ++iMuS){
                    const float v = table.values[idx4(nuDim, muDim, muSDim, lambdaDim, iNu, iMu, iMuS, iL)];
                    packed[idxXYZ(nuDim, muDim, muSDim, iNu, iMu, iMuS)] = v;
                }
            }
        }

        outTextures[iL].createFromHost(
            packed.data(),
            nuDim,
            muDim,
            muSDim
        );

    }

    std::vector<cudaTextureObject_t> handles(lambdaDim);
    for(uint32_t i = 0; i < lambdaDim; ++i){
        handles[i] = outTextures[i].texture();
    }
    outHandleBuffer.allocAndUpload(handles);
}

void AtmosphericLUTs::uploadWavelengths(const std::vector<float>& wavelengthsNm)
{
    m_wavelengthBuffer.allocAndUpload(wavelengthsNm);
    m_deviceData.wavelengthsNm = m_wavelengthBuffer.getDevicePointer();
}

void AtmosphericLUTs::setDirectory(const std::filesystem::path& dir)
{
    m_directory = dir;
}

void AtmosphericLUTs::uploadFromHost()
{
    freeDevice();

    // if(m_directory.empty()) {
    //     outError = "Atmosphere LUT directory is empty.";
    //     return false;
    // }


    validateInputs();

    uploadSunTransmittance(m_hostLUTs.sunTransmittance);
    uploadDirectIrradiance(m_hostLUTs.directIrradiance);

    uploadSkyTable(
        m_hostLUTs.skyRayleighSingle, 
        m_skyRayleighDirect, 
        m_skyRayleighHandleBuffer
    );
    
    uploadSkyTable(
        m_hostLUTs.skyMieSingle, 
        m_skyMieDirect, 
        m_skyMieHandleBuffer
    );
    
    uploadSkyTable(
        m_hostLUTs.skyMultiple, 
        m_skyMultiple, 
        m_skyMultipleHandleBuffer
    );

    uploadWavelengths(m_metaData.wavelengthsNm);

    m_deviceData.skyRayleighTexHandles  = m_skyRayleighHandleBuffer.getDevicePointer();
    m_deviceData.skyMieTexHandles       = m_skyMieHandleBuffer.getDevicePointer();
    m_deviceData.skyMultipleTexHandles  = m_skyMultipleHandleBuffer.getDevicePointer();

    m_deviceData.lambdaCount = static_cast<uint32_t>(m_metaData.wavelengthsNm.size());

    m_deviceData.bottomRadius_m = m_metaData.bottomRadius_m;
    m_deviceData.topRadius_m    = m_metaData.topRadius_m;
    m_deviceData.observerAltitude_m = m_metaData.observerAltitude_m;
    m_deviceData.muSMin         = m_metaData.muSMin;

    m_deviceData.skyNu  = m_metaData.skyNu;
    m_deviceData.skyMu  = m_metaData.skyMu;
    m_deviceData.skyMuS = m_metaData.skyMuS;

    m_deviceData.irradianceR = m_metaData.irradianceR;
    m_deviceData.irradianceMuS = m_metaData.irradianceMuS;

    m_ready = true;
    return;
}

bool AtmosphericLUTs::load(std::string& outError)
{
    try {
        free();

        if(m_directory.empty()) {
            outError = "Atmosphere LUT directory is empty.";
            return false;
        }

        if(!loadFromFile(outError)) {
            free();
            return false;
        }

        uploadFromHost();

        m_ready = true;
        return true;
    }
    catch(const std::exception& e) {
        free();
        outError = e.what();
        return false;
    }
}
