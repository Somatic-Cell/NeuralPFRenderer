#pragma once
#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "launch_params.h"



namespace mie {

static inline void trim_cr(std::string& s)
{
    if (!s.empty() && s.back() == '\r') s.pop_back();
}

// 先頭から float を順に読む（タブ/空白区切り、末尾タブOK）
static inline bool next_float(const char*& p, float& out)
{
    while (*p && (*p==' ' || *p=='\t' || *p=='\r')) ++p;
    if (!*p) return false;
    char* end = nullptr;
    out = std::strtof(p, &end);
    if (end == p) return false;
    p = end;
    return true;
}

static inline bool starts_with(const std::string& s, const char* prefix)
{
    return s.rfind(prefix, 0) == 0;
}

struct MieHostTables
{
    int Nd = 0;
    int Nlambda = 0;
    int Ntheta = 0;

    std::vector<float> diam_um;     // Nd
    std::vector<float> lambda_nm;   // Nlambda
    std::vector<float> theta_deg;   // Ntheta（読み込み検証用）

    // layout:
    // g     : [d][lambda]
    // pdf/cdf: [d][lambda][theta]  (theta が最内周)
    std::vector<float> g;    // Nd*Nlambda
    std::vector<float> pdf;  // Nd*Nlambda*Ntheta  (solid-angle PDF 期待)
    std::vector<float> cdf;  // Nd*Nlambda*Ntheta  (F(theta) = ∫ 2π p(θ) sinθ dθ)

    inline size_t idx2(int id, int il) const
    {
        return (size_t)id * (size_t)Nlambda + (size_t)il;
    }
    inline size_t idx3(int id, int il, int it) const
    {
        return ((size_t)id * (size_t)Nlambda + (size_t)il) * (size_t)Ntheta + (size_t)it;
    }
};

struct MieGpuTextures
{
    // cuda arrays
    cudaArray_t arr_g   = nullptr;   // 2D
    cudaArray_t arr_pdf = nullptr;   // 3D
    cudaArray_t arr_cdf = nullptr;   // 3D

    // texture objects
    cudaTextureObject_t tex_g   = 0;
    cudaTextureObject_t tex_pdf = 0;
    cudaTextureObject_t tex_cdf = 0;

    // dimensions (device side mappingのために launch params にも載せる想定)
    int Nd = 0, Nlambda = 0, Ntheta = 0;

    void destroy()
    {
        if (tex_g)   CUDA_CHECK(cudaDestroyTextureObject(tex_g));
        if (tex_pdf) CUDA_CHECK(cudaDestroyTextureObject(tex_pdf));
        if (tex_cdf) CUDA_CHECK(cudaDestroyTextureObject(tex_cdf));
        tex_g = tex_pdf = tex_cdf = 0;

        if (arr_g)   CUDA_CHECK(cudaFreeArray(arr_g));
        if (arr_pdf) CUDA_CHECK(cudaFreeArray(arr_pdf));
        if (arr_cdf) CUDA_CHECK(cudaFreeArray(arr_cdf));
        arr_g = arr_pdf = arr_cdf = nullptr;
    }

    ~MieGpuTextures() { destroy(); }

    // コピー禁止（重要）
    MieGpuTextures(const MieGpuTextures&) = delete;
    MieGpuTextures& operator=(const MieGpuTextures&) = delete;

    // ムーブ可
    MieGpuTextures() = default;
    MieGpuTextures(MieGpuTextures&& o) noexcept { *this = std::move(o); }

    MieGpuTextures& operator=(MieGpuTextures&& o) noexcept
    {
        if (this == &o) return *this;
        destroy();
        arr_g=o.arr_g; arr_pdf=o.arr_pdf; arr_cdf=o.arr_cdf;
        tex_g=o.tex_g; tex_pdf=o.tex_pdf; tex_cdf=o.tex_cdf;
        Nd=o.Nd; Nlambda=o.Nlambda; Ntheta=o.Ntheta;
        o.arr_g=o.arr_pdf=o.arr_cdf=nullptr;
        o.tex_g=o.tex_pdf=o.tex_cdf=0;
        o.Nd=o.Nlambda=o.Ntheta=0;
        return *this;
    }
};

// -------------------------
// txt パース（1つの code に対して）
// -------------------------

static inline float parse_mean_diameter_um_from_param_header(const std::filesystem::path& path)
{
    std::ifstream ifs(path);
    if (!ifs) {
        std::fprintf(stderr, "Failed to open %s\n", path.string().c_str());
        std::abort();
    }

    std::string line;
    while (std::getline(ifs, line)) {
        trim_cr(line);
        // 例: "Mean diameter of spheres: 5 um"
        auto pos = line.find("Mean diameter of spheres:");
        if (pos != std::string::npos) {
            // 数字を拾う（単位はひとまず um 想定）
            const char* p = line.c_str() + pos;
            // ':' の次へ
            const char* colon = std::strchr(p, ':');
            if (!colon) continue;
            p = colon + 1;

            float v = 0.0f;
            if (!next_float(p, v)) continue;
            return v; // um
        }
    }

    std::fprintf(stderr, "Could not find mean diameter in %s\n", path.string().c_str());
    std::abort();
}

static inline void parse_g_table_from_param_file(
    const std::filesystem::path& path,
    std::vector<float>& out_lambda_nm,
    std::vector<float>& out_g)
{
    std::ifstream ifs(path);
    if (!ifs) {
        std::fprintf(stderr, "Failed to open %s\n", path.string().c_str());
        std::abort();
    }

    std::string line;
    bool in_table = false;

    out_lambda_nm.clear();
    out_g.clear();

    while (std::getline(ifs, line)) {
        trim_cr(line);

        if (!in_table) {
            // 例: "WL(nm)\tus(mm^-1)\tg\tus'(mm^-1)"
            if (line.find("WL(nm)") != std::string::npos && line.find("\tg\t") != std::string::npos) {
                in_table = true;
            }
            continue;
        }

        if (line.empty()) break;

        const char* p = line.c_str();
        float wl=0, us=0, g=0, usp=0;
        if (!next_float(p, wl)) continue;
        if (!next_float(p, us)) continue;
        if (!next_float(p, g))  continue;
        // usp はあってもなくても良い
        (void)next_float(p, usp);

        out_lambda_nm.push_back(wl);
        out_g.push_back(g);
    }

    if (out_lambda_nm.empty() || out_g.empty() || out_lambda_nm.size() != out_g.size()) {
        std::fprintf(stderr, "Failed to parse g table in %s\n", path.string().c_str());
        std::abort();
    }
}

static inline void parse_phase_pdf_from_phase_file(
    const std::filesystem::path& path,
    std::vector<float>& out_lambda_nm,
    std::vector<float>& out_theta_deg,
    std::vector<float>& out_pdf_theta_major /* [lambda][theta] or rowwise? -> here: [theta][lambda] temp */)
{
    std::ifstream ifs(path);
    if (!ifs) {
        std::fprintf(stderr, "Failed to open %s\n", path.string().c_str());
        std::abort();
    }

    std::string line;
    bool got_header = false;

    out_lambda_nm.clear();
    out_theta_deg.clear();
    out_pdf_theta_major.clear();

    while (std::getline(ifs, line)) {
        trim_cr(line);

        if (!got_header) {
            // ヘッダ行: "Angle(deg)\t390\t392.5\t..."
            if (line.find("Angle(deg)") != std::string::npos) {
                got_header = true;

                const char* p = line.c_str();
                // "Angle(deg)" を飛ばす（数値パースできるところまで進める）
                while (*p && *p!='\t' && *p!=' ') ++p;

                float wl = 0.0f;
                while (next_float(p, wl)) {
                    out_lambda_nm.push_back(wl);
                }

                if (out_lambda_nm.empty()) {
                    std::fprintf(stderr, "Phase header wavelengths not found in %s\n", path.string().c_str());
                    std::abort();
                }
            }
            continue;
        }

        if (line.empty()) continue;

        const char* p = line.c_str();
        float theta = 0.0f;
        if (!next_float(p, theta)) continue;

        out_theta_deg.push_back(theta);

        // 1行に Nlambda 個の値
        for (size_t il=0; il<out_lambda_nm.size(); ++il) {
            float v = 0.0f;
            if (!next_float(p, v)) {
                std::fprintf(stderr, "Phase row missing values at theta=%g in %s\n", theta, path.string().c_str());
                std::abort();
            }
            out_pdf_theta_major.push_back(v); // row-major: [theta][lambda]
        }
    }

    if (!got_header) {
        std::fprintf(stderr, "Phase header not found in %s\n", path.string().c_str());
        std::abort();
    }

    const int Ntheta = (int)out_theta_deg.size();
    const int Nlambda = (int)out_lambda_nm.size();
    if ((int)out_pdf_theta_major.size() != Ntheta * Nlambda) {
        std::fprintf(stderr, "Phase size mismatch in %s\n", path.string().c_str());
        std::abort();
    }
}

// -------------------------
// CDF 生成（必要なら PDF も再正規化）
// -------------------------
static inline void build_cdf_and_renormalize_pdf_inplace(
    int Ntheta,
    const float* theta_deg, // Ntheta
    float* pdf_theta,       // Ntheta (solid-angle PDF: p(θ) [sr^-1])
    float* cdf_theta)       // Ntheta
{
    // θ は 0..180deg を想定
    // CDF: F(θ)=∫0^θ 2π p(t) sin(t) dt
    constexpr float PI = 3.14159265358979323846f;

    auto deg2rad = [](float d) { return d * (3.14159265358979323846f / 180.0f); };

    cdf_theta[0] = 0.0f;
    float accum = 0.0f;

    for (int i=1; i<Ntheta; ++i) {
        float t0 = deg2rad(theta_deg[i-1]);
        float t1 = deg2rad(theta_deg[i]);
        float p0 = pdf_theta[i-1];
        float p1 = pdf_theta[i];

        float w0 = 2.0f * PI * p0 * std::sinf(t0);
        float w1 = 2.0f * PI * p1 * std::sinf(t1);
        float dt = (t1 - t0);

        accum += 0.5f * (w0 + w1) * dt;
        cdf_theta[i] = accum;
    }

    // 正規化（accum が 1 からズレていても合わせる）
    float norm = cdf_theta[Ntheta-1];
    if (norm <= 0.0f || !std::isfinite(norm)) {
        // 最悪の保険：均一にする（ただしここに来るなら入力データ不正）
        for (int i=0; i<Ntheta; ++i) cdf_theta[i] = (float)i / (float)(Ntheta-1);
        return;
    }

    // PDF も norm で割って整合させる
    for (int i=0; i<Ntheta; ++i) pdf_theta[i] /= norm;
    for (int i=0; i<Ntheta; ++i) cdf_theta[i] /= norm;

    // 単調性の強制（浮動小数誤差対策）
    float prev = 0.0f;
    for (int i=0; i<Ntheta; ++i) {
        float v = cdf_theta[i];
        if (v < prev) v = prev;
        if (v < 0.0f) v = 0.0f;
        if (v > 1.0f) v = 1.0f;
        cdf_theta[i] = v;
        prev = v;
    }
    cdf_theta[Ntheta-1] = 1.0f;
}

// -------------------------
// ディレクトリから全ファイルをロード
// -------------------------
static inline int extract_code_3digits(const std::string& filename)
{
    // 例: Mie_PhaseFunctionData_050.txt -> 50
    // 末尾から3桁を拾う（命名が一定と仮定）
    // より堅牢にするなら regex を使ってください。
    auto pos_us = filename.find_last_of('_');
    auto pos_dot = filename.find_last_of('.');
    if (pos_us == std::string::npos || pos_dot == std::string::npos || pos_dot <= pos_us+1) return -1;
    std::string code = filename.substr(pos_us+1, pos_dot-(pos_us+1));
    if (code.size() != 3) return -1;
    for (char c: code) if (c<'0' || c>'9') return -1;
    return std::atoi(code.c_str());
}

static inline MieHostTables load_all_txt_tables_from_directory(const std::filesystem::path& dir)
{
    using std::filesystem::directory_iterator;
    std::unordered_map<int, std::filesystem::path> phaseFiles;
    std::unordered_map<int, std::filesystem::path> paramFiles;

    for (auto& e : directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        const auto fn = e.path().filename().string();

        if (fn.find("Mie_PhaseFunctionData_") != std::string::npos && fn.find(".txt") != std::string::npos) {
            int code = extract_code_3digits(fn);
            if (code >= 0) phaseFiles[code] = e.path();
        }
        if (fn.find("Mie_ScatteringParameters_") != std::string::npos && fn.find(".txt") != std::string::npos) {
            int code = extract_code_3digits(fn);
            if (code >= 0) paramFiles[code] = e.path();
        }
    }

    if (phaseFiles.empty()) {
        std::fprintf(stderr, "No phase files found in %s\n", dir.string().c_str());
        std::abort();
    }

    // code を昇順に並べる
    std::vector<int> codes;
    codes.reserve(phaseFiles.size());
    for (auto& kv : phaseFiles) codes.push_back(kv.first);
    std::sort(codes.begin(), codes.end());

    // まず 1つ目で軸サイズを確定
    std::vector<float> lambda_phase, theta_deg, pdf_theta_major;
    parse_phase_pdf_from_phase_file(phaseFiles[codes[0]], lambda_phase, theta_deg, pdf_theta_major);

    // g 側の波長も読む（一致確認用）
    std::vector<float> lambda_param, g_one;
    if (paramFiles.find(codes[0]) == paramFiles.end()) {
        std::fprintf(stderr, "Missing param file for code %03d\n", codes[0]);
        std::abort();
    }
    parse_g_table_from_param_file(paramFiles[codes[0]], lambda_param, g_one);

    // 波長一致チェック（完全一致でなくてもよいが、ズレがあるなら危険）
    if (lambda_phase.size() != lambda_param.size()) {
        std::fprintf(stderr, "Wavelength count mismatch phase=%zu param=%zu\n",
            lambda_phase.size(), lambda_param.size());
        std::abort();
    }
    for (size_t i=0; i<lambda_phase.size(); ++i) {
        float a = lambda_phase[i], b = lambda_param[i];
        if (std::fabs(a - b) > 1e-3f) {
            std::fprintf(stderr, "Wavelength mismatch at %zu: phase=%g param=%g\n", i, a, b);
            std::abort();
        }
    }

    MieHostTables ht;
    ht.Nd = (int)codes.size();
    ht.Nlambda = (int)lambda_phase.size();
    ht.Ntheta = (int)theta_deg.size();
    ht.lambda_nm = lambda_phase;
    ht.theta_deg = theta_deg;

    ht.diam_um.resize(ht.Nd);

    ht.g.resize((size_t)ht.Nd * (size_t)ht.Nlambda, 0.0f);
    ht.pdf.resize((size_t)ht.Nd * (size_t)ht.Nlambda * (size_t)ht.Ntheta, 0.0f);
    ht.cdf.resize((size_t)ht.Nd * (size_t)ht.Nlambda * (size_t)ht.Ntheta, 0.0f);

    // 全 code を読む
    for (int id=0; id<ht.Nd; ++id) {
        int code = codes[id];

        auto itP = phaseFiles.find(code);
        auto itS = paramFiles.find(code);
        if (itP == phaseFiles.end() || itS == paramFiles.end()) {
            std::fprintf(stderr, "Missing pair files for code %03d\n", code);
            std::abort();
        }

        // mean diameter（um）
        float mean_um = parse_mean_diameter_um_from_param_header(itS->second);
        ht.diam_um[id] = mean_um;

        // g table
        std::vector<float> lambda_nm, gvals;
        parse_g_table_from_param_file(itS->second, lambda_nm, gvals);
        if ((int)gvals.size() != ht.Nlambda) {
            std::fprintf(stderr, "g table size mismatch for code %03d\n", code);
            std::abort();
        }
        for (int il=0; il<ht.Nlambda; ++il) {
            ht.g[ht.idx2(id, il)] = gvals[il];
        }

        // phase pdf
        std::vector<float> lambda_nm2, theta_deg2, pdf_major;
        parse_phase_pdf_from_phase_file(itP->second, lambda_nm2, theta_deg2, pdf_major);
        if ((int)theta_deg2.size() != ht.Ntheta || (int)lambda_nm2.size() != ht.Nlambda) {
            std::fprintf(stderr, "phase size mismatch for code %03d\n", code);
            std::abort();
        }

        // pdf_major は [theta][lambda] で並んでいるので、最終レイアウト [d][lambda][theta] に詰め替え
        for (int it=0; it<ht.Ntheta; ++it) {
            for (int il=0; il<ht.Nlambda; ++il) {
                float v = pdf_major[(size_t)it * (size_t)ht.Nlambda + (size_t)il];
                ht.pdf[ht.idx3(id, il, it)] = v;
            }
        }

        // CDF 生成 & PDF 再正規化（d,lambda ごと）
        std::vector<float> tmp_pdf(ht.Ntheta);
        std::vector<float> tmp_cdf(ht.Ntheta);

        for (int il=0; il<ht.Nlambda; ++il) {
            for (int it=0; it<ht.Ntheta; ++it) {
                tmp_pdf[it] = ht.pdf[ht.idx3(id, il, it)];
            }

            build_cdf_and_renormalize_pdf_inplace(ht.Ntheta, ht.theta_deg.data(), tmp_pdf.data(), tmp_cdf.data());

            for (int it=0; it<ht.Ntheta; ++it) {
                ht.pdf[ht.idx3(id, il, it)] = tmp_pdf[it];
                ht.cdf[ht.idx3(id, il, it)] = tmp_cdf[it];
                // std::cout << "[" << it << "] pdf: " << tmp_pdf[it] << ", cdf: " << tmp_cdf[it] << std::endl; 
            }
        }

        std::printf("Loaded code %03d: mean_diameter=%g um\n", code, mean_um);
    }

    std::printf("Mie tables loaded: Nd=%d Nlambda=%d Ntheta=%d\n", ht.Nd, ht.Nlambda, ht.Ntheta);
    std::printf("lambda range: %g .. %g (nm)\n", ht.lambda_nm.front(), ht.lambda_nm.back());
    std::printf("theta range : %g .. %g (deg)\n", ht.theta_deg.front(), ht.theta_deg.back());

    std::cout << "Nd=" << ht.Nd
          << " Nlambda=" << ht.Nlambda
          << " Ntheta=" << ht.Ntheta
          << " pdf.size=" << ht.pdf.size()
          << " expected=" << (size_t)ht.Nd*ht.Nlambda*ht.Ntheta
          << std::endl;

    return ht;
}

// -------------------------
// GPU アップロード & texture object 生成
// -------------------------

static inline cudaTextureObject_t create_texture_object_from_array(
    cudaArray_t arr,
    cudaTextureFilterMode filterMode,
    bool normalizedCoords,
    int dims /*2 or 3*/)
{
    cudaResourceDesc resDesc{};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    cudaTextureDesc texDesc{};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = filterMode;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = normalizedCoords ? 1 : 0;

    // CDF は point 推奨、PDF は最初は point で OK（必要なら linear に）
    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
    return tex;
}

static inline MieGpuTextures upload_to_gpu_textures(const MieHostTables& ht)
{
    MieGpuTextures gpu;
    gpu.Nd = ht.Nd;
    gpu.Nlambda = ht.Nlambda;
    gpu.Ntheta = ht.Ntheta;

    // --- g: 2D array (width=Nlambda, height=Nd)
    {
        std::cout << "Uploading g..." << std::endl;
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        CUDA_CHECK(cudaMallocArray(&gpu.arr_g, &desc, (size_t)ht.Nlambda, (size_t)ht.Nd));

        const size_t widthBytes = (size_t)ht.Nlambda * sizeof(float);
        const size_t pitchBytes = widthBytes;
        CUDA_CHECK(cudaMemcpy2DToArray(
            gpu.arr_g,
            0, 0,
            ht.g.data(),
            pitchBytes,
            widthBytes,
            (size_t)ht.Nd,
            cudaMemcpyHostToDevice));

        // g は連続補間したいなら linear、まずは point でも可
        gpu.tex_g = create_texture_object_from_array(gpu.arr_g, cudaFilterModeLinear, true, 2);
    }

    // --- pdf: 3D array (width=Ntheta, height=Nlambda, depth=Nd)
    auto upload3D = [&](const float* src, cudaArray_t* outArr, cudaTextureObject_t* outTex, cudaTextureFilterMode filter)
    {
        std::cout << "Uploading 3D..." << std::endl;

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
        cudaExtent extentElem = make_cudaExtent((size_t)ht.Ntheta, (size_t)ht.Nlambda, (size_t)ht.Nd);

        CUDA_CHECK(cudaMalloc3DArray(outArr, &desc, extentElem, 0));
        std::cout << "cudaMalloc3DArray OK\n";


        size_t pitchBytes = (size_t)ht.Ntheta * sizeof(float);
        
        // copy は width がバイト
        cudaMemcpy3DParms cp{};
        cp.srcPtr = make_cudaPitchedPtr(
            (void*)src, pitchBytes, 
            (size_t)ht.Ntheta, 
            (size_t)ht.Nlambda);

        cp.dstArray = *outArr;
        cp.extent = extentElem;
        cp.kind = cudaMemcpyHostToDevice;

        CUDA_CHECK(cudaMemcpy3D(&cp));
        std::cout << "cudaMemcpy3D OK\n";

        *outTex = create_texture_object_from_array(*outArr, filter, true, 3);
        std::cout << "cudaCreateTextureObject OK\n";
    };
// #if defined(PHASE_FUNCTION_TABULATED)
    upload3D(ht.pdf.data(), &gpu.arr_pdf, &gpu.tex_pdf, cudaFilterModePoint);
    upload3D(ht.cdf.data(), &gpu.arr_cdf, &gpu.tex_cdf, cudaFilterModePoint); // CDF は point 推奨
// #endif
    return gpu;
}

} // namespace mie