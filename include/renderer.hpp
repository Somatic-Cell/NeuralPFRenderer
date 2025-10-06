#ifndef RENDERER_H_
#define RENDERER_H_

#include <optix.h>
#include <optix_stubs.h>
#include "helper_math.h"
#include "my_math.hpp"
#include "cuda_buffer.h"
#include "model.h"
#include "launch_params.h"
#include "sceneDescIO.hpp"
#include <cuda_runtime.h>

enum class OptixModuleIdentifier{
    OPTIX_MODULE_ID_RAYGEN=0,
    OPTIX_MODULE_ID_AH_RADIANCE,
    OPTIX_MODULE_ID_AH_SHADOW,
    OPTIX_MODULE_ID_CH_RADIANCE,
    OPTIX_MODULE_ID_CH_SHADOW,
    OPTIX_MODULE_ID_MISS_RADIANCE,
    OPTIX_MODULE_ID_MISS_SHADOW,
    OPTIX_MODULE_ID_BXDF_DIFFUSE,
    OPTIX_MODULE_ID_BXDF_PRINCIPLED,
    OPTIX_MODULE_ID_BXDF_GLASS,
    OPTIX_MODULE_ID_LIGHTSAMPLE,
    OPTIX_MODULE_ID_LENS,
    NUM_OPTIX_MODULE_IDENTIFIERS
};

enum class PostProcessCudaModuleIdentifier{
    CUDA_MODULE_ID_TONEMAP=0,
    NUM_CUDA_MODULE_IDENTIFIERS
};

enum RenderBufferType{
    COLOR=0,
    NORMAL,
    ALBEDO,
    NUM_RENDER_BUFFER_TYPES
};

struct Camera {
    // extrinsics
    float3 from         {make_float3(0.0f, 1.0f, 3.0f)};
    float3 at           {make_float3(0.0f, 1.0f, -1.0f)};
    float3 up           {make_float3(0.f, 1.0f, 0.f)};

    //intrinsics
    float focalLength   {50.0f};
    float fValue        {2.8f};
    float fov           {50.f};      // degrees
    float pintDist      {1.0f};
    float sensitivity   {1.0f};

    void setExtrinsics(const float3 _from, const float3 _at, const float3 _up){
        from = _from;
        at = _at;
        up = _up;
    }

    void setIntrinsics(const float _focalLength, const float _fValue, const float _fov, const float _pintDist, const float _sensitivity){
        focalLength = _focalLength;
        fValue = _fValue;
        fov = _fov;
        pintDist = _pintDist;
        sensitivity = _sensitivity;
    }
};

struct SpectrumData{
    float lambdaMin;
    float lambdaMax;
    std::array<std::vector<float>, 3> data;
};

class Renderer 
{
public:
    Renderer(std::vector<const Model*> models, sceneIO::Scene sceneDesc);

    void render();
    void resize(const int2 & newSize);
    void downloadPixels(uint32_t m_h_pixels[]);
    void setCamera(const Camera &camera);
    void setEnvMap(const std::string& envMapFileName);
    void setNumDevices(const int numDevices);
    int  getNumDevices() const;
    void setCameraModel(const int cameraModel);
    void setRenderBufferType(const int renderBufferType);
    int getRenderBufferType() const;
    const LaunchParams  getLaunchParams() const;
    const CUDABuffer&   getFinalColorBuffer() const;

    // Tonemap 用
    void setWhite(const float white);
    float getWhite() const;
    void setExposure(const float exposure);
    float getExposure() const;

    

protected:
    void computeFinalPixelColors();
    void initOptix();
    void createContext();
    void createOptiXModule();
    void createRaygenPrograms();
    void createMissPrograms();
    void createHitgroupPrograms();
    void createCallablePrograms();
    void createPipeline();
    void buildSBT();

    bool buildAccel();
    
    void createTextures();
    void createCUDAModule();

    void createLightTable();

    // Spectral rendering 用
    void uploadSpectrumData();
    SpectrumData loadSpectrumDataFromCSV(const std::string path);

    CUcontext           m_cudaContext;
    CUstream            m_stream;
    cudaDeviceProp      m_deviceProps;

    OptixDeviceContext  m_optixContext;

    // ビルドするパイプライン
    OptixPipeline               m_pipeline                  {};
    OptixPipelineCompileOptions m_pipelineCompileOptions    {};
    OptixPipelineLinkOptions    m_pipelineLinkOptions       {};

    std::vector<OptixModule>    m_module                    {};
    OptixModuleCompileOptions   m_moduleCompileOptions      {};

    std::vector<OptixProgramGroup>  m_raygenPrograms        {};
    CUDABuffer                      m_raygenRecordsBuffer   {};
    std::vector<OptixProgramGroup>  m_missPrograms          {};
    CUDABuffer                      m_missRecordsBuffer     {};
    std::vector<OptixProgramGroup>  m_hitgroupPrograms      {};
    CUDABuffer                      m_hitgroupRecordsBuffer {};
    std::vector<OptixProgramGroup>  m_callablePrograms      {};
    CUDABuffer                      m_callableRecordsBuffer {};
    OptixShaderBindingTable         m_sbt                   {};

    LaunchParams    m_launchParams;
    CUDABuffer      m_launchParamsBuffer;

    // デノイザに入れるバッファ
    CUDABuffer  m_fbColor;
    CUDABuffer  m_fbNormal;
    CUDABuffer  m_fbAlbedo;

    CUDABuffer  m_denoisedBuffer;
    CUDABuffer  m_finalColorBuffer;

    // OptiX デノイズ用
    OptixDenoiser   m_denoiser  {nullptr};
    CUDABuffer      m_denoiserScratch;
    CUDABuffer      m_denoiserState;
    CUDABuffer      m_denoiserIntensity;

    CUmodule                    m_cudaModule;
    std::vector<CUfunction>     m_cudaFunction;

    Camera          m_lastSetCamera;

    std::vector<const Model*>   m_models;
    CUDABuffer                  m_objectMatrix;
    CUDABuffer                  m_normalMatrix;

    std::vector<CUDABuffer> m_vertexBuffer;
    std::vector<CUDABuffer> m_indexBuffer;
    std::vector<CUDABuffer> m_diffuseTexcoordBuffer;
    std::vector<CUDABuffer> m_normalTexcoordBuffer;
    std::vector<CUDABuffer> m_emissiveTexcoordBuffer;
    std::vector<CUDABuffer> m_normalBuffer;
    std::vector<CUDABuffer> m_tangentBuffer;
    std::vector<CUDABuffer> m_colorBuffer;

    std::vector<CUDABuffer> m_GASBuffer;
    CUDABuffer              m_IASBuffer;
    CUDABuffer              m_instance;

    std::vector<OptixTraversableHandle> m_gasHandle;
    OptixTraversableHandle              m_iasHandle;


    std::vector<cudaArray_t>            m_textureArrays;    // テクスチャの実体が入ったデータのベクトル
    std::vector<cudaTextureObject_t>    m_textureObjects;   // テクスチャにアクセスするためのハンドル
    
    CUDABuffer  m_envMapBuffer;
    int m_numDevices            {0};
    bool m_isAccumulate         {true};

    std::vector<std::string> m_optixModuleFileNames;    // optix 用の .ptx .optixir のコード一覧
    std::vector<std::string> m_cudaModuleFileNames;     // cuda 用の.ptx のコード一覧

    int m_renderBufferType      {COLOR};

    CUDABuffer  m_envMap;
    cudaArray_t m_envMapArray   {nullptr};
    cudaTextureObject_t m_envMapTex;
    
    std::vector<LightDefinition>    m_lightDefinitionTable;
    std::vector<TriangleLightData>  m_triangleLightDataTable;
    CUDABuffer  m_lightDefinitionBuffer;
    CUDABuffer  m_triangleLightDataBuffer;

    int         m_envPatchWidth         {1024};
    int         m_envPatchHeight        {512};
    CUDABuffer  m_envCDFCoarseMarginal;     // H
    CUDABuffer  m_envCDFCoarseConditional;  // W x H
    CUDABuffer  m_envPatchWeight;           // W x H

    float m_exposure    {0.5f};
    float m_white       {5.0f};

    sceneIO::Scene m_sceneDesc;

    // for spectral rendering
    SpectrumData    m_xyz;
    SpectrumData    m_rgbUpSamplingBasis;

    // XYZ 等色関数
    std::vector<cudaArray_t>            m_xyzFuncArrays;    // 関数の実体が入ったデータのベクトル
    std::vector<cudaTextureObject_t>    m_xyzFuncObjects;   // 関数にアクセスするためのハンドル
    
    // RGB テクスチャのアップサンプリング用の関数
    std::vector<cudaArray_t>            m_rgbUpSampleFuncArrays;    // 関数の実体が入ったデータのベクトル
    std::vector<cudaTextureObject_t>    m_rgbUpSampleFuncObjects;   // 関数にアクセスするためのハンドル
};



#endif // RENDERER_H_