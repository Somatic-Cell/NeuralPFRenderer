#ifndef RENDERER_H_
#define RENDERER_H_

#include <optix.h>
#include <optix_stubs.h>
#include "helper_math.h"
#include "cuda_buffer.h"
#include "model.h"
#include "launch_params.h"
#include <cuda_runtime.h>

enum class OptixModuleIdentifier{
    OPTIX_MODULE_ID_HIT=0,
    OPTIX_MODULE_ID_LENS,
    OPTIX_MODULE_ID_BXDF_DIFFUSE,
    OPTIX_MODULE_ID_BXDF_PRINCIPLED,
    OPTIX_MODULE_ID_BXDF_GLASS,
    OPTIX_MODULE_ID_LIGHTSAMPLE,
    // MODULE_ID_MISS,
    // MODULE_ID_CLOSESTHIT,
    // MODULE_ID_ANYHIT,
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
    float3 from         {make_float3(-15.0f, 15.0f, -7.0f)};
    float3 at           {make_float3(0.0f, 0.0f, 0.0f)};
    float3 up           {make_float3(0.f, 1.0f, 0.f)};

    //intrinsics
    float focalLength   {50.0f};
    float fValue        {2.8f};
    float fov           {50.f};      // degrees
    float pintDist      {1.0f};
    float sensitivity   {1.0f};
};


class Renderer 
{
public:
    Renderer(const Model* model);

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

    const Model     *m_model;

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
    
};



#endif // RENDERER_H_