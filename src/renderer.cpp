#include "renderer.hpp"
#include "texture.hpp"
#include <optix_function_table_definition.h>
// #include <filesystem>
// #include <iostream>
// #include <fstream>
#include <algorithm>
#include <mutex>
#include <string.h>


#define STB_IMAGE_IMPLEMENTATION
#include "ptx_data_reader.hpp"
#include "stb_image.h"
#include "my_debug_tool.hpp"

#define _USE_MATH_DEFINES
#include <math.h>

#if defined(_WIN32)
#include <windows.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#else
#include <unistd.h>
#endif

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) CallableRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void* data;
};

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    TriangleMeshSBTData data;
};


Renderer::Renderer(const Model* model) : m_model(model)
{
    m_optixModuleFileNames.resize(static_cast<int>(OptixModuleIdentifier::NUM_OPTIX_MODULE_IDENTIFIERS));
#if defined(USE_OPTIX_IR)
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_HIT)]             = std::string("hit.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LENS)]            = std::string("lens.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_DIFFUSE)]    = std::string("diffuse_brdf.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_PRINCIPLED)] = std::string("disney_principled_brdf.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_GLASS)]      = std::string("glass_bsdf.optixir");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LIGHTSAMPLE)]     = std::string("light_sample.optixir");
#else
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_HIT)]             = std::string("hit.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LENS)]            = std::string("lens.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_DIFFUSE)]    = std::string("diffuse_brdf.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_PRINCIPLED)] = std::string("disney_principled_brdf.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_GLASS)]      = std::string("glass_bsdf.ptx");
    m_optixModuleFileNames[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LIGHTSAMPLE)]     = std::string("light_sample.ptx");
#endif
    
    m_cudaModuleFileNames.resize(static_cast<int>(PostProcessCudaModuleIdentifier::NUM_CUDA_MODULE_IDENTIFIERS));
    m_cudaModuleFileNames[static_cast<int>(PostProcessCudaModuleIdentifier::CUDA_MODULE_ID_TONEMAP)] = std::string("tonemap.ptx");



    initOptix();
    
    std::cout << "# Simple Renderer: creating OptiX context..." << std::endl;
    createContext();

    std::cout << "# Simple Renderer: setting up OptiX module..." << std::endl;
    createOptiXModule();

    std::cout << "# Simple Renderer: creating raygen programs ..." << std::endl;
    createRaygenPrograms();
    std::cout << "# Simple Renderer: creating miss programs ..." << std::endl;
    createMissPrograms();
    std::cout << "# Simple Renderer: creating callable programs ..." << std::endl;
    createCallablePrograms();
    std::cout << "# Simple Renderer: creating hitgroup programs ..." << std::endl;
    createHitgroupPrograms();

    buildAccel();
    m_launchParams.traversable = m_iasHandle;

    std::cout << "# Simple Renderer: setting up OptiX pipeline ..." << std::endl;
    createPipeline();
    
    createTextures();

    std::cout << "# Simple Renderer: building shader binding table..." << std::endl;
    buildSBT();

    m_launchParamsBuffer.alloc(sizeof(m_launchParams)); // alloc だけ？
    std::cout << "# Simple Renderer: context, module, pipeline, etc, all set up ..." << std::endl;
    std::cout << "# Simple Renderer: Optix 8 fully set up..." << std::endl;

    
    std::cout << "# Simple Renderer: setting up CUDA module..." << std::endl;
    createCUDAModule();

    std::cout << "# Simple Renderer: create light table..." << std::endl;
    createLightTable();
    std::cout << "# Simple Renderer: CUDA kernel fully set up..." << std::endl;
}


void Renderer::createTextures()
{
    int numTextures = (int)m_model->textures.size();
  
    m_textureArrays.resize(numTextures);
    m_textureObjects.resize(numTextures);

    for(int textureID = 0; textureID < numTextures; textureID++){
        auto texture = m_model->textures[textureID];

        cudaResourceDesc resDesc = {};

        cudaChannelFormatDesc channelDesc;
        int32_t width = texture->resolution.x;
        int32_t height = texture->resolution.y;
        int32_t numComponents = 4; //rgba
        int32_t pitch = width * numComponents * sizeof(uint8_t);
        channelDesc = cudaCreateChannelDesc<uchar4>();

        cudaArray_t &pixelArray = m_textureArrays[textureID];
        CUDA_CHECK(cudaMallocArray(&pixelArray, &channelDesc, width, height));
        CUDA_CHECK(cudaMemcpy2DToArray(pixelArray, 0, 0, texture->pixel, pitch, pitch, height, cudaMemcpyHostToDevice));
        
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = pixelArray;

        // テクスチャのふるまいを決定
        cudaTextureDesc texDesc = {};                               
        texDesc.addressMode[0]      = cudaAddressModeWrap;          // テクスチャが範囲外になったときの対処法 (タイリング)
        texDesc.addressMode[1]      = cudaAddressModeWrap;          // テクスチャが範囲外になったときの対処法 (タイリング)
        texDesc.filterMode          = cudaFilterModeLinear;         // テクセルの補間方法
        texDesc.readMode            = cudaReadModeNormalizedFloat;  // GPUg　側で読み込むときのフォーマット (0-1)
        texDesc.normalizedCoords    = 1;                            // テクスチャ座標を [0,1] 範囲で指定
        texDesc.maxAnisotropy       = 1;                            // 異方性フィルタリングの強度 (1:無効)
        texDesc.maxMipmapLevelClamp = 99;                           // ミップレベルの最大
        texDesc.minMipmapLevelClamp = 0;                            // ミップレベルの最小
        texDesc.mipmapFilterMode    = cudaFilterModeLinear;         // ミップマップ間の補間
        texDesc.borderColor[0]      = 1.0f;                         // 範囲外で指定される色
        if(texture->isDiffuseTexture){
            texDesc.sRGB            = 1;                            // 1: 読み込む画像は sRGB 画像と考え，ガンマ補正を適用して読み込み
        } else {
            texDesc.sRGB            = 0;                            // そのまま読み込み
        }

        cudaTextureObject_t cudaTex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&cudaTex, &resDesc, &texDesc, nullptr));
        m_textureObjects[textureID] = cudaTex;
    }
}

bool Renderer::buildAccel()
{
    PING;
    PRINT(m_model->meshes.size());
    
    const int numMeshes = (int)m_model->meshes.size();

    m_vertexBuffer.resize(numMeshes);
    m_indexBuffer.resize(numMeshes);
    m_normalBuffer.resize(numMeshes);
    m_diffuseTexcoordBuffer.resize(numMeshes);
    m_normalTexcoordBuffer.resize(numMeshes);
    m_emissiveTexcoordBuffer.resize(numMeshes);
    m_tangentBuffer.resize(numMeshes);
    m_colorBuffer.resize(numMeshes);
    
    m_GASBuffer.resize(numMeshes);
    m_gasHandle.resize(numMeshes);
  
    // ===========================
    // GAS (BLAS) を構築　（メッシュごとに 1 つ）
    // ===========================
    for(int meshID = 0; meshID < numMeshes; meshID++){
        
        // 三角形の入力
        OptixBuildInput     triangleInput;
        CUdeviceptr         d_vertices;
        CUdeviceptr         d_indices;
        uint32_t            triangleInputFlags;

        TriangleMesh &mesh = *m_model->meshes[meshID];
        m_vertexBuffer[meshID].allocAndUpload(mesh.vertex);
        m_indexBuffer[meshID].allocAndUpload(mesh.index);
        if(!mesh.normal.empty()){
            m_normalBuffer[meshID].allocAndUpload(mesh.normal);
        }
        if(!mesh.diffuseTexcoord.empty()){
            m_diffuseTexcoordBuffer[meshID].allocAndUpload(mesh.diffuseTexcoord);
        }
        if(!mesh.normalTexcoord.empty()){
            m_normalTexcoordBuffer[meshID].allocAndUpload(mesh.normalTexcoord);
        }
        if(!mesh.emissiveTexcoord.empty()){
            m_emissiveTexcoordBuffer[meshID].allocAndUpload(mesh.emissiveTexcoord);
        }
        if(!mesh.tangent.empty()){
            m_tangentBuffer[meshID].allocAndUpload(mesh.tangent);
        }


        triangleInput = {}; // ここに情報を入れていく
        triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        d_vertices = m_vertexBuffer[meshID].getDevicePointer();
        d_indices = m_indexBuffer[meshID].getDevicePointer();

        // 頂点情報
        triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
        triangleInput.triangleArray.numVertices         = (int)mesh.vertex.size();
        triangleInput.triangleArray.vertexBuffers       = &d_vertices;

        // 頂点のインデックス情報
        triangleInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput.triangleArray.indexStrideInBytes  = sizeof(uint3);
        triangleInput.triangleArray.numIndexTriplets    = (int)mesh.index.size();
        triangleInput.triangleArray.indexBuffer         = d_indices;

        triangleInputFlags = OPTIX_GEOMETRY_FLAG_NONE;  // 特別な設定を行わない
        // MEMO: 
        // OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT : any_hit シェーダを無効化
        // OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL : 1回だけ any-hit をよぶ
        // OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING :  バックフェースカリングを無効化　
        //                                                      （薄いマテリアルをレンダリングしたいときにたてる）

        // 単一の SBT レコードを使用
        triangleInput.triangleArray.flags                       = &triangleInputFlags;
        triangleInput.triangleArray.numSbtRecords               = 1;        // 単一の SBT レコード
        // 単一の場合，未使用なので 0 
        triangleInput.triangleArray.sbtIndexOffsetBuffer        = 0;        
        triangleInput.triangleArray.sbtIndexOffsetSizeInBytes   = 0;
        triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;


        // GAS のセットアップ
        OptixAccelBuildOptions  accelOptions    = {};
        accelOptions.buildFlags                 = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        if(getNumDevices() == 1){
            // 複数 GPU を使うと性能が悪化する恐れがあるため，単一 GPU の場合のみオプションを追加
            accelOptions.buildFlags             |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        }
        accelOptions.motionOptions.numKeys      = 1;    // モーションブラーなし． numKeys > 1 でブラー補間
        accelOptions.operation                  = OPTIX_BUILD_OPERATION_BUILD; // 新規構築． ..._UPDATE を使うと更新


        // AS に必要なバッファサイズの見積もり
        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK(
            optixAccelComputeMemoryUsage(
                m_optixContext,
                &accelOptions,
                &triangleInput,
                1,              // メッシュの個数．今回は1個ずつ作成しているので1
                &gasBufferSizes
            )
        );

        // Compaction の準備
        // AS を最悪のケースを想定して作るので，作成後に不要となった部分を圧縮することで VRAM を節約可能
        
        CUDABuffer compactedSizeBuffer;
        compactedSizeBuffer.alloc(sizeof(uint64_t));

        OptixAccelEmitDesc emitDesc;                        // AS 構築時の圧縮後サイズ，AABB の範囲，インスタンスの変換行列... を出力してくれる補助出力の構造体
        emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE; // 圧縮後のサイズを返すように指定
        emitDesc.result = compactedSizeBuffer.getDevicePointer();

        // GAS の構築
        CUDABuffer tempBuffer;
        tempBuffer.alloc(gasBufferSizes.tempSizeInBytes);

        CUDABuffer outputBuffer;
        outputBuffer.alloc(gasBufferSizes.outputSizeInBytes);

        OPTIX_CHECK(
            optixAccelBuild(
                m_optixContext,
                0, // stream
                &accelOptions,
                &triangleInput,
                1,              // メッシュの個数．今回は1個ずつ作成しているので1
                
                // temp
                tempBuffer.getDevicePointer(),
                tempBuffer.getSizeInBytes(),

                // output
                outputBuffer.getDevicePointer(),
                outputBuffer.getSizeInBytes(),

                &m_gasHandle[meshID],

                &emitDesc, 1
            )
        );

        CUDA_SYNC_CHECK();
        // Compaction の実行
        uint64_t compactedSize;
        compactedSizeBuffer.download(&compactedSize, 1);
        m_GASBuffer[meshID].alloc(compactedSize);
        OPTIX_CHECK(
            optixAccelCompact(
                m_optixContext,
                0,
                m_gasHandle[meshID],
                m_GASBuffer[meshID].getDevicePointer(),
                m_GASBuffer[meshID].getSizeInBytes(),
                &m_gasHandle[meshID]
            )
        );

        CUDA_SYNC_CHECK();

        // クリーンアップ
        outputBuffer.free();
        tempBuffer.free();
        compactedSizeBuffer.free();
    }
    
    // ==========================
    // IAS (TLAS) の構築
    // ==========================

    std::vector<OptixInstance> instances(numMeshes);

    for(unsigned int meshID = 0; meshID < numMeshes; meshID++){
        OptixInstance & inst = instances[meshID];
        memset(&inst, 0, sizeof(OptixInstance));

        float transform[12] = {
            1, 0, 0, 0,
            0, 1, 0, 0, 
            0, 0, 1, 0
        };

        memcpy(inst.transform, transform, sizeof(float) * 12);

        inst.instanceId     = meshID;
        inst.sbtOffset      = meshID * RAY_TYPE_COUNT;
        inst.visibilityMask = 255;
        inst.flags          = OPTIX_INSTANCE_FLAG_NONE;     // インスタンスの挙動を指定．
                                                            // OPTIX_INSTNCE_FLAG_NONE:                 デフォルト
                                                            // OPTIX_INSTNCE_FLAG_DISABLE_TRANSFORM:    transform 行列を無視 (ワールド座標に直置き)
                                                            // OPTIX_INSTNCE_FLAG_DISABLE_ANYHIT:       AnyHit シェーダを無視
                                                            // OPTIX_INSTNCE_FLAG_ENFORCE_ANYHIT:       AnyHit シェーダを必ず呼ぶ
        inst.traversableHandle  = m_gasHandle[meshID];
    }

    m_instance.allocAndUpload(instances);

    OptixBuildInput instanceInput = {};
    instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceInput.instanceArray.instances    = m_instance.getDevicePointer();
    instanceInput.instanceArray.numInstances = static_cast<unsigned int>(instances.size());


    // IAS のセットアップ
    OptixAccelBuildOptions  accelOptions    = {};
    accelOptions.buildFlags                 = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    if(getNumDevices() == 1){
        // 複数 GPU を使うと性能が悪化する恐れがあるため，単一 GPU の場合のみオプションを追加
        accelOptions.buildFlags             |= OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    }
    accelOptions.motionOptions.numKeys      = 1;    // モーションブラーなし． numKeys > 1 でブラー補間
    accelOptions.operation                  = OPTIX_BUILD_OPERATION_BUILD; // 新規構築． ..._UPDATE を使うと更新
    
    // AS に必要なバッファサイズの見積もり
    OptixAccelBufferSizes iasBufferSizes;
    OPTIX_CHECK(
        optixAccelComputeMemoryUsage(
            m_optixContext,
            &accelOptions,
            &instanceInput,
            1,              // メッシュの個数．今回は1個ずつ作成しているので1
            &iasBufferSizes
        )
    );

    // Compaction の準備
    // IAS を最悪のケースを想定して作るので，作成後に不要となった部分を圧縮することで VRAM を節約可能
    CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;                        // IAS 構築時の圧縮後サイズ，AABB の範囲，インスタンスの変換行列... を出力してくれる補助出力の構造体
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE; // 圧縮後のサイズを返すように指定
    emitDesc.result = compactedSizeBuffer.getDevicePointer();

    // IAS の構築
    CUDABuffer tempBuffer;
    tempBuffer.alloc(iasBufferSizes.tempSizeInBytes);

    CUDABuffer outputBuffer;
    outputBuffer.alloc(iasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(
        optixAccelBuild(
            m_optixContext,
            0, // stream
            &accelOptions,
            &instanceInput,
            1,         
            
            // temp
            tempBuffer.getDevicePointer(),
            tempBuffer.getSizeInBytes(),

            // output
            outputBuffer.getDevicePointer(),
            outputBuffer.getSizeInBytes(),

            &m_iasHandle,

            &emitDesc, 1
        )
    );

    CUDA_SYNC_CHECK();
    
    // Compaction の実行
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    m_IASBuffer.alloc(compactedSize);
    OPTIX_CHECK(
        optixAccelCompact(
            m_optixContext,
            0,
            m_iasHandle,
            m_IASBuffer.getDevicePointer(),
            m_IASBuffer.getSizeInBytes(),
            &m_iasHandle
        )
    );

    CUDA_SYNC_CHECK();

    // クリーンアップ
    outputBuffer.free();
    tempBuffer.free();
    compactedSizeBuffer.free();

    return true;
}

// OptiX の初期化とエラーチェック
void Renderer::initOptix()
{
    std::cout << "Initializing optix ..." << std::endl;

    // CUDA を使える GPU が搭載されているかどうか確認
    cudaFree(0);
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if(numDevices == 0){
        throw std::runtime_error("# no CUDA capable devices are found :(");
    }

    std::cout << "found" << numDevices << "CUDA devices" << std::endl;

    // Optix の初期化
    OPTIX_CHECK(optixInit());
    std::cout << "Successfully initialised OptiX!" << std::endl;

    setNumDevices(numDevices);
}

//callback 関数
static void contextLogCallback( unsigned int level,
                            const char *tag,
                            const char *message,
                            void *)
{
    fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

void Renderer::createContext()
{
    const int deviceID = 0;
    CUDA_CHECK(cudaSetDevice(deviceID));
    CUDA_CHECK(cudaStreamCreate(&m_stream));

    cudaGetDeviceProperties(&m_deviceProps, deviceID);
    std::cout << "Running on device: " << m_deviceProps.name << std::endl;

    CUresult cuRes = cuCtxGetCurrent(&m_cudaContext);
    if(cuRes != CUDA_SUCCESS){
        fprintf(stderr, "ERROR querying current context: error code %d\n", cuRes);
    }

    OPTIX_CHECK(
        optixDeviceContextCreate(
            m_cudaContext, 
            0,              // OptixDeviceContextOption を省略し，デフォルト設定を使用
            &m_optixContext // 生成されたコンテキストをここに出力，格納
        )
    );
    

    // OptixDeviceContextOptions options = {};
    // options.logCallbackFunction = contextLogCallback;
    // options.


    OPTIX_CHECK(
        optixDeviceContextSetLogCallback (
            m_optixContext,         // 対象のコンテキスト
            contextLogCallback,     // ログを受け取るコールバック関数
            nullptr,                // コールバックされるデータ
            4                       // ログレベル (4 が最も詳細)
        )
    );
}

void Renderer::createOptiXModule()
{
    m_moduleCompileOptions                      = {};
    m_moduleCompileOptions.maxRegisterCount     = 50;
    m_moduleCompileOptions.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    m_moduleCompileOptions.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    
    m_pipelineCompileOptions                        = {};
    m_pipelineCompileOptions.traversableGraphFlags  =                               // 設計した AS に合わせて最小限のフラグを使うのが望ましい
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;                 // トップレベル IAS + GAS
    m_pipelineCompileOptions.usesMotionBlur         = false;                        // モーションブラーなし
    m_pipelineCompileOptions.numPayloadValues       = 2;                            // payload のスロット数．
    m_pipelineCompileOptions.numAttributeValues     = 2;                            // ヒットしたプリミティブから返される補助情報数．例えば，重心座標 (u, v)
    m_pipelineCompileOptions.exceptionFlags         =                               // 例外処理を有効化するフラグ
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW                                             // コールスタックが溢れた場合の例外
        | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH                                              // 最大再帰深度を超えた場合に発生する例外をキャッチ
        | OPTIX_EXCEPTION_FLAG_USER;                                                    // 
    m_pipelineCompileOptions.pipelineLaunchParamsVariableName   = "optixLaunchParams";  // パラメータのグローバル変数名．CUDA 側と一致させる必要があるので注意
    
    m_pipelineLinkOptions.maxTraceDepth             = 4;                            // __raygen__ から始まるレイパスで，最大何回 OptixTrace() を呼び出すか
                                                                                    // NEE などの可視性の判断を行う
                                                                                    // 追跡可能なレイの最大深さ．(セカンダリレイまで)



    m_module.resize(m_optixModuleFileNames.size());

    for(size_t i = 0; i < m_optixModuleFileNames.size(); i++){
        std::vector<char> ptxCode = readData(m_optixModuleFileNames[i]);
        char log[2048];
        size_t sizeOfLog = sizeof(log);

    // モジュールの作成
#if OPTIX_VERSION >= 70700
        OPTIX_CHECK(optixModuleCreate(m_optixContext,
                                        &m_moduleCompileOptions,
                                        &m_pipelineCompileOptions,
                                        ptxCode.data(),
                                        ptxCode.size(),
                                        log,
                                        &sizeOfLog,
                                        &m_module[i]
                                        ));
        
#else
        OPTIX_CHECK(optixModuleCreateFromPTX(m_optixContext,
                                        &m_moduleCompileOptions,
                                        &m_pipelineCompileOptions,
                                        ptxCode.data(),
                                        ptxCode.size(),
                                        log,
                                        &sizeOfLog,
                                        &m_module[i]
                                        ));
#endif
        if(sizeOfLog > 1) PRINT(log);
    }
        
}

void Renderer::createRaygenPrograms()
{
    m_raygenPrograms.resize(1);

    OptixProgramGroupOptions    pgOptions   = {};
    OptixProgramGroupDesc       pgDesc      = {};
    pgDesc.kind                             = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module                    = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_HIT)];
    pgDesc.raygen.entryFunctionName         = "__raygen__renderFrame";

    // m_raygenPrograms に 登録
    char log[2048];
    size_t sizeOfLog = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &sizeOfLog,
                                        &m_raygenPrograms[0]
                                        ));
    if (sizeOfLog > 1) PRINT(log);
}

void Renderer::createMissPrograms()
{
    m_missPrograms.resize(RAY_TYPE_COUNT);

    OptixProgramGroupOptions    pgOptions   = {};
    OptixProgramGroupDesc       pgDesc      = {};
    pgDesc.kind                             = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module                      = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_HIT)];

    char log[2048];
    size_t sizeOfLog = sizeof(log);
    
    // m_missPrograms に radiance ray を登録
    pgDesc.miss.entryFunctionName = "__miss__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &sizeOfLog,
                                        &m_missPrograms[RADIANCE_RAY_TYPE]
                                        ));
    if (sizeOfLog > 1) PRINT(log);

    // m_missPrograms に shadow ray を登録
    pgDesc.miss.entryFunctionName = "__miss__shadow";

    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &sizeOfLog,
                                        &m_missPrograms[SHADOW_RAY_TYPE]
                                        ));
    if (sizeOfLog > 1) PRINT(log);
    
}

void Renderer::createHitgroupPrograms()
{
    m_hitgroupPrograms.resize(RAY_TYPE_COUNT);

    OptixProgramGroupOptions    pgOptions   = {};
    OptixProgramGroupDesc       pgDesc      = {};
    pgDesc.kind                             = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH                = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_HIT)];
    pgDesc.hitgroup.moduleAH                = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_HIT)];

    char log[2048];
    size_t sizeOfLog = sizeof(log);
    
    // radiance ray の登録
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &sizeOfLog,
                                        &m_hitgroupPrograms[RADIANCE_RAY_TYPE]
                                        ));
    if (sizeOfLog > 1) PRINT(log);

    // shadow ray の登録
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

    OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                        &pgDesc,
                                        1,
                                        &pgOptions,
                                        log,
                                        &sizeOfLog,
                                        &m_hitgroupPrograms[SHADOW_RAY_TYPE]
                                        ));
    if (sizeOfLog > 1) PRINT(log);
}

void Renderer::createCallablePrograms()
{
    const int numCallablePrograms = NUM_LENS_TYPE + NUM_BXDF + NUM_LIGHT_TYPE;
    m_callablePrograms.resize(numCallablePrograms);

    OptixProgramGroupOptions            pgOptions   = {};
    std::vector<OptixProgramGroupDesc>  pgDesc(numCallablePrograms);
    OptixProgramGroupDesc*              pgd;
    char log[2048];
    size_t sizeOfLog = sizeof(log);
    
    
    // レンズシェーダ
    pgd = &pgDesc[LENS_TYPE_PINHOLE];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LENS)];
    pgd->callables.entryFunctionNameDC    = "__direct_callable__lens_pinhole";

    pgd = &pgDesc[LENS_TYPE_THIN_LENS];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LENS)];
    pgd->callables.entryFunctionNameDC    = "__direct_callable__lens_thinLens";

    // BRDF のサンプリングと評価
    int offset = NUM_LENS_TYPE;
    // Diffuse
    pgd = &pgDesc[offset + BXDF_TYPE_DIFFUSE_SAMPLE];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_DIFFUSE)];
    pgd->callables.entryFunctionNameDC    = "__direct_callable__brdf_diffuse_sample";

    pgd = &pgDesc[offset + BXDF_TYPE_DIFFUSE_EVAL];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_DIFFUSE)];
    pgd->callables.entryFunctionNameDC    = "__direct_callable__brdf_diffuse_eval";

    // Disney principled
    pgd = &pgDesc[offset + BXDF_TYPE_PRINCIPLED_SAMPLE];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_PRINCIPLED)];
    pgd->callables.entryFunctionNameDC    = "__direct_callable__brdf_principled_sample";

    pgd = &pgDesc[offset + BXDF_TYPE_PRINCIPLED_EVAL];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_PRINCIPLED)];
    pgd->callables.entryFunctionNameDC    = "__direct_callable__brdf_principled_eval";

    // Glass
    pgd = &pgDesc[offset + BXDF_TYPE_GLASS_SAMPLE];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_GLASS)];
    pgd->callables.entryFunctionNameDC    = "__direct_callable__bsdf_glass_sample";

    pgd = &pgDesc[offset + BXDF_TYPE_GLASS_EVAL];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_BXDF_GLASS)];
    pgd->callables.entryFunctionNameDC    = "__direct_callable__bsdf_glass_eval";

    // ライトサンプリング
    offset = NUM_LENS_TYPE + NUM_BXDF; 
    
    // 環境マップ
    pgd = &pgDesc[offset + LIGHT_TYPE_ENV_SPHERE];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LIGHTSAMPLE)];
    pgd->callables.entryFunctionNameDC    = "__direct_callable__light_env_sphere_constant";

    // メッシュ
    pgd = &pgDesc[offset + LIGHT_TYPE_TRIANGLE];
    pgd->kind                             = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    pgd->flags                            = OPTIX_PROGRAM_GROUP_FLAGS_NONE;
    pgd->callables.moduleDC               = m_module[static_cast<int>(OptixModuleIdentifier::OPTIX_MODULE_ID_LIGHTSAMPLE)];
    pgd->callables.entryFunctionNameDC    = "__direct_callable__light_triangle";

    
    // 登録
    for(int callableID = 0; callableID < numCallablePrograms; callableID++){
        OPTIX_CHECK(optixProgramGroupCreate(m_optixContext,
                                            &pgDesc[callableID],
                                            1,
                                            &pgOptions,
                                            log,
                                            &sizeOfLog,
                                            &m_callablePrograms[callableID]
                                            ));
    }  
}

void Renderer::createPipeline()
{
    std::vector<OptixProgramGroup> programGroups;
    for(auto pg : m_raygenPrograms)
    {
        programGroups.push_back(pg);
    }
    for(auto pg : m_missPrograms)
    {
        programGroups.push_back(pg);
    }
    for(auto pg : m_hitgroupPrograms)
    {
        programGroups.push_back(pg);
    }
    for(auto pg : m_callablePrograms)
    {
        programGroups.push_back(pg);
    }

    char log[2048];
    size_t sizeOfLog = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        m_optixContext,
        &m_pipelineCompileOptions,
        &m_pipelineLinkOptions,
        programGroups.data(),
        (int)programGroups.size(),
        log,
        &sizeOfLog,
        &m_pipeline
    ));
    if (sizeOfLog > 1) PRINT(log);

    OPTIX_CHECK(optixPipelineSetStackSize
                    (
                        // スタックサイズを設定するパイプライン
                        m_pipeline,
                        // IS AH から呼び出される Direct callable のスタックサイズの要件
                        2*1024,
                        // Raygen, Miss, Hit から呼び出される Direct callable のスタックサイズの要件
                        2*1024,
                        // レイトレの深さや Any0Hit に備えるスタックの要件
                        2*1024,
                        // トレースに渡される操作可能なグラフの最大深さ
                        2 // SINGLE IAS
                    )
    );
    if(sizeOfLog > 1) PRINT(log);
    
}

void Renderer::buildSBT()
{
    // Raygen records の登録
    std::vector<RaygenRecord> raygenRecords;
    for(int i = 0; i < m_raygenPrograms.size(); i++)
    {
        RaygenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_raygenPrograms[i], &rec));   // RaygenRecord.header に， m_raygenPrograms[i] を登録
        rec.data = nullptr;
        raygenRecords.push_back(rec);
    }
    m_raygenRecordsBuffer.allocAndUpload(raygenRecords);        // GPU に情報を転送
    m_sbt.raygenRecord  = m_raygenRecordsBuffer.getDevicePointer();   // raygenRecords バッファのデバイス上の先頭アドレスを sbt に登録

    // Miss records の登録
    std::vector<MissRecord> missRecords;
    for(int i = 0; i < m_missPrograms.size(); i++)
    {
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_missPrograms[i], &rec));   // MissRecord.header に， m_missPrograms[i] を登録
        rec.data = nullptr;
        missRecords.push_back(rec);
    }
    m_missRecordsBuffer.allocAndUpload(missRecords);                        // GPU に情報を転送
    m_sbt.missRecordBase            = m_missRecordsBuffer.getDevicePointer();     // missRecords バッファのデバイス上の先頭アドレスを sbt に登録
    m_sbt.missRecordStrideInBytes   = sizeof(MissRecord);
    m_sbt.missRecordCount           = (int)missRecords.size();

    // Callables records の登録
    std::vector<CallableRecord> callableRecords;
    for(int i =0; i < m_callablePrograms.size(); i++){
        CallableRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(m_callablePrograms[i], &rec));   // CallableRecord.header に， m_callablePrograms[i] を登録
        rec.data = nullptr;
        callableRecords.push_back(rec);
    }
    m_callableRecordsBuffer.allocAndUpload(callableRecords);
    m_sbt.callablesRecordBase            = m_callableRecordsBuffer.getDevicePointer();
    m_sbt.callablesRecordStrideInBytes   = sizeof(CallableRecord);
    m_sbt.callablesRecordCount           = (int)callableRecords.size();

    // Hitgroup records の登録
    int numObjects = (int)m_model->meshes.size();
    std::vector<HitgroupRecord> hitgroupRecords;
    for(int meshID = 0; meshID < numObjects; meshID++)
    {
        auto mesh = m_model->meshes[meshID];
        for(int rayID = 0; rayID < RAY_TYPE_COUNT; rayID++)
        {
            HitgroupRecord rec;
            OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroupPrograms[rayID], &rec));   // HitgroupRecord.header に， m_hitgroupPrograms[rayID] を登録
            rec.data.vertex             = (float3*)m_vertexBuffer[meshID].getDevicePointer();
            rec.data.index              = (uint3*)m_indexBuffer[meshID].getDevicePointer();
            rec.data.normal             = (float3*)m_normalBuffer[meshID].getDevicePointer();
            rec.data.tangent            = (float4*)m_tangentBuffer[meshID].getDevicePointer();
            rec.data.diffuseTexcoord    = (float2*)m_diffuseTexcoordBuffer[meshID].getDevicePointer();
            rec.data.normalTexcoord     = (float2*)m_normalTexcoordBuffer[meshID].getDevicePointer();
            rec.data.emissiveTexcoord   = (float2*)m_emissiveTexcoordBuffer[meshID].getDevicePointer();
            rec.data.color              = (float3*)m_colorBuffer[meshID].getDevicePointer();
            // Material 情報の登録
            int materialID = mesh->materialID;
            if(materialID >= 0){
                auto material = m_model->materials[materialID];
                rec.data.roughness  = material->roughness;
                rec.data.metallic   = material->metallic;
                rec.data.emissive   = material->emissive;
                // Material Type
                if(material->isLight){
                    rec.data.materialType = MATERIAL_TYPE_LIGHT;
                } else if (material->isGlass){
                    rec.data.materialType = MATERIAL_TYPE_GLASS;
                } else if (material->rmTextureID >= 0){
                    rec.data.materialType = MATERIAL_TYPE_PRINCIPLED_BRDF;
                } else {
                    rec.data.materialType = MATERIAL_TYPE_DIFFUSE;
                }
                // Texture の登録
                if(material->diffuseTextureID >= 0){
                    rec.data.diffuseTexture.hasTexture  = true;
                    rec.data.diffuseTexture.texture     = m_textureObjects[material->diffuseTextureID];
                } 
                if(material->normalTextureID >= 0){
                    rec.data.normalTexture.hasTexture   = true;
                    rec.data.normalTexture.texture      = m_textureObjects[material->normalTextureID];
                }
                if(material->rmTextureID >= 0){
                    rec.data.rmTexture.hasTexture       = true;
                    rec.data.rmTexture.texture          = m_textureObjects[material->rmTextureID];
                }
                if(material->emissiveTextureID >= 0){
                    rec.data.emissiveTexture.hasTexture  = true;
                    rec.data.emissiveTexture.texture      = m_textureObjects[material->emissiveTextureID];
                }

            }
            hitgroupRecords.push_back(rec);
        }
    }
    
    m_hitgroupRecordsBuffer.allocAndUpload(hitgroupRecords);                        // GPU に情報を転送
    m_sbt.hitgroupRecordBase            = m_hitgroupRecordsBuffer.getDevicePointer();    // hitgroupRecords バッファのデバイス上の先頭アドレスを sbt に登録
    m_sbt.hitgroupRecordStrideInBytes   = sizeof(HitgroupRecord);                   // 
    m_sbt.hitgroupRecordCount           = (int)hitgroupRecords.size();
}

void Renderer::render()
{
    if(m_launchParams.frame.size.x == 0) return;

    if(! m_isAccumulate){
        m_launchParams.frame.frameID = 0;
    }
    m_launchParamsBuffer.upload(&m_launchParams, 1);
    m_launchParams.frame.frameID ++;

    OPTIX_CHECK(optixLaunch(
        m_pipeline, m_stream,
        m_launchParamsBuffer.getDevicePointer(),
        m_launchParamsBuffer.getSizeInBytes(),
        &m_sbt,
        m_launchParams.frame.size.x,
        m_launchParams.frame.size.y,
        1
    ));

    CUDA_SYNC_CHECK();

    computeFinalPixelColors();

}

void Renderer::setCamera(const Camera &camera)
{
    m_launchParams.frame.frameID = 0;
    
    // レンダラで使用するカメラパラメータを計算してセット

    // MEMO: fov を計算する
    m_lastSetCamera = camera;
    m_launchParams.camera.position  = camera.from;
    m_launchParams.camera.direction = normalize(camera.at - camera.from);
    m_launchParams.camera.horizontal    
        = normalize(cross(m_launchParams.camera.direction, camera.up));
    m_launchParams.camera.vertical      
        = normalize(cross(m_launchParams.camera.horizontal, m_launchParams.camera.direction));
    m_launchParams.camera.fValue        = camera.fValue;
    m_launchParams.camera.focalLength   = camera.focalLength;
    m_launchParams.camera.pintDist      = camera.pintDist;
    m_launchParams.camera.fov           = camera.fov;
    m_launchParams.camera.sensitivity   = camera.sensitivity;

}


void Renderer::downloadPixels(uint32_t m_h_pixels[])
{
    m_finalColorBuffer.download(m_h_pixels, m_launchParams.frame.size.x * m_launchParams.frame.size.y);
}

void Renderer::resize(const int2 &newSize)
{
    if ((newSize.x == 0) || (newSize.y == 0)) return;

    m_denoisedBuffer.resize(newSize.x * newSize.y * sizeof(float4));
    m_fbColor.resize(newSize.x * newSize.y * sizeof(float4));
    m_fbNormal.resize(newSize.x * newSize.y * sizeof(float4));
    m_fbAlbedo.resize(newSize.x * newSize.y * sizeof(float4));
    m_finalColorBuffer.resize(newSize.x * newSize.y * sizeof(uint32_t));

    m_launchParams.frame.size           = newSize;
    m_launchParams.frame.colorBuffer    = (float4*)m_fbColor.getDevicePointer(); 
    m_launchParams.frame.normalBuffer   = (float4*)m_fbNormal.getDevicePointer(); 
    m_launchParams.frame.albedoBuffer   = (float4*)m_fbAlbedo.getDevicePointer(); 

    setCamera(m_lastSetCamera);
}

void Renderer::setNumDevices(const int numDevices)
{
    m_numDevices = numDevices;
}

int Renderer::getNumDevices() const 
{
    return m_numDevices;
}

// 1 ファイルにつき 1 関数と考える
void Renderer::createCUDAModule()
{
    const int numCudaModule = m_cudaModuleFileNames.size();
    m_cudaFunction.resize(numCudaModule);

    // ptx ファイルの読み出し
    std::vector<char> ptxCode = readData(m_cudaModuleFileNames[static_cast<int>(PostProcessCudaModuleIdentifier::CUDA_MODULE_ID_TONEMAP)]);
    if (ptxCode.empty()) {
        throw std::runtime_error("PTX code is empty. Check file path: " + m_cudaModuleFileNames[static_cast<int>(PostProcessCudaModuleIdentifier::CUDA_MODULE_ID_TONEMAP)]);
    }

    /////////////////////////////

    std::filesystem::path exe_path;
    char path_buffer[MAX_PATH] = {};
#if defined(_WIN32)
    if(GetModuleFileNameA(NULL, path_buffer, MAX_PATH) == 0){
        std::cerr << "ERROR: GetModuleFileNameA failed" << std::endl;
    }
    exe_path = std::filesystem::path(path_buffer);
#else
    ssize_t count = readlink("/proc/self/exe", exe_path, PATH_MAX);
    if(count == -1) {
        std::cerr << "ERROR: readlink() failed" << std::endl;
    }
#endif
    std::filesystem::path ptx_dir = exe_path.parent_path().parent_path() / "ptxes" / m_cudaModuleFileNames[static_cast<int>(PostProcessCudaModuleIdentifier::CUDA_MODULE_ID_TONEMAP)];

    std::cout << "Loading CUDA module..." << std::endl;
    CUDA_DRIVER_CHECK(cuModuleLoad(&m_cudaModule, ptx_dir.string().c_str()));
    std::cout << "Loaded CUDA module." << std::endl;

    // 登録
    std::cout << "Fetching kernel function..." << std::endl;
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&m_cudaFunction[static_cast<int>(PostProcessCudaModuleIdentifier::CUDA_MODULE_ID_TONEMAP)], m_cudaModule, "computeFinalPixelColorsKernel"));
    std::cout << "Fetched kernel function." << std::endl;
}

void Renderer::computeFinalPixelColors()
{
    int2 fbSize = m_launchParams.frame.size;
    int2 blockSize = make_int2(32);
    int2 numBlocks = make_int2(std::ceil((float)fbSize.x / (float)blockSize.x), std::ceil((float)fbSize.y / (float)blockSize.y));
    
    CUdeviceptr finalColorBufferPtr = m_finalColorBuffer.getDevicePointer();
    CUdeviceptr renderTargetBufferPtr;
    switch(m_renderBufferType){
        case COLOR:
            renderTargetBufferPtr = m_fbColor.getDevicePointer();
            break;
        case NORMAL:
            renderTargetBufferPtr = m_fbNormal.getDevicePointer();
            break;
        case ALBEDO:
            renderTargetBufferPtr = m_fbAlbedo.getDevicePointer();
            break;
        default:
            renderTargetBufferPtr = m_fbColor.getDevicePointer();
            break;
    }

    void* arg0 = &finalColorBufferPtr;
    void* arg1 = &renderTargetBufferPtr;
    void* arg2 = &fbSize;
    void* args[] = {
        arg0, arg1, arg2
    };
    

    CUDA_DRIVER_CHECK(
        cuLaunchKernel(
            m_cudaFunction[static_cast<int>(PostProcessCudaModuleIdentifier::CUDA_MODULE_ID_TONEMAP)],
            numBlocks.x, numBlocks.y, 1,    // スレッドのブロック数
            blockSize.x, blockSize.y, 1,    // 各ブロック内のスレッド数
            0,
            0,
            args,
            nullptr 
        )
    );
    CUDA_SYNC_CHECK();

}

void Renderer::setCameraModel(const int cameraModel){
    m_launchParams.camera.cameraMode = cameraModel;
};

void Renderer::setRenderBufferType(const int renderBufferType){
    m_renderBufferType = renderBufferType;
};

int Renderer::getRenderBufferType() const{
    return m_renderBufferType;
};

void Renderer::setEnvMap(const std::string& envMapFileName)
{

    std::filesystem::path exePath;
    char pathBuffer[MAX_PATH] = {};
#if defined(_WIN32)
    if(GetModuleFileNameA(NULL, pathBuffer, MAX_PATH) == 0){
        std::cerr << "ERROR: GetModuleFileNameA failed" << std::endl;
    }
    exePath = std::filesystem::path(pathBuffer);
#else
    ssize_t count = readlink("/proc/self/exe", exePath, PATH_MAX);
    if(count == -1) {
        std::cerr << "ERROR: readlink() failed" << std::endl;
    }
#endif
    std::filesystem::path envMapDir = exePath.parent_path().parent_path().parent_path().parent_path() / "envMap" / envMapFileName;

    int2 res;
    const char* err = NULL;
    float* image;
    int compornents_per_pixel = 3;
    image = stbi_loadf(envMapDir.string().c_str(), &res.x, &res.y, &compornents_per_pixel, compornents_per_pixel);

    // int ret = LoadEXR(&image, &res.x, &res.y, envMapFileName.c_str(), &err);
    if(image){
        int bytes_per_scanline = compornents_per_pixel * res.x;
        float4*  h_tex_env = (float4*)malloc(res.x * res.y  * sizeof(float4));
        for (int i = 0; i < res.y; i++) {
            for (int j = 0; j < res.x; j++) {
                auto pixel = image + i * bytes_per_scanline + j * compornents_per_pixel;
                float4 tmp;
        
                tmp.x = pixel[0];
                tmp.y = pixel[1];
                tmp.z = pixel[2];
                tmp.w = 1.f;
                h_tex_env[i * res.x + j] = tmp;

            }
        }

        cudaResourceDesc res_desc = {};

        cudaChannelFormatDesc channel_desc;
        int32_t width  = (int32_t)res.x;
        int32_t height = (int32_t)res.y;
        int32_t pitch = width * sizeof(float4);
        channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

        cudaArray_t pixelArrayEnv;
        CUDA_CHECK(cudaMallocArray(&pixelArrayEnv, &channel_desc, width, height));
        CUDA_CHECK(cudaMemcpy2DToArray(pixelArrayEnv, 0, 0, h_tex_env, pitch, pitch, height, cudaMemcpyHostToDevice));


        res_desc.resType    = cudaResourceTypeArray;
        res_desc.res.array.array = pixelArrayEnv;
        // textureArrays.push_back(pixelArrayEnv);

        cudaTextureDesc tex_desc = {};
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode     = cudaFilterModeLinear;
        tex_desc.readMode       = cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;
        tex_desc.maxAnisotropy  = 1;
        tex_desc.maxMipmapLevelClamp = 99;
        tex_desc.minMipmapLevelClamp = 0;
        tex_desc.mipmapFilterMode    = cudaFilterModePoint;
        tex_desc.borderColor[0]      = 1.0f;
        tex_desc.sRGB                = 0;
    
        // Create texture object
        cudaTextureObject_t cudaTex = 0;
        CUDA_CHECK(cudaCreateTextureObject(&cudaTex, &res_desc, &tex_desc, nullptr));
        m_envMapArray = pixelArrayEnv;
        m_envMapTex = cudaTex;
        m_launchParams.envMap = cudaTex;
        stbi_image_free(image);
        free(h_tex_env);

    } else {
        std::cerr << envMapDir.string() << std::endl;
        std::cerr << "ERROR: can't find hdr file" << std::endl;
    }
}

void Renderer::createLightTable()
{
    // 三角形の面光源を登録
    const int numMeshes = (int)m_model->meshes.size();

    for(int meshID = 0; meshID < numMeshes; meshID++){
        const TriangleMesh &mesh = *m_model->meshes[meshID];
        const int materialID = mesh.materialID;
        const Material& material = *m_model->materials[materialID];

        // //環境マップを光源として登録 (1枚だけ) 
        LightDefinition lightDefinition;
        lightDefinition.lightType = LIGHT_TYPE_ENV_SPHERE;
        lightDefinition.lightIndexInType = 0;

        m_lightDefinitionTable.push_back(lightDefinition);


        if(material.isLight ==  true){
            const float3 constantEmission = material.emissive;
            const bool hasEmissiveTexture = (material.emissiveTextureID >= 0);
            for(int triangleID = 0; triangleID < mesh.index.size(); triangleID++){
                
                const uint3 index = mesh.index[triangleID]; 
                
                TriangleLightData triangleLightData;
                const float3 v0 = mesh.vertex[index.x];
                const float3 v1 = mesh.vertex[index.y];
                const float3 v2 = mesh.vertex[index.z];
                triangleLightData.v0 = v0;
                triangleLightData.v1 = v1;
                triangleLightData.v2 = v2;
                triangleLightData.normal = normalize(cross(v1 - v0, v2 - v0));
                triangleLightData.area = 0.5f * length(cross(v1 - v0, v2 - v0));
                triangleLightData.constantEmission = constantEmission;
                triangleLightData.uv0 = mesh.emissiveTexcoord[index.x];
                triangleLightData.uv1 = mesh.emissiveTexcoord[index.y];
                triangleLightData.uv2 = mesh.emissiveTexcoord[index.z];
                triangleLightData.emissiveTexture.hasTexture = hasEmissiveTexture;
                triangleLightData.emissiveTexture.texture = m_textureObjects[material.emissiveTextureID];

                LightDefinition lightDefinition;
                lightDefinition.lightType = LIGHT_TYPE_TRIANGLE;
                lightDefinition.lightIndexInType = m_triangleLightDataTable.size();

                m_triangleLightDataTable.push_back(triangleLightData);
                m_lightDefinitionTable.push_back(lightDefinition);
            }
            
        }

    }

    // バッファのアップロード
    m_lightDefinitionBuffer.allocAndUpload(m_lightDefinitionTable);
    m_triangleLightDataBuffer.allocAndUpload(m_triangleLightDataTable);

    // コンスタントメモリに登録し，GPU から読み込めるようにする
    m_launchParams.light.lightDefinition = (LightDefinition*)m_lightDefinitionBuffer.getDevicePointer();
    m_launchParams.light.triangleLightData = (TriangleLightData*)m_triangleLightDataBuffer.getDevicePointer();
    m_launchParams.light.numLights = m_lightDefinitionTable.size();

}

const LaunchParams Renderer::getLaunchParams() const
{
    return m_launchParams;
};

const CUDABuffer&    Renderer::getFinalColorBuffer() const
{
    return m_finalColorBuffer;
};
