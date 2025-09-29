#include "config.cuh"

#include <optix.h>

#include "per_ray_data.cuh"
#include "shader_common.cuh"
#include "random_number_generator.cuh"
#include "../include/launch_params.h"
#include <stdio.h>

extern "C" __global__ void __closesthit__radiance()
{

    const TriangleMeshSBTData &sbtData =*(const TriangleMeshSBTData*) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    // 基本的な交差点の情報を取得
    const int primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // レイの進行方向を取得
    const float3 rayDirection = normalize(optixGetWorldRayDirection());
    // レイとプリミティブの交差点を取得
    const float rayLength = optixGetRayTmax();
    const float3 intersectedPoint = optixGetWorldRayOrigin() + rayLength * rayDirection;

    // 形状処理用. 面法線を取得 
    const float3 &V1 = sbtData.vertex[index.x];
    const float3 &V2 = sbtData.vertex[index.y];
    const float3 &V3 = sbtData.vertex[index.z];
    float3 Ng = normalize(cross(V2-V1, V3-V1));

    // Diffuse テクスチャ座標を取得
    const float2 &UVDiffuse1 = sbtData.diffuseTexcoord[index.x];
    const float2 &UVDiffuse2 = sbtData.diffuseTexcoord[index.y];
    const float2 &UVDiffuse3 = sbtData.diffuseTexcoord[index.z];

    const float2 diffuseTextureCoordinate = (1.0f - u - v) * UVDiffuse1
        + u * UVDiffuse2
        + v * UVDiffuse3;

    // Normal テクスチャ座標を取得
    const float2 &UVNormal1 = sbtData.normalTexcoord[index.x];
    const float2 &UVNormal2 = sbtData.normalTexcoord[index.y];
    const float2 &UVNormal3 = sbtData.normalTexcoord[index.z];
    
    const float2 normalTextureCoordinate = (1.0f - u - v) * UVNormal1
        + u * UVNormal2
        + v * UVNormal3;

    // Emissive テクスチャ座標を取得
    const float2 &UVEmissive1 = sbtData.emissiveTexcoord[index.x];
    const float2 &UVEmissive2 = sbtData.emissiveTexcoord[index.y];
    const float2 &UVEmissive3 = sbtData.emissiveTexcoord[index.z];
    const float2 emissiveTextureCoordinate = (1.0f - u - v) * UVEmissive1
        + u * UVEmissive2
        + v * UVEmissive3;

    // シェーディング用．頂点法線があれば頂点法線を，なければ面法線を使用
    float3 Ns = (sbtData.normal) 
        ?   (1.0f - u - v) * sbtData.normal[index.x]
            +            u * sbtData.normal[index.y]
            +            v * sbtData.normal[index.z]
        :   Ng;
    Ns = normalize(Ns);
    float3 nml = make_float3(0.0f);
    if(sbtData.normalTexture.hasTexture) {
        float4 fromTexture = tex2D<float4>(sbtData.normalTexture.texture, normalTextureCoordinate.x, 1.0f - normalTextureCoordinate.y);
        nml = (make_float3(fromTexture) - make_float3(0.5f)) * 2.0f ;
        nml.y = -nml.y;
        nml.z = sqrtf(fmaxf(1.0f - nml.y * nml.y - nml.x * nml.x, 0.0001f));
    }

    // 接ベクトル空間の計算
    const float4 &T1 = sbtData.tangent[index.x];
    const float4 &T2 = sbtData.tangent[index.y];
    const float4 &T3 = sbtData.tangent[index.z];

    const float4 tan =  (1.0f - u - v) * T1
            +            u * T2
            +            v * T3;
    const float3 T = normalize(make_float3(tan));
    const float3 N = normalize(Ns);
    const float3 B = normalize(cross(N, T) * tan.w);
    
    Ns = normalize(T * nml.x + B * nml.y + N * nml.z);
    const float3 binormal = normalize(cross(T, Ns)); 
    const float3 tangent = normalize(cross(Ns, binormal)); 

    // 法線の向きを調整
    if(dot(rayDirection, Ng) > 0.f) {
        Ng = -Ng;
    }
    Ng = normalize(Ng);

    // if(dot(Ng, Ns) < 0.f){
    //     Ns -= 2.f * dot(Ng, Ns) * Ng;  
    // }

    // Albedo の計算
    float3 albedo = make_float3(1.0f);
    if(sbtData.diffuseTexture.hasTexture){
        float4 fromTexture = tex2D<float4>(
            sbtData.diffuseTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y);
        albedo = make_float3(fromTexture);
    }

    float4 arm = make_float4(0.0f);
    
    if(sbtData.rmTexture.hasTexture){
        arm = tex2D<float4>(sbtData.rmTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y);
    }

    IntersectedData matData;
    matData.ior = 1.5f; // glass
    matData.metallic = arm.z;
    matData.roughness = arm.y;
    matData.baseColor = albedo;

    prd.position = intersectedPoint; // 後で置き場を考える
    // メッシュの光源と交差した場合の処理
    if(sbtData.materialType == MATERIAL_TYPE_LIGHT)
    {
        const float cosTheta = dot(-1.0f * rayDirection, Ns);
        if(cosTheta > 1e-4f)
        {
            float3 emission = sbtData.emissive;
            if(sbtData.emissiveTexture.hasTexture){
               float4 fromTexture = tex2D<float4>(
                sbtData.emissiveTexture.texture, emissiveTextureCoordinate.x, 1.0f - emissiveTextureCoordinate.y);
                emission = make_float3(fromTexture);
            }
            if(prd.bounce != 0)
            {
                const float areaThisTriangle = 0.5f * fmaxf(length(cross(V2 - V1, V3 - V1)), 1e-5f);
                float pdfLight = 1.0f / (areaThisTriangle * float(optixLaunchParams.light.numLights));
                const float geometricTerm = cosTheta / (rayLength * rayLength);
                pdfLight /=  geometricTerm;
                
                const float weight = balanceHeuristicWeight(1, fmaxf(prd.pdf.bxdf, 1e-7f), 1, fmaxf(pdfLight, 1e-7f));
                emission *= weight;

                
            }
            emission *= optixLaunchParams.light.lightIntensityFactor;
            prd.contribution += emission * prd.albedo;
        }

        if(prd.bounce == 0){
            prd.primaryNormal = Ns;
            prd.primaryAlbedo = albedo;
        }
        prd.continueTrace = false;
        return;
    }


    const float3 woLocal = worldToLocal(-1.0f * rayDirection, tangent, Ns, binormal);
    // ガラスのメッシュと交差した場合の処理（レイトレーシングを続行）
    if(sbtData.materialType == MATERIAL_TYPE_GLASS)
    {
        int callableFunctionOffset = NUM_LENS_TYPE + MATERIAL_TYPE_GLASS * 2; // 登録した順番が lens の後で，glass * 2 が 対応するマテリアルの sample 関数
        prd.position = intersectedPoint;
        float3 wiLocal = optixDirectCall<float3, const float3, const IntersectedData& , PRD*>(callableFunctionOffset, woLocal, matData, &prd);
        prd.wi = localToWorld(wiLocal, tangent, Ns, binormal);
        prd.albedo *= albedo;
        prd.continueTrace = true;
        prd.pdf.bxdf = 1.0f;
        return;
    }


    const int numLights = optixLaunchParams.light.numLights; // 後で置き場を考える
    // MIS によるシェーディング
    // ライトサンプリング
    if (numLights > 0){

        //RIS
        const int indexLight = (numLights > 1) ? clamp(static_cast<int>(floor(prd.random() * (float)numLights)), 0, numLights - 1) : 0;
        LightDefinition light = optixLaunchParams.light.lightDefinition[indexLight];

        const int callLight = NUM_LENS_TYPE + NUM_BXDF + light.lightType;
        LightSample lightSample = optixDirectCall<LightSample, LightDefinition, PRD*>(callLight, light, &prd);
        const float3 wiLocal = worldToLocal(lightSample.direction, tangent, Ns, binormal);

        if(lightSample.pdf > 0.0f && wiLocal.y > 1e-5f)
        {
            ShadowPRD shadowPrd;
            shadowPrd.visible = false;
            uint32_t u0, u1;
            packPointer(&shadowPrd, u0, u1);
            

            // BSDF の計算
            int callableFunctionOffset = NUM_LENS_TYPE + sbtData.materialType * 2 + 1; // 登録した順番が lens の後で，materialType * 2 + 1 が 対応するマテリアルの eval 関数 
            float3 bxdfValue = optixDirectCall<float3, const float3, const float3, const IntersectedData& , PRD*>(callableFunctionOffset, wiLocal, woLocal, matData, &prd);
            float bxdfPdf = prd.pdf.bxdf;
            const float3 wi = localToWorld(wiLocal, tangent, Ns, binormal); 


            // 光源へ接続して可視性を判断
            optixTrace( optixLaunchParams.traversable,
                        intersectedPoint + 1e-3f * Ng, // 出射位置
                        wi,
                        1e-3f,
                        lightSample.distance - 1e-3f,
                        0.0f,
                        OptixVisibilityMask( 255 ),
                        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                            | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                        SHADOW_RAY_TYPE,    // SBT offset
                        RAY_TYPE_COUNT,     // SBT stride
                        SHADOW_RAY_TYPE,    // miss SBT Index
                        u0, u1
            );

            if(shadowPrd.visible){
                float weight = balanceHeuristicWeight(1, fmaxf(lightSample.pdf, 1e-7f), 1, fmaxf(bxdfPdf, 1e-7f));
                prd.contribution += prd.albedo * lightSample.emission * bxdfValue * wiLocal.y * weight / fmaxf(lightSample.pdf, 1e-7f);
            }
        }
        prd.continueTrace = true;
    }

    
    // 次の方向の決定
    // BSDF のサンプリング とアルベドの変更
    int callableFunctionOffset = NUM_LENS_TYPE + sbtData.materialType * 2;
    const float3 sampledWiLocal = optixDirectCall<float3, const float3, const IntersectedData&, PRD*>(callableFunctionOffset, woLocal, matData, &prd);
    callableFunctionOffset = NUM_LENS_TYPE + sbtData.materialType * 2 + 1;
    const float3 bxdfValue = optixDirectCall<float3, const float3, const float3, const IntersectedData& , PRD*>(callableFunctionOffset, sampledWiLocal, woLocal, matData, &prd);
    const float3 sampledWi = localToWorld(sampledWiLocal, tangent, Ns, binormal);

    if(sampledWiLocal.y < 1e-7f){
        prd.continueTrace = false;
        return;
    }
    
    prd.position += 1e-3f * Ng;
    prd.albedo *= bxdfValue * sampledWiLocal.y / fmaxf(prd.pdf.bxdf, 1e-7f);
    prd.wi = sampledWi; 
    prd.continueTrace = true;
    
    if(prd.bounce == 0){
        prd.primaryNormal = Ns;
        prd.primaryAlbedo = albedo;
    }
}

extern "C" __global__ void __closesthit__shadow()
{

}



extern "C" __global__ void __anyhit__radiance()
{
    const TriangleMeshSBTData &sbtData =*(const TriangleMeshSBTData*) optixGetSbtDataPointer();
    PRD &prd = *getPRD<PRD>();

    // 基本的な交差点の情報を取得
    const int primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // Diffuse テクスチャ座標を取得
    const float2 &UVDiffuse1 = sbtData.diffuseTexcoord[index.x];
    const float2 &UVDiffuse2 = sbtData.diffuseTexcoord[index.y];
    const float2 &UVDiffuse3 = sbtData.diffuseTexcoord[index.z];

    const float2 diffuseTextureCoordinate = (1.0f - u - v) * UVDiffuse1
        + u * UVDiffuse2
        + v * UVDiffuse3;

    const float opacity = tex2D<float4>(sbtData.diffuseTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y).w;

    if(opacity < 1.0f && opacity <= prd.random()){
        optixIgnoreIntersection();
        return;
    }
}

extern "C" __global__ void __anyhit__shadow()
{
    const TriangleMeshSBTData &sbtData =*(const TriangleMeshSBTData*) optixGetSbtDataPointer();
    if(sbtData.materialType == MATERIAL_TYPE_GLASS){
        optixIgnoreIntersection();
        return;
    }
    PRD &prd = *getPRD<PRD>();

    // 基本的な交差点の情報を取得
    const int primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // Diffuse テクスチャ座標を取得
    const float2 &UVDiffuse1 = sbtData.diffuseTexcoord[index.x];
    const float2 &UVDiffuse2 = sbtData.diffuseTexcoord[index.y];
    const float2 &UVDiffuse3 = sbtData.diffuseTexcoord[index.z];

    const float2 diffuseTextureCoordinate = (1.0f - u - v) * UVDiffuse1
        + u * UVDiffuse2
        + v * UVDiffuse3;

    const float opacity = tex2D<float4>(sbtData.diffuseTexture.texture, diffuseTextureCoordinate.x, 1.0f - diffuseTextureCoordinate.y).w;

    if(opacity < 1.0f && opacity <= prd.random()){
        optixIgnoreIntersection();
        return;
    }
}

// radiance ray が交差しない = 環境マップにあたる
extern "C" __global__ void __miss__radiance()
{
    PRD &prd = *getPRD<PRD>(); 
    float u, v;
    orthogonalToSphericalCoord(prd.wi, &u, &v);
    float3 emission =  make_float3(tex2D<float4>(optixLaunchParams.envMap, u, v));
    
    const float pdfLight = 1.0f / (2.0f * M_PI);
    if(prd.bounce != 0){
        const float weight = balanceHeuristicWeight(1, fmaxf(prd.pdf.bxdf, 1e-7f), 1, fmaxf(pdfLight, 1e-7f));
        emission *= weight; 
    }
    prd.contribution += emission * prd.albedo / fmaxf(pdfLight, 1e-7f);
    prd.continueTrace = false;
}

// shadow ray が交差しない = 可視
extern "C" __global__ void __miss__shadow()
{
    ShadowPRD &prd = *getPRD<ShadowPRD>(); 
    // 何にもヒットしないので visible
    prd.visible = true; 
}

extern "C" __global__ void __raygen__renderFrame()
{
    // // ピクセル位置の取得
    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;

    const int accumID = optixLaunchParams.frame.accumID;
    const int frameID = optixLaunchParams.frame.frameID;
    const auto &camera = optixLaunchParams.camera;

    const uint32_t fbIndex = ix + iy * optixLaunchParams.frame.size.x;


    // PRD: per-ray-data
    PRD prd;
    // 乱数をピクセル位置とフレーム数をシード値として初期化
    prd.random.init(fbIndex, optixLaunchParams.frame.frameID);
    
    uint32_t u0, u1;
    packPointer( &prd, u0, u1);

    float3 pixelColor = make_float3(0.f); 
    float3 pixelNormal = make_float3(0.f);
    float3 pixelAlbedo = make_float3(0.f);

    for(int sampleID = 0; sampleID < optixLaunchParams.frame.numPixelSamples; sampleID ++){

        // prd.emission = make_float3(0.f);
        prd.contribution = make_float3(0.f);
        prd.albedo   = make_float3(1.f);
        prd.continueTrace = true;
        prd.bounce = 0;
        prd.pdf.bxdf = 1.0f;
        prd.pdf.light = 1.0f;

        // スクリーン空間上のサンプル点をサブピクセル精度でサンプリング
        const float2 screen = make_float2(
            ((float)(optixLaunchParams.frame.size.x - ix - 1) + prd.random()) / (float)optixLaunchParams.frame.size.x, 
            ((float)(optixLaunchParams.frame.size.y - iy - 1) + prd.random()) / (float)optixLaunchParams.frame.size.y 
        );


        // Attention! : レンズ系の direct callable 関数は，プログラム全体でオフセット：0
        const LensRay ray = 
            optixDirectCall<LensRay, const float2, const float, const float>(optixLaunchParams.camera.cameraMode, screen, prd.random(), prd.random());

        prd.position = ray.org;
        prd.wi = ray.dir;

        

        while(prd.bounce < optixLaunchParams.frame.maxBounce)
        {
            optixTrace( optixLaunchParams.traversable,
                prd.position,
                prd.wi,
                0.f,    // tmin
                1e20f,  // tmax
                0.0f,   // rayTime
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_NONE,
                0,
                RAY_TYPE_COUNT,
                RADIANCE_RAY_TYPE,
                u0, u1);

            if(!prd.continueTrace){
                break;
            }
            prd.bounce ++;
        }

        pixelColor = pixelColor + (prd.contribution * ray.weight - pixelColor) / (sampleID + 1.0f);
        pixelNormal = pixelNormal + (prd.primaryNormal - pixelNormal) / (sampleID + 1.0f);
        pixelAlbedo = pixelAlbedo + (prd.primaryAlbedo - pixelAlbedo) / (sampleID + 1.0f);

    }
    pixelColor = clamp(pixelColor, 0.0f, 1000.0f);
    if(!isfinite(pixelColor.x) || !isfinite(pixelColor.y) || !isfinite(pixelColor.z)){
        pixelColor = make_float3(0.0f);
    }

    float4 rgba = make_float4(pixelColor, 1.0f);

    
    if (optixLaunchParams.frame.frameID > 0){
        float4 rgba_now = optixLaunchParams.frame.colorBuffer[fbIndex];
        rgba = rgba_now + (rgba - rgba_now) / (optixLaunchParams.frame.frameID + 1.0f); 
    }
    
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
    optixLaunchParams.frame.albedoBuffer[fbIndex] = make_float4(pixelAlbedo, 1.0f);
    optixLaunchParams.frame.normalBuffer[fbIndex] = make_float4(pixelNormal, 1.0f);
}
