#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include "../device/trace_volume.cuh"
#include <stdio.h>


static __forceinline__ __device__
float evalPhaseFunction(){
    return 1.f / (4.0f * M_PI);
}
extern "C" __global__ void __closesthit__vdb_radiance_rgb()
{
    PRDRGB & prd = *getPRD<PRDRGB>();
    const HitgroupSBTData &sbtData =*(const HitgroupSBTData*) optixGetSbtDataPointer();

    const uint32_t vdbIndex         = sbtData.vdb.vdbIndex;
    // const uint32_t materialIndex    = sbtData.vdb.materialIndex;

    const float tEnter  = optixGetRayTmax();
    const float tExit   = __uint_as_float(optixGetAttribute_0());
    if(!(tExit > tEnter)){
        prd.continueTrace = false;
        return;
    }
    // ray の情報
    const float3 rayDirectionWorld  = normalize(optixGetWorldRayDirection());
    const float3 rayOriginWorld     = optixGetWorldRayOrigin();

    const float3 rayDirectionObject = normalize(optixGetObjectRayDirection());
    const float3 rayOriginObject    = optixGetObjectRayOrigin();

    prd.instanceID = optixGetInstanceId();

    // AABB
    const int primitiveIndex = optixGetPrimitiveIndex();
    const OptixAabb aabb = optixLaunchParams.vdbAABBs[primitiveIndex];
    const float3 bMin = make_float3(aabb.minX, aabb.minY, aabb.minZ);
    const float3 bMax = make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);

    // --------------------------
    // 媒質のパラメータ
    // --------------------------
    const float sigmaTScale = optixLaunchParams.vdbs[primitiveIndex].densityScale;

    // デルタトラッキング
    float tScatter;
    const bool isScatter = deltaTrack_localMajorant<PRDRGB>(
        prd, vdbIndex,
        rayOriginObject, rayDirectionObject,
        tEnter, tExit,
        sigmaTScale,
        tScatter
    );

    if(!isScatter)
    {
        // 散乱しなかった場合，radiance ray をもう一度飛ばしてメッシュと交差させる
        const float3 newOrigin = rayOriginWorld + tExit * rayDirectionWorld;

        uint32_t u0, u1;
        packPointer(&prd, u0, u1);

        optixTrace(
            optixLaunchParams.traversable,
            newOrigin,
            rayDirectionWorld,
            1e-4f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            RADIANCE_RAY_TYPE,
            RAY_TYPE_COUNT,
            RADIANCE_RAY_TYPE,
            u0, u1
        );

        return;
    }

    // 散乱する位置
    const float3 scatteredPointWorld = rayOriginWorld + tScatter * rayDirectionWorld;
    const float3 scatteredPointObject = rayOriginObject + tScatter * rayDirectionObject;

    prd.position = scatteredPointWorld;

    // 本当は，ここで散乱あるべとを反映させる
    // prd.albedo *= mediumAlbedo;

    // --------------------------
    // NEE
    // --------------------------
    const int numLights = optixLaunchParams.light.numLights;
    if(numLights > 0){
        const int indexLight = (numLights > 1) ? clamp(static_cast<int>(floor(prd.random() * (float)numLights)), 0, numLights - 1) : 0;
        LightDefinition light = optixLaunchParams.light.lightDefinition[indexLight];

        const int callLightType = NUM_LENS_TYPE + NUM_BXDF + light.lightType;
        LightSample_RGB lightSample = optixDirectCall<LightSample_RGB, LightDefinition, PRDRGB*>(callLightType, light, &prd);

        if(lightSample.pdf > 0.f)
        {
            const float3 wiWorld = normalize(lightSample.direction);
            const float distance = lightSample.distance;

            // ratio tracking
            const float3 wiObject   = optixTransformVectorFromWorldToObjectSpace(wiWorld);
            const float transmittance = ratioTrack_localMajorant<PRDRGB>(
                prd, vdbIndex,
                rayOriginObject, rayDirectionObject,
                tEnter, tExit,
                sigmaTScale
            );

            const float3 xExitW = rayOriginWorld + tExit * rayDirectionWorld;
            const float3 newOrigin = xExitW + rayDirectionWorld * 1e-3f;

            ShadowPRD shadowPrd;
            uint32_t u0, u1;
            packPointer(&shadowPrd, u0, u1);
            
            // 光源へ接続して可視性を判断
            optixTrace( 
                optixLaunchParams.traversable,
                newOrigin, // 出射位置
                wiWorld,
                1e-3f,
                distance - 1e-3f,
                0.0f,
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                    | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                SHADOW_RAY_TYPE,    // SBT offset
                RAY_TYPE_COUNT,     // SBT stride
                SHADOW_RAY_TYPE,    // miss SBT Index
                u0, u1
            );

            if(shadowPrd.visible && transmittance > 0.0f){
                const float phasePdf    = evalPhaseFunction();
                const float phaseValue  = phasePdf;

                float weight = balanceHeuristicWeight(1, fmaxf(lightSample.pdf, 1e-7f), 1, fmaxf(phasePdf, 1e-7f));
                prd.contribution += prd.albedo * (transmittance * lightSample.emission) * phaseValue * weight / fmaxf(lightSample.pdf, 1e-7f);
            }
        }
        prd.continueTrace = true;
    }

    // --------------------------
    // 次の方向の決定（位相関数のサンプリング）
    // --------------------------
    const float3 newDirection = random_unit_sphere(prd.random(), prd.random());

    prd.wi = newDirection;
    prd.position += 1e-3f * newDirection;
    prd.pdf.bxdf = evalPhaseFunction();
    prd.continueTrace = true;

    if(prd.bounce == 0){
        prd.primaryNormal = make_float3(0.0f); // 媒質なので 0 埋め等
        prd.primaryAlbedo = make_float3(0.8f);
    }
    
}


extern "C" __global__ void __closesthit__vdb_radiance_spectral()
{
    PRDSpectral & prd = *getPRD<PRDSpectral>();
    const HitgroupSBTData &sbtData =*(const HitgroupSBTData*) optixGetSbtDataPointer();

    const uint32_t vdbIndex         = sbtData.vdb.vdbIndex;
    // const uint32_t materialIndex    = sbtData.vdb.materialIndex;

    const float tEnter  = optixGetRayTmax();
    const float tExit   = __uint_as_float(optixGetAttribute_0());
    if(!(tExit > tEnter)){
        prd.continueTrace = false;
        return;
    }
    // ray の情報
    const float3 rayDirectionWorld  = normalize(optixGetWorldRayDirection());
    const float3 rayOriginWorld     = optixGetWorldRayOrigin();

    const float3 rayDirectionObject = normalize(optixGetObjectRayDirection());
    const float3 rayOriginObject    = optixGetObjectRayOrigin();

    prd.instanceID = optixGetInstanceId();

    // AABB
    const int primitiveIndex = optixGetPrimitiveIndex();
    const OptixAabb aabb = optixLaunchParams.vdbAABBs[primitiveIndex];
    const float3 bMin = make_float3(aabb.minX, aabb.minY, aabb.minZ);
    const float3 bMax = make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);

    // --------------------------
    // 媒質のパラメータ
    // --------------------------
    const float sigmaTScale = optixLaunchParams.vdbs[primitiveIndex].densityScale;

    // デルタトラッキング
    float tScatter;
    const bool isScatter = deltaTrack_localMajorant<PRDSpectral>(
        prd, vdbIndex, rayOriginObject, rayDirectionObject,
        tEnter, tExit,
        sigmaTScale, tScatter
    );

    if(!isScatter)
    {
        // 散乱しなかった場合，radiance ray をもう一度飛ばしてメッシュと交差させる
        const float3 newOrigin = rayOriginWorld + tExit * rayDirectionWorld;

        uint32_t u0, u1;
        packPointer(&prd, u0, u1);

        optixTrace(
            optixLaunchParams.traversable,
            newOrigin,
            rayDirectionWorld,
            1e-4f,    // tmin
            1e20f,  // tmax
            0.0f,   // rayTime
            OptixVisibilityMask( 255 ),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            RADIANCE_RAY_TYPE,
            RAY_TYPE_COUNT,
            RADIANCE_RAY_TYPE,
            u0, u1
        );

        return;
    }

    // 散乱する位置
    const float3 scatteredPoint = rayOriginWorld + tScatter * rayDirectionWorld;

    prd.position = scatteredPoint;

    // 本当は，ここで散乱あるべとを反映させる
    // prd.albedo *= mediumAlbedo;

    // --------------------------
    // NEE
    // --------------------------
    const int numLights = optixLaunchParams.light.numLights;
    if(numLights > 0){
        const int indexLight = (numLights > 1) ? clamp(static_cast<int>(floor(prd.random() * (float)numLights)), 0, numLights - 1) : 0;
        LightDefinition light = optixLaunchParams.light.lightDefinition[indexLight];

        const int callLightType = NUM_LENS_TYPE + NUM_BXDF + light.lightType;
        LightSample_Spectral lightSample = optixDirectCall<LightSample_Spectral, LightDefinition, PRDSpectral*>(callLightType, light, &prd);

        if(lightSample.pdf > 0.f)
        {
            const float3 wiWorld = normalize(lightSample.direction);
            const float distance = lightSample.distance;

            // ratio tracking
            const float3 wiObject   = optixTransformVectorFromWorldToObjectSpace(wiWorld);
            const float transmittance = ratioTrack_localMajorant<PRDSpectral>(
                prd, vdbIndex, rayOriginObject, rayDirectionObject,
                tEnter, tExit, sigmaTScale
            );

            const float3 xExitW = rayOriginWorld + tExit * rayDirectionWorld;
            const float3 newOrigin = xExitW + rayDirectionWorld * 1e-3f;


            ShadowPRD shadowPrd;
            uint32_t u0, u1;
            packPointer(&shadowPrd, u0, u1);
            
            // 光源へ接続して可視性を判断
            optixTrace( 
                optixLaunchParams.traversable,
                newOrigin, // 出射位置
                wiWorld,
                1e-3f,
                distance - 1e-3f,
                0.0f,
                OptixVisibilityMask( 255 ),
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                    | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
                SHADOW_RAY_TYPE,    // SBT offset
                RAY_TYPE_COUNT,     // SBT stride
                SHADOW_RAY_TYPE,    // miss SBT Index
                u0, u1
            );

            if(shadowPrd.visible && transmittance > 0.0f){
                const float phasePdf    = evalPhaseFunction();
                const float phaseValue  = phasePdf;

                float weight = balanceHeuristicWeight(1, fmaxf(lightSample.pdf, 1e-7f), 1, fmaxf(phasePdf, 1e-7f));
                prd.contribution += prd.albedo * (transmittance * lightSample.emission) * phaseValue * weight / fmaxf(lightSample.pdf, 1e-7f);
            }
        }
        prd.continueTrace = true;
    }

    // --------------------------
    // 次の方向の決定（位相関数のサンプリング）
    // --------------------------
    const float3 newDirection = random_unit_sphere(prd.random(), prd.random());

    prd.wi = newDirection;
    prd.position += 1e-3f * newDirection;
    prd.pdf.bxdf = evalPhaseFunction();
    prd.continueTrace = true;

    if(prd.bounce == 0){
        prd.primaryNormal = make_float3(0.0f); // 媒質なので 0 埋め等
        prd.primaryAlbedo = make_float3(0.8f);
    }
}