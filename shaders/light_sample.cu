#include "config.cuh"

#include <optix.h>

#include "per_ray_data.cuh"
#include "shader_common.cuh"
#include "random_number_generator.cuh"
#include "../include/launch_params.h"

extern "C" __device__ LightSample __direct_callable__light_env_sphere_constant(LightDefinition light, PRD* prd)
{
    LightSample lightSample;
    lightSample.pdf = 1.0f /  (2.0f * M_PI) / float(optixLaunchParams.light.numLights);
    lightSample.distance = 1e7f;
    
    const float3 direction = random_unit_sphere(prd->random(), prd->random()); 
    lightSample.direction = direction; 
    float u, v;
    orthogonalToSphericalCoord(direction, &u, &v);
    lightSample.emission = make_float3(tex2D<float4>(optixLaunchParams.envMap, u, v));
    return lightSample;
}


// extern "C" __device__ LightSample __direct_callable__light_env_sphere_is(LightDefinition light, PRD* prd)
// {
//     LightSample lightSample;
//     lightSample.pdf = 1.0f / M_2_PI;
    
//     const float3 direction = random_unit_sphere(prd->random(), prd->random()); 
//     lightSample.direction = direction; 
//     float u, v;
//     orthogonalToSphericalCoord(direction, &u, &v);
//     lightSample.emission = make_float3(tex2D<float4>(optixLaunchParams.envMap, u, v)) * float(optixLaunchParams.light.numLights);
//     return lightSample;
// }


extern "C" __device__ LightSample __direct_callable__light_triangle(LightDefinition light, PRD* prd)
{
    LightSample lightSample; // emission = (0, 0, 0) , pdf = 0 に初期化される 

    TriangleLightData& triangleLightData = optixLaunchParams.light.triangleLightData[light.lightIndexInType];
    
    // 三角形上の一様サンプリング
    const float3 v0 = triangleLightData.v0;
    const float3 v1 = triangleLightData.v1;
    const float3 v2 = triangleLightData.v2;

    const float3 a = v1 - v0;
    const float3 b = v2 - v0;

    const float ta = clamp(1.0f - sqrtf(prd->random()), 0.0f, 1.0f);
    const float tb = clamp((1.0f - ta) * prd->random(), 0.0f, 1.0f);
    const float3 sampledPosition = v0 + ta * a + tb * b;

    float3 emission = triangleLightData.constantEmission;

    float distance = length(sampledPosition - prd->position);
    float3 lightDirection = normalize(sampledPosition - prd->position);

    const float cosTheta = dot(-lightDirection, triangleLightData.normal);

    if(cosTheta > 1e-5f){
        // テクスチャの参照位置
        if(triangleLightData.emissiveTexture.hasTexture){
            const float2 u0 = triangleLightData.uv0;
            const float2 u1 = triangleLightData.uv1;
            const float2 u2 = triangleLightData.uv2;

            const float2 sampledUVCoodinate = u0 + ta * (u1 - u0) + tb * (u2 - u0);
            emission *= make_float3(tex2D<float4>(triangleLightData.emissiveTexture.texture, sampledUVCoodinate.x, 1.0 - sampledUVCoodinate.y));
        }

        float pdfInTriangle = 1.0f / triangleLightData.area; // 面積
        float geometricTerm = cosTheta / fmaxf(distance * distance, 1e-4f);
        lightSample.pdf         = pdfInTriangle /  float(optixLaunchParams.light.numLights) /geometricTerm;
        lightSample.distance    = distance;
        lightSample.position    = sampledPosition;
        lightSample.direction   = lightDirection;
        lightSample.emission    = emission * optixLaunchParams.light.lightIntensityFactor; // MEMO: pdfChoseLight とするのが分かりやすいい？
    }

    return lightSample;
}