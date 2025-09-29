#include "config.cuh"

#include <optix.h>

#include "per_ray_data.cuh"
#include "shader_common.cuh"
#include "random_number_generator.cuh"
#include "../include/launch_params.h"

extern "C" __device__ float3 __direct_callable__bsdf_glass_sample(const float3 wo, const IntersectedData& material, PRD* prd)
{
    float cosThetaI = wo.y;
    float eta = 1.0f / material.ior;
    float3 N = make_float3(0.f, 1.0f, 0.f);
    if(cosThetaI < 0.0f){
        eta = 1.0f / eta;
        N = -N;
        cosThetaI = -cosThetaI;
    }

    float F0 = powf(__fdividef((1.0f - eta), (1.f + eta)), 2.f);
    float fresnel = schlick(wo, N, F0);

    float sin2ThetaT = (1.0f - cosThetaI * cosThetaI) * eta * eta;

    if(sin2ThetaT > 1.0f || prd->random() < fresnel){
        prd->position +=  N * 1e-3f;
        return normalize(- 1.0f * wo + N * 2.f * dot(N, wo));
    } else {
        float cosThetaT = sqrtf(fmaxf(1.0f - sin2ThetaT, 1e-7f));
        prd->position -=  N * 1e-3f;
        return normalize(eta * (-1.0f * wo) + (eta * cosThetaI - cosThetaT) * N);   
    }
    prd->lastHitMaterialType = MATERIAL_TYPE_GLASS;
}

extern "C" __device__ float3 __direct_callable__bsdf_glass_eval(const float3 wi, const float3 wo, const IntersectedData& material, PRD* prd)
{
    const float pdf = getLambertPdf(wo, wi);
    prd->pdf.bxdf = pdf;
    return evalLambertBRDF(material.baseColor, wo, wi);
}