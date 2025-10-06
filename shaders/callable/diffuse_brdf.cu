#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"

extern "C" __device__ float3 __direct_callable__brdf_diffuse_sample_rgb(const float3 wo, const IntersectedData_RGB & material, PRDRGB* prd)
{
    const float3 wi = randomCosineHemisphere(prd->random(), prd->random());
    const float pdf = getLambertPdf(wo, wi);
    prd->pdf.bxdf = pdf;
    prd->lastHitMaterialType = MATERIAL_TYPE_DIFFUSE;
    return wi;
}

extern "C" __device__ float3  __direct_callable__brdf_diffuse_eval_rgb(const float3 wi, const float3 wo, const IntersectedData_RGB & material, PRDRGB* prd)
{
    const float pdf = getLambertPdf(wo, wi);
    prd->pdf.bxdf = pdf;
    prd->lastHitMaterialType = MATERIAL_TYPE_DIFFUSE;
    return evalLambertBRDF_RGB(material.baseColor, wo, wi);
}

extern "C" __device__ float3 __direct_callable__brdf_diffuse_sample_spectral(const float3 wo, const IntersectedData_Spectral & material, PRDSpectral* prd)
{
    const float3 wi = randomCosineHemisphere(prd->random(), prd->random());
    const float pdf = getLambertPdf(wo, wi);
    prd->pdf.bxdf = pdf;
    prd->lastHitMaterialType = MATERIAL_TYPE_DIFFUSE;
    return wi;
}

extern "C" __device__ float  __direct_callable__brdf_diffuse_eval_spectral(const float3 wi, const float3 wo, const IntersectedData_Spectral & material, PRDSpectral* prd)
{
    const float pdf = getLambertPdf(wo, wi);
    prd->pdf.bxdf = pdf;
    prd->lastHitMaterialType = MATERIAL_TYPE_DIFFUSE;
    return evalLambertBRDF_Spectral(material.baseColor, wo, wi);
}