#include "config.cuh"

#include <optix.h>

#include "per_ray_data.cuh"
#include "shader_common.cuh"
#include "random_number_generator.cuh"
#include "../include/launch_params.h"

extern "C" __device__ float3 __direct_callable__brdf_diffuse_sample(const float3 wo, const IntersectedData & material, PRD* prd)
{
    const float3 wi = random_unit_hemisphere(prd->random(), prd->random());
    const float pdf = getLambertPdf(wo, wi);
    prd->pdf.bxdf = pdf;
    prd->lastHitMaterialType = MATERIAL_TYPE_DIFFUSE;
    return wi;
}

extern "C" __device__ float3  __direct_callable__brdf_diffuse_eval(const float3 wi, const float3 wo, const IntersectedData & material, PRD* prd)
{
    const float pdf = getLambertPdf(wo, wi);
    prd->pdf.bxdf = pdf;
    prd->lastHitMaterialType = MATERIAL_TYPE_DIFFUSE;
    return evalLambertBRDF(material.baseColor, wo, wi);
}