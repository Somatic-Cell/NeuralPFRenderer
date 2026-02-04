#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include <stdio.h>


extern "C" __global__ void __intersection__vdb()
{
    const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    const OptixAabb aabb = optixLaunchParams.vdbAABBs[primitiveIndex];

    const float3 bmin = make_float3(aabb.minX, aabb.minY, aabb.minZ);
    const float3 bmax = make_float3(aabb.maxX, aabb.maxY, aabb.maxZ);

    const float3 rayOrigin      = optixGetObjectRayOrigin();
    const float3 rayDirection   = optixGetObjectRayDirection();

    const float tMin = optixGetRayTmin();
    const float tMax = optixGetRayTmax();

    float tEnter, tExit;
    if(!intersectAABB(rayOrigin, rayDirection, bmin, bmax, tMin, tMax, tEnter, tExit)) return;
    
    const bool inside = 
        ( rayOrigin.x >= bmin.x && rayOrigin.x <= bmax.x) &&
        ( rayOrigin.y >= bmin.y && rayOrigin.y <= bmax.y) &&
        ( rayOrigin.z >= bmin.z && rayOrigin.z <= bmax.z);

    const unsigned int hitKind = inside ? HIT_KIND_VDB_INSIDE : HIT_KIND_VDB_ENTER;

    const unsigned int a0 = __float_as_uint(tExit);

    // 後段の optixGetRayTmax() では tEnter を取得
    // tExit も欲しいので，a0 に埋め込む
    optixReportIntersection(tEnter, hitKind, a0);
}