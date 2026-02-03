#ifndef TRACE_VOLUME_CUH_
#define TRACE_VOLUME_CUH_

#include <optix.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/NanoVDB.h>

#include "../config.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../utils/helper_math.h"
#include "../../utils/my_math.hpp"
#include "../params/per_ray_data.cuh"
#include "shader_common.cuh"
#define _USE_MATH_DEFINES
#include <math.h>

#include <optix_device.h>  
#ifdef __CUDACC__
#include <cuda_fp16.h>      // half を使うなら（OptixCoopVec<half, N> 等）
#endif


template <typename AccT>
static __forceinline__ __device__
auto makeSampler(const AccT& acc)
-> decltype(nanovdb::math::createSampler<0>(acc))
{
    return nanovdb::math::createSampler<0>(acc);
}

static __forceinline__ __device__
int floorToInt(float x)
{
    return __float2int_rd(x);
}

static __forceinline__ __device__
nanovdb::Coord floorToCoord(const nanovdb::Vec3f& p)
{
    return nanovdb::Coord(floorToInt(p[0]), floorToInt(p[1]), floorToInt(p[2]));
}

// world 空間から vdb の index への変換
static __forceinline__ __device__ 
nanovdb::Vec3f worldToIndexPoint(
    const nanovdb::FloatGrid* grid,
    const nanovdb::Vec3f& pW
)
{
    return grid->map().applyInverseMap(pW);
}

static __forceinline__ __device__ 
nanovdb::Vec3f worldToIndexVector(
    const nanovdb::FloatGrid* grid,
    const nanovdb::Vec3f& pW,
    const nanovdb::Vec3f& vW
)
{
    return grid->map().applyInverseJacobian(vW);
    // const nanovdb::Vec3f a = grid->map().applyInverseMap(pW);
    // const nanovdb::Vec3f b = grid->map().applyInverseMap(pW + vW);
    // return b - a;
}

static __forceinline__ __device__
bool intersectAABBVec3(
    const nanovdb::Vec3f& origin,
    const nanovdb::Vec3f& direction,
    const nanovdb::Vec3f& bMin, 
    const nanovdb::Vec3f& bMax,
    float tMin, float tMax,
    float& tEnter,  // aabb との交差位置（手前）
    float& tExit    // aabb との交差位置（奥）
)
{
    float t0 = tMin, t1 = tMax;

    for(int ax = 0; ax < 3; ++ax){
        const float invD = (fabsf(direction[ax]) > 1e-20f) ? 1.0f / direction[ax] : 1e20f;
        float tNear = (bMin[ax] - origin[ax]) * invD;
        float tFar  = (bMax[ax] - origin[ax]) * invD;

        if(tNear > tFar){float tmp = tNear; tNear = tFar; tFar=tmp;}
        t0 = fmaxf(t0, tNear);
        t1 = fminf(t1, tFar);
        if(t0 > t1) return false;
    }

    tEnter = t0;
    tExit = t1;
    return true;
}


struct LocalSegment
{
    float tEnd;     // セグメントの終わりまでの距離
    float majorant; // セグメント内の局所的な上界
};

template<typename AccT>
static __forceinline__ __device__
LocalSegment getLocalSegment(
    const nanovdb::FloatGrid* grid,
    const AccT & acc,
    const nanovdb::Vec3f& rayOriginWorld,
    const nanovdb::Vec3f& rayDirectionWorld,
    const nanovdb::Vec3f& rayOriginIndex,
    const nanovdb::Vec3f& rayDirectionIndex,
    float t,            // 現在の距離
    float tExitWorld,   // 外までの距離
    float densityScale,
    float sigmaTScale
)
{
    LocalSegment seg;
    seg.tEnd = tExitWorld;
    seg.majorant = 0.0f;

    // 少しだけ前進した位置のセル情報を取得
    const float tt = fminf(tExitWorld, t + 1e-3f);
    const nanovdb::Vec3f pI = rayOriginIndex + tt * rayDirectionIndex;
    const nanovdb::Coord ijk = floorToCoord(pI);

    // NodeInfo をとる
    const auto info = acc.getNodeInfo(ijk);

    // const int dim = info.mDim;          // ノードのサイズ (おそらく 8 の倍数)
    const float vmax = info.maximum;   // ノード内の最大密度
    const nanovdb::Coord bbMinC = info.bbox.min(); 
    const nanovdb::Coord bbMaxC = info.bbox.max();
    
    const nanovdb::Vec3f bbMin((float)bbMinC[0], (float)bbMinC[1], (float)bbMinC[2]);
    const nanovdb::Vec3f bbMax((float)(bbMaxC[0] + 1), (float)(bbMaxC[1] + 1), (float)(bbMaxC[2] + 1));

    float tN, tF;
    if(intersectAABBVec3(rayOriginIndex, rayDirectionIndex, bbMin, bbMax, t, tExitWorld, tN, tF)){
        seg.tEnd = fminf(tF, tExitWorld);
    } else {
        seg.tEnd = fminf(tt, tExitWorld);
    }

    seg.majorant = fmaxf(vmax * densityScale * sigmaTScale, 0.0f);
    return seg;
}

static __forceinline__ __device__
float sampleFreeFlight(
    float u,
    float sigma
)
{
    u = fminf(fmaxf(u, 0.0f), 0.99999994f);
    sigma = fmaxf(sigma, 1e-7f);

    return - log1pf(-u) / sigma;
}


template <class PRD>
static __forceinline__ __device__
bool deltaTrack_localMajorant(
    PRD& prd,
    uint32_t vdbIndex,
    const float3& rayOriginObject,
    const float3& rayDirectionObject,
    float tEnter,
    float tExit,
    float sigmaTScale,
    float& outT
)
{
    if(!(tExit > tEnter)) return false;

    const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(
        optixLaunchParams.vdbs[vdbIndex].nanoGrid
    );
    const float densityScale = optixLaunchParams.vdbs[vdbIndex].densityScale;

    // 1回だけ呼ぶのを推奨されているやつ
    auto acc = grid->getAccessor();
    auto sampler = makeSampler(acc);

    const nanovdb::Vec3f rayOriginWorld(rayOriginObject.x, rayOriginObject.y, rayOriginObject.z);
    const nanovdb::Vec3f rayDirectionWorld(rayDirectionObject.x, rayDirectionObject.y, rayDirectionObject.z);
    
    const nanovdb::Vec3f rayOriginIndex     = worldToIndexPoint(grid, rayOriginWorld);
    const nanovdb::Vec3f rayDirectionIndex  = worldToIndexVector(grid, rayOriginWorld, rayDirectionWorld);
    
    float t = tEnter;
    for(int i = 0; i < 4096; ++i){
        // はみ出た場合
        if(t >= tExit) return false;

        const LocalSegment seg = getLocalSegment(
            grid, acc, rayOriginWorld, rayDirectionWorld, rayOriginIndex, rayDirectionIndex,
            t, tExit, densityScale, sigmaTScale

        );

        if(seg.majorant <= 0.0f){
            t = seg.tEnd;
            continue;
        }

        const float tCand = t + sampleFreeFlight(prd.random(), seg.majorant);
        // セルをはみ出すくらい大きな工程をサンプリングしたら，境界で止める
        if(tCand >= seg.tEnd){
            t = seg.tEnd;
            continue;
        }

        t = tCand;

        const nanovdb::Vec3f positionWorld = rayOriginWorld + t * rayDirectionWorld;
        const nanovdb::Vec3f positionIndex = worldToIndexPoint(grid, positionWorld);
        const float density = sampler(positionIndex);

        const float sigmaT = fmaxf(density * densityScale * sigmaTScale, 0.0f);

        // null collision かどうか
        float ratio = sigmaT / seg.majorant;
        ratio = fminf(fmaxf(ratio, 0.0f), 1.0f);

        if(prd.random() < ratio){
            outT = t;
            return true;
        }
    }

    return false;

}

template <class PRD>
static __forceinline__ __device__
float ratioTrack_localMajorant(
    PRD& prd,
    uint32_t vdbIndex,
    const float3& rayOriginObject,
    const float3& rayDirectionObject,
    float tEnter,
    float tExit,
    float sigmaTScale
)
{
    if(!(tExit > tEnter)) return 1.0f;

    const auto* grid = reinterpret_cast<const nanovdb::FloatGrid*>(
        optixLaunchParams.vdbs[vdbIndex].nanoGrid
    );
    const float densityScale = optixLaunchParams.vdbs[vdbIndex].densityScale;

    // 1回だけ呼ぶのを推奨されているやつ
    auto acc = grid->getAccessor();
    auto sampler = makeSampler(acc);

    const nanovdb::Vec3f rayOriginWorld(rayOriginObject.x, rayOriginObject.y, rayOriginObject.z);
    const nanovdb::Vec3f rayDirectionWorld(rayDirectionObject.x, rayDirectionObject.y, rayDirectionObject.z);
    const nanovdb::Vec3f rayOriginIndex     = worldToIndexPoint(grid, rayOriginWorld);
    const nanovdb::Vec3f rayDirectionIndex  = worldToIndexVector(grid, rayOriginWorld, rayDirectionWorld);
    
    float t = tEnter;
    float transmittance = 1.0f;

    for(int i = 0; i < 4096; ++i){
        // はみ出た場合はレイマーチング修了
        if(t >= tExit) break;

        const LocalSegment seg = getLocalSegment(
            grid, acc, rayOriginWorld, rayDirectionWorld, rayOriginIndex, rayDirectionIndex,
            t, tExit, densityScale, sigmaTScale

        );

        // 空だった場合はまとめてスキップ
        if(seg.majorant <= 0.0f){
            t = seg.tEnd;
            continue;
        }

        // 自由行程サンプリング
        const float tCand = t + sampleFreeFlight(prd.random(), seg.majorant);
        
        // セルをはみ出すくらい大きな工程をサンプリングしたら，境界で止める
        if(tCand >= seg.tEnd){
            t = seg.tEnd;
            continue;
        }

        // 候補点で密度を評価
        t = tCand;
        const nanovdb::Vec3f positionWorld = rayOriginWorld + t * rayDirectionWorld;
        const nanovdb::Vec3f positionIndex = worldToIndexPoint(grid, positionWorld);

        const float density = sampler(positionIndex);
        const float sigmaT = fmaxf(density * densityScale * sigmaTScale, 0.0f);

        // null collision かどうか
        float ratio = 1.0f - (sigmaT / seg.majorant);
        ratio = fminf(fmaxf(ratio, 0.0f), 1.0f);

        transmittance *= ratio;

        if(transmittance < 1e-6f) return 0.0f; // ほぼ透過しない場合は打ち切る

    }

    return transmittance;

}
#endif // TRACE_VOLUME_CUH_