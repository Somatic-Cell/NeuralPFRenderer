#ifndef TRACE_VOLUME_CUH_
#define TRACE_VOLUME_CUH_

#include <optix.h>
#include <nanovdb/math/SampleFromVoxels.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/math/HDDA.h> 

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

static __forceinline__ __device__
int floorDivInt(int a, int b)
{
    int q = a / b;
    int r = a % b;
    if (r != 0 && ((r > 0) != (b > 0))) --q; // C/C++ の /,% は 0 方向丸めなので補正
    return q;
}

template <typename AccT>
static __forceinline__ __device__
auto makeSampler(const AccT& acc)
-> decltype(nanovdb::math::createSampler<1>(acc))
{
    return nanovdb::math::createSampler<1>(acc);
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
    float t0 = tMin;
    float t1 = tMax;
    auto slab = [&](float oA, float dA, float mn, float mx) -> bool 
    { 
        if (fabsf(dA) < 1e-20f) { // 平行：原点がスラブ外なら不交差 
            return (oA >= mn && oA <= mx); 
        } const float invD = 1.0f / dA; 
        float tNear = (mn - oA) * invD; 
        float tFar  = (mx - oA) * invD;

        if (tNear > tFar) { 
            float tmp = tNear; 
            tNear = tFar; 
            tFar = tmp; 
        } 
        t0 = fmaxf(t0, tNear); 
        t1 = fminf(t1, tFar); 
        return (t0 < t1); 
    };

    for(int ax = 0; ax < 3; ++ax){
        if(!slab(origin[ax], direction[ax], bMin[ax], bMax[ax])) return false;
    }

    float tE = t0;
    float tX = t1;

    if(tX <= tE) return false;

    tEnter  = fmaxf(tMin, tE);
    tExit   = fminf(tMax, tX);

    return (tEnter < tExit);
}


struct LocalSegment
{
    float tEnd;     // セグメントの終わりまでの距離
    float majorant; // セグメント内の局所的な上界
};

struct IndexRay
{
    using RealType = float;
    using Vec3Type = nanovdb::Vec3f;

    __hostdev__ IndexRay() = default;

    __hostdev__ IndexRay(const Vec3Type& eye, const Vec3Type& dir, float t0, float t1)
        : mEye(eye), mDir(dir), mT0(t0), mT1(t1)
    {
        // invDir の 0 割り回避
        mInv = Vec3Type(
            fabsf(mDir[0]) > 1e-20f ? 1.0f / mDir[0] : nanovdb::math::Maximum<float>::value(),
            fabsf(mDir[1]) > 1e-20f ? 1.0f / mDir[1] : nanovdb::math::Maximum<float>::value(),
            fabsf(mDir[2]) > 1e-20f ? 1.0f / mDir[2] : nanovdb::math::Maximum<float>::value()
        );
    }

    __hostdev__ Vec3Type operator()(float t) const { return mEye + t * mDir; }
    __hostdev__ const Vec3Type& dir()    const { return mDir; }
    __hostdev__ const Vec3Type& invDir() const { return mInv; }
    __hostdev__ float t0() const { return mT0; }
    __hostdev__ float t1() const { return mT1; }

    Vec3Type mEye, mDir, mInv;
    float    mT0 = 0.0f, mT1 = 0.0f;
};

#ifndef VDB_TRACKING_HALO_MAJORANT
#define VDB_TRACKING_HALO_MAJORANT 1
#endif

template<typename AccT>
struct LeafSegmentStepper
{
    using TreeType = nanovdb::FloatGrid::TreeType;
    static constexpr int LEAF_DIM = TreeType::LeafNodeType::DIM;

    using RayT = IndexRay;
    using DDA  = nanovdb::math::DDA<RayT, nanovdb::Coord, LEAF_DIM>;

    __device__ LeafSegmentStepper(
        const nanovdb::FloatGrid* grid,
        const AccT& acc,
        const nanovdb::Vec3f& rayOriginIndex,
        const nanovdb::Vec3f& rayDirectionIndex,
        float tEnter,
        float tExit,
        float densityScale,
        float sigmaTScale
    )
        : mGrid(grid)
        , mAcc(acc)
        , mRay(rayOriginIndex, rayDirectionIndex, tEnter, tExit)
        , mDensityScale(densityScale)
        , mSigmaTScale(sigmaTScale)
    {
        mDda.init(mRay, tEnter, tExit);
        updateCurrentSegment();
    }

    __device__ bool valid() const
    {
        // mDda.time() は現在境界（セグメント開始相当）
        return mDda.time() < mRay.t1();
    }

    __device__ const LocalSegment& seg() const { return mSeg; }

    __device__ bool advance() // 次ブロックへ
    {
        if (!mDda.step()) return false;
        updateCurrentSegment();
        return true;
    }

private:
    __device__ void updateCurrentSegment()
    {
        // 次境界時刻（＝セグメント終端）
        float tEnd = mDda.next();

        // 数値誤差で tEnd==現在になったときに停滞しないように少し前進
        const float t0 = mDda.time();
        tEnd = fmaxf(tEnd, nextafterf(t0, mRay.t1()));

        mSeg.tEnd = tEnd;

        // ブロック原点（LEAF_DIM アライン）
        const nanovdb::Coord base = mDda.voxel();

        float vmax = mAcc.getNodeInfo(base).maximum;

#if VDB_TRACKING_HALO_MAJORANT
        // trilinear の halo を簡易に保守化（+方向の隣接 leaf を見る）
        // まずは 8 個（0/1 の組合せ）で十分なことが多いです
        #pragma unroll
        for (int dz = 0; dz <= 1; ++dz)
        #pragma unroll
        for (int dy = 0; dy <= 1; ++dy)
        #pragma unroll
        for (int dx = 0; dx <= 1; ++dx) {
            const nanovdb::Coord nb = base + nanovdb::Coord(dx * LEAF_DIM, dy * LEAF_DIM, dz * LEAF_DIM);
            vmax = fmaxf(vmax, mAcc.getNodeInfo(nb).maximum);
        }
#endif

        mSeg.majorant = fmaxf(vmax * mDensityScale * mSigmaTScale, 0.0f);
    }

    const nanovdb::FloatGrid* mGrid = nullptr;
    const AccT&               mAcc;
    RayT                      mRay;
    DDA                       mDda;
    LocalSegment              mSeg;
    float                     mDensityScale = 1.0f;
    float                     mSigmaTScale  = 1.0f;
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
    float sigmaTScale,
    float globalMajorant
)
{
    LocalSegment seg;
    seg.tEnd = tExitWorld;
    seg.majorant = 0.0f;

    // 現在よりも少しだけ前進した位置のセル情報を取得
    const float epsIndex = 1e-3f;
    const float dirIndexLength = fmaxf(length(make_float3(rayDirectionIndex[0], rayDirectionIndex[1], rayDirectionIndex[2])), 1e-20f);
    const float dt = epsIndex / dirIndexLength;

    const float tt = fminf(tExitWorld, nextafterf(t, tExitWorld) + dt);

    const nanovdb::Vec3f pI = rayOriginIndex + tt * rayDirectionIndex;
    const nanovdb::Coord ijk = floorToCoord(pI + nanovdb::Vec3f(0.5f));

    // NodeInfo をとる
    const auto info = acc.getNodeInfo(ijk);

    // const int dim = info.mDim();          // ノードのサイズ (おそらく 8 の倍数)
    // printf("ndim : %d\n", dim);
    const float vmax = info.maximum;   // この node / tile の最大密度

    // using TreeType = nanovdb::FloatGrid::TreeType;
    // constexpr int LEAF_DIM = TreeType::LeafNodeType::DIM;

    // const int bx = floorDivInt(ijk[0], LEAF_DIM) * LEAF_DIM;
    // const int by = floorDivInt(ijk[1], LEAF_DIM) * LEAF_DIM;
    // const int bz = floorDivInt(ijk[2], LEAF_DIM) * LEAF_DIM;

    // const nanovdb::Coord bbMinC(bx, by, bz); 
    // const nanovdb::Coord bbMaxC(bx + LEAF_DIM - 1, by  + LEAF_DIM - 1, bz  + LEAF_DIM - 1);
    
    const nanovdb::Coord bbMinC = info.bbox.min(); 
    const nanovdb::Coord bbMaxC = info.bbox.max();
    
    const nanovdb::Vec3f bbMin(
        (float)bbMinC[0] - 1.0f,
        (float)bbMinC[1] - 1.0f, 
        (float)bbMinC[2] - 1.0f);
    const nanovdb::Vec3f bbMax(
        (float)bbMaxC[0] + 1.0f, 
        (float)bbMaxC[1] + 1.0f, 
        (float)bbMaxC[2] + 1.0f);

    float tNear, tFar;
    if(intersectAABBVec3(rayOriginIndex, rayDirectionIndex, bbMin, bbMax, t, tExitWorld, tNear, tFar)){
        seg.tEnd = fminf(tFar, tExitWorld);
        seg.tEnd = fmaxf(seg.tEnd, nextafterf(t, tExitWorld));
    } else {
        seg.tEnd = tExitWorld;
    }

    // seg.majorant = fmaxf(vmax * densityScale * sigmaTScale, 0.0f);
    // local majorant にせず，empty span だけ飛ばし，active span 内では global majorant を使う
    const float localUpper = fmaxf(vmax * densityScale * sigmaTScale, 0.0f);
    seg.majorant = (localUpper > 0.0f) ? globalMajorant : 0.0f;
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


template <class PRD, class AccT, class SamplerT>
static __forceinline__ __device__
bool deltaTrack_localMajorant(
    PRD& prd,
    const float densityScale,
    const nanovdb::FloatGrid* grid,
    const AccT& acc,
    const SamplerT& sampler,
    const float3& rayOriginObject,
    const float3& rayDirectionObject,
    float tEnter,
    float tExit,
    float sigmaTScale,
    float& outT
)
{
    if(!(tExit > tEnter)) return false;

    if (!optixLaunchParams.vdbs) return false;

    

    const nanovdb::Vec3f rayOriginWorld(rayOriginObject.x, rayOriginObject.y, rayOriginObject.z);
    const nanovdb::Vec3f rayDirectionWorld(rayDirectionObject.x, rayDirectionObject.y, rayDirectionObject.z);
    
    const nanovdb::Vec3f rayOriginIndex     = worldToIndexPoint(grid, rayOriginWorld);
    const nanovdb::Vec3f rayDirectionIndex  = worldToIndexVector(grid, rayOriginWorld, rayDirectionWorld);
    
    float t = tEnter;
    LocalSegment seg;
    
    // seg = getLocalSegment(
    //     grid, acc, rayOriginWorld, rayDirectionWorld, rayOriginIndex, rayDirectionIndex,
    //     t, tExit, densityScale, sigmaTScale
    // );
    
    seg.tEnd = tExit;
    seg.majorant = grid->tree().root().maximum() * densityScale * sigmaTScale;

    // RNG state をローカルへ退避（レジスタに乗りやすい）
    Random random = prd.random;

    for(int i = 0; i < 4096; ++i){
        // はみ出た場合
        if(t >= tExit) return false;

        // if(seg.majorant <= 0.0f){
        //     t = seg.tEnd;
        //     if(t >= tExit) return false;
        //     seg = getLocalSegment(
        //         grid, acc, rayOriginWorld, rayDirectionWorld, rayOriginIndex, rayDirectionIndex,
        //         t, tExit, densityScale, sigmaTScale
        //     );
        //     continue;
        // }
        
        const float tCand = t + sampleFreeFlight(random(), seg.majorant);
        // セルをはみ出すくらい大きな工程をサンプリングしたら，境界で止める
        if(tCand >= seg.tEnd){
            t = seg.tEnd;
            if(t >= tExit) {
                prd.random = random;
                return false;
            }
            // seg = getLocalSegment(
            //     grid, acc, rayOriginWorld, rayDirectionWorld, rayOriginIndex, rayDirectionIndex,
            //     t, tExit, densityScale, sigmaTScale
            // );
            continue;
        }

        t = tCand;

        // const nanovdb::Vec3f positionWorld = rayOriginWorld + t * rayDirectionWorld;
        const nanovdb::Vec3f positionIndex = rayOriginIndex + t * rayDirectionIndex;
        const float density = sampler(positionIndex);

        const float sigmaT = fmaxf(density * densityScale * sigmaTScale, 0.0f);

        // null collision かどうか
        float ratio = sigmaT / seg.majorant;
        ratio = fminf(fmaxf(ratio, 0.0f), 1.0f);

        if(random() < ratio){
            outT = t;
            prd.random = random;
            return true;
        }
    }
    prd.random = random;
    return false;

}

template <class PRD, class AccT, class SamplerT>
static __forceinline__ __device__
float ratioTrack_localMajorant(
    PRD& prd,
    const float densityScale,
    const nanovdb::FloatGrid* grid,
    const AccT& acc,
    const SamplerT& sampler,
    const float3& rayOriginObject,
    const float3& rayDirectionObject,
    float tEnter,
    float tExit,
    float sigmaTScale
)
{
    if(!(tExit > tEnter)) return 1.0f;

    if (!optixLaunchParams.vdbs) return 1.0f;

    if(!(tExit > tEnter)) return 1.0f;

    const nanovdb::Vec3f rayOriginWorld(rayOriginObject.x, rayOriginObject.y, rayOriginObject.z);
    const nanovdb::Vec3f rayDirectionWorld(rayDirectionObject.x, rayDirectionObject.y, rayDirectionObject.z);
    const nanovdb::Vec3f rayOriginIndex     = worldToIndexPoint(grid, rayOriginWorld);
    const nanovdb::Vec3f rayDirectionIndex  = worldToIndexVector(grid, rayOriginWorld, rayDirectionWorld);
    
    float t = tEnter;
    float transmittance = 1.0f;

    LocalSegment seg;
    seg.tEnd = tExit;
    seg.majorant = grid->tree().root().maximum() * densityScale * sigmaTScale;

    // seg = getLocalSegment(
    //     grid, acc, rayOriginWorld, rayDirectionWorld, rayOriginIndex, rayDirectionIndex,
    //     t, tExit, densityScale, sigmaTScale
    // );

    Random random = prd.random;

    for(int i = 0; i < 4096; ++i){
        // はみ出た場合はレイマーチング修了
        if(t >= tExit) break;

        // 空だった場合はまとめてスキップ
        // if(seg.majorant <= 0.0f){
        //     t = seg.tEnd;
        //     if(t >= tExit) break;
        //     seg = getLocalSegment(
        //         grid, acc, rayOriginWorld, rayDirectionWorld, rayOriginIndex, rayDirectionIndex,
        //         t, tExit, densityScale, sigmaTScale
        //     );
        //     continue;
        // }
        
        // 自由行程サンプリング
        const float tCand = t + sampleFreeFlight(random(), seg.majorant);

        // セルをはみ出すくらい大きな工程をサンプリングしたら，境界で止める
        if(tCand >= seg.tEnd){
            t = seg.tEnd;
            if(t >= tExit) break;
            // seg = getLocalSegment(
            //     grid, acc, rayOriginWorld, rayDirectionWorld, rayOriginIndex, rayDirectionIndex,
            //     t, tExit, densityScale, sigmaTScale
            // );
            continue;
        }

        // 候補点で密度を評価
        t = tCand;

        // const nanovdb::Vec3f positionWorld = rayOriginWorld + t * rayDirectionWorld;
        const nanovdb::Vec3f positionIndex = rayOriginIndex + t * rayDirectionIndex;

        const float density = sampler(positionIndex);
        const float sigmaT = fmaxf(density * densityScale * sigmaTScale, 0.0f);

        // null collision かどうか
        float ratio = 1.0f - (sigmaT / seg.majorant);
        ratio = fminf(fmaxf(ratio, 0.0f), 1.0f);

        transmittance *= ratio;

        if(transmittance < 1e-6f) {
            prd.random = random;
            return 0.0f;
        }
             // ほぼ透過しない場合は打ち切る
    }
    prd.random = random;
    return transmittance;

}
#endif // TRACE_VOLUME_CUH_