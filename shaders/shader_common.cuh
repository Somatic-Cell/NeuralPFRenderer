#ifndef SHADER_COMMON_CUH_
#define SHADER_COMMON_CUH_

#include <cuda_runtime.h>
#include "../utils/helper_math.h"
#define _USE_MATH_DEFINES
#include <math.h>

struct LensRay
{
    float3 org;
    float3 dir;
    float weight    {1.0f};
};

struct IntersectedData
{
    float3 baseColor;
    float ior;
    float roughness;
    float metallic;
    float3 wiLocal;
    float3 normal;
};

struct LightSample
{
    float3  position;
    float3  direction;
    float distance      {1e7f};
    float3  emission    {make_float3(0.0f)};
    float   pdf         {0.0f};
};

inline __device__ __host__ void rotate(const float3 _u, const float r, float3& vx, float3& vy, float3& vz)
{
    float3 u = normalize(_u);
    float s = sin(r);
    float c = cos(r);
    const float3 rx = make_float3(u.x * u.x + (1.f - u.x * u.x) * c,   u.x * u.y * (1.f - c) - u.z * s,    u.x * u.z * (1.f - c) + u.y * s);
    const float3 ry = make_float3(u.x * u.y * (1.f - c) + u.z * s,     u.y * u.y + (1.f - u.y * u.y) * c,  u.y * u.z * (1.f - c) - u.x * s);
    const float3 rz = make_float3(u.x * u.z * (1.f - c) * u.y * s,     u.y * u.z * (1.f - c) + u.x * s,    u.z * u.z + (1.f - u.z * u.z) * c);

    // | rx.x  ry.x  rz.x |   | vx.x  vy.x  vz.x |
    // | rx.y  ry.y  rz.y | * | vx.y  vy.y  vz.y |
    // | rx.z  ry.z  rz.z |   | vx.z  vy.z  vz.z |
    //
    const float3 tmpx = make_float3(rx.x * vx.x + ry.x * vx.y + rz.x * vx.z, rx.x * vy.x + ry.x * vy.y + rz.x * vy.z, rx.x * vz.x + ry.x * vz.y + rz.x * vz.z);
    const float3 tmpy = make_float3(rx.y * vx.x + ry.y * vx.y + rz.y * vx.z, rx.y * vy.x + ry.y * vy.y + rz.y * vy.z, rx.y * vz.x + ry.y * vz.y + rz.y * vz.z);
    const float3 tmpz = make_float3(rx.z * vx.x + ry.z * vx.y + rz.z * vx.z, rx.z * vy.x + ry.z * vy.y + rz.z * vy.z, rx.z * vz.x + ry.z * vz.y + rz.z * vz.z);

    vx = tmpx;
    vy = tmpy;
    vz = tmpz;
}

inline __device__ __host__ float3 random_unit_hemisphere(float rnd1, float rnd2)
{
    float a = rnd1 * 2.f * M_PI;
    float z = 2.f * rnd2 - 1.f;
    float r = sqrtf(1.f - z * z);
    return make_float3(r * cosf(a), z, r * sinf(a));
}

inline __device__ __host__ float3 random_unit_sphere(float rnd1, float rnd2)
{
    float3 p;
    float costheta, phi;
    float sintheta;

    // 球上の一点をサンプリング（極座標表現）
    costheta = 2.f * rnd1 - 1.f; // 0 < theta < pi
    phi = 2.f * M_PI * rnd2;

    sintheta = sqrtf(1.f - costheta * costheta);

    p.x = sintheta * cosf(phi);
    p.y = sintheta * sinf(phi);
    p.z = costheta;

    return p;
}

inline __device__ __host__ float2 random_unit_disk(float rnd1, float rnd2)
{
    float theta = 2.0f * float(M_PI) * rnd1;
    float r = sqrtf(rnd2);
    float2 p = r * make_float2(cosf(theta), sinf(theta));
    return p;
}


// 基底変換
inline __device__ __host__ float3 localToWorld(const float3 v, const float3 localX, const float3 localY, const float3 localZ) {
    return normalize(v.x * localX + v.y * localY + v.z * localZ);
}

inline __device__ __host__ float3 worldToLocal(const float3 v, const float3 localX, const float3 localY, const float3 localZ) {
    return normalize(make_float3(dot(v, localX), dot(v, localY), dot(v, localZ)));
}

// 球面座標
inline __device__ __host__ void orthogonalToSphericalCoord(const float3 dir, float* u, float* v) {
    // const float3 dir = normalize(_dir);
    *u = (M_PI - atan2f(dir.x, dir.z)) / (2.f * M_PI);
    *v = acosf(dir.y) / M_PI;
}


// wi を返す
inline __device__ float3 cosineSampling(const float u, const float v)
{
    const float theta = 0.5f * acosf(1.0f - 2.0f * u);
    const float phi = 2.0f * M_PI * v;

    const float sinTheta    = sinf(theta);
    const float cosTheta    = cosf(theta);
    const float sinPhi      = sinf(phi);
    const float cosPhi      = cosf(phi);
    return make_float3(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi);
}

// wm を返す
inline __device__ float3 visibleNormalSampling( const float alpha, 
                                                const float3 wo, 
                                                const float rnd1, 
                                                const float rnd2)
{
    float3 Vh = normalize(make_float3(alpha * wo.x, wo.y, alpha * wo.z));

    float3 normal = make_float3(0.f, 1.f, 0.f);
    if (Vh.y > 0.99f) {
        normal = make_float3(0.f, 0.f, -1.f);
    }

    float3 T1 = normalize(cross(Vh, normal));
    float3 T2 = cross(T1, Vh);

    float r = sqrtf(fmaxf(rnd1, 0.f));
    float phi = 2.f * M_PI * rnd2;
    float t1 = r * cosf(phi);
    float t2 = r * sinf(phi);
    float s = 0.5f * (1.f + Vh.y);
    t2 = (1.f - s) * sqrtf(fmaxf(1.f - t1 * t1, 0.f)) + s * t2;

    float3 Nh = t1 * T1 + t2 * T2 + sqrtf(fmaxf(1e-7f, 1.f - t1 * t1 - t2 * t2)) * Vh;
    float3 Ne = normalize(make_float3(alpha * Nh.x, Nh.y, alpha * Nh.z));
    return Ne;
}

inline __device__ float3 evalLambertBRDF(const float3 baseColor, const float3 wo, const float3 wi)
{
    return baseColor / M_PI;
}

inline __device__ float getLambertPdf(const float3 wo, const float3 wi){
    return fmaxf(wi.y, 1e-4f) / M_PI ;
}

inline __device__ float ggx_D(const float alpha, const float3 wm)
{
    const float t = wm.y * wm.y + (wm.x * wm.x + wm.z * wm.z) / (alpha * alpha);
    return 1.f / (M_PI * alpha * alpha * t * t);
}

inline __device__ float lambda(const float alpha, const float3 w)
{
    float vy = fmaxf(w.y, 1e-5f);
    float t = alpha * alpha * (w.x * w.x + w.z * w.z) / (vy * vy);
    return (-1.f + sqrtf(1.f + t)) / 2.f;
}

inline __device__ float smith_G1(const float alpha, const float3 w)
{
    return 1.0f / (1.0f + lambda(alpha, w));
}

inline __device__ float smith_G2(const float alpha, const float3 wo, const float3 wi)
{
    return 1.0f / (1.0f + lambda(alpha, wo) + lambda(alpha, wi));
}

inline __device__ float3 schlick(const float3 wo, const float3 n, const float3 F0){
    return F0 + (make_float3(1.0f) - F0) * powf(1.f - dot(wo, n), 5);
}

inline __device__ float schlick(const float3 wo, const float3 n, const float F0){
    return F0 + (1.0f - F0) * powf(1.f - dot(wo, n), 5);
}

inline __device__ float3 evalSpecularBRDF( const float alpha, 
                                        const float3 F0, 
                                        const float3& wo, 
                                        const float3& wi)
{
    float3 wm = normalize(wo + wi);

    float D = ggx_D(alpha, wm);
    float G = smith_G2(alpha, wo, wi);
    float3 F = schlick(wo, wm, F0);

    float in = fmaxf(wi.y, 1e-4f);
    float on = fmaxf(wo.y, 1e-4f);

    float3 brdf = F * G * D / (4.f * in * on + 1e-5f);

    return brdf;
}

inline __device__ float getGGXPdf(const float alpha, const float3 wo, const float3 wi){
    float3 wm = normalize(wo + wi);
    return ggx_D(alpha, wm) * smith_G1(alpha, wo) * fmaxf(dot(wo, wm), 1e-7f) / (fmaxf(wo.y, 1e-7f) * 4.0f * fmaxf(dot(wi, wm), 1e-7f));
}

__forceinline__ __device__ float balanceHeuristicWeight(const int num_a, const float pdf_a, const int num_b, const float pdf_b){
    float np_a = (float)num_a * pdf_a;
    float np_b = (float)num_b * pdf_b;
    return np_a / (np_a + np_b);
}



#endif // SHADER_COMMON_CUH_