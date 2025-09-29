#ifndef MY_MATH_HPP_
#define MY_MATH_HPP_

#include <cuda_runtime.h>
#include "helper_math.h"
#define _USE_MATH_DEFINES
#include <math.h>


namespace mymath{
    inline __device__ __host__ void rotate(const float3 _u, const float theta, float3& vx, float3& vy, float3& vz)
  {
    float3 u = normalize(_u);
    float s = sin(theta);
    float c = cos(theta);
    const float3 rx = make_float3(u.x * u.x + (1.f - u.x * u.x) * c,   u.x * u.y * (1.f - c) - u.z * s,    u.x * u.z * (1.f - c) + u.y * s);
    const float3 ry = make_float3(u.x * u.y * (1.f - c) + u.z * s,     u.y * u.y + (1.f - u.y * u.y) * c,  u.y * u.z * (1.f - c) - u.x * s);
    const float3 rz = make_float3(u.x * u.z * (1.f - c) * u.y * s,     u.y * u.z * (1.f - c) + u.x * s,    u.z * u.z + (1.f - u.z * u.z) * c);

    // | rx.x  ry.x  rz.x |   | vx.x  vy.x  vz.x |
    // | rx.y  ry.y  rz.y | * | vx.y  vy.y  vz.y |
    // | rx.z  ry.z  rz.z |   | vx.z  vy.z  vz.z |
    // R * X
    const float3 tmpx = make_float3(rx.x * vx.x + ry.x * vx.y + rz.x * vx.z, rx.x * vy.x + ry.x * vy.y + rz.x * vy.z, rx.x * vz.x + ry.x * vz.y + rz.x * vz.z);
    const float3 tmpy = make_float3(rx.y * vx.x + ry.y * vx.y + rz.y * vx.z, rx.y * vy.x + ry.y * vy.y + rz.y * vy.z, rx.y * vz.x + ry.y * vz.y + rz.y * vz.z);
    const float3 tmpz = make_float3(rx.z * vx.x + ry.z * vx.y + rz.z * vx.z, rx.z * vy.x + ry.z * vy.y + rz.z * vy.z, rx.z * vz.x + ry.z * vz.y + rz.z * vz.z);

    vx = tmpx;
    vy = tmpy;
    vz = tmpz;
  }
}
#endif // MY_MATH_HPP_