#include "../config.cuh"

#include <optix.h>

#include "../params/per_ray_data.cuh"
#include "../device/shader_common.cuh"
#include "../device/random_number_generator.cuh"
#include "../../include/launch_params.h"
#include <stdio.h>

// shadow ray が交差しない = 可視
extern "C" __global__ void __miss__shadow()
{
    ShadowPRD &prd = *getPRD<ShadowPRD>(); 
    // 何にもヒットしないので visible
    prd.visible = true; 
}