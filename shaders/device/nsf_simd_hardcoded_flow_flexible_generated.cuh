#pragma once
// Generated from safetensors.
//
// This file is a standalone replacement for the top MLP / phi cache section inside
// `neural_phase_function.cuh`. It keeps the same public surface used by the rest of the file:
//   - using VOut = ...
//   - float phi_f(const VOut&, int)
//   - void evalHyperMLP_phi62_vec(const LaunchParams&, int, float, float, float, VOut&)
//
// Weights are emitted as immediate float literals inside FMA chains, so there is no
// runtime access to a packed weight blob and no dependence on OptiX CoopVec.
#ifndef NSF_T
#error "Include the generated hardcoded header from neural_phase_function.cuh after NSF_T/C/H/K/PHI/PHI_PAD are defined."
#endif
#include <cuda_fp16.h>

namespace nsf_simd_hardcoded_generated {
static constexpr int kTransforms     = 2;
static constexpr int kContext        = 3;
static constexpr int kHidden         = 32;
static constexpr int kBins           = 16;
static constexpr int kPhi            = 31;
static constexpr int kPhiPad         = 32;
static constexpr int kLayer0InputPad = 4;

static_assert(NSF_T   == kTransforms,   "NSF_T mismatch between neural_phase_function.cuh and generated header.");
static_assert(C       == kContext,      "C mismatch between neural_phase_function.cuh and generated header.");
static_assert(H       == kHidden,       "H mismatch between neural_phase_function.cuh and generated header.");
static_assert(K       == kBins,         "K mismatch between neural_phase_function.cuh and generated header.");
static_assert(PHI     == kPhi,          "PHI mismatch between neural_phase_function.cuh and generated header.");
static_assert(PHI_PAD == kPhiPad,       "PHI_PAD mismatch between neural_phase_function.cuh and generated header.");

struct alignas(16) PhiVec {
    half lane[PHI_PAD];
    __device__ __forceinline__ half& operator[](int i) { return lane[i]; }
    __device__ __forceinline__ const half& operator[](int i) const { return lane[i]; }
};

static __forceinline__ __device__ half nsf_relu_to_half(float x)
{
    return __float2half_rn(fmaxf(x, 0.0f));
}

static __forceinline__ __device__ float nsf_dot4_acc(
    float acc,
    float w0, float w1, float w2, float w3,
    float x0, float x1, float x2, float x3)
{
    acc = fmaf(w0, x0, acc);
    acc = fmaf(w1, x1, acc);
    acc = fmaf(w2, x2, acc);
    acc = fmaf(w3, x3, acc);
    return acc;
}

static __forceinline__ __device__ void eval_transform_0(float c0, float c1, float c2, PhiVec& y_out)
{
    const float x0 = c0;
    const float x1 = c1;
    const float x2 = c2;
    const float x3 = 0.0f;
    alignas(16) float h0[32];
    alignas(16) float h1[32];

    {
        float acc = 0.280029297f;
        acc = nsf_dot4_acc(acc, -0.00540924072f, 0.322509766f, -0.485351562f, 0.0f, x0, x1, x2, x3);
        h0[0] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.220947266f;
        acc = nsf_dot4_acc(acc, -0.400146484f, -0.21496582f, 0.257080078f, 0.0f, x0, x1, x2, x3);
        h0[1] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.450439453f;
        acc = nsf_dot4_acc(acc, -0.0114364624f, 0.457763672f, -0.0512390137f, 0.0f, x0, x1, x2, x3);
        h0[2] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.332519531f;
        acc = nsf_dot4_acc(acc, 0.152832031f, -0.174438477f, -0.113464355f, 0.0f, x0, x1, x2, x3);
        h0[3] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.581542969f;
        acc = nsf_dot4_acc(acc, -0.451904297f, -0.470214844f, -0.199951172f, 0.0f, x0, x1, x2, x3);
        h0[4] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.483398438f;
        acc = nsf_dot4_acc(acc, 0.0499572754f, 0.0858154297f, 0.440185547f, 0.0f, x0, x1, x2, x3);
        h0[5] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.210571289f;
        acc = nsf_dot4_acc(acc, -0.297607422f, -0.389892578f, 0.252929688f, 0.0f, x0, x1, x2, x3);
        h0[6] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0452270508f;
        acc = nsf_dot4_acc(acc, 0.411132812f, -0.180786133f, 0.619140625f, 0.0f, x0, x1, x2, x3);
        h0[7] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.55859375f;
        acc = nsf_dot4_acc(acc, -0.083984375f, 0.0520019531f, 0.514160156f, 0.0f, x0, x1, x2, x3);
        h0[8] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0205688477f;
        acc = nsf_dot4_acc(acc, -0.641113281f, -0.278076172f, -0.158813477f, 0.0f, x0, x1, x2, x3);
        h0[9] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.434814453f;
        acc = nsf_dot4_acc(acc, -0.225097656f, 0.498779297f, -0.374267578f, 0.0f, x0, x1, x2, x3);
        h0[10] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.4453125f;
        acc = nsf_dot4_acc(acc, -0.265869141f, -0.403320312f, -0.540527344f, 0.0f, x0, x1, x2, x3);
        h0[11] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0242462158f;
        acc = nsf_dot4_acc(acc, -0.232543945f, 0.413818359f, 0.317626953f, 0.0f, x0, x1, x2, x3);
        h0[12] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.185791016f;
        acc = nsf_dot4_acc(acc, 0.304931641f, 0.0030632019f, -0.197875977f, 0.0f, x0, x1, x2, x3);
        h0[13] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.119750977f;
        acc = nsf_dot4_acc(acc, 0.0953369141f, -0.622558594f, -0.302246094f, 0.0f, x0, x1, x2, x3);
        h0[14] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.43359375f;
        acc = nsf_dot4_acc(acc, -0.329833984f, 0.313964844f, 0.428710938f, 0.0f, x0, x1, x2, x3);
        h0[15] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.361083984f;
        acc = nsf_dot4_acc(acc, -0.361083984f, 0.0927124023f, 0.356933594f, 0.0f, x0, x1, x2, x3);
        h0[16] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.62109375f;
        acc = nsf_dot4_acc(acc, 0.572753906f, 0.0368347168f, 0.176147461f, 0.0f, x0, x1, x2, x3);
        h0[17] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.522460938f;
        acc = nsf_dot4_acc(acc, 0.390380859f, -0.353027344f, 0.231689453f, 0.0f, x0, x1, x2, x3);
        h0[18] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.458740234f;
        acc = nsf_dot4_acc(acc, -0.478027344f, -0.419189453f, -0.269042969f, 0.0f, x0, x1, x2, x3);
        h0[19] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0497131348f;
        acc = nsf_dot4_acc(acc, 0.365234375f, 0.287109375f, -0.246948242f, 0.0f, x0, x1, x2, x3);
        h0[20] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.00854492188f;
        acc = nsf_dot4_acc(acc, 0.155639648f, 0.291259766f, -0.0892944336f, 0.0f, x0, x1, x2, x3);
        h0[21] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.169067383f;
        acc = nsf_dot4_acc(acc, 0.0310974121f, 0.032989502f, 0.444335938f, 0.0f, x0, x1, x2, x3);
        h0[22] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.275634766f;
        acc = nsf_dot4_acc(acc, 0.467773438f, -0.621582031f, -0.0753173828f, 0.0f, x0, x1, x2, x3);
        h0[23] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.314453125f;
        acc = nsf_dot4_acc(acc, 0.296875f, 0.368896484f, 0.590332031f, 0.0f, x0, x1, x2, x3);
        h0[24] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0827026367f;
        acc = nsf_dot4_acc(acc, 0.551757812f, 0.0684204102f, -0.452880859f, 0.0f, x0, x1, x2, x3);
        h0[25] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.281738281f;
        acc = nsf_dot4_acc(acc, 0.0531005859f, -0.361083984f, -0.538085938f, 0.0f, x0, x1, x2, x3);
        h0[26] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.29296875f;
        acc = nsf_dot4_acc(acc, 0.490966797f, 0.378662109f, -0.556640625f, 0.0f, x0, x1, x2, x3);
        h0[27] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.553710938f;
        acc = nsf_dot4_acc(acc, 0.108093262f, -0.0972900391f, -0.0950317383f, 0.0f, x0, x1, x2, x3);
        h0[28] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.342285156f;
        acc = nsf_dot4_acc(acc, -0.264404297f, 0.222045898f, -0.342041016f, 0.0f, x0, x1, x2, x3);
        h0[29] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.101623535f;
        acc = nsf_dot4_acc(acc, 0.239013672f, 0.169677734f, 0.455810547f, 0.0f, x0, x1, x2, x3);
        h0[30] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.132568359f;
        acc = nsf_dot4_acc(acc, 0.264404297f, -0.689453125f, -0.227172852f, 0.0f, x0, x1, x2, x3);
        h0[31] = __half2float(nsf_relu_to_half(acc));
    }

    {
        float acc = -0.14855957f;
        acc = nsf_dot4_acc(acc, -0.0618286133f, -0.144897461f, -0.0375976562f, 0.0377807617f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.11517334f, -0.0090713501f, 0.126586914f, -0.0181732178f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.00491333008f, -0.0152511597f, 0.0357666016f, 0.112426758f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.167480469f, 0.112243652f, 0.16784668f, -0.0127868652f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.158813477f, -0.0838012695f, 0.120361328f, -0.00114631653f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.087890625f, -0.135498047f, -0.165405273f, -0.149169922f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.0358581543f, 0.0969238281f, 0.0955810547f, -0.170532227f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.110290527f, -0.138305664f, -0.0373840332f, -0.0716552734f, h0[28], h0[29], h0[30], h0[31]);
        h1[0] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0538635254f;
        acc = nsf_dot4_acc(acc, -0.0737304688f, -0.00101184845f, -0.158691406f, -0.152587891f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0183258057f, 0.0513916016f, -0.0474243164f, 0.131347656f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.162841797f, -0.0357055664f, 0.155517578f, -0.0720825195f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.114257812f, 0.045135498f, -0.153686523f, 0.137329102f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.145874023f, -0.0154037476f, 0.214355469f, -0.201904297f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0188903809f, -0.00148200989f, 0.17565918f, 0.0870361328f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.128173828f, -0.0108184814f, -0.0425109863f, -0.00359916687f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.145629883f, 0.0957641602f, 0.177124023f, 0.0946044922f, h0[28], h0[29], h0[30], h0[31]);
        h1[1] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.134765625f;
        acc = nsf_dot4_acc(acc, -0.0536499023f, 0.0245361328f, -0.124633789f, -0.0972900391f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.0944213867f, 0.0530090332f, -0.0863647461f, -0.0693969727f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0124206543f, 0.198974609f, -0.133666992f, -0.121337891f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.050994873f, 0.188598633f, -0.0729370117f, 0.167236328f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.168945312f, -0.0533447266f, -0.0671386719f, -0.0899658203f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0697631836f, -0.00328445435f, -0.0101242065f, -0.0383605957f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.150024414f, -0.0948486328f, 0.142700195f, -0.112915039f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0126953125f, 0.174926758f, 0.0334777832f, 0.0270080566f, h0[28], h0[29], h0[30], h0[31]);
        h1[2] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0134963989f;
        acc = nsf_dot4_acc(acc, -0.0550231934f, 0.253417969f, -0.125854492f, -0.0501708984f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.110656738f, 0.130493164f, 0.123840332f, 0.282226562f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0244293213f, 0.17956543f, 0.108154297f, -0.111755371f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.135498047f, -0.136352539f, -0.0383300781f, 0.209350586f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.0395202637f, 0.256103516f, 0.0919189453f, -0.0365905762f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.0248565674f, -0.130493164f, 0.28515625f, -0.0211029053f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.134521484f, -0.133911133f, -0.0930786133f, 0.031829834f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.146240234f, -0.0516662598f, -0.00634384155f, -0.0497741699f, h0[28], h0[29], h0[30], h0[31]);
        h1[3] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.00915527344f;
        acc = nsf_dot4_acc(acc, -0.176269531f, 0.119384766f, -0.0848999023f, -0.117980957f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.0375061035f, 0.189697266f, 0.119628906f, 0.267822266f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.065612793f, -0.110168457f, -0.049407959f, 0.0522155762f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.0974731445f, 0.115905762f, 0.00122737885f, -0.0434570312f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.331542969f, 0.162963867f, 0.0430603027f, -0.159545898f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.105651855f, -0.0787963867f, 0.0689697266f, 0.163574219f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0867919922f, -0.132324219f, -0.142944336f, 0.171508789f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.028793335f, 0.146606445f, 0.0284423828f, 0.146850586f, h0[28], h0[29], h0[30], h0[31]);
        h1[4] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.213012695f;
        acc = nsf_dot4_acc(acc, -0.0192108154f, 0.0249328613f, -0.107666016f, 0.0597839355f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.100280762f, 0.0536804199f, -0.0347595215f, -0.0371398926f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.116394043f, -0.134277344f, 0.0724487305f, -0.0594482422f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.0955200195f, 0.105529785f, 0.0476379395f, 0.0771484375f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.00649642944f, -0.063659668f, 0.104370117f, 0.00771331787f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0230865479f, 0.0024394989f, -0.0872192383f, 0.0678710938f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.114257812f, -0.0325012207f, 0.0670776367f, 0.132446289f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0922851562f, 0.00174331665f, 0.119873047f, 0.16027832f, h0[28], h0[29], h0[30], h0[31]);
        h1[5] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.029876709f;
        acc = nsf_dot4_acc(acc, 0.049621582f, 0.0546569824f, -0.138916016f, 0.0138931274f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.168579102f, 0.217407227f, 0.117553711f, 0.0918579102f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0764770508f, -0.124572754f, -0.146484375f, -0.151977539f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.159912109f, 0.126953125f, -0.0158081055f, 0.109008789f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.0281982422f, 0.241210938f, 0.226196289f, 0.0543212891f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.10333252f, -0.0227508545f, 0.202880859f, 0.173095703f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.0536193848f, 0.0841064453f, 0.0853881836f, 0.107177734f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.110778809f, 0.0507202148f, -0.0175476074f, 0.214477539f, h0[28], h0[29], h0[30], h0[31]);
        h1[6] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.158203125f;
        acc = nsf_dot4_acc(acc, 0.0169525146f, 0.186401367f, 0.0314025879f, 0.0751953125f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0651855469f, 0.158935547f, -0.00975036621f, 0.0844726562f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.114135742f, -0.122314453f, -0.141967773f, 0.139526367f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.108825684f, 0.100341797f, 0.0971679688f, -0.0771484375f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.120544434f, 0.00601196289f, -0.0509643555f, 0.075378418f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.28125f, -0.0911865234f, -0.0991821289f, -0.0693359375f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.192382812f, -0.0373840332f, 0.126586914f, 0.0911254883f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.157226562f, -0.0452575684f, 0.0961914062f, 0.0866699219f, h0[28], h0[29], h0[30], h0[31]);
        h1[7] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.135498047f;
        acc = nsf_dot4_acc(acc, 0.0427856445f, 0.189453125f, 0.0916748047f, 0.109863281f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.0446777344f, 0.123596191f, 0.01953125f, 0.0225372314f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.0948486328f, -0.0971069336f, -0.135986328f, 0.118652344f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.149902344f, 0.0466918945f, -0.0734863281f, 0.167480469f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.122314453f, 0.0647583008f, -0.0281982422f, 0.150268555f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0580749512f, -0.148071289f, 0.206665039f, 0.192504883f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.021194458f, 0.108581543f, -0.100463867f, -0.00548171997f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.150146484f, 0.00730514526f, -0.0734863281f, -0.00824737549f, h0[28], h0[29], h0[30], h0[31]);
        h1[8] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0556030273f;
        acc = nsf_dot4_acc(acc, -0.0479736328f, -0.0341186523f, 0.0169219971f, 0.163452148f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.00946807861f, -0.109130859f, 0.0090637207f, 0.084777832f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0877075195f, -0.161621094f, -0.0316467285f, -0.131347656f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.0754394531f, 0.0637207031f, -0.125488281f, 0.0657348633f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.150024414f, 0.0115966797f, -0.117797852f, -0.0633544922f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0386047363f, -0.134765625f, 0.0878295898f, -0.160522461f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.169921875f, -0.17175293f, -0.0358581543f, 0.118896484f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.167358398f, 0.146972656f, -0.0707397461f, 0.0517883301f, h0[28], h0[29], h0[30], h0[31]);
        h1[9] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0438537598f;
        acc = nsf_dot4_acc(acc, 0.0455932617f, -0.0259094238f, 0.146606445f, 0.0951538086f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.317626953f, 0.218139648f, -0.0302429199f, 0.247192383f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.00846099854f, 0.0716552734f, -0.141723633f, -0.0508422852f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.148193359f, -0.133422852f, 0.000633239746f, -0.0135879517f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0422058105f, 0.194458008f, 0.141479492f, -0.252197266f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.182250977f, -0.0204315186f, 0.207397461f, 0.218139648f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.210083008f, -0.151733398f, 0.0619812012f, -0.150268555f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.151000977f, 0.0824584961f, -0.0299224854f, -0.00402832031f, h0[28], h0[29], h0[30], h0[31]);
        h1[10] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.208007812f;
        acc = nsf_dot4_acc(acc, -0.151855469f, -0.0403137207f, -0.145629883f, 0.0922241211f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0356445312f, 0.196655273f, 0.140625f, 0.185180664f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.011932373f, -0.0733032227f, -0.0384216309f, 0.0234985352f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.0950317383f, -0.0841064453f, 0.109924316f, 0.0308685303f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.225952148f, -0.0263824463f, 0.237182617f, 0.113830566f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.179077148f, 0.114562988f, 0.149047852f, 0.14453125f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.207397461f, -0.118591309f, 0.0558166504f, 0.0966796875f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0522460938f, 0.059173584f, 0.00750732422f, 0.0972900391f, h0[28], h0[29], h0[30], h0[31]);
        h1[11] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.154296875f;
        acc = nsf_dot4_acc(acc, -0.032989502f, -0.0039024353f, 0.0444946289f, 0.0626831055f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.014793396f, 0.0968017578f, 0.134155273f, 0.0623474121f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0449829102f, -0.0681762695f, 0.0619506836f, -0.128051758f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.0953369141f, -0.0550537109f, 0.23840332f, 0.018081665f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0756225586f, -0.0012216568f, 0.263916016f, 0.100830078f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.184814453f, -0.0966796875f, -0.0517578125f, 0.0748901367f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0309753418f, -0.123779297f, -0.0634155273f, 0.0667724609f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0923461914f, 0.0391845703f, 0.024887085f, -0.0208435059f, h0[28], h0[29], h0[30], h0[31]);
        h1[12] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0794677734f;
        acc = nsf_dot4_acc(acc, 0.0283355713f, 0.273925781f, -0.119506836f, 0.00820922852f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0582275391f, 0.210327148f, -0.0146942139f, -0.197753906f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.168579102f, 0.0807495117f, 0.0339050293f, -0.0152587891f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.0420227051f, -0.047454834f, 0.0775146484f, 0.0756225586f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.0286560059f, -0.14855957f, -0.0871582031f, 0.193725586f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.108276367f, 0.133178711f, -0.0011920929f, -0.00764083862f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.150146484f, 0.100646973f, 0.149902344f, 0.0740966797f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0536193848f, -0.11730957f, -0.02734375f, 0.113708496f, h0[28], h0[29], h0[30], h0[31]);
        h1[13] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0625610352f;
        acc = nsf_dot4_acc(acc, -0.137695312f, -0.0933837891f, 0.0724487305f, 0.172119141f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0202484131f, -0.116455078f, -0.113464355f, 0.0781860352f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.013381958f, -0.0531311035f, 0.035736084f, 0.0671386719f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.127441406f, 0.165527344f, 0.0194244385f, -0.131225586f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0601806641f, 0.184814453f, 0.116516113f, 0.11505127f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.18359375f, 0.0999755859f, 0.196899414f, 0.202148438f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.131225586f, -0.0782470703f, -0.121582031f, 0.148071289f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0759277344f, -0.0806884766f, 0.104736328f, -0.0299377441f, h0[28], h0[29], h0[30], h0[31]);
        h1[14] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0277862549f;
        acc = nsf_dot4_acc(acc, 0.0202331543f, -0.0208587646f, -0.016998291f, -0.137695312f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.151855469f, -0.046875f, 0.0472717285f, 0.209472656f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0891723633f, -0.00282859802f, 0.150390625f, 0.0237579346f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.0986328125f, 0.166870117f, 0.051940918f, -0.160888672f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.120422363f, -0.0662841797f, 0.208862305f, -0.0609436035f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.178833008f, 0.0539550781f, 0.141479492f, 0.0950927734f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.094543457f, 0.00513458252f, 0.0659179688f, -0.0189361572f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0703125f, -0.165283203f, -0.0935058594f, 0.0868530273f, h0[28], h0[29], h0[30], h0[31]);
        h1[15] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0955200195f;
        acc = nsf_dot4_acc(acc, -0.200195312f, 0.0208892822f, 0.056427002f, 0.0125198364f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.109741211f, 0.145874023f, 0.0141372681f, 0.144165039f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.145751953f, -0.0773925781f, 0.0961914062f, -0.0722045898f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.0870361328f, -0.130493164f, -0.134521484f, -0.116149902f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0802612305f, 0.150390625f, -0.0856933594f, 0.0632324219f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.00568771362f, -0.058807373f, 0.10760498f, -0.11730957f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.0054397583f, 0.158081055f, -0.0242919922f, 0.13293457f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0544433594f, 0.142333984f, -0.19128418f, -0.145874023f, h0[28], h0[29], h0[30], h0[31]);
        h1[16] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0263214111f;
        acc = nsf_dot4_acc(acc, -0.0261688232f, 0.130493164f, 0.0693359375f, 0.135742188f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.046875f, 0.120910645f, 0.231567383f, 0.0806884766f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.155029297f, -0.118225098f, -0.0191345215f, -0.00949859619f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.132324219f, 0.0830078125f, -0.0205993652f, -0.0294342041f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.033416748f, 0.168823242f, 0.205200195f, 0.0135879517f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0817260742f, 0.226318359f, 0.14440918f, -0.0866699219f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.238891602f, -0.228271484f, 0.0406494141f, 0.0527954102f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0827636719f, 0.0615539551f, -0.0545654297f, -0.0320739746f, h0[28], h0[29], h0[30], h0[31]);
        h1[17] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0336303711f;
        acc = nsf_dot4_acc(acc, -0.0758666992f, 0.00425338745f, 0.0227355957f, 0.150634766f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.0639648438f, 0.140258789f, -0.00973510742f, -0.0414428711f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.151855469f, -0.122314453f, 0.172607422f, -0.0440979004f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.065246582f, 0.10559082f, 0.141235352f, 0.190307617f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.228027344f, -0.118530273f, -0.00955200195f, -0.0908203125f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.118286133f, 0.0269622803f, 0.191894531f, -0.0775756836f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.134887695f, -0.102172852f, -0.0344238281f, 0.15222168f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0399780273f, 0.0160980225f, -0.0991210938f, 0.186889648f, h0[28], h0[29], h0[30], h0[31]);
        h1[18] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.127929688f;
        acc = nsf_dot4_acc(acc, -0.0318908691f, -0.110656738f, 0.0517578125f, -0.125732422f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.12121582f, -0.00314903259f, 0.168945312f, -0.0951538086f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.080078125f, 0.195068359f, 0.117248535f, -0.0927124023f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.0201568604f, -0.0681152344f, -0.185668945f, -0.00826263428f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.159301758f, 0.126342773f, -0.0137252808f, 0.0555725098f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.0371704102f, 0.12512207f, -0.0860595703f, -0.129516602f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.00830078125f, 0.150390625f, -0.159423828f, 0.162963867f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0635986328f, -0.123840332f, -0.0646972656f, 0.118652344f, h0[28], h0[29], h0[30], h0[31]);
        h1[19] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.126708984f;
        acc = nsf_dot4_acc(acc, -0.135620117f, -0.0516662598f, 0.0579833984f, -0.154907227f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0969238281f, 0.0920410156f, 0.109619141f, -0.112731934f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.176025391f, -0.104797363f, 0.176513672f, -0.169677734f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.157470703f, 0.108581543f, 0.0184783936f, 0.0102005005f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0979003906f, -0.0741577148f, -0.0516967773f, -0.172241211f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.00918579102f, 0.03125f, -0.000149726868f, 0.0570983887f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.167724609f, 0.0469970703f, -0.0646972656f, -0.0727539062f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.113098145f, -0.122558594f, -0.0284729004f, -0.0312805176f, h0[28], h0[29], h0[30], h0[31]);
        h1[20] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.112487793f;
        acc = nsf_dot4_acc(acc, 0.0779418945f, -0.0137634277f, 0.140869141f, -0.124023438f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0559692383f, 0.16796875f, 0.241455078f, -0.121704102f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0308837891f, 0.111633301f, -0.0828857422f, -0.0610656738f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.0966796875f, -0.118041992f, -0.00719833374f, 0.109924316f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.063293457f, 0.111999512f, -0.0329284668f, 0.110412598f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.170532227f, -0.101013184f, 0.105407715f, -0.0420837402f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.0306549072f, -0.00730514526f, -0.00109863281f, -0.0846557617f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0414428711f, 0.00666046143f, 0.0897827148f, -0.0563049316f, h0[28], h0[29], h0[30], h0[31]);
        h1[21] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0316162109f;
        acc = nsf_dot4_acc(acc, -0.0318603516f, -0.0638427734f, -0.148071289f, 0.0715332031f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0437927246f, -0.0487976074f, -0.0699462891f, 0.128295898f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.0267181396f, 0.0596008301f, 0.0100097656f, 0.0667724609f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.0107727051f, 0.0106811523f, 0.133178711f, 0.120178223f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.128173828f, 0.0905151367f, 0.0826416016f, 0.0877685547f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.14465332f, -0.125488281f, 0.0983276367f, -0.0740356445f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0207672119f, -0.071472168f, 0.116699219f, -0.157714844f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.107177734f, 0.174926758f, 0.0595703125f, 0.0671386719f, h0[28], h0[29], h0[30], h0[31]);
        h1[22] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0438232422f;
        acc = nsf_dot4_acc(acc, 0.0928955078f, -0.0654296875f, -0.0141677856f, -0.0773925781f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.00896453857f, 0.0756225586f, 0.0327148438f, 0.0672607422f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.174560547f, -0.0450744629f, -0.00523757935f, 0.0265960693f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.0684814453f, 0.203491211f, -0.0203552246f, -0.0149383545f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.216674805f, -0.0865478516f, -0.0219268799f, 0.0677490234f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0277862549f, 0.0788574219f, 0.136474609f, -0.0150299072f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0474243164f, 0.0916748047f, -0.0986328125f, 0.158203125f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0464782715f, 0.124389648f, -0.0501708984f, 0.165893555f, h0[28], h0[29], h0[30], h0[31]);
        h1[23] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.00364685059f;
        acc = nsf_dot4_acc(acc, -0.0970458984f, -0.114624023f, 0.0589599609f, 0.0530090332f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0589904785f, 0.133789062f, 0.0534667969f, 0.121337891f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.122009277f, -0.142456055f, -0.137207031f, -0.065612793f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.0889892578f, -0.0694580078f, 0.10357666f, -0.00735473633f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0598449707f, 0.0520324707f, 0.151245117f, -0.0263824463f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.171386719f, -0.0888061523f, 0.036315918f, 0.0751953125f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0665893555f, -0.156982422f, -0.163085938f, 0.111633301f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.113098145f, -0.159912109f, 0.0680541992f, -0.101928711f, h0[28], h0[29], h0[30], h0[31]);
        h1[24] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.00313949585f;
        acc = nsf_dot4_acc(acc, 0.131713867f, 0.053527832f, 0.174804688f, -0.174316406f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.170166016f, 0.117370605f, 0.156982422f, 0.106201172f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.117553711f, -0.148925781f, 0.127929688f, -0.166870117f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.037322998f, -0.115783691f, -0.0701904297f, 0.0291290283f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.069519043f, -0.0301361084f, -0.169677734f, -0.166625977f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0684814453f, -0.076965332f, -0.109680176f, -0.0697631836f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.146728516f, -0.141357422f, 0.105773926f, 0.0877075195f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0235443115f, -0.0965576172f, 0.056640625f, 0.00794219971f, h0[28], h0[29], h0[30], h0[31]);
        h1[25] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0123519897f;
        acc = nsf_dot4_acc(acc, -0.176025391f, 0.0520935059f, 0.161010742f, -0.173828125f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.144897461f, 0.0759277344f, 0.120300293f, -0.0233612061f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0333557129f, 0.0684814453f, 0.0545043945f, 0.038269043f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.13293457f, 0.155273438f, -0.0518493652f, 0.0352783203f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0145339966f, 0.231811523f, -0.0814208984f, 0.0426025391f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.0484008789f, -0.158569336f, 0.0934448242f, -0.00294685364f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0726928711f, -0.105895996f, -0.133056641f, 0.0157623291f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.059387207f, 0.103393555f, 0.0601806641f, 0.179321289f, h0[28], h0[29], h0[30], h0[31]);
        h1[26] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.110351562f;
        acc = nsf_dot4_acc(acc, 0.0331420898f, -0.0599365234f, 0.119750977f, 0.00965881348f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0300445557f, -0.0599975586f, -0.00231552124f, -0.0371398926f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.0150299072f, -0.137817383f, -0.00108718872f, -0.146728516f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.0380249023f, 0.0993041992f, -0.126098633f, -0.122253418f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0431518555f, -0.0516357422f, 0.026763916f, 0.0293426514f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0896606445f, -0.121765137f, -0.0156173706f, 0.095703125f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.149780273f, -0.0159301758f, -0.102294922f, 0.0508117676f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.100646973f, -0.110412598f, -0.105712891f, 0.075012207f, h0[28], h0[29], h0[30], h0[31]);
        h1[27] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0532531738f;
        acc = nsf_dot4_acc(acc, -0.0146560669f, -0.208007812f, -0.101928711f, 0.0717163086f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.0985717773f, 0.102111816f, 0.11932373f, 0.00253295898f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.0139770508f, -0.0603942871f, -0.00894165039f, 0.0321655273f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.127807617f, -0.0618591309f, -0.166625977f, 0.0916748047f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.114196777f, 0.141845703f, -0.0697631836f, -0.101257324f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.102416992f, 0.0321960449f, 0.102722168f, 0.0458374023f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.187866211f, 0.115112305f, -0.0332946777f, -0.0784912109f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.112243652f, 0.12890625f, -0.171875f, -0.0456542969f, h0[28], h0[29], h0[30], h0[31]);
        h1[28] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0664672852f;
        acc = nsf_dot4_acc(acc, 0.151733398f, 0.0589904785f, 0.148925781f, -0.0605773926f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.0931396484f, 0.153930664f, 0.065612793f, 0.202270508f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.0720214844f, -0.160766602f, -0.0207214355f, -0.166503906f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.00303840637f, 0.120788574f, -0.0018491745f, -0.0397644043f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.0105056763f, -0.0848388672f, 0.211425781f, -0.285400391f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.186035156f, -0.0227508545f, 0.146118164f, 0.0803833008f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0703735352f, -0.11505127f, 0.064453125f, -0.00556564331f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0708618164f, 0.124816895f, 0.078918457f, -0.0435791016f, h0[28], h0[29], h0[30], h0[31]);
        h1[29] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.248779297f;
        acc = nsf_dot4_acc(acc, -0.037109375f, -0.00692367554f, -0.0639648438f, 0.114868164f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0444030762f, 0.210205078f, 0.0129165649f, 0.0562133789f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.00484085083f, -0.15246582f, -0.00176525116f, -0.0178222656f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.013671875f, -0.038269043f, 0.131347656f, 0.216186523f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.21484375f, 0.173461914f, -0.0262756348f, -0.222045898f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0116043091f, 0.0260925293f, 0.0193634033f, -0.10546875f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0143280029f, -0.000477790833f, -0.163574219f, -0.145263672f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0224304199f, -0.127319336f, -0.0960083008f, -0.0917358398f, h0[28], h0[29], h0[30], h0[31]);
        h1[30] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.172119141f;
        acc = nsf_dot4_acc(acc, -0.0284576416f, -0.160644531f, -0.118041992f, 0.0115890503f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.166748047f, -0.0248413086f, -0.114135742f, -0.122558594f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0694580078f, -0.136108398f, -0.133911133f, 0.100830078f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.0330505371f, 0.161010742f, -0.0472412109f, -0.0853881836f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0163574219f, 0.000894546509f, 0.134765625f, -0.176025391f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.012588501f, 0.0781860352f, -0.0973510742f, -0.08984375f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0854492188f, -0.169433594f, 0.0419006348f, -0.13684082f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.087890625f, -0.151123047f, 0.108703613f, -0.167358398f, h0[28], h0[29], h0[30], h0[31]);
        h1[31] = __half2float(nsf_relu_to_half(acc));
    }

    {
        float acc = 0.108337402f;
        acc = nsf_dot4_acc(acc, -0.112731934f, -0.130615234f, -0.0306854248f, 0.168701172f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.174438477f, -0.140869141f, 0.00252342224f, 0.108459473f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.114929199f, 0.144775391f, -0.0239715576f, -0.117248535f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.107971191f, -0.0396118164f, 0.112426758f, -0.038269043f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0267791748f, 0.108154297f, 0.188964844f, -0.177001953f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.116943359f, -0.0804443359f, -0.0946655273f, 0.216186523f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.0107116699f, -0.117248535f, -0.0439453125f, -0.09765625f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.0833129883f, -0.143798828f, -0.155273438f, -0.174316406f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[0] = __float2half_rn(acc);
    }
    {
        float acc = -0.0312347412f;
        acc = nsf_dot4_acc(acc, 0.0761108398f, 0.0974121094f, -0.113647461f, 0.0706787109f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.146606445f, 0.0353393555f, -0.0304870605f, 0.0143966675f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0146484375f, 0.0889892578f, -0.140625f, 0.087890625f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0882568359f, -0.0905761719f, 0.16418457f, 0.176391602f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.11932373f, -0.101867676f, 0.109191895f, -0.156494141f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0698852539f, 0.0293884277f, 0.126342773f, -0.109741211f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.14465332f, 0.0794067383f, 0.0955200195f, -0.164550781f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.0872192383f, -0.154296875f, -0.120849609f, -0.0797729492f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[1] = __float2half_rn(acc);
    }
    {
        float acc = -0.0740356445f;
        acc = nsf_dot4_acc(acc, 0.146240234f, -0.0251312256f, 0.101379395f, 0.1953125f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0957641602f, -0.147583008f, 0.0967407227f, 0.0682373047f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.171508789f, -0.174438477f, 0.0888061523f, 0.143554688f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.175170898f, 0.0359802246f, 0.169921875f, -0.123046875f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.00701522827f, 0.024597168f, 0.0667114258f, 0.155883789f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.144775391f, 0.140014648f, -0.0068397522f, 0.190185547f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.161987305f, -0.0953979492f, -0.00769424438f, -0.0524902344f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.173706055f, 0.0911254883f, 0.119628906f, 0.120239258f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[2] = __float2half_rn(acc);
    }
    {
        float acc = -0.0422973633f;
        acc = nsf_dot4_acc(acc, -0.132080078f, -0.0220489502f, 0.0107421875f, 0.0606079102f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0962524414f, -0.0718994141f, -0.0573730469f, 0.137329102f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.0262298584f, -0.122802734f, 0.0893554688f, 0.106811523f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0762939453f, -0.0323791504f, -0.118041992f, -0.150878906f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0123825073f, -0.137817383f, 0.147827148f, -0.167236328f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.0244293213f, -0.0209960938f, 0.0943603516f, 0.174682617f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.111999512f, 0.0778198242f, -0.0623779297f, -0.126220703f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.14831543f, 0.00569534302f, 0.164794922f, -0.0335693359f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[3] = __float2half_rn(acc);
    }
    {
        float acc = 0.0463256836f;
        acc = nsf_dot4_acc(acc, -0.00744247437f, -0.138183594f, 0.0843505859f, 0.119628906f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.187011719f, -0.136474609f, 0.116577148f, 0.0256958008f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0148849487f, 0.0315856934f, -0.0763549805f, -0.148071289f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.209838867f, 0.151000977f, -0.021270752f, -0.0955810547f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.105957031f, -0.159790039f, -0.0259857178f, -0.153686523f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.132324219f, 0.0661621094f, -0.0980224609f, -0.0276184082f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0500183105f, -0.0654296875f, 0.065246582f, 0.0486755371f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.104858398f, 0.0254058838f, -0.00207519531f, 0.154418945f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[4] = __float2half_rn(acc);
    }
    {
        float acc = -0.0639038086f;
        acc = nsf_dot4_acc(acc, -0.0639648438f, -0.0693359375f, 0.130249023f, -0.157592773f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.00105571747f, 0.145141602f, -0.10369873f, 0.139892578f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.155395508f, 0.164428711f, -0.0595092773f, 0.124755859f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.111755371f, -0.0940551758f, 0.0868530273f, 0.0278015137f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0441589355f, 0.109985352f, -0.1328125f, 0.0556335449f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0591430664f, 0.0309448242f, 0.0617370605f, -0.0590515137f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.171142578f, 0.141479492f, -0.105834961f, 0.113037109f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.152709961f, -0.0549621582f, -0.0339355469f, 0.0305938721f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[5] = __float2half_rn(acc);
    }
    {
        float acc = 0.0952148438f;
        acc = nsf_dot4_acc(acc, 0.014289856f, 0.0836791992f, -0.0849609375f, 0.0708618164f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.22644043f, -0.0938720703f, 0.0116958618f, -0.0346984863f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.189575195f, 0.151367188f, 0.00827026367f, -0.0467834473f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0925292969f, 0.0881347656f, 0.0709838867f, 0.196166992f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.077331543f, 0.161132812f, 0.155395508f, -0.108276367f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0517578125f, -0.12121582f, 0.114013672f, 0.0401611328f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0292816162f, -0.122253418f, -0.110717773f, 0.0665893555f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.0618591309f, 0.0451660156f, 0.134887695f, -0.0737915039f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[6] = __float2half_rn(acc);
    }
    {
        float acc = -0.161987305f;
        acc = nsf_dot4_acc(acc, 0.0627441406f, -0.0837402344f, 0.0473632812f, -0.0672607422f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.132080078f, 0.146850586f, 0.162231445f, 0.0246582031f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.113464355f, -0.176635742f, -0.0963134766f, 0.0357055664f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0127639771f, -0.121337891f, 0.0402832031f, -0.071105957f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.17175293f, 0.107116699f, 0.126098633f, 0.0813598633f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.083984375f, -0.187133789f, -0.145751953f, 0.0265808105f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.111206055f, -0.0715942383f, -0.067199707f, -0.0942993164f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.00961303711f, -0.155517578f, -0.177124023f, 0.174072266f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[7] = __float2half_rn(acc);
    }
    {
        float acc = -0.0916748047f;
        acc = nsf_dot4_acc(acc, -0.0842895508f, -0.0179290771f, -0.0244750977f, -0.127563477f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0228729248f, 0.0626831055f, -0.0655517578f, 0.0215759277f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0525817871f, -0.176025391f, 0.114929199f, 0.140991211f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0975341797f, -0.00109386444f, 0.102600098f, -0.17980957f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.09765625f, -0.0616149902f, -0.00735092163f, 0.0338134766f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.0805664062f, -0.0133743286f, -0.0344848633f, -0.106567383f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.0395812988f, -0.149780273f, 0.144165039f, 0.0944824219f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.163330078f, -0.0529174805f, 0.0258789062f, 0.0346374512f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[8] = __float2half_rn(acc);
    }
    {
        float acc = 0.130615234f;
        acc = nsf_dot4_acc(acc, 0.176513672f, 0.156616211f, -0.0545349121f, -0.137207031f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.167114258f, -0.0896606445f, 0.138793945f, 0.0724487305f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.190307617f, 0.130981445f, -0.00295829773f, -0.118896484f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.157226562f, 0.102111816f, 0.068359375f, -0.115112305f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0715332031f, 0.0212860107f, 0.145507812f, 0.0455627441f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0938720703f, -0.148681641f, -0.0974121094f, 0.108764648f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.0444030762f, -0.153808594f, 0.162109375f, -0.00378417969f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.0804443359f, -0.142944336f, 0.0664672852f, -0.0665893555f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[9] = __float2half_rn(acc);
    }
    {
        float acc = -0.00500488281f;
        acc = nsf_dot4_acc(acc, -0.0744018555f, -0.0674438477f, -0.0116500854f, 0.0549316406f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0864257812f, 0.0982666016f, 0.0185699463f, -0.0670166016f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.0849609375f, -0.161865234f, -0.103210449f, -0.190917969f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0942382812f, 0.128540039f, 0.0681762695f, -0.123291016f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0844116211f, 0.173706055f, -0.0218505859f, -0.0980224609f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.11114502f, 0.155517578f, -0.187744141f, -0.0433349609f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.0588684082f, -0.0640258789f, 0.144165039f, -0.0261230469f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.137451172f, -0.0358886719f, -0.0726318359f, -0.103088379f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[10] = __float2half_rn(acc);
    }
    {
        float acc = 0.0170898438f;
        acc = nsf_dot4_acc(acc, -0.0587768555f, 0.0079498291f, 0.0090713501f, -0.0725708008f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0678100586f, -0.178833008f, 0.0135421753f, 0.0666503906f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.0875244141f, 0.140380859f, -0.105895996f, -0.176147461f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.199951172f, 0.222290039f, -0.176025391f, -0.0687866211f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.240112305f, 0.166625977f, -0.0125045776f, 0.0708007812f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.101867676f, -0.0780029297f, -0.183959961f, -0.152954102f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.00595092773f, 0.0344238281f, -0.105834961f, -0.172973633f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.047088623f, -0.152587891f, -0.0812988281f, -0.148193359f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[11] = __float2half_rn(acc);
    }
    {
        float acc = 0.00103282928f;
        acc = nsf_dot4_acc(acc, -0.0997314453f, 0.0221252441f, -0.284912109f, -0.139648438f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0983276367f, -0.122375488f, -0.117919922f, -0.092590332f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0283966064f, -0.115966797f, -0.0200500488f, 0.0238494873f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0444946289f, -0.0723266602f, -0.126464844f, 0.0405273438f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0587768555f, 0.05078125f, -0.177978516f, -0.00420761108f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.115356445f, -0.0672607422f, 0.0516357422f, 0.0609741211f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.100952148f, -0.0122756958f, 0.157714844f, 0.000822067261f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.193115234f, -0.0653686523f, 0.0523986816f, -0.0577087402f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[12] = __float2half_rn(acc);
    }
    {
        float acc = -0.0708618164f;
        acc = nsf_dot4_acc(acc, -0.092590332f, 0.124267578f, 0.0538024902f, -0.071472168f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0051574707f, -0.0379943848f, 0.02293396f, -0.0269622803f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.173217773f, 0.0327453613f, -0.0120849609f, -0.00255393982f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.067199707f, -0.0471496582f, -0.150390625f, 0.0513916016f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.114318848f, -0.0102539062f, -0.192626953f, 0.00325965881f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.000250816345f, -0.203735352f, -0.142822266f, 0.119018555f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.120666504f, 0.0979614258f, 0.0840454102f, 0.0680541992f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.113647461f, 0.104614258f, -0.0731811523f, -0.0224761963f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[13] = __float2half_rn(acc);
    }
    {
        float acc = -0.0829467773f;
        acc = nsf_dot4_acc(acc, 0.126098633f, 0.00332069397f, -0.0729370117f, -0.130859375f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.135986328f, 0.149414062f, 0.14440918f, 0.051574707f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0960083008f, -0.162353516f, -0.0879516602f, -0.162719727f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.184448242f, -0.0814819336f, -0.0534057617f, 0.100341797f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.0663452148f, -0.147583008f, 0.010635376f, 0.0409545898f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.126586914f, -0.166992188f, 0.0868530273f, 0.0876464844f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0121612549f, -0.0888061523f, 0.0606994629f, 0.0308074951f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.108459473f, -0.10534668f, 0.0326538086f, 0.109375f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[14] = __float2half_rn(acc);
    }
    {
        float acc = 0.156860352f;
        acc = nsf_dot4_acc(acc, -0.0567932129f, 0.184448242f, -0.07421875f, 0.320068359f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.250732422f, 0.147338867f, 0.17980957f, 0.184814453f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.00587463379f, -0.0015411377f, 0.181030273f, 0.0666503906f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.309082031f, -0.0715332031f, -0.11895752f, 0.0211181641f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.104980469f, 0.164550781f, -0.0227966309f, -0.0814819336f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.0191040039f, -0.0409851074f, 0.0214080811f, 0.15234375f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.028427124f, 0.0833740234f, 0.208740234f, -0.000424146652f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.10357666f, 0.00263595581f, 0.270263672f, 0.17175293f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[15] = __float2half_rn(acc);
    }
    {
        float acc = 0.0335083008f;
        acc = nsf_dot4_acc(acc, 0.0420227051f, 0.233276367f, -0.121826172f, 0.0616760254f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0938720703f, -0.0816650391f, 0.158203125f, 0.00531768799f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0136947632f, -0.140258789f, 0.139404297f, -0.0562438965f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0252227783f, 0.162353516f, 0.155883789f, 0.0726318359f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.0285491943f, -0.0166015625f, 0.173339844f, 0.0814208984f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.148803711f, 0.151123047f, 0.212158203f, -0.0770874023f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.206054688f, -0.081237793f, 0.0246734619f, 0.00105762482f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.0293273926f, 0.0979614258f, 0.235351562f, -0.105651855f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[16] = __float2half_rn(acc);
    }
    {
        float acc = -0.0809326172f;
        acc = nsf_dot4_acc(acc, 0.0546264648f, -0.151000977f, 0.0273590088f, 0.138793945f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.111694336f, 0.139892578f, 0.0869750977f, -0.0223083496f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0296478271f, -0.0738525391f, 0.150146484f, 0.0798950195f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.131958008f, 0.130004883f, -0.0016784668f, -0.00393676758f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.0625610352f, -0.0442504883f, -0.011177063f, -0.119689941f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.164428711f, -0.107727051f, -0.0403747559f, 0.110046387f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.122192383f, -0.105895996f, -0.0161437988f, -0.0510253906f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.0989990234f, -0.0386962891f, -0.0714111328f, 0.098815918f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[17] = __float2half_rn(acc);
    }
    {
        float acc = 0.0407409668f;
        acc = nsf_dot4_acc(acc, -0.0876464844f, 0.0258331299f, -0.0163879395f, -0.0375366211f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.132202148f, -0.0792236328f, 0.00264167786f, 0.158081055f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0726928711f, -0.0899658203f, -0.09375f, -0.0551452637f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0409240723f, 0.125366211f, -0.0521240234f, 0.17578125f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.159667969f, -0.0887451172f, 0.0570983887f, 0.11517334f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.169067383f, 0.0442810059f, 0.0110397339f, -0.0983276367f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0988769531f, -0.102478027f, -0.0581665039f, -0.147705078f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.154418945f, 0.0957641602f, 0.0079498291f, 0.140869141f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[18] = __float2half_rn(acc);
    }
    {
        float acc = 0.0157318115f;
        acc = nsf_dot4_acc(acc, 0.140625f, 0.0420837402f, -0.110046387f, -0.0848999023f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0297698975f, 0.0348205566f, 0.00292396545f, -0.044342041f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.049987793f, 0.000784397125f, -0.19921875f, -0.00261878967f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.178466797f, 0.0427246094f, -0.299804688f, -0.142456055f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.0949707031f, 0.0487365723f, -0.163330078f, 0.0206298828f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.088684082f, 0.132446289f, 0.0285644531f, 0.0579528809f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.11706543f, 0.0348205566f, 0.00712203979f, -0.138183594f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.0900268555f, -0.0299377441f, 0.00894927979f, -0.0624694824f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[19] = __float2half_rn(acc);
    }
    {
        float acc = -0.0672607422f;
        acc = nsf_dot4_acc(acc, -0.175292969f, -0.0860595703f, 0.187133789f, 0.0952148438f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.207763672f, 0.0412597656f, -0.207275391f, -0.214477539f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.0995483398f, -0.0233459473f, -0.0141525269f, -0.184204102f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0569458008f, 0.196655273f, 0.0661621094f, 0.118652344f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.164916992f, -0.133422852f, -0.0146255493f, 0.00969696045f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.123291016f, -0.175537109f, 0.0195770264f, 0.0277709961f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.102661133f, -0.0720825195f, 0.0451660156f, -0.0713500977f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.0902099609f, -0.0553588867f, 0.0678710938f, 0.0982055664f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[20] = __float2half_rn(acc);
    }
    {
        float acc = -0.163574219f;
        acc = nsf_dot4_acc(acc, 0.147094727f, 0.0371704102f, 0.0156860352f, 0.100708008f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.17565918f, 0.10559082f, 0.104919434f, 0.108764648f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.135864258f, -0.0936889648f, -0.0631103516f, -0.00624465942f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.00651931763f, 0.0415649414f, 0.0743408203f, -0.14440918f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0219573975f, 0.0158691406f, 0.0321350098f, -0.0377197266f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0171051025f, -0.141601562f, -0.117614746f, -0.0900268555f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.106811523f, -0.159179688f, -0.145019531f, -0.172973633f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.0242767334f, -0.197631836f, 0.0132369995f, -0.122802734f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[21] = __float2half_rn(acc);
    }
    {
        float acc = 0.0417785645f;
        acc = nsf_dot4_acc(acc, -0.113220215f, -0.0310058594f, 0.149780273f, 0.21862793f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.200317383f, 0.0423583984f, -0.130249023f, 0.00708770752f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0258178711f, -0.0375976562f, 0.0645751953f, 0.133789062f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0817871094f, 0.204223633f, -0.121520996f, -0.0349731445f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0112915039f, -0.0769042969f, 0.12890625f, -0.0452880859f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.166992188f, 0.252929688f, 0.0753173828f, -0.0651245117f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.183959961f, -0.0950927734f, 0.148071289f, 0.175292969f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.0344543457f, 0.100463867f, 0.163330078f, 0.133666992f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[22] = __float2half_rn(acc);
    }
    {
        float acc = -0.202758789f;
        acc = nsf_dot4_acc(acc, -0.0542297363f, -0.171142578f, -0.0308837891f, 0.0298461914f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0747680664f, 0.0248718262f, -0.0942993164f, 0.0237731934f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.0559692383f, 0.00479888916f, -0.247192383f, -0.244018555f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.105651855f, 0.0626831055f, 0.0994873047f, -0.168945312f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0787963867f, 0.135742188f, -0.157836914f, -0.107788086f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.137084961f, -0.0718383789f, -0.227539062f, -0.125366211f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.172363281f, 0.0689697266f, -0.0152359009f, 0.16394043f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.20715332f, -0.219848633f, -0.0709228516f, 0.0291900635f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[23] = __float2half_rn(acc);
    }
    {
        float acc = 0.103637695f;
        acc = nsf_dot4_acc(acc, 0.154296875f, -0.0374755859f, -0.0125656128f, -0.0793457031f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.162597656f, -0.0319824219f, -0.0531311035f, 0.150878906f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.155395508f, -0.00745010376f, -0.0942993164f, 0.157226562f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0248260498f, -0.0809326172f, -0.113647461f, -0.0774536133f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.0977172852f, -0.0791625977f, -0.142700195f, 0.0178527832f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.071105957f, -0.107788086f, 0.01537323f, -0.0614624023f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0902709961f, 0.153076172f, -0.00464630127f, -0.0889892578f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.11920166f, -0.0911254883f, 0.0718994141f, 0.0405578613f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[24] = __float2half_rn(acc);
    }
    {
        float acc = -0.0650634766f;
        acc = nsf_dot4_acc(acc, 0.0952148438f, -0.0198669434f, 0.297363281f, -0.121948242f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0188598633f, 0.167724609f, -0.173217773f, -0.111083984f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.121520996f, -0.027633667f, -0.121887207f, 0.111755371f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0240020752f, 0.21081543f, -0.113098145f, 0.0657958984f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.143188477f, 0.017288208f, 0.0682373047f, 0.150390625f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.110107422f, 0.0634765625f, -0.0105743408f, 0.0985107422f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.081237793f, 0.165283203f, -0.0870361328f, 0.0815429688f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.0980834961f, 0.047088623f, 0.132202148f, 0.0213165283f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[25] = __float2half_rn(acc);
    }
    {
        float acc = 0.0974121094f;
        acc = nsf_dot4_acc(acc, -0.0177154541f, 0.0451049805f, -0.00402832031f, -0.204833984f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.265869141f, -0.184448242f, -0.0571289062f, -0.186279297f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.00308990479f, 0.00596618652f, 0.0214996338f, -0.145874023f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.188110352f, 0.252929688f, -0.129760742f, 0.0717773438f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0124816895f, 0.0123138428f, -0.0653076172f, 0.179931641f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0677490234f, -0.0554199219f, 0.00946044922f, -0.0367126465f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.139648438f, -0.00356674194f, 0.0711669922f, 0.173461914f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.119262695f, -0.0582885742f, 0.0121002197f, -0.0739135742f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[26] = __float2half_rn(acc);
    }
    {
        float acc = 0.00902557373f;
        acc = nsf_dot4_acc(acc, -0.0924072266f, -0.109130859f, -0.329589844f, -0.0178985596f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0540771484f, -0.130371094f, -0.0837402344f, -0.0452575684f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.091003418f, 0.157836914f, -0.205810547f, 0.0109634399f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0206604004f, 0.00842285156f, -0.136230469f, -0.0489807129f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.110412598f, 0.169433594f, -0.228881836f, -0.254638672f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.11895752f, -0.175048828f, 0.0151596069f, 0.0389404297f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.157592773f, 0.114990234f, 0.0799560547f, -0.0871582031f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.0220794678f, -0.12512207f, 0.000524997711f, -0.0819702148f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[27] = __float2half_rn(acc);
    }
    {
        float acc = -0.172851562f;
        acc = nsf_dot4_acc(acc, -0.0688476562f, -0.120666504f, -0.0311126709f, 0.123901367f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.00449371338f, -0.00130558014f, -0.083190918f, 0.0862426758f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.099609375f, -0.071472168f, -0.131958008f, -0.154663086f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.162719727f, -0.13269043f, 0.0265045166f, 0.00917053223f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0469970703f, -0.0706176758f, -0.190917969f, 0.196777344f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.176147461f, -0.0492553711f, -0.0551147461f, -0.0382080078f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.116760254f, 0.149291992f, 0.0112075806f, -0.0624389648f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.177246094f, 0.120056152f, -0.015045166f, -0.0483398438f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[28] = __float2half_rn(acc);
    }
    {
        float acc = 0.117858887f;
        acc = nsf_dot4_acc(acc, 0.00745391846f, 0.142089844f, -0.177734375f, -0.0673828125f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.159912109f, -0.108215332f, 0.108337402f, -0.0592346191f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.0747070312f, 0.0688476562f, -0.12322998f, -0.026763916f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0525512695f, -0.120239258f, 0.102355957f, 0.127075195f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.184204102f, -0.245849609f, -0.182495117f, 0.009765625f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.150146484f, -0.0648803711f, -0.154785156f, 0.020690918f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.158081055f, -0.151245117f, 0.0757446289f, 0.0824584961f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.0961914062f, 0.165649414f, -0.178833008f, -0.119018555f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[29] = __float2half_rn(acc);
    }
    {
        float acc = -0.120544434f;
        acc = nsf_dot4_acc(acc, -0.0879516602f, -0.00466918945f, 0.098449707f, -0.0136108398f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.191040039f, -0.0116958618f, 0.128417969f, 0.133178711f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.100158691f, 0.0256652832f, 0.0172424316f, -0.13671875f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0551757812f, -0.0608215332f, -0.189086914f, -0.0924682617f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.133544922f, -0.0202178955f, -0.0196228027f, 0.0286254883f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.146728516f, 0.122497559f, 0.059173584f, 0.00720977783f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.00302505493f, 0.108032227f, -0.135498047f, -0.0892333984f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.159301758f, 0.033996582f, 0.0171966553f, 0.0915527344f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[30] = __float2half_rn(acc);
    }
    {
        float acc = 0.0f;
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[31] = __float2half_rn(acc);
    }
}

static __forceinline__ __device__ void eval_transform_1(float c0, float c1, float c2, PhiVec& y_out)
{
    const float x0 = c0;
    const float x1 = c1;
    const float x2 = c2;
    const float x3 = 0.0f;
    alignas(16) float h0[32];
    alignas(16) float h1[32];

    {
        float acc = 0.309326172f;
        acc = nsf_dot4_acc(acc, 0.03515625f, -0.329833984f, -0.537109375f, 0.0f, x0, x1, x2, x3);
        h0[0] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.3828125f;
        acc = nsf_dot4_acc(acc, 0.267578125f, -0.317871094f, 0.584472656f, 0.0f, x0, x1, x2, x3);
        h0[1] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.454589844f;
        acc = nsf_dot4_acc(acc, -0.653320312f, 0.383056641f, 0.406738281f, 0.0f, x0, x1, x2, x3);
        h0[2] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.553710938f;
        acc = nsf_dot4_acc(acc, 0.633300781f, -0.247558594f, -0.135742188f, 0.0f, x0, x1, x2, x3);
        h0[3] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.359375f;
        acc = nsf_dot4_acc(acc, -0.471435547f, 0.405761719f, -0.308105469f, 0.0f, x0, x1, x2, x3);
        h0[4] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.437744141f;
        acc = nsf_dot4_acc(acc, 0.0848999023f, -0.0704345703f, 0.119750977f, 0.0f, x0, x1, x2, x3);
        h0[5] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.192993164f;
        acc = nsf_dot4_acc(acc, -0.560058594f, 0.355957031f, -0.418457031f, 0.0f, x0, x1, x2, x3);
        h0[6] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.239379883f;
        acc = nsf_dot4_acc(acc, 0.640136719f, -0.37890625f, -0.161987305f, 0.0f, x0, x1, x2, x3);
        h0[7] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.574707031f;
        acc = nsf_dot4_acc(acc, 0.0655517578f, 0.135009766f, 0.00620651245f, 0.0f, x0, x1, x2, x3);
        h0[8] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.465576172f;
        acc = nsf_dot4_acc(acc, -0.599609375f, 0.398925781f, 0.637207031f, 0.0f, x0, x1, x2, x3);
        h0[9] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.425048828f;
        acc = nsf_dot4_acc(acc, 0.173217773f, 0.534179688f, 0.157836914f, 0.0f, x0, x1, x2, x3);
        h0[10] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.309570312f;
        acc = nsf_dot4_acc(acc, 0.294921875f, -0.52734375f, -0.378662109f, 0.0f, x0, x1, x2, x3);
        h0[11] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.445556641f;
        acc = nsf_dot4_acc(acc, -0.395263672f, -0.13671875f, 0.275878906f, 0.0f, x0, x1, x2, x3);
        h0[12] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.260253906f;
        acc = nsf_dot4_acc(acc, -0.114440918f, -0.0452575684f, 0.108825684f, 0.0f, x0, x1, x2, x3);
        h0[13] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0736694336f;
        acc = nsf_dot4_acc(acc, -0.246948242f, -0.0558166504f, 0.642578125f, 0.0f, x0, x1, x2, x3);
        h0[14] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0346679688f;
        acc = nsf_dot4_acc(acc, 0.423095703f, 0.532714844f, 0.137939453f, 0.0f, x0, x1, x2, x3);
        h0[15] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.00354385376f;
        acc = nsf_dot4_acc(acc, 0.0770874023f, 0.334716797f, -0.124938965f, 0.0f, x0, x1, x2, x3);
        h0[16] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.216064453f;
        acc = nsf_dot4_acc(acc, -0.344970703f, 0.422851562f, 0.0852661133f, 0.0f, x0, x1, x2, x3);
        h0[17] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.569824219f;
        acc = nsf_dot4_acc(acc, -0.296142578f, 0.470458984f, 0.159667969f, 0.0f, x0, x1, x2, x3);
        h0[18] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.638671875f;
        acc = nsf_dot4_acc(acc, 0.464599609f, -0.463378906f, 0.247924805f, 0.0f, x0, x1, x2, x3);
        h0[19] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0291595459f;
        acc = nsf_dot4_acc(acc, 0.436523438f, -0.218139648f, -0.236572266f, 0.0f, x0, x1, x2, x3);
        h0[20] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.305664062f;
        acc = nsf_dot4_acc(acc, 0.60546875f, 0.227661133f, 0.369873047f, 0.0f, x0, x1, x2, x3);
        h0[21] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0777587891f;
        acc = nsf_dot4_acc(acc, 0.291748047f, 0.0292816162f, 0.443115234f, 0.0f, x0, x1, x2, x3);
        h0[22] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.142456055f;
        acc = nsf_dot4_acc(acc, 0.129760742f, -0.392089844f, 0.326171875f, 0.0f, x0, x1, x2, x3);
        h0[23] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.413818359f;
        acc = nsf_dot4_acc(acc, 0.040802002f, 0.267578125f, -0.0383911133f, 0.0f, x0, x1, x2, x3);
        h0[24] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0919799805f;
        acc = nsf_dot4_acc(acc, 0.0263061523f, -0.0570678711f, 0.22265625f, 0.0f, x0, x1, x2, x3);
        h0[25] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.336181641f;
        acc = nsf_dot4_acc(acc, -0.278808594f, -0.541992188f, 0.383544922f, 0.0f, x0, x1, x2, x3);
        h0[26] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.24621582f;
        acc = nsf_dot4_acc(acc, -0.284912109f, -0.0315856934f, 0.249267578f, 0.0f, x0, x1, x2, x3);
        h0[27] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.290039062f;
        acc = nsf_dot4_acc(acc, 0.217163086f, 0.36328125f, 0.113891602f, 0.0f, x0, x1, x2, x3);
        h0[28] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.358154297f;
        acc = nsf_dot4_acc(acc, 0.142944336f, -0.0516662598f, 0.0852050781f, 0.0f, x0, x1, x2, x3);
        h0[29] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.516113281f;
        acc = nsf_dot4_acc(acc, 0.211669922f, 0.393798828f, -0.546875f, 0.0f, x0, x1, x2, x3);
        h0[30] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0911865234f;
        acc = nsf_dot4_acc(acc, -0.025680542f, -0.586425781f, -0.130859375f, 0.0f, x0, x1, x2, x3);
        h0[31] = __half2float(nsf_relu_to_half(acc));
    }

    {
        float acc = 0.0407714844f;
        acc = nsf_dot4_acc(acc, 0.0861206055f, -0.0270690918f, 0.224243164f, -0.0697631836f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.0244598389f, 0.0185241699f, 0.00498580933f, -0.0899658203f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.164306641f, 0.0165863037f, 0.137817383f, 0.0764160156f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.0657348633f, 0.143554688f, -0.0865478516f, -0.0489196777f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0126800537f, 0.124938965f, -0.149780273f, -0.0338134766f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.218261719f, 0.0413818359f, 0.0162658691f, 0.0459289551f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0954589844f, -0.0546569824f, -0.00534439087f, 0.16027832f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.185913086f, -0.122192383f, 0.0129318237f, -0.0755004883f, h0[28], h0[29], h0[30], h0[31]);
        h1[0] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.000241160393f;
        acc = nsf_dot4_acc(acc, 0.0231170654f, 0.330566406f, -0.151489258f, 0.0708618164f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.151123047f, 0.271728516f, -0.151855469f, 0.143798828f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.193359375f, -0.0969848633f, -0.0160217285f, 0.251953125f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.175292969f, 0.186523438f, 0.099609375f, 0.0684814453f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0272216797f, 0.0416259766f, -0.0116500854f, 0.19543457f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.050201416f, 0.0377807617f, 0.266357422f, 0.089050293f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0589294434f, -0.0438537598f, 0.135986328f, -0.0176544189f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.19921875f, 0.170532227f, 0.0859985352f, -0.112182617f, h0[28], h0[29], h0[30], h0[31]);
        h1[1] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.14050293f;
        acc = nsf_dot4_acc(acc, -0.12902832f, 0.170776367f, 0.000738143921f, -0.106323242f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.101745605f, -0.0368347168f, 0.0121078491f, -0.0921020508f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.149780273f, -0.117431641f, -0.0194091797f, -0.127929688f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.174194336f, 0.00191307068f, -0.0676879883f, 0.0698852539f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.156738281f, -0.0903930664f, 0.06640625f, -0.150146484f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0881347656f, 0.0148620605f, -0.147338867f, -0.175415039f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0558776855f, 0.0835571289f, -0.0353393555f, -0.15637207f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.101867676f, 0.0187835693f, 0.00643157959f, 0.0452575684f, h0[28], h0[29], h0[30], h0[31]);
        h1[2] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.167480469f;
        acc = nsf_dot4_acc(acc, 0.00541687012f, -0.0416564941f, 0.0882568359f, 0.146118164f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.147216797f, 0.137817383f, -0.0659790039f, -0.0877075195f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.0428771973f, 0.041015625f, 0.0559692383f, -0.0508422852f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.131103516f, -0.159545898f, 0.145751953f, -0.0422668457f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0238952637f, -0.110473633f, -0.108215332f, 0.0138092041f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.029296875f, -0.076171875f, -0.089050293f, 0.114074707f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.109436035f, -0.157348633f, -0.0530395508f, 0.144042969f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0384521484f, 0.0244293213f, -0.0220184326f, 0.0794067383f, h0[28], h0[29], h0[30], h0[31]);
        h1[3] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.11114502f;
        acc = nsf_dot4_acc(acc, -0.0876464844f, 0.119873047f, 0.0417480469f, 0.0196533203f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0243377686f, -0.109008789f, 0.131347656f, 0.0928344727f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.138305664f, -0.164550781f, -0.045501709f, 0.117431641f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.0870361328f, 0.163818359f, -0.0888671875f, -0.159667969f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.0777587891f, -0.106811523f, -0.0278625488f, -0.0551452637f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.0140457153f, -0.154052734f, -0.162109375f, 0.0759887695f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.02003479f, -0.106628418f, -0.00450515747f, -0.161376953f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0469360352f, 0.0839233398f, 0.171386719f, -0.144042969f, h0[28], h0[29], h0[30], h0[31]);
        h1[4] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0331115723f;
        acc = nsf_dot4_acc(acc, 0.0418395996f, 0.292236328f, 0.0112075806f, 0.0832519531f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.0375366211f, 0.196044922f, -0.0311431885f, 0.0940551758f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.229370117f, -0.190429688f, -0.0188903809f, 0.169067383f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.128173828f, -0.0402832031f, 0.154907227f, 0.0487365723f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.0899047852f, 0.0232849121f, 0.0795288086f, 0.039276123f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0576477051f, 0.109924316f, 0.0387878418f, 0.184936523f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.0016078949f, 0.0959472656f, 0.154663086f, -0.0903930664f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0593566895f, 0.0806274414f, -0.174804688f, 0.1875f, h0[28], h0[29], h0[30], h0[31]);
        h1[5] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0083694458f;
        acc = nsf_dot4_acc(acc, -0.152709961f, -0.00580978394f, 0.0362854004f, 0.050201416f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.109741211f, 0.00581359863f, 0.0761108398f, 0.130859375f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.151245117f, -0.087890625f, 0.0348815918f, -0.0161437988f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.118774414f, -0.156738281f, -0.17199707f, -0.142578125f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0776367188f, -0.122741699f, -0.169555664f, -0.00538635254f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0712890625f, -0.106933594f, -0.00106811523f, -0.0211486816f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0659790039f, -0.14074707f, -0.00923156738f, 0.149169922f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.12286377f, -0.175170898f, -0.0266113281f, -0.00628662109f, h0[28], h0[29], h0[30], h0[31]);
        h1[6] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0451965332f;
        acc = nsf_dot4_acc(acc, -0.158203125f, -0.151367188f, -0.102905273f, 0.17199707f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.111999512f, 0.0602722168f, 0.0801391602f, -0.0860595703f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.102905273f, 0.125f, -0.0177764893f, 0.00244522095f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.0120391846f, 0.176635742f, 0.200439453f, 0.0621032715f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0256195068f, -0.131958008f, 0.169677734f, 0.167358398f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.126586914f, -0.187011719f, 0.211425781f, 0.0155715942f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.126220703f, 0.0269165039f, 0.171630859f, 0.264160156f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0953369141f, -0.0562438965f, 0.123901367f, 0.0342712402f, h0[28], h0[29], h0[30], h0[31]);
        h1[7] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.100646973f;
        acc = nsf_dot4_acc(acc, 0.0172424316f, -0.142333984f, -0.0759277344f, -0.174926758f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.118225098f, 0.174682617f, 0.102966309f, 0.0564880371f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0417785645f, 0.170166016f, -0.0895996094f, -0.0593566895f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.158691406f, -0.0175018311f, 0.0121231079f, 0.0484008789f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.12298584f, -0.0204620361f, -0.145629883f, -0.143798828f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0230407715f, -0.172607422f, 0.0470581055f, -0.162841797f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.12298584f, -0.0366821289f, -0.168701172f, 0.142822266f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.144897461f, -0.0923461914f, 0.0370178223f, 0.0904541016f, h0[28], h0[29], h0[30], h0[31]);
        h1[8] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0275878906f;
        acc = nsf_dot4_acc(acc, 0.0249786377f, -0.0664672852f, -0.175415039f, 0.102172852f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.155029297f, -0.0615234375f, 0.000231742859f, -0.0874023438f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.0938110352f, -0.170166016f, -0.0127944946f, -0.068359375f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.123535156f, -0.0941162109f, 0.103271484f, -0.0134124756f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.100524902f, -0.0537414551f, -0.0475158691f, -0.0310211182f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.061126709f, 0.0302581787f, 0.117614746f, -0.00947570801f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0407104492f, 0.0748291016f, 0.00143146515f, 0.0479736328f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0754394531f, 0.165405273f, -0.161376953f, 0.0165710449f, h0[28], h0[29], h0[30], h0[31]);
        h1[9] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.031036377f;
        acc = nsf_dot4_acc(acc, 0.00246810913f, -0.00349235535f, -0.0596313477f, 0.157836914f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.315429688f, -0.11138916f, 0.110900879f, 0.141845703f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0552062988f, -0.00696563721f, 0.140014648f, 0.0388183594f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.152954102f, 0.0642700195f, 0.112915039f, 0.0556030273f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0120544434f, 0.0759277344f, 0.0657958984f, -0.114501953f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.019241333f, 0.115966797f, 0.0332946777f, 0.0980224609f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.140625f, -0.108215332f, 0.0670776367f, -0.0241699219f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.170166016f, -0.171020508f, 0.0175323486f, 0.0200958252f, h0[28], h0[29], h0[30], h0[31]);
        h1[10] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.178710938f;
        acc = nsf_dot4_acc(acc, 0.00971984863f, 0.210083008f, -0.174926758f, 0.265136719f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.00297737122f, -0.0231933594f, -0.0816650391f, 0.182373047f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.083984375f, 0.0338439941f, -0.00680923462f, 0.194946289f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.253173828f, 0.107116699f, 0.0169372559f, 0.106140137f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0219421387f, -0.170410156f, 0.0336608887f, 0.0605163574f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.214477539f, 0.125366211f, 0.23449707f, -0.0329284668f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.0363769531f, -0.0949707031f, 0.0669555664f, -0.141967773f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.148681641f, -0.0893554688f, -0.149291992f, -0.0688476562f, h0[28], h0[29], h0[30], h0[31]);
        h1[11] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.145629883f;
        acc = nsf_dot4_acc(acc, -0.11126709f, -0.0906982422f, 0.16809082f, -0.0333251953f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.146606445f, 0.00392150879f, -0.0227355957f, -0.0517578125f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.126342773f, -0.142089844f, 0.122375488f, -0.054473877f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.172363281f, 0.162231445f, 0.0032787323f, -0.118286133f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.138916016f, -0.1484375f, -0.155029297f, 0.159912109f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.121887207f, 0.0144577026f, -0.150634766f, -0.0462341309f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.166015625f, 0.0609130859f, -0.0469360352f, -0.0378417969f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0551452637f, 0.0828857422f, 0.095703125f, 0.0387268066f, h0[28], h0[29], h0[30], h0[31]);
        h1[12] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0598144531f;
        acc = nsf_dot4_acc(acc, -0.0989990234f, -0.0909423828f, 0.0508117676f, -0.110595703f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0768432617f, 0.0350036621f, 0.157714844f, -0.105041504f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.133300781f, 0.143310547f, -0.0178375244f, 0.00232887268f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.168334961f, 0.145263672f, 0.0925292969f, 0.133544922f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.165405273f, 0.055480957f, 0.149414062f, -0.00609588623f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.035369873f, -0.0901489258f, 0.178955078f, -0.00423431396f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.268554688f, 0.244506836f, 0.193725586f, 0.135742188f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0952758789f, -0.161132812f, 0.143310547f, -0.107788086f, h0[28], h0[29], h0[30], h0[31]);
        h1[13] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.113586426f;
        acc = nsf_dot4_acc(acc, 0.064453125f, -0.0299377441f, -0.0593566895f, -0.0830078125f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.114013672f, -0.0961914062f, -0.0311737061f, 0.0891723633f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0178222656f, -0.148681641f, -0.17578125f, -0.0859375f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.163085938f, -0.125732422f, 0.0974121094f, 0.126586914f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0733032227f, -0.0780639648f, -0.127197266f, 0.0275726318f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.0247192383f, 0.00859069824f, 0.0766601562f, 0.07421875f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.0557861328f, 0.0121383667f, 0.0419921875f, 0.00376319885f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.035369873f, -0.0715942383f, 0.144897461f, -0.0181274414f, h0[28], h0[29], h0[30], h0[31]);
        h1[14] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0785522461f;
        acc = nsf_dot4_acc(acc, 0.00489807129f, -0.139160156f, -0.134033203f, 0.0621643066f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.0375061035f, -0.0352478027f, 0.095703125f, 0.0261077881f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.120849609f, 0.136474609f, -0.0251159668f, -0.0324401855f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.00706481934f, 0.0433654785f, -0.0339355469f, -0.0360412598f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0311279297f, 0.0820922852f, 0.219726562f, 0.186279297f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0212097168f, -0.0246276855f, 0.095703125f, 0.0526428223f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.031829834f, 0.11932373f, -0.0932617188f, -0.0920410156f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0127563477f, -0.0535888672f, -0.0874633789f, -0.139770508f, h0[28], h0[29], h0[30], h0[31]);
        h1[15] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.00294303894f;
        acc = nsf_dot4_acc(acc, 0.105773926f, -0.0187072754f, -0.0953369141f, -0.0740356445f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0464782715f, -0.14050293f, 0.00719451904f, -0.00358009338f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0473022461f, 0.0578308105f, -0.116699219f, 0.0611877441f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.120117188f, -0.00252532959f, 0.0765380859f, -0.0669555664f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.0351257324f, 0.0516662598f, -0.094543457f, 0.0963134766f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.167114258f, -0.06640625f, -0.114807129f, -0.118530273f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.113098145f, -0.180053711f, 0.016784668f, -0.0484619141f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.148803711f, -7.27176666e-05f, -0.0294494629f, 0.0580444336f, h0[28], h0[29], h0[30], h0[31]);
        h1[16] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0536193848f;
        acc = nsf_dot4_acc(acc, -0.0883789062f, 0.0534057617f, -0.0244750977f, 0.175292969f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.04296875f, 0.111572266f, 0.121704102f, 0.0282745361f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.185546875f, 0.0817260742f, -0.0932006836f, 0.150146484f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.104736328f, 0.0223693848f, 0.140380859f, 0.0430908203f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.123779297f, -0.0349731445f, -0.0703735352f, 0.214233398f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.12322998f, 0.166503906f, 0.122619629f, 0.165039062f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.171630859f, 0.0983276367f, 0.00620651245f, 0.0809326172f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.127319336f, 0.00603866577f, -0.0568237305f, 0.0509338379f, h0[28], h0[29], h0[30], h0[31]);
        h1[17] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.338623047f;
        acc = nsf_dot4_acc(acc, -0.0549316406f, 0.270263672f, -0.270996094f, 0.310058594f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.0304718018f, 0.0169525146f, 0.0697631836f, 0.0863647461f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.137573242f, 0.0198364258f, 0.0960083008f, 0.308105469f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.0800170898f, 0.133178711f, 0.0114898682f, 0.0509338379f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.174804688f, 0.0214385986f, 0.124938965f, 0.222045898f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0057144165f, 0.264404297f, 0.188964844f, 0.107971191f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.00619888306f, 0.327880859f, 0.216186523f, 0.225952148f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0483703613f, -0.119934082f, 0.0723876953f, 0.0978393555f, h0[28], h0[29], h0[30], h0[31]);
        h1[18] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0934448242f;
        acc = nsf_dot4_acc(acc, 0.0333557129f, -0.0833740234f, -0.089050293f, 0.113525391f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0592651367f, -0.0966796875f, 0.00768280029f, -0.0731811523f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.0421447754f, 0.0836791992f, -0.15637207f, 0.12109375f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.0961303711f, -0.0289459229f, -0.100158691f, 0.0321960449f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.117004395f, 0.114807129f, -0.120788574f, 0.00522232056f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.218261719f, 0.0850830078f, 0.0917358398f, 0.103759766f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.00573730469f, 0.0755004883f, 0.0321044922f, 0.090637207f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0997314453f, -0.154663086f, 0.0186309814f, -0.0927734375f, h0[28], h0[29], h0[30], h0[31]);
        h1[19] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0324401855f;
        acc = nsf_dot4_acc(acc, -0.0862426758f, 0.120666504f, -0.0252990723f, 0.0202789307f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.111328125f, 0.112304688f, 0.0943603516f, -0.0371704102f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.166870117f, -0.169677734f, 0.0457458496f, -0.00470352173f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.112487793f, 0.0705566406f, 0.154541016f, 0.0948486328f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0332946777f, 0.0160064697f, -0.122802734f, 0.0775756836f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.143066406f, 0.114807129f, -0.0888671875f, -0.0733642578f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.157226562f, -0.0497131348f, -0.0200958252f, -0.0874023438f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.117492676f, 0.10144043f, -0.156005859f, -0.0196075439f, h0[28], h0[29], h0[30], h0[31]);
        h1[20] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.115600586f;
        acc = nsf_dot4_acc(acc, 0.0257110596f, 0.189819336f, -0.0914916992f, 0.202148438f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.0459899902f, 0.00696182251f, 0.0684814453f, -0.0818481445f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.220947266f, 0.137084961f, 0.0173797607f, 0.087097168f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.0782470703f, 0.122009277f, 0.126342773f, 0.0412902832f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.137817383f, 0.00368118286f, -0.0425109863f, 0.24230957f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.0388183594f, 0.0220031738f, 0.0381164551f, 0.173583984f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0684814453f, 0.177001953f, 0.0093536377f, -0.0484008789f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0441589355f, -0.0994873047f, -0.145141602f, 0.029006958f, h0[28], h0[29], h0[30], h0[31]);
        h1[21] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.204956055f;
        acc = nsf_dot4_acc(acc, -0.0538024902f, 0.0118331909f, -0.0621032715f, 0.0667114258f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.10168457f, -0.0120620728f, -0.0546264648f, 0.186645508f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.116455078f, -0.0640258789f, -0.0489501953f, 0.0739746094f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.0841674805f, 0.0293426514f, 0.140136719f, 0.172363281f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.0914916992f, 0.069519043f, 0.152587891f, 0.00321388245f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.158081055f, 0.142333984f, -0.00463104248f, 0.0608825684f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.00443267822f, -0.0777587891f, 0.114624023f, 0.178466797f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.100646973f, 0.0662231445f, 0.0541992188f, 0.0715942383f, h0[28], h0[29], h0[30], h0[31]);
        h1[22] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.183227539f;
        acc = nsf_dot4_acc(acc, 0.0259552002f, 0.189331055f, 0.075012207f, -0.127319336f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.201171875f, 0.13269043f, -0.151123047f, 0.053894043f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.0286865234f, 0.0143356323f, 0.140258789f, 0.0460205078f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.0476074219f, -0.00861358643f, 0.113098145f, 0.10723877f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.00903320312f, -0.143798828f, 0.191650391f, 0.000912189484f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0982666016f, 0.104919434f, 0.0670166016f, -0.116271973f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0773925781f, -0.0393066406f, 0.0568237305f, -0.0501403809f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.138183594f, -0.159545898f, -0.0475769043f, 0.00131893158f, h0[28], h0[29], h0[30], h0[31]);
        h1[23] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0201873779f;
        acc = nsf_dot4_acc(acc, 0.0168609619f, 0.145263672f, -0.055847168f, -0.141113281f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.11706543f, 0.049407959f, 0.116455078f, -0.104614258f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.103820801f, -0.0714111328f, 0.0125656128f, 0.0599060059f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.1015625f, -0.0166473389f, -0.0305480957f, 0.161254883f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0637817383f, 0.171264648f, -0.108398438f, -0.0728149414f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.13293457f, 0.130249023f, -0.121582031f, 0.0731811523f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.0649414062f, 0.176147461f, 0.0249328613f, -0.153564453f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0304260254f, 0.0214996338f, -0.171020508f, 0.0354919434f, h0[28], h0[29], h0[30], h0[31]);
        h1[24] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.0668945312f;
        acc = nsf_dot4_acc(acc, 0.168701172f, 0.0508728027f, -0.000940322876f, -0.0366516113f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0469665527f, -0.0787353516f, 0.174072266f, 0.152832031f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0825805664f, -0.0658569336f, -0.166015625f, -0.101745605f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, 0.171875f, -0.162231445f, -0.168823242f, -0.0127410889f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.0562438965f, -0.17590332f, -0.1640625f, 0.000961303711f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.0592956543f, 0.123168945f, 0.0507507324f, -0.153442383f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.159301758f, 0.121398926f, -0.0313415527f, -0.077331543f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0556945801f, -0.0347290039f, 0.0845336914f, 0.0448303223f, h0[28], h0[29], h0[30], h0[31]);
        h1[25] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.0791625977f;
        acc = nsf_dot4_acc(acc, -0.0985107422f, -0.146118164f, 0.103393555f, -0.107971191f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.0226745605f, 0.0308685303f, -0.087097168f, 0.0777587891f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.0983886719f, 0.0408325195f, 0.0690917969f, 0.0783691406f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.175537109f, 0.00806427002f, 0.0932617188f, -0.0974731445f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.0552978516f, 0.0358581543f, 0.188842773f, -0.0444335938f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, -0.00415420532f, -0.0995483398f, -0.0354614258f, 0.0346374512f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.122558594f, 0.139038086f, 0.111755371f, 0.0894775391f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.109741211f, -0.0550231934f, 0.0402526855f, -0.0142974854f, h0[28], h0[29], h0[30], h0[31]);
        h1[26] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.118469238f;
        acc = nsf_dot4_acc(acc, -0.0687866211f, 0.0766601562f, -0.0426635742f, 0.0386657715f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.114990234f, -0.112365723f, -0.0152740479f, 0.0834350586f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.159790039f, 0.0352478027f, 0.0782470703f, 0.230834961f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.00646591187f, -0.129760742f, -0.0697021484f, -0.10144043f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.0135269165f, 0.0377502441f, 0.0065536499f, 0.157592773f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.2890625f, -0.0700683594f, 0.163574219f, 0.171142578f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0176086426f, -0.128295898f, -0.0905151367f, 0.0309448242f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0191802979f, 0.164794922f, 0.0899658203f, 0.0816650391f, h0[28], h0[29], h0[30], h0[31]);
        h1[27] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.061706543f;
        acc = nsf_dot4_acc(acc, 0.0232086182f, 0.235107422f, -0.114868164f, 0.163085938f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.109313965f, 0.0618286133f, 0.0575561523f, 0.192260742f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0234527588f, -0.0428161621f, -0.0311126709f, 0.147338867f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.0991210938f, 0.0880126953f, -0.146850586f, -0.0675048828f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.137695312f, -0.0545959473f, 0.119140625f, 0.220703125f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0880126953f, -0.041015625f, -0.0609130859f, 0.138671875f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.165283203f, 0.0436096191f, -0.134521484f, -0.0903320312f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, -0.0113449097f, 0.0806884766f, -0.0592346191f, -0.099609375f, h0[28], h0[29], h0[30], h0[31]);
        h1[28] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.132324219f;
        acc = nsf_dot4_acc(acc, -0.11529541f, 0.211181641f, -0.0322570801f, -0.0737304688f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.221435547f, -0.122314453f, -0.0591125488f, -0.0211486816f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, 0.0374450684f, -0.115600586f, -0.0873413086f, -0.10333252f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.197753906f, -0.0767211914f, 0.0268249512f, -0.00240325928f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.020904541f, -0.0809326172f, 0.179077148f, -0.0337524414f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.11151123f, -0.102661133f, -0.0888061523f, 0.0832519531f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.0431213379f, 0.119812012f, -0.101074219f, 0.141357422f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.192382812f, 0.0113143921f, 0.0377807617f, 0.179077148f, h0[28], h0[29], h0[30], h0[31]);
        h1[29] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = 0.224243164f;
        acc = nsf_dot4_acc(acc, 0.142089844f, 0.00899505615f, -0.0490722656f, -0.0431213379f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, -0.0127944946f, 0.0456237793f, -0.0318908691f, 0.0160369873f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.0162963867f, -0.00531387329f, 0.233032227f, 0.0263824463f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.151245117f, 0.200683594f, 0.192871094f, 0.0321044922f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, 0.172851562f, 0.0494384766f, 0.204345703f, 0.104858398f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0200500488f, 0.00250434875f, 0.17590332f, 0.163818359f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, -0.0828857422f, 0.219360352f, -0.120117188f, 0.138793945f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.145019531f, 0.043548584f, -0.0333557129f, 0.116516113f, h0[28], h0[29], h0[30], h0[31]);
        h1[30] = __half2float(nsf_relu_to_half(acc));
    }
    {
        float acc = -0.100952148f;
        acc = nsf_dot4_acc(acc, -0.0551452637f, -0.13671875f, -0.126586914f, 0.0785522461f, h0[0], h0[1], h0[2], h0[3]);
        acc = nsf_dot4_acc(acc, 0.135009766f, 0.000904083252f, -0.17565918f, -0.176269531f, h0[4], h0[5], h0[6], h0[7]);
        acc = nsf_dot4_acc(acc, -0.0460205078f, 0.0892944336f, -0.121520996f, 0.0405578613f, h0[8], h0[9], h0[10], h0[11]);
        acc = nsf_dot4_acc(acc, -0.10345459f, 0.137451172f, -0.0527038574f, -0.0368347168f, h0[12], h0[13], h0[14], h0[15]);
        acc = nsf_dot4_acc(acc, -0.155029297f, -0.158325195f, 0.00587463379f, 0.0222473145f, h0[16], h0[17], h0[18], h0[19]);
        acc = nsf_dot4_acc(acc, 0.0955200195f, -0.154541016f, 0.0249786377f, -0.0320129395f, h0[20], h0[21], h0[22], h0[23]);
        acc = nsf_dot4_acc(acc, 0.0981445312f, -0.086730957f, 0.0461425781f, 0.0648803711f, h0[24], h0[25], h0[26], h0[27]);
        acc = nsf_dot4_acc(acc, 0.0888061523f, -0.0606384277f, 0.0568847656f, -0.0417480469f, h0[28], h0[29], h0[30], h0[31]);
        h1[31] = __half2float(nsf_relu_to_half(acc));
    }

    {
        float acc = -0.0960083008f;
        acc = nsf_dot4_acc(acc, -0.00339508057f, -0.0933227539f, 0.0405883789f, 0.121337891f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0343322754f, 0.0739135742f, 0.0765380859f, 0.053894043f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.137451172f, 0.0319519043f, -0.165893555f, -0.0565185547f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.175170898f, 0.0718383789f, 0.0955810547f, 0.0417480469f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.16015625f, 0.0385131836f, 0.0905151367f, 0.16784668f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0919189453f, 0.0576477051f, 0.224975586f, -0.071472168f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0658569336f, 0.111633301f, -0.0556030273f, -0.110046387f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.126342773f, -0.00654983521f, -0.0434570312f, 0.0328979492f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[0] = __float2half_rn(acc);
    }
    {
        float acc = 0.0974731445f;
        acc = nsf_dot4_acc(acc, -0.0933837891f, 0.221069336f, -0.0158691406f, -0.061706543f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.147827148f, 0.0922851562f, 0.0524902344f, -0.0304107666f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.168457031f, -0.0483703613f, 0.153320312f, 0.0737304688f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.088684082f, 0.185668945f, -0.131591797f, -0.0552368164f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.173583984f, 0.194213867f, 0.203491211f, -0.0566101074f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.0839233398f, 0.0895385742f, 0.146240234f, 0.0752563477f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.0368347168f, -0.112426758f, 0.201416016f, -0.0615234375f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.103027344f, -0.013381958f, 0.0783081055f, 0.063293457f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[1] = __float2half_rn(acc);
    }
    {
        float acc = -0.125976562f;
        acc = nsf_dot4_acc(acc, 0.101806641f, -0.0396728516f, -0.0678710938f, -0.0821533203f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0856323242f, -0.106750488f, 0.0514221191f, 0.00274467468f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.170166016f, -0.0982055664f, -0.104492188f, 0.00622940063f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.10925293f, 0.032409668f, 0.0633544922f, -0.0823364258f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.0140686035f, 0.0133666992f, -0.0397644043f, -0.0308532715f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.0729980469f, 0.0638427734f, -0.0777587891f, 0.0633544922f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.162353516f, 0.146484375f, -0.0271911621f, 0.0979614258f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.104248047f, 0.210327148f, -0.133666992f, -0.16015625f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[2] = __float2half_rn(acc);
    }
    {
        float acc = -0.0872192383f;
        acc = nsf_dot4_acc(acc, -0.139526367f, -0.110412598f, 0.02394104f, -0.051361084f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0422973633f, 0.0184783936f, 0.0152053833f, -0.022567749f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.104858398f, -0.142456055f, 0.0812988281f, -0.0136260986f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.035949707f, 0.131469727f, -0.0437927246f, 0.00196647644f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.116516113f, -0.0961914062f, 0.121948242f, -0.0310211182f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.113586426f, 0.0587158203f, 0.0383605957f, -0.142700195f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0039024353f, -0.0399475098f, -0.0630493164f, 0.00702285767f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.0668945312f, -0.0885009766f, 0.0581054688f, 0.120788574f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[3] = __float2half_rn(acc);
    }
    {
        float acc = -0.0461730957f;
        acc = nsf_dot4_acc(acc, 0.0565185547f, 0.205200195f, 0.0825195312f, -0.145385742f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0695800781f, -0.024017334f, 0.0723266602f, 0.0240478516f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.142211914f, 0.0253143311f, -0.120727539f, 0.000777244568f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.147094727f, -0.0100631714f, 0.0869140625f, 0.0435791016f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.167602539f, -0.110107422f, 0.0902099609f, -0.147338867f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.138427734f, -0.0692749023f, 0.195678711f, -0.0412902832f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.0728759766f, -0.119384766f, 0.139160156f, 0.00559616089f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.0809936523f, 0.0771484375f, 0.069519043f, 0.110778809f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[4] = __float2half_rn(acc);
    }
    {
        float acc = 0.00663375854f;
        acc = nsf_dot4_acc(acc, 0.0364990234f, 0.0638427734f, -0.0114822388f, -0.142089844f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0404663086f, 0.185913086f, -0.10723877f, -0.0764160156f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.175415039f, 0.0146636963f, -0.163085938f, -0.0125427246f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.162841797f, 0.113952637f, 0.0941772461f, 0.0741577148f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.149169922f, -0.0317687988f, -0.116088867f, -0.101257324f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0628051758f, -0.0101242065f, 0.20715332f, -0.112670898f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.150878906f, 0.174560547f, -0.120544434f, 0.200439453f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.106262207f, -0.137573242f, 0.17565918f, -0.0504455566f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[5] = __float2half_rn(acc);
    }
    {
        float acc = -0.165283203f;
        acc = nsf_dot4_acc(acc, -0.0838012695f, -0.0932617188f, -0.149536133f, 0.0382385254f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.173583984f, 0.0667114258f, -0.0179901123f, 0.0765991211f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.0432739258f, 0.0679321289f, -0.041595459f, 0.141967773f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0718994141f, -0.112915039f, 0.0225067139f, -0.0938720703f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.107910156f, -0.128173828f, -0.0494384766f, -0.110839844f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.140258789f, 0.151733398f, 0.0602722168f, 0.146240234f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.085144043f, -0.166503906f, -0.00867462158f, 0.0752563477f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.0302276611f, 0.0466918945f, -0.0534362793f, -0.0872802734f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[6] = __float2half_rn(acc);
    }
    {
        float acc = -0.054473877f;
        acc = nsf_dot4_acc(acc, -0.0697021484f, -0.0996704102f, 0.146362305f, 0.0989379883f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.167358398f, -0.0408325195f, 0.0576477051f, -0.0551452637f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.146240234f, 0.0653076172f, -0.124572754f, -0.0439758301f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0791625977f, -0.219604492f, 0.00634002686f, -0.0454711914f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.0182647705f, 0.118713379f, -0.0227661133f, -0.0497741699f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.0982055664f, -0.13684082f, -0.0617370605f, -0.132080078f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.116638184f, -0.0233764648f, -0.0925292969f, -0.0147018433f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.106994629f, 0.0712280273f, -0.182128906f, 0.0786132812f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[7] = __float2half_rn(acc);
    }
    {
        float acc = 0.0182189941f;
        acc = nsf_dot4_acc(acc, -0.0905151367f, 0.0355224609f, 0.165039062f, 0.0231628418f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.168579102f, 0.154418945f, 0.0198669434f, -0.0823364258f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.0933837891f, 0.147827148f, -0.0210876465f, 0.168945312f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.116210938f, -0.052947998f, 0.015007019f, -0.116455078f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.143920898f, -0.0646362305f, 0.111999512f, 0.0472717285f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.110412598f, -0.117126465f, 0.00933837891f, 0.0928955078f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.00705718994f, -0.00253677368f, 0.0980834961f, -0.0491943359f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.016204834f, 0.0783691406f, 0.0761108398f, 0.162719727f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[8] = __float2half_rn(acc);
    }
    {
        float acc = -0.0829467773f;
        acc = nsf_dot4_acc(acc, -0.0519104004f, 0.0834350586f, -0.0869750977f, -0.0373535156f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0887451172f, 0.111083984f, -0.00370407104f, 0.0684204102f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0972290039f, 0.0175476074f, -0.0341186523f, 0.0180664062f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.149780273f, 0.0160217285f, -0.0391540527f, -0.158081055f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.158203125f, -0.010093689f, -0.0899047852f, -0.184692383f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.165649414f, 0.0603942871f, -0.0756225586f, -0.0770874023f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.102294922f, 0.142700195f, -0.0437011719f, 0.135864258f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.162475586f, -0.131591797f, 0.117797852f, -0.0290374756f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[9] = __float2half_rn(acc);
    }
    {
        float acc = 0.0131759644f;
        acc = nsf_dot4_acc(acc, 0.130981445f, 0.00228118896f, -0.171264648f, 0.0140228271f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0621337891f, -0.0969848633f, 0.110656738f, -0.200927734f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.0560913086f, 0.147583008f, 0.221679688f, 0.0178527832f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.105529785f, -0.232543945f, -0.143676758f, -0.123962402f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0925292969f, 0.00266838074f, -0.0767822266f, 0.137817383f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.135742188f, -0.132446289f, -0.118713379f, -0.0272369385f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0366821289f, -0.00126552582f, -0.0102233887f, 0.0437927246f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.126708984f, 0.00772857666f, -0.018737793f, 0.071105957f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[10] = __float2half_rn(acc);
    }
    {
        float acc = 0.157104492f;
        acc = nsf_dot4_acc(acc, 0.0379943848f, -0.056854248f, -0.0787963867f, 0.0590820312f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.11315918f, 0.160522461f, -0.139282227f, 0.133544922f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0709228516f, -0.0583190918f, 0.0900268555f, -0.0847167969f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0376586914f, 0.0689697266f, 0.0180511475f, 0.0166015625f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.0107803345f, -0.0591125488f, 0.141845703f, 0.0535888672f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0622253418f, 0.114440918f, -0.0852661133f, -0.0198059082f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.00503921509f, 0.142700195f, -0.129272461f, -0.128051758f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.0841674805f, -0.125854492f, -0.133422852f, 0.120422363f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[11] = __float2half_rn(acc);
    }
    {
        float acc = -0.00894165039f;
        acc = nsf_dot4_acc(acc, 0.0567321777f, -0.0880126953f, -0.0291900635f, -0.000988006592f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0628662109f, -0.105041504f, 0.0702514648f, 0.0101852417f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.172607422f, -0.141967773f, -0.0382995605f, -0.0461730957f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.12310791f, 0.169311523f, 0.13671875f, 0.16796875f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.10723877f, 0.116882324f, 0.162109375f, -0.0861206055f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0159912109f, 0.0844726562f, 0.163330078f, -0.105895996f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.176025391f, -0.0444335938f, 0.104187012f, -0.109130859f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.166503906f, -0.0985107422f, 0.159179688f, -0.086730957f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[12] = __float2half_rn(acc);
    }
    {
        float acc = -0.0772094727f;
        acc = nsf_dot4_acc(acc, 0.184204102f, -0.0344543457f, 0.0703125f, -0.0251617432f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0559082031f, -0.1640625f, -0.0804443359f, 0.161376953f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.00960540771f, -0.0592346191f, -0.143676758f, -0.0756835938f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.175415039f, 0.0325317383f, 0.111633301f, -0.0777587891f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.103515625f, -0.0891723633f, -0.18371582f, -0.162109375f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0896606445f, -0.156005859f, 0.081237793f, 0.0188446045f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0960693359f, -0.13293457f, 0.208129883f, -0.315429688f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.217529297f, 0.166259766f, 0.00326156616f, -0.0998535156f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[13] = __float2half_rn(acc);
    }
    {
        float acc = -0.275146484f;
        acc = nsf_dot4_acc(acc, -0.106079102f, -0.382324219f, -0.153808594f, -0.0247039795f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.134887695f, -0.0959472656f, 0.0498046875f, -0.194824219f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.160888672f, 0.0188751221f, -0.0335998535f, -0.164916992f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0275878906f, -0.0362548828f, 0.135620117f, -0.0147857666f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0292510986f, -0.0765380859f, -0.201049805f, 0.136962891f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0523681641f, 0.0108642578f, -0.171142578f, -0.324707031f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0349121094f, 0.0718994141f, 0.0239562988f, -0.0880126953f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.0751953125f, 0.0352783203f, -0.188598633f, 0.156738281f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[14] = __float2half_rn(acc);
    }
    {
        float acc = -0.073059082f;
        acc = nsf_dot4_acc(acc, 0.0809936523f, 0.19934082f, 0.126098633f, -0.121398926f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0921020508f, 0.247680664f, -0.102966309f, 0.0600585938f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.107910156f, -0.0238800049f, 0.0313110352f, 0.210693359f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0513305664f, -0.0693359375f, 0.0954589844f, 0.185302734f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.0253753662f, 0.232177734f, 0.231323242f, -0.0201873779f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.00901794434f, 0.115539551f, 0.0410766602f, 0.0109710693f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.0797729492f, -0.100524902f, 0.14453125f, 0.062286377f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.0595703125f, 0.224243164f, 0.0960083008f, 0.127929688f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[15] = __float2half_rn(acc);
    }
    {
        float acc = -0.0891113281f;
        acc = nsf_dot4_acc(acc, 0.184204102f, -0.032989502f, 0.0263671875f, 0.16394043f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.121643066f, -0.0542907715f, 0.0250549316f, -0.00693511963f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0470581055f, 0.090637207f, -0.0693359375f, -0.00649642944f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.159912109f, -0.0829467773f, -0.119873047f, -0.123840332f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.105651855f, -0.164306641f, -0.0842285156f, 0.119384766f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.105957031f, 0.142700195f, -0.0795898438f, 0.0355529785f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.023147583f, -0.00433731079f, 0.127685547f, 0.110290527f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.136352539f, 0.123168945f, 0.0530395508f, -0.00329399109f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[16] = __float2half_rn(acc);
    }
    {
        float acc = -0.183959961f;
        acc = nsf_dot4_acc(acc, -0.185668945f, 0.00956726074f, 0.128295898f, -0.154296875f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.163208008f, -0.227172852f, 0.0389404297f, -0.105041504f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.054901123f, 0.159179688f, -0.103393555f, -0.149414062f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0158996582f, -0.163818359f, -0.0517883301f, 0.0406799316f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0398864746f, -0.169677734f, 0.110839844f, 0.116943359f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.120483398f, 0.0252075195f, 0.0194396973f, -0.182983398f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.124267578f, -0.146362305f, 0.0183868408f, 0.00727462769f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.039276123f, -0.147583008f, 0.073425293f, 0.022567749f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[17] = __float2half_rn(acc);
    }
    {
        float acc = -0.065246582f;
        acc = nsf_dot4_acc(acc, -0.101623535f, -0.0639648438f, 0.0902099609f, 0.0466003418f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0270385742f, -0.123962402f, -0.108154297f, 0.154907227f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.101257324f, 0.129882812f, -0.0486755371f, -0.0204925537f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.172119141f, -0.0396118164f, 0.149047852f, -0.148925781f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.089050293f, -0.151367188f, 0.0270996094f, -0.157592773f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.139892578f, -0.139526367f, -0.0648803711f, 0.0193328857f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.129638672f, 0.0447998047f, 0.131835938f, 0.142089844f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.142456055f, 0.131347656f, -0.0475769043f, -0.0162963867f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[18] = __float2half_rn(acc);
    }
    {
        float acc = 0.0381774902f;
        acc = nsf_dot4_acc(acc, -0.0205535889f, 0.0789794922f, -0.046875f, 0.129638672f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.154541016f, 0.106994629f, 0.09375f, -0.167602539f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.144165039f, 0.138427734f, 0.0254211426f, 0.0115890503f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0690917969f, 0.00798034668f, 0.0797729492f, -0.12109375f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.141235352f, 0.0743408203f, -0.00228691101f, 0.0872192383f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.165893555f, -0.128295898f, -0.0245513916f, -0.0183563232f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.0241699219f, 0.0667724609f, 0.000317811966f, -0.149414062f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.0523376465f, -0.0612182617f, -0.180175781f, 0.00596618652f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[19] = __float2half_rn(acc);
    }
    {
        float acc = 0.0554199219f;
        acc = nsf_dot4_acc(acc, -0.00312042236f, -0.10357666f, -0.161132812f, 0.143432617f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.10736084f, 0.0847167969f, 0.141845703f, -0.0342102051f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.115844727f, -0.171264648f, 0.141113281f, -0.0531921387f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0128936768f, 0.0610351562f, -0.0500793457f, 0.14465332f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.0244140625f, -0.0227966309f, -0.0720825195f, 0.0462646484f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.110351562f, 0.0829467773f, -0.0132675171f, 0.110595703f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.0866699219f, -0.087097168f, -0.0475158691f, 0.0286407471f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.131713867f, -0.130004883f, 0.0741577148f, 0.102416992f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[20] = __float2half_rn(acc);
    }
    {
        float acc = 0.121826172f;
        acc = nsf_dot4_acc(acc, 0.00362205505f, -0.0162353516f, 0.096496582f, -0.094909668f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.176147461f, -0.032989502f, -0.142822266f, -0.166992188f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.044921875f, 0.142089844f, -0.146484375f, 0.00807952881f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.128051758f, -0.037322998f, 0.0503845215f, -0.0671386719f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.131591797f, -0.166137695f, -0.0466918945f, -0.0975341797f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.153808594f, 0.11730957f, 0.147216797f, 0.140258789f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0700073242f, -0.135375977f, -0.191650391f, 0.0333862305f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.140258789f, 0.0718383789f, 0.042175293f, 0.046875f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[21] = __float2half_rn(acc);
    }
    {
        float acc = 0.158447266f;
        acc = nsf_dot4_acc(acc, -0.205322266f, 0.071105957f, -0.0187072754f, -0.0762939453f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0922851562f, -0.0662231445f, -0.0803833008f, 0.02293396f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.117858887f, 0.173461914f, -0.011390686f, 0.104187012f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0935058594f, -0.250976562f, -0.0982666016f, -0.169799805f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.0904541016f, -0.185546875f, 0.071105957f, 0.0993652344f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.105957031f, -0.125366211f, 0.100585938f, -0.0526428223f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.0458068848f, 0.0951538086f, 0.0791625977f, -0.141357422f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.216552734f, -0.056427002f, 0.0287780762f, -0.179443359f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[22] = __float2half_rn(acc);
    }
    {
        float acc = -0.144775391f;
        acc = nsf_dot4_acc(acc, -0.0725097656f, -0.0681152344f, -0.110656738f, -0.0743408203f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.000823497772f, -0.150268555f, -0.128173828f, 0.0553894043f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0525817871f, -0.187866211f, -0.0641479492f, 0.0620117188f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.00649642944f, -0.0632324219f, -0.0148010254f, -0.00765609741f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.128051758f, -0.133544922f, 0.0409851074f, -0.127319336f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0838623047f, 0.0684204102f, -0.21496582f, -0.0888061523f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.150878906f, -0.0106887817f, -0.0679321289f, 0.0803222656f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.0775146484f, -0.178344727f, -0.0618286133f, 0.123657227f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[23] = __float2half_rn(acc);
    }
    {
        float acc = -0.0362548828f;
        acc = nsf_dot4_acc(acc, -0.0401611328f, -0.110595703f, 0.0610961914f, -0.0398864746f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0372924805f, 0.0756835938f, 0.0477905273f, -0.0565185547f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.0816040039f, -0.0141830444f, -0.126464844f, 0.0530090332f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.120849609f, 0.128662109f, 0.0916748047f, 0.0676269531f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0455322266f, 0.0605773926f, 0.138305664f, 0.046661377f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.030670166f, -0.0703125f, 0.105651855f, -0.112854004f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0557556152f, -0.153198242f, -0.0383300781f, 0.122009277f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.0220031738f, 0.173217773f, -0.0649414062f, 0.166870117f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[24] = __float2half_rn(acc);
    }
    {
        float acc = -0.0323486328f;
        acc = nsf_dot4_acc(acc, -0.0891723633f, -0.0333862305f, 0.107788086f, -0.14855957f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.111999512f, -0.219360352f, 0.0627441406f, -0.0233306885f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.152099609f, 0.166137695f, -0.140136719f, -0.0422058105f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.00422286987f, -0.181640625f, 0.0487670898f, -0.0068359375f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.0505065918f, -0.00810241699f, 0.0162963867f, -0.120239258f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.0380859375f, -0.075378418f, -0.153686523f, -0.0342102051f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.00649642944f, -0.142822266f, -0.0169372559f, 0.256835938f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.128295898f, -0.154296875f, 0.0694580078f, -0.120666504f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[25] = __float2half_rn(acc);
    }
    {
        float acc = -0.185058594f;
        acc = nsf_dot4_acc(acc, -0.0328674316f, -0.0473327637f, 0.140625f, 0.157348633f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.15222168f, -0.121948242f, 0.0235290527f, -0.183837891f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.00390434265f, 0.117370605f, 0.292236328f, -0.283203125f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, -0.0760498047f, -0.197265625f, 0.0832519531f, -0.079284668f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.135620117f, -0.116088867f, 0.0100784302f, 0.00452423096f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.163330078f, -0.0473937988f, -0.014793396f, -0.11932373f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0210876465f, 0.00633239746f, 0.079284668f, 0.0250244141f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.224243164f, -0.274902344f, -0.219970703f, -0.0643920898f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[26] = __float2half_rn(acc);
    }
    {
        float acc = 0.128417969f;
        acc = nsf_dot4_acc(acc, 0.0969238281f, 0.189697266f, -0.017364502f, 0.0615844727f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.08984375f, -0.121826172f, -0.182861328f, 0.092590332f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0823974609f, -0.0854492188f, 0.0528564453f, -0.183349609f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0022277832f, 0.107177734f, 0.00167369843f, 0.00409698486f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0709228516f, -0.0394287109f, 0.159790039f, -0.164794922f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.103271484f, -0.129394531f, -0.131347656f, 0.159912109f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.043762207f, -0.138916016f, 0.0471191406f, -0.216186523f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.00929260254f, 0.11340332f, -0.000953197479f, 0.090637207f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[27] = __float2half_rn(acc);
    }
    {
        float acc = 0.0450134277f;
        acc = nsf_dot4_acc(acc, 0.193725586f, -0.181762695f, 0.0347900391f, 0.12286377f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0409851074f, -0.0310058594f, 0.0250854492f, 0.0980224609f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0459594727f, -0.157592773f, 0.128540039f, -0.0112457275f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0218353271f, 0.151611328f, -0.142578125f, -0.00189495087f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, -0.00148010254f, 0.0123291016f, -0.112854004f, 0.0296783447f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.0555114746f, 0.10949707f, -0.125610352f, 0.0839233398f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.119750977f, -0.137329102f, 0.104736328f, -0.154541016f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.041595459f, 0.146850586f, 0.129882812f, -0.124694824f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[28] = __float2half_rn(acc);
    }
    {
        float acc = -0.297607422f;
        acc = nsf_dot4_acc(acc, 0.0214385986f, -0.226928711f, 0.114257812f, 0.0299530029f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, -0.0196685791f, -0.0282287598f, 0.160400391f, -0.0336608887f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.148803711f, 0.0224151611f, -0.115112305f, -0.0895996094f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.126953125f, -0.285888672f, -0.0102157593f, -0.0293884277f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.116516113f, -0.264648438f, -0.00778579712f, -0.0261383057f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, -0.149780273f, -0.0386657715f, -0.0951538086f, -0.161621094f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.120178223f, -0.128051758f, -0.341552734f, -0.0196228027f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.336669922f, -0.16784668f, -0.180297852f, 0.0795288086f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[29] = __float2half_rn(acc);
    }
    {
        float acc = -0.389160156f;
        acc = nsf_dot4_acc(acc, -0.17199707f, -0.354736328f, 0.0308837891f, 0.169189453f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0290985107f, -0.350341797f, 0.00522232056f, -0.426269531f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, -0.11340332f, -0.0208129883f, -0.333496094f, -0.0720825195f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0343017578f, -0.146240234f, -0.0402526855f, -0.2890625f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0457763672f, -0.298828125f, -0.341796875f, -0.165283203f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.134765625f, -0.0876464844f, -0.429199219f, -0.339111328f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, -0.133666992f, -0.0717773438f, -0.130981445f, -0.0888671875f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, -0.219848633f, -0.187866211f, -0.2109375f, 0.0959472656f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[30] = __float2half_rn(acc);
    }
    {
        float acc = 0.0f;
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[0], h1[1], h1[2], h1[3]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[4], h1[5], h1[6], h1[7]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[8], h1[9], h1[10], h1[11]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[12], h1[13], h1[14], h1[15]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[16], h1[17], h1[18], h1[19]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[20], h1[21], h1[22], h1[23]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[24], h1[25], h1[26], h1[27]);
        acc = nsf_dot4_acc(acc, 0.0f, 0.0f, 0.0f, 0.0f, h1[28], h1[29], h1[30], h1[31]);
        y_out.lane[31] = __float2half_rn(acc);
    }
}

} // namespace nsf_simd_hardcoded_generated

using VOut = nsf_simd_hardcoded_generated::PhiVec;

static __forceinline__ __device__
float phi_f(const VOut& y, int i)
{
    return __half2float(y.lane[i]);
}

static __noinline__ __device__
void evalHyperMLP_phi62_vec(const LaunchParams& lp, int t,
                            float c0, float c1, float c2,
                            VOut& y_out)
{
    (void)lp;
    switch (t) {
        case 0: nsf_simd_hardcoded_generated::eval_transform_0(c0, c1, c2, y_out); break;
        case 1: nsf_simd_hardcoded_generated::eval_transform_1(c0, c1, c2, y_out); break;
        default:
            #pragma unroll
            for (int i = 0; i < PHI_PAD; ++i) y_out.lane[i] = __float2half_rn(0.0f);
            break;
    }
}