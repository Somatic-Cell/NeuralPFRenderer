#pragma once

#include "phase_function.cuh"

#include <optix_device.h> 
#include <cuda_fp16.h>      // half を使うなら（OptixCoopVec<half, N> 等）

// 例：固定ネット（config 前提）
static constexpr int NSF_T  = 3;    // transforms
static constexpr int C      = 3;    // context
static constexpr int H      = 64;   // hidden
static constexpr int K      = 32;   // bins
static constexpr int PHI    = 3*K-1; // 95

// 実際の CoopVec サイズは padding 後（launchParams.nsf.inputPad や N/K を参照して決める）
// 典型：inputPad=16, phiPad=96 など
static constexpr int IN_PAD  = 16;
static constexpr int PHI_PAD = 96;

// 型は状況に応じて（half/float）
// まずは matmul は half のまま、後段の RQS は float に落とすのが無難です
using VIn   = OptixCoopVec<half, IN_PAD>;
using VHid  = OptixCoopVec<half, H>;
using VOut  = OptixCoopVec<half, PHI_PAD>;

static __forceinline__ __device__
VHid relu(const VHid& x)
{
    // ReLU: max(x, 0)
    return optixCoopVecMax(x, half(0.0f)); // あなたの OptiX ヘッダの名前に合わせてください
}

static __forceinline__ __device__
VHid evalHyperLayer0(const LaunchParams& lp, int t, const VIn& in)
{
    const CUdeviceptr base = lp.nsf.packedBase;
    const uint32_t wOff = lp.nsf.wOffset[t][0];
    const uint32_t bOff = lp.nsf.bOffset[t][0];

    VHid out;
    out = optixCoopVecMatMul<
        VHid, VIn,
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
        OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
        false,
        H, IN_PAD,
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16
    >(in, base, wOff, base, bOff);

    return out;
}

static __forceinline__ __device__
VHid evalHyperLayer1(const LaunchParams& lp, int t, const VHid& in)
{
    const CUdeviceptr base = lp.nsf.packedBase;
    const uint32_t wOff = lp.nsf.wOffset[t][1];
    const uint32_t bOff = lp.nsf.bOffset[t][1];

    VHid out;
    out = optixCoopVecMatMul<
        VHid, VHid,
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
        OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
        false,
        H, H,
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16
    >(in, base, wOff, base, bOff);

    return out;
}

static __forceinline__ __device__
VOut evalHyperLayer2(const LaunchParams& lp, int t, const VHid& in)
{
    const CUdeviceptr base = lp.nsf.packedBase;
    const uint32_t wOff = lp.nsf.wOffset[t][2];
    const uint32_t bOff = lp.nsf.bOffset[t][2];

    VOut out;
    out = optixCoopVecMatMul<
        VOut, VHid,
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
        OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
        false,
        PHI_PAD, H,
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16, OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16
    >(in, base, wOff, base, bOff);

    return out;
}

static __forceinline__ __device__
void evalHyperMLP_phi95(const LaunchParams& lp, int t, const float c0, const float c1, const float c2,
                        float phi_out[PHI])
{
    // 入力パディング
    alignas(16) VIn x(half(0.0f));
    x[0] = __float2half(c0);
    x[1] = __float2half(c1);
    x[2] = __float2half(c2);

    // 3->64->64->phiPad
    alignas(16) VHid h0 = relu(evalHyperLayer0(lp, t, x));
    alignas(16) VHid h1 = relu(evalHyperLayer1(lp, t, h0));
    alignas(16) VOut y  = evalHyperLayer2(lp, t, h1);

    // 必要な 95 次元だけ float に取り出す
    #pragma unroll
    for (int i = 0; i < PHI; ++i)
        phi_out[i] = __half2float(y[i]);
}

// ---- NSF constants (zuko 1.5.0 defaults unless overridden) ----
static constexpr int   NSF_BINS  = 32;
static constexpr int   NSF_PHI   = 3 * NSF_BINS - 1; // 95
static constexpr float NSF_BOUND = 5.0f;
static constexpr float NSF_SLOPE = 1e-3f;

static constexpr float NSF_SCALE = 3.0f;   // t = atanh(mu) / scale
static constexpr float NSF_EPS   = 1e-6f;  // clamp

// zuko: searchsorted(seq, x) = sum(seq < x)
static __forceinline__ __device__ int searchsorted_strict_lt(const float* seq, int n, float x) {
    int cnt = 0;
    #pragma unroll
    for (int i = 0; i < n; ++i) cnt += (seq[i] < x);
    return cnt;
}

static __forceinline__ __device__
void split_phi(const float phi[NSF_PHI], float w[NSF_BINS], float h[NSF_BINS], float d[NSF_BINS - 1])
{
    #pragma unroll
    for (int i = 0; i < NSF_BINS; ++i) w[i] = phi[i];
    #pragma unroll
    for (int i = 0; i < NSF_BINS; ++i) h[i] = phi[NSF_BINS + i];
    #pragma unroll
    for (int i = 0; i < NSF_BINS - 1; ++i) d[i] = phi[2 * NSF_BINS + i];
}

struct Rqs1D {
    float xk[NSF_BINS + 1]; // horizontal knots
    float yk[NSF_BINS + 1]; // vertical knots
    float dk[NSF_BINS + 1]; // derivatives at knots (positive)
};

static __forceinline__ __device__
float saturate_param(float v, float log_slope, float factor) {
    // zuko:
    // widths  = w / (1 + abs(2*w/log(slope)))
    // heights = h / (1 + abs(2*h/log(slope)))
    // deriv   = d / (1 + abs(d/log(slope)))
    const float denom = 1.0f + fabsf(factor * v / log_slope);
    return v / denom;
}

static __forceinline__ __device__
void softmax_32(const float in[NSF_BINS], float out[NSF_BINS])
{
    float m = in[0];
    #pragma unroll
    for (int i = 1; i < NSF_BINS; ++i) m = fmaxf(m, in[i]);

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < NSF_BINS; ++i) {
        out[i] = __expf(in[i] - m);
        sum += out[i];
    }
    const float inv = 1.0f / fmaxf(sum, 1e-20f);
    #pragma unroll
    for (int i = 0; i < NSF_BINS; ++i) out[i] *= inv;
}

// zuko 1.5.0 __init__ を反映して RQS のノット列/導関数を構築
static __forceinline__ __device__
void build_rqs_from_phi(const float phi[NSF_PHI], Rqs1D& rqs,
                        float bound = NSF_BOUND, float slope = NSF_SLOPE)
{
    float w_raw[NSF_BINS], h_raw[NSF_BINS], d_raw[NSF_BINS - 1];
    split_phi(phi, w_raw, h_raw, d_raw);

    const float log_slope = __logf(slope); // < 0

    // saturation
    #pragma unroll
    for (int i = 0; i < NSF_BINS; ++i) {
        w_raw[i] = saturate_param(w_raw[i], log_slope, 2.0f);
        h_raw[i] = saturate_param(h_raw[i], log_slope, 2.0f);
    }
    #pragma unroll
    for (int i = 0; i < NSF_BINS - 1; ++i) {
        d_raw[i] = saturate_param(d_raw[i], log_slope, 1.0f);
    }

    // widths/heights: softmax then pad left with 0, then cumsum, then scale to [-B,B]
    float w_sm[NSF_BINS], h_sm[NSF_BINS];
    softmax_32(w_raw, w_sm);
    softmax_32(h_raw, h_sm);

    // cumulative with pad(1,0) value=0:
    float cw = 0.0f;
    float ch = 0.0f;

    // i=0 is padded 0 -> cumsum(0)=0
    rqs.xk[0] = -bound;
    rqs.yk[0] = -bound;

    #pragma unroll
    for (int i = 1; i <= NSF_BINS; ++i) {
        cw += w_sm[i - 1];
        ch += h_sm[i - 1];
        rqs.xk[i] = bound * (2.0f * cw - 1.0f);
        rqs.yk[i] = bound * (2.0f * ch - 1.0f);
    }
    // 理論上 rqs.xk[NSF_BINS]=+B, rqs.yk[NSF_BINS]=+B

    // derivatives: pad(1,1) with 0 then exp
    rqs.dk[0] = 1.0f; // exp(0)
    #pragma unroll
    for (int i = 1; i < NSF_BINS; ++i) {
        rqs.dk[i] = __expf(d_raw[i - 1]);
    }
    rqs.dk[NSF_BINS] = 1.0f; // exp(0)
}

static __forceinline__ __device__
void rqs_forward_and_ladj(const Rqs1D& rqs, float x, float& y, float& ladj)
{
    // k = searchsorted(horizontal, x) - 1
    const int nKnots = NSF_BINS + 1;
    int k = searchsorted_strict_lt(rqs.xk, nKnots, x) - 1;

    // mask: 0 <= k < bins
    if (k < 0 || k >= NSF_BINS) {
        y = x;
        ladj = 0.0f;
        return;
    }

    const float x0 = rqs.xk[k];
    const float x1 = rqs.xk[k + 1];
    const float y0 = rqs.yk[k];
    const float y1 = rqs.yk[k + 1];
    const float d0 = rqs.dk[k];
    const float d1 = rqs.dk[k + 1];

    const float inv_dx = 1.0f / fmaxf(x1 - x0, 1e-20f);
    const float s = (y1 - y0) * inv_dx;

    const float z = (x - x0) * inv_dx; // in [0,1]
    const float omz = 1.0f - z;
    const float z_omz = z * omz;

    const float denom = s + (d0 + d1 - 2.0f * s) * z_omz;
    const float numer = s * z * z + d0 * z * omz;

    y = y0 + (y1 - y0) * numer / fmaxf(denom, 1e-20f);

    // jacobian (zuko)
    const float term = 2.0f * s * z_omz + d0 * omz * omz + d1 * z * z;
    const float J = (s * s) * term / fmaxf(denom * denom, 1e-20f);

    ladj = logf(fmaxf(J, 1e-30f));
}

static __forceinline__ __device__
void rqs_inverse_and_ladj_fwd(const Rqs1D& rqs, float y, float& x, float& ladj_fwd)
{
    const int nKnots = NSF_BINS + 1;
    int k = searchsorted_strict_lt(rqs.yk, nKnots, y) - 1;

    if (k < 0 || k >= NSF_BINS) {
        x = y;
        ladj_fwd = 0.0f;
        return;
    }

    const float x0 = rqs.xk[k];
    const float x1 = rqs.xk[k + 1];
    const float y0 = rqs.yk[k];
    const float y1 = rqs.yk[k + 1];
    const float d0 = rqs.dk[k];
    const float d1 = rqs.dk[k + 1];

    const float inv_dx = 1.0f / fmaxf(x1 - x0, 1e-20f);
    const float s = (y1 - y0) * inv_dx;

    const float y_ = (y - y0); // mask * (y-y0)

    const float a = (y1 - y0) * (s - d0) + y_ * (d0 + d1 - 2.0f * s);
    const float b = (y1 - y0) * d0       - y_ * (d0 + d1 - 2.0f * s);
    const float c = -s * y_;

    const float disc = fmaxf(b * b - 4.0f * a * c, 0.0f);
    const float sqrt_disc = sqrtf(disc);

    // zuko: z = 2c / (-b - sqrt(...))
    const float denom = (-b - sqrt_disc);
    const float z = (fabsf(denom) > 1e-20f) ? (2.0f * c / denom) : 0.0f;

    x = x0 + z * (x1 - x0);

    // ladj_fwd at x: same as forward jacobian formula, using z
    const float omz = 1.0f - z;
    const float z_omz = z * omz;

    const float denom2 = s + (d0 + d1 - 2.0f * s) * z_omz;
    const float term = 2.0f * s * z_omz + d0 * omz * omz + d1 * z * z;
    const float J = (s * s) * term / fmaxf(denom2 * denom2, 1e-20f);

    ladj_fwd = logf(fmaxf(J, 1e-30f));
}

static __forceinline__ __device__
float sampleMuHG(float u, float g)
{
    g = clamp(g, -0.999999f, 0.999999f);
    u = clamp(u, 0.0f, 0.99999994f);

    if (fabsf(g) < 1e-4f) {
        return 1.0f - 2.0f * u;
    } else {
        const float gg = g * g;
        const float t  = (1.0f - gg) / (1.0f - g + 2.0f * g * u);
        float mu = (1.0f + gg - t * t) / (2.0f * g);
        return clamp(mu, -1.0f, 1.0f);
    }
}

static __forceinline__ __device__
float logpdfMuHG(float mu, float g)
{
    g  = clamp(g, -0.999999f, 0.999999f);
    mu = clamp(mu, -1.0f, 1.0f);

    if (fabsf(g) < 1e-4f) {
        return logf(0.5f);
    } else {
        const float gg = g * g;
        const float denom = 1.0f + gg - 2.0f * g * mu;
        const float denom32 = denom * sqrtf(fmaxf(denom, 1e-20f));
        const float q = (1.0f - gg) / (2.0f * fmaxf(denom32, 1e-20f));
        return logf(fmaxf(q, 1e-30f));
    }
}

static __forceinline__ __device__
float logpdfBaseT(float t, float g)
{
    // mu = tanh(scale*t)
    const float x = NSF_SCALE * t;
    float mu = tanhf(x);
    mu = clamp(mu, -1.0f + NSF_EPS, 1.0f - NSF_EPS);

    const float logq = logpdfMuHG(mu, g);
    const float jac  = NSF_SCALE * fmaxf(1.0f - mu * mu, 1e-30f); // dmu/dt
    return logq + logf(jac);
}

static __forceinline__ __device__
float sampleBaseT(float u, float g)
{
    float mu = sampleMuHG(u, g);
    mu = clamp(mu, -1.0f + NSF_EPS, 1.0f - NSF_EPS);
    return atanhf(mu) / NSF_SCALE;
}


// static __forceinline__ __device__
// float flow_logpdf_t(const LaunchParams& lp, float t, float c0, float c1, float c2)
// {
//     // inverse 方向で base に戻しつつ、各段の ladj_fwd を積算
//     float y = t;
//     float sum_ladj_fwd = 0.0f;

//     #pragma unroll
//     for (int i = 2; i >= 0; --i) {
//         alignas(16) float phi[NSF_PHI];
//         evalHyperMLP_phi95(lp, i, c0, c1, c2, phi);

//         Rqs1D rqs;
//         build_rqs_from_phi(phi, rqs, NSF_BOUND, NSF_SLOPE);

//         float x, ladj_fwd;
//         rqs_inverse_and_ladj_fwd(rqs, y, x, ladj_fwd);

//         sum_ladj_fwd += ladj_fwd;
//         y = x;
//     }

    // y は t0（base の変数）
//     const float logp0 = logpdfBaseT(y, c2);
//     return logp0 - sum_ladj_fwd;
// }

static __forceinline__ __device__
float flow_logpdf_t(const LaunchParams& lp, float t, float c0, float c1, float c2)
{
    // forward 方向で base(z) に写しつつ、各段の ladj_fwd = log|dz/dx| を積算
    float x = t;
    float sum_ladj_fwd = 0.0f;

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        alignas(16) float phi[NSF_PHI];
        evalHyperMLP_phi95(lp, i, c0, c1, c2, phi);

        Rqs1D rqs;
        build_rqs_from_phi(phi, rqs, NSF_BOUND, NSF_SLOPE);

        float y, ladj_fwd;
        rqs_forward_and_ladj(rqs, x, y, ladj_fwd);   // y = f(x), ladj_fwd = log|dy/dx|
        sum_ladj_fwd += ladj_fwd;
        x = y;
    }

    // x は base の変数 z
    const float logp0 = logpdfBaseT(x, c2);
    return logp0 + sum_ladj_fwd;  // log p(t) = log p(z) + log|dz/dt|
}

// static __forceinline__ __device__
// float flow_sample_t_and_logpdf(const LaunchParams& lp, float u_base, float c0, float c1, float c2, float& out_logp_t)
// {
//     float x = sampleBaseT(u_base, c2);
//     float logp = logpdfBaseT(x, c2);

//     #pragma unroll
//     for (int i = 0; i < 3; ++i) {
//     // for (int i = 0; i < 3; ++i) {
//         alignas(16) float phi[NSF_PHI];
//         evalHyperMLP_phi95(lp, i, c0, c1, c2, phi);

//         alignas(16) Rqs1D rqs;
//         build_rqs_from_phi(phi, rqs, NSF_BOUND, NSF_SLOPE);

//         float y, ladj_fwd;
//         rqs_forward_and_ladj(rqs, x, y, ladj_fwd);

//         // p(y)=p(x)*|dx/dy| => logp -= log|dy/dx|
//         logp -= ladj_fwd;
//         x = y;

//     }

//     out_logp_t = logp;
//     return x; // t
// }


static __forceinline__ __device__
float flow_sample_t_and_logpdf(const LaunchParams& lp,
                               float u_base, float c0, float c1, float c2,
                               float& out_logp_t)
{
    // z ~ base
    float y = sampleBaseT(u_base, c2);
    float logp = logpdfBaseT(y, c2);

    // inverse 方向で data(t) に戻す。ただし logp は forward の ladj_fwd を足す。
    #pragma unroll
    for (int i = 2; i >= 0; --i) {
        alignas(16) float phi[NSF_PHI];
        evalHyperMLP_phi95(lp, i, c0, c1, c2, phi);

        alignas(16) Rqs1D rqs;
        build_rqs_from_phi(phi, rqs, NSF_BOUND, NSF_SLOPE);

        float x, ladj_fwd;
        rqs_inverse_and_ladj_fwd(rqs, y, x, ladj_fwd);  // x = f^{-1}(y), ladj_fwd = log|dy/dx| (forward)
        logp += ladj_fwd;                                // log p(x) = log p(y) + log|dy/dx|
        y = x;
    }

    out_logp_t = logp;
    return y; // t
}

static __forceinline__ __device__
float evalPhaseFunctionNF(const LaunchParams& lp, float cosTheta, float c0, float c1, float g)
{
    // cosTheta は与えられるので t に戻す（atanh 安定化）
    float mu = clamp(cosTheta, -1.0f + NSF_EPS, 1.0f - NSF_EPS);
    const float t = atanhf(mu) / NSF_SCALE;

    const float logp_t = flow_logpdf_t(lp, t, c0, c1, g);

    // p_omega = p_t / (2π * scale * (1-mu^2))
    const float denom = (2.0f * float(M_PI)) * NSF_SCALE * fmaxf(1.0f - mu * mu, 1e-30f);
    const float pdf = __expf(logp_t) / denom;
    return pdf;
}

static __forceinline__ __device__
PhaseSample samplePhaseFunctionNF(const LaunchParams& lp,
                                  const float3& woWorld, float u1, float u2,
                                  float c0, float c1, float g)
{
    PhaseSample ps;
    const float3 wo = normalize(woWorld);

    float logp_t;
    const float t = flow_sample_t_and_logpdf(lp, u1, c0, c1, g, logp_t);

    // t -> mu
    float mu = tanhf(NSF_SCALE * t);
    mu = clamp(mu, -1.0f + NSF_EPS, 1.0f - NSF_EPS);

    const float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - mu * mu));

    const float phi = 2.0f * M_PI * u2;
    float sinPhi, cosPhi;
    sincosf(phi, &sinPhi, &cosPhi);

    // wo を z 軸にしたローカル方向（z=mu）
    const float3 local = make_float3(cosPhi * sinTheta, sinPhi * sinTheta, mu);

    float3 tvec, bvec;
    makeONB(wo, tvec, bvec);
    ps.wi = normalize(local.x * tvec + local.y * bvec + local.z * wo);

    // pdf over solid angle
    const float denom = (2.0f * M_PI) * NSF_SCALE * fmaxf(1.0f - mu * mu, 1e-30f);
    ps.pdf = __expf(logp_t) / denom;

    return ps;
}
