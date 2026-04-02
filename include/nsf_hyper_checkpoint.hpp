#pragma once
#include <cstdint>
#include <cstring>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

#include "safetensors_loader.hpp" // load_safetensors / SafeTensorsFile / TensorView / DType

// -----------------------------
// NSF Hyper-only checkpoint
// -----------------------------
class NsfHyperCheckpoint {
public:
    // FP16 linear: weight(out,in), bias(out)
    struct LinearFP16 {
        int out = 0;
        int in  = 0;
        std::vector<uint16_t> W; // size = out*in, row-major (PyTorchの(out,in)そのまま)
        std::vector<uint16_t> b; // size = out
        bool hasW() const { return !W.empty(); }
        bool hasB() const { return !b.empty(); }
    };

    // hyper MLP: 3->64->64->(3K-1)
    struct HyperMLP {
        LinearFP16 l0; // hyper.0 (Linear)
        LinearFP16 l1; // hyper.2 (Linear)
        LinearFP16 l2; // hyper.4 (Linear)
    };

public:
    // ---- load interface ----
    static NsfHyperCheckpoint Load(
        const std::string& safetensors_path,
        int expected_transforms = -1,
        int expected_bins       = -1,
        int expected_context    = -1,
        int expected_hidden     = -1
    ) {
        NsfHyperCheckpoint ck;
        ck.loadImpl(safetensors_path);

        // optional assertions (事故防止)
        if (expected_transforms >= 0 && ck.transforms_ != expected_transforms)
            throw std::runtime_error("transforms mismatch");
        if (expected_bins >= 0 && ck.bins_ != expected_bins)
            throw std::runtime_error("bins mismatch");
        if (expected_context >= 0 && ck.context_ != expected_context)
            throw std::runtime_error("context mismatch");
        if (expected_hidden >= 0 && ck.hidden_ != expected_hidden)
            throw std::runtime_error("hidden mismatch");

        return ck;
    }

    // ---- getters ----
    int transforms() const { return transforms_; }
    int bins() const { return bins_; }       // K
    int context() const { return context_; } // 3
    int hidden() const { return hidden_; }   // 32
    int spline_out_dim() const { return spline_out_dim_; } // fixed-x : 2K -1

    const HyperMLP& hyperAt(int t) const { return hyper_.at(static_cast<std::size_t>(t)); }
    HyperMLP& hyperAt(int t) { return hyper_.at(static_cast<std::size_t>(t)); }
    
private:
    int transforms_ = 0;
    int bins_ = 0;
    int context_ = 0;
    int hidden_ = 0;
    int spline_out_dim_ = 0;
    std::vector<HyperMLP> hyper_;

private:
    static size_t numel(const TensorView& tv) {
        size_t n = 1;
        for (auto d : tv.shape) n *= (size_t)d;
        return n;
    }

    static void ensure_dtype_f16(const TensorView& tv, const std::string& name) {
        if (tv.dtype != DType::F16) {
            throw std::runtime_error("dtype must be F16: " + name);
        }
    }

    static void copy_f16_1d(const TensorView& tv, const std::string& name, std::vector<uint16_t>& out) {
        ensure_dtype_f16(tv, name);
        if (tv.shape.size() != 1) {
            throw std::runtime_error("expected 1D tensor: " + name);
        }
        const size_t n = numel(tv);
        if (tv.nbytes != n * 2) {
            throw std::runtime_error("nbytes mismatch: " + name);
        }
        out.resize(n);
        std::memcpy(out.data(), tv.data, n * 2);
    }

    static void copy_f16_2d(const TensorView& tv, const std::string& name, int& out_dim, int& in_dim, std::vector<uint16_t>& out) {
        ensure_dtype_f16(tv, name);
        if (tv.shape.size() != 2) {
            throw std::runtime_error("expected 2D tensor: " + name);
        }
        out_dim = (int)tv.shape[0];
        in_dim  = (int)tv.shape[1];
        const size_t n = (size_t)out_dim * (size_t)in_dim;
        if (tv.nbytes != n * 2) {
            throw std::runtime_error("nbytes mismatch: " + name);
        }
        out.resize(n);
        std::memcpy(out.data(), tv.data, n * 2);
    }

    void loadImpl(const std::string& path) {
        SafeTensorsFile st = load_safetensors(path);

        // key例: transform.transforms.0.hyper.4.weight
        const std::regex re(R"(transform\.transforms\.(\d+)\.hyper\.(\d+)\.(weight|bias))");

        // transforms 数を推定（最大 idx + 1）
        int max_t = -1;
        for (const auto& kv : st.tensors) {
            std::smatch m;
            if (std::regex_match(kv.first, m, re)) {
                const int t = std::stoi(m[1].str());
                max_t = std::max(max_t, t);
            }
        }
        if (max_t < 0) {
            throw std::runtime_error("No NSF hyper tensors found in safetensors.");
        }

        transforms_ = max_t + 1;
        hyper_.assign((size_t)transforms_, HyperMLP{});

        // 収集
        for (const auto& kv : st.tensors) {
            const std::string& name = kv.first;
            const TensorView& tv = kv.second;

            std::smatch m;
            if (!std::regex_match(name, m, re)) continue;

            const int t = std::stoi(m[1].str());
            const int layer = std::stoi(m[2].str()); // 0,2,4
            const std::string kind = m[3].str();     // weight/bias

            HyperMLP& hm = hyper_.at((size_t)t);
            LinearFP16* L = nullptr;
            if (layer == 0) L = &hm.l0;
            else if (layer == 2) L = &hm.l1;
            else if (layer == 4) L = &hm.l2;
            else continue;

            if (kind == "weight") {
                copy_f16_2d(tv, name, L->out, L->in, L->W);
            } else {
                copy_f16_1d(tv, name, L->b);
            }
        }

        // 完全性チェック：各 transform に 6 テンソルあること
        for (int t = 0; t < transforms_; ++t) {
            const auto& hm = hyper_.at((size_t)t);
            if (!hm.l0.hasW() || !hm.l0.hasB() ||
                !hm.l1.hasW() || !hm.l1.hasB() ||
                !hm.l2.hasW() || !hm.l2.hasB()) {
                throw std::runtime_error("Missing hyper tensors at transform " + std::to_string(t));
            }
        }

        // 形状から context/hidden/bins を確定（transform間一致をチェック）
        hidden_  = hyper_[0].l0.out;
        context_ = hyper_[0].l0.in;

        // l2.out = 2K - 1 から K を逆算
        spline_out_dim_ = hyper_[0].l2.out;
        if ((spline_out_dim_ + 1) % 2 != 0) {
            throw std::runtime_error("Cannot infer bins: l2.out is not (2K-1). out=" + std::to_string(spline_out_dim_));
        }
        bins_ = (spline_out_dim_ + 1) / 2;

        // Sanity check：全 transform で一致
        for (int t = 0; t < transforms_; ++t) {
            const auto& hm = hyper_[(size_t)t];

            if (hm.l0.out != hidden_ || hm.l0.in != context_)
                throw std::runtime_error("shape mismatch l0 at transform " + std::to_string(t));
            if (hm.l1.out != hidden_ || hm.l1.in != hidden_)
                throw std::runtime_error("shape mismatch l1 at transform " + std::to_string(t));
            if (hm.l2.in != hidden_ || hm.l2.out != spline_out_dim_)
                throw std::runtime_error("shape mismatch l2 at transform " + std::to_string(t));

            if ((int)hm.l0.b.size() != hidden_)         throw std::runtime_error("bias mismatch l0");
            if ((int)hm.l1.b.size() != hidden_)         throw std::runtime_error("bias mismatch l1");
            if ((int)hm.l2.b.size() != spline_out_dim_) throw std::runtime_error("bias mismatch l2");
        }
    }
};