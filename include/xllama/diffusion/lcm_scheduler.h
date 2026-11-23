// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// LCMScheduler — header-only Latent Consistency Model step schedule (Luo et al.,
// "Latent Consistency Models", 2023) using the public discrete-timestep
// construction also documented by Hugging Face Diffusers' LCMScheduler:
//   origin[i] = (i+1) * (N / original_steps) - 1     i = 0 .. original_steps-1
//   timesteps = origin[::-1][::len(origin)/n][:n]
//   c_skip = σ_d² / ((t·s)² + σ_d²)
//   c_out  = (t·s) / sqrt((t·s)² + σ_d²)
//   x0 = (x - sqrt(1-α_t)·ε) / sqrt(α_t)
//   denoised = c_skip·x + c_out·x0
//   last step: x' = denoised; otherwise x' = sqrt(α_prev)·denoised + sqrt(1-α_prev)·z
// Same scaled_linear betas as EulerDiscreteScheduler so SD-Turbo's train grid
// lines up. scale_model_input is identity (DDPM α-space, not Euler σ-space).
// Default diffusion stays on EulerDiscreteScheduler; this type is unused until
// an operator opts in.
#pragma once

#include <cmath>
#include <cstddef>
#include <random>
#include <string_view>
#include <vector>

namespace xllama::diffusion {

struct LcmConfig {
    int num_train_timesteps = 1000;
    int original_inference_steps = 50;
    double beta_start = 0.00085;
    double beta_end = 0.012;
    double timestep_scaling = 10.0;
    double sigma_data = 0.5;
};

// LocalState\diffuse-lcm.txt: only an exact "1" (optional trailing newline)
// selects LCM. Absent, empty, or any other text keeps Euler.
inline bool lcm_scheduler_requested(std::string_view text) {
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' '))
        text.remove_suffix(1);
    return text == "1";
}

class LCMScheduler {
  public:
    explicit LCMScheduler(const LcmConfig& cfg = {}) : cfg_(cfg) {
        const int N = cfg_.num_train_timesteps;
        const double sb0 = std::sqrt(cfg_.beta_start);
        const double sb1 = std::sqrt(cfg_.beta_end);
        double acp = 1.0;
        alpha_full_.resize(N);
        for (int i = 0; i < N; ++i) {
            const double s = sb0 + (sb1 - sb0) * (double)i / (double)(N - 1);
            const double beta = s * s;
            acp *= (1.0 - beta);
            alpha_full_[i] = acp;
        }
    }

    void set_timesteps(int n) {
        timesteps_.clear();
        step_index_ = 0;
        if (n <= 0)
            return;
        const int N = cfg_.num_train_timesteps;
        const int orig = cfg_.original_inference_steps;
        const int c = N / orig;
        std::vector<int> origin;
        origin.reserve(orig);
        for (int i = 1; i <= orig; ++i)
            origin.push_back(i * c - 1);
        int skip = (int)origin.size() / n;
        if (skip < 1)
            skip = 1;
        for (int i = (int)origin.size() - 1; i >= 0 && (int)timesteps_.size() < n; i -= skip)
            timesteps_.push_back((double)origin[i]);
    }

    const std::vector<double>& timesteps() const {
        return timesteps_;
    }
    int step_index() const {
        return step_index_;
    }
    double init_noise_sigma() const {
        return 1.0;
    }

    void set_rng(std::mt19937* rng) {
        rng_ = rng;
    }

    // LCM feeds the UNet the unscaled sample (identity).
    template <typename T> void scale_model_input(std::vector<T>&) const {}

    // One LCM step. `eps` is epsilon prediction; `x` is updated in place.
    // Intermediate steps inject Gaussian noise when set_rng was called.
    template <typename T> void step(const std::vector<T>& eps, std::vector<T>& x) {
        const int t = (int)timesteps_[step_index_];
        const bool last = step_index_ + 1 >= (int)timesteps_.size();
        const double alpha_t = alpha_at(t);
        const double sqrt_a = std::sqrt(alpha_t);
        const double sqrt_b = std::sqrt(1.0 - alpha_t);
        const double scaled = (double)t * cfg_.timestep_scaling;
        const double d2 = scaled * scaled + cfg_.sigma_data * cfg_.sigma_data;
        const double c_skip = cfg_.sigma_data * cfg_.sigma_data / d2;
        const double c_out = scaled / std::sqrt(d2);
        std::vector<double> denoised(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            const double x0 = ((double)x[i] - sqrt_b * (double)eps[i]) / sqrt_a;
            denoised[i] = c_skip * (double)x[i] + c_out * x0;
        }
        if (last) {
            for (size_t i = 0; i < x.size(); ++i)
                x[i] = (T)denoised[i];
        } else {
            const int t_prev = (int)timesteps_[step_index_ + 1];
            const double alpha_prev = alpha_at(t_prev);
            const double sqrt_ap = std::sqrt(alpha_prev);
            const double sqrt_bp = std::sqrt(1.0 - alpha_prev);
            std::normal_distribution<double> gauss(0.0, 1.0);
            for (size_t i = 0; i < x.size(); ++i) {
                const double z = rng_ ? gauss(*rng_) : 0.0;
                x[i] = (T)(sqrt_ap * denoised[i] + sqrt_bp * z);
            }
        }
        ++step_index_;
    }

  private:
    double alpha_at(int t) const {
        if (t <= 0)
            return alpha_full_.front();
        if (t >= (int)alpha_full_.size())
            return alpha_full_.back();
        return alpha_full_[t];
    }

    LcmConfig cfg_;
    std::vector<double> alpha_full_;
    std::vector<double> timesteps_;
    int step_index_ = 0;
    std::mt19937* rng_ = nullptr;
};

} // namespace xllama::diffusion
