// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
//
// EulerDiscreteScheduler — a header-only, dependency-free reimplementation of the
// diffusers 0.31.0 scheduler used by SD-Turbo (epsilon prediction, "trailing"
// timestep spacing, scaled_linear betas, final_sigmas_type="zero"). Pure float
// math so it builds and is unit-tested on the host (tests/test_diffusion.cpp)
// against golden vectors captured from the Python reference
// (diffusion/gen_golden_vectors.py) — the correctness gate before it ships in the
// un-runtime-testable console pipeline (uwp/diffuse.cpp).
//
// Reference (diffusers/schedulers/scheduling_euler_discrete.py):
//   betas   = (linspace(sqrt(b0), sqrt(b1), N))^2                  # scaled_linear
//   acp     = cumprod(1 - betas)
//   sig[i]  = sqrt((1 - acp[i]) / acp[i])
//   set_timesteps(n), spacing="trailing":
//     step = N / n;  ts = round(arange(N, 0, -step)) - 1
//     sigmas = interp(ts, arange(N), sig) then append 0 (final_sigmas_type=zero)
//     init_noise_sigma = max(sigmas)                               # trailing branch
//   scale_model_input(x, i) = x / sqrt(sigmas[i]^2 + 1)
//   step(eps, i, x): x0 = x - sigmas[i]*eps; x' = x + eps*(sigmas[i+1]-sigmas[i])
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace xllama::diffusion {

struct EulerConfig {
    int num_train_timesteps = 1000;
    double beta_start = 0.00085;
    double beta_end = 0.012;
    // Only the SD-Turbo configuration is implemented: beta_schedule="scaled_linear",
    // timestep_spacing="trailing", final_sigmas_type="zero", prediction_type="epsilon".
};

class EulerDiscreteScheduler {
  public:
    explicit EulerDiscreteScheduler(const EulerConfig& cfg = {}) : cfg_(cfg) {
        // Precompute the full per-train-timestep sigma table.
        const int N = cfg_.num_train_timesteps;
        const double sb0 = std::sqrt(cfg_.beta_start);
        const double sb1 = std::sqrt(cfg_.beta_end);
        double acp = 1.0;
        sigma_full_.resize(N);
        for (int i = 0; i < N; ++i) {
            const double s = sb0 + (sb1 - sb0) * (double)i / (double)(N - 1);
            const double beta = s * s;
            acp *= (1.0 - beta); // cumulative product of alphas
            sigma_full_[i] = std::sqrt((1.0 - acp) / acp);
        }
    }

    // Configure the inference schedule (SD-Turbo: n == 1).
    void set_timesteps(int n) {
        const int N = cfg_.num_train_timesteps;
        const double step = (double)N / (double)n;
        timesteps_.clear();
        sigmas_.clear();
        // "trailing": ts = round(arange(N, 0, -step)) - 1, descending.
        std::vector<double> ts;
        for (double v = (double)N; v > 0.0; v -= step)
            ts.push_back(std::floor(v + 0.5) - 1.0);
        for (double t : ts) {
            timesteps_.push_back(t);
            sigmas_.push_back(interp_sigma(t));
        }
        sigmas_.push_back(0.0); // final_sigmas_type = "zero"
        step_index_ = 0;
        // init_noise_sigma: "trailing"/"linspace" branch -> max(sigmas).
        init_noise_sigma_ = 0.0;
        for (double s : sigmas_)
            init_noise_sigma_ = std::max(init_noise_sigma_, s);
    }

    const std::vector<double>& sigmas() const {
        return sigmas_;
    }
    const std::vector<double>& timesteps() const {
        return timesteps_;
    }
    double init_noise_sigma() const {
        return init_noise_sigma_;
    }
    int step_index() const {
        return step_index_;
    }

    // Scale the latent before feeding the UNet: x / sqrt(sigma^2 + 1).
    // Operates in place on a flat buffer for the current step.
    template <typename T> void scale_model_input(std::vector<T>& x) const {
        const double sigma = sigmas_[step_index_];
        const double d = std::sqrt(sigma * sigma + 1.0);
        for (auto& v : x)
            v = (T)((double)v / d);
    }

    // One Euler step (epsilon prediction). x and eps are flat buffers of equal
    // length; x is updated in place to the previous sample. Advances step_index.
    //   x0 = x - sigma*eps ; dt = sigma_next - sigma ; x' = x + eps*dt
    template <typename T> void step(const std::vector<T>& eps, std::vector<T>& x) {
        const double sigma = sigmas_[step_index_];
        const double sigma_next = sigmas_[step_index_ + 1];
        const double dt = sigma_next - sigma;
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = (T)((double)x[i] + (double)eps[i] * dt);
        ++step_index_;
    }

  private:
    // Linear interpolation of sigma at a (possibly fractional) train timestep,
    // matching np.interp over arange(N) — SD-Turbo lands on the integer t=999.
    double interp_sigma(double t) const {
        if (t <= 0.0)
            return sigma_full_.front();
        const int N = (int)sigma_full_.size();
        if (t >= N - 1)
            return sigma_full_.back();
        const int lo = (int)std::floor(t);
        const double frac = t - lo;
        return sigma_full_[lo] * (1.0 - frac) + sigma_full_[lo + 1] * frac;
    }

    EulerConfig cfg_;
    std::vector<double> sigma_full_;
    std::vector<double> sigmas_;
    std::vector<double> timesteps_;
    double init_noise_sigma_ = 0.0;
    int step_index_ = 0;
};

} // namespace xllama::diffusion
