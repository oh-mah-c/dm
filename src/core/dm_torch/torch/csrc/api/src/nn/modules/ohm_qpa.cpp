#include "torch/nn/modules/ohm_qpa.h"
#include <torch/types.h>
#include <torch/utils.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace torch {
namespace nn {

OhmQPAOptions::OhmQPAOptions(int64_t hidden_dim) : hidden_dim_(hidden_dim) {}

OhmQPAImpl::OhmQPAImpl(const OhmQPAOptions& options_) : options(options_) {
    reset();
}

void OhmQPAImpl::reset() {
    lambda1 = register_parameter("lambda1", torch::randn({1, 1, 1, options.hidden_dim()}) * 0.02);
    lambda2 = register_parameter("lambda2", torch::randn({1, 1, 1, options.hidden_dim()}) * 0.02);
    alpha = register_parameter("alpha", torch::randn({1, 1, 1, options.hidden_dim()}) * 0.02);
    beta = register_parameter("beta", torch::randn({1}) * 0.02);
}

void OhmQPAImpl::reset_parameters() {
    reset();
}

torch::Tensor OhmQPAImpl::forward(const torch::Tensor& query, const torch::Tensor& key) {
    // query: [B, H, N, D]
    // key:   [B, H, M, D]
    // We want output: [B, H, N, M]
    
    // Broadcast for pairwise computation
    auto q = query.unsqueeze(3); // [B, H, N, 1, D]
    auto k = key.unsqueeze(2);   // [B, H, 1, M, D]
    
    // Qubit 0 and 1 rotation angles
    auto phi0 = M_PI / 4.0 + lambda1 * q + lambda2 * k;
    auto phi1 = M_PI / 4.0 + lambda2 * q + lambda1 * k;
    
    // Initial state |00>
    // Apply RY(phi0) on q0 and RY(phi1) on q1
    auto c00 = torch::cos(phi1 / 2.0) * torch::cos(phi0 / 2.0);
    auto c01 = torch::cos(phi1 / 2.0) * torch::sin(phi0 / 2.0);
    auto c10 = torch::sin(phi1 / 2.0) * torch::cos(phi0 / 2.0);
    auto c11 = torch::sin(phi1 / 2.0) * torch::sin(phi0 / 2.0);
    
    // CNOT(0, 1) -> flips q1 if q0 is 1. (Swaps c01 and c11)
    auto n_c00 = c00;
    auto n_c10 = c10;
    auto n_c01 = c11;
    auto n_c11 = c01;
    
    // RY(alpha * (q+k)) on q1
    auto theta = alpha * (q + k);
    auto cos_t = torch::cos(theta / 2.0);
    auto sin_t = torch::sin(theta / 2.0);
    
    auto n2_c00 = cos_t * n_c00 - sin_t * n_c10;
    auto n2_c10 = sin_t * n_c00 + cos_t * n_c10;
    auto n2_c01 = cos_t * n_c01 - sin_t * n_c11;
    auto n2_c11 = sin_t * n_c01 + cos_t * n_c11;
    
    // CNOT(1, 0) -> flips q0 if q1 is 1. (Swaps c10 and c11)
    auto v00 = n2_c00;
    auto v01 = n2_c01;
    auto v10 = n2_c11;
    auto v11 = n2_c10;
    
    // RX(2*beta) on both qubits.
    auto cos_b = torch::cos(beta);
    auto sin_b = torch::sin(beta);
    auto cos2_b = cos_b * cos_b;
    auto sin2_b = sin_b * sin_b;
    auto sincos_b = sin_b * cos_b;
    
    // Final state amplitudes (real and imaginary parts)
    auto f00_re = cos2_b * v00 - sin2_b * v11;
    auto f00_im = -sincos_b * (v01 + v10);
    
    auto f11_re = -sin2_b * v00 + cos2_b * v11;
    auto f11_im = -sincos_b * (v01 + v10);
    
    // Measurement probability P(|00>) + P(|11>)
    auto mu = (f00_re * f00_re + f00_im * f00_im) + (f11_re * f11_re + f11_im * f11_im); // Shape: [B, H, N, M, D]
    
    // Average over the feature dimension D to get the final bounded attention score in [0, 1]
    auto score = mu.mean(/*dim=*/-1); // Shape: [B, H, N, M]
    
    return score;
}

} // namespace nn
} // namespace torch
