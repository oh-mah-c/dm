// ─────────────────────────────────────────────────────────────────────────────
// VICReg implementation
// Bardes, Ponce & LeCun, arXiv:2105.04906v3, ICLR 2022
//
// Equations referenced:
//   Invariance:  s(Z,Z′) = (1/n) Σᵢ ||zᵢ − zᵢ′||²          (Eq. 5)
//   Variance:    v(Z)    = (1/d) Σⱼ max(0, γ−S(zʲ,ε))       (Eq. 1–2)
//                S(x,ε) = √(Var(x)+ε)
//   Covariance:  C(Z)    = (1/(n−1)) Σᵢ (zᵢ−z̄)(zᵢ−z̄)ᵀ     (Eq. 3)
//                c(Z)    = (1/d) Σᵢ≠ⱼ [C(Z)]²ᵢⱼ              (Eq. 4)
//   Total loss:  ℓ = λ·s + μ·(v(Z)+v(Z′)) + ν·(c(Z)+c(Z′)) (Eq. 6)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/vicreg/vicreg.h"

#include <torch/torch.h>
#include <stdexcept>

namespace dm {
namespace models {
namespace vision {

// ─────────────────────────────────────────────────────────────────────────────
// Expander  h_φ
// Architecture: Linear(repr→exp) → BN → ReLU
//             → Linear(exp→exp)  → BN → ReLU
//             → Linear(exp→exp)              (last layer: no BN/ReLU)
// Section 4.2: 3 layers of size 8192.
// ─────────────────────────────────────────────────────────────────────────────

ExpanderImpl::ExpanderImpl(int64_t repr_dim, int64_t exp_dim) {
    fc1 = register_module("fc1", torch::nn::Linear(repr_dim, exp_dim));
    bn1 = register_module("bn1", torch::nn::BatchNorm1d(exp_dim));
    fc2 = register_module("fc2", torch::nn::Linear(exp_dim,  exp_dim));
    bn2 = register_module("bn2", torch::nn::BatchNorm1d(exp_dim));
    fc3 = register_module("fc3", torch::nn::Linear(exp_dim,  exp_dim));
}

torch::Tensor ExpanderImpl::forward(torch::Tensor x) {
    x = torch::relu(bn1(fc1(x)));
    x = torch::relu(bn2(fc2(x)));
    x = fc3(x);
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP encoder  (for demos and unit tests)
// ─────────────────────────────────────────────────────────────────────────────

MLPEncoderImpl::MLPEncoderImpl(int64_t in_dim, int64_t hidden, int64_t repr_dim) {
    fc1 = register_module("fc1", torch::nn::Linear(in_dim, hidden));
    fc2 = register_module("fc2", torch::nn::Linear(hidden, hidden));
    fc3 = register_module("fc3", torch::nn::Linear(hidden, repr_dim));
}

torch::Tensor MLPEncoderImpl::forward(torch::Tensor x) {
    // Flatten spatial dims if needed
    if (x.dim() > 2) x = x.flatten(1);
    x = torch::relu(fc1(x));
    x = torch::relu(fc2(x));
    return fc3(x);
}

// ─────────────────────────────────────────────────────────────────────────────
// VICReg
// ─────────────────────────────────────────────────────────────────────────────

VICRegImpl::VICRegImpl(torch::nn::AnyModule enc1,
                       torch::nn::AnyModule enc2,
                       VICRegConfig c)
    : cfg(c)
{
    shared_encoder = enc2.is_empty();

    encoder = std::move(enc1);
    register_module("encoder", encoder.ptr());

    if (!shared_encoder) {
        encoder2 = std::move(enc2);
        register_module("encoder2", encoder2.ptr());
    }

    expander  = register_module("expander",  Expander(cfg.repr_dim, cfg.expander_dim));
    expander2 = register_module("expander2", Expander(cfg.repr_dim, cfg.expander_dim));
}

// ── Static loss helpers ───────────────────────────────────────────────────────

// Invariance: Eq. 5 — mean squared Euclidean distance.
torch::Tensor VICRegImpl::invariance_loss(torch::Tensor z, torch::Tensor z_prime) {
    return torch::nn::functional::mse_loss(z, z_prime,
               torch::nn::functional::MSELossFuncOptions()
                   .reduction(torch::kMean));
}

// Variance hinge: Eq. 1–2.
// v(Z) = (1/d) Σⱼ max(0, γ − √(Var(zʲ) + ε))
// z: [B, d]
torch::Tensor VICRegImpl::variance_loss(torch::Tensor z, double gamma, double eps) {
    // var over batch dim (dim=0) → [d]
    auto var  = z.var(0);  // unbiased by default
    auto std  = (var + eps).sqrt();
    auto hinge = torch::clamp(gamma - std, /*min=*/0.0);
    return hinge.mean();
}

// Covariance penalty: Eq. 3–4.
// C(Z) = (1/(n−1)) (Z − z̄)ᵀ (Z − z̄)   →  [d, d]
// c(Z) = (1/d) Σᵢ≠ⱼ [C(Z)]²ᵢⱼ
// z: [B, d]
torch::Tensor VICRegImpl::covariance_loss(torch::Tensor z) {
    int64_t n = z.size(0);
    int64_t d = z.size(1);

    // Centre along batch dim
    auto z_c = z - z.mean(0, /*keepdim=*/true);  // [B, d]

    // Covariance matrix  [d, d]
    auto cov = torch::mm(z_c.t(), z_c) / static_cast<double>(n - 1);

    // Sum of squared off-diagonal elements, scaled by 1/d (Eq. 4)
    auto off_diag = cov.pow(2);
    // Zero out diagonal
    off_diag = off_diag - off_diag.diagonal().diag();
    return off_diag.sum() / static_cast<double>(d);
}

// ── Forward ──────────────────────────────────────────────────────────────────

VICRegLoss VICRegImpl::forward(torch::Tensor x, torch::Tensor x_prime) {
    // ── Encode ───────────────────────────────────────────────────────────────
    // Flatten spatial dims before passing to encoder if input is 4-D and the
    // registered encoder is our MLP (AnyModule does dynamic dispatch).
    auto y  = encoder.forward<torch::Tensor>(x);
    auto y_prime = shared_encoder
                 ? encoder.forward<torch::Tensor>(x_prime)
                 : encoder2.forward<torch::Tensor>(x_prime);

    // ── Expand ───────────────────────────────────────────────────────────────
    auto z       = expander(y);         // [B, exp_dim]
    auto z_prime = expander2(y_prime);  // [B, exp_dim]

    // ── Loss terms ───────────────────────────────────────────────────────────
    // Invariance (Eq. 5)
    auto inv = invariance_loss(z, z_prime);

    // Variance (Eq. 1–2) — applied to both branches independently
    auto var = variance_loss(z,       cfg.gamma, cfg.var_eps)
             + variance_loss(z_prime, cfg.gamma, cfg.var_eps);

    // Covariance (Eq. 3–4) — applied to both branches independently
    auto cov = covariance_loss(z) + covariance_loss(z_prime);

    // Weighted sum (Eq. 6)
    auto total = cfg.lambda_inv * inv
               + cfg.mu_var    * var
               + cfg.nu_cov    * cov;

    return VICRegLoss{total, inv, var, cov};
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory helpers
// ─────────────────────────────────────────────────────────────────────────────

VICReg make_vicreg_mlp(int64_t in_dim, int64_t repr_dim, int64_t expander_dim) {
    VICRegConfig cfg;
    cfg.repr_dim     = repr_dim;
    cfg.expander_dim = expander_dim;

    auto enc = MLPEncoder(in_dim, /*hidden=*/512, repr_dim);
    torch::nn::AnyModule m1(enc);
    return VICReg(std::move(m1), torch::nn::AnyModule{}, cfg);
}

VICReg make_vicreg_siamese(torch::nn::AnyModule backbone, VICRegConfig cfg) {
    return VICReg(std::move(backbone), torch::nn::AnyModule{}, cfg);
}

}  // namespace vision
}  // namespace models
}  // namespace dm
