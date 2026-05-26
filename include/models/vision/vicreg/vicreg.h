#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// VICReg — Variance-Invariance-Covariance Regularization for Self-Supervised Learning
// Bardes, Ponce & LeCun, arXiv:2105.04906v3, ICLR 2022
// https://arxiv.org/abs/2105.04906
//
// Architecture (Figure 1):
//   encoder  f_θ   : backbone (ResNet-50 default; 2048-dim output)
//   expander h_φ   : 3-layer MLP [rep_dim → 8192 → 8192 → 8192], BN+ReLU, last linear
//   Two branches (Siamese by default; may differ in arch / weights)
//
// Loss function (Section 4, Eq. 6):
//   ℓ(Z, Z′) = λ·s(Z,Z′) + μ·[v(Z)+v(Z′)] + ν·[c(Z)+c(Z′)]
//
//   s(Z,Z′)  — Invariance: mean squared Euclidean distance (Eq. 5)
//              s = (1/n) Σᵢ ||zᵢ − zᵢ′||²
//
//   v(Z)     — Variance hinge (Eq. 1–2):
//              v = (1/d) Σⱼ max(0, γ − S(zʲ, ε))
//              S(x,ε) = √(Var(x) + ε)   — regularised std-dev
//              γ=1, ε=1e-4 (paper default)
//
//   c(Z)     — Covariance penalty (Eq. 3–4):
//              C(Z) = (1/(n−1)) Σᵢ (zᵢ−z̄)(zᵢ−z̄)ᵀ
//              c = (1/d) Σᵢ≠ⱼ [C(Z)]²ᵢⱼ
//
// Default coefficients (Section 4.2): λ=25, μ=25, ν=1
// Training:  LARS, lr = batch/256 × 0.2, cosine decay, weight-decay=1e-6,
//            1000 epochs, batch=2048 (Section 4.2)
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <cstdint>
#include <string>

namespace dm {
namespace models {
namespace vision {

// ── VICRegConfig ─────────────────────────────────────────────────────────────

struct VICRegConfig {
    // Encoder output dimension (set to match backbone)
    int64_t repr_dim      = 2048;
    // Expander hidden/output dimension (paper: 8192)
    int64_t expander_dim  = 8192;
    // Loss coefficients (Section 4.2: λ=25, μ=25, ν=1)
    double  lambda_inv    = 25.0;   // invariance weight
    double  mu_var        = 25.0;   // variance weight
    double  nu_cov        = 1.0;    // covariance weight
    // Variance hinge target γ and stability ε (Eq. 1–2)
    double  gamma         = 1.0;
    double  var_eps       = 1e-4;

    // ── Convenience constructors ─────────────────────────────────────────────
    // Standard config matching paper (ResNet-50, ImageNet)
    static VICRegConfig standard() {
        return VICRegConfig{};  // all defaults match paper
    }
    // Lightweight config for fast smoke-tests
    static VICRegConfig tiny(int64_t repr = 64, int64_t exp = 128) {
        VICRegConfig c;
        c.repr_dim     = repr;
        c.expander_dim = exp;
        return c;
    }
};

// ── Expander (h_φ) ───────────────────────────────────────────────────────────
// Three fully-connected layers of size expander_dim.
// Inner layers: Linear → BN → ReLU.   Last layer: Linear only (Section 4.2).

struct ExpanderImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
    torch::nn::BatchNorm1d bn1{nullptr}, bn2{nullptr};

    ExpanderImpl(int64_t repr_dim, int64_t exp_dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Expander);

// ── VICReg model ─────────────────────────────────────────────────────────────
// Wraps two branches, each = encoder + expander.
// Encoder is provided externally (plug in any ResNet / ViT backbone).
// If no second encoder is provided the branches share weights (Siamese).

struct VICRegLoss {
    torch::Tensor total;
    torch::Tensor inv;   // invariance term s(Z,Z')
    torch::Tensor var;   // variance term   v(Z)+v(Z')
    torch::Tensor cov;   // covariance term c(Z)+c(Z')
};

struct VICRegImpl : torch::nn::Module {
    VICRegConfig cfg;

    torch::nn::AnyModule encoder;       // branch-1 encoder (f_θ)
    torch::nn::AnyModule encoder2;      // branch-2 encoder (f_θ'; may be same module)
    Expander expander{nullptr};         // branch-1 expander (h_φ)
    Expander expander2{nullptr};        // branch-2 expander (h_φ'; independent)

    bool shared_encoder;  // true → both branches use `encoder`

    // ── Constructor ──────────────────────────────────────────────────────────
    // enc1       : backbone for branch 1 (must output [B, repr_dim])
    // enc2       : backbone for branch 2 (pass {} for weight sharing)
    // cfg        : VICRegConfig
    VICRegImpl(torch::nn::AnyModule enc1,
               torch::nn::AnyModule enc2,
               VICRegConfig cfg = VICRegConfig{});

    // ── Forward ──────────────────────────────────────────────────────────────
    // x, x_prime : two augmented views of the same batch [B, C, H, W]
    // Returns VICRegLoss with individual terms (useful for logging).
    VICRegLoss forward(torch::Tensor x, torch::Tensor x_prime);

    // ── Loss helpers (static, usable stand-alone) ─────────────────────────────
    // Invariance: Eq. 5
    static torch::Tensor invariance_loss(torch::Tensor z, torch::Tensor z_prime);
    // Variance hinge: Eq. 1-2
    static torch::Tensor variance_loss(torch::Tensor z, double gamma, double eps);
    // Covariance penalty: Eq. 3-4
    static torch::Tensor covariance_loss(torch::Tensor z);
};
TORCH_MODULE(VICReg);

// ── Simple MLP encoder for demos / tests ─────────────────────────────────────
// fc1 → ReLU → fc2 → ReLU → fc3.  Output dim = repr_dim.

struct MLPEncoderImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    MLPEncoderImpl(int64_t in_dim, int64_t hidden, int64_t repr_dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MLPEncoder);

// ── Factory helpers ───────────────────────────────────────────────────────────

// Create a VICReg model with a simple MLP encoder (for testing / demos).
// in_dim   : flattened input size (e.g. 3*32*32 for CIFAR-like)
// repr_dim : encoder output dimension
VICReg make_vicreg_mlp(int64_t in_dim,
                        int64_t repr_dim    = 2048,
                        int64_t expander_dim = 8192);

// Create a VICReg model with a pre-built backbone.
// The same backbone is shared between the two branches (Siamese).
VICReg make_vicreg_siamese(torch::nn::AnyModule backbone,
                            VICRegConfig cfg = VICRegConfig{});

}  // namespace vision
}  // namespace models
}  // namespace dm
