#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// KAN — Kolmogorov-Arnold Networks
// Liu et al., "KAN: Kolmogorov-Arnold Networks", ICLR 2025
//
// Architecture (Eq. 7, Fig. 1d):
//   KAN(x) = (Φ_{L-1} ∘ Φ_{L-2} ∘ ... ∘ Φ_1 ∘ Φ_0)(x)
//
//   Each layer Φ_l: ℝ^{n_l} → ℝ^{n_{l+1}} is a KANLinear (dm::prim):
//     x_{l+1,j} = Σ_i φ_{l,j,i}(x_{l,i})
//
//   φ_{l,j,i}(x) = w_b · silu(x) + w_s · spline(x)       (Eq. 15)
//   spline(x) = Σ_m  c_m · B_m(x)                         (Eq. 17)
//
//   B_m: B-spline basis functions of order k on G uniform intervals.
//   Grid is adaptive (updated from input statistics, Sect. 2.5).
//   Grid extension: coarsen→fine via least-squares (Appendix L, Eq. 18).
//
// Sparsity regularisation (Eqs. 19-22, Section 2.4):
//   l_total = l_pred + λ · Σ_l (μ1·|Φ_l|_1 + μ2·S(Φ_l))
//
// The reusable primitive dm::prim::KANLinear lives in:
//   <torch/nn/modules/kan_linear.h>
//
// Architecture notation [n_0, n_1, ..., n_L]:
//   n_0 = input dim, n_L = output dim, L = number of layers.
//   Paper examples: [2,5,1], [2,1,1], [784,100,10], [4,2,1,1].
//
// Training: Adam (lr=1e-3) or LBFGS (lr=1e-1..1).
//   Grid is updated from running stats each forward pass.
//   Grid extension schedule: start with G=5, extend to G=10,20,... .
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/nn/modules/kan_linear.h>
#include <torch/torch.h>
#include <vector>
#include <string>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// KANImpl — full Kolmogorov-Arnold Network
// ─────────────────────────────────────────────────────────────────────────────

struct KANImpl : torch::nn::Module {
    // layers[l]: KANLinear mapping ℝ^{n_l} → ℝ^{n_{l+1}}
    torch::nn::ModuleList layers;

    // widths: [n_0, n_1, ..., n_L]  (length L+1)
    std::vector<int64_t> widths;

    // ── Construction ──────────────────────────────────────────────────────
    // widths_:   layer width sequence [n_0, n_1, ..., n_L]
    // G_:        B-spline grid intervals (default 5)
    // k_:        B-spline order (default 3)
    // grid_min/max: initial grid range (default [-1,1])
    // update_grid_: whether to adapt grid each forward pass
    KANImpl(std::vector<int64_t> widths_,
            int64_t G_            = 5,
            int64_t k_            = 3,
            double  grid_min      = -1.0,
            double  grid_max      =  1.0,
            bool    update_grid_  = true);

    // x: [B, n_0]  →  [B, n_L]
    torch::Tensor forward(torch::Tensor x);

    // Sparsity regularisation term (Eqs. 19-22).
    // Pass the network input x: [B, n_0].
    // Returns scalar  λ * Σ_l (μ1·|Φ_l|_1 + μ2·S(Φ_l)).
    // Add to prediction loss during training.
    torch::Tensor sparsity_loss(const torch::Tensor& x,
                                double lambda_  = 1e-3,
                                double mu1      = 1.0,
                                double mu2      = 1.0) const;

    // Grid extension: refine all layers from current G to new_G.
    // x_samples: [N, n_0] — representative inputs (used to project coeffs).
    void extend_grid(int64_t new_G, const torch::Tensor& x_samples);

    // Convenience: number of KAN layers (L)
    int64_t depth() const { return static_cast<int64_t>(widths.size()) - 1; }
};
TORCH_MODULE(KAN);

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration (paper defaults, Appendix N / Section 4)
// ─────────────────────────────────────────────────────────────────────────────

struct KANTrainConfig {
    // Optimiser: "adam" or "lbfgs"
    std::string optimizer = "adam";
    double lr             = 1e-3;    // Adam lr; LBFGS: 1e-1
    int64_t max_iter      = 2000;
    int64_t batch_size    = 0;       // 0 = full-batch

    // Sparsity regularisation (Eq. 22)
    double lambda_        = 0.0;     // overall regularisation weight
    double mu1            = 1.0;     // L1 coefficient
    double mu2            = 1.0;     // entropy coefficient

    // Grid extension schedule.
    // G_schedule: list of grid sizes to extend through (e.g. {5,10,20,50})
    std::vector<int64_t> G_schedule = {5};

    // Steps between grid extensions (each entry of G_schedule uses this many steps)
    int64_t steps_per_grid = 200;

    torch::Device device = torch::kCPU;
};

} // namespace nlp
} // namespace models
} // namespace dm
