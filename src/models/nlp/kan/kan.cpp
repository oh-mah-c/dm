// ─────────────────────────────────────────────────────────────────────────────
// KAN — Kolmogorov-Arnold Networks
// Liu et al., ICLR 2025
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/kan/kan.h"

#include <torch/torch.h>
#include <stdexcept>
#include <string>

namespace dm {
namespace models {
namespace nlp {

// ═══════════════════════════════════════════════════════════════════════════
// KANImpl
// ═══════════════════════════════════════════════════════════════════════════

KANImpl::KANImpl(std::vector<int64_t> widths_,
                 int64_t G_,
                 int64_t k_,
                 double  grid_min,
                 double  grid_max,
                 bool    update_grid_)
    : widths(std::move(widths_))
{
    if (widths.size() < 2)
        throw std::invalid_argument("KAN: widths must have at least 2 entries");

    layers = register_module("layers", torch::nn::ModuleList());

    for (int64_t l = 0; l < static_cast<int64_t>(widths.size()) - 1; ++l) {
        dm::prim::KANLinearOptions opts(widths[l], widths[l + 1]);
        opts.G(G_)
            .k(k_)
            .grid_min(grid_min)
            .grid_max(grid_max)
            .update_grid(update_grid_);
        layers->push_back(dm::prim::KANLinear(opts));
    }
}

torch::Tensor KANImpl::forward(torch::Tensor x) {
    for (int64_t l = 0; l < depth(); ++l) {
        auto layer = layers->at<dm::prim::KANLinearImpl>(l);
        x = layer.forward(x);
    }
    return x;
}

torch::Tensor KANImpl::sparsity_loss(const torch::Tensor& x,
                                     double lambda_,
                                     double mu1,
                                     double mu2) const {
    auto loss = torch::zeros({}, x.options());
    auto xi   = x;
    for (int64_t l = 0; l < depth(); ++l) {
        auto layer_mod = layers->ptr(l);
        // Cast to KANLinear module holder
        auto kan_layer = std::dynamic_pointer_cast<dm::prim::KANLinearImpl>(layer_mod);
        if (!kan_layer)
            throw std::runtime_error("KAN::sparsity_loss: unexpected layer type");

        dm::prim::KANLinear holder(kan_layer);
        loss = loss + dm::prim::kan_linear_sparsity(holder, xi, mu1, mu2);
        // Propagate x through this layer (detached) for the next layer's input
        xi = kan_layer->forward(xi.detach()).detach();
    }
    return static_cast<float>(lambda_) * loss;
}

void KANImpl::extend_grid(int64_t new_G, const torch::Tensor& x_samples) {
    auto xi = x_samples;
    for (int64_t l = 0; l < depth(); ++l) {
        auto& layer = layers->at<dm::prim::KANLinearImpl>(l);
        layer.extend_grid(new_G, xi);
        // Pass samples through updated layer (no grad)
        torch::NoGradGuard ng;
        xi = layer.forward(xi);
    }
}

} // namespace nlp
} // namespace models
} // namespace dm
