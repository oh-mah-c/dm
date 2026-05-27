// ─────────────────────────────────────────────────────────────────────────────
// CCNet — Criss-Cross Network for Semantic Segmentation
// Huang et al., IEEE TPAMI 2020  (arXiv:1811.11721v2)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/ccnet/ccnet.h"

#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dm {
namespace models {
namespace vision {

// ═══════════════════════════════════════════════════════════════════════════
// RCCAModule
// ═══════════════════════════════════════════════════════════════════════════

RCCAModuleImpl::RCCAModuleImpl(int64_t in_channels,
                               int64_t key_channels,
                               int     loops)
    : rcca_loops(loops)
{
    cca = register_module("cca",
        dm::prim::CrissCrossAttention(in_channels, key_channels));
}

torch::Tensor RCCAModuleImpl::forward(torch::Tensor x) {
    for (int r = 0; r < rcca_loops; ++r)
        x = cca->forward(x);
    return x;
}

// ═══════════════════════════════════════════════════════════════════════════
// CCNetHead
// ═══════════════════════════════════════════════════════════════════════════

CCNetHeadImpl::CCNetHeadImpl(int64_t in_channels,
                             int64_t mid_channels,
                             int64_t num_classes)
{
    fuse = register_module("fuse", torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, mid_channels, 3)
            .padding(1).bias(false)),
        torch::nn::BatchNorm2d(mid_channels),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Dropout2d(0.1)
    ));
    seg_conv = register_module("seg_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(mid_channels, num_classes, 1)));
}

torch::Tensor CCNetHeadImpl::forward(torch::Tensor x) {
    return seg_conv->forward(fuse->forward(x));
}

// ═══════════════════════════════════════════════════════════════════════════
// CCNet
// ═══════════════════════════════════════════════════════════════════════════

CCNetImpl::CCNetImpl(torch::nn::AnyModule backbone_,
                     int64_t backbone_out_ch,
                     int64_t num_classes,
                     int64_t rcca_ch,
                     int64_t key_ch,
                     int64_t mid_ch,
                     int     loops)
    : rcca_channels_(rcca_ch)
{
    // Store backbone (not registered as named sub-module since it's AnyModule;
    // user is responsible for its parameters appearing in an optimizer group).
    backbone = std::move(backbone_);

    // Reduction: backbone_out_ch -> rcca_ch (1x1 conv + BN + ReLU)
    reduction = register_module("reduction",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(backbone_out_ch, rcca_ch, 1)
            .bias(false)));
    reduction_bn = register_module("reduction_bn",
        torch::nn::BatchNorm2d(rcca_ch));

    // RCCA
    rcca = register_module("rcca", RCCAModule(rcca_ch, key_ch, loops));

    // Head: concat(H'', X) = rcca_ch + backbone_out_ch channels
    head = register_module("head",
        CCNetHead(rcca_ch + backbone_out_ch, mid_ch, num_classes));
}

torch::Tensor CCNetImpl::forward(torch::Tensor x) {
    // Backbone: [B,3,H,W] -> [B, backbone_out_ch, H/8, W/8]
    auto X = backbone.forward<torch::Tensor>(x); // [B, C_b, H', W']

    // Reduction: [B, C_b, H', W'] -> [B, rcca_ch, H', W']
    auto H = torch::relu(reduction_bn->forward(reduction->forward(X)));

    // RCCA: [B, rcca_ch, H', W'] -> [B, rcca_ch, H', W']
    auto H_pp = rcca->forward(H);

    // Concat H'' and X: [B, rcca_ch + C_b, H', W']
    auto fused = torch::cat({H_pp, X}, /*dim=*/1);

    // Head -> logits: [B, num_classes, H', W']
    return head->forward(fused);
}

// ═══════════════════════════════════════════════════════════════════════════
// Category Consistent Loss  (Eqs. 3-7, Section 3.4)
// ═══════════════════════════════════════════════════════════════════════════

torch::Tensor ccl_loss(const torch::Tensor& features,
                       const torch::Tensor& targets,
                       const CCLLossOptions& opts,
                       int64_t ignore_index)
{
    // features: [B, C, H, W]
    // targets:  [B, H, W]  (int64)
    const int64_t B = features.size(0);
    const int64_t C = features.size(1);
    const int64_t H = features.size(2);
    const int64_t W = features.size(3);

    // Flatten spatial: [B*H*W, C] and [B*H*W]
    auto feat_flat = features.permute({0, 2, 3, 1})
                             .contiguous()
                             .view({B * H * W, C});
    auto tgt_flat  = targets.contiguous().view({B * H * W});

    // Valid mask (exclude ignore_index)
    auto valid_mask = tgt_flat.ne(ignore_index);
    auto feat_valid = feat_flat.index({valid_mask}); // [N_valid, C]
    auto tgt_valid  = tgt_flat.index({valid_mask});  // [N_valid]

    if (feat_valid.size(0) == 0) {
        return torch::zeros({}, features.options());
    }

    // Unique classes in this batch
    auto classes = std::get<0>(at::_unique(tgt_valid, /*sorted=*/true));
    const int64_t n_classes = classes.size(0);

    auto l_var  = torch::zeros({}, features.options());
    auto l_dis  = torch::zeros({}, features.options());
    auto l_reg  = torch::zeros({}, features.options());

    // Compute per-class means (mu_c)
    std::vector<torch::Tensor> means;
    means.reserve(n_classes);
    for (int64_t ci = 0; ci < n_classes; ++ci) {
        int64_t cls = classes[ci].template item<int64_t>();
        auto mask_c = tgt_valid.eq(cls);
        auto feat_c = feat_valid.index({mask_c}); // [N_c, C]
        means.push_back(feat_c.mean(0));           // [C]
    }

    // l_var  (Eq. 3): mean over classes of mean over elements of phi_var
    // phi_var (Eq. 6): piecewise distance from center mu_c
    for (int64_t ci = 0; ci < n_classes; ++ci) {
        int64_t cls = classes[ci].template item<int64_t>();
        auto mask_c = tgt_valid.eq(cls);
        auto feat_c = feat_valid.index({mask_c}); // [N_c, C]
        auto mu_c   = means[ci];                  // [C]
        auto N_c    = feat_c.size(0);

        // dist = ||h_i - mu_c||  for each element
        auto diff = feat_c - mu_c.unsqueeze(0); // [N_c, C]
        auto dist = diff.norm(2, /*dim=*/1);     // [N_c]

        // phi_var (Eq. 6):
        //   0                                if dist <= delta_v
        //   (dist - delta_v)^2               if delta_v < dist <= delta_d
        //   (delta_d - delta_v)^2 + linear   if dist > delta_d
        // Simplified piece-wise (paper Eq. 6):
        //   0                  if ||mu-h|| <= delta_v
        //   (||mu-h||-delta_v)^2   if delta_v < ||mu-h|| <= delta_d
        //   linear             if > delta_d
        // We implement the quadratic+linear version from Eq. 6 exactly.
        float dv = opts.delta_v;
        float dd = opts.delta_d;
        auto hinge_v = dist - dv;
        // phi_var = 0 if dist <= dv; (dist-dv)^2 if dv < dist <= dd;
        //           (dd-dv)^2 + (dist-dd)*2*(dd-dv) if dist > dd   [linear region]
        auto in_quad = (dist > dv) & (dist <= dd);
        auto in_lin  = dist > dd;
        float quad_at_dd = (dd - dv) * (dd - dv);
        auto phi = torch::zeros_like(dist);
        phi = phi + in_quad.to(dist.dtype()) * hinge_v.clamp_min(0).pow(2);
        phi = phi + in_lin.to(dist.dtype()) *
                    (quad_at_dd + (dist - dd) * 2.0f * (dd - dv));

        l_var = l_var + phi.mean();
    }
    if (n_classes > 0)
        l_var = l_var / n_classes;

    // l_dis (Eq. 4): mean over pairs of phi_dis
    // phi_dis (Eq. 7): (2*delta_d - ||mu_a - mu_b||)^2 if < 2*delta_d else 0
    if (n_classes > 1) {
        auto centers = torch::stack(means, 0); // [n_classes, C]
        int64_t n_pairs = 0;
        for (int64_t ci = 0; ci < n_classes; ++ci) {
            for (int64_t cj = ci + 1; cj < n_classes; ++cj) {
                auto diff = centers[ci] - centers[cj];
                auto dist = diff.norm(2);
                auto margin = 2.0f * opts.delta_d - dist;
                auto phi_d = torch::where(
                    margin > 0.0f,
                    margin.pow(2),
                    torch::zeros_like(margin));
                l_dis = l_dis + phi_d;
                ++n_pairs;
            }
        }
        if (n_pairs > 0)
            l_dis = l_dis / n_pairs;
    }

    // l_reg (Eq. 5): mean ||mu_c||
    for (int64_t ci = 0; ci < n_classes; ++ci)
        l_reg = l_reg + means[ci].norm(2);
    if (n_classes > 0)
        l_reg = l_reg / n_classes;

    return opts.alpha * l_var + opts.beta * l_dis + opts.gamma * l_reg;
}

} // namespace vision
} // namespace models
} // namespace dm
