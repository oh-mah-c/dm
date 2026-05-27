#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// CCNet — Criss-Cross Network for Semantic Segmentation
// Huang et al., IEEE TPAMI 2020  (arXiv:1811.11721v2)
//
// Architecture (Fig. 2, Section 3.1):
//   input image
//     -> CNN backbone (feature extractor, output stride 8)
//     -> Reduction conv (1x1, C -> reduced_channels)
//     -> Recurrent Criss-Cross Attention (RCCA), R=2 loops
//     -> concat(H'', X)
//     -> head convs (BN-ReLU) + 1x1 segmentation output
//
// Key components:
//   CrissCrossAttention (dm::prim): single loop of criss-cross attention
//     (Section 3.2, Eqs. 1-2).  Parameters are shared across both loops.
//   RCCA module: applies CrissCrossAttention R times recurrently (Section 3.3).
//     R=2 achieves full-image context with ~11x less memory than non-local.
//   Category Consistent Loss (CCL, Section 3.4): auxiliary loss that enforces
//     intra-class compactness and inter-class separation in feature space
//     (Eqs. 3-7).  Only applied during training.
//
// The backbone is modular: any feature extractor that returns an output-stride-8
// feature map can be used.  CCNetImpl accepts a user-supplied backbone (AnyModule)
// or can be constructed with a built-in dilated ResNet-101 stub.
//
// Training: SGD, lr=0.01 (poly schedule), momentum=0.9, weight_decay=0.0001.
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <torch/nn/modules/criss_cross_attention.h>
#include <vector>
#include <string>

namespace dm {
namespace models {
namespace vision {

// ─────────────────────────────────────────────────────────────────────────────
// RCCAModuleImpl — Recurrent Criss-Cross Attention (Section 3.3)
//
// Applies a single CrissCrossAttention module R times sequentially.
// Parameters are shared across all R iterations (paper: "Two criss-cross
// attention modules before and after share the same parameters").
//
// Input:  x: [B, in_channels, H, W]
// Output: H'': [B, in_channels, H, W]
// ─────────────────────────────────────────────────────────────────────────────
struct RCCAModuleImpl : torch::nn::Module {
    dm::prim::CrissCrossAttention cca{nullptr};
    int rcca_loops;

    // in_channels: C (feature channels fed into RCCA)
    // key_channels: C' (query/key projection dim, paper default C/8)
    // rcca_loops:   R (paper default 2)
    RCCAModuleImpl(int64_t in_channels,
                   int64_t key_channels,
                   int     rcca_loops = 2);

    // x: [B, in_channels, H, W] -> [B, in_channels, H, W]
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(RCCAModule);

// ─────────────────────────────────────────────────────────────────────────────
// CCNetHeadImpl — segmentation head after RCCA
//
// Takes concatenated [H'', X] (2*in_channels) and produces per-pixel logits.
// Paper: "one or several convolutional layers with batch normalization and
// activation for feature fusion, then a 1x1 segmentation layer".
// ─────────────────────────────────────────────────────────────────────────────
struct CCNetHeadImpl : torch::nn::Module {
    torch::nn::Sequential fuse{};
    torch::nn::Conv2d     seg_conv{nullptr};

    // in_channels: channels entering the head (= 2 * rcca_in after concat)
    // mid_channels: intermediate channels in fuse convs (paper: 512)
    // num_classes: segmentation output channels
    CCNetHeadImpl(int64_t in_channels,
                  int64_t mid_channels,
                  int64_t num_classes);

    // x: [B, in_channels, H, W] -> [B, num_classes, H, W]
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(CCNetHead);

// ─────────────────────────────────────────────────────────────────────────────
// CCNetImpl — full CCNet for semantic segmentation
//
// Architecture:
//   backbone          -> X:  [B, backbone_out_ch, H/8, W/8]
//   reduction conv    -> H:  [B, rcca_channels, H/8, W/8]
//   RCCA (R loops)    -> H'': [B, rcca_channels, H/8, W/8]
//   concat(H'', X)    -> fused: [B, rcca_channels+backbone_out_ch, H/8, W/8]
//   head              -> logits: [B, num_classes, H/8, W/8]
//   bilinear upsample -> [B, num_classes, H, W]   (at inference)
//
// The backbone is passed as a torch::nn::AnyModule (forward: Tensor->Tensor).
// ─────────────────────────────────────────────────────────────────────────────
struct CCNetImpl : torch::nn::Module {
    torch::nn::AnyModule  backbone;
    torch::nn::Conv2d     reduction{nullptr};   // backbone_out -> rcca_channels
    torch::nn::BatchNorm2d reduction_bn{nullptr};
    RCCAModule            rcca{nullptr};
    CCNetHead             head{nullptr};

    // backbone_out_ch: channels output by backbone (ResNet-101 dilated = 2048)
    // rcca_channels:   C fed into RCCA after reduction (paper: 512)
    // key_channels:    C' for Q/K projections (paper: 64 = 512/8)
    // mid_channels:    head fusion channels (paper: 512)
    // num_classes:     segmentation output channels
    // rcca_loops:      R (default 2)
    CCNetImpl(torch::nn::AnyModule backbone_,
              int64_t backbone_out_ch,
              int64_t num_classes,
              int64_t rcca_channels = 512,
              int64_t key_channels  = 64,
              int64_t mid_channels  = 512,
              int     rcca_loops    = 2);

    // x: [B, 3, H, W] -> [B, num_classes, H, W]  (logits, not upsampled)
    // Returns raw logits at output stride 8 (H/8 x W/8).
    // Call torch::nn::functional::interpolate to upsample to full resolution.
    torch::Tensor forward(torch::Tensor x);

 private:
    int64_t rcca_channels_{512};
};
TORCH_MODULE(CCNet);

// ─────────────────────────────────────────────────────────────────────────────
// Category Consistent Loss (CCL, Section 3.4, Eqs. 3-7)
//
// Discriminative loss for intra-class compactness and inter-class separation.
// Applied to the RCCA output features (after dim reduction) during training.
//
// l_var: penalises large intra-class feature spread   (Eq. 3, 6)
// l_dis: penalises small inter-class center distance  (Eq. 4, 7)
// l_reg: draws all centers toward the origin          (Eq. 5)
// total: l_seg + alpha*l_var + beta*l_dis + gamma*l_reg  (Eq. 8)
//
// Paper defaults: delta_v=0.5, delta_d=1.5, alpha=beta=1, gamma=0.001,
//                 dim_reduction to 16 channels before computing loss.
// ─────────────────────────────────────────────────────────────────────────────
struct CCLLossOptions {
    // Margins
    float delta_v = 0.5f;
    float delta_d = 1.5f;
    // Loss weights (Eq. 8)
    float alpha   = 1.0f;
    float beta    = 1.0f;
    float gamma   = 0.001f;
};

// Compute CCL.
// features: [B, C, H, W]  — the RCCA output features (already dim-reduced)
// targets:  [B, H, W]     — ground-truth class labels (int64, ignore_index=-1)
// Returns: scalar tensor (total CCL value, differentiable)
torch::Tensor ccl_loss(const torch::Tensor& features,
                       const torch::Tensor& targets,
                       const CCLLossOptions& opts = CCLLossOptions{},
                       int64_t ignore_index = 255);

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration — defaults from paper Section 4.2
// ─────────────────────────────────────────────────────────────────────────────
struct CCNetTrainConfig {
    // SGD poly LR schedule: lr * (1 - iter/max_iter)^power
    double  lr          = 0.01;
    double  momentum    = 0.9;
    double  weight_decay= 0.0001;
    double  lr_power    = 0.9;
    int64_t max_iter    = 40000;

    // CCL loss weights (Eq. 8)
    float   ccl_alpha   = 1.0f;
    float   ccl_beta    = 1.0f;
    float   ccl_gamma   = 0.001f;

    // Ignore label for segmentation loss (Cityscapes: 255)
    int64_t ignore_index= 255;

    // Input crop size (Cityscapes training: 769x769)
    int64_t crop_size   = 769;

    torch::Device device = torch::kCPU;
};

} // namespace vision
} // namespace models
} // namespace dm
