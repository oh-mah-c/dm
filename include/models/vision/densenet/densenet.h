#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// DenseNet — Densely Connected Convolutional Networks
// Huang, Liu, van der Maaten & Weinberger, CVPR 2017  (arXiv:1608.06993v5)
//
// Key equation (Eq. 2):
//   x_l = H_l([x_0, x_1, ..., x_{l-1}])
//
// Each layer concatenates feature maps from ALL preceding layers within the
// same dense block rather than summing them (cf. ResNet).
//
// H_l = BN → ReLU → Conv (plain DenseNet)
//       BN → ReLU → Conv(1×1) → BN → ReLU → Conv(3×3)  (DenseNet-B bottleneck)
//
// Growth rate k: each layer appends exactly k new feature maps to the
// "collective knowledge" tensor (Section 3, paragraph "Growth rate").
//
// Compression factor θ ∈ (0, 1]: transition layers halve channels when θ < 1.
// DenseNet-BC uses both bottleneck + compression (θ = 0.5, paper Section 3).
//
// ImageNet architectures (Table 1):
//   DenseNet-121: [6, 12, 24, 16]  k=32
//   DenseNet-169: [6, 12, 32, 32]  k=32
//   DenseNet-201: [6, 12, 48, 32]  k=32
//   DenseNet-264: [6, 12, 64, 48]  k=32
//   All start with 7×7 conv(stride 2) → 3×3 maxpool(stride 2) → 4 dense blocks
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <vector>
#include <string>

namespace dm {
namespace models {
namespace vision {

// ─────────────────────────────────────────────────────────────────────────────
// DenseLayer  — one H_l composite function inside a dense block.
//
// Plain (bottleneck=false): BN-ReLU-Conv(3×3, k output channels)
// Bottleneck (bottleneck=true): BN-ReLU-Conv(1×1, 4k) → BN-ReLU-Conv(3×3, k)
//   per paper Section 3, "Bottleneck layers".
// Drop rate (dropout after Conv, applied in training only).
// ─────────────────────────────────────────────────────────────────────────────
struct DenseLayerImpl : torch::nn::Module {
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Conv2d      conv1{nullptr};
    torch::nn::BatchNorm2d bn2{nullptr};    // bottleneck only
    torch::nn::Conv2d      conv2{nullptr};  // bottleneck only
    torch::nn::Dropout2d   drop{nullptr};   // no-op when drop_rate=0
    bool use_bottleneck;

    // in_channels: total channels entering this layer (k0 + k*(l-1))
    // growth_rate: k, number of output channels this layer produces
    DenseLayerImpl(int64_t in_channels,
                   int64_t growth_rate,
                   bool    bottleneck   = true,
                   double  drop_rate    = 0.0);

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DenseLayer);

// ─────────────────────────────────────────────────────────────────────────────
// DenseBlock  — L densely-connected layers at the same feature-map resolution.
//
// After forward() the output tensor has in_channels + num_layers*growth_rate
// channels (all layers cat'd together, identical to the final x_L).
// ─────────────────────────────────────────────────────────────────────────────
struct DenseBlockImpl : torch::nn::Module {
    torch::nn::ModuleList layers{};
    int64_t out_channels_;  // in_channels + num_layers * growth_rate

    DenseBlockImpl(int64_t in_channels,
                   int64_t num_layers,
                   int64_t growth_rate,
                   bool    bottleneck = true,
                   double  drop_rate  = 0.0);

    // input: [B, in_channels, H, W]
    // output: [B, in_channels + num_layers*growth_rate, H, W]
    torch::Tensor forward(torch::Tensor x);

    int64_t out_channels() const { return out_channels_; }
};
TORCH_MODULE(DenseBlock);

// ─────────────────────────────────────────────────────────────────────────────
// TransitionLayer  — BN-ReLU-Conv(1×1)-AvgPool(2×2, stride 2).
//
// Compression: output channels = floor(theta * in_channels).
// theta=0.5 for DenseNet-C/BC; theta=1.0 (no compression) otherwise.
// Paper Section 3, "Compression".
// ─────────────────────────────────────────────────────────────────────────────
struct TransitionLayerImpl : torch::nn::Module {
    torch::nn::BatchNorm2d bn{nullptr};
    torch::nn::Conv2d      conv{nullptr};
    torch::nn::AvgPool2d   pool{nullptr};
    int64_t out_channels_;

    TransitionLayerImpl(int64_t in_channels, double theta = 0.5);

    torch::Tensor forward(torch::Tensor x);

    int64_t out_channels() const { return out_channels_; }
};
TORCH_MODULE(TransitionLayer);

// ─────────────────────────────────────────────────────────────────────────────
// DenseNet
//
// Paper Table 1 (ImageNet) / Section 3 "Implementation Details" (CIFAR):
//
// ImageNet layout:
//   conv1  : 7×7, stride=2, out=2k, BN, ReLU       → 112×112
//   maxpool: 3×3, stride=2, padding=1               →  56×56
//   dense1 + trans1                                 →  28×28
//   dense2 + trans2                                 →  14×14
//   dense3 + trans3                                 →   7×7
//   dense4                                          →   7×7
//   BN-ReLU (final norm before pooling)
//   global average pool                             →   1×1
//   fully-connected → num_classes
//
// CIFAR layout (small_input=true):
//   conv1  : 3×3, stride=1, out=16 (or 2k for DenseNet-BC), padding=1
//   3 dense blocks + 2 transition layers
//   feature-map sizes: 32×32, 16×16, 8×8
// ─────────────────────────────────────────────────────────────────────────────
struct DenseNetImpl : torch::nn::Module {
    torch::nn::Conv2d      conv0{nullptr};
    torch::nn::BatchNorm2d bn0{nullptr};
    torch::nn::MaxPool2d   pool0{nullptr};   // ImageNet only (registered when !small_input)

    // Dense blocks and transition layers stored in a sequential-style list.
    // Indexed as: dense0, trans0, dense1, trans1, ..., denseN-1 (no trans last)
    torch::nn::ModuleList features{};

    torch::nn::BatchNorm2d norm_final{nullptr};
    torch::nn::Linear      classifier{nullptr};
    bool small_input_{false};

    // block_cfg[i] = number of layers in dense block i
    DenseNetImpl(const std::vector<int64_t>& block_cfg,
                 int64_t    growth_rate  = 32,
                 double     theta        = 0.5,
                 bool       bottleneck   = true,
                 int64_t    num_classes  = 1000,
                 bool       small_input  = false,
                 double     drop_rate    = 0.0);

    // x: [B, C, H, W] → [B, num_classes]
    torch::Tensor forward(torch::Tensor x);

private:
    // Track channel count during construction.
    int64_t feature_channels_{0};
};

TORCH_MODULE(DenseNet);

// ─────────────────────────────────────────────────────────────────────────────
// Factory functions — exact paper Table 1 configurations (ImageNet, k=32)
// ─────────────────────────────────────────────────────────────────────────────
std::shared_ptr<DenseNetImpl> make_densenet121(int64_t num_classes = 1000);
std::shared_ptr<DenseNetImpl> make_densenet169(int64_t num_classes = 1000);
std::shared_ptr<DenseNetImpl> make_densenet201(int64_t num_classes = 1000);
std::shared_ptr<DenseNetImpl> make_densenet264(int64_t num_classes = 1000);

// CIFAR configurations from Table 2 of the paper.
// All use small_input=true (3×3 stem, no maxpool, 3 dense blocks).
// DenseNet-BC uses bottleneck=true, theta=0.5.
std::shared_ptr<DenseNetImpl> make_densenet_cifar(
    int64_t depth, int64_t growth_rate, bool bottleneck = false,
    int64_t num_classes = 10);

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration — defaults match paper Section 4.2 (ImageNet).
// ─────────────────────────────────────────────────────────────────────────────
struct DenseNetTrainConfig {
    // SGD — paper: lr=0.1, momentum=0.9, weight_decay=1e-4
    double   lr           = 0.1;
    double   momentum     = 0.9;
    double   weight_decay = 1e-4;

    // LR schedule — paper: divide by 10 at epoch 30 and 60 (ImageNet)
    std::vector<int64_t> lr_milestones = {30, 60};
    double   lr_gamma     = 0.1;

    // Batch size — paper: 256 on 8 GPUs; use what fits locally
    int64_t  batch_size   = 256;

    // Total epochs — paper: 90 epochs (ImageNet)
    int64_t  max_epochs   = 90;

    torch::Device device  = torch::kCPU;
};

} // namespace vision
} // namespace models
} // namespace dm
