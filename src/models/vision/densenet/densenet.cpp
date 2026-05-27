// ─────────────────────────────────────────────────────────────────────────────
// DenseNet — Densely Connected Convolutional Networks
// Huang et al., CVPR 2017  (arXiv:1608.06993v5)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/densenet/densenet.h"

#include <torch/torch.h>
#include <cassert>
#include <cmath>

namespace dm {
namespace models {
namespace vision {

// ═══════════════════════════════════════════════════════════════════════════
// DenseLayer
// ═══════════════════════════════════════════════════════════════════════════

DenseLayerImpl::DenseLayerImpl(int64_t in_channels,
                               int64_t growth_rate,
                               bool    bottleneck,
                               double  drop_rate)
    : use_bottleneck(bottleneck)
{
    if (bottleneck) {
        // BN → ReLU → Conv(1×1, 4k, no bias) → BN → ReLU → Conv(3×3, k, no bias)
        int64_t inter = 4 * growth_rate;
        bn1   = register_module("bn1",   torch::nn::BatchNorm2d(in_channels));
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, inter, 1)
                .bias(false)));
        bn2   = register_module("bn2",   torch::nn::BatchNorm2d(inter));
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(inter, growth_rate, 3)
                .padding(1).bias(false)));
    } else {
        // BN → ReLU → Conv(3×3, k, no bias)
        bn1   = register_module("bn1",   torch::nn::BatchNorm2d(in_channels));
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, growth_rate, 3)
                .padding(1).bias(false)));
    }

    drop = register_module("drop", torch::nn::Dropout2d(drop_rate));
}

torch::Tensor DenseLayerImpl::forward(torch::Tensor x) {
    torch::Tensor out;

    if (use_bottleneck) {
        out = conv1->forward(torch::relu(bn1->forward(x)));
        out = conv2->forward(torch::relu(bn2->forward(out)));
    } else {
        out = conv1->forward(torch::relu(bn1->forward(x)));
    }

    out = drop->forward(out);

    // Dense connection: concatenate new features with all prior input channels.
    return torch::cat({x, out}, /*dim=*/1);
}

// ═══════════════════════════════════════════════════════════════════════════
// DenseBlock
// ═══════════════════════════════════════════════════════════════════════════

DenseBlockImpl::DenseBlockImpl(int64_t in_channels,
                               int64_t num_layers,
                               int64_t growth_rate,
                               bool    bottleneck,
                               double  drop_rate)
{
    layers = register_module("layers", torch::nn::ModuleList());
    int64_t ch = in_channels;
    for (int64_t i = 0; i < num_layers; ++i) {
        layers->push_back(
            DenseLayer(ch, growth_rate, bottleneck, drop_rate));
        ch += growth_rate;
    }
    out_channels_ = ch;
}

torch::Tensor DenseBlockImpl::forward(torch::Tensor x) {
    for (auto& mod : *layers)
        x = mod->as<DenseLayerImpl>()->forward(x);
    return x;
}

// ═══════════════════════════════════════════════════════════════════════════
// TransitionLayer
// ═══════════════════════════════════════════════════════════════════════════

TransitionLayerImpl::TransitionLayerImpl(int64_t in_channels, double theta) {
    out_channels_ = static_cast<int64_t>(std::floor(theta * in_channels));
    bn   = register_module("bn",
        torch::nn::BatchNorm2d(in_channels));
    conv = register_module("conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels_, 1)
            .bias(false)));
    // 2×2 average pool, stride 2
    pool = register_module("pool",
        torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(2).stride(2)));
}

torch::Tensor TransitionLayerImpl::forward(torch::Tensor x) {
    return pool->forward(conv->forward(torch::relu(bn->forward(x))));
}

// ═══════════════════════════════════════════════════════════════════════════
// DenseNet
// ═══════════════════════════════════════════════════════════════════════════

DenseNetImpl::DenseNetImpl(const std::vector<int64_t>& block_cfg,
                           int64_t  growth_rate,
                           double   theta,
                           bool     bottleneck,
                           int64_t  num_classes,
                           bool     small_input,
                           double   drop_rate)
{
    features = register_module("features", torch::nn::ModuleList());

    small_input_ = small_input;
    int64_t init_ch;
    if (small_input) {
        // CIFAR: 3×3 conv, stride=1, no maxpool
        // Paper Section 3 "Implementation details": 16 channels (or 2k for BC)
        init_ch = bottleneck ? 2 * growth_rate : 16;
        conv0 = register_module("conv0",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, init_ch, 3)
                .stride(1).padding(1).bias(false)));
        bn0 = register_module("bn0", torch::nn::BatchNorm2d(init_ch));
    } else {
        // ImageNet: 7×7 conv, stride=2 → 3×3 maxpool, stride=2
        // Initial channels = 2k (Section 3)
        init_ch = 2 * growth_rate;
        conv0 = register_module("conv0",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, init_ch, 7)
                .stride(2).padding(3).bias(false)));
        bn0 = register_module("bn0", torch::nn::BatchNorm2d(init_ch));
        pool0 = register_module("pool0",
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
    }

    int64_t ch = init_ch;
    for (size_t i = 0; i < block_cfg.size(); ++i) {
        auto blk = DenseBlock(ch, block_cfg[i], growth_rate, bottleneck, drop_rate);
        ch = blk->out_channels();
        features->push_back(std::move(blk));

        // Transition layer after every block except the last.
        if (i + 1 < block_cfg.size()) {
            auto trans = TransitionLayer(ch, theta);
            ch = trans->out_channels();
            features->push_back(std::move(trans));
        }
    }

    feature_channels_ = ch;
    norm_final   = register_module("norm_final", torch::nn::BatchNorm2d(ch));
    classifier   = register_module("classifier", torch::nn::Linear(ch, num_classes));

    // Weight initialisation — paper Section 4.2 / He et al. 2015
    for (auto& m : modules(/*include_self=*/false)) {
        if (auto* conv = m->as<torch::nn::Conv2dImpl>()) {
            torch::nn::init::kaiming_normal_(
                conv->weight, 0.0, torch::kFanOut, torch::kReLU);
        } else if (auto* bn = m->as<torch::nn::BatchNorm2dImpl>()) {
            torch::nn::init::ones_(bn->weight);
            torch::nn::init::zeros_(bn->bias);
        } else if (auto* fc = m->as<torch::nn::LinearImpl>()) {
            torch::nn::init::constant_(fc->bias, 0);
        }
    }
}

torch::Tensor DenseNetImpl::forward(torch::Tensor x) {
    // Stem
    x = torch::relu(bn0->forward(conv0->forward(x)));
    if (!small_input_)
        x = pool0->forward(x);

    // Dense blocks + transitions
    for (size_t i = 0; i < features->size(); ++i) {
        auto mod = (*features)[i];
        if (auto* blk = mod->as<DenseBlockImpl>())
            x = blk->forward(x);
        else if (auto* tr = mod->as<TransitionLayerImpl>())
            x = tr->forward(x);
    }

    // Final BN-ReLU
    x = torch::relu(norm_final->forward(x));

    // Global average pool → flatten → fc
    x = torch::adaptive_avg_pool2d(x, {1, 1});
    x = x.view({x.size(0), -1});
    return classifier->forward(x);
}

// ═══════════════════════════════════════════════════════════════════════════
// Factory — ImageNet (Table 1, k=32, theta=0.5, bottleneck=true)
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<DenseNetImpl> make_densenet121(int64_t num_classes) {
    return std::make_shared<DenseNetImpl>(
        std::vector<int64_t>{6, 12, 24, 16}, 32, 0.5, true, num_classes, false, 0.0);
}

std::shared_ptr<DenseNetImpl> make_densenet169(int64_t num_classes) {
    return std::make_shared<DenseNetImpl>(
        std::vector<int64_t>{6, 12, 32, 32}, 32, 0.5, true, num_classes, false, 0.0);
}

std::shared_ptr<DenseNetImpl> make_densenet201(int64_t num_classes) {
    return std::make_shared<DenseNetImpl>(
        std::vector<int64_t>{6, 12, 48, 32}, 32, 0.5, true, num_classes, false, 0.0);
}

std::shared_ptr<DenseNetImpl> make_densenet264(int64_t num_classes) {
    return std::make_shared<DenseNetImpl>(
        std::vector<int64_t>{6, 12, 64, 48}, 32, 0.5, true, num_classes, false, 0.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// Factory — CIFAR (Table 2)
//
// Total depth L for a plain DenseNet with 3 blocks of equal size:
//   L = 3 * num_layers_per_block + 4  (3 conv in each block + 1 stem + 1 fc)
// For bottleneck: each "layer" is 2 convs, so:
//   L = 3 * num_layers_per_block * 2 + 4
// The paper parameterizes by depth L and k.
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<DenseNetImpl> make_densenet_cifar(
    int64_t depth, int64_t growth_rate, bool bottleneck, int64_t num_classes)
{
    int64_t n;
    if (bottleneck) {
        // Each "layer" contributes 2 conv ops; L = 3*n*2 + 4
        assert((depth - 4) % 6 == 0);
        n = (depth - 4) / 6;
    } else {
        assert((depth - 4) % 3 == 0);
        n = (depth - 4) / 3;
    }
    return std::make_shared<DenseNetImpl>(
        std::vector<int64_t>{n, n, n}, growth_rate,
        bottleneck ? 0.5 : 1.0,
        bottleneck, num_classes, /*small_input=*/true, 0.0);
}

} // namespace vision
} // namespace models
} // namespace dm
