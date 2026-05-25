// ─────────────────────────────────────────────────────────────────────────────
// VGGNet — Very Deep Convolutional Networks for Large-Scale Image Recognition
// Simonyan & Zisserman, ICLR 2015  (arXiv:1409.1556v6)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/vgg/vgg.h"

#include <torch/torch.h>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace dm {
namespace models {
namespace vision {

// ═══════════════════════════════════════════════════════════════════════════
// Feature block definitions (Table 1)
//
// Each config is described as a flat sequence of layer descriptors:
//   positive int  → conv3×3 with that many output channels
//   -1            → conv1×1 with |value| output channels  (config C)
//   0             → MaxPool 2×2, stride 2
//
// Paper Section 2.1:
//   "filters with a very small receptive field: 3×3"
//   "The convolution stride is fixed to 1 pixel"
//   "spatial padding of conv layer input is 1 pixel for 3×3 conv. layers"
//   "Max-pooling is performed over a 2×2 pixel window, with stride 2"
// ═══════════════════════════════════════════════════════════════════════════

// Sentinel used internally: positive = conv3×3 out_ch, negative = conv1×1 |out_ch|
struct LayerDesc {
    int channels;   // >0: output channels; 0: maxpool
    bool conv1x1;   // true if 1×1 conv (config C)
};

// Encode Table 1 configurations
static std::vector<LayerDesc> get_config(VGGConfig cfg) {
    // 0 = maxpool; {ch, false} = conv3×3; {ch, true} = conv1×1
    using L = LayerDesc;
    switch (cfg) {
    case VGGConfig::A:  // 11 weight layers
        return {
            {64,false},  {0,false},
            {128,false}, {0,false},
            {256,false},{256,false},{0,false},
            {512,false},{512,false},{0,false},
            {512,false},{512,false},{0,false},
        };
    case VGGConfig::B:  // 13 weight layers
        return {
            {64,false},{64,false},   {0,false},
            {128,false},{128,false}, {0,false},
            {256,false},{256,false}, {0,false},
            {512,false},{512,false}, {0,false},
            {512,false},{512,false}, {0,false},
        };
    case VGGConfig::C:  // 16 weight layers — conv1×1 in blocks 3,4,5
        return {
            {64,false},{64,false},         {0,false},
            {128,false},{128,false},       {0,false},
            {256,false},{256,false},{256,true}, {0,false},
            {512,false},{512,false},{512,true}, {0,false},
            {512,false},{512,false},{512,true}, {0,false},
        };
    case VGGConfig::D:  // 16 weight layers — "VGG-16"
        return {
            {64,false},{64,false},           {0,false},
            {128,false},{128,false},         {0,false},
            {256,false},{256,false},{256,false}, {0,false},
            {512,false},{512,false},{512,false}, {0,false},
            {512,false},{512,false},{512,false}, {0,false},
        };
    case VGGConfig::E:  // 19 weight layers — "VGG-19"
        return {
            {64,false},{64,false},                   {0,false},
            {128,false},{128,false},                 {0,false},
            {256,false},{256,false},{256,false},{256,false}, {0,false},
            {512,false},{512,false},{512,false},{512,false}, {0,false},
            {512,false},{512,false},{512,false},{512,false}, {0,false},
        };
    }
    return {};
}

// ═══════════════════════════════════════════════════════════════════════════
// Build the feature extractor (conv blocks + maxpools)
// ═══════════════════════════════════════════════════════════════════════════
/* static */ torch::nn::Sequential VGGImpl::make_features(VGGConfig cfg, bool batch_norm) {
    torch::nn::Sequential seq;
    auto layers = get_config(cfg);
    int64_t in_ch = 3;

    for (auto& ld : layers) {
        if (ld.channels == 0) {
            // MaxPool 2×2, stride 2  (paper Section 2.1)
            seq->push_back(
                torch::nn::MaxPool2d(
                    torch::nn::MaxPool2dOptions(2).stride(2)));
        } else {
            int64_t out_ch = ld.channels;
            int     ksize  = ld.conv1x1 ? 1 : 3;
            int     pad    = ld.conv1x1 ? 0 : 1;

            seq->push_back(
                torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(in_ch, out_ch, ksize)
                        .stride(1).padding(pad)));

            if (batch_norm)
                seq->push_back(torch::nn::BatchNorm2d(out_ch));

            // Paper Section 2.1: "All hidden layers are equipped with ReLU"
            seq->push_back(torch::nn::ReLU(
                torch::nn::ReLUOptions().inplace(true)));

            in_ch = out_ch;
        }
    }
    return seq;
}

// ═══════════════════════════════════════════════════════════════════════════
// VGGImpl constructor
// ═══════════════════════════════════════════════════════════════════════════
VGGImpl::VGGImpl(VGGConfig cfg, int64_t num_classes, bool batch_norm) {
    // Feature extractor — conv blocks + maxpools
    features = register_module("features", make_features(cfg, batch_norm));

    // Adaptive average pool — maps last feature map to 7×7 regardless of input
    // (for dense/multi-scale evaluation, Section 3.2)
    avgpool = register_module("avgpool",
        torch::nn::AdaptiveAvgPool2d(
            torch::nn::AdaptiveAvgPool2dOptions({7, 7})));

    // Classifier — paper Section 2.1:
    //   "three Fully-Connected (FC) layers: the first two have 4096 channels …
    //    the third performs 1000-way classification"
    //   "Dropout regularisation for the first two FC layers (dropout ratio 0.5)"
    classifier = register_module("classifier", torch::nn::Sequential(
        torch::nn::Linear(512 * 7 * 7, 4096),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Dropout(0.5),
        torch::nn::Linear(4096, 4096),
        torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)),
        torch::nn::Dropout(0.5),
        torch::nn::Linear(4096, num_classes)
    ));

    // Weight init — paper Section 3.1:
    //   "weights sampled from a normal distribution with zero mean and 10^{-2} variance"
    //   "biases initialised with zero"
    for (auto& m : modules(false)) {
        if (auto* c = m->as<torch::nn::Conv2dImpl>()) {
            torch::nn::init::normal_(c->weight, 0.0, 0.01);
            torch::nn::init::constant_(c->bias, 0.0);
        } else if (auto* l = m->as<torch::nn::LinearImpl>()) {
            torch::nn::init::normal_(l->weight, 0.0, 0.01);
            torch::nn::init::constant_(l->bias, 0.0);
        } else if (auto* b = m->as<torch::nn::BatchNorm2dImpl>()) {
            torch::nn::init::constant_(b->weight, 1.0);
            torch::nn::init::constant_(b->bias,   0.0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward pass
// ═══════════════════════════════════════════════════════════════════════════
torch::Tensor VGGImpl::forward(torch::Tensor x) {
    x = features->forward(x);       // conv blocks
    x = avgpool->forward(x);        // 7×7 adaptive pool
    x = torch::flatten(x, 1);       // flatten to 512*7*7
    x = classifier->forward(x);     // FC layers (includes dropout)
    return x;
    // Note: paper uses softmax at test time; cross-entropy loss applies
    // log-softmax internally so we return raw logits here.
}

// ═══════════════════════════════════════════════════════════════════════════
// Factory functions (Table 1)
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<VGGImpl> make_vgg_a(int64_t num_classes, bool batch_norm) {
    return std::make_shared<VGGImpl>(VGGConfig::A, num_classes, batch_norm);
}
std::shared_ptr<VGGImpl> make_vgg_b(int64_t num_classes, bool batch_norm) {
    return std::make_shared<VGGImpl>(VGGConfig::B, num_classes, batch_norm);
}
std::shared_ptr<VGGImpl> make_vgg_c(int64_t num_classes, bool batch_norm) {
    return std::make_shared<VGGImpl>(VGGConfig::C, num_classes, batch_norm);
}
std::shared_ptr<VGGImpl> make_vgg16(int64_t num_classes, bool batch_norm) {
    return std::make_shared<VGGImpl>(VGGConfig::D, num_classes, batch_norm);
}
std::shared_ptr<VGGImpl> make_vgg19(int64_t num_classes, bool batch_norm) {
    return std::make_shared<VGGImpl>(VGGConfig::E, num_classes, batch_norm);
}

// ═══════════════════════════════════════════════════════════════════════════
// Training helpers
// ═══════════════════════════════════════════════════════════════════════════

static std::pair<int64_t, int64_t> topk_correct(const torch::Tensor& output,
                                                  const torch::Tensor& target,
                                                  int64_t k) {
    auto pred    = std::get<1>(output.topk(k, 1, true, true)); // [N, k]
    auto correct = pred.eq(target.view({-1, 1}).expand_as(pred));
    return {correct.any(1).sum().item<int64_t>(), target.size(0)};
}

float vgg_train_epoch(
    torch::nn::AnyModule& model,
    torch::optim::SGD&    optimizer,
    torch::Device         device,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batches)
{
    model.ptr()->train();
    double  total_loss = 0.0;
    int64_t n          = 0;

    for (auto& [data, target] : batches) {
        auto x = data.to(device);
        auto y = target.to(device);

        optimizer.zero_grad();
        auto output = model.forward<torch::Tensor>(x);
        // Paper Section 3.1: "multinomial logistic regression objective"
        auto loss = torch::nn::functional::cross_entropy(output, y);
        loss.backward();
        optimizer.step();

        total_loss += loss.item<double>();
        ++n;
    }
    return n > 0 ? static_cast<float>(total_loss / n) : 0.f;
}

std::pair<float, float> vgg_evaluate(
    torch::nn::AnyModule& model,
    torch::Device         device,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batches)
{
    model.ptr()->eval();
    torch::NoGradGuard no_grad;
    int64_t top1 = 0, top5 = 0, total = 0;

    for (auto& [data, target] : batches) {
        auto x      = data.to(device);
        auto y      = target.to(device);
        auto output = model.forward<torch::Tensor>(x);

        auto [c1, n]  = topk_correct(output, y, 1);
        top1  += c1;
        total += n;

        auto [c5, _n] = topk_correct(output, y,
                                      std::min<int64_t>(5, output.size(1)));
        top5 += c5;
    }

    if (total == 0) return {0.f, 0.f};
    return {static_cast<float>(top1) / total,
            static_cast<float>(top5) / total};
}

void vgg_train(
    torch::nn::AnyModule&   model,
    const VGGTrainConfig&   cfg,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& train_batches,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& val_batches,
    const std::string& save_path)
{
    model.ptr()->to(cfg.device);

    // Paper Section 3.1: "SGD, momentum 0.9, weight decay 5·10^{−4}"
    torch::optim::SGD optimizer(
        model.ptr()->parameters(),
        torch::optim::SGDOptions(cfg.lr)
            .momentum(cfg.momentum)
            .weight_decay(cfg.weight_decay));

    float best_top1 = 0.f;

    for (int64_t epoch = 1; epoch <= cfg.max_epochs; ++epoch) {
        // LR schedule: paper decreases LR 3× when val accuracy stops improving
        for (int64_t ms : cfg.lr_milestones) {
            if (epoch == ms) {
                for (auto& pg : optimizer.param_groups()) {
                    auto& opts = static_cast<torch::optim::SGDOptions&>(pg.options());
                    opts.lr(opts.lr() * cfg.lr_gamma);
                }
                double cur_lr = static_cast<torch::optim::SGDOptions&>(
                    optimizer.param_groups()[0].options()).lr();
                std::cout << "[VGG] epoch " << epoch << " LR → " << cur_lr << "\n";
            }
        }

        float loss        = vgg_train_epoch(model, optimizer, cfg.device, train_batches);
        auto [top1, top5] = vgg_evaluate(model, cfg.device, val_batches);

        std::cout << "[VGG] " << epoch << "/" << cfg.max_epochs
                  << "  loss=" << loss
                  << "  top1=" << top1 * 100.f << "%"
                  << "  top5=" << top5 * 100.f << "%\n";

        if (top1 > best_top1) {
            best_top1 = top1;
            torch::serialize::OutputArchive archive;
            model.ptr()->save(archive);
            archive.save_to(save_path);
            std::cout << "[VGG] checkpoint saved → " << save_path << "\n";
        }
    }
}

} // namespace vision
} // namespace models
} // namespace dm
