// ─────────────────────────────────────────────────────────────────────────────
// ResNet — Deep Residual Learning for Image Recognition
// He et al., CVPR 2016  (https://arxiv.org/abs/1512.03385)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/resnet/resnet.h"

#include <torch/torch.h>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace dm {
namespace models {
namespace vision {

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

static torch::nn::Conv2d conv3x3(int64_t in_c, int64_t out_c, int64_t stride = 1) {
    return torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_c, out_c, 3)
            .stride(stride).padding(1).bias(false));
}

static torch::nn::Conv2d conv1x1(int64_t in_c, int64_t out_c, int64_t stride = 1) {
    return torch::nn::Conv2d(
        torch::nn::Conv2dOptions(in_c, out_c, 1)
            .stride(stride).bias(false));
}

// ═══════════════════════════════════════════════════════════════════════════
// BasicBlock
// ═══════════════════════════════════════════════════════════════════════════

BasicBlockImpl::BasicBlockImpl(int64_t in_planes, int64_t planes,
                                int64_t stride, torch::nn::Sequential ds)
    : conv1(conv3x3(in_planes, planes, stride))
    , conv2(conv3x3(planes,    planes))
    , bn1(torch::nn::BatchNorm2d(planes))
    , bn2(torch::nn::BatchNorm2d(planes))
    , downsample(std::move(ds))
{
    register_module("conv1", conv1);
    register_module("bn1",   bn1);
    register_module("conv2", conv2);
    register_module("bn2",   bn2);
    if (!downsample->is_empty())
        register_module("downsample", downsample);
}

torch::Tensor BasicBlockImpl::forward(torch::Tensor x) {
    torch::Tensor identity = x;

    auto out = torch::relu(bn1->forward(conv1->forward(x)));
    out = bn2->forward(conv2->forward(out));

    if (!downsample->is_empty())
        identity = downsample->forward(x);

    out += identity;
    return torch::relu(out);
}

// ═══════════════════════════════════════════════════════════════════════════
// Bottleneck
// ═══════════════════════════════════════════════════════════════════════════

BottleneckImpl::BottleneckImpl(int64_t in_planes, int64_t planes,
                                int64_t stride, torch::nn::Sequential ds)
    : conv1(conv1x1(in_planes, planes))
    , conv2(conv3x3(planes,    planes,              stride))
    , conv3(conv1x1(planes,    planes * expansion))
    , bn1(torch::nn::BatchNorm2d(planes))
    , bn2(torch::nn::BatchNorm2d(planes))
    , bn3(torch::nn::BatchNorm2d(planes * expansion))
    , downsample(std::move(ds))
{
    register_module("conv1", conv1);
    register_module("bn1",   bn1);
    register_module("conv2", conv2);
    register_module("bn2",   bn2);
    register_module("conv3", conv3);
    register_module("bn3",   bn3);
    if (!downsample->is_empty())
        register_module("downsample", downsample);
}

torch::Tensor BottleneckImpl::forward(torch::Tensor x) {
    torch::Tensor identity = x;

    auto out = torch::relu(bn1->forward(conv1->forward(x)));
    out = torch::relu(bn2->forward(conv2->forward(out)));
    out = bn3->forward(conv3->forward(out));

    if (!downsample->is_empty())
        identity = downsample->forward(x);

    out += identity;
    return torch::relu(out);
}

// ═══════════════════════════════════════════════════════════════════════════
// ResNet<Block>
// Note: Block is a TORCH_MODULE holder; Block::ContainedType is the Impl.
// We use Block::ContainedType::expansion to reach the static constexpr.
// We push Block::ContainedType into Sequential (the holder auto-wraps it).
// ═══════════════════════════════════════════════════════════════════════════

template <typename Block>
torch::nn::Sequential ResNetImpl<Block>::_make_layer(int64_t planes,
                                                      int64_t num_blocks,
                                                      int64_t stride) {
    using Impl = typename Block::ContainedType;
    constexpr int64_t exp = Impl::expansion;

    torch::nn::Sequential ds{};
    if (stride != 1 || in_planes_ != planes * exp) {
        ds = torch::nn::Sequential(
            conv1x1(in_planes_, planes * exp, stride),
            torch::nn::BatchNorm2d(planes * exp));
    }

    torch::nn::Sequential seq;
    seq->push_back(Impl(in_planes_, planes, stride, ds));
    in_planes_ = planes * exp;

    for (int64_t i = 1; i < num_blocks; ++i)
        seq->push_back(Impl(in_planes_, planes));

    return seq;
}

template <typename Block>
ResNetImpl<Block>::ResNetImpl(const std::vector<int64_t>& layers,
                               int64_t num_classes) {
    using Impl = typename Block::ContainedType;
    constexpr int64_t exp = Impl::expansion;

    assert(layers.size() == 4);

    // conv1: 7×7, 64, stride 2, BN, ReLU  (paper Table 1 / Section 3.3)
    conv1 = register_module("conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7)
            .stride(2).padding(3).bias(false)));
    bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
    // maxpool: 3×3, stride 2
    maxpool = register_module("maxpool",
        torch::nn::MaxPool2d(
            torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

    layer1 = register_module("layer1", _make_layer(64,  layers[0], 1));
    layer2 = register_module("layer2", _make_layer(128, layers[1], 2));
    layer3 = register_module("layer3", _make_layer(256, layers[2], 2));
    layer4 = register_module("layer4", _make_layer(512, layers[3], 2));

    // global average pool → fc
    avgpool = register_module("avgpool",
        torch::nn::AdaptiveAvgPool2d(
            torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
    fc = register_module("fc",
        torch::nn::Linear(512 * exp, num_classes));

    // Kaiming normal init (paper Section 3.4: "initialize weights as in [12]")
    for (auto& m : modules(false)) {
        if (auto* c = m->as<torch::nn::Conv2dImpl>()) {
            torch::nn::init::kaiming_normal_(
                c->weight, 0.0, torch::kFanOut, torch::kReLU);
        } else if (auto* b = m->as<torch::nn::BatchNorm2dImpl>()) {
            torch::nn::init::constant_(b->weight, 1.0);
            torch::nn::init::constant_(b->bias,   0.0);
        }
    }
}

template <typename Block>
torch::Tensor ResNetImpl<Block>::forward(torch::Tensor x) {
    x = torch::relu(bn1->forward(conv1->forward(x)));
    x = maxpool->forward(x);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = avgpool->forward(x);
    x = torch::flatten(x, 1);
    x = fc->forward(x);
    return x;
}

// Explicit instantiations
template struct ResNetImpl<BasicBlock>;
template struct ResNetImpl<Bottleneck>;

// ═══════════════════════════════════════════════════════════════════════════
// Factory functions (paper Table 1)
// ═══════════════════════════════════════════════════════════════════════════

std::shared_ptr<ResNetBasicImpl> make_resnet18(int64_t num_classes) {
    return std::make_shared<ResNetBasicImpl>(
        std::vector<int64_t>{2, 2, 2, 2}, num_classes);
}
std::shared_ptr<ResNetBasicImpl> make_resnet34(int64_t num_classes) {
    return std::make_shared<ResNetBasicImpl>(
        std::vector<int64_t>{3, 4, 6, 3}, num_classes);
}
std::shared_ptr<ResNetBottleneckImpl> make_resnet50(int64_t num_classes) {
    return std::make_shared<ResNetBottleneckImpl>(
        std::vector<int64_t>{3, 4, 6, 3}, num_classes);
}
std::shared_ptr<ResNetBottleneckImpl> make_resnet101(int64_t num_classes) {
    return std::make_shared<ResNetBottleneckImpl>(
        std::vector<int64_t>{3, 4, 23, 3}, num_classes);
}
std::shared_ptr<ResNetBottleneckImpl> make_resnet152(int64_t num_classes) {
    return std::make_shared<ResNetBottleneckImpl>(
        std::vector<int64_t>{3, 8, 36, 3}, num_classes);
}

// ═══════════════════════════════════════════════════════════════════════════
// Training helpers
// ═══════════════════════════════════════════════════════════════════════════

static std::pair<int64_t, int64_t> topk_correct(const torch::Tensor& output,
                                                  const torch::Tensor& target,
                                                  int64_t k) {
    auto pred    = std::get<1>(output.topk(k, /*dim=*/1, /*largest=*/true,
                                            /*sorted=*/true));   // [N, k]
    auto correct = pred.eq(target.view({-1, 1}).expand_as(pred));
    return {correct.any(1).sum().item<int64_t>(), target.size(0)};
}

float resnet_train_epoch(
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
        auto loss   = torch::nn::functional::cross_entropy(output, y);
        loss.backward();
        optimizer.step();

        total_loss += loss.item<double>();
        ++n;
    }
    return n > 0 ? static_cast<float>(total_loss / n) : 0.f;
}

std::pair<float, float> resnet_evaluate(
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

        auto [c1, n] = topk_correct(output, y, 1);
        top1  += c1;
        total += n;
        auto [c5, _n2] = topk_correct(output, y,
                                       std::min<int64_t>(5, output.size(1)));
        top5 += c5;
    }

    if (total == 0) return {0.f, 0.f};
    return {static_cast<float>(top1) / total,
            static_cast<float>(top5) / total};
}

void resnet_train(
    torch::nn::AnyModule&   model,
    const ResNetTrainConfig& cfg,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& train_batches,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& val_batches,
    const std::string& save_path)
{
    model.ptr()->to(cfg.device);

    torch::optim::SGD optimizer(
        model.ptr()->parameters(),
        torch::optim::SGDOptions(cfg.lr)
            .momentum(cfg.momentum)
            .weight_decay(cfg.weight_decay));

    float best_top1 = 0.f;

    for (int64_t epoch = 1; epoch <= cfg.max_epochs; ++epoch) {
        for (int64_t ms : cfg.lr_milestones) {
            if (epoch == ms) {
                for (auto& pg : optimizer.param_groups()) {
                    auto& opts = static_cast<torch::optim::SGDOptions&>(pg.options());
                    opts.lr(opts.lr() * cfg.lr_gamma);
                }
                double cur_lr = static_cast<torch::optim::SGDOptions&>(
                    optimizer.param_groups()[0].options()).lr();
                std::cout << "[ResNet] epoch " << epoch << " LR → " << cur_lr << "\n";
            }
        }

        float loss        = resnet_train_epoch(model, optimizer, cfg.device, train_batches);
        auto [top1, top5] = resnet_evaluate(model, cfg.device, val_batches);

        std::cout << "[ResNet] " << epoch << "/" << cfg.max_epochs
                  << "  loss=" << loss
                  << "  top1=" << top1 * 100.f << "%"
                  << "  top5=" << top5 * 100.f << "%\n";

        if (top1 > best_top1) {
            best_top1 = top1;
            // Save all named parameters to file
            torch::serialize::OutputArchive archive;
            model.ptr()->save(archive);
            archive.save_to(save_path);
            std::cout << "[ResNet] checkpoint saved → " << save_path << "\n";
        }
    }
}

} // namespace vision
} // namespace models
} // namespace dm
