// ─────────────────────────────────────────────────────────────────────────────
// MobileNet v1 implementation — Howard et al., arXiv:1704.04861v1
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/mobilenet/mobilenet.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>

namespace dm {
namespace models {
namespace vision {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: round channel count to nearest multiple of 8 (common practice)
// ─────────────────────────────────────────────────────────────────────────────
static int64_t ch(int64_t base, float alpha) {
    return std::max<int64_t>(8,
        (int64_t)(std::round((float)base * alpha / 8.0f) * 8));
}

// ─────────────────────────────────────────────────────────────────────────────
// DepthwiseSeparableBlockImpl
// ─────────────────────────────────────────────────────────────────────────────
DepthwiseSeparableBlockImpl::DepthwiseSeparableBlockImpl(int64_t in_ch,
                                                         int64_t out_ch,
                                                         int64_t stride) {
    // Depthwise 3×3 conv: groups = in_ch (one filter per input channel)
    dw = register_module("dw",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, in_ch, 3)
                              .stride(stride).padding(1)
                              .groups(in_ch).bias(false)));
    dw_bn = register_module("dw_bn", torch::nn::BatchNorm2d(in_ch));

    // Pointwise 1×1 conv
    pw = register_module("pw",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, 1)
                              .stride(1).padding(0).bias(false)));
    pw_bn = register_module("pw_bn", torch::nn::BatchNorm2d(out_ch));
}

torch::Tensor DepthwiseSeparableBlockImpl::forward(torch::Tensor x) {
    x = torch::relu(dw_bn->forward(dw->forward(x)));   // DW + BN + ReLU
    x = torch::relu(pw_bn->forward(pw->forward(x)));   // PW + BN + ReLU
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// MobileNetImpl — 28-layer body (Table 1)
//
// Table 1 specification (input 224×224):
// Conv/s2    3×3×3×32        224→112
// DW/s1      3×3×32dw        112→112
// PW/s1      1×1×32×64       112→112
// DW/s2      3×3×64dw        112→56
// PW/s1      1×1×64×128      56→56
// DW/s1      3×3×128dw       56→56
// PW/s1      1×1×128×128     56→56
// DW/s2      3×3×128dw       56→28
// PW/s1      1×1×128×256     28→28
// DW/s1      3×3×256dw       28→28
// PW/s1      1×1×256×256     28→28
// DW/s2      3×3×256dw       28→14
// PW/s1      1×1×256×512     14→14
// 5× DW/s1   3×3×512dw       14→14
//    PW/s1   1×1×512×512     14→14
// DW/s2      3×3×512dw       14→7
// PW/s1      1×1×512×1024    7→7
// DW/s1      3×3×1024dw      7→7
// PW/s1      1×1×1024×1024   7→7
// AvgPool/s1  Pool 7×7        7→1
// FC/s1      1024×1000
// Softmax
// ─────────────────────────────────────────────────────────────────────────────
MobileNetImpl::MobileNetImpl(int64_t num_classes_, float alpha_)
    : alpha(alpha_), num_classes(num_classes_) {

    // Channel widths scaled by α
    int64_t c32   = ch(32,   alpha);
    int64_t c64   = ch(64,   alpha);
    int64_t c128  = ch(128,  alpha);
    int64_t c256  = ch(256,  alpha);
    int64_t c512  = ch(512,  alpha);
    int64_t c1024 = ch(1024, alpha);

    torch::nn::Sequential feat;

    // Layer 1: standard Conv 3×3, stride 2
    feat->push_back(torch::nn::Conv2d(
        torch::nn::Conv2dOptions(3, c32, 3).stride(2).padding(1).bias(false)));
    feat->push_back(torch::nn::BatchNorm2d(c32));
    feat->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

    // DW+PW blocks following Table 1
    // s=1: 32→64
    feat->push_back(DepthwiseSeparableBlock(c32,  c64,  1));
    // s=2: 64→128
    feat->push_back(DepthwiseSeparableBlock(c64,  c128, 2));
    // s=1: 128→128
    feat->push_back(DepthwiseSeparableBlock(c128, c128, 1));
    // s=2: 128→256
    feat->push_back(DepthwiseSeparableBlock(c128, c256, 2));
    // s=1: 256→256
    feat->push_back(DepthwiseSeparableBlock(c256, c256, 1));
    // s=2: 256→512
    feat->push_back(DepthwiseSeparableBlock(c256, c512, 2));
    // 5× s=1: 512→512  (14×14 feature maps)
    for (int i = 0; i < 5; ++i)
        feat->push_back(DepthwiseSeparableBlock(c512, c512, 1));
    // s=2: 512→1024  (14→7)
    feat->push_back(DepthwiseSeparableBlock(c512,  c1024, 2));
    // s=1: 1024→1024
    feat->push_back(DepthwiseSeparableBlock(c1024, c1024, 1));

    // Global average pooling 7×7 → 1×1
    feat->push_back(torch::nn::AdaptiveAvgPool2d(
        torch::nn::AdaptiveAvgPool2dOptions({1, 1})));

    features   = register_module("features",   feat);
    classifier = register_module("classifier",
                     torch::nn::Linear(c1024, num_classes_));
}

torch::Tensor MobileNetImpl::forward(torch::Tensor x) {
    x = features->forward(x);       // [B, 1024α, 1, 1]
    x = x.flatten(1);               // [B, 1024α]
    x = classifier->forward(x);     // [B, num_classes]
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// Factories
// ─────────────────────────────────────────────────────────────────────────────
MobileNet make_mobilenet_1_0(int64_t num_classes)  { return MobileNet(num_classes, 1.00f); }
MobileNet make_mobilenet_0_75(int64_t num_classes) { return MobileNet(num_classes, 0.75f); }
MobileNet make_mobilenet_0_5(int64_t num_classes)  { return MobileNet(num_classes, 0.50f); }
MobileNet make_mobilenet_0_25(int64_t num_classes) { return MobileNet(num_classes, 0.25f); }

// ─────────────────────────────────────────────────────────────────────────────
// Training helpers
// ─────────────────────────────────────────────────────────────────────────────
float mobilenet_train_epoch(
    MobileNet& model,
    torch::optim::RMSprop& optimizer,
    torch::Device device,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batches) {

    model->train();
    double total = 0.0;
    int64_t steps = 0;
    for (auto& [imgs, labels] : batches) {
        auto x = imgs.to(device);
        auto y = labels.to(device);
        optimizer.zero_grad();
        auto logits = model->forward(x);
        auto loss   = torch::nn::functional::cross_entropy(logits, y);
        loss.backward();
        optimizer.step();
        total += loss.item<double>();
        ++steps;
    }
    return steps > 0 ? (float)(total / steps) : 0.f;
}

float mobilenet_evaluate(
    MobileNet& model,
    torch::Device device,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batches) {

    model->eval();
    torch::NoGradGuard ng;
    double total = 0.0;
    int64_t steps = 0;
    for (auto& [imgs, labels] : batches) {
        auto x = imgs.to(device);
        auto y = labels.to(device);
        auto logits = model->forward(x);
        auto loss   = torch::nn::functional::cross_entropy(logits, y);
        total += loss.item<double>();
        ++steps;
    }
    return steps > 0 ? (float)(total / steps) : 0.f;
}

void mobilenet_train(
    MobileNet& model,
    const MobileNetTrainConfig& cfg,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& train_batches,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& val_batches,
    const std::string& save_path) {

    model->to(cfg.device);

    // RMSprop — same optimiser as used in training (Section 3.2)
    torch::optim::RMSprop optimizer(
        model->parameters(),
        torch::optim::RMSpropOptions(cfg.lr)
            .alpha(cfg.rms_alpha)
            .momentum(cfg.momentum)
            .weight_decay(cfg.weight_decay));

    float best_val = std::numeric_limits<float>::max();

    for (int64_t ep = 0; ep < cfg.max_epochs; ++ep) {
        float train_loss = mobilenet_train_epoch(model, optimizer, cfg.device, train_batches);
        float val_loss   = mobilenet_evaluate(model, cfg.device, val_batches);

        std::cout << "[MobileNet] epoch " << ep + 1 << "/" << cfg.max_epochs
                  << "  train_loss=" << train_loss
                  << "  val_loss="   << val_loss << "\n";

        if (val_loss < best_val) {
            best_val = val_loss;
            torch::serialize::OutputArchive ar;
            model->save(ar);
            ar.save_to(save_path);
            std::cout << "[MobileNet] saved → " << save_path << "\n";
        }
    }
}

} // namespace vision
} // namespace models
} // namespace dm
