#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// ResNet — Deep Residual Learning for Image Recognition
// He et al., CVPR 2016  (https://arxiv.org/abs/1512.03385)
//
// Implements all five variants from Table 1 of the paper:
//   ResNet-18  / 34   →  BasicBlock   (2-layer: 3×3, 3×3)
//   ResNet-50  / 101 / 152  →  Bottleneck  (3-layer: 1×1, 3×3, 1×1)
//
// dm internal modules include this header to use torch:: types directly.
// Public dm users only see the pure-C dm_c_api; torch never leaks outward.
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <vector>
#include <string>

namespace dm {
namespace models {
namespace vision {

// ─────────────────────────────────────────────────────────────────────────────
// BasicBlock  (ResNet-18, ResNet-34)
//
// Paper Section 3.3 / Fig. 2 / Fig. 5 (left):
//   y = F(x) + x
//   F = BN(conv3×3(BN(conv3×3(x))))   (biases omitted per paper)
//   Second ReLU applied after addition.
//   Shortcut is identity when dims match; 1×1 conv projection otherwise.
// ─────────────────────────────────────────────────────────────────────────────
struct BasicBlockImpl : torch::nn::Module {
    static constexpr int64_t expansion = 1;

    torch::nn::Conv2d      conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr},   bn2{nullptr};
    torch::nn::Sequential  downsample{};   // empty → identity shortcut

    BasicBlockImpl(int64_t in_planes,
                   int64_t planes,
                   int64_t stride = 1,
                   torch::nn::Sequential ds = {});

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(BasicBlock);

// ─────────────────────────────────────────────────────────────────────────────
// Bottleneck  (ResNet-50, ResNet-101, ResNet-152)
//
// Paper Section 3.3 / Fig. 5 (right):
//   F = BN(conv1×1) → ReLU → BN(conv3×3) → ReLU → BN(conv1×1)
//   1×1 layers reduce then restore dimensions; 3×3 is the bottleneck.
//   expansion = 4: output channels = planes * 4
// ─────────────────────────────────────────────────────────────────────────────
struct BottleneckImpl : torch::nn::Module {
    static constexpr int64_t expansion = 4;

    torch::nn::Conv2d      conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr},   bn2{nullptr},   bn3{nullptr};
    torch::nn::Sequential  downsample{};

    BottleneckImpl(int64_t in_planes,
                   int64_t planes,
                   int64_t stride = 1,
                   torch::nn::Sequential ds = {});

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Bottleneck);

// ─────────────────────────────────────────────────────────────────────────────
// ResNet<Block>
//
// Paper Table 1 full architecture:
//   conv1:    7×7 conv, 64 filters, stride 2, BN, ReLU    → 112×112
//   pool:     3×3 maxpool, stride 2                        →  56×56
//   layer1-4: stacked Block groups (see make_resnetXX for counts)
//   avgpool:  global average pool                          →  1×1
//   fc:       fully connected → num_classes
//
// Paper Section 3.4:
//   BN after every conv, before activation.
//   Weight init: paper [12] (Kaiming normal).
//   No dropout used.
// ─────────────────────────────────────────────────────────────────────────────
template <typename Block>
struct ResNetImpl : torch::nn::Module {
    torch::nn::Conv2d            conv1{nullptr};
    torch::nn::BatchNorm2d       bn1{nullptr};
    torch::nn::MaxPool2d         maxpool{nullptr};
    torch::nn::Sequential        layer1, layer2, layer3, layer4;
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear            fc{nullptr};

    // layers[i] = number of blocks in conv(i+2)_x
    ResNetImpl(const std::vector<int64_t>& layers, int64_t num_classes = 1000);

    torch::Tensor forward(torch::Tensor x);

private:
    int64_t in_planes_{64};
    torch::nn::Sequential _make_layer(int64_t planes, int64_t num_blocks, int64_t stride);
};

extern template struct ResNetImpl<BasicBlock>;
extern template struct ResNetImpl<Bottleneck>;

// Module holder aliases (LibTorch convention)
using ResNetBasicImpl      = ResNetImpl<BasicBlock>;
using ResNetBottleneckImpl = ResNetImpl<Bottleneck>;
TORCH_MODULE(ResNetBasic);
TORCH_MODULE(ResNetBottleneck);

// ─────────────────────────────────────────────────────────────────────────────
// Factory functions — paper Table 1 exact configs
//
//  Variant   Block        layer counts
//  ──────────────────────────────────────
//  ResNet-18  BasicBlock   [2, 2,  2,  2]   1.8 B FLOPs
//  ResNet-34  BasicBlock   [3, 4,  6,  3]   3.6 B FLOPs
//  ResNet-50  Bottleneck   [3, 4,  6,  3]   3.8 B FLOPs
//  ResNet-101 Bottleneck   [3, 4, 23,  3]   7.6 B FLOPs
//  ResNet-152 Bottleneck   [3, 8, 36,  3]  11.3 B FLOPs
// ─────────────────────────────────────────────────────────────────────────────
std::shared_ptr<ResNetBasicImpl>      make_resnet18 (int64_t num_classes = 1000);
std::shared_ptr<ResNetBasicImpl>      make_resnet34 (int64_t num_classes = 1000);
std::shared_ptr<ResNetBottleneckImpl> make_resnet50 (int64_t num_classes = 1000);
std::shared_ptr<ResNetBottleneckImpl> make_resnet101(int64_t num_classes = 1000);
std::shared_ptr<ResNetBottleneckImpl> make_resnet152(int64_t num_classes = 1000);

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration
// Defaults taken directly from paper Section 3.4 (ImageNet).
// ─────────────────────────────────────────────────────────────────────────────
struct ResNetTrainConfig {
    // Optimiser — paper: SGD, lr=0.1, momentum=0.9, weight_decay=1e-4
    double   lr           = 0.1;
    double   momentum     = 0.9;
    double   weight_decay = 1e-4;

    // Schedule — paper: "divided by 10 when error plateaus"
    // Milestones in epochs; halve at each.
    std::vector<int64_t> lr_milestones = {30, 60};  // ImageNet
    double   lr_gamma     = 0.1;

    // Data — paper: batch 256 on 8 GPUs; use what fits locally
    int64_t  batch_size   = 256;

    // Total epochs — paper: "up to 60×10^4 iterations" ≈ 90 epochs @1.28M imgs
    int64_t  max_epochs   = 90;

    torch::Device device  = torch::kCPU;
};

// ─────────────────────────────────────────────────────────────────────────────
// Training and evaluation entry points.
//
// These work with any dataset that produces batches of:
//   data   : FloatTensor [N, C, H, W]
//   target : LongTensor  [N]
//
// Call site is responsible for constructing the DataLoader.
// See src/models/vision/resnet/train.cpp for a complete example.
// ─────────────────────────────────────────────────────────────────────────────

// Train one full epoch.
// Returns average cross-entropy loss over all batches.
// `model` is a torch::nn::AnyModule wrapping a ResNet (see make_resnetXX).
float resnet_train_epoch(torch::nn::AnyModule& model,
                         torch::optim::SGD&    optimizer,
                         torch::Device         device,
                         const std::vector<std::pair<torch::Tensor,
                                                     torch::Tensor>>& batches);

// Evaluate top-1 and top-5 accuracy.
// Returns {top1, top5} as fractions in [0, 1].
std::pair<float, float> resnet_evaluate(
    torch::nn::AnyModule& model,
    torch::Device         device,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batches);

// Full training loop with LR schedule.
// Saves checkpoint to `save_path` after each epoch improvement.
void resnet_train(torch::nn::AnyModule&   model,
                  const ResNetTrainConfig& cfg,
                  const std::vector<std::pair<torch::Tensor, torch::Tensor>>& train_batches,
                  const std::vector<std::pair<torch::Tensor, torch::Tensor>>& val_batches,
                  const std::string&      save_path = "resnet_best.pt");

} // namespace vision
} // namespace models
} // namespace dm
