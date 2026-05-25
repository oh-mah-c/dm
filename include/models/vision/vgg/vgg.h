#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// VGGNet — Very Deep Convolutional Networks for Large-Scale Image Recognition
// Simonyan & Zisserman, ICLR 2015  (arXiv:1409.1556v6)
//
// Implements all 5 configurations from Table 1 of the paper:
//   VGG-A  (11 weight layers)
//   VGG-B  (13 weight layers)
//   VGG-C  (16 weight layers, contains conv1×1 layers)
//   VGG-D  (16 weight layers) — commonly called "VGG-16"
//   VGG-E  (19 weight layers) — commonly called "VGG-19"
//
// Architecture (Section 2.1):
//   - Input: 224×224 RGB image, per-pixel mean subtracted
//   - Conv layers: 3×3 filters (stride 1, padding 1), ReLU, no LRN
//   - Config C only: some 1×1 conv layers
//   - MaxPool: 2×2, stride 2 — after each block (5 pools total)
//   - FC: 4096 → 4096 → num_classes
//   - Dropout(0.5) before each FC (Section 3.1)
//   - Softmax output
//
// dm internal modules include this header to use torch:: types directly.
// Public dm users only see the pure-C dm_c_api.
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <string>
#include <vector>

namespace dm {
namespace models {
namespace vision {

// ─────────────────────────────────────────────────────────────────────────────
// VGG configurations (Table 1)
// Each inner vector describes one convolutional block.
// 0 = MaxPool marker; positive int = number of output channels for a conv3×3;
// negative int = number of output channels for a conv1×1 (config C only).
// ─────────────────────────────────────────────────────────────────────────────
enum class VGGConfig { A, B, C, D, E };

// ─────────────────────────────────────────────────────────────────────────────
// VGGNet module
//
// Paper Table 1 generic layout:
//   Block1: [64, 64*]  → maxpool
//   Block2: [128, 128*] → maxpool
//   Block3: [256, 256, 256*, 256*] → maxpool
//   Block4: [512, 512, 512*, 512*] → maxpool
//   Block5: [512, 512, 512*, 512*] → maxpool
//   FC-4096 → Dropout(0.5)
//   FC-4096 → Dropout(0.5)
//   FC-num_classes → Softmax
// (* = present only in deeper configs)
// ─────────────────────────────────────────────────────────────────────────────
struct VGGImpl : torch::nn::Module {
    torch::nn::Sequential features{nullptr};    // conv blocks + maxpool
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr}; // 7×7 adaptive pool (dense eval)
    torch::nn::Sequential classifier{nullptr};  // FC layers

    // cfg      — which of the 5 paper configurations to build
    // num_classes — 1000 for ImageNet (default); 10 for CIFAR-10 etc.
    // batch_norm  — not in original paper but commonly added; default false
    VGGImpl(VGGConfig cfg, int64_t num_classes = 1000, bool batch_norm = false);

    torch::Tensor forward(torch::Tensor x);

private:
    static torch::nn::Sequential make_features(VGGConfig cfg, bool batch_norm);
};
TORCH_MODULE(VGG);

// ─────────────────────────────────────────────────────────────────────────────
// Factory functions — paper Table 1 exact configurations
//
//  Config  Name    Weight layers  Channels per block
//  ──────────────────────────────────────────────────────────────────────
//  A       VGG-11  11             [64] [128] [256,256] [512,512] [512,512]
//  B       VGG-13  13             [64,64] [128,128] [256,256] [512,512] [512,512]
//  C       VGG-16  16             [64,64] [128,128] [256,256,256(1×1)] [512,512,512(1×1)] [512,512,512(1×1)]
//  D       VGG-16  16             [64,64] [128,128] [256,256,256] [512,512,512] [512,512,512]
//  E       VGG-19  19             [64,64] [128,128] [256,256,256,256] [512,512,512,512] [512,512,512,512]
// ─────────────────────────────────────────────────────────────────────────────
std::shared_ptr<VGGImpl> make_vgg_a (int64_t num_classes = 1000, bool batch_norm = false);
std::shared_ptr<VGGImpl> make_vgg_b (int64_t num_classes = 1000, bool batch_norm = false);
std::shared_ptr<VGGImpl> make_vgg_c (int64_t num_classes = 1000, bool batch_norm = false);
std::shared_ptr<VGGImpl> make_vgg16(int64_t num_classes = 1000, bool batch_norm = false); // config D
std::shared_ptr<VGGImpl> make_vgg19(int64_t num_classes = 1000, bool batch_norm = false); // config E

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration
// Defaults from paper Section 3.1:
//   SGD, lr=0.01, momentum=0.9, weight_decay=5e-4
//   Dropout(0.5) before first two FC layers
//   LR divided by 10 when val accuracy stops improving (3 times total)
//   Training stopped after 74 epochs (370K iterations at batch=256)
// ─────────────────────────────────────────────────────────────────────────────
struct VGGTrainConfig {
    double   lr           = 0.01;    // Paper: "initial learning rate 10^{-2}"
    double   momentum     = 0.9;     // Paper: "momentum to 0.9"
    double   weight_decay = 5e-4;    // Paper: "L2 penalty multiplier 5·10^{-4}"
    double   dropout      = 0.5;     // Paper: "dropout ratio 0.5"

    std::vector<int64_t> lr_milestones = {25, 50};  // divide LR by 10 at each
    double   lr_gamma     = 0.1;

    int64_t  batch_size   = 256;     // Paper: "batch size 256"
    int64_t  max_epochs   = 74;      // Paper: "370K iterations ≈ 74 epochs"

    torch::Device device  = torch::kCPU;
};

// ─────────────────────────────────────────────────────────────────────────────
// Training / evaluation helpers
// Batches: vector of {data [N,C,H,W] float, target [N] long}
// ─────────────────────────────────────────────────────────────────────────────

// Train one epoch. Returns mean cross-entropy loss.
float vgg_train_epoch(torch::nn::AnyModule&  model,
                      torch::optim::SGD&     optimizer,
                      torch::Device          device,
                      const std::vector<std::pair<torch::Tensor,
                                                  torch::Tensor>>& batches);

// Evaluate top-1 and top-5 accuracy. Returns {top1, top5} in [0,1].
std::pair<float, float> vgg_evaluate(
    torch::nn::AnyModule&  model,
    torch::Device          device,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batches);

// Full training loop with LR schedule + checkpoint saving.
void vgg_train(torch::nn::AnyModule&   model,
               const VGGTrainConfig&   cfg,
               const std::vector<std::pair<torch::Tensor, torch::Tensor>>& train_batches,
               const std::vector<std::pair<torch::Tensor, torch::Tensor>>& val_batches,
               const std::string&      save_path = "vgg_best.pt");

} // namespace vision
} // namespace models
} // namespace dm
