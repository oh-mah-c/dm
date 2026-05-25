#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// MobileNet v1 — Efficient CNNs for Mobile Vision Applications
// A.G. Howard, M. Zhu, B. Chen et al., arXiv:1704.04861v1, 2017
//
// ── Core building block: Depthwise Separable Convolution (Section 3.1) ───────
//
// Factorises a standard D_K×D_K×M×N conv into:
//   1. Depthwise conv:   D_K×D_K×1  per input channel (groups=M)
//                        Cost: D_K·D_K·M·D_F·D_F           (Eq. 4)
//   2. Pointwise conv:   1×1×M×N  (standard conv, groups=1)
//                        Cost: M·N·D_F·D_F
//   Total DS cost:       D_K·D_K·M·D_F·D_F + M·N·D_F·D_F  (Eq. 5)
//
// Reduction over standard conv: 1/N + 1/D_K²  (≈8–9× for D_K=3)
// Both layers followed by BN + ReLU  (Section 3.2, Figure 3)
//
// ── Network architecture (Section 3.2, Table 1) ──────────────────────────────
//
// Layer 1: Conv 3×3, stride 2, 3→32                 BN+ReLU
// Then 13 depthwise-separable blocks:
//   DW  3×3, stride s, 32→32 dw                     BN+ReLU
//   PW  1×1, stride 1, 32→64                         BN+ReLU
//   DW  3×3, stride 1, 64→64 dw                     BN+ReLU
//   PW  1×1, stride 1, 64→128                        BN+ReLU
//   ... (see Table 1 for complete spec)
//   5× (DW 512→512, PW 512→512)  at 14×14
//   DW s=2  512→512              14→7
//   PW 512→1024
//   DW s=1  1024→1024
//   PW 1024→1024
// GlobalAvgPool 7×7 → 1×1
// FC 1024→num_classes
// Softmax
//
// Total: 28 layers (counting DW and PW separately), 4.2M params, 569M MAdds
//
// ── Hyper-parameters (Section 3.3 / 3.4) ────────────────────────────────────
// Width multiplier α ∈ {1, 0.75, 0.5, 0.25}: scales all channel widths
// Resolution multiplier ρ: set via input size {224, 192, 160, 128}
//
// ── Training (Section 3.2) ───────────────────────────────────────────────────
// Optimizer: RMSprop (similar to Inception V3)
// Less regularisation and data augmentation than large models
// Little/no weight decay on depthwise filter weights
// Input: 224×224 (baseline)
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <string>
#include <vector>

namespace dm {
namespace models {
namespace vision {

// ─────────────────────────────────────────────────────────────────────────────
// DepthwiseSeparableBlock — DW conv + PW conv, each with BN+ReLU (Figure 3)
//
// Input:  [B, in_ch, H, W]
// Output: [B, out_ch, H/stride, W/stride]
// ─────────────────────────────────────────────────────────────────────────────
struct DepthwiseSeparableBlockImpl : torch::nn::Module {
    torch::nn::Conv2d      dw{nullptr};   // depthwise 3×3, groups=in_ch
    torch::nn::BatchNorm2d dw_bn{nullptr};
    torch::nn::Conv2d      pw{nullptr};   // pointwise 1×1
    torch::nn::BatchNorm2d pw_bn{nullptr};

    DepthwiseSeparableBlockImpl(int64_t in_ch, int64_t out_ch,
                                int64_t stride = 1);

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(DepthwiseSeparableBlock);

// ─────────────────────────────────────────────────────────────────────────────
// MobileNetImpl — full MobileNet v1 classification network (Table 1)
//
// Constructor arguments:
//   num_classes — output classes (default 1000 for ImageNet)
//   alpha       — width multiplier ∈ {1.0, 0.75, 0.5, 0.25} (Section 3.3)
// ─────────────────────────────────────────────────────────────────────────────
struct MobileNetImpl : torch::nn::Module {
    float   alpha;        // width multiplier
    int64_t num_classes;

    torch::nn::Sequential features{nullptr};   // conv + 13 DS blocks + pool
    torch::nn::Linear      classifier{nullptr}; // FC 1024α → num_classes

    MobileNetImpl(int64_t num_classes = 1000, float alpha = 1.0f);

    // Input: [B, 3, H, W]  (H=W=224 for baseline)
    // Output: [B, num_classes]
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MobileNet);

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factories (Table 6 / Table 8)
// ─────────────────────────────────────────────────────────────────────────────
MobileNet make_mobilenet_1_0(int64_t num_classes = 1000);    // α=1.0, 4.2M
MobileNet make_mobilenet_0_75(int64_t num_classes = 1000);   // α=0.75, 2.6M
MobileNet make_mobilenet_0_5(int64_t num_classes = 1000);    // α=0.5,  1.3M
MobileNet make_mobilenet_0_25(int64_t num_classes = 1000);   // α=0.25, 0.5M

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration (Section 3.2)
// ─────────────────────────────────────────────────────────────────────────────
struct MobileNetTrainConfig {
    double  lr            = 1e-2;        // RMSprop initial lr
    double  weight_decay  = 4e-5;        // small WD (no WD on DW filters)
    double  momentum      = 0.9;
    double  rms_alpha     = 0.9;         // RMSprop smoothing
    int64_t batch_size    = 256;
    int64_t max_epochs    = 100;
    torch::Device device  = torch::kCPU;
};

// ─────────────────────────────────────────────────────────────────────────────
// Training helpers
// ─────────────────────────────────────────────────────────────────────────────
float mobilenet_train_epoch(
    MobileNet& model,
    torch::optim::RMSprop& optimizer,
    torch::Device device,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batches);

float mobilenet_evaluate(
    MobileNet& model,
    torch::Device device,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& batches);

void mobilenet_train(
    MobileNet& model,
    const MobileNetTrainConfig& cfg,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& train_batches,
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& val_batches,
    const std::string& save_path = "mobilenet_best.pt");

} // namespace vision
} // namespace models
} // namespace dm
