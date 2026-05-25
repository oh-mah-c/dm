#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
// Ze Liu et al., ICCV 2021
// https://arxiv.org/abs/2103.14030
//
// Architecture (Section 3, Figure 3):
//   - Patch partition: split H×W image into (H/4)×(W/4) non-overlapping
//     patches of size 4×4; each patch is a 48-dim token (4×4×3 raw RGB)
//   - Linear embedding: project 48-dim tokens to arbitrary dimension C
//   - Stage 1: 2 Swin Transformer Blocks, tokens kept at H/4 × W/4
//   - Patch merging: merge 2×2 neighbours → H/8 × W/8, dim 2C
//   - Stage 2: 2 blocks
//   - Patch merging → H/16 × W/16, dim 4C
//   - Stage 3: 6 blocks (Swin-T/S) or 18 (Swin-B/L)
//   - Patch merging → H/32 × W/32, dim 8C
//   - Stage 4: 2 blocks
//   - Global average pool + FC head for classification
//
// Each Swin Transformer Block (Eq. 3):
//   ẑ^l = W-MSA(LN(z^{l-1})) + z^{l-1}   (regular window, even layers)
//   z^l  = MLP(LN(ẑ^l))      + ẑ^l
//   ẑ^{l+1} = SW-MSA(LN(z^l)) + z^l       (shifted window, odd layers)
//   z^{l+1} = MLP(LN(ẑ^{l+1})) + ẑ^{l+1}
//
// W-MSA / SW-MSA: window-based multi-head self-attention with relative
//   position bias B ∈ ℝ^{M²×M²} (Eq. 4). M=7 by default.
//   Attention(Q,K,V) = SoftMax(QK^T/sqrt(d) + B)V
//
// Efficient batched shifted-window: cyclic shift + attention masking (Fig. 4)
//
// Variants (Section 3.3):
//   Swin-T: C=96,  depths={2,2,6,2},  heads={3,6,12,24}
//   Swin-S: C=96,  depths={2,2,18,2}, heads={3,6,12,24}
//   Swin-B: C=128, depths={2,2,18,2}, heads={4,8,16,32}
//   Swin-L: C=192, depths={2,2,18,2}, heads={6,12,24,48}
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <string>
#include <vector>

namespace dm {
namespace models {
namespace vision {

// ─────────────────────────────────────────────────────────────────────────────
// MLP block (2-layer, GELU, ratio=4 per Section 3.1)
// ─────────────────────────────────────────────────────────────────────────────
struct SwinMlpImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::Dropout drop{nullptr};

    SwinMlpImpl(int64_t in_features, int64_t hidden_features, double dropout = 0.0);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SwinMlp);

// ─────────────────────────────────────────────────────────────────────────────
// Window-based multi-head self-attention with relative position bias
// (Section 3.2, Eq. 4)
// ─────────────────────────────────────────────────────────────────────────────
struct WindowAttentionImpl : torch::nn::Module {
    int64_t dim;
    int64_t window_size;   // M — default 7
    int64_t num_heads;
    double  scale;

    torch::nn::Linear qkv{nullptr};
    torch::nn::Linear proj{nullptr};
    torch::nn::Dropout attn_drop{nullptr};
    torch::nn::Dropout proj_drop{nullptr};

    // Relative position bias table: (2M-1)*(2M-1) × num_heads
    torch::Tensor relative_position_bias_table;
    // Precomputed index buffer: M²×M²
    torch::Tensor relative_position_index;

    WindowAttentionImpl(int64_t dim, int64_t window_size, int64_t num_heads,
                        double attn_drop = 0.0, double proj_drop = 0.0);

    torch::Tensor forward(torch::Tensor x,
                          torch::optional<torch::Tensor> mask = torch::nullopt);

private:
    void _build_relative_position_index();
};
TORCH_MODULE(WindowAttention);

// ─────────────────────────────────────────────────────────────────────────────
// Single Swin Transformer Block
// (Figure 3b; implements one of the W-MSA or SW-MSA blocks)
// ─────────────────────────────────────────────────────────────────────────────
struct SwinBlockImpl : torch::nn::Module {
    int64_t dim;
    int64_t input_resolution_h;
    int64_t input_resolution_w;
    int64_t window_size;
    int64_t shift_size;   // 0 for W-MSA, window_size/2 for SW-MSA

    torch::nn::LayerNorm norm1{nullptr};
    WindowAttention attn{nullptr};
    torch::nn::LayerNorm norm2{nullptr};
    SwinMlp mlp{nullptr};

    // Attention mask for shifted windows (nullptr if shift_size==0)
    torch::Tensor attn_mask;

    SwinBlockImpl(int64_t dim, int64_t input_resolution_h, int64_t input_resolution_w,
                  int64_t num_heads, int64_t window_size = 7, int64_t shift_size = 0,
                  double mlp_ratio = 4.0, double dropout = 0.0, double attn_drop = 0.0);

    torch::Tensor forward(torch::Tensor x);

private:
    void _compute_attn_mask();
};
TORCH_MODULE(SwinBlock);

// ─────────────────────────────────────────────────────────────────────────────
// Patch Merging layer (Section 3.1)
// Concatenates 2×2 neighbouring patches, applies LayerNorm, then Linear 4C→2C
// ─────────────────────────────────────────────────────────────────────────────
struct PatchMergingImpl : torch::nn::Module {
    int64_t input_resolution_h;
    int64_t input_resolution_w;
    int64_t dim;

    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Linear reduction{nullptr};

    PatchMergingImpl(int64_t input_resolution_h, int64_t input_resolution_w, int64_t dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PatchMerging);

// ─────────────────────────────────────────────────────────────────────────────
// Stage: sequence of SwinBlocks followed by optional PatchMerging
// ─────────────────────────────────────────────────────────────────────────────
struct SwinStageImpl : torch::nn::Module {
    torch::nn::Sequential blocks{nullptr};
    PatchMerging downsample{nullptr};  // nullptr for last stage

    SwinStageImpl(int64_t dim, int64_t input_resolution_h, int64_t input_resolution_w,
                  int64_t depth, int64_t num_heads, int64_t window_size = 7,
                  double mlp_ratio = 4.0, double dropout = 0.0, double attn_drop = 0.0,
                  bool use_downsample = true);

    // Returns (output_tensor, out_H, out_W)
    std::tuple<torch::Tensor, int64_t, int64_t> forward(torch::Tensor x);

    // Access resolution of first block (used internally in forward)
    int64_t input_resolution_h() const;
    int64_t input_resolution_w() const;
};
TORCH_MODULE(SwinStage);

// ─────────────────────────────────────────────────────────────────────────────
// Patch Partition + Linear Embedding (Section 3.1 / Figure 3)
// Input: [N, 3, H, W] — H,W must be divisible by patch_size (4)
// Output: [N, H/P × W/P, embed_dim]
// ─────────────────────────────────────────────────────────────────────────────
struct PatchEmbedImpl : torch::nn::Module {
    int64_t patch_size;   // 4 per paper
    int64_t embed_dim;    // C

    // Use Conv2d(3, embed_dim, kernel=patch_size, stride=patch_size) as patch partition
    torch::nn::Conv2d proj{nullptr};
    torch::nn::LayerNorm norm{nullptr};

    PatchEmbedImpl(int64_t img_size = 224, int64_t patch_size = 4,
                   int64_t in_chans = 3, int64_t embed_dim = 96);

    int64_t num_patches_h;
    int64_t num_patches_w;

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PatchEmbed);

// ─────────────────────────────────────────────────────────────────────────────
// Full Swin Transformer (image classification head)
// ─────────────────────────────────────────────────────────────────────────────
struct SwinTransformerImpl : torch::nn::Module {
    PatchEmbed patch_embed{nullptr};
    torch::nn::Dropout pos_drop{nullptr};

    // 4 stages (Section 3.1)
    torch::nn::ModuleList stages{nullptr};

    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Linear head{nullptr};

    // Store resolutions for each stage output (needed at forward)
    std::vector<std::pair<int64_t,int64_t>> stage_resolutions;

    SwinTransformerImpl(
        int64_t img_size     = 224,
        int64_t patch_size   = 4,
        int64_t in_chans     = 3,
        int64_t num_classes  = 1000,
        int64_t embed_dim    = 96,      // C
        std::vector<int64_t> depths     = {2, 2, 6, 2},   // Swin-T
        std::vector<int64_t> num_heads  = {3, 6, 12, 24},
        int64_t window_size  = 7,
        double  mlp_ratio    = 4.0,
        double  dropout      = 0.0,
        double  attn_drop    = 0.0
    );

    torch::Tensor forward(torch::Tensor x);

private:
    int64_t _num_features;  // 8C (output dim of last stage before head)
};
TORCH_MODULE(SwinTransformer);

// ─────────────────────────────────────────────────────────────────────────────
// Factory functions — paper Section 3.3
// ─────────────────────────────────────────────────────────────────────────────
// Swin-T: C=96, depths={2,2,6,2},   heads={3,6,12,24},  ~29M params
std::shared_ptr<SwinTransformerImpl> make_swin_t(int64_t num_classes = 1000,
                                                  int64_t img_size   = 224);
// Swin-S: C=96, depths={2,2,18,2},  heads={3,6,12,24},  ~50M params
std::shared_ptr<SwinTransformerImpl> make_swin_s(int64_t num_classes = 1000,
                                                  int64_t img_size   = 224);
// Swin-B: C=128, depths={2,2,18,2}, heads={4,8,16,32},  ~88M params
std::shared_ptr<SwinTransformerImpl> make_swin_b(int64_t num_classes = 1000,
                                                  int64_t img_size   = 224);
// Swin-L: C=192, depths={2,2,18,2}, heads={6,12,24,48}, ~197M params
std::shared_ptr<SwinTransformerImpl> make_swin_l(int64_t num_classes = 1000,
                                                  int64_t img_size   = 224);

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration
// Defaults: AdamW, lr=0.001, cosine LR, 20-epoch warm-up, 300 epochs
// (Paper Section 4.1: "regular ImageNet-1K training")
// ─────────────────────────────────────────────────────────────────────────────
struct SwinTrainConfig {
    double   lr           = 1e-3;       // initial LR
    double   weight_decay = 0.05;       // AdamW weight decay
    double   min_lr       = 1e-5;       // cosine LR lower bound

    int64_t  warmup_epochs = 20;
    int64_t  max_epochs    = 300;
    int64_t  batch_size    = 1024;

    torch::Device device  = torch::kCPU;
};

// ─────────────────────────────────────────────────────────────────────────────
// Training / evaluation helpers
// ─────────────────────────────────────────────────────────────────────────────
float swin_train_epoch(torch::nn::AnyModule& model,
                       torch::optim::AdamW&  optimizer,
                       torch::Device         device,
                       const std::vector<std::pair<torch::Tensor,
                                                   torch::Tensor>>& batches);

std::pair<float,float> swin_evaluate(
    torch::nn::AnyModule& model,
    torch::Device         device,
    const std::vector<std::pair<torch::Tensor,torch::Tensor>>& batches);

void swin_train(torch::nn::AnyModule&   model,
                const SwinTrainConfig&  cfg,
                const std::vector<std::pair<torch::Tensor,torch::Tensor>>& train_batches,
                const std::vector<std::pair<torch::Tensor,torch::Tensor>>& val_batches,
                const std::string&      save_path = "swin_best.pt");

} // namespace vision
} // namespace models
} // namespace dm
