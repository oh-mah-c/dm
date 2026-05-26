#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// I-JEPA — Image-based Joint-Embedding Predictive Architecture
// Assran et al., arXiv:2301.08243v3, CVPR 2023
// https://arxiv.org/abs/2301.08243
//
// Architecture (Figure 3, Appendix A):
//   context-encoder  f_θ   : standard ViT (no [CLS], processes visible patches)
//   target-encoder   f_θ̄   : identical ViT, weights = EMA of context-encoder
//   predictor        g_φ   : narrow ViT (embed_dim=384 for ViT-{L,H,H14})
//
// Forward pass per iteration:
//   1. Sample M=4 target blocks (scale 0.15–0.20, aspect 0.75–1.5) from image.
//   2. Sample 1 context block (scale 0.85–1.0, unit aspect); remove overlap.
//   3. f_θ encodes context patches → s_x  (patch tokens at context positions)
//   4. f_θ̄  encodes full image   → s_y  (patch tokens at ALL positions)
//   5. For each target block i:
//        predictor g_φ(s_x, mask_tokens_i) → ŝ_y(i)
//   6. Loss = (1/M) Σ_i Σ_{j∈B_i} ||ŝ_y_j − s_y_j||²   (L2 in repr. space)
//   7. θ updated via AdamW; θ̄ updated via EMA: θ̄ ← τ·θ̄ + (1-τ)·θ
//
// Supported model sizes (Table 1 / Appendix A):
//   vit_b16  : embed=768,  depth=12, heads=12, patch=16, pred_depth=6
//   vit_l16  : embed=1024, depth=24, heads=16, patch=16, pred_depth=12
//   vit_h14  : embed=1280, depth=32, heads=16, patch=14, pred_depth=12
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <vector>
#include <string>
#include <utility>

namespace dm {
namespace models {
namespace vision {

// ─────────────────────────────────────────────────────────────────────────────
// ViT building blocks
// ─────────────────────────────────────────────────────────────────────────────

// Multi-head self-attention (Vaswani et al. 2017).
struct AttentionImpl : torch::nn::Module {
    int64_t num_heads, head_dim;
    double  scale;
    torch::nn::Linear qkv{nullptr}, proj{nullptr};

    AttentionImpl(int64_t dim, int64_t num_heads);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Attention);

// FFN: LayerNorm → MLP with GELU (ratio 4:1).
struct MLPImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    MLPImpl(int64_t dim, int64_t hidden_dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MLP);

// Standard ViT block: pre-LN, attention + MLP with residual.
struct ViTBlockImpl : torch::nn::Module {
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    Attention attn{nullptr};
    MLP       mlp{nullptr};

    ViTBlockImpl(int64_t dim, int64_t num_heads, int64_t mlp_hidden);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(ViTBlock);

// ─────────────────────────────────────────────────────────────────────────────
// ViTEncoder
//
// Processes a subset of patch positions (the context block or the full image).
// No [CLS] token (per Appendix A: "I-JEPA is pretrained without a [CLS] token").
// Positional embeddings are learned sinusoidal-style (via nn::Embedding indexed
// by patch position).
// ─────────────────────────────────────────────────────────────────────────────
struct ViTEncoderImpl : torch::nn::Module {
    int64_t           embed_dim, patch_size, num_patches;
    torch::nn::Linear patch_embed{nullptr};   // conv-equivalent linear projection
    torch::nn::Embedding pos_embed{nullptr};  // learnable positional embeddings
    torch::nn::ModuleList blocks;
    torch::nn::LayerNorm norm{nullptr};

    // img_size: full image side length (assumed square).
    ViTEncoderImpl(int64_t img_size, int64_t patch_size,
                   int64_t embed_dim, int64_t depth,
                   int64_t num_heads);

    // Forward over a subset of patches selected by `indices` (shape [B, L]).
    // If indices is empty (undefined), processes all patches.
    torch::Tensor forward(torch::Tensor patches,            // [B, N, C_in]
                          torch::Tensor indices = {});      // [B, L] patch positions
};
TORCH_MODULE(ViTEncoder);

// ─────────────────────────────────────────────────────────────────────────────
// IJEPAPredictor
//
// Narrow ViT (embed_dim=384 for L/H, 12 for B) that takes:
//   - context encoder output s_x at context positions (projected to pred_dim)
//   - learnable mask tokens at target positions  (paper Fig 3)
// and produces predictions ŝ_y at target positions.
// ─────────────────────────────────────────────────────────────────────────────
struct IJEPAPredictorImpl : torch::nn::Module {
    int64_t  pred_dim, num_patches;
    torch::nn::Linear    input_proj{nullptr};   // encoder_dim → pred_dim
    torch::Tensor        mask_token;            // learnable [1,1,pred_dim]
    torch::nn::Embedding pos_embed{nullptr};    // same grid as encoder
    torch::nn::ModuleList blocks;
    torch::nn::LayerNorm norm{nullptr};
    torch::nn::Linear    output_proj{nullptr};  // pred_dim → encoder_dim

    IJEPAPredictorImpl(int64_t encoder_dim, int64_t pred_dim,
                       int64_t num_patches, int64_t depth, int64_t num_heads);

    // s_x: [B, L_ctx, encoder_dim]  context encoder output
    // ctx_idx: [B, L_ctx]           patch indices of the context block
    // tgt_idx: [B, L_tgt]           patch indices of one target block
    // Returns: [B, L_tgt, encoder_dim]  predictions in encoder repr. space
    torch::Tensor forward(torch::Tensor s_x,
                          torch::Tensor ctx_idx,
                          torch::Tensor tgt_idx);
};
TORCH_MODULE(IJEPAPredictor);

// ─────────────────────────────────────────────────────────────────────────────
// IJEPAConfig  —  model-size presets (Appendix A, Table 1)
// ─────────────────────────────────────────────────────────────────────────────
struct IJEPAConfig {
    std::string name;
    int64_t img_size      = 224;
    int64_t patch_size    = 16;
    int64_t embed_dim     = 768;
    int64_t encoder_depth = 12;
    int64_t encoder_heads = 12;
    int64_t pred_dim      = 384;
    int64_t pred_depth    = 6;
    int64_t pred_heads    = 12;   // same as encoder heads (Appendix A)

    // Multi-block masking defaults (Table 8/9/10 optimal values)
    int     num_target_blocks  = 4;
    double  target_scale_min   = 0.15;
    double  target_scale_max   = 0.20;
    double  target_aspect_min  = 0.75;
    double  target_aspect_max  = 1.50;
    double  context_scale_min  = 0.85;
    double  context_scale_max  = 1.00;

    // EMA momentum (Appendix A: 0.996, linearly → 1.0)
    double  ema_momentum = 0.996;

    static IJEPAConfig vit_b16(int64_t img_size = 224);
    static IJEPAConfig vit_l16(int64_t img_size = 224);
    static IJEPAConfig vit_h14(int64_t img_size = 224);
};

// ─────────────────────────────────────────────────────────────────────────────
// MaskSampler  —  multi-block masking strategy (Section 3, Fig. 4)
// ─────────────────────────────────────────────────────────────────────────────
struct MaskSampler {
    int64_t grid_h, grid_w;  // patch grid dimensions

    MaskSampler(int64_t grid_h, int64_t grid_w)
        : grid_h(grid_h), grid_w(grid_w) {}

    // Sample one block of patches.
    // Returns flat patch indices on the grid [h*grid_w + w].
    std::vector<int64_t> sample_block(double scale_min, double scale_max,
                                      double aspect_min, double aspect_max) const;

    // Sample context + M target blocks with overlap removal (Fig. 4).
    // Returns: {context_indices, {target_0_indices, ..., target_{M-1}_indices}}
    std::pair<std::vector<int64_t>,
              std::vector<std::vector<int64_t>>>
    sample(int M, double tgt_scale_min, double tgt_scale_max,
           double tgt_aspect_min, double tgt_aspect_max,
           double ctx_scale_min, double ctx_scale_max) const;
};

// ─────────────────────────────────────────────────────────────────────────────
// IJEPA  —  full model: context-encoder + target-encoder + predictor
// ─────────────────────────────────────────────────────────────────────────────
struct IJEPAImpl : torch::nn::Module {
    IJEPAConfig  cfg;
    int64_t      num_patches;
    MaskSampler  mask_sampler;

    ViTEncoder   context_encoder{nullptr};
    ViTEncoder   target_encoder{nullptr};   // EMA copy; not in optimizer
    IJEPAPredictor predictor{nullptr};

    IJEPAImpl(const IJEPAConfig &cfg);

    // Patchify a batch of images [B, C, H, W] → [B, N, patch_size²*C].
    torch::Tensor patchify(torch::Tensor img) const;

    // Compute I-JEPA loss for a batch.
    // img: [B, 3, H, W] raw pixel tensor (normalised externally).
    torch::Tensor forward(torch::Tensor img);

    // Update target encoder weights: θ̄ ← τ·θ̄ + (1-τ)·θ.
    void update_target_encoder(double momentum);
};
TORCH_MODULE(IJEPA);

// ─────────────────────────────────────────────────────────────────────────────
// Factory helpers
// ─────────────────────────────────────────────────────────────────────────────
inline IJEPA make_ijepa_vit_b16(int64_t img_size = 224) {
    return IJEPA(IJEPAConfig::vit_b16(img_size));
}
inline IJEPA make_ijepa_vit_l16(int64_t img_size = 224) {
    return IJEPA(IJEPAConfig::vit_l16(img_size));
}
inline IJEPA make_ijepa_vit_h14(int64_t img_size = 224) {
    return IJEPA(IJEPAConfig::vit_h14(img_size));
}

} // namespace vision
} // namespace models
} // namespace dm
