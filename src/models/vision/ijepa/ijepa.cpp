// ─────────────────────────────────────────────────────────────────────────────
// I-JEPA implementation
// Assran et al., arXiv:2301.08243v3, CVPR 2023
//
// Equations referenced:
//   Loss:        Section 3 — (1/M) Σ_i Σ_{j∈B_i} ||ŝ_y_j − s_y_j||²
//   EMA update:  θ̄ ← τ·θ̄ + (1-τ)·θ    (Appendix A.1, τ = 0.996 → 1.0)
//   Masking:     M=4 target blocks (scale 0.15-0.20, aspect 0.75-1.5)
//                1 context block  (scale 0.85-1.0,  aspect ≈1; overlap removed)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/ijepa/ijepa.h"

#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace dm {
namespace models {
namespace vision {

// ─────────────────────────────────────────────────────────────────────────────
// Attention
// ─────────────────────────────────────────────────────────────────────────────

AttentionImpl::AttentionImpl(int64_t dim, int64_t num_heads)
    : num_heads(num_heads),
      head_dim(dim / num_heads),
      scale(1.0 / std::sqrt(static_cast<double>(dim / num_heads))) {
    qkv  = register_module("qkv",  torch::nn::Linear(
               torch::nn::LinearOptions(dim, dim * 3).bias(false)));
    proj = register_module("proj", torch::nn::Linear(dim, dim));
}

torch::Tensor AttentionImpl::forward(torch::Tensor x) {
    auto [B, N, C] = std::make_tuple(x.size(0), x.size(1), x.size(2));
    // QKV: [B, N, 3*C] → [B, N, 3, heads, head_dim] → [3, B, heads, N, head_dim]
    auto qkv_t = qkv(x).reshape({B, N, 3, num_heads, head_dim})
                        .permute({2, 0, 3, 1, 4});
    auto q = qkv_t[0], k = qkv_t[1], v = qkv_t[2];
    // Scaled dot-product attention
    auto attn = torch::softmax((q * scale).matmul(k.transpose(-2, -1)), -1);
    auto out  = attn.matmul(v)                          // [B, heads, N, head_dim]
                    .transpose(1, 2)                    // [B, N, heads, head_dim]
                    .reshape({B, N, C});                // [B, N, C]
    return proj(out);
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP
// ─────────────────────────────────────────────────────────────────────────────

MLPImpl::MLPImpl(int64_t dim, int64_t hidden_dim) {
    fc1 = register_module("fc1", torch::nn::Linear(dim, hidden_dim));
    fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, dim));
}

torch::Tensor MLPImpl::forward(torch::Tensor x) {
    return fc2(torch::gelu(fc1(x)));
}

// ─────────────────────────────────────────────────────────────────────────────
// ViTBlock
// ─────────────────────────────────────────────────────────────────────────────

ViTBlockImpl::ViTBlockImpl(int64_t dim, int64_t num_heads, int64_t mlp_hidden) {
    norm1 = register_module("norm1", torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({dim})));
    norm2 = register_module("norm2", torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({dim})));
    attn  = register_module("attn", Attention(dim, num_heads));
    mlp   = register_module("mlp",  MLP(dim, mlp_hidden));
}

torch::Tensor ViTBlockImpl::forward(torch::Tensor x) {
    x = x + attn(norm1(x));
    x = x + mlp(norm2(x));
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// ViTEncoder
// ─────────────────────────────────────────────────────────────────────────────

ViTEncoderImpl::ViTEncoderImpl(int64_t img_size, int64_t patch_size,
                                int64_t embed_dim, int64_t depth,
                                int64_t num_heads)
    : embed_dim(embed_dim),
      patch_size(patch_size),
      num_patches((img_size / patch_size) * (img_size / patch_size)) {
    int64_t patch_dim = 3 * patch_size * patch_size;
    patch_embed = register_module("patch_embed",
                    torch::nn::Linear(patch_dim, embed_dim));
    pos_embed   = register_module("pos_embed",
                    torch::nn::Embedding(num_patches, embed_dim));
    norm        = register_module("norm", torch::nn::LayerNorm(
                    torch::nn::LayerNormOptions({embed_dim})));
    int64_t mlp_hidden = embed_dim * 4;
    for (int64_t i = 0; i < depth; i++) {
        auto blk = ViTBlock(embed_dim, num_heads, mlp_hidden);
        blocks->push_back(blk);
    }
    register_module("blocks", blocks);
}

torch::Tensor ViTEncoderImpl::forward(torch::Tensor patches, torch::Tensor indices) {
    // patches: [B, N, patch_dim] — either full N or a subset L
    // indices: [B, L] or undefined (→ use all 0..N-1)
    auto x = patch_embed(patches);  // [B, L, embed_dim]

    if (indices.defined()) {
        // Gather positional embeddings at the given positions
        // indices: [B, L] → look up pos_embed [num_patches, D]
        auto B = x.size(0), L = x.size(1);
        // Flatten indices for embedding lookup, then reshape
        auto flat_idx = indices.reshape({-1});                       // [B*L]
        auto pe = pos_embed(flat_idx).reshape({B, L, embed_dim});   // [B, L, D]
        x = x + pe;
    } else {
        // All patches: arange [0..N-1]
        auto N = x.size(1);
        auto idx = torch::arange(N, x.options().dtype(torch::kLong));
        x = x + pos_embed(idx).unsqueeze(0);  // [1, N, D] broadcast
    }

    for (auto &blk : *blocks)
        x = blk->as<ViTBlock>()->forward(x);
    return norm(x);  // [B, L, embed_dim]
}

// ─────────────────────────────────────────────────────────────────────────────
// IJEPAPredictor
// ─────────────────────────────────────────────────────────────────────────────

IJEPAPredictorImpl::IJEPAPredictorImpl(int64_t encoder_dim, int64_t pred_dim,
                                       int64_t num_patches, int64_t depth,
                                       int64_t num_heads)
    : pred_dim(pred_dim), num_patches(num_patches) {
    input_proj  = register_module("input_proj",
                    torch::nn::Linear(encoder_dim, pred_dim));
    mask_token  = register_parameter("mask_token",
                    torch::zeros({1, 1, pred_dim}));
    pos_embed   = register_module("pos_embed",
                    torch::nn::Embedding(num_patches, pred_dim));
    norm        = register_module("norm", torch::nn::LayerNorm(
                    torch::nn::LayerNormOptions({pred_dim})));
    output_proj = register_module("output_proj",
                    torch::nn::Linear(pred_dim, encoder_dim));
    int64_t mlp_hidden = pred_dim * 4;
    for (int64_t i = 0; i < depth; i++) {
        auto blk = ViTBlock(pred_dim, num_heads, mlp_hidden);
        blocks->push_back(blk);
    }
    register_module("blocks", blocks);
}

torch::Tensor IJEPAPredictorImpl::forward(torch::Tensor s_x,
                                           torch::Tensor ctx_idx,
                                           torch::Tensor tgt_idx) {
    // s_x:     [B, L_ctx, encoder_dim]
    // ctx_idx: [B, L_ctx]
    // tgt_idx: [B, L_tgt]
    auto B      = s_x.size(0);
    auto L_ctx  = s_x.size(1);
    auto L_tgt  = tgt_idx.size(1);

    // Project context tokens to pred_dim and add positional embeddings
    auto ctx = input_proj(s_x);                                  // [B, L_ctx, pred_dim]
    auto ctx_pe = pos_embed(ctx_idx.reshape({-1}))
                      .reshape({B, L_ctx, pred_dim});
    ctx = ctx + ctx_pe;

    // Build mask tokens for target positions and add positional embeddings
    auto tgt = mask_token.expand({B, L_tgt, pred_dim});          // [B, L_tgt, pred_dim]
    auto tgt_pe = pos_embed(tgt_idx.reshape({-1}))
                      .reshape({B, L_tgt, pred_dim});
    tgt = tgt + tgt_pe;

    // Concatenate: [context tokens | mask tokens] → run predictor
    auto x = torch::cat({ctx, tgt}, 1);                          // [B, L_ctx+L_tgt, pred_dim]
    for (auto &blk : *blocks)
        x = blk->as<ViTBlock>()->forward(x);
    x = norm(x);

    // Extract only the target-position outputs and project back to encoder dim
    auto pred = x.narrow(1, L_ctx, L_tgt);                      // [B, L_tgt, pred_dim]
    return output_proj(pred);                                     // [B, L_tgt, encoder_dim]
}

// ─────────────────────────────────────────────────────────────────────────────
// IJEPAConfig presets  (Appendix A, Table 1)
// ─────────────────────────────────────────────────────────────────────────────

IJEPAConfig IJEPAConfig::vit_b16(int64_t img_size) {
    IJEPAConfig c;
    c.name          = "vit_b16";
    c.img_size      = img_size;
    c.patch_size    = 16;
    c.embed_dim     = 768;
    c.encoder_depth = 12;
    c.encoder_heads = 12;
    c.pred_dim      = 384;
    c.pred_depth    = 6;   // Appendix A: "set the depth of the predictor to 6"
    c.pred_heads    = 12;
    return c;
}

IJEPAConfig IJEPAConfig::vit_l16(int64_t img_size) {
    IJEPAConfig c;
    c.name          = "vit_l16";
    c.img_size      = img_size;
    c.patch_size    = 16;
    c.embed_dim     = 1024;
    c.encoder_depth = 24;
    c.encoder_heads = 16;
    c.pred_dim      = 384;
    c.pred_depth    = 12;  // Appendix A: depth 12 for L/H
    c.pred_heads    = 16;
    return c;
}

IJEPAConfig IJEPAConfig::vit_h14(int64_t img_size) {
    IJEPAConfig c;
    c.name          = "vit_h14";
    c.img_size      = img_size;
    c.patch_size    = 14;
    c.embed_dim     = 1280;
    c.encoder_depth = 32;
    c.encoder_heads = 16;
    c.pred_dim      = 384;
    c.pred_depth    = 12;
    c.pred_heads    = 16;
    return c;
}

// ─────────────────────────────────────────────────────────────────────────────
// MaskSampler
// ─────────────────────────────────────────────────────────────────────────────

static thread_local std::mt19937 rng{std::random_device{}()};

std::vector<int64_t> MaskSampler::sample_block(double scale_min, double scale_max,
                                                double aspect_min, double aspect_max) const {
    int64_t total = grid_h * grid_w;
    std::uniform_real_distribution<double> scale_dist(scale_min, scale_max);
    std::uniform_real_distribution<double> aspect_dist(aspect_min, aspect_max);

    for (int attempt = 0; attempt < 20; attempt++) {
        double scale  = scale_dist(rng);
        double aspect = aspect_dist(rng);
        int64_t area  = static_cast<int64_t>(std::round(scale * total));
        int64_t h     = static_cast<int64_t>(std::round(std::sqrt(area * aspect)));
        int64_t w     = static_cast<int64_t>(std::round(std::sqrt(area / aspect)));
        h = std::max<int64_t>(1, std::min(h, grid_h));
        w = std::max<int64_t>(1, std::min(w, grid_w));

        if (h > grid_h || w > grid_w) continue;

        std::uniform_int_distribution<int64_t> row_dist(0, grid_h - h);
        std::uniform_int_distribution<int64_t> col_dist(0, grid_w - w);
        int64_t r0 = row_dist(rng), c0 = col_dist(rng);

        std::vector<int64_t> indices;
        indices.reserve(static_cast<size_t>(h * w));
        for (int64_t r = r0; r < r0 + h; r++)
            for (int64_t c = c0; c < c0 + w; c++)
                indices.push_back(r * grid_w + c);
        return indices;
    }
    // Fallback: all patches
    std::vector<int64_t> all(static_cast<size_t>(total));
    std::iota(all.begin(), all.end(), 0);
    return all;
}

std::pair<std::vector<int64_t>, std::vector<std::vector<int64_t>>>
MaskSampler::sample(int M,
                    double tgt_scale_min, double tgt_scale_max,
                    double tgt_aspect_min, double tgt_aspect_max,
                    double ctx_scale_min,  double ctx_scale_max) const {
    // Sample M target blocks
    std::vector<std::vector<int64_t>> targets;
    targets.reserve(static_cast<size_t>(M));
    std::unordered_set<int64_t> all_tgt_set;
    for (int i = 0; i < M; i++) {
        auto t = sample_block(tgt_scale_min, tgt_scale_max,
                              tgt_aspect_min, tgt_aspect_max);
        targets.push_back(t);
        for (auto idx : t) all_tgt_set.insert(idx);
    }

    // Sample context block and remove overlapping patches (Fig. 4)
    auto ctx_raw = sample_block(ctx_scale_min, ctx_scale_max, 1.0, 1.0);
    std::vector<int64_t> context;
    context.reserve(ctx_raw.size());
    for (auto idx : ctx_raw)
        if (all_tgt_set.find(idx) == all_tgt_set.end())
            context.push_back(idx);

    // Ensure context is non-empty (fallback: keep all context patches)
    if (context.empty()) context = ctx_raw;

    return {context, targets};
}

// ─────────────────────────────────────────────────────────────────────────────
// IJEPA
// ─────────────────────────────────────────────────────────────────────────────

IJEPAImpl::IJEPAImpl(const IJEPAConfig &cfg)
    : cfg(cfg),
      num_patches((cfg.img_size / cfg.patch_size) * (cfg.img_size / cfg.patch_size)),
      mask_sampler(cfg.img_size / cfg.patch_size, cfg.img_size / cfg.patch_size) {

    context_encoder = register_module("context_encoder",
        ViTEncoder(cfg.img_size, cfg.patch_size,
                   cfg.embed_dim, cfg.encoder_depth, cfg.encoder_heads));
    target_encoder  = register_module("target_encoder",
        ViTEncoder(cfg.img_size, cfg.patch_size,
                   cfg.embed_dim, cfg.encoder_depth, cfg.encoder_heads));
    predictor       = register_module("predictor",
        IJEPAPredictor(cfg.embed_dim, cfg.pred_dim,
                       num_patches, cfg.pred_depth, cfg.pred_heads));

    // Initialise target encoder = context encoder; disable its gradients
    {
        torch::NoGradGuard ng;
        auto ctx_params = context_encoder->parameters();
        auto tgt_params = target_encoder->parameters();
        for (size_t i = 0; i < ctx_params.size(); i++)
            tgt_params[i].copy_(ctx_params[i]);
        for (auto &p : target_encoder->parameters())
            p.requires_grad_(false);
    }
}

torch::Tensor IJEPAImpl::patchify(torch::Tensor img) const {
    // img: [B, 3, H, W] → [B, N, 3*P*P]
    auto B = img.size(0);
    auto P = cfg.patch_size;
    auto H = img.size(2), W = img.size(3);
    auto gh = H / P, gw = W / P;
    // [B, 3, gh, P, gw, P] → [B, gh, gw, 3, P, P] → [B, gh*gw, 3*P*P]
    auto x = img.reshape({B, 3LL, gh, P, gw, P})
                .permute({0, 2, 4, 1, 3, 5})
                .reshape({B, gh * gw, 3 * P * P});
    return x.contiguous();
}

torch::Tensor IJEPAImpl::forward(torch::Tensor img) {
    auto B     = img.size(0);
    auto dev   = img.device();
    auto dtype = img.dtype();
    auto patches = patchify(img);  // [B, N, 3*P*P]

    // Sample one mask per image — for simplicity use a single shared mask per
    // batch (standard practice in most implementations).
    auto [ctx_idx_vec, tgt_idx_vecs] = mask_sampler.sample(
        cfg.num_target_blocks,
        cfg.target_scale_min, cfg.target_scale_max,
        cfg.target_aspect_min, cfg.target_aspect_max,
        cfg.context_scale_min, cfg.context_scale_max);

    // Build index tensors
    auto make_idx = [&](const std::vector<int64_t> &v) {
        auto t = torch::from_blob(
                    const_cast<int64_t*>(v.data()),
                    {static_cast<int64_t>(v.size())},
                    torch::kLong).clone().to(dev);
        return t.unsqueeze(0).expand({B, -1});  // [B, L]
    };

    auto ctx_idx = make_idx(ctx_idx_vec);  // [B, L_ctx]

    // Gather context patches: [B, L_ctx, 3*P*P]
    auto ctx_patches = patches.index_select(
        1, torch::from_blob(
               const_cast<int64_t*>(ctx_idx_vec.data()),
               {static_cast<int64_t>(ctx_idx_vec.size())},
               torch::kLong).clone().to(dev));

    // ── Context encoder forward (with gradients) ──────────────────────────
    auto s_x = context_encoder(ctx_patches, ctx_idx);  // [B, L_ctx, D]

    // ── Target encoder forward (no gradients) ─────────────────────────────
    torch::Tensor s_y;
    {
        torch::NoGradGuard ng;
        s_y = target_encoder(patches);  // [B, N, D] — full image
    }

    // ── Predictor + loss ──────────────────────────────────────────────────
    torch::Tensor loss = torch::zeros({}, torch::TensorOptions().device(dev).dtype(dtype));
    for (const auto &tgt_v : tgt_idx_vecs) {
        auto tgt_idx_t  = make_idx(tgt_v);              // [B, L_tgt]
        auto tgt_long   = torch::from_blob(
                              const_cast<int64_t*>(tgt_v.data()),
                              {static_cast<int64_t>(tgt_v.size())},
                              torch::kLong).clone().to(dev);

        // Predicted representations at target positions
        auto s_hat = predictor(s_x, ctx_idx, tgt_idx_t); // [B, L_tgt, D]

        // Target representations at target positions (gathered from s_y)
        auto s_tgt = s_y.index_select(1, tgt_long);       // [B, L_tgt, D]
        // Detach: targets come from EMA encoder, no gradient flows through them
        s_tgt = s_tgt.detach();

        // L2 loss in representation space (Section 3, loss equation)
        loss = loss + torch::mean((s_hat - s_tgt).pow(2));
    }
    return loss / static_cast<double>(cfg.num_target_blocks);
}

void IJEPAImpl::update_target_encoder(double momentum) {
    // θ̄ ← τ·θ̄ + (1-τ)·θ   (Appendix A: τ=0.996, linearly → 1.0)
    torch::NoGradGuard ng;
    auto ctx_params = context_encoder->parameters();
    auto tgt_params = target_encoder->parameters();
    for (size_t i = 0; i < ctx_params.size(); i++)
        tgt_params[i].mul_(momentum).add_(ctx_params[i] * (1.0 - momentum));
}

} // namespace vision
} // namespace models
} // namespace dm
