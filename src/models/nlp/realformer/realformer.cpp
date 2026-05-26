// ─────────────────────────────────────────────────────────────────────────────
// RealFormer implementation
// He et al., ACL-IJCNLP 2021 Findings — arXiv:2012.11747
//
// The residual attention primitive (dm::prim::ResidualMultiheadAttention) lives
// in the PyTorch include tree:
//   torch/csrc/api/include/torch/nn/modules/residual_attention.h
//
// This file implements the higher-level block / encoder / decoder stacks.
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/realformer/realformer.h"

#include <torch/torch.h>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// RealFormerConfig factories
// ─────────────────────────────────────────────────────────────────────────────
RealFormerConfig RealFormerConfig::bert_base() {
    RealFormerConfig c;
    c.d_model = 768; c.n_heads = 12; c.ffn_dim = 3072; c.n_layers = 12;
    c.dropout = 0.1f; c.attn_drop = 0.1f; c.causal = false; c.use_mean = false;
    return c;
}

RealFormerConfig RealFormerConfig::bert_large() {
    RealFormerConfig c;
    c.d_model = 1024; c.n_heads = 16; c.ffn_dim = 4096; c.n_layers = 24;
    c.dropout = 0.1f; c.attn_drop = 0.1f; c.causal = false; c.use_mean = false;
    return c;
}

RealFormerConfig RealFormerConfig::small() {
    RealFormerConfig c;
    c.d_model = 64; c.n_heads = 4; c.ffn_dim = 256; c.n_layers = 2;
    c.dropout = 0.0f; c.attn_drop = 0.0f; c.causal = false; c.use_mean = false;
    return c;
}

RealFormerConfig RealFormerConfig::gpt2_small() {
    RealFormerConfig c;
    c.d_model = 768; c.n_heads = 12; c.ffn_dim = 3072; c.n_layers = 12;
    c.dropout = 0.1f; c.attn_drop = 0.1f; c.causal = true; c.use_mean = false;
    return c;
}

// ─────────────────────────────────────────────────────────────────────────────
// RealFormerBlock
// ─────────────────────────────────────────────────────────────────────────────
RealFormerBlockImpl::RealFormerBlockImpl(const RealFormerConfig& cfg)
    : causal_(cfg.causal)
{
    dm::prim::ResidualMultiheadAttentionOptions attn_opts(cfg.d_model, cfg.n_heads);
    attn_opts.dropout(cfg.attn_drop).use_mean(cfg.use_mean);
    attn  = register_module("attn",  dm::prim::ResidualMultiheadAttention(attn_opts));

    ff1   = register_module("ff1",   torch::nn::Linear(cfg.d_model, cfg.ffn_dim));
    ff2   = register_module("ff2",   torch::nn::Linear(cfg.ffn_dim, cfg.d_model));
    norm1 = register_module("norm1", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({cfg.d_model})));
    norm2 = register_module("norm2", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({cfg.d_model})));
    drop  = register_module("drop",  torch::nn::Dropout(
        torch::nn::DropoutOptions(cfg.dropout)));
}

std::pair<torch::Tensor, torch::Tensor>
RealFormerBlockImpl::forward(const torch::Tensor& x,
                              const torch::Tensor& prev,
                              const torch::Tensor& key_mask,
                              int64_t              layer)
{
    // Residual self-attention (Post-LN)
    auto [h, new_prev] = attn->forward(x, x, x, prev, key_mask, causal_, layer);
    auto x2 = norm1(x + drop(h));

    // FFN: GELU(W1·x)·W2  (standard BERT FFN) — Post-LN
    auto out = norm2(x2 + drop(ff2(torch::gelu(ff1(x2)))));

    return {out, new_prev};
}

// ─────────────────────────────────────────────────────────────────────────────
// RealFormerEncoder
// ─────────────────────────────────────────────────────────────────────────────
RealFormerEncoderImpl::RealFormerEncoderImpl(const RealFormerConfig& cfg)
    : cfg_(cfg)
{
    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_layers; ++i)
        layers_->push_back(RealFormerBlock(cfg));

    drop_ = register_module("drop",
        torch::nn::Dropout(torch::nn::DropoutOptions(cfg.dropout)));
}

std::pair<torch::Tensor, torch::Tensor>
RealFormerEncoderImpl::forward(const torch::Tensor& x,
                                const torch::Tensor& key_mask)
{
    torch::Tensor h    = drop_(x);
    torch::Tensor prev;

    for (size_t i = 0; i < layers_->size(); ++i) {
        auto block = layers_->ptr<RealFormerBlockImpl>(i);
        auto [h2, new_prev] = block->forward(h, prev, key_mask,
                                              static_cast<int64_t>(i + 1));
        h    = h2;
        prev = new_prev;
    }
    return {h, prev};
}

// ─────────────────────────────────────────────────────────────────────────────
// RealFormerDecoder
// ─────────────────────────────────────────────────────────────────────────────
RealFormerDecoderImpl::RealFormerDecoderImpl(const RealFormerConfig& cfg)
    : cfg_(cfg)
{
    RealFormerConfig dcfg = cfg;
    dcfg.causal = true;   // force causal masking for all decoder blocks

    layers_ = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_layers; ++i)
        layers_->push_back(RealFormerBlock(dcfg));

    drop_ = register_module("drop",
        torch::nn::Dropout(torch::nn::DropoutOptions(cfg.dropout)));
}

std::pair<torch::Tensor, torch::Tensor>
RealFormerDecoderImpl::forward(const torch::Tensor& x,
                                const torch::Tensor& key_mask)
{
    torch::Tensor h    = drop_(x);
    torch::Tensor prev;

    for (size_t i = 0; i < layers_->size(); ++i) {
        auto block = layers_->ptr<RealFormerBlockImpl>(i);
        auto [h2, new_prev] = block->forward(h, prev, key_mask,
                                              static_cast<int64_t>(i + 1));
        h    = h2;
        prev = new_prev;
    }
    return {h, prev};
}

} // namespace nlp
} // namespace models
} // namespace dm
