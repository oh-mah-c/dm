#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// RealFormer: Residual Attention Layer Transformer
// Ruining He, Anirudh Ravula, Bhargav Kanagal, Joshua Ainslie (Google Research)
// ACL-IJCNLP 2021 Findings — https://aclanthology.org/2021.findings-acl.81
//
// ── What this file provides ───────────────────────────────────────────────────
//
// The core residual attention primitive lives in the dm-extended PyTorch tree:
//   #include <torch/nn/modules/residual_attention.h>
//   → dm::prim::ResidualMultiheadAttention
//
// This header builds higher-level RealFormer building blocks on top of it:
//
//   RealFormerBlock    — Post-LN encoder block with residual MHA + FFN.
//                        Returns (hidden, new_prev) so callers thread the
//                        score tensor through all layers.
//
//   RealFormerEncoder  — Stack of N RealFormerBlocks (bidirectional).
//                        Handles prev-score threading internally.
//
//   RealFormerDecoder  — Same stack with forced causal masking (for decoders).
//
// ── Quick start ───────────────────────────────────────────────────────────────
//
//   // Encoder (e.g. BERT-like pre-training)
//   auto enc = RealFormerEncoder(RealFormerConfig::bert_base());
//   auto [out, _] = enc->forward(embedded_tokens);   // [B,T,D]
//
//   // Decoder (causal language model)
//   auto dec = RealFormerDecoder(RealFormerConfig::gpt2_small());
//   auto [logits, _] = dec->forward(embedded_tokens);
//
// ── Residual attention API (single layer) ─────────────────────────────────────
//
//   dm::prim::ResidualMultiheadAttention mha(512, 8);
//   torch::Tensor prev;   // empty on first call
//   for (int l = 1; l <= n_layers; ++l) {
//       auto [h, new_prev] = mha(x, x, x, prev,
//                                /*key_padding_mask=*/{},
//                                /*causal=*/false,
//                                /*layer=*/l);
//       x    = norm(x + h);
//       prev = new_prev;
//   }
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/nn/modules/residual_attention.h>  // dm::prim primitive
#include <torch/torch.h>

#include <cstdint>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────
struct RealFormerConfig {
    int64_t d_model    = 768;    ///< Hidden / embedding dimension
    int64_t n_heads    = 12;     ///< Number of attention heads
    int64_t ffn_dim    = 3072;   ///< FFN intermediate dimension (usually 4×d_model)
    int64_t n_layers   = 12;     ///< Number of stacked blocks
    float   dropout    = 0.1f;   ///< Residual + FFN dropout
    float   attn_drop  = 0.1f;   ///< Attention-weight dropout
    bool    causal     = false;  ///< Causal (decoder) masking
    bool    use_mean   = false;  ///< Use running-mean instead of running-sum

    int64_t head_dim() const { return d_model / n_heads; }

    static RealFormerConfig bert_base();   ///< 12L 768H 12A
    static RealFormerConfig bert_large();  ///< 24L 1024H 16A
    static RealFormerConfig small();       ///< 2L 64H 4A  (smoke tests)
    static RealFormerConfig gpt2_small();  ///< 12L 768H 12A causal
};

// ─────────────────────────────────────────────────────────────────────────────
// RealFormerBlock — Post-LN encoder/decoder block
//
// Forward formula (Post-LN):
//   h, new_prev = ResidualMHA(x, x, x, prev, ...)
//   x           = LayerNorm( x + Dropout(h) )
//   x           = LayerNorm( x + Dropout( FFN(x) ) )
//   return (x, new_prev)
//
// The `prev` tensor ([B, H, T, T]) is threaded through the block unchanged
// except that the internal MHA updates it with the current layer's scores.
// ─────────────────────────────────────────────────────────────────────────────
struct RealFormerBlockImpl : torch::nn::Module {
    explicit RealFormerBlockImpl(const RealFormerConfig& cfg);

    /// @param x         [B, T, d_model]
    /// @param prev      Previous raw scores [B, H, T, T] or undefined (layer 1)
    /// @param key_mask  Padding mask [B, T] — true = ignore
    /// @param layer     1-indexed layer depth (for use_mean mode)
    /// @returns (out [B, T, d_model], new_prev [B, H, T, T])
    std::pair<torch::Tensor, torch::Tensor>
    forward(const torch::Tensor& x,
            const torch::Tensor& prev     = {},
            const torch::Tensor& key_mask = {},
            int64_t              layer    = 1);

    dm::prim::ResidualMultiheadAttention  attn{nullptr};
    torch::nn::Linear                     ff1{nullptr}, ff2{nullptr};
    torch::nn::LayerNorm                  norm1{nullptr}, norm2{nullptr};
    torch::nn::Dropout                    drop{nullptr};
    bool                                  causal_{false};
};
TORCH_MODULE(RealFormerBlock);

// ─────────────────────────────────────────────────────────────────────────────
// RealFormerEncoder — bidirectional stack of RealFormerBlocks
// ─────────────────────────────────────────────────────────────────────────────
struct RealFormerEncoderImpl : torch::nn::Module {
    explicit RealFormerEncoderImpl(const RealFormerConfig& cfg);

    /// @param x        [B, T, d_model]  (caller provides embeddings)
    /// @param key_mask Padding mask [B, T] optional
    /// @returns (out [B, T, d_model], final_prev [B, H, T, T])
    std::pair<torch::Tensor, torch::Tensor>
    forward(const torch::Tensor& x,
            const torch::Tensor& key_mask = {});

    RealFormerConfig      cfg_;
    torch::nn::ModuleList layers_{nullptr};
    torch::nn::Dropout    drop_{nullptr};
};
TORCH_MODULE(RealFormerEncoder);

// ─────────────────────────────────────────────────────────────────────────────
// RealFormerDecoder — causal decoder stack (causal=true forced on all blocks)
// ─────────────────────────────────────────────────────────────────────────────
struct RealFormerDecoderImpl : torch::nn::Module {
    explicit RealFormerDecoderImpl(const RealFormerConfig& cfg);

    /// @param x        [B, T, d_model]
    /// @returns (out [B, T, d_model], final_prev [B, H, T, T])
    std::pair<torch::Tensor, torch::Tensor>
    forward(const torch::Tensor& x,
            const torch::Tensor& key_mask = {});

    RealFormerConfig      cfg_;
    torch::nn::ModuleList layers_{nullptr};
    torch::nn::Dropout    drop_{nullptr};
};
TORCH_MODULE(RealFormerDecoder);

} // namespace nlp
} // namespace models
} // namespace dm
