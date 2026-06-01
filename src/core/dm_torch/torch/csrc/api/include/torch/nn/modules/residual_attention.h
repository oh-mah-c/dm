#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// ResidualMultiheadAttention — Residual Attention primitive  (dm::prim)
//
// From: "RealFormer: Transformer Likes Residual Attention"
//   Ruining He, Anirudh Ravula, Bhargav Kanagal, Joshua Ainslie (Google)
//   ACL-IJCNLP 2021 Findings — https://aclanthology.org/2021.findings-acl.81
//
// ── Core mechanism (Eq. 1) ────────────────────────────────────────────────────
//
//   ResidualAttention(Q, K, V, Prev) =
//       Softmax( Q K^T / sqrt(d_k)  +  Prev ) * V
//
// where Prev is the *raw pre-softmax score tensor* [B, H, T_q, T_k] passed in
// from the previous layer.  The updated scores are returned alongside the
// output so the caller can thread them to the next layer:
//
//   new_prev = Q K^T / sqrt(d_k) + Prev     (running sum; or Prev=0 at L=0)
//
// This is equivalent to adding a direct skip-edge on the attention-score
// stream across Transformer layers, analogous to residual connections on
// activations.  It adds zero extra learnable parameters vs standard MHA.
//
// ── Running-mean variant ──────────────────────────────────────────────────────
//
// For deep networks the paper notes that replacing the running sum with a
// running mean can be more stable:
//
//   new_prev = (old_prev * (layer - 1) + raw_scores) / layer
//
// Enable with ResidualMultiheadAttentionOptions::use_mean(true) and pass the
// current (1-indexed) layer number via the `layer` argument of forward().
// Ignored (no-op) when use_mean=false.
//
// ── Drop-in API ───────────────────────────────────────────────────────────────
//
//   #include <torch/nn/modules/residual_attention.h>
//
//   dm::prim::ResidualMultiheadAttention mha(512, 8);
//   auto [out, new_prev] = mha(query, key, value);            // first layer
//   auto [out2, new_prev2] = mha(query2, key2, value2,        // next layer
//                                /*prev=*/new_prev,
//                                /*key_padding_mask=*/{},
//                                /*causal=*/false,
//                                /*layer=*/2);
//
// The prev tensor is undefined (empty) on first call — the module inserts
// zeros automatically.  Subsequent layers pass the returned new_prev.
//
// ── Usage inside a RealFormer stack ───────────────────────────────────────────
//
//   torch::Tensor prev;   // initially undefined
//   for (auto& block : blocks) {
//       auto [h, new_prev] = block.attn(x, x, x, prev);
//       x    = norm(x + h);
//       prev = new_prev;
//   }
//
// ── Parameters ────────────────────────────────────────────────────────────────
//   wq  [d_model, d_model]   Q projection (no bias)
//   wk  [d_model, d_model]   K projection (no bias)
//   wv  [d_model, d_model]   V projection (no bias)
//   wo  [d_model, d_model]   output projection (no bias)
//
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>
#include <torch/csrc/Export.h>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace dm {
namespace prim {

// ─────────────────────────────────────────────────────────────────────────────
// Options
// ─────────────────────────────────────────────────────────────────────────────

struct ResidualMultiheadAttentionOptions {
    /// @param embed_dim   Total embedding dimension (d_model).
    /// @param num_heads   Number of attention heads H; embed_dim % num_heads == 0.
    ResidualMultiheadAttentionOptions(int64_t embed_dim, int64_t num_heads)
        : embed_dim_(embed_dim), num_heads_(num_heads) {}

    /// Total embedding dimension.
    TORCH_ARG(int64_t, embed_dim);

    /// Number of attention heads.
    TORCH_ARG(int64_t, num_heads);

    /// Attention-weight dropout probability.  Default: 0.0 (no dropout).
    TORCH_ARG(double, dropout) = 0.0;

    /// If true, use running *mean* of scores instead of running sum.
    /// Pass the 1-indexed current layer depth to forward() as `layer`.
    TORCH_ARG(bool, use_mean) = false;
};

// ─────────────────────────────────────────────────────────────────────────────
// Implementation
// ─────────────────────────────────────────────────────────────────────────────

/// ResidualMultiheadAttentionImpl — residual attention as a torch::nn module.
///
/// Inherits Module so it integrates with `to()`, `save()`/`load()`,
/// `parameters()`, and autograd automatically.
class TORCH_API ResidualMultiheadAttentionImpl
    : public torch::nn::Module {
 public:
  explicit ResidualMultiheadAttentionImpl(
      int64_t embed_dim, int64_t num_heads)
      : ResidualMultiheadAttentionImpl(
            ResidualMultiheadAttentionOptions(embed_dim, num_heads)) {}

  explicit ResidualMultiheadAttentionImpl(
      ResidualMultiheadAttentionOptions options_);

  void reset();

  void pretty_print(std::ostream& stream) const override;

  /// Forward pass.
  ///
  /// @param query          [B, T_q, embed_dim]
  /// @param key            [B, T_k, embed_dim]
  /// @param value          [B, T_k, embed_dim]
  /// @param prev           Previous raw scores [B, H, T_q, T_k], or undefined
  ///                       (treated as zeros on the first layer).
  /// @param key_padding_mask  Boolean mask [B, T_k]; true = ignore position.
  /// @param causal         If true, apply an upper-triangular causal mask so
  ///                       each query position only attends to positions <= it.
  /// @param layer          1-indexed layer depth, used only when use_mean=true.
  ///
  /// @returns std::pair:
  ///   .first   output      [B, T_q, embed_dim]
  ///   .second  new_prev    [B, H, T_q, T_k]  — pass to next layer's `prev`
  std::pair<torch::Tensor, torch::Tensor>
  forward(const torch::Tensor& query,
          const torch::Tensor& key,
          const torch::Tensor& value,
          const torch::Tensor& prev              = {},
          const torch::Tensor& key_padding_mask  = {},
          bool                 causal            = false,
          int64_t              layer             = 1);

  /// The options used to construct this module.
  ResidualMultiheadAttentionOptions options;

  // ── Learnable projections ─────────────────────────────────────────────────
  torch::nn::Linear wq{nullptr}, wk{nullptr}, wv{nullptr}, wo{nullptr};
  torch::nn::Dropout attn_drop{nullptr};

 private:
  int64_t head_dim_;
};

TORCH_MODULE(ResidualMultiheadAttention);

// ─────────────────────────────────────────────────────────────────────────────
// Inline implementation — header-only so any dm module can include it without
// linking a separate translation unit.
// ─────────────────────────────────────────────────────────────────────────────

inline ResidualMultiheadAttentionImpl::ResidualMultiheadAttentionImpl(
    ResidualMultiheadAttentionOptions options_)
    : options(std::move(options_))
{
  if (options.embed_dim() % options.num_heads() != 0)
    throw std::invalid_argument(
        "ResidualMultiheadAttention: embed_dim must be divisible by num_heads");
  head_dim_ = options.embed_dim() / options.num_heads();
  reset();
}

inline void ResidualMultiheadAttentionImpl::reset() {
  const int64_t D = options.embed_dim();
  wq = register_module("wq",
      torch::nn::Linear(torch::nn::LinearOptions(D, D).bias(false)));
  wk = register_module("wk",
      torch::nn::Linear(torch::nn::LinearOptions(D, D).bias(false)));
  wv = register_module("wv",
      torch::nn::Linear(torch::nn::LinearOptions(D, D).bias(false)));
  wo = register_module("wo",
      torch::nn::Linear(torch::nn::LinearOptions(D, D).bias(false)));
  attn_drop = register_module("attn_drop",
      torch::nn::Dropout(torch::nn::DropoutOptions(options.dropout())));
}

inline void ResidualMultiheadAttentionImpl::pretty_print(
    std::ostream& stream) const {
  stream << "dm::prim::ResidualMultiheadAttention("
         << "embed_dim=" << options.embed_dim()
         << ", num_heads=" << options.num_heads()
         << ", dropout=" << options.dropout()
         << ", use_mean=" << (options.use_mean() ? "true" : "false")
         << ")";
}

inline std::pair<torch::Tensor, torch::Tensor>
ResidualMultiheadAttentionImpl::forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& prev,
    const torch::Tensor& key_padding_mask,
    bool                 causal,
    int64_t              layer)
{
  const int64_t B  = query.size(0);
  const int64_t Tq = query.size(1);
  const int64_t Tk = key.size(1);
  const int64_t H  = options.num_heads();
  const double  scale = 1.0 / std::sqrt(static_cast<double>(head_dim_));

  // Project and split into heads: [B, T, D] -> [B, H, T, head_dim]
  auto Q = wq(query).view({B, Tq, H, head_dim_}).transpose(1, 2);
  auto K = wk(key  ).view({B, Tk, H, head_dim_}).transpose(1, 2);
  auto V = wv(value).view({B, Tk, H, head_dim_}).transpose(1, 2);

  // Raw scores: [B, H, Tq, Tk]
  auto raw = torch::matmul(Q, K.transpose(-2, -1)) * scale;

  // ── Residual accumulation ─────────────────────────────────────────────────
  torch::Tensor new_prev;
  if (!prev.defined() || prev.numel() == 0) {
    // First layer: accumulated scores are just the raw scores
    new_prev = raw;
  } else if (options.use_mean() && layer > 1) {
    // Running mean: keeps scores bounded regardless of depth
    new_prev = (prev * static_cast<double>(layer - 1) + raw)
               / static_cast<double>(layer);
  } else {
    // Running sum (default, paper Eq. 1)
    new_prev = raw + prev;
  }

  // The scores fed into softmax are the accumulated residual scores
  auto scores = new_prev;

  // ── Key-padding mask: [B, Tk] → broadcast [B, 1, 1, Tk] ─────────────────
  if (key_padding_mask.defined() && key_padding_mask.numel() > 0) {
    scores = scores.masked_fill(
        key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9);
  }

  // ── Causal mask: upper-triangular (positions > t are masked) ─────────────
  if (causal) {
    auto mask = torch::ones({Tq, Tk}, query.options())
                    .triu(1)
                    .to(torch::kBool);
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9);
  }

  // Softmax → dropout → aggregate V
  auto attn_w = torch::softmax(scores, /*dim=*/-1);
  attn_w      = attn_drop(attn_w);

  // [B,H,Tq,hd] → [B,Tq,D]
  auto ctx = torch::matmul(attn_w, V)
                 .transpose(1, 2).contiguous()
                 .view({B, Tq, H * head_dim_});

  return {wo(ctx), new_prev};
}

}  // namespace prim
}  // namespace dm
