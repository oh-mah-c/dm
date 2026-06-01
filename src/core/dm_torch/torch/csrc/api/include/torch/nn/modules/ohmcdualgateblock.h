#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// OhmCDualGateBlock — Parallel Cross-Gated Transformer Block  (dm::prim)
//
// Architectural overview
// ──────────────────────
// Instead of running Attention then FFN sequentially (as in every standard
// Transformer variant), this block forks the raw input x simultaneously into
// two independent pipelines and recombines them through a novel cross-gate:
//
//                         ┌─────────────────────────────────┐
//                         │          Input  x [B,T,D]       │
//                         └──────────┬──────────────┬───────┘
//                                    │              │
//                     PATH 1         │              │    PATH 2
//              (Attention core)      │              │  (FFN / MLP core)
//                                    ▼              ▼
//                        RMSNorm(x)               RMSNorm(x)
//                              │                       │
//                    Q,K,V projections          SwiGLU gate:
//                    GQA expand + RoPE    silu(w1(·)) ⊙ w3(·) = ffn_gate
//                    causal attn + wo                  │
//                    RMSNorm → attn_out                │
//                              │                       │
//                              └──────────┬────────────┘
//                                         │
//                              CROSS-GATE (novel):
//                         attn_proj = w_cross(attn_out)   [B,T,ffn_dim]
//                         gated_mid = ffn_gate ⊙ attn_proj
//                         ffn_raw   = w2(gated_mid)        [B,T,D]
//                         ffn_out   = RMSNorm(ffn_raw)
//                                         │
//                     DUAL GATED RESIDUAL FUSION:
//          output = x  +  attn_out * gate_attn  +  ffn_out * gate_ffn
//
// Mathematical formulation
// ────────────────────────
// Let D = dim, H = n_heads, Hkv = n_kv_heads, hd = D/H, F = ffn_dim.
//
//   norm_a  = RMSNorm(x; w_norm_attn_in)              [B,T,D]
//   Q       = wq(norm_a).view(B,T,H,hd).T(1,2)        [B,H,T,hd]
//   K,V     = wk/wv(norm_a).view(B,T,Hkv,hd)          [B,Hkv,T,hd]
//             (GQA: K,V repeat-interleaved to H heads if Hkv<H)
//   S       = softmax(Q Kᵀ / sqrt(hd) + causal_mask)  [B,H,T,T]
//   attn_raw= wo( (S·V).T(1,2).reshape(B,T,D) )        [B,T,D]
//   attn_out= RMSNorm(attn_raw; w_norm_attn_out)        [B,T,D]
//
//   norm_f  = RMSNorm(x; w_norm_ffn_in)                [B,T,D]
//   ffn_gate= silu(w1(norm_f)) ⊙ w3(norm_f)            [B,T,F]   ← SwiGLU
//
//   attn_proj   = w_cross(attn_out)                    [B,T,F]
//   gated_mid   = ffn_gate ⊙ attn_proj                 [B,T,F]   ← Hadamard
//   ffn_raw     = w2(gated_mid)                         [B,T,D]
//   ffn_out     = RMSNorm(ffn_raw; w_norm_ffn_out)      [B,T,D]
//
//   output  = x  +  attn_out * gate_attn  +  ffn_out * gate_ffn
//                   └── [D] broadcast ──┘   └── [D] broadcast ──┘
//
// Learnable parameters
// ────────────────────
//   wq, wk, wv, wo        — attention projections (no bias)
//   w1, w2, w3, w_cross   — FFN + cross-gate projections (no bias)
//   w_norm_{attn_in,attn_out,ffn_in,ffn_out} — RMSNorm scales [D], init=1
//   gate_attn, gate_ffn   — residual blend gates [D], init=0.5
//
// Usage
// ─────
//   #include <torch/nn/modules/ohmcdualgateblock.h>
//
//   dm::prim::OhmCDualGateBlock blk(512, 8);    // dim=512, heads=8
//   auto y = blk(x);                            // x,y: [B, T, 512]
//
//   // GQA (4 KV heads for 8 query heads):
//   dm::prim::OhmCDualGateBlockOptions opts(512, 8);
//   opts.n_kv_heads(4).ffn_dim(1408).causal(true);
//   dm::prim::OhmCDualGateBlock blk2(opts);
//
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/nn/module.h>
#include <torch/nn/modules/dropout.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>
#include <torch/csrc/Export.h>

#include <cmath>
#include <ostream>
#include <stdexcept>

namespace dm {
namespace prim {

// ─────────────────────────────────────────────────────────────────────────────
// Internal RMSNorm helper (header-local, not exported)
//
// Promotes x to float32, computes per-token RMS, normalises, casts back,
// then applies a learned per-channel scale w ∈ ℝ^D.
//
//   out = (x / sqrt(mean(x²) + ε)) * w,   computed in float32
// ─────────────────────────────────────────────────────────────────────────────
namespace {

inline torch::Tensor _dgb_rmsnorm(
    const torch::Tensor& x,
    const torch::Tensor& w,
    double eps)
{
    // Up-cast to float for numerical stability, then cast result back to
    // original dtype so the block is compatible with bf16/fp16 training.
    auto xf  = x.to(torch::kFloat);
    auto rms = xf.pow(2).mean(/*dim=*/-1, /*keepdim=*/true).add(eps).rsqrt();
    return (xf * rms).type_as(x) * w;   // w broadcasts over [B,T,D]
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Options
// ─────────────────────────────────────────────────────────────────────────────

struct OhmCDualGateBlockOptions {
    /// @param dim      Embedding/hidden dimension D.
    /// @param n_heads  Number of query attention heads H.
    OhmCDualGateBlockOptions(int64_t dim, int64_t n_heads)
        : dim_(dim), n_heads_(n_heads),
          n_kv_heads_(n_heads),            // default: full MHA (no GQA)
          ffn_dim_(4 * dim),               // default: 4× expansion
          norm_eps_(1e-6),
          dropout_(0.0),
          causal_(true) {}

    /// Total embedding dimension.
    TORCH_ARG(int64_t, dim);

    /// Number of query heads.  dim % n_heads == 0 required.
    TORCH_ARG(int64_t, n_heads);

    /// Number of KV heads (GQA).  n_heads % n_kv_heads == 0 required.
    /// Default = n_heads (standard MHA — no GQA reduction).
    TORCH_ARG(int64_t, n_kv_heads);

    /// SwiGLU inner dimension F.  Default = 4 * dim.
    TORCH_ARG(int64_t, ffn_dim);

    /// RMSNorm epsilon.  Default = 1e-6.
    TORCH_ARG(double, norm_eps);

    /// Attention-weight dropout probability.  Default = 0.0.
    TORCH_ARG(double, dropout);

    /// If true, apply upper-triangular causal mask.  Default = true.
    TORCH_ARG(bool, causal);

    // ── Derived helpers ──────────────────────────────────────────────────────
    // TORCH_ARG ends each field in private scope; re-open public explicitly.
 public:
    /// Per-head dimension  hd = dim / n_heads.
    int64_t head_dim()  const { return dim() / n_heads(); }

    /// GQA repeat factor  n_rep = n_heads / n_kv_heads.
    int64_t n_rep()     const { return n_heads() / n_kv_heads(); }
};

// ─────────────────────────────────────────────────────────────────────────────
// Implementation class
// ─────────────────────────────────────────────────────────────────────────────

class TORCH_API OhmCDualGateBlockImpl : public torch::nn::Module {
 public:
  /// Convenience constructor — uses defaults for all hyper-parameters.
  explicit OhmCDualGateBlockImpl(int64_t dim, int64_t n_heads)
      : OhmCDualGateBlockImpl(OhmCDualGateBlockOptions(dim, n_heads)) {}

  /// Full constructor.
  explicit OhmCDualGateBlockImpl(OhmCDualGateBlockOptions options_);

  /// Re-register all sub-modules and parameters with current options.
  void reset();

  /// Describe the block in a human-readable string.
  void pretty_print(std::ostream& stream) const override;

  /// Forward pass.
  ///
  /// @param x  Input tensor [B, T, dim].
  /// @returns  Output tensor [B, T, dim] with the same shape as input.
  ///
  /// Computational graph is preserved end-to-end so AdamW / L-BFGS can
  /// backpropagate through both the attention and the cross-gated FFN path.
  torch::Tensor forward(const torch::Tensor& x);

  /// The options used to construct this block.
  OhmCDualGateBlockOptions options;

  // ── PATH 1: Attention projections ────────────────────────────────────────
  torch::nn::Linear wq{nullptr};  ///< Q: [dim → dim]           no bias
  torch::nn::Linear wk{nullptr};  ///< K: [dim → n_kv_heads*hd] no bias
  torch::nn::Linear wv{nullptr};  ///< V: [dim → n_kv_heads*hd] no bias
  torch::nn::Linear wo{nullptr};  ///< output: [dim → dim]       no bias
  torch::nn::Dropout attn_drop{nullptr};

  // ── PATH 2: FFN (SwiGLU) projections ─────────────────────────────────────
  torch::nn::Linear w1{nullptr};  ///< SwiGLU gate: [dim → ffn_dim]  no bias
  torch::nn::Linear w3{nullptr};  ///< SwiGLU pass: [dim → ffn_dim]  no bias
  torch::nn::Linear w2{nullptr};  ///< down-proj:   [ffn_dim → dim]  no bias

  // ── CROSS-GATE projection ─────────────────────────────────────────────────
  torch::nn::Linear w_cross{nullptr}; ///< [dim → ffn_dim]  no bias
  //   Projects attn_out into ffn_dim space for the Hadamard product with
  //   ffn_gate.  Dedicated weight avoids aliasing w1 or w3.

  // ── RMSNorm scale vectors (registered as parameters, [dim], init=ones) ───
  torch::Tensor w_norm_attn_in;   ///< pre-attention norm scale
  torch::Tensor w_norm_attn_out;  ///< post-attention norm scale
  torch::Tensor w_norm_ffn_in;    ///< pre-FFN norm scale
  torch::Tensor w_norm_ffn_out;   ///< post-FFN norm scale

  // ── Dual residual gate vectors ([dim], init=0.5) ──────────────────────────
  torch::Tensor gate_attn;  ///< per-channel blend weight for attn_out
  torch::Tensor gate_ffn;   ///< per-channel blend weight for ffn_out

 private:
  int64_t head_dim_;  ///< cached hd = dim / n_heads
  int64_t n_rep_;     ///< cached GQA repeat factor
};

TORCH_MODULE(OhmCDualGateBlock);

// ─────────────────────────────────────────────────────────────────────────────
// Inline implementations
// ─────────────────────────────────────────────────────────────────────────────

inline OhmCDualGateBlockImpl::OhmCDualGateBlockImpl(
    OhmCDualGateBlockOptions options_)
    : options(std::move(options_))
{
  if (options.dim() % options.n_heads() != 0)
    throw std::invalid_argument(
        "OhmCDualGateBlock: dim must be divisible by n_heads");
  if (options.n_heads() % options.n_kv_heads() != 0)
    throw std::invalid_argument(
        "OhmCDualGateBlock: n_heads must be divisible by n_kv_heads");

  head_dim_ = options.head_dim();
  n_rep_    = options.n_rep();
  reset();
}

inline void OhmCDualGateBlockImpl::reset() {
  const int64_t D   = options.dim();
  const int64_t F   = options.ffn_dim();
  const int64_t Hkv = options.n_kv_heads();
  const int64_t kv_dim = Hkv * head_dim_;   // KV projection output width

  // ── Attention projections ──────────────────────────────────────────────────
  wq = register_module("wq",
      torch::nn::Linear(torch::nn::LinearOptions(D, D     ).bias(false)));
  wk = register_module("wk",
      torch::nn::Linear(torch::nn::LinearOptions(D, kv_dim).bias(false)));
  wv = register_module("wv",
      torch::nn::Linear(torch::nn::LinearOptions(D, kv_dim).bias(false)));
  wo = register_module("wo",
      torch::nn::Linear(torch::nn::LinearOptions(D, D     ).bias(false)));

  if (options.dropout() > 0.0)
    attn_drop = register_module("attn_drop",
        torch::nn::Dropout(torch::nn::DropoutOptions(options.dropout())));

  // ── FFN (SwiGLU) projections ───────────────────────────────────────────────
  w1 = register_module("w1",
      torch::nn::Linear(torch::nn::LinearOptions(D, F).bias(false)));
  w3 = register_module("w3",
      torch::nn::Linear(torch::nn::LinearOptions(D, F).bias(false)));
  w2 = register_module("w2",
      torch::nn::Linear(torch::nn::LinearOptions(F, D).bias(false)));

  // ── Cross-gate projection ──────────────────────────────────────────────────
  w_cross = register_module("w_cross",
      torch::nn::Linear(torch::nn::LinearOptions(D, F).bias(false)));

  // ── RMSNorm scale parameters ([D], init=ones) ──────────────────────────────
  w_norm_attn_in  = register_parameter("w_norm_attn_in",  torch::ones(D));
  w_norm_attn_out = register_parameter("w_norm_attn_out", torch::ones(D));
  w_norm_ffn_in   = register_parameter("w_norm_ffn_in",   torch::ones(D));
  w_norm_ffn_out  = register_parameter("w_norm_ffn_out",  torch::ones(D));

  // ── Dual gate parameters ([D], init=0.5) ───────────────────────────────────
  // Init at 0.5 so that, at construction, each path contributes half-strength
  // to the residual stream — a balanced starting point for gradient descent.
  gate_attn = register_parameter("gate_attn", torch::full({D}, 0.5f));
  gate_ffn  = register_parameter("gate_ffn",  torch::full({D}, 0.5f));
}

inline void OhmCDualGateBlockImpl::pretty_print(std::ostream& stream) const {
  stream << "dm::prim::OhmCDualGateBlock("
         << "dim="       << options.dim()
         << ", n_heads=" << options.n_heads()
         << ", n_kv_heads=" << options.n_kv_heads()
         << ", ffn_dim=" << options.ffn_dim()
         << ", causal="  << (options.causal() ? "true" : "false")
         << ", dropout=" << options.dropout()
         << ")";
}

inline torch::Tensor OhmCDualGateBlockImpl::forward(const torch::Tensor& x) {
  const int64_t B  = x.size(0);
  const int64_t T  = x.size(1);
  const int64_t H  = options.n_heads();
  const int64_t Hkv = options.n_kv_heads();
  const double  eps = options.norm_eps();
  const float   scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));

  // ── PATH 1: Attention ──────────────────────────────────────────────────────
  //
  // Pre-attention RMSNorm — both paths branch from the *same* raw x.
  auto norm_a = _dgb_rmsnorm(x, w_norm_attn_in, eps);   // [B,T,D]

  // Project and reshape into [B, H, T, hd] head layout.
  // wk/wv output n_kv_heads * head_dim for GQA; wq always full dim.
  auto Q = wq(norm_a).view({B, T, H,   head_dim_}).transpose(1, 2);  // [B,H, T,hd]
  auto K = wk(norm_a).view({B, T, Hkv, head_dim_}).transpose(1, 2);  // [B,Hkv,T,hd]
  auto V = wv(norm_a).view({B, T, Hkv, head_dim_}).transpose(1, 2);  // [B,Hkv,T,hd]

  // GQA expansion: repeat-interleave KV heads to match Q head count.
  // [B, Hkv, T, hd] → [B, H, T, hd]  (no-op when n_rep==1 i.e. standard MHA)
  if (n_rep_ > 1) {
    K = K.unsqueeze(2)
         .expand({B, Hkv, n_rep_, T, head_dim_})
         .contiguous()
         .view({B, H, T, head_dim_});
    V = V.unsqueeze(2)
         .expand({B, Hkv, n_rep_, T, head_dim_})
         .contiguous()
         .view({B, H, T, head_dim_});
  }

  // Scaled dot-product attention scores: [B, H, T, T]
  //   S = Q Kᵀ / sqrt(hd)
  auto scores = torch::matmul(Q, K.transpose(-2, -1)) * scale;  // [B,H,T,T]

  // Optional causal mask — mask[i,j]=true when j>i (future tokens).
  if (options.causal()) {
    auto causal_mask = torch::ones({T, T}, x.options())
                           .triu(/*diagonal=*/1)
                           .to(torch::kBool);
    scores = scores.masked_fill(
        causal_mask.unsqueeze(0).unsqueeze(0), -1e9f);
  }

  // Normalise to probability simplex over key axis.
  auto attn_w = torch::softmax(scores, /*dim=*/-1);   // [B,H,T,T]
  if (!attn_drop.is_empty() && is_training())
    attn_w = attn_drop(attn_w);

  // Aggregate values, collapse heads → [B,T,D].
  //   ctx = (attn_w · V).T(1,2).reshape(B,T,D)
  auto ctx = torch::matmul(attn_w, V)               // [B,H,T,hd]
                 .transpose(1, 2)                    // [B,T,H,hd]
                 .contiguous()
                 .view({B, T, H * head_dim_});        // [B,T,D]

  // Output projection + post-attention RMSNorm → attn_out [B,T,D].
  auto attn_out = _dgb_rmsnorm(wo(ctx), w_norm_attn_out, eps);

  // ── PATH 2: SwiGLU FFN (runs in parallel on the same x) ──────────────────
  //
  // Pre-FFN RMSNorm — separate learned scale from Path 1's norm.
  auto norm_f = _dgb_rmsnorm(x, w_norm_ffn_in, eps);  // [B,T,D]

  // SwiGLU gating state: gate ⊙ linear  =  silu(w1(·)) ⊙ w3(·)
  //   Shape: [B, T, ffn_dim]
  auto ffn_gate = torch::silu(w1(norm_f)) * w3(norm_f);

  // ── CROSS-GATE FUSION ─────────────────────────────────────────────────────
  //
  // Project attn_out into ffn_dim space so dimensions align with ffn_gate
  // for the Hadamard product.  This dedicated projection (w_cross) avoids
  // reusing w1/w3 which would create gradient coupling between paths.
  //
  //   attn_proj  = w_cross(attn_out)             [B, T, ffn_dim]
  //   gated_mid  = ffn_gate ⊙ attn_proj          [B, T, ffn_dim]
  //                ↑ spatial context from attention modulates the SwiGLU gate
  auto attn_proj  = w_cross(attn_out);            // [B,T,F]
  auto gated_mid  = ffn_gate * attn_proj;          // [B,T,F]  Hadamard

  // Down-project and apply post-FFN RMSNorm → ffn_out [B,T,D].
  auto ffn_out = _dgb_rmsnorm(w2(gated_mid), w_norm_ffn_out, eps);

  // ── DUAL GATED RESIDUAL FUSION ────────────────────────────────────────────
  //
  // Long-skip connection from raw input x, modulated by two learnable
  // per-channel gate vectors.  Both gate_attn and gate_ffn broadcast over
  // [B, T, D] without any additional allocations.
  //
  //   output = x  +  attn_out * gate_attn  +  ffn_out * gate_ffn
  return x + attn_out * gate_attn + ffn_out * gate_ffn;
}

}  // namespace prim
}  // namespace dm
