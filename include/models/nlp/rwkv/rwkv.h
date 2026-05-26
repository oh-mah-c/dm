#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// RWKV: Reinventing RNNs for the Transformer Era
// Bo Peng et al., arXiv:2305.13048v2, 2023
// https://arxiv.org/abs/2305.13048
//
// ── Core idea ────────────────────────────────────────────────────────────────
// RWKV combines Transformer parallelism with RNN inference efficiency.
// Named after its four learnable vectors per block:
//   R (Receptance) — receiver of past information
//   W (Weight)     — channel-wise time-decay, trainable, non-negative → e^{-w}≤1
//   K (Key)        — analogous to K in attention
//   V (Value)      — analogous to V in attention
//
// ── WKV operator (§3.1.2, eq.16) ────────────────────────────────────────────
// Replaces softmax attention with an AFT-style linear attention:
//
//   wkv_t = [Σ_{i<t} e^{-(t-1-i)w+k_i} ⊙ v_i  +  e^{u+k_t} ⊙ v_t]
//           / [Σ_{i<t} e^{-(t-1-i)w+k_i}        +  e^{u+k_t}       ]
//
// where w ∈ R^D (time-decay, non-negative) and u ∈ R^D ("bonus"/direct token
// weight). The ⊙ is element-wise multiplication (channel-independent).
//
// ── RNN recurrence (App. D, eqs. 19–28) — numerically stable form ───────────
// State: (a', b', p) per layer, each ∈ R^D
//   a'_0 = 0, b'_0 = 0, p_0 = -∞
// At each step t (sharing exponent p for numerical precision):
//   wkv_t = (e^{p_{t-1}-q} a'_{t-1} + e^{u+k_t-q} v_t)
//           / (e^{p_{t-1}-q} b'_{t-1} + e^{u+k_t-q})
//     where q = max(p_{t-1}, u+k_t)
//   q' = max(p_{t-1}-w, k_t)
//   a'_t = e^{p_{t-1}-w-q'} a'_{t-1} + e^{k_t-q'} v_t
//   b'_t = e^{p_{t-1}-w-q'} b'_{t-1} + e^{k_t-q'}
//   p_t  = q'
//
// ── Token shift (§3.1.1) ─────────────────────────────────────────────────────
// All RWKV inputs are linearly interpolated with the previous timestep:
//   r_t = W_r (μ_r ⊙ x_t + (1−μ_r) ⊙ x_{t-1})
//   k_t = W_k (μ_k ⊙ x_t + (1−μ_k) ⊙ x_{t-1})
//   v_t = W_v (μ_v ⊙ x_t + (1−μ_v) ⊙ x_{t-1})
// Implemented as a one-position causal shift (ZeroPad2d in training mode).
//
// ── Channel mixing (§3.1, eq.18) ─────────────────────────────────────────────
// FFN-like sub-block using squared-ReLU:
//   r'_t = W_r' (μ_r' ⊙ x_t + (1−μ_r') ⊙ x_{t-1})
//   k'_t = W_k' (μ_k' ⊙ x_t + (1−μ_k') ⊙ x_{t-1})
//   o'_t = σ(r'_t) ⊙ (W_v' · max(k'_t, 0)²)
//
// ── Full block ───────────────────────────────────────────────────────────────
//   x ← x + TimeMix(LayerNorm(x))
//   x ← x + ChannelMix(LayerNorm(x))
//
// ── Model sizes (Table 2, §4) ─────────────────────────────────────────────────
//   169M : 12L, D=768   (d_ff = 4*D = 3072)
//   430M : 24L, D=1024
//   1.5B : 24L, D=2048
//   3B   : 32L, D=2560
//   7B   : 32L, D=4096
//   14B  : 40L, D=5120
//
// ── Training (§4.1, App. G) ──────────────────────────────────────────────────
//   Optimizer : Adam, ε=(0.9, 0.99), no weight decay
//   LR        : constant warmup → exponential decay
//   Context   : 1024 tokens
//   Embedding : small init (U(±1e-4)) + post-embed LayerNorm (§3.4)
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <vector>
#include <string>
#include <cstdint>
#include <tuple>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────
struct RWKVConfig {
    int64_t vocab_size = 50277;  // GPT-NeoX tokenizer default
    int64_t d_model    = 512;    // model dimension D  (paper uses d=4s, s=D/4)
    int64_t n_layers   = 12;     // number of RWKV blocks L
    int64_t ctx_len    = 1024;   // training context length

    // d_ff = 4*d_model  (paper uses 4× expansion in channel mixing)
    int64_t d_ff() const { return 4 * d_model; }

    // ── Named presets (Table 2) ───────────────────────────────────────────
    static RWKVConfig tiny() {
        return {1000, 128, 2, 256};
    }
    static RWKVConfig r169m() {
        return {50277, 768,  12, 1024};
    }
    static RWKVConfig r430m() {
        return {50277, 1024, 24, 1024};
    }
    static RWKVConfig r1p5b() {
        return {50277, 2048, 24, 1024};
    }
    static RWKVConfig r3b() {
        return {50277, 2560, 32, 1024};
    }
    static RWKVConfig r7b() {
        return {50277, 4096, 32, 1024};
    }
    static RWKVConfig r14b() {
        return {50277, 5120, 40, 1024};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Time-mixing block (§3.1, §3.1.1, §3.1.2)
//
// Training:   processes full sequence [B, T, D] in parallel using
//             cumulative WKV scan (O(BT d) serial scan, parallelisable
//             along B and D dimensions).
// Inference:  step() — O(d) constant cost per token; state = (a', b', p).
// ─────────────────────────────────────────────────────────────────────────────
struct RWKVTimeMixImpl : torch::nn::Module {
    RWKVTimeMixImpl(const RWKVConfig& cfg, int64_t layer_id);

    // Training forward.  x: [B, T, D]  →  out: [B, T, D]
    // x_prev: shifted-by-1 input for token shift (passed externally so
    // block can share it with channel-mix); if empty, uses zero padding.
    torch::Tensor forward(torch::Tensor x, torch::Tensor x_prev);

    // Inference step — one token at a time.
    // x_t:  [B, D]    current input
    // x_tm1:[B, D]    previous input (token-shift state)
    // aa:   [B, D]    WKV numerator state a'
    // bb:   [B, D]    WKV denominator state b'
    // pp:   [B, D]    WKV exponent state p
    // Updates aa, bb, pp in-place.  Returns output [B, D].
    torch::Tensor step(torch::Tensor x_t,
                       torch::Tensor x_tm1,
                       torch::Tensor& aa,
                       torch::Tensor& bb,
                       torch::Tensor& pp);

    // Token-shift interpolation parameters μ_r, μ_k, μ_v
    torch::Tensor mu_r, mu_k, mu_v;

    // Time-decay w and bonus u  (both ∈ R^D)
    torch::Tensor time_decay;   // w, non-negative after parameterization
    torch::Tensor time_first;   // u

    // Linear projections (no bias, §E)
    torch::nn::Linear Wr{nullptr}, Wk{nullptr}, Wv{nullptr}, Wo{nullptr};

    int64_t d_model;

private:
    // WKV sequential scan for one sequence position
    // Returns [B, D] given current k_t, v_t and running state
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
    wkv_step(torch::Tensor k_t, torch::Tensor v_t,
             torch::Tensor aa, torch::Tensor bb, torch::Tensor pp);
};
TORCH_MODULE(RWKVTimeMix);

// ─────────────────────────────────────────────────────────────────────────────
// Channel-mixing block (§3.1, eq.18)
// FFN-like using squared-ReLU activation.
// ─────────────────────────────────────────────────────────────────────────────
struct RWKVChannelMixImpl : torch::nn::Module {
    RWKVChannelMixImpl(const RWKVConfig& cfg, int64_t layer_id);

    // x: [B, T, D],  x_prev: [B, T, D]  →  [B, T, D]
    torch::Tensor forward(torch::Tensor x, torch::Tensor x_prev);

    // Inference step.  x_t: [B,D], x_tm1: [B,D]  →  [B,D]
    torch::Tensor step(torch::Tensor x_t, torch::Tensor x_tm1);

    // Token-shift interpolation for R', K'
    torch::Tensor mu_r, mu_k;

    // Projections
    torch::nn::Linear Wr{nullptr}, Wk{nullptr}, Wv{nullptr};

    int64_t d_model, d_ff;
};
TORCH_MODULE(RWKVChannelMix);

// ─────────────────────────────────────────────────────────────────────────────
// Single RWKV residual block
// ─────────────────────────────────────────────────────────────────────────────
struct RWKVBlockImpl : torch::nn::Module {
    RWKVBlockImpl(const RWKVConfig& cfg, int64_t layer_id);

    // x: [B, T, D]  →  [B, T, D]
    torch::Tensor forward(torch::Tensor x);

    // Inference step.
    // x_t: [B,D]  current token embedding
    // State: (x_tm_tm, x_cm_tm, aa, bb, pp) — all [B,D]
    //   x_tm_tm: previous x entering time-mix (token-shift)
    //   x_cm_tm: previous x entering channel-mix (token-shift)
    //   aa, bb, pp: WKV state
    // Returns (out [B,D], updated state)
    torch::Tensor step(torch::Tensor x_t,
                       torch::Tensor& x_tm_tm,
                       torch::Tensor& x_cm_tm,
                       torch::Tensor& aa,
                       torch::Tensor& bb,
                       torch::Tensor& pp);

    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    RWKVTimeMix    time_mix{nullptr};
    RWKVChannelMix chan_mix{nullptr};
};
TORCH_MODULE(RWKVBlock);

// ─────────────────────────────────────────────────────────────────────────────
// Full RWKV language model
// ─────────────────────────────────────────────────────────────────────────────
struct RWKVModelImpl : torch::nn::Module {
    explicit RWKVModelImpl(const RWKVConfig& cfg);

    // tokens: [B, T] int64  →  logits: [B, T, vocab_size]
    torch::Tensor forward(torch::Tensor tokens);

    // Autoregressive generation; returns [B, max_new] new tokens
    torch::Tensor generate(torch::Tensor prompt,
                           int64_t max_new = 100,
                           double temperature = 1.0,
                           double top_p = 0.9);

    RWKVConfig             cfg;
    torch::nn::Embedding   embedding{nullptr};
    torch::nn::LayerNorm   emb_ln{nullptr};    // post-embed LN (§3.4 small init)
    torch::nn::ModuleList  blocks{nullptr};
    torch::nn::LayerNorm   ln_out{nullptr};
    torch::nn::Linear      head{nullptr};
};
TORCH_MODULE(RWKVModel);

// ─────────────────────────────────────────────────────────────────────────────
// Training utilities
// ─────────────────────────────────────────────────────────────────────────────
struct RWKVTrainConfig {
    double  lr         = 6e-4;   // initial LR (Table 3: 169M uses 6e-4)
    double  end_lr     = 1e-5;   // final LR after exponential decay
    double  beta1      = 0.9;
    double  beta2      = 0.99;   // Adam ε₂=0.99 (§4.1)
    int64_t warmup     = 0;
    int64_t total_steps= 10000;
};

torch::optim::Adam make_rwkv_optimizer(RWKVModel& model,
                                       const RWKVTrainConfig& tcfg);

// Exponential LR decay from lr → end_lr over total_steps
double rwkv_lr_schedule(int64_t step, const RWKVTrainConfig& tcfg);

// Single training step.  tokens: [B, T+1]
std::pair<torch::Tensor, double>
rwkv_train_step(RWKVModel& model,
                torch::optim::Adam& optimizer,
                torch::Tensor tokens,
                const RWKVTrainConfig& tcfg,
                int64_t step);

} // namespace nlp
} // namespace models
} // namespace dm
