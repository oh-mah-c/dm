#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
// Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell,
// Quoc V. Le, Ruslan Salakhutdinov
// ACL 2019 — https://aclanthology.org/P19-1285
//
// ── Key contributions ────────────────────────────────────────────────────────
// 1. Segment-level recurrence with state reuse (§3.2)
//    Previous segment's hidden states are cached (stop-gradient) and
//    concatenated to the current segment's hidden states before attention.
//    Effective context length grows as O(N × L) with N layers, L seg length.
//
// 2. Relative positional encoding (§3.3)
//    Instead of absolute position embeddings, relative distances are injected
//    into every attention score via sinusoidal matrix R and two global
//    trainable bias vectors u (content) and v (position).
//    Attention score decomposition (single head):
//      A_{i,j} = (a) E_xi Wq^T Wk,E E_xj   -- content-based key
//              + (b) E_xi Wq^T Wk,R R_{i-j} -- content-based position
//              + (c) u^T Wk,E E_xj           -- global content bias
//              + (d) v^T Wk,R R_{i-j}        -- global position bias
//
// ── Architecture ─────────────────────────────────────────────────────────────
//   N-layer Transformer with pre-norm LayerNorm + residual.
//   Each layer has:
//     • Multi-head relative attention  (§3.3)
//     • Positionwise feed-forward      (§3, same as vanilla Transformer)
//   No absolute positional encoding is added to the token embeddings.
//
// ── Model sizes (paper, Table 1 / Table 2) ───────────────────────────────────
//   small  (12L, WikiText-103 151M):  d=512,  n_heads=8,  d_head=64, d_ff=2048
//   base   (12L, enwiki8 41M):        d=410,  n_heads=10, d_head=41, d_ff=2100
//   large  (18L, enwiki8 88M):        d=512,  n_heads=8,  d_head=64, d_ff=2048
//   xl     (24L, enwiki8 277M):       d=1024, n_heads=16, d_head=64, d_ff=4096
//
// ── Training ─────────────────────────────────────────────────────────────────
//   Optimizer : Adam, lr=2.5e-4, clip=0.25
//   Scheduler : linear warmup (default 0), cosine or constant decay
//   Memory    : M = L (segment length) during training, up to 6L at eval
//   Dropout   : applied after attention softmax and after FFN
//
// ── Inference ────────────────────────────────────────────────────────────────
//   At each segment step, feed new segment, update memory banks.
//   Memory can be extended during eval for longer effective context.
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <vector>
#include <string>
#include <cstdint>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────
struct TransformerXLConfig {
    int64_t vocab_size  = 10000;  // vocabulary size
    int64_t d_model     = 512;    // hidden dimension
    int64_t n_heads     = 8;      // number of attention heads
    int64_t d_head      = 64;     // head dimension (d_model / n_heads normally)
    int64_t d_inner     = 2048;   // FFN inner dimension
    int64_t n_layers    = 12;     // number of Transformer-XL layers
    int64_t seg_len     = 128;    // segment (context) length during training
    int64_t mem_len     = 128;    // memory length (cached segments)
    float   dropout     = 0.1f;   // general dropout
    float   attn_drop   = 0.0f;   // attention weight dropout
    bool    tie_weights = true;   // tie input embedding with output projection

    // ── Named size presets ────────────────────────────────────────────────
    // Small:  12L, 151M params, WikiText-103 word-level (Table 1)
    static TransformerXLConfig small() {
        return {267735, 512, 8, 64, 2048, 12, 384, 384, 0.1f, 0.0f, true};
    }
    // Base: 12L, 41M params, enwiki8 char-level (Table 2)
    static TransformerXLConfig base() {
        return {200, 410, 10, 41, 2100, 12, 512, 512, 0.1f, 0.0f, true};
    }
    // Large: 18L, 88M params, enwiki8 (Table 2)
    static TransformerXLConfig large() {
        return {200, 512, 8, 64, 2048, 18, 512, 512, 0.1f, 0.0f, true};
    }
    // XL: 24L, 277M params, enwiki8 (Table 2)
    static TransformerXLConfig xl() {
        return {200, 1024, 16, 64, 4096, 24, 512, 1024, 0.1f, 0.0f, true};
    }
    // Tiny — fast unit-test config
    static TransformerXLConfig tiny() {
        return {1000, 64, 4, 16, 128, 2, 32, 32, 0.0f, 0.0f, true};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Sinusoidal relative positional encoding table
// Returns R ∈ R^{(seg_len + mem_len) × d_model}  (§3.3)
// R_i encodes relative distance i; i=0 is the most recent position.
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor make_rel_pos_encoding(int64_t klen, int64_t d_model,
                                    torch::Device device);

// ─────────────────────────────────────────────────────────────────────────────
// Multi-head relative attention (§3.3)
//
// Each head computes:
//   A_{i,j} = q_i Wk,E e_j  +  q_i Wk,R r_{i-j}
//            + u Wk,E e_j   +  v Wk,R r_{i-j}
//
// where q_i = h_i Wq,  e_j is the key embedding (current + memory),
//       r_{i-j} is the sinusoid row for relative distance i-j,
//       u, v are global learned bias vectors (one per head).
// ─────────────────────────────────────────────────────────────────────────────
struct TXLRelAttnImpl : torch::nn::Module {
    explicit TXLRelAttnImpl(const TransformerXLConfig& cfg);

    // h_cur:  [B, q_len, d_model]   — current segment hidden
    // h_mem:  [B, m_len, d_model]   — cached memory (may be empty)
    // r:      [klen, d_model]        — sinusoid table for relative distances
    // attn_mask: [q_len, klen] bool  — true = masked out (causal)
    torch::Tensor forward(torch::Tensor h_cur,
                          torch::Tensor h_mem,
                          torch::Tensor r,
                          torch::Tensor attn_mask);

    // projections — separate content key (Wk,E) and position key (Wk,R)
    torch::nn::Linear Wq{nullptr}, Wk_E{nullptr}, Wk_R{nullptr}, Wv{nullptr};
    torch::nn::Linear Wo{nullptr};  // output projection
    torch::nn::Dropout attn_drop_layer{nullptr};

    // global bias vectors u (content) and v (position) — (n_heads, d_head)
    torch::Tensor u, v;

    int64_t n_heads, d_head, d_model;

private:
    // Relative shift: efficiently computes all relative position scores
    // using the trick from Appendix B of the paper.
    torch::Tensor rel_shift(torch::Tensor x);
};
TORCH_MODULE(TXLRelAttn);

// ─────────────────────────────────────────────────────────────────────────────
// Positionwise feed-forward (identical to vanilla Transformer)
// FFN(x) = dropout(relu(x W1 + b1)) W2 + b2
// ─────────────────────────────────────────────────────────────────────────────
struct TXLFFNImpl : torch::nn::Module {
    explicit TXLFFNImpl(const TransformerXLConfig& cfg);
    torch::Tensor forward(torch::Tensor x);

    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::Dropout drop{nullptr};
};
TORCH_MODULE(TXLFFN);

// ─────────────────────────────────────────────────────────────────────────────
// Single Transformer-XL layer (§3, eq. block)
//
// h_τ+1^n = LayerNorm( o_τ^n )  (post-attn residual)
// h_τ+1^n = LayerNorm( FFN(h) + h )
//
// (pre-norm variant used here for training stability — same as most
// modern implementations; both pre- and post-norm produce similar results)
// ─────────────────────────────────────────────────────────────────────────────
struct TXLLayerImpl : torch::nn::Module {
    explicit TXLLayerImpl(const TransformerXLConfig& cfg);

    // h_cur:   [B, q_len, d_model]
    // h_mem:   [B, m_len, d_model]  (empty tensor if first segment)
    // r:       [klen, d_model]
    // mask:    [q_len, klen] bool
    // returns: new hidden [B, q_len, d_model]
    torch::Tensor forward(torch::Tensor h_cur,
                          torch::Tensor h_mem,
                          torch::Tensor r,
                          torch::Tensor attn_mask);

    TXLRelAttn  attn{nullptr};
    TXLFFN      ffn{nullptr};
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
    torch::nn::Dropout   drop{nullptr};
};
TORCH_MODULE(TXLLayer);

// ─────────────────────────────────────────────────────────────────────────────
// Full Transformer-XL model
//
// Maintains a memory bank: one tensor per layer, shape [B, mem_len, d_model].
// Call reset_memory() at the start of each new document.
// ─────────────────────────────────────────────────────────────────────────────
struct TransformerXLModelImpl : torch::nn::Module {
    explicit TransformerXLModelImpl(const TransformerXLConfig& cfg);

    // Forward over one segment.
    // tokens: [B, seg_len] int64
    // Returns logits: [B, seg_len, vocab_size]
    // Side effect: updates internal memory banks.
    torch::Tensor forward(torch::Tensor tokens);

    // Clear all cached memories (call between documents)
    void reset_memory();

    // Access current memory for layer n (0-indexed)
    const torch::Tensor& memory(int64_t n) const { return mems_[n]; }

    TransformerXLConfig             cfg;
    torch::nn::Embedding            embedding{nullptr};
    torch::nn::ModuleList           layers{nullptr};
    torch::nn::LayerNorm            norm_out{nullptr};
    torch::nn::Linear               output{nullptr};
    torch::nn::Dropout              drop_emb{nullptr};

private:
    // mems_[n]: [B, mem_len, d_model] — cached hidden states for layer n
    std::vector<torch::Tensor>      mems_;

    // Build causal mask for [q_len × klen] where klen = q_len + m_len
    torch::Tensor causal_mask(int64_t q_len, int64_t klen, torch::Device dev);

    // Update memory for layer n with new hidden states
    void update_memory(int64_t n, torch::Tensor new_h);
};
TORCH_MODULE(TransformerXLModel);

// ─────────────────────────────────────────────────────────────────────────────
// Training utilities
// ─────────────────────────────────────────────────────────────────────────────
struct TransformerXLTrainConfig {
    double  lr          = 2.5e-4;
    double  clip        = 0.25;
    int64_t warmup      = 0;       // linear warmup steps
    int64_t total_steps = 400000;
    int64_t eval_mem    = -1;      // -1 = same as mem_len; else expand at eval
};

// Returns Adam optimizer (paper uses Adam, not AdamW for base results)
torch::optim::Adam make_txl_optimizer(TransformerXLModel& model,
                                      const TransformerXLTrainConfig& tcfg);

// LR schedule: linear warmup then constant (simple version used in paper)
double txl_lr_schedule(int64_t step, const TransformerXLTrainConfig& tcfg);

// Single training step.
// tokens: [B, seg_len+1] — input is tokens[0..T-1], target is tokens[1..T]
// Resets memory if new_doc=true.
// Returns (loss_scalar, nll_per_token).
std::pair<torch::Tensor, double>
txl_train_step(TransformerXLModel& model,
               torch::optim::Adam& optimizer,
               torch::Tensor tokens,
               const TransformerXLTrainConfig& tcfg,
               int64_t step,
               bool new_doc = false);

} // namespace nlp
} // namespace models
} // namespace dm
