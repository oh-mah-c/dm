#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Llama 3: The Llama 3 Herd of Models
// Llama Team, AI @ Meta, arXiv:2407.21783v3, 2024
// https://arxiv.org/abs/2407.21783
//
// ── Architecture (§3.2, Table 3) ─────────────────────────────────────────────
// Dense decoder-only Transformer with:
//   • RMSNorm (pre-norm)                                              (§3.2)
//   • RoPE positional embeddings, θ = 500,000 (increased from 10k)   (§3.2)
//   • Grouped-Query Attention: 8 KV heads for ALL model sizes         (§3.2)
//   • SwiGLU feed-forward: FFN(x) = W2(SiLU(W1·x) ⊙ W3·x)           (Table 3)
//   • No bias in attention/FFN layers
//   • Vocabulary: 128,000 tokens (tiktoken BPE + 28K non-English)     (§3.2)
//
// ── Model sizes (Table 3) ─────────────────────────────────────────────────────
//   8B : dim=4096,  n_layers=32,  n_heads=32,  n_kv_heads=8, ffn=14336
//  70B : dim=8192,  n_layers=80,  n_heads=64,  n_kv_heads=8, ffn=28672
// 405B : dim=16384, n_layers=126, n_heads=128, n_kv_heads=8, ffn=53248
//
// ── Training recipe (§3.4) ───────────────────────────────────────────────────
//   Optimizer : AdamW (β=(0.9,0.95), ε=1e-5, weight_decay=0.1)
//   Scheduler : cosine with linear warmup (8000 steps) → decay to 1e-5
//   Peak LR   : 8B → 3e-4, 70B → 1.5e-4, 405B → 8e-5
//   Grad clip : 1.0
//   Tokens    : 15.6T pre-training + 800B long-context stage (128K)
//
// ── Key differences from Llama 2 ─────────────────────────────────────────────
//   • RoPE θ=500k (Llama 2 used 10k)     → longer context support (§3.2)
//   • GQA on ALL sizes (Llama 2 only 70B had GQA)
//   • Vocab 128K (was 32K)
//   • FFN dims updated (8B: 14336; Llama-2-7B used 11008)
//   • No tied input/output embeddings
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <string>
#include <vector>
#include <cstdint>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────
struct Llama3Config {
    int64_t dim         = 4096;   // transformer hidden dimension
    int64_t ffn_dim     = 14336;  // FFN inner dimension (SwiGLU gate + up)
    int64_t n_layers    = 32;     // number of transformer blocks
    int64_t n_heads     = 32;     // number of query heads
    int64_t n_kv_heads  = 8;      // key/value heads (GQA; all sizes use 8)
    int64_t vocab_size  = 128000; // 128K token vocabulary
    int64_t seq_len     = 8192;   // context window (standard pre-training)
    float   norm_eps    = 1e-5f;  // RMSNorm epsilon
    float   rope_theta  = 500000.0f; // RoPE base frequency (§3.2)
    float   dropout     = 0.0f;

    // ── Named presets (Table 3) ──────────────────────────────────────────────
    // Tiny: smallest architecture for unit testing / CI
    static Llama3Config tiny() {
        Llama3Config c;
        c.dim        = 256;
        c.ffn_dim    = 512;
        c.n_layers   = 2;
        c.n_heads    = 4;
        c.n_kv_heads = 2;
        c.vocab_size = 128000;
        c.seq_len    = 64;
        return c;
    }
    static Llama3Config llama3_8b() {
        Llama3Config c;
        c.dim        = 4096;
        c.ffn_dim    = 14336;
        c.n_layers   = 32;
        c.n_heads    = 32;
        c.n_kv_heads = 8;
        c.vocab_size = 128000;
        c.seq_len    = 8192;
        return c;
    }
    static Llama3Config llama3_70b() {
        Llama3Config c;
        c.dim        = 8192;
        c.ffn_dim    = 28672;
        c.n_layers   = 80;
        c.n_heads    = 64;
        c.n_kv_heads = 8;
        c.vocab_size = 128000;
        c.seq_len    = 8192;
        return c;
    }
    static Llama3Config llama3_405b() {
        Llama3Config c;
        c.dim        = 16384;
        c.ffn_dim    = 53248;
        c.n_layers   = 126;
        c.n_heads    = 128;
        c.n_kv_heads = 8;
        c.vocab_size = 128000;
        c.seq_len    = 8192;
        return c;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm — pre-norm (§3.2)
//   y = x / RMS(x) * weight,   RMS(x) = sqrt(mean(x²) + ε)
// ─────────────────────────────────────────────────────────────────────────────
struct Llama3RMSNormImpl : torch::nn::Module {
    torch::Tensor w;   // learnable scale (dim,)
    float eps;

    Llama3RMSNormImpl(int64_t dim, float eps = 1e-5f);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Llama3RMSNorm);

// ─────────────────────────────────────────────────────────────────────────────
// Grouped-Query Attention with RoPE (§3.2, Table 3)
//
//   head_dim  = dim / n_heads
//   Q  : [B, T, n_heads  , head_dim]
//   K,V: [B, T, n_kv_heads, head_dim]   n_kv_heads = 8 for all sizes
//   K/V are repeated (n_heads / n_kv_heads) times before dot-product.
//   KV-cache supported for autoregressive inference.
// ─────────────────────────────────────────────────────────────────────────────
struct Llama3AttentionImpl : torch::nn::Module {
    int64_t n_heads, n_kv_heads, head_dim, n_rep;
    torch::nn::Linear wq{nullptr}, wk{nullptr}, wv{nullptr}, wo{nullptr};
    torch::nn::Dropout attn_drop{nullptr}, resid_drop{nullptr};

    Llama3AttentionImpl(const Llama3Config& cfg);

    // freqs_cos/sin: precomputed RoPE frequencies [T, head_dim/2]
    // kv_cache_{k,v}: optional [B, max_seq, n_kv_heads, head_dim] (inference)
    // cache_pos: current token position in the cache (-1 = training/prefill)
    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& freqs_cos,
        const torch::Tensor& freqs_sin,
        torch::Tensor* kv_cache_k = nullptr,
        torch::Tensor* kv_cache_v = nullptr,
        int64_t        cache_pos  = -1);
};
TORCH_MODULE(Llama3Attention);

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU Feed-Forward (Table 3, §3.2)
//   FFN(x) = W2( SiLU(W1·x) ⊙ W3·x )
// ─────────────────────────────────────────────────────────────────────────────
struct Llama3FFNImpl : torch::nn::Module {
    torch::nn::Linear w1{nullptr}, w2{nullptr}, w3{nullptr};
    torch::nn::Dropout drop{nullptr};

    Llama3FFNImpl(int64_t dim, int64_t ffn_dim, float dropout = 0.0f);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Llama3FFN);

// ─────────────────────────────────────────────────────────────────────────────
// Transformer block (pre-norm, §3.2)
//   h   = x + Attention(RMSNorm(x))
//   out = h + FFN(RMSNorm(h))
// ─────────────────────────────────────────────────────────────────────────────
struct Llama3BlockImpl : torch::nn::Module {
    Llama3RMSNorm   attention_norm{nullptr};
    Llama3RMSNorm   ffn_norm{nullptr};
    Llama3Attention attention{nullptr};
    Llama3FFN       feed_forward{nullptr};

    Llama3BlockImpl(const Llama3Config& cfg);

    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& freqs_cos,
        const torch::Tensor& freqs_sin,
        torch::Tensor* kv_cache_k = nullptr,
        torch::Tensor* kv_cache_v = nullptr,
        int64_t        cache_pos  = -1);
};
TORCH_MODULE(Llama3Block);

// ─────────────────────────────────────────────────────────────────────────────
// Full Llama 3 Transformer
//
// forward(tokens, targets):
//   tokens  [B, T] int64  — input token ids
//   targets [B, T] int64  — next-token targets (optional; if given, sets last_loss)
//   returns logits [B, T, vocab_size]
//
// forward_one(token, pos, kv_caches_k, kv_caches_v):
//   Single-token autoregressive inference with KV-cache.
//   Returns logits [1, vocab_size].
//
// Note: Llama 3 does NOT tie input/output embedding weights (unlike Llama 2).
// ─────────────────────────────────────────────────────────────────────────────
struct Llama3ModelImpl : torch::nn::Module {
    Llama3Config cfg;

    torch::nn::Embedding    tok_embeddings{nullptr};
    torch::nn::Dropout      drop{nullptr};
    torch::nn::ModuleList   layers{nullptr};
    Llama3RMSNorm           norm{nullptr};
    torch::nn::Linear       output{nullptr};  // NOT tied; separate lm head

    // Pre-computed RoPE frequencies [seq_len, head_dim/2]
    torch::Tensor freqs_cos, freqs_sin;

    torch::Tensor last_loss;  // set after forward() with targets

    explicit Llama3ModelImpl(const Llama3Config& cfg = Llama3Config());

    // Training forward: returns logits [B, T, vocab_size]; sets last_loss if targets given
    torch::Tensor forward(const torch::Tensor& tokens,
                          const torch::Tensor& targets = torch::Tensor{});

    // Single-token inference with KV-cache
    // kv_caches_k / kv_caches_v: vectors of n_layers tensors,
    //   each [1, seq_len, n_kv_heads, head_dim]
    torch::Tensor forward_one(int64_t token, int64_t pos,
                              std::vector<torch::Tensor>& kv_caches_k,
                              std::vector<torch::Tensor>& kv_caches_v);

    // Allocate per-layer KV caches for inference (batch=1)
    void init_kv_caches(std::vector<torch::Tensor>& kv_caches_k,
                        std::vector<torch::Tensor>& kv_caches_v) const;

    // Generate max_new_tokens autoregressively; greedy (temp≤0) or sampled
    std::vector<int64_t> generate(const std::vector<int64_t>& prompt_tokens,
                                  int64_t max_new_tokens = 200,
                                  float   temperature    = 1.0f,
                                  float   top_p          = 0.9f,
                                  int64_t eos_id         = 128001); // <|eot_id|>

private:
    void _init_weights();
    // Precompute RoPE cos/sin tables: [seq_len, head_dim/2]
    // Uses θ = rope_theta (500k for Llama 3)
    static std::pair<torch::Tensor, torch::Tensor>
    _precompute_freqs(int64_t head_dim, int64_t seq_len, float theta);
};
TORCH_MODULE(Llama3Model);

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factories (Table 3)
// ─────────────────────────────────────────────────────────────────────────────
Llama3Model make_llama3_tiny();    // tiny, for testing
Llama3Model make_llama3_8b();     // 8B  (architecture only — random weights)
Llama3Model make_llama3_70b();    // 70B
Llama3Model make_llama3_405b();   // 405B

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration (§3.4.1)
// ─────────────────────────────────────────────────────────────────────────────
struct Llama3TrainConfig {
    double  lr           = 3e-4;   // peak LR (8B: 3e-4, 70B: 1.5e-4, 405B: 8e-5)
    double  min_lr       = 1e-5;   // cosine decay floor
    double  weight_decay = 0.1;    // AdamW WD for 2-D params
    double  beta1        = 0.9;
    double  beta2        = 0.95;
    double  eps          = 1e-5;
    double  grad_clip    = 1.0;
    int64_t warmup_iters = 8000;   // linear warmup (§3.4.1)
    int64_t max_iters    = 1200000; // 405B: 1.2M steps
    int64_t batch_size   = 1024;
    torch::Device device = torch::kCPU;
};

// ─────────────────────────────────────────────────────────────────────────────
// Training helpers
// ─────────────────────────────────────────────────────────────────────────────
// Build two AdamW param groups: 2-D params (weight decay) + 1-D (no decay)
torch::optim::AdamW make_llama3_optimizer(Llama3Model& model,
                                          const Llama3TrainConfig& cfg);

// Cosine LR schedule with linear warmup; returns absolute LR
double llama3_lr_schedule(int64_t iter, const Llama3TrainConfig& cfg);

// One training step; returns cross-entropy loss
float llama3_train_step(Llama3Model&             model,
                        torch::optim::AdamW&     optimizer,
                        const torch::Tensor&     tokens,   // [B, T+1]
                        const Llama3TrainConfig& cfg,
                        int64_t                  iter);

} // namespace nlp
} // namespace models
} // namespace dm
