#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Llama 2 — Open Foundation and Fine-Tuned Chat Models
// Hugo Touvron et al., Meta AI, arXiv:2307.09288, 2023
//
// Reference implementation: Andrej Karpathy, llama2.c
//   https://github.com/karpathy/llama2.c
//
// ── Architecture ─────────────────────────────────────────────────────────────
// Decoder-only Transformer with:
//   • RMSNorm (pre-norm) instead of LayerNorm                (§2.1)
//   • RoPE (Rotary Positional Embeddings) on Q and K        (§2.1, Su et al.)
//   • Grouped-Query Attention (GQA): n_kv_heads ≤ n_heads   (§2.2, Ainslie et al.)
//   • SwiGLU feed-forward: FFN(x) = W2(SiLU(W1·x) ⊙ W3·x) (§2.1, Shazeer)
//   • Tied token embedding / output projection weights       (Press & Wolf 2017)
//
// ── Model sizes ──────────────────────────────────────────────────────────────
//   7B:  dim=4096, n_layers=32, n_heads=32, n_kv_heads=32, hidden=11008
//  13B:  dim=5120, n_layers=40, n_heads=40, n_kv_heads=40, hidden=13824
//  70B:  dim=8192, n_layers=80, n_heads=64, n_kv_heads= 8, hidden=28672  (GQA)
//
// ── Training ─────────────────────────────────────────────────────────────────
//   Optimizer : AdamW, β=(0.9, 0.95), ε=1e-5
//   Scheduler : cosine with warmup, lr=3e-4 (7B), min_lr=lr/10
//   Grad clip : 1.0 (global norm)
//   Weight dec : 0.1 (2-D tensors only; 1-D biases/norms decay=0)
//   Context   : 4096 tokens
//   Tokenizer : SentencePiece BPE, 32000-token vocabulary
//
// ── Checkpoint format (.bin) ─────────────────────────────────────────────────
//   The binary checkpoint format is compatible with karpathy/llama2.c:
//   [Config header (7 × int32)][TransformerWeights (float32 arrays, mmap-able)]
//   vocab_size < 0 signals unshared cls weights (abs value is true vocab size).
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <string>
#include <vector>
#include <cstdint>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Model configuration
// ─────────────────────────────────────────────────────────────────────────────
struct Llama2Config {
    int64_t dim         = 288;    // transformer hidden dimension
    int64_t hidden_dim  = 768;    // FFN inner dimension
    int64_t n_layers    = 6;      // number of transformer blocks
    int64_t n_heads     = 6;      // number of query heads
    int64_t n_kv_heads  = 6;      // number of key/value heads (GQA: may be < n_heads)
    int64_t vocab_size  = 32000;  // vocabulary size
    int64_t seq_len     = 256;    // max sequence length (context window)
    float   norm_eps    = 1e-5f;  // RMSNorm epsilon
    float   dropout     = 0.0f;   // dropout (train > 0, inference = 0)

    // Pre-defined size variants (approximate)
    static Llama2Config stories110k() { // tiny, for testing
        return {288, 768, 6, 6, 6, 32000, 256, 1e-5f, 0.0f};
    }
    static Llama2Config llama_7b() {
        return {4096, 11008, 32, 32, 32, 32000, 4096, 1e-5f, 0.0f};
    }
    static Llama2Config llama_13b() {
        return {5120, 13824, 40, 40, 40, 32000, 4096, 1e-5f, 0.0f};
    }
    static Llama2Config llama_70b() { // GQA: 8 kv heads
        return {8192, 28672, 80, 64, 8, 32000, 4096, 1e-5f, 0.0f};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm — pre-norm, no bias (§2.1)
// y = x / RMS(x) * weight,  RMS(x) = sqrt(mean(x²) + ε)
// ─────────────────────────────────────────────────────────────────────────────
struct Llama2RMSNormImpl : torch::nn::Module {
    torch::Tensor w;   // learnable scale (dim,)
    float eps;

    Llama2RMSNormImpl(int64_t dim, float eps = 1e-5f);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Llama2RMSNorm);

// ─────────────────────────────────────────────────────────────────────────────
// Grouped-Query Attention with RoPE (§2.1, §2.2)
//
//   head_dim = dim / n_heads
//   Q : [B, T, n_heads  × head_dim]
//   K : [B, T, n_kv_heads × head_dim]   (fewer heads than Q in GQA)
//   V : [B, T, n_kv_heads × head_dim]
//   K/V are repeated (n_heads / n_kv_heads) times before attention.
//   KV-cache supported for autoregressive inference.
// ─────────────────────────────────────────────────────────────────────────────
struct Llama2AttentionImpl : torch::nn::Module {
    int64_t n_heads, n_kv_heads, head_dim, n_rep;
    torch::nn::Linear wq{nullptr}, wk{nullptr}, wv{nullptr}, wo{nullptr};
    torch::nn::Dropout attn_drop{nullptr}, resid_drop{nullptr};

    Llama2AttentionImpl(const Llama2Config& cfg);

    // freqs_cos/sin: precomputed RoPE frequencies [T, head_dim/2]
    // kv_cache_{k,v}: optional [B, max_seq, n_kv_heads, head_dim] (inference)
    // cache_pos: current token position in the cache (inference mode)
    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& freqs_cos,
        const torch::Tensor& freqs_sin,
        torch::Tensor* kv_cache_k = nullptr,
        torch::Tensor* kv_cache_v = nullptr,
        int64_t        cache_pos  = -1);
};
TORCH_MODULE(Llama2Attention);

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU Feed-Forward (§2.1, Shazeer 2020)
//   FFN(x) = W2( SiLU(W1·x) ⊙ W3·x )
//   hidden_dim is rounded to multiple_of (default 256) after 2/3 × 4d scaling
// ─────────────────────────────────────────────────────────────────────────────
struct Llama2FFNImpl : torch::nn::Module {
    torch::nn::Linear w1{nullptr}, w2{nullptr}, w3{nullptr};
    torch::nn::Dropout drop{nullptr};

    Llama2FFNImpl(int64_t dim, int64_t hidden_dim, float dropout = 0.0f);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Llama2FFN);

// ─────────────────────────────────────────────────────────────────────────────
// Transformer block (pre-norm)
//   h  = x + Attention(RMSNorm(x))
//   out = h + FFN(RMSNorm(h))
// ─────────────────────────────────────────────────────────────────────────────
struct Llama2BlockImpl : torch::nn::Module {
    Llama2RMSNorm   attention_norm{nullptr};
    Llama2RMSNorm   ffn_norm{nullptr};
    Llama2Attention attention{nullptr};
    Llama2FFN       feed_forward{nullptr};

    Llama2BlockImpl(const Llama2Config& cfg);

    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& freqs_cos,
        const torch::Tensor& freqs_sin,
        torch::Tensor* kv_cache_k = nullptr,
        torch::Tensor* kv_cache_v = nullptr,
        int64_t        cache_pos  = -1);
};
TORCH_MODULE(Llama2Block);

// ─────────────────────────────────────────────────────────────────────────────
// Full Llama 2 Transformer
//
// forward(tokens, targets):
//   tokens  [B, T] int64  — input token ids
//   targets [B, T] int64  — next-token targets (optional; if given, loss is set)
//   returns logits [B, T, vocab_size]
//
// forward_one(token, pos, kv_caches_k, kv_caches_v):
//   Single-token inference with KV cache. Returns logits [1, vocab_size].
// ─────────────────────────────────────────────────────────────────────────────
struct Llama2ModelImpl : torch::nn::Module {
    Llama2Config cfg;

    torch::nn::Embedding    tok_embeddings{nullptr};
    torch::nn::Dropout      drop{nullptr};
    torch::nn::ModuleList   layers{nullptr};
    Llama2RMSNorm           norm{nullptr};
    torch::nn::Linear       output{nullptr};   // tied to tok_embeddings.weight

    // Pre-computed RoPE frequencies [max_seq_len, head_dim/2]
    torch::Tensor freqs_cos, freqs_sin;

    torch::Tensor last_loss;  // set after forward() with targets

    explicit Llama2ModelImpl(const Llama2Config& cfg = Llama2Config());

    // Training forward: returns logits [B, T, vocab_size]; sets last_loss
    torch::Tensor forward(const torch::Tensor& tokens,
                          const torch::Tensor& targets = torch::Tensor{});

    // Inference forward: single token, uses KV caches
    // kv_caches_k / kv_caches_v: vectors of length n_layers,
    //   each [1, seq_len, n_kv_heads, head_dim]
    torch::Tensor forward_one(int64_t token, int64_t pos,
                              std::vector<torch::Tensor>& kv_caches_k,
                              std::vector<torch::Tensor>& kv_caches_v);

    // Allocate per-layer KV caches for inference (batch=1)
    void init_kv_caches(std::vector<torch::Tensor>& kv_caches_k,
                        std::vector<torch::Tensor>& kv_caches_v) const;

    // Generate max_new_tokens autoregressively; greedy (temp=0) or sampled
    std::vector<int64_t> generate(const std::vector<int64_t>& prompt_tokens,
                                  int64_t max_new_tokens = 200,
                                  float   temperature    = 1.0f,
                                  float   top_p          = 0.9f,
                                  int64_t eos_id         = 2);

private:
    void _init_weights();
    static torch::Tensor _precompute_freqs(int64_t head_dim,
                                           int64_t seq_len,
                                           float   theta = 10000.0f);
};
TORCH_MODULE(Llama2Model);

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factories
// ─────────────────────────────────────────────────────────────────────────────
Llama2Model make_llama2_stories110k();  // 15M tiny (TinyStories)
Llama2Model make_llama2_7b();           // 7B  (architecture only — random weights)
Llama2Model make_llama2_13b();          // 13B
Llama2Model make_llama2_70b();          // 70B (GQA)

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration
// ─────────────────────────────────────────────────────────────────────────────
struct Llama2TrainConfig {
    double  lr            = 3e-4;    // peak learning rate
    double  min_lr        = 3e-5;    // cosine decay floor (lr/10)
    double  weight_decay  = 0.1;     // AdamW WD for 2-D tensors
    double  beta1         = 0.9;
    double  beta2         = 0.95;
    double  eps           = 1e-5;
    double  grad_clip     = 1.0;
    int64_t warmup_iters  = 100;
    int64_t max_iters     = 100000;
    int64_t batch_size    = 64;
    torch::Device device  = torch::kCPU;
};

// ─────────────────────────────────────────────────────────────────────────────
// Training helpers
// ─────────────────────────────────────────────────────────────────────────────
// Build two AdamW param groups: 2-D params (weight decay) + 1-D (no decay)
torch::optim::AdamW make_llama2_optimizer(Llama2Model& model,
                                          const Llama2TrainConfig& cfg);

// Cosine LR schedule with linear warmup (returns multiplier for base lr)
double llama2_lr_schedule(int64_t iter, const Llama2TrainConfig& cfg);

// One training step; returns cross-entropy loss
float llama2_train_step(Llama2Model&           model,
                        torch::optim::AdamW&   optimizer,
                        const torch::Tensor&   tokens,   // [B, T+1]
                        const Llama2TrainConfig& cfg,
                        int64_t                iter);

} // namespace nlp
} // namespace models
} // namespace dm
