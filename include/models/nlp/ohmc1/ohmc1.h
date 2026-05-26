// ─────────────────────────────────────────────────────────────────────────────
// OhmC1 — dm-native autoregressive LLM family (tiny / small / medium / large)
//
// Architectural distinctions from Llama 2/3 in this repo:
//   1. Sandwich-norm blocks: 4 RMSNorms per block (pre+post for both attn/FFN)
//   2. QKNorm: per-head RMS norm on Q and K after projection, before RoPE
//   3. Tiered GQA: n_kv_heads = 1/2/4/8 for tiny/small/medium/large
//   4. Size-scaled RoPE base: 10k / 50k / 200k / 500k per size
//   5. 64K vocabulary, embeddings never tied to lm_head
//
// oh-mah-c, dm/OhmC1, 2026. [151]
// ─────────────────────────────────────────────────────────────────────────────

#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace dm {
namespace models {
namespace nlp {

// ── Config ────────────────────────────────────────────────────────────────────

struct OhmC1Config {
    int64_t dim         = 2048;    // transformer hidden dimension
    int64_t ffn_dim     = 5632;    // SwiGLU inner dimension
    int64_t n_layers    = 24;      // number of transformer blocks
    int64_t n_heads     = 16;      // query heads
    int64_t n_kv_heads  = 4;       // KV heads (GQA); must divide n_heads
    int64_t vocab_size  = 64000;   // 64K BPE vocabulary
    int64_t seq_len     = 4096;    // context window (max sequence length)
    float   norm_eps    = 1e-6f;   // RMSNorm epsilon
    float   rope_theta  = 200000.0f; // RoPE base frequency (size-scaled)
    int64_t window_size = 0;       // 0 = full causal attention; >0 = sliding window
    float   dropout     = 0.0f;    // dropout rate (0.0 for inference)

    // head_dim derived from dim / n_heads — must be even for RoPE
    int64_t head_dim() const { return dim / n_heads; }
    // repetition factor for GQA expansion
    int64_t n_rep() const { return n_heads / n_kv_heads; }

    // Named presets
    static OhmC1Config tiny();    // <10M params, seq_len=128, for unit tests
    static OhmC1Config small();   // ~145M params, seq_len=2048
    static OhmC1Config medium();  // ~1.7B params, seq_len=4096 (default)
    static OhmC1Config large();   // ~7.1B params, seq_len=8192
};

// ── RMSNorm ───────────────────────────────────────────────────────────────────

struct OhmC1RMSNormImpl : torch::nn::Module {
    torch::Tensor w;  // (dim,) learnable scale, init=ones
    float eps;

    OhmC1RMSNormImpl(int64_t dim, float eps = 1e-6f);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(OhmC1RMSNorm);

// ── QKNorm ────────────────────────────────────────────────────────────────────
// Per-head RMS normalization applied to Q and K after projection.
// Stabilizes attention logit scale as a function of depth (inspired by Gemma 2).

struct OhmC1QKNormImpl : torch::nn::Module {
    torch::Tensor w;  // (n_heads, head_dim) learnable gain, init=ones
    int64_t n_heads_, head_dim_;
    float eps;

    OhmC1QKNormImpl(int64_t n_heads, int64_t head_dim, float eps = 1e-6f);
    // x: [B, T, n_heads, head_dim] -> same shape, per-head L2-normalized then scaled
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(OhmC1QKNorm);

// ── Attention ─────────────────────────────────────────────────────────────────

struct OhmC1AttentionImpl : torch::nn::Module {
    int64_t n_heads_, n_kv_heads_, head_dim_, n_rep_;
    int64_t window_size_;

    torch::nn::Linear wq{nullptr}, wk{nullptr}, wv{nullptr}, wo{nullptr};
    OhmC1QKNorm q_norm{nullptr}, k_norm{nullptr};
    torch::nn::Dropout attn_drop{nullptr}, resid_drop{nullptr};

    explicit OhmC1AttentionImpl(const OhmC1Config& cfg);

    // freqs_cos, freqs_sin: precomputed RoPE [T, head_dim/2]
    // kv_cache_k/v: [1, seq_len, n_kv_heads, head_dim] — inference KV cache
    // cache_pos: current token position; -1 for training (full sequence)
    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& freqs_cos,
        const torch::Tensor& freqs_sin,
        torch::Tensor* kv_cache_k = nullptr,
        torch::Tensor* kv_cache_v = nullptr,
        int64_t        cache_pos  = -1);
};
TORCH_MODULE(OhmC1Attention);

// ── FFN (SwiGLU) ──────────────────────────────────────────────────────────────

struct OhmC1FFNImpl : torch::nn::Module {
    torch::nn::Linear w1{nullptr}, w2{nullptr}, w3{nullptr};
    torch::nn::Dropout drop{nullptr};

    OhmC1FFNImpl(int64_t dim, int64_t ffn_dim, float dropout = 0.0f);
    // SwiGLU: W2( SiLU(W1(x)) * W3(x) )
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(OhmC1FFN);

// ── Block (sandwich-norm) ─────────────────────────────────────────────────────
//
// h   = x + post_attn_norm( attention( pre_attn_norm(x) ) )
// out = h + post_ffn_norm ( ffn      ( pre_ffn_norm (h) ) )

struct OhmC1BlockImpl : torch::nn::Module {
    OhmC1RMSNorm  pre_attn_norm{nullptr};
    OhmC1RMSNorm  post_attn_norm{nullptr};
    OhmC1RMSNorm  pre_ffn_norm{nullptr};
    OhmC1RMSNorm  post_ffn_norm{nullptr};
    OhmC1Attention attention{nullptr};
    OhmC1FFN       feed_forward{nullptr};

    explicit OhmC1BlockImpl(const OhmC1Config& cfg);

    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& freqs_cos,
        const torch::Tensor& freqs_sin,
        torch::Tensor* kv_cache_k = nullptr,
        torch::Tensor* kv_cache_v = nullptr,
        int64_t        cache_pos  = -1);
};
TORCH_MODULE(OhmC1Block);

// ── Model ─────────────────────────────────────────────────────────────────────

struct OhmC1ModelImpl : torch::nn::Module {
    OhmC1Config cfg;

    torch::nn::Embedding  tok_embeddings{nullptr};
    torch::nn::Dropout    drop{nullptr};
    torch::nn::ModuleList layers{nullptr};
    OhmC1RMSNorm          norm{nullptr};    // final pre-lm_head norm
    torch::nn::Linear     lm_head{nullptr}; // never tied to tok_embeddings

    torch::Tensor freqs_cos, freqs_sin;  // registered buffers [seq_len, head_dim/2]
    torch::Tensor last_loss;             // set when targets are provided to forward()

    explicit OhmC1ModelImpl(const OhmC1Config& cfg = OhmC1Config());

    // Training forward: tokens [B,T] -> logits [B,T,vocab_size]
    // If targets [B,T] provided, also computes cross-entropy into last_loss.
    torch::Tensor forward(const torch::Tensor& tokens,
                          const torch::Tensor& targets = torch::Tensor{});

    // Single-token inference with KV-cache.
    // kv_caches_k/v: per-layer caches of shape [1, seq_len, n_kv_heads, head_dim]
    torch::Tensor forward_one(int64_t token, int64_t pos,
                              std::vector<torch::Tensor>& kv_caches_k,
                              std::vector<torch::Tensor>& kv_caches_v);

    // Allocate per-layer KV caches on the model's device.
    void init_kv_caches(std::vector<torch::Tensor>& kv_caches_k,
                        std::vector<torch::Tensor>& kv_caches_v) const;

    // Autoregressive text generation.
    std::vector<int64_t> generate(const std::vector<int64_t>& prompt_tokens,
                                  int64_t max_new_tokens = 200,
                                  float   temperature    = 1.0f,
                                  float   top_p          = 0.9f,
                                  int64_t eos_id         = 1);

 private:
    void _init_weights();
    static std::pair<torch::Tensor, torch::Tensor>
    _precompute_freqs(int64_t head_dim, int64_t seq_len, float theta);
};
TORCH_MODULE(OhmC1Model);

// ── Factory functions ─────────────────────────────────────────────────────────

OhmC1Model make_ohmc1_tiny();
OhmC1Model make_ohmc1_small();
OhmC1Model make_ohmc1_medium();
OhmC1Model make_ohmc1_large();

// ── Training configuration ────────────────────────────────────────────────────

struct OhmC1TrainConfig {
    double  lr           = 3e-4;
    double  min_lr       = 1e-5;
    double  weight_decay = 0.1;
    double  beta1        = 0.9;
    double  beta2        = 0.95;
    double  eps          = 1e-8;
    double  grad_clip    = 1.0;
    int64_t warmup_iters = 2000;
    int64_t max_iters    = 100000;
    int64_t batch_size   = 32;
    torch::Device device = torch::kCPU;
};

// Build AdamW with two param groups: 2-D tensors (decay) and 1-D norms/gains (no decay).
torch::optim::AdamW make_ohmc1_optimizer(OhmC1Model& model,
                                          const OhmC1TrainConfig& cfg);

// Cosine LR with linear warmup. Returns absolute LR for the given iteration.
double ohmc1_lr_schedule(int64_t iter, const OhmC1TrainConfig& cfg);

// One AdamW gradient step. Returns cross-entropy loss value.
float ohmc1_train_step(OhmC1Model&             model,
                       torch::optim::AdamW&    optimizer,
                       const torch::Tensor&    tokens,  // [B, T+1] int64
                       const OhmC1TrainConfig& cfg,
                       int64_t                 iter);

// ── BPE Tokenizer (optional — model works with raw int64 IDs without it) ──────
//
// Reads a tiktoken-compatible vocab.json ({"token": id, ...}) and merges.txt
// ("TOKEN1 TOKEN2\n" one merge rule per line) from disk. No external dependencies.

class OhmC1BPETokenizer {
 public:
    static constexpr int64_t BOS_ID = 0;
    static constexpr int64_t EOS_ID = 1;
    static constexpr int64_t PAD_ID = 2;
    static constexpr int64_t UNK_ID = 3;

    explicit OhmC1BPETokenizer(const std::string& vocab_path,
                                const std::string& merges_path);

    // Encode UTF-8 text to token IDs (prepends BOS by default).
    std::vector<int64_t> encode(const std::string& text,
                                 bool add_bos = false) const;

    // Decode token IDs to UTF-8 text.
    std::string decode(const std::vector<int64_t>& ids) const;

    int64_t vocab_size() const;

 private:
    std::unordered_map<std::string, int64_t>         token_to_id_;
    std::vector<std::string>                          id_to_token_;
    std::vector<std::pair<std::string, std::string>>  merges_;

    // BPE encode a single whitespace-split word.
    std::vector<std::string> _bpe_encode_word(const std::string& word) const;
    // Split text into pre-tokenization units (words + punctuation).
    std::vector<std::string> _pretokenize(const std::string& text) const;
};

}  // namespace nlp
}  // namespace models
}  // namespace dm
