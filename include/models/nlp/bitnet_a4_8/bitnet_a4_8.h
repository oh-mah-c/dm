#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// BitNet a4.8 — 4-bit Activations for 1-bit LLMs
// Wang et al., arXiv:2411.04965v1, 2024
//
// ── Overview ─────────────────────────────────────────────────────────────────
// Builds on BitNet b1.58 (1.58-bit ternary weights {-1,0,+1}) and introduces
// hybrid quantization + sparsification of activations to reach 4-bit average
// activation precision while matching b1.58 perplexity.
//
// ── Weight quantization (shared with b1.58) ──────────────────────────────────
// Q_w(W) = α · RoundClip(W/(α+ε), -1, 1),  α = mean(|W|)
// Weights stored as ternary {-1,0,+1} scaled by a per-tensor float α.
//
// ── Activation quantization (hybrid, Figure 1) ───────────────────────────────
// Three flavours:
//
//   INT4 (absmean): used for Attention Q/K/V inputs and FFN Up/Gate inputs.
//     Q_INT4(X) = β/√7 · RoundClip(√7/(β+ε) · X, -8, 7),  β = mean(|X|)
//     Range [-8,7] (asymmetric) maps to INT4 signed; scaling keeps RMS equal.
//
//   INT8 + TopK sparsification: used for Attention Out-projection input.
//     Q_INT8(X) = γ/127 · Round(127/γ · X),  γ = max(|X|)  (absmax)
//     M = TopK(|X|, k=50%) binary mask  →  Y = (Q_INT8(X) ⊙ M) · W^T
//
//   INT8 only (no sparsification): used for FFN Down-projection input.
//     Same absmax INT8 formula. High sparsity comes naturally from ReLU²GLU.
//
// ── FFN: ReLU²GLU (§2.1) ─────────────────────────────────────────────────────
// FFN(X) = ReLU²GLU(X) W_down^T
//   where ReLU²GLU(X) = (X W_up^T) ⊙ ReLU²(X W_gate^T)
//   ReLU² = (max(0,·))² — achieves >80% sparsity in down-projection input.
//
// ── Model sizes (Table 6, Appendix A) ────────────────────────────────────────
// 700M:  hidden=1536, glu=4096,  heads=24, layers=24
// 1.3B:  hidden=2048, glu=5460,  heads=32, layers=24
// 3B:    hidden=3200, glu=8640,  heads=32, layers=26
// 7B:    hidden=4096, glu=11008, heads=32, layers=32
//
// ── Training recipe (§2.2) ───────────────────────────────────────────────────
// Stage 1: continue-train b1.58 checkpoint with INT8 activations + ReLU²GLU
//          (95B tokens, RedPajama)
// Stage 2: switch to 4-bit + sparse activations (5B tokens)
// STE (straight-through estimator) for quantization gradients.
// AdamW β=(0.9,0.95), warmup 375 steps, cosine lr decay, weight-decay 0.1→0.
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <vector>
#include <cstdint>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Configuration (Table 6)
// ─────────────────────────────────────────────────────────────────────────────
struct BitNetA48Config {
    int64_t hidden_size   = 2048;   // d_model
    int64_t glu_size      = 5460;   // intermediate FFN size (up/gate)
    int64_t n_heads       = 32;     // attention heads
    int64_t n_kv_heads    = 32;     // key/value heads (no GQA in original)
    int64_t n_layers      = 24;     // transformer layers
    int64_t vocab_size    = 32000;  // default vocabulary (b1.58 compatible)
    int64_t max_seq_len   = 2048;   // maximum sequence length
    float   rms_eps       = 1e-6f;  // RMSNorm epsilon
    int64_t topk_percent  = 50;     // TopK sparsification for Out-proj (%)

    // Pre-defined size variants (Table 6)
    static BitNetA48Config size_700m() { return {1536,  4096, 24, 24, 24, 32000, 2048, 1e-6f, 50}; }
    static BitNetA48Config size_1b3()  { return {2048,  5460, 32, 32, 24, 32000, 2048, 1e-6f, 50}; }
    static BitNetA48Config size_3b()   { return {3200,  8640, 32, 32, 26, 32000, 2048, 1e-6f, 50}; }
    static BitNetA48Config size_7b()   { return {4096, 11008, 32, 32, 32, 32000, 2048, 1e-6f, 50}; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Quantization utilities (equations 1–9)
// ─────────────────────────────────────────────────────────────────────────────

// Weight quantization: Q_w(W) = α·RoundClip(W/(α+ε),-1,1), α=mean(|W|)
// Returns float32 fake-quantized tensor (for training with STE).
torch::Tensor quantize_weight_ternary(const torch::Tensor& W);

// INT4 absmean: Q_INT4(X) = β/√7 · RoundClip(√7/(β+ε)·X, -8, 7), β=mean(|X|)
// Used for attention QKV inputs and FFN up/gate inputs.
torch::Tensor quantize_int4_absmean(const torch::Tensor& X);

// INT8 absmax: Q_INT8(X) = γ/127 · Round(127/γ·X), γ=max(|X|)
// Used for attention out-proj and FFN down-proj inputs.
torch::Tensor quantize_int8_absmax(const torch::Tensor& X);

// TopK sparsification mask: binary mask with top-k% largest |X| set to 1.
// k_percent in [0,100]. Applied to quantized tensor for out-proj input.
torch::Tensor topk_mask(const torch::Tensor& X, int64_t k_percent);

// ─────────────────────────────────────────────────────────────────────────────
// BitLinear — 1.58-bit weight linear layer with activation quantization
// ─────────────────────────────────────────────────────────────────────────────
enum class ActQuant {
    INT4_ABSMEAN,    // for QKV and FFN up/gate
    INT8_ABSMAX,     // for FFN down
    INT8_TOPK,       // for attention out-proj (+ TopK sparsification)
};

struct BitLinearImpl : torch::nn::Module {
    int64_t  in_features, out_features;
    bool     bias_;
    ActQuant act_quant;
    int64_t  topk_percent;

    torch::nn::Linear linear{nullptr};  // stores float weights, quantized on-the-fly

    BitLinearImpl(int64_t in_f, int64_t out_f, bool bias,
                  ActQuant aq, int64_t topk_pct = 50);

    // Forward: RMSNorm(x) → quant(x) → fake-ternary-weight matmul
    // norm_weight: the per-layer sub-LN weight (passed from parent block)
    torch::Tensor forward(const torch::Tensor& x,
                          const torch::Tensor& norm_weight);
};
TORCH_MODULE(BitLinear);

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm
// ─────────────────────────────────────────────────────────────────────────────
struct RMSNormImpl : torch::nn::Module {
    int64_t d;
    float   eps;
    torch::Tensor weight;  // gain parameter

    RMSNormImpl(int64_t dim, float eps = 1e-6f);
    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(RMSNorm);

// ─────────────────────────────────────────────────────────────────────────────
// Rotary Position Embedding (RoPE)
// ─────────────────────────────────────────────────────────────────────────────
// Returns cosine and sine caches [seq_len, head_dim/2]
std::pair<torch::Tensor, torch::Tensor> build_rope_cache(
    int64_t seq_len, int64_t head_dim, torch::Device device,
    float theta = 10000.0f);

// Apply RoPE to q or k of shape [B, n_heads, T, head_dim]
torch::Tensor apply_rope(const torch::Tensor& x,
                         const torch::Tensor& cos_cache,
                         const torch::Tensor& sin_cache);

// ─────────────────────────────────────────────────────────────────────────────
// BitNet a4.8 Attention block
// ─────────────────────────────────────────────────────────────────────────────
struct BitNetA48AttentionImpl : torch::nn::Module {
    int64_t hidden, n_heads, n_kv_heads, head_dim;
    int64_t topk_percent;

    // BitLinear projections
    BitLinear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, o_proj{nullptr};

    // Sub-layer norms (pre-projection, folded into BitLinear)
    RMSNorm q_norm{nullptr}, k_norm{nullptr}, v_norm{nullptr}, o_norm{nullptr};

    BitNetA48AttentionImpl(const BitNetA48Config& cfg);

    // x: [B, T, hidden]   mask: causal additive mask [T, T] or null
    // Returns: [B, T, hidden]
    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& cos_cache,
        const torch::Tensor& sin_cache,
        const torch::Tensor& mask);
};
TORCH_MODULE(BitNetA48Attention);

// ─────────────────────────────────────────────────────────────────────────────
// FFN: ReLU²GLU + 1.58-bit weights + INT4/INT8 activations
// ─────────────────────────────────────────────────────────────────────────────
struct BitNetA48FFNImpl : torch::nn::Module {
    // up / gate / down projections
    BitLinear up_proj{nullptr}, gate_proj{nullptr}, down_proj{nullptr};
    RMSNorm   up_norm{nullptr},  gate_norm{nullptr}, down_norm{nullptr};

    BitNetA48FFNImpl(const BitNetA48Config& cfg);

    // x: [B, T, hidden]  →  [B, T, hidden]
    torch::Tensor forward(const torch::Tensor& x);
};
TORCH_MODULE(BitNetA48FFN);

// ─────────────────────────────────────────────────────────────────────────────
// Transformer decoder block
// ─────────────────────────────────────────────────────────────────────────────
struct BitNetA48BlockImpl : torch::nn::Module {
    RMSNorm          attn_norm{nullptr};
    RMSNorm          ffn_norm{nullptr};
    BitNetA48Attention attn{nullptr};
    BitNetA48FFN       ffn{nullptr};

    BitNetA48BlockImpl(const BitNetA48Config& cfg);

    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& cos_cache,
        const torch::Tensor& sin_cache,
        const torch::Tensor& mask);
};
TORCH_MODULE(BitNetA48Block);

// ─────────────────────────────────────────────────────────────────────────────
// BitNet a4.8 full model
// ─────────────────────────────────────────────────────────────────────────────
struct BitNetA48ModelImpl : torch::nn::Module {
    BitNetA48Config cfg;

    torch::nn::Embedding tok_emb{nullptr};
    torch::nn::ModuleList blocks{nullptr};
    RMSNorm  final_norm{nullptr};
    // Output head — weight-tied to tok_emb
    int64_t vocab_size_;

    explicit BitNetA48ModelImpl(const BitNetA48Config& cfg = BitNetA48Config{});

    // tokens: [B, T] int64  →  logits [B, T, vocab_size]
    torch::Tensor forward(const torch::Tensor& tokens);

    // Greedy decode: returns token-id sequence starting with sot_id
    std::vector<int64_t> greedy_decode(
        const torch::Tensor& prompt_tokens,  // [1, L] int64
        int64_t eot_id,
        int64_t max_new_tokens = 200);
};
TORCH_MODULE(BitNetA48Model);

// Convenience factories
BitNetA48Model make_bitnet_a48_700m();
BitNetA48Model make_bitnet_a48_1b3();
BitNetA48Model make_bitnet_a48_3b();
BitNetA48Model make_bitnet_a48_7b();

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration (Table 7)
// ─────────────────────────────────────────────────────────────────────────────
struct BitNetA48TrainConfig {
    double  lr_start       = 1.2e-3;   // 1.3B default (Table 7)
    double  lr_end         = 8e-4;
    double  weight_decay   = 0.1;      // cosine decay → 0
    double  beta1          = 0.9;
    double  beta2          = 0.95;
    double  grad_clip      = 1.0;
    int64_t warmup_steps   = 375;
    int64_t batch_size     = 1 << 20;  // 1M tokens (Table 6)
    int64_t seq_len        = 2048;
    torch::Device device   = torch::kCPU;
};

// One gradient update; returns cross-entropy loss.
float bitnet_a48_train_step(
    BitNetA48Model&         model,
    torch::optim::AdamW&    optimizer,
    const torch::Tensor&    tokens_in,   // [B, T] int64
    const torch::Tensor&    tokens_tgt,  // [B, T] int64
    double                  lr_scale = 1.0);

} // namespace nlp
} // namespace models
} // namespace dm
