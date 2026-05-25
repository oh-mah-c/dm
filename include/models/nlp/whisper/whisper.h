#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Whisper — Robust Speech Recognition via Large-Scale Weak Supervision
// A. Radford et al., ICML 2023 (radford23a)
//
// ── Audio front-end (Section 2) ─────────────────────────────────────────────
// 16 kHz audio, 80-channel log-mel spectrogram
// 25 ms Hann window, 10 ms hop, globally normalised to [-1, 1]
//
// ── Encoder (Section 2.1) ────────────────────────────────────────────────────
// Conv1d(80→d_model, k=3, pad=1) + GELU   (stem, no downsampling)
// Conv1d(d_model→d_model, k=3, stride=2, pad=1) + GELU  (halves time)
// + sinusoidal positional embeddings (fixed)
// N × pre-norm Transformer encoder block:
//   LayerNorm → MultiHeadSelfAttn (causal=false) → residual
//   LayerNorm → MLP (4× GELU) → residual
// Final LayerNorm
//
// ── Decoder (Section 2.1) ────────────────────────────────────────────────────
// Learned token embeddings (vocab_size × d_model) + learned positional
// M × pre-norm Transformer decoder block:
//   LayerNorm → MaskedMultiHeadSelfAttn (causal) → residual
//   LayerNorm → MultiHeadCrossAttn (queries from decoder, keys/values from encoder) → residual
//   LayerNorm → MLP (4× GELU) → residual
// Final LayerNorm → linear projection (tied to token embedding weights)
//
// ── Model sizes (Table 1) ────────────────────────────────────────────────────
// Tiny:   d=384,  heads=6,  enc_layers=4,  dec_layers=4,   39M params
// Base:   d=512,  heads=8,  enc_layers=6,  dec_layers=6,   74M params
// Small:  d=768,  heads=12, enc_layers=12, dec_layers=12, 244M params
// Medium: d=1024, heads=16, enc_layers=24, dec_layers=24, 769M params
// Large:  d=1280, heads=20, enc_layers=32, dec_layers=32, 1550M params
//
// ── Training (Section 2.2) ───────────────────────────────────────────────────
// AdamW, β=(0.9,0.999), linear lr warmup 2048 steps, batch=256 segments
// Gradient norm clipping, 1e-6 weight decay, ~2^20 update steps
// Byte-level BPE, GPT-2 vocab (50257 tokens) + 99 language/task tokens
//
// ── Multitask tokens (Section 2.3) ──────────────────────────────────────────
// <|startoftranscript|>  →  language_tag  →  <|transcribe|>/<|translate|>
// [<|notimestamps|> | timestamp tokens]  →  text tokens  →  <|endoftranscript|>
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <string>
#include <vector>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Model configuration (Table 1)
// ─────────────────────────────────────────────────────────────────────────────
struct WhisperConfig {
    int64_t n_mels        = 80;        // log-mel channels (fixed)
    int64_t d_model       = 512;       // hidden dimension
    int64_t n_heads       = 8;         // attention heads
    int64_t enc_layers    = 6;         // encoder transformer layers
    int64_t dec_layers    = 6;         // decoder transformer layers
    int64_t vocab_size    = 51865;     // GPT-2 BPE + 1608 special tokens
    int64_t max_source_positions = 1500; // encoder context (T/2 frames)
    int64_t max_target_positions = 448; // decoder context (tokens)
    double  dropout       = 0.0;       // inference-time dropout = 0

    // Pre-defined size variants
    static WhisperConfig tiny()   { return {80, 384,  6,  4,  4,  51865, 1500, 448}; }
    static WhisperConfig base()   { return {80, 512,  8,  6,  6,  51865, 1500, 448}; }
    static WhisperConfig small()  { return {80, 768, 12, 12, 12, 51865, 1500, 448}; }
    static WhisperConfig medium() { return {80,1024, 16, 24, 24, 51865, 1500, 448}; }
    static WhisperConfig large()  { return {80,1280, 20, 32, 32, 51865, 1500, 448}; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Special token IDs (Section 2.3)
// ─────────────────────────────────────────────────────────────────────────────
struct WhisperTokens {
    static constexpr int64_t SOT          = 50258; // <|startoftranscript|>
    static constexpr int64_t EOT          = 50257; // <|endoftext|>
    static constexpr int64_t TRANSLATE    = 50358; // <|translate|>
    static constexpr int64_t TRANSCRIBE   = 50359; // <|transcribe|>
    static constexpr int64_t NOSPEECH     = 50362; // <|nospeech|>
    static constexpr int64_t NOTIMESTAMPS = 50363; // <|notimestamps|>
    // language tags: SOT+1 .. SOT+99
    static int64_t lang(int64_t lang_id) { return SOT + 1 + lang_id; }
    // timestamp tokens: 50364 + frame_idx (20 ms resolution)
    static int64_t timestamp(int64_t frame) { return 50364 + frame; }
};

// ─────────────────────────────────────────────────────────────────────────────
// Multi-head attention (self or cross)
// ─────────────────────────────────────────────────────────────────────────────
struct WhisperAttentionImpl : torch::nn::Module {
    int64_t d_model, n_heads, head_dim;
    torch::nn::Linear q_proj{nullptr}, k_proj{nullptr},
                      v_proj{nullptr}, out_proj{nullptr};

    WhisperAttentionImpl(int64_t d_model, int64_t n_heads);

    // Returns [B, T_q, d_model].
    // key_value: if null, use x as keys/values (self-attention);
    //            otherwise cross-attention (encoder hidden states).
    // mask: additive attention mask (−∞ at positions to block).
    torch::Tensor forward(
        const torch::Tensor& x,
        const torch::Tensor& key_value,   // may be undefined
        const torch::Tensor& mask         // may be undefined
    );
};
TORCH_MODULE(WhisperAttention);

// ─────────────────────────────────────────────────────────────────────────────
// Feed-forward MLP (4× expansion, GELU)
// ─────────────────────────────────────────────────────────────────────────────
struct WhisperMLPImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    WhisperMLPImpl(int64_t d_model);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(WhisperMLP);

// ─────────────────────────────────────────────────────────────────────────────
// Encoder block (pre-norm)
// ─────────────────────────────────────────────────────────────────────────────
struct WhisperEncoderBlockImpl : torch::nn::Module {
    torch::nn::LayerNorm  ln1{nullptr}, ln2{nullptr};
    WhisperAttention      self_attn{nullptr};
    WhisperMLP            mlp{nullptr};

    WhisperEncoderBlockImpl(int64_t d_model, int64_t n_heads);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(WhisperEncoderBlock);

// ─────────────────────────────────────────────────────────────────────────────
// Decoder block (pre-norm, masked self-attn + cross-attn)
// ─────────────────────────────────────────────────────────────────────────────
struct WhisperDecoderBlockImpl : torch::nn::Module {
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr}, ln3{nullptr};
    WhisperAttention     self_attn{nullptr};
    WhisperAttention     cross_attn{nullptr};
    WhisperMLP           mlp{nullptr};

    WhisperDecoderBlockImpl(int64_t d_model, int64_t n_heads);
    torch::Tensor forward(
        torch::Tensor x,
        const torch::Tensor& enc_out,
        const torch::Tensor& mask    // causal mask for self-attn
    );
};
TORCH_MODULE(WhisperDecoderBlock);

// ─────────────────────────────────────────────────────────────────────────────
// Whisper Encoder
// Input:  log-mel spectrogram [B, 80, T]
// Output: encoder hidden states [B, T/2, d_model]
// ─────────────────────────────────────────────────────────────────────────────
struct WhisperEncoderImpl : torch::nn::Module {
    // Conv1D stem
    torch::nn::Conv1d conv1{nullptr};  // 80 → d_model, k=3, pad=1
    torch::nn::Conv1d conv2{nullptr};  // d_model → d_model, k=3, stride=2, pad=1
    // Positional embedding (sinusoidal, not learned)
    torch::Tensor pos_emb;             // [1, max_source_positions, d_model]
    // Transformer encoder blocks
    torch::nn::ModuleList blocks{nullptr};
    torch::nn::LayerNorm  final_ln{nullptr};

    WhisperEncoderImpl(const WhisperConfig& cfg);
    torch::Tensor forward(torch::Tensor mel);  // [B, 80, T] → [B, T/2, d_model]
};
TORCH_MODULE(WhisperEncoder);

// ─────────────────────────────────────────────────────────────────────────────
// Whisper Decoder
// Input:  token_ids [B, L], encoder output [B, T, d_model]
// Output: logits [B, L, vocab_size]
// ─────────────────────────────────────────────────────────────────────────────
struct WhisperDecoderImpl : torch::nn::Module {
    torch::nn::Embedding   token_emb{nullptr};   // vocab_size × d_model
    torch::Tensor          pos_emb;              // [1, max_target_positions, d_model] (learned)
    torch::nn::ModuleList  blocks{nullptr};
    torch::nn::LayerNorm   final_ln{nullptr};
    // output projection — weight tied to token_emb.weight
    int64_t vocab_size, d_model;

    WhisperDecoderImpl(const WhisperConfig& cfg);
    // tokens: [B, L] int64   enc_out: [B, T, d_model]
    torch::Tensor forward(const torch::Tensor& tokens, const torch::Tensor& enc_out);
};
TORCH_MODULE(WhisperDecoder);

// ─────────────────────────────────────────────────────────────────────────────
// Full Whisper model
// ─────────────────────────────────────────────────────────────────────────────
struct WhisperModelImpl : torch::nn::Module {
    WhisperConfig  cfg;
    WhisperEncoder encoder{nullptr};
    WhisperDecoder decoder{nullptr};

    explicit WhisperModelImpl(const WhisperConfig& cfg = WhisperConfig::base());

    // mel: [B, 80, T]   tokens: [B, L]
    // Returns logits [B, L, vocab_size]
    torch::Tensor forward(const torch::Tensor& mel, const torch::Tensor& tokens);

    // Greedy decode (no beam search) — returns token id sequence
    std::vector<int64_t> greedy_decode(
        const torch::Tensor& mel,       // [1, 80, T]
        int64_t              sot_id,
        int64_t              eot_id,
        int64_t              max_tokens = 224);
};
TORCH_MODULE(WhisperModel);

// Convenience factories
WhisperModel make_whisper_tiny()   ;
WhisperModel make_whisper_base()   ;
WhisperModel make_whisper_small()  ;
WhisperModel make_whisper_medium() ;
WhisperModel make_whisper_large()  ;

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration (Section 2.2)
// ─────────────────────────────────────────────────────────────────────────────
struct WhisperTrainConfig {
    double  lr             = 1e-3;
    double  weight_decay   = 1e-6;
    double  beta1          = 0.9;
    double  beta2          = 0.999;
    double  grad_clip      = 1.0;    // gradient norm clipping
    int64_t warmup_steps   = 2048;   // linear lr warmup
    int64_t max_steps      = 1 << 20;
    int64_t batch_size     = 256;
    torch::Device device   = torch::kCPU;
};

// ─────────────────────────────────────────────────────────────────────────────
// Training helpers
// ─────────────────────────────────────────────────────────────────────────────
// One gradient update; returns cross-entropy loss value.
float whisper_train_step(
    WhisperModel&              model,
    torch::optim::AdamW&       optimizer,
    const torch::Tensor&       mel,       // [B, 80, T]
    const torch::Tensor&       tokens_in, // [B, L] decoder input  (teacher-forced)
    const torch::Tensor&       tokens_tgt,// [B, L] target token ids
    double                     lr_scale = 1.0);

} // namespace nlp
} // namespace models
} // namespace dm
