// ─────────────────────────────────────────────────────────────────────────────
// Whisper implementation — Radford et al., ICML 2023 (radford23a)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/whisper/whisper.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Sinusoidal positional embedding  (same as original Transformer / Whisper)
// Returns [1, length, d_model]
// ─────────────────────────────────────────────────────────────────────────────
static torch::Tensor sinusoidal_pe(int64_t length, int64_t d_model) {
    auto positions = torch::arange(length, torch::kFloat).unsqueeze(1); // [L, 1]
    auto dims      = torch::arange(0, d_model, 2, torch::kFloat);       // [d/2]
    auto freqs     = torch::exp(-dims * (std::log(10000.0) / d_model)); // [d/2]
    auto angles    = positions * freqs;                                  // [L, d/2]
    auto pe        = torch::zeros({length, d_model});
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, torch::indexing::None, 2)},
                  torch::sin(angles));
    pe.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None, 2)},
                  torch::cos(angles));
    return pe.unsqueeze(0); // [1, L, d_model]
}

// ─────────────────────────────────────────────────────────────────────────────
// Causal mask  [T, T], fill upper triangle with -inf
// ─────────────────────────────────────────────────────────────────────────────
static torch::Tensor causal_mask(int64_t size, torch::Device device) {
    auto mask = torch::full({size, size}, -std::numeric_limits<float>::infinity());
    mask = torch::triu(mask, 1);   // keep upper triangle (excluding diagonal) as -inf
    return mask.to(device);
}

// ─────────────────────────────────────────────────────────────────────────────
// WhisperAttentionImpl
// ─────────────────────────────────────────────────────────────────────────────
WhisperAttentionImpl::WhisperAttentionImpl(int64_t d_model_, int64_t n_heads_)
    : d_model(d_model_), n_heads(n_heads_), head_dim(d_model_ / n_heads_) {
    q_proj   = register_module("q_proj",   torch::nn::Linear(d_model, d_model));
    k_proj   = register_module("k_proj",
        torch::nn::Linear(torch::nn::LinearOptions(d_model, d_model).bias(false)));
    v_proj   = register_module("v_proj",   torch::nn::Linear(d_model, d_model));
    out_proj = register_module("out_proj", torch::nn::Linear(d_model, d_model));
}

torch::Tensor WhisperAttentionImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& key_value,
        const torch::Tensor& mask) {

    int64_t B  = x.size(0);
    int64_t Tq = x.size(1);

    // For cross-attention use encoder output for K/V, else self-attention
    const auto& kv_src = key_value.defined() ? key_value : x;
    int64_t Tk = kv_src.size(1);

    // Project
    auto Q = q_proj->forward(x);        // [B, Tq, d_model]
    auto K = k_proj->forward(kv_src);   // [B, Tk, d_model]
    auto V = v_proj->forward(kv_src);   // [B, Tk, d_model]

    // Reshape to [B, heads, T, head_dim]
    auto reshape = [&](torch::Tensor t, int64_t T) {
        return t.view({B, T, n_heads, head_dim}).transpose(1, 2);
    };
    Q = reshape(Q, Tq);
    K = reshape(K, Tk);
    V = reshape(V, Tk);

    // Scaled dot-product attention
    float scale = 1.0f / std::sqrt((float)head_dim);
    auto scores = torch::matmul(Q, K.transpose(-2, -1)) * scale; // [B, h, Tq, Tk]

    if (mask.defined()) {
        // mask: [Tq, Tk] or [B, h, Tq, Tk]
        scores = scores + mask;
    }

    auto attn   = torch::softmax(scores, -1);                     // [B, h, Tq, Tk]
    auto ctx    = torch::matmul(attn, V);                         // [B, h, Tq, head_dim]
    ctx = ctx.transpose(1, 2).contiguous().view({B, Tq, d_model}); // [B, Tq, d_model]

    return out_proj->forward(ctx);
}

// ─────────────────────────────────────────────────────────────────────────────
// WhisperMLPImpl
// ─────────────────────────────────────────────────────────────────────────────
WhisperMLPImpl::WhisperMLPImpl(int64_t d_model) {
    fc1 = register_module("fc1", torch::nn::Linear(d_model, d_model * 4));
    fc2 = register_module("fc2", torch::nn::Linear(d_model * 4, d_model));
}

torch::Tensor WhisperMLPImpl::forward(torch::Tensor x) {
    x = fc1->forward(x);
    x = torch::gelu(x);
    x = fc2->forward(x);
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// WhisperEncoderBlockImpl
// ─────────────────────────────────────────────────────────────────────────────
WhisperEncoderBlockImpl::WhisperEncoderBlockImpl(int64_t d_model, int64_t n_heads) {
    ln1       = register_module("ln1",       torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    ln2       = register_module("ln2",       torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    self_attn = register_module("self_attn", WhisperAttention(d_model, n_heads));
    mlp       = register_module("mlp",       WhisperMLP(d_model));
}

torch::Tensor WhisperEncoderBlockImpl::forward(torch::Tensor x) {
    // Pre-norm residual: self-attention
    auto residual = x;
    x = ln1->forward(x);
    x = self_attn->forward(x, torch::Tensor{}, torch::Tensor{});
    x = x + residual;
    // Pre-norm residual: MLP
    residual = x;
    x = ln2->forward(x);
    x = mlp->forward(x);
    x = x + residual;
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// WhisperDecoderBlockImpl
// ─────────────────────────────────────────────────────────────────────────────
WhisperDecoderBlockImpl::WhisperDecoderBlockImpl(int64_t d_model, int64_t n_heads) {
    ln1        = register_module("ln1",        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    ln2        = register_module("ln2",        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    ln3        = register_module("ln3",        torch::nn::LayerNorm(torch::nn::LayerNormOptions({d_model})));
    self_attn  = register_module("self_attn",  WhisperAttention(d_model, n_heads));
    cross_attn = register_module("cross_attn", WhisperAttention(d_model, n_heads));
    mlp        = register_module("mlp",        WhisperMLP(d_model));
}

torch::Tensor WhisperDecoderBlockImpl::forward(
        torch::Tensor x,
        const torch::Tensor& enc_out,
        const torch::Tensor& mask) {
    // Masked self-attention (causal)
    auto residual = x;
    x = ln1->forward(x);
    x = self_attn->forward(x, torch::Tensor{}, mask);
    x = x + residual;
    // Cross-attention (encoder output as K/V)
    residual = x;
    x = ln2->forward(x);
    x = cross_attn->forward(x, enc_out, torch::Tensor{});
    x = x + residual;
    // MLP
    residual = x;
    x = ln3->forward(x);
    x = mlp->forward(x);
    x = x + residual;
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// WhisperEncoderImpl
// ─────────────────────────────────────────────────────────────────────────────
WhisperEncoderImpl::WhisperEncoderImpl(const WhisperConfig& cfg) {
    // Convolutional stem
    conv1 = register_module("conv1",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(cfg.n_mels, cfg.d_model, 3).padding(1)));
    conv2 = register_module("conv2",
        torch::nn::Conv1d(torch::nn::Conv1dOptions(cfg.d_model, cfg.d_model, 3).stride(2).padding(1)));

    // Sinusoidal positional embedding (fixed, not a parameter)
    pos_emb = sinusoidal_pe(cfg.max_source_positions, cfg.d_model);
    register_buffer("pos_emb", pos_emb);

    // Transformer encoder blocks
    blocks = register_module("blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.enc_layers; ++i)
        blocks->push_back(WhisperEncoderBlock(cfg.d_model, cfg.n_heads));

    final_ln = register_module("final_ln",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.d_model})));
}

torch::Tensor WhisperEncoderImpl::forward(torch::Tensor mel) {
    // mel: [B, 80, T]
    // Stem
    auto x = torch::gelu(conv1->forward(mel));   // [B, d_model, T]
    x      = torch::gelu(conv2->forward(x));     // [B, d_model, T/2]
    x      = x.transpose(1, 2);                  // [B, T/2, d_model]

    // Add sinusoidal PE (truncate to actual length)
    int64_t T = x.size(1);
    x = x + pos_emb.to(x.device()).slice(1, 0, T);

    // Transformer blocks
    for (int64_t i = 0; i < (int64_t)blocks->size(); ++i)
        x = blocks->at<WhisperEncoderBlockImpl>(i).forward(x);

    return final_ln->forward(x);  // [B, T/2, d_model]
}

// ─────────────────────────────────────────────────────────────────────────────
// WhisperDecoderImpl
// ─────────────────────────────────────────────────────────────────────────────
WhisperDecoderImpl::WhisperDecoderImpl(const WhisperConfig& cfg)
    : vocab_size(cfg.vocab_size), d_model(cfg.d_model) {

    token_emb = register_module("token_emb",
        torch::nn::Embedding(cfg.vocab_size, cfg.d_model));

    // Learned positional embeddings
    pos_emb = register_parameter("pos_emb",
        torch::zeros({1, cfg.max_target_positions, cfg.d_model}));

    blocks = register_module("blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.dec_layers; ++i)
        blocks->push_back(WhisperDecoderBlock(cfg.d_model, cfg.n_heads));

    final_ln = register_module("final_ln",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({cfg.d_model})));
}

torch::Tensor WhisperDecoderImpl::forward(
        const torch::Tensor& tokens,
        const torch::Tensor& enc_out) {
    int64_t L = tokens.size(1);

    // Embed tokens + positional embedding
    auto x = token_emb->forward(tokens);                       // [B, L, d_model]
    x = x + pos_emb.to(x.device()).slice(1, 0, L);

    // Causal mask
    auto mask = causal_mask(L, x.device());

    // Decoder blocks
    for (int64_t i = 0; i < (int64_t)blocks->size(); ++i)
        x = blocks->at<WhisperDecoderBlockImpl>(i).forward(x, enc_out, mask);

    x = final_ln->forward(x);  // [B, L, d_model]

    // Tied output projection: logits = x @ token_emb.weight.T
    auto logits = torch::matmul(x, token_emb->weight.t()); // [B, L, vocab_size]
    return logits;
}

// ─────────────────────────────────────────────────────────────────────────────
// WhisperModelImpl
// ─────────────────────────────────────────────────────────────────────────────
WhisperModelImpl::WhisperModelImpl(const WhisperConfig& cfg_) : cfg(cfg_) {
    encoder = register_module("encoder", WhisperEncoder(cfg));
    decoder = register_module("decoder", WhisperDecoder(cfg));
}

torch::Tensor WhisperModelImpl::forward(
        const torch::Tensor& mel,
        const torch::Tensor& tokens) {
    auto enc_out = encoder->forward(mel);
    return decoder->forward(tokens, enc_out);
}

std::vector<int64_t> WhisperModelImpl::greedy_decode(
        const torch::Tensor& mel,
        int64_t sot_id,
        int64_t eot_id,
        int64_t max_tokens) {
    this->eval();
    torch::NoGradGuard ng;

    auto enc_out = encoder->forward(mel.unsqueeze(0));  // [1, T/2, d]

    std::vector<int64_t> seq = {sot_id};
    for (int64_t step = 0; step < max_tokens; ++step) {
        auto tok_tensor = torch::tensor(seq, torch::kLong)
                              .unsqueeze(0).to(mel.device()); // [1, L]
        auto logits = decoder->forward(tok_tensor, enc_out); // [1, L, vocab]
        auto next_id = logits.select(1, -1).argmax(-1).item<int64_t>();
        seq.push_back(next_id);
        if (next_id == eot_id) break;
    }
    return seq;
}

// ─────────────────────────────────────────────────────────────────────────────
// Factories
// ─────────────────────────────────────────────────────────────────────────────
WhisperModel make_whisper_tiny()   { return WhisperModel(WhisperConfig::tiny());   }
WhisperModel make_whisper_base()   { return WhisperModel(WhisperConfig::base());   }
WhisperModel make_whisper_small()  { return WhisperModel(WhisperConfig::small());  }
WhisperModel make_whisper_medium() { return WhisperModel(WhisperConfig::medium()); }
WhisperModel make_whisper_large()  { return WhisperModel(WhisperConfig::large());  }

// ─────────────────────────────────────────────────────────────────────────────
// Training step
// ─────────────────────────────────────────────────────────────────────────────
float whisper_train_step(
        WhisperModel&          model,
        torch::optim::AdamW&   optimizer,
        const torch::Tensor&   mel,
        const torch::Tensor&   tokens_in,
        const torch::Tensor&   tokens_tgt,
        double                 lr_scale) {
    model->train();

    // Apply lr scale (warmup / decay)
    for (auto& pg : optimizer.param_groups()) {
        auto& opts = static_cast<torch::optim::AdamWOptions&>(pg.options());
        opts.lr(opts.lr() * lr_scale);
    }

    optimizer.zero_grad();
    auto logits = model->forward(mel, tokens_in);  // [B, L, vocab]

    int64_t B = logits.size(0);
    int64_t L = logits.size(1);
    int64_t V = logits.size(2);

    auto loss = torch::nn::functional::cross_entropy(
        logits.view({B * L, V}),
        tokens_tgt.view({B * L}));
    loss.backward();

    // Gradient norm clipping (Section 2.2)
    torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);

    optimizer.step();
    return loss.item<float>();
}

} // namespace nlp
} // namespace models
} // namespace dm
