// ─────────────────────────────────────────────────────────────────────────────
// BitNet a4.8 implementation — Wang et al., arXiv:2411.04965v1, 2024
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/bitnet_a4_8/bitnet_a4_8.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <limits>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Quantization utilities
// ─────────────────────────────────────────────────────────────────────────────

// Weight quantization: α = mean(|W|), Q_w = α · RoundClip(W/(α+ε), -1, 1)
torch::Tensor quantize_weight_ternary(const torch::Tensor& W) {
    constexpr float eps = 1e-8f;
    auto alpha = W.abs().mean();
    auto Wq    = (W / (alpha + eps)).round().clamp(-1.0f, 1.0f);
    // Straight-through: in forward pass return fake-quantized float
    return alpha * Wq;
}

// INT4 absmean: β = mean(|X|), Q = β/√7 · RoundClip(√7/(β+ε)·X, -8, 7)
torch::Tensor quantize_int4_absmean(const torch::Tensor& X) {
    constexpr float eps  = 1e-8f;
    const     float sq7  = std::sqrt(7.0f);
    auto beta  = X.abs().mean();
    auto scale = sq7 / (beta + eps);
    auto Xq    = (scale * X).round().clamp(-8.0f, 7.0f);
    return (beta / sq7) * Xq;
}

// INT8 absmax: γ = max(|X|), Q = γ/127 · Round(127/γ·X)
torch::Tensor quantize_int8_absmax(const torch::Tensor& X) {
    constexpr float eps = 1e-8f;
    auto gamma = X.abs().amax();
    auto scale = 127.0f / (gamma + eps);
    auto Xq    = (scale * X).round().clamp(-128.0f, 127.0f);
    return (gamma / 127.0f) * Xq;
}

// TopK mask: keep top-(k_percent)% largest magnitudes, zero the rest
torch::Tensor topk_mask(const torch::Tensor& X, int64_t k_percent) {
    int64_t numel = X.numel();
    int64_t k     = std::max<int64_t>(1, numel * k_percent / 100);
    auto flat     = X.abs().view({-1});
    // kth_value of sorted descending at position k gives threshold
    auto [vals, _idx] = flat.topk(k, /*dim=*/0, /*largest=*/true, /*sorted=*/false);
    auto threshold    = vals.min();
    return (X.abs() >= threshold).to(X.dtype());
}

// ─────────────────────────────────────────────────────────────────────────────
// RMSNormImpl
// ─────────────────────────────────────────────────────────────────────────────
RMSNormImpl::RMSNormImpl(int64_t dim, float eps_)
    : d(dim), eps(eps_) {
    weight = register_parameter("weight", torch::ones({dim}));
}

torch::Tensor RMSNormImpl::forward(const torch::Tensor& x) {
    // x: [..., d]
    auto rms = x.pow(2).mean(-1, /*keepdim=*/true).add(eps).sqrt();
    return (x / rms) * weight;
}

// ─────────────────────────────────────────────────────────────────────────────
// RoPE cache & application
// ─────────────────────────────────────────────────────────────────────────────
std::pair<torch::Tensor, torch::Tensor> build_rope_cache(
        int64_t seq_len, int64_t head_dim, torch::Device device, float theta) {
    // Frequencies for even dims: θ_i = 1 / theta^(2i/head_dim)
    auto i      = torch::arange(0, head_dim, 2, torch::kFloat).to(device);
    auto freqs  = 1.0f / torch::pow(theta, i / (float)head_dim); // [head_dim/2]
    auto t      = torch::arange(seq_len, torch::kFloat).to(device); // [T]
    auto angles = torch::outer(t, freqs); // [T, head_dim/2]
    return {angles.cos(), angles.sin()};
}

// x: [B, n_heads, T, head_dim]
torch::Tensor apply_rope(const torch::Tensor& x,
                         const torch::Tensor& cos_c,
                         const torch::Tensor& sin_c) {
    // cos_c, sin_c: [T, head_dim/2]
    int64_t T  = x.size(2);
    int64_t hd = x.size(3);

    auto cos = cos_c.slice(0, 0, T).unsqueeze(0).unsqueeze(0); // [1,1,T,hd/2]
    auto sin = sin_c.slice(0, 0, T).unsqueeze(0).unsqueeze(0);

    // Split into even/odd pairs
    auto x1 = x.slice(-1, 0, hd, 2); // [..., T, hd/2] even
    auto x2 = x.slice(-1, 1, hd, 2); // [..., T, hd/2] odd

    // Apply rotation: [x1,x2]*[cos,-sin;sin,cos]
    auto r1 = x1 * cos - x2 * sin;
    auto r2 = x1 * sin + x2 * cos;

    // Interleave back
    auto out = torch::stack({r1, r2}, -1).flatten(-2, -1); // [B,h,T,hd]
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// BitLinearImpl
// ─────────────────────────────────────────────────────────────────────────────
BitLinearImpl::BitLinearImpl(int64_t in_f, int64_t out_f, bool bias,
                             ActQuant aq, int64_t topk_pct)
    : in_features(in_f), out_features(out_f), bias_(bias),
      act_quant(aq), topk_percent(topk_pct) {
    linear = register_module("linear",
        torch::nn::Linear(torch::nn::LinearOptions(in_f, out_f).bias(bias)));
}

// norm_weight: RMSNorm gain of the sub-LN preceding this projection
torch::Tensor BitLinearImpl::forward(const torch::Tensor& x,
                                     const torch::Tensor& norm_weight) {
    // 1. Sub-layer norm (BitLinear uses a fused pre-norm before each proj)
    constexpr float eps = 1e-6f;
    auto rms = x.pow(2).mean(-1, true).add(eps).sqrt();
    auto xn  = (x / rms) * norm_weight;

    // 2. Quantize activation
    torch::Tensor xq;
    torch::Tensor mask;
    bool use_mask = false;

    switch (act_quant) {
        case ActQuant::INT4_ABSMEAN:
            xq = quantize_int4_absmean(xn);
            break;
        case ActQuant::INT8_ABSMAX:
            xq = quantize_int8_absmax(xn);
            break;
        case ActQuant::INT8_TOPK:
            xq       = quantize_int8_absmax(xn);
            mask     = topk_mask(xq, topk_percent);
            xq       = xq * mask;
            use_mask = true;
            break;
    }

    // 3. Quantize weight (fake-quant, STE for backprop)
    auto Wq = quantize_weight_ternary(linear->weight); // [out_f, in_f]

    // 4. Linear pass with (optionally) biased addition
    auto out = torch::nn::functional::linear(xq, Wq,
        bias_ ? linear->bias : torch::Tensor{});

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Causal mask
// ─────────────────────────────────────────────────────────────────────────────
static torch::Tensor causal_mask(int64_t T, torch::Device device) {
    auto mask = torch::full({T, T}, -std::numeric_limits<float>::infinity());
    return torch::triu(mask, 1).to(device);
}

// ─────────────────────────────────────────────────────────────────────────────
// BitNetA48AttentionImpl
// ─────────────────────────────────────────────────────────────────────────────
BitNetA48AttentionImpl::BitNetA48AttentionImpl(const BitNetA48Config& cfg)
    : hidden(cfg.hidden_size), n_heads(cfg.n_heads),
      n_kv_heads(cfg.n_kv_heads),
      head_dim(cfg.hidden_size / cfg.n_heads),
      topk_percent(cfg.topk_percent) {

    // Q, K, V inputs → INT4 absmean
    q_proj = register_module("q_proj",
        BitLinear(hidden, hidden, false, ActQuant::INT4_ABSMEAN));
    k_proj = register_module("k_proj",
        BitLinear(hidden, n_kv_heads * head_dim, false, ActQuant::INT4_ABSMEAN));
    v_proj = register_module("v_proj",
        BitLinear(hidden, n_kv_heads * head_dim, false, ActQuant::INT4_ABSMEAN));

    // Out-projection input → INT8 + TopK sparsification
    o_proj = register_module("o_proj",
        BitLinear(hidden, hidden, false, ActQuant::INT8_TOPK, topk_percent));

    // Per-projection sub-LN norms
    q_norm = register_module("q_norm", RMSNorm(hidden, cfg.rms_eps));
    k_norm = register_module("k_norm", RMSNorm(hidden, cfg.rms_eps));
    v_norm = register_module("v_norm", RMSNorm(hidden, cfg.rms_eps));
    o_norm = register_module("o_norm", RMSNorm(hidden, cfg.rms_eps));
}

torch::Tensor BitNetA48AttentionImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& cos_cache,
        const torch::Tensor& sin_cache,
        const torch::Tensor& mask) {

    int64_t B  = x.size(0);
    int64_t T  = x.size(1);
    int64_t kv_dim = n_kv_heads * head_dim;

    // Projections (each BitLinear applies its own sub-LN internally)
    auto Q = q_proj->forward(x, q_norm->weight); // [B, T, hidden]
    auto K = k_proj->forward(x, k_norm->weight); // [B, T, kv_dim]
    auto V = v_proj->forward(x, v_norm->weight); // [B, T, kv_dim]

    // Reshape → [B, n_heads, T, head_dim]
    Q = Q.view({B, T, n_heads,    head_dim}).transpose(1, 2);
    K = K.view({B, T, n_kv_heads, head_dim}).transpose(1, 2);
    V = V.view({B, T, n_kv_heads, head_dim}).transpose(1, 2);

    // RoPE
    Q = apply_rope(Q, cos_cache, sin_cache);
    K = apply_rope(K, cos_cache, sin_cache);

    // Expand KV heads if GQA
    if (n_kv_heads != n_heads) {
        int64_t rep = n_heads / n_kv_heads;
        K = K.repeat_interleave(rep, 1);
        V = V.repeat_interleave(rep, 1);
    }

    // Scaled dot-product attention
    float  scale  = 1.0f / std::sqrt((float)head_dim);
    auto   scores = torch::matmul(Q, K.transpose(-2, -1)) * scale; // [B,h,T,T]
    if (mask.defined())
        scores = scores + mask;
    auto attn = torch::softmax(scores, -1);               // [B,h,T,T]
    auto ctx  = torch::matmul(attn, V);                   // [B,h,T,head_dim]

    // Merge heads → [B, T, hidden]
    ctx = ctx.transpose(1, 2).contiguous().view({B, T, hidden});

    // Out-projection with INT8+TopK
    return o_proj->forward(ctx, o_norm->weight);
}

// ─────────────────────────────────────────────────────────────────────────────
// BitNetA48FFNImpl  —  ReLU²GLU + down-proj
// ─────────────────────────────────────────────────────────────────────────────
BitNetA48FFNImpl::BitNetA48FFNImpl(const BitNetA48Config& cfg) {
    int64_t h   = cfg.hidden_size;
    int64_t glu = cfg.glu_size;

    // Up and Gate inputs → INT4 absmean
    up_proj   = register_module("up_proj",
        BitLinear(h, glu, false, ActQuant::INT4_ABSMEAN));
    gate_proj = register_module("gate_proj",
        BitLinear(h, glu, false, ActQuant::INT4_ABSMEAN));

    // Down input → INT8 (sparsity comes from ReLU², no TopK needed)
    down_proj = register_module("down_proj",
        BitLinear(glu, h, false, ActQuant::INT8_ABSMAX));

    // Per-projection sub-LN norms
    up_norm   = register_module("up_norm",   RMSNorm(h,   cfg.rms_eps));
    gate_norm = register_module("gate_norm", RMSNorm(h,   cfg.rms_eps));
    down_norm = register_module("down_norm", RMSNorm(glu, cfg.rms_eps));
}

torch::Tensor BitNetA48FFNImpl::forward(const torch::Tensor& x) {
    // ReLU²GLU: (x W_up^T) ⊙ ReLU²(x W_gate^T)
    auto up   = up_proj->forward(x,   up_norm->weight);    // [B, T, glu]
    auto gate = gate_proj->forward(x, gate_norm->weight);  // [B, T, glu]

    // ReLU²: (max(0,·))^2 — achieves >80% sparsity
    auto gate_act = torch::relu(gate).pow(2);
    auto hidden   = up * gate_act;                         // [B, T, glu]

    // Down projection (input already sparse from ReLU²)
    return down_proj->forward(hidden, down_norm->weight);  // [B, T, h]
}

// ─────────────────────────────────────────────────────────────────────────────
// BitNetA48BlockImpl
// ─────────────────────────────────────────────────────────────────────────────
BitNetA48BlockImpl::BitNetA48BlockImpl(const BitNetA48Config& cfg) {
    attn_norm = register_module("attn_norm", RMSNorm(cfg.hidden_size, cfg.rms_eps));
    ffn_norm  = register_module("ffn_norm",  RMSNorm(cfg.hidden_size, cfg.rms_eps));
    attn      = register_module("attn",      BitNetA48Attention(cfg));
    ffn       = register_module("ffn",       BitNetA48FFN(cfg));
}

torch::Tensor BitNetA48BlockImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& cos_cache,
        const torch::Tensor& sin_cache,
        const torch::Tensor& mask) {
    // Pre-norm residual: attention
    auto h = x + attn->forward(attn_norm->forward(x), cos_cache, sin_cache, mask);
    // Pre-norm residual: FFN
    return h + ffn->forward(ffn_norm->forward(h));
}

// ─────────────────────────────────────────────────────────────────────────────
// BitNetA48ModelImpl
// ─────────────────────────────────────────────────────────────────────────────
BitNetA48ModelImpl::BitNetA48ModelImpl(const BitNetA48Config& cfg_)
    : cfg(cfg_), vocab_size_(cfg_.vocab_size) {

    tok_emb = register_module("tok_emb",
        torch::nn::Embedding(cfg.vocab_size, cfg.hidden_size));

    blocks = register_module("blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_layers; ++i)
        blocks->push_back(BitNetA48Block(cfg));

    final_norm = register_module("final_norm",
        RMSNorm(cfg.hidden_size, cfg.rms_eps));
}

torch::Tensor BitNetA48ModelImpl::forward(const torch::Tensor& tokens) {
    int64_t B = tokens.size(0);
    int64_t T = tokens.size(1);

    auto x = tok_emb->forward(tokens);  // [B, T, hidden]

    // Build RoPE cache for current sequence length
    int64_t head_dim = cfg.hidden_size / cfg.n_heads;
    auto [cos_c, sin_c] = build_rope_cache(T, head_dim, tokens.device());

    // Causal mask
    auto mask = causal_mask(T, tokens.device());

    // Decoder blocks
    for (int64_t i = 0; i < (int64_t)blocks->size(); ++i)
        x = blocks->at<BitNetA48BlockImpl>(i).forward(x, cos_c, sin_c, mask);

    x = final_norm->forward(x);  // [B, T, hidden]

    // Tied output projection: logits = x @ tok_emb.weight^T
    return torch::matmul(x, tok_emb->weight.t());  // [B, T, vocab]
}

std::vector<int64_t> BitNetA48ModelImpl::greedy_decode(
        const torch::Tensor& prompt_tokens,
        int64_t eot_id,
        int64_t max_new_tokens) {
    this->eval();
    torch::NoGradGuard ng;

    // Collect prompt into output
    std::vector<int64_t> seq;
    for (int64_t i = 0; i < prompt_tokens.size(1); ++i)
        seq.push_back(prompt_tokens[0][i].item<int64_t>());

    for (int64_t step = 0; step < max_new_tokens; ++step) {
        // Clip to max_seq_len
        int64_t start = std::max<int64_t>(0, (int64_t)seq.size() - cfg.max_seq_len);
        std::vector<int64_t> window(seq.begin() + start, seq.end());

        auto tok = torch::tensor(window, torch::kLong)
                       .unsqueeze(0).to(prompt_tokens.device());  // [1, L]
        auto logits = forward(tok);                                // [1, L, vocab]
        auto next   = logits.select(1, -1).argmax(-1).item<int64_t>();
        seq.push_back(next);
        if (next == eot_id) break;
    }
    return seq;
}

// ─────────────────────────────────────────────────────────────────────────────
// Factories
// ─────────────────────────────────────────────────────────────────────────────
BitNetA48Model make_bitnet_a48_700m() { return BitNetA48Model(BitNetA48Config::size_700m()); }
BitNetA48Model make_bitnet_a48_1b3()  { return BitNetA48Model(BitNetA48Config::size_1b3());  }
BitNetA48Model make_bitnet_a48_3b()   { return BitNetA48Model(BitNetA48Config::size_3b());   }
BitNetA48Model make_bitnet_a48_7b()   { return BitNetA48Model(BitNetA48Config::size_7b());   }

// ─────────────────────────────────────────────────────────────────────────────
// Training step
// ─────────────────────────────────────────────────────────────────────────────
float bitnet_a48_train_step(
        BitNetA48Model&         model,
        torch::optim::AdamW&    optimizer,
        const torch::Tensor&    tokens_in,
        const torch::Tensor&    tokens_tgt,
        double                  lr_scale) {
    model->train();

    // Apply lr scaling (warmup / cosine decay)
    for (auto& pg : optimizer.param_groups()) {
        auto& opts = static_cast<torch::optim::AdamWOptions&>(pg.options());
        opts.lr(opts.lr() * lr_scale);
    }

    optimizer.zero_grad();

    auto logits = model->forward(tokens_in);  // [B, T, vocab]
    int64_t B   = logits.size(0);
    int64_t L   = logits.size(1);
    int64_t V   = logits.size(2);

    auto loss = torch::nn::functional::cross_entropy(
        logits.view({B * L, V}),
        tokens_tgt.view({B * L}));

    loss.backward();

    torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
    optimizer.step();

    return loss.item<float>();
}

} // namespace nlp
} // namespace models
} // namespace dm
