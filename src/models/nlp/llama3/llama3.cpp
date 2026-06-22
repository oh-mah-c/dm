// ─────────────────────────────────────────────────────────────────────────────
// Llama 3 implementation
//
// Llama Team, AI @ Meta, "The Llama 3 Herd of Models,"
// arXiv:2407.21783v3, 2024. https://arxiv.org/abs/2407.21783
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/llama3/llama3.h"

#include <torch/torch.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// RoPE helpers
// ─────────────────────────────────────────────────────────────────────────────

// Precompute cos/sin tables [seq_len, head_dim/2].
// Llama 3 uses θ = 500,000 (§3.2) vs 10,000 in Llama 1/2.
std::pair<torch::Tensor, torch::Tensor>
Llama3ModelImpl::_precompute_freqs(int64_t head_dim,
                                    int64_t seq_len,
                                    float   theta) {
    // freqs[i] = 1 / θ^(2i / head_dim),  i=0..head_dim/2-1
    auto d = torch::arange(0, head_dim, 2, torch::kFloat);
    auto freqs = 1.0f / torch::pow(theta, d / static_cast<float>(head_dim));
    auto t   = torch::arange(seq_len, torch::kFloat);
    auto mat = torch::outer(t, freqs); // [seq_len, head_dim/2]
    return {torch::cos(mat), torch::sin(mat)};
}

// Apply RoPE rotation to x: [B, T, n_heads, head_dim]
static torch::Tensor apply_rotary_emb(
    const torch::Tensor& x,
    const torch::Tensor& freqs_cos,  // [T, head_dim/2]
    const torch::Tensor& freqs_sin)
{
    auto xf = x.to(torch::kFloat);
    // reshape: [B, T, H, head_dim/2, 2]
    auto xs = xf.reshape({xf.size(0), xf.size(1), xf.size(2), -1, 2});
    auto xr = xs.select(-1, 0);  // real part
    auto xi = xs.select(-1, 1);  // imag part

    // broadcast: [1, T, 1, head_dim/2]
    auto fc = freqs_cos.unsqueeze(0).unsqueeze(2);
    auto fs = freqs_sin.unsqueeze(0).unsqueeze(2);

    auto out_r = xr * fc - xi * fs;
    auto out_i = xr * fs + xi * fc;

    // stack then flatten last two dims: [B, T, H, head_dim]
    auto out = torch::stack({out_r, out_i}, -1).flatten(3);
    return out.type_as(x);
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama3RMSNorm
// ─────────────────────────────────────────────────────────────────────────────
Llama3RMSNormImpl::Llama3RMSNormImpl(int64_t dim, float eps_) : eps(eps_) {
    w = register_parameter("w", torch::ones(dim));
}

torch::Tensor Llama3RMSNormImpl::forward(torch::Tensor x) {
    auto xf  = x.to(torch::kFloat);
    auto rms = xf.pow(2).mean(-1, /*keepdim=*/true).add(eps).rsqrt();
    return (xf * rms).type_as(x) * w;
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama3Attention — GQA with RoPE
// ─────────────────────────────────────────────────────────────────────────────
Llama3AttentionImpl::Llama3AttentionImpl(const Llama3Config& cfg)
    : n_heads(cfg.n_heads)
    , n_kv_heads(cfg.n_kv_heads)
    , head_dim(cfg.dim / cfg.n_heads)
    , n_rep(cfg.n_heads / cfg.n_kv_heads)
{
    wq = register_module("wq",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.n_heads    * head_dim).bias(false)));
    wk = register_module("wk",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.n_kv_heads * head_dim).bias(false)));
    wv = register_module("wv",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.n_kv_heads * head_dim).bias(false)));
    wo = register_module("wo",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.n_heads * head_dim, cfg.dim).bias(false)));

    if (cfg.dropout > 0.0f) {
        attn_drop  = register_module("attn_drop",  torch::nn::Dropout(cfg.dropout));
        resid_drop = register_module("resid_drop", torch::nn::Dropout(cfg.dropout));
    }
}

torch::Tensor Llama3AttentionImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& freqs_cos,
    const torch::Tensor& freqs_sin,
    torch::Tensor* kv_cache_k,
    torch::Tensor* kv_cache_v,
    int64_t        cache_pos)
{
    int64_t B = x.size(0), T = x.size(1);

    // Project Q, K, V
    auto q = wq->forward(x).view({B, T, n_heads,    head_dim});
    auto k = wk->forward(x).view({B, T, n_kv_heads, head_dim});
    auto v = wv->forward(x).view({B, T, n_kv_heads, head_dim});

    // Apply RoPE
    if (cache_pos >= 0) {
        // Inference: single token — slice freq tables at position
        auto fc = freqs_cos.slice(0, cache_pos, cache_pos + 1);
        auto fs = freqs_sin.slice(0, cache_pos, cache_pos + 1);
        q = apply_rotary_emb(q, fc, fs);
        k = apply_rotary_emb(k, fc, fs);
    } else {
        // Training / prefill: use first T positions
        auto fc = freqs_cos.slice(0, 0, T);
        auto fs = freqs_sin.slice(0, 0, T);
        q = apply_rotary_emb(q, fc, fs);
        k = apply_rotary_emb(k, fc, fs);
    }

    // KV-cache update (inference)
    if (kv_cache_k && kv_cache_v && cache_pos >= 0) {
        kv_cache_k->slice(1, cache_pos, cache_pos + 1).copy_(k);
        kv_cache_v->slice(1, cache_pos, cache_pos + 1).copy_(v);
        k = kv_cache_k->slice(1, 0, cache_pos + 1);
        v = kv_cache_v->slice(1, 0, cache_pos + 1);
    }

    // GQA: repeat K, V to match n_heads
    // k: [B, T_kv, n_kv_heads, head_dim] → [B, T_kv, n_heads, head_dim]
    if (n_rep > 1) {
        k = k.unsqueeze(3).expand({-1, -1, -1, n_rep, -1})
             .contiguous().view({B, k.size(1), n_heads, head_dim});
        v = v.unsqueeze(3).expand({-1, -1, -1, n_rep, -1})
             .contiguous().view({B, v.size(1), n_heads, head_dim});
    }

    // [B, n_heads, T, head_dim]
    q = q.transpose(1, 2);
    k = k.transpose(1, 2);
    v = v.transpose(1, 2);

    // Scaled dot-product attention
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    auto attn = q.matmul(k.transpose(-2, -1)) * scale; // [B, H, T, T_kv]

    // Causal mask (training / prefill only)
    if (cache_pos < 0) {
        int64_t Tkv = k.size(2);
        auto mask = torch::triu(
            torch::ones({T, Tkv}, x.options()), 1).to(torch::kBool);
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9f);
    }

    attn = torch::softmax(attn, -1);

    if (!attn_drop.is_empty() && is_training())
        attn = attn_drop->forward(attn);

    auto out = attn.matmul(v)                           // [B, H, T, head_dim]
                   .transpose(1, 2)                      // [B, T, H, head_dim]
                   .contiguous()
                   .view({B, T, n_heads * head_dim});    // [B, T, dim]

    auto h = wo->forward(out);

    if (!resid_drop.is_empty() && is_training())
        h = resid_drop->forward(h);

    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama3FFN — SwiGLU
// ─────────────────────────────────────────────────────────────────────────────
Llama3FFNImpl::Llama3FFNImpl(int64_t dim, int64_t ffn_dim, float dropout) {
    w1 = register_module("w1",
        torch::nn::Linear(torch::nn::LinearOptions(dim, ffn_dim).bias(false)));
    w2 = register_module("w2",
        torch::nn::Linear(torch::nn::LinearOptions(ffn_dim, dim).bias(false)));
    w3 = register_module("w3",
        torch::nn::Linear(torch::nn::LinearOptions(dim, ffn_dim).bias(false)));
    if (dropout > 0.0f)
        drop = register_module("drop", torch::nn::Dropout(dropout));
}

torch::Tensor Llama3FFNImpl::forward(torch::Tensor x) {
    // FFN(x) = W2( SiLU(W1·x) ⊙ W3·x )
    auto h = torch::silu(w1->forward(x)) * w3->forward(x);
    if (!drop.is_empty() && is_training())
        h = drop->forward(h);
    return w2->forward(h);
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama3Block
// ─────────────────────────────────────────────────────────────────────────────
Llama3BlockImpl::Llama3BlockImpl(const Llama3Config& cfg) {
    attention_norm = register_module("attention_norm",
        Llama3RMSNorm(cfg.dim, cfg.norm_eps));
    ffn_norm = register_module("ffn_norm",
        Llama3RMSNorm(cfg.dim, cfg.norm_eps));
    attention = register_module("attention",
        Llama3Attention(cfg));
    feed_forward = register_module("feed_forward",
        Llama3FFN(cfg.dim, cfg.ffn_dim, cfg.dropout));
}

torch::Tensor Llama3BlockImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& freqs_cos,
    const torch::Tensor& freqs_sin,
    torch::Tensor* kv_cache_k,
    torch::Tensor* kv_cache_v,
    int64_t        cache_pos)
{
    auto h   = x + attention->forward(attention_norm->forward(x),
                                      freqs_cos, freqs_sin,
                                      kv_cache_k, kv_cache_v, cache_pos);
    auto out = h + feed_forward->forward(ffn_norm->forward(h));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama3Model
// ─────────────────────────────────────────────────────────────────────────────
void Llama3ModelImpl::_init_weights() {
    // Embedding: normal, std ≈ 1/√dim (same as Llama 2)
    torch::nn::init::normal_(tok_embeddings->weight, 0.0, 0.02);

    // Output head: normal, std = 0.02
    torch::nn::init::normal_(output->weight, 0.0, 0.02);

    // Init all Linear layers inside blocks
    for (auto& block : layers->children()) {
        for (auto& item : block->named_modules()) {
            if (auto* lin = dynamic_cast<torch::nn::LinearImpl*>(item.value().get())) {
                torch::nn::init::normal_(lin->weight, 0.0, 0.02);
                if (lin->bias.defined())
                    torch::nn::init::zeros_(lin->bias);
            }
        }
    }
}

Llama3ModelImpl::Llama3ModelImpl(const Llama3Config& c) : cfg(c) {
    tok_embeddings = register_module("tok_embeddings",
        torch::nn::Embedding(cfg.vocab_size, cfg.dim));

    if (cfg.dropout > 0.0f)
        drop = register_module("drop", torch::nn::Dropout(cfg.dropout));

    layers = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_layers; ++i)
        layers->push_back(Llama3Block(cfg));

    norm = register_module("norm", Llama3RMSNorm(cfg.dim, cfg.norm_eps));

    // Separate output projection (no weight tying in Llama 3)
    output = register_module("output",
        torch::nn::Linear(
            torch::nn::LinearOptions(cfg.dim, cfg.vocab_size).bias(false)));

    // Precompute RoPE tables
    int64_t head_dim = cfg.dim / cfg.n_heads;
    auto [fc, fs] = _precompute_freqs(head_dim, cfg.seq_len, cfg.rope_theta);
    freqs_cos = register_buffer("freqs_cos", fc);
    freqs_sin = register_buffer("freqs_sin", fs);

    _init_weights();
}

torch::Tensor Llama3ModelImpl::forward(
    const torch::Tensor& tokens,
    const torch::Tensor& targets)
{
    int64_t B = tokens.size(0);
    int64_t T = tokens.size(1);

    auto x = tok_embeddings->forward(tokens); // [B, T, dim]
    if (!drop.is_empty() && is_training())
        x = drop->forward(x);

    // Slice RoPE tables to current sequence length
    auto fc = freqs_cos.slice(0, 0, T);
    auto fs = freqs_sin.slice(0, 0, T);

    for (int64_t i = 0; i < (int64_t)layers->size(); ++i)
        x = layers->at<Llama3BlockImpl>(i).forward(x, fc, fs);

    x = norm->forward(x);
    auto logits = output->forward(x); // [B, T, vocab_size]

    if (targets.defined()) {
        last_loss = torch::nn::functional::cross_entropy(
            logits.reshape({-1, cfg.vocab_size}),
            targets.reshape({-1}));
    }
    return logits;
}

torch::Tensor Llama3ModelImpl::forward_embeds(const torch::Tensor& embeds) {
    int64_t B = embeds.size(0);
    int64_t T = embeds.size(1);

    auto x = embeds; // [B, T, dim]
    if (!drop.is_empty() && is_training())
        x = drop->forward(x);

    auto fc = freqs_cos.slice(0, 0, T);
    auto fs = freqs_sin.slice(0, 0, T);

    for (int64_t i = 0; i < (int64_t)layers->size(); ++i) {
        // Run transformer block
        x = layers->at<Llama3BlockImpl>(i).forward(x, fc, fs);
    }

    x = norm->forward(x);
    // Note: We do NOT pass through `output` linear layer (LM Head), 
    // we return the final hidden state to be used by Embodied ActionHead!
    return x;
}

torch::Tensor Llama3ModelImpl::forward_one(
    int64_t token, int64_t pos,
    std::vector<torch::Tensor>& kv_caches_k,
    std::vector<torch::Tensor>& kv_caches_v)
{
    auto tokens = torch::tensor({{token}},
        torch::TensorOptions().dtype(torch::kLong).device(freqs_cos.device()));

    auto x = tok_embeddings->forward(tokens); // [1, 1, dim]

    for (int64_t i = 0; i < (int64_t)layers->size(); ++i) {
        x = layers->at<Llama3BlockImpl>(i).forward(
            x, freqs_cos, freqs_sin,
            &kv_caches_k[i], &kv_caches_v[i], pos);
    }

    x = norm->forward(x);                 // [1, 1, dim]
    return output->forward(x.squeeze(1)); // [1, vocab_size]
}

void Llama3ModelImpl::init_kv_caches(
    std::vector<torch::Tensor>& kv_caches_k,
    std::vector<torch::Tensor>& kv_caches_v) const
{
    kv_caches_k.clear();
    kv_caches_v.clear();
    int64_t head_dim = cfg.dim / cfg.n_heads;
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        kv_caches_k.push_back(torch::zeros(
            {1, cfg.seq_len, cfg.n_kv_heads, head_dim},
            freqs_cos.options()));
        kv_caches_v.push_back(torch::zeros(
            {1, cfg.seq_len, cfg.n_kv_heads, head_dim},
            freqs_cos.options()));
    }
}

std::vector<int64_t> Llama3ModelImpl::generate(
    const std::vector<int64_t>& prompt_tokens,
    int64_t max_new_tokens,
    float   temperature,
    float   top_p,
    int64_t eos_id)
{
    this->eval();
    torch::NoGradGuard ng;

    std::vector<torch::Tensor> kv_k, kv_v;
    init_kv_caches(kv_k, kv_v);

    std::vector<int64_t> tokens = prompt_tokens;
    int64_t pos = 0;

    // Prefill
    for (int64_t tok : prompt_tokens) {
        forward_one(tok, pos++, kv_k, kv_v);
    }

    // Generate
    for (int64_t s = 0; s < max_new_tokens; ++s) {
        auto logits = forward_one(tokens.back(), pos++, kv_k, kv_v); // [1, vocab]

        int64_t next_tok;
        if (temperature <= 0.0f) {
            next_tok = logits.argmax(-1).item<int64_t>();
        } else {
            auto scaled = logits / temperature;
            auto probs  = torch::softmax(scaled, -1).squeeze(0); // [vocab]

            // Top-p nucleus sampling
            auto [sorted_probs, sorted_idx] = probs.sort(-1, /*descending=*/true);
            auto cum = sorted_probs.cumsum(-1);
            // Keep tokens where cumulative prob ≤ top_p (plus first token always)
            auto mask = cum - sorted_probs > top_p;
            sorted_probs.masked_fill_(mask, 0.0f);
            sorted_probs.div_(sorted_probs.sum());

            next_tok = sorted_idx[torch::multinomial(sorted_probs, 1).item<int64_t>()]
                           .item<int64_t>();
        }

        tokens.push_back(next_tok);
        if (next_tok == eos_id) break;
        if (pos >= cfg.seq_len) break;
    }

    return tokens;
}

// ─────────────────────────────────────────────────────────────────────────────
// Convenience factories
// ─────────────────────────────────────────────────────────────────────────────
Llama3Model make_llama3_tiny()  { return Llama3Model(Llama3Config::tiny()); }
Llama3Model make_llama3_8b()    { return Llama3Model(Llama3Config::llama3_8b()); }
Llama3Model make_llama3_70b()   { return Llama3Model(Llama3Config::llama3_70b()); }
Llama3Model make_llama3_405b()  { return Llama3Model(Llama3Config::llama3_405b()); }

// ─────────────────────────────────────────────────────────────────────────────
// Training utilities
// ─────────────────────────────────────────────────────────────────────────────
torch::optim::AdamW make_llama3_optimizer(Llama3Model& model,
                                           const Llama3TrainConfig& cfg)
{
    std::vector<torch::Tensor> decay_params, nodecay_params;
    for (auto& item : model->named_parameters()) {
        auto& p = item.value();
        if (!p.requires_grad()) continue;
        if (p.dim() >= 2)
            decay_params.push_back(p);
        else
            nodecay_params.push_back(p);
    }

    std::vector<torch::optim::OptimizerParamGroup> groups;
    {
        auto opts = std::make_unique<torch::optim::AdamWOptions>(cfg.lr);
        opts->betas({cfg.beta1, cfg.beta2});
        opts->eps(cfg.eps);
        opts->weight_decay(cfg.weight_decay);
        groups.emplace_back(decay_params, std::move(opts));
    }
    {
        auto opts = std::make_unique<torch::optim::AdamWOptions>(cfg.lr);
        opts->betas({cfg.beta1, cfg.beta2});
        opts->eps(cfg.eps);
        opts->weight_decay(0.0);
        groups.emplace_back(nodecay_params, std::move(opts));
    }
    return torch::optim::AdamW(groups);
}

double llama3_lr_schedule(int64_t iter, const Llama3TrainConfig& cfg) {
    if (cfg.warmup_iters > 0 && iter < cfg.warmup_iters) {
        return cfg.lr * static_cast<double>(iter + 1) / cfg.warmup_iters;
    }
    int64_t s = iter - cfg.warmup_iters;
    int64_t T = std::max(cfg.max_iters - cfg.warmup_iters, (int64_t)1);
    double ratio = std::min(1.0, static_cast<double>(s) / T);
    // cosine: lr → min_lr
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + std::cos(M_PI * ratio));
}

float llama3_train_step(Llama3Model&             model,
                        torch::optim::AdamW&     optimizer,
                        const torch::Tensor&     tokens,
                        const Llama3TrainConfig& cfg,
                        int64_t                  iter)
{
    // Update LR
    double lr_now = llama3_lr_schedule(iter, cfg);
    for (auto& pg : optimizer.param_groups()) {
        static_cast<torch::optim::AdamWOptions&>(pg.options()).lr(lr_now);
    }

    optimizer.zero_grad();
    model->train();

    int64_t T  = tokens.size(1) - 1;
    auto inp   = tokens.slice(1, 0, T);
    auto tgt   = tokens.slice(1, 1, T + 1);

    auto logits = model->forward(inp);
    auto loss   = torch::nn::functional::cross_entropy(
                      logits.reshape({-1, model->cfg.vocab_size}),
                      tgt.reshape({-1}));
    loss.backward();

    // Gradient clipping
    torch::nn::utils::clip_grad_norm_(model->parameters(), cfg.grad_clip);

    optimizer.step();
    return loss.item<float>();
}

} // namespace nlp
} // namespace models
} // namespace dm
