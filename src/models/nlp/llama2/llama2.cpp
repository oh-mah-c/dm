// ─────────────────────────────────────────────────────────────────────────────
// Llama 2 implementation — Touvron et al., arXiv:2307.09288, 2023
// Based on karpathy/llama2.c (https://github.com/karpathy/llama2.c)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/llama2/llama2.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <stdexcept>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Helper: round up to nearest multiple
// ─────────────────────────────────────────────────────────────────────────────
static int64_t round_up(int64_t n, int64_t mult) {
    return ((n + mult - 1) / mult) * mult;
}

// ─────────────────────────────────────────────────────────────────────────────
// RoPE: precompute cos/sin frequency table
// Returns (freqs_cos, freqs_sin) each [seq_len, head_dim/2]
// ─────────────────────────────────────────────────────────────────────────────
static std::pair<torch::Tensor, torch::Tensor>
precompute_freqs_cis(int64_t head_dim, int64_t seq_len, float theta = 10000.0f) {
    // freqs: [head_dim/2]
    auto d = torch::arange(0, head_dim, 2, torch::kFloat);
    auto freqs = 1.0f / torch::pow(theta, d / (float)head_dim); // [head_dim/2]
    // positions: [seq_len]
    auto t = torch::arange(seq_len, torch::kFloat);
    // outer product: [seq_len, head_dim/2]
    auto mat = torch::outer(t, freqs);
    return {torch::cos(mat), torch::sin(mat)};
}

// Apply RoPE rotation to x: [B, T, n_heads, head_dim]
static torch::Tensor apply_rotary_emb(
        const torch::Tensor& x,
        const torch::Tensor& freqs_cos,  // [T, head_dim/2]
        const torch::Tensor& freqs_sin)  // [T, head_dim/2]
{
    // Split last dim into pairs
    auto xf = x.to(torch::kFloat);
    auto xs = xf.reshape({xf.size(0), xf.size(1), xf.size(2), -1, 2});
    auto xr = xs.select(-1, 0);  // [B, T, H, head_dim/2]
    auto xi = xs.select(-1, 1);

    // Broadcast freqs: [1, T, 1, head_dim/2]
    auto fc = freqs_cos.unsqueeze(0).unsqueeze(2);
    auto fs = freqs_sin.unsqueeze(0).unsqueeze(2);

    auto out_r = xr * fc - xi * fs;
    auto out_i = xr * fs + xi * fc;

    // stack and flatten: [B, T, H, head_dim]
    std::vector<torch::Tensor> parts = {out_r, out_i};
    auto out = torch::stack(parts, -1).flatten(3);
    return out.type_as(x);
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama2RMSNormImpl
// ─────────────────────────────────────────────────────────────────────────────
Llama2RMSNormImpl::Llama2RMSNormImpl(int64_t dim, float eps_) : eps(eps_) {
    w = register_parameter("w", torch::ones(dim));
}

torch::Tensor Llama2RMSNormImpl::forward(torch::Tensor x) {
    // x: [B, T, dim]
    auto xf  = x.to(torch::kFloat);
    auto rms = xf.pow(2).mean(-1, /*keepdim=*/true).add(eps).rsqrt();
    return (xf * rms).type_as(x) * w;
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama2AttentionImpl
// ─────────────────────────────────────────────────────────────────────────────
Llama2AttentionImpl::Llama2AttentionImpl(const Llama2Config& cfg)
    : n_heads(cfg.n_heads),
      n_kv_heads(cfg.n_kv_heads),
      head_dim(cfg.dim / cfg.n_heads),
      n_rep(cfg.n_heads / cfg.n_kv_heads)
{
    using Opt = torch::nn::LinearOptions;
    wq = register_module("wq", torch::nn::Linear(Opt(cfg.dim, cfg.n_heads   * head_dim).bias(false)));
    wk = register_module("wk", torch::nn::Linear(Opt(cfg.dim, cfg.n_kv_heads * head_dim).bias(false)));
    wv = register_module("wv", torch::nn::Linear(Opt(cfg.dim, cfg.n_kv_heads * head_dim).bias(false)));
    wo = register_module("wo", torch::nn::Linear(Opt(cfg.n_heads * head_dim, cfg.dim).bias(false)));
    attn_drop  = register_module("attn_drop",  torch::nn::Dropout(cfg.dropout));
    resid_drop = register_module("resid_drop", torch::nn::Dropout(cfg.dropout));
}

torch::Tensor Llama2AttentionImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& freqs_cos,
        const torch::Tensor& freqs_sin,
        torch::Tensor*       kv_cache_k,
        torch::Tensor*       kv_cache_v,
        int64_t              cache_pos)
{
    int64_t B = x.size(0), T = x.size(1);
    auto dev = x.device();

    // Project Q, K, V
    auto xq = wq->forward(x).view({B, T, n_heads,    head_dim}); // [B, T, Hq, D]
    auto xk = wk->forward(x).view({B, T, n_kv_heads, head_dim}); // [B, T, Hk, D]
    auto xv = wv->forward(x).view({B, T, n_kv_heads, head_dim}); // [B, T, Hk, D]

    // Apply RoPE
    xq = apply_rotary_emb(xq, freqs_cos, freqs_sin);
    xk = apply_rotary_emb(xk, freqs_cos, freqs_sin);

    // KV cache (inference mode)
    int64_t cache_len = 0;
    if (kv_cache_k && kv_cache_v && cache_pos >= 0) {
        // Write current K/V into cache at position cache_pos
        kv_cache_k->select(1, cache_pos).copy_(xk.squeeze(1));
        kv_cache_v->select(1, cache_pos).copy_(xv.squeeze(1));
        // Use all cached K/V up to (and including) current position
        cache_len = cache_pos + 1;
        xk = kv_cache_k->slice(1, 0, cache_len);  // [B, cache_len, Hk, D]
        xv = kv_cache_v->slice(1, 0, cache_len);  // [B, cache_len, Hk, D]
    }

    // GQA: expand K/V to match Q head count
    if (n_rep > 1) {
        int64_t kT = xk.size(1);
        xk = xk.unsqueeze(3).expand({B, kT, n_kv_heads, n_rep, head_dim})
                .reshape({B, kT, n_heads, head_dim});
        xv = xv.unsqueeze(3).expand({B, kT, n_kv_heads, n_rep, head_dim})
                .reshape({B, kT, n_heads, head_dim});
    }

    // Transpose to [B, H, T, D] for batched matmul
    xq = xq.transpose(1, 2);
    xk = xk.transpose(1, 2);
    xv = xv.transpose(1, 2);

    // Scaled dot-product attention
    float scale = 1.0f / std::sqrt((float)head_dim);
    auto scores = torch::matmul(xq, xk.transpose(-2, -1)) * scale; // [B, H, T, kT]

    // Causal mask (only needed in training / non-cache mode)
    if (cache_pos < 0 && T > 1) {
        auto mask = torch::full({T, T}, -std::numeric_limits<float>::infinity(),
                                x.options()).triu(1);
        scores = scores + mask;
    }

    auto attn = torch::softmax(scores.to(torch::kFloat), -1).type_as(x);
    attn = attn_drop->forward(attn);

    auto out = torch::matmul(attn, xv);  // [B, H, T, D]
    out = out.transpose(1, 2).contiguous().view({B, T, n_heads * head_dim});
    out = wo->forward(out);
    out = resid_drop->forward(out);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama2FFNImpl  (SwiGLU)
// ─────────────────────────────────────────────────────────────────────────────
Llama2FFNImpl::Llama2FFNImpl(int64_t dim, int64_t hidden_dim_, float dropout) {
    // Scale hidden_dim: 2/3 × 4d, rounded to multiple of 256
    if (hidden_dim_ == 0) {
        hidden_dim_ = round_up((int64_t)(2.0 * 4 * dim / 3), 256);
    }
    using Opt = torch::nn::LinearOptions;
    w1 = register_module("w1", torch::nn::Linear(Opt(dim, hidden_dim_).bias(false)));
    w2 = register_module("w2", torch::nn::Linear(Opt(hidden_dim_, dim).bias(false)));
    w3 = register_module("w3", torch::nn::Linear(Opt(dim, hidden_dim_).bias(false)));
    drop = register_module("drop", torch::nn::Dropout(dropout));
}

torch::Tensor Llama2FFNImpl::forward(torch::Tensor x) {
    // SwiGLU: W2( SiLU(W1·x) ⊙ W3·x )
    return drop->forward(w2->forward(torch::silu(w1->forward(x)) * w3->forward(x)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama2BlockImpl
// ─────────────────────────────────────────────────────────────────────────────
Llama2BlockImpl::Llama2BlockImpl(const Llama2Config& cfg) {
    attention_norm = register_module("attention_norm", Llama2RMSNorm(cfg.dim, cfg.norm_eps));
    ffn_norm       = register_module("ffn_norm",       Llama2RMSNorm(cfg.dim, cfg.norm_eps));
    attention      = register_module("attention",      Llama2Attention(cfg));
    feed_forward   = register_module("feed_forward",   Llama2FFN(cfg.dim, cfg.hidden_dim, cfg.dropout));
}

torch::Tensor Llama2BlockImpl::forward(
        const torch::Tensor& x,
        const torch::Tensor& freqs_cos,
        const torch::Tensor& freqs_sin,
        torch::Tensor*       kv_cache_k,
        torch::Tensor*       kv_cache_v,
        int64_t              cache_pos)
{
    auto h = x + attention->forward(attention_norm->forward(x),
                                    freqs_cos, freqs_sin,
                                    kv_cache_k, kv_cache_v, cache_pos);
    return h + feed_forward->forward(ffn_norm->forward(h));
}

// ─────────────────────────────────────────────────────────────────────────────
// Llama2ModelImpl
// ─────────────────────────────────────────────────────────────────────────────
Llama2ModelImpl::Llama2ModelImpl(const Llama2Config& cfg_) : cfg(cfg_) {
    tok_embeddings = register_module("tok_embeddings",
        torch::nn::Embedding(cfg.vocab_size, cfg.dim));
    drop   = register_module("drop",   torch::nn::Dropout(cfg.dropout));
    layers = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_layers; ++i)
        layers->push_back(Llama2Block(cfg));
    norm   = register_module("norm",   Llama2RMSNorm(cfg.dim, cfg.norm_eps));
    output = register_module("output",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.vocab_size).bias(false)));

    // Tie output projection to token embedding (Press & Wolf 2017)
    output->weight = tok_embeddings->weight;

    // Pre-compute RoPE frequencies
    int64_t head_dim = cfg.dim / cfg.n_heads;
    auto rope = precompute_freqs_cis(head_dim, cfg.seq_len);
    freqs_cos = register_buffer("freqs_cos", rope.first);
    freqs_sin = register_buffer("freqs_sin", rope.second);

    _init_weights();
}

void Llama2ModelImpl::_init_weights() {
    for (auto& mod : modules(/*include_self=*/false)) {
        if (auto* lin = mod->as<torch::nn::LinearImpl>()) {
            torch::nn::init::normal_(lin->weight, 0.0, 0.02);
            if (lin->bias.defined())
                torch::nn::init::zeros_(lin->bias);
        } else if (auto* emb = mod->as<torch::nn::EmbeddingImpl>()) {
            torch::nn::init::normal_(emb->weight, 0.0, 0.02);
        }
    }
    // Scaled init for residual projections (GPT-2 style)
    double std_scale = 0.02 / std::sqrt(2.0 * cfg.n_layers);
    for (auto& item : named_parameters()) {
        if (item.key().find("wo.weight") != std::string::npos ||
            item.key().find("w3.weight") != std::string::npos) {
            torch::nn::init::normal_(item.value(), 0.0, std_scale);
        }
    }
}

torch::Tensor Llama2ModelImpl::forward(const torch::Tensor& tokens,
                                       const torch::Tensor& targets) {
    int64_t B = tokens.size(0), T = tokens.size(1);
    auto dev = tokens.device();

    auto h = tok_embeddings->forward(tokens);  // [B, T, dim]
    h = drop->forward(h);

    auto fc = freqs_cos.to(dev).slice(0, 0, T);
    auto fs = freqs_sin.to(dev).slice(0, 0, T);

    for (int64_t i = 0; i < (int64_t)layers->size(); ++i)
        h = layers->at<Llama2BlockImpl>(i).forward(h, fc, fs);

    h = norm->forward(h);  // [B, T, dim]

    torch::Tensor logits;
    if (targets.defined()) {
        logits = output->forward(h);  // [B, T, vocab]
        last_loss = torch::nn::functional::cross_entropy(
            logits.view({B * T, cfg.vocab_size}),
            targets.reshape({B * T}),
            torch::nn::functional::CrossEntropyFuncOptions().ignore_index(-1));
    } else {
        // inference: only last token
        logits = output->forward(h.select(1, -1).unsqueeze(1)); // [B, 1, vocab]
        last_loss = torch::Tensor{};
    }
    return logits;
}

void Llama2ModelImpl::init_kv_caches(std::vector<torch::Tensor>& kc_k,
                                     std::vector<torch::Tensor>& kc_v) const {
    int64_t head_dim = cfg.dim / cfg.n_heads;
    kc_k.clear(); kc_v.clear();
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        kc_k.push_back(torch::zeros({1, cfg.seq_len, cfg.n_kv_heads, head_dim}));
        kc_v.push_back(torch::zeros({1, cfg.seq_len, cfg.n_kv_heads, head_dim}));
    }
}

torch::Tensor Llama2ModelImpl::forward_one(int64_t        token,
                                           int64_t        pos,
                                           std::vector<torch::Tensor>& kc_k,
                                           std::vector<torch::Tensor>& kc_v) {
    auto dev = parameters().front().device();
    auto tok_t = torch::tensor({{token}}, torch::TensorOptions().dtype(torch::kLong).device(dev));

    auto h = tok_embeddings->forward(tok_t);  // [1, 1, dim]

    auto fc = freqs_cos.to(dev).slice(0, pos, pos + 1);
    auto fs = freqs_sin.to(dev).slice(0, pos, pos + 1);

    for (int64_t i = 0; i < (int64_t)layers->size(); ++i)
        h = layers->at<Llama2BlockImpl>(i).forward(
                h, fc, fs, &kc_k[i], &kc_v[i], pos);

    h = norm->forward(h);                      // [1, 1, dim]
    return output->forward(h.squeeze(1));      // [1, vocab]
}

std::vector<int64_t> Llama2ModelImpl::generate(
        const std::vector<int64_t>& prompt_tokens,
        int64_t max_new_tokens,
        float   temperature,
        float   top_p,
        int64_t eos_id)
{
    this->eval();
    torch::NoGradGuard ng;

    std::vector<torch::Tensor> kc_k, kc_v;
    init_kv_caches(kc_k, kc_v);

    auto dev = parameters().front().device();
    std::vector<int64_t> seq(prompt_tokens);

    // Prefill: feed the prompt
    for (int64_t pos = 0; pos < (int64_t)prompt_tokens.size() - 1; ++pos)
        forward_one(prompt_tokens[pos], pos, kc_k, kc_v);

    // Decode
    int64_t cur_pos = (int64_t)prompt_tokens.size() - 1;
    int64_t cur_tok = prompt_tokens.back();

    for (int64_t step = 0; step < max_new_tokens; ++step) {
        auto logits = forward_one(cur_tok, cur_pos, kc_k, kc_v); // [1, vocab]
        ++cur_pos;

        int64_t next;
        if (temperature == 0.0f) {
            next = logits.argmax(-1).item<int64_t>();
        } else {
            auto lf = logits.squeeze(0).to(torch::kFloat) / temperature; // [vocab]
            // top-p nucleus sampling
            if (top_p > 0.0f && top_p < 1.0f) {
                auto [sorted, idx] = lf.sort(-1, /*descending=*/true);
                auto probs = torch::softmax(sorted, -1);
                auto cum   = probs.cumsum(-1);
                // zero out tokens beyond top_p cutoff
                auto mask  = cum - probs > top_p;
                sorted.masked_fill_(mask, -std::numeric_limits<float>::infinity());
                lf = torch::zeros_like(lf).scatter_(-1, idx, sorted);
            }
            auto probs = torch::softmax(lf, -1);
            next = torch::multinomial(probs, 1).item().toLong();
        }
        seq.push_back(next);
        if (next == eos_id) break;
        cur_tok = next;
    }
    return seq;
}

// ─────────────────────────────────────────────────────────────────────────────
// Factories
// ─────────────────────────────────────────────────────────────────────────────
Llama2Model make_llama2_stories110k() { return Llama2Model(Llama2Config::stories110k()); }
Llama2Model make_llama2_7b()          { return Llama2Model(Llama2Config::llama_7b());    }
Llama2Model make_llama2_13b()         { return Llama2Model(Llama2Config::llama_13b());   }
Llama2Model make_llama2_70b()         { return Llama2Model(Llama2Config::llama_70b());   }

// ─────────────────────────────────────────────────────────────────────────────
// Training helpers
// ─────────────────────────────────────────────────────────────────────────────
torch::optim::AdamW make_llama2_optimizer(Llama2Model& model,
                                          const Llama2TrainConfig& cfg) {
    // Separate 2-D (weight decay) vs 1-D (no weight decay) params
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

double llama2_lr_schedule(int64_t iter, const Llama2TrainConfig& cfg) {
    if (iter < cfg.warmup_iters)
        return (double)(iter + 1) / (double)cfg.warmup_iters;
    if (iter > cfg.max_iters)
        return cfg.min_lr / cfg.lr;
    double progress = (double)(iter - cfg.warmup_iters) /
                      (double)(cfg.max_iters - cfg.warmup_iters);
    double cosine   = 0.5 * (1.0 + std::cos(M_PI * progress));
    return cfg.min_lr / cfg.lr + cosine * (1.0 - cfg.min_lr / cfg.lr);
}

float llama2_train_step(Llama2Model&           model,
                        torch::optim::AdamW&   optimizer,
                        const torch::Tensor&   data,
                        const Llama2TrainConfig& cfg,
                        int64_t                iter) {
    model->train();

    // Apply LR schedule
    double scale = llama2_lr_schedule(iter, cfg);
    for (auto& pg : optimizer.param_groups()) {
        auto& opts = static_cast<torch::optim::AdamWOptions&>(pg.options());
        opts.lr(cfg.lr * scale);
    }

    // Slice: tokens = data[:, :-1], targets = data[:, 1:]
    int64_t T1 = data.size(1) - 1;
    auto tokens  = data.slice(1, 0, T1);
    auto targets = data.slice(1, 1, T1 + 1);

    optimizer.zero_grad();
    model->forward(tokens, targets);
    auto loss = model->last_loss;
    loss.backward();
    torch::nn::utils::clip_grad_norm_(model->parameters(), cfg.grad_clip);
    optimizer.step();
    return loss.item<float>();
}

} // namespace nlp
} // namespace models
} // namespace dm
