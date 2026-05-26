// ─────────────────────────────────────────────────────────────────────────────
// OhmC1 implementation
//
// oh-mah-c, "OhmC1: A dm-Native Autoregressive Language Model Family with
// QK-Normalization and Sandwich-Norm Blocks," dm repository, 2026. [151]
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/ohmc1/ohmc1.h"

#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// OhmC1Config presets
// ─────────────────────────────────────────────────────────────────────────────

OhmC1Config OhmC1Config::tiny() {
    OhmC1Config c;
    c.dim        = 256;
    c.ffn_dim    = 768;
    c.n_layers   = 2;
    c.n_heads    = 4;
    c.n_kv_heads = 1;     // MQA for tiny
    c.seq_len    = 128;
    c.rope_theta = 10000.0f;
    return c;
}

OhmC1Config OhmC1Config::small() {
    OhmC1Config c;
    c.dim        = 768;
    c.ffn_dim    = 2048;
    c.n_layers   = 12;
    c.n_heads    = 12;
    c.n_kv_heads = 2;
    c.seq_len    = 2048;
    c.rope_theta = 50000.0f;
    return c;
}

OhmC1Config OhmC1Config::medium() {
    // default constructor values already define medium
    return OhmC1Config{};
}

OhmC1Config OhmC1Config::large() {
    OhmC1Config c;
    c.dim        = 4096;
    c.ffn_dim    = 11008;
    c.n_layers   = 32;
    c.n_heads    = 32;
    c.n_kv_heads = 8;
    c.seq_len    = 8192;
    c.rope_theta = 500000.0f;
    return c;
}

// ─────────────────────────────────────────────────────────────────────────────
// RoPE helpers (same math as Llama 3)
// ─────────────────────────────────────────────────────────────────────────────

std::pair<torch::Tensor, torch::Tensor>
OhmC1ModelImpl::_precompute_freqs(int64_t head_dim, int64_t seq_len, float theta) {
    auto d    = torch::arange(0, head_dim, 2, torch::kFloat);
    auto freq = 1.0f / torch::pow(theta, d / static_cast<float>(head_dim));
    auto t    = torch::arange(seq_len, torch::kFloat);
    auto mat  = torch::outer(t, freq);  // [seq_len, head_dim/2]
    return {torch::cos(mat), torch::sin(mat)};
}

static torch::Tensor apply_rotary_emb(
    const torch::Tensor& x,
    const torch::Tensor& freqs_cos,
    const torch::Tensor& freqs_sin)
{
    auto xf = x.to(torch::kFloat);
    auto xs = xf.reshape({xf.size(0), xf.size(1), xf.size(2), -1, 2});
    auto xr = xs.select(-1, 0);
    auto xi = xs.select(-1, 1);

    auto fc = freqs_cos.unsqueeze(0).unsqueeze(2);
    auto fs = freqs_sin.unsqueeze(0).unsqueeze(2);

    auto out_r = xr * fc - xi * fs;
    auto out_i = xr * fs + xi * fc;
    return torch::stack({out_r, out_i}, -1).flatten(3).type_as(x);
}

// ─────────────────────────────────────────────────────────────────────────────
// OhmC1RMSNorm
// ─────────────────────────────────────────────────────────────────────────────

OhmC1RMSNormImpl::OhmC1RMSNormImpl(int64_t dim, float eps_) : eps(eps_) {
    w = register_parameter("w", torch::ones(dim));
}

torch::Tensor OhmC1RMSNormImpl::forward(torch::Tensor x) {
    auto xf  = x.to(torch::kFloat);
    auto rms = xf.pow(2).mean(-1, /*keepdim=*/true).add(eps).rsqrt();
    return (xf * rms).type_as(x) * w;
}

// ─────────────────────────────────────────────────────────────────────────────
// OhmC1QKNorm
// ─────────────────────────────────────────────────────────────────────────────

OhmC1QKNormImpl::OhmC1QKNormImpl(int64_t n_heads, int64_t head_dim, float eps_)
    : n_heads_(n_heads), head_dim_(head_dim), eps(eps_)
{
    w = register_parameter("w", torch::ones({n_heads, head_dim}));
}

torch::Tensor OhmC1QKNormImpl::forward(torch::Tensor x) {
    // x: [B, T, n_heads, head_dim]
    auto xf  = x.to(torch::kFloat);
    auto rms = xf.pow(2).mean(-1, /*keepdim=*/true).add(eps).rsqrt();
    // w: [n_heads, head_dim] -> broadcast [1, 1, n_heads, head_dim]
    return (xf * rms).type_as(x) * w.unsqueeze(0).unsqueeze(0);
}

// ─────────────────────────────────────────────────────────────────────────────
// OhmC1Attention
// ─────────────────────────────────────────────────────────────────────────────

OhmC1AttentionImpl::OhmC1AttentionImpl(const OhmC1Config& cfg)
    : n_heads_(cfg.n_heads)
    , n_kv_heads_(cfg.n_kv_heads)
    , head_dim_(cfg.head_dim())
    , n_rep_(cfg.n_rep())
    , window_size_(cfg.window_size)
{
    wq = register_module("wq",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.n_heads    * head_dim_).bias(false)));
    wk = register_module("wk",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.n_kv_heads * head_dim_).bias(false)));
    wv = register_module("wv",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.n_kv_heads * head_dim_).bias(false)));
    wo = register_module("wo",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.n_heads * head_dim_, cfg.dim).bias(false)));

    q_norm = register_module("q_norm", OhmC1QKNorm(cfg.n_heads,    head_dim_, cfg.norm_eps));
    k_norm = register_module("k_norm", OhmC1QKNorm(cfg.n_kv_heads, head_dim_, cfg.norm_eps));

    if (cfg.dropout > 0.0f) {
        attn_drop  = register_module("attn_drop",  torch::nn::Dropout(cfg.dropout));
        resid_drop = register_module("resid_drop", torch::nn::Dropout(cfg.dropout));
    }
}

torch::Tensor OhmC1AttentionImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& freqs_cos,
    const torch::Tensor& freqs_sin,
    torch::Tensor* kv_cache_k,
    torch::Tensor* kv_cache_v,
    int64_t        cache_pos)
{
    int64_t B = x.size(0), T = x.size(1);

    auto q = wq->forward(x).view({B, T, n_heads_,    head_dim_});
    auto k = wk->forward(x).view({B, T, n_kv_heads_, head_dim_});
    auto v = wv->forward(x).view({B, T, n_kv_heads_, head_dim_});

    // QKNorm before RoPE
    q = q_norm->forward(q);
    k = k_norm->forward(k);

    // RoPE
    if (cache_pos >= 0) {
        auto fc = freqs_cos.slice(0, cache_pos, cache_pos + 1);
        auto fs = freqs_sin.slice(0, cache_pos, cache_pos + 1);
        q = apply_rotary_emb(q, fc, fs);
        k = apply_rotary_emb(k, fc, fs);
    } else {
        auto fc = freqs_cos.slice(0, 0, T);
        auto fs = freqs_sin.slice(0, 0, T);
        q = apply_rotary_emb(q, fc, fs);
        k = apply_rotary_emb(k, fc, fs);
    }

    // KV-cache (inference)
    if (kv_cache_k && kv_cache_v && cache_pos >= 0) {
        kv_cache_k->slice(1, cache_pos, cache_pos + 1).copy_(k);
        kv_cache_v->slice(1, cache_pos, cache_pos + 1).copy_(v);
        k = kv_cache_k->slice(1, 0, cache_pos + 1);
        v = kv_cache_v->slice(1, 0, cache_pos + 1);
    }

    // GQA expansion
    if (n_rep_ > 1) {
        k = k.unsqueeze(3).expand({-1, -1, -1, n_rep_, -1})
             .contiguous().view({B, k.size(1), n_heads_, head_dim_});
        v = v.unsqueeze(3).expand({-1, -1, -1, n_rep_, -1})
             .contiguous().view({B, v.size(1), n_heads_, head_dim_});
    }

    // [B, n_heads, T, head_dim]
    q = q.transpose(1, 2);
    k = k.transpose(1, 2);
    v = v.transpose(1, 2);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    auto attn = q.matmul(k.transpose(-2, -1)) * scale;  // [B, H, T, T_kv]

    // Causal mask (training/prefill)
    if (cache_pos < 0) {
        int64_t Tkv = k.size(2);
        auto causal = torch::triu(
            torch::ones({T, Tkv}, x.options()), 1).to(torch::kBool);

        // Sliding-window: also mask tokens too far in the past
        if (window_size_ > 0) {
            // position j is masked for query i if j < i - window_size_ + 1
            auto rows = torch::arange(T, x.options()).unsqueeze(1);  // [T, 1]
            auto cols = torch::arange(Tkv, x.options()).unsqueeze(0); // [1, Tkv]
            auto sw   = (cols < rows - window_size_ + 1).to(torch::kBool);
            causal    = causal | sw;
        }

        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), -1e9f);
    }

    attn = torch::softmax(attn, -1);
    if (!attn_drop.is_empty() && is_training())
        attn = attn_drop->forward(attn);

    auto out = attn.matmul(v)
                   .transpose(1, 2)
                   .contiguous()
                   .view({B, T, n_heads_ * head_dim_});
    auto h = wo->forward(out);
    if (!resid_drop.is_empty() && is_training())
        h = resid_drop->forward(h);
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// OhmC1FFN — SwiGLU
// ─────────────────────────────────────────────────────────────────────────────

OhmC1FFNImpl::OhmC1FFNImpl(int64_t dim, int64_t ffn_dim, float dropout) {
    w1 = register_module("w1",
        torch::nn::Linear(torch::nn::LinearOptions(dim, ffn_dim).bias(false)));
    w2 = register_module("w2",
        torch::nn::Linear(torch::nn::LinearOptions(ffn_dim, dim).bias(false)));
    w3 = register_module("w3",
        torch::nn::Linear(torch::nn::LinearOptions(dim, ffn_dim).bias(false)));
    if (dropout > 0.0f)
        drop = register_module("drop", torch::nn::Dropout(dropout));
}

torch::Tensor OhmC1FFNImpl::forward(torch::Tensor x) {
    auto h = torch::silu(w1->forward(x)) * w3->forward(x);
    if (!drop.is_empty() && is_training())
        h = drop->forward(h);
    return w2->forward(h);
}

// ─────────────────────────────────────────────────────────────────────────────
// OhmC1Block — sandwich-norm
// ─────────────────────────────────────────────────────────────────────────────

OhmC1BlockImpl::OhmC1BlockImpl(const OhmC1Config& cfg) {
    pre_attn_norm  = register_module("pre_attn_norm",  OhmC1RMSNorm(cfg.dim, cfg.norm_eps));
    post_attn_norm = register_module("post_attn_norm", OhmC1RMSNorm(cfg.dim, cfg.norm_eps));
    pre_ffn_norm   = register_module("pre_ffn_norm",   OhmC1RMSNorm(cfg.dim, cfg.norm_eps));
    post_ffn_norm  = register_module("post_ffn_norm",  OhmC1RMSNorm(cfg.dim, cfg.norm_eps));
    attention    = register_module("attention",    OhmC1Attention(cfg));
    feed_forward = register_module("feed_forward", OhmC1FFN(cfg.dim, cfg.ffn_dim, cfg.dropout));
}

torch::Tensor OhmC1BlockImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& freqs_cos,
    const torch::Tensor& freqs_sin,
    torch::Tensor* kv_cache_k,
    torch::Tensor* kv_cache_v,
    int64_t        cache_pos)
{
    auto h   = x + post_attn_norm->forward(
                       attention->forward(pre_attn_norm->forward(x),
                                          freqs_cos, freqs_sin,
                                          kv_cache_k, kv_cache_v, cache_pos));
    auto out = h + post_ffn_norm->forward(
                       feed_forward->forward(pre_ffn_norm->forward(h)));
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// OhmC1Model
// ─────────────────────────────────────────────────────────────────────────────

void OhmC1ModelImpl::_init_weights() {
    torch::nn::init::normal_(tok_embeddings->weight, 0.0, 0.02);
    torch::nn::init::normal_(lm_head->weight, 0.0, 0.02);
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

OhmC1ModelImpl::OhmC1ModelImpl(const OhmC1Config& c) : cfg(c) {
    tok_embeddings = register_module("tok_embeddings",
        torch::nn::Embedding(cfg.vocab_size, cfg.dim));

    if (cfg.dropout > 0.0f)
        drop = register_module("drop", torch::nn::Dropout(cfg.dropout));

    layers = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_layers; ++i)
        layers->push_back(OhmC1Block(cfg));

    norm = register_module("norm", OhmC1RMSNorm(cfg.dim, cfg.norm_eps));

    // lm_head is never tied to tok_embeddings
    lm_head = register_module("lm_head",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.vocab_size).bias(false)));

    auto [fc, fs] = _precompute_freqs(cfg.head_dim(), cfg.seq_len, cfg.rope_theta);
    freqs_cos = register_buffer("freqs_cos", fc);
    freqs_sin = register_buffer("freqs_sin", fs);

    _init_weights();
}

torch::Tensor OhmC1ModelImpl::forward(
    const torch::Tensor& tokens,
    const torch::Tensor& targets)
{
    int64_t T = tokens.size(1);
    auto x = tok_embeddings->forward(tokens);
    if (!drop.is_empty() && is_training())
        x = drop->forward(x);

    auto fc = freqs_cos.slice(0, 0, T);
    auto fs = freqs_sin.slice(0, 0, T);

    for (int64_t i = 0; i < (int64_t)layers->size(); ++i)
        x = layers->at<OhmC1BlockImpl>(i).forward(x, fc, fs);

    x = norm->forward(x);
    auto logits = lm_head->forward(x);

    if (targets.defined()) {
        last_loss = torch::nn::functional::cross_entropy(
            logits.reshape({-1, cfg.vocab_size}),
            targets.reshape({-1}));
    }
    return logits;
}

torch::Tensor OhmC1ModelImpl::forward_one(
    int64_t token, int64_t pos,
    std::vector<torch::Tensor>& kv_caches_k,
    std::vector<torch::Tensor>& kv_caches_v)
{
    auto tokens = torch::tensor({{token}},
        torch::TensorOptions().dtype(torch::kLong).device(freqs_cos.device()));

    auto x = tok_embeddings->forward(tokens);

    for (int64_t i = 0; i < (int64_t)layers->size(); ++i)
        x = layers->at<OhmC1BlockImpl>(i).forward(
            x, freqs_cos, freqs_sin,
            &kv_caches_k[i], &kv_caches_v[i], pos);

    x = norm->forward(x);
    return lm_head->forward(x.squeeze(1));  // [1, vocab_size]
}

void OhmC1ModelImpl::init_kv_caches(
    std::vector<torch::Tensor>& kv_caches_k,
    std::vector<torch::Tensor>& kv_caches_v) const
{
    kv_caches_k.clear();
    kv_caches_v.clear();
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        kv_caches_k.push_back(torch::zeros(
            {1, cfg.seq_len, cfg.n_kv_heads, cfg.head_dim()},
            freqs_cos.options()));
        kv_caches_v.push_back(torch::zeros(
            {1, cfg.seq_len, cfg.n_kv_heads, cfg.head_dim()},
            freqs_cos.options()));
    }
}

std::vector<int64_t> OhmC1ModelImpl::generate(
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

    for (int64_t tok : prompt_tokens)
        forward_one(tok, pos++, kv_k, kv_v);

    for (int64_t s = 0; s < max_new_tokens; ++s) {
        auto logits = forward_one(tokens.back(), pos++, kv_k, kv_v);  // [1, vocab]

        int64_t next_tok;
        if (temperature <= 0.0f) {
            next_tok = logits.argmax(-1).item<int64_t>();
        } else {
            auto scaled = logits / temperature;
            auto probs  = torch::softmax(scaled, -1).squeeze(0);

            auto sorted_res = probs.sort(-1, /*descending=*/true);
            auto sorted_probs = std::get<0>(sorted_res);
            auto sorted_idx   = std::get<1>(sorted_res);
            auto cum  = sorted_probs.cumsum(-1);
            auto mask = (cum - sorted_probs) > top_p;
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

OhmC1Model make_ohmc1_tiny()   { return OhmC1Model(OhmC1Config::tiny()); }
OhmC1Model make_ohmc1_small()  { return OhmC1Model(OhmC1Config::small()); }
OhmC1Model make_ohmc1_medium() { return OhmC1Model(OhmC1Config::medium()); }
OhmC1Model make_ohmc1_large()  { return OhmC1Model(OhmC1Config::large()); }

// ─────────────────────────────────────────────────────────────────────────────
// Training utilities
// ─────────────────────────────────────────────────────────────────────────────

torch::optim::AdamW make_ohmc1_optimizer(OhmC1Model& model,
                                          const OhmC1TrainConfig& cfg)
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

double ohmc1_lr_schedule(int64_t iter, const OhmC1TrainConfig& cfg) {
    if (cfg.warmup_iters > 0 && iter < cfg.warmup_iters)
        return cfg.lr * static_cast<double>(iter + 1) / cfg.warmup_iters;
    int64_t s = iter - cfg.warmup_iters;
    int64_t T = std::max(cfg.max_iters - cfg.warmup_iters, (int64_t)1);
    double ratio = std::min(1.0, static_cast<double>(s) / T);
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + std::cos(M_PI * ratio));
}

float ohmc1_train_step(OhmC1Model&             model,
                       torch::optim::AdamW&    optimizer,
                       const torch::Tensor&    tokens,
                       const OhmC1TrainConfig& cfg,
                       int64_t                 iter)
{
    double lr_now = ohmc1_lr_schedule(iter, cfg);
    for (auto& pg : optimizer.param_groups())
        static_cast<torch::optim::AdamWOptions&>(pg.options()).lr(lr_now);

    optimizer.zero_grad();
    model->train();

    int64_t T = tokens.size(1) - 1;
    auto inp = tokens.slice(1, 0, T);
    auto tgt = tokens.slice(1, 1, T + 1);

    auto logits = model->forward(inp);
    auto loss   = torch::nn::functional::cross_entropy(
                      logits.reshape({-1, model->cfg.vocab_size}),
                      tgt.reshape({-1}));
    loss.backward();
    torch::nn::utils::clip_grad_norm_(model->parameters(), cfg.grad_clip);
    optimizer.step();
    return loss.item<float>();
}

// ─────────────────────────────────────────────────────────────────────────────
// OhmC1BPETokenizer
// ─────────────────────────────────────────────────────────────────────────────

// Simple JSON value extractor for flat {"token": id, ...} maps.
// Avoids pulling in a JSON library.
static std::unordered_map<std::string, int64_t> parse_vocab_json(
    const std::string& path)
{
    std::unordered_map<std::string, int64_t> map;
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("OhmC1BPETokenizer: cannot open " + path);
    std::string line;
    while (std::getline(f, line)) {
        // expect: "  \"TOKEN\": 123," or similar
        auto q1 = line.find('"');
        if (q1 == std::string::npos) continue;
        auto q2 = line.find('"', q1 + 1);
        if (q2 == std::string::npos) continue;
        std::string token = line.substr(q1 + 1, q2 - q1 - 1);
        auto colon = line.find(':', q2 + 1);
        if (colon == std::string::npos) continue;
        int64_t id = std::stoll(line.substr(colon + 1));
        map[token] = id;
    }
    return map;
}

OhmC1BPETokenizer::OhmC1BPETokenizer(const std::string& vocab_path,
                                       const std::string& merges_path)
{
    token_to_id_ = parse_vocab_json(vocab_path);

    // Build reverse map
    int64_t max_id = -1;
    for (auto& p : token_to_id_)
        max_id = std::max(max_id, p.second);
    id_to_token_.resize(max_id + 1);
    for (auto& p : token_to_id_)
        id_to_token_[p.second] = p.first;

    // Read BPE merge rules: "TOKEN1 TOKEN2\n"
    std::ifstream f(merges_path);
    if (!f.is_open())
        throw std::runtime_error("OhmC1BPETokenizer: cannot open " + merges_path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto sp = line.find(' ');
        if (sp == std::string::npos) continue;
        merges_.push_back({line.substr(0, sp), line.substr(sp + 1)});
    }
}

std::vector<std::string> OhmC1BPETokenizer::_pretokenize(
    const std::string& text) const
{
    // Split on whitespace; prepend 'Ġ' (0xC4 0xa0) to non-leading words
    // to match tiktoken GPT-2 pre-tokenization convention.
    std::vector<std::string> words;
    std::istringstream ss(text);
    std::string word;
    bool first = true;
    while (ss >> word) {
        if (!first) word = "\xc4\xa0" + word;
        words.push_back(word);
        first = false;
    }
    return words;
}

std::vector<std::string> OhmC1BPETokenizer::_bpe_encode_word(
    const std::string& word) const
{
    // Byte-level initial split: each UTF-8 byte is a symbol
    std::vector<std::string> syms;
    for (unsigned char c : word)
        syms.push_back(std::string(1, (char)c));

    // Apply BPE merge rules in priority order
    for (auto& rule : merges_) {
        std::vector<std::string> out;
        for (size_t i = 0; i < syms.size(); ) {
            if (i + 1 < syms.size() && syms[i] == rule.first && syms[i + 1] == rule.second) {
                out.push_back(syms[i] + syms[i + 1]);
                i += 2;
            } else {
                out.push_back(syms[i]);
                ++i;
            }
        }
        syms = out;
    }
    return syms;
}

std::vector<int64_t> OhmC1BPETokenizer::encode(
    const std::string& text,
    bool add_bos) const
{
    std::vector<int64_t> ids;
    if (add_bos) ids.push_back(BOS_ID);
    for (auto& word : _pretokenize(text)) {
        for (auto& sym : _bpe_encode_word(word)) {
            auto it = token_to_id_.find(sym);
            ids.push_back(it != token_to_id_.end() ? it->second : UNK_ID);
        }
    }
    return ids;
}

std::string OhmC1BPETokenizer::decode(const std::vector<int64_t>& ids) const {
    std::string out;
    for (int64_t id : ids) {
        if (id >= 0 && id < (int64_t)id_to_token_.size())
            out += id_to_token_[id];
    }
    // Replace leading Ġ (GPT-2 space marker) with actual space
    std::string result;
    for (size_t i = 0; i < out.size(); ) {
        if ((unsigned char)out[i] == 0xc4 && i + 1 < out.size()
                && (unsigned char)out[i + 1] == 0xa0) {
            if (!result.empty()) result += ' ';
            i += 2;
        } else {
            result += out[i++];
        }
    }
    return result;
}

int64_t OhmC1BPETokenizer::vocab_size() const {
    return static_cast<int64_t>(id_to_token_.size());
}

}  // namespace nlp
}  // namespace models
}  // namespace dm
