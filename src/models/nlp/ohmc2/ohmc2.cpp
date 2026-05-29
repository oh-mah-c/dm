#include "models/nlp/ohmc2/ohmc2.h"

#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace dm {
namespace models {
namespace nlp {

namespace {

torch::Tensor chronological_cache(const torch::Tensor& cache,
                                  int64_t pos,
                                  int64_t valid_len,
                                  int64_t max_seq) {
    if (valid_len < max_seq)
        return cache.slice(1, 0, valid_len);
    const int64_t start = (pos + 1) % max_seq;
    auto idx = (torch::arange(0, max_seq, torch::TensorOptions()
                    .dtype(torch::kLong).device(cache.device())) + start)
                   .remainder(max_seq);
    return cache.index_select(1, idx);
}

torch::Tensor chronological_scores(const torch::Tensor& cache,
                                   int64_t pos,
                                   int64_t valid_len,
                                   int64_t max_seq) {
    (void)pos;
    if (valid_len < max_seq)
        return cache.slice(2, 0, valid_len);
    return cache;
}

bool is_valid_ffn_type(const std::string& s) {
    return s == "swiglu" || s == "kan" || s == "kan_moe";
}

torch::Tensor apply_rope(torch::Tensor x, int64_t pos_start, double theta) {
    const int64_t T = x.size(1);
    const int64_t hd = x.size(3);
    if (hd % 2 != 0)
        throw std::invalid_argument("OhmC2: RoPE requires even head_dim");
    const int64_t half = hd / 2;
    auto opts = x.options().dtype(torch::kFloat);
    auto pos = torch::arange(pos_start, pos_start + T, opts).unsqueeze(1); // [T,1]
    auto idx = torch::arange(0, half, opts).unsqueeze(0);                  // [1,half]
    auto inv = torch::pow(torch::full({1, half}, theta, opts),
                          -(2.0 * idx) / static_cast<double>(hd));
    auto ang = pos * inv; // [T,half]
    auto cos = torch::cos(ang).view({1, T, 1, half}).type_as(x);
    auto sin = torch::sin(ang).view({1, T, 1, half}).type_as(x);
    auto x1 = x.slice(3, 0, half);
    auto x2 = x.slice(3, half, hd);
    return torch::cat({x1 * cos - x2 * sin, x1 * sin + x2 * cos}, 3);
}

}  // namespace

OhmC2Config OhmC2Config::tiny() {
    OhmC2Config c;
    c.dim = 256;
    c.ffn_dim = 1024;
    c.n_layers = 8;
    c.n_heads = 4;
    c.n_experts = 1;
    c.kan_grid = 3;
    c.max_seq = 512;
    c.ffn_type = "kan";
    return c;
}

OhmC2Config OhmC2Config::small() {
    OhmC2Config c;
    c.dim = 512;
    c.ffn_dim = 2048;
    c.n_layers = 16;
    c.n_heads = 8;
    c.n_experts = 4;
    c.kan_grid = 5;
    c.max_seq = 2048;
    c.ffn_type = "kan_moe";
    return c;
}

OhmC2Config OhmC2Config::medium() {
    OhmC2Config c;
    c.dim = 1024;
    c.ffn_dim = 4096;
    c.n_layers = 24;
    c.n_heads = 16;
    c.n_experts = 8;
    c.kan_grid = 8;
    c.max_seq = 4096;
    c.ffn_type = "kan_moe";
    return c;
}

OhmC2Config OhmC2Config::large() {
    OhmC2Config c;
    c.dim = 2048;
    c.ffn_dim = 8192;
    c.n_layers = 32;
    c.n_heads = 32;
    c.n_experts = 16;
    c.kan_grid = 12;
    c.max_seq = 8192;
    c.ffn_type = "kan_moe";
    return c;
}

OhmC2Config OhmC2Config::cpu_quality() {
    OhmC2Config c;
    c.dim = 128;
    c.ffn_dim = 512;
    c.n_layers = 4;
    c.n_heads = 4;
    c.n_experts = 1;
    c.kan_grid = 3;
    c.kan_order = 3;
    c.vocab_size = 256;
    c.max_seq = 128;
    c.ffn_type = "swiglu";
    c.use_rope = true;
    c.rope_theta = 10000.0;
    return c;
}

void OhmC2GenerationState::reset() {
    for (auto& t : score_cache) t.zero_();
    pos = 0;
    valid_len = 0;
}

OhmC2RMSNormImpl::OhmC2RMSNormImpl(int64_t dim, double eps_) : eps(eps_) {
    weight = register_parameter("weight", torch::ones({dim}));
}

torch::Tensor OhmC2RMSNormImpl::forward(torch::Tensor x) {
    auto xf = x.to(torch::kFloat);
    auto rms = xf.pow(2).mean(-1, true).add(eps).rsqrt();
    return (xf * rms).type_as(x) * weight;
}

OhmC2DualGateBlockImpl::OhmC2DualGateBlockImpl(
    const OhmC2Config& cfg_, int64_t layer_index_)
    : cfg(cfg_), layer_index(layer_index_) {
    if (cfg.dim <= 0 || cfg.n_heads <= 0 || cfg.dim % cfg.n_heads != 0)
        throw std::invalid_argument("OhmC2: dim must be divisible by n_heads");
    if (cfg.use_rope && cfg.head_dim() % 2 != 0)
        throw std::invalid_argument("OhmC2: RoPE requires even head_dim");
    if (cfg.n_experts <= 0)
        throw std::invalid_argument("OhmC2: n_experts must be positive");
    if (!is_valid_ffn_type(cfg.ffn_type))
        throw std::invalid_argument("OhmC2: ffn_type must be swiglu, kan, or kan_moe");

    norm_attn_in = register_module("norm_attn_in",
        OhmC2RMSNorm(cfg.dim, cfg.norm_eps));
    norm_attn_out = register_module("norm_attn_out",
        OhmC2RMSNorm(cfg.dim, cfg.norm_eps));
    norm_ffn_in = register_module("norm_ffn_in",
        OhmC2RMSNorm(cfg.dim, cfg.norm_eps));
    norm_ffn_out = register_module("norm_ffn_out",
        OhmC2RMSNorm(cfg.dim, cfg.norm_eps));

    wq = register_module("wq",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.dim).bias(false)));
    wk = register_module("wk",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.dim).bias(false)));
    wv = register_module("wv",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.dim).bias(false)));
    wo = register_module("wo",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.dim).bias(false)));
    router = register_module("router",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.n_experts).bias(true)));

    if (cfg.ffn_type == "swiglu") {
        ffn_up = register_module("ffn_up",
            torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.ffn_dim).bias(false)));
        ffn_gate_proj = register_module("ffn_gate_proj",
            torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.ffn_dim).bias(false)));
    } else {
        std::vector<torch::nn::AnyModule> exp_vec;
        const int64_t n_exp = (cfg.ffn_type == "kan") ? 1 : cfg.n_experts;
        for (int64_t i = 0; i < n_exp; ++i) {
            dm::prim::KANLinearOptions opts(cfg.dim, cfg.ffn_dim);
            opts.G(cfg.kan_grid).k(cfg.kan_order).update_grid(false);
            exp_vec.push_back(torch::nn::AnyModule(dm::prim::KANLinear(opts)));
        }
        ohm_router = register_module("ohm_router", torch::nn::OhmHardRouter(cfg.dim, n_exp, exp_vec));
    }

    w_cross = register_module("w_cross",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.ffn_dim).bias(false)));
    w2 = register_module("w2",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.ffn_dim, cfg.dim).bias(false)));
    block_drop = register_module("block_drop",
        torch::nn::OhmBlockDrop(cfg.dropout, std::max<int64_t>(1, cfg.max_seq / 16), 1));

    gate_attn = register_parameter("gate_attn", torch::full({cfg.dim}, 0.5));
    gate_ffn = register_parameter("gate_ffn", torch::full({cfg.dim}, 0.5));
    last_route_counts = torch::zeros({cfg.n_experts}, torch::kLong);
}

std::pair<torch::Tensor, torch::Tensor>
OhmC2DualGateBlockImpl::attention_full(torch::Tensor norm_a,
                                       torch::Tensor prev_scores) {
    const int64_t B = norm_a.size(0);
    const int64_t T = norm_a.size(1);
    const int64_t H = cfg.n_heads;
    const int64_t hd = cfg.head_dim();
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    auto q = wq->forward(norm_a).view({B, T, H, hd}).transpose(1, 2);
    auto k = wk->forward(norm_a).view({B, T, H, hd}).transpose(1, 2);
    auto v = wv->forward(norm_a).view({B, T, H, hd}).transpose(1, 2);
    if (cfg.use_rope) {
        q = apply_rope(q.transpose(1, 2), 0, cfg.rope_theta).transpose(1, 2);
        k = apply_rope(k.transpose(1, 2), 0, cfg.rope_theta).transpose(1, 2);
    }
    auto raw = torch::matmul(q, k.transpose(-2, -1)) * scale;

    torch::Tensor new_prev;
    if (prev_scores.defined() && prev_scores.numel() > 0) {
        if (cfg.realformer_mean)
            new_prev = (prev_scores * static_cast<double>(layer_index) + raw) /
                       static_cast<double>(layer_index + 1);
        else
            new_prev = prev_scores + raw;
    } else {
        new_prev = raw;
    }

    auto scores = new_prev;
    auto mask = torch::ones({T, T}, norm_a.options()).triu(1).to(torch::kBool);
    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9f);
    auto attn = torch::softmax(scores, -1);
    auto ctx = torch::matmul(attn, v)
                   .transpose(1, 2).contiguous()
                   .view({B, T, cfg.dim});
    return {norm_attn_out->forward(wo->forward(ctx)), new_prev};
}

torch::Tensor OhmC2DualGateBlockImpl::ffn_route(torch::Tensor norm_f) {
    const int64_t B = norm_f.size(0);
    const int64_t T = norm_f.size(1);
    auto flat = norm_f.contiguous().view({B * T, cfg.dim});
    auto out = ohm_router->forward(flat);
    return out.view({B, T, cfg.ffn_dim});
}

torch::Tensor OhmC2DualGateBlockImpl::ffn_forward(torch::Tensor norm_f) {
    if (cfg.ffn_type == "swiglu")
        return torch::silu(ffn_gate_proj->forward(norm_f)) * ffn_up->forward(norm_f);
    return ffn_route(norm_f);
}

std::pair<torch::Tensor, torch::Tensor>
OhmC2DualGateBlockImpl::forward(torch::Tensor x, torch::Tensor prev_scores) {
    auto norm_a = norm_attn_in->forward(x);
    auto attn_pair = attention_full(norm_a, prev_scores);
    auto attn_out = attn_pair.first;
    auto new_prev = attn_pair.second;

    auto norm_f = norm_ffn_in->forward(x);
    auto ffn_gate = ffn_forward(norm_f);
    auto gated_mid = ffn_gate * w_cross->forward(attn_out);
    auto ffn_out = norm_ffn_out->forward(w2->forward(gated_mid));
    auto out = x + attn_out * gate_attn + ffn_out * gate_ffn;
    if (is_training() && cfg.dropout > 0.0)
        out = block_drop->forward(out);
    return {out, new_prev};
}

torch::Tensor OhmC2DualGateBlockImpl::forward_step(
    torch::Tensor x, OhmC2GenerationState& state) {
    const int64_t B = x.size(0);
    const int64_t H = cfg.n_heads;
    const int64_t hd = cfg.head_dim();
    const int64_t slot = state.pos % state.max_seq;
    const int64_t next_valid = std::min(state.valid_len + 1, state.max_seq);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    auto norm_a = norm_attn_in->forward(x);
    auto q = wq->forward(norm_a).view({B, 1, H, hd});
    auto k_new = wk->forward(norm_a).view({B, 1, H, hd});
    auto v_new = wv->forward(norm_a).view({B, 1, H, hd});
    if (cfg.use_rope) {
        q = apply_rope(q, state.pos, cfg.rope_theta);
        k_new = apply_rope(k_new, state.pos, cfg.rope_theta);
    }
    auto k_hist = state.k_ring[layer_index]->forward(k_new);
    auto v_hist = state.v_ring[layer_index]->forward(v_new);

    auto qh = q.transpose(1, 2);       // [B,H,1,hd]
    auto kh = k_hist.transpose(1, 2);  // [B,H,L,hd]
    auto vh = v_hist.transpose(1, 2);
    auto raw = torch::matmul(qh, kh.transpose(-2, -1)).squeeze(2) * scale;

    torch::Tensor new_scores;
    if (layer_index > 0) {
        auto prev = chronological_scores(state.score_cache[layer_index - 1],
                                         state.pos, next_valid, state.max_seq);
        if (cfg.realformer_mean)
            new_scores = (prev * static_cast<double>(layer_index) + raw) /
                         static_cast<double>(layer_index + 1);
        else
            new_scores = prev + raw;
    } else {
        new_scores = raw;
    }

    state.score_cache[layer_index].zero_();
    state.score_cache[layer_index].slice(2, 0, next_valid).copy_(new_scores);

    auto attn = torch::softmax(new_scores, -1).unsqueeze(2);
    auto ctx = torch::matmul(attn, vh)
                   .transpose(1, 2).contiguous()
                   .view({B, 1, cfg.dim});
    auto attn_out = norm_attn_out->forward(wo->forward(ctx));
    auto norm_f = norm_ffn_in->forward(x);
    auto ffn_gate = ffn_forward(norm_f);
    auto gated_mid = ffn_gate * w_cross->forward(attn_out);
    auto ffn_out = norm_ffn_out->forward(w2->forward(gated_mid));
    return x + attn_out * gate_attn + ffn_out * gate_ffn;
}

OhmC2LLMImpl::OhmC2LLMImpl(const OhmC2Config& cfg_) : cfg(cfg_) {
    tok_embeddings = register_module("tok_embeddings",
        torch::nn::Embedding(cfg.vocab_size, cfg.dim));
    layers = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_layers; ++i)
        layers->push_back(OhmC2DualGateBlock(cfg, i));
    norm = register_module("norm", OhmC2RMSNorm(cfg.dim, cfg.norm_eps));
    lm_head = register_module("lm_head",
        torch::nn::Linear(torch::nn::LinearOptions(cfg.dim, cfg.vocab_size).bias(false)));
    torch::nn::init::normal_(tok_embeddings->weight, 0.0, 0.02);
    torch::nn::init::normal_(lm_head->weight, 0.0, 0.02);
}

torch::Tensor OhmC2LLMImpl::forward(torch::Tensor tokens, torch::Tensor targets) {
    auto x = tok_embeddings->forward(tokens);
    torch::Tensor prev;
    last_prev_scores.clear();
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        auto& block = layers->at<OhmC2DualGateBlockImpl>(i);
        auto pair = block.forward(x, prev);
        x = pair.first;
        prev = pair.second;
        last_prev_scores.push_back(prev);
    }
    auto logits = lm_head->forward(norm->forward(x));
    if (targets.defined() && targets.numel() > 0) {
        last_loss = torch::nn::functional::cross_entropy(
            logits.reshape({-1, cfg.vocab_size}), targets.reshape({-1}));
    }
    return logits;
}

OhmC2GenerationState OhmC2LLMImpl::init_generation_state(int64_t batch_size) const {
    OhmC2GenerationState st;
    st.max_seq = cfg.max_seq;
    st.batch_size = batch_size;
    auto opts = tok_embeddings->weight.options();
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        st.k_ring.push_back(torch::nn::OhmRingKV(batch_size, cfg.max_seq, cfg.n_heads, cfg.head_dim(), tok_embeddings->weight.device()));
        st.v_ring.push_back(torch::nn::OhmRingKV(batch_size, cfg.max_seq, cfg.n_heads, cfg.head_dim(), tok_embeddings->weight.device()));
        st.score_cache.push_back(torch::zeros(
            {batch_size, cfg.n_heads, cfg.max_seq}, opts));
    }
    return st;
}

torch::Tensor OhmC2LLMImpl::forward_step(torch::Tensor token_ids,
                                         OhmC2GenerationState& state) {
    if (token_ids.dim() == 1)
        token_ids = token_ids.unsqueeze(1);
    auto x = tok_embeddings->forward(token_ids);
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        auto& block = layers->at<OhmC2DualGateBlockImpl>(i);
        x = block.forward_step(x, state);
    }
    state.pos += 1;
    state.valid_len = std::min(state.valid_len + 1, state.max_seq);
    return lm_head->forward(norm->forward(x).squeeze(1));
}

std::vector<int64_t> OhmC2LLMImpl::generate(
    const std::vector<int64_t>& prompt_ids,
    int64_t max_new_tokens,
    float temperature,
    float top_p,
    int64_t eos_id) {
    eval();
    torch::NoGradGuard ng;
    auto state = init_generation_state(1);
    std::vector<int64_t> out = prompt_ids;
    torch::Tensor logits;
    for (int64_t id : prompt_ids)
        logits = forward_step(torch::tensor({id}, torch::kLong), state);
    for (int64_t i = 0; i < max_new_tokens; ++i) {
        int64_t next;
        if (temperature <= 0.0f) {
            next = logits.argmax(-1).item<int64_t>();
        } else {
            auto probs = torch::softmax(logits / temperature, -1).squeeze(0);
            auto sorted = probs.sort(-1, true);
            auto sorted_probs = std::get<0>(sorted);
            auto sorted_idx = std::get<1>(sorted);
            auto mask = (sorted_probs.cumsum(-1) - sorted_probs) > top_p;
            sorted_probs.masked_fill_(mask, 0.0);
            sorted_probs.div_(sorted_probs.sum().clamp_min(1e-12));
            next = sorted_idx[torch::multinomial(sorted_probs, 1).item<int64_t>()]
                       .item<int64_t>();
        }
        out.push_back(next);
        if (next == eos_id) break;
        logits = forward_step(torch::tensor({next}, torch::kLong), state);
    }
    return out;
}

OhmC2LLM make_ohmc2_tiny() { return OhmC2LLM(OhmC2Config::tiny()); }
OhmC2LLM make_ohmc2_small() { return OhmC2LLM(OhmC2Config::small()); }
OhmC2LLM make_ohmc2_medium() { return OhmC2LLM(OhmC2Config::medium()); }
OhmC2LLM make_ohmc2_large() { return OhmC2LLM(OhmC2Config::large()); }

torch::optim::AdamW make_ohmc2_optimizer(OhmC2LLM& model,
                                          const OhmC2TrainConfig& cfg) {
    std::vector<torch::Tensor> decay, nodecay;
    for (auto& item : model->named_parameters()) {
        auto& p = item.value();
        if (!p.requires_grad()) continue;
        if (p.dim() >= 2) decay.push_back(p);
        else nodecay.push_back(p);
    }
    std::vector<torch::optim::OptimizerParamGroup> groups;
    auto opt1 = std::make_unique<torch::optim::AdamWOptions>(cfg.lr);
    opt1->betas({cfg.beta1, cfg.beta2});
    opt1->eps(cfg.eps);
    opt1->weight_decay(cfg.weight_decay);
    groups.emplace_back(decay, std::move(opt1));
    auto opt2 = std::make_unique<torch::optim::AdamWOptions>(cfg.lr);
    opt2->betas({cfg.beta1, cfg.beta2});
    opt2->eps(cfg.eps);
    opt2->weight_decay(0.0);
    groups.emplace_back(nodecay, std::move(opt2));
    return torch::optim::AdamW(groups);
}

double ohmc2_lr_schedule(int64_t iter, const OhmC2TrainConfig& cfg) {
    if (cfg.warmup_iters > 0 && iter < cfg.warmup_iters)
        return cfg.lr * static_cast<double>(iter + 1) / cfg.warmup_iters;
    const int64_t span = std::max<int64_t>(1, cfg.max_iters - cfg.warmup_iters);
    const double t = std::min(1.0, static_cast<double>(iter - cfg.warmup_iters) / span);
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + std::cos(M_PI * t));
}

float ohmc2_train_step(OhmC2LLM& model,
                       torch::optim::AdamW& optimizer,
                       torch::Tensor tokens,
                       const OhmC2TrainConfig& cfg,
                       int64_t iter) {
    for (auto& pg : optimizer.param_groups())
        static_cast<torch::optim::AdamWOptions&>(pg.options()).lr(
            ohmc2_lr_schedule(iter, cfg));
    model->train();
    optimizer.zero_grad();
    int64_t T = tokens.size(1) - 1;
    auto inp = tokens.slice(1, 0, T);
    auto tgt = tokens.slice(1, 1, T + 1);
    model->forward(inp, tgt);
    model->last_loss.backward();
    torch::nn::utils::clip_grad_norm_(model->parameters(), cfg.grad_clip);
    optimizer.step();
    return model->last_loss.item<float>();
}

}  // namespace nlp
}  // namespace models
}  // namespace dm
