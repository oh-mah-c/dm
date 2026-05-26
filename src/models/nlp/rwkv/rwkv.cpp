// ─────────────────────────────────────────────────────────────────────────────
// rwkv.cpp — RWKV implementation
//
// B. Peng et al., "RWKV: Reinventing RNNs for the Transformer Era,"
// arXiv:2305.13048v2, 2023. https://arxiv.org/abs/2305.13048
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/rwkv/rwkv.h"

#include <torch/torch.h>
#include <cmath>
#include <algorithm>
#include <tuple>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// RWKVTimeMix
// ─────────────────────────────────────────────────────────────────────────────
RWKVTimeMixImpl::RWKVTimeMixImpl(const RWKVConfig& cfg, int64_t layer_id)
    : d_model(cfg.d_model) {

    int64_t D = cfg.d_model;
    int64_t L = cfg.n_layers;
    int64_t l = layer_id;   // 0-indexed

    // ── Token-shift interpolation factors μ (App. E) ───────────────────
    // μ_k[i] = (i/D)^{1 - l/L}   (eq given in App. E for time-mix K)
    // μ_v[i] = (i/D)^{1 - l/L} + 0.3l/(L-1)
    // μ_r[i] = 0.5 * (i/D)^{1 - l/L}
    auto ratio_0_to_1 = static_cast<double>(l) / std::max((int64_t)1, L - 1);
    auto exponent     = 1.0 - ratio_0_to_1;

    auto idx = torch::arange(D, torch::kDouble) / D;  // [D] in [0,1)
    auto base = torch::pow(idx, exponent);              // [D]

    mu_k = register_parameter("mu_k",
               (base).to(torch::kFloat));
    mu_v = register_parameter("mu_v",
               (base + 0.3 * ratio_0_to_1).clamp(0.0, 1.0).to(torch::kFloat));
    mu_r = register_parameter("mu_r",
               (0.5 * base).to(torch::kFloat));

    // ── Time decay w (App. E) ─────────────────────────────────────────
    // w[i] = -5 + 8 * (i/(D-1))^{0.7 + 1.3l/(L-1)}
    // We parameterize w directly (it must be non-negative after exp(-w)≤1
    // but in practice is initialised as described and trained freely).
    auto d_idx = torch::arange(D, torch::kDouble) / std::max((int64_t)1, D - 1);
    double exp_w = 0.7 + 1.3 * ratio_0_to_1;
    auto w_init = -5.0 + 8.0 * torch::pow(d_idx, exp_w);
    time_decay = register_parameter("time_decay", w_init.to(torch::kFloat));

    // ── Bonus u (App. E) ─────────────────────────────────────────────
    // u[i] = 0.5 * (((i+1) mod 3) - 1) + log(0.3)
    auto u_init = torch::zeros(D, torch::kFloat);
    for (int64_t i = 0; i < D; ++i) {
        u_init[i] = 0.5f * static_cast<float>(((i + 1) % 3) - 1)
                    + static_cast<float>(std::log(0.3));
    }
    time_first = register_parameter("time_first", u_init);

    // ── Projections (no bias, weights init to 0 except Wo, App. E) ───
    Wr = register_module("Wr", torch::nn::Linear(
             torch::nn::LinearOptions(D, D).bias(false)));
    Wk = register_module("Wk", torch::nn::Linear(
             torch::nn::LinearOptions(D, D).bias(false)));
    Wv = register_module("Wv", torch::nn::Linear(
             torch::nn::LinearOptions(D, D).bias(false)));
    Wo = register_module("Wo", torch::nn::Linear(
             torch::nn::LinearOptions(D, D).bias(false)));

    // Init: Wr, Wk, Wv → zeros;  Wo → N(0, sqrt(d/s)) where s=d (App. E)
    torch::nn::init::zeros_(Wr->weight);
    torch::nn::init::zeros_(Wk->weight);
    torch::nn::init::zeros_(Wv->weight);
    torch::nn::init::normal_(Wo->weight, 0.0, std::sqrt(static_cast<double>(D) / D));
}

// Numerically stable WKV step (App. D, eqs. 23–28)
// Returns (wkv_t, aa', bb', pp')
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RWKVTimeMixImpl::wkv_step(torch::Tensor k_t, torch::Tensor v_t,
                            torch::Tensor aa, torch::Tensor bb, torch::Tensor pp) {
    // u = time_first, w = time_decay (non-neg desired but trained freely)
    auto u = time_first;   // [D]
    auto w = time_decay;   // [D]

    // q = max(p_{t-1}, u+k_t)   (for wkv numerator/denominator)
    auto uk   = u + k_t;                        // [B, D]
    auto q    = torch::maximum(pp, uk);          // [B, D]

    // wkv_t = (e^{pp-q} * aa + e^{uk-q} * v_t) /
    //         (e^{pp-q} * bb + e^{uk-q})
    auto e_pp = torch::exp(pp - q);             // [B, D]
    auto e_uk = torch::exp(uk - q);             // [B, D]
    auto wkv  = (e_pp * aa + e_uk * v_t) /
                (e_pp * bb + e_uk).clamp_min(1e-30f);  // [B, D]

    // Update state
    // q' = max(p_{t-1} - w, k_t)
    auto qp    = torch::maximum(pp - w, k_t);    // [B, D]
    auto e_ppw = torch::exp(pp - w - qp);        // [B, D]  e^{p-w-q'}
    auto e_k   = torch::exp(k_t - qp);           // [B, D]  e^{k_t - q'}

    auto aa_new = e_ppw * aa + e_k * v_t;
    auto bb_new = e_ppw * bb + e_k;
    auto pp_new = qp;

    return {wkv, aa_new, bb_new, pp_new};
}

// Training forward: full sequence parallel scan
torch::Tensor RWKVTimeMixImpl::forward(torch::Tensor x, torch::Tensor x_prev) {
    // x:     [B, T, D]
    // x_prev:[B, T, D]  — x shifted right by 1 (x_{t-1}), padded with zeros

    // Token-shifted inputs
    auto xr = mu_r * x + (1.0f - mu_r) * x_prev;   // [B,T,D]
    auto xk = mu_k * x + (1.0f - mu_k) * x_prev;
    auto xv = mu_v * x + (1.0f - mu_v) * x_prev;

    auto r = torch::sigmoid(Wr->forward(xr));        // [B,T,D]
    auto k = Wk->forward(xk);                        // [B,T,D]
    auto v = Wv->forward(xv);                        // [B,T,D]

    int64_t B = x.size(0);
    int64_t T = x.size(1);

    // WKV sequential scan along T
    auto aa = torch::zeros({B, d_model}, x.options());
    auto bb = torch::zeros({B, d_model}, x.options());
    // pp init: very negative so e^{pp}≈0 (acts like p_0=-inf)
    auto pp = torch::full({B, d_model}, -1e30f, x.options());

    std::vector<torch::Tensor> wkv_list;
    wkv_list.reserve(T);
    for (int64_t t = 0; t < T; ++t) {
        auto k_t = k.select(1, t);   // [B,D]
        auto v_t = v.select(1, t);   // [B,D]
        auto [wkv_t, aa_n, bb_n, pp_n] = wkv_step(k_t, v_t, aa, bb, pp);
        aa = aa_n; bb = bb_n; pp = pp_n;
        wkv_list.push_back(wkv_t);
    }

    std::vector<torch::Tensor> wkv_vec(wkv_list.begin(), wkv_list.end());
    auto wkv = torch::stack(wkv_vec, 1);     // [B,T,D]

    // Output gating + projection (eq. 17)
    auto out = Wo->forward(r * wkv);          // [B,T,D]
    return out;
}

// Inference step
torch::Tensor RWKVTimeMixImpl::step(torch::Tensor x_t,
                                     torch::Tensor x_tm1,
                                     torch::Tensor& aa,
                                     torch::Tensor& bb,
                                     torch::Tensor& pp) {
    auto xr = mu_r * x_t + (1.0f - mu_r) * x_tm1;
    auto xk = mu_k * x_t + (1.0f - mu_k) * x_tm1;
    auto xv = mu_v * x_t + (1.0f - mu_v) * x_tm1;

    auto r = torch::sigmoid(Wr->forward(xr));
    auto k = Wk->forward(xk);
    auto v = Wv->forward(xv);

    auto [wkv_t, aa_n, bb_n, pp_n] = wkv_step(k, v, aa, bb, pp);
    aa = aa_n; bb = bb_n; pp = pp_n;

    return Wo->forward(r * wkv_t);
}

// ─────────────────────────────────────────────────────────────────────────────
// RWKVChannelMix
// ─────────────────────────────────────────────────────────────────────────────
RWKVChannelMixImpl::RWKVChannelMixImpl(const RWKVConfig& cfg, int64_t layer_id)
    : d_model(cfg.d_model), d_ff(cfg.d_ff()) {

    int64_t D = cfg.d_model;
    int64_t L = cfg.n_layers;
    int64_t l = layer_id;

    double ratio = static_cast<double>(l) / std::max((int64_t)1, L - 1);
    double exponent = 1.0 - ratio;
    auto idx  = torch::arange(D, torch::kDouble) / D;
    auto base = torch::pow(idx, exponent);  // [D]

    // μ_k' = μ_r' = (i/D)^{1-l/(L-1)}  (App. E, channel-mixing)
    mu_k = register_parameter("mu_k", base.to(torch::kFloat));
    mu_r = register_parameter("mu_r", base.to(torch::kFloat));

    // Projections: Wk, Wv → 0; Wr → 0 (App. E)
    Wr = register_module("Wr", torch::nn::Linear(
             torch::nn::LinearOptions(D, D).bias(false)));
    Wk = register_module("Wk", torch::nn::Linear(
             torch::nn::LinearOptions(D, d_ff).bias(false)));
    Wv = register_module("Wv", torch::nn::Linear(
             torch::nn::LinearOptions(d_ff, D).bias(false)));

    torch::nn::init::zeros_(Wr->weight);
    torch::nn::init::zeros_(Wk->weight);
    torch::nn::init::normal_(Wv->weight, 0.0, std::sqrt(static_cast<double>(D) / D));
}

torch::Tensor RWKVChannelMixImpl::forward(torch::Tensor x, torch::Tensor x_prev) {
    auto xr = mu_r * x + (1.0f - mu_r) * x_prev;
    auto xk = mu_k * x + (1.0f - mu_k) * x_prev;

    auto r = torch::sigmoid(Wr->forward(xr));
    // squared-ReLU activation (eq.18, So et al. 2021)
    auto k = Wk->forward(xk);
    auto kk = torch::pow(torch::relu(k), 2.0);
    return r * Wv->forward(kk);
}

torch::Tensor RWKVChannelMixImpl::step(torch::Tensor x_t, torch::Tensor x_tm1) {
    auto xr = mu_r * x_t + (1.0f - mu_r) * x_tm1;
    auto xk = mu_k * x_t + (1.0f - mu_k) * x_tm1;

    auto r  = torch::sigmoid(Wr->forward(xr));
    auto k  = Wk->forward(xk);
    auto kk = torch::pow(torch::relu(k), 2.0);
    return r * Wv->forward(kk);
}

// ─────────────────────────────────────────────────────────────────────────────
// RWKVBlock
// ─────────────────────────────────────────────────────────────────────────────
RWKVBlockImpl::RWKVBlockImpl(const RWKVConfig& cfg, int64_t layer_id) {
    ln1 = register_module("ln1", torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({cfg.d_model})));
    ln2 = register_module("ln2", torch::nn::LayerNorm(
              torch::nn::LayerNormOptions({cfg.d_model})));
    time_mix = register_module("time_mix", RWKVTimeMix(cfg, layer_id));
    chan_mix = register_module("chan_mix", RWKVChannelMix(cfg, layer_id));

    // LN init: weight=1, bias=0 (App. E: "All LayerNorm weights start from 1")
    torch::nn::init::ones_(ln1->weight);
    torch::nn::init::zeros_(ln1->bias);
    torch::nn::init::ones_(ln2->weight);
    torch::nn::init::zeros_(ln2->bias);
}

torch::Tensor RWKVBlockImpl::forward(torch::Tensor x) {
    // x: [B, T, D]
    int64_t T = x.size(1);

    // Token shift: x_prev = [zeros, x[:,0..T-2,:]]
    // Implemented by padding left with zero and dropping last
    auto x_tm = torch::zeros_like(x);
    if (T > 1) x_tm.slice(1, 1) = x.slice(1, 0, T - 1);

    auto ln1_x = ln1->forward(x);
    auto tm_out = time_mix->forward(ln1_x, ln1->forward(x_tm));
    x = x + tm_out;

    auto x_cm = torch::zeros_like(x);
    if (T > 1) x_cm.slice(1, 1) = x.slice(1, 0, T - 1);

    auto ln2_x = ln2->forward(x);
    auto cm_out = chan_mix->forward(ln2_x, ln2->forward(x_cm));
    x = x + cm_out;

    return x;
}

torch::Tensor RWKVBlockImpl::step(torch::Tensor x_t,
                                   torch::Tensor& x_tm_tm,
                                   torch::Tensor& x_cm_tm,
                                   torch::Tensor& aa,
                                   torch::Tensor& bb,
                                   torch::Tensor& pp) {
    auto ln1_x  = ln1->forward(x_t.unsqueeze(1)).squeeze(1);
    auto ln1_xm = ln1->forward(x_tm_tm.unsqueeze(1)).squeeze(1);
    auto tm_out = time_mix->step(ln1_x, ln1_xm, aa, bb, pp);
    x_tm_tm = x_t;   // update token-shift state for time-mix
    x_t = x_t + tm_out;

    auto ln2_x  = ln2->forward(x_t.unsqueeze(1)).squeeze(1);
    auto ln2_xm = ln2->forward(x_cm_tm.unsqueeze(1)).squeeze(1);
    auto cm_out = chan_mix->step(ln2_x, ln2_xm);
    x_cm_tm = x_t - tm_out;  // state before channel-mix residual
    x_t = x_t + cm_out;

    return x_t;
}

// ─────────────────────────────────────────────────────────────────────────────
// RWKVModel
// ─────────────────────────────────────────────────────────────────────────────
RWKVModelImpl::RWKVModelImpl(const RWKVConfig& c) : cfg(c) {
    // Small init embedding (§3.4): U(±1e-4) + post-embed LN
    embedding = register_module("embedding",
                    torch::nn::Embedding(cfg.vocab_size, cfg.d_model));
    torch::nn::init::uniform_(embedding->weight, -1e-4, 1e-4);

    emb_ln = register_module("emb_ln",
                 torch::nn::LayerNorm(
                     torch::nn::LayerNormOptions({cfg.d_model})));
    torch::nn::init::ones_(emb_ln->weight);
    torch::nn::init::zeros_(emb_ln->bias);

    blocks = register_module("blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        blocks->push_back(RWKVBlock(cfg, i));
    }

    ln_out = register_module("ln_out",
                 torch::nn::LayerNorm(
                     torch::nn::LayerNormOptions({cfg.d_model})));
    torch::nn::init::ones_(ln_out->weight);
    torch::nn::init::zeros_(ln_out->bias);

    head = register_module("head",
               torch::nn::Linear(
                   torch::nn::LinearOptions(cfg.d_model, cfg.vocab_size)
                       .bias(false)));
    torch::nn::init::zeros_(head->weight);
}

torch::Tensor RWKVModelImpl::forward(torch::Tensor tokens) {
    auto x = embedding->forward(tokens);   // [B, T, D]
    x = emb_ln->forward(x);               // post-embed LN

    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        x = blocks->at<RWKVBlockImpl>(i).forward(x);
    }

    x = ln_out->forward(x);               // [B, T, D]
    return head->forward(x);              // [B, T, vocab]
}

torch::Tensor RWKVModelImpl::generate(torch::Tensor prompt,
                                       int64_t max_new,
                                       double temperature,
                                       double top_p) {
    this->eval();
    torch::NoGradGuard ng;

    int64_t B   = prompt.size(0);
    int64_t T_p = prompt.size(1);
    auto    dev = prompt.device();

    // Per-layer states: x_tm_tm, x_cm_tm, aa, bb, pp
    std::vector<torch::Tensor> x_tm_tm(cfg.n_layers),
                                x_cm_tm(cfg.n_layers),
                                aa(cfg.n_layers),
                                bb(cfg.n_layers),
                                pp(cfg.n_layers);
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        auto z = torch::zeros({B, cfg.d_model},
                              torch::TensorOptions().dtype(torch::kFloat).device(dev));
        x_tm_tm[i] = z.clone();
        x_cm_tm[i] = z.clone();
        aa[i]      = z.clone();
        bb[i]      = z.clone();
        pp[i]      = z.clone().fill_(-1e30f);
    }

    // Prefill
    torch::Tensor last_tok = prompt.select(1, T_p - 1);
    for (int64_t t = 0; t < T_p; ++t) {
        auto tok = prompt.select(1, t);   // [B]
        auto x   = emb_ln->forward(embedding->forward(tok).unsqueeze(1)).squeeze(1);
        for (int64_t i = 0; i < cfg.n_layers; ++i) {
            x = blocks->at<RWKVBlockImpl>(i).step(
                    x, x_tm_tm[i], x_cm_tm[i], aa[i], bb[i], pp[i]);
        }
        last_tok = tok;
    }

    // Decode
    std::vector<torch::Tensor> generated;
    auto next_tok = last_tok;

    for (int64_t s = 0; s < max_new; ++s) {
        auto x = emb_ln->forward(embedding->forward(next_tok).unsqueeze(1)).squeeze(1);
        for (int64_t i = 0; i < cfg.n_layers; ++i) {
            x = blocks->at<RWKVBlockImpl>(i).step(
                    x, x_tm_tm[i], x_cm_tm[i], aa[i], bb[i], pp[i]);
        }
        auto logits = head->forward(ln_out->forward(x.unsqueeze(1)).squeeze(1));
        logits = logits / static_cast<float>(temperature);
        auto probs = torch::softmax(logits.to(torch::kFloat), -1);

        // Top-p sampling
        auto sorted  = torch::sort(probs, -1, true);
        auto sp      = std::get<0>(sorted);
        auto si      = std::get<1>(sorted);
        auto cum     = sp.cumsum(-1);
        auto cum_sh  = torch::cat(
            {torch::zeros({B,1}, probs.options()), cum.slice(1,0,-1)}, 1);
        sp = sp.masked_fill(cum_sh > static_cast<float>(top_p), 0.0f);
        sp = sp / sp.sum(-1, true).clamp_min(1e-8f);
        auto idx  = torch::multinomial(sp, 1).squeeze(1);
        next_tok  = si.gather(1, idx.unsqueeze(1)).squeeze(1);
        generated.push_back(next_tok);
    }

    if (generated.empty()) return prompt;
    std::vector<torch::Tensor> gv(generated.begin(), generated.end());
    return torch::stack(gv, 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Training utilities
// ─────────────────────────────────────────────────────────────────────────────
torch::optim::Adam make_rwkv_optimizer(RWKVModel& model,
                                        const RWKVTrainConfig& tcfg) {
    // No weight decay (§4.1)
    return torch::optim::Adam(
        model->parameters(),
        torch::optim::AdamOptions(tcfg.lr)
            .betas({tcfg.beta1, tcfg.beta2})
            .eps(1e-8)
            .weight_decay(0.0));
}

double rwkv_lr_schedule(int64_t step, const RWKVTrainConfig& tcfg) {
    if (tcfg.warmup > 0 && step < tcfg.warmup) {
        return tcfg.lr * (static_cast<double>(step + 1) / tcfg.warmup);
    }
    // Exponential decay: lr * (end_lr/lr)^{t/total}
    int64_t s = step - tcfg.warmup;
    int64_t T = std::max(tcfg.total_steps - tcfg.warmup, (int64_t)1);
    double ratio = static_cast<double>(s) / T;
    double log_r = std::log(tcfg.end_lr / tcfg.lr);
    return tcfg.lr * std::exp(log_r * ratio);
}

std::pair<torch::Tensor, double>
rwkv_train_step(RWKVModel& model,
                torch::optim::Adam& optimizer,
                torch::Tensor tokens,
                const RWKVTrainConfig& tcfg,
                int64_t step) {
    model->train();

    double lr_now = rwkv_lr_schedule(step, tcfg);
    for (auto& pg : optimizer.param_groups()) {
        static_cast<torch::optim::AdamOptions&>(pg.options()).lr(lr_now);
    }

    optimizer.zero_grad();

    int64_t T = tokens.size(1) - 1;
    auto inp = tokens.slice(1, 0, T);
    auto tgt = tokens.slice(1, 1, T + 1);

    auto logits = model->forward(inp);
    int64_t V   = logits.size(-1);
    auto loss   = torch::nn::functional::cross_entropy(
        logits.reshape({-1, V}),
        tgt.reshape({-1}));

    loss.backward();
    // No gradient clipping mentioned in paper for main training
    optimizer.step();

    return {loss.detach(), loss.item<double>()};
}

} // namespace nlp
} // namespace models
} // namespace dm
