// ─────────────────────────────────────────────────────────────────────────────
// mamba.cpp — Mamba selective SSM implementation
//
// A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State
// Spaces," arXiv:2312.00752v2, 2023. https://arxiv.org/abs/2312.00752
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/mamba/mamba.h"

#include <torch/torch.h>
#include <cmath>
#include <algorithm>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// MambaRMSNorm
// ─────────────────────────────────────────────────────────────────────────────
MambaRMSNormImpl::MambaRMSNormImpl(int64_t dim, float eps_)
    : eps(eps_) {
    w = register_parameter("w", torch::ones(dim));
}

torch::Tensor MambaRMSNormImpl::forward(torch::Tensor x) {
    auto xf  = x.to(torch::kFloat);
    auto rms = xf.pow(2).mean(-1, /*keepdim=*/true).add(eps).rsqrt();
    return (xf * rms).type_as(x) * w;
}

// ─────────────────────────────────────────────────────────────────────────────
// MambaSSM — selective state space (Algorithm 2, §3.2)
// ─────────────────────────────────────────────────────────────────────────────
MambaSSMImpl::MambaSSMImpl(const MambaConfig& cfg)
    : d_inner(cfg.d_inner()), d_state(cfg.d_state), dt_rank_(cfg.dt_rank()) {

    // A is a fixed diagonal matrix; we store log(-A) and exponentiate.
    // S4D-Real init: A[i] = -(i+1) for i in [0, N)  (§3.6)
    auto A_init = torch::arange(1, cfg.d_state + 1, torch::kFloat)
                      .unsqueeze(0)
                      .expand({d_inner, cfg.d_state});  // [d_inner, N]
    A_log = register_parameter("A_log", torch::log(A_init));  // log(i+1) > 0

    // D: skip connection weight, init to ones  (§3.4)
    D = register_parameter("D", torch::ones(d_inner));

    // x_proj: u → [dt_rank + 2*N]  (Δ_raw, B, C packed together)
    x_proj = register_module("x_proj",
                 torch::nn::Linear(
                     torch::nn::LinearOptions(d_inner, dt_rank_ + 2 * cfg.d_state)
                         .bias(false)));

    // dt_proj: dt_rank → d_inner  WITH bias (the bias is the dt_bias)
    // Initialized so that softplus(bias) ≈ Uniform([0.001, 0.1])  (§3.6)
    dt_proj = register_module("dt_proj",
                  torch::nn::Linear(
                      torch::nn::LinearOptions(dt_rank_, d_inner)
                          .bias(true)));

    // Δ bias init: inverse softplus of Uniform([0.001, 0.1])
    // softplus^{-1}(x) = log(exp(x) - 1)
    {
        torch::NoGradGuard ng;
        auto u = torch::rand(d_inner) * (0.1 - 0.001) + 0.001; // Uniform(0.001, 0.1)
        // inv_softplus(u) = log(exp(u) - 1)
        auto dt_bias_init = torch::log(torch::exp(u) - 1.0f);
        dt_proj->bias.copy_(dt_bias_init);
    }

    // x_proj weight init
    torch::nn::init::xavier_uniform_(x_proj->weight);

    // dt_proj weight: special init — near-identity-like for stability
    // Paper uses random init; we use kaiming uniform
    torch::nn::init::kaiming_uniform_(dt_proj->weight, std::sqrt(5.0));
}

// Parallel selective scan (training mode, processes whole sequence at once)
// u: [B, L, d_inner]
// Returns y: [B, L, d_inner]
torch::Tensor MambaSSMImpl::forward(torch::Tensor u) {
    int64_t B = u.size(0);
    int64_t L = u.size(1);

    // -A from log storage (always positive since we stored log(|A|))
    // A: [d_inner, N]  (negative diagonal entries)
    auto A = -torch::exp(A_log.to(torch::kFloat));  // [d_inner, N]

    // Project u to (Δ_raw, B_ssm, C_ssm) — pack all three into one projection
    // x_proj_out: [B, L, dt_rank + 2*N]
    auto x_proj_out = x_proj->forward(u);

    // Split: dt_raw [B,L,dt_rank], B_ssm [B,L,N], C_ssm [B,L,N]
    auto splits = x_proj_out.split({dt_rank_, d_state, d_state}, /*dim=*/-1);
    auto dt_raw  = splits[0];   // [B, L, dt_rank]
    auto B_ssm   = splits[1];   // [B, L, N]
    auto C_ssm   = splits[2];   // [B, L, N]

    // Δ = softplus(dt_bias + dt_proj(dt_raw))   [B, L, d_inner]
    auto dt = torch::nn::functional::softplus(
        dt_proj->forward(dt_raw));   // [B, L, d_inner]

    // ── Discretization (ZOH, eq.4) ──────────────────────────────────────
    // Ā_t = exp(Δ_t ⊗ A)                      [B, L, d_inner, N]
    // B̄_t = Δ_t ⊗ B_t  (simplified: (ΔA)^{-1}(exp(ΔA)-I)ΔB ≈ Δ·B for small Δ)
    //   The paper's efficient implementation uses the exact ZOH for A
    //   and the simplified Δ·B for B (common in practice).
    // dA: [B, L, d_inner, N]
    auto dA = torch::exp(
        dt.unsqueeze(-1) *           // [B, L, d_inner, 1]
        A.unsqueeze(0).unsqueeze(0)  // [1, 1, d_inner, N]
    );
    // dB: [B, L, d_inner, N]
    auto dB = dt.unsqueeze(-1) *        // [B, L, d_inner, 1]
              B_ssm.unsqueeze(2);       // [B, L, 1,      N]

    // ── Parallel sequential scan ─────────────────────────────────────────
    // h_t = Ā_t h_{t-1} + B̄_t u_t,  y_t = C_t h_t
    // We implement the sequential scan in a simple loop.
    // (A hardware-aware parallel scan requires custom CUDA — §3.3, App D.
    //  For a LibTorch-portable pure-PyTorch training path, the sequential
    //  loop on CPU is correct; for GPU training, torch.compile or a
    //  hand-written CUDA kernel replaces this section.)
    auto h = torch::zeros({B, d_inner, d_state},
                          torch::TensorOptions()
                              .dtype(torch::kFloat)
                              .device(u.device()));

    std::vector<torch::Tensor> ys;
    ys.reserve(L);

    for (int64_t t = 0; t < L; ++t) {
        // dA_t: [B, d_inner, N]
        auto dA_t = dA.select(1, t);
        // dB_t: [B, d_inner, N]
        auto dB_t = dB.select(1, t);
        // u_t:  [B, d_inner]
        auto u_t  = u.select(1, t);

        // h = Ā·h + B̄·u_t
        // u_t.unsqueeze(-1): [B, d_inner, 1]  → broadcast over N
        h = dA_t * h + dB_t * u_t.unsqueeze(-1);  // [B, d_inner, N]

        // y_t = C_t · h_t  (sum over N)
        // C_ssm[:,t,:]: [B, N], h: [B, d_inner, N]
        // y_t = sum_n C[b,n] * h[b,d,n]  → [B, d_inner]
        auto C_t = C_ssm.select(1, t);              // [B, N]
        // einsum: b n, b d n -> b d
        auto y_t = (h * C_t.unsqueeze(1)).sum(-1);  // [B, d_inner]
        ys.push_back(y_t);
    }

    // Stack: [B, L, d_inner]
    std::vector<torch::Tensor> ys_vec(ys.begin(), ys.end());
    auto y = torch::stack(ys_vec, 1).type_as(u);

    // Add skip connection: y += D ⊙ u  (§3.4)
    y = y + u * D.unsqueeze(0).unsqueeze(0);  // [B, L, d_inner]
    return y;
}

// Single-step recurrent inference
// u_t: [B, d_inner],  h: [B, d_inner, N]  (in-place update)
torch::Tensor MambaSSMImpl::step(torch::Tensor u_t, torch::Tensor& h) {
    auto A = -torch::exp(A_log.to(torch::kFloat));   // [d_inner, N]

    // Project
    auto xp_out = x_proj->forward(u_t);              // [B, dt_rank+2N]
    auto splits  = xp_out.split({dt_rank_, d_state, d_state}, -1);
    auto dt_raw  = splits[0];  // [B, dt_rank]
    auto B_t     = splits[1];  // [B, N]
    auto C_t     = splits[2];  // [B, N]

    auto dt = torch::nn::functional::softplus(
        dt_proj->forward(dt_raw));  // [B, d_inner]

    auto dA = torch::exp(dt.unsqueeze(-1) *
                         A.unsqueeze(0));             // [B, d_inner, N]
    auto dB = dt.unsqueeze(-1) *
              B_t.unsqueeze(1);                       // [B, d_inner, N]

    // Update hidden state in-place
    h = dA * h + dB * u_t.unsqueeze(-1);             // [B, d_inner, N]

    // Output: y = C·h + D·u
    auto y = (h * C_t.unsqueeze(1)).sum(-1)           // [B, d_inner]
             + D.unsqueeze(0) * u_t;
    return y.type_as(u_t);
}

// ─────────────────────────────────────────────────────────────────────────────
// MambaBlock
// ─────────────────────────────────────────────────────────────────────────────
MambaBlockImpl::MambaBlockImpl(const MambaConfig& cfg)
    : d_model(cfg.d_model), d_inner(cfg.d_inner()), d_conv(cfg.d_conv) {

    norm    = register_module("norm",     MambaRMSNorm(cfg.d_model));
    // in_proj: d_model → 2*d_inner  (x branch + z gate branch, packed)
    in_proj = register_module("in_proj",
                  torch::nn::Linear(
                      torch::nn::LinearOptions(cfg.d_model, 2 * d_inner)
                          .bias(false)));
    // Depthwise conv1d: groups=d_inner, kernel=d_conv, padding=d_conv-1
    conv1d  = register_module("conv1d",
                  torch::nn::Conv1d(
                      torch::nn::Conv1dOptions(d_inner, d_inner, cfg.d_conv)
                          .groups(d_inner)
                          .padding(cfg.d_conv - 1)
                          .bias(true)));
    ssm     = register_module("ssm",     MambaSSM(cfg));
    out_proj= register_module("out_proj",
                  torch::nn::Linear(
                      torch::nn::LinearOptions(d_inner, cfg.d_model)
                          .bias(false)));

    // Init
    torch::nn::init::xavier_uniform_(in_proj->weight);
    torch::nn::init::xavier_uniform_(out_proj->weight);
    torch::nn::init::normal_(conv1d->weight, 0.0, 0.02);
    torch::nn::init::zeros_(conv1d->bias);
}

torch::Tensor MambaBlockImpl::forward(torch::Tensor x) {
    // x: [B, L, d_model]
    auto residual = x;

    // Pre-norm
    x = norm->forward(x);

    // Input projection → split into x and z branches
    auto xz = in_proj->forward(x);             // [B, L, 2*d_inner]
    auto chunks = xz.chunk(2, /*dim=*/-1);
    auto x_br = chunks[0];                      // [B, L, d_inner]
    auto z_br = chunks[1];                      // [B, L, d_inner]

    // Depthwise conv1d on x branch
    // Conv1d expects [B, C, L] — transpose in/out
    x_br = x_br.transpose(1, 2);               // [B, d_inner, L]
    x_br = conv1d->forward(x_br);              // [B, d_inner, L + d_conv-1]
    x_br = x_br.slice(2, 0, residual.size(1)); // [B, d_inner, L]
    x_br = x_br.transpose(1, 2);               // [B, L, d_inner]

    // SiLU activation (§3.4: "SiLU/Swish")
    x_br = torch::silu(x_br);

    // Selective SSM scan
    auto y = ssm->forward(x_br);               // [B, L, d_inner]

    // Gated output: y ⊙ SiLU(z)
    y = y * torch::silu(z_br);                 // [B, L, d_inner]

    // Output projection + residual
    return out_proj->forward(y) + residual;     // [B, L, d_model]
}

torch::Tensor MambaBlockImpl::step(torch::Tensor x_t,
                                    torch::Tensor& h,
                                    torch::Tensor& conv_state) {
    // x_t: [B, d_model]
    auto residual = x_t;

    x_t = norm->forward(x_t.unsqueeze(1)).squeeze(1);  // [B, d_model]

    auto xz = in_proj->forward(x_t);           // [B, 2*d_inner]
    auto chunks = xz.chunk(2, -1);
    auto x_br = chunks[0];                      // [B, d_inner]
    auto z_br = chunks[1];                      // [B, d_inner]

    // Shift conv_state left, append new x
    // conv_state: [B, d_inner, d_conv]
    conv_state = torch::cat(
        {conv_state.slice(2, 1),               // [B, d_inner, d_conv-1]
         x_br.unsqueeze(2)},                   // [B, d_inner, 1]
        2);                                     // [B, d_inner, d_conv]

    // Apply conv: sum over kernel positions
    // conv1d weight: [d_inner, 1, d_conv] (depthwise)
    auto w = conv1d->weight.squeeze(1);        // [d_inner, d_conv]
    auto b = conv1d->bias;                     // [d_inner]
    // x_conv = sum_k w[d,k] * conv_state[b,d,k]  + bias[d]
    x_br = (conv_state * w.unsqueeze(0)).sum(-1) + b.unsqueeze(0); // [B, d_inner]

    x_br = torch::silu(x_br);

    // SSM step
    auto y = ssm->step(x_br, h);              // [B, d_inner]

    y = y * torch::silu(z_br);               // [B, d_inner]

    return out_proj->forward(y) + residual;   // [B, d_model]
}

// ─────────────────────────────────────────────────────────────────────────────
// MambaModel
// ─────────────────────────────────────────────────────────────────────────────
MambaModelImpl::MambaModelImpl(const MambaConfig& c) : cfg(c) {
    embedding = register_module("embedding",
                    torch::nn::Embedding(cfg.vocab_size, cfg.d_model));
    blocks    = register_module("blocks", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        blocks->push_back(MambaBlock(cfg));
    }
    norm_f  = register_module("norm_f",  MambaRMSNorm(cfg.d_model));
    lm_head = register_module("lm_head",
                  torch::nn::Linear(
                      torch::nn::LinearOptions(cfg.d_model, cfg.vocab_size)
                          .bias(false)));

    // Tie weights
    if (cfg.tie_weights) {
        lm_head->weight = embedding->weight;
    }

    // Init
    torch::nn::init::normal_(embedding->weight, 0.0, 0.02);
    if (!cfg.tie_weights) {
        torch::nn::init::normal_(lm_head->weight, 0.0,
                                  1.0 / std::sqrt(cfg.d_model));
    }
}

torch::Tensor MambaModelImpl::forward(torch::Tensor tokens) {
    // tokens: [B, L]
    auto x = embedding->forward(tokens);      // [B, L, d_model]
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        x = blocks->at<MambaBlockImpl>(i).forward(x);
    }
    x = norm_f->forward(x);                   // [B, L, d_model]
    return lm_head->forward(x);               // [B, L, vocab_size]
}

torch::Tensor MambaModelImpl::generate(torch::Tensor prompt,
                                        int64_t max_new,
                                        double temperature,
                                        double top_p) {
    this->eval();
    torch::NoGradGuard ng;

    int64_t B   = prompt.size(0);
    auto    dev = prompt.device();

    // Allocate per-layer hidden states and conv states
    std::vector<torch::Tensor> hs(cfg.n_layers);
    std::vector<torch::Tensor> convs(cfg.n_layers);
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        hs[i]    = torch::zeros({B, cfg.d_inner(), cfg.d_state},
                                torch::TensorOptions().dtype(torch::kFloat).device(dev));
        convs[i] = torch::zeros({B, cfg.d_inner(), cfg.d_conv},
                                torch::TensorOptions().dtype(torch::kFloat).device(dev));
    }

    // Prefill: feed prompt tokens one by one to warm up the state
    int64_t T_prompt = prompt.size(1);
    torch::Tensor last_logits;
    for (int64_t t = 0; t < T_prompt; ++t) {
        auto tok = prompt.select(1, t);                      // [B]
        auto x   = embedding->forward(tok);                  // [B, d_model]
        for (int64_t i = 0; i < cfg.n_layers; ++i) {
            x = blocks->at<MambaBlockImpl>(i).step(x, hs[i], convs[i]);
        }
        x = norm_f->forward(x.unsqueeze(1)).squeeze(1);
        last_logits = lm_head->forward(x);                   // [B, vocab]
    }

    // Decode
    std::vector<torch::Tensor> generated;
    auto next_tok = prompt.select(1, T_prompt - 1);           // [B]

    for (int64_t s = 0; s < max_new; ++s) {
        auto x = embedding->forward(next_tok);                // [B, d_model]
        for (int64_t i = 0; i < cfg.n_layers; ++i) {
            x = blocks->at<MambaBlockImpl>(i).step(x, hs[i], convs[i]);
        }
        x = norm_f->forward(x.unsqueeze(1)).squeeze(1);
        auto logits = lm_head->forward(x);                    // [B, vocab]

        // Temperature scaling + top-p sampling
        logits = logits / static_cast<float>(temperature);
        auto probs = torch::softmax(logits.to(torch::kFloat), -1);

        // Top-p (nucleus) sampling
        auto sorted = torch::sort(probs, -1, /*descending=*/true);
        auto sorted_probs   = std::get<0>(sorted);
        auto sorted_indices = std::get<1>(sorted);
        auto cum_probs = sorted_probs.cumsum(-1);
        // Shift right so first element is always kept
        auto cum_shift = torch::cat(
            {torch::zeros({B, 1}, probs.options()),
             cum_probs.slice(1, 0, -1)}, 1);
        auto mask_out  = cum_shift > static_cast<float>(top_p);
        sorted_probs   = sorted_probs.masked_fill(mask_out, 0.0f);
        sorted_probs   = sorted_probs / sorted_probs.sum(-1, true).clamp_min(1e-8f);

        auto sampled_idx = torch::multinomial(sorted_probs, 1).squeeze(1); // [B]
        // Map back to vocab indices
        next_tok = sorted_indices.gather(1, sampled_idx.unsqueeze(1)).squeeze(1);
        generated.push_back(next_tok);
    }

    if (generated.empty()) return prompt;
    std::vector<torch::Tensor> gv(generated.begin(), generated.end());
    return torch::stack(gv, 1);  // [B, max_new]
}

// ─────────────────────────────────────────────────────────────────────────────
// Training utilities
// ─────────────────────────────────────────────────────────────────────────────
torch::optim::AdamW make_mamba_optimizer(MambaModel& model,
                                          const MambaTrainConfig& tcfg) {
    // Separate 2-D (weight_decay) from 1-D (no decay) — same as Llama 2 / GPT
    std::vector<torch::Tensor> decay_params, nodecay_params;
    for (auto& item : model->named_parameters()) {
        if (item.value().dim() >= 2) {
            decay_params.push_back(item.value());
        } else {
            nodecay_params.push_back(item.value());
        }
    }
    std::vector<torch::optim::OptimizerParamGroup> groups;
    auto make_opts = [&](double wd) {
        auto opts = std::make_unique<torch::optim::AdamWOptions>(tcfg.lr);
        opts->weight_decay(wd);
        opts->eps(1e-8);
        return opts;
    };
    groups.emplace_back(decay_params,   make_opts(tcfg.weight_decay));
    groups.emplace_back(nodecay_params, make_opts(0.0));
    return torch::optim::AdamW(std::move(groups));
}

double mamba_lr_schedule(int64_t step, const MambaTrainConfig& tcfg) {
    if (tcfg.warmup > 0 && step < tcfg.warmup) {
        return tcfg.lr * (static_cast<double>(step + 1) / tcfg.warmup);
    }
    // Cosine decay
    int64_t decay_steps = std::max(tcfg.total_steps - tcfg.warmup, (int64_t)1);
    int64_t s = step - tcfg.warmup;
    double ratio = std::min(1.0, static_cast<double>(s) / decay_steps);
    double coeff = 0.5 * (1.0 + std::cos(M_PI * ratio));
    return tcfg.min_lr + coeff * (tcfg.lr - tcfg.min_lr);
}

std::pair<torch::Tensor, double>
mamba_train_step(MambaModel& model,
                 torch::optim::AdamW& optimizer,
                 torch::Tensor tokens,
                 const MambaTrainConfig& tcfg,
                 int64_t step) {
    model->train();

    // Update LR
    double lr_now = mamba_lr_schedule(step, tcfg);
    for (auto& pg : optimizer.param_groups()) {
        static_cast<torch::optim::AdamWOptions&>(pg.options()).lr(lr_now);
    }

    optimizer.zero_grad();

    int64_t T = tokens.size(1) - 1;
    auto inp = tokens.slice(1, 0, T);    // [B, T]
    auto tgt = tokens.slice(1, 1, T+1);  // [B, T]

    auto logits = model->forward(inp);   // [B, T, vocab]
    int64_t V   = logits.size(-1);
    auto loss   = torch::nn::functional::cross_entropy(
        logits.reshape({-1, V}),
        tgt.reshape({-1}));

    loss.backward();
    torch::nn::utils::clip_grad_norm_(model->parameters(), tcfg.clip);
    optimizer.step();

    double nll = loss.item<double>();
    return {loss.detach(), nll};
}

} // namespace nlp
} // namespace models
} // namespace dm
