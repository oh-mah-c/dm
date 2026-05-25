// ─────────────────────────────────────────────────────────────────────────────
// Tiny Recursion Model (TRM) implementation
// Jolicoeur-Martineau, arXiv:2510.04871v1, 2025
// ─────────────────────────────────────────────────────────────────────────────

#include "models/reasoning/trm/trm.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace dm {
namespace models {
namespace reasoning {

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm
// ─────────────────────────────────────────────────────────────────────────────
RMSNormImpl::RMSNormImpl(int64_t dim, float eps_) : eps(eps_) {
    weight = register_parameter("weight", torch::ones({dim}));
}

torch::Tensor RMSNormImpl::forward(torch::Tensor x) {
    // rms = sqrt(mean(x²) + eps),  out = x / rms * weight
    auto rms = x.pow(2).mean(-1, /*keepdim=*/true).add(eps).sqrt();
    return (x / rms) * weight;
}

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU MLP  —  out = (gate(x) * silu(up(x))) @ down
// ─────────────────────────────────────────────────────────────────────────────
SwiGLUMlpImpl::SwiGLUMlpImpl(int64_t dim, int64_t hidden_dim) {
    gate = register_module("gate",
               torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false)));
    up   = register_module("up",
               torch::nn::Linear(torch::nn::LinearOptions(dim, hidden_dim).bias(false)));
    down = register_module("down",
               torch::nn::Linear(torch::nn::LinearOptions(hidden_dim, dim).bias(false)));
}

torch::Tensor SwiGLUMlpImpl::forward(torch::Tensor x) {
    return down->forward(gate->forward(x) * torch::silu(up->forward(x)));
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotary Position Embedding
// ─────────────────────────────────────────────────────────────────────────────
std::pair<torch::Tensor,torch::Tensor>
build_rope_cache(int64_t seq_len, int64_t head_dim, torch::Device device) {
    // θ_i = 1 / 10000^(2i / head_dim)
    auto half = head_dim / 2;
    auto idx  = torch::arange(half, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto theta = torch::pow(10000.0f, -2.0f * idx / head_dim);   // [half]
    auto pos   = torch::arange(seq_len,
                     torch::TensorOptions().dtype(torch::kFloat32).device(device)); // [L]
    auto freqs = torch::outer(pos, theta);  // [L, half]
    return {freqs.cos(), freqs.sin()};      // each [L, half]
}

// x: [B, nh, L, head_dim]
torch::Tensor apply_rotary_emb(torch::Tensor x, torch::Tensor cos, torch::Tensor sin) {
    auto half = x.size(-1) / 2;
    auto x1 = x.slice(-1, 0, half);
    auto x2 = x.slice(-1, half, x.size(-1));
    // cos/sin: [L, half] → broadcast to [1, 1, L, half]
    auto c = cos.unsqueeze(0).unsqueeze(0);
    auto s = sin.unsqueeze(0).unsqueeze(0);
    auto rot = torch::cat({x1 * c - x2 * s,
                           x1 * s + x2 * c}, /*dim=*/-1);
    return rot;
}

// ─────────────────────────────────────────────────────────────────────────────
// TRMAttention  (multi-head self-attention with RoPE, no bias)
// ─────────────────────────────────────────────────────────────────────────────
TRMAttentionImpl::TRMAttentionImpl(int64_t dim, int64_t num_heads_)
    : num_heads(num_heads_), head_dim(dim / num_heads_) {
    qkv = register_module("qkv",
              torch::nn::Linear(torch::nn::LinearOptions(dim, 3 * dim).bias(false)));
    out = register_module("out",
              torch::nn::Linear(torch::nn::LinearOptions(dim, dim).bias(false)));
}

torch::Tensor TRMAttentionImpl::forward(torch::Tensor x) {
    auto B = x.size(0), L = x.size(1), D = x.size(2);
    // [B, L, 3D] → [B, L, 3, nh, hd] → [3, B, nh, L, hd]
    auto qkv_out = qkv->forward(x)
                       .reshape({B, L, 3, num_heads, head_dim})
                       .permute({2, 0, 3, 1, 4});
    auto q = qkv_out[0], k = qkv_out[1], v = qkv_out[2];

    // Apply RoPE
    auto [c, s] = build_rope_cache(L, head_dim, x.device());
    q = apply_rotary_emb(q, c, s);
    k = apply_rotary_emb(k, c, s);

    // Scaled dot-product attention
    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    auto attn = torch::matmul(q * scale, k.transpose(-2, -1));  // [B, nh, L, L]
    attn = torch::softmax(attn, -1);

    auto y = torch::matmul(attn, v)           // [B, nh, L, hd]
                 .transpose(1, 2)             // [B, L, nh, hd]
                 .contiguous()
                 .view({B, L, D});
    return out->forward(y);
}

// ─────────────────────────────────────────────────────────────────────────────
// TRMMixer  (MLP-Mixer replacement for self-attention, Section 4.5)
// ─────────────────────────────────────────────────────────────────────────────
TRMMixerImpl::TRMMixerImpl(int64_t seq_len) {
    mix = register_module("mix",
              torch::nn::Linear(torch::nn::LinearOptions(seq_len, seq_len).bias(false)));
}

torch::Tensor TRMMixerImpl::forward(torch::Tensor x) {
    // x: [B, L, D]  — mix over L (token axis)
    // transpose → [B, D, L], apply mix → [B, D, L], transpose back
    return mix->forward(x.transpose(1, 2)).transpose(1, 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// TRMBlock
// ─────────────────────────────────────────────────────────────────────────────
TRMBlockImpl::TRMBlockImpl(int64_t dim, int64_t num_heads, int64_t mlp_hidden,
                            bool use_attn, int64_t seq_len)
    : use_attention(use_attn) {
    norm1 = register_module("norm1", RMSNorm(dim));
    norm2 = register_module("norm2", RMSNorm(dim));
    mlp   = register_module("mlp",   SwiGLUMlp(dim, mlp_hidden));

    if (use_attention) {
        attn  = register_module("attn",  TRMAttention(dim, num_heads));
    } else {
        if (seq_len <= 0)
            throw std::invalid_argument("TRMBlock: seq_len must be >0 for MLP-Mixer");
        mixer = register_module("mixer", TRMMixer(seq_len));
    }
}

torch::Tensor TRMBlockImpl::forward(torch::Tensor x) {
    // Pre-norm, residual
    if (use_attention) {
        x = x + attn->forward(norm1->forward(x));
    } else {
        x = x + mixer->forward(norm1->forward(x));
    }
    x = x + mlp->forward(norm2->forward(x));
    return x;
}

// ─────────────────────────────────────────────────────────────────────────────
// TRMNet  (the single shared "net" in the pseudocode)
// ─────────────────────────────────────────────────────────────────────────────
TRMNetImpl::TRMNetImpl(int64_t vocab_size_, int64_t seq_len_, int64_t embed_dim_,
                        int64_t n_layers, int64_t num_heads, bool use_attention)
    : vocab_size(vocab_size_), embed_dim(embed_dim_), seq_len(seq_len_) {

    input_embed = register_module("input_embed",
                      torch::nn::Embedding(vocab_size_, embed_dim_));

    blocks = register_module("blocks", torch::nn::ModuleList());
    int64_t mlp_hidden = 4 * embed_dim_;   // standard FFN ratio

    // The network processes concatenated sequences (e.g. [x; y; z] = 3*L tokens)
    // For Mixer variant we pass 3*seq_len as seq_len to each block
    int64_t full_len = 3 * seq_len_;       // conservative: always 3 slots

    for (int64_t i = 0; i < n_layers; ++i) {
        blocks->push_back(TRMBlock(embed_dim_, num_heads, mlp_hidden,
                                   use_attention, full_len));
    }

    final_norm  = register_module("final_norm", RMSNorm(embed_dim_));
    output_head = register_module("output_head",
                      torch::nn::Linear(torch::nn::LinearOptions(embed_dim_, vocab_size_).bias(false)));
    halt_head   = register_module("halt_head",
                      torch::nn::Linear(torch::nn::LinearOptions(embed_dim_, 1).bias(true)));

    // Init output_head and halt_head with small weights
    torch::nn::init::normal_(output_head->weight, 0.0, 0.02);
    torch::nn::init::zeros_(halt_head->bias);
    torch::nn::init::normal_(halt_head->weight, 0.0, 0.02);
}

torch::Tensor TRMNetImpl::embed(torch::Tensor tokens) {
    return input_embed->forward(tokens);
}

torch::Tensor TRMNetImpl::forward(torch::Tensor x) {
    // x: [B, L_in, D]  (already embedded)
    for (size_t i = 0; i < blocks->size(); ++i) {
        x = blocks->ptr(i)->as<TRMBlockImpl>()->forward(x);
    }
    return final_norm->forward(x);
}

torch::Tensor TRMNetImpl::to_logits(torch::Tensor x) {
    // Take first seq_len positions (the answer slot)
    return output_head->forward(x.slice(1, 0, seq_len));
}

torch::Tensor TRMNetImpl::to_halt(torch::Tensor x) {
    // Use first token of first slot as halt signal
    return halt_head->forward(x.select(1, 0));  // [B, 1]
}

// ─────────────────────────────────────────────────────────────────────────────
// TRMImpl
// ─────────────────────────────────────────────────────────────────────────────
TRMImpl::TRMImpl(int64_t vocab_size_, int64_t seq_len_, int64_t embed_dim_,
                  int64_t n_layers, int64_t num_heads, bool use_attention,
                  int64_t n_recursions_, int64_t T_, double ema_rate_)
    : vocab_size(vocab_size_), seq_len(seq_len_), embed_dim(embed_dim_),
      n_recursions(n_recursions_), T(T_), ema_rate(ema_rate_) {

    net = register_module("net",
              TRMNet(vocab_size_, seq_len_, embed_dim_, n_layers,
                     num_heads, use_attention));
}

torch::Tensor TRMImpl::init_z(int64_t batch, torch::Device device) const {
    return torch::zeros({batch, seq_len, embed_dim},
                        torch::TensorOptions().device(device));
}

void TRMImpl::init_ema() {
    ema_params.clear();
    for (auto& p : net->parameters()) {
        ema_params.push_back(p.data().clone());
    }
}

void TRMImpl::update_ema() {
    if (ema_params.empty()) init_ema();
    torch::NoGradGuard ng;
    auto params = net->parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        ema_params[i].mul_(ema_rate).add_(params[i].data(), 1.0 - ema_rate);
    }
}

void TRMImpl::apply_ema() {
    if (ema_params.empty()) return;
    _live_backup.clear();
    auto params = net->parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        _live_backup.push_back(params[i].data().clone());
        params[i].data().copy_(ema_params[i]);
    }
}

void TRMImpl::restore_live() {
    if (_live_backup.empty()) return;
    auto params = net->parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        params[i].data().copy_(_live_backup[i]);
    }
    _live_backup.clear();
}

// ── Latent recursion ─────────────────────────────────────────────────────────
// Figure 3 pseudocode:
//   def latent_recursion(x, y, z, n=6):
//     for i in range(n+1):  z = net(x, y, z)
//     return y, z
//
// Here: concatenate [y ; z] for the "refine answer" slot, passing x as context.
// Actually from the pseudocode, net takes (x, y, z) — we concatenate along L.
std::pair<torch::Tensor,torch::Tensor>
TRMImpl::latent_recursion(const torch::Tensor& x_emb,
                           torch::Tensor y_emb,
                           torch::Tensor z,
                           int64_t n_steps) {
    for (int64_t i = 0; i <= n_steps; ++i) {
        // Concatenate [x_emb; y_emb; z] along seq dim → [B, 3L, D]
        auto inp = torch::cat({x_emb, y_emb, z}, 1);
        auto out = net->forward(inp);          // [B, 3L, D]
        // Split: first L → new y, last L → new z
        y_emb = out.slice(1, 0,       seq_len);
        z     = out.slice(1, seq_len, 2*seq_len);
    }
    return {y_emb, z};
}

// ── Deep recursion ───────────────────────────────────────────────────────────
// Figure 3:
//   def deep_recursion(x, y, z, n=6, T=3):
//     with no_grad:
//       for j in range(T-1): y,z = latent_recursion(x, y, z, n)
//     y,z = latent_recursion(x, y, z, n)  # with grad
//     return y.detach(), z.detach(), output_head(y), halt_head(y)
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
TRMImpl::deep_recursion(const torch::Tensor& x_emb,
                         torch::Tensor y_emb,
                         torch::Tensor z) {
    // T-1 no-grad warm-up recursions
    {
        torch::NoGradGuard ng;
        for (int64_t j = 0; j < T - 1; ++j) {
            auto [ny, nz] = latent_recursion(x_emb, y_emb, z, n_recursions);
            y_emb = ny; z = nz;
        }
    }
    // One full recursion with gradients
    auto [ny, nz] = latent_recursion(x_emb, y_emb, z, n_recursions);
    y_emb = ny; z = nz;

    // Output head: logits over vocab  [B, L, V]
    auto logits = net->to_logits(y_emb);
    // Halt head: [B, 1]
    auto q      = net->to_halt(y_emb);

    return {y_emb.detach(), z.detach(), logits, q};
}

// ── Forward (token inputs) ───────────────────────────────────────────────────
std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
TRMImpl::forward(torch::Tensor x_tokens, torch::Tensor y_tokens, torch::Tensor z) {
    auto x_emb = net->embed(x_tokens);   // [B, L, D]
    auto y_emb = net->embed(y_tokens);   // [B, L, D]
    return deep_recursion(x_emb, y_emb, z);
}

// ── Inference ────────────────────────────────────────────────────────────────
// Figure 3 outer loop:
//   z = z_init
//   for step in range(N_sup):
//     y,z,q = deep_recursion(x, y_init, z)
//     if q[0] > 0: break
//   return argmax(output_head(y))
torch::Tensor TRMImpl::predict(torch::Tensor x_tokens, int64_t n_sup) {
    net->eval();
    torch::NoGradGuard ng;
    auto B      = x_tokens.size(0);
    auto device = x_tokens.device();

    auto x_emb = net->embed(x_tokens);
    // Start from zero latent and zero answer embedding
    auto y_emb = torch::zeros({B, seq_len, embed_dim},
                     torch::TensorOptions().device(device));
    auto z     = init_z(B, device);

    torch::Tensor last_logits;
    for (int64_t step = 0; step < n_sup; ++step) {
        auto [ny, nz, logits, q] = deep_recursion(x_emb, y_emb, z);
        y_emb = ny; z = nz; last_logits = logits;
        // Early stop: ACT halt signal (Section 4.6)
        if (q.select(1, 0).mean().item<float>() > 0.f) break;
    }
    return last_logits.argmax(-1);  // [B, L]
}

// ─────────────────────────────────────────────────────────────────────────────
// Stable cross-entropy  (Section 6 — "stable-max loss")
// Clip logits to [-10, 10] before softmax to stabilise training on tiny data
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor stable_cross_entropy(torch::Tensor logits, torch::Tensor targets) {
    logits = torch::clamp(logits, -10.0f, 10.0f);
    auto B = logits.size(0);
    auto L = logits.size(1);
    auto V = logits.size(2);
    return torch::nn::functional::cross_entropy(
        logits.view({B * L, V}),
        targets.view({B * L}));
}

// ─────────────────────────────────────────────────────────────────────────────
// Learning-rate warmup schedule
// ─────────────────────────────────────────────────────────────────────────────
static void set_lr(torch::optim::AdamW& opt, double lr) {
    for (auto& pg : opt.param_groups()) pg.options().set_lr(lr);
}

static double warmup_lr(double base_lr, int64_t step, int64_t warmup_steps) {
    if (step < warmup_steps)
        return base_lr * (step + 1.0) / warmup_steps;
    return base_lr;
}

// ─────────────────────────────────────────────────────────────────────────────
// Training helpers
// ─────────────────────────────────────────────────────────────────────────────
float trm_train_epoch(
        TRM&                  model,
        torch::optim::AdamW&  optimizer,
        const TRMTrainConfig& cfg,
        int64_t               global_step,
        const std::vector<std::pair<torch::Tensor,torch::Tensor>>& batches) {

    model->train();
    double total_loss = 0.0;
    int64_t steps     = 0;

    for (auto& [x_tok, y_true] : batches) {
        auto x = x_tok.to(cfg.device);
        auto y = y_true.to(cfg.device);
        auto B = x.size(0);

        // LR warmup
        set_lr(optimizer, warmup_lr(cfg.lr, global_step, cfg.warmup_iters));
        ++global_step;

        // Initialise latent and answer embedding for this batch
        auto z     = model->init_z(B, cfg.device);
        auto y_emb = model->net->embed(
                         torch::zeros({B, model->seq_len},
                             torch::TensorOptions().dtype(torch::kLong).device(cfg.device)));
        auto x_emb = model->net->embed(x);

        // ── Deep supervision loop (Figure 3 outer loop) ──────────────────────
        // for step in range(N_sup):
        //   (y_hat, q_hat) = deep_recursion(...)
        //   loss = cross_entropy(y_hat, y_true)
        //        + 0.5 * bce(q_hat, y_hat==y_true)
        //   z = z.detach()
        //   backward; step; zero_grad
        //   if q_hat > 0: break
        for (int64_t sup = 0; sup < cfg.n_sup; ++sup) {
            auto [ny, nz, logits, q] = model->deep_recursion(x_emb, y_emb, z);

            // Prediction loss
            auto loss = stable_cross_entropy(logits, y);

            // Halt loss: binary CE whether prediction matches target
            auto pred_correct = (logits.argmax(-1) == y).all(1).to(torch::kFloat);
            loss = loss + 0.5f * torch::nn::functional::binary_cross_entropy_with_logits(
                             q.squeeze(1), pred_correct);

            optimizer.zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
            optimizer.step();

            total_loss += loss.item<double>();
            ++steps;

            // Update EMA after each optimiser step
            model->update_ema();

            // Detach latent for next supervision step (no BPTT across steps)
            y_emb = ny;
            z     = nz;  // already detached from deep_recursion

            // ACT early stop (Section 4.6): if model signals halt, move on
            if (q.mean().item<float>() > 0.f) break;
        }
    }
    return steps > 0 ? static_cast<float>(total_loss / steps) : 0.f;
}

float trm_evaluate(
        TRM&          model,
        torch::Device device,
        const std::vector<std::pair<torch::Tensor,torch::Tensor>>& batches,
        int64_t n_sup) {

    // Use EMA weights for evaluation (Section 4.7)
    model->apply_ema();
    model->eval();

    int64_t correct = 0, total = 0;
    {
        torch::NoGradGuard ng;
        for (auto& [x_tok, y_true] : batches) {
            auto x   = x_tok.to(device);
            auto y   = y_true.to(device);
            auto pred = model->predict(x, n_sup);
            // Sequence-level accuracy: entire output must match
            correct += (pred == y).all(1).sum().item<int64_t>();
            total   += x.size(0);
        }
    }
    model->restore_live();
    return total > 0 ? static_cast<float>(correct) / total : 0.f;
}

void trm_train(
        TRM&                  model,
        const TRMTrainConfig& cfg,
        const std::vector<std::pair<torch::Tensor,torch::Tensor>>& train_batches,
        const std::vector<std::pair<torch::Tensor,torch::Tensor>>& val_batches,
        const std::string&    save_path) {

    model->to(cfg.device);
    model->init_ema();

    // Separate LR for embeddings (Section 6: 1e-2 for embeddings)
    std::vector<torch::optim::OptimizerParamGroup> param_groups;

    // Embedding parameters
    std::vector<torch::Tensor> embed_params;
    for (auto& item : model->net->named_parameters()) {
        if (item.key().find("embed") != std::string::npos) {
            embed_params.push_back(item.value());
        }
    }
    // All other parameters
    std::vector<torch::Tensor> other_params;
    for (auto& item : model->net->named_parameters()) {
        if (item.key().find("embed") == std::string::npos) {
            other_params.push_back(item.value());
        }
    }

    auto embed_opts = torch::optim::AdamWOptions(cfg.embed_lr)
                          .betas({cfg.beta1, cfg.beta2})
                          .weight_decay(0.0);  // no WD on embeddings
    auto other_opts = torch::optim::AdamWOptions(cfg.lr)
                          .betas({cfg.beta1, cfg.beta2})
                          .weight_decay(cfg.weight_decay);

    torch::optim::AdamW optimizer(
        {torch::optim::OptimizerParamGroup(embed_params,
             std::make_unique<torch::optim::AdamWOptions>(embed_opts)),
         torch::optim::OptimizerParamGroup(other_params,
             std::make_unique<torch::optim::AdamWOptions>(other_opts))});

    float best_acc = 0.f;
    int64_t global_step = 0;

    for (int64_t epoch = 0; epoch < cfg.max_epochs; ++epoch) {
        float loss = trm_train_epoch(model, optimizer, cfg, global_step, train_batches);
        global_step += static_cast<int64_t>(train_batches.size()) * cfg.n_sup;

        if ((epoch + 1) % 1000 == 0 || epoch == 0) {
            float acc = trm_evaluate(model, cfg.device, val_batches, cfg.n_sup);
            std::cout << "[TRM] epoch " << epoch + 1
                      << "  loss=" << loss
                      << "  val_acc=" << acc * 100.f << "%\n";
            if (acc > best_acc) {
                best_acc = acc;
                torch::serialize::OutputArchive archive;
                model->save(archive);
                archive.save_to(save_path);
                std::cout << "[TRM] saved checkpoint → " << save_path << "\n";
            }
        } else if ((epoch + 1) % 100 == 0) {
            std::cout << "[TRM] epoch " << epoch + 1
                      << "  loss=" << loss << "\n";
        }
    }
}

} // namespace reasoning
} // namespace models
} // namespace dm
