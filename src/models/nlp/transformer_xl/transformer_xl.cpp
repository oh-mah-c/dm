// ─────────────────────────────────────────────────────────────────────────────
// transformer_xl.cpp — Transformer-XL implementation
//
// Dai et al., "Transformer-XL: Attentive Language Models Beyond a Fixed-Length
// Context," ACL 2019. https://aclanthology.org/P19-1285
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/transformer_xl/transformer_xl.h"

#include <torch/torch.h>
#include <cmath>
#include <algorithm>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Sinusoidal relative position table  R ∈ R^{klen × d_model}  (§3.3)
//
// R_i encodes relative distance i (i=0 → most recent).
// Uses the same sinusoid formula as the original Transformer PE but applied
// to relative distances, giving an inductive positional bias that can
// generalise to longer sequences at test time.
//
// PE(i, 2k)   = sin(i / 10000^{2k/d})
// PE(i, 2k+1) = cos(i / 10000^{2k/d})
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor make_rel_pos_encoding(int64_t klen, int64_t d_model,
                                    torch::Device device) {
    // positions: 0, 1, …, klen-1  (index 0 = most-recent)
    auto pos = torch::arange(klen, torch::dtype(torch::kFloat).device(device))
                   .unsqueeze(1);                           // [klen, 1]
    int64_t half = d_model / 2;
    auto div = torch::exp(
        torch::arange(0, half, torch::dtype(torch::kFloat).device(device)) *
        (-std::log(10000.0) / half));                      // [half]

    auto angles = pos * div;                               // [klen, half]
    auto R = torch::zeros({klen, d_model},
                          torch::dtype(torch::kFloat).device(device));
    R.slice(1, 0, half)      = torch::sin(angles);
    R.slice(1, half, d_model) = torch::cos(angles);
    return R;  // [klen, d_model]
}

// ─────────────────────────────────────────────────────────────────────────────
// TXLRelAttn
// ─────────────────────────────────────────────────────────────────────────────
TXLRelAttnImpl::TXLRelAttnImpl(const TransformerXLConfig& cfg)
    : n_heads(cfg.n_heads), d_head(cfg.d_head), d_model(cfg.d_model) {

    int64_t inner = cfg.n_heads * cfg.d_head;

    // Query, content-key, position-key, value projections (no bias — paper)
    Wq   = register_module("Wq",   torch::nn::Linear(
                torch::nn::LinearOptions(cfg.d_model, inner).bias(false)));
    Wk_E = register_module("Wk_E", torch::nn::Linear(
                torch::nn::LinearOptions(cfg.d_model, inner).bias(false)));
    Wk_R = register_module("Wk_R", torch::nn::Linear(
                torch::nn::LinearOptions(cfg.d_model, inner).bias(false)));
    Wv   = register_module("Wv",   torch::nn::Linear(
                torch::nn::LinearOptions(cfg.d_model, inner).bias(false)));
    Wo   = register_module("Wo",   torch::nn::Linear(
                torch::nn::LinearOptions(inner, cfg.d_model).bias(false)));

    if (cfg.attn_drop > 0.0f)
        attn_drop_layer = register_module("attn_drop",
                              torch::nn::Dropout(cfg.attn_drop));

    // Global bias vectors u ∈ R^{n_heads×d_head}, v ∈ R^{n_heads×d_head}
    // Initialised to zero (Appendix B default)
    u = register_parameter("u", torch::zeros({n_heads, d_head}));
    v = register_parameter("v", torch::zeros({n_heads, d_head}));

    // Init projections
    torch::nn::init::xavier_uniform_(Wq->weight);
    torch::nn::init::xavier_uniform_(Wk_E->weight);
    torch::nn::init::xavier_uniform_(Wk_R->weight);
    torch::nn::init::xavier_uniform_(Wv->weight);
    torch::nn::init::xavier_uniform_(Wo->weight);
}

// Relative shift — Appendix B trick
// Input x: [B, n_heads, q_len, klen+1]  (zero-padded left column)
// Output:  [B, n_heads, q_len, klen]    after shifting
torch::Tensor TXLRelAttnImpl::rel_shift(torch::Tensor x) {
    // x shape: [B, H, q, klen+1]
    auto sizes = x.sizes();
    int64_t B    = sizes[0];
    int64_t H    = sizes[1];
    int64_t q    = sizes[2];
    int64_t kp1  = sizes[3];

    // Reshape to [B, H, klen+1, q] then slice
    auto x_pad = x.view({B, H, kp1, q});
    // Remove the first row (the zero-pad column, now the first row after reshape)
    auto x_shifted = x_pad.slice(2, 1);    // [B, H, klen, q]
    return x_shifted.view({B, H, q, kp1 - 1});  // [B, H, q, klen]
}

torch::Tensor TXLRelAttnImpl::forward(torch::Tensor h_cur,
                                       torch::Tensor h_mem,
                                       torch::Tensor r,
                                       torch::Tensor attn_mask) {
    // h_cur: [B, q, d_model]
    // h_mem: [B, m, d_model]  — may be empty (m=0)
    // r:     [klen, d_model]   klen = q + m
    // attn_mask: [q, klen] bool — true = mask out

    int64_t B = h_cur.size(0);
    int64_t q = h_cur.size(1);

    // Concatenate memory + current segment for key/value context
    torch::Tensor h_ext;
    if (h_mem.numel() > 0) {
        h_ext = torch::cat({h_mem, h_cur}, 1);  // [B, m+q, d]
    } else {
        h_ext = h_cur;
    }
    int64_t klen = h_ext.size(1);

    // ── Projections ───────────────────────────────────────────────────────
    // Queries from current segment only
    auto Q = Wq->forward(h_cur);     // [B, q, H*dh]
    auto K = Wk_E->forward(h_ext);   // [B, klen, H*dh]
    auto V = Wv->forward(h_ext);     // [B, klen, H*dh]
    auto R = Wk_R->forward(r);       // [klen, H*dh]

    // Reshape to multi-head
    Q = Q.view({B, q,    n_heads, d_head}).transpose(1, 2); // [B, H, q, dh]
    K = K.view({B, klen, n_heads, d_head}).transpose(1, 2); // [B, H, klen, dh]
    V = V.view({B, klen, n_heads, d_head}).transpose(1, 2); // [B, H, klen, dh]
    R = R.view({klen, n_heads, d_head}).permute({1, 0, 2}); // [H, klen, dh]

    // Bias vectors: broadcast over batch and query positions
    // u: [n_heads, d_head] → [1, H, 1, dh]
    auto u_b = u.unsqueeze(0).unsqueeze(2);  // [1, H, 1, dh]
    auto v_b = v.unsqueeze(0).unsqueeze(2);  // [1, H, 1, dh]

    // Term (a)+(c): (Q + u) · K^T  — content-based addressing
    auto AC = torch::matmul(Q + u_b, K.transpose(-2, -1));  // [B, H, q, klen]

    // Term (b)+(d): (Q + v) · R^T  — position-based addressing
    // Need to apply relative shift trick (Appendix B).
    // Compute (Q + v_b) @ R^T:  [B,H,q,dh] × [H,dh,klen] → [B,H,q,klen]
    // Pad left with a zero column for the shift trick
    auto BD_full = torch::matmul(
        Q + v_b,
        R.transpose(-2, -1));   // [B, H, q, klen]

    // Pad a zero column on the left → [B, H, q, klen+1]
    auto zero_pad = torch::zeros({B, n_heads, q, 1},
                                 BD_full.options());
    auto BD_pad = torch::cat({zero_pad, BD_full}, -1);  // [B, H, q, klen+1]
    auto BD = rel_shift(BD_pad);                         // [B, H, q, klen]

    // Scale and add
    double scale = 1.0 / std::sqrt(static_cast<double>(d_head));
    auto attn_score = (AC + BD) * scale;  // [B, H, q, klen]

    // Apply causal mask (mask out future + mem positions beyond klen)
    if (attn_mask.numel() > 0) {
        // attn_mask: [q, klen], bool — true = masked
        attn_score = attn_score.masked_fill(
            attn_mask.unsqueeze(0).unsqueeze(0), -1e9f);
    }

    auto attn_w = torch::softmax(attn_score, -1);  // [B, H, q, klen]
    if (attn_drop_layer) {
        attn_w = attn_drop_layer->forward(attn_w);
    }

    // Weighted sum of values
    auto out = torch::matmul(attn_w, V);          // [B, H, q, dh]
    out = out.transpose(1, 2).contiguous()         // [B, q, H, dh]
              .view({B, q, n_heads * d_head});      // [B, q, H*dh]
    return Wo->forward(out);                        // [B, q, d_model]
}

// ─────────────────────────────────────────────────────────────────────────────
// TXLFFN
// ─────────────────────────────────────────────────────────────────────────────
TXLFFNImpl::TXLFFNImpl(const TransformerXLConfig& cfg) {
    fc1  = register_module("fc1",  torch::nn::Linear(cfg.d_model, cfg.d_inner));
    fc2  = register_module("fc2",  torch::nn::Linear(cfg.d_inner, cfg.d_model));
    drop = register_module("drop", torch::nn::Dropout(cfg.dropout));
}

torch::Tensor TXLFFNImpl::forward(torch::Tensor x) {
    return fc2->forward(drop->forward(torch::relu(fc1->forward(x))));
}

// ─────────────────────────────────────────────────────────────────────────────
// TXLLayer — pre-norm residual layer
// ─────────────────────────────────────────────────────────────────────────────
TXLLayerImpl::TXLLayerImpl(const TransformerXLConfig& cfg) {
    attn = register_module("attn", TXLRelAttn(cfg));
    ffn  = register_module("ffn",  TXLFFN(cfg));
    ln1  = register_module("ln1",  torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({cfg.d_model})));
    ln2  = register_module("ln2",  torch::nn::LayerNorm(
                torch::nn::LayerNormOptions({cfg.d_model})));
    drop = register_module("drop", torch::nn::Dropout(cfg.dropout));
}

torch::Tensor TXLLayerImpl::forward(torch::Tensor h_cur,
                                     torch::Tensor h_mem,
                                     torch::Tensor r,
                                     torch::Tensor attn_mask) {
    // Pre-norm attention
    auto attn_in = ln1->forward(h_cur);
    // Memory is also layer-normed before being used as keys/values
    torch::Tensor mem_normed;
    if (h_mem.numel() > 0) {
        mem_normed = ln1->forward(h_mem);
    }
    auto a = drop->forward(attn->forward(attn_in, mem_normed, r, attn_mask));
    auto h = h_cur + a;

    // Pre-norm FFN
    auto h2 = h + drop->forward(ffn->forward(ln2->forward(h)));
    return h2;
}

// ─────────────────────────────────────────────────────────────────────────────
// TransformerXLModel
// ─────────────────────────────────────────────────────────────────────────────
TransformerXLModelImpl::TransformerXLModelImpl(const TransformerXLConfig& c)
    : cfg(c) {

    embedding = register_module("embedding",
                    torch::nn::Embedding(cfg.vocab_size, cfg.d_model));
    layers    = register_module("layers", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        layers->push_back(TXLLayer(cfg));
    }
    norm_out = register_module("norm_out",
                   torch::nn::LayerNorm(
                       torch::nn::LayerNormOptions({cfg.d_model})));
    output   = register_module("output",
                   torch::nn::Linear(
                       torch::nn::LinearOptions(cfg.d_model, cfg.vocab_size)
                           .bias(false)));
    drop_emb = register_module("drop_emb",
                   torch::nn::Dropout(cfg.dropout));

    // Tie input embedding weights with output projection (§3 / Press & Wolf)
    if (cfg.tie_weights) {
        output->weight = embedding->weight;
    }

    // Embedding init
    torch::nn::init::normal_(embedding->weight, 0.0, 0.02);
    // Output init — only if not tied (xavier on the output projection)
    if (!cfg.tie_weights) {
        torch::nn::init::xavier_uniform_(output->weight);
    }
    // FFN layer init (attn projections already initialised in TXLRelAttnImpl ctor)
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        auto& layer = layers->at<TXLLayerImpl>(i);
        torch::nn::init::xavier_uniform_(layer.ffn->fc1->weight);
        torch::nn::init::zeros_(layer.ffn->fc1->bias);
        torch::nn::init::xavier_uniform_(layer.ffn->fc2->weight);
        torch::nn::init::zeros_(layer.ffn->fc2->bias);
    }

    // Initialise empty memory banks
    mems_.resize(cfg.n_layers);
}

void TransformerXLModelImpl::reset_memory() {
    for (auto& m : mems_) {
        m = torch::Tensor();  // empty
    }
}

void TransformerXLModelImpl::update_memory(int64_t n, torch::Tensor new_h) {
    // new_h: [B, q_len, d_model]  (detached — stop gradient, §3.2)
    torch::NoGradGuard ng;
    if (mems_[n].numel() == 0) {
        // First segment — just store (up to mem_len)
        int64_t keep = std::min(new_h.size(1), cfg.mem_len);
        mems_[n] = new_h.slice(1, new_h.size(1) - keep).detach();
    } else {
        // Concatenate old memory + new hidden, then keep last mem_len
        auto cat = torch::cat({mems_[n], new_h.detach()}, 1);
        int64_t total = cat.size(1);
        int64_t keep  = std::min(total, cfg.mem_len);
        mems_[n] = cat.slice(1, total - keep);
    }
}

torch::Tensor TransformerXLModelImpl::causal_mask(int64_t q_len, int64_t klen,
                                                    torch::Device dev) {
    // mask[i, j] = true (masked) if position j is in the "future" of query i.
    // Memory positions (j < m_len) are always visible.
    // For causal LM: query i can attend to key j if j <= m_len + i.
    // Equivalently, mask[i,j] = (j > m_len + i)  where m_len = klen - q_len.
    int64_t m_len = klen - q_len;
    // Build upper-triangular causal mask over the q_len × q_len current block
    // then prepend zeros (False) for the memory block.
    auto cur_mask = torch::ones({q_len, q_len}, torch::dtype(torch::kBool).device(dev))
                        .triu(1);  // upper triangle (strictly above diagonal)
    // Memory is always visible → all-false mask
    auto mem_mask = torch::zeros({q_len, m_len}, torch::dtype(torch::kBool).device(dev));
    return torch::cat({mem_mask, cur_mask}, 1);  // [q_len, klen]
}

torch::Tensor TransformerXLModelImpl::forward(torch::Tensor tokens) {
    // tokens: [B, T]  int64
    int64_t B = tokens.size(0);
    int64_t T = tokens.size(1);
    auto dev  = tokens.device();

    auto h = drop_emb->forward(embedding->forward(tokens));  // [B, T, d]

    // Memory length for this step
    int64_t m_len = (mems_[0].numel() > 0) ? mems_[0].size(1) : 0;
    int64_t klen  = T + m_len;

    // Sinusoidal relative positional encoding  [klen, d_model]
    auto r = make_rel_pos_encoding(klen, cfg.d_model, dev);

    // Causal attention mask
    auto mask = causal_mask(T, klen, dev);  // [T, klen]

    // Forward through N layers, collecting new hidden states for memory update
    std::vector<torch::Tensor> new_hids;
    new_hids.reserve(cfg.n_layers);

    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        new_hids.push_back(h);  // store input to each layer as new memory

        auto& layer_mod = layers->at<TXLLayerImpl>(i);
        torch::Tensor mem_i;
        if (mems_[i].numel() > 0) {
            // Expand memory to batch if stored without batch (1-batch edge case)
            mem_i = mems_[i];
            if (mem_i.size(0) == 1 && B > 1) {
                mem_i = mem_i.expand({B, -1, -1});
            }
        }
        h = layer_mod.forward(h, mem_i, r, mask);
    }

    // Update memories with the *inputs* to each layer (SG applied inside update)
    for (int64_t i = 0; i < cfg.n_layers; ++i) {
        update_memory(i, new_hids[i]);
    }

    // Output norm + projection
    h = norm_out->forward(h);          // [B, T, d]
    auto logits = output->forward(h);  // [B, T, vocab]
    return logits;
}

// ─────────────────────────────────────────────────────────────────────────────
// Training utilities
// ─────────────────────────────────────────────────────────────────────────────
torch::optim::Adam make_txl_optimizer(TransformerXLModel& model,
                                       const TransformerXLTrainConfig& tcfg) {
    // Paper uses Adam with lr=2.5e-4 for character-level; simple flat decay.
    return torch::optim::Adam(model->parameters(),
                              torch::optim::AdamOptions(tcfg.lr)
                                  .betas({0.9, 0.999})
                                  .eps(1e-8)
                                  .weight_decay(0.0));
}

double txl_lr_schedule(int64_t step, const TransformerXLTrainConfig& tcfg) {
    if (tcfg.warmup > 0 && step < tcfg.warmup) {
        // Linear warmup
        return tcfg.lr * (static_cast<double>(step + 1) / tcfg.warmup);
    }
    return tcfg.lr;
}

std::pair<torch::Tensor, double>
txl_train_step(TransformerXLModel& model,
               torch::optim::Adam& optimizer,
               torch::Tensor tokens,
               const TransformerXLTrainConfig& tcfg,
               int64_t step,
               bool new_doc) {
    // tokens: [B, T+1]
    // input  = tokens[:, 0..T-1],  target = tokens[:, 1..T]
    if (new_doc) model->reset_memory();

    // Update LR
    double lr_now = txl_lr_schedule(step, tcfg);
    for (auto& pg : optimizer.param_groups()) {
        static_cast<torch::optim::AdamOptions&>(pg.options()).lr(lr_now);
    }

    model->train();
    optimizer.zero_grad();

    int64_t T = tokens.size(1) - 1;
    auto inp = tokens.slice(1, 0, T);   // [B, T]
    auto tgt = tokens.slice(1, 1, T+1); // [B, T]

    auto logits = model->forward(inp);  // [B, T, vocab]

    int64_t V = logits.size(-1);
    auto loss = torch::nn::functional::cross_entropy(
        logits.reshape({-1, V}),
        tgt.reshape({-1}));

    loss.backward();

    // Gradient clipping (paper: clip=0.25)
    torch::nn::utils::clip_grad_norm_(model->parameters(), tcfg.clip);

    optimizer.step();

    double nll = loss.item<double>();
    return {loss.detach(), nll};
}

} // namespace nlp
} // namespace models
} // namespace dm
