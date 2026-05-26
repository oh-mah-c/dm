// ─────────────────────────────────────────────────────────────────────────────
// lora.cpp — LoRA implementation
//
// E. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and
// W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models,"
// arXiv:2106.09685v2, 2021. https://arxiv.org/abs/2106.09685
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/lora/lora.h"

#include <torch/torch.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// LoRALinear
// ─────────────────────────────────────────────────────────────────────────────
LoRALinearImpl::LoRALinearImpl(int64_t in_f,
                               int64_t out_f,
                               const LoRAConfig& c,
                               bool unfreeze_base)
    : cfg(c), in_features(in_f), out_features(out_f)
{
    // Base linear — bias according to cfg
    base = register_module("base",
               torch::nn::Linear(
                   torch::nn::LinearOptions(in_f, out_f).bias(cfg.bias)));

    // Freeze base weights (W₀) by default
    if (!unfreeze_base) {
        base->weight.set_requires_grad(false);
        if (base->bias.defined())
            base->bias.set_requires_grad(false);
    }

    // LoRA matrices — registered as parameters so optimizer can update them
    // A ∈ ℝ^{r × k},  B ∈ ℝ^{d × r}
    lora_A = register_parameter("lora_A",
                 torch::empty({cfg.rank, in_f}));
    lora_B = register_parameter("lora_B",
                 torch::empty({out_f, cfg.rank}));

    // Dropout on LoRA path
    if (cfg.dropout > 0.0) {
        dropout_ = register_module("dropout",
                       torch::nn::Dropout(cfg.dropout));
    }

    reset_lora();
}

void LoRALinearImpl::reset_lora() {
    // A ~ N(0, 1/√r)   (Kaiming-style; paper uses N(0,σ²) with σ=1 and
    // notes it "doesn't significantly affect performance" — we use 1/√r
    // which matches the standard fan_in init)
    torch::nn::init::kaiming_uniform_(lora_A, std::sqrt(5.0));
    // B = 0  (so ΔW = BA = 0 at training start — §4.1)
    torch::nn::init::zeros_(lora_B);
    merged_ = false;
}

torch::Tensor LoRALinearImpl::forward(torch::Tensor x) {
    // Base path
    auto h = base->forward(x);   // [..., d]

    if (!merged_ && cfg.rank > 0) {
        // LoRA path:  scale · B (A (dropout(x)))
        auto lx = (dropout_.is_empty() || !is_training())
                  ? x
                  : dropout_->forward(x);
        // lx: [..., k]
        // A: [r, k] → lx @ A.T: [..., r]
        // B: [d, r] → (...) @ B.T: [..., d]
        auto lora_out = torch::nn::functional::linear(
                            torch::nn::functional::linear(lx, lora_A),
                            lora_B);  // [..., d]
        h = h + static_cast<float>(cfg.scale()) * lora_out;
    }
    return h;
}

void LoRALinearImpl::merge() {
    if (merged_) return;
    // W₀ ← W₀ + scale · B A
    // lora_A: [r, k], lora_B: [d, r]  →  lora_B @ lora_A: [d, k]
    torch::NoGradGuard ng;
    base->weight.data() += static_cast<float>(cfg.scale()) * lora_B.mm(lora_A);
    merged_ = true;
}

void LoRALinearImpl::unmerge() {
    if (!merged_) return;
    torch::NoGradGuard ng;
    base->weight.data() -= static_cast<float>(cfg.scale()) * lora_B.mm(lora_A);
    merged_ = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// LoRAEmbedding
// ─────────────────────────────────────────────────────────────────────────────
LoRAEmbeddingImpl::LoRAEmbeddingImpl(int64_t num_emb,
                                     int64_t emb_dim,
                                     const LoRAConfig& c)
    : cfg(c), num_embeddings(num_emb), embedding_dim(emb_dim)
{
    base = register_module("base",
               torch::nn::Embedding(num_emb, emb_dim));
    base->weight.set_requires_grad(false);

    // A_e: [vocab, r] — init 0;  B_e: [r, d] — init N(0,1)
    lora_A = register_parameter("lora_A",
                 torch::zeros({num_emb, cfg.rank}));
    lora_B = register_parameter("lora_B",
                 torch::empty({cfg.rank, emb_dim}));
    torch::nn::init::normal_(lora_B, 0.0, 1.0);
}

void LoRAEmbeddingImpl::reset_lora() {
    torch::nn::init::zeros_(lora_A);
    torch::nn::init::normal_(lora_B, 0.0, 1.0);
    merged_ = false;
}

torch::Tensor LoRAEmbeddingImpl::forward(torch::Tensor indices) {
    auto h = base->forward(indices);   // [..., d]
    if (!merged_ && cfg.rank > 0) {
        // Embedding lookup of lora_A rows, then project through lora_B
        auto lora_emb = torch::nn::functional::embedding(
                            indices, lora_A);           // [..., r]
        auto lora_out = lora_emb.matmul(lora_B);        // [..., d]
        h = h + static_cast<float>(cfg.scale()) * lora_out;
    }
    return h;
}

void LoRAEmbeddingImpl::merge() {
    if (merged_) return;
    torch::NoGradGuard ng;
    // lora_A: [vocab, r], lora_B: [r, d] → lora_A @ lora_B: [vocab, d]
    base->weight.data() += static_cast<float>(cfg.scale()) * lora_A.mm(lora_B);
    merged_ = true;
}

void LoRAEmbeddingImpl::unmerge() {
    if (!merged_) return;
    torch::NoGradGuard ng;
    base->weight.data() -= static_cast<float>(cfg.scale()) * lora_A.mm(lora_B);
    merged_ = false;
}

// ─────────────────────────────────────────────────────────────────────────────
// LoRAModel
// ─────────────────────────────────────────────────────────────────────────────
LoRAModelImpl::LoRAModelImpl(std::shared_ptr<torch::nn::Module> base,
                              const std::vector<std::string>&    target_names,
                              const LoRAConfig&                  c)
    : base_model(base), cfg(c)
{
    register_module("base_model", base_model);

    // Inject LoRA into matching Linear sub-modules
    auto injected = inject_lora(*base_model, target_names, cfg);
    for (auto& [path, ll] : injected) {
        lora_layers.push_back(ll);
        // The LoRALinear modules are already owned inside base_model's
        // module tree via inject_lora; register them here too so their
        // parameters appear in named_parameters().
        register_module("lora_" + path, ll);
    }
}

torch::Tensor LoRAModelImpl::forward(torch::Tensor /*x*/) {
    // LoRAModel::forward is a no-op placeholder — call base_model's forward
    // directly by holding a reference of the concrete type.  In practice,
    // users inject LoRA into their model via inject_lora() and call that
    // model's own forward; LoRAModel is used as a container for managing
    // LoRA parameter lists and merge/unmerge.
    throw std::logic_error(
        "LoRAModel::forward: call the base model's forward() directly. "
        "LoRAModel is a parameter-management wrapper, not a forward wrapper.");
}

std::vector<torch::Tensor> LoRAModelImpl::lora_parameters() const {
    std::vector<torch::Tensor> params;
    for (const auto& ll : lora_layers) {
        params.push_back(ll->lora_A);
        params.push_back(ll->lora_B);
    }
    return params;
}

int64_t LoRAModelImpl::lora_param_count() const {
    int64_t n = 0;
    for (const auto& ll : lora_layers)
        n += ll->lora_A.numel() + ll->lora_B.numel();
    return n;
}

int64_t LoRAModelImpl::total_param_count() const {
    int64_t n = 0;
    for (auto& item : base_model->parameters())
        n += item.numel();
    return n;
}

void LoRAModelImpl::merge_all() {
    for (auto& ll : lora_layers) ll->merge();
}

void LoRAModelImpl::unmerge_all() {
    for (auto& ll : lora_layers) ll->unmerge();
}

// ─────────────────────────────────────────────────────────────────────────────
// inject_lora
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::pair<std::string, LoRALinear>>
inject_lora(torch::nn::Module&             module,
            const std::vector<std::string>& target_patterns,
            const LoRAConfig&               cfg)
{
    std::vector<std::pair<std::string, LoRALinear>> result;

    // Collect all named children (immediate only — we recurse manually)
    // We do a BFS/DFS via named_modules which returns all descendants.
    // However, we need parent references to replace children.  Instead we
    // walk named_children of each sub-module recursively.

    std::function<void(torch::nn::Module&, const std::string&)> walk;
    walk = [&](torch::nn::Module& mod, const std::string& prefix) {
        // Iterate immediate children
        for (auto& child_pair : mod.named_children()) {
            const std::string& child_name = child_pair.key();
            auto& child_mod = child_pair.value();
            std::string full_path = prefix.empty()
                                    ? child_name
                                    : prefix + "." + child_name;

            // Check if this child is an nn::Linear AND its path matches
            bool is_linear = (dynamic_cast<torch::nn::LinearImpl*>(
                                  child_mod.get()) != nullptr);
            bool matches   = false;
            if (is_linear) {
                for (const auto& pat : target_patterns) {
                    if (full_path.find(pat) != std::string::npos) {
                        matches = true;
                        break;
                    }
                }
            }

            if (matches) {
                // Cast to Linear and copy weights
                auto* lin = dynamic_cast<torch::nn::LinearImpl*>(
                                child_mod.get());
                int64_t out_f = lin->weight.size(0);
                int64_t in_f  = lin->weight.size(1);
                bool    has_b = lin->bias.defined();

                // Build LoRALinear with the same dimensions
                LoRAConfig child_cfg = cfg;
                // Cap rank at min(in,out) for safety
                child_cfg.rank = std::min(cfg.rank, std::min(in_f, out_f));

                auto ll = LoRALinear(in_f, out_f, child_cfg, false);
                // Copy pre-trained weights into base
                {
                    torch::NoGradGuard ng;
                    ll->base->weight.data().copy_(lin->weight.data());
                    if (has_b && ll->base->bias.defined())
                        ll->base->bias.data().copy_(lin->bias.data());
                }
                ll->base->weight.set_requires_grad(false);
                if (ll->base->bias.defined())
                    ll->base->bias.set_requires_grad(false);

                // Replace child in parent module (replace_module handles
                // the case where the name is already registered)
                mod.replace_module(child_name, ll);

                result.emplace_back(full_path, ll);
            } else {
                // Recurse
                walk(*child_mod, full_path);
            }
        }
    };

    walk(module, "");
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Training utilities
// ─────────────────────────────────────────────────────────────────────────────
torch::optim::Adam make_lora_optimizer(
    const std::vector<torch::Tensor>& lora_params,
    const LoRATrainConfig& tcfg)
{
    return torch::optim::Adam(
        lora_params,
        torch::optim::AdamOptions(tcfg.lr)
            .betas({tcfg.beta1, tcfg.beta2})
            .eps(tcfg.eps)
            .weight_decay(tcfg.weight_decay));
}

double lora_lr_schedule(int64_t step, const LoRATrainConfig& tcfg) {
    // Linear warmup
    if (tcfg.warmup > 0 && step < tcfg.warmup) {
        return tcfg.lr * (static_cast<double>(step + 1) / tcfg.warmup);
    }
    // Cosine decay from lr → 0 (or min_lr = 0 as no separate param here)
    int64_t s = step - tcfg.warmup;
    int64_t T = std::max(tcfg.total_steps - tcfg.warmup, (int64_t)1);
    double ratio = static_cast<double>(s) / T;
    if (ratio >= 1.0) ratio = 1.0;
    return 0.5 * tcfg.lr * (1.0 + std::cos(M_PI * ratio));
}

std::pair<torch::Tensor, double>
lora_train_step(torch::nn::AnyModule&   model,
                torch::optim::Adam&     optimizer,
                torch::Tensor           tokens,
                const LoRATrainConfig&  tcfg,
                int64_t                 step)
{
    // Update LR
    double lr_now = lora_lr_schedule(step, tcfg);
    for (auto& pg : optimizer.param_groups())
        static_cast<torch::optim::AdamOptions&>(pg.options()).lr(lr_now);

    optimizer.zero_grad();

    int64_t T  = tokens.size(1) - 1;
    auto inp   = tokens.slice(1, 0, T);
    auto tgt   = tokens.slice(1, 1, T + 1);

    auto logits = model.forward<torch::Tensor>(inp);  // [B, T, vocab]
    int64_t V   = logits.size(-1);
    auto loss   = torch::nn::functional::cross_entropy(
                      logits.reshape({-1, V}),
                      tgt.reshape({-1}));

    loss.backward();
    optimizer.step();

    return {loss.detach(), loss.item().toDouble()};
}

} // namespace nlp
} // namespace models
} // namespace dm
