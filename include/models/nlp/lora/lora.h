#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// LoRA: Low-Rank Adaptation of Large Language Models
// Edward Hu et al., arXiv:2106.09685v2, 2021
// https://arxiv.org/abs/2106.09685
//
// ── Core idea (§4.1, eq.3) ───────────────────────────────────────────────────
// For a frozen pre-trained weight W₀ ∈ ℝ^{d×k}, constrain its update ΔW to a
// low-rank decomposition:
//
//   h = W₀ x  +  ΔW x  =  W₀ x  +  (B A) x
//
// where  B ∈ ℝ^{d×r},  A ∈ ℝ^{r×k},  r ≪ min(d, k).
//
// ── Initialization (§4.1) ────────────────────────────────────────────────────
//   A ~ N(0, σ²)    (random Gaussian; paper uses σ=1/√r by default)
//   B = 0           (so ΔW = BA = 0 at training start)
//
// ── Scaling (§4.1) ───────────────────────────────────────────────────────────
//   The LoRA update is scaled:  h += (α / r) · B A x
//   α is a constant (default = r, so scale = 1). Setting α ≠ r lets you
//   re-use the same set of hyperparameters across rank choices without
//   retuning the learning rate.
//
// ── Where to apply (§4.2, §7.1) ──────────────────────────────────────────────
//   Applied to attention weight matrices Wq, Wk, Wv, Wo.  Paper focuses on
//   Wq and Wv; best results come from adapting all four attention projections
//   with a lower r each (Table 5).  MLP/FFN weights are frozen.
//
// ── Merge for inference (§4.1) ───────────────────────────────────────────────
//   W_merged = W₀ + (α/r) · B A
//   After merging, forward pass is identical to a standard Linear — zero
//   additional inference latency.  Merge/unmerge supported here.
//
// ── Number of trainable parameters ───────────────────────────────────────────
//   Per LoRA layer:  r·(d + k)   (vs d·k for full fine-tuning)
//   Example: d=k=1024, r=4 → 8192 vs 1,048,576 (128× reduction)
//
// ── Recommended rank r (Table 6) ─────────────────────────────────────────────
//   r=1 or r=2 already competitive when adapting both Wq and Wv;
//   r=4 a safe general default; r=8 for harder tasks or single-matrix adapt.
//
// ── Training (§4.1, §D) ──────────────────────────────────────────────────────
//   Optimizer : Adam (β=(0.9,0.999), no weight decay on LoRA params)
//   LR        : typically 1e-4 to 3e-4
//   Only A and B are updated; W₀ receives no gradients (requires_grad=false)
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────
struct LoRAConfig {
    int64_t rank    = 4;     // r: LoRA rank (paper recommends 1–8)
    double  alpha   = 4.0;   // α: scaling numerator (scale = α/r)
    double  dropout = 0.0;   // dropout on the LoRA path (optional regulariser)
    bool    bias    = false; // LoRA layers have no bias (§4.1)

    // scale = alpha / rank
    double scale() const {
        return (rank > 0) ? (alpha / static_cast<double>(rank)) : 1.0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// LoRALinear — wraps a frozen nn::Linear with a trainable low-rank branch.
//
// Forward (training or before merge):
//   h = W₀ x  +  scale · B A x          (eq.3)
//
// After merge():
//   h = (W₀ + scale·BA) x               (zero-latency inference)
//
// Usage:
//   LoRALinear layer(in_features, out_features, cfg);
//   // layer->weight is W₀ (frozen by default unless unfreeze_base=true)
//   // layer->lora_A, layer->lora_B are trainable
// ─────────────────────────────────────────────────────────────────────────────
struct LoRALinearImpl : torch::nn::Module {
    LoRALinearImpl(int64_t in_features,
                   int64_t out_features,
                   const LoRAConfig& cfg,
                   bool    unfreeze_base = false);

    // Forward pass: W₀x + scale·BAx  (or merged Wx when merged_)
    torch::Tensor forward(torch::Tensor x);

    // Merge BA into W₀ in-place: W₀ ← W₀ + scale·BA
    // After this, forward() runs as a plain Linear (no lora overhead).
    void merge();

    // Unmerge: W₀ ← W₀ − scale·BA  (restore original weights)
    void unmerge();

    bool is_merged() const { return merged_; }

    // Reset LoRA weights to paper initialization (A~N, B=0)
    void reset_lora();

    // ─── Public members ────────────────────────────────────────────────
    // Base linear (W₀, bias optional)
    torch::nn::Linear   base{nullptr};

    // LoRA parameters
    torch::Tensor lora_A;  // [r, k=in_features]
    torch::Tensor lora_B;  // [d=out_features, r]

    LoRAConfig cfg;

    int64_t in_features;
    int64_t out_features;

private:
    bool merged_ = false;
    torch::nn::Dropout dropout_{nullptr};
};
TORCH_MODULE(LoRALinear);

// ─────────────────────────────────────────────────────────────────────────────
// LoRAEmbedding — LoRA adaptation for an Embedding layer.
// ΔW has same shape as embedding table [vocab, d]:
//   h_embed = E₀ x  +  scale · A_e B_e x
// where A_e ∈ ℝ^{vocab×r}, B_e ∈ ℝ^{r×d}.
// Init: A_e = 0, B_e ~ N(0,1)   (reversed from Linear so ΔEmbed=0 at start)
// ─────────────────────────────────────────────────────────────────────────────
struct LoRAEmbeddingImpl : torch::nn::Module {
    LoRAEmbeddingImpl(int64_t num_embeddings,
                      int64_t embedding_dim,
                      const LoRAConfig& cfg);

    torch::Tensor forward(torch::Tensor indices);

    void merge();
    void unmerge();
    bool is_merged() const { return merged_; }
    void reset_lora();

    torch::nn::Embedding  base{nullptr};
    torch::Tensor lora_A;  // [num_embeddings, r]  — init to 0
    torch::Tensor lora_B;  // [r, embedding_dim]   — init to N(0,1)

    LoRAConfig cfg;
    int64_t num_embeddings, embedding_dim;

private:
    bool merged_ = false;
};
TORCH_MODULE(LoRAEmbedding);

// ─────────────────────────────────────────────────────────────────────────────
// LoRAModel — wraps any torch::nn::Module and injects LoRA into named
// nn::Linear sub-modules whose names match a target list.
//
// Usage:
//   auto model = SomeLargeModel(cfg);
//   model->eval();
//   // Freeze base model
//   for (auto& p : model->parameters()) p.set_requires_grad(false);
//   // Inject LoRA into Wq, Wv
//   LoRAModel lora_model(model, {"attn.W_q", "attn.W_v"}, lora_cfg);
//   // Only lora_model->lora_parameters() requires grad
//   auto opt = make_lora_optimizer(lora_model, train_cfg);
// ─────────────────────────────────────────────────────────────────────────────
struct LoRAModelImpl : torch::nn::Module {
    // target_names: list of sub-module name substrings to wrap with LoRA.
    //   Any nn::Linear whose full path contains one of these strings is replaced.
    LoRAModelImpl(std::shared_ptr<torch::nn::Module> base_model,
                  const std::vector<std::string>&    target_names,
                  const LoRAConfig&                  cfg);

    // NOTE: Do not call forward() directly — call base_model's own forward().
    // LoRAModel is a parameter-management wrapper; forward() throws logic_error.
    // Use: base_model->forward(x) after injection.
    torch::Tensor forward(torch::Tensor x);

    // Return only LoRA trainable parameters (A and B matrices)
    std::vector<torch::Tensor> lora_parameters() const;

    // Number of trainable LoRA parameters
    int64_t lora_param_count() const;

    // Total parameters in base model (frozen + LoRA)
    int64_t total_param_count() const;

    // Merge all LoRA layers
    void merge_all();

    // Unmerge all LoRA layers
    void unmerge_all();

    std::shared_ptr<torch::nn::Module> base_model;
    std::vector<LoRALinear>            lora_layers;
    LoRAConfig                         cfg;
};
TORCH_MODULE(LoRAModel);

// ─────────────────────────────────────────────────────────────────────────────
// Training utilities
// ─────────────────────────────────────────────────────────────────────────────
struct LoRATrainConfig {
    double  lr          = 1e-4;    // typical LoRA LR (§D)
    double  beta1       = 0.9;
    double  beta2       = 0.999;
    double  eps         = 1e-8;
    double  weight_decay= 0.0;     // no WD on LoRA params (§4.1)
    int64_t warmup      = 0;
    int64_t total_steps = 10000;
};

// Optimizer that only updates LoRA A/B parameters
torch::optim::Adam make_lora_optimizer(
    const std::vector<torch::Tensor>& lora_params,
    const LoRATrainConfig& tcfg);

// LR schedule: linear warmup then cosine decay
double lora_lr_schedule(int64_t step, const LoRATrainConfig& tcfg);

// Single training step.  tokens: [B, T+1]  (input = tokens[:,:-1], target = tokens[:,1:])
// model must be a callable that takes [B,T] → [B,T,vocab_size] logits.
// Returns (loss_tensor, loss_scalar).
std::pair<torch::Tensor, double>
lora_train_step(torch::nn::AnyModule&     model,
                torch::optim::Adam&       optimizer,
                torch::Tensor             tokens,
                const LoRATrainConfig&    tcfg,
                int64_t                   step);

// ─────────────────────────────────────────────────────────────────────────────
// Utility: inject LoRA into any Module's named Linear children
//
// Replaces each nn::Linear sub-module whose full dotted path contains any
// string in target_patterns with a LoRALinear.  The original weight is
// copied into the LoRALinear's base and frozen.
//
// Returns list of (path, LoRALinear) pairs for bookkeeping.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::pair<std::string, LoRALinear>>
inject_lora(torch::nn::Module&             module,
            const std::vector<std::string>& target_patterns,
            const LoRAConfig&               cfg);

} // namespace nlp
} // namespace models
} // namespace dm
