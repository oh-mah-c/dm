#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Tiny Recursion Model (TRM)
// Alexia Jolicoeur-Martineau, Samsung SAIL Montréal, 2025
// arXiv:2510.04871v1
//
// TRM recursively refines a latent reasoning vector z and answer y using a
// single tiny network.  It outperforms 27M-param HRM (and most LLMs) on
// hard puzzle tasks while using only 7M parameters with 2 Transformer layers.
//
// ── Architecture (Section 4, Figure 1) ──────────────────────────────────────
// Single shared network `net` with n_layers=2 Transformer blocks:
//   Each block: RMSNorm → Self-Attention → Add,  RMSNorm → SwiGLU MLP → Add
//   Rotary positional embeddings (RoPE), no bias (Section 2.1)
//   hidden_size D=512 (Section 6 hyperparameters)
//
// Two modes of the network depending on what is updated:
//   Latent update (n recursions, no grad):  z  ← net(x, y, z)
//   Answer update (1 recursion, with grad): y  ← net(y, z)        [refine]
// where x = embedded question, y = current answer embedding, z = latent.
// Inputs are concatenated along the sequence dimension before being fed.
//
// ── Deep Supervision (Section 2.4 / Figure 3) ────────────────────────────────
// N_sup=16 supervision steps per mini-batch element:
//   1. Run T-1 gradient-free recursions to improve (y, z)
//   2. Run 1 full recursion (with grad) to get (y_hat, q_hat)
//   3. loss = softmax_cross_entropy(y_hat, y_true)
//            + binary_cross_entropy(q_hat, y_hat==y_true)   [halt signal]
//   4. z = z.detach()
//   5. Early-stop if q_hat[0] > 0  (ACT: model thinks it has the answer)
//
// ── EMA (Section 4.7) ────────────────────────────────────────────────────────
// Exponential Moving Average of weights (rate=0.999) to improve stability.
//
// ── Variants (Table 1) ───────────────────────────────────────────────────────
//   TRM-Att (attention):  use self-attention in the blocks    (~7M params)
//   TRM-MLP (mlp-mixer):  replace self-attention with MLP    (~5M/19M params)
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <string>
#include <vector>

namespace dm {
namespace models {
namespace reasoning {

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm  (Section 2.1 — "RMSNorm" cited from Zhang & Sennrich 2019)
// ─────────────────────────────────────────────────────────────────────────────
struct RMSNormImpl : torch::nn::Module {
    torch::Tensor weight;
    float eps;
    RMSNormImpl(int64_t dim, float eps = 1e-6f);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(RMSNorm);

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU MLP  (Section 2.1 — "SwiGLU activation function")
// out = (W1·x ⊙ swish(W2·x)) · W3
// hidden_mult=4 gives the standard FFN expansion
// ─────────────────────────────────────────────────────────────────────────────
struct SwiGLUMlpImpl : torch::nn::Module {
    torch::nn::Linear gate{nullptr};   // W1
    torch::nn::Linear up{nullptr};     // W2
    torch::nn::Linear down{nullptr};   // W3

    SwiGLUMlpImpl(int64_t dim, int64_t hidden_dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(SwiGLUMlp);

// ─────────────────────────────────────────────────────────────────────────────
// Rotary Position Embedding helper (Section 2.1)
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor apply_rotary_emb(torch::Tensor x, torch::Tensor cos, torch::Tensor sin);
// Build (cos, sin) cache for a given seq_len and head_dim
std::pair<torch::Tensor,torch::Tensor> build_rope_cache(int64_t seq_len,
                                                         int64_t head_dim,
                                                         torch::Device device);

// ─────────────────────────────────────────────────────────────────────────────
// Self-Attention block  (used in TRM-Att variant)
// Multi-head attention with RoPE, no bias
// ─────────────────────────────────────────────────────────────────────────────
struct TRMAttentionImpl : torch::nn::Module {
    int64_t num_heads;
    int64_t head_dim;

    torch::nn::Linear qkv{nullptr};   // projects to 3 * dim, no bias
    torch::nn::Linear out{nullptr};   // projects back to dim, no bias

    TRMAttentionImpl(int64_t dim, int64_t num_heads);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(TRMAttention);

// ─────────────────────────────────────────────────────────────────────────────
// MLP-Mixer attention replacement  (TRM-MLP variant, Section 4.5)
// Simple linear across the sequence dimension (cheap for small context)
// ─────────────────────────────────────────────────────────────────────────────
struct TRMMixerImpl : torch::nn::Module {
    torch::nn::Linear mix{nullptr};  // [L, L] token mixing
    TRMMixerImpl(int64_t seq_len);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(TRMMixer);

// ─────────────────────────────────────────────────────────────────────────────
// Single Transformer block
// ─────────────────────────────────────────────────────────────────────────────
struct TRMBlockImpl : torch::nn::Module {
    bool use_attention;

    RMSNorm  norm1{nullptr};
    RMSNorm  norm2{nullptr};
    SwiGLUMlp mlp{nullptr};

    // attention variant
    TRMAttention attn{nullptr};
    // mixer variant
    TRMMixer     mixer{nullptr};

    TRMBlockImpl(int64_t dim, int64_t num_heads, int64_t mlp_hidden,
                 bool use_attention, int64_t seq_len = 0);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(TRMBlock);

// ─────────────────────────────────────────────────────────────────────────────
// Core TRM network  (the single shared "net" in the paper's pseudocode)
//
// Input:  concatenation of two token sequences along dim=1
//         e.g. [x ; z]  or  [y ; z]  — each [B, L, D]
// Output: updated embedding, same shape as input
//
// The network has n_layers=2 TRMBlock layers + input/output projections.
// Input embedding and output "reverse embedding" (output head) are separate
// linear layers (Section 2.1: "output head f_O").
// ─────────────────────────────────────────────────────────────────────────────
struct TRMNetImpl : torch::nn::Module {
    int64_t vocab_size;
    int64_t embed_dim;
    int64_t seq_len;     // L (single sequence length, concatenated input is 2L or 3L)

    torch::nn::Embedding input_embed{nullptr};     // token → D
    torch::nn::ModuleList blocks{nullptr};
    RMSNorm final_norm{nullptr};
    torch::nn::Linear output_head{nullptr};        // D → vocab_size (reverse embedding)
    torch::nn::Linear halt_head{nullptr};          // D → 1 (ACT halt signal)

    TRMNetImpl(int64_t vocab_size, int64_t seq_len, int64_t embed_dim = 512,
               int64_t n_layers = 2, int64_t num_heads = 8,
               bool use_attention = true);

    // Embed raw token indices → float tensor [B, L, D]
    torch::Tensor embed(torch::Tensor tokens);

    // Forward through all blocks; returns [B, L_in, D]
    torch::Tensor forward(torch::Tensor x);

    // output_head(x[:, :seq_len, :]) → logits [B, L, vocab_size]
    torch::Tensor to_logits(torch::Tensor x);

    // halt_head(x[:, 0, :]) → scalar logit [B, 1]
    torch::Tensor to_halt(torch::Tensor x);
};
TORCH_MODULE(TRMNet);

// ─────────────────────────────────────────────────────────────────────────────
// Full TRM model  (wraps TRMNet + EMA + recursion logic)
// ─────────────────────────────────────────────────────────────────────────────
struct TRMImpl : torch::nn::Module {
    int64_t vocab_size;
    int64_t seq_len;     // L
    int64_t embed_dim;
    int64_t n_recursions; // n=6 (paper Section 4.8: T=3, n=6)
    int64_t T;            // T=3 (supervision depth multiplier)

    TRMNet net{nullptr};

    // EMA shadow copy (Section 4.7, rate=0.999)
    // Stored as a separate non-grad parameter dict (updated manually)
    std::vector<torch::Tensor> ema_params;
    double ema_rate;

    TRMImpl(int64_t vocab_size, int64_t seq_len,
            int64_t embed_dim     = 512,
            int64_t n_layers      = 2,
            int64_t num_heads     = 8,
            bool    use_attention = true,
            int64_t n_recursions  = 6,   // n in pseudocode
            int64_t T             = 3,   // T in pseudocode
            double  ema_rate      = 0.999);

    // Initialise EMA shadow from current weights
    void init_ema();
    // Update EMA after each optimiser step
    void update_ema();
    // Copy EMA weights into net for inference
    void apply_ema();
    // Restore live weights (after inference with EMA)
    void restore_live();

    // ── Core recursion (Figure 3 pseudocode) ─────────────────────────────────
    //
    // latent_recursion(x_emb, y_emb, z, n):
    //   for i in range(n+1):  z = net([x_emb; y_emb; z])
    //   returns z
    //
    // deep_recursion(x_emb, y_emb, z, n, T):
    //   with no_grad:  for j in range(T-1): y_emb,z = latent_recursion(...)
    //   y_emb, z = latent_recursion(...)   [with grad for last call]
    //   return y_emb.detach(), z.detach(), output_head(y_emb), halt_head(y_emb)

    // Single latent recursion step: returns updated (y_emb, z)
    std::pair<torch::Tensor,torch::Tensor>
    latent_recursion(const torch::Tensor& x_emb,
                     torch::Tensor y_emb,
                     torch::Tensor z,
                     int64_t n_steps);

    // Full deep recursion with T-1 no-grad warm-ups
    // Returns: (y_emb_detach, z_detach, y_logits, q_logit)
    std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
    deep_recursion(const torch::Tensor& x_emb,
                   torch::Tensor y_emb,
                   torch::Tensor z);

    // Convenience: embed tokens and run deep_recursion
    std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
    forward(torch::Tensor x_tokens,
            torch::Tensor y_tokens,
            torch::Tensor z);

    // Inference: run N_sup supervision steps, return best y token prediction
    torch::Tensor predict(torch::Tensor x_tokens, int64_t n_sup = 16);

    // Zero-init latent z [B, L, D] on given device
    torch::Tensor init_z(int64_t batch, torch::Device device) const;

private:
    // live parameter backup for EMA swap
    std::vector<torch::Tensor> _live_backup;
};
TORCH_MODULE(TRM);

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration  (Section 6 hyperparameters)
// ─────────────────────────────────────────────────────────────────────────────
struct TRMTrainConfig {
    double   lr            = 1e-4;      // Section 6: "learning rate 1e-4"
    double   embed_lr      = 1e-2;      // Section 6: "1e-2 for embeddings"
    double   weight_decay  = 0.1;       // Section 6
    double   beta1         = 0.9;       // Section 6
    double   beta2         = 0.95;      // Section 6
    int64_t  warmup_iters  = 2000;      // Section 6: "2K iterations warm-up"
    int64_t  max_epochs    = 60000;     // Section 6: Sudoku-Extreme/Maze

    int64_t  batch_size    = 768;       // Section 6
    int64_t  n_sup         = 16;        // N_sup: max supervision steps
    double   ema_rate      = 0.999;     // Section 4.7

    torch::Device device   = torch::kCPU;
};

// ─────────────────────────────────────────────────────────────────────────────
// Stable-max cross-entropy loss  (Section 6 — "stable-max loss")
// Clips logits before softmax to avoid gradient explosion on small datasets
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor stable_cross_entropy(torch::Tensor logits, torch::Tensor targets);

// ─────────────────────────────────────────────────────────────────────────────
// Training helpers
// ─────────────────────────────────────────────────────────────────────────────

// One epoch of deep-supervision training.
// batches: vector of {x_tokens [B,L], y_true_tokens [B,L]}
// Returns mean loss over the epoch.
float trm_train_epoch(
    TRM&                     model,
    torch::optim::AdamW&     optimizer,
    const TRMTrainConfig&    cfg,
    int64_t                  global_step,
    const std::vector<std::pair<torch::Tensor,torch::Tensor>>& batches);

// Evaluate token-level accuracy (fraction of fully correct sequences).
float trm_evaluate(
    TRM&          model,
    torch::Device device,
    const std::vector<std::pair<torch::Tensor,torch::Tensor>>& batches,
    int64_t       n_sup = 16);

// Full training loop.
void trm_train(
    TRM&                     model,
    const TRMTrainConfig&    cfg,
    const std::vector<std::pair<torch::Tensor,torch::Tensor>>& train_batches,
    const std::vector<std::pair<torch::Tensor,torch::Tensor>>& val_batches,
    const std::string&       save_path = "trm_best.pt");

} // namespace reasoning
} // namespace models
} // namespace dm
