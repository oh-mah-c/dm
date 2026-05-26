#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Mamba: Linear-Time Sequence Modeling with Selective State Spaces
// Albert Gu and Tri Dao
// arXiv:2312.00752v2, 2023  —  https://arxiv.org/abs/2312.00752
//
// ── Core idea ────────────────────────────────────────────────────────────────
// Structured State Space Models (S4) define a sequence-to-sequence map via:
//   Continuous:  h'(t) = A h(t) + B x(t),  y(t) = C h(t)          (§2, eq.1)
//   Discrete:    h_t   = Ā h_{t-1} + B̄ x_t, y_t  = C h_t          (§2, eq.2)
//   where Ā = exp(ΔA), B̄ = (ΔA)⁻¹(exp(ΔA)-I)·ΔB   (ZOH, §2, eq.4)
//
// S4 is Linear Time Invariant (LTI): A, B, C, Δ are fixed across time.
// This enables convolution-mode training but limits content-based reasoning.
//
// Mamba's key contribution — Selective SSM (S6, §3.2):
//   B, C are functions of input x  (B = Linear_N(x), C = Linear_N(x))
//   Δ is input-dependent: Δ = softplus(Parameter + Broadcast_D(Linear_1(x)))
//   A remains a fixed parameter (diagonal N×N matrix, §3.6)
// This makes the model time-varying → no convolution, only recurrence (scan).
//
// ── Mamba block architecture (§3.4, Fig. 3) ──────────────────────────────────
// Each Mamba block expands dimension by factor E=2:
//   x_proj = Linear(d_model → E*d_model)   [in_proj, no bias]
//   z_proj = Linear(d_model → E*d_model)   [in_proj gate branch, no bias]
//   conv1d  : depthwise conv on x_proj, kernel d_conv=4
//   x_ssm  = SiLU(conv1d(x_proj))
//   SSM scan on x_ssm  →  y_ssm
//   output = y_ssm ⊙ SiLU(z_proj)
//   out_proj = Linear(E*d_model → d_model) [no bias]
//
// RMSNorm (pre-norm) + residual wraps each block.
//
// ── SSM dimensions (§2, §3.6) ────────────────────────────────────────────────
//   d_model  D  : model dimension
//   d_state  N  : SSM state dimension (default 16)
//   d_conv   4  : depthwise conv kernel width
//   expand   E  : channel expansion factor (default 2)
//   d_inner  = E * d_model  : inner SSM dimension
//
// ── Model sizes (Table 12 / §E.2) ────────────────────────────────────────────
//   130M : 24 layers, d_model=768   (matches GPT3-125M spec)
//   370M : 48 layers, d_model=1024
//   790M : 48 layers, d_model=1536
//   1.4B : 48 layers, d_model=2048
//
// ── Training (§E.2) ──────────────────────────────────────────────────────────
//   Optimizer : AdamW, β=(0.9, 0.95), clip=1.0, weight_decay=0.1
//   Scheduler : linear warmup + cosine decay (min_lr=1e-5)
//   No dropout; no bias in Linear projections
//   A init    : -1/2 + n·i and -(n+1) (S4D-Lin / S4D-Real); Mamba uses real:
//               A[i] = -(i+1)  for i in [0, N)              (§3.6)
//   Δ init    : inverse softplus of Uniform([0.001, 0.1])   (§3.6)
//
// ── Inference ────────────────────────────────────────────────────────────────
//   Constant-time per token: only the hidden state h_t ∈ R^{d_inner × N}
//   is maintained. No KV-cache growth.
//   step() method implements one-token recurrent inference step.
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <vector>
#include <string>
#include <cstdint>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────
struct MambaConfig {
    int64_t vocab_size  = 50277;  // default GPT-NeoX tokenizer vocab
    int64_t d_model     = 768;    // model (token embedding) dimension D
    int64_t n_layers    = 24;     // number of Mamba blocks
    int64_t d_state     = 16;     // SSM state dimension N
    int64_t d_conv      = 4;      // depthwise conv kernel width
    int64_t expand      = 2;      // channel expansion factor E (d_inner = E*D)
    float   dt_rank_scale = 1.0f; // dt_rank = ceil(D / 16) * dt_rank_scale
    bool    tie_weights = false;  // tie embedding / output projection

    // Computed field helpers
    int64_t d_inner()  const { return expand * d_model; }
    int64_t dt_rank()  const {
        return static_cast<int64_t>(
            std::ceil(static_cast<double>(d_model) / 16.0) * dt_rank_scale);
    }

    // ── Named presets (Table 12 / §E.2) ──────────────────────────────────
    // Tiny — fast unit-test config
    static MambaConfig tiny() {
        return {1000, 64, 2, 16, 4, 2, 1.0f, false};
    }
    // 130M — 24L, d_model=768
    static MambaConfig m130() {
        return {50277, 768, 24, 16, 4, 2, 1.0f, false};
    }
    // 370M — 48L, d_model=1024
    static MambaConfig m370() {
        return {50277, 1024, 48, 16, 4, 2, 1.0f, false};
    }
    // 790M — 48L, d_model=1536
    static MambaConfig m790() {
        return {50277, 1536, 48, 16, 4, 2, 1.0f, false};
    }
    // 1.4B — 48L, d_model=2048
    static MambaConfig m1p4b() {
        return {50277, 2048, 48, 16, 4, 2, 1.0f, false};
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm — identical to Llama 2 variant; field named `w` to avoid c10 clash
// ─────────────────────────────────────────────────────────────────────────────
struct MambaRMSNormImpl : torch::nn::Module {
    explicit MambaRMSNormImpl(int64_t dim, float eps = 1e-5f);
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor w;
    float eps;
};
TORCH_MODULE(MambaRMSNorm);

// ─────────────────────────────────────────────────────────────────────────────
// Selective SSM core (Algorithm 2, §3.2)
//
// Given:   u ∈ R^{B × L × D_inner}   (input after conv + activation)
// Compute:
//   B_t  = Linear_N(u_t)              [B, L, N]   input-dependent
//   C_t  = Linear_N(u_t)              [B, L, N]
//   Δ_t  = softplus(dt_bias + dt_proj(u_t))  [B, L, D_inner]
//   Ā_t  = exp(Δ_t ⊗ A)              [B, L, D_inner, N]  (A is diagonal→N)
//   B̄_t  = Δ_t ⊗ B_t                 [B, L, D_inner, N]
//   h_t  = Ā_t h_{t-1} + B̄_t u_t     recurrence
//   y_t  = C_t h_t                    output
//
// Returns y ∈ R^{B × L × D_inner}
// ─────────────────────────────────────────────────────────────────────────────
struct MambaSSMImpl : torch::nn::Module {
    explicit MambaSSMImpl(const MambaConfig& cfg);

    // Training / parallel forward: processes the full sequence at once.
    // u: [B, L, d_inner]
    // Returns y: [B, L, d_inner]
    torch::Tensor forward(torch::Tensor u);

    // Inference step: one token at a time.
    // u_t: [B, d_inner]     — single time-step input
    // h:   [B, d_inner, N]  — running hidden state (updated in-place)
    // Returns y_t: [B, d_inner]
    torch::Tensor step(torch::Tensor u_t, torch::Tensor& h);

    // SSM parameters
    torch::Tensor A_log;   // [d_inner, N] — log(-A), A is fixed diagonal <0
    torch::Tensor D;       // [d_inner]    — skip connection weight

    // Input-dependent projections
    torch::nn::Linear x_proj{nullptr};  // u → [dt_rank + 2*N]
    torch::nn::Linear dt_proj{nullptr}; // dt_rank → d_inner  (with bias = dt_bias)

    int64_t d_inner, d_state, dt_rank_;
};
TORCH_MODULE(MambaSSM);

// ─────────────────────────────────────────────────────────────────────────────
// Mamba block (§3.4, Fig. 3)
//
// Residual(x):
//   z, x  ← split(in_proj(x), [d_inner, d_inner])
//   x     ← SiLU(conv1d(x))       depthwise, kernel=d_conv
//   x     ← SSM(x)
//   y     ← x ⊙ SiLU(z)
//   out   ← out_proj(y)
// ─────────────────────────────────────────────────────────────────────────────
struct MambaBlockImpl : torch::nn::Module {
    explicit MambaBlockImpl(const MambaConfig& cfg);

    // x: [B, L, d_model]  → y: [B, L, d_model]
    torch::Tensor forward(torch::Tensor x);

    // One-token inference step.
    // x_t: [B, d_model]
    // h:   [B, d_inner, N]  — SSM hidden state (updated in-place)
    // conv_state: [B, d_inner, d_conv]  — conv sliding window (updated in-place)
    torch::Tensor step(torch::Tensor x_t,
                       torch::Tensor& h,
                       torch::Tensor& conv_state);

    MambaRMSNorm   norm{nullptr};
    torch::nn::Linear in_proj{nullptr};   // d_model → 2*d_inner, no bias
    torch::nn::Conv1d conv1d{nullptr};    // depthwise, groups=d_inner, kernel=d_conv
    MambaSSM       ssm{nullptr};
    torch::nn::Linear out_proj{nullptr};  // d_inner → d_model, no bias

    int64_t d_model, d_inner, d_conv;
};
TORCH_MODULE(MambaBlock);

// ─────────────────────────────────────────────────────────────────────────────
// Full Mamba language model
// ─────────────────────────────────────────────────────────────────────────────
struct MambaModelImpl : torch::nn::Module {
    explicit MambaModelImpl(const MambaConfig& cfg);

    // tokens: [B, L]  int64
    // Returns logits: [B, L, vocab_size]
    torch::Tensor forward(torch::Tensor tokens);

    // Autoregressive generation (one token at a time, O(1) state).
    // prompt: [B, T_prompt]  — conditioning tokens
    // max_new: number of new tokens to generate
    // temperature, top_p: sampling params
    torch::Tensor generate(torch::Tensor prompt,
                           int64_t max_new = 100,
                           double temperature = 1.0,
                           double top_p = 0.9);

    MambaConfig              cfg;
    torch::nn::Embedding     embedding{nullptr};
    torch::nn::ModuleList    blocks{nullptr};
    MambaRMSNorm             norm_f{nullptr};
    torch::nn::Linear        lm_head{nullptr};
};
TORCH_MODULE(MambaModel);

// ─────────────────────────────────────────────────────────────────────────────
// Training utilities
// ─────────────────────────────────────────────────────────────────────────────
struct MambaTrainConfig {
    double  lr          = 6e-4;   // peak LR (GPT3 spec for 125M)
    double  min_lr      = 1e-5;   // cosine decay floor
    double  clip        = 1.0;
    double  weight_decay= 0.1;
    double  beta1       = 0.9;
    double  beta2       = 0.95;
    int64_t warmup      = 0;
    int64_t total_steps = 4800;
};

torch::optim::AdamW make_mamba_optimizer(MambaModel& model,
                                         const MambaTrainConfig& tcfg);

// LR: linear warmup then cosine decay to min_lr
double mamba_lr_schedule(int64_t step, const MambaTrainConfig& tcfg);

// Single training step.
// tokens: [B, L+1]  — input=tokens[:,0..L-1], target=tokens[:,1..L]
// Returns (loss_tensor, nll_per_token)
std::pair<torch::Tensor, double>
mamba_train_step(MambaModel& model,
                 torch::optim::AdamW& optimizer,
                 torch::Tensor tokens,
                 const MambaTrainConfig& tcfg,
                 int64_t step);

} // namespace nlp
} // namespace models
} // namespace dm
