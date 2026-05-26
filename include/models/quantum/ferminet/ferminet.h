#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// FermiNet — Ab-Initio Solution of the Many-Electron Schrödinger Equation
//            with Deep Neural Networks
// Pfau, Spencer, Matthews & Foulkes (DeepMind), arXiv:1909.02487v3, 2021
// https://arxiv.org/abs/1909.02487
//
// Two-stream architecture (Algorithm 1):
//   Input features (single-electron stream, Eq 4a):
//     h_i^{0α} = (r_i^α − R_I, |r_i^α − R_I|  ∀I)   ∈ R^{4·M}
//   Input features (two-electron stream, Eq 4b):
//     h_ij^{0αβ} = (r_i^α − r_j^β, |r_i^α − r_j^β|)  ∈ R^4
//
//   Layer update (Eq 5):
//     g^{ℓ↑} = mean_i h_i^{ℓ↑}               [global spin-up mean]
//     g^{ℓ↓} = mean_i h_i^{ℓ↓}               [global spin-down mean]
//     g_i^{ℓα↑} = mean_j h_ij^{ℓα↑}          [per-electron same-spin 2e mean]
//     g_i^{ℓα↓} = mean_j h_ij^{ℓα↓}          [per-electron opp-spin 2e mean]
//     f_i^{ℓα} = (h_i^{ℓα}, g^{ℓ↑}, g^{ℓ↓}, g_i^{ℓα↑}, g_i^{ℓα↓})
//     h_i^{ℓ+1,α} = tanh(V^ℓ f_i^{ℓα} + b^ℓ) + h_i^{ℓα}   (residual)
//     h_ij^{ℓ+1,αβ} = tanh(W^ℓ h_ij^{ℓαβ} + c^ℓ) + h_ij^{ℓαβ}  (residual)
//
//   Orbital with exponential envelope (Eq 6):
//     φ_i^kα(r_j) = (w_i^kα · h_j^{Lα} + g_i^kα)
//                 · Σ_m π_im^kα exp(−||Σ_im^kα (r_j − R_m)||)
//     Σ_im^kα ∈ R^{3×3} — anisotropic decay matrix
//
//   Wave function (Eq 7):
//     ψ = Σ_k ω_k det[Φ^{k↑}] · det[Φ^{k↓}]
//
//   VMC loss/gradient (Eq 9):
//     ∇_θ L = 2·E_{r~|ψ|²}[(E_L − Ē) · ∇_θ log|ψ|]
//
//   Local energy (Eq 2):
//     E_L = −½ Σ_i ∇²_i log|ψ| − ½ Σ_i |∇_i log|ψ||² + V(r)
//
// Hyperparameters (Table V):
//   L=4 layers, n_1=256 (1e hidden), n_2=32 (2e hidden), n_k=16 dets,
//   batch=4096, 2×10^5 iters, lr=(1e4+t)^{−1}, KFAC (decay=0.95,
//   norm_constraint=1e-3, damping=1e-3), MCMC step=0.02 bohr, 10 MCMC
//   steps per update, clip=5×MAD
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <vector>
#include <string>
#include <cstdint>

namespace dm {
namespace models {
namespace quantum {

// ── FermiNetConfig ────────────────────────────────────────────────────────────

struct FermiNetConfig {
    // System definition
    int64_t n_up       = 1;   // N↑ spin-up electrons
    int64_t n_down     = 0;   // N↓ spin-down electrons
    int64_t n_nuclei   = 1;   // M nuclei

    // Network architecture (Table V)
    int64_t n_layers   = 4;   // L interaction layers
    int64_t dim_1e     = 256; // n_1: single-electron stream hidden dim
    int64_t dim_2e     = 32;  // n_2: two-electron stream hidden dim
    int64_t n_det      = 16;  // n_k: number of determinants ω_k

    // VMC training
    int64_t n_walkers         = 2000;
    int64_t n_steps           = 200000;
    int64_t batch_size        = 4096;
    double  lr_decay_offset   = 1e4;   // lr = 1/(lr_decay_offset + t)
    double  clip_factor       = 5.0;   // E_L clipping: ± clip_factor × MAD
    int64_t mcmc_steps_per_update = 10;
    double  mcmc_step_size    = 0.02;  // bohr
    double  target_acceptance = 0.57;
    int64_t n_discard         = 200;   // burn-in

    // KFAC parameters
    double  kfac_decay            = 0.95;
    double  kfac_norm_constraint  = 1e-3;
    double  kfac_damping          = 1e-3;

    // Pretraining (Appendix A)
    bool    pretrain           = false;
    int64_t pretrain_steps     = 1000;
    double  pretrain_lr        = 1e-3;

    // Convenience constructors
    static FermiNetConfig hydrogen();   // H: 1e↑, Z=1
    static FermiNetConfig helium();     // He: 1e↑+1e↓, Z=2
    static FermiNetConfig h2();         // H₂: 1e↑+1e↓, 2 protons
    static FermiNetConfig lih();        // LiH: 2e↑+2e↓
    static FermiNetConfig carbon();     // C: 3e↑+3e↓, Z=6 (from Table I)
};

// ── FermiNetLayer — one two-stream interaction block ──────────────────────────
// Implements Eq 5 update for both streams.

struct FermiNetLayerImpl : torch::nn::Module {
    // Single-electron linear (f → h): V^ℓ x + b^ℓ
    torch::nn::Linear lin_1e_up{nullptr};    // for ↑ electrons
    torch::nn::Linear lin_1e_down{nullptr};  // for ↓ electrons (if n_down > 0)

    // Two-electron linear (h_ij → h_ij): W^ℓ x + c^ℓ
    torch::nn::Linear lin_2e{nullptr};

    int64_t n_up, n_down;
    int64_t dim_1e_in;   // input dim of f (single-electron concat)
    int64_t dim_1e_out;  // output dim
    int64_t dim_2e_in;
    int64_t dim_2e_out;
    bool    residual;    // apply residual only when dims match (ℓ ≥ 1)

    FermiNetLayerImpl(int64_t n_up, int64_t n_down,
                      int64_t dim_1e_in, int64_t dim_1e_out,
                      int64_t dim_2e_in, int64_t dim_2e_out,
                      bool residual);

    // x_up:    [n_up,   dim_1e_in]  single-electron features (↑)
    // x_down:  [n_down, dim_1e_in]  single-electron features (↓)
    // h_ee_uu: [n_up,   n_up,   dim_2e_in]  2e features ↑↑
    // h_ee_ud: [n_up,   n_down, dim_2e_in]  2e features ↑↓
    // h_ee_du: [n_down, n_up,   dim_2e_in]  2e features ↓↑
    // h_ee_dd: [n_down, n_down, dim_2e_in]  2e features ↓↓
    // Returns {x_up_new, x_down_new, h_ee_uu_new, h_ee_ud_new, h_ee_du_new, h_ee_dd_new}
    std::tuple<torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor>
    forward(torch::Tensor x_up,  torch::Tensor x_down,
            torch::Tensor h_uu, torch::Tensor h_ud,
            torch::Tensor h_du, torch::Tensor h_dd);
};
TORCH_MODULE(FermiNetLayer);

// ── OrbitalLayer — backflow-modified orbitals with exponential envelope ───────
// Implements Eq 6: φ_i^kα(r_j) = linear(h_j) · envelope(r_j, {R_m})

struct OrbitalLayerImpl : torch::nn::Module {
    int64_t n_up, n_down, n_nuclei, n_det, dim_1e;

    // Linear weights: w_i^kα · h_j + g_i^kα
    // For ↑ electrons: output dim = n_det * n_up
    // For ↓ electrons: output dim = n_det * n_down
    torch::nn::Linear linear_up{nullptr};    // [dim_1e → n_det * n_up]
    torch::nn::Linear linear_down{nullptr};  // [dim_1e → n_det * n_down]

    // Exponential envelope parameters (Eq 6)
    // π_im^kα ∈ R^{n_det × N↑/↓ × M}
    torch::Tensor pi_up;    // [n_det, n_up,   n_nuclei]
    torch::Tensor pi_down;  // [n_det, n_down, n_nuclei]
    // Σ_im^kα ∈ R^{n_det × N × M × 3} (diagonal approx: one scalar per dim)
    // We use diagonal Σ → [n_det, N, M] scale per dimension
    torch::Tensor sigma_up;   // [n_det, n_up,   n_nuclei, 3]  anisotropic decay
    torch::Tensor sigma_down; // [n_det, n_down, n_nuclei, 3]

    // Determinant combination weights ω_k (Eq 7)
    torch::Tensor det_weights;  // [n_det]

    OrbitalLayerImpl(int64_t n_up, int64_t n_down,
                     int64_t n_nuclei, int64_t n_det, int64_t dim_1e);

    // h_up:   [n_up,   dim_1e]  final single-electron features for ↑
    // h_down: [n_down, dim_1e]  final single-electron features for ↓
    // r:      [N, 3]            electron positions (bohr)
    // npos:   [M, 3]            nuclear positions
    // Returns {slater_up [n_det, n_up, n_up], slater_down [n_det, n_down, n_down]}
    std::pair<torch::Tensor, torch::Tensor>
    forward(torch::Tensor h_up, torch::Tensor h_down,
            torch::Tensor r, torch::Tensor npos);

private:
    // Compute envelope: [n_det, N_spin, n_nuclei] → [n_det, N_spin, 1] (summed)
    torch::Tensor envelope_up(torch::Tensor r_spin, torch::Tensor npos);
    torch::Tensor envelope_down(torch::Tensor r_spin, torch::Tensor npos);
};
TORCH_MODULE(OrbitalLayer);

// ── FermiNet ──────────────────────────────────────────────────────────────────

struct FermiNetImpl : torch::nn::Module {
    FermiNetConfig cfg;
    int64_t n_elec;  // N = n_up + n_down

    // Nuclear geometry (fixed)
    torch::Tensor nuclei_pos;     // [M, 3]
    torch::Tensor nuclear_charge; // [M]

    // Two-stream interaction layers
    torch::nn::ModuleList layers{nullptr};

    // Orbital + envelope layer
    OrbitalLayer orbital{nullptr};

    FermiNetImpl(FermiNetConfig cfg,
                 torch::Tensor nuclei_pos,
                 torch::Tensor nuclear_charge);

    // r: [N, 3] electron positions
    // Returns {log_abs_psi, sign_psi} — both scalar tensors
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor r);

    // Local energy E_L(r) = kinetic + potential
    torch::Tensor local_energy(torch::Tensor r);

    // Potential energy V(r) = e-n + e-e + n-n
    torch::Tensor potential_energy(torch::Tensor r);

    // VMC step: compute VMC gradient (Eq 9), set .grad on params, return mean E
    double vmc_step(torch::Tensor r_batch);

private:
    // Build input features from r
    // Returns {x_up [n_up, dim0], x_down [n_down, dim0],
    //          h_uu, h_ud, h_du, h_dd}  (all 2e: [N_a, N_b, 4])
    std::tuple<torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor,
               torch::Tensor, torch::Tensor>
    input_features(torch::Tensor r);

    // Run all layers; return {h_up_final, h_down_final}
    std::pair<torch::Tensor, torch::Tensor>
    run_layers(torch::Tensor r);

    // Combine determinants: log|Σ_k ω_k det↑_k det↓_k|, sign
    std::pair<torch::Tensor, torch::Tensor>
    log_wavefunction(torch::Tensor slater_up, torch::Tensor slater_down);
};
TORCH_MODULE(FermiNet);

// ── MCMC sampler for FermiNet ─────────────────────────────────────────────────

struct FermiNetMCMC {
    int64_t n_walkers;
    int64_t n_elec;
    double  step_size;
    double  target_acceptance;

    torch::Tensor walkers;   // [W, N, 3]
    torch::Tensor log_psi2;  // [W] cached 2·log|ψ|

    FermiNetMCMC(int64_t n_walkers, int64_t n_elec,
                 double step_size, double target_acceptance,
                 torch::Tensor init_positions);

    // Run n_steps Metropolis-Hastings, adapt step_size
    double step(FermiNet& model, int64_t n_steps);

    // Sample random batch: [batch_size, N, 3]
    torch::Tensor sample_batch(int64_t batch_size) const;
};

// ── VMC Trainer ───────────────────────────────────────────────────────────────

struct FermiNetTrainer {
    FermiNetConfig cfg;
    FermiNet       model;
    FermiNetMCMC   sampler;
    torch::optim::Adam optimizer;  // Adam used as KFAC approximation

    FermiNetTrainer(FermiNetConfig cfg,
                    torch::Tensor nuclei_pos,
                    torch::Tensor nuclear_charge);

    // One VMC optimisation step: MCMC → sample → grad → optimizer step
    std::pair<double, double> train_step();

    // Full training loop
    void train(int64_t n_steps, bool verbose = true);

    // Pretrain: match HF-like orbitals (Appendix A simplified)
    void pretrain(int64_t n_steps);
};

// ── Factory helpers ───────────────────────────────────────────────────────────

FermiNet make_ferminet_hydrogen();
FermiNet make_ferminet_helium();
FermiNet make_ferminet_h2(double bond_length_bohr = 1.401);
FermiNet make_ferminet_lih();

}  // namespace quantum
}  // namespace models
}  // namespace dm
