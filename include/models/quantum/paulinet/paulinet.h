#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// PauliNet — Deep Neural Network Solution of the Electronic Schrödinger Equation
// Hermann, Schätzle & Noé, arXiv:1909.08423v5, Nature Chemistry 2020
// https://arxiv.org/abs/1909.08423
//
// Wave function ansatz (Eq. 1):
//   ψ_θ(r) = e^{J(r) + γ(r)} · Σ_p c_p · det[φ̃↑_{θ,μ_p}(r)] · det[φ̃↓_{θ,μ_p}(r)]
//   φ̃_μ(r) = φ_μ(r_i) · f_{θ,μi}(r)        — backflow-modified orbitals
//
// Three physics components:
//   1. Cusp factor γ(r): enforces electron–electron cusp conditions (Eq. 9)
//      γ(r) = -Σ_{i<j} c_{ij} / (1 + |r_i − r_j|)
//      c_{ij} = 1/2 (same spin), 1/4 (opposite spin)
//
//   2. Jastrow factor J(r): symmetric correlation function (Eq. 5)
//      J = η_θ(Σ_i x_i^(L))    — MLP on summed electron features
//
//   3. Backflow f_i(r): antisymmetric orbital modification (Eq. 5)
//      f_i = κ_θ(x_i^(L))      — MLP on per-electron features
//
// Electron features via SchNet-like iterations (Eq. 11):
//   z_i^{(n,±)} = Σ_{j≠i} w_θ^{(n,n)}(e(|r_i−r_j|)) ⊙ h_θ^n(x_j^{(n)})
//   z_i^{(n)}   = Σ_I  w_θ^{(n,n)}(e(|r_i−R_I|))   ⊙ Y_{θ,I}
//   x_i^{(n+1)} = x_i^{(n)} + Σ_± g_θ^{(n,±)}(z_i^{(n,±)}) + g_θ^n(z_i^{(n)})
//
// Distance features (Eqs. 12–14):
//   e_k(r) = r² · exp(-(r - μ_k)² / σ_k²)
//   μ_k := r_c · q_k²,   σ_k := (1/7)(1 + r_c · q_k),   q_k ∈ (0,1) uniform
//
// VMC loss & gradient (Eqs. 6–8):
//   E[ψ] = E_{r~|ψ|²}[E_loc[ψ](r)]
//   ∇_θ L = 2·E_{r~|ψ|²}[(E_loc[ψ](r) − E[ψ]) · ∇_θ ln|ψ_θ(r)|]
//
// Hyperparameters (Table 2):
//   dim_e=32, dim_x=128, dim_z=64, L=3 interaction layers
//   batch=10000, walkers=2000, steps=7000, lr=0.01, AdamW
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <vector>
#include <string>
#include <cstdint>

namespace dm {
namespace models {
namespace quantum {

// ── PauliNetConfig ────────────────────────────────────────────────────────────

struct PauliNetConfig {
    // System definition
    int64_t n_up        = 1;    // N↑ spin-up electrons
    int64_t n_down      = 0;    // N↓ spin-down electrons
    int64_t n_nuclei    = 1;    // M nuclei
    int64_t n_det       = 1;    // number of Slater determinants

    // Network dimensions (Table 2)
    int64_t dim_e       = 32;   // distance embedding dim
    int64_t dim_x       = 128;  // electron feature dim
    int64_t dim_z       = 64;   // message dim
    int64_t n_layers    = 3;    // SchNet interaction layers L

    // MLP depths (Table 2)
    int64_t n_eta       = 3;    // layers in η_θ (Jastrow)
    int64_t n_kappa     = 3;    // layers in κ_θ (backflow)
    int64_t n_w         = 2;    // layers in w_θ (message weight)
    int64_t n_h         = 1;    // layers in h_θ (feature transform)
    int64_t n_g         = 1;    // layers in g_θ (aggregation)

    // Distance featurisation (Eqs. 12–14)
    int64_t n_rbf       = 32;   // number of radial basis functions
    double  r_cutoff    = 5.0;  // r_c in bohr

    // VMC training (Table 2)
    int64_t n_walkers   = 2000;
    int64_t n_steps     = 7000;
    int64_t batch_size  = 1000; // walkers sampled per gradient step
    double  lr          = 0.01;
    int64_t lr_decay_period = 200;
    double  clip_window = 5.0;  // local energy clipping (×median deviance)
    int64_t resample_period  = 100;
    int64_t n_discard        = 50;
    int64_t n_decorrelate    = 1;
    double  target_acceptance = 0.57;
    double  mcmc_step_size   = 0.02; // initial MCMC step size (bohr)

    // Convenience constructors
    static PauliNetConfig hydrogen();   // H atom: 1e↑, 1 nucleus Z=1
    static PauliNetConfig h2();         // H₂: 1e↑+1e↓, 2 nuclei, 1 det
    static PauliNetConfig lih();        // LiH: 2e↑+2e↓, 2 nuclei, 4 det
};

// ── Distance featurisation (Eqs. 12–14) ──────────────────────────────────────
// e_k(r) = r² · exp(-(r - μ_k)² / σ_k²)
// Cuspless: e_k(0)=0, (d/dr)e_k(0)=0  (Eq. 10 + text below Eq. 14)

struct DistanceFeatureImpl : torch::nn::Module {
    torch::Tensor mu;     // [K] centres
    torch::Tensor sigma;  // [K] widths

    DistanceFeatureImpl(int64_t n_rbf, double r_cutoff);
    // r: [...] non-negative distances → [..., K]
    torch::Tensor forward(torch::Tensor r);
};
TORCH_MODULE(DistanceFeature);

// ── Simple MLP helper ─────────────────────────────────────────────────────────

struct MLPImpl : torch::nn::Module {
    torch::nn::Sequential net{nullptr};

    // dims: [in, h1, h2, ..., out]; tanh activations except last layer linear
    MLPImpl(std::vector<int64_t> dims);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MLP);

// ── SchNet interaction block (Eq. 11) ─────────────────────────────────────────
// One iteration of the electron feature update.
// Handles three message channels: same-spin (+), opposite-spin (−), nucleus (n).

struct SchNetLayerImpl : torch::nn::Module {
    // w_θ: distance features → message weights  [dim_e → dim_z]
    MLP w_same{nullptr}, w_opp{nullptr}, w_nuc{nullptr};
    // h_θ: electron features → values            [dim_x → dim_z]
    MLP h_same{nullptr}, h_opp{nullptr};
    // g_θ: aggregated messages → feature update  [dim_z → dim_x]
    MLP g_same{nullptr}, g_opp{nullptr}, g_nuc{nullptr};

    int64_t n_up, n_down, dim_z, dim_x;

    SchNetLayerImpl(int64_t n_up, int64_t n_down,
                    int64_t dim_e, int64_t dim_x, int64_t dim_z,
                    int64_t n_w, int64_t n_h, int64_t n_g);

    // x:     [N, dim_x]  electron features
    // e_ee:  [N, N, dim_e] electron-electron distance features
    // e_en:  [N, M, dim_e] electron-nucleus distance features
    // Returns updated x: [N, dim_x]
    torch::Tensor forward(torch::Tensor x,
                          torch::Tensor e_ee,
                          torch::Tensor e_en);
};
TORCH_MODULE(SchNetLayer);

// ── Cusp factor γ(r) (Eq. 9) ─────────────────────────────────────────────────
// γ(r) = -Σ_{i<j} c_{ij} / (1 + |r_i − r_j|)

struct ElectronCuspImpl : torch::nn::Module {
    int64_t n_up, n_down;

    ElectronCuspImpl(int64_t n_up, int64_t n_down);
    // r_ee: [N, N] pairwise distances
    // Returns scalar γ
    torch::Tensor forward(torch::Tensor r_ee);
};
TORCH_MODULE(ElectronCusp);

// ── PauliNet ──────────────────────────────────────────────────────────────────

struct PauliNetImpl : torch::nn::Module {
    PauliNetConfig cfg;
    int64_t n_elec;   // N = n_up + n_down

    // ── Nuclear geometry (fixed for a given molecule) ─────────────────────────
    torch::Tensor nuclei_pos;    // [M, 3]  nuclear positions (bohr)
    torch::Tensor nuclear_charge;// [M]     atomic numbers Z

    // ── Trainable nuclear embeddings Y_{θ,I} (Eq. 11, point iv) ─────────────
    torch::Tensor nuclear_emb;   // [M, dim_x]

    // ── Electron initial features x_i^(0) (one per spin type) ────────────────
    torch::Tensor init_up, init_down;  // [dim_x]

    // ── Distance featurisation ────────────────────────────────────────────────
    DistanceFeature dist_feat{nullptr};

    // ── SchNet interaction layers ─────────────────────────────────────────────
    torch::nn::ModuleList schnet_layers{nullptr};

    // ── Jastrow η_θ (Eq. 5): [dim_x → ... → 1] ──────────────────────────────
    MLP jastrow{nullptr};

    // ── Backflow κ_θ (Eq. 5): [dim_x → ... → n_det·n_orb] ───────────────────
    // Separate for spin-up and spin-down orbitals
    MLP backflow_up{nullptr}, backflow_down{nullptr};

    // ── Electronic cusp γ(r) ─────────────────────────────────────────────────
    ElectronCusp cusp{nullptr};

    // ── Hartree–Fock orbitals φ_μ: represented as learned linear layer ────────
    // [3 → n_det·N↑] and [3 → n_det·N↓]  (simplified single-zeta-like basis)
    torch::nn::Linear hf_up{nullptr}, hf_down{nullptr};

    // ── Determinant coefficients c_p (Eq. 1) ─────────────────────────────────
    torch::Tensor det_coeffs;  // [n_det]

    PauliNetImpl(PauliNetConfig cfg,
                 torch::Tensor nuclei_pos,
                 torch::Tensor nuclear_charge);

    // ── Forward: compute log|ψ| and sign  ────────────────────────────────────
    // r: [N, 3] electron positions (bohr)
    // Returns {log_abs_psi, sign_psi}  both scalars
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor r);

    // ── Compute local energy E_loc[ψ](r) = Ĥψ(r)/ψ(r) ───────────────────────
    // Uses automatic differentiation for kinetic energy (∇²ψ/ψ)
    // and explicit nuclear/electron repulsion for potential energy.
    torch::Tensor local_energy(torch::Tensor r);

    // ── VMC energy estimate (Eq. 7) over a batch of electron configs ─────────
    torch::Tensor vmc_energy(torch::Tensor r_batch);

    // ── VMC gradient (Eq. 8) ─────────────────────────────────────────────────
    // Returns mean local energy (for logging) and sets .grad on parameters
    double vmc_step(torch::Tensor r_batch);

    // ── Potential energy V(r) = electron–nucleus + electron–electron + nuc–nuc
    torch::Tensor potential_energy(torch::Tensor r);

private:
    // Compute all pairwise distances
    std::pair<torch::Tensor, torch::Tensor>
    compute_distances(torch::Tensor r);

    // Run SchNet to get final electron features x^(L)  [N, dim_x]
    torch::Tensor electron_features(torch::Tensor r,
                                    torch::Tensor r_ee,
                                    torch::Tensor r_en);

    // Build backflow-modified Slater matrices  [n_det, N↑, N↑] and [n_det, N↓, N↓]
    std::pair<torch::Tensor, torch::Tensor>
    slater_matrices(torch::Tensor r, torch::Tensor x_final);

    // Multi-determinant expansion log|Σ c_p det↑_p det↓_p|
    torch::Tensor log_wavefunction(torch::Tensor slater_up,
                                   torch::Tensor slater_down,
                                   torch::Tensor jastrow_val,
                                   torch::Tensor cusp_val);
};
TORCH_MODULE(PauliNet);

// ── MCMC sampler ─────────────────────────────────────────────────────────────
// Metropolis–Hastings sampler for |ψ|² (Methods: Langevin / simple random walk)

struct MCMCSampler {
    int64_t n_walkers;
    int64_t n_elec;
    double  step_size;
    double  target_acceptance;

    torch::Tensor walkers;    // [W, N, 3] current electron positions
    torch::Tensor log_psi2;   // [W]       cached 2·log|ψ| per walker

    MCMCSampler(int64_t n_walkers, int64_t n_elec,
                double step_size, double target_acceptance,
                torch::Tensor init_positions);

    // Run n_steps of Metropolis–Hastings; adapt step size toward target acceptance.
    // Returns acceptance rate.
    double step(PauliNet& model, int64_t n_steps);

    // Return a random batch of walker positions: [batch_size, N, 3]
    torch::Tensor sample_batch(int64_t batch_size) const;
};

// ── VMC trainer ──────────────────────────────────────────────────────────────

struct VMCTrainer {
    PauliNetConfig cfg;
    PauliNet       model;
    MCMCSampler    sampler;
    torch::optim::AdamW optimizer;

    VMCTrainer(PauliNetConfig cfg,
               torch::Tensor nuclei_pos,
               torch::Tensor nuclear_charge);

    // Run one full VMC optimisation step:
    //   1. MCMC decorrelation
    //   2. Sample batch
    //   3. Compute VMC gradient (Eq. 8)
    //   4. Adam step
    // Returns {mean_energy, variance}
    std::pair<double, double> train_step();

    // Full training loop (n_steps iterations)
    void train(int64_t n_steps, bool verbose = true);
};

// ── Factory helpers ───────────────────────────────────────────────────────────

// Hydrogen atom: 1 electron (spin-up), 1 proton at origin
PauliNet make_paulinet_hydrogen();

// H₂ molecule at equilibrium (d=1.401 a₀): 2 electrons, 2 protons
PauliNet make_paulinet_h2(double bond_length_bohr = 1.401);

// Helium atom: 2 electrons (1↑+1↓), 1 nucleus Z=2
PauliNet make_paulinet_helium();

}  // namespace quantum
}  // namespace models
}  // namespace dm
