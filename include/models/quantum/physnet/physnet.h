#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// PhysNet — A neural network for predicting energies, forces, dipole moments
// and partial charges
// Unke & Meuwly, arXiv:1902.08408v2, 2019.
//
// Implements the paper architecture:
//   - Atom embeddings x_i^0 = e_Zi
//   - Nmodule modular blocks
//   - Interaction block with gated feature update and RBF attention mask
//   - Pre-activation residual blocks
//   - Module-wise output summation with element-specific scale/shift
//   - Optional energy+charge output with charge correction
//   - Long-range damped Coulomb energy term
//   - Dipole and force prediction by differentiating energy
//
// The optional DFT-D3 dispersion correction in Eq. 12 is not implemented here:
// the paper references DFT-D3 but does not provide the full parameterization.
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dm {
namespace models {
namespace quantum {

struct PhysNetConfig {
    int64_t max_z = 100;
    int64_t feature_dim = 128;       // F, Table 1
    int64_t n_rbf = 64;              // K, Table 1
    int64_t n_modules = 5;           // Nmodule, Table 1
    int64_t n_atomic_residual = 2;   // Nresidual atomic
    int64_t n_interaction_residual = 3;
    int64_t n_output_residual = 1;
    double cutoff = 10.0;            // Angstrom, Table 1
    bool predict_charges = true;
    bool long_range = true;
    double coulomb_const = 1.0;

    double w_energy = 1.0;
    double w_force = 100.0;
    double w_charge = 1.0;
    double w_dipole = 1.0;
    double lambda_nonhierarchical = 1e-2;

    int64_t batch_size = 32;
    int64_t max_epochs = 100;
    double lr = 1e-3;
    double ema_decay = 0.999;

    static PhysNetConfig default_config();
    static PhysNetConfig energy_only();
};

torch::Tensor physnet_shifted_softplus(torch::Tensor x);
torch::Tensor physnet_cutoff(torch::Tensor r, double cutoff);

struct PhysNetResidualBlockImpl : torch::nn::Module {
    torch::nn::Linear dense1{nullptr};
    torch::nn::Linear dense2{nullptr};

    explicit PhysNetResidualBlockImpl(int64_t dim);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PhysNetResidualBlock);

struct PhysNetRBFImpl : torch::nn::Module {
    int64_t n_rbf;
    double cutoff;
    torch::Tensor mu;
    torch::Tensor beta;

    PhysNetRBFImpl(int64_t n_rbf = 64, double cutoff = 10.0);
    torch::Tensor forward(torch::Tensor distances);
};
TORCH_MODULE(PhysNetRBF);

struct PhysNetInteractionBlockImpl : torch::nn::Module {
    int64_t feature_dim;
    int64_t n_rbf;
    double cutoff;

    torch::nn::Linear wi{nullptr};
    torch::nn::Linear wj{nullptr};
    torch::nn::Linear g{nullptr};
    torch::nn::ModuleList residuals{nullptr};
    torch::nn::Linear out{nullptr};
    torch::Tensor gate;

    PhysNetInteractionBlockImpl(int64_t feature_dim,
                                int64_t n_rbf,
                                int64_t n_residual,
                                double cutoff);

    torch::Tensor forward(torch::Tensor x, torch::Tensor rbf,
                          torch::Tensor distances);
};
TORCH_MODULE(PhysNetInteractionBlock);

struct PhysNetOutputBlockImpl : torch::nn::Module {
    torch::nn::ModuleList residuals{nullptr};
    torch::nn::Linear out{nullptr};

    PhysNetOutputBlockImpl(int64_t feature_dim,
                           int64_t n_residual,
                           int64_t n_out);

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PhysNetOutputBlock);

struct PhysNetModuleBlockImpl : torch::nn::Module {
    PhysNetInteractionBlock interaction{nullptr};
    torch::nn::ModuleList atomic_residuals{nullptr};
    PhysNetOutputBlock output{nullptr};

    PhysNetModuleBlockImpl(const PhysNetConfig& cfg, int64_t n_out);

    std::pair<torch::Tensor, torch::Tensor>
    forward(torch::Tensor x, torch::Tensor rbf, torch::Tensor distances);
};
TORCH_MODULE(PhysNetModuleBlock);

struct PhysNetImpl : torch::nn::Module {
    PhysNetConfig cfg;
    int64_t n_out;

    torch::nn::Embedding embedding{nullptr};
    PhysNetRBF rbf{nullptr};
    torch::nn::ModuleList modules_list{nullptr};
    torch::nn::Embedding scale{nullptr};
    torch::nn::Embedding shift{nullptr};

    explicit PhysNetImpl(PhysNetConfig cfg = PhysNetConfig::default_config());

    torch::Tensor module_outputs(torch::Tensor atomic_numbers,
                                 torch::Tensor positions);
    torch::Tensor atomic_properties(torch::Tensor atomic_numbers,
                                    torch::Tensor positions);
    torch::Tensor corrected_charges(torch::Tensor atomic_numbers,
                                    torch::Tensor positions,
                                    double total_charge = 0.0);
    torch::Tensor energy(torch::Tensor atomic_numbers,
                         torch::Tensor positions,
                         double total_charge = 0.0);
    torch::Tensor dipole(torch::Tensor atomic_numbers,
                         torch::Tensor positions,
                         double total_charge = 0.0);
    std::pair<torch::Tensor, torch::Tensor>
    energy_and_forces(torch::Tensor atomic_numbers,
                      torch::Tensor positions,
                      double total_charge = 0.0);

    torch::Tensor forward(torch::Tensor atomic_numbers,
                          torch::Tensor positions);
};
TORCH_MODULE(PhysNet);

struct PhysNetBatch {
    torch::Tensor atomic_numbers;  // [N] int64
    torch::Tensor positions;       // [N,3] float
    torch::Tensor energy;          // scalar
    torch::Tensor forces;          // [N,3], optional
    torch::Tensor dipole;          // [3], optional
    double total_charge = 0.0;
};

torch::Tensor physnet_loss(PhysNet& model, const PhysNetBatch& batch,
                           bool use_forces = true,
                           bool use_dipole = true);

float physnet_train_epoch(PhysNet& model,
                          torch::optim::Adam& optimizer,
                          const std::vector<PhysNetBatch>& batches,
                          bool use_forces = true,
                          bool use_dipole = true);

float physnet_evaluate_mae(PhysNet& model,
                           const std::vector<PhysNetBatch>& batches);

PhysNet make_physnet();
PhysNet make_physnet_energy_only();

}  // namespace quantum
}  // namespace models
}  // namespace dm
