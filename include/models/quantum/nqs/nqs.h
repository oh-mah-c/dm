#pragma once
// -----------------------------------------------------------------------------
// Neural Quantum States / VMC utilities
// Medvidovic & Robledo Moreno, "Neural-network quantum states for many-body
// physics", arXiv:2402.11014v2.
//
// This module implements the review paper's explicit RBM neural quantum-state
// ansatz and the standard VMC ingredients it describes:
//   - psi_theta(s) = exp(s.a) prod_h 2 cosh((sW + b)_h)
//   - p_theta(s) proportional to |psi_theta(s)|^2
//   - local-energy estimator E_loc(s) = <s|H|psi>/<s|psi>
//   - Metropolis-Hastings sampling with spin-flip proposals
//   - exact small-system TFIM energy for deterministic tests/training smoke
// -----------------------------------------------------------------------------

#include <torch/torch.h>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace dm {
namespace models {
namespace quantum {

struct NQSRBMConfig {
    int64_t n_visible = 16;
    int64_t n_hidden  = 32;
    double init_std   = 0.01;

    static NQSRBMConfig tiny(int64_t n_visible = 4, int64_t n_hidden = 4);
};

struct NQSTFIM {
    double j = 1.0;          // Ising coupling in -J sum_i sz_i sz_{i+1}
    double h = 1.0;          // transverse field in -h sum_i sx_i
    bool periodic = true;
};

struct NQSRBMImpl : torch::nn::Module {
    NQSRBMConfig cfg;

    torch::Tensor visible_bias;  // [N]
    torch::Tensor hidden_bias;   // [H]
    torch::Tensor weight;        // [N,H]

    explicit NQSRBMImpl(NQSRBMConfig cfg = NQSRBMConfig{});

    torch::Tensor log_amplitude(torch::Tensor spins);
    torch::Tensor amplitude(torch::Tensor spins);
    torch::Tensor log_prob(torch::Tensor spins);
    torch::Tensor forward(torch::Tensor spins);
};
TORCH_MODULE(NQSRBM);

torch::Tensor nqs_all_spin_configs(int64_t n_visible);
torch::Tensor nqs_random_spin_config(int64_t n_visible);
torch::Tensor nqs_flip_spin(torch::Tensor spins, int64_t index);

torch::Tensor nqs_tfim_local_energy(NQSRBM& model,
                                    torch::Tensor spins,
                                    const NQSTFIM& hamiltonian);

torch::Tensor nqs_exact_energy_tfim(NQSRBM& model,
                                    const NQSTFIM& hamiltonian);

struct NQSMetropolisSampler {
    torch::Tensor spins;  // [N], values are -1 or +1
    std::mt19937 rng;

    explicit NQSMetropolisSampler(int64_t n_visible,
                                  uint64_t seed = 1234);
    explicit NQSMetropolisSampler(torch::Tensor initial_spins,
                                  uint64_t seed = 1234);

    double step(NQSRBM& model, int64_t n_steps = 1);
    torch::Tensor sample_batch(NQSRBM& model,
                               int64_t batch_size,
                               int64_t decorrelation_steps = 1);
};

float nqs_train_exact_step(NQSRBM& model,
                           torch::optim::Optimizer& optimizer,
                           const NQSTFIM& hamiltonian);

NQSRBM make_nqs_rbm(int64_t n_visible, int64_t n_hidden);

}  // namespace quantum
}  // namespace models
}  // namespace dm
