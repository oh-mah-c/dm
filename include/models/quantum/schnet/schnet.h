#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// SchNet — A continuous-filter convolutional neural network for modeling
// quantum interactions
// Schutt, Kindermans, Sauceda, Chmiela, Tkatchenko & Muller, NeurIPS 2017.
//
// Implements the paper architecture for molecular energy and force prediction:
//   - Atom embeddings x_i^0 = a_Zi
//   - Continuous-filter convolution x_i^{l+1} = sum_j x_j^l ◦ W^l(||r_i-r_j||)
//   - Three residual interaction blocks by default
//   - Shifted softplus ssp(x)=softplus(x)-ln(2)
//   - RBF distances with centers 0..30 Angstrom every 0.1 Angstrom, gamma=10
//   - Atom-wise energy head and sum pooling
//   - Forces as negative gradient of energy w.r.t. positions
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dm {
namespace models {
namespace quantum {

struct SchNetConfig {
    int64_t max_z          = 100;
    int64_t hidden_dim     = 64;
    int64_t n_interactions = 3;
    int64_t n_gaussians    = 301;
    double  cutoff         = 30.0;
    double  rbf_step       = 0.1;
    double  gamma          = 10.0;
    double  rho_energy     = 0.01;

    int64_t batch_size     = 32;
    int64_t max_steps      = 100000;
    double  lr             = 1e-3;
    double  lr_decay       = 0.96;
    int64_t lr_decay_steps = 100000;
    double  ema_decay      = 0.99;

    static SchNetConfig qm9();
    static SchNetConfig md17();
};

torch::Tensor schnet_shifted_softplus(torch::Tensor x);

struct SchNetGaussianSmearingImpl : torch::nn::Module {
    int64_t n_gaussians;
    double  gamma;
    torch::Tensor offsets;

    SchNetGaussianSmearingImpl(int64_t n_gaussians = 301,
                               double start = 0.0,
                               double stop = 30.0,
                               double gamma = 10.0);

    torch::Tensor forward(torch::Tensor distances);
};
TORCH_MODULE(SchNetGaussianSmearing);

struct SchNetFilterNetworkImpl : torch::nn::Module {
    torch::nn::Linear dense1{nullptr};
    torch::nn::Linear dense2{nullptr};

    SchNetFilterNetworkImpl(int64_t n_gaussians, int64_t hidden_dim);
    torch::Tensor forward(torch::Tensor rbf);
};
TORCH_MODULE(SchNetFilterNetwork);

struct SchNetCFConvImpl : torch::nn::Module {
    SchNetGaussianSmearing rbf{nullptr};
    SchNetFilterNetwork filter_net{nullptr};

    int64_t hidden_dim;

    SchNetCFConvImpl(int64_t hidden_dim,
                     int64_t n_gaussians,
                     double cutoff,
                     double gamma);

    torch::Tensor forward(torch::Tensor x, torch::Tensor positions);
};
TORCH_MODULE(SchNetCFConv);

struct SchNetInteractionBlockImpl : torch::nn::Module {
    torch::nn::Linear atom_in{nullptr};
    SchNetCFConv cfconv{nullptr};
    torch::nn::Linear atom_hidden{nullptr};
    torch::nn::Linear atom_out{nullptr};

    int64_t hidden_dim;

    SchNetInteractionBlockImpl(int64_t hidden_dim,
                               int64_t n_gaussians,
                               double cutoff,
                               double gamma);

    torch::Tensor forward(torch::Tensor x, torch::Tensor positions);
};
TORCH_MODULE(SchNetInteractionBlock);

struct SchNetImpl : torch::nn::Module {
    SchNetConfig cfg;

    torch::nn::Embedding embedding{nullptr};
    torch::nn::ModuleList interactions{nullptr};
    torch::nn::Linear atomwise1{nullptr};
    torch::nn::Linear atomwise2{nullptr};

    explicit SchNetImpl(SchNetConfig cfg = SchNetConfig{});

    torch::Tensor atom_features(torch::Tensor atomic_numbers,
                                torch::Tensor positions);
    torch::Tensor atomic_energies(torch::Tensor atomic_numbers,
                                  torch::Tensor positions);
    torch::Tensor forward(torch::Tensor atomic_numbers,
                          torch::Tensor positions);
    std::pair<torch::Tensor, torch::Tensor>
    energy_and_forces(torch::Tensor atomic_numbers, torch::Tensor positions);
};
TORCH_MODULE(SchNet);

struct SchNetBatch {
    torch::Tensor atomic_numbers;  // [N] int64
    torch::Tensor positions;       // [N,3] float
    torch::Tensor energy;          // scalar float
    torch::Tensor forces;          // [N,3] float, optional/undefined allowed
};

torch::Tensor schnet_loss(SchNet& model, const SchNetBatch& batch,
                          double rho_energy = 0.01,
                          bool use_forces = true);

float schnet_train_epoch(SchNet& model,
                         torch::optim::Adam& optimizer,
                         const std::vector<SchNetBatch>& batches,
                         bool use_forces = true);

float schnet_evaluate_mae(SchNet& model,
                          const std::vector<SchNetBatch>& batches);

SchNet make_schnet_qm9();
SchNet make_schnet_md17();

}  // namespace quantum
}  // namespace models
}  // namespace dm
