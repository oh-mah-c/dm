// ─────────────────────────────────────────────────────────────────────────────
// SchNet implementation
// Schutt et al., "SchNet: A continuous-filter convolutional neural network
// for modeling quantum interactions", NeurIPS 2017.
// ─────────────────────────────────────────────────────────────────────────────

#include "models/quantum/schnet/schnet.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace dm {
namespace models {
namespace quantum {

namespace {

constexpr double kLog2 = 0.69314718055994530942;

void validate_molecule_inputs(const torch::Tensor& z, const torch::Tensor& r) {
    if (!z.defined() || !r.defined()) {
        throw std::invalid_argument("SchNet input tensors must be defined");
    }
    if (z.dim() != 1) {
        throw std::invalid_argument("SchNet atomic_numbers must have shape [N]");
    }
    if (r.dim() != 2 || r.size(1) != 3) {
        throw std::invalid_argument("SchNet positions must have shape [N,3]");
    }
    if (z.size(0) != r.size(0)) {
        throw std::invalid_argument("SchNet atomic_numbers and positions disagree on N");
    }
}

void init_linear(torch::nn::Linear& linear) {
    torch::nn::init::xavier_uniform_(linear->weight);
    torch::nn::init::constant_(linear->bias, 0.0);
}

}  // namespace

SchNetConfig SchNetConfig::qm9() {
    SchNetConfig c;
    c.hidden_dim = 64;
    c.n_interactions = 3;
    c.cutoff = 30.0;
    c.rbf_step = 0.1;
    c.n_gaussians = 301;
    c.gamma = 10.0;
    c.rho_energy = 0.01;
    return c;
}

SchNetConfig SchNetConfig::md17() {
    return qm9();
}

torch::Tensor schnet_shifted_softplus(torch::Tensor x) {
    return torch::softplus(x) - kLog2;
}

SchNetGaussianSmearingImpl::SchNetGaussianSmearingImpl(
    int64_t n_gaussians_, double start, double stop, double gamma_)
    : n_gaussians(n_gaussians_), gamma(gamma_) {
    if (n_gaussians <= 1) {
        throw std::invalid_argument("SchNet n_gaussians must be greater than one");
    }
    offsets = register_buffer("offsets",
        torch::linspace(start, stop, n_gaussians, torch::kFloat32));
}

torch::Tensor SchNetGaussianSmearingImpl::forward(torch::Tensor distances) {
    auto d = distances.unsqueeze(-1) - offsets.to(distances.device());
    return torch::exp(-gamma * d.pow(2));
}

SchNetFilterNetworkImpl::SchNetFilterNetworkImpl(int64_t n_gaussians,
                                                 int64_t hidden_dim) {
    dense1 = register_module("dense1", torch::nn::Linear(n_gaussians, hidden_dim));
    dense2 = register_module("dense2", torch::nn::Linear(hidden_dim, hidden_dim));
    init_linear(dense1);
    init_linear(dense2);
}

torch::Tensor SchNetFilterNetworkImpl::forward(torch::Tensor rbf) {
    auto x = dense1(rbf);
    x = schnet_shifted_softplus(x);
    x = dense2(x);
    return x;
}

SchNetCFConvImpl::SchNetCFConvImpl(int64_t hidden_dim_,
                                   int64_t n_gaussians,
                                   double cutoff,
                                   double gamma)
    : hidden_dim(hidden_dim_) {
    rbf = register_module("rbf",
        SchNetGaussianSmearing(n_gaussians, 0.0, cutoff, gamma));
    filter_net = register_module("filter_net",
        SchNetFilterNetwork(n_gaussians, hidden_dim));
}

torch::Tensor SchNetCFConvImpl::forward(torch::Tensor x,
                                        torch::Tensor positions) {
    // x: [N,F], positions: [N,3]
    auto diff = positions.unsqueeze(1) - positions.unsqueeze(0);  // [N,N,3]
    auto distances = diff.pow(2).sum(-1).add(1e-16).sqrt();       // [N,N]
    auto filters = filter_net->forward(rbf->forward(distances));  // [N,N,F]

    // cfconv_i = sum_j x_j ◦ W(d_ij)
    auto messages = x.unsqueeze(0) * filters;                    // [N_i,N_j,F]
    return messages.sum(1);                                      // [N,F]
}

SchNetInteractionBlockImpl::SchNetInteractionBlockImpl(
    int64_t hidden_dim_, int64_t n_gaussians, double cutoff, double gamma)
    : hidden_dim(hidden_dim_) {
    atom_in = register_module("atom_in",
        torch::nn::Linear(hidden_dim, hidden_dim));
    cfconv = register_module("cfconv",
        SchNetCFConv(hidden_dim, n_gaussians, cutoff, gamma));
    atom_hidden = register_module("atom_hidden",
        torch::nn::Linear(hidden_dim, hidden_dim));
    atom_out = register_module("atom_out",
        torch::nn::Linear(hidden_dim, hidden_dim));

    init_linear(atom_in);
    init_linear(atom_hidden);
    init_linear(atom_out);
}

torch::Tensor SchNetInteractionBlockImpl::forward(torch::Tensor x,
                                                  torch::Tensor positions) {
    auto v = atom_in(x);
    v = cfconv->forward(v, positions);
    v = atom_hidden(v);
    v = schnet_shifted_softplus(v);
    v = atom_out(v);
    return x + v;
}

SchNetImpl::SchNetImpl(SchNetConfig cfg_) : cfg(cfg_) {
    if (cfg.max_z <= 0 || cfg.hidden_dim <= 0 || cfg.n_interactions <= 0) {
        throw std::invalid_argument("Invalid SchNet configuration");
    }

    embedding = register_module("embedding",
        torch::nn::Embedding(cfg.max_z + 1, cfg.hidden_dim));
    interactions = register_module("interactions", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_interactions; ++i) {
        interactions->push_back(SchNetInteractionBlock(
            cfg.hidden_dim, cfg.n_gaussians, cfg.cutoff, cfg.gamma));
    }

    atomwise1 = register_module("atomwise1",
        torch::nn::Linear(cfg.hidden_dim, cfg.hidden_dim / 2));
    atomwise2 = register_module("atomwise2",
        torch::nn::Linear(cfg.hidden_dim / 2, 1));

    torch::nn::init::normal_(embedding->weight, 0.0, 1.0);
    init_linear(atomwise1);
    init_linear(atomwise2);
}

torch::Tensor SchNetImpl::atom_features(torch::Tensor atomic_numbers,
                                        torch::Tensor positions) {
    validate_molecule_inputs(atomic_numbers, positions);
    auto z = atomic_numbers.to(torch::kLong).to(positions.device());
    auto x = embedding(z);  // [N,F]
    for (size_t i = 0; i < interactions->size(); ++i) {
        auto& block = interactions->at<SchNetInteractionBlockImpl>(i);
        x = block.forward(x, positions);
    }
    return x;
}

torch::Tensor SchNetImpl::atomic_energies(torch::Tensor atomic_numbers,
                                          torch::Tensor positions) {
    auto x = atom_features(atomic_numbers, positions);
    x = atomwise1(x);
    x = schnet_shifted_softplus(x);
    return atomwise2(x).squeeze(-1);  // [N]
}

torch::Tensor SchNetImpl::forward(torch::Tensor atomic_numbers,
                                  torch::Tensor positions) {
    return atomic_energies(atomic_numbers, positions).sum();
}

std::pair<torch::Tensor, torch::Tensor>
SchNetImpl::energy_and_forces(torch::Tensor atomic_numbers,
                              torch::Tensor positions) {
    torch::AutoGradMode grad_mode(true);
    auto pos = positions.detach().clone().set_requires_grad(true);
    auto energy = forward(atomic_numbers, pos);
    auto grad = torch::autograd::grad(
        {energy}, {pos}, {torch::ones_like(energy)},
        /*retain_graph=*/true,
        /*create_graph=*/true)[0];
    return {energy, -grad};
}

torch::Tensor schnet_loss(SchNet& model, const SchNetBatch& batch,
                          double rho_energy, bool use_forces) {
    auto pred = model->energy_and_forces(batch.atomic_numbers, batch.positions);
    auto energy_loss = (pred.first - batch.energy.to(pred.first.device())).pow(2);
    auto loss = rho_energy * energy_loss;
    if (use_forces && batch.forces.defined() && batch.forces.numel() > 0) {
        auto target_f = batch.forces.to(pred.second.device());
        loss = loss + (pred.second - target_f).pow(2).mean();
    }
    return loss;
}

float schnet_train_epoch(SchNet& model,
                         torch::optim::Adam& optimizer,
                         const std::vector<SchNetBatch>& batches,
                         bool use_forces) {
    model->train();
    double total = 0.0;
    for (const auto& batch : batches) {
        optimizer.zero_grad();
        auto loss = schnet_loss(model, batch, model->cfg.rho_energy, use_forces);
        loss.backward();
        optimizer.step();
        total += loss.detach().item<double>();
    }
    return batches.empty() ? 0.0f : static_cast<float>(total / batches.size());
}

float schnet_evaluate_mae(SchNet& model,
                          const std::vector<SchNetBatch>& batches) {
    model->eval();
    torch::NoGradGuard ng;
    double total = 0.0;
    for (const auto& batch : batches) {
        auto pred = model->forward(batch.atomic_numbers, batch.positions);
        total += (pred - batch.energy.to(pred.device())).abs().item<double>();
    }
    return batches.empty() ? 0.0f : static_cast<float>(total / batches.size());
}

SchNet make_schnet_qm9() {
    return SchNet(SchNetConfig::qm9());
}

SchNet make_schnet_md17() {
    return SchNet(SchNetConfig::md17());
}

}  // namespace quantum
}  // namespace models
}  // namespace dm
