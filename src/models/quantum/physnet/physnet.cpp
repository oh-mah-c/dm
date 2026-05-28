// ─────────────────────────────────────────────────────────────────────────────
// PhysNet implementation
// Unke & Meuwly, arXiv:1902.08408v2.
// ─────────────────────────────────────────────────────────────────────────────

#include "models/quantum/physnet/physnet.h"

#include <torch/torch.h>
#include <cmath>
#include <stdexcept>

namespace dm {
namespace models {
namespace quantum {

namespace {

constexpr double kLog2 = 0.69314718055994530942;

void validate_inputs(const torch::Tensor& z, const torch::Tensor& r) {
    if (!z.defined() || !r.defined())
        throw std::invalid_argument("PhysNet inputs must be defined");
    if (z.dim() != 1)
        throw std::invalid_argument("PhysNet atomic_numbers must have shape [N]");
    if (r.dim() != 2 || r.size(1) != 3)
        throw std::invalid_argument("PhysNet positions must have shape [N,3]");
    if (z.size(0) != r.size(0))
        throw std::invalid_argument("PhysNet atomic_numbers and positions disagree on N");
}

void init_linear(torch::nn::Linear& layer, bool zero = false) {
    if (zero) {
        torch::nn::init::constant_(layer->weight, 0.0);
    } else {
        torch::nn::init::orthogonal_(layer->weight);
    }
    torch::nn::init::constant_(layer->bias, 0.0);
}

torch::Tensor pairwise_distances(torch::Tensor positions) {
    auto diff = positions.unsqueeze(1) - positions.unsqueeze(0);
    return diff.pow(2).sum(-1).add(1e-16).sqrt();
}

}  // namespace

PhysNetConfig PhysNetConfig::default_config() {
    return PhysNetConfig{};
}

PhysNetConfig PhysNetConfig::energy_only() {
    PhysNetConfig c;
    c.predict_charges = false;
    c.long_range = false;
    return c;
}

torch::Tensor physnet_shifted_softplus(torch::Tensor x) {
    return torch::softplus(x) - kLog2;
}

torch::Tensor physnet_cutoff(torch::Tensor r, double cutoff) {
    auto x = r / cutoff;
    auto poly = 1.0 - 6.0 * x.pow(5) + 15.0 * x.pow(4) - 10.0 * x.pow(3);
    return torch::where(r < cutoff, poly, torch::zeros_like(r));
}

PhysNetResidualBlockImpl::PhysNetResidualBlockImpl(int64_t dim) {
    dense1 = register_module("dense1", torch::nn::Linear(dim, dim));
    dense2 = register_module("dense2", torch::nn::Linear(dim, dim));
    init_linear(dense1);
    init_linear(dense2);
}

torch::Tensor PhysNetResidualBlockImpl::forward(torch::Tensor x) {
    auto y = dense1(physnet_shifted_softplus(x));
    y = dense2(physnet_shifted_softplus(y));
    return x + y;
}

PhysNetRBFImpl::PhysNetRBFImpl(int64_t n_rbf_, double cutoff_)
    : n_rbf(n_rbf_), cutoff(cutoff_) {
    if (n_rbf <= 1 || cutoff <= 0.0)
        throw std::invalid_argument("Invalid PhysNet RBF configuration");
    auto start = std::exp(-cutoff);
    mu = register_buffer("mu",
        torch::linspace(start, 1.0, n_rbf, torch::kFloat32));
    double beta0 = std::pow(2.0 / n_rbf * (1.0 - std::exp(-cutoff)), -2.0);
    beta = register_buffer("beta",
        torch::full({n_rbf}, beta0, torch::kFloat32));
}

torch::Tensor PhysNetRBFImpl::forward(torch::Tensor distances) {
    auto exp_r = torch::exp(-distances).unsqueeze(-1);
    auto vals = torch::exp(-beta.to(distances.device()) *
                           (exp_r - mu.to(distances.device())).pow(2));
    return physnet_cutoff(distances, cutoff).unsqueeze(-1) * vals;
}

PhysNetInteractionBlockImpl::PhysNetInteractionBlockImpl(
    int64_t feature_dim_, int64_t n_rbf_, int64_t n_residual, double cutoff_)
    : feature_dim(feature_dim_), n_rbf(n_rbf_), cutoff(cutoff_) {
    wi = register_module("wi", torch::nn::Linear(feature_dim, feature_dim));
    wj = register_module("wj", torch::nn::Linear(feature_dim, feature_dim));
    g = register_module("g", torch::nn::Linear(n_rbf, feature_dim));
    residuals = register_module("residuals", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_residual; ++i)
        residuals->push_back(PhysNetResidualBlock(feature_dim));
    out = register_module("out", torch::nn::Linear(feature_dim, feature_dim));
    gate = register_parameter("gate", torch::ones({feature_dim}));

    init_linear(wi);
    init_linear(wj);
    init_linear(g, true);
    init_linear(out);
}

torch::Tensor PhysNetInteractionBlockImpl::forward(torch::Tensor x,
                                                   torch::Tensor rbf_vals,
                                                   torch::Tensor distances) {
    auto central = physnet_shifted_softplus(wi(physnet_shifted_softplus(x)));
    auto neigh = physnet_shifted_softplus(wj(physnet_shifted_softplus(x)));
    auto mask = g(rbf_vals);  // [N,N,F]
    auto valid = (distances > 1e-8).to(x.dtype()).unsqueeze(-1);
    auto msg = (mask * neigh.unsqueeze(0) * valid).sum(1);
    auto v = central + msg;
    for (size_t i = 0; i < residuals->size(); ++i) {
        auto& block = residuals->at<PhysNetResidualBlockImpl>(i);
        v = block.forward(v);
    }
    return gate.unsqueeze(0) * x + out(physnet_shifted_softplus(v));
}

PhysNetOutputBlockImpl::PhysNetOutputBlockImpl(int64_t feature_dim,
                                               int64_t n_residual,
                                               int64_t n_out) {
    residuals = register_module("residuals", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_residual; ++i)
        residuals->push_back(PhysNetResidualBlock(feature_dim));
    out = register_module("out", torch::nn::Linear(feature_dim, n_out));
    init_linear(out, true);
}

torch::Tensor PhysNetOutputBlockImpl::forward(torch::Tensor x) {
    for (size_t i = 0; i < residuals->size(); ++i) {
        auto& block = residuals->at<PhysNetResidualBlockImpl>(i);
        x = block.forward(x);
    }
    return out(physnet_shifted_softplus(x));
}

PhysNetModuleBlockImpl::PhysNetModuleBlockImpl(const PhysNetConfig& cfg,
                                               int64_t n_out) {
    interaction = register_module("interaction",
        PhysNetInteractionBlock(cfg.feature_dim, cfg.n_rbf,
                                cfg.n_interaction_residual, cfg.cutoff));
    atomic_residuals = register_module("atomic_residuals", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_atomic_residual; ++i)
        atomic_residuals->push_back(PhysNetResidualBlock(cfg.feature_dim));
    output = register_module("output",
        PhysNetOutputBlock(cfg.feature_dim, cfg.n_output_residual, n_out));
}

std::pair<torch::Tensor, torch::Tensor>
PhysNetModuleBlockImpl::forward(torch::Tensor x, torch::Tensor rbf_vals,
                                torch::Tensor distances) {
    x = interaction->forward(x, rbf_vals, distances);
    for (size_t i = 0; i < atomic_residuals->size(); ++i) {
        auto& block = atomic_residuals->at<PhysNetResidualBlockImpl>(i);
        x = block.forward(x);
    }
    return {x, output->forward(x)};
}

PhysNetImpl::PhysNetImpl(PhysNetConfig cfg_) : cfg(cfg_) {
    n_out = cfg.predict_charges ? 2 : 1;
    if (cfg.feature_dim <= 0 || cfg.n_modules <= 0 || cfg.n_rbf <= 1)
        throw std::invalid_argument("Invalid PhysNet configuration");

    embedding = register_module("embedding",
        torch::nn::Embedding(cfg.max_z + 1, cfg.feature_dim));
    rbf = register_module("rbf", PhysNetRBF(cfg.n_rbf, cfg.cutoff));
    modules_list = register_module("modules_list", torch::nn::ModuleList());
    for (int64_t i = 0; i < cfg.n_modules; ++i)
        modules_list->push_back(PhysNetModuleBlock(cfg, n_out));

    scale = register_module("scale",
        torch::nn::Embedding(cfg.max_z + 1, n_out));
    shift = register_module("shift",
        torch::nn::Embedding(cfg.max_z + 1, n_out));

    torch::nn::init::uniform_(embedding->weight, -std::sqrt(3.0), std::sqrt(3.0));
    torch::nn::init::constant_(scale->weight, 1.0);
    torch::nn::init::constant_(shift->weight, 0.0);
}

torch::Tensor PhysNetImpl::module_outputs(torch::Tensor atomic_numbers,
                                          torch::Tensor positions) {
    validate_inputs(atomic_numbers, positions);
    auto z = atomic_numbers.to(torch::kLong).to(positions.device());
    auto x = embedding(z);
    auto distances = pairwise_distances(positions);
    auto rbf_vals = rbf->forward(distances);

    std::vector<torch::Tensor> outs;
    for (size_t i = 0; i < modules_list->size(); ++i) {
        auto& mod = modules_list->at<PhysNetModuleBlockImpl>(i);
        auto result = mod.forward(x, rbf_vals, distances);
        x = result.first;
        outs.push_back(result.second);
    }
    return torch::stack(outs, 0);  // [M,N,n_out]
}

torch::Tensor PhysNetImpl::atomic_properties(torch::Tensor atomic_numbers,
                                             torch::Tensor positions) {
    auto z = atomic_numbers.to(torch::kLong).to(positions.device());
    auto summed = module_outputs(atomic_numbers, positions).sum(0);
    return summed * scale(z) + shift(z);
}

torch::Tensor PhysNetImpl::corrected_charges(torch::Tensor atomic_numbers,
                                             torch::Tensor positions,
                                             double total_charge) {
    if (!cfg.predict_charges)
        return torch::zeros({atomic_numbers.size(0)}, positions.options());
    auto q = atomic_properties(atomic_numbers, positions).select(1, 1);
    auto correction = (q.sum() - total_charge) / q.size(0);
    return q - correction;
}

torch::Tensor PhysNetImpl::energy(torch::Tensor atomic_numbers,
                                  torch::Tensor positions,
                                  double total_charge) {
    auto props = atomic_properties(atomic_numbers, positions);
    auto e = props.select(1, 0).sum();
    if (!cfg.predict_charges || !cfg.long_range)
        return e;

    auto q = corrected_charges(atomic_numbers, positions, total_charge);
    auto distances = pairwise_distances(positions);
    auto phi2 = physnet_cutoff(2.0 * distances, cfg.cutoff);
    auto chi = phi2 / (distances.pow(2) + 1.0).sqrt() +
               (1.0 - phi2) / distances.clamp_min(1e-8);
    auto qq = q.unsqueeze(1) * q.unsqueeze(0);
    auto upper = torch::triu(torch::ones_like(distances), 1);
    return e + cfg.coulomb_const * (qq * chi * upper).sum();
}

torch::Tensor PhysNetImpl::dipole(torch::Tensor atomic_numbers,
                                  torch::Tensor positions,
                                  double total_charge) {
    auto q = corrected_charges(atomic_numbers, positions, total_charge);
    return (q.unsqueeze(-1) * positions).sum(0);
}

std::pair<torch::Tensor, torch::Tensor>
PhysNetImpl::energy_and_forces(torch::Tensor atomic_numbers,
                               torch::Tensor positions,
                               double total_charge) {
    torch::AutoGradMode grad_mode(true);
    auto pos = positions.detach().clone().set_requires_grad(true);
    auto e = energy(atomic_numbers, pos, total_charge);
    auto grad = torch::autograd::grad({e}, {pos}, {torch::ones_like(e)},
                                      true, true)[0];
    return {e, -grad};
}

torch::Tensor PhysNetImpl::forward(torch::Tensor atomic_numbers,
                                   torch::Tensor positions) {
    return energy(atomic_numbers, positions, 0.0);
}

torch::Tensor physnet_loss(PhysNet& model, const PhysNetBatch& batch,
                           bool use_forces, bool use_dipole) {
    auto ef = model->energy_and_forces(batch.atomic_numbers, batch.positions,
                                       batch.total_charge);
    auto loss = model->cfg.w_energy *
        (ef.first - batch.energy.to(ef.first.device())).pow(2);

    if (use_forces && batch.forces.defined() && batch.forces.numel() > 0) {
        auto target = batch.forces.to(ef.second.device());
        loss = loss + model->cfg.w_force * (ef.second - target).pow(2).mean();
    }
    if (model->cfg.predict_charges) {
        auto q = model->corrected_charges(batch.atomic_numbers, batch.positions,
                                          batch.total_charge);
        auto charge_err = q.sum() - batch.total_charge;
        loss = loss + model->cfg.w_charge * charge_err.pow(2);
        if (use_dipole && batch.dipole.defined() && batch.dipole.numel() == 3) {
            auto p = model->dipole(batch.atomic_numbers, batch.positions,
                                   batch.total_charge);
            loss = loss + model->cfg.w_dipole *
                (p - batch.dipole.to(p.device())).pow(2).mean();
        }
    }
    return loss;
}

float physnet_train_epoch(PhysNet& model,
                          torch::optim::Adam& optimizer,
                          const std::vector<PhysNetBatch>& batches,
                          bool use_forces,
                          bool use_dipole) {
    model->train();
    double total = 0.0;
    for (const auto& batch : batches) {
        optimizer.zero_grad();
        auto loss = physnet_loss(model, batch, use_forces, use_dipole);
        loss.backward();
        optimizer.step();
        total += loss.detach().item<double>();
    }
    return batches.empty() ? 0.0f : static_cast<float>(total / batches.size());
}

float physnet_evaluate_mae(PhysNet& model,
                           const std::vector<PhysNetBatch>& batches) {
    model->eval();
    torch::NoGradGuard ng;
    double total = 0.0;
    for (const auto& batch : batches) {
        auto e = model->energy(batch.atomic_numbers, batch.positions,
                               batch.total_charge);
        total += (e - batch.energy.to(e.device())).abs().item<double>();
    }
    return batches.empty() ? 0.0f : static_cast<float>(total / batches.size());
}

PhysNet make_physnet() {
    return PhysNet(PhysNetConfig::default_config());
}

PhysNet make_physnet_energy_only() {
    return PhysNet(PhysNetConfig::energy_only());
}

}  // namespace quantum
}  // namespace models
}  // namespace dm
