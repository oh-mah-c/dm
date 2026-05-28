// -----------------------------------------------------------------------------
// Neural Quantum States / RBM VMC implementation
// -----------------------------------------------------------------------------

#include "models/quantum/nqs/nqs.h"

#include <torch/torch.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace dm {
namespace models {
namespace quantum {

namespace {

constexpr double kLog2 = 0.69314718055994530942;

void validate_spin_tensor(const torch::Tensor& spins, int64_t n_visible) {
    if (!spins.defined())
        throw std::invalid_argument("NQS spin tensor must be defined");
    if (spins.dim() != 1 && spins.dim() != 2)
        throw std::invalid_argument("NQS spins must have shape [N] or [B,N]");
    const int64_t last = spins.size(spins.dim() - 1);
    if (last != n_visible)
        throw std::invalid_argument("NQS spin tensor has wrong visible dimension");
}

torch::Tensor stable_log_2cosh(torch::Tensor x) {
    return torch::logaddexp(x, -x);
}

torch::Tensor as_model_spins(torch::Tensor spins, const torch::Tensor& ref) {
    return spins.to(ref.device()).to(ref.dtype());
}

}  // namespace

NQSRBMConfig NQSRBMConfig::tiny(int64_t n_visible, int64_t n_hidden) {
    NQSRBMConfig cfg;
    cfg.n_visible = n_visible;
    cfg.n_hidden = n_hidden;
    cfg.init_std = 0.01;
    return cfg;
}

NQSRBMImpl::NQSRBMImpl(NQSRBMConfig cfg_) : cfg(cfg_) {
    if (cfg.n_visible <= 0 || cfg.n_hidden <= 0)
        throw std::invalid_argument("NQSRBM requires positive visible/hidden sizes");
    if (cfg.init_std < 0.0)
        throw std::invalid_argument("NQSRBM init_std must be non-negative");

    visible_bias = register_parameter("visible_bias",
        torch::zeros({cfg.n_visible}, torch::kFloat32));
    hidden_bias = register_parameter("hidden_bias",
        torch::zeros({cfg.n_hidden}, torch::kFloat32));
    weight = register_parameter("weight",
        cfg.init_std * torch::randn({cfg.n_visible, cfg.n_hidden}, torch::kFloat32));
}

torch::Tensor NQSRBMImpl::log_amplitude(torch::Tensor spins) {
    validate_spin_tensor(spins, cfg.n_visible);
    const bool scalar_input = spins.dim() == 1;
    auto x = scalar_input ? spins.unsqueeze(0) : spins;
    x = as_model_spins(x, weight);

    auto theta = torch::matmul(x, weight) + hidden_bias;
    auto visible = torch::matmul(x, visible_bias);
    auto hidden = stable_log_2cosh(theta).sum(-1);
    auto out = visible + hidden;
    return scalar_input ? out.squeeze(0) : out;
}

torch::Tensor NQSRBMImpl::amplitude(torch::Tensor spins) {
    return torch::exp(log_amplitude(spins));
}

torch::Tensor NQSRBMImpl::log_prob(torch::Tensor spins) {
    return 2.0 * log_amplitude(spins);
}

torch::Tensor NQSRBMImpl::forward(torch::Tensor spins) {
    return log_amplitude(spins);
}

torch::Tensor nqs_all_spin_configs(int64_t n_visible) {
    if (n_visible <= 0 || n_visible > 30)
        throw std::invalid_argument("nqs_all_spin_configs supports 1..30 spins");
    const int64_t n_states = int64_t{1} << n_visible;
    std::vector<float> data(static_cast<size_t>(n_states * n_visible));
    for (int64_t state = 0; state < n_states; ++state) {
        for (int64_t i = 0; i < n_visible; ++i) {
            const bool bit = ((state >> i) & 1) != 0;
            data[static_cast<size_t>(state * n_visible + i)] = bit ? 1.0f : -1.0f;
        }
    }
    return torch::from_blob(data.data(), {n_states, n_visible}, torch::kFloat32).clone();
}

torch::Tensor nqs_random_spin_config(int64_t n_visible) {
    if (n_visible <= 0)
        throw std::invalid_argument("nqs_random_spin_config requires positive n_visible");
    auto bits = torch::randint(0, 2, {n_visible}, torch::kFloat32);
    return bits * 2.0 - 1.0;
}

torch::Tensor nqs_flip_spin(torch::Tensor spins, int64_t index) {
    if (!spins.defined() || spins.dim() != 1)
        throw std::invalid_argument("nqs_flip_spin expects shape [N]");
    if (index < 0 || index >= spins.size(0))
        throw std::out_of_range("nqs_flip_spin index out of range");
    auto out = spins.clone();
    out.index_put_({index}, -out.index({index}));
    return out;
}

torch::Tensor nqs_tfim_local_energy(NQSRBM& model,
                                    torch::Tensor spins,
                                    const NQSTFIM& hamiltonian) {
    if (!model)
        throw std::invalid_argument("nqs_tfim_local_energy requires a model");
    validate_spin_tensor(spins, model->cfg.n_visible);
    if (spins.dim() != 1)
        throw std::invalid_argument("nqs_tfim_local_energy expects one spin config [N]");

    auto s = as_model_spins(spins, model->weight);
    const int64_t n = s.size(0);
    auto diag = torch::zeros({}, s.options());
    const int64_t bonds = hamiltonian.periodic ? n : n - 1;
    for (int64_t i = 0; i < bonds; ++i) {
        const int64_t j = (i + 1) % n;
        diag = diag - hamiltonian.j * s.index({i}) * s.index({j});
    }

    auto base_log = model->log_amplitude(s);
    auto offdiag = torch::zeros({}, s.options());
    for (int64_t i = 0; i < n; ++i) {
        auto flipped = nqs_flip_spin(s, i);
        offdiag = offdiag + torch::exp(model->log_amplitude(flipped) - base_log);
    }
    return diag - hamiltonian.h * offdiag;
}

torch::Tensor nqs_exact_energy_tfim(NQSRBM& model,
                                    const NQSTFIM& hamiltonian) {
    if (!model)
        throw std::invalid_argument("nqs_exact_energy_tfim requires a model");
    auto states = nqs_all_spin_configs(model->cfg.n_visible).to(model->weight.device());
    auto logp = model->log_prob(states);
    auto max_logp = logp.max();
    auto weights = torch::exp(logp - max_logp);

    std::vector<torch::Tensor> locals;
    locals.reserve(static_cast<size_t>(states.size(0)));
    for (int64_t i = 0; i < states.size(0); ++i)
        locals.push_back(nqs_tfim_local_energy(model, states.index({i}), hamiltonian));
    auto local_energy = torch::stack(locals).to(weights.dtype());
    return (weights * local_energy).sum() / weights.sum();
}

NQSMetropolisSampler::NQSMetropolisSampler(int64_t n_visible, uint64_t seed)
    : spins(nqs_random_spin_config(n_visible)), rng(static_cast<uint32_t>(seed)) {}

NQSMetropolisSampler::NQSMetropolisSampler(torch::Tensor initial_spins,
                                           uint64_t seed)
    : spins(initial_spins.clone().to(torch::kFloat32)),
      rng(static_cast<uint32_t>(seed)) {
    if (!spins.defined() || spins.dim() != 1)
        throw std::invalid_argument("NQSMetropolisSampler initial spins must be [N]");
}

double NQSMetropolisSampler::step(NQSRBM& model, int64_t n_steps) {
    if (!model)
        throw std::invalid_argument("NQSMetropolisSampler::step requires a model");
    if (n_steps <= 0)
        throw std::invalid_argument("NQSMetropolisSampler::step requires positive n_steps");
    validate_spin_tensor(spins, model->cfg.n_visible);

    torch::NoGradGuard no_grad;
    std::uniform_int_distribution<int64_t> pick(0, spins.size(0) - 1);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    int64_t accepted = 0;
    for (int64_t t = 0; t < n_steps; ++t) {
        const int64_t idx = pick(rng);
        auto proposal = nqs_flip_spin(spins, idx);
        auto log_ratio = 2.0 * (model->log_amplitude(proposal) -
                                model->log_amplitude(spins));
        const double lr = log_ratio.item<double>();
        const double accept_prob = lr >= 0.0 ? 1.0 : std::exp(lr);
        if (uniform(rng) <= accept_prob) {
            spins = proposal.detach().clone();
            ++accepted;
        }
    }
    return static_cast<double>(accepted) / static_cast<double>(n_steps);
}

torch::Tensor NQSMetropolisSampler::sample_batch(NQSRBM& model,
                                                 int64_t batch_size,
                                                 int64_t decorrelation_steps) {
    if (batch_size <= 0 || decorrelation_steps <= 0)
        throw std::invalid_argument("NQS sample_batch requires positive sizes");
    std::vector<torch::Tensor> samples;
    samples.reserve(static_cast<size_t>(batch_size));
    for (int64_t i = 0; i < batch_size; ++i) {
        step(model, decorrelation_steps);
        samples.push_back(spins.clone());
    }
    return torch::stack(samples, 0);
}

float nqs_train_exact_step(NQSRBM& model,
                           torch::optim::Optimizer& optimizer,
                           const NQSTFIM& hamiltonian) {
    if (!model)
        throw std::invalid_argument("nqs_train_exact_step requires a model");
    model->train();
    optimizer.zero_grad();
    auto energy = nqs_exact_energy_tfim(model, hamiltonian);
    energy.backward();
    optimizer.step();
    return energy.item<float>();
}

NQSRBM make_nqs_rbm(int64_t n_visible, int64_t n_hidden) {
    return NQSRBM(NQSRBMConfig::tiny(n_visible, n_hidden));
}

}  // namespace quantum
}  // namespace models
}  // namespace dm
