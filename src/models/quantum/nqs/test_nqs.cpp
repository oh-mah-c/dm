// -----------------------------------------------------------------------------
// Neural Quantum States unit tests
// -----------------------------------------------------------------------------

#include "models/quantum/nqs/nqs.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>

using namespace dm::models::quantum;

static int passed = 0;
static int failed = 0;

#define NQS_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << (msg) << "\n"; \
            failed++; \
        } else { \
            std::cout << "  [PASS] " << (msg) << "\n"; \
            passed++; \
        } \
    } while (0)

static NQSRBM make_zero_rbm(int64_t n_visible, int64_t n_hidden) {
    NQSRBMConfig cfg = NQSRBMConfig::tiny(n_visible, n_hidden);
    cfg.init_std = 0.0;
    auto model = NQSRBM(cfg);
    torch::NoGradGuard no_grad;
    model->visible_bias.zero_();
    model->hidden_bias.zero_();
    model->weight.zero_();
    return model;
}

static void test_config_and_forward() {
    auto cfg = NQSRBMConfig::tiny(4, 3);
    NQS_CHECK(cfg.n_visible == 4, "tiny config visible size");
    NQS_CHECK(cfg.n_hidden == 3, "tiny config hidden size");

    auto model = make_zero_rbm(4, 3);
    auto s = torch::tensor({1.0f, -1.0f, 1.0f, -1.0f});
    auto log_amp = model->log_amplitude(s).item<float>();
    NQS_CHECK(std::abs(log_amp - 3.0f * std::log(2.0f)) < 1e-5f,
              "zero RBM log amplitude equals H log 2");
    NQS_CHECK(std::isfinite(model->amplitude(s).item<float>()),
              "RBM amplitude finite");
}

static void test_all_spin_configs() {
    auto states = nqs_all_spin_configs(3);
    NQS_CHECK(states.sizes() == torch::IntArrayRef({8, 3}),
              "all spin configs shape [2^N,N]");
    auto abs_vals = states.abs();
    NQS_CHECK((abs_vals == 1.0).all().item<bool>(),
              "all spin configs contain only +/-1");
}

static void test_flip_spin() {
    auto s = torch::tensor({1.0f, -1.0f, 1.0f});
    auto f = nqs_flip_spin(s, 1);
    NQS_CHECK(f.equal(torch::tensor({1.0f, 1.0f, 1.0f})),
              "flip_spin flips exactly one spin");
    NQS_CHECK(s.equal(torch::tensor({1.0f, -1.0f, 1.0f})),
              "flip_spin leaves input unchanged");
}

static void test_tfim_local_energy_uniform() {
    auto model = make_zero_rbm(4, 2);
    NQSTFIM ham;
    ham.j = 1.0;
    ham.h = 0.5;
    ham.periodic = true;
    auto up = torch::ones({4});
    auto e = nqs_tfim_local_energy(model, up, ham).item<float>();
    NQS_CHECK(std::abs(e - (-6.0f)) < 1e-5f,
              "uniform RBM all-up TFIM local energy");
}

static void test_tfim_exact_energy_uniform() {
    auto model = make_zero_rbm(4, 2);
    NQSTFIM ham;
    ham.j = 1.0;
    ham.h = 0.5;
    ham.periodic = true;
    auto e = nqs_exact_energy_tfim(model, ham).item<float>();
    NQS_CHECK(std::abs(e - (-2.0f)) < 1e-5f,
              "uniform RBM exact TFIM energy averages to -h*N");
}

static void test_metropolis_sampler() {
    auto model = make_zero_rbm(4, 2);
    NQSMetropolisSampler sampler(torch::ones({4}), 7);
    auto rate = sampler.step(model, 16);
    NQS_CHECK(std::abs(rate - 1.0) < 1e-12,
              "uniform RBM accepts every spin flip");
    auto batch = sampler.sample_batch(model, 5, 2);
    NQS_CHECK(batch.sizes() == torch::IntArrayRef({5, 4}),
              "sampler batch shape [B,N]");
    NQS_CHECK((batch.abs() == 1.0).all().item<bool>(),
              "sampler emits +/-1 spins");
}

static void test_train_exact_step() {
    auto cfg = NQSRBMConfig::tiny(4, 4);
    cfg.init_std = 0.02;
    auto model = NQSRBM(cfg);
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-2));
    NQSTFIM ham;
    ham.j = 1.0;
    ham.h = 0.5;
    auto e0 = nqs_exact_energy_tfim(model, ham).item<float>();
    float e_train = nqs_train_exact_step(model, opt, ham);
    auto e1 = nqs_exact_energy_tfim(model, ham).item<float>();
    NQS_CHECK(std::isfinite(e0) && std::isfinite(e_train) && std::isfinite(e1),
              "exact-energy train step stays finite");
}

int main() {
    torch::manual_seed(17);
    std::cout << "=== Neural Quantum States Tests ===\n\n";

    std::cout << "-- RBM --\n";
    test_config_and_forward();
    test_all_spin_configs();
    test_flip_spin();

    std::cout << "\n-- TFIM / VMC --\n";
    test_tfim_local_energy_uniform();
    test_tfim_exact_energy_uniform();
    test_metropolis_sampler();
    test_train_exact_step();

    std::cout << "\n=== Results: " << passed << " passed, "
              << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
