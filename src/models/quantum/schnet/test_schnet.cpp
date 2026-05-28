// ─────────────────────────────────────────────────────────────────────────────
// SchNet unit tests
// ─────────────────────────────────────────────────────────────────────────────

#include "models/quantum/schnet/schnet.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>

using namespace dm::models::quantum;

static int passed = 0;
static int failed = 0;

#define SN_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << (msg) << "\n"; \
            failed++; \
        } else { \
            std::cout << "  [PASS] " << (msg) << "\n"; \
            passed++; \
        } \
    } while (0)

static void test_config() {
    auto cfg = SchNetConfig::qm9();
    SN_CHECK(cfg.hidden_dim == 64, "QM9 config: F=64");
    SN_CHECK(cfg.n_interactions == 3, "QM9 config: 3 interaction blocks");
    SN_CHECK(cfg.n_gaussians == 301, "QM9 config: RBF centers 0..30 step 0.1");
    SN_CHECK(std::abs(cfg.gamma - 10.0) < 1e-12, "QM9 config: gamma=10");
    SN_CHECK(std::abs(cfg.rho_energy - 0.01) < 1e-12, "QM9 config: rho=0.01");
}

static void test_shifted_softplus() {
    auto z = torch::zeros({4});
    auto y = schnet_shifted_softplus(z);
    SN_CHECK(y.abs().max().item<float>() < 1e-6f,
             "shifted softplus: ssp(0)=0");
}

static void test_rbf_shape() {
    SchNetGaussianSmearing rbf(301, 0.0, 30.0, 10.0);
    auto d = torch::tensor({{0.0f, 1.0f}, {2.0f, 3.0f}});
    auto y = rbf(d);
    SN_CHECK(y.sizes() == torch::IntArrayRef({2, 2, 301}),
             "RBF expansion shape [N,N,K]");
    SN_CHECK(std::isfinite(y.sum().item<float>()), "RBF values finite");
}

static void test_cfconv_shape() {
    SchNetCFConv conv(16, 31, 3.0, 10.0);
    auto x = torch::randn({5, 16});
    auto r = torch::randn({5, 3});
    auto y = conv(x, r);
    SN_CHECK(y.sizes() == torch::IntArrayRef({5, 16}),
             "cfconv output shape [N,F]");
    SN_CHECK(y.isfinite().all().item<bool>(), "cfconv output finite");
}

static SchNet make_tiny_schnet() {
    SchNetConfig cfg;
    cfg.hidden_dim = 16;
    cfg.n_interactions = 2;
    cfg.n_gaussians = 31;
    cfg.cutoff = 3.0;
    cfg.gamma = 10.0;
    return SchNet(cfg);
}

static void test_forward_energy_scalar() {
    auto model = make_tiny_schnet();
    auto z = torch::tensor({6, 1, 1}, torch::kLong);
    auto r = torch::tensor({{0.0f, 0.0f, 0.0f},
                            {0.9f, 0.0f, 0.0f},
                            {-0.9f, 0.0f, 0.0f}});
    auto e = model(z, r);
    SN_CHECK(e.dim() == 0, "energy prediction is scalar");
    SN_CHECK(std::isfinite(e.item<float>()), "energy prediction finite");
}

static void test_permutation_invariance() {
    auto model = make_tiny_schnet();
    auto z = torch::tensor({6, 1, 8, 1}, torch::kLong);
    auto r = torch::randn({4, 3});
    auto idx = torch::tensor({2, 0, 3, 1}, torch::kLong);
    auto e1 = model(z, r);
    auto e2 = model(z.index({idx}), r.index({idx}));
    SN_CHECK(std::abs((e1 - e2).item<float>()) < 1e-4f,
             "energy invariant to atom indexing");
}

static void test_translation_invariance() {
    auto model = make_tiny_schnet();
    auto z = torch::tensor({6, 1, 1}, torch::kLong);
    auto r = torch::randn({3, 3});
    auto shift = torch::tensor({{2.0f, -1.0f, 0.5f}});
    auto e1 = model(z, r);
    auto e2 = model(z, r + shift);
    SN_CHECK(std::abs((e1 - e2).item<float>()) < 1e-4f,
             "energy invariant to translation");
}

static void test_rotation_invariance_and_force_equivariance() {
    auto model = make_tiny_schnet();
    auto z = torch::tensor({8, 1, 1}, torch::kLong);
    auto r = torch::tensor({{0.0f, 0.0f, 0.0f},
                            {0.8f, 0.1f, 0.0f},
                            {-0.3f, 0.7f, 0.2f}});
    auto rot = torch::tensor({{0.0f, -1.0f, 0.0f},
                              {1.0f,  0.0f, 0.0f},
                              {0.0f,  0.0f, 1.0f}});
    auto r_rot = torch::matmul(r, rot.t());
    auto out1 = model->energy_and_forces(z, r);
    auto out2 = model->energy_and_forces(z, r_rot);
    auto f1_rot = torch::matmul(out1.second, rot.t());

    SN_CHECK(std::abs((out1.first - out2.first).item<float>()) < 1e-4f,
             "energy invariant to rotation");
    SN_CHECK((f1_rot - out2.second).abs().max().item<float>() < 1e-4f,
             "forces equivariant to rotation");
}

static void test_loss_and_train_step() {
    auto model = make_tiny_schnet();
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-3));
    SchNetBatch b;
    b.atomic_numbers = torch::tensor({1, 1}, torch::kLong);
    b.positions = torch::tensor({{0.0f, 0.0f, 0.0f},
                                 {0.74f, 0.0f, 0.0f}});
    b.energy = torch::tensor(-1.0f);
    b.forces = torch::zeros({2, 3});

    auto loss0 = schnet_loss(model, b, 0.01, true).item<float>();
    float train_loss = schnet_train_epoch(model, opt, {b}, true);
    auto loss1 = schnet_loss(model, b, 0.01, true).item<float>();
    SN_CHECK(std::isfinite(loss0) && std::isfinite(train_loss) && std::isfinite(loss1),
             "loss/train step finite");
}

int main() {
    torch::manual_seed(7);
    std::cout << "=== SchNet Tests ===\n\n";

    std::cout << "-- Config --\n";
    test_config();

    std::cout << "\n-- Core ops --\n";
    test_shifted_softplus();
    test_rbf_shape();
    test_cfconv_shape();

    std::cout << "\n-- Model --\n";
    test_forward_energy_scalar();
    test_permutation_invariance();
    test_translation_invariance();
    test_rotation_invariance_and_force_equivariance();
    test_loss_and_train_step();

    std::cout << "\n=== Results: " << passed << " passed, "
              << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
