// ─────────────────────────────────────────────────────────────────────────────
// PhysNet unit tests
// ─────────────────────────────────────────────────────────────────────────────

#include "models/quantum/physnet/physnet.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>

using namespace dm::models::quantum;

static int passed = 0;
static int failed = 0;

#define PH_CHECK(cond, msg) \
    do { \
        if (!(cond)) { std::cerr << "  [FAIL] " << (msg) << "\n"; failed++; } \
        else { std::cout << "  [PASS] " << (msg) << "\n"; passed++; } \
    } while (0)

static PhysNetConfig tiny_cfg(bool charges = true) {
    PhysNetConfig cfg;
    cfg.feature_dim = 16;
    cfg.n_rbf = 16;
    cfg.n_modules = 2;
    cfg.n_atomic_residual = 1;
    cfg.n_interaction_residual = 1;
    cfg.n_output_residual = 1;
    cfg.cutoff = 5.0;
    cfg.predict_charges = charges;
    cfg.long_range = charges;
    cfg.w_force = 1.0;
    return cfg;
}

static PhysNet tiny_model(bool charges = true) {
    return PhysNet(tiny_cfg(charges));
}

static void test_config() {
    auto cfg = PhysNetConfig::default_config();
    PH_CHECK(cfg.feature_dim == 128, "Table 1: F=128");
    PH_CHECK(cfg.n_rbf == 64, "Table 1: K=64");
    PH_CHECK(cfg.n_modules == 5, "Table 1: Nmodule=5");
    PH_CHECK(cfg.n_atomic_residual == 2, "Table 1: atomic residual blocks=2");
    PH_CHECK(cfg.n_interaction_residual == 3, "Table 1: interaction residual blocks=3");
    PH_CHECK(cfg.n_output_residual == 1, "Table 1: output residual blocks=1");
    PH_CHECK(std::abs(cfg.cutoff - 10.0) < 1e-12, "Table 1: cutoff=10 A");
}

static void test_shifted_softplus_and_cutoff() {
    auto y = physnet_shifted_softplus(torch::zeros({3}));
    PH_CHECK(y.abs().max().item<float>() < 1e-6f, "shifted softplus: sigma(0)=0");

    auto r = torch::tensor({0.0f, 2.5f, 5.0f, 6.0f});
    auto c = physnet_cutoff(r, 5.0);
    PH_CHECK(std::abs(c[0].item<float>() - 1.0f) < 1e-6f, "cutoff phi(0)=1");
    PH_CHECK(c[2].item<float>() == 0.0f && c[3].item<float>() == 0.0f,
             "cutoff zero at and beyond rcut");
}

static void test_rbf_shape() {
    PhysNetRBF rbf(16, 5.0);
    auto d = torch::rand({4, 4}) * 4.0;
    auto g = rbf(d);
    PH_CHECK(g.sizes() == torch::IntArrayRef({4, 4, 16}), "RBF shape [N,N,K]");
    PH_CHECK(g.isfinite().all().item<bool>(), "RBF finite");
}

static void test_forward_and_properties() {
    auto model = tiny_model(true);
    auto z = torch::tensor({8, 1, 1}, torch::kLong);
    auto r = torch::tensor({{0.0f, 0.0f, 0.0f},
                            {0.8f, 0.0f, 0.0f},
                            {-0.8f, 0.0f, 0.0f}});
    auto props = model->atomic_properties(z, r);
    auto e = model->energy(z, r, 0.0);
    auto q = model->corrected_charges(z, r, 0.0);
    auto p = model->dipole(z, r, 0.0);
    PH_CHECK(props.sizes() == torch::IntArrayRef({3, 2}), "atomic properties [N,2]");
    PH_CHECK(e.dim() == 0 && std::isfinite(e.item<float>()), "energy finite scalar");
    PH_CHECK(std::abs(q.sum().item<float>()) < 1e-5f, "corrected charges conserve Q");
    PH_CHECK(p.sizes() == torch::IntArrayRef({3}) && p.isfinite().all().item<bool>(),
             "dipole shape [3] and finite");
}

static void test_invariances() {
    auto model = tiny_model(true);
    auto z = torch::tensor({6, 1, 8, 1}, torch::kLong);
    auto r = torch::randn({4, 3});
    auto idx = torch::tensor({2, 0, 3, 1}, torch::kLong);
    auto e1 = model->energy(z, r, 0.0);
    auto e2 = model->energy(z.index({idx}), r.index({idx}), 0.0);
    PH_CHECK(std::abs((e1 - e2).item<float>()) < 1e-4f,
             "energy invariant to atom permutation");

    auto shift = torch::tensor({{1.0f, -2.0f, 0.5f}});
    auto e3 = model->energy(z, r + shift, 0.0);
    PH_CHECK(std::abs((e1 - e3).item<float>()) < 1e-4f,
             "energy invariant to translation");
}

static void test_force_rotation_equivariance() {
    auto model = tiny_model(false);
    auto z = torch::tensor({8, 1, 1}, torch::kLong);
    auto r = torch::tensor({{0.0f, 0.0f, 0.0f},
                            {0.8f, 0.1f, 0.0f},
                            {-0.3f, 0.7f, 0.2f}});
    auto rot = torch::tensor({{0.0f, -1.0f, 0.0f},
                              {1.0f,  0.0f, 0.0f},
                              {0.0f,  0.0f, 1.0f}});
    auto r_rot = torch::matmul(r, rot.t());
    auto a = model->energy_and_forces(z, r, 0.0);
    auto b = model->energy_and_forces(z, r_rot, 0.0);
    auto fa_rot = torch::matmul(a.second, rot.t());
    PH_CHECK(std::abs((a.first - b.first).item<float>()) < 1e-4f,
             "energy invariant to rotation");
    PH_CHECK((fa_rot - b.second).abs().max().item<float>() < 1e-4f,
             "forces equivariant to rotation");
}

static void test_loss_and_training() {
    auto model = tiny_model(true);
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-3));
    PhysNetBatch b;
    b.atomic_numbers = torch::tensor({1, 1}, torch::kLong);
    b.positions = torch::tensor({{-0.37f, 0.0f, 0.0f},
                                 { 0.37f, 0.0f, 0.0f}});
    b.energy = torch::tensor(-1.0f);
    b.forces = torch::zeros({2, 3});
    b.dipole = torch::zeros({3});
    b.total_charge = 0.0;
    auto loss0 = physnet_loss(model, b, true, true).item<float>();
    auto train_loss = physnet_train_epoch(model, opt, {b}, true, true);
    PH_CHECK(std::isfinite(loss0) && std::isfinite(train_loss),
             "loss and train step finite");
}

int main() {
    torch::manual_seed(11);
    std::cout << "=== PhysNet Tests ===\n\n";

    std::cout << "-- Config --\n";
    test_config();

    std::cout << "\n-- Core ops --\n";
    test_shifted_softplus_and_cutoff();
    test_rbf_shape();

    std::cout << "\n-- Model --\n";
    test_forward_and_properties();
    test_invariances();
    test_force_rotation_equivariance();
    test_loss_and_training();

    std::cout << "\n=== Results: " << passed << " passed, "
              << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
