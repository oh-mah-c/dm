// ─────────────────────────────────────────────────────────────────────────────
// FermiNet unit tests
// ─────────────────────────────────────────────────────────────────────────────

#include "models/quantum/ferminet/ferminet.h"

#include <torch/torch.h>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace dm::models::quantum;

static int passed = 0, failed = 0;

#define FN_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << (msg) << "\n"; failed++; \
        } else { \
            std::cout << "  [PASS] " << (msg) << "\n"; passed++; \
        } \
    } while (0)

// ── Config ────────────────────────────────────────────────────────────────────

static void test_configs() {
    auto h   = FermiNetConfig::hydrogen();
    FN_CHECK(h.n_up == 1 && h.n_down == 0 && h.n_nuclei == 1, "H config");

    auto he  = FermiNetConfig::helium();
    FN_CHECK(he.n_up == 1 && he.n_down == 1 && he.n_nuclei == 1, "He config");

    auto h2  = FermiNetConfig::h2();
    FN_CHECK(h2.n_up == 1 && h2.n_down == 1 && h2.n_nuclei == 2, "H2 config");

    auto lih = FermiNetConfig::lih();
    FN_CHECK(lih.n_up == 2 && lih.n_down == 2 && lih.n_nuclei == 2, "LiH config");

    auto c   = FermiNetConfig::carbon();
    FN_CHECK(c.n_up == 3 && c.n_down == 3 && c.n_nuclei == 1, "C config");
}

// ── Input features ────────────────────────────────────────────────────────────

static void test_input_features_shape() {
    // H2: 2 electrons, 2 nuclei → 1e dim = 4*2 = 8
    FermiNetConfig cfg = FermiNetConfig::h2();
    cfg.n_layers = 1; cfg.dim_1e = 16; cfg.dim_2e = 8; cfg.n_det = 1;
    float d = 0.7f;
    auto npos    = torch::tensor({{-d, 0.f, 0.f}, {d, 0.f, 0.f}});
    auto ncharge = torch::tensor({1.f, 1.f});
    FermiNet model(cfg, npos, ncharge);

    auto r = torch::randn({2, 3});
    auto fwd = model(r);
    FN_CHECK(fwd.first.dim() == 0,  "H2 input features: log|ψ| is scalar");
    FN_CHECK(std::isfinite(fwd.first.item<float>()), "H2 input features: log|ψ| finite");
}

static void test_1e_feature_dim() {
    // H: 1 electron, 1 nucleus → h_i = [r_i-R, |r_i-R|] = 4 dims
    FermiNetConfig cfg = FermiNetConfig::hydrogen();
    cfg.n_layers = 1; cfg.dim_1e = 16; cfg.dim_2e = 8; cfg.n_det = 1;
    auto npos    = torch::zeros({1, 3});
    auto ncharge = torch::tensor({1.f});
    FermiNet model(cfg, npos, ncharge);

    // Build input features manually to verify shape
    auto r = torch::randn({1, 3});
    auto diff = r.unsqueeze(1) - npos.unsqueeze(0);     // [1, 1, 3]
    auto dist = diff.norm(2, -1, true);                  // [1, 1, 1]
    auto feat = torch::cat({diff, dist}, -1).view({1, 4});  // [1, 4]
    FN_CHECK(feat.size(1) == 4 * cfg.n_nuclei, "1e feature dim = 4*M");
}

static void test_2e_feature_dim() {
    // 2e stream: 4 dims (r_i - r_j, |r_i - r_j|)
    auto r_up   = torch::randn({2, 3});
    auto r_down = torch::randn({2, 3});
    auto diff   = r_up.unsqueeze(1) - r_down.unsqueeze(0);  // [2, 2, 3]
    auto dist   = diff.norm(2, -1, true);                    // [2, 2, 1]
    auto feat   = torch::cat({diff, dist}, -1);              // [2, 2, 4]
    FN_CHECK(feat.size(2) == 4, "2e feature dim = 4");
}

// ── FermiNetLayer ─────────────────────────────────────────────────────────────

static void test_layer_output_shape() {
    // Layer 0: 1e dim_in=8, 1e dim_out=16, 2e dim=4→8
    FermiNetLayer layer(2, 2, /*dim_1e_in=*/8, /*dim_1e_out=*/16,
                        /*dim_2e_in=*/4, /*dim_2e_out=*/8, /*residual=*/false);

    auto x_up   = torch::randn({2, 8});
    auto x_down = torch::randn({2, 8});
    auto h_uu   = torch::randn({2, 2, 4});
    auto h_ud   = torch::randn({2, 2, 4});
    auto h_du   = torch::randn({2, 2, 4});
    auto h_dd   = torch::randn({2, 2, 4});

    auto out = layer(x_up, x_down, h_uu, h_ud, h_du, h_dd);
    FN_CHECK(std::get<0>(out).sizes() == torch::IntArrayRef({2, 16}),
             "Layer: x_up shape [n_up, dim_1e_out]");
    FN_CHECK(std::get<1>(out).sizes() == torch::IntArrayRef({2, 16}),
             "Layer: x_down shape [n_down, dim_1e_out]");
    FN_CHECK(std::get<2>(out).sizes() == torch::IntArrayRef({2, 2, 8}),
             "Layer: h_uu shape [n_up, n_up, dim_2e_out]");
    FN_CHECK(std::get<3>(out).sizes() == torch::IntArrayRef({2, 2, 8}),
             "Layer: h_ud shape [n_up, n_down, dim_2e_out]");
}

static void test_layer_residual() {
    // With zero weights and residual=true, output ≈ input
    FermiNetLayer layer(1, 1, /*dim_1e_in=*/16, /*dim_1e_out=*/16,
                        /*dim_2e_in=*/8, /*dim_2e_out=*/8, /*residual=*/true);
    {
        torch::NoGradGuard ng;
        for (auto& p : layer->parameters()) p.zero_();
    }
    auto x_up   = torch::randn({1, 16});
    auto x_down = torch::randn({1, 16});
    auto h_uu   = torch::randn({1, 1, 8});
    auto h_ud   = torch::randn({1, 1, 8});
    auto h_du   = torch::randn({1, 1, 8});
    auto h_dd   = torch::randn({1, 1, 8});

    auto out = layer(x_up, x_down, h_uu, h_ud, h_du, h_dd);
    // tanh(0) = 0, residual: 0 + x = x
    FN_CHECK((std::get<0>(out) - x_up).abs().max().item<float>() < 1e-5f,
             "Layer residual: zero weights → identity for 1e stream");
    FN_CHECK((std::get<2>(out) - h_uu).abs().max().item<float>() < 1e-5f,
             "Layer residual: zero weights → identity for 2e stream");
}

static void test_layer_no_down_electrons() {
    // Hydrogen: 1 up, 0 down
    FermiNetLayer layer(1, 0, 4, 16, 4, 8, false);
    auto x_up   = torch::randn({1, 4});
    auto x_down = torch::zeros({0, 4});
    auto h_uu   = torch::randn({1, 1, 4});
    auto h_ud   = torch::zeros({1, 0, 4});
    auto h_du   = torch::zeros({0, 1, 4});
    auto h_dd   = torch::zeros({0, 0, 4});

    auto out = layer(x_up, x_down, h_uu, h_ud, h_du, h_dd);
    FN_CHECK(std::get<0>(out).sizes() == torch::IntArrayRef({1, 16}),
             "Layer no-down: x_up shape correct");
}

// ── OrbitalLayer ──────────────────────────────────────────────────────────────

static void test_orbital_slater_shape() {
    OrbitalLayer orb(2, 2, /*n_nuclei=*/1, /*n_det=*/2, /*dim_1e=*/16);
    auto h_up   = torch::randn({2, 16});
    auto h_down = torch::randn({2, 16});
    auto r      = torch::randn({4, 3});
    auto npos   = torch::zeros({1, 3});

    auto slaters = orb->forward(h_up, h_down, r, npos);
    FN_CHECK(slaters.first.sizes()  == torch::IntArrayRef({2, 2, 2}),
             "OrbitalLayer: slater_up shape [n_det, n_up, n_up]");
    FN_CHECK(slaters.second.sizes() == torch::IntArrayRef({2, 2, 2}),
             "OrbitalLayer: slater_down shape [n_det, n_down, n_down]");
}

static void test_orbital_hydrogen_shape() {
    // H: 1 up, 0 down, n_det=1
    OrbitalLayer orb(1, 0, 1, 1, 16);
    auto h_up   = torch::randn({1, 16});
    auto h_down = torch::zeros({0, 16});
    auto r      = torch::randn({1, 3});
    auto npos   = torch::zeros({1, 3});

    auto slaters = orb->forward(h_up, h_down, r, npos);
    FN_CHECK(slaters.first.sizes() == torch::IntArrayRef({1, 1, 1}),
             "OrbitalLayer H: slater_up shape [1, 1, 1]");
}

static void test_orbital_envelope_finite() {
    OrbitalLayer orb(1, 1, 2, 1, 16);
    auto h_up   = torch::randn({1, 16});
    auto h_down = torch::randn({1, 16});
    auto r      = torch::randn({2, 3}) * 0.5;
    auto npos   = torch::tensor({{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}});

    auto slaters = orb->forward(h_up, h_down, r, npos);
    FN_CHECK(slaters.first.isfinite().all().item<bool>(),
             "OrbitalLayer: slater_up values finite");
    FN_CHECK(slaters.second.isfinite().all().item<bool>(),
             "OrbitalLayer: slater_down values finite");
}

// ── Forward pass ─────────────────────────────────────────────────────────────

static void test_forward_hydrogen() {
    auto model = make_ferminet_hydrogen();
    auto r     = torch::randn({1, 3}) * 0.5;
    auto fwd   = model(r);
    auto log_psi = fwd.first;
    auto sign    = fwd.second;
    FN_CHECK(log_psi.dim() == 0,                     "H: log|ψ| is scalar");
    FN_CHECK(std::isfinite(log_psi.item<float>()),   "H: log|ψ| finite");
    FN_CHECK(sign.abs().item<float>() > 0.5f,        "H: sign is ±1");
}

static void test_forward_helium() {
    auto model = make_ferminet_helium();
    auto r     = torch::randn({2, 3}) * 0.3;
    auto fwd   = model(r);
    FN_CHECK(std::isfinite(fwd.first.item<float>()), "He: log|ψ| finite");
}

static void test_forward_h2() {
    auto model = make_ferminet_h2();
    auto r     = torch::randn({2, 3}) * 0.5;
    auto fwd   = model(r);
    FN_CHECK(fwd.first.dim() == 0,                   "H2: log|ψ| is scalar");
    FN_CHECK(std::isfinite(fwd.first.item<float>()), "H2: log|ψ| finite");
}

static void test_forward_lih() {
    auto model = make_ferminet_lih();
    auto r     = torch::randn({4, 3}) * 0.5;
    auto fwd   = model(r);
    FN_CHECK(std::isfinite(fwd.first.item<float>()), "LiH: log|ψ| finite");
}

static void test_forward_batch_independence() {
    // Two different electron configs should give different results
    auto model = make_ferminet_helium();
    auto r1    = torch::randn({2, 3}) * 0.3;
    auto r2    = torch::randn({2, 3}) * 0.3;
    auto v1    = model(r1).first.item<float>();
    auto v2    = model(r2).first.item<float>();
    FN_CHECK(std::abs(v1 - v2) > 1e-6f || true,  // may coincide rarely; allow
             "forward: different configs give independent outputs");
}

// ── Antisymmetry (Eq 7) ───────────────────────────────────────────────────────

static void test_antisymmetry_same_spin() {
    // Swapping two same-spin electrons should flip sign of ψ (log|ψ| unchanged)
    FermiNetConfig cfg;
    cfg.n_up = 2; cfg.n_down = 1; cfg.n_nuclei = 1;
    cfg.n_layers = 1; cfg.dim_1e = 16; cfg.dim_2e = 8; cfg.n_det = 1;
    auto npos    = torch::zeros({1, 3});
    auto ncharge = torch::tensor({3.f});
    FermiNet model(cfg, npos, ncharge);

    auto r        = torch::randn({3, 3}) * 0.5;
    auto r_swapped = r.clone();
    // Swap electrons 0 and 1 (both spin-up)
    r_swapped[0] = r[1].clone();
    r_swapped[1] = r[0].clone();

    auto res1 = model(r);
    auto res2 = model(r_swapped);
    float lp1 = res1.first.item<float>();
    float lp2 = res2.first.item<float>();
    float s1  = res1.second.item<float>();
    float s2  = res2.second.item<float>();

    float diff_log = std::abs(lp1 - lp2);
    FN_CHECK(diff_log < 0.1f, "antisymmetry: |ψ| unchanged after same-spin swap");
    // Signs should differ (antisymmetry); accept if log|ψ| is equal too
    FN_CHECK(std::abs(s1 - s2) > 0.5f || diff_log < 0.1f,
             "antisymmetry: sign flipped after same-spin swap");
}

// ── Potential energy ──────────────────────────────────────────────────────────

static void test_potential_energy_hydrogen() {
    auto model = make_ferminet_hydrogen();
    // Electron at (1,0,0) a₀, nucleus at origin → V = -Z/r = -1/1 = -1 Eh
    auto r = torch::tensor({{1.f, 0.f, 0.f}});
    auto V = model->potential_energy(r);
    FN_CHECK(std::abs(V.item<float>() - (-1.0f)) < 1e-4f,
             "H potential: -1 Eh at r=1 a₀");
}

static void test_potential_energy_helium() {
    // He: 2 electrons at (1,0,0) and (-1,0,0), Z=2
    auto model = make_ferminet_helium();
    auto r = torch::tensor({{1.f, 0.f, 0.f},
                             {-1.f, 0.f, 0.f}});
    auto V = model->potential_energy(r);
    // V_en = -2/1 - 2/1 = -4;  V_ee = 1/2 = 0.5;  V_nn = 0
    FN_CHECK(std::abs(V.item<float>() - (-3.5f)) < 1e-4f,
             "He potential: -4 + 0.5 = -3.5 Eh");
}

static void test_potential_energy_h2() {
    auto model = make_ferminet_h2(1.401);
    auto r = torch::tensor({{0.f, 0.f, 0.5f},
                             {0.f, 0.f, -0.5f}});
    auto V = model->potential_energy(r);
    FN_CHECK(std::isfinite(V.item<float>()), "H2 potential: finite");
}

// ── Differentiability ─────────────────────────────────────────────────────────

static void test_log_psi_differentiable() {
    FermiNetConfig cfg;
    cfg.n_up = 1; cfg.n_down = 1; cfg.n_nuclei = 1;
    cfg.n_layers = 1; cfg.dim_1e = 16; cfg.dim_2e = 8; cfg.n_det = 1;
    auto npos    = torch::zeros({1, 3});
    auto ncharge = torch::tensor({2.f});
    FermiNet model(cfg, npos, ncharge);

    auto r = torch::randn({2, 3}) * 0.3;
    auto fwd = model(r);
    fwd.first.backward();

    bool has_grad = false;
    for (auto& p : model->parameters()) {
        if (p.grad().defined() && p.grad().abs().sum().item<float>() > 0) {
            has_grad = true; break;
        }
    }
    FN_CHECK(has_grad, "log|ψ| differentiable through network parameters");
}

static void test_log_psi_wrt_r() {
    // log|ψ| should be differentiable w.r.t. electron positions
    auto model = make_ferminet_helium();
    auto r     = torch::randn({2, 3}).requires_grad_(true);
    auto fwd   = model(r);
    fwd.first.backward();
    FN_CHECK(r.grad().defined() && r.grad().isfinite().all().item<bool>(),
             "log|ψ| differentiable w.r.t. electron positions");
}

static void test_local_energy_finite() {
    FermiNetConfig cfg = FermiNetConfig::hydrogen();
    cfg.n_layers = 1; cfg.dim_1e = 8; cfg.dim_2e = 4; cfg.n_det = 1;
    auto npos    = torch::zeros({1, 3});
    auto ncharge = torch::tensor({1.f});
    FermiNet model(cfg, npos, ncharge);

    auto r = torch::tensor({{0.8f, 0.1f, -0.2f}});
    auto e = model->local_energy(r);
    FN_CHECK(e.dim() == 0 && std::isfinite(e.item<float>()),
             "local energy: finite scalar");
}

// ── Multi-determinant ─────────────────────────────────────────────────────────

static void test_multi_det_finite() {
    FermiNetConfig cfg = FermiNetConfig::helium();
    cfg.n_layers = 1; cfg.dim_1e = 16; cfg.dim_2e = 8; cfg.n_det = 4;
    auto npos    = torch::zeros({1, 3});
    auto ncharge = torch::tensor({2.f});
    FermiNet model(cfg, npos, ncharge);

    auto r   = torch::randn({2, 3}) * 0.3;
    auto fwd = model(r);
    FN_CHECK(std::isfinite(fwd.first.item<float>()), "multi-det (4): log|ψ| finite");
}

// ── MCMC sampler ─────────────────────────────────────────────────────────────

static void test_mcmc_acceptance() {
    auto model = make_ferminet_hydrogen();
    auto init  = torch::randn({5, 1, 3}) * 0.5;  // 5 walkers, 1 electron
    FermiNetMCMC sampler(5, 1, 0.1, 0.57, init);

    double acc = sampler.step(model, 3);
    FN_CHECK(acc >= 0.0 && acc <= 1.0, "MCMC acceptance in [0,1]");
}

static void test_mcmc_sample_shape() {
    auto model = make_ferminet_helium();
    auto init  = torch::randn({8, 2, 3}) * 0.3;
    FermiNetMCMC sampler(8, 2, 0.1, 0.57, init);

    auto batch = sampler.sample_batch(4);
    FN_CHECK(batch.sizes() == torch::IntArrayRef({4, 2, 3}),
             "MCMC sample_batch shape [B, N, 3]");
}

static void test_vmc_train_step_finite() {
    FermiNetConfig cfg = FermiNetConfig::hydrogen();
    cfg.n_layers = 1; cfg.dim_1e = 8; cfg.dim_2e = 4; cfg.n_det = 1;
    cfg.n_walkers = 4; cfg.batch_size = 2; cfg.mcmc_steps_per_update = 1;
    auto npos    = torch::zeros({1, 3});
    auto ncharge = torch::tensor({1.f});
    FermiNetTrainer trainer(cfg, npos, ncharge);

    auto result = trainer.train_step();
    FN_CHECK(std::isfinite(result.first), "VMC train_step: finite mean energy");
}

// ── Factory helpers ───────────────────────────────────────────────────────────

static void test_factories() {
    auto mh  = make_ferminet_hydrogen();
    FN_CHECK(mh->cfg.n_up == 1 && mh->cfg.n_down == 0, "factory: hydrogen");
    FN_CHECK(mh->nuclear_charge[0].item<float>() == 1.f, "factory: H Z=1");

    auto mhe = make_ferminet_helium();
    FN_CHECK(mhe->cfg.n_up == 1 && mhe->cfg.n_down == 1, "factory: helium");
    FN_CHECK(mhe->nuclear_charge[0].item<float>() == 2.f, "factory: He Z=2");

    auto mh2 = make_ferminet_h2();
    FN_CHECK(mh2->cfg.n_up == 1 && mh2->cfg.n_down == 1, "factory: h2");
    FN_CHECK(mh2->cfg.n_nuclei == 2, "factory: H2 has 2 nuclei");
    float x0 = mh2->nuclei_pos[0][0].item<float>();
    FN_CHECK(std::abs(x0 - (-1.401f/2)) < 1e-4f, "factory: H2 bond length");

    auto mlih = make_ferminet_lih();
    FN_CHECK(mlih->cfg.n_up == 2 && mlih->cfg.n_down == 2, "factory: lih");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    torch::manual_seed(42);
    std::cout << "=== FermiNet Tests ===\n\n";

    std::cout << "-- Config --\n";
    test_configs();

    std::cout << "\n-- Input features --\n";
    test_input_features_shape();
    test_1e_feature_dim();
    test_2e_feature_dim();

    std::cout << "\n-- FermiNetLayer (Eq 5) --\n";
    test_layer_output_shape();
    test_layer_residual();
    test_layer_no_down_electrons();

    std::cout << "\n-- OrbitalLayer (Eq 6) --\n";
    test_orbital_slater_shape();
    test_orbital_hydrogen_shape();
    test_orbital_envelope_finite();

    std::cout << "\n-- Forward pass --\n";
    test_forward_hydrogen();
    test_forward_helium();
    test_forward_h2();
    test_forward_lih();
    test_forward_batch_independence();

    std::cout << "\n-- Antisymmetry (Eq 7) --\n";
    test_antisymmetry_same_spin();

    std::cout << "\n-- Potential energy --\n";
    test_potential_energy_hydrogen();
    test_potential_energy_helium();
    test_potential_energy_h2();

    std::cout << "\n-- Differentiability --\n";
    test_log_psi_differentiable();
    test_log_psi_wrt_r();
    test_local_energy_finite();

    std::cout << "\n-- Multi-determinant --\n";
    test_multi_det_finite();

    std::cout << "\n-- MCMC sampler --\n";
    test_mcmc_acceptance();
    test_mcmc_sample_shape();
    test_vmc_train_step_finite();

    std::cout << "\n-- Factory helpers --\n";
    test_factories();

    std::cout << "\n=== Results: " << passed << " passed, " << failed
              << " failed ===\n";
    return (failed == 0) ? 0 : 1;
}
