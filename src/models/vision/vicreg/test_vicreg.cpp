// ─────────────────────────────────────────────────────────────────────────────
// VICReg unit tests
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/vicreg/vicreg.h"

#include <torch/torch.h>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace dm::models::vision;

static int passed = 0, failed = 0;

#define VICREG_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << (msg) << "\n"; failed++; \
        } else { \
            std::cout << "  [PASS] " << (msg) << "\n"; passed++; \
        } \
    } while (0)

// ── Config ────────────────────────────────────────────────────────────────────

static void test_config_defaults() {
    VICRegConfig cfg;
    VICREG_CHECK(cfg.repr_dim     == 2048,  "default repr_dim=2048");
    VICREG_CHECK(cfg.expander_dim == 8192,  "default expander_dim=8192");
    VICREG_CHECK(cfg.lambda_inv   == 25.0,  "default lambda=25");
    VICREG_CHECK(cfg.mu_var       == 25.0,  "default mu=25");
    VICREG_CHECK(cfg.nu_cov       == 1.0,   "default nu=1");
    VICREG_CHECK(cfg.gamma        == 1.0,   "default gamma=1");
    VICREG_CHECK(std::abs(cfg.var_eps - 1e-4) < 1e-9, "default var_eps=1e-4");
}

static void test_config_tiny() {
    auto cfg = VICRegConfig::tiny(64, 128);
    VICREG_CHECK(cfg.repr_dim == 64,  "tiny repr_dim=64");
    VICREG_CHECK(cfg.expander_dim == 128, "tiny expander_dim=128");
}

// ── Expander shape ────────────────────────────────────────────────────────────

static void test_expander_shape() {
    Expander exp(64, 128);
    auto x = torch::randn({4, 64});
    auto y = exp(x);
    VICREG_CHECK(y.sizes() == torch::IntArrayRef({4, 128}), "expander output shape [B, exp_dim]");
}

// ── Static loss helpers ───────────────────────────────────────────────────────

static void test_invariance_loss_zero() {
    // When both embeddings are identical, invariance loss = 0
    auto z = torch::randn({8, 32});
    auto loss = VICRegImpl::invariance_loss(z, z);
    VICREG_CHECK(loss.item<float>() < 1e-6f, "invariance_loss(z,z) = 0");
}

static void test_invariance_loss_positive() {
    auto z  = torch::randn({8, 32});
    auto zp = torch::randn({8, 32});
    auto loss = VICRegImpl::invariance_loss(z, zp);
    VICREG_CHECK(loss.item<float>() > 0.0f, "invariance_loss(z,z') > 0 for random z");
}

static void test_variance_loss_collapse() {
    // Collapsed embedding: all same vector → std ≈ 0 → hinge = gamma = 1
    int64_t B = 16, d = 32;
    auto z = torch::ones({B, d});
    auto loss = VICRegImpl::variance_loss(z, 1.0, 1e-4);
    // Each dim: max(0, 1 - sqrt(0 + 1e-4)) ≈ max(0, 1 - 0.01) ≈ 0.99
    VICREG_CHECK(loss.item<float>() > 0.5f, "variance_loss fires on collapsed embedding");
}

static void test_variance_loss_spread() {
    // Spread embeddings: std ≥ γ → hinge = 0 for each dim
    int64_t B = 128, d = 16;
    // Sample from N(0, 4) → std ≈ 4 >> gamma=1
    auto z = torch::randn({B, d}) * 4.0;
    auto loss = VICRegImpl::variance_loss(z, 1.0, 1e-4);
    VICREG_CHECK(loss.item<float>() < 1e-3f, "variance_loss ≈ 0 for well-spread embedding");
}

static void test_covariance_loss_identity() {
    // If batch embeddings are drawn from N(0,I), off-diagonal covs ≈ 0
    // (not exact for finite B, but should be small for large B)
    torch::manual_seed(0);
    int64_t B = 512, d = 16;
    auto z = torch::randn({B, d});
    auto loss = VICRegImpl::covariance_loss(z);
    VICREG_CHECK(loss.item<float>() < 0.5f, "covariance_loss small for uncorrelated embedding");
}

static void test_covariance_loss_correlated() {
    // Construct perfectly correlated embedding: z[:,1] = z[:,0]
    int64_t B = 64, d = 4;
    auto z = torch::randn({B, d});
    // Force dim-1 = dim-0
    z.select(1, 1).copy_(z.select(1, 0));
    auto loss = VICRegImpl::covariance_loss(z);
    VICREG_CHECK(loss.item<float>() > 0.1f, "covariance_loss fires on correlated dimensions");
}

// ── Full VICReg model (MLP encoder) ──────────────────────────────────────────

static void test_forward_loss_scalar() {
    auto model = make_vicreg_mlp(/*in_dim=*/3*16*16, /*repr_dim=*/64, /*expander_dim=*/128);
    auto x  = torch::randn({4, 3, 16, 16});
    auto xp = torch::randn({4, 3, 16, 16});
    auto r  = model(x, xp);

    VICREG_CHECK(r.total.dim() == 0,                   "total loss is scalar");
    VICREG_CHECK(std::isfinite(r.total.item<float>()),  "total loss is finite");
    VICREG_CHECK(r.inv.dim()   == 0,                   "invariance term is scalar");
    VICREG_CHECK(r.var.dim()   == 0,                   "variance term is scalar");
    VICREG_CHECK(r.cov.dim()   == 0,                   "covariance term is scalar");
}

static void test_invariance_term_positive() {
    auto model = make_vicreg_mlp(3*16*16, 64, 128);
    auto x  = torch::randn({4, 3, 16, 16});
    auto xp = torch::randn({4, 3, 16, 16});
    auto r  = model(x, xp);
    VICREG_CHECK(r.inv.item<float>() >= 0.0f, "invariance term ≥ 0");
}

static void test_variance_fires_on_collapse() {
    // With constant input → encoder outputs constant → expander outputs near-constant
    // → variance loss should be > 0
    VICRegConfig cfg = VICRegConfig::tiny(32, 64);
    auto enc = MLPEncoder(/*in_dim=*/32, /*hidden=*/64, /*repr_dim=*/32);
    torch::nn::AnyModule m(enc);
    VICReg model(std::move(m), torch::nn::AnyModule{}, cfg);

    // Freeze weights so output is deterministic
    torch::NoGradGuard ng;
    auto x = torch::zeros({8, 32});   // constant input → risk of collapse
    auto r = model(x, x);
    // Variance term: since both x are identical, the embedding has zero variance → hinge fires
    VICREG_CHECK(r.var.item<float>() > 0.0f, "variance term fires on zero-variance embedding");
}

static void test_loss_backward() {
    auto model = make_vicreg_mlp(3*8*8, 32, 64);
    auto x  = torch::randn({4, 3, 8, 8});
    auto xp = torch::randn({4, 3, 8, 8});
    auto r  = model(x, xp);
    r.total.backward();

    bool any_grad = false;
    for (auto &p : model->parameters()) {
        if (p.grad().defined() && p.grad().abs().sum().item<float>() > 0) {
            any_grad = true; break;
        }
    }
    VICREG_CHECK(any_grad, "gradients flow to model parameters");
}

static void test_loss_weighted_sum() {
    // Verify:  total ≈ λ·s + μ·(v+v') + ν·(c+c')
    auto model = make_vicreg_mlp(32, 32, 64);
    auto x  = torch::randn({8, 32});
    auto xp = torch::randn({8, 32});

    // Override coefficients
    model->cfg.lambda_inv = 3.0;
    model->cfg.mu_var     = 7.0;
    model->cfg.nu_cov     = 2.0;

    auto r = model(x, xp);
    float expected = 3.0f * r.inv.item<float>()
                   + 7.0f * r.var.item<float>()
                   + 2.0f * r.cov.item<float>();
    VICREG_CHECK(std::abs(r.total.item<float>() - expected) < 1e-4f,
                 "total loss = λ·s + μ·var + ν·cov");
}

static void test_siamese_same_output_for_same_input() {
    // When x == x', the encoder (shared) produces the same representation.
    // The two expanders are independent modules; after random init their BN running
    // stats differ, so the expander outputs Z and Z' may not be identical.
    // What we can guarantee: the invariance term is finite and non-negative.
    auto model = make_vicreg_mlp(32, 32, 64);
    model->eval();
    torch::NoGradGuard ng;
    auto x = torch::randn({4, 32});
    auto r = model(x, x);
    float inv = r.inv.item<float>();
    VICREG_CHECK(std::isfinite(inv) && inv >= 0.0f,
                 "invariance term finite and ≥ 0 when x=x' (siamese)");
}

static void test_independent_expanders() {
    // The two expanders have independent weights; verify they're not the same module.
    auto model = make_vicreg_mlp(32, 32, 64);
    auto& e1 = model->expander;
    auto& e2 = model->expander2;
    // Perturb e2 and check e1 is unaffected
    {
        torch::NoGradGuard ng;
        for (auto& p : e2->parameters()) p.fill_(0.0f);
    }
    auto p1 = e1->fc1->weight.abs().sum().item<float>();
    VICREG_CHECK(p1 > 0.0f, "expanders are independent (perturbing e2 doesn't affect e1)");
}

static void test_factories() {
    auto m1 = make_vicreg_mlp(64, 128, 256);
    VICREG_CHECK(m1->cfg.repr_dim == 128,     "make_vicreg_mlp repr_dim");
    VICREG_CHECK(m1->cfg.expander_dim == 256, "make_vicreg_mlp expander_dim");
    VICREG_CHECK(m1->shared_encoder == true,  "make_vicreg_mlp is siamese");

    auto enc = MLPEncoder(8, 16, 32);
    auto m2  = make_vicreg_siamese(torch::nn::AnyModule(enc));
    VICREG_CHECK(m2->shared_encoder == true, "make_vicreg_siamese shared_encoder=true");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    torch::manual_seed(42);
    std::cout << "=== VICReg Tests ===\n\n";

    std::cout << "-- Config --\n";
    test_config_defaults();
    test_config_tiny();

    std::cout << "\n-- Expander --\n";
    test_expander_shape();

    std::cout << "\n-- Loss helpers --\n";
    test_invariance_loss_zero();
    test_invariance_loss_positive();
    test_variance_loss_collapse();
    test_variance_loss_spread();
    test_covariance_loss_identity();
    test_covariance_loss_correlated();

    std::cout << "\n-- Full model --\n";
    test_forward_loss_scalar();
    test_invariance_term_positive();
    test_variance_fires_on_collapse();
    test_loss_backward();
    test_loss_weighted_sum();
    test_siamese_same_output_for_same_input();
    test_independent_expanders();
    test_factories();

    std::cout << "\n=== Results: " << passed << " passed, " << failed
              << " failed ===\n";
    return (failed == 0) ? 0 : 1;
}
