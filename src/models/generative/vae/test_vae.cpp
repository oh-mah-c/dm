// ─────────────────────────────────────────────────────────────────────────────
// test_vae.cpp — C++ structural tests for VAE
//
// Paper: D.P. Kingma & M. Welling,
//        "Auto-Encoding Variational Bayes", arXiv:1312.6114v11, 2022
//
// Tests verify:
//  1.  Encoder output shapes: μ and log_var both [B, latent_dim]
//  2.  Decoder output shape: [B, input_dim], values in (0,1)
//  3.  Reparameterisation: z = μ + σ ⊙ ε gives correct shape
//  4.  In eval mode reparameterisation returns μ exactly (no noise)
//  5.  Forward pass output shapes {recon_x, μ, log_var}
//  6.  ELBO loss is finite and positive
//  7.  KL term ≥ 0 (lower bound on D_KL)
//  8.  Reconstruction term ≥ 0
//  9.  Gradient flow: no NaN/Inf through encoder + decoder
// 10.  Adam optimiser step changes weights
// 11.  sample() returns [N, input_dim] in (0,1)
// 12.  encode() returns [B, latent_dim]
// 13.  Checkpoint save/load round-trip
// ─────────────────────────────────────────────────────────────────────────────

#include "models/generative/vae/vae.h"

#include <torch/torch.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace dm::models::generative;

// ── Minimal test framework ───────────────────────────────────────────────────
static int s_passed = 0, s_failed = 0;
#define ASSERT_TRUE(cond, msg) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "  FAIL: %s\n", (msg)); \
        ++s_failed; return; \
    } } while(0)

static void begin_test(const char* name) {
    std::printf("[test] %s ...", name); std::fflush(stdout);
}
static void end_test() { ++s_passed; std::printf(" PASS\n"); }

// ── Shared small model dimensions ────────────────────────────────────────────
static const int64_t INPUT  = 784;   // 28×28
static const int64_t HIDDEN = 128;   // reduced for fast tests
static const int64_t LATENT = 20;    // Section 5
static const int64_t BATCH  = 8;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Encoder output shapes
// ─────────────────────────────────────────────────────────────────────────────
static void test_encoder_shapes() {
    begin_test("encoder output shapes [B, latent_dim] (Appendix C.2)");
    VAEEncoder enc(INPUT, HIDDEN, LATENT);
    enc->eval();
    torch::NoGradGuard ng;
    auto x = torch::rand({BATCH, INPUT});
    auto [mu, lv] = enc->forward(x);
    ASSERT_TRUE(mu.sizes() == torch::IntArrayRef({BATCH, LATENT}), "mu shape wrong");
    ASSERT_TRUE(lv.sizes() == torch::IntArrayRef({BATCH, LATENT}), "log_var shape wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Decoder output shape + range (0,1)
// ─────────────────────────────────────────────────────────────────────────────
static void test_decoder_shape_range() {
    begin_test("decoder output shape [B, input_dim] in (0,1) (Appendix C.1)");
    VAEDecoder dec(LATENT, HIDDEN, INPUT);
    dec->eval();
    torch::NoGradGuard ng;
    auto z   = torch::randn({BATCH, LATENT});
    auto out = dec->forward(z);
    ASSERT_TRUE(out.sizes() == torch::IntArrayRef({BATCH, INPUT}), "decoder shape wrong");
    ASSERT_TRUE((out > 0.f).all().item<bool>() && (out < 1.f).all().item<bool>(),
                "decoder output not in (0,1)");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Reparameterisation: shape correct in train mode
// ─────────────────────────────────────────────────────────────────────────────
static void test_reparam_shape() {
    begin_test("reparameterisation z shape [B, latent_dim] (Section 2.4)");
    auto model = VAE(INPUT, HIDDEN, LATENT);
    model->train();
    auto mu      = torch::zeros({BATCH, LATENT});
    auto log_var = torch::zeros({BATCH, LATENT});
    auto z       = model->reparameterise(mu, log_var);
    ASSERT_TRUE(z.sizes() == torch::IntArrayRef({BATCH, LATENT}), "z shape wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Eval mode: reparameterisation returns μ exactly (deterministic)
// ─────────────────────────────────────────────────────────────────────────────
static void test_reparam_eval_deterministic() {
    begin_test("eval mode: z == μ (no noise) (Section 2.4)");
    auto model = VAE(INPUT, HIDDEN, LATENT);
    model->eval();
    auto mu      = torch::randn({BATCH, LATENT});
    auto log_var = torch::randn({BATCH, LATENT});
    auto z       = model->reparameterise(mu, log_var);
    ASSERT_TRUE(torch::allclose(z, mu), "eval reparameterisation should return mu exactly");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Full forward pass output shapes
// ─────────────────────────────────────────────────────────────────────────────
static void test_forward_shapes() {
    begin_test("forward pass output shapes {recon_x, mu, log_var}");
    auto model = VAE(INPUT, HIDDEN, LATENT);
    model->eval();
    torch::NoGradGuard ng;
    auto x = torch::rand({BATCH, INPUT});
    auto [recon_x, mu, lv] = model->forward(x);
    ASSERT_TRUE(recon_x.sizes() == torch::IntArrayRef({BATCH, INPUT}), "recon_x shape");
    ASSERT_TRUE(mu.sizes()      == torch::IntArrayRef({BATCH, LATENT}), "mu shape");
    ASSERT_TRUE(lv.sizes()      == torch::IntArrayRef({BATCH, LATENT}), "log_var shape");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. ELBO loss is finite and positive
// ─────────────────────────────────────────────────────────────────────────────
static void test_loss_finite() {
    begin_test("ELBO loss is finite and positive (Eq. 10)");
    auto model = VAE(INPUT, HIDDEN, LATENT);
    model->train();
    auto x = torch::rand({BATCH, INPUT});
    auto [recon_x, mu, lv] = model->forward(x);
    auto loss = vae_loss(recon_x, x, mu, lv);
    ASSERT_TRUE(std::isfinite(loss.item<float>()), "loss is not finite");
    ASSERT_TRUE(loss.item<float>() > 0.f,          "loss should be > 0");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. KL term ≥ 0  (Appendix B)
// ─────────────────────────────────────────────────────────────────────────────
static void test_kl_nonnegative() {
    begin_test("KL term >= 0 (Appendix B analytical solution)");
    // KL = -½ Σ(1 + log σ² - μ² - σ²); minimised at μ=0, σ=1 → KL=0
    auto mu      = torch::randn({BATCH, LATENT});
    auto log_var = torch::randn({BATCH, LATENT});
    auto kl = -0.5f * torch::sum(1.0f + log_var - mu.pow(2) - log_var.exp())
              / BATCH;
    ASSERT_TRUE(kl.item<float>() >= -1e-4f, "KL term is negative");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Reconstruction term ≥ 0  (BCE is always ≥ 0)
// ─────────────────────────────────────────────────────────────────────────────
static void test_recon_nonnegative() {
    begin_test("reconstruction BCE >= 0 (Appendix C.1)");
    auto recon = torch::rand({BATCH, INPUT});        // in (0,1)
    auto x     = torch::rand({BATCH, INPUT});
    auto bce   = torch::nn::functional::binary_cross_entropy(
                     recon, x,
                     torch::nn::functional::BinaryCrossEntropyFuncOptions()
                         .reduction(torch::kSum))
                 / BATCH;
    ASSERT_TRUE(bce.item<float>() >= 0.f, "BCE reconstruction is negative");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Gradient flow: no NaN/Inf
// ─────────────────────────────────────────────────────────────────────────────
static void test_gradient_flow() {
    begin_test("gradient flow: no NaN/Inf");
    auto model = VAE(INPUT, HIDDEN, LATENT);
    model->train();
    auto x = torch::rand({BATCH, INPUT});
    auto [recon_x, mu, lv] = model->forward(x);
    auto loss = vae_loss(recon_x, x, mu, lv);
    loss.backward();

    bool ok = true;
    for (auto& p : model->parameters()) {
        if (!p.grad().defined()) continue;
        if (p.grad().isnan().any().item<bool>() ||
            p.grad().isinf().any().item<bool>()) { ok = false; break; }
    }
    ASSERT_TRUE(ok, "NaN/Inf in gradients");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Adam optimiser step changes weights
// ─────────────────────────────────────────────────────────────────────────────
static void test_weight_update() {
    begin_test("Adam optimiser step changes weights (Section 5)");
    auto model = VAE(INPUT, HIDDEN, LATENT);
    model->train();
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-3));

    auto w_before = model->encoder->fc_mu->weight.clone().detach();

    auto x = torch::rand({BATCH, INPUT});
    auto [recon_x, mu, lv] = model->forward(x);
    auto loss = vae_loss(recon_x, x, mu, lv);
    opt.zero_grad();
    loss.backward();
    opt.step();

    ASSERT_TRUE(!torch::allclose(w_before, model->encoder->fc_mu->weight.detach()),
                "weights unchanged after Adam step");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. sample() returns [N, input_dim] in (0,1)
// ─────────────────────────────────────────────────────────────────────────────
static void test_sample_shape_range() {
    begin_test("sample() returns [N, input_dim] in (0,1)");
    auto model = VAE(INPUT, HIDDEN, LATENT);
    int64_t N  = 5;
    auto samples = model->sample(N, torch::kCPU);
    ASSERT_TRUE(samples.sizes() == torch::IntArrayRef({N, INPUT}),
                "sample shape wrong");
    ASSERT_TRUE((samples > 0.f).all().item<bool>() &&
                (samples < 1.f).all().item<bool>(),
                "samples not in (0,1)");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. encode() returns latent mean [B, latent_dim]
// ─────────────────────────────────────────────────────────────────────────────
static void test_encode_shape() {
    begin_test("encode() returns [B, latent_dim]");
    auto model = VAE(INPUT, HIDDEN, LATENT);
    auto x  = torch::rand({BATCH, INPUT});
    auto mu = model->encode(x);
    ASSERT_TRUE(mu.sizes() == torch::IntArrayRef({BATCH, LATENT}),
                "encode output shape wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Checkpoint save/load round-trip
// ─────────────────────────────────────────────────────────────────────────────
static void test_checkpoint() {
    begin_test("checkpoint save/load round-trip");
    auto m1 = VAE(INPUT, HIDDEN, LATENT);
    m1->eval();
    torch::NoGradGuard ng;
    auto x = torch::rand({2, INPUT});
    auto [r1, mu1, lv1] = m1->forward(x);

    const std::string path = "/tmp/test_vae_ckpt.pt";
    {
        torch::serialize::OutputArchive ar;
        m1->save(ar);
        ar.save_to(path);
    }
    auto m2 = VAE(INPUT, HIDDEN, LATENT);
    {
        torch::serialize::InputArchive ar;
        ar.load_from(path);
        m2->load(ar);
    }
    m2->eval();
    auto [r2, mu2, lv2] = m2->forward(x);

    ASSERT_TRUE(torch::allclose(r1,  r2,  1e-5f, 1e-5f), "recon_x differs after reload");
    ASSERT_TRUE(torch::allclose(mu1, mu2, 1e-5f, 1e-5f), "mu differs after reload");
    std::filesystem::remove(path);
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("\n=== VAE tests (Kingma & Welling, arXiv:1312.6114v11) ===\n");

    test_encoder_shapes();
    test_decoder_shape_range();
    test_reparam_shape();
    test_reparam_eval_deterministic();
    test_forward_shapes();
    test_loss_finite();
    test_kl_nonnegative();
    test_recon_nonnegative();
    test_gradient_flow();
    test_weight_update();
    test_sample_shape_range();
    test_encode_shape();
    test_checkpoint();

    std::printf("=== %d passed, %d failed ===\n\n", s_passed, s_failed);
    return s_failed == 0 ? 0 : 1;
}
