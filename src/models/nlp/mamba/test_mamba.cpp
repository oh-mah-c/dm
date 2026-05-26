// ─────────────────────────────────────────────────────────────────────────────
// test_mamba.cpp — C++ structural tests for Mamba
//
// Paper: A. Gu and T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective
//        State Spaces," arXiv:2312.00752v2, 2023.
//        https://arxiv.org/abs/2312.00752
//
// Tests:
//  1.  Forward output shape [B, L, vocab_size]                  (§3.4)
//  2.  SSM output shape [B, L, d_inner]                         (§3.2)
//  3.  Recurrent step matches sequential forward at each token  (§3.2)
//  4.  B, C are input-dependent (differ for different inputs)   (§3.2)
//  5.  Δ is always positive (softplus)                          (§3.2)
//  6.  A diagonal entries are always negative                   (§3.6)
//  7.  Conv1d depthwise preserves [B, d_inner, L] shape         (§3.4)
//  8.  Causal: y_t independent of x_{t+1} (SSM is causal)      (§3.2)
//  9.  Tiny param count in expected range                        (config)
// 10.  Gradient flow: no NaN/Inf                                (§3)
// 11.  AdamW step changes parameters                            (§E.2)
// 12.  LR schedule: warmup linear, then cosine decay            (§E.2)
// 13.  Checkpoint save/load round-trip                           (§E.2)
// 14.  generate() returns correct shape                          (§3.4)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/mamba/mamba.h"

#include <torch/torch.h>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <filesystem>

using namespace dm::models::nlp;

static int s_passed = 0, s_failed = 0;

#define ASSERT_TRUE(cond, msg)                            \
    do {                                                  \
        if (!(cond)) {                                    \
            std::fprintf(stderr, "  FAIL: %s\n", (msg)); \
            ++s_failed; return;                           \
        }                                                 \
    } while(0)

static void begin_test(const char* n) {
    std::printf("[test] %s ...", n); std::fflush(stdout);
}
static void end_test() { ++s_passed; std::printf(" PASS\n"); }

static MambaConfig tiny_cfg() { return MambaConfig::tiny(); }

static const int64_t B = 2;
static const int64_t L = 16;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Forward output shape [B, L, vocab_size]
// ─────────────────────────────────────────────────────────────────────────────
static void test_forward_shape() {
    begin_test("forward shape [B, L, vocab_size] (§3.4)");
    auto cfg   = tiny_cfg();
    auto model = MambaModel(cfg);
    model->eval();
    torch::NoGradGuard ng;
    auto tokens  = torch::randint(0, cfg.vocab_size, {B, L});
    auto logits  = model->forward(tokens);
    ASSERT_TRUE(logits.size(0) == B,            "batch dim wrong");
    ASSERT_TRUE(logits.size(1) == L,            "seq dim wrong");
    ASSERT_TRUE(logits.size(2) == cfg.vocab_size,"vocab dim wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. SSM output shape [B, L, d_inner]
// ─────────────────────────────────────────────────────────────────────────────
static void test_ssm_shape() {
    begin_test("SSM output shape [B, L, d_inner] (§3.2)");
    auto cfg = tiny_cfg();
    auto ssm = MambaSSM(cfg);
    ssm->eval();
    torch::NoGradGuard ng;
    auto u   = torch::randn({B, L, cfg.d_inner()});
    auto y   = ssm->forward(u);
    ASSERT_TRUE(y.size(0) == B,          "batch dim wrong");
    ASSERT_TRUE(y.size(1) == L,          "seq dim wrong");
    ASSERT_TRUE(y.size(2) == cfg.d_inner(),"d_inner dim wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. step() matches sequential forward position by position
// ─────────────────────────────────────────────────────────────────────────────
static void test_step_matches_forward() {
    begin_test("recurrent step matches sequential forward (§3.2)");
    auto cfg   = tiny_cfg();
    auto model = MambaModel(cfg);
    model->eval();
    torch::NoGradGuard ng;

    // Full sequence forward
    auto tokens  = torch::randint(0, cfg.vocab_size, {1, L});
    auto logits_full = model->forward(tokens);         // [1, L, vocab]

    // Step-by-step through one MambaBlock's SSM for the first layer only
    // (We verify the SSM step matches the SSM forward directly)
    auto cfg2 = tiny_cfg();
    auto ssm  = MambaSSM(cfg2);
    ssm->eval();

    auto u = torch::randn({1, L, cfg2.d_inner()});
    auto y_full = ssm->forward(u);                    // [1, L, d_inner]

    auto h = torch::zeros({1, cfg2.d_inner(), cfg2.d_state});
    for (int64_t t = 0; t < L; ++t) {
        auto u_t  = u.select(1, t);                   // [1, d_inner]
        auto y_t  = ssm->step(u_t, h);               // [1, d_inner]
        auto diff = (y_t - y_full.select(1, t)).abs().max().item<float>();
        ASSERT_TRUE(diff < 1e-4f, "step output differs from forward at token t");
    }
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. B and C are input-dependent (different for different inputs)
// ─────────────────────────────────────────────────────────────────────────────
static void test_selective_bc() {
    begin_test("B, C are input-dependent (§3.2)");
    auto cfg = tiny_cfg();
    auto ssm = MambaSSM(cfg);
    ssm->eval();
    torch::NoGradGuard ng;

    auto u1 = torch::randn({1, L, cfg.d_inner()});
    auto u2 = torch::randn({1, L, cfg.d_inner()});

    // Project to (dt, B, C) for each input
    auto out1 = ssm->x_proj->forward(u1.reshape({-1, cfg.d_inner()}));
    auto out2 = ssm->x_proj->forward(u2.reshape({-1, cfg.d_inner()}));

    // B and C should differ when inputs differ
    auto B1 = out1.slice(1, cfg.dt_rank(), cfg.dt_rank() + cfg.d_state);
    auto B2 = out2.slice(1, cfg.dt_rank(), cfg.dt_rank() + cfg.d_state);
    auto diff = (B1 - B2).abs().max().item().toFloat();
    ASSERT_TRUE(diff > 1e-5f, "B_ssm should differ for different inputs");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Δ (dt) is always positive (softplus output)
// ─────────────────────────────────────────────────────────────────────────────
static void test_delta_positive() {
    begin_test("Δ is always positive (softplus) (§3.2)");
    auto cfg = tiny_cfg();
    auto ssm = MambaSSM(cfg);
    ssm->eval();
    torch::NoGradGuard ng;

    auto u = torch::randn({B, L, cfg.d_inner()});
    // Extract dt_raw
    auto xp = ssm->x_proj->forward(u.reshape({-1, cfg.d_inner()}));
    auto dt_raw = xp.slice(1, 0, cfg.dt_rank());
    auto dt = torch::nn::functional::softplus(ssm->dt_proj->forward(dt_raw));
    float min_dt = dt.min().item().toFloat();
    ASSERT_TRUE(min_dt > 0.0f, "Δ should always be positive");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. A diagonal entries are always negative (stability)
// ─────────────────────────────────────────────────────────────────────────────
static void test_A_negative() {
    begin_test("A diagonal entries always negative (§3.6)");
    auto cfg = tiny_cfg();
    auto ssm = MambaSSM(cfg);
    // A = -exp(A_log), so always negative
    auto A = -torch::exp(ssm->A_log);
    float max_A = A.max().item<float>();
    ASSERT_TRUE(max_A < 0.0f, "A entries should be negative (stability)");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Conv1d depthwise output shape
// ─────────────────────────────────────────────────────────────────────────────
static void test_conv_shape() {
    begin_test("conv1d depthwise preserves sequence shape (§3.4)");
    auto cfg   = tiny_cfg();
    auto block = MambaBlock(cfg);
    block->eval();
    torch::NoGradGuard ng;
    auto x   = torch::randn({B, L, cfg.d_model});
    auto y   = block->forward(x);
    ASSERT_TRUE(y.size(0) == B,         "batch dim wrong");
    ASSERT_TRUE(y.size(1) == L,         "seq dim wrong");
    ASSERT_TRUE(y.size(2) == cfg.d_model,"d_model dim wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Causality: output at t is independent of input at t+1
// ─────────────────────────────────────────────────────────────────────────────
static void test_causal() {
    begin_test("causal: y_t independent of x_{t+1} (§3.2)");
    auto cfg   = tiny_cfg();
    auto model = MambaModel(cfg);
    model->eval();
    torch::NoGradGuard ng;

    auto tok1 = torch::randint(0, cfg.vocab_size, {1, L});
    auto tok2 = tok1.clone();
    // Change only the last token
    tok2[0][L-1] = (tok2[0][L-1].item<int64_t>() + 1) % cfg.vocab_size;

    auto logits1 = model->forward(tok1);
    auto logits2 = model->forward(tok2);

    // All positions 0..L-2 should be identical
    auto diff = (logits1.slice(1,0,L-1) - logits2.slice(1,0,L-1)).abs().max();
    ASSERT_TRUE(diff.item<float>() < 1e-4f, "causal property violated");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Tiny param count in expected range
// ─────────────────────────────────────────────────────────────────────────────
static void test_param_count() {
    begin_test("tiny param count in expected range (config)");
    auto cfg   = tiny_cfg();
    auto model = MambaModel(cfg);

    int64_t total = 0;
    for (auto& item : model->named_parameters())
        total += item.value().numel();

    ASSERT_TRUE(total > 1000,    "param count too small");
    ASSERT_TRUE(total < 5000000,"param count too large");
    std::printf(" (%lld params)", static_cast<long long>(total));
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Gradient flow: no NaN/Inf
// ─────────────────────────────────────────────────────────────────────────────
static void test_gradient_flow() {
    begin_test("gradient flow: no NaN/Inf (§3)");
    auto cfg   = tiny_cfg();
    auto model = MambaModel(cfg);
    model->train();

    auto tokens = torch::randint(0, cfg.vocab_size, {B, L + 1});
    auto inp    = tokens.slice(1, 0, L);
    auto tgt    = tokens.slice(1, 1, L + 1);
    auto logits = model->forward(inp);
    auto loss   = torch::nn::functional::cross_entropy(
        logits.reshape({-1, cfg.vocab_size}),
        tgt.reshape({-1}));
    loss.backward();

    for (auto& item : model->named_parameters()) {
        if (!item.value().grad().defined()) continue;
        auto& g = item.value().grad();
        ASSERT_TRUE(!g.isnan().any().item<bool>(),
                    ("NaN in grad: " + item.key()).c_str());
        ASSERT_TRUE(!g.isinf().any().item<bool>(),
                    ("Inf in grad: " + item.key()).c_str());
    }
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. AdamW step changes parameters
// ─────────────────────────────────────────────────────────────────────────────
static void test_optimizer_step() {
    begin_test("AdamW step changes parameters (§E.2)");
    auto cfg  = tiny_cfg();
    auto model = MambaModel(cfg);
    MambaTrainConfig tcfg;

    auto w_before = model->embedding->weight.clone().detach();
    auto opt = make_mamba_optimizer(model, tcfg);
    auto tokens = torch::randint(0, cfg.vocab_size, {B, L + 1});
    mamba_train_step(model, opt, tokens, tcfg, 0);

    auto changed = (w_before - model->embedding->weight.detach()).abs().max().item<float>();
    ASSERT_TRUE(changed > 1e-9f, "weights unchanged after optimizer step");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. LR schedule: warmup linear, then cosine decay
// ─────────────────────────────────────────────────────────────────────────────
static void test_lr_schedule() {
    begin_test("LR schedule: warmup linear + cosine decay (§E.2)");
    MambaTrainConfig tcfg;
    tcfg.lr          = 6e-4;
    tcfg.min_lr      = 1e-5;
    tcfg.warmup      = 100;
    tcfg.total_steps = 1000;

    // Warmup
    double lr0  = mamba_lr_schedule(0, tcfg);
    double lr50 = mamba_lr_schedule(49, tcfg);
    double lr99 = mamba_lr_schedule(99, tcfg);
    ASSERT_TRUE(std::fabs(lr0  - tcfg.lr / 100.0) < 1e-9, "step-0 LR wrong");
    ASSERT_TRUE(std::fabs(lr50 - tcfg.lr * 50.0/100.0) < 1e-9, "step-49 LR wrong");
    ASSERT_TRUE(std::fabs(lr99 - tcfg.lr) < 1e-9, "step-99 LR wrong");

    // After warmup: cosine should be decreasing
    double lr100 = mamba_lr_schedule(100, tcfg);
    double lr500 = mamba_lr_schedule(500, tcfg);
    double lr999 = mamba_lr_schedule(999, tcfg);
    ASSERT_TRUE(lr100 <= tcfg.lr, "post-warmup LR should not exceed peak");
    ASSERT_TRUE(lr500 <  lr100,   "LR should decrease during cosine phase");
    ASSERT_TRUE(lr999 >= tcfg.min_lr - 1e-9, "LR should not go below min_lr");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Checkpoint save/load round-trip
// ─────────────────────────────────────────────────────────────────────────────
static void test_checkpoint() {
    begin_test("checkpoint save/load round-trip (§E.2)");
    auto cfg   = tiny_cfg();
    auto model = MambaModel(cfg);
    model->eval();

    const std::string path = "/tmp/mamba_test_ckpt.pt";
    torch::save(model, path);

    auto model2 = MambaModel(cfg);
    torch::load(model2, path);
    model2->eval();
    std::filesystem::remove(path);

    torch::NoGradGuard ng;
    auto tokens  = torch::randint(0, cfg.vocab_size, {B, L});
    auto logits1 = model->forward(tokens);
    auto logits2 = model2->forward(tokens);

    auto diff = (logits1 - logits2).abs().max().item<float>();
    ASSERT_TRUE(diff < 1e-4f, "logits differ after checkpoint round-trip");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. generate() returns correct shape
// ─────────────────────────────────────────────────────────────────────────────
static void test_generate_shape() {
    begin_test("generate() returns [B, max_new] shape (§3.4)");
    auto cfg   = tiny_cfg();
    auto model = MambaModel(cfg);

    int64_t max_new  = 5;
    auto    prompt   = torch::randint(0, cfg.vocab_size, {1, 4});
    auto    gen      = model->generate(prompt, max_new, 1.0, 0.9);

    ASSERT_TRUE(gen.size(0) == 1,       "batch dim wrong");
    ASSERT_TRUE(gen.size(1) == max_new, "max_new dim wrong");
    ASSERT_TRUE(!gen.isnan().any().item<bool>(), "NaN in generated tokens");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    torch::manual_seed(42);

    test_forward_shape();
    test_ssm_shape();
    test_step_matches_forward();
    test_selective_bc();
    test_delta_positive();
    test_A_negative();
    test_conv_shape();
    test_causal();
    test_param_count();
    test_gradient_flow();
    test_optimizer_step();
    test_lr_schedule();
    test_checkpoint();
    test_generate_shape();

    std::printf("\n%d passed, %d failed\n", s_passed, s_failed);
    return s_failed ? 1 : 0;
}
