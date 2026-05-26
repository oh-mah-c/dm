// ─────────────────────────────────────────────────────────────────────────────
// test_rwkv.cpp — C++ structural tests for RWKV
//
// Paper: B. Peng et al., "RWKV: Reinventing RNNs for the Transformer Era,"
//        arXiv:2305.13048v2, 2023. https://arxiv.org/abs/2305.13048
//
// Tests:
//  1.  Forward output shape [B, T, vocab_size]                    (§3)
//  2.  TimeMix output shape [B, T, D]                             (§3.1.2)
//  3.  ChannelMix output shape [B, T, D]                          (§3.1)
//  4.  step() matches forward position-by-position (TimeMix)      (App.D)
//  5.  WKV state updates (aa, bb, pp) are non-trivial             (App.D)
//  6.  Token-shift: output changes when x_prev changes            (§3.1.1)
//  7.  Causal: y_t independent of x_{t+1}                         (§3.1)
//  8.  Tiny param count in expected range                          (config)
//  9.  Gradient flow: no NaN/Inf                                  (§3)
// 10.  Adam step changes parameters                               (§4.1)
// 11.  LR schedule: warmup linear + exponential decay             (§4.1)
// 12.  Checkpoint save/load round-trip                             (§4.1)
// 13.  generate() returns correct shape                            (§3)
// 14.  time_decay w trainable, time_first u trainable             (§3.1.2)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/rwkv/rwkv.h"

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

static RWKVConfig tiny_cfg() { return RWKVConfig::tiny(); }

static const int64_t B = 2;
static const int64_t T = 16;

// ─────────────────────────────────────────────────────────────────────────────
// 1. Forward output shape [B, T, vocab_size]
// ─────────────────────────────────────────────────────────────────────────────
static void test_forward_shape() {
    begin_test("forward shape [B, T, vocab_size] (§3)");
    auto cfg   = tiny_cfg();
    auto model = RWKVModel(cfg);
    model->eval();
    torch::NoGradGuard ng;
    auto tokens = torch::randint(0, cfg.vocab_size,
                                 {B, T},
                                 torch::TensorOptions().dtype(torch::kLong));
    auto logits = model->forward(tokens);
    ASSERT_TRUE(logits.size(0) == B,              "batch dim wrong");
    ASSERT_TRUE(logits.size(1) == T,              "seq dim wrong");
    ASSERT_TRUE(logits.size(2) == cfg.vocab_size, "vocab dim wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. TimeMix output shape [B, T, D]
// ─────────────────────────────────────────────────────────────────────────────
static void test_timemix_shape() {
    begin_test("TimeMix output shape [B, T, D] (§3.1.2)");
    auto cfg = tiny_cfg();
    auto tm  = RWKVTimeMix(cfg, 0);
    tm->eval();
    torch::NoGradGuard ng;
    auto x      = torch::randn({B, T, cfg.d_model});
    auto x_prev = torch::zeros_like(x);
    auto y      = tm->forward(x, x_prev);
    ASSERT_TRUE(y.size(0) == B,          "batch dim wrong");
    ASSERT_TRUE(y.size(1) == T,          "seq dim wrong");
    ASSERT_TRUE(y.size(2) == cfg.d_model,"d_model dim wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. ChannelMix output shape [B, T, D]
// ─────────────────────────────────────────────────────────────────────────────
static void test_channelmix_shape() {
    begin_test("ChannelMix output shape [B, T, D] (§3.1)");
    auto cfg = tiny_cfg();
    auto cm  = RWKVChannelMix(cfg, 0);
    cm->eval();
    torch::NoGradGuard ng;
    auto x      = torch::randn({B, T, cfg.d_model});
    auto x_prev = torch::zeros_like(x);
    auto y      = cm->forward(x, x_prev);
    ASSERT_TRUE(y.size(0) == B,          "batch dim wrong");
    ASSERT_TRUE(y.size(1) == T,          "seq dim wrong");
    ASSERT_TRUE(y.size(2) == cfg.d_model,"d_model dim wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. TimeMix step() matches sequential forward position-by-position
// ─────────────────────────────────────────────────────────────────────────────
static void test_step_matches_forward() {
    begin_test("TimeMix step() matches sequential forward (App.D)");
    auto cfg = tiny_cfg();
    auto tm  = RWKVTimeMix(cfg, 0);
    tm->eval();
    torch::NoGradGuard ng;

    int64_t Bs = 1;
    auto x      = torch::randn({Bs, T, cfg.d_model});
    auto x_prev = torch::zeros_like(x);

    // Full-sequence forward
    auto y_full = tm->forward(x, x_prev);   // [1, T, D]

    // Step-by-step
    auto aa = torch::zeros({Bs, cfg.d_model});
    auto bb = torch::zeros({Bs, cfg.d_model});
    auto pp = torch::full({Bs, cfg.d_model}, -1e30f);
    auto prev = torch::zeros({Bs, cfg.d_model});

    for (int64_t t = 0; t < T; ++t) {
        auto x_t = x.select(1, t);           // [1, D]
        auto y_t = tm->step(x_t, prev, aa, bb, pp);
        prev = x_t;
        auto diff = (y_t - y_full.select(1, t)).abs().max().item<float>();
        ASSERT_TRUE(diff < 1e-4f, "step output differs from forward at t");
    }
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. WKV state (aa, bb, pp) non-trivially updated
// ─────────────────────────────────────────────────────────────────────────────
static void test_wkv_state_updates() {
    begin_test("WKV state (aa, bb, pp) non-trivially updated (App.D)");
    auto cfg = tiny_cfg();
    auto tm  = RWKVTimeMix(cfg, 0);
    // Re-init Wv to non-zero so state gets updated (zero-init is paper default,
    // but would leave aa=0 since v_t=Wv*xv=0; test needs non-zero v projection)
    torch::nn::init::xavier_uniform_(tm->Wv->weight);
    torch::nn::init::xavier_uniform_(tm->Wk->weight);
    tm->eval();
    torch::NoGradGuard ng;

    auto aa = torch::zeros({1, cfg.d_model});
    auto bb = torch::zeros({1, cfg.d_model});
    auto pp = torch::full({1, cfg.d_model}, -1e30f);
    auto prev = torch::zeros({1, cfg.d_model});
    auto x_t  = torch::randn({1, cfg.d_model});

    auto aa_before = aa.clone();
    auto pp_before = pp.clone();
    tm->step(x_t, prev, aa, bb, pp);

    auto da = (aa - aa_before).abs().max().item<float>();
    auto dp = (pp - pp_before).abs().max().item<float>();
    ASSERT_TRUE(da > 0.0f, "WKV aa state unchanged after step");
    ASSERT_TRUE(dp > 0.0f, "WKV pp state unchanged after step");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Token-shift: output changes when x_prev changes
// ─────────────────────────────────────────────────────────────────────────────
static void test_token_shift() {
    begin_test("token-shift: output changes with different x_prev (§3.1.1)");
    auto cfg = tiny_cfg();
    auto tm  = RWKVTimeMix(cfg, 0);
    // Re-init projections to non-zero so the token shift is visible
    // (paper uses zero-init for Wr/Wk/Wv but that makes mu_r/k terms cancel)
    torch::nn::init::xavier_uniform_(tm->Wr->weight);
    torch::nn::init::xavier_uniform_(tm->Wk->weight);
    torch::nn::init::xavier_uniform_(tm->Wv->weight);
    tm->eval();
    torch::NoGradGuard ng;

    auto x      = torch::randn({1, T, cfg.d_model});
    auto x_prev = torch::zeros_like(x);
    auto x_prev2= torch::randn_like(x);

    auto y1 = tm->forward(x, x_prev);
    auto y2 = tm->forward(x, x_prev2);

    auto diff = (y1 - y2).abs().max().item<float>();
    ASSERT_TRUE(diff > 1e-6f, "output unchanged with different x_prev");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Causal: y_t independent of x_{t+1}
// ─────────────────────────────────────────────────────────────────────────────
static void test_causal() {
    begin_test("causal: y_t independent of x_{t+1} (§3.1)");
    auto cfg   = tiny_cfg();
    auto model = RWKVModel(cfg);
    model->eval();
    torch::NoGradGuard ng;

    auto tok1 = torch::randint(0, cfg.vocab_size, {1, T},
                               torch::TensorOptions().dtype(torch::kLong));
    auto tok2 = tok1.clone();
    tok2[0][T-1] = (tok2[0][T-1].item<int64_t>() + 1) % cfg.vocab_size;

    auto logits1 = model->forward(tok1);
    auto logits2 = model->forward(tok2);

    // Positions 0..T-2 should be identical (causal model)
    auto diff = (logits1.slice(1, 0, T-1) - logits2.slice(1, 0, T-1)).abs().max();
    ASSERT_TRUE(diff.item<float>() < 1e-4f, "causal property violated");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Tiny param count in expected range
// ─────────────────────────────────────────────────────────────────────────────
static void test_param_count() {
    begin_test("tiny param count in expected range (config)");
    auto cfg   = tiny_cfg();
    auto model = RWKVModel(cfg);

    int64_t total = 0;
    for (auto& item : model->named_parameters())
        total += item.value().numel();

    ASSERT_TRUE(total > 1000,     "param count too small");
    ASSERT_TRUE(total < 5000000, "param count too large");
    std::printf(" (%lld params)", static_cast<long long>(total));
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Gradient flow: no NaN/Inf
// ─────────────────────────────────────────────────────────────────────────────
static void test_gradient_flow() {
    begin_test("gradient flow: no NaN/Inf (§3)");
    auto cfg   = tiny_cfg();
    auto model = RWKVModel(cfg);
    model->train();

    auto tokens = torch::randint(0, cfg.vocab_size, {B, T + 1},
                                 torch::TensorOptions().dtype(torch::kLong));
    auto inp    = tokens.slice(1, 0, T);
    auto tgt    = tokens.slice(1, 1, T + 1);
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
// 10. Adam step changes parameters
// ─────────────────────────────────────────────────────────────────────────────
static void test_optimizer_step() {
    begin_test("Adam step changes parameters (§4.1)");
    auto cfg  = tiny_cfg();
    auto model = RWKVModel(cfg);
    // The head is zero-initialized (§3.4 / App.E) which means logits are always
    // zero → constant loss → zero gradient for embedding. Re-init head to
    // kaiming so gradient actually flows; this tests the optimizer mechanics.
    torch::nn::init::kaiming_uniform_(model->head->weight);
    RWKVTrainConfig tcfg;

    // Track a non-embedding parameter that will receive gradient via head
    auto head_w_before = model->head->weight.clone().detach();
    auto opt = make_rwkv_optimizer(model, tcfg);
    auto tokens = torch::randint(0, cfg.vocab_size, {B, T + 1},
                                 torch::TensorOptions().dtype(torch::kLong));
    rwkv_train_step(model, opt, tokens, tcfg, 0);

    auto changed = (head_w_before - model->head->weight.detach()).abs().max().item<float>();
    ASSERT_TRUE(changed > 1e-9f, "weights unchanged after optimizer step");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. LR schedule: warmup linear + exponential decay
// ─────────────────────────────────────────────────────────────────────────────
static void test_lr_schedule() {
    begin_test("LR schedule: warmup linear + exponential decay (§4.1)");
    RWKVTrainConfig tcfg;
    tcfg.lr          = 6e-4;
    tcfg.end_lr      = 1e-5;
    tcfg.warmup      = 100;
    tcfg.total_steps = 1000;

    // Warmup phase: step 0 → lr/warmup, step warmup-1 → lr
    double lr0  = rwkv_lr_schedule(0, tcfg);
    double lr49 = rwkv_lr_schedule(49, tcfg);
    double lr99 = rwkv_lr_schedule(99, tcfg);
    ASSERT_TRUE(std::fabs(lr0  - tcfg.lr / 100.0) < 1e-9, "step-0 LR wrong");
    ASSERT_TRUE(std::fabs(lr49 - tcfg.lr * 50.0/100.0) < 1e-9, "step-49 LR wrong");
    ASSERT_TRUE(std::fabs(lr99 - tcfg.lr) < 1e-9, "step-99 LR wrong");

    // After warmup: exponential decay, decreasing toward end_lr
    double lr100 = rwkv_lr_schedule(100, tcfg);
    double lr500 = rwkv_lr_schedule(500, tcfg);
    double lr999 = rwkv_lr_schedule(999, tcfg);
    ASSERT_TRUE(lr100 <= tcfg.lr,         "post-warmup LR should not exceed peak");
    ASSERT_TRUE(lr500 < lr100,            "LR should decrease during exp decay");
    ASSERT_TRUE(lr999 >= tcfg.end_lr - 1e-9, "LR should not go below end_lr");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Checkpoint save/load round-trip
// ─────────────────────────────────────────────────────────────────────────────
static void test_checkpoint() {
    begin_test("checkpoint save/load round-trip (§4.1)");
    auto cfg   = tiny_cfg();
    auto model = RWKVModel(cfg);
    model->eval();

    const std::string path = "/tmp/rwkv_test_ckpt.pt";
    torch::save(model, path);

    auto model2 = RWKVModel(cfg);
    torch::load(model2, path);
    model2->eval();
    std::filesystem::remove(path);

    torch::NoGradGuard ng;
    auto tokens  = torch::randint(0, cfg.vocab_size, {B, T},
                                  torch::TensorOptions().dtype(torch::kLong));
    auto logits1 = model->forward(tokens);
    auto logits2 = model2->forward(tokens);

    auto diff = (logits1 - logits2).abs().max().item<float>();
    ASSERT_TRUE(diff < 1e-4f, "logits differ after checkpoint round-trip");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. generate() returns correct shape
// ─────────────────────────────────────────────────────────────────────────────
static void test_generate_shape() {
    begin_test("generate() returns [B, max_new] shape (§3)");
    auto cfg   = tiny_cfg();
    auto model = RWKVModel(cfg);

    int64_t max_new = 5;
    auto prompt = torch::randint(0, cfg.vocab_size, {1, 4},
                                 torch::TensorOptions().dtype(torch::kLong));
    auto gen    = model->generate(prompt, max_new, 1.0, 0.9);

    ASSERT_TRUE(gen.size(0) == 1,       "batch dim wrong");
    ASSERT_TRUE(gen.size(1) == max_new, "max_new dim wrong");
    ASSERT_TRUE(!gen.isnan().any().item<bool>(), "NaN in generated tokens");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. time_decay and time_first are learnable parameters
// ─────────────────────────────────────────────────────────────────────────────
static void test_rwkv_params_learnable() {
    begin_test("time_decay w and time_first u are learnable params (§3.1.2)");
    auto cfg   = tiny_cfg();
    auto model = RWKVModel(cfg);
    model->train();

    // Run a forward + backward
    auto tokens = torch::randint(0, cfg.vocab_size, {1, T + 1},
                                 torch::TensorOptions().dtype(torch::kLong));
    auto inp    = tokens.slice(1, 0, T);
    auto tgt    = tokens.slice(1, 1, T + 1);
    auto logits = model->forward(inp);
    auto loss   = torch::nn::functional::cross_entropy(
        logits.reshape({-1, cfg.vocab_size}),
        tgt.reshape({-1}));
    loss.backward();

    // Check that time_decay and time_first have gradients
    bool found_td = false, found_tf = false;
    for (auto& item : model->named_parameters()) {
        if (item.key().find("time_decay") != std::string::npos &&
            item.value().grad().defined()) {
            found_td = true;
        }
        if (item.key().find("time_first") != std::string::npos &&
            item.value().grad().defined()) {
            found_tf = true;
        }
    }
    ASSERT_TRUE(found_td, "time_decay has no gradient");
    ASSERT_TRUE(found_tf, "time_first has no gradient");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    torch::manual_seed(42);

    test_forward_shape();
    test_timemix_shape();
    test_channelmix_shape();
    test_step_matches_forward();
    test_wkv_state_updates();
    test_token_shift();
    test_causal();
    test_param_count();
    test_gradient_flow();
    test_optimizer_step();
    test_lr_schedule();
    test_checkpoint();
    test_generate_shape();
    test_rwkv_params_learnable();

    std::printf("\n%d passed, %d failed\n", s_passed, s_failed);
    return s_failed ? 1 : 0;
}
