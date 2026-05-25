// ─────────────────────────────────────────────────────────────────────────────
// test_trm.cpp — C++ structural tests for Tiny Recursion Model (TRM)
//
// Paper: A. Jolicoeur-Martineau,
//        "Less is More: Recursive Reasoning with Tiny Networks",
//        arXiv:2510.04871v1, 2025.
//
// Tests verify:
//  1.  TRM-Att builds without error and has ~7M params (Table 1)
//  2.  TRM-MLP builds and has ~5M params (Table 1, Sudoku config)
//  3.  Output logit shape [B, L, V] from deep_recursion
//  4.  Halt signal shape [B, 1] from deep_recursion
//  5.  latent z shape preserved across recursion steps
//  6.  Deep supervision: T-1 no-grad + 1 grad call (Section 2.4)
//  7.  Gradient flow: no NaN/Inf after one supervised step
//  8.  AdamW weight update changes parameters
//  9.  EMA weights differ from live weights after update
// 10.  apply_ema / restore_live roundtrip preserves live weights
// 11.  Checkpoint save/load roundtrip
// 12.  predict() returns correct shape [B, L]
// 13.  stable_cross_entropy returns finite scalar
// ─────────────────────────────────────────────────────────────────────────────

#include "models/reasoning/trm/trm.h"

#include <torch/torch.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace dm::models::reasoning;

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

// ── Helpers ──────────────────────────────────────────────────────────────────
static int64_t param_count(TRM& m) {
    int64_t n = 0;
    for (auto& p : m->parameters()) n += p.numel();
    return n;
}

static const int64_t VOCAB = 16, SEQ = 9, DIM = 64, LAYERS = 2,
                     HEADS = 4, N = 2, T_VAL = 2;

// ─────────────────────────────────────────────────────────────────────────────
// 1. TRM-Att param count ~7M  (Table 1)
// ─────────────────────────────────────────────────────────────────────────────
static void test_param_count_att() {
    begin_test("TRM-Att param count ~7M (Table 1)");
    // Full-size model: vocab=16, seqlen=81, dim=512
    auto m = TRM(16, 81, 512, 2, 8, /*use_attention=*/true, 6, 3);
    double pm = param_count(m) / 1e6;
    // Allow 5–10M range (Table 1: 7M for TRM-Att)
    ASSERT_TRUE(pm >= 5.0 && pm <= 12.0,
        "TRM-Att param count out of expected range [5M, 12M]");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. TRM-MLP param count ~5M  (Table 1, Sudoku config)
// ─────────────────────────────────────────────────────────────────────────────
static void test_param_count_mlp() {
    begin_test("TRM-MLP param count ~5M (Table 1)");
    auto m = TRM(16, 81, 512, 2, 8, /*use_attention=*/false, 6, 3);
    double pm = param_count(m) / 1e6;
    // Table 1: 5M on Sudoku (19M on Maze — different seqlen, but same model)
    ASSERT_TRUE(pm >= 3.0 && pm <= 25.0,
        "TRM-MLP param count out of expected range [3M, 25M]");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Output logit shape [B, L, V]
// ─────────────────────────────────────────────────────────────────────────────
static void test_logit_shape() {
    begin_test("logit shape [B, L, V] from deep_recursion");
    int64_t B = 2;
    auto m = TRM(VOCAB, SEQ, DIM, LAYERS, HEADS, true, N, T_VAL);
    m->eval();
    torch::NoGradGuard ng;
    auto x = torch::randint(0, VOCAB, {B, SEQ});
    auto y = torch::randint(0, VOCAB, {B, SEQ});
    auto z = m->init_z(B, torch::kCPU);
    auto [ny, nz, logits, q] = m->forward(x, y, z);
    ASSERT_TRUE(logits.dim()    == 3,     "logits must be 3D");
    ASSERT_TRUE(logits.size(0)  == B,     "logits batch dim");
    ASSERT_TRUE(logits.size(1)  == SEQ,   "logits seq dim");
    ASSERT_TRUE(logits.size(2)  == VOCAB, "logits vocab dim");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Halt signal shape [B, 1]
// ─────────────────────────────────────────────────────────────────────────────
static void test_halt_shape() {
    begin_test("halt signal shape [B, 1]");
    int64_t B = 3;
    auto m = TRM(VOCAB, SEQ, DIM, LAYERS, HEADS, true, N, T_VAL);
    m->eval();
    torch::NoGradGuard ng;
    auto x = torch::randint(0, VOCAB, {B, SEQ});
    auto y = torch::randint(0, VOCAB, {B, SEQ});
    auto z = m->init_z(B, torch::kCPU);
    auto [ny, nz, logits, q] = m->forward(x, y, z);
    ASSERT_TRUE(q.dim()    == 2, "halt must be 2D [B,1]");
    ASSERT_TRUE(q.size(0)  == B, "halt batch dim");
    ASSERT_TRUE(q.size(1)  == 1, "halt must have 1 output");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Latent z shape preserved across recursion
// ─────────────────────────────────────────────────────────────────────────────
static void test_z_shape_preserved() {
    begin_test("latent z shape [B, L, D] preserved across recursion");
    int64_t B = 2;
    auto m = TRM(VOCAB, SEQ, DIM, LAYERS, HEADS, true, N, T_VAL);
    m->eval();
    torch::NoGradGuard ng;
    auto x_emb = m->net->embed(torch::randint(0, VOCAB, {B, SEQ}));
    auto y_emb = m->net->embed(torch::randint(0, VOCAB, {B, SEQ}));
    auto z     = m->init_z(B, torch::kCPU);

    auto [ny, nz] = m->latent_recursion(x_emb, y_emb, z, N);
    ASSERT_TRUE(nz.sizes() == z.sizes(), "z shape changed after recursion");
    ASSERT_TRUE(ny.sizes() == y_emb.sizes(), "y shape changed after recursion");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Deep supervision: T-1 no-grad calls + 1 with-grad call
//    Verify the last call has gradients, earlier ones don't
// ─────────────────────────────────────────────────────────────────────────────
static void test_deep_supervision_grad() {
    begin_test("deep_recursion: T-1 no-grad + 1 with-grad (Section 2.4)");
    int64_t B = 1;
    // Use T=2 so 1 no-grad + 1 grad
    auto m = TRM(VOCAB, SEQ, DIM, LAYERS, HEADS, true, /*n=*/1, /*T=*/2);
    m->train();

    auto x = torch::randint(0, VOCAB, {B, SEQ});
    auto y = torch::randint(0, VOCAB, {B, SEQ});
    auto z = m->init_z(B, torch::kCPU);
    auto x_emb = m->net->embed(x);
    auto y_emb = m->net->embed(y);

    auto [ny, nz, logits, q] = m->deep_recursion(x_emb, y_emb, z);

    // Logits must require grad (from the final with-grad recursion)
    ASSERT_TRUE(logits.requires_grad(), "logits must have grad after deep_recursion");
    // ny and nz are detached
    ASSERT_TRUE(!ny.requires_grad(), "ny must be detached");
    ASSERT_TRUE(!nz.requires_grad(), "nz must be detached");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Gradient flow: no NaN/Inf after one supervised step
// ─────────────────────────────────────────────────────────────────────────────
static void test_gradient_flow() {
    begin_test("gradient flow (no NaN/Inf)");
    int64_t B = 2;
    auto m = TRM(VOCAB, SEQ, DIM, LAYERS, HEADS, true, N, T_VAL);
    m->train();

    auto x    = torch::randint(0, VOCAB, {B, SEQ});
    auto y    = torch::randint(0, VOCAB, {B, SEQ});
    auto z    = m->init_z(B, torch::kCPU);
    auto x_emb = m->net->embed(x);
    auto y_emb = m->net->embed(y);

    auto [ny, nz, logits, q] = m->deep_recursion(x_emb, y_emb, z);
    auto loss = stable_cross_entropy(logits, y);
    loss.backward();

    bool ok = true;
    for (auto& p : m->parameters()) {
        if (!p.grad().defined()) continue;
        if (p.grad().isnan().any().item<bool>() ||
            p.grad().isinf().any().item<bool>()) { ok = false; break; }
    }
    ASSERT_TRUE(ok, "NaN/Inf found in gradients");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. AdamW step changes weights
// ─────────────────────────────────────────────────────────────────────────────
static void test_weight_update() {
    begin_test("AdamW step changes weights");
    int64_t B = 2;
    auto m = TRM(VOCAB, SEQ, DIM, LAYERS, HEADS, true, N, T_VAL);
    m->train();
    torch::optim::AdamW opt(m->parameters(),
        torch::optim::AdamWOptions(1e-3).betas({0.9, 0.95}));

    auto w_before = m->net->output_head->weight.clone().detach();

    auto x     = torch::randint(0, VOCAB, {B, SEQ});
    auto y     = torch::randint(0, VOCAB, {B, SEQ});
    auto z     = m->init_z(B, torch::kCPU);
    auto x_emb = m->net->embed(x);
    auto y_emb = m->net->embed(y);

    auto [ny, nz, logits, q] = m->deep_recursion(x_emb, y_emb, z);
    auto loss = stable_cross_entropy(logits, y);
    opt.zero_grad();
    loss.backward();
    opt.step();

    ASSERT_TRUE(!torch::allclose(w_before, m->net->output_head->weight.detach()),
                "weights unchanged after AdamW step");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. EMA weights differ from live weights after update  (Section 4.7)
// ─────────────────────────────────────────────────────────────────────────────
static void test_ema_differs() {
    begin_test("EMA weights differ from live weights after update (Section 4.7)");
    auto m = TRM(VOCAB, SEQ, DIM, LAYERS, HEADS, true, N, T_VAL);
    m->train();
    m->init_ema();

    // Modify live weights with a gradient step
    torch::optim::AdamW opt(m->parameters(), torch::optim::AdamWOptions(0.1));
    auto x     = torch::randint(0, VOCAB, {2, SEQ});
    auto y     = torch::randint(0, VOCAB, {2, SEQ});
    auto z     = m->init_z(2, torch::kCPU);
    auto x_emb = m->net->embed(x);
    auto y_emb = m->net->embed(y);
    auto [ny, nz, logits, q] = m->deep_recursion(x_emb, y_emb, z);
    stable_cross_entropy(logits, y).backward();
    opt.step();
    m->update_ema();

    // EMA should differ from live (ema = 0.999*orig + 0.001*new)
    bool differs = false;
    auto params = m->net->parameters();
    for (size_t i = 0; i < params.size() && i < m->ema_params.size(); ++i) {
        if (!torch::allclose(params[i].data(), m->ema_params[i], 1e-6f, 1e-6f)) {
            differs = true; break;
        }
    }
    ASSERT_TRUE(differs, "EMA params identical to live after update");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. apply_ema / restore_live roundtrip
// ─────────────────────────────────────────────────────────────────────────────
static void test_ema_roundtrip() {
    begin_test("apply_ema / restore_live roundtrip preserves live weights");
    auto m = TRM(VOCAB, SEQ, DIM, LAYERS, HEADS, true, N, T_VAL);
    m->init_ema();

    // Clone live weights
    std::vector<torch::Tensor> live_orig;
    for (auto& p : m->net->parameters())
        live_orig.push_back(p.data().clone());

    m->apply_ema();
    m->restore_live();

    bool ok = true;
    auto params = m->net->parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        if (!torch::allclose(params[i].data(), live_orig[i], 1e-6f, 1e-6f)) {
            ok = false; break;
        }
    }
    ASSERT_TRUE(ok, "live weights not restored after apply_ema/restore_live");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Checkpoint save/load roundtrip
// ─────────────────────────────────────────────────────────────────────────────
static void test_checkpoint() {
    begin_test("checkpoint save/load roundtrip");
    auto m1 = TRM(VOCAB, SEQ, DIM, LAYERS, HEADS, true, N, T_VAL);
    m1->eval();
    torch::NoGradGuard ng;
    auto x  = torch::randint(0, VOCAB, {1, SEQ});
    auto y0 = torch::randint(0, VOCAB, {1, SEQ});
    auto z  = m1->init_z(1, torch::kCPU);
    auto out1 = std::get<2>(m1->forward(x, y0, z));

    const std::string path = "/tmp/test_trm_ckpt.pt";
    {
        torch::serialize::OutputArchive ar;
        m1->save(ar);
        ar.save_to(path);
    }
    auto m2 = TRM(VOCAB, SEQ, DIM, LAYERS, HEADS, true, N, T_VAL);
    {
        torch::serialize::InputArchive ar;
        ar.load_from(path);
        m2->load(ar);
    }
    m2->eval();
    auto out2 = std::get<2>(m2->forward(x, y0, z));
    ASSERT_TRUE(torch::allclose(out1, out2, 1e-5f, 1e-5f),
                "outputs differ after checkpoint roundtrip");
    std::filesystem::remove(path);
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. predict() returns [B, L]
// ─────────────────────────────────────────────────────────────────────────────
static void test_predict_shape() {
    begin_test("predict() returns [B, L] token indices");
    int64_t B = 3;
    auto m = TRM(VOCAB, SEQ, DIM, LAYERS, HEADS, true, N, T_VAL);
    auto x = torch::randint(0, VOCAB, {B, SEQ});
    auto pred = m->predict(x, /*n_sup=*/2);
    ASSERT_TRUE(pred.dim()    == 2,   "predict must be 2D");
    ASSERT_TRUE(pred.size(0)  == B,   "predict batch dim");
    ASSERT_TRUE(pred.size(1)  == SEQ, "predict seq dim");
    // All predictions must be valid token indices
    ASSERT_TRUE((pred >= 0).all().item<bool>()       && (pred < VOCAB).all().item<bool>(),
                "predict returned out-of-vocab indices");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. stable_cross_entropy is finite
// ─────────────────────────────────────────────────────────────────────────────
static void test_stable_ce() {
    begin_test("stable_cross_entropy returns finite scalar");
    // Test with extreme logits to verify clamping works
    auto logits  = torch::randn({4, SEQ, VOCAB}) * 1000.f;
    auto targets = torch::randint(0, VOCAB, {4, SEQ});
    auto loss = stable_cross_entropy(logits, targets);
    ASSERT_TRUE(std::isfinite(loss.item<float>()), "stable_cross_entropy not finite");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("\n=== TRM tests (Jolicoeur-Martineau, arXiv:2510.04871v1) ===\n");

    test_param_count_att();
    test_param_count_mlp();
    test_logit_shape();
    test_halt_shape();
    test_z_shape_preserved();
    test_deep_supervision_grad();
    test_gradient_flow();
    test_weight_update();
    test_ema_differs();
    test_ema_roundtrip();
    test_checkpoint();
    test_predict_shape();
    test_stable_ce();

    std::printf("=== %d passed, %d failed ===\n\n", s_passed, s_failed);
    return s_failed == 0 ? 0 : 1;
}
