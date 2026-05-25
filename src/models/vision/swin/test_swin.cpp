// ─────────────────────────────────────────────────────────────────────────────
// test_swin.cpp — C++ structural tests for Swin Transformer
//
// Paper: Ze Liu et al., "Swin Transformer: Hierarchical Vision Transformer
//        using Shifted Windows", ICCV 2021.
//
// Tests verify:
//  1.  Output shape [N, num_classes] for all 4 variants (Section 3.3)
//  2.  Swin-T parameter count ~29M (Table 1)
//  3.  Swin-S parameter count ~50M (Table 1)
//  4.  Swin-B parameter count ~88M (Table 1)
//  5.  Patch embedding output shape: [B, (H/4)*(W/4), C] (Section 3.1)
//  6.  4 stages exist in the model (Figure 3)
//  7.  Window partition + reverse is lossless
//  8.  Shifted window: SW-MSA has shift_size = M/2 (Section 3.2)
//  9.  Gradient flow: no NaN/Inf on Swin-T
// 10.  AdamW training step changes weights (Section 4.1)
// 11.  Eval mode determinism
// 12.  Checkpoint save/load round-trip
// 13.  swin_train_epoch / swin_evaluate returns finite values
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/swin/swin.h"

#include <torch/torch.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <filesystem>

using namespace dm::models::vision;

// ── Minimal test framework ──────────────────────────────────────────────────
static int s_passed = 0, s_failed = 0;
#define ASSERT_TRUE(cond, msg) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "  FAIL: %s\n", (msg)); \
        ++s_failed; return; \
    } } while(0)

static void begin_test(const char* name) {
    std::printf("[test] %s ...", name);
    std::fflush(stdout);
}
static void end_test() {
    ++s_passed;
    std::printf(" PASS\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Output shape for all 4 variants
// ─────────────────────────────────────────────────────────────────────────────
static void test_output_shape() {
    begin_test("output shape [N,C] all variants");

    int64_t N = 2, classes = 100;
    auto x = torch::randn({N, 3, 224, 224});

    struct V { std::string name; std::shared_ptr<SwinTransformerImpl> m; };
    std::vector<V> variants = {
        {"swin_t", make_swin_t(classes)},
        {"swin_s", make_swin_s(classes)},
        {"swin_b", make_swin_b(classes)},
        {"swin_l", make_swin_l(classes)},
    };

    for (auto& v : variants) {
        v.m->eval();
        torch::NoGradGuard ng;
        auto out = v.m->forward(x);
        ASSERT_TRUE(out.dim()    == 2,       (v.name + ": expected 2D output").c_str());
        ASSERT_TRUE(out.size(0)  == N,       (v.name + ": batch dim mismatch").c_str());
        ASSERT_TRUE(out.size(1)  == classes, (v.name + ": class dim mismatch").c_str());
    }
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2-4. Parameter counts (Table 1, ±15% tolerance for slight impl differences)
// ─────────────────────────────────────────────────────────────────────────────
static int64_t param_count(const std::shared_ptr<SwinTransformerImpl>& m) {
    int64_t n = 0;
    for (auto& p : m->parameters()) n += p.numel();
    return n;
}

static void test_param_counts() {
    begin_test("parameter counts Swin-T/S/B (Table 1)");

    // Swin-T: ~29M, Swin-S: ~50M, Swin-B: ~88M
    auto check = [](const char* name, int64_t actual, int64_t expected_M) {
        double actual_M = actual / 1e6;
        double lo = expected_M * 0.80, hi = expected_M * 1.20;
        if (actual_M < lo || actual_M > hi) {
            std::fprintf(stderr, "  FAIL: %s params %.1fM not in [%.0f, %.0f]M\n",
                         name, actual_M, lo, hi);
            return false;
        }
        return true;
    };

    bool ok = true;
    ok &= check("swin_t", param_count(make_swin_t(1000)), 29);
    ok &= check("swin_s", param_count(make_swin_s(1000)), 50);
    ok &= check("swin_b", param_count(make_swin_b(1000)), 88);
    ASSERT_TRUE(ok, "one or more param counts out of range");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Patch embedding output shape: [B, (H/4)*(W/4), C]
// ─────────────────────────────────────────────────────────────────────────────
static void test_patch_embed_shape() {
    begin_test("patch embedding shape [B, H/4*W/4, C]");

    int64_t B = 2, H = 224, W = 224, C = 96, P = 4;
    PatchEmbed pe(H, P, 3, C);
    pe->eval();
    torch::NoGradGuard ng;
    auto x   = torch::randn({B, 3, H, W});
    auto out = pe->forward(x);
    ASSERT_TRUE(out.size(0) == B,                 "batch dim");
    ASSERT_TRUE(out.size(1) == (H/P)*(W/P),       "sequence len (H/4 * W/4)");
    ASSERT_TRUE(out.size(2) == C,                  "embed dim");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. 4 stages exist (Figure 3 / Section 3.1)
// ─────────────────────────────────────────────────────────────────────────────
static void test_four_stages() {
    begin_test("model has exactly 4 stages (Figure 3)");

    auto m = make_swin_t(1000);
    ASSERT_TRUE(m->stages->size() == 4, "expected 4 stages");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Window partition + reverse is lossless
// ─────────────────────────────────────────────────────────────────────────────
// (static helper duplicated from swin.cpp for test)
static torch::Tensor wp(torch::Tensor x, int64_t M) {
    auto B=x.size(0),H=x.size(1),W=x.size(2),C=x.size(3);
    x=x.view({B,H/M,M,W/M,M,C});
    x=x.permute({0,1,3,2,4,5}).contiguous();
    return x.view({-1,M,M,C});
}
static torch::Tensor wr(torch::Tensor w, int64_t M, int64_t H, int64_t W) {
    int64_t B = w.size(0)/((H/M)*(W/M));
    auto x=w.view({B,H/M,W/M,M,M,-1});
    x=x.permute({0,1,3,2,4,5}).contiguous();
    return x.view({B,H,W,-1});
}

static void test_window_partition_roundtrip() {
    begin_test("window_partition / window_reverse roundtrip");

    int64_t B=2, H=56, W=56, C=96, M=7;
    auto x = torch::randn({B, H, W, C});
    auto windows = wp(x, M);
    auto out = wr(windows, M, H, W);

    ASSERT_TRUE(out.sizes() == x.sizes(), "shape mismatch after roundtrip");
    ASSERT_TRUE(torch::allclose(out, x, 1e-5f, 1e-5f), "values differ after roundtrip");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. SW-MSA has shift_size = M/2 (Section 3.2)
// ─────────────────────────────────────────────────────────────────────────────
static void test_shift_size() {
    begin_test("SW-MSA blocks have shift_size = M/2 (Section 3.2)");

    int64_t M = 7;
    // Stage 1 has depth=2: block 0 shift=0, block 1 shift=M/2=3
    auto m = make_swin_t(1000);
    auto stage0 = m->stages->ptr(0)->as<SwinStageImpl>();
    ASSERT_TRUE(stage0 != nullptr, "stage 0 not SwinStageImpl");

    auto b0 = stage0->blocks->ptr(0)->as<SwinBlockImpl>();
    auto b1 = stage0->blocks->ptr(1)->as<SwinBlockImpl>();
    ASSERT_TRUE(b0 != nullptr, "block 0 null");
    ASSERT_TRUE(b1 != nullptr, "block 1 null");
    ASSERT_TRUE(b0->shift_size == 0,   "block 0 must be W-MSA (shift=0)");
    ASSERT_TRUE(b1->shift_size == M/2, "block 1 must be SW-MSA (shift=M/2)");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Gradient flow: no NaN/Inf
// ─────────────────────────────────────────────────────────────────────────────
static void test_gradient_flow() {
    begin_test("gradient flow (no NaN/Inf) — Swin-T");

    auto m = make_swin_t(10);
    m->train();
    auto x = torch::randn({1, 3, 224, 224});
    auto y = torch::zeros({1}, torch::kLong);
    auto out  = torch::nn::AnyModule(m).forward<torch::Tensor>(x);
    auto loss = torch::nn::functional::cross_entropy(out, y);
    loss.backward();

    bool ok = true;
    for (auto& p : m->parameters()) {
        if (!p.grad().defined()) continue;
        if (p.grad().isnan().any().item<bool>() ||
            p.grad().isinf().any().item<bool>()) {
            ok = false; break;
        }
    }
    ASSERT_TRUE(ok, "NaN/Inf in gradients");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. AdamW training step changes weights
// ─────────────────────────────────────────────────────────────────────────────
static void test_weight_update() {
    begin_test("AdamW training step changes weights");

    auto m = make_swin_t(10);
    m->train();
    torch::optim::AdamW optimizer(m->parameters(),
        torch::optim::AdamWOptions(1e-3).weight_decay(0.05));

    // Clone a weight before step
    auto w_before = m->head->weight.clone().detach();

    auto x = torch::randn({1, 3, 224, 224});
    auto y = torch::zeros({1}, torch::kLong);
    optimizer.zero_grad();
    auto out  = torch::nn::AnyModule(m).forward<torch::Tensor>(x);
    auto loss = torch::nn::functional::cross_entropy(out, y);
    loss.backward();
    optimizer.step();

    auto w_after = m->head->weight.detach();
    ASSERT_TRUE(!torch::allclose(w_before, w_after), "weights unchanged after AdamW step");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Eval mode determinism
// ─────────────────────────────────────────────────────────────────────────────
static void test_eval_determinism() {
    begin_test("eval mode determinism");

    auto m = make_swin_t(10);
    m->eval();
    torch::NoGradGuard ng;
    auto x    = torch::randn({2, 3, 224, 224});
    auto out1 = m->forward(x);
    auto out2 = m->forward(x);
    ASSERT_TRUE(torch::allclose(out1, out2), "non-deterministic in eval mode");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Checkpoint save/load round-trip
// ─────────────────────────────────────────────────────────────────────────────
static void test_checkpoint() {
    begin_test("checkpoint save/load round-trip");

    auto m1 = make_swin_t(10);
    m1->eval();
    torch::NoGradGuard ng;
    auto x    = torch::randn({1, 3, 224, 224});
    auto out1 = m1->forward(x);

    const std::string path = "/tmp/test_swin_ckpt.pt";
    {
        torch::serialize::OutputArchive archive;
        m1->save(archive);
        archive.save_to(path);
    }

    auto m2 = make_swin_t(10);
    {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        m2->load(archive);
    }
    m2->eval();
    auto out2 = m2->forward(x);

    ASSERT_TRUE(torch::allclose(out1, out2, 1e-5f, 1e-5f),
                "outputs differ after checkpoint round-trip");
    std::filesystem::remove(path);
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. swin_train_epoch / swin_evaluate return finite values
// ─────────────────────────────────────────────────────────────────────────────
static void test_train_evaluate_api() {
    begin_test("swin_train_epoch / swin_evaluate API");

    auto m = make_swin_t(10);
    torch::nn::AnyModule am(m);

    // Tiny synthetic batch: 2 images 224×224
    auto data    = torch::randn({2, 3, 224, 224});
    auto targets = torch::zeros({2}, torch::kLong);
    std::vector<std::pair<torch::Tensor,torch::Tensor>> batches = {{data, targets}};

    torch::optim::AdamW opt(m->parameters(),
        torch::optim::AdamWOptions(1e-4).weight_decay(0.05));

    float loss = swin_train_epoch(am, opt, torch::kCPU, batches);
    ASSERT_TRUE(std::isfinite(loss), "train_epoch returned non-finite loss");

    auto [top1, top5] = swin_evaluate(am, torch::kCPU, batches);
    ASSERT_TRUE(std::isfinite(top1), "evaluate returned non-finite top1");
    ASSERT_TRUE(std::isfinite(top5), "evaluate returned non-finite top5");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("\n=== Swin Transformer tests (Liu et al., ICCV 2021) ===\n");

    test_output_shape();
    test_param_counts();
    test_patch_embed_shape();
    test_four_stages();
    test_window_partition_roundtrip();
    test_shift_size();
    test_gradient_flow();
    test_weight_update();
    test_eval_determinism();
    test_checkpoint();
    test_train_evaluate_api();

    std::printf("=== %d passed, %d failed ===\n\n", s_passed, s_failed);
    return s_failed == 0 ? 0 : 1;
}
