// ─────────────────────────────────────────────────────────────────────────────
// test_mobilenet.cpp — C++ structural tests for MobileNet v1
//
// Paper: A.G. Howard, M. Zhu, B. Chen et al.,
//        "MobileNets: Efficient Convolutional Neural Networks for Mobile
//         Vision Applications", arXiv:1704.04861v1, 2017
//
// Tests verify:
//  1.  DS block output shape + stride=1  (Section 3.1)
//  2.  DS block stride=2 halves spatial dims  (Section 3.1)
//  3.  MobileNet (α=1) output shape [B, 1000]  (Table 1)
//  4.  α=1.0 parameter count ≈ 4.2M  (Table 1)
//  5.  α=0.75 parameter count ≈ 2.6M  (Table 6)
//  6.  α=0.5  parameter count ≈ 1.3M  (Table 6)
//  7.  α=0.25 parameter count ≈ 0.5M  (Table 6)
//  8.  Width multiplier α scales channels quadratically (Section 3.3, Eq. 6)
//  9.  DW conv uses groups=in_ch (one filter per channel)  (Section 3.1)
// 10.  Depthwise filters have no bias (BN absorbs bias)  (Section 3.2)
// 11.  Gradient flow: no NaN/Inf  (Section 3.2)
// 12.  RMSprop step changes weights  (Section 3.2)
// 13.  Checkpoint save/load round-trip
// 14.  All factory variants produce correct output shape
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/mobilenet/mobilenet.h"

#include <torch/torch.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>

using namespace dm::models::vision;

static int s_passed = 0, s_failed = 0;
#define ASSERT_TRUE(cond, msg) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "  FAIL: %s\n", (msg)); \
        ++s_failed; return; \
    } } while(0)
static void begin_test(const char* n) {
    std::printf("[test] %s ...", n); std::fflush(stdout);
}
static void end_test() { ++s_passed; std::printf(" PASS\n"); }

static const int64_t B = 2;
static const int64_t C = 10;   // small num_classes for fast tests

// ─────────────────────────────────────────────────────────────────────────────
// 1. DS block stride=1: output spatial unchanged
// ─────────────────────────────────────────────────────────────────────────────
static void test_ds_block_stride1() {
    begin_test("DS block stride=1 preserves spatial dims (Section 3.1)");
    auto block = DepthwiseSeparableBlock(32, 64, 1);
    block->eval();
    torch::NoGradGuard ng;
    auto x   = torch::rand({B, 32, 56, 56});
    auto out = block->forward(x);
    ASSERT_TRUE(out.sizes() == torch::IntArrayRef({B, 64, 56, 56}),
                "DS block stride=1 shape wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. DS block stride=2 halves spatial dims (Section 3.1)
// ─────────────────────────────────────────────────────────────────────────────
static void test_ds_block_stride2() {
    begin_test("DS block stride=2 halves spatial (Section 3.1)");
    auto block = DepthwiseSeparableBlock(64, 128, 2);
    block->eval();
    torch::NoGradGuard ng;
    auto x   = torch::rand({B, 64, 112, 112});
    auto out = block->forward(x);
    ASSERT_TRUE(out.sizes() == torch::IntArrayRef({B, 128, 56, 56}),
                "DS block stride=2 shape wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. MobileNet (α=1) output shape [B, 1000] (Table 1)
// ─────────────────────────────────────────────────────────────────────────────
static void test_output_shape() {
    begin_test("MobileNet α=1 output [B, 1000] (Table 1)");
    auto model = make_mobilenet_1_0(1000);
    model->eval();
    torch::NoGradGuard ng;
    auto x   = torch::rand({B, 3, 224, 224});
    auto out = model->forward(x);
    ASSERT_TRUE(out.sizes() == torch::IntArrayRef({B, 1000}),
                "output shape wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. α=1.0: params ≈ 4.2M  (Table 1)
// ─────────────────────────────────────────────────────────────────────────────
static void test_params_1_0() {
    begin_test("α=1.0 params ≈ 4.2M (Table 1)");
    auto model = make_mobilenet_1_0(1000);
    int64_t n = 0;
    for (auto& p : model->parameters()) n += p.numel();
    // Accept range [3.8M, 4.6M]
    ASSERT_TRUE(n >= 3'800'000 && n <= 4'600'000,
                "α=1.0 param count out of expected range [3.8M, 4.6M]");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. α=0.75: params ≈ 2.6M  (Table 6)
// ─────────────────────────────────────────────────────────────────────────────
static void test_params_0_75() {
    begin_test("α=0.75 params ≈ 2.6M (Table 6)");
    auto model = make_mobilenet_0_75(1000);
    int64_t n = 0;
    for (auto& p : model->parameters()) n += p.numel();
    ASSERT_TRUE(n >= 2'200'000 && n <= 3'000'000,
                "α=0.75 param count out of expected range [2.2M, 3.0M]");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. α=0.5: params ≈ 1.3M  (Table 6)
// ─────────────────────────────────────────────────────────────────────────────
static void test_params_0_5() {
    begin_test("α=0.5 params ≈ 1.3M (Table 6)");
    auto model = make_mobilenet_0_5(1000);
    int64_t n = 0;
    for (auto& p : model->parameters()) n += p.numel();
    ASSERT_TRUE(n >= 1'000'000 && n <= 1'600'000,
                "α=0.5 param count out of expected range [1.0M, 1.6M]");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. α=0.25: params ≈ 0.5M  (Table 6)
// ─────────────────────────────────────────────────────────────────────────────
static void test_params_0_25() {
    begin_test("α=0.25 params ≈ 0.5M (Table 6)");
    auto model = make_mobilenet_0_25(1000);
    int64_t n = 0;
    for (auto& p : model->parameters()) n += p.numel();
    ASSERT_TRUE(n >= 350'000 && n <= 700'000,
                "α=0.25 param count out of expected range [350K, 700K]");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Width multiplier α reduces params roughly quadratically  (Section 3.3)
//    params(α=0.5) ≈ params(α=1.0) / 4
// ─────────────────────────────────────────────────────────────────────────────
static void test_alpha_quadratic() {
    begin_test("α scales params ~quadratically (Section 3.3)");
    auto count_params = [](float a) {
        auto m = MobileNet(1000, a);
        int64_t n = 0;
        for (auto& p : m->parameters()) n += p.numel();
        return n;
    };
    double p1_0 = (double)count_params(1.00f);
    double p0_5 = (double)count_params(0.50f);
    double ratio = p1_0 / p0_5;
    // Expect ratio ≈ 4 (±25%)
    ASSERT_TRUE(ratio >= 3.0 && ratio <= 5.5,
                "α=0.5 should give ~1/4 params of α=1.0");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. DW conv uses groups = in_ch (one filter per channel)  (Section 3.1, Eq. 3)
// ─────────────────────────────────────────────────────────────────────────────
static void test_dw_groups() {
    begin_test("DW conv groups = in_ch (Section 3.1)");
    auto block = DepthwiseSeparableBlock(64, 128, 1);
    auto dw_groups = block->dw->options.groups();
    ASSERT_TRUE(dw_groups == 64, "DW groups != in_ch");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. DW and PW convs have no bias (BN subsumes bias)  (Section 3.2)
// ─────────────────────────────────────────────────────────────────────────────
static void test_no_bias() {
    begin_test("DW and PW convs have no bias (Section 3.2)");
    auto block = DepthwiseSeparableBlock(32, 64, 1);
    // bias() returns bool in this LibTorch version
    ASSERT_TRUE(!block->dw->options.bias(), "DW conv should have no bias");
    ASSERT_TRUE(!block->pw->options.bias(), "PW conv should have no bias");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Gradient flow: no NaN/Inf  (Section 3.2)
// ─────────────────────────────────────────────────────────────────────────────
static void test_gradient_flow() {
    begin_test("gradient flow: no NaN/Inf (Section 3.2)");
    auto model = MobileNet(C, 1.0f);
    model->train();
    auto x      = torch::rand({B, 3, 224, 224});
    auto labels = torch::randint(0, C, {B});
    auto logits = model->forward(x);
    auto loss   = torch::nn::functional::cross_entropy(logits, labels);
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
// 12. RMSprop step changes weights  (Section 3.2)
// ─────────────────────────────────────────────────────────────────────────────
static void test_rmsprop_update() {
    begin_test("RMSprop step changes weights (Section 3.2)");
    auto model = MobileNet(C, 1.0f);
    model->train();
    torch::optim::RMSprop opt(model->parameters(),
        torch::optim::RMSpropOptions(1e-3).momentum(0.9));

    // Snapshot first DS block DW weights
    auto w_before = model->features->ptr(3)
                        ->as<DepthwiseSeparableBlockImpl>()
                        ->dw->weight.clone().detach();

    auto x      = torch::rand({B, 3, 224, 224});
    auto labels = torch::randint(0, C, {B});
    auto logits = model->forward(x);
    auto loss   = torch::nn::functional::cross_entropy(logits, labels);
    opt.zero_grad();
    loss.backward();
    opt.step();

    auto w_after = model->features->ptr(3)
                       ->as<DepthwiseSeparableBlockImpl>()
                       ->dw->weight.detach();
    ASSERT_TRUE(!torch::allclose(w_before, w_after),
                "DW weights unchanged after RMSprop step");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Checkpoint save/load round-trip
// ─────────────────────────────────────────────────────────────────────────────
static void test_checkpoint() {
    begin_test("checkpoint save/load round-trip");
    auto m1 = MobileNet(C, 1.0f);
    m1->eval();
    torch::NoGradGuard ng;
    auto x  = torch::rand({1, 3, 224, 224});
    auto p1 = m1->forward(x);

    const std::string path = "/tmp/test_mobilenet_ckpt.pt";
    {
        torch::serialize::OutputArchive ar;
        m1->save(ar);
        ar.save_to(path);
    }
    auto m2 = MobileNet(C, 1.0f);
    {
        torch::serialize::InputArchive ar;
        ar.load_from(path);
        m2->load(ar);
    }
    m2->eval();
    auto p2 = m2->forward(x);
    ASSERT_TRUE(torch::allclose(p1, p2, 1e-5f, 1e-5f),
                "predictions differ after checkpoint reload");
    std::filesystem::remove(path);
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. All four factory variants produce correct output shape
// ─────────────────────────────────────────────────────────────────────────────
static void test_all_variants() {
    begin_test("all α variants produce [B, 1000] output");
    torch::NoGradGuard ng;
    auto x = torch::rand({1, 3, 224, 224});
    for (auto fn : {make_mobilenet_1_0, make_mobilenet_0_75,
                    make_mobilenet_0_5, make_mobilenet_0_25}) {
        auto m   = fn(1000);
        m->eval();
        auto out = m->forward(x);
        ASSERT_TRUE(out.sizes() == torch::IntArrayRef({1, 1000}),
                    "variant output shape wrong");
    }
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("\n=== MobileNet v1 tests (Howard et al., arXiv:1704.04861v1) ===\n");

    test_ds_block_stride1();
    test_ds_block_stride2();
    test_output_shape();
    test_params_1_0();
    test_params_0_75();
    test_params_0_5();
    test_params_0_25();
    test_alpha_quadratic();
    test_dw_groups();
    test_no_bias();
    test_gradient_flow();
    test_rmsprop_update();
    test_checkpoint();
    test_all_variants();

    std::printf("=== %d passed, %d failed ===\n\n", s_passed, s_failed);
    return s_failed == 0 ? 0 : 1;
}
