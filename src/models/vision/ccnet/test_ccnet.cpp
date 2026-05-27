// ─────────────────────────────────────────────────────────────────────────────
// CCNet — Unit Tests
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/ccnet/ccnet.h"

#include <torch/torch.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace dm::models::vision;

static int pass_count = 0;
static int fail_count = 0;

#define TCHECK(cond, name) do { \
    if (cond) { std::cout << "[PASS] " << (name) << "\n"; ++pass_count; } \
    else { std::cout << "[FAIL] " << (name) << "\n"; ++fail_count; } \
} while(0)

// ─────────────────────────────────────────────────────────────────────────────
// Minimal backbone for forward-pass tests
// ─────────────────────────────────────────────────────────────────────────────

struct TinyBackboneImpl : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};
    TinyBackboneImpl(int64_t out_ch) {
        conv = register_module("conv",
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(3, out_ch, 3).padding(1).bias(false)));
    }
    torch::Tensor forward(torch::Tensor x) {
        return torch::relu(conv->forward(x));
    }
};
TORCH_MODULE(TinyBackbone);

// ─────────────────────────────────────────────────────────────────────────────
// CrissCrossAttention primitive
// ─────────────────────────────────────────────────────────────────────────────

static void test_cca_output_shape() {
    dm::prim::CrissCrossAttention cca(64, 8);
    auto x = torch::randn({2, 64, 12, 15});
    auto y = cca->forward(x);
    TCHECK(y.sizes() == x.sizes(), "CCA output shape matches input");
}

static void test_cca_square_spatial() {
    dm::prim::CrissCrossAttention cca(32, 4);
    auto x = torch::randn({1, 32, 8, 8});
    auto y = cca->forward(x);
    TCHECK(y.sizes() == x.sizes(), "CCA square spatial shape");
}

static void test_cca_batch() {
    dm::prim::CrissCrossAttention cca(16, 4);
    auto x = torch::randn({4, 16, 6, 6});
    auto y = cca->forward(x);
    TCHECK(y.size(0) == 4 && y.size(1) == 16, "CCA batch size preserved");
}

static void test_cca_gradient() {
    dm::prim::CrissCrossAttention cca(16, 4);
    auto x = torch::randn({1, 16, 5, 5}).requires_grad_(true);
    auto y = cca->forward(x);
    y.sum().backward();
    TCHECK(x.grad().defined() && !x.grad().isnan().any().template item<bool>(),
          "CCA gradient flows");
}

static void test_cca_eval_mode() {
    dm::prim::CrissCrossAttention cca(32, 8);
    cca->eval();
    auto x = torch::randn({1, 32, 8, 8});
    bool ok = true;
    try { cca->forward(x); } catch (...) { ok = false; }
    TCHECK(ok, "CCA eval mode no throw");
}

static void test_cca_independent_params() {
    dm::prim::CrissCrossAttention cca1(32, 4);
    dm::prim::CrissCrossAttention cca2(32, 4);
    for (auto& p : cca1->parameters()) p.data().fill_(0.0f);
    bool has_nonzero = false;
    for (auto& p : cca2->parameters())
        if (p.abs().max().toType(torch::kFloat).template item<float>() > 1e-6f) { has_nonzero = true; break; }
    TCHECK(has_nonzero, "CCA instances have independent parameters");
}

// ─────────────────────────────────────────────────────────────────────────────
// RCCAModule
// ─────────────────────────────────────────────────────────────────────────────

static void test_rcca_output_shape() {
    RCCAModule rcca(64, 8, 2);
    auto x = torch::randn({2, 64, 12, 12});
    auto y = rcca->forward(x);
    TCHECK(y.sizes() == x.sizes(), "RCCA output shape");
}

static void test_rcca_single_loop() {
    RCCAModule rcca(32, 4, 1);
    auto x = torch::randn({1, 32, 8, 8});
    auto y = rcca->forward(x);
    TCHECK(y.sizes() == x.sizes(), "RCCA R=1 shape");
}

static void test_rcca_shared_params() {
    // R=2 shares CCA params — param count must equal R=1
    RCCAModule rcca1(32, 4, 1);
    RCCAModule rcca2(32, 4, 2);
    int64_t p1 = 0, p2 = 0;
    for (auto& p : rcca1->parameters()) p1 += p.numel();
    for (auto& p : rcca2->parameters()) p2 += p.numel();
    TCHECK(p1 == p2, "RCCA R=2 has same param count as R=1 (shared params)");
}

static void test_rcca_gradient() {
    RCCAModule rcca(16, 4, 2);
    auto x = torch::randn({1, 16, 6, 6}).requires_grad_(true);
    auto y = rcca->forward(x);
    y.sum().backward();
    TCHECK(x.grad().defined() && !x.grad().isnan().any().template item<bool>(),
          "RCCA gradient flows");
}

// ─────────────────────────────────────────────────────────────────────────────
// CCNetHead
// ─────────────────────────────────────────────────────────────────────────────

static void test_head_output_shape() {
    CCNetHead head(1536, 512, 19);
    auto x = torch::randn({2, 1536, 8, 8});
    auto y = head->forward(x);
    TCHECK(y.size(0) == 2 && y.size(1) == 19 &&
          y.size(2) == 8 && y.size(3) == 8, "Head output shape");
}

static void test_head_gradient() {
    CCNetHead head(512, 256, 10);
    auto x = torch::randn({1, 512, 4, 4}).requires_grad_(true);
    auto y = head->forward(x);
    y.sum().backward();
    TCHECK(x.grad().defined(), "Head gradient flows");
}

// ─────────────────────────────────────────────────────────────────────────────
// CCNet (full model)
// ─────────────────────────────────────────────────────────────────────────────

static void test_ccnet_output_shape() {
    auto bb = TinyBackbone(512);
    CCNet model(torch::nn::AnyModule(bb), 512, 19, 256, 32, 256, 2);
    model->eval();
    auto x = torch::randn({1, 3, 16, 16});
    auto y = model->forward(x);
    TCHECK(y.size(0) == 1 && y.size(1) == 19 &&
          y.size(2) == 16 && y.size(3) == 16, "CCNet output shape [1,19,16,16]");
}

static void test_ccnet_batch2() {
    auto bb = TinyBackbone(512);
    CCNet model(torch::nn::AnyModule(bb), 512, 19, 256, 32, 256, 2);
    model->eval();
    auto x = torch::randn({2, 3, 16, 16});
    auto y = model->forward(x);
    TCHECK(y.size(0) == 2, "CCNet batch=2 preserved");
}

static void test_ccnet_num_classes() {
    auto bb = TinyBackbone(512);
    CCNet model(torch::nn::AnyModule(bb), 512, 150, 256, 32, 256, 2);
    model->eval();
    auto x = torch::randn({1, 3, 8, 8});
    auto y = model->forward(x);
    TCHECK(y.size(1) == 150, "CCNet num_classes=150");
}

static void test_ccnet_gradient() {
    auto bb = TinyBackbone(512);
    CCNet model(torch::nn::AnyModule(bb), 512, 19, 256, 32, 256, 2);
    auto x = torch::randn({1, 3, 8, 8}).requires_grad_(true);
    auto y = model->forward(x);
    y.sum().backward();
    TCHECK(x.grad().defined() && !x.grad().isnan().any().template item<bool>(),
          "CCNet gradient flows");
}

static void test_ccnet_no_nan() {
    auto bb = TinyBackbone(512);
    CCNet model(torch::nn::AnyModule(bb), 512, 19, 256, 32, 256, 2);
    model->eval();
    auto x = torch::randn({1, 3, 16, 16});
    auto y = model->forward(x);
    TCHECK(!y.isnan().any().template item<bool>() && !y.isinf().any().template item<bool>(),
          "CCNet output has no NaN/Inf");
}

static void test_ccnet_train_eval_shape() {
    auto bb = TinyBackbone(512);
    CCNet model(torch::nn::AnyModule(bb), 512, 19, 256, 32, 256, 2);
    auto x = torch::randn({1, 3, 8, 8});
    model->train();
    auto yt = model->forward(x);
    model->eval();
    auto ye = model->forward(x);
    TCHECK(yt.sizes() == ye.sizes(), "CCNet train/eval output shape consistent");
}

// ─────────────────────────────────────────────────────────────────────────────
// Category Consistent Loss
// ─────────────────────────────────────────────────────────────────────────────

static void test_ccl_scalar() {
    auto feat = torch::randn({2, 16, 8, 8});
    auto tgt  = torch::randint(0, 5, {2, 8, 8});
    auto loss = ccl_loss(feat, tgt);
    TCHECK(loss.dim() == 0, "CCL returns scalar");
}

static void test_ccl_nonnegative() {
    auto feat = torch::randn({2, 16, 6, 6});
    auto tgt  = torch::randint(0, 4, {2, 6, 6});
    auto loss = ccl_loss(feat, tgt);
    TCHECK(loss.template item<float>() >= 0.0f, "CCL non-negative");
}

static void test_ccl_gradient() {
    auto feat = torch::randn({2, 16, 6, 6}).requires_grad_(true);
    auto tgt  = torch::randint(0, 4, {2, 6, 6});
    auto loss = ccl_loss(feat, tgt);
    loss.backward();
    TCHECK(feat.grad().defined() && !feat.grad().isnan().any().template item<bool>(),
          "CCL gradient flows");
}

static void test_ccl_all_ignore() {
    auto feat = torch::randn({2, 16, 4, 4});
    auto tgt  = torch::full({2, 4, 4}, 255, torch::kLong);
    auto loss = ccl_loss(feat, tgt, CCLLossOptions{}, 255);
    TCHECK(std::abs(loss.template item<float>()) < 1e-6f, "CCL all-ignore returns 0");
}

static void test_ccl_single_class() {
    auto feat = torch::randn({1, 16, 4, 4});
    auto tgt  = torch::zeros({1, 4, 4}, torch::kLong);
    bool ok = true;
    try { ccl_loss(feat, tgt); } catch (...) { ok = false; }
    TCHECK(ok, "CCL single class no throw");
}

static void test_ccl_multi_class() {
    auto feat = torch::randn({2, 16, 8, 8});
    auto tgt  = torch::randint(0, 5, {2, 8, 8});
    bool ok = true;
    try { ccl_loss(feat, tgt); } catch (...) { ok = false; }
    TCHECK(ok, "CCL multi-class no throw");
}

static void test_ccl_zero_weights() {
    auto feat = torch::randn({1, 16, 4, 4});
    auto tgt  = torch::randint(0, 3, {1, 4, 4});
    CCLLossOptions opts;
    opts.alpha = 0.0f; opts.beta = 0.0f; opts.gamma = 0.0f;
    auto loss = ccl_loss(feat, tgt, opts);
    TCHECK(std::abs(loss.template item<float>()) < 1e-5f, "CCL zero weights = 0 loss");
}

static void test_ccl_large_delta_v() {
    // delta_v >> feature norms → l_var = 0, isolate by zeroing beta/gamma
    auto feat = torch::randn({1, 16, 4, 4});
    auto tgt  = torch::randint(0, 3, {1, 4, 4});
    CCLLossOptions opts;
    opts.delta_v = 1e6f; opts.delta_d = 2e6f;
    opts.beta = 0.0f; opts.gamma = 0.0f;
    auto loss = ccl_loss(feat, tgt, opts);
    TCHECK(std::abs(loss.template item<float>()) < 1e-3f, "CCL large delta_v gives ~0 l_var");
}

static void test_ccl_mixed_ignore() {
    auto feat = torch::randn({1, 16, 4, 4});
    auto tgt  = torch::zeros({1, 4, 4}, torch::kLong);
    tgt.slice(2, 0, 2).fill_(255);   // first two columns ignored
    bool ok = true;
    try { ccl_loss(feat, tgt, CCLLossOptions{}, 255); } catch (...) { ok = false; }
    TCHECK(ok, "CCL mixed ignore/valid no throw");
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== CCNet Tests ===\n";

    // CrissCrossAttention
    test_cca_output_shape();
    test_cca_square_spatial();
    test_cca_batch();
    test_cca_gradient();
    test_cca_eval_mode();
    test_cca_independent_params();

    // RCCAModule
    test_rcca_output_shape();
    test_rcca_single_loop();
    test_rcca_shared_params();
    test_rcca_gradient();

    // CCNetHead
    test_head_output_shape();
    test_head_gradient();

    // CCNet
    test_ccnet_output_shape();
    test_ccnet_batch2();
    test_ccnet_num_classes();
    test_ccnet_gradient();
    test_ccnet_no_nan();
    test_ccnet_train_eval_shape();

    // CCL Loss
    test_ccl_scalar();
    test_ccl_nonnegative();
    test_ccl_gradient();
    test_ccl_all_ignore();
    test_ccl_single_class();
    test_ccl_multi_class();
    test_ccl_zero_weights();
    test_ccl_large_delta_v();
    test_ccl_mixed_ignore();

    std::cout << "\n" << pass_count << " passed, " << fail_count << " failed\n";
    return fail_count == 0 ? 0 : 1;
}
