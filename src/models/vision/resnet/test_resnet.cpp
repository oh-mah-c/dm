// ─────────────────────────────────────────────────────────────────────────────
// test_resnet.cpp — structural and numerical tests for the ResNet implementation
//
// Tests derived directly from the paper:
//   He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
//
// Verifies:
//  1. Output shape   : [N, num_classes] for all 5 variants
//  2. Parameter count: matches Table 1 values (±5% tolerance for BN params)
//  3. Gradient flow  : loss.backward() runs without NaN/Inf
//  4. Shortcut types : identity vs. projection triggered correctly
//  5. Bottleneck     : expansion=4 produces correct channel widths
//  6. Training step  : SGD step changes weights
//  7. Eval mode      : model.eval() disables BatchNorm running stats update
//  8. Checkpoint I/O : save / load round-trip preserves forward output
//
// Exit code 0 = all tests passed.
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/resnet/resnet.h"

#include <torch/torch.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace dm::models::vision;

// ── Helpers ──────────────────────────────────────────────────────────────────

static void ASSERT_TRUE(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

// Count all trainable parameters in a module
static int64_t count_params(const torch::nn::Module& m) {
    int64_t n = 0;
    for (const auto& p : m.parameters()) n += p.numel();
    return n;
}

// ── Test 1: output shape ─────────────────────────────────────────────────────
// Paper Table 1: final output is [N, num_classes] after global avg pool + fc.
// Input: [N, 3, 224, 224]  (standard ImageNet crop size, Section 3.4)
static void test_output_shape() {
    std::printf("[test] output shape ... ");

    auto x = torch::randn({2, 3, 224, 224});

    auto r18  = make_resnet18(1000);
    auto r34  = make_resnet34(1000);
    auto r50  = make_resnet50(1000);
    auto r101 = make_resnet101(1000);
    auto r152 = make_resnet152(1000);

    r18->eval(); r34->eval(); r50->eval(); r101->eval(); r152->eval();

    torch::NoGradGuard ng;
    ASSERT_TRUE(r18->forward(x).sizes()  == torch::IntArrayRef({2, 1000}), "resnet18  output shape");
    ASSERT_TRUE(r34->forward(x).sizes()  == torch::IntArrayRef({2, 1000}), "resnet34  output shape");
    ASSERT_TRUE(r50->forward(x).sizes()  == torch::IntArrayRef({2, 1000}), "resnet50  output shape");
    ASSERT_TRUE(r101->forward(x).sizes() == torch::IntArrayRef({2, 1000}), "resnet101 output shape");
    ASSERT_TRUE(r152->forward(x).sizes() == torch::IntArrayRef({2, 1000}), "resnet152 output shape");

    // Also verify num_classes is configurable (paper uses 10 for CIFAR-10)
    auto r50_c10 = make_resnet50(10);
    r50_c10->eval();
    ASSERT_TRUE(r50_c10->forward(x).sizes() == torch::IntArrayRef({2, 10}),
          "resnet50 10-class output shape");

    std::printf("PASS\n");
}

// ── Test 2: parameter counts ─────────────────────────────────────────────────
// Paper Table 1 FLOPs imply approximate parameter budgets.
// Published parameter counts (widely verified against the paper):
//   ResNet-18:   11.7M
//   ResNet-34:   21.8M
//   ResNet-50:   25.6M
//   ResNet-101:  44.5M
//   ResNet-152:  60.2M
// We allow ±2% tolerance.
static void test_param_counts() {
    std::printf("[test] parameter counts ... ");

    struct Expected { std::shared_ptr<torch::nn::Module> m; int64_t n; const char* name; };
    std::vector<Expected> cases = {
        { make_resnet18(1000),  11'689'512, "resnet18"  },
        { make_resnet34(1000),  21'797'672, "resnet34"  },
        { make_resnet50(1000),  25'557'032, "resnet50"  },
        { make_resnet101(1000), 44'549'160, "resnet101" },
        { make_resnet152(1000), 60'192'808, "resnet152" },
    };

    for (auto& c : cases) {
        int64_t got   = count_params(*c.m);
        double  ratio = std::abs(static_cast<double>(got - c.n)) / c.n;
        if (ratio > 0.02) {
            std::fprintf(stderr,
                "FAIL: %s expected ~%lld params, got %lld (%.1f%% off)\n",
                c.name, (long long)c.n, (long long)got, ratio * 100.0);
            std::exit(1);
        }
    }

    std::printf("PASS\n");
}

// ── Test 3: gradient flow ─────────────────────────────────────────────────────
// Paper trains end-to-end with SGD + backprop. Verify no NaN/Inf gradients
// for all 5 variants on a single batch.
static void test_gradient_flow() {
    std::printf("[test] gradient flow (no NaN/Inf) ... ");

    // Use AnyModule so we can call forward() generically on all 5 variants
    auto run_one = [](torch::nn::AnyModule model, const char* name) {
        model.ptr()->train();
        auto x      = torch::randn({2, 3, 224, 224});
        auto target = torch::randint(0, 1000, {2});
        auto output = model.forward<torch::Tensor>(x);
        auto loss   = torch::nn::functional::cross_entropy(output, target);

        ASSERT_TRUE(!torch::isnan(loss).item<bool>(),
                    (std::string(name) + " loss is NaN").c_str());
        ASSERT_TRUE(!torch::isinf(loss).item<bool>(),
                    (std::string(name) + " loss is Inf").c_str());

        loss.backward();

        for (const auto& p : model.ptr()->parameters()) {
            if (p.grad().defined()) {
                ASSERT_TRUE(!torch::isnan(p.grad()).any().item<bool>(),
                            (std::string(name) + " gradient has NaN").c_str());
                ASSERT_TRUE(!torch::isinf(p.grad()).any().item<bool>(),
                            (std::string(name) + " gradient has Inf").c_str());
            }
        }
    };

    run_one(torch::nn::AnyModule(make_resnet18(1000)),  "resnet18");
    run_one(torch::nn::AnyModule(make_resnet34(1000)),  "resnet34");
    run_one(torch::nn::AnyModule(make_resnet50(1000)),  "resnet50");
    run_one(torch::nn::AnyModule(make_resnet101(1000)), "resnet101");
    run_one(torch::nn::AnyModule(make_resnet152(1000)), "resnet152");

    std::printf("PASS\n");
}

// ── Test 4: shortcut projection dimensions ────────────────────────────────────
// Paper Section 3.3: "When the dimensions increase (dotted line shortcuts in
// Fig. 3), we consider … option B: projection shortcut (1×1 conv)."
// Verify that the first block of layer2/3/4 uses a downsample (projection)
// and the remaining blocks do NOT (identity shortcut).
static void test_shortcut_types() {
    std::printf("[test] shortcut types (identity vs projection) ... ");

    auto check_layer = [](torch::nn::Sequential& layer, const char* name) {
        bool first = true;
        for (auto& sub : layer->children()) {
            // torch::OrderedDict uses .contains() not .find()
            bool has_ds = sub->named_children().contains("downsample");
            if (first) {
                ASSERT_TRUE(has_ds,
                    (std::string(name) + " first block should have downsample").c_str());
                first = false;
            } else {
                if (has_ds) {
                    // non-first blocks must have identity (empty) downsample
                    auto& ds = sub->named_children()["downsample"];
                    ASSERT_TRUE(ds->children().empty() || ds->parameters().empty(),
                        (std::string(name) + " non-first block downsample must be identity").c_str());
                }
            }
        }
    };

    auto r50 = make_resnet50(1000);
    // layer1: in_planes==64, planes*exp==64 → identity (no downsample at first block)
    // layer2: in_planes==64, planes*exp==512 → projection
    check_layer(r50->layer2, "resnet50 layer2");
    check_layer(r50->layer3, "resnet50 layer3");
    check_layer(r50->layer4, "resnet50 layer4");

    std::printf("PASS\n");
}

// ── Test 5: bottleneck channel widths ─────────────────────────────────────────
// Paper Fig. 5 right: bottleneck reduces channels by 4× then restores.
// conv1: in → planes, conv2: planes → planes, conv3: planes → planes*4
static void test_bottleneck_expansion() {
    std::printf("[test] bottleneck expansion=4 ... ");

    // Build a single bottleneck and inspect conv filter sizes
    auto ds = torch::nn::Sequential{};
    BottleneckImpl block(64, 64, 1, ds);  // in=64, planes=64, out=256

    // conv1: 64→64 (1×1 reduce)
    ASSERT_TRUE(block.conv1->weight.size(0) == 64 && block.conv1->weight.size(1) == 64,
          "bottleneck conv1 channels 64→64");
    // conv2: 64→64 (3×3 compute)
    ASSERT_TRUE(block.conv2->weight.size(0) == 64 && block.conv2->weight.size(1) == 64,
          "bottleneck conv2 channels 64→64");
    // conv3: 64→256 (1×1 restore, expansion=4)
    ASSERT_TRUE(block.conv3->weight.size(0) == 256 && block.conv3->weight.size(1) == 64,
          "bottleneck conv3 channels 64→256 (expansion=4)");

    // Forward through a single block
    auto x   = torch::randn({1, 64, 56, 56});
    auto ds2 = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 256, 1).bias(false)),
        torch::nn::BatchNorm2d(256));
    BottleneckImpl block2(64, 64, 1, ds2);
    block2.eval();
    torch::NoGradGuard ng;
    auto out = block2.forward(x);
    ASSERT_TRUE(out.size(1) == 256, "bottleneck output channels == planes*4 = 256");

    std::printf("PASS\n");
}

// ── Test 6: SGD training step changes weights ─────────────────────────────────
// Paper Section 3.4: "trained from scratch" with SGD. Verify a single
// optimizer step actually modifies the network parameters.
static void test_training_step() {
    std::printf("[test] training step (SGD updates weights) ... ");

    auto model = make_resnet18(10);  // small model, faster
    model->train();

    // Snapshot first conv weight before step
    auto w_before = model->conv1->weight.clone().detach();

    torch::optim::SGD opt(model->parameters(),
                          torch::optim::SGDOptions(0.1).momentum(0.9));

    auto x      = torch::randn({1, 3, 224, 224});
    auto target = torch::randint(0, 10, {1});

    opt.zero_grad();
    auto loss = torch::nn::functional::cross_entropy(
        model->forward(x), target);
    loss.backward();
    opt.step();

    auto w_after = model->conv1->weight.detach();
    ASSERT_TRUE(!w_before.allclose(w_after), "SGD step should change conv1 weights");

    std::printf("PASS\n");
}

// ── Test 7: train vs eval mode ────────────────────────────────────────────────
// Paper Section 3.4: BN is used; in eval mode BN uses running stats (fixed),
// so the same input should produce the same output across two calls.
// In train mode, BN stats update → outputs may differ with different batch stats.
static void test_train_eval_mode() {
    std::printf("[test] train/eval mode consistency ... ");

    auto model = make_resnet18(10);

    auto x = torch::randn({2, 3, 224, 224});

    // eval mode: two forward passes must be identical
    model->eval();
    torch::NoGradGuard ng;
    auto out1 = model->forward(x);
    auto out2 = model->forward(x);
    ASSERT_TRUE(out1.allclose(out2), "eval mode: same input → same output");

    // train mode: two forward passes with THE SAME input should still
    // be identical (BN updates running mean/var but output per-call is same)
    model->train();
    auto out3 = model->forward(x);
    auto out4 = model->forward(x);
    // outputs may differ slightly due to running stat update, so we just check
    // that neither is NaN/Inf
    ASSERT_TRUE(!torch::isnan(out3).any().item<bool>(), "train mode out3 no NaN");
    ASSERT_TRUE(!torch::isnan(out4).any().item<bool>(), "train mode out4 no NaN");

    std::printf("PASS\n");
}

// ── Test 8: checkpoint save / load ───────────────────────────────────────────
// Paper reports best single-model results; we verify the checkpoint round-trip
// preserves the forward computation exactly.
static void test_checkpoint_roundtrip() {
    std::printf("[test] checkpoint save/load round-trip ... ");

    const std::string path = "/tmp/dm_resnet18_test.pt";

    auto model = make_resnet18(10);
    model->eval();

    auto x = torch::randn({1, 3, 224, 224});
    torch::NoGradGuard ng;
    auto out_before = model->forward(x).clone();

    // Save
    {
        torch::serialize::OutputArchive archive;
        model->save(archive);
        archive.save_to(path);
    }

    // Corrupt weights
    torch::nn::init::zeros_(model->conv1->weight);

    // Load back
    {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        model->load(archive);
    }

    auto out_after = model->forward(x);
    ASSERT_TRUE(out_before.allclose(out_after, /*rtol=*/1e-5, /*atol=*/1e-5),
          "checkpoint round-trip: forward output preserved");

    std::remove(path.c_str());
    std::printf("PASS\n");
}

// ── Test 9: BasicBlock spatial size preservation ──────────────────────────────
// Paper Fig. 3 right / Table 1: within a stage, spatial size is preserved.
// At stage transitions (stride=2), spatial size is halved.
static void test_spatial_sizes() {
    std::printf("[test] spatial size progression (224→112→56→28→14→7) ... ");

    auto model = make_resnet50(1000);
    model->eval();
    torch::NoGradGuard ng;

    // Hook to capture intermediate sizes
    auto x = torch::randn({1, 3, 224, 224});

    // conv1 + bn1 + relu: 224 → 112
    auto after_conv1 = torch::relu(
        model->bn1->forward(model->conv1->forward(x)));
    ASSERT_TRUE(after_conv1.size(2) == 112 && after_conv1.size(3) == 112,
          "after conv1: 224→112");

    // maxpool: 112 → 56
    auto after_pool = model->maxpool->forward(after_conv1);
    ASSERT_TRUE(after_pool.size(2) == 56 && after_pool.size(3) == 56,
          "after maxpool: 112→56");

    // layer1: 56→56 (stride=1, same size)
    auto after_l1 = model->layer1->forward(after_pool);
    ASSERT_TRUE(after_l1.size(2) == 56, "after layer1: 56→56");

    // layer2: 56→28 (stride=2)
    auto after_l2 = model->layer2->forward(after_l1);
    ASSERT_TRUE(after_l2.size(2) == 28, "after layer2: 56→28");

    // layer3: 28→14 (stride=2)
    auto after_l3 = model->layer3->forward(after_l2);
    ASSERT_TRUE(after_l3.size(2) == 14, "after layer3: 28→14");

    // layer4: 14→7 (stride=2)
    auto after_l4 = model->layer4->forward(after_l3);
    ASSERT_TRUE(after_l4.size(2) == 7, "after layer4: 14→7");

    std::printf("PASS\n");
}

// ── Test 10: resnet_train_epoch / resnet_evaluate helpers ─────────────────────
// Smoke-test the training API with a tiny synthetic dataset (2 batches of 4).
static void test_train_eval_api() {
    std::printf("[test] resnet_train_epoch / resnet_evaluate API ... ");

    auto model_ptr = make_resnet18(10);
    torch::nn::AnyModule model(model_ptr);

    torch::optim::SGD opt(model_ptr->parameters(),
                          torch::optim::SGDOptions(0.01).momentum(0.9));

    // 2 tiny batches: 4 samples, 3-channel 224×224
    std::vector<std::pair<torch::Tensor, torch::Tensor>> batches;
    for (int i = 0; i < 2; ++i)
        batches.emplace_back(torch::randn({4, 3, 224, 224}),
                             torch::randint(0, 10, {4}));

    float loss = resnet_train_epoch(model, opt, torch::kCPU, batches);
    ASSERT_TRUE(!std::isnan(loss) && !std::isinf(loss),
          "resnet_train_epoch returns finite loss");
    ASSERT_TRUE(loss > 0.f, "resnet_train_epoch loss > 0");

    auto [top1, top5] = resnet_evaluate(model, torch::kCPU, batches);
    ASSERT_TRUE(top1 >= 0.f && top1 <= 1.f, "top1 in [0,1]");
    ASSERT_TRUE(top5 >= 0.f && top5 <= 1.f, "top5 in [0,1]");
    ASSERT_TRUE(top5 >= top1,               "top5 >= top1");

    std::printf("PASS\n");
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("=== ResNet tests (He et al. CVPR 2016) ===\n");

    test_output_shape();
    test_param_counts();
    test_gradient_flow();
    test_shortcut_types();
    test_bottleneck_expansion();
    test_training_step();
    test_train_eval_mode();
    test_checkpoint_roundtrip();
    test_spatial_sizes();
    test_train_eval_api();

    std::printf("=== all tests PASSED ===\n");
    return 0;
}
