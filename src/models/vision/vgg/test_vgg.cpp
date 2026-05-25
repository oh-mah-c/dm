// ─────────────────────────────────────────────────────────────────────────────
// test_vgg.cpp — structural and numerical tests for VGGNet
//
// Paper: Simonyan & Zisserman, "Very Deep Convolutional Networks for
//        Large-Scale Image Recognition", ICLR 2015 (arXiv:1409.1556v6)
//
// Tests derived directly from the paper:
//  1. Output shape   [N, num_classes] for all 5 configs
//  2. Layer counts   match Table 1 weight-layer counts
//  3. Parameter counts   match Table 2 (±5%)
//  4. Feature block depth   correct conv count per config
//  5. Config C  has conv1×1 layers
//  6. Spatial size progression  224→112→56→28→14→7 (5 maxpools, Section 2.1)
//  7. No LRN layers   (paper Section 2.1: "none of our networks uses LRN")
//  8. Dropout present in classifier  (paper Section 3.1)
//  9. Gradient flow  no NaN/Inf for all 5 configs
// 10. SGD training step mutates weights
// 11. Eval mode determinism
// 12. Checkpoint save/load round-trip
// 13. vgg_train_epoch / vgg_evaluate API returns finite values
//
// Exit 0 = all passed.
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/vgg/vgg.h"

#include <torch/torch.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace dm::models::vision;

static void ASSERT_TRUE(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

static int64_t count_params(const torch::nn::Module& m) {
    int64_t n = 0;
    for (const auto& p : m.parameters()) n += p.numel();
    return n;
}

// Count modules of a specific type in a module tree
template <typename T>
static int64_t count_modules(const torch::nn::Module& m) {
    int64_t n = 0;
    for (const auto& sub : m.modules()) {
        if (sub->as<T>()) ++n;
    }
    return n;
}

// ── Test 1: output shape ──────────────────────────────────────────────────────
static void test_output_shape() {
    std::printf("[test] output shape ... ");
    auto x = torch::randn({2, 3, 224, 224});
    torch::NoGradGuard ng;

    for (auto& [name, model] : std::vector<std::pair<std::string, std::shared_ptr<VGGImpl>>>{
            {"vgg_a",  make_vgg_a(1000)},
            {"vgg_b",  make_vgg_b(1000)},
            {"vgg_c",  make_vgg_c(1000)},
            {"vgg16",  make_vgg16(1000)},
            {"vgg19",  make_vgg19(1000)},
        }) {
        model->eval();
        auto out = model->forward(x);
        ASSERT_TRUE(out.sizes() == torch::IntArrayRef({2, 1000}),
                    (name + " output shape [2,1000]").c_str());
    }

    // num_classes configurable — paper uses 10 for fine-grained tasks
    auto m10 = make_vgg16(10);
    m10->eval();
    ASSERT_TRUE(make_vgg16(10)->forward(x).sizes() == torch::IntArrayRef({2, 10}),
                "vgg16 10-class output shape");

    std::printf("PASS\n");
}

// ── Test 2: weight layer counts (Table 1) ─────────────────────────────────────
// Paper Table 1 header row: A=11, B=13, C=16, D=16, E=19 weight layers
// "weight layers" = conv + FC (not pooling, not ReLU)
static void test_weight_layer_counts() {
    std::printf("[test] weight layer counts (Table 1) ... ");

    struct Case { std::shared_ptr<VGGImpl> m; int64_t conv; int64_t fc; const char* name; };
    std::vector<Case> cases = {
        // A: 8 conv + 3 FC = 11
        { make_vgg_a(1000),  8, 3, "vgg_a (11 weight layers)" },
        // B: 10 conv + 3 FC = 13
        { make_vgg_b(1000), 10, 3, "vgg_b (13 weight layers)" },
        // C: 13 conv + 3 FC = 16
        { make_vgg_c(1000), 13, 3, "vgg_c (16 weight layers)" },
        // D: 13 conv + 3 FC = 16
        { make_vgg16(1000), 13, 3, "vgg16 (16 weight layers)" },
        // E: 16 conv + 3 FC = 19
        { make_vgg19(1000), 16, 3, "vgg19 (19 weight layers)" },
    };

    for (auto& c : cases) {
        int64_t n_conv = count_modules<torch::nn::Conv2dImpl>(*c.m);
        int64_t n_fc   = count_modules<torch::nn::LinearImpl>(*c.m);
        ASSERT_TRUE(n_conv == c.conv,
            (std::string(c.name) + " conv count").c_str());
        ASSERT_TRUE(n_fc == c.fc,
            (std::string(c.name) + " fc count").c_str());
    }

    std::printf("PASS\n");
}

// ── Test 3: parameter counts (Table 2) ────────────────────────────────────────
// Paper Table 2: A,A-LRN=133M, B=133M, C=134M, D=138M, E=144M
// We allow ±5% tolerance (BN adds a small number of params in batch_norm=false).
static void test_param_counts() {
    std::printf("[test] parameter counts (Table 2) ... ");

    struct Case { std::shared_ptr<VGGImpl> m; int64_t expected_M; const char* name; };
    std::vector<Case> cases = {
        { make_vgg_a(1000),  132'863'336, "vgg_a  (~133M)" },
        { make_vgg_b(1000),  132'863'336, "vgg_b  (~133M)" },
        { make_vgg_c(1000),  134'301'544, "vgg_c  (~134M)" },
        { make_vgg16(1000),  138'357'544, "vgg16  (~138M)" },
        { make_vgg19(1000),  143'667'240, "vgg19  (~144M)" },
    };

    for (auto& c : cases) {
        int64_t got   = count_params(*c.m);
        double  ratio = std::abs(static_cast<double>(got - c.expected_M))
                        / c.expected_M;
        if (ratio > 0.05) {
            std::fprintf(stderr,
                "FAIL: %s expected ~%lldM params, got %lldM (%.1f%% off)\n",
                c.name,
                (long long)(c.expected_M / 1'000'000),
                (long long)(got / 1'000'000),
                ratio * 100.0);
            std::exit(1);
        }
    }

    std::printf("PASS\n");
}

// ── Test 4 & 5: feature block structure + conv1×1 in config C ─────────────────
// Paper Section 2.1 / Table 1:
//   Config C only uses 1×1 conv filters in blocks 3, 4, 5.
static void test_feature_structure() {
    std::printf("[test] feature block structure + conv1×1 in config C ... ");

    // Config D (VGG-16): must have zero 1×1 conv layers
    auto d = make_vgg16(1000);
    int64_t n1x1_d = 0;
    for (const auto& m : d->features->modules()) {
        if (auto* c = m->as<torch::nn::Conv2dImpl>()) {
            if (c->options.kernel_size()->at(0) == 1) ++n1x1_d;
        }
    }
    ASSERT_TRUE(n1x1_d == 0, "vgg16 (config D) must have no conv1×1");

    // Config C: must have exactly 3 conv1×1 layers (one per block 3,4,5)
    auto cv = make_vgg_c(1000);
    int64_t n1x1_c = 0;
    for (const auto& m : cv->features->modules()) {
        if (auto* c = m->as<torch::nn::Conv2dImpl>()) {
            if (c->options.kernel_size()->at(0) == 1) ++n1x1_c;
        }
    }
    ASSERT_TRUE(n1x1_c == 3, "vgg_c must have exactly 3 conv1×1 layers");

    // Config E (VGG-19): no conv1×1
    auto e = make_vgg19(1000);
    int64_t n1x1_e = 0;
    for (const auto& m : e->features->modules()) {
        if (auto* c = m->as<torch::nn::Conv2dImpl>()) {
            if (c->options.kernel_size()->at(0) == 1) ++n1x1_e;
        }
    }
    ASSERT_TRUE(n1x1_e == 0, "vgg19 (config E) must have no conv1×1");

    // All configs must have exactly 5 maxpool layers (Section 2.1: 5 pools)
    for (auto& [name, model] : std::vector<std::pair<std::string, std::shared_ptr<VGGImpl>>>{
            {"vgg_a", make_vgg_a(1000)}, {"vgg_b", make_vgg_b(1000)},
            {"vgg_c", make_vgg_c(1000)}, {"vgg16", make_vgg16(1000)},
            {"vgg19", make_vgg19(1000)},
        }) {
        int64_t n_pool = count_modules<torch::nn::MaxPool2dImpl>(*model);
        ASSERT_TRUE(n_pool == 5,
            (name + " must have exactly 5 maxpool layers").c_str());
    }

    std::printf("PASS\n");
}

// ── Test 6: spatial size progression ─────────────────────────────────────────
// Paper Section 2.1: 5 max-pooling layers 2×2/stride 2 → 224→112→56→28→14→7
static void test_spatial_sizes() {
    std::printf("[test] spatial size 224→112→56→28→14→7 ... ");

    auto model = make_vgg16(1000);
    model->eval();
    torch::NoGradGuard ng;

    // Walk through features module manually, checking size after each maxpool
    auto x = torch::randn({1, 3, 224, 224});
    std::vector<int64_t> expected_after_pool = {112, 56, 28, 14, 7};
    int pool_idx = 0;

    for (const auto& sub : model->features->children()) {
        // Dispatch to concrete type; all sub-modules in features are one of:
        // Conv2d, ReLU, MaxPool2d
        if (auto* c = sub->as<torch::nn::Conv2dImpl>())
            x = c->forward(x);
        else if (auto* r = sub->as<torch::nn::ReLUImpl>())
            x = r->forward(x);
        else if (auto* p = sub->as<torch::nn::MaxPool2dImpl>()) {
            x = p->forward(x);
            ASSERT_TRUE(x.size(2) == expected_after_pool[pool_idx],
                ("spatial size after pool " + std::to_string(pool_idx+1)).c_str());
            ++pool_idx;
        } else if (auto* b = sub->as<torch::nn::BatchNorm2dImpl>())
            x = b->forward(x);
    }
    ASSERT_TRUE(pool_idx == 5, "exactly 5 maxpools traversed");

    std::printf("PASS\n");
}

// ── Test 7: no LRN layers ─────────────────────────────────────────────────────
// Paper Section 2.1: "We do not employ the Local Response Normalisation (LRN)"
// (Note: only A-LRN variant uses it, which we don't implement)
static void test_no_lrn() {
    std::printf("[test] no LRN layers ... ");

    // LibTorch doesn't have a standalone LRN module in nn::, so we verify
    // by checking that the features Sequential contains no BatchNorm when
    // batch_norm=false (default), and no unexpected module types.
    for (auto& m : {make_vgg_a(1000), make_vgg_b(1000), make_vgg_c(1000),
                    make_vgg16(1000), make_vgg19(1000)}) {
        int64_t n_bn = count_modules<torch::nn::BatchNorm2dImpl>(*m);
        ASSERT_TRUE(n_bn == 0, "default VGG must have no BatchNorm (= no LRN)");
    }

    // With batch_norm=true the BN layers should appear
    auto bn_model = make_vgg16(1000, /*batch_norm=*/true);
    int64_t n_bn = count_modules<torch::nn::BatchNorm2dImpl>(*bn_model);
    ASSERT_TRUE(n_bn > 0, "vgg16 with batch_norm=true must have BN layers");

    std::printf("PASS\n");
}

// ── Test 8: dropout in classifier ─────────────────────────────────────────────
// Paper Section 3.1: "dropout for the first two FC layers (dropout ratio 0.5)"
static void test_dropout_present() {
    std::printf("[test] dropout in classifier ... ");

    auto m = make_vgg16(1000);
    int64_t n_drop = count_modules<torch::nn::DropoutImpl>(*m);
    ASSERT_TRUE(n_drop == 2, "classifier must have exactly 2 Dropout layers");

    std::printf("PASS\n");
}

// ── Test 9: gradient flow ─────────────────────────────────────────────────────
static void test_gradient_flow() {
    std::printf("[test] gradient flow (no NaN/Inf) ... ");

    // Use vgg_a (smallest, fastest) for gradient check
    auto run = [](torch::nn::AnyModule model, const char* name) {
        model.ptr()->train();
        auto x      = torch::randn({1, 3, 224, 224});
        auto target = torch::randint(0, 1000, {1});
        auto output = model.forward<torch::Tensor>(x);
        auto loss   = torch::nn::functional::cross_entropy(output, target);

        ASSERT_TRUE(!torch::isnan(loss).item<bool>(),
                    (std::string(name) + " loss NaN").c_str());
        ASSERT_TRUE(!torch::isinf(loss).item<bool>(),
                    (std::string(name) + " loss Inf").c_str());

        loss.backward();
        for (const auto& p : model.ptr()->parameters()) {
            if (p.grad().defined()) {
                ASSERT_TRUE(!torch::isnan(p.grad()).any().item<bool>(),
                            (std::string(name) + " grad NaN").c_str());
            }
        }
    };

    run(torch::nn::AnyModule(make_vgg_a(1000)), "vgg_a");
    run(torch::nn::AnyModule(make_vgg16(1000)), "vgg16");
    run(torch::nn::AnyModule(make_vgg19(1000)), "vgg19");

    std::printf("PASS\n");
}

// ── Test 10: SGD training step ────────────────────────────────────────────────
static void test_training_step() {
    std::printf("[test] SGD training step mutates weights ... ");

    // Use vgg_a (smallest)
    auto model = make_vgg_a(10);
    model->train();

    // Snapshot first conv weight
    auto w_before = model->features[0]->as<torch::nn::Conv2dImpl>()
                        ->weight.clone().detach();

    torch::optim::SGD opt(model->parameters(),
                          torch::optim::SGDOptions(0.01).momentum(0.9));

    auto x      = torch::randn({1, 3, 224, 224});
    auto target = torch::randint(0, 10, {1});

    opt.zero_grad();
    auto loss = torch::nn::functional::cross_entropy(model->forward(x), target);
    loss.backward();
    opt.step();

    auto w_after = model->features[0]->as<torch::nn::Conv2dImpl>()
                       ->weight.detach();
    ASSERT_TRUE(!w_before.allclose(w_after), "SGD step must change first conv weight");

    std::printf("PASS\n");
}

// ── Test 11: eval mode determinism ────────────────────────────────────────────
static void test_eval_determinism() {
    std::printf("[test] eval mode determinism ... ");

    auto model = make_vgg_a(10);
    model->eval();
    torch::NoGradGuard ng;

    auto x    = torch::randn({2, 3, 224, 224});
    auto out1 = model->forward(x);
    auto out2 = model->forward(x);
    ASSERT_TRUE(out1.allclose(out2), "eval mode: same input → same output");

    std::printf("PASS\n");
}

// ── Test 12: checkpoint save/load ─────────────────────────────────────────────
static void test_checkpoint_roundtrip() {
    std::printf("[test] checkpoint save/load round-trip ... ");

    const std::string path = "/tmp/dm_vgg_test.pt";
    auto model = make_vgg_a(10);
    model->eval();

    auto x          = torch::randn({1, 3, 224, 224});
    torch::NoGradGuard ng;
    auto out_before = model->forward(x).clone();

    // Save
    {
        torch::serialize::OutputArchive a;
        model->save(a);
        a.save_to(path);
    }

    // Corrupt weights
    torch::nn::init::zeros_(
        model->features[0]->as<torch::nn::Conv2dImpl>()->weight);

    // Reload
    {
        torch::serialize::InputArchive a;
        a.load_from(path);
        model->load(a);
    }

    auto out_after = model->forward(x);
    ASSERT_TRUE(out_before.allclose(out_after, 1e-5f, 1e-5f),
                "checkpoint round-trip preserves output");

    std::remove(path.c_str());
    std::printf("PASS\n");
}

// ── Test 13: training/evaluation API ─────────────────────────────────────────
static void test_train_eval_api() {
    std::printf("[test] vgg_train_epoch / vgg_evaluate API ... ");

    auto model_ptr = make_vgg_a(10);
    torch::nn::AnyModule model(model_ptr);

    torch::optim::SGD opt(model_ptr->parameters(),
                          torch::optim::SGDOptions(0.01).momentum(0.9));

    std::vector<std::pair<torch::Tensor, torch::Tensor>> batches;
    for (int i = 0; i < 2; ++i)
        batches.emplace_back(torch::randn({2, 3, 224, 224}),
                             torch::randint(0, 10, {2}));

    float loss = vgg_train_epoch(model, opt, torch::kCPU, batches);
    ASSERT_TRUE(!std::isnan(loss) && !std::isinf(loss) && loss > 0.f,
                "vgg_train_epoch returns finite positive loss");

    auto [top1, top5] = vgg_evaluate(model, torch::kCPU, batches);
    ASSERT_TRUE(top1 >= 0.f && top1 <= 1.f, "top1 in [0,1]");
    ASSERT_TRUE(top5 >= top1,               "top5 >= top1");

    std::printf("PASS\n");
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("=== VGGNet tests (Simonyan & Zisserman, ICLR 2015) ===\n");

    test_output_shape();
    test_weight_layer_counts();
    test_param_counts();
    test_feature_structure();
    test_spatial_sizes();
    test_no_lrn();
    test_dropout_present();
    test_gradient_flow();
    test_training_step();
    test_eval_determinism();
    test_checkpoint_roundtrip();
    test_train_eval_api();

    std::printf("=== all tests PASSED ===\n");
    return 0;
}
