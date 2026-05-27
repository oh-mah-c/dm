// ─────────────────────────────────────────────────────────────────────────────
// Tests for DenseNet (arXiv:1608.06993v5)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/densenet/densenet.h"

#include <torch/torch.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

namespace dm { namespace models { namespace vision {

static int pass_count = 0;
static int fail_count = 0;

static void check(bool cond, const std::string& name) {
    if (cond) {
        std::cout << "  PASS  " << name << "\n";
        ++pass_count;
    } else {
        std::cerr << "  FAIL  " << name << "\n";
        ++fail_count;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DenseLayer output shape tests
// ─────────────────────────────────────────────────────────────────────────────

static void test_dense_layer_shapes() {
    std::cout << "\n[DenseLayer]\n";

    // Plain layer: in_channels=16, growth_rate=12 → out = 16+12 = 28
    {
        DenseLayer layer(16, 12, /*bottleneck=*/false);
        auto x   = torch::randn({2, 16, 8, 8});
        auto out = layer(x);
        check(out.size(1) == 28, "plain layer out channels = in+k");
        check(out.size(2) == 8 && out.size(3) == 8, "plain layer spatial unchanged");
    }

    // Bottleneck layer: in=32, k=12 → out = 32+12 = 44
    {
        DenseLayer layer(32, 12, /*bottleneck=*/true);
        auto x   = torch::randn({1, 32, 16, 16});
        auto out = layer(x);
        check(out.size(1) == 44, "bottleneck out channels = in+k");
    }

    // Input x is preserved via cat — first 32 channels must equal x
    {
        DenseLayer layer(32, 12, /*bottleneck=*/false);
        auto x   = torch::randn({1, 32, 8, 8});
        auto out = layer(x);
        check(out.narrow(1, 0, 32).allclose(x), "dense cat preserves input channels");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DenseBlock output shape tests
// ─────────────────────────────────────────────────────────────────────────────

static void test_dense_block_shapes() {
    std::cout << "\n[DenseBlock]\n";

    // 5 layers, k=4, in=16 → out = 16 + 5*4 = 36
    {
        DenseBlock blk(16, 5, 4, /*bottleneck=*/false);
        check(blk->out_channels() == 36, "block out_channels() = in + num_layers*k");
        auto x   = torch::randn({2, 16, 8, 8});
        auto out = blk(x);
        check(out.size(1) == 36, "block forward out channels correct");
        check(out.size(2) == 8, "block forward spatial unchanged");
    }

    // Bottleneck, 6 layers, k=32, in=64 → out = 64 + 6*32 = 256
    {
        DenseBlock blk(64, 6, 32, /*bottleneck=*/true);
        check(blk->out_channels() == 256, "bottleneck block out_channels correct");
        auto x   = torch::randn({1, 64, 7, 7});
        auto out = blk(x);
        check(out.size(1) == 256, "bottleneck block forward correct");
    }

    // Growth rate k=12, paper Figure 1: 5-layer block from 16 → 76
    {
        DenseBlock blk(16, 5, 12, false);
        auto x   = torch::randn({1, 16, 4, 4});
        auto out = blk(x);
        check(out.size(1) == 76, "paper fig 1 analog: 16 + 5*12 = 76 channels");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TransitionLayer tests
// ─────────────────────────────────────────────────────────────────────────────

static void test_transition_layer() {
    std::cout << "\n[TransitionLayer]\n";

    // theta=0.5: in=128 → out=64, spatial halved
    {
        TransitionLayer trans(128, 0.5);
        check(trans->out_channels() == 64, "transition theta=0.5 halves channels");
        auto x   = torch::randn({1, 128, 16, 16});
        auto out = trans(x);
        check(out.size(1) == 64,  "transition forward channels correct");
        check(out.size(2) == 8 && out.size(3) == 8, "transition halves spatial");
    }

    // theta=1.0: no compression
    {
        TransitionLayer trans(64, 1.0);
        check(trans->out_channels() == 64, "transition theta=1.0 no compression");
        auto x   = torch::randn({1, 64, 8, 8});
        auto out = trans(x);
        check(out.size(1) == 64, "theta=1.0 channels unchanged");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CIFAR factory tests (small_input=true)
// ─────────────────────────────────────────────────────────────────────────────

static void test_cifar_factory() {
    std::cout << "\n[CIFAR factory]\n";

    // DenseNet(L=40, k=12): plain, 3 blocks of (40-4)/3=12 layers each
    {
        auto net = make_densenet_cifar(40, 12, /*bottleneck=*/false, 10);
        net->eval();
        auto x   = torch::randn({2, 3, 32, 32});
        auto out = net->forward(x);
        check(out.sizes() == torch::IntArrayRef({2, 10}),
              "cifar L=40 k=12 output shape [2,10]");
    }

    // DenseNet-BC(L=100, k=12): 3 blocks of 16 bottleneck layers each
    {
        auto net = make_densenet_cifar(100, 12, /*bottleneck=*/true, 10);
        net->eval();
        auto x   = torch::randn({1, 3, 32, 32});
        auto out = net->forward(x);
        check(out.sizes() == torch::IntArrayRef({1, 10}),
              "cifar BC L=100 k=12 output shape [1,10]");
    }

    // DenseNet(L=100, k=24): plain, 3 blocks of 32 layers
    {
        auto net = make_densenet_cifar(100, 24, /*bottleneck=*/false, 100);
        net->eval();
        auto x   = torch::randn({1, 3, 32, 32});
        auto out = net->forward(x);
        check(out.sizes() == torch::IntArrayRef({1, 100}),
              "cifar L=100 k=24 C=100 output shape correct");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ImageNet factory tests (spot-check output shape + param count order)
// ─────────────────────────────────────────────────────────────────────────────

static void test_imagenet_factory() {
    std::cout << "\n[ImageNet factory]\n";

    // DenseNet-121: output [B, 1000], param count ~8M
    {
        auto net = make_densenet121(1000);
        net->eval();
        auto x   = torch::randn({1, 3, 224, 224});
        auto out = net->forward(x);
        check(out.sizes() == torch::IntArrayRef({1, 1000}),
              "densenet-121 output shape [1,1000]");

        int64_t params = 0;
        for (auto& p : net->parameters()) params += p.numel();
        // Paper quotes ~8M; check rough range [5M, 12M]
        check(params > 5'000'000 && params < 12'000'000,
              "densenet-121 param count ~8M");
    }

    // DenseNet-169: output [1, 1000]
    {
        auto net = make_densenet169(1000);
        net->eval();
        auto x   = torch::randn({1, 3, 224, 224});
        auto out = net->forward(x);
        check(out.sizes() == torch::IntArrayRef({1, 1000}),
              "densenet-169 output shape [1,1000]");

        int64_t params = 0;
        for (auto& p : net->parameters()) params += p.numel();
        // Paper ~14M; check [10M, 20M]
        check(params > 10'000'000 && params < 20'000'000,
              "densenet-169 param count ~14M");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Architecture / connectivity tests
// ─────────────────────────────────────────────────────────────────────────────

static void test_dense_connectivity() {
    std::cout << "\n[Dense connectivity]\n";

    // The output of a 3-layer block must equal cumulative concatenation.
    // We verify by checking that total out-channels = in + 3*k.
    {
        DenseBlock blk(8, 3, 4, false);
        auto x   = torch::zeros({1, 8, 4, 4});
        auto out = blk(x);
        check(out.size(1) == 8 + 3 * 4, "3-layer block channels = in + 3k");
    }

    // Each successive DenseLayer in a block receives one more k channels.
    // Verify by looking at the registered layer parameter shapes.
    {
        DenseBlock blk(16, 4, 8, /*bottleneck=*/false);
        // layer 0: in=16, out=8 → conv weight [8, 16, 3, 3]
        // layer 1: in=24, out=8 → conv weight [8, 24, 3, 3]
        // layer 2: in=32, out=8
        // layer 3: in=40, out=8
        auto& layers = blk->layers;
        for (size_t i = 0; i < layers->size(); ++i) {
            auto* dl = (*layers)[i]->as<DenseLayerImpl>();
            int64_t expected_in = 16 + static_cast<int64_t>(i) * 8;
            check(dl->conv1->weight.size(1) == expected_in,
                  "layer " + std::to_string(i) + " receives correct in-channels");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Gradient flow test
// ─────────────────────────────────────────────────────────────────────────────

static void test_gradients() {
    std::cout << "\n[Gradients]\n";

    auto net = make_densenet_cifar(40, 12, false, 10);
    net->train();
    auto x      = torch::randn({2, 3, 32, 32}, torch::requires_grad(false));
    auto logits = net->forward(x);
    auto loss   = torch::cross_entropy_loss(logits,
                      torch::zeros({2}, torch::kLong));
    loss.backward();

    bool all_grads = true;
    for (auto& p : net->parameters()) {
        if (!p.grad().defined() || p.grad().abs().sum().item<float>() == 0.0f) {
            all_grads = false;
            break;
        }
    }
    check(all_grads, "gradients flow to all parameters");
}

// ─────────────────────────────────────────────────────────────────────────────
// Train vs eval mode
// ─────────────────────────────────────────────────────────────────────────────

static void test_train_eval_mode() {
    std::cout << "\n[Train/eval mode]\n";

    auto net = make_densenet_cifar(40, 12, false, 10);
    auto x   = torch::randn({2, 3, 32, 32});

    net->eval();
    auto out1 = net->forward(x);
    auto out2 = net->forward(x);
    check(out1.allclose(out2), "eval mode: deterministic output");

    net->train();
    // In train mode BN running stats diverge between eval/train; just check shape.
    auto out3 = net->forward(x);
    check(out3.sizes() == torch::IntArrayRef({2, 10}), "train mode: shape correct");
}

// ─────────────────────────────────────────────────────────────────────────────
// No weight sharing between DenseNet-121 instances
// ─────────────────────────────────────────────────────────────────────────────

static void test_no_weight_sharing() {
    std::cout << "\n[No weight sharing]\n";

    auto a = make_densenet121();
    auto b = make_densenet121();
    auto pa = a->parameters()[0].data_ptr<float>();
    auto pb = b->parameters()[0].data_ptr<float>();
    check(pa != pb, "separate instances have separate parameter storage");
}

// ─────────────────────────────────────────────────────────────────────────────
// Bottleneck intermediate channel count (4k, paper Section 3)
// ─────────────────────────────────────────────────────────────────────────────

static void test_bottleneck_4k() {
    std::cout << "\n[Bottleneck 4k intermediate]\n";

    int64_t k = 32;
    DenseLayer layer(64, k, /*bottleneck=*/true);
    // First conv (1×1) output channels should be 4k
    check(layer->conv1->weight.size(0) == 4 * k,
          "bottleneck conv1 output = 4k");
    // Second conv (3×3) output channels = k
    check(layer->conv2->weight.size(0) == k,
          "bottleneck conv2 output = k");
}

// ─────────────────────────────────────────────────────────────────────────────
// Theta compression check (paper equation: floor(theta * m))
// ─────────────────────────────────────────────────────────────────────────────

static void test_theta_compression() {
    std::cout << "\n[Theta compression]\n";

    // m=100, theta=0.5 → floor(50) = 50
    TransitionLayer t1(100, 0.5);
    check(t1->out_channels() == 50, "theta=0.5 floor(100*0.5)=50");

    // m=100, theta=0.3 → floor(30) = 30
    TransitionLayer t2(100, 0.3);
    check(t2->out_channels() == 30, "theta=0.3 floor(100*0.3)=30");

    // m=64, theta=1.0 → 64
    TransitionLayer t3(64, 1.0);
    check(t3->out_channels() == 64, "theta=1.0 no compression");
}

} // namespace vision
} // namespace models
} // namespace dm

int main() {
    using namespace dm::models::vision;
    std::cout << "=== DenseNet tests ===\n";

    test_dense_layer_shapes();
    test_dense_block_shapes();
    test_transition_layer();
    test_cifar_factory();
    test_imagenet_factory();
    test_dense_connectivity();
    test_gradients();
    test_train_eval_mode();
    test_no_weight_sharing();
    test_bottleneck_4k();
    test_theta_compression();

    std::cout << "\n─────────────────────────\n";
    std::cout << "Passed: " << pass_count << "\n";
    if (fail_count)
        std::cerr << "FAILED: " << fail_count << "\n";
    else
        std::cout << "All tests passed.\n";

    return fail_count ? 1 : 0;
}
