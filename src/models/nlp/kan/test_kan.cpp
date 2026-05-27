// ─────────────────────────────────────────────────────────────────────────────
// KAN — Unit Tests
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/kan/kan.h"

#include <torch/torch.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

using namespace dm::models::nlp;

static int pass_count = 0;
static int fail_count = 0;

#define TCHECK(cond, name) do { \
    if (cond) { std::cout << "[PASS] " << (name) << "\n"; ++pass_count; } \
    else { std::cout << "[FAIL] " << (name) << "\n"; ++fail_count; } \
} while(0)

// ─────────────────────────────────────────────────────────────────────────────
// KANLinear primitive
// ─────────────────────────────────────────────────────────────────────────────

static void test_kan_linear_output_shape() {
    dm::prim::KANLinear layer(3, 5);
    auto x = torch::randn({4, 3});
    auto y = layer->forward(x);
    TCHECK(y.size(0) == 4 && y.size(1) == 5, "KANLinear output shape [4,5]");
}

static void test_kan_linear_batch1() {
    dm::prim::KANLinear layer(2, 4);
    auto x = torch::randn({1, 2});
    auto y = layer->forward(x);
    TCHECK(y.size(0) == 1 && y.size(1) == 4, "KANLinear batch=1 shape");
}

static void test_kan_linear_no_nan() {
    dm::prim::KANLinear layer(4, 3);
    layer->eval();
    auto x = torch::randn({8, 4});
    auto y = layer->forward(x);
    TCHECK(!y.isnan().any().template item<bool>() &&
           !y.isinf().any().template item<bool>(),
           "KANLinear output has no NaN/Inf");
}

static void test_kan_linear_gradient() {
    dm::prim::KANLinear layer(3, 2);
    auto x = torch::randn({4, 3}).requires_grad_(true);
    auto y = layer->forward(x);
    y.sum().backward();
    TCHECK(x.grad().defined() && !x.grad().isnan().any().template item<bool>(),
           "KANLinear gradient flows");
}

static void test_kan_linear_param_grad() {
    dm::prim::KANLinear layer(2, 3);
    auto x = torch::randn({8, 2});
    auto y = layer->forward(x);
    y.sum().backward();
    bool all_grads = true;
    for (auto& p : layer->parameters())
        if (!p.grad().defined()) { all_grads = false; break; }
    TCHECK(all_grads, "KANLinear all parameters have gradients");
}

static void test_kan_linear_custom_G_k() {
    dm::prim::KANLinearOptions opts(2, 4);
    opts.G(10).k(4);
    dm::prim::KANLinear layer(opts);
    auto x = torch::randn({3, 2});
    auto y = layer->forward(x);
    TCHECK(y.sizes() == torch::IntArrayRef({3, 4}),
           "KANLinear custom G=10,k=4 shape");
}

static void test_kan_linear_spline_weight_shape() {
    dm::prim::KANLinearOptions opts(3, 5);
    opts.G(7).k(3);
    dm::prim::KANLinear layer(opts);
    // spline_weight: [out, in, G+k] = [5, 3, 10]
    auto sw = layer->spline_weight;
    TCHECK(sw.size(0) == 5 && sw.size(1) == 3 && sw.size(2) == 10,
           "KANLinear spline_weight shape [5,3,G+k=10]");
}

static void test_kan_linear_grid_shape() {
    dm::prim::KANLinearOptions opts(4, 2);
    opts.G(5).k(3);
    dm::prim::KANLinear layer(opts);
    // grid: [n_in, G+2k+1] = [4, 12]
    auto g = layer->grid;
    TCHECK(g.size(0) == 4 && g.size(1) == 12,
           "KANLinear grid shape [n_in, G+2k+1=12]");
}

static void test_kan_linear_eval_no_grid_update() {
    dm::prim::KANLinear layer(2, 3);
    layer->eval();
    auto g_before = layer->grid.clone();
    auto x = torch::randn({10, 2}) * 10.0; // large range
    layer->forward(x);
    // In eval mode, grid should NOT update
    TCHECK((layer->grid - g_before).abs().max().template item<float>() < 1e-6f,
           "KANLinear grid not updated in eval mode");
}

static void test_kan_linear_train_grid_update() {
    dm::prim::KANLinear layer(2, 3);
    layer->train();
    auto g_before = layer->grid.clone();
    // Data far outside initial [-1,1] range should update the grid
    auto x = torch::randn({50, 2}) * 5.0 + 3.0;
    layer->forward(x);
    bool changed = (layer->grid - g_before).abs().max().template item<float>() > 1e-4f;
    TCHECK(changed, "KANLinear grid updates in train mode with out-of-range input");
}

static void test_kan_linear_sparsity_loss_nonnegative() {
    dm::prim::KANLinear layer(3, 4);
    auto x = torch::randn({16, 3});
    auto reg = dm::prim::kan_linear_sparsity(layer, x);
    TCHECK(reg.template item<float>() >= 0.0f,
           "KANLinear sparsity loss non-negative");
}

static void test_kan_linear_sparsity_loss_differentiable() {
    dm::prim::KANLinear layer(3, 4);
    auto x = torch::randn({8, 3});
    auto y = layer->forward(x);
    auto reg = dm::prim::kan_linear_sparsity(layer, x);
    (y.sum() + reg).backward();
    bool ok = true;
    for (auto& p : layer->parameters())
        if (!p.grad().defined()) { ok = false; break; }
    TCHECK(ok, "KANLinear sparsity loss is differentiable");
}

static void test_kan_linear_extend_grid() {
    dm::prim::KANLinearOptions opts(2, 3);
    opts.G(5).k(3);
    dm::prim::KANLinear layer(opts);
    layer->eval();
    auto x = torch::randn({50, 2});
    // Forward before extension to get reference output
    auto y_before = layer->forward(x).detach();
    // Extend grid from G=5 to G=10
    layer->extend_grid(10, x);
    TCHECK(layer->G() == 10, "KANLinear grid extended G=5->10");
    // After extension, output should be approximately the same
    auto y_after = layer->forward(x).detach();
    auto diff = (y_after - y_before).abs().max().template item<float>();
    TCHECK(diff < 0.5f, "KANLinear grid extension preserves output approx");
}

// ─────────────────────────────────────────────────────────────────────────────
// KAN network
// ─────────────────────────────────────────────────────────────────────────────

static void test_kan_output_shape_2layer() {
    KAN model(std::vector<int64_t>{2, 5, 1});
    auto x = torch::randn({8, 2});
    auto y = model->forward(x);
    TCHECK(y.size(0) == 8 && y.size(1) == 1, "KAN [2,5,1] output shape [8,1]");
}

static void test_kan_output_shape_3layer() {
    KAN model(std::vector<int64_t>{4, 10, 5, 2});
    auto x = torch::randn({4, 4});
    auto y = model->forward(x);
    TCHECK(y.size(0) == 4 && y.size(1) == 2, "KAN [4,10,5,2] output shape [4,2]");
}

static void test_kan_depth() {
    KAN model(std::vector<int64_t>{3, 8, 4, 1});
    TCHECK(model->depth() == 3, "KAN depth = L = 3");
}

static void test_kan_gradient() {
    KAN model(std::vector<int64_t>{2, 4, 1});
    auto x = torch::randn({6, 2}).requires_grad_(true);
    auto y = model->forward(x);
    y.sum().backward();
    TCHECK(x.grad().defined() && !x.grad().isnan().any().template item<bool>(),
           "KAN gradient flows to input");
}

static void test_kan_no_nan() {
    KAN model(std::vector<int64_t>{3, 5, 2});
    model->eval();
    auto x = torch::randn({10, 3});
    auto y = model->forward(x);
    TCHECK(!y.isnan().any().template item<bool>() &&
           !y.isinf().any().template item<bool>(),
           "KAN output has no NaN/Inf");
}

static void test_kan_sparsity_loss_scalar() {
    KAN model(std::vector<int64_t>{2, 5, 1});
    model->eval();
    auto x = torch::randn({16, 2});
    auto reg = model->sparsity_loss(x, 1e-3);
    TCHECK(reg.dim() == 0, "KAN sparsity loss is scalar");
}

static void test_kan_sparsity_loss_nonneg() {
    KAN model(std::vector<int64_t>{2, 4, 1});
    model->eval();
    auto x = torch::randn({8, 2});
    auto reg = model->sparsity_loss(x, 1.0);
    TCHECK(reg.template item<float>() >= 0.0f,
           "KAN sparsity loss non-negative");
}

static void test_kan_training_step() {
    // One Adam step on a simple 1D function f(x) = x^2
    KAN model(std::vector<int64_t>{1, 5, 1});
    torch::optim::Adam opt(model->parameters(),
                           torch::optim::AdamOptions(1e-3));
    auto x = torch::linspace(-1.f, 1.f, 20).unsqueeze(1); // [20,1]
    auto y_target = x.pow(2);
    opt.zero_grad();
    auto y_pred = model->forward(x);
    auto loss = torch::mse_loss(y_pred, y_target);
    loss.backward();
    float loss0 = loss.template item<float>();
    opt.step();
    opt.zero_grad();
    auto loss1 = torch::mse_loss(model->forward(x), y_target);
    TCHECK(loss1.template item<float>() != loss0,
           "KAN loss changes after optimizer step");
}

static void test_kan_extend_grid_all_layers() {
    KAN model(std::vector<int64_t>{2, 4, 1}, /*G=*/5, /*k=*/3);
    model->eval();
    auto x = torch::randn({30, 2});
    model->extend_grid(10, x);
    bool ok = true;
    for (int64_t l = 0; l < model->depth(); ++l) {
        auto& layer = model->layers->at<dm::prim::KANLinearImpl>(l);
        if (layer.G() != 10) { ok = false; break; }
    }
    TCHECK(ok, "KAN extend_grid updates all layers to G=10");
}

static void test_kan_lbfgs_step() {
    // One L-BFGS step on f(x,y) = x + y
    KAN model(std::vector<int64_t>{2, 3, 1});
    torch::optim::LBFGS opt(model->parameters(),
                            torch::optim::LBFGSOptions(1e-1));
    auto x = torch::randn({10, 2});
    auto y_target = x.sum(1, true);
    float loss_val = 0.0f;
    auto closure = [&]() {
        opt.zero_grad();
        auto pred = model->forward(x);
        auto loss = torch::mse_loss(pred, y_target);
        loss.backward();
        loss_val = loss.template item<float>();
        return loss;
    };
    opt.step(closure);
    TCHECK(loss_val >= 0.0f, "KAN L-BFGS step runs without error");
}

static void test_kan_wide_shallow() {
    // Paper MNIST setup: [784, 100, 10]
    KAN model(std::vector<int64_t>{16, 8, 4}, /*G=*/3);
    model->eval();
    auto x = torch::randn({2, 16});
    auto y = model->forward(x);
    TCHECK(y.size(0) == 2 && y.size(1) == 4,
           "KAN wide-shallow [16,8,4] output shape");
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== KAN Tests ===\n";

    // KANLinear primitive
    test_kan_linear_output_shape();
    test_kan_linear_batch1();
    test_kan_linear_no_nan();
    test_kan_linear_gradient();
    test_kan_linear_param_grad();
    test_kan_linear_custom_G_k();
    test_kan_linear_spline_weight_shape();
    test_kan_linear_grid_shape();
    test_kan_linear_eval_no_grid_update();
    test_kan_linear_train_grid_update();
    test_kan_linear_sparsity_loss_nonnegative();
    test_kan_linear_sparsity_loss_differentiable();
    test_kan_linear_extend_grid();

    // KAN network
    test_kan_output_shape_2layer();
    test_kan_output_shape_3layer();
    test_kan_depth();
    test_kan_gradient();
    test_kan_no_nan();
    test_kan_sparsity_loss_scalar();
    test_kan_sparsity_loss_nonneg();
    test_kan_training_step();
    test_kan_extend_grid_all_layers();
    test_kan_lbfgs_step();
    test_kan_wide_shallow();

    std::cout << "\n" << pass_count << " passed, " << fail_count << " failed\n";
    return fail_count == 0 ? 0 : 1;
}
