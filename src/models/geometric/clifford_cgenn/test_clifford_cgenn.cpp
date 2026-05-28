// -----------------------------------------------------------------------------
// Clifford CGENN tests
// -----------------------------------------------------------------------------

#include "models/geometric/clifford_cgenn/clifford_cgenn.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>

using namespace dm::models::geometric;

static int passed = 0;
static int failed = 0;

#define CG_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << (msg) << "\n"; \
            failed++; \
        } else { \
            std::cout << "  [PASS] " << (msg) << "\n"; \
            passed++; \
        } \
    } while (0)

static void test_product_table_2d() {
    auto table = clifford_build_product_table(CliffordAlgebraConfig::euclidean(2));
    CG_CHECK(table.n_blades == 4, "2D Clifford algebra has 4 blades");
    auto e1 = torch::tensor({0.0f, 1.0f, 0.0f, 0.0f});
    auto e2 = torch::tensor({0.0f, 0.0f, 1.0f, 0.0f});
    auto e12 = torch::tensor({0.0f, 0.0f, 0.0f, 1.0f});
    auto one = torch::tensor({1.0f, 0.0f, 0.0f, 0.0f});
    CG_CHECK(torch::allclose(clifford_geometric_product(e1, e1, table), one),
             "e1*e1=1 in Euclidean metric");
    CG_CHECK(torch::allclose(clifford_geometric_product(e1, e2, table), e12),
             "e1*e2=e12");
    CG_CHECK(torch::allclose(clifford_geometric_product(e2, e1, table), -e12),
             "e2*e1=-e12");
}

static void test_embed_and_qbar() {
    auto table = clifford_build_product_table(CliffordAlgebraConfig::euclidean(3));
    auto v = torch::tensor({{1.0f, 2.0f, 3.0f}});
    auto mv = clifford_embed_vector(v, table);
    CG_CHECK(mv.sizes() == torch::IntArrayRef({1, 8}), "vector embeds into multivector");
    auto q = clifford_qbar(mv, table, 1);
    CG_CHECK(std::abs(q.item<float>() - 14.0f) < 1e-5f,
             "grade-1 qbar equals squared vector norm");
}

static void test_rotation_equivariance_linear() {
    torch::manual_seed(1);
    auto table = clifford_build_product_table(CliffordAlgebraConfig::euclidean(2));
    CliffordLinear layer(table, 2, 3);
    auto x = torch::randn({5, 2, table.n_blades});
    const double angle = 0.37;
    auto y1 = layer->forward(clifford_rotate_2d(x, angle, table));
    auto y2 = clifford_rotate_2d(layer->forward(x), angle, table);
    CG_CHECK((y1 - y2).abs().max().item<float>() < 1e-5f,
             "grade-wise linear layer is rotation equivariant");
}

static void test_rotation_equivariance_product_and_activation() {
    torch::manual_seed(2);
    auto table = clifford_build_product_table(CliffordAlgebraConfig::euclidean(2));
    CliffordGradeNorm norm(table);
    CliffordProductLayer prod(table, 2);
    CliffordGatedActivation act(table);
    auto x = torch::randn({4, 2, table.n_blades});
    const double angle = -0.61;
    auto f1 = act->forward(prod->forward(norm->forward(clifford_rotate_2d(x, angle, table))));
    auto f2 = clifford_rotate_2d(act->forward(prod->forward(norm->forward(x))), angle, table);
    CG_CHECK((f1 - f2).abs().max().item<float>() < 1e-4f,
             "product/norm/gated activation stack is rotation equivariant");
}

static void test_cgenn_forward_train_step() {
    torch::manual_seed(3);
    auto table = clifford_build_product_table(CliffordAlgebraConfig::euclidean(2));
    CGENN model(table, 1, 2, 1, 1);
    auto x = torch::randn({6, 1, table.n_blades});
    auto target = torch::zeros({6, 1, table.n_blades});
    auto y = model->forward(x);
    CG_CHECK(y.sizes() == torch::IntArrayRef({6, 1, table.n_blades}),
             "CGENN output shape [B,C,blades]");
    auto loss = torch::mse_loss(y, target);
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(1e-3));
    opt.zero_grad();
    loss.backward();
    opt.step();
    CG_CHECK(std::isfinite(loss.item<float>()), "CGENN train step finite");
}

int main() {
    std::cout << "=== Clifford CGENN Tests ===\n\n";

    std::cout << "-- Algebra --\n";
    test_product_table_2d();
    test_embed_and_qbar();

    std::cout << "\n-- Equivariance --\n";
    test_rotation_equivariance_linear();
    test_rotation_equivariance_product_and_activation();

    std::cout << "\n-- Model --\n";
    test_cgenn_forward_train_step();

    std::cout << "\n=== Results: " << passed << " passed, "
              << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
