// ─────────────────────────────────────────────────────────────────────────────
// QuanvNd C++ unit tests — wired into the PyTorch gtest suite
// ─────────────────────────────────────────────────────────────────────────────

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch/nn/modules/quanv_nd.h>

#include <cmath>

using namespace dm::prim;

// ── Construction ──────────────────────────────────────────────────────────────

TEST(QuanvNdTest, Construct1d) {
  QuanvNdImpl<1> layer(4);
  ASSERT_EQ(layer.options.in_channels(), 4);
  ASSERT_TRUE(layer.delta_theta.defined());
  ASSERT_EQ(layer.delta_theta.size(0), 4);
  ASSERT_EQ(layer.delta_theta.size(1), 2);  // N_QUBITS = 2^1
}

TEST(QuanvNdTest, Construct2d) {
  QuanvNdImpl<2> layer(3);
  ASSERT_EQ(layer.delta_theta.size(0), 3);
  ASSERT_EQ(layer.delta_theta.size(1), 4);  // N_QUBITS = 2^2
}

TEST(QuanvNdTest, Construct3d) {
  QuanvNdImpl<3> layer(2);
  ASSERT_EQ(layer.delta_theta.size(0), 2);
  ASSERT_EQ(layer.delta_theta.size(1), 8);  // N_QUBITS = 2^3
}

// ── Output shape ─────────────────────────────────────────────────────────────

TEST(QuanvNdTest, OutputShape1d) {
  torch::manual_seed(0);
  QuanvNdImpl<1> layer(3);
  auto x = torch::randn({2, 3, 16});          // [B=2, C=3, L=16]
  auto y = layer.forward(x);
  // Expected: [2, 3*2, 8]
  ASSERT_EQ(y.dim(), 3);
  ASSERT_EQ(y.size(0), 2);
  ASSERT_EQ(y.size(1), 6);   // 3 * 2 qubits
  ASSERT_EQ(y.size(2), 8);   // 16/2
}

TEST(QuanvNdTest, OutputShape2d) {
  torch::manual_seed(0);
  QuanvNdImpl<2> layer(3);
  auto x = torch::randn({1, 3, 8, 8});        // [B=1, C=3, H=8, W=8]
  auto y = layer.forward(x);
  // Expected: [1, 12, 4, 4]
  ASSERT_EQ(y.dim(), 4);
  ASSERT_EQ(y.size(0), 1);
  ASSERT_EQ(y.size(1), 12);  // 3 * 4 qubits
  ASSERT_EQ(y.size(2), 4);
  ASSERT_EQ(y.size(3), 4);
}

TEST(QuanvNdTest, OutputShape3d) {
  torch::manual_seed(0);
  QuanvNdImpl<3> layer(2);
  auto x = torch::randn({1, 2, 4, 4, 4});     // [B=1, C=2, D=4, H=4, W=4]
  auto y = layer.forward(x);
  // Expected: [1, 16, 2, 2, 2]
  ASSERT_EQ(y.dim(), 5);
  ASSERT_EQ(y.size(0), 1);
  ASSERT_EQ(y.size(1), 16);  // 2 * 8 qubits
  ASSERT_EQ(y.size(2), 2);
  ASSERT_EQ(y.size(3), 2);
  ASSERT_EQ(y.size(4), 2);
}

// ── Values lie in [0, 1] (Born-rule probabilities) ───────────────────────────

TEST(QuanvNdTest, OutputRangeD1) {
  torch::manual_seed(1);
  QuanvNdImpl<1> layer(2);
  auto x = torch::randn({4, 2, 32}) * 5.0;   // deliberately large inputs
  auto y = layer.forward(x);
  ASSERT_TRUE((y >= 0.0f).all().item<bool>());
  ASSERT_TRUE((y <= 1.0f).all().item<bool>());
}

TEST(QuanvNdTest, OutputRangeD2) {
  torch::manual_seed(2);
  QuanvNdImpl<2> layer(1);
  auto x = torch::randn({2, 1, 16, 16}) * 3.0;
  auto y = layer.forward(x);
  ASSERT_TRUE((y >= 0.0f).all().item<bool>());
  ASSERT_TRUE((y <= 1.0f).all().item<bool>());
}

TEST(QuanvNdTest, OutputRangeD3) {
  torch::manual_seed(3);
  QuanvNdImpl<3> layer(1);
  auto x = torch::randn({1, 1, 4, 4, 4});
  auto y = layer.forward(x);
  ASSERT_TRUE((y >= 0.0f).all().item<bool>());
  ASSERT_TRUE((y <= 1.0f).all().item<bool>());
}

// ── Gradient flows through δθ ─────────────────────────────────────────────────

TEST(QuanvNdTest, GradientThroughDeltaTheta1d) {
  torch::manual_seed(10);
  QuanvNdImpl<1> layer(2);
  auto x = torch::randn({1, 2, 8});
  auto y = layer.forward(x).sum();
  y.backward();
  ASSERT_TRUE(layer.delta_theta.grad().defined());
  ASSERT_FALSE(layer.delta_theta.grad().isnan().any().item<bool>());
  ASSERT_GT(layer.delta_theta.grad().abs().sum().item<float>(), 0.0f);
}

TEST(QuanvNdTest, GradientThroughDeltaTheta2d) {
  torch::manual_seed(11);
  QuanvNdImpl<2> layer(2);
  auto x = torch::randn({1, 2, 8, 8});
  auto y = layer.forward(x).sum();
  y.backward();
  ASSERT_TRUE(layer.delta_theta.grad().defined());
  ASSERT_GT(layer.delta_theta.grad().abs().sum().item<float>(), 0.0f);
}

TEST(QuanvNdTest, GradientThroughDeltaTheta3d) {
  torch::manual_seed(12);
  QuanvNdImpl<3> layer(1);
  auto x = torch::randn({1, 1, 4, 4, 4});
  auto y = layer.forward(x).sum();
  y.backward();
  ASSERT_TRUE(layer.delta_theta.grad().defined());
  ASSERT_GT(layer.delta_theta.grad().abs().sum().item<float>(), 0.0f);
}

// ── Parameters can be updated by Adam ────────────────────────────────────────

TEST(QuanvNdTest, AdamStep2d) {
  torch::manual_seed(20);
  QuanvNdImpl<2> layer(1);
  torch::optim::Adam opt(layer.parameters(),
                         torch::optim::AdamOptions(1e-2));
  auto before = layer.delta_theta.clone().detach();

  auto x = torch::randn({1, 1, 8, 8});
  opt.zero_grad();
  layer.forward(x).sum().backward();
  opt.step();

  auto after = layer.delta_theta.detach();
  ASSERT_FALSE((after - before).abs().max().item<float>() < 1e-9f)
    << "delta_theta did not change after Adam step";
}

// ── Determinism: same input + same weights → same output ─────────────────────

TEST(QuanvNdTest, Deterministic2d) {
  torch::manual_seed(30);
  QuanvNdImpl<2> layer(2);
  auto x = torch::randn({2, 2, 8, 8});
  auto y1 = layer.forward(x);
  auto y2 = layer.forward(x);
  ASSERT_TRUE((y1 - y2).abs().max().item<float>() < 1e-6f);
}

// ── assume_normalised flag disables tanh pre-scaling ─────────────────────────

TEST(QuanvNdTest, AssumeNormalisedFlag) {
  torch::manual_seed(40);
  // With assume_normalised=true, inputs in (0, π/2) should yield output
  // values closer to exact trig values (no tanh warp).
  QuanvNdImpl<1> layerN(QuanvNdOptions(1).assume_normalised(true));
  QuanvNdImpl<1> layerU(QuanvNdOptions(1).assume_normalised(false));
  // Copy same weights
  {
    torch::NoGradGuard ng;
    layerU.delta_theta.copy_(layerN.delta_theta);
  }
  // Tiny uniform input: tanh(x) ≈ x for small x, so both should be close
  auto x = torch::full({1, 1, 4}, 0.01f);
  auto yN = layerN.forward(x);
  auto yU = layerU.forward(x);
  ASSERT_LT((yN - yU).abs().max().item<float>(), 0.05f);
}

// ── TORCH_MODULE wrappers ─────────────────────────────────────────────────────

TEST(QuanvNdTest, ModuleHolderQuanv1d) {
  Quanv1d m(2);
  auto x = torch::randn({1, 2, 8});
  auto y = m(x);
  ASSERT_EQ(y.size(1), 4);   // 2 ch × 2 qubits
}

TEST(QuanvNdTest, ModuleHolderQuanv2d) {
  Quanv2d m(3);
  auto x = torch::randn({1, 3, 8, 8});
  auto y = m(x);
  ASSERT_EQ(y.size(1), 12);  // 3 × 4
  ASSERT_EQ(y.size(2), 4);
}

TEST(QuanvNdTest, ModuleHolderQuanv3d) {
  Quanv3d m(1);
  auto x = torch::randn({1, 1, 4, 4, 4});
  auto y = m(x);
  ASSERT_EQ(y.size(1), 8);   // 1 × 8
}

// ── pretty_print smoke test ───────────────────────────────────────────────────

TEST(QuanvNdTest, PrettyPrint) {
  std::ostringstream ss;
  QuanvNdImpl<2> layer(3);
  layer.pretty_print(ss);
  auto s = ss.str();
  ASSERT_NE(s.find("QuanvNd"), std::string::npos);
  ASSERT_NE(s.find("in_channels=3"), std::string::npos);
  ASSERT_NE(s.find("n_qubits=4"), std::string::npos);
}
