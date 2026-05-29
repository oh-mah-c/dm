#include "models/nlp/ohmc2/ohmc2.h"

#include <torch/torch.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace dm::models::nlp;

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        std::printf("  PASS  %s\n", name);
        ++passed;
    } else {
        std::printf("  FAIL  %s\n", name);
        ++failed;
    }
}

static OhmC2Config test_cfg(int64_t experts = 2, int64_t layers = 2) {
    auto c = OhmC2Config::tiny();
    c.dim = 32;
    c.ffn_dim = 64;
    c.n_layers = layers;
    c.n_heads = 4;
    c.n_experts = experts;
    c.kan_grid = 2;
    c.kan_order = 2;
    c.vocab_size = 128;
    c.max_seq = 8;
    c.dropout = 0.0;
    return c;
}

static void test_presets() {
    auto t = OhmC2Config::tiny();
    auto s = OhmC2Config::small();
    auto m = OhmC2Config::medium();
    auto l = OhmC2Config::large();
    check(t.dim == 256 && t.n_layers == 8 && t.n_heads == 4 &&
          t.n_experts == 1 && t.kan_grid == 3 && t.max_seq == 512,
          "preset_tiny");
    check(s.dim == 512 && s.n_layers == 16 && s.n_heads == 8 &&
          s.n_experts == 4 && s.kan_grid == 5 && s.max_seq == 2048,
          "preset_small");
    check(m.dim == 1024 && m.n_layers == 24 && m.n_heads == 16 &&
          m.n_experts == 8 && m.kan_grid == 8 && m.max_seq == 4096,
          "preset_medium");
    check(l.dim == 2048 && l.n_layers == 32 && l.n_heads == 32 &&
          l.n_experts == 16 && l.kan_grid == 12 && l.max_seq == 8192,
          "preset_large");
}

static void test_forward_and_loss() {
    torch::manual_seed(1);
    auto cfg = test_cfg(2, 2);
    auto model = OhmC2LLM(cfg);
    model->eval();
    auto tokens = torch::randint(0, cfg.vocab_size, {2, 4}, torch::kLong);
    auto logits = model->forward(tokens);
    check(logits.dim() == 3 && logits.size(0) == 2 && logits.size(1) == 4 &&
          logits.size(2) == cfg.vocab_size, "forward_shape");

    model->train();
    auto targets = torch::randint(0, cfg.vocab_size, {2, 4}, torch::kLong);
    model->forward(tokens, targets);
    check(model->last_loss.defined() &&
          std::isfinite(model->last_loss.item<float>()) &&
          model->last_loss.item<float>() > 0.0f, "loss_finite_positive");
}

static void test_per_token_routing() {
    torch::manual_seed(2);
    auto cfg = test_cfg(3, 1);
    cfg.ffn_type = "kan_moe";
    OhmC2DualGateBlock block(cfg, 0);
    {
        torch::NoGradGuard ng;
        block->router->weight.zero_();
        block->router->bias.zero_();
        block->router->weight.index_put_({0, 0}, 1.0f);
        block->router->weight.index_put_({1, 0}, -1.0f);
    }
    auto x = torch::zeros({1, 2, cfg.dim});
    x.index_put_({0, 0, 0}, 2.0f);
    x.index_put_({0, 1, 0}, -2.0f);
    auto y = block->ffn_route(x);
    check(y.size(0) == 1 && y.size(1) == 2 && y.size(2) == cfg.ffn_dim,
          "routing_output_shape");
}

static void test_dense_single_expert_path() {
    torch::manual_seed(3);
    auto cfg = test_cfg(1, 1);
    cfg.ffn_type = "kan";
    OhmC2DualGateBlock block(cfg, 0);
    auto x = torch::randn({2, 3, cfg.dim});
    auto y = block->ffn_route(x);
    check(y.size(0) == 2 && y.size(1) == 3 && y.size(2) == cfg.ffn_dim,
          "single_expert_shape");
    check(torch::isfinite(y).all().item<bool>(), "single_expert_finite");
}

static void test_cpu_quality_swiglu() {
    torch::manual_seed(7);
    auto cfg = OhmC2Config::cpu_quality();
    cfg.dim = 32;
    cfg.ffn_dim = 64;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.vocab_size = 128;
    cfg.max_seq = 8;
    auto model = OhmC2LLM(cfg);
    auto tokens = torch::randint(0, cfg.vocab_size, {1, 5}, torch::kLong);
    auto logits = model->forward(tokens, tokens);
    check(cfg.ffn_type == "swiglu" && cfg.use_rope, "cpu_quality_defaults");
    check(logits.size(0) == 1 && logits.size(1) == 5 &&
          logits.size(2) == cfg.vocab_size, "swiglu_forward_shape");
    check(model->last_loss.defined() && std::isfinite(model->last_loss.item<float>()),
          "swiglu_loss_finite");
}

static void test_kan_modes() {
    torch::manual_seed(8);
    auto cfg = test_cfg(1, 1);
    cfg.ffn_type = "kan";
    auto model_kan = OhmC2LLM(cfg);
    auto tokens = torch::randint(0, cfg.vocab_size, {1, 3}, torch::kLong);
    check(torch::isfinite(model_kan->forward(tokens)).all().item<bool>(), "kan_mode_runs");

    cfg.n_experts = 2;
    cfg.ffn_type = "kan_moe";
    auto model_moe = OhmC2LLM(cfg);
    check(torch::isfinite(model_moe->forward(tokens)).all().item<bool>(), "kan_moe_mode_runs");
}

static void test_realformer_state_shapes() {
    torch::manual_seed(4);
    auto cfg = test_cfg(2, 2);
    auto model = OhmC2LLM(cfg);
    auto tokens = torch::randint(0, cfg.vocab_size, {1, 4}, torch::kLong);
    model->forward(tokens);
    check((int64_t)model->last_prev_scores.size() == cfg.n_layers,
          "prev_scores_layer_count");
    bool ok = true;
    for (const auto& p : model->last_prev_scores) {
        ok = ok && p.dim() == 4 && p.size(0) == 1 && p.size(1) == cfg.n_heads &&
             p.size(2) == 4 && p.size(3) == 4;
    }
    check(ok, "prev_scores_shape_BHTT");

    auto st = model->init_generation_state(1);
    check((int64_t)st.k_ring.size() == cfg.n_layers &&
          st.score_cache[0].size(0) == 1 &&
          st.score_cache[0].size(1) == cfg.n_heads &&
          st.score_cache[0].size(2) == cfg.max_seq,
          "generation_score_cache_shape");
}

static void test_forward_step_matches_prefill_before_wrap() {
    torch::manual_seed(5);
    auto cfg = test_cfg(2, 2);
    auto model = OhmC2LLM(cfg);
    model->eval();
    torch::NoGradGuard ng;
    auto tokens = torch::randint(0, cfg.vocab_size, {1, 4}, torch::kLong);
    auto batch_last = model->forward(tokens).select(1, 3);
    auto st = model->init_generation_state(1);
    torch::Tensor step_logits;
    for (int64_t i = 0; i < tokens.size(1); ++i)
        step_logits = model->forward_step(tokens.select(1, i), st);
    auto diff = (batch_last - step_logits).abs().max().item<float>();
    check(diff < 1e-4f, "forward_step_matches_prefill");
}

static void test_ring_wrap_finite() {
    torch::manual_seed(6);
    auto cfg = test_cfg(2, 2);
    cfg.max_seq = 4;
    auto model = OhmC2LLM(cfg);
    model->eval();
    auto st = model->init_generation_state(1);
    torch::Tensor logits;
    for (int64_t i = 0; i < cfg.max_seq + 2; ++i) {
        auto tok = torch::tensor({i % cfg.vocab_size}, torch::kLong);
        logits = model->forward_step(tok, st);
    }
    check(st.pos == cfg.max_seq + 2 && st.valid_len == cfg.max_seq,
          "ring_state_wrap_counters");
    check(torch::isfinite(logits).all().item<bool>(), "ring_wrap_logits_finite");
}

static void test_block_drop_train_only() {
    auto cfg = test_cfg(2, 1);
    cfg.dropout = 1.0;
    cfg.max_seq = 8;
    OhmC2DualGateBlock block(cfg, 0);
    auto x = torch::ones({1, 4, cfg.dim});
    block->train();
    auto y_train = block->block_drop->forward(x);
    block->eval();
    auto y_eval = block->block_drop->forward(x);
    check((y_train - x).abs().sum().item<float>() > 0.0f,
          "blockdrop_changes_training");
    check((y_eval - x).abs().max().item<float>() == 0.0f,
          "blockdrop_identity_eval");
}

int main() {
    try {
        test_presets();
        test_forward_and_loss();
        test_per_token_routing();
        test_dense_single_expert_path();
        test_cpu_quality_swiglu();
        test_kan_modes();
        test_realformer_state_shapes();
        test_forward_step_matches_prefill_before_wrap();
        test_ring_wrap_finite();
        test_block_drop_train_only();
    } catch (const std::exception& e) {
        std::printf("  FAIL  exception: %s\n", e.what());
        ++failed;
    }
    std::printf("\nOhmC2 tests: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
