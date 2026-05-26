// OhmC1 unit tests
// oh-mah-c, dm/OhmC1, 2026. [151]

#include "models/nlp/ohmc1/ohmc1.h"

#include <torch/torch.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>

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

// ── 1. Config presets ─────────────────────────────────────────────────────────
static void test_config_presets() {
    auto t = OhmC1Config::tiny();
    auto s = OhmC1Config::small();
    auto m = OhmC1Config::medium();
    auto l = OhmC1Config::large();

    check(t.n_layers == 2 && t.dim == 256,      "config_presets_tiny");
    check(s.n_layers == 12 && s.dim == 768,      "config_presets_small");
    check(m.n_layers == 24 && m.dim == 2048,     "config_presets_medium");
    check(l.n_layers == 32 && l.dim == 4096,     "config_presets_large");
    check(t.vocab_size == 64000 && l.vocab_size == 64000, "config_vocab_size");
}

// ── 2. Tiered GQA n_kv_heads ─────────────────────────────────────────────────
static void test_config_n_kv_heads_tiered() {
    check(OhmC1Config::tiny().n_kv_heads  == 1, "kv_heads_tiny_is_1");
    check(OhmC1Config::small().n_kv_heads == 2, "kv_heads_small_is_2");
    check(OhmC1Config::medium().n_kv_heads == 4, "kv_heads_medium_is_4");
    check(OhmC1Config::large().n_kv_heads == 8,  "kv_heads_large_is_8");
}

// ── 3. Size-scaled RoPE theta ─────────────────────────────────────────────────
static void test_config_rope_theta_scaled() {
    check(OhmC1Config::tiny().rope_theta   == 10000.0f,  "rope_theta_tiny");
    check(OhmC1Config::small().rope_theta  == 50000.0f,  "rope_theta_small");
    check(OhmC1Config::medium().rope_theta == 200000.0f, "rope_theta_medium");
    check(OhmC1Config::large().rope_theta  == 500000.0f, "rope_theta_large");
}

// ── 4. RMSNorm shape ──────────────────────────────────────────────────────────
static void test_rmsnorm_shape() {
    OhmC1RMSNorm norm(256);
    auto x = torch::randn({2, 8, 256});
    auto y = norm->forward(x);
    check(y.sizes() == x.sizes(), "rmsnorm_shape");
}

// ── 5. RMSNorm unit norm ──────────────────────────────────────────────────────
static void test_rmsnorm_unit_norm() {
    // With unit weight, the output RMS should be ~1.0 per token vector
    OhmC1RMSNorm norm(64);
    // Force weight to ones (it already is, but be explicit)
    {
        torch::NoGradGuard ng;
        norm->w.fill_(1.0f);
    }
    auto x = torch::randn({4, 10, 64});
    auto y = norm->forward(x);
    auto rms = y.pow(2).mean(-1).sqrt();  // [4, 10]
    auto mean_rms = rms.mean().item<float>();
    check(std::abs(mean_rms - 1.0f) < 0.1f, "rmsnorm_unit_norm");
}

// ── 6. QKNorm shape ───────────────────────────────────────────────────────────
static void test_qknorm_shape() {
    OhmC1QKNorm qknorm(4, 64);
    auto x = torch::randn({2, 8, 4, 64});
    auto y = qknorm->forward(x);
    check(y.sizes() == x.sizes(), "qknorm_shape");
}

// ── 7. QKNorm unit norm ───────────────────────────────────────────────────────
static void test_qknorm_unit_norm() {
    OhmC1QKNorm qknorm(4, 64);
    {
        torch::NoGradGuard ng;
        qknorm->w.fill_(1.0f);
    }
    auto x = torch::randn({2, 8, 4, 64});
    auto y = qknorm->forward(x);
    // Per-head RMS should be ~1.0
    auto norms = y.pow(2).mean(-1).sqrt();  // [2, 8, 4]
    auto mean_norm = norms.mean().item<float>();
    check(std::abs(mean_norm - 1.0f) < 0.1f, "qknorm_unit_norm");
}

// ── 8. Model construction ─────────────────────────────────────────────────────
static void test_model_construction() {
    auto model = make_ohmc1_tiny();
    int64_t total = 0;
    for (auto& p : model->parameters()) total += p.numel();
    check(total > 0, "model_construction_param_count");
}

// ── 9. No tied weights ────────────────────────────────────────────────────────
static void test_no_tied_weights() {
    auto model = make_ohmc1_tiny();
    bool different = model->tok_embeddings->weight.data_ptr()
                  != model->lm_head->weight.data_ptr();
    check(different, "no_tied_weights");
}

// ── 10. Forward shape ─────────────────────────────────────────────────────────
static void test_forward_shape() {
    torch::manual_seed(0);
    auto model = make_ohmc1_tiny();
    model->eval();
    auto tokens = torch::randint(0, 64000, {2, 16});
    auto logits = model->forward(tokens);
    check(logits.dim() == 3, "forward_dim3");
    check(logits.size(0) == 2 && logits.size(1) == 16
          && logits.size(2) == 64000, "forward_shape");
}

// ── 11. Loss computation ──────────────────────────────────────────────────────
static void test_loss_computation() {
    torch::manual_seed(1);
    auto model = make_ohmc1_tiny();
    model->train();
    auto inp = torch::randint(0, 64000, {2, 15});
    auto tgt = torch::randint(0, 64000, {2, 15});
    model->forward(inp, tgt);
    bool defined = model->last_loss.defined();
    bool finite  = defined && std::isfinite(model->last_loss.item<float>());
    bool positive = defined && model->last_loss.item<float>() > 0.0f;
    check(defined && finite && positive, "loss_computation");
}

// ── 12. GQA weight sizes ──────────────────────────────────────────────────────
static void test_gqa_weight_sizes() {
    auto cfg = OhmC1Config::tiny();
    OhmC1Attention attn(cfg);
    int64_t expected_kv_out = cfg.n_kv_heads * cfg.head_dim();
    check(attn->wk->weight.size(0) == expected_kv_out, "gqa_wk_size");
    check(attn->wv->weight.size(0) == expected_kv_out, "gqa_wv_size");
}

// ── 13. GQA rep ratio ────────────────────────────────────────────────────────
static void test_gqa_rep_ratio() {
    auto cfg = OhmC1Config::tiny();
    OhmC1Attention attn(cfg);
    check(attn->n_rep_ == cfg.n_heads / cfg.n_kv_heads, "gqa_rep_ratio");
}

// ── 14. Causal mask — no future leak ─────────────────────────────────────────
static void test_causal_mask_no_leak() {
    torch::manual_seed(2);
    auto model = make_ohmc1_tiny();
    model->eval();
    torch::NoGradGuard ng;

    auto tokens_a = torch::randint(0, 64000, {1, 16});
    auto tokens_b = tokens_a.clone();
    // Change positions 8..15; logits at position 0 should be unaffected
    tokens_b.slice(1, 8, 16) = torch::randint(0, 64000, {1, 8});

    auto logits_a = model->forward(tokens_a);
    auto logits_b = model->forward(tokens_b);

    // logits at t=0 must be identical
    float diff = (logits_a.select(1, 0) - logits_b.select(1, 0)).abs().max().item<float>();
    check(diff < 1e-5f, "causal_mask_no_leak");
}

// ── 15. Sandwich-norm count per block ─────────────────────────────────────────
static void test_sandwich_norm_count() {
    auto cfg = OhmC1Config::tiny();
    OhmC1Block block(cfg);
    int rms_count = 0;
    for (auto& item : block->named_children()) {
        if (dynamic_cast<OhmC1RMSNormImpl*>(item.value().get()))
            ++rms_count;
    }
    check(rms_count == 4, "sandwich_norm_count");
}

// ── 16. SwiGLU nonlinearity ───────────────────────────────────────────────────
static void test_swiglu_nonlinearity() {
    torch::manual_seed(3);
    OhmC1FFN ffn(64, 128);
    auto x = torch::randn({2, 8, 64});
    auto y1 = ffn->forward(x);
    auto y2 = ffn->forward(x * 2.0f);
    // If FFN were linear, y2 = 2*y1. SwiGLU is not linear.
    float rel_diff = ((y2 - 2.0f * y1).abs().mean() / y1.abs().mean()).item<float>();
    check(rel_diff > 0.01f, "swiglu_nonlinearity");
}

// ── 17. KV-cache matches batch prefill ───────────────────────────────────────
static void test_kv_cache_matches_batch() {
    torch::manual_seed(4);
    auto model = make_ohmc1_tiny();
    model->eval();

    int64_t T = 8;
    auto tokens = torch::randint(0, 64000, {1, T});

    // Batch forward logits at last position
    torch::Tensor batch_logits;
    {
        torch::NoGradGuard ng;
        auto all_logits = model->forward(tokens);
        batch_logits = all_logits.select(1, T - 1);  // [1, vocab]
    }

    // KV-cache forward
    std::vector<torch::Tensor> kv_k, kv_v;
    model->init_kv_caches(kv_k, kv_v);
    torch::Tensor cache_logits;
    {
        torch::NoGradGuard ng;
        for (int64_t i = 0; i < T; ++i)
            cache_logits = model->forward_one(tokens[0][i].item<int64_t>(), i, kv_k, kv_v);
    }

    float diff = (batch_logits - cache_logits).abs().max().item<float>();
    check(diff < 1e-4f, "kv_cache_matches_batch");
}

// ── 18. Gradient flow ─────────────────────────────────────────────────────────
static void test_gradient_flow() {
    torch::manual_seed(5);
    auto model = make_ohmc1_tiny();
    model->train();

    auto tokens = torch::randint(0, 64000, {1, 8});
    auto tgt    = torch::randint(0, 64000, {1, 8});
    model->forward(tokens, tgt);
    model->last_loss.backward();

    bool any_grad = false;
    for (auto& p : model->parameters()) {
        if (p.grad().defined() && p.grad().abs().sum().item<float>() > 0.0f) {
            any_grad = true;
            break;
        }
    }
    check(any_grad, "gradient_flow");
}

// ── 19. LR schedule ───────────────────────────────────────────────────────────
static void test_lr_schedule() {
    OhmC1TrainConfig cfg;
    cfg.lr           = 3e-4;
    cfg.min_lr       = 1e-5;
    cfg.warmup_iters = 100;
    cfg.max_iters    = 1000;

    double lr_0   = ohmc1_lr_schedule(0, cfg);
    double lr_50  = ohmc1_lr_schedule(50, cfg);
    double lr_100 = ohmc1_lr_schedule(100, cfg);
    double lr_500 = ohmc1_lr_schedule(500, cfg);
    double lr_end = ohmc1_lr_schedule(1000, cfg);

    check(lr_0 < lr_50 && lr_50 < lr_100, "lr_schedule_warmup_increasing");
    check(lr_500 < lr_100,                 "lr_schedule_cosine_decreasing");
    check(lr_end <= cfg.min_lr + 1e-8,     "lr_schedule_reaches_min");
}

// ── 20. Generation prefix ─────────────────────────────────────────────────────
static void test_generation_prefix() {
    torch::manual_seed(6);
    auto model = make_ohmc1_tiny();
    model->eval();

    std::vector<int64_t> prompt = {10, 20, 30};
    auto gen = model->generate(prompt, /*max_new_tokens=*/5,
                               /*temperature=*/1.0f, /*top_p=*/0.9f,
                               /*eos_id=*/-1);  // eos=-1 so it never stops early

    bool starts_with_prompt = gen.size() >= prompt.size();
    if (starts_with_prompt) {
        for (size_t i = 0; i < prompt.size(); ++i)
            if (gen[i] != prompt[i]) { starts_with_prompt = false; break; }
    }
    check(starts_with_prompt, "generation_prefix");
    check(gen.size() >= prompt.size(), "generation_length");
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== OhmC1 Tests ===\n";
    torch::manual_seed(42);

    test_config_presets();
    test_config_n_kv_heads_tiered();
    test_config_rope_theta_scaled();
    test_rmsnorm_shape();
    test_rmsnorm_unit_norm();
    test_qknorm_shape();
    test_qknorm_unit_norm();
    test_model_construction();
    test_no_tied_weights();
    test_forward_shape();
    test_loss_computation();
    test_gqa_weight_sizes();
    test_gqa_rep_ratio();
    test_causal_mask_no_leak();
    test_sandwich_norm_count();
    test_swiglu_nonlinearity();
    test_kv_cache_matches_batch();
    test_gradient_flow();
    test_lr_schedule();
    test_generation_prefix();

    std::printf("\n=== %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
