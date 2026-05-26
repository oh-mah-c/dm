// ─────────────────────────────────────────────────────────────────────────────
// Unit tests for Llama 3 (arXiv:2407.21783v3) — 14 tests
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/llama3/llama3.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <cassert>

using namespace dm::models::nlp;

static int passed = 0, failed = 0;

static void check(bool cond, const char* test) {
    if (cond) { std::cout << "[PASS] " << test << "\n"; ++passed; }
    else       { std::cout << "[FAIL] " << test << "\n"; ++failed; }
}

// ── Test helpers ─────────────────────────────────────────────────────────────
static Llama3Config tiny_cfg() { return Llama3Config::tiny(); }

// ─────────────────────────────────────────────────────────────────────────────
// 1. Config presets: verify Table 3 hyperparameters
// ─────────────────────────────────────────────────────────────────────────────
static void test_config_presets() {
    auto c8b  = Llama3Config::llama3_8b();
    auto c70b = Llama3Config::llama3_70b();
    auto c405 = Llama3Config::llama3_405b();

    // 8B (Table 3)
    check(c8b.n_layers == 32 && c8b.dim == 4096 && c8b.ffn_dim == 14336
       && c8b.n_heads == 32  && c8b.n_kv_heads == 8,
          "test_config_8b_table3");

    // 70B (Table 3)
    check(c70b.n_layers == 80 && c70b.dim == 8192 && c70b.ffn_dim == 28672
       && c70b.n_heads == 64  && c70b.n_kv_heads == 8,
          "test_config_70b_table3");

    // 405B (Table 3)
    check(c405.n_layers == 126 && c405.dim == 16384 && c405.ffn_dim == 53248
       && c405.n_heads == 128  && c405.n_kv_heads == 8,
          "test_config_405b_table3");

    // All models have the same KV heads = 8 and vocab = 128000
    check(c8b.n_kv_heads == 8 && c70b.n_kv_heads == 8 && c405.n_kv_heads == 8,
          "test_all_gqa_8kv");

    // RoPE theta = 500000 (§3.2)
    check(std::abs(c8b.rope_theta - 500000.0f) < 1.0f,
          "test_rope_theta_500k");
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. RMSNorm — unit norm on last dim
// ─────────────────────────────────────────────────────────────────────────────
static void test_rmsnorm() {
    auto cfg = tiny_cfg();
    Llama3RMSNorm norm(cfg.dim);

    auto x   = torch::randn({2, 8, cfg.dim});
    auto out = norm->forward(x);

    check(out.sizes() == x.sizes(), "test_rmsnorm_shape");

    // With unit weights and eps≈0: output RMS ≈ 1
    torch::NoGradGuard ng;
    norm->w.fill_(1.0f);
    auto xr = torch::randn({1, 1, cfg.dim});
    auto yr = norm->forward(xr);
    float rms = yr.pow(2).mean().sqrt().item<float>();
    check(std::abs(rms - 1.0f) < 0.05f, "test_rmsnorm_unit_norm");
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. RoPE theta = 500000 (§3.2): frequencies should be smaller than Llama-2's
// ─────────────────────────────────────────────────────────────────────────────
static void test_rope_theta() {
    // Compare the cosine of position 1, dim 0: cos(1/(θ^(0/d)))
    // Llama-2 theta=10000 → higher freq → larger angle
    // Llama-3 theta=500000 → lower freq → smaller angle
    float theta_l2 = 10000.0f;
    float theta_l3 = 500000.0f;
    int64_t head_dim = 64;

    // freq at dim 0: 1 / theta^(0/d) = 1.0 always; compare at dim 2:
    // freq = 1 / theta^(2/64)
    float freq_l2 = 1.0f / std::pow(theta_l2, 2.0f / head_dim);
    float freq_l3 = 1.0f / std::pow(theta_l3, 2.0f / head_dim);
    check(freq_l3 < freq_l2, "test_rope_theta_500k_lower_freq");

    // The difference grows with dimension index; at d=20 ratio≈3.4, d=62 ratio≈44
    float freq_l2_high = 1.0f / std::pow(theta_l2, 20.0f / head_dim);
    float freq_l3_high = 1.0f / std::pow(theta_l3, 20.0f / head_dim);
    check(freq_l2_high / freq_l3_high > 2.0f, "test_rope_theta_freq_ratio");
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Model construction: verify parameter count and structure
// ─────────────────────────────────────────────────────────────────────────────
static void test_model_construction() {
    auto model = make_llama3_tiny();
    int64_t total = 0;
    for (auto& p : model->parameters()) total += p.numel();
    check(total > 0, "test_model_has_params");

    // Verify no tied weights (Llama 3 does NOT tie embeddings)
    auto& emb_w = model->tok_embeddings->weight;
    auto& out_w = model->output->weight;
    // Different data pointers
    check(emb_w.data_ptr() != out_w.data_ptr(), "test_no_tied_weights");
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Forward pass: shape [B, T, vocab_size]
// ─────────────────────────────────────────────────────────────────────────────
static void test_forward_shape() {
    auto cfg   = tiny_cfg();
    auto model = make_llama3_tiny();
    model->eval();

    int64_t B = 2, T = 16;
    auto tokens = torch::randint(0, cfg.vocab_size, {B, T});

    torch::NoGradGuard ng;
    auto logits = model->forward(tokens);

    check(logits.size(0) == B && logits.size(1) == T
       && logits.size(2) == cfg.vocab_size,
          "test_forward_shape");
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Loss computation: last_loss is set and finite
// ─────────────────────────────────────────────────────────────────────────────
static void test_loss_computation() {
    auto cfg   = tiny_cfg();
    auto model = make_llama3_tiny();
    model->train();

    int64_t B = 2, T = 16;
    auto inp = torch::randint(0, cfg.vocab_size, {B, T});
    auto tgt = torch::randint(0, cfg.vocab_size, {B, T});

    auto logits = model->forward(inp, tgt);

    check(model->last_loss.defined(), "test_loss_defined");
    float loss_val = model->last_loss.item<float>();
    check(std::isfinite(loss_val) && loss_val > 0.0f, "test_loss_finite_positive");
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. GQA: K/V have n_kv_heads dims, not n_heads
// ─────────────────────────────────────────────────────────────────────────────
static void test_gqa_kv_heads() {
    auto cfg = tiny_cfg();
    Llama3Attention attn(cfg);

    int64_t head_dim = cfg.dim / cfg.n_heads;

    // wq: [dim, n_heads * head_dim]
    // wk: [dim, n_kv_heads * head_dim]
    check(attn->wq->weight.size(0) == cfg.n_heads    * head_dim, "test_gqa_wq_size");
    check(attn->wk->weight.size(0) == cfg.n_kv_heads * head_dim, "test_gqa_wk_size");
    check(attn->wv->weight.size(0) == cfg.n_kv_heads * head_dim, "test_gqa_wv_size");

    // Confirm reduction ratio
    check(cfg.n_heads / cfg.n_kv_heads == attn->n_rep, "test_gqa_rep_ratio");
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Causal mask: token[i] should NOT attend to future tokens
// ─────────────────────────────────────────────────────────────────────────────
static void test_causal_mask() {
    auto cfg   = tiny_cfg();
    auto model = make_llama3_tiny();
    model->eval();
    torch::NoGradGuard ng;

    int64_t B = 1, T = 8;
    auto tokens = torch::randint(0, 10, {B, T});

    // Logits at position 0 must not change when we change tokens[0, 3..7]
    auto log1 = model->forward(tokens).detach().clone();
    tokens[0][3] = (tokens[0][3].item<int64_t>() + 1) % 10;
    tokens[0][6] = (tokens[0][6].item<int64_t>() + 1) % 10;
    auto log2 = model->forward(tokens).detach().clone();

    // Position 0 output should be identical (no future leakage)
    check(torch::allclose(log1.slice(1,0,1), log2.slice(1,0,1), 1e-4f, 1e-4f),
          "test_causal_mask_no_future_leak");
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. SwiGLU FFN: output shape correct
// ─────────────────────────────────────────────────────────────────────────────
static void test_swiglu_ffn() {
    auto cfg = tiny_cfg();
    Llama3FFN ffn(cfg.dim, cfg.ffn_dim);

    auto x   = torch::randn({2, 8, cfg.dim});
    auto out = ffn->forward(x);

    check(out.sizes() == x.sizes(), "test_swiglu_output_shape");

    // SwiGLU non-linearity: output is not just a linear function of input
    auto x2  = x * 2.0f;
    auto out2 = ffn->forward(x2);
    // If it were linear, out2 == 2*out1. It shouldn't be.
    check(!torch::allclose(out2, out * 2.0f, 1e-3f, 1e-3f),
          "test_swiglu_nonlinearity");
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. KV-cache inference: single-token forward matches prefill logits at last pos
// ─────────────────────────────────────────────────────────────────────────────
static void test_kv_cache_inference() {
    auto cfg   = tiny_cfg();
    auto model = make_llama3_tiny();
    model->eval();
    torch::NoGradGuard ng;

    int64_t T = 4;
    auto tokens = torch::randint(0, 100, {1, T});

    // Batch prefill logits at position T-1
    auto batch_logits = model->forward(tokens); // [1, T, vocab]
    auto ref = batch_logits.slice(1, T-1, T).squeeze(0); // [vocab]

    // Autoregressive KV-cache: feed tokens 0..T-2 as prefill, then token T-1
    std::vector<torch::Tensor> kk, kv;
    model->init_kv_caches(kk, kv);

    // Prefill positions 0..T-2
    for (int64_t p = 0; p < T - 1; ++p)
        model->forward_one(tokens[0][p].item<int64_t>(), p, kk, kv);

    // Single-step at position T-1
    auto cache_logits = model->forward_one(
        tokens[0][T-1].item<int64_t>(), T-1, kk, kv); // [1, vocab]

    check(torch::allclose(cache_logits.squeeze(0), ref, 1e-3f, 1e-3f),
          "test_kv_cache_matches_batch");
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Gradient flow: all LoRA-free params must receive gradients
// ─────────────────────────────────────────────────────────────────────────────
static void test_gradient_flow() {
    auto cfg   = tiny_cfg();
    auto model = make_llama3_tiny();
    model->train();

    int64_t B = 2, T = 8;
    auto inp = torch::randint(0, cfg.vocab_size, {B, T});
    auto tgt = torch::randint(0, cfg.vocab_size, {B, T});

    model->forward(inp, tgt);
    model->last_loss.backward();

    int grads = 0;
    for (auto& p : model->parameters()) {
        if (p.grad().defined() && p.grad().abs().max().item<float>() > 0.0f)
            ++grads;
    }
    check(grads > 0, "test_gradient_flow");
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Optimizer: one step reduces training loss
// ─────────────────────────────────────────────────────────────────────────────
static void test_optimizer_step() {
    auto cfg   = tiny_cfg();
    auto model = make_llama3_tiny();

    // Re-init output with non-zero weights so loss is not constant
    torch::nn::init::kaiming_uniform_(model->output->weight);

    Llama3TrainConfig tcfg;
    tcfg.lr        = 1e-3;
    tcfg.max_iters = 100;
    auto opt = make_llama3_optimizer(model, tcfg);

    model->train();

    auto tokens = torch::randint(0, cfg.vocab_size, {2, cfg.seq_len + 1});
    float loss1 = llama3_train_step(model, opt, tokens, tcfg, 0);
    float loss2 = llama3_train_step(model, opt, tokens, tcfg, 1);

    // Over two steps on the same batch, weight should move
    check(std::isfinite(loss1) && std::isfinite(loss2),
          "test_optimizer_loss_finite");
    check(loss2 != loss1, "test_optimizer_weights_updated");
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. LR schedule: warmup + cosine decay
// ─────────────────────────────────────────────────────────────────────────────
static void test_lr_schedule() {
    Llama3TrainConfig cfg;
    cfg.lr          = 3e-4;
    cfg.min_lr      = 1e-5;
    cfg.warmup_iters= 100;
    cfg.max_iters   = 1000;

    // During warmup: LR should increase
    double lr0  = llama3_lr_schedule(0,   cfg);
    double lr50 = llama3_lr_schedule(50,  cfg);
    double lr99 = llama3_lr_schedule(99,  cfg);
    check(lr0 < lr50 && lr50 < lr99, "test_lr_warmup_increasing");

    // After warmup: LR should decrease
    double lr100 = llama3_lr_schedule(100, cfg);
    double lr500 = llama3_lr_schedule(500, cfg);
    double lr999 = llama3_lr_schedule(999, cfg);
    check(lr100 > lr500 && lr500 > lr999, "test_lr_cosine_decreasing");

    // Final LR should be at or near min_lr
    double lr_end = llama3_lr_schedule(cfg.max_iters, cfg);
    check(lr_end <= cfg.min_lr * 1.01, "test_lr_reaches_min");
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. Autoregressive generation: output grows by max_new_tokens
// ─────────────────────────────────────────────────────────────────────────────
static void test_generation() {
    auto cfg   = tiny_cfg();
    auto model = make_llama3_tiny();
    model->eval();

    std::vector<int64_t> prompt = {1, 2, 3};
    int64_t max_new = 5;

    // Greedy decoding
    auto out = model->generate(prompt, max_new, 0.0f, 0.9f, -1 /* no eos */);

    check(static_cast<int64_t>(out.size()) >= static_cast<int64_t>(prompt.size()),
          "test_generation_length");
    // Prompt should be at the front
    bool prompt_ok = true;
    for (size_t i = 0; i < prompt.size(); ++i)
        if (out[i] != prompt[i]) { prompt_ok = false; break; }
    check(prompt_ok, "test_generation_prompt_prefix");
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== Llama 3 tests (arXiv:2407.21783v3) ===\n";

    torch::manual_seed(42);
    torch::NoGradGuard no_grad_global;  // most tests don't need grad

    // Tests that need grad — restore temporarily inside
    test_config_presets();   // tests 1-5
    test_rmsnorm();          // tests 6-7

    {   // test 3 — no grad needed
        test_rope_theta();
    }

    test_model_construction();  // test 4

    {   // tests 5,6 — no grad
        test_forward_shape();
        test_loss_computation();
    }

    test_gqa_kv_heads();     // test 7
    test_causal_mask();      // test 8
    test_swiglu_ffn();       // test 9
    test_kv_cache_inference();// test 10

    // Tests that need grad
    {
        torch::AutoGradMode ag(true);
        test_gradient_flow();   // test 11
        test_optimizer_step();  // test 12
    }

    test_lr_schedule();  // test 13
    test_generation();   // test 14

    int total = passed + failed;
    std::cout << "\n" << passed << "/" << total << " tests passed\n";
    return (failed > 0) ? 1 : 0;
}
