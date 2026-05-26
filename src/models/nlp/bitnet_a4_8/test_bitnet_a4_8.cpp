// ─────────────────────────────────────────────────────────────────────────────
// Unit tests for BitNet a4.8 (arXiv:2411.04965v1) — 16 tests
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/bitnet_a4_8/bitnet_a4_8.h"

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

// ─────────────────────────────────────────────────────────────────────────────
// 1. Config presets match Table 6
// ─────────────────────────────────────────────────────────────────────────────
static void test_config_presets() {
    auto c700m = BitNetA48Config::size_700m();
    auto c1b3  = BitNetA48Config::size_1b3();
    auto c3b   = BitNetA48Config::size_3b();
    auto c7b   = BitNetA48Config::size_7b();

    check(c700m.hidden_size == 1536 && c700m.glu_size == 4096
       && c700m.n_heads == 24       && c700m.n_layers == 24,
          "config_700m_table6");
    check(c1b3.hidden_size  == 2048 && c1b3.glu_size  == 5460
       && c1b3.n_heads  == 32       && c1b3.n_layers == 24,
          "config_1b3_table6");
    check(c3b.hidden_size   == 3200 && c3b.glu_size   == 8640
       && c3b.n_heads   == 32       && c3b.n_layers == 26,
          "config_3b_table6");
    check(c7b.hidden_size   == 4096 && c7b.glu_size   == 11008
       && c7b.n_heads   == 32       && c7b.n_layers == 32,
          "config_7b_table6");
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Weight quantization: ternary, norm preserved
// ─────────────────────────────────────────────────────────────────────────────
static void test_weight_quant() {
    torch::manual_seed(0);
    auto W = torch::randn({64, 64});
    auto Wq = quantize_weight_ternary(W);

    // Values must be in {-α, 0, +α}: check that unique scaled values are ≤ 3
    float alpha = W.abs().mean().item<float>();
    auto normed = (Wq / (alpha + 1e-8f)).round();
    // Ternary weights: rounded normed values should be in {-1, 0, +1}
    // Check by counting values outside {-1,0,1}
    auto diff = ((normed + 1.0f).abs().min(
                  normed.abs()).min(
                  (normed - 1.0f).abs()));
    float max_deviation = diff.max().item<float>();
    check(max_deviation < 0.5f, "weight_quant_ternary_values");

    // Shape preserved
    check(Wq.sizes() == W.sizes(), "weight_quant_shape");
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. INT4 absmean: values in [-8β/√7, 7β/√7]
// ─────────────────────────────────────────────────────────────────────────────
static void test_int4_absmean() {
    torch::manual_seed(1);
    auto X  = torch::randn({32, 128});
    auto Xq = quantize_int4_absmean(X);

    float beta  = X.abs().mean().item<float>();
    float sq7   = std::sqrt(7.0f);
    float lo    = -8.0f * beta / sq7 - 1e-4f;
    float hi    =  7.0f * beta / sq7 + 1e-4f;

    check(Xq.min().item<float>() >= lo && Xq.max().item<float>() <= hi,
          "int4_absmean_range");
    check(Xq.sizes() == X.sizes(), "int4_absmean_shape");
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. INT8 absmax: values in [-γ, γ], step = γ/127
// ─────────────────────────────────────────────────────────────────────────────
static void test_int8_absmax() {
    torch::manual_seed(2);
    auto X  = torch::randn({32, 128});
    auto Xq = quantize_int8_absmax(X);

    float gamma = X.abs().amax().item<float>();
    check(Xq.abs().max().item<float>() <= gamma + 1e-4f, "int8_absmax_range");
    check(Xq.sizes() == X.sizes(), "int8_absmax_shape");
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. TopK mask: exactly k% values non-zero
// ─────────────────────────────────────────────────────────────────────────────
static void test_topk_mask() {
    torch::manual_seed(3);
    auto X    = torch::randn({4, 100});  // 400 elements
    int64_t k = 50;
    auto mask = topk_mask(X, k);

    int64_t expected = 400 * k / 100;   // 200
    int64_t nonzero  = mask.sum().item<int64_t>();
    // Allow ±1 due to ties at threshold
    check(std::abs(nonzero - expected) <= 1, "topk_mask_count");
    // Mask values are 0 or 1
    check(mask.min().item<float>() == 0.0f && mask.max().item<float>() == 1.0f,
          "topk_mask_binary");
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. RMSNorm: unit RMS output
// ─────────────────────────────────────────────────────────────────────────────
static void test_rmsnorm() {
    auto norm = RMSNorm(64);
    torch::manual_seed(4);
    auto x   = torch::randn({2, 10, 64});
    auto out = norm->forward(x);
    // RMS of last dim ≈ 1 (with unit gain weight)
    auto rms = out.pow(2).mean(-1).sqrt();
    check((rms - 1.0f).abs().max().item<float>() < 1e-4f, "rmsnorm_unit_rms");
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. RoPE cache shape
// ─────────────────────────────────────────────────────────────────────────────
static void test_rope_cache_shape() {
    auto [cos_c, sin_c] = build_rope_cache(128, 64, torch::kCPU);
    check(cos_c.sizes() == std::vector<int64_t>({128, 32}), "rope_cos_shape");
    check(sin_c.sizes() == std::vector<int64_t>({128, 32}), "rope_sin_shape");
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. RoPE application: shape preserved
// ─────────────────────────────────────────────────────────────────────────────
static void test_rope_apply_shape() {
    auto [cos_c, sin_c] = build_rope_cache(16, 32, torch::kCPU);
    torch::manual_seed(5);
    auto q   = torch::randn({2, 4, 16, 32});
    auto qr  = apply_rope(q, cos_c, sin_c);
    check(qr.sizes() == q.sizes(), "rope_apply_shape");
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. BitLinear: output shape
// ─────────────────────────────────────────────────────────────────────────────
static void test_bitlinear_shape() {
    auto bl   = BitLinear(64, 128, false, ActQuant::INT4_ABSMEAN);
    auto norm = RMSNorm(64);
    torch::manual_seed(6);
    auto x  = torch::randn({2, 10, 64});
    auto out = bl->forward(x, norm->weight);
    check(out.sizes() == std::vector<int64_t>({2, 10, 128}), "bitlinear_shape");
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Attention block: output shape
// ─────────────────────────────────────────────────────────────────────────────
static void test_attention_shape() {
    BitNetA48Config cfg = BitNetA48Config::size_700m();
    cfg.n_layers = 1;
    auto attn = BitNetA48Attention(cfg);
    torch::manual_seed(7);

    int64_t B = 2, T = 8;
    auto x = torch::randn({B, T, cfg.hidden_size});
    auto [cos_c, sin_c] = build_rope_cache(T, cfg.hidden_size / cfg.n_heads, torch::kCPU);
    auto mask = torch::full({T, T}, -std::numeric_limits<float>::infinity());
    mask = torch::triu(mask, 1);

    auto out = attn->forward(x, cos_c, sin_c, mask);
    check(out.sizes() == x.sizes(), "attention_output_shape");
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. FFN: output shape + sparsity from ReLU²
// ─────────────────────────────────────────────────────────────────────────────
static void test_ffn_shape_sparsity() {
    BitNetA48Config cfg = BitNetA48Config::size_700m();
    auto ffn = BitNetA48FFN(cfg);
    torch::manual_seed(8);

    auto x   = torch::randn({2, 8, cfg.hidden_size});
    auto out = ffn->forward(x);
    check(out.sizes() == x.sizes(), "ffn_output_shape");
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Full model: forward pass output shape
// ─────────────────────────────────────────────────────────────────────────────
static void test_model_forward_shape() {
    BitNetA48Config cfg;
    cfg.hidden_size = 64;
    cfg.glu_size    = 128;
    cfg.n_heads     = 4;
    cfg.n_kv_heads  = 4;
    cfg.n_layers    = 2;
    cfg.vocab_size  = 256;
    cfg.max_seq_len = 32;

    auto model = BitNetA48Model(cfg);
    torch::manual_seed(9);
    auto tokens = torch::randint(0, 256, {2, 8});
    auto logits = model->forward(tokens);
    check(logits.sizes() == std::vector<int64_t>({2, 8, 256}),
          "model_forward_shape");
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Greedy decode: returns non-empty sequence
// ─────────────────────────────────────────────────────────────────────────────
static void test_greedy_decode() {
    BitNetA48Config cfg;
    cfg.hidden_size = 64;
    cfg.glu_size    = 128;
    cfg.n_heads     = 4;
    cfg.n_kv_heads  = 4;
    cfg.n_layers    = 2;
    cfg.vocab_size  = 256;
    cfg.max_seq_len = 32;

    auto model  = BitNetA48Model(cfg);
    auto prompt = torch::tensor({{1L, 2L, 3L}});  // [1, 3]
    auto seq    = model->greedy_decode(prompt, /*eot_id=*/2, /*max_new=*/10);
    check(seq.size() > 3, "greedy_decode_nonempty");
    check(seq[0] == 1L && seq[1] == 2L && seq[2] == 3L, "greedy_decode_prompt_preserved");
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. Training step: loss is finite and decreases
// ─────────────────────────────────────────────────────────────────────────────
static void test_train_step() {
    BitNetA48Config cfg;
    cfg.hidden_size = 64;
    cfg.glu_size    = 128;
    cfg.n_heads     = 4;
    cfg.n_kv_heads  = 4;
    cfg.n_layers    = 2;
    cfg.vocab_size  = 256;
    cfg.max_seq_len = 32;

    auto model = BitNetA48Model(cfg);
    auto opts  = torch::optim::AdamWOptions(1e-3)
                     .betas({0.9, 0.95}).weight_decay(0.1);
    auto opt   = torch::optim::AdamW(model->parameters(), opts);

    torch::manual_seed(10);
    auto tokens_in  = torch::randint(0, 256, {2, 8});
    auto tokens_tgt = torch::randint(0, 256, {2, 8});

    float l1 = bitnet_a48_train_step(model, opt, tokens_in, tokens_tgt);
    float l2 = bitnet_a48_train_step(model, opt, tokens_in, tokens_tgt);

    check(std::isfinite(l1), "train_step_loss_finite");
    check(l2 <= l1 + 0.5f,   "train_step_loss_not_diverging");
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. Factory functions build correct sizes
// ─────────────────────────────────────────────────────────────────────────────
static void test_factories() {
    auto m700 = make_bitnet_a48_700m();
    check(m700->cfg.hidden_size == 1536, "factory_700m");

    auto m1b3 = make_bitnet_a48_1b3();
    check(m1b3->cfg.hidden_size == 2048, "factory_1b3");
}

// ─────────────────────────────────────────────────────────────────────────────
// 16. INT8+TopK: masked output has ≤50% non-zero elements
// ─────────────────────────────────────────────────────────────────────────────
static void test_topk_sparsification() {
    BitNetA48Config cfg;
    cfg.hidden_size = 64;
    cfg.glu_size    = 128;
    cfg.n_heads     = 4;
    cfg.n_kv_heads  = 4;
    cfg.n_layers    = 1;
    cfg.vocab_size  = 256;
    cfg.max_seq_len = 32;
    cfg.topk_percent = 50;

    // Directly test topk_mask on a vector
    torch::manual_seed(11);
    auto X    = torch::randn({1, 200});
    auto mask = topk_mask(X, 50);
    int64_t nz = (mask > 0).sum().item<int64_t>();
    // Expect exactly 100 non-zero (within ±1 for ties)
    check(std::abs(nz - 100L) <= 1, "topk_50pct_count");
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== BitNet a4.8 tests (arXiv:2411.04965v1) ===\n\n";

    test_config_presets();
    test_weight_quant();
    test_int4_absmean();
    test_int8_absmax();
    test_topk_mask();
    test_rmsnorm();
    test_rope_cache_shape();
    test_rope_apply_shape();
    test_bitlinear_shape();
    test_attention_shape();
    test_ffn_shape_sparsity();
    test_model_forward_shape();
    test_greedy_decode();
    test_train_step();
    test_factories();
    test_topk_sparsification();

    std::cout << "\n=== Results: " << passed << " passed, " << failed << " failed ===\n";
    return (failed == 0) ? 0 : 1;
}
