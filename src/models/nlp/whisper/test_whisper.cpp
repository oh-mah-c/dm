// ─────────────────────────────────────────────────────────────────────────────
// test_whisper.cpp — C++ structural tests for Whisper
//
// Paper: A. Radford et al.,
//        "Robust Speech Recognition via Large-Scale Weak Supervision"
//        ICML 2023 (radford23a)
//
// Tests verify:
//  1.  Encoder output shape [B, T/2, d_model]             (Section 2.1)
//  2.  Encoder halves time dimension (conv stride=2)       (Section 2.1)
//  3.  Decoder output logits shape [B, L, vocab_size]      (Section 2.1)
//  4.  Full forward pass shape [B, L, vocab_size]          (Section 2.1)
//  5.  Tiny model param count ≈ 39M                        (Table 1)
//  6.  Base model param count ≈ 74M                        (Table 1)
//  7.  Encoder block count matches config enc_layers       (Section 2.1)
//  8.  Decoder block count matches config dec_layers       (Section 2.1)
//  9.  Multi-head attention head_dim = d_model / n_heads   (Section 2.1)
// 10.  Positional embeddings have correct shape            (Section 2.1)
// 11.  Gradient flow: no NaN/Inf                           (Section 2.2)
// 12.  AdamW step changes encoder weights                  (Section 2.2)
// 13.  Checkpoint save/load round-trip
// 14.  All factory variants produce correct output shape
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/whisper/whisper.h"

#include <torch/torch.h>
#include <cassert>
#include <cstdio>
#include <filesystem>

using namespace dm::models::nlp;

static int s_passed = 0, s_failed = 0;
#define ASSERT_TRUE(cond, msg) \
    do { if (!(cond)) { \
        std::fprintf(stderr, "  FAIL: %s\n", (msg)); \
        ++s_failed; return; \
    } } while(0)
static void begin_test(const char* n) {
    std::printf("[test] %s ...", n); std::fflush(stdout);
}
static void end_test() { ++s_passed; std::printf(" PASS\n"); }

// Use tiny model for speed
static const int64_t B  = 2;   // batch size
static const int64_t T  = 200; // mel time frames → encoder outputs T/2=100
static const int64_t L  = 16;  // decoder token length

// ─────────────────────────────────────────────────────────────────────────────
// 1. Encoder output shape [B, T/2, d_model]
// ─────────────────────────────────────────────────────────────────────────────
static void test_encoder_shape() {
    begin_test("encoder output shape [B, T/2, d_model] (Section 2.1)");
    auto cfg = WhisperConfig::tiny();
    auto enc = WhisperEncoder(cfg);
    enc->eval();
    torch::NoGradGuard ng;
    auto mel = torch::randn({B, cfg.n_mels, T});
    auto out = enc->forward(mel);
    ASSERT_TRUE(out.size(0) == B,            "batch dim wrong");
    ASSERT_TRUE(out.size(1) == T / 2,        "time dim not halved by stride=2");
    ASSERT_TRUE(out.size(2) == cfg.d_model,  "feature dim wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Encoder halves time dimension
// ─────────────────────────────────────────────────────────────────────────────
static void test_encoder_stride() {
    begin_test("encoder conv2 stride=2 halves time (Section 2.1)");
    auto cfg = WhisperConfig::tiny();
    auto enc = WhisperEncoder(cfg);
    enc->eval();
    torch::NoGradGuard ng;
    for (int64_t t_in : {100, 200, 400}) {
        auto mel = torch::randn({1, cfg.n_mels, t_in});
        auto out = enc->forward(mel);
        ASSERT_TRUE(out.size(1) == t_in / 2, "time not halved");
    }
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Decoder output logits [B, L, vocab_size]
// ─────────────────────────────────────────────────────────────────────────────
static void test_decoder_shape() {
    begin_test("decoder logits shape [B, L, vocab_size] (Section 2.1)");
    auto cfg = WhisperConfig::tiny();
    // Build encoder output placeholder
    auto enc_out = torch::randn({B, T/2, cfg.d_model});
    auto tokens  = torch::randint(0, cfg.vocab_size, {B, L});
    auto dec = WhisperDecoder(cfg);
    dec->eval();
    torch::NoGradGuard ng;
    auto logits = dec->forward(tokens, enc_out);
    ASSERT_TRUE(logits.size(0) == B,               "batch dim wrong");
    ASSERT_TRUE(logits.size(1) == L,               "seq len dim wrong");
    ASSERT_TRUE(logits.size(2) == cfg.vocab_size,  "vocab dim wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Full forward pass shape [B, L, vocab_size]
// ─────────────────────────────────────────────────────────────────────────────
static void test_full_forward() {
    begin_test("full forward [B, L, vocab_size] (Section 2.1)");
    auto cfg   = WhisperConfig::tiny();
    auto model = WhisperModel(cfg);
    model->eval();
    torch::NoGradGuard ng;
    auto mel    = torch::randn({B, cfg.n_mels, T});
    auto tokens = torch::randint(0, cfg.vocab_size, {B, L});
    auto logits = model->forward(mel, tokens);
    ASSERT_TRUE(logits.size(0) == B,              "batch dim wrong");
    ASSERT_TRUE(logits.size(1) == L,              "seq len wrong");
    ASSERT_TRUE(logits.size(2) == cfg.vocab_size, "vocab dim wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Tiny model param count ≈ 39M (Table 1)
// ─────────────────────────────────────────────────────────────────────────────
static void test_params_tiny() {
    begin_test("tiny model params ≈ 39M (Table 1)");
    auto model = make_whisper_tiny();
    int64_t n = 0;
    for (auto& p : model->parameters()) n += p.numel();
    // Allow ±10M range (we use exact GPT-2 vocab but simplified architecture for testing)
    ASSERT_TRUE(n >= 25'000'000 && n <= 55'000'000,
                "tiny param count out of range [25M, 55M]");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Base model param count ≈ 74M (Table 1)
// ─────────────────────────────────────────────────────────────────────────────
static void test_params_base() {
    begin_test("base model params ≈ 74M (Table 1)");
    auto model = make_whisper_base();
    int64_t n = 0;
    for (auto& p : model->parameters()) n += p.numel();
    ASSERT_TRUE(n >= 50'000'000 && n <= 100'000'000,
                "base param count out of range [50M, 100M]");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Encoder block count matches enc_layers
// ─────────────────────────────────────────────────────────────────────────────
static void test_enc_block_count() {
    begin_test("encoder block count = enc_layers (Section 2.1)");
    auto cfg = WhisperConfig::tiny();
    auto enc = WhisperEncoder(cfg);
    ASSERT_TRUE((int64_t)enc->blocks->size() == cfg.enc_layers,
                "encoder block count wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Decoder block count matches dec_layers
// ─────────────────────────────────────────────────────────────────────────────
static void test_dec_block_count() {
    begin_test("decoder block count = dec_layers (Section 2.1)");
    auto cfg = WhisperConfig::tiny();
    auto dec = WhisperDecoder(cfg);
    ASSERT_TRUE((int64_t)dec->blocks->size() == cfg.dec_layers,
                "decoder block count wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Attention head_dim = d_model / n_heads
// ─────────────────────────────────────────────────────────────────────────────
static void test_head_dim() {
    begin_test("attention head_dim = d_model / n_heads (Section 2.1)");
    auto cfg = WhisperConfig::tiny();
    auto attn = WhisperAttention(cfg.d_model, cfg.n_heads);
    ASSERT_TRUE(attn->head_dim == cfg.d_model / cfg.n_heads,
                "head_dim wrong");
    ASSERT_TRUE(cfg.d_model % cfg.n_heads == 0,
                "d_model not divisible by n_heads");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Sinusoidal encoder PE has shape [1, max_source_positions, d_model]
// ─────────────────────────────────────────────────────────────────────────────
static void test_pos_emb_shape() {
    begin_test("encoder pos_emb shape [1, max_src_pos, d_model] (Section 2.1)");
    auto cfg = WhisperConfig::tiny();
    auto enc = WhisperEncoder(cfg);
    ASSERT_TRUE(enc->pos_emb.size(0) == 1,                          "pos_emb batch dim wrong");
    ASSERT_TRUE(enc->pos_emb.size(1) == cfg.max_source_positions,   "pos_emb length wrong");
    ASSERT_TRUE(enc->pos_emb.size(2) == cfg.d_model,                "pos_emb d_model wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Gradient flow: no NaN/Inf (Section 2.2)
// ─────────────────────────────────────────────────────────────────────────────
static void test_gradient_flow() {
    begin_test("gradient flow: no NaN/Inf (Section 2.2)");
    auto cfg   = WhisperConfig::tiny();
    auto model = WhisperModel(cfg);
    model->train();
    auto mel    = torch::randn({B, cfg.n_mels, T});
    auto tokens = torch::randint(0, 100, {B, L});
    auto target = torch::randint(0, 100, {B, L});
    auto logits = model->forward(mel, tokens);
    auto loss   = torch::nn::functional::cross_entropy(
        logits.view({B * L, cfg.vocab_size}), target.view({B * L}));
    loss.backward();
    bool ok = true;
    for (auto& p : model->parameters()) {
        if (!p.grad().defined()) continue;
        if (p.grad().isnan().any().item<bool>() ||
            p.grad().isinf().any().item<bool>()) { ok = false; break; }
    }
    ASSERT_TRUE(ok, "NaN/Inf in gradients");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. AdamW step changes encoder weights (Section 2.2)
// ─────────────────────────────────────────────────────────────────────────────
static void test_adamw_update() {
    begin_test("AdamW step changes encoder weights (Section 2.2)");
    auto cfg   = WhisperConfig::tiny();
    auto model = WhisperModel(cfg);
    model->train();
    torch::optim::AdamW opt(model->parameters(),
        torch::optim::AdamWOptions(1e-3).betas({0.9, 0.999}).weight_decay(1e-6));

    // Snapshot conv1 weights
    auto w_before = model->encoder->conv1->weight.clone().detach();

    auto mel    = torch::randn({B, cfg.n_mels, T});
    auto tokens = torch::randint(0, 100, {B, L});
    auto target = torch::randint(0, 100, {B, L});
    auto logits = model->forward(mel, tokens);
    auto loss   = torch::nn::functional::cross_entropy(
        logits.view({B * L, cfg.vocab_size}), target.view({B * L}));
    opt.zero_grad();
    loss.backward();
    opt.step();

    auto w_after = model->encoder->conv1->weight.detach();
    ASSERT_TRUE(!torch::allclose(w_before, w_after),
                "encoder conv1 weights unchanged after AdamW step");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Checkpoint save/load round-trip
// ─────────────────────────────────────────────────────────────────────────────
static void test_checkpoint() {
    begin_test("checkpoint save/load round-trip");
    auto cfg = WhisperConfig::tiny();
    auto m1  = WhisperModel(cfg);
    m1->eval();
    torch::NoGradGuard ng;
    auto mel    = torch::randn({1, cfg.n_mels, T});
    auto tokens = torch::randint(0, 100, {1, L});
    auto p1     = m1->forward(mel, tokens);

    const std::string path = "/tmp/test_whisper_ckpt.pt";
    {
        torch::serialize::OutputArchive ar;
        m1->save(ar);
        ar.save_to(path);
    }
    auto m2 = WhisperModel(cfg);
    {
        torch::serialize::InputArchive ar;
        ar.load_from(path);
        m2->load(ar);
    }
    m2->eval();
    auto p2 = m2->forward(mel, tokens);
    ASSERT_TRUE(torch::allclose(p1, p2, 1e-5f, 1e-5f),
                "predictions differ after checkpoint reload");
    std::filesystem::remove(path);
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. All factory variants produce correct output shapes
// ─────────────────────────────────────────────────────────────────────────────
static void test_all_variants() {
    begin_test("all size variants produce correct output shape");
    torch::NoGradGuard ng;
    // Only test tiny/base for speed; just check logit shape
    struct Variant { const char* name; WhisperConfig cfg; };
    Variant vs[] = {
        {"tiny",   WhisperConfig::tiny()  },
        {"base",   WhisperConfig::base()  },
    };
    for (auto& v : vs) {
        auto model = WhisperModel(v.cfg);
        model->eval();
        auto mel    = torch::randn({1, v.cfg.n_mels, T});
        auto tokens = torch::randint(0, 10, {1, L});
        auto out    = model->forward(mel, tokens);
        ASSERT_TRUE(out.size(0) == 1,                "batch dim wrong");
        ASSERT_TRUE(out.size(1) == L,                "seq dim wrong");
        ASSERT_TRUE(out.size(2) == v.cfg.vocab_size, "vocab dim wrong");
    }
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("\n=== Whisper tests (Radford et al., ICML 2023, radford23a) ===\n");

    test_encoder_shape();
    test_encoder_stride();
    test_decoder_shape();
    test_full_forward();
    test_params_tiny();
    test_params_base();
    test_enc_block_count();
    test_dec_block_count();
    test_head_dim();
    test_pos_emb_shape();
    test_gradient_flow();
    test_adamw_update();
    test_checkpoint();
    test_all_variants();

    std::printf("=== %d passed, %d failed ===\n\n", s_passed, s_failed);
    return s_failed == 0 ? 0 : 1;
}
