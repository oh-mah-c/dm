// ─────────────────────────────────────────────────────────────────────────────
// test_transformer_xl.cpp — C++ structural tests for Transformer-XL
//
// Paper: Z. Dai et al., "Transformer-XL: Attentive Language Models Beyond a
//        Fixed-Length Context," ACL 2019. https://aclanthology.org/P19-1285
//
// Tests:
//  1.  Forward output shape [B, T, vocab_size]                  (§3)
//  2.  Memory is populated after first forward pass             (§3.2)
//  3.  Memory length does not exceed mem_len after many steps   (§3.2)
//  4.  Relative positional encoding shape [klen, d_model]       (§3.3)
//  5.  Relative encoding values are bounded (sin/cos ∈ [-1, 1]) (§3.3)
//  6.  Attention output shape through one layer                  (§3.3)
//  7.  Causal: output at position t unchanged by future tokens  (§3.2)
//  8.  Memory reset clears all caches                           (§3.2)
//  9.  Tiny param count is in expected range                     (config)
// 10.  Gradient flow: no NaN/Inf in grads                       (§3)
// 11.  Adam step changes parameters                             (§4)
// 12.  LR warmup schedule is linear                             (§4)
// 13.  Checkpoint save/load round-trip                           (§4)
// 14.  Extended memory at eval (mem_len > seg_len)              (§3.2, §4.5)
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/transformer_xl/transformer_xl.h"

#include <torch/torch.h>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <filesystem>

using namespace dm::models::nlp;

static int s_passed = 0, s_failed = 0;

#define ASSERT_TRUE(cond, msg)                          \
    do {                                                \
        if (!(cond)) {                                  \
            std::fprintf(stderr, "  FAIL: %s\n", (msg)); \
            ++s_failed; return;                         \
        }                                               \
    } while(0)

static void begin_test(const char* n) {
    std::printf("[test] %s ...", n); std::fflush(stdout);
}
static void end_test() { ++s_passed; std::printf(" PASS\n"); }

// Tiny config for speed
static TransformerXLConfig tiny_cfg() { return TransformerXLConfig::tiny(); }

static const int64_t B  = 2;
static const int64_t T  = 16;   // seg_len

// ─────────────────────────────────────────────────────────────────────────────
// 1. Forward output shape [B, T, vocab_size]
// ─────────────────────────────────────────────────────────────────────────────
static void test_forward_shape() {
    begin_test("forward shape [B, T, vocab_size] (§3)");
    auto cfg   = tiny_cfg();
    auto model = TransformerXLModel(cfg);
    model->eval();
    torch::NoGradGuard ng;
    auto tokens = torch::randint(0, cfg.vocab_size, {B, T});
    auto logits = model->forward(tokens);
    ASSERT_TRUE(logits.size(0) == B,            "batch dim wrong");
    ASSERT_TRUE(logits.size(1) == T,            "seq dim wrong");
    ASSERT_TRUE(logits.size(2) == cfg.vocab_size,"vocab dim wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Memory is populated after first forward pass
// ─────────────────────────────────────────────────────────────────────────────
static void test_memory_populated() {
    begin_test("memory populated after forward (§3.2)");
    auto cfg   = tiny_cfg();
    auto model = TransformerXLModel(cfg);
    model->eval();
    torch::NoGradGuard ng;

    // Before any forward: memory should be empty
    ASSERT_TRUE(model->memory(0).numel() == 0, "memory should start empty");

    auto tokens = torch::randint(0, cfg.vocab_size, {B, T});
    model->forward(tokens);

    // After one forward pass, memory for layer 0 should be populated
    ASSERT_TRUE(model->memory(0).numel() > 0, "memory not populated after forward");
    ASSERT_TRUE(model->memory(0).size(-1) == cfg.d_model, "memory dim wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Memory length bounded by mem_len
// ─────────────────────────────────────────────────────────────────────────────
static void test_memory_bounded() {
    begin_test("memory length bounded by mem_len after many steps (§3.2)");
    auto cfg   = tiny_cfg();  // mem_len=32, seg_len=32
    auto model = TransformerXLModel(cfg);
    model->eval();
    torch::NoGradGuard ng;

    // Run 5 segments
    for (int i = 0; i < 5; ++i) {
        auto tokens = torch::randint(0, cfg.vocab_size, {B, T});
        model->forward(tokens);
    }

    // Memory for every layer should be at most mem_len
    for (int64_t n = 0; n < cfg.n_layers; ++n) {
        ASSERT_TRUE(model->memory(n).size(1) <= cfg.mem_len,
                    "memory exceeds mem_len");
    }
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Relative positional encoding shape
// ─────────────────────────────────────────────────────────────────────────────
static void test_rel_pos_shape() {
    begin_test("relative PE shape [klen, d_model] (§3.3)");
    auto cfg = tiny_cfg();
    int64_t klen = T + cfg.mem_len;
    auto r = make_rel_pos_encoding(klen, cfg.d_model, torch::kCPU);
    ASSERT_TRUE(r.size(0) == klen,       "klen dim wrong");
    ASSERT_TRUE(r.size(1) == cfg.d_model,"d_model dim wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Relative encoding values ∈ [-1, 1]  (sin/cos)
// ─────────────────────────────────────────────────────────────────────────────
static void test_rel_pos_bounded() {
    begin_test("relative PE values bounded in [-1, 1] (§3.3)");
    auto cfg = tiny_cfg();
    auto r = make_rel_pos_encoding(64, cfg.d_model, torch::kCPU);
    float max_val = r.abs().max().item<float>();
    ASSERT_TRUE(max_val <= 1.001f, "relative PE values out of [-1, 1]");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Attention output shape through one layer
// ─────────────────────────────────────────────────────────────────────────────
static void test_attn_shape() {
    begin_test("attention output shape [B, T, d_model] (§3.3)");
    auto cfg  = tiny_cfg();
    auto attn = TXLRelAttn(cfg);
    attn->eval();
    torch::NoGradGuard ng;

    auto h_cur = torch::randn({B, T, cfg.d_model});
    auto r     = make_rel_pos_encoding(T, cfg.d_model, torch::kCPU);
    auto mask  = torch::zeros({T, T}, torch::kBool)
                     .triu(1);
    torch::Tensor h_mem;  // empty
    auto out = attn->forward(h_cur, h_mem, r, mask);
    ASSERT_TRUE(out.size(0) == B,          "batch dim wrong");
    ASSERT_TRUE(out.size(1) == T,          "seq dim wrong");
    ASSERT_TRUE(out.size(2) == cfg.d_model,"d_model dim wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Causal: output at position t-1 is unaffected by token at position t
// ─────────────────────────────────────────────────────────────────────────────
static void test_causal_mask() {
    begin_test("causal mask: position t-1 unaffected by position t (§3.2)");
    auto cfg   = tiny_cfg();
    auto model = TransformerXLModel(cfg);
    model->eval();
    torch::NoGradGuard ng;

    // Two token sequences differing only at position T-1
    auto tok1 = torch::randint(0, cfg.vocab_size, {1, T});
    auto tok2 = tok1.clone();
    tok2[0][T-1] = (tok2[0][T-1].item<int64_t>() + 1) % cfg.vocab_size;

    auto logits1 = model->forward(tok1);
    model->reset_memory();
    auto logits2 = model->forward(tok2);

    // Logits at positions 0..T-2 should be identical
    auto diff = (logits1.slice(1, 0, T-1) - logits2.slice(1, 0, T-1)).abs().max();
    ASSERT_TRUE(diff.item<float>() < 1e-4f, "causal mask violated");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. reset_memory clears all caches
// ─────────────────────────────────────────────────────────────────────────────
static void test_reset_memory() {
    begin_test("reset_memory clears all caches (§3.2)");
    auto cfg   = tiny_cfg();
    auto model = TransformerXLModel(cfg);
    model->eval();
    torch::NoGradGuard ng;

    auto tokens = torch::randint(0, cfg.vocab_size, {B, T});
    model->forward(tokens);
    ASSERT_TRUE(model->memory(0).numel() > 0, "memory should be populated");

    model->reset_memory();
    for (int64_t n = 0; n < cfg.n_layers; ++n) {
        ASSERT_TRUE(model->memory(n).numel() == 0,
                    "memory not cleared after reset");
    }
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Tiny param count in expected range
// ─────────────────────────────────────────────────────────────────────────────
static void test_param_count() {
    begin_test("tiny param count in expected range (config)");
    auto cfg   = tiny_cfg();
    auto model = TransformerXLModel(cfg);

    int64_t total = 0;
    for (auto& item : model->named_parameters()) {
        total += item.value().numel();
    }
    // Tiny: vocab=1000, d=64, d_head=16, H=4, d_inner=128, n_layers=2
    // Expect roughly tens of thousands — sanity check 10K < total < 2M
    ASSERT_TRUE(total > 10000,   "param count too small");
    ASSERT_TRUE(total < 2000000, "param count too large");
    std::printf(" (%lld params)", static_cast<long long>(total));
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Gradient flow: no NaN/Inf
// ─────────────────────────────────────────────────────────────────────────────
static void test_gradient_flow() {
    begin_test("gradient flow: no NaN/Inf (§3)");
    auto cfg   = tiny_cfg();
    auto model = TransformerXLModel(cfg);
    model->train();

    auto tokens = torch::randint(0, cfg.vocab_size, {B, T + 1});
    auto inp    = tokens.slice(1, 0, T);
    auto tgt    = tokens.slice(1, 1, T+1);
    auto logits = model->forward(inp);
    auto loss   = torch::nn::functional::cross_entropy(
        logits.reshape({-1, cfg.vocab_size}),
        tgt.reshape({-1}));
    loss.backward();

    for (auto& item : model->named_parameters()) {
        if (!item.value().grad().defined()) continue;
        auto& g = item.value().grad();
        ASSERT_TRUE(!g.isnan().any().item<bool>(),
                    ("NaN in grad: " + item.key()).c_str());
        ASSERT_TRUE(!g.isinf().any().item<bool>(),
                    ("Inf in grad: " + item.key()).c_str());
    }
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Adam step changes parameters
// ─────────────────────────────────────────────────────────────────────────────
static void test_optimizer_step() {
    begin_test("Adam step changes parameters (§4)");
    auto cfg  = tiny_cfg();
    auto model = TransformerXLModel(cfg);
    TransformerXLTrainConfig tcfg;

    // Snapshot embedding weight before
    auto w_before = model->embedding->weight.clone().detach();

    auto opt = make_txl_optimizer(model, tcfg);
    auto tokens = torch::randint(0, cfg.vocab_size, {B, T + 1});
    txl_train_step(model, opt, tokens, tcfg, 0, true);

    auto w_after = model->embedding->weight.detach();
    auto changed = (w_before - w_after).abs().max().item<float>();
    ASSERT_TRUE(changed > 1e-9f, "weights unchanged after optimizer step");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. LR warmup schedule is linear
// ─────────────────────────────────────────────────────────────────────────────
static void test_lr_schedule() {
    begin_test("LR warmup schedule is linear (§4)");
    TransformerXLTrainConfig tcfg;
    tcfg.lr      = 2.5e-4;
    tcfg.warmup  = 100;

    double lr0  = txl_lr_schedule(0, tcfg);
    double lr50 = txl_lr_schedule(49, tcfg);
    double lr99 = txl_lr_schedule(99, tcfg);
    double lr100= txl_lr_schedule(100, tcfg);

    // Step 0: lr0 = lr * 1/100
    ASSERT_TRUE(std::fabs(lr0 - tcfg.lr / 100.0) < 1e-9, "step-0 LR wrong");
    // Step 49: lr * 50/100
    ASSERT_TRUE(std::fabs(lr50 - tcfg.lr * 50.0 / 100.0) < 1e-9, "step-49 LR wrong");
    // Step 99: lr * 100/100 = lr
    ASSERT_TRUE(std::fabs(lr99 - tcfg.lr) < 1e-9, "step-99 LR wrong");
    // Step 100 (post-warmup): lr stays at peak
    ASSERT_TRUE(std::fabs(lr100 - tcfg.lr) < 1e-9, "post-warmup LR wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Checkpoint save/load round-trip
// ─────────────────────────────────────────────────────────────────────────────
static void test_checkpoint() {
    begin_test("checkpoint save/load round-trip (§4)");
    auto cfg   = tiny_cfg();
    auto model = TransformerXLModel(cfg);
    model->eval();

    const std::string path = "/tmp/txl_test_ckpt.pt";
    torch::save(model, path);

    auto model2 = TransformerXLModel(cfg);
    torch::load(model2, path);
    model2->eval();
    std::filesystem::remove(path);

    torch::NoGradGuard ng;
    auto tokens  = torch::randint(0, cfg.vocab_size, {B, T});
    auto logits1 = model->forward(tokens);
    model2->reset_memory();
    auto logits2 = model2->forward(tokens);

    auto diff = (logits1 - logits2).abs().max().item<float>();
    ASSERT_TRUE(diff < 1e-4f, "logits differ after checkpoint round-trip");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. Extended memory at eval: memory > seg_len accepted
// ─────────────────────────────────────────────────────────────────────────────
static void test_extended_memory() {
    begin_test("extended memory at eval (§3.2, §4.5)");
    // Create a model with larger mem_len than seg_len (evaluation setting)
    auto cfg       = tiny_cfg();
    cfg.mem_len    = cfg.seg_len * 3;   // 3× memory
    auto model = TransformerXLModel(cfg);
    model->eval();
    torch::NoGradGuard ng;

    // Run 4 segments — memory should accumulate
    torch::Tensor last_logits;
    for (int i = 0; i < 4; ++i) {
        auto tokens  = torch::randint(0, cfg.vocab_size, {B, T});
        last_logits  = model->forward(tokens);
    }

    // Memory should be min(3*seg_len, 3*seg_len) = 3*seg_len after 4 segs
    // (each seg stores up to seg_len tokens; after 3+ segs memory is full)
    int64_t mem_t = model->memory(0).size(1);
    ASSERT_TRUE(mem_t <= cfg.mem_len, "memory exceeds extended mem_len");
    ASSERT_TRUE(mem_t > 0,            "memory empty after 4 segments");

    // Final logits should be valid
    ASSERT_TRUE(!last_logits.isnan().any().item<bool>(), "NaN in logits");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    torch::manual_seed(42);

    test_forward_shape();
    test_memory_populated();
    test_memory_bounded();
    test_rel_pos_shape();
    test_rel_pos_bounded();
    test_attn_shape();
    test_causal_mask();
    test_reset_memory();
    test_param_count();
    test_gradient_flow();
    test_optimizer_step();
    test_lr_schedule();
    test_checkpoint();
    test_extended_memory();

    std::printf("\n%d passed, %d failed\n", s_passed, s_failed);
    return s_failed ? 1 : 0;
}
