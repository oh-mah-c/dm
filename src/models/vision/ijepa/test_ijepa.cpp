// ─────────────────────────────────────────────────────────────────────────────
// I-JEPA unit tests
// ─────────────────────────────────────────────────────────────────────────────

#include "models/vision/ijepa/ijepa.h"

#include <torch/torch.h>
#include <iostream>
#include <cassert>
#include <cmath>
#include <string>

using namespace dm::models::vision;

static int passed = 0, failed = 0;

#undef CHECK   // torch headers define CHECK; use our own name
#define IJEPA_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << (msg) << "\n"; failed++; \
        } else { \
            std::cout << "  [PASS] " << (msg) << "\n"; passed++; \
        } \
    } while (0)

// ── Config presets ────────────────────────────────────────────────────────────

static void test_config_presets() {
    auto b = IJEPAConfig::vit_b16();
    IJEPA_CHECK(b.embed_dim == 768  && b.encoder_depth == 12 && b.pred_depth == 6,
          "vit_b16 config");

    auto l = IJEPAConfig::vit_l16();
    IJEPA_CHECK(l.embed_dim == 1024 && l.encoder_depth == 24 && l.pred_depth == 12,
          "vit_l16 config");

    auto h = IJEPAConfig::vit_h14();
    IJEPA_CHECK(h.embed_dim == 1280 && h.encoder_depth == 32 && h.patch_size == 14,
          "vit_h14 config");
}

// ── Attention forward shape ───────────────────────────────────────────────────

static void test_attention_shape() {
    Attention attn(64, 8);
    auto x = torch::randn({2, 16, 64});
    auto y = attn(x);
    IJEPA_CHECK(y.sizes() == torch::IntArrayRef({2, 16, 64}), "attention output shape");
}

// ── ViTBlock forward shape ────────────────────────────────────────────────────

static void test_vitblock_shape() {
    ViTBlock blk(128, 8, 512);
    auto x = torch::randn({3, 10, 128});
    auto y = blk(x);
    IJEPA_CHECK(y.sizes() == torch::IntArrayRef({3, 10, 128}), "ViTBlock output shape");
}

// ── ViTEncoder — full image ───────────────────────────────────────────────────

static void test_vit_encoder_full() {
    // Small encoder: img=32, patch=8 → 16 patches; depth=2
    ViTEncoder enc(32, 8, 64, 2, 4);
    auto patches = torch::randn({2, 16, 8*8*3});  // [B, N, P^2*C]
    auto out = enc(patches);
    IJEPA_CHECK(out.sizes() == torch::IntArrayRef({2, 16, 64}),
          "ViTEncoder full-image output shape");
}

// ── ViTEncoder — partial patches with indices ─────────────────────────────────

static void test_vit_encoder_partial() {
    ViTEncoder enc(32, 8, 64, 2, 4);
    // 6 context patches out of 16
    auto patches = torch::randn({2, 6, 8*8*3});
    auto idx     = torch::randint(0, 16, {2, 6});
    auto out     = enc(patches, idx);
    IJEPA_CHECK(out.sizes() == torch::IntArrayRef({2, 6, 64}),
          "ViTEncoder partial output shape");
}

// ── MaskSampler ──────────────────────────────────────────────────────────────

static void test_mask_sampler_shapes() {
    MaskSampler ms(14, 14);  // 14×14 patch grid
    auto [ctx, tgts] = ms.sample(4, 0.15, 0.20, 0.75, 1.5, 0.85, 1.0);

    IJEPA_CHECK(!ctx.empty(),   "mask sampler: context non-empty");
    IJEPA_CHECK(tgts.size() == 4, "mask sampler: 4 target blocks");
    for (auto &t : tgts)
        IJEPA_CHECK(!t.empty(), "mask sampler: target block non-empty");
}

static void test_mask_sampler_range() {
    MaskSampler ms(14, 14);
    int64_t total = 14 * 14;
    for (int trial = 0; trial < 10; trial++) {
        auto [ctx, tgts] = ms.sample(4, 0.15, 0.20, 0.75, 1.5, 0.85, 1.0);
        for (auto idx : ctx)
            IJEPA_CHECK(idx >= 0 && idx < total, "context index in range");
        for (auto &t : tgts)
            for (auto idx : t)
                IJEPA_CHECK(idx >= 0 && idx < total, "target index in range");
    }
}

static void test_mask_sampler_no_overlap() {
    // Context patches must not overlap with any target patch.
    MaskSampler ms(14, 14);
    for (int trial = 0; trial < 20; trial++) {
        auto [ctx, tgts] = ms.sample(4, 0.15, 0.20, 0.75, 1.5, 0.85, 1.0);
        std::unordered_set<int64_t> tgt_set;
        for (auto &t : tgts) for (auto i : t) tgt_set.insert(i);
        bool ok = true;
        for (auto i : ctx) if (tgt_set.count(i)) { ok = false; break; }
        IJEPA_CHECK(ok, "mask sampler: no ctx-target overlap");
    }
}

// ── IJEPAPredictor output shape ───────────────────────────────────────────────

static void test_predictor_shape() {
    // encoder_dim=64, pred_dim=32, num_patches=16, depth=2, heads=4
    IJEPAPredictor pred(64, 32, 16, 2, 4);
    auto s_x     = torch::randn({2, 6, 64});  // context encoder output
    auto ctx_idx = torch::randint(0, 16, {2, 6});
    auto tgt_idx = torch::randint(0, 16, {2, 4});
    auto out     = pred(s_x, ctx_idx, tgt_idx);
    IJEPA_CHECK(out.sizes() == torch::IntArrayRef({2, 4, 64}),
          "predictor output shape [B, L_tgt, encoder_dim]");
}

// ── Patchify ─────────────────────────────────────────────────────────────────

static void test_patchify() {
    auto cfg = IJEPAConfig::vit_b16(32);  // tiny 32×32 image
    IJEPA model(cfg);
    auto img     = torch::randn({2, 3, 32, 32});
    auto patches = model->patchify(img);
    int64_t n    = (32/16) * (32/16);   // 4 patches
    int64_t dim  = 3 * 16 * 16;
    IJEPA_CHECK(patches.sizes() == torch::IntArrayRef({2, n, dim}), "patchify shape");
}

// ── Full forward pass: loss is scalar and finite ──────────────────────────────

static void test_forward_loss_scalar() {
    // Use tiny config for speed
    IJEPAConfig cfg;
    cfg.img_size      = 32;
    cfg.patch_size    = 8;   // 4×4 = 16 patches
    cfg.embed_dim     = 32;
    cfg.encoder_depth = 1;
    cfg.encoder_heads = 4;
    cfg.pred_dim      = 16;
    cfg.pred_depth    = 1;
    cfg.pred_heads    = 4;
    cfg.num_target_blocks = 2;

    IJEPA model(cfg);
    auto img  = torch::randn({2, 3, 32, 32});
    auto loss = model(img);
    IJEPA_CHECK(loss.dim() == 0,                  "loss is scalar");
    IJEPA_CHECK(std::isfinite(loss.item<float>()), "loss is finite");
    IJEPA_CHECK(loss.item<float>() >= 0.0f,        "loss is non-negative");
}

// ── Loss is differentiable (backward works) ───────────────────────────────────

static void test_loss_backward() {
    IJEPAConfig cfg;
    cfg.img_size      = 32;
    cfg.patch_size    = 8;
    cfg.embed_dim     = 32;
    cfg.encoder_depth = 1;
    cfg.encoder_heads = 4;
    cfg.pred_dim      = 16;
    cfg.pred_depth    = 1;
    cfg.pred_heads    = 4;
    cfg.num_target_blocks = 2;

    IJEPA model(cfg);
    auto img  = torch::randn({2, 3, 32, 32});
    auto loss = model(img);
    loss.backward();

    bool any_grad = false;
    for (auto &p : model->context_encoder->parameters())
        if (p.grad().defined() && p.grad().abs().sum().item<float>() > 0) {
            any_grad = true; break;
        }
    IJEPA_CHECK(any_grad, "gradients flow to context encoder");

    // Target encoder should have no gradient
    bool tgt_grad = false;
    for (auto &p : model->target_encoder->parameters())
        if (p.grad().defined() && p.grad().abs().sum().item<float>() > 0) {
            tgt_grad = true; break;
        }
    IJEPA_CHECK(!tgt_grad, "target encoder has no gradient (EMA-only)");
}

// ── EMA update changes target encoder ────────────────────────────────────────

static void test_ema_update() {
    IJEPAConfig cfg;
    cfg.img_size      = 32;
    cfg.patch_size    = 8;
    cfg.embed_dim     = 32;
    cfg.encoder_depth = 1;
    cfg.encoder_heads = 4;
    cfg.pred_dim      = 16;
    cfg.pred_depth    = 1;
    cfg.pred_heads    = 4;

    IJEPA model(cfg);

    // Perturb context encoder
    {
        torch::NoGradGuard ng;
        for (auto &p : model->context_encoder->parameters())
            p.add_(torch::randn_like(p) * 0.1);
    }

    // Snapshot target encoder before EMA
    auto tgt_before = model->target_encoder->parameters()[0].clone();
    model->update_target_encoder(0.9);
    auto tgt_after = model->target_encoder->parameters()[0];

    float diff = (tgt_after - tgt_before).abs().mean().item<float>();
    IJEPA_CHECK(diff > 1e-6f, "EMA update changes target encoder weights");
}

// ── Target encoder frozen during forward ─────────────────────────────────────

static void test_target_encoder_no_grad() {
    IJEPAConfig cfg;
    cfg.img_size      = 32;
    cfg.patch_size    = 8;
    cfg.embed_dim     = 32;
    cfg.encoder_depth = 1;
    cfg.encoder_heads = 4;
    cfg.pred_dim      = 16;
    cfg.pred_depth    = 1;
    cfg.pred_heads    = 4;
    cfg.num_target_blocks = 2;

    IJEPA model(cfg);
    bool all_frozen = true;
    for (auto &p : model->target_encoder->parameters())
        if (p.requires_grad()) { all_frozen = false; break; }
    IJEPA_CHECK(all_frozen, "all target encoder params are frozen (requires_grad=false)");
}

// ── Factory helpers ───────────────────────────────────────────────────────────

static void test_factories() {
    auto b = make_ijepa_vit_b16(32);
    IJEPA_CHECK(b->cfg.embed_dim == 768,  "make_ijepa_vit_b16 embed_dim");

    auto l = make_ijepa_vit_l16(32);
    IJEPA_CHECK(l->cfg.embed_dim == 1024, "make_ijepa_vit_l16 embed_dim");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    torch::manual_seed(42);
    std::cout << "=== I-JEPA Tests ===\n\n";

    std::cout << "-- Config --\n";
    test_config_presets();

    std::cout << "\n-- Building blocks --\n";
    test_attention_shape();
    test_vitblock_shape();
    test_vit_encoder_full();
    test_vit_encoder_partial();

    std::cout << "\n-- Mask sampler --\n";
    test_mask_sampler_shapes();
    test_mask_sampler_range();
    test_mask_sampler_no_overlap();

    std::cout << "\n-- Predictor --\n";
    test_predictor_shape();

    std::cout << "\n-- Full model --\n";
    test_patchify();
    test_forward_loss_scalar();
    test_loss_backward();
    test_ema_update();
    test_target_encoder_no_grad();
    test_factories();

    std::cout << "\n=== Results: " << passed << " passed, " << failed
              << " failed ===\n";
    return (failed == 0) ? 0 : 1;
}
