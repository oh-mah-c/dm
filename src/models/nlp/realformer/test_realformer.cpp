// ─────────────────────────────────────────────────────────────────────────────
// RealFormer unit tests (20 tests)
// He et al., ACL-IJCNLP 2021 Findings
//
// Tests exercise both the dm::prim pytorch primitive (ResidualMultiheadAttention)
// and the higher-level dm::models::nlp blocks (RealFormerBlock/Encoder/Decoder).
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/realformer/realformer.h"
// The pytorch primitive is already included transitively via realformer.h:
//   #include <torch/nn/modules/residual_attention.h>

#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <string>

using namespace dm::models::nlp;
using dm::prim::ResidualMultiheadAttention;
using dm::prim::ResidualMultiheadAttentionOptions;

// ── Test harness ─────────────────────────────────────────────────────────────
static int g_pass = 0, g_fail = 0;

static void check(bool cond, const std::string& name) {
    if (cond) { ++g_pass; std::cout << "  [PASS] " << name << "\n"; }
    else       { ++g_fail; std::cout << "  [FAIL] " << name << "\n"; }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Config presets produce correct fields
// ─────────────────────────────────────────────────────────────────────────────
static void test_config_presets() {
    auto bs  = RealFormerConfig::bert_base();
    auto bl  = RealFormerConfig::bert_large();
    auto sm  = RealFormerConfig::small();
    auto gpt = RealFormerConfig::gpt2_small();
    check(bs.d_model == 768  && bs.n_heads == 12  && bs.n_layers == 12, "bert_base dims");
    check(bl.d_model == 1024 && bl.n_heads == 16  && bl.n_layers == 24, "bert_large dims");
    check(sm.d_model == 64   && sm.n_heads == 4   && sm.n_layers == 2,  "small dims");
    check(gpt.causal && gpt.n_heads == 12 && gpt.d_model == 768,        "gpt2_small causal");
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. ResidualMultiheadAttention: options round-trip
// ─────────────────────────────────────────────────────────────────────────────
static void test_primitive_options() {
    ResidualMultiheadAttentionOptions opts(512, 8);
    opts.dropout(0.2).use_mean(true);
    auto mha = ResidualMultiheadAttention(opts);
    check(mha->options.embed_dim()  == 512,  "primitive embed_dim=512");
    check(mha->options.num_heads()  == 8,    "primitive num_heads=8");
    check(mha->options.use_mean()   == true, "primitive use_mean=true");
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. ResidualMultiheadAttention: output + prev shapes
// ─────────────────────────────────────────────────────────────────────────────
static void test_primitive_shapes() {
    int64_t B=2, T=8, D=64, H=4;
    auto mha = ResidualMultiheadAttention(D, H);
    mha->eval();
    auto x = torch::randn({B, T, D});
    auto [out, prev] = mha->forward(x, x, x);
    check(out.sizes()  == torch::IntArrayRef({B, T, D}),    "primitive out [B,T,D]");
    check(prev.sizes() == torch::IntArrayRef({B, H, T, T}), "primitive prev [B,H,T,T]");
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Prev accumulates: second call with prev1 != first call result
// ─────────────────────────────────────────────────────────────────────────────
static void test_prev_accumulates() {
    auto mha = ResidualMultiheadAttention(64, 4);
    mha->eval();
    auto x = torch::randn({1, 4, 64});
    auto [o1, prev1] = mha->forward(x, x, x);
    auto [o2, prev2] = mha->forward(x, x, x, prev1);
    check(!torch::allclose(prev1, prev2), "prev accumulates across layers");
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Undefined prev == zero prev (first layer behaviour)
// ─────────────────────────────────────────────────────────────────────────────
static void test_no_prev_equiv_zero() {
    torch::manual_seed(42);
    auto mha = ResidualMultiheadAttention(64, 4);
    mha->eval();
    auto x = torch::randn({1, 4, 64});
    auto [out_none, prev_none] = mha->forward(x, x, x);
    auto zero_prev = torch::zeros({1, 4, 4, 4});
    auto [out_zero, prev_zero] = mha->forward(x, x, x, zero_prev);
    check(torch::allclose(out_none,  out_zero,  1e-5f, 1e-5f), "undefined prev == zero output");
    check(torch::allclose(prev_none, prev_zero, 1e-5f, 1e-5f), "undefined prev == zero prev");
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Key-padding mask changes output
// ─────────────────────────────────────────────────────────────────────────────
static void test_key_padding_mask() {
    auto mha = ResidualMultiheadAttention(64, 4);
    mha->eval();
    int64_t B=1, T=6;
    auto x    = torch::randn({B, T, 64});
    auto mask = torch::zeros({B, T}, torch::kBool);
    mask.slice(1, 4) = true;   // mask last 2 positions
    auto [out_masked, _m]   = mha->forward(x, x, x, {}, mask);
    auto [out_unmasked, _u] = mha->forward(x, x, x, {}, {});
    check(!torch::allclose(out_masked, out_unmasked), "padding mask changes output");
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Causal flag: early positions unaffected by future tokens
// ─────────────────────────────────────────────────────────────────────────────
static void test_causal_flag() {
    torch::manual_seed(7);
    auto mha = ResidualMultiheadAttention(64, 4);
    mha->eval();
    int64_t T = 8;
    auto x1 = torch::randn({1, T, 64});
    auto x2 = x1.clone();
    x2.narrow(1, 5, T-5).copy_(torch::randn({1, T-5, 64}));
    auto [out1, _1] = mha->forward(x1, x1, x1, {}, {}, /*causal=*/true);
    auto [out2, _2] = mha->forward(x2, x2, x2, {}, {}, /*causal=*/true);
    auto s1 = out1.slice(1, 0, 5);
    auto s2 = out2.slice(1, 0, 5);
    check(torch::allclose(s1, s2, 1e-5f, 1e-5f), "causal: early positions unaffected by future");
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. use_mean mode: output/prev shapes unchanged
// ─────────────────────────────────────────────────────────────────────────────
static void test_use_mean_shapes() {
    ResidualMultiheadAttentionOptions opts(64, 4);
    opts.use_mean(true);
    auto mha = ResidualMultiheadAttention(opts);
    mha->eval();
    int64_t B=1, T=4, H=4;
    auto x = torch::randn({B, T, 64});
    auto [o1, prev1] = mha->forward(x, x, x, {},  {}, false, 1);
    auto [o2, prev2] = mha->forward(x, x, x, prev1, {}, false, 2);
    check(o2.sizes()    == torch::IntArrayRef({B, T, 64}),   "use_mean out shape");
    check(prev2.sizes() == torch::IntArrayRef({B, H, T, T}), "use_mean prev shape");
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. use_mean: scores bounded — |prev2| <= |prev1| + |raw| (not diverging)
// ─────────────────────────────────────────────────────────────────────────────
static void test_use_mean_bounded() {
    ResidualMultiheadAttentionOptions opts(64, 4);
    opts.use_mean(true);
    auto mha = ResidualMultiheadAttention(opts);
    mha->eval();
    auto x = torch::randn({1, 4, 64});
    torch::Tensor prev;
    for (int64_t l = 1; l <= 10; ++l) {
        auto [o, p] = mha->forward(x, x, x, prev, {}, false, l);
        prev = p;
    }
    // Running mean should not exceed the scale of a single raw-score matrix
    float max_val = prev.abs().max().item<float>();
    check(max_val < 1000.0f, "use_mean: scores don't diverge after 10 layers");
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Primitive: no tied projection weights
// ─────────────────────────────────────────────────────────────────────────────
static void test_no_tied_weights() {
    auto mha = ResidualMultiheadAttention(64, 4);
    auto* q = mha->wq->weight.data_ptr();
    auto* k = mha->wk->weight.data_ptr();
    auto* v = mha->wv->weight.data_ptr();
    auto* o = mha->wo->weight.data_ptr();
    check(q!=k && q!=v && q!=o && k!=v && k!=o && v!=o, "wq/wk/wv/wo are distinct");
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. RealFormerBlock output shape + prev shape
// ─────────────────────────────────────────────────────────────────────────────
static void test_block_shapes() {
    auto cfg = RealFormerConfig::small();
    auto blk = RealFormerBlock(cfg);
    blk->eval();
    int64_t B=2, T=6, D=cfg.d_model, H=cfg.n_heads;
    auto x = torch::randn({B, T, D});
    auto [out, prev] = blk->forward(x);
    check(out.sizes()  == torch::IntArrayRef({B, T, D}),    "block out [B,T,D]");
    check(prev.sizes() == torch::IntArrayRef({B, H, T, T}), "block prev [B,H,T,T]");
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Block has exactly 2 LayerNorms (Post-LN)
// ─────────────────────────────────────────────────────────────────────────────
static void test_block_two_layernorms() {
    auto cfg = RealFormerConfig::small();
    auto blk = RealFormerBlock(cfg);
    int cnt = 0;
    for (auto& item : blk->named_modules())
        if (dynamic_cast<torch::nn::LayerNormImpl*>(item.value().get()))
            ++cnt;
    check(cnt == 2, "block has exactly 2 LayerNorm modules");
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Block uses dm::prim::ResidualMultiheadAttention internally
// ─────────────────────────────────────────────────────────────────────────────
static void test_block_uses_primitive() {
    auto cfg = RealFormerConfig::small();
    auto blk = RealFormerBlock(cfg);
    bool found = (dynamic_cast<dm::prim::ResidualMultiheadAttentionImpl*>(
                      blk->attn.get()) != nullptr);
    check(found, "block.attn is dm::prim::ResidualMultiheadAttention");
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. Encoder output shape + correct block count
// ─────────────────────────────────────────────────────────────────────────────
static void test_encoder_shape_and_count() {
    auto cfg = RealFormerConfig::small();   // 2 layers
    auto enc = RealFormerEncoder(cfg);
    enc->eval();
    int64_t B=2, T=10, D=cfg.d_model, H=cfg.n_heads;
    auto [out, prev] = enc->forward(torch::randn({B, T, D}));
    check(out.sizes()  == torch::IntArrayRef({B, T, D}),    "encoder out [B,T,D]");
    check(prev.sizes() == torch::IntArrayRef({B, H, T, T}), "encoder prev [B,H,T,T]");

    int block_count = 0;
    for (auto& item : enc->named_children())
        if (item.key() == "layers")
            block_count = static_cast<int>(
                std::dynamic_pointer_cast<torch::nn::ModuleListImpl>(
                    item.value())->size());
    check(block_count == cfg.n_layers, "encoder has n_layers blocks");
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. Decoder enforces causal masking
// ─────────────────────────────────────────────────────────────────────────────
static void test_decoder_causal() {
    torch::manual_seed(13);
    auto cfg = RealFormerConfig::small();
    cfg.causal = false;          // decoder constructor forces true internally
    auto dec = RealFormerDecoder(cfg);
    dec->eval();
    int64_t T = 8;
    auto x1 = torch::randn({1, T, cfg.d_model});
    auto x2 = x1.clone();
    x2.narrow(1, 5, T-5).copy_(torch::randn({1, T-5, cfg.d_model}));
    auto [out1, _1] = dec->forward(x1);
    auto [out2, _2] = dec->forward(x2);
    check(torch::allclose(out1.slice(1,0,5), out2.slice(1,0,5), 1e-5f, 1e-5f),
          "decoder: early positions independent of future");
}

// ─────────────────────────────────────────────────────────────────────────────
// 16. Gradient flows through encoder
// ─────────────────────────────────────────────────────────────────────────────
static void test_gradient_flow() {
    auto cfg = RealFormerConfig::small();
    auto enc = RealFormerEncoder(cfg);
    enc->train();
    auto x = torch::randn({1, 4, cfg.d_model}, torch::requires_grad(true));
    auto [out, _] = enc->forward(x);
    out.sum().backward();
    bool any_grad = false;
    for (auto& p : enc->parameters())
        if (p.grad().defined() && p.grad().abs().max().item<float>() > 0.0f)
            any_grad = true;
    check(any_grad, "encoder: params have non-zero grad after backward");
}

// ─────────────────────────────────────────────────────────────────────────────
// 17. Prev scores differ layer-to-layer (residual is doing something)
// ─────────────────────────────────────────────────────────────────────────────
static void test_prev_differs_across_layers() {
    auto cfg = RealFormerConfig::small();
    cfg.n_layers = 3;
    auto enc = RealFormerEncoder(cfg);
    enc->eval();
    auto x = torch::randn({1, 5, cfg.d_model});
    auto blk0 = enc->layers_->ptr<RealFormerBlockImpl>(0);
    auto blk1 = enc->layers_->ptr<RealFormerBlockImpl>(1);
    auto [h0, prev0] = blk0->forward(x);
    auto [h1, prev1] = blk1->forward(h0, prev0);
    check(!torch::allclose(prev0, prev1), "prev scores differ between layers");
}

// ─────────────────────────────────────────────────────────────────────────────
// 18. Encoder eval/train modes produce same-shaped outputs
// ─────────────────────────────────────────────────────────────────────────────
static void test_eval_train_modes() {
    auto cfg = RealFormerConfig::small();
    auto enc = RealFormerEncoder(cfg);
    auto x   = torch::randn({1, 4, cfg.d_model});
    enc->train(); auto [ot, _t] = enc->forward(x);
    enc->eval();  auto [oe, _e] = enc->forward(x);
    check(ot.sizes() == oe.sizes(), "eval/train: same output shape");
}

// ─────────────────────────────────────────────────────────────────────────────
// 19. Primitive pretty_print doesn't crash
// ─────────────────────────────────────────────────────────────────────────────
static void test_pretty_print() {
    auto mha = ResidualMultiheadAttention(256, 8);
    std::ostringstream ss;
    mha->pretty_print(ss);
    bool ok = ss.str().find("ResidualMultiheadAttention") != std::string::npos;
    check(ok, "pretty_print contains module name");
}

// ─────────────────────────────────────────────────────────────────────────────
// 20. Param count sanity
// ─────────────────────────────────────────────────────────────────────────────
static void test_param_count() {
    auto cfg = RealFormerConfig::small();
    auto enc = RealFormerEncoder(cfg);
    int64_t total = 0;
    for (auto& p : enc->parameters()) total += p.numel();
    check(total > 0,          "encoder has > 0 parameters");
    check(total < 1000000LL,  "small encoder < 1M params");
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    torch::manual_seed(0);
    std::cout << "=== RealFormer tests ===\n";

    test_config_presets();
    test_primitive_options();
    test_primitive_shapes();
    test_prev_accumulates();
    test_no_prev_equiv_zero();
    test_key_padding_mask();
    test_causal_flag();
    test_use_mean_shapes();
    test_use_mean_bounded();
    test_no_tied_weights();
    test_block_shapes();
    test_block_two_layernorms();
    test_block_uses_primitive();
    test_encoder_shape_and_count();
    test_decoder_causal();
    test_gradient_flow();
    test_prev_differs_across_layers();
    test_eval_train_modes();
    test_pretty_print();
    test_param_count();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.\n";
    return g_fail ? 1 : 0;
}
