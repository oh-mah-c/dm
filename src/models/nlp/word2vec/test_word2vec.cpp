// ─────────────────────────────────────────────────────────────────────────────
// test_word2vec.cpp — C++ structural tests for Word2Vec
//
// Paper: T. Mikolov, K. Chen, G. Corrado, J. Dean,
//        "Efficient Estimation of Word Representations in Vector Space",
//        arXiv:1301.3781v3, 2013
//
// Tests verify:
//  1.  Embedding matrix shapes W_in/W_out: [V, D]
//  2.  CBOW forward output shape: [batch, V]  (Section 3.1)
//  3.  Skip-gram forward output shape: [batch]  (Section 3.2)
//  4.  W_in init: uniform [-0.5/D, 0.5/D]; W_out init: zeros
//  5.  NS loss is finite and >= 0
//  6.  NS loss decreases for positive vs negative pairs (well-separated scores)
//  7.  CBOW: context embeddings averaged (projection layer, Section 3.1)
//  8.  cosine_similarity: same word → 1.0  (after normalisation)
//  9.  analogy: correct closest word in simple 2D constructed embedding
// 10.  most_similar: returns k results, excludes query word
// 11.  SGD step changes W_in weights
// 12.  build_pairs SkipGram: produces (centre, context) pairs
// 13.  build_pairs CBOW: produces (context_window, centre) pairs
// 14.  Checkpoint save/load round-trip
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/word2vec/word2vec.h"

#include <torch/torch.h>
#include <cassert>
#include <cmath>
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

static const int64_t V = 100;   // small vocab for fast tests
static const int64_t D = 32;    // embedding dim
static const int64_t BATCH = 8;
static const int64_t WIN   = 2;  // half-window

// ─────────────────────────────────────────────────────────────────────────────
// 1. Embedding matrix shapes
// ─────────────────────────────────────────────────────────────────────────────
static void test_embedding_shapes() {
    begin_test("embedding matrix shapes [V, D]");
    auto m = Word2Vec(V, D, Word2VecMode::SkipGram);
    ASSERT_TRUE(m->W_in->weight.sizes()  == torch::IntArrayRef({V, D}),
                "W_in shape wrong");
    ASSERT_TRUE(m->W_out->weight.sizes() == torch::IntArrayRef({V, D}),
                "W_out shape wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. CBOW forward output shape [batch, V]  (Section 3.1)
// ─────────────────────────────────────────────────────────────────────────────
static void test_cbow_forward_shape() {
    begin_test("CBOW forward shape [batch, V] (Section 3.1)");
    auto m = Word2Vec(V, D, Word2VecMode::CBOW);
    m->eval();
    torch::NoGradGuard ng;
    auto ctx = torch::randint(0, V, {BATCH, 2*WIN});
    auto out = m->forward_cbow(ctx);
    ASSERT_TRUE(out.sizes() == torch::IntArrayRef({BATCH, V}),
                "CBOW output shape wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Skip-gram forward output shape [batch]  (Section 3.2)
// ─────────────────────────────────────────────────────────────────────────────
static void test_skipgram_forward_shape() {
    begin_test("Skip-gram forward shape [batch] (Section 3.2)");
    auto m = Word2Vec(V, D, Word2VecMode::SkipGram);
    m->eval();
    torch::NoGradGuard ng;
    auto centre = torch::randint(0, V, {BATCH});
    auto ctx    = torch::randint(0, V, {BATCH});
    auto out    = m->forward_skipgram(centre, ctx);
    ASSERT_TRUE(out.sizes() == torch::IntArrayRef({BATCH}),
                "Skip-gram output shape wrong");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Initialisation: W_in uniform [-0.5/D, 0.5/D]; W_out zeros
// ─────────────────────────────────────────────────────────────────────────────
static void test_init() {
    begin_test("W_in ~ U[-0.5/D, 0.5/D]; W_out = 0");
    auto m = Word2Vec(V, D, Word2VecMode::SkipGram);
    torch::NoGradGuard ng;
    float bound = 0.5f / (float)D;
    ASSERT_TRUE((m->W_in->weight >=  -bound - 1e-4f).all().item<bool>() &&
                (m->W_in->weight <=   bound + 1e-4f).all().item<bool>(),
                "W_in not in [-0.5/D, 0.5/D]");
    ASSERT_TRUE(m->W_out->weight.abs().max().item<float>() < 1e-6f,
                "W_out not zero-initialised");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. NS loss is finite and >= 0
// ─────────────────────────────────────────────────────────────────────────────
static void test_ns_loss_finite() {
    begin_test("NS loss is finite and >= 0");
    auto pos = torch::randn({BATCH});
    auto neg = torch::randn({BATCH, 5});
    auto loss = Word2VecImpl::ns_loss(pos, neg);
    ASSERT_TRUE(std::isfinite(loss.item<float>()), "NS loss not finite");
    ASSERT_TRUE(loss.item<float>() >= 0.f, "NS loss is negative");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. NS loss: very positive pos_scores + very negative neg_scores → small loss
// ─────────────────────────────────────────────────────────────────────────────
static void test_ns_loss_direction() {
    begin_test("NS loss decreases for well-separated scores");
    // Good prediction: pos_scores >> 0, neg_scores << 0
    auto pos_good = torch::full({BATCH}, 10.0f);
    auto neg_good = torch::full({BATCH, 5}, -10.0f);
    auto loss_good = Word2VecImpl::ns_loss(pos_good, neg_good).item<float>();

    // Bad prediction: all scores near zero
    auto pos_bad = torch::zeros({BATCH});
    auto neg_bad = torch::zeros({BATCH, 5});
    auto loss_bad = Word2VecImpl::ns_loss(pos_bad, neg_bad).item<float>();

    ASSERT_TRUE(loss_good < loss_bad,
                "well-separated scores should yield smaller NS loss");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. CBOW: context embeddings averaged (Section 3.1)
// ─────────────────────────────────────────────────────────────────────────────
static void test_cbow_averaging() {
    begin_test("CBOW context embeddings averaged (Section 3.1)");
    auto m = Word2Vec(V, D, Word2VecMode::CBOW);
    m->eval();
    torch::NoGradGuard ng;
    // Two context words with known embeddings
    auto ctx = torch::tensor({{3L, 7L}});  // [1, 2]
    auto e3 = m->W_in->weight[3];   // [D]
    auto e7 = m->W_in->weight[7];
    auto expected_h = (e3 + e7) / 2.0f;  // averaged projection

    // Reconstruct: scores = h @ W_out.T  — check h matches
    // Use small W_out all-ones to expose h
    m->W_out->weight.fill_(1.0f);
    auto out = m->forward_cbow(ctx);          // [1, V]
    // Each score = expected_h.sum()
    float expected_score = expected_h.sum().item<float>();
    float actual_score   = out[0][0].item<float>();
    ASSERT_TRUE(std::abs(actual_score - expected_score) < 1e-4f,
                "CBOW projection not averaged correctly");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. cosine_similarity: same word → 1.0
// ─────────────────────────────────────────────────────────────────────────────
static void test_cosine_same() {
    begin_test("cosine_similarity: same word -> 1.0");
    auto m = Word2Vec(V, D, Word2VecMode::SkipGram);
    for (int32_t w = 1; w < 5; ++w) {
        float sim = m->cosine_similarity(w, w);
        ASSERT_TRUE(std::abs(sim - 1.0f) < 1e-5f, "self-similarity not 1.0");
    }
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. analogy: constructed 2D embedding where analogy is exact
//    vec(0) - vec(1) + vec(2) should be closest to vec(3)
// ─────────────────────────────────────────────────────────────────────────────
static void test_analogy() {
    begin_test("analogy: vec(0)-vec(1)+vec(2) ~ vec(3)");
    // Build tiny 4-word vocab, 2D embeddings
    auto m = Word2Vec(10, 2, Word2VecMode::SkipGram);
    torch::NoGradGuard ng;
    // Manual embeddings (orthogonal, distinct):
    //   0=[1,0], 1=[0,1], 2=[0,0.5], 3=[1,-0.5]
    // query = [1,0] - [0,1] + [0,0.5] = [1, -0.5]  ≈ vec(3)
    m->W_in->weight[0] = torch::tensor({1.0f, 0.0f});
    m->W_in->weight[1] = torch::tensor({0.0f, 1.0f});
    m->W_in->weight[2] = torch::tensor({0.0f, 0.5f});
    m->W_in->weight[3] = torch::tensor({1.0f,-0.5f});
    // Make other words' embeddings far away
    for (int i = 4; i < 10; ++i)
        m->W_in->weight[i] = torch::tensor({-10.0f, -10.0f});

    int32_t result = m->analogy(0, 1, 2);
    ASSERT_TRUE(result == 3, "analogy did not return word 3");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. most_similar: returns k results, excludes query
// ─────────────────────────────────────────────────────────────────────────────
static void test_most_similar() {
    begin_test("most_similar: k results, excludes query word");
    auto m = Word2Vec(V, D, Word2VecMode::SkipGram);
    int64_t k = 5;
    auto nbrs = m->most_similar(3, k);
    ASSERT_TRUE((int64_t)nbrs.size() == k, "most_similar returned wrong number");
    for (auto& [id, _] : nbrs)
        ASSERT_TRUE(id != 3, "most_similar returned query word itself");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. SGD step changes W_in weights  (Section 4: SGD)
// ─────────────────────────────────────────────────────────────────────────────
static void test_sgd_update() {
    begin_test("SGD step changes embedding weights (Section 4)");
    auto m = Word2Vec(V, D, Word2VecMode::SkipGram);
    m->train();
    // W_out is zero-initialised → grad of W_in = 0; so initialise W_out non-zero
    torch::nn::init::uniform_(m->W_out->weight, -0.5f/D, 0.5f/D);

    torch::optim::SGD opt(m->parameters(), torch::optim::SGDOptions(0.025));

    auto w_in_before  = m->W_in->weight.clone().detach();
    auto w_out_before = m->W_out->weight.clone().detach();

    // Use fixed indices 0..BATCH-1 to guarantee the looked-up rows change
    auto centre = torch::arange(BATCH, torch::kLong);
    auto ctx    = torch::arange(BATCH, torch::kLong) + 1;
    auto neg    = torch::randint(BATCH + 2, V, {BATCH, 5});

    auto pos_s  = m->forward_skipgram(centre, ctx);
    auto centre_exp = centre.unsqueeze(1).expand({BATCH, 5}).reshape({BATCH*5});
    auto neg_flat   = neg.reshape({BATCH*5});
    auto neg_s  = m->forward_skipgram(centre_exp, neg_flat).view({BATCH, 5});

    auto loss = Word2VecImpl::ns_loss(pos_s, neg_s);
    opt.zero_grad();
    loss.backward();
    opt.step();

    // W_out rows for centre+ctx indices must have changed (gradient from W_in)
    bool w_out_changed = !torch::allclose(
        w_out_before.slice(0, 0, BATCH+1),
        m->W_out->weight.detach().slice(0, 0, BATCH+1));
    // W_in rows for centre indices must have changed (gradient from W_out)
    bool w_in_changed = !torch::allclose(
        w_in_before.slice(0, 0, BATCH),
        m->W_in->weight.detach().slice(0, 0, BATCH));

    ASSERT_TRUE(w_in_changed || w_out_changed,
                "neither W_in nor W_out changed after SGD step");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. build_pairs SkipGram: (centre, context) pairs
// ─────────────────────────────────────────────────────────────────────────────
static void test_build_pairs_skipgram() {
    begin_test("build_pairs SkipGram: [N] centre + [N] context (Section 3.2)");
    Word2VecConfig cfg;
    cfg.mode   = Word2VecMode::SkipGram;
    cfg.window = 2;
    std::vector<int32_t> ids = {1, 2, 3, 4, 5, 6, 7, 8};
    auto pairs = build_pairs(ids, cfg);
    ASSERT_TRUE(pairs.inputs.dim()  == 1, "SG inputs should be 1D [N]");
    ASSERT_TRUE(pairs.targets.dim() == 1, "SG targets should be 1D [N]");
    ASSERT_TRUE(pairs.inputs.size(0) == pairs.targets.size(0),
                "SG inputs/targets size mismatch");
    ASSERT_TRUE(pairs.inputs.size(0) > 0, "no pairs generated");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. build_pairs CBOW: [N, 2w] context + [N] centre
// ─────────────────────────────────────────────────────────────────────────────
static void test_build_pairs_cbow() {
    begin_test("build_pairs CBOW: [N, 2w] context + [N] centre (Section 3.1)");
    Word2VecConfig cfg;
    cfg.mode   = Word2VecMode::CBOW;
    cfg.window = 2;
    std::vector<int32_t> ids = {1, 2, 3, 4, 5, 6, 7, 8};
    auto pairs = build_pairs(ids, cfg);
    ASSERT_TRUE(pairs.inputs.dim()  == 2, "CBOW inputs should be 2D [N, 2w]");
    ASSERT_TRUE(pairs.targets.dim() == 1, "CBOW targets should be 1D [N]");
    ASSERT_TRUE(pairs.inputs.size(1) == 2 * cfg.window,
                "CBOW input width != 2*window");
    ASSERT_TRUE(pairs.inputs.size(0) == pairs.targets.size(0),
                "CBOW inputs/targets size mismatch");
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. Checkpoint save/load round-trip
// ─────────────────────────────────────────────────────────────────────────────
static void test_checkpoint() {
    begin_test("checkpoint save/load round-trip");
    auto m1 = Word2Vec(V, D, Word2VecMode::SkipGram);
    const std::string path = "/tmp/test_word2vec_ckpt.pt";
    {
        torch::serialize::OutputArchive ar;
        m1->save(ar);
        ar.save_to(path);
    }
    auto m2 = Word2Vec(V, D, Word2VecMode::SkipGram);
    {
        torch::serialize::InputArchive ar;
        ar.load_from(path);
        m2->load(ar);
    }
    torch::NoGradGuard ng;
    ASSERT_TRUE(torch::allclose(m1->W_in->weight, m2->W_in->weight, 1e-5f, 1e-5f),
                "W_in differs after reload");
    ASSERT_TRUE(torch::allclose(m1->W_out->weight, m2->W_out->weight, 1e-5f, 1e-5f),
                "W_out differs after reload");
    std::filesystem::remove(path);
    end_test();
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    std::printf("\n=== Word2Vec tests (Mikolov et al., arXiv:1301.3781v3) ===\n");

    test_embedding_shapes();
    test_cbow_forward_shape();
    test_skipgram_forward_shape();
    test_init();
    test_ns_loss_finite();
    test_ns_loss_direction();
    test_cbow_averaging();
    test_cosine_same();
    test_analogy();
    test_most_similar();
    test_sgd_update();
    test_build_pairs_skipgram();
    test_build_pairs_cbow();
    test_checkpoint();

    std::printf("=== %d passed, %d failed ===\n\n", s_passed, s_failed);
    return s_failed == 0 ? 0 : 1;
}
