#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Word2Vec — Efficient Estimation of Word Representations in Vector Space
// T. Mikolov, K. Chen, G. Corrado, J. Dean — arXiv:1301.3781v3, 2013
//
// ── Two architectures  (Section 3, Figure 1) ────────────────────────────────
//
// CBOW  (Continuous Bag-of-Words, Section 3.1):
//   - Averages/sums embeddings of 2×window context words
//   - Predicts the centre word
//   - Complexity:  Q = N×D + D×log₂(V)  (Eq. 4)
//   - Window: 4 history + 4 future = 8 context words (paper default)
//
// Skip-gram  (Section 3.2):
//   - Uses centre word to predict each surrounding word
//   - Window up to C=10 (paper), actual window sampled uniformly from [1,C]
//   - Complexity:  Q = C×(D + D×log₂(V))  (Eq. 5)
//
// ── Shared design ────────────────────────────────────────────────────────────
//   - Embedding matrix  W_in  [V, D]  — input  / centre-word lookup
//   - Embedding matrix  W_out [V, D]  — output / context-word lookup
//   - No hidden layer, no non-linearity (log-linear model, Section 3)
//   - Output: dot(centre, context) + optional full softmax or NS/HS
//
// ── Training  (Section 4, Table 5) ──────────────────────────────────────────
//   - SGD, initial lr = 0.025, linearly decayed to 0 over training
//   - 3 epochs (or 1 epoch on 2× data — comparable, Section 4.3)
//   - Vocabulary: most frequent 1M words
//   - Vector dimensionality D = 300 (best single-machine result, Table 4)
//   - Negative sampling (follow-up / practical default) or hierarchical softmax
//
// ── Evaluation ───────────────────────────────────────────────────────────────
//   - Cosine similarity
//   - Analogy: vec(King) - vec(Man) + vec(Woman) ≈ vec(Queen)
//     find argmax_w cos(v_w, v_King - v_Man + v_Woman), excluding input words
// ─────────────────────────────────────────────────────────────────────────────

#include <torch/torch.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Vocabulary — maps tokens ↔ integer ids, stores unigram counts for NS
// ─────────────────────────────────────────────────────────────────────────────
struct Vocabulary {
    std::unordered_map<std::string, int32_t> word2id;
    std::vector<std::string>                 id2word;
    std::vector<int64_t>                     counts;     // unigram frequency

    // Build from a flat token list; keeps the top max_vocab most-frequent words
    // (plus a special <UNK> token at index 0).
    static Vocabulary build(const std::vector<std::string>& tokens,
                            int64_t max_vocab = 1'000'000);

    int32_t size() const { return static_cast<int32_t>(id2word.size()); }
    int32_t unk_id() const { return 0; }

    int32_t encode(const std::string& w) const {
        auto it = word2id.find(w);
        return it == word2id.end() ? 0 : it->second;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Word2Vec model variants
// ─────────────────────────────────────────────────────────────────────────────
enum class Word2VecMode { CBOW, SkipGram };

// ─────────────────────────────────────────────────────────────────────────────
// Word2VecImpl — shared embedding tables for CBOW and Skip-gram
//
// Trainable parameters:
//   W_in  [vocab_size, embed_dim]  — centre/input embeddings (initialised U[-0.5/D, 0.5/D])
//   W_out [vocab_size, embed_dim]  — context/output embeddings (initialised 0)
// ─────────────────────────────────────────────────────────────────────────────
struct Word2VecImpl : torch::nn::Module {
    int64_t        vocab_size;
    int64_t        embed_dim;
    Word2VecMode   mode;

    torch::nn::Embedding W_in{nullptr};   // input embeddings  [V, D]
    torch::nn::Embedding W_out{nullptr};  // output embeddings [V, D]

    Word2VecImpl(int64_t vocab_size,
                 int64_t embed_dim  = 300,
                 Word2VecMode mode  = Word2VecMode::SkipGram);

    // ── CBOW forward (Section 3.1) ──────────────────────────────────────────
    // context_ids: [batch, 2*window]  — indices of context words
    // Returns:     [batch, vocab_size] — unnormalised scores (before softmax)
    torch::Tensor forward_cbow(const torch::Tensor& context_ids);

    // ── Skip-gram forward (Section 3.2) ─────────────────────────────────────
    // centre_ids:  [batch]            — centre word indices
    // context_ids: [batch]            — one target context word per sample
    // Returns:     [batch]            — dot product score (positive pair)
    torch::Tensor forward_skipgram(const torch::Tensor& centre_ids,
                                   const torch::Tensor& context_ids);

    // ── Negative-sampling loss (practical training objective) ────────────────
    // Computes -log σ(pos_score) - Σ log σ(-neg_score) per sample; mean over batch
    // pos_scores: [batch]
    // neg_scores: [batch, n_neg]
    static torch::Tensor ns_loss(const torch::Tensor& pos_scores,
                                 const torch::Tensor& neg_scores);

    // ── Lookup: return normalised embedding for a word id ────────────────────
    torch::Tensor embedding(int32_t word_id) const;

    // ── Cosine similarity between two word ids ───────────────────────────────
    float cosine_similarity(int32_t a, int32_t b) const;

    // ── Analogy: v(a) - v(b) + v(c), find closest word (excluding a,b,c) ────
    // Implements: vec(King) - vec(Man) + vec(Woman) ≈ vec(Queen)  (Section 4)
    int32_t analogy(int32_t a, int32_t b, int32_t c) const;

    // ── Top-k most similar words to query_id ────────────────────────────────
    std::vector<std::pair<int32_t, float>> most_similar(int32_t query_id,
                                                         int64_t k = 10) const;
};
TORCH_MODULE(Word2Vec);

// ─────────────────────────────────────────────────────────────────────────────
// Training configuration  (Section 4, Table 5)
// ─────────────────────────────────────────────────────────────────────────────
struct Word2VecConfig {
    Word2VecMode mode       = Word2VecMode::SkipGram;
    int64_t  embed_dim      = 300;        // Table 4: D=300 best single-machine
    int64_t  window         = 5;          // context window half-size (C=10 → ±5)
    int64_t  neg_samples    = 5;          // negative samples per positive pair
    int64_t  max_vocab      = 1'000'000;  // top-1M most frequent words
    double   lr_start       = 0.025;      // Section 4.3: lr decreases to 0
    int64_t  epochs         = 3;          // Table 5: 3 epochs
    int64_t  batch_size     = 512;
    torch::Device device    = torch::kCPU;
};

// ─────────────────────────────────────────────────────────────────────────────
// Negative-sampling table: drawn proportional to count^(3/4)  (standard word2vec)
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor build_ns_table(const Vocabulary& vocab,
                             int64_t table_size = 1'000'000);

// ─────────────────────────────────────────────────────────────────────────────
// Build training pairs from a token sequence
//
// CBOW:      returns ({context_ids [N, 2*window], centre_ids [N]})
// Skip-gram: returns ({centre_ids [N], context_ids [N]})
//            window is sampled uniformly in [1, config.window] per centre word
// ─────────────────────────────────────────────────────────────────────────────
struct TrainingPairs {
    torch::Tensor inputs;   // CBOW: [N, 2w] context; SG: [N] centre
    torch::Tensor targets;  // CBOW: [N] centre;       SG: [N] context
};

TrainingPairs build_pairs(const std::vector<int32_t>& ids,
                          const Word2VecConfig& cfg);

// ─────────────────────────────────────────────────────────────────────────────
// Train one epoch. Returns average loss.
// ns_table: [table_size] int32 — pre-built negative-sampling table
// ─────────────────────────────────────────────────────────────────────────────
float word2vec_train_epoch(Word2Vec& model,
                           const TrainingPairs& pairs,
                           const torch::Tensor& ns_table,
                           const Word2VecConfig& cfg,
                           double lr);

// ─────────────────────────────────────────────────────────────────────────────
// Full training loop
// tokens: flat token sequence (all sentences concatenated)
// ─────────────────────────────────────────────────────────────────────────────
void word2vec_train(Word2Vec& model,
                    const Vocabulary& vocab,
                    const std::vector<int32_t>& token_ids,
                    const Word2VecConfig& cfg,
                    const std::string& save_path = "word2vec.pt");

} // namespace nlp
} // namespace models
} // namespace dm
