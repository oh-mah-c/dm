// ─────────────────────────────────────────────────────────────────────────────
// Word2Vec implementation — Mikolov et al., arXiv:1301.3781v3
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/word2vec/word2vec.h"

#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_map>

namespace dm {
namespace models {
namespace nlp {

// ─────────────────────────────────────────────────────────────────────────────
// Vocabulary::build
// ─────────────────────────────────────────────────────────────────────────────
Vocabulary Vocabulary::build(const std::vector<std::string>& tokens,
                             int64_t max_vocab) {
    // Count frequencies
    std::unordered_map<std::string, int64_t> freq;
    for (auto& t : tokens) ++freq[t];

    // Sort by descending frequency
    std::vector<std::pair<int64_t, std::string>> sorted;
    sorted.reserve(freq.size());
    for (auto& kv : freq) sorted.push_back({kv.second, kv.first});
    std::sort(sorted.begin(), sorted.end(),
              [](auto& a, auto& b){ return a.first > b.first; });

    Vocabulary v;
    // id=0 → <UNK>
    v.id2word.push_back("<UNK>");
    v.counts.push_back(0);
    v.word2id["<UNK>"] = 0;

    int64_t kept = std::min<int64_t>((int64_t)sorted.size(), max_vocab - 1);
    for (int64_t i = 0; i < kept; ++i) {
        int32_t id = (int32_t)(i + 1);
        v.id2word.push_back(sorted[i].second);
        v.counts.push_back(sorted[i].first);
        v.word2id[sorted[i].second] = id;
    }
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Word2VecImpl constructor
// ─────────────────────────────────────────────────────────────────────────────
Word2VecImpl::Word2VecImpl(int64_t vocab_size_, int64_t embed_dim_,
                           Word2VecMode mode_)
    : vocab_size(vocab_size_), embed_dim(embed_dim_), mode(mode_) {

    W_in  = register_module("W_in",
                torch::nn::Embedding(vocab_size_, embed_dim_));
    W_out = register_module("W_out",
                torch::nn::Embedding(vocab_size_, embed_dim_));

    // Initialise W_in: uniform [-0.5/D, 0.5/D]  (standard word2vec init)
    torch::nn::init::uniform_(W_in->weight,
        -0.5f / (float)embed_dim_, 0.5f / (float)embed_dim_);
    // W_out initialised to zero
    torch::nn::init::zeros_(W_out->weight);
}

// ─────────────────────────────────────────────────────────────────────────────
// CBOW forward  (Section 3.1)
// context_ids: [batch, 2*window]
// Returns:     [batch, vocab_size]
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor Word2VecImpl::forward_cbow(const torch::Tensor& context_ids) {
    // Look up + sum context embeddings  (the paper says "sum/average projected")
    auto ctx = W_in->forward(context_ids);           // [B, 2w, D]
    auto h   = ctx.mean(1);                          // [B, D]  — averaged projection
    // Score against all output embeddings
    auto scores = torch::matmul(h, W_out->weight.t()); // [B, V]
    return scores;
}

// ─────────────────────────────────────────────────────────────────────────────
// Skip-gram forward  (Section 3.2)
// centre_ids:  [batch]
// context_ids: [batch]
// Returns:     [batch]  — unnormalised dot-product score
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor Word2VecImpl::forward_skipgram(const torch::Tensor& centre_ids,
                                              const torch::Tensor& context_ids) {
    auto h_in  = W_in->forward(centre_ids);    // [B, D]
    auto h_out = W_out->forward(context_ids);  // [B, D]
    return (h_in * h_out).sum(1);              // [B]  dot product per sample
}

// ─────────────────────────────────────────────────────────────────────────────
// Negative-sampling loss  (binary cross-entropy over positive + negative pairs)
// -log σ(pos) - Σ_k log σ(-neg_k)
// pos_scores: [batch]
// neg_scores: [batch, n_neg]
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor Word2VecImpl::ns_loss(const torch::Tensor& pos_scores,
                                    const torch::Tensor& neg_scores) {
    auto pos_loss = torch::nn::functional::binary_cross_entropy_with_logits(
        pos_scores,
        torch::ones_like(pos_scores),
        torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
            .reduction(torch::kMean));

    auto neg_loss = torch::nn::functional::binary_cross_entropy_with_logits(
        neg_scores,
        torch::zeros_like(neg_scores),
        torch::nn::functional::BinaryCrossEntropyWithLogitsFuncOptions()
            .reduction(torch::kMean));

    return pos_loss + neg_loss;
}

// ─────────────────────────────────────────────────────────────────────────────
// embedding: return L2-normalised W_in vector for word_id  [D]
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor Word2VecImpl::embedding(int32_t word_id) const {
    torch::NoGradGuard ng;
    auto v = W_in->weight[word_id];
    return v / (v.norm() + 1e-8f);
}

// ─────────────────────────────────────────────────────────────────────────────
// cosine_similarity
// ─────────────────────────────────────────────────────────────────────────────
float Word2VecImpl::cosine_similarity(int32_t a, int32_t b) const {
    torch::NoGradGuard ng;
    auto va = embedding(a);
    auto vb = embedding(b);
    return (va * vb).sum().item<float>();
}

// ─────────────────────────────────────────────────────────────────────────────
// analogy: vec(a) - vec(b) + vec(c), find closest word (excl. a,b,c)
// Section 4: vec(King) - vec(Man) + vec(Woman) ≈ vec(Queen)
// ─────────────────────────────────────────────────────────────────────────────
int32_t Word2VecImpl::analogy(int32_t a, int32_t b, int32_t c) const {
    torch::NoGradGuard ng;
    auto query = embedding(a) - embedding(b) + embedding(c);  // [D]
    query = query / (query.norm() + 1e-8f);

    // Normalise all embeddings and compute cosine similarity
    auto W = W_in->weight;                           // [V, D]
    auto norms = W.norm(2, 1, true).clamp_min(1e-8f); // [V, 1]
    auto W_norm = W / norms;                          // [V, D]
    auto scores = torch::matmul(W_norm, query);       // [V]

    // Exclude input words
    scores[a] = -1e9f;
    scores[b] = -1e9f;
    scores[c] = -1e9f;

    return scores.argmax().item<int32_t>();
}

// ─────────────────────────────────────────────────────────────────────────────
// most_similar: top-k words by cosine similarity to query
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::pair<int32_t, float>>
Word2VecImpl::most_similar(int32_t query_id, int64_t k) const {
    torch::NoGradGuard ng;
    auto W = W_in->weight;
    auto norms = W.norm(2, 1, true).clamp_min(1e-8f);
    auto W_norm = W / norms;
    auto q = W_norm[query_id];                        // [D]
    auto scores = torch::matmul(W_norm, q);           // [V]
    scores[query_id] = -1e9f;                         // exclude self

    auto topk = torch::topk(scores, k);
    auto vals  = std::get<0>(topk);
    auto idxs  = std::get<1>(topk);

    std::vector<std::pair<int32_t, float>> result;
    result.reserve(k);
    for (int64_t i = 0; i < k; ++i)
        result.push_back({idxs[i].item<int32_t>(), vals[i].item<float>()});
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// build_ns_table: unigram^(3/4) negative sampling table
// ─────────────────────────────────────────────────────────────────────────────
torch::Tensor build_ns_table(const Vocabulary& vocab, int64_t table_size /*=1'000'000*/) {
    std::vector<int32_t> table;
    table.reserve(table_size);

    // Compute count^(3/4) total
    double total = 0.0;
    for (int64_t c : vocab.counts) total += std::pow((double)c, 0.75);

    int32_t V = vocab.size();
    int64_t idx = 0;
    double cumulative = 0.0;
    for (int32_t w = 0; w < V; ++w) {
        cumulative += std::pow((double)vocab.counts[w], 0.75) / total;
        int64_t until = (int64_t)(cumulative * table_size);
        while ((int64_t)table.size() < until && (int64_t)table.size() < table_size)
            table.push_back(w);
        if ((int64_t)table.size() >= table_size) break;
    }
    while ((int64_t)table.size() < table_size)
        table.push_back(V - 1);

    return torch::tensor(table, torch::kInt32);
}

// ─────────────────────────────────────────────────────────────────────────────
// build_pairs: create training (input, target) pairs
// ─────────────────────────────────────────────────────────────────────────────
TrainingPairs build_pairs(const std::vector<int32_t>& ids,
                          const Word2VecConfig& cfg) {
    std::mt19937 rng(42);
    int64_t N = (int64_t)ids.size();
    int64_t w = cfg.window;

    std::vector<int32_t> inputs_v, targets_v;

    if (cfg.mode == Word2VecMode::SkipGram) {
        // For each centre word, sample window size r in [1, w], emit (centre, ctx)
        std::uniform_int_distribution<int64_t> wdist(1, w);
        for (int64_t i = 0; i < N; ++i) {
            int64_t r = wdist(rng);
            for (int64_t j = -r; j <= r; ++j) {
                if (j == 0) continue;
                int64_t ctx = i + j;
                if (ctx < 0 || ctx >= N) continue;
                inputs_v.push_back(ids[i]);
                targets_v.push_back(ids[ctx]);
            }
        }
    } else {
        // CBOW: for each centre, collect 2w context words (pad with <UNK>=0)
        // Return context sum as single averaged vector lookup
        // We store the 2w context ids for each centre
        // inputs: [N, 2w], targets: [N]
        // We flatten here and reconstruct shape in train loop
        for (int64_t i = 0; i < N; ++i) {
            // Check at least one neighbour exists
            bool any = false;
            for (int64_t j = -w; j <= w; ++j) {
                if (j == 0) continue;
                if (i+j >= 0 && i+j < N) { any = true; break; }
            }
            if (!any) continue;
            for (int64_t j = -w; j <= w; ++j) {
                if (j == 0) continue;
                int64_t ctx = i + j;
                inputs_v.push_back(ctx >= 0 && ctx < N ? ids[ctx] : 0);
            }
            targets_v.push_back(ids[i]);
        }
    }

    if (cfg.mode == Word2VecMode::SkipGram) {
        return {
            torch::tensor(inputs_v,  torch::kInt32),
            torch::tensor(targets_v, torch::kInt32)
        };
    } else {
        int64_t n_pairs = (int64_t)targets_v.size();
        int64_t ctx_len = 2 * w;
        return {
            torch::tensor(inputs_v, torch::kInt32).view({n_pairs, ctx_len}),
            torch::tensor(targets_v, torch::kInt32)
        };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// word2vec_train_epoch
// ─────────────────────────────────────────────────────────────────────────────
float word2vec_train_epoch(Word2Vec& model,
                           const TrainingPairs& pairs,
                           const torch::Tensor& ns_table,
                           const Word2VecConfig& cfg,
                           double lr) {
    model->train();
    torch::optim::SGD opt(model->parameters(),
        torch::optim::SGDOptions(lr));

    auto& inputs  = pairs.inputs;
    auto& targets = pairs.targets;
    int64_t N     = targets.size(0);
    int64_t B     = cfg.batch_size;
    int64_t table_size = ns_table.size(0);

    double total_loss = 0.0;
    int64_t steps = 0;

    // Random shuffled indices
    auto perm = torch::randperm(N, torch::kLong);
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int64_t> ns_dist(0, table_size - 1);

    for (int64_t start = 0; start + B <= N; start += B) {
        auto idx    = perm.slice(0, start, start + B);
        auto tgt    = targets.index_select(0, idx).to(torch::kLong).to(cfg.device);

        torch::Tensor pos_scores;

        if (cfg.mode == Word2VecMode::SkipGram) {
            auto inp = inputs.index_select(0, idx).to(torch::kLong).to(cfg.device);
            pos_scores = model->forward_skipgram(inp, tgt);  // [B]
        } else {
            auto inp = inputs.index_select(0, idx).to(torch::kLong).to(cfg.device);
            auto scores = model->forward_cbow(inp);           // [B, V]
            // Gather score for the target word
            pos_scores = scores.gather(1, tgt.unsqueeze(1)).squeeze(1);  // [B]
        }

        // Sample negative vocab ids via NS table (unigram^3/4 distribution)
        std::vector<int64_t> ns_pos_v(B * cfg.neg_samples);
        for (auto& v : ns_pos_v) v = ns_dist(rng);
        auto ns_pos = torch::tensor(ns_pos_v, torch::kLong);  // positions in ns_table
        // Look up vocab ids from the NS table: [B * n_neg]
        auto neg_vocab_flat = ns_table.index_select(0, ns_pos).to(torch::kLong).to(cfg.device);
        auto neg_vocab      = neg_vocab_flat.view({B, cfg.neg_samples});

        torch::Tensor neg_scores;
        if (cfg.mode == Word2VecMode::SkipGram) {
            auto inp = inputs.index_select(0, idx).to(torch::kLong).to(cfg.device);
            // Expand centre for each negative sample
            auto inp_exp  = inp.unsqueeze(1).expand({B, cfg.neg_samples}).reshape({B * cfg.neg_samples});
            neg_scores = model->forward_skipgram(inp_exp, neg_vocab_flat).view({B, cfg.neg_samples});
        } else {
            auto inp = inputs.index_select(0, idx).to(torch::kLong).to(cfg.device);
            auto all_scores = model->forward_cbow(inp);         // [B, V]
            neg_scores = all_scores.gather(1, neg_vocab);       // [B, n_neg] — vocab ids in [0,V)
        }

        auto loss = Word2VecImpl::ns_loss(pos_scores, neg_scores);
        opt.zero_grad();
        loss.backward();
        opt.step();

        total_loss += loss.item<double>();
        ++steps;
    }
    return steps > 0 ? (float)(total_loss / steps) : 0.f;
}

// ─────────────────────────────────────────────────────────────────────────────
// word2vec_train — full training loop
// ─────────────────────────────────────────────────────────────────────────────
void word2vec_train(Word2Vec& model,
                    const Vocabulary& vocab,
                    const std::vector<int32_t>& token_ids,
                    const Word2VecConfig& cfg,
                    const std::string& save_path) {
    model->to(cfg.device);

    std::cout << "[Word2Vec] mode="
              << (cfg.mode == Word2VecMode::SkipGram ? "SkipGram" : "CBOW")
              << "  vocab=" << vocab.size()
              << "  dim=" << cfg.embed_dim
              << "  window=" << cfg.window
              << "  neg=" << cfg.neg_samples << "\n";

    auto ns_table = build_ns_table(vocab);
    auto pairs    = build_pairs(token_ids, cfg);
    std::cout << "[Word2Vec] training pairs: " << pairs.targets.size(0) << "\n";

    for (int64_t ep = 0; ep < cfg.epochs; ++ep) {
        // Linear LR decay: starts at lr_start, reaches 0 at last step
        double progress = (double)ep / std::max<int64_t>(cfg.epochs - 1, 1);
        double lr       = cfg.lr_start * (1.0 - progress);
        lr              = std::max(lr, cfg.lr_start * 0.0001);  // min lr floor

        float loss = word2vec_train_epoch(model, pairs, ns_table, cfg, lr);
        std::cout << "[Word2Vec] epoch " << ep + 1 << "/" << cfg.epochs
                  << "  lr=" << lr
                  << "  loss=" << loss << "\n";
    }

    torch::serialize::OutputArchive ar;
    model->save(ar);
    ar.save_to(save_path);
    std::cout << "[Word2Vec] saved → " << save_path << "\n";
}

} // namespace nlp
} // namespace models
} // namespace dm
