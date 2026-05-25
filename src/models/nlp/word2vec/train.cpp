// ─────────────────────────────────────────────────────────────────────────────
// Word2Vec training entry point — Mikolov et al., arXiv:1301.3781v3
//
// Usage:
//   ./word2vec_train [options]
//   --mode    cbow|skipgram        (default: skipgram)
//   --dim     <int>                (default: 300  — Table 4)
//   --window  <int>                (default: 5    — half-window; C=10)
//   --neg     <int>                (default: 5    — negative samples)
//   --epochs  <int>                (default: 3    — Table 5)
//   --batch   <int>                (default: 512)
//   --lr      <float>              (default: 0.025)
//   --vocab   <int>                (default: 1000000)
//   --data    <path>               text file, one token per line or space-sep
//   --save    <path>               (default: word2vec.pt)
//   --cuda
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/word2vec/word2vec.h"

#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace dm::models::nlp;

struct Args {
    std::string  mode    = "skipgram";
    int64_t      dim     = 300;
    int64_t      window  = 5;
    int64_t      neg     = 5;
    int64_t      epochs  = 3;
    int64_t      batch   = 512;
    double       lr      = 0.025;
    int64_t      vocab   = 1'000'000;
    std::string  data    = "./data/corpus.txt";
    std::string  save    = "word2vec.pt";
    bool         cuda    = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--mode"   && i+1<argc) a.mode   = argv[++i];
        else if (k == "--dim"    && i+1<argc) a.dim     = std::stoll(argv[++i]);
        else if (k == "--window" && i+1<argc) a.window  = std::stoll(argv[++i]);
        else if (k == "--neg"    && i+1<argc) a.neg     = std::stoll(argv[++i]);
        else if (k == "--epochs" && i+1<argc) a.epochs  = std::stoll(argv[++i]);
        else if (k == "--batch"  && i+1<argc) a.batch   = std::stoll(argv[++i]);
        else if (k == "--lr"     && i+1<argc) a.lr      = std::stod(argv[++i]);
        else if (k == "--vocab"  && i+1<argc) a.vocab   = std::stoll(argv[++i]);
        else if (k == "--data"   && i+1<argc) a.data    = argv[++i];
        else if (k == "--save"   && i+1<argc) a.save    = argv[++i];
        else if (k == "--cuda")               a.cuda    = true;
    }
    return a;
}

static std::vector<std::string> load_tokens(const std::string& path) {
    std::vector<std::string> tokens;
    std::ifstream f(path);
    if (!f.is_open()) return tokens;
    std::string line, tok;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        while (ss >> tok) tokens.push_back(tok);
    }
    return tokens;
}

int main(int argc, char* argv[]) {
    auto args   = parse_args(argc, argv);
    auto device = (args.cuda && torch::cuda::is_available())
                  ? torch::kCUDA : torch::kCPU;
    std::cout << "[Word2Vec] device: " << device << "\n";

    auto tokens = load_tokens(args.data);
    if (tokens.empty()) {
        std::cout << "[Word2Vec] no data at " << args.data
                  << " — using synthetic tokens for smoke-test\n";
        for (int i = 0; i < 1000; ++i)
            tokens.push_back("word" + std::to_string(i % 50));
    }
    std::cout << "[Word2Vec] loaded " << tokens.size() << " tokens\n";

    auto vocab = Vocabulary::build(tokens, args.vocab);
    std::cout << "[Word2Vec] vocab size: " << vocab.size() << "\n";

    // Encode token sequence
    std::vector<int32_t> ids;
    ids.reserve(tokens.size());
    for (auto& t : tokens) ids.push_back(vocab.encode(t));

    Word2VecConfig cfg;
    cfg.mode       = (args.mode == "cbow") ? Word2VecMode::CBOW : Word2VecMode::SkipGram;
    cfg.embed_dim  = args.dim;
    cfg.window     = args.window;
    cfg.neg_samples = args.neg;
    cfg.epochs     = args.epochs;
    cfg.batch_size = args.batch;
    cfg.lr_start   = args.lr;
    cfg.max_vocab  = args.vocab;
    cfg.device     = device;

    auto model = Word2Vec(vocab.size(), cfg.embed_dim, cfg.mode);
    word2vec_train(model, vocab, ids, cfg, args.save);
    return 0;
}
