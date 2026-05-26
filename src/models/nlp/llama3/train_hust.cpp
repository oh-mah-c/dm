// ─────────────────────────────────────────────────────────────────────────────
// Llama 3 training on HUST-tokenized text
//
// Reads a HUST tokenizer output file (space-separated integer token IDs,
// one transaction per line) and trains the Llama 3 tiny model on it.
//
// Usage:
//   ./llama3_train_hust -input <tokens_file>
//                       [-dim <int>]        (default: 256)
//                       [-layers <int>]     (default: 4)
//                       [-heads <int>]      (default: 4)
//                       [-kvheads <int>]    (default: 2)
//                       [-ffn <int>]        (default: 512)
//                       [-seqlen <int>]     (default: 64)
//                       [-epochs <int>]     (default: 20)
//                       [-batch <int>]      (default: 32)
//                       [-lr <float>]       (default: 3e-4)
//                       [-warmup <int>]     (default: 200)
//                       [-maxiter <int>]    (default: 5000)
//                       [-save <path>]      (default: llama3_hust.pt)
//                       [--cuda]
//                       [--benchmark]
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/llama3/llama3.h"

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <chrono>

using namespace dm::models::nlp;

// ─────────────────────────────────────────────────────────────────────────────
// Dataset: read HUST token file → flat int64 vector
// ─────────────────────────────────────────────────────────────────────────────
struct HUSTDataset {
    std::vector<int64_t> tokens;   // flat token sequence
    int64_t vocab_size = 0;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "Cannot open: " << path << "\n";
            return false;
        }
        int64_t max_id = 0;
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            int64_t id;
            while (ss >> id) {
                tokens.push_back(id);
                if (id > max_id) max_id = id;
            }
        }
        vocab_size = max_id + 1;
        return !tokens.empty();
    }

    // Sample a random batch of [B, seqlen+1] token windows
    torch::Tensor sample_batch(int64_t B, int64_t seqlen,
                                std::mt19937& rng) const {
        int64_t N = (int64_t)tokens.size();
        assert(N > seqlen + 1);

        std::uniform_int_distribution<int64_t> dist(0, N - seqlen - 2);
        std::vector<int64_t> buf;
        buf.reserve(B * (seqlen + 1));

        for (int64_t i = 0; i < B; ++i) {
            int64_t start = dist(rng);
            for (int64_t j = 0; j <= seqlen; ++j)
                buf.push_back(tokens[start + j]);
        }

        return torch::tensor(buf, torch::kLong).view({B, seqlen + 1});
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Args
// ─────────────────────────────────────────────────────────────────────────────
struct Args {
    std::string input;
    int64_t dim        = 256;
    int64_t layers     = 4;
    int64_t heads      = 4;
    int64_t kvheads    = 2;
    int64_t ffn        = 512;
    int64_t seqlen     = 64;
    int64_t epochs     = 20;
    int64_t batch      = 32;
    double  lr         = 3e-4;
    int64_t warmup     = 200;
    int64_t maxiter    = 5000;
    std::string save   = "llama3_hust.pt";
    bool    cuda       = false;
    bool    benchmark  = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "-input"   && i+1<argc) a.input   = argv[++i];
        else if (k == "-dim"     && i+1<argc) a.dim      = std::stoll(argv[++i]);
        else if (k == "-layers"  && i+1<argc) a.layers   = std::stoll(argv[++i]);
        else if (k == "-heads"   && i+1<argc) a.heads    = std::stoll(argv[++i]);
        else if (k == "-kvheads" && i+1<argc) a.kvheads  = std::stoll(argv[++i]);
        else if (k == "-ffn"     && i+1<argc) a.ffn      = std::stoll(argv[++i]);
        else if (k == "-seqlen"  && i+1<argc) a.seqlen   = std::stoll(argv[++i]);
        else if (k == "-epochs"  && i+1<argc) a.epochs   = std::stoll(argv[++i]);
        else if (k == "-batch"   && i+1<argc) a.batch    = std::stoll(argv[++i]);
        else if (k == "-lr"      && i+1<argc) a.lr       = std::stod(argv[++i]);
        else if (k == "-warmup"  && i+1<argc) a.warmup   = std::stoll(argv[++i]);
        else if (k == "-maxiter" && i+1<argc) a.maxiter  = std::stoll(argv[++i]);
        else if (k == "-save"    && i+1<argc) a.save     = argv[++i];
        else if (k == "--cuda")               a.cuda     = true;
        else if (k == "--benchmark")          a.benchmark = true;
    }
    return a;
}

int main(int argc, char* argv[]) {
    auto a = parse_args(argc, argv);

    if (a.input.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " -input <tokens_file> [options]\n";
        return 1;
    }

    // ── Load dataset ─────────────────────────────────────────────────────────
    auto t_load_start = std::chrono::steady_clock::now();
    HUSTDataset dataset;
    if (!dataset.load(a.input)) {
        std::cerr << "Failed to load token file\n";
        return 1;
    }
    auto t_load_end = std::chrono::steady_clock::now();
    double load_sec = std::chrono::duration<double>(t_load_end - t_load_start).count();

    std::printf("=== HUST-tokenized dataset loaded ===\n");
    std::printf("  Tokens  : %zu\n",   dataset.tokens.size());
    std::printf("  Vocab   : %lld\n",  (long long)dataset.vocab_size);
    std::printf("  Load    : %.3f s\n", load_sec);

    // ── Device ───────────────────────────────────────────────────────────────
    torch::Device device = (a.cuda && torch::cuda::is_available())
                         ? torch::kCUDA : torch::kCPU;
    std::printf("  Device  : %s\n\n", device == torch::kCUDA ? "CUDA" : "CPU");

    // ── Build model ──────────────────────────────────────────────────────────
    Llama3Config cfg;
    cfg.dim        = a.dim;
    cfg.ffn_dim    = a.ffn;
    cfg.n_layers   = a.layers;
    cfg.n_heads    = a.heads;
    cfg.n_kv_heads = a.kvheads;
    cfg.vocab_size = dataset.vocab_size;  // use actual HUST vocab size
    cfg.seq_len    = a.seqlen;
    cfg.dropout    = 0.1f;

    auto model = Llama3Model(cfg);
    model->to(device);

    int64_t n_params = 0;
    for (auto& p : model->parameters()) n_params += p.numel();

    std::printf("=== Llama 3 (HUST vocab) ===\n");
    std::printf("  Architecture : layers=%lld  dim=%lld  heads=%lld  kv_heads=%lld  ffn=%lld\n",
                (long long)cfg.n_layers, (long long)cfg.dim,
                (long long)cfg.n_heads,  (long long)cfg.n_kv_heads,
                (long long)cfg.ffn_dim);
    std::printf("  Vocab        : %lld\n",  (long long)cfg.vocab_size);
    std::printf("  SeqLen       : %lld\n",  (long long)cfg.seq_len);
    std::printf("  Params       : %.3fM\n", n_params / 1e6);

    // ── Optimizer ────────────────────────────────────────────────────────────
    Llama3TrainConfig tcfg;
    tcfg.lr          = a.lr;
    tcfg.min_lr      = a.lr / 10.0;
    tcfg.warmup_iters= a.warmup;
    tcfg.max_iters   = a.maxiter;
    tcfg.device      = device;
    auto opt = make_llama3_optimizer(model, tcfg);

    // ── Training loop ────────────────────────────────────────────────────────
    std::printf("\n=== Training ===\n");
    std::printf("  Epochs  : %lld\n",  (long long)a.epochs);
    std::printf("  Batch   : %lld\n",  (long long)a.batch);
    std::printf("  MaxIter : %lld\n",  (long long)a.maxiter);
    std::printf("  LR peak : %.2e  floor: %.2e\n\n", a.lr, a.lr/10.0);
    std::fflush(stdout);

    std::mt19937 rng(42);
    double best_loss = 1e9;
    int64_t step     = 0;
    double loss_sum  = 0.0;
    int64_t loss_cnt = 0;
    auto t_train_start = std::chrono::steady_clock::now();

    model->train();

    for (int64_t epoch = 0; epoch < a.epochs && step < a.maxiter; ++epoch) {
        // Estimate steps per epoch based on dataset size
        int64_t steps_per_epoch = std::max((int64_t)1,
            (int64_t)dataset.tokens.size() / (a.batch * a.seqlen));
        steps_per_epoch = std::min(steps_per_epoch, a.maxiter - step);

        for (int64_t si = 0; si < steps_per_epoch && step < a.maxiter; ++si, ++step) {
            auto tokens_batch = dataset.sample_batch(a.batch, a.seqlen, rng);
            tokens_batch = tokens_batch.to(device);

            // LR update
            double lr_now = llama3_lr_schedule(step, tcfg);
            for (auto& pg : opt.param_groups())
                static_cast<torch::optim::AdamWOptions&>(pg.options()).lr(lr_now);

            opt.zero_grad();
            auto inp    = tokens_batch.slice(1, 0, a.seqlen);
            auto tgt    = tokens_batch.slice(1, 1, a.seqlen + 1);
            model->forward(inp, tgt);
            auto loss = model->last_loss;
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), tcfg.grad_clip);
            opt.step();

            double lv = loss.item<double>();
            loss_sum += lv;
            ++loss_cnt;

            if (step % 50 == 0) {
                double avg_loss = loss_sum / loss_cnt;
                double bpc      = avg_loss / std::log(2.0);
                double ppl      = std::exp(avg_loss);
                auto t_now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(t_now - t_train_start).count();
                std::printf("epoch %2lld  step %5lld  loss %.4f  bpc %.4f  ppl %7.2f  lr %.2e  t %.1fs\n",
                    (long long)epoch, (long long)step,
                    avg_loss, bpc, ppl, lr_now, elapsed);
                std::fflush(stdout);
                loss_sum = 0.0;
                loss_cnt = 0;
            }

            if (lv < best_loss) {
                best_loss = lv;
                torch::save(model, a.save);
            }
        }
    }

    auto t_train_end = std::chrono::steady_clock::now();
    double train_sec = std::chrono::duration<double>(t_train_end - t_train_start).count();

    // ── Final eval ───────────────────────────────────────────────────────────
    model->eval();
    torch::NoGradGuard ng;
    double eval_loss_sum = 0.0;
    int64_t eval_steps   = 50;
    for (int64_t i = 0; i < eval_steps; ++i) {
        auto tb = dataset.sample_batch(a.batch, a.seqlen, rng).to(device);
        model->forward(tb.slice(1, 0, a.seqlen), tb.slice(1, 1, a.seqlen + 1));
        eval_loss_sum += model->last_loss.item<double>();
    }
    double eval_loss = eval_loss_sum / eval_steps;

    std::printf("\n=== Results ===\n");
    std::printf("  Train time     : %.2f s\n", train_sec);
    std::printf("  Best train loss: %.4f\n",   best_loss);
    std::printf("  Eval loss      : %.4f\n",   eval_loss);
    std::printf("  Eval BPC       : %.4f\n",   eval_loss / std::log(2.0));
    std::printf("  Eval PPL       : %.2f\n",   std::exp(eval_loss));
    std::printf("  Checkpoint     : %s\n",     a.save.c_str());

    if (a.benchmark) {
        double tokens_sec = (double)(step * a.batch * a.seqlen) / train_sec;
        std::printf("\n=== Benchmark ===\n");
        std::printf("  Steps          : %lld\n",   (long long)step);
        std::printf("  Tokens/sec     : %.0f\n",   tokens_sec);
        std::printf("  Params         : %.3fM\n",  n_params / 1e6);
    }

    return 0;
}
