// ─────────────────────────────────────────────────────────────────────────────
// BitNet a4.8 training entry-point  (Wang et al., arXiv:2411.04965v1)
//
// 2-stage training recipe (§2.2):
//   Stage 1: INT8 activations + ReLU²GLU  (95B tokens)
//   Stage 2: 4-bit + sparse activations    (5B tokens)
//
// This binary implements stage 2 training on a user-supplied token file
// (raw int32 little-endian, as produced by the tokeniser).
//
// Usage:
//   bitnet_a4_8_train --data <tokens.bin> --size 700m|1b3|3b|7b \
//                     [--steps N] [--batch B] [--lr LR] [--device cpu|cuda]
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/bitnet_a4_8/bitnet_a4_8.h"

#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <random>

using namespace dm::models::nlp;

// ─────────────────────────────────────────────────────────────────────────────
// Simple argument parsing
// ─────────────────────────────────────────────────────────────────────────────
struct Args {
    std::string data_path;
    std::string size     = "700m";
    int64_t     steps    = 10000;
    int64_t     batch    = 4;
    double      lr       = 1.5e-3;
    std::string device   = "cpu";
    int64_t     seq_len  = 2048;
    int64_t     log_every= 100;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc - 1; ++i) {
        std::string k = argv[i];
        std::string v = argv[i + 1];
        if (k == "--data")     { a.data_path = v; ++i; }
        else if (k == "--size")     { a.size      = v; ++i; }
        else if (k == "--steps")    { a.steps     = std::stoll(v); ++i; }
        else if (k == "--batch")    { a.batch     = std::stoll(v); ++i; }
        else if (k == "--lr")       { a.lr        = std::stod(v); ++i; }
        else if (k == "--device")   { a.device    = v; ++i; }
        else if (k == "--seq_len")  { a.seq_len   = std::stoll(v); ++i; }
        else if (k == "--log_every"){ a.log_every = std::stoll(v); ++i; }
    }
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────
// Load a flat int32 token file into a 1-D int64 tensor
// ─────────────────────────────────────────────────────────────────────────────
static torch::Tensor load_tokens(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open token file: " + path);
    std::streamsize bytes = f.tellg();
    f.seekg(0);
    int64_t n = bytes / sizeof(int32_t);
    std::vector<int32_t> buf(n);
    f.read(reinterpret_cast<char*>(buf.data()), bytes);

    // Convert to int64
    std::vector<int64_t> buf64(buf.begin(), buf.end());
    return torch::tensor(buf64, torch::kLong);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sample a random batch: [B, seq_len+1]
// ─────────────────────────────────────────────────────────────────────────────
static std::pair<torch::Tensor, torch::Tensor> sample_batch(
        const torch::Tensor& data, int64_t B, int64_t seq_len,
        std::mt19937& rng) {
    int64_t N = data.size(0);
    std::uniform_int_distribution<int64_t> dist(0, N - seq_len - 2);

    std::vector<int64_t> idxs;
    idxs.reserve(B * (seq_len + 1));
    for (int64_t b = 0; b < B; ++b) {
        int64_t start = dist(rng);
        for (int64_t t = 0; t <= seq_len; ++t)
            idxs.push_back(data[start + t].item<int64_t>());
    }

    auto chunk  = torch::tensor(idxs, torch::kLong).view({B, seq_len + 1});
    auto inp    = chunk.slice(1, 0, seq_len);    // [B, seq_len]
    auto tgt    = chunk.slice(1, 1, seq_len + 1); // [B, seq_len]
    return {inp, tgt};
}

// ─────────────────────────────────────────────────────────────────────────────
// Cosine LR with warmup (Table 7: warmup=375 steps)
// ─────────────────────────────────────────────────────────────────────────────
static double lr_schedule(int64_t step, int64_t warmup, int64_t total,
                           double lr_start, double lr_end) {
    if (step < warmup)
        return lr_start * (double)(step + 1) / (double)warmup;
    double t = (double)(step - warmup) / (double)(total - warmup);
    return lr_end + 0.5 * (lr_start - lr_end) * (1.0 + std::cos(M_PI * t));
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    // Device
    torch::Device device = (args.device == "cuda" && torch::cuda::is_available())
                         ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << (device == torch::kCUDA ? "cuda" : "cpu") << "\n";

    // Model
    BitNetA48Config cfg;
    if      (args.size == "700m") cfg = BitNetA48Config::size_700m();
    else if (args.size == "1b3")  cfg = BitNetA48Config::size_1b3();
    else if (args.size == "3b")   cfg = BitNetA48Config::size_3b();
    else if (args.size == "7b")   cfg = BitNetA48Config::size_7b();
    else { std::cerr << "Unknown size: " << args.size << "\n"; return 1; }

    cfg.max_seq_len = args.seq_len;
    auto model = BitNetA48Model(cfg);
    model->to(device);
    std::cout << "Model: BitNet a4.8 " << args.size << "\n";

    // Optimizer (Table 7: AdamW β=(0.9,0.95))
    auto adam_opts = torch::optim::AdamWOptions(args.lr)
                         .betas({0.9, 0.95})
                         .weight_decay(0.1);
    auto optimizer = torch::optim::AdamW(model->parameters(), adam_opts);

    // Data
    if (args.data_path.empty()) {
        std::cerr << "No --data provided. Use --data <tokens.bin>\n";
        std::cerr << "Running synthetic smoke-test for 10 steps...\n";

        // Synthetic smoke test
        for (int64_t step = 0; step < 10; ++step) {
            auto tokens_in  = torch::randint(0, cfg.vocab_size, {args.batch, args.seq_len});
            auto tokens_tgt = torch::randint(0, cfg.vocab_size, {args.batch, args.seq_len});
            tokens_in  = tokens_in.to(device);
            tokens_tgt = tokens_tgt.to(device);

            double lr_now = lr_schedule(step, 375, args.steps, args.lr, args.lr * 0.1);
            // Set absolute lr (train_step applies scale=1.0)
            for (auto& pg : optimizer.param_groups()) {
                static_cast<torch::optim::AdamWOptions&>(pg.options()).lr(lr_now);
            }
            float loss = bitnet_a48_train_step(model, optimizer, tokens_in, tokens_tgt);
            std::cout << "step " << step << "  loss=" << loss << "\n";
        }
        return 0;
    }

    std::cout << "Loading data: " << args.data_path << "\n";
    auto data = load_tokens(args.data_path);
    std::cout << "Tokens: " << data.size(0) << "\n";

    std::mt19937 rng(42);
    float  smooth_loss = -1.0f;
    const  double alpha = 0.99;

    for (int64_t step = 0; step < args.steps; ++step) {
        auto [inp, tgt] = sample_batch(data, args.batch, args.seq_len, rng);
        inp = inp.to(device);
        tgt = tgt.to(device);

        double lr_now = lr_schedule(step, 375, args.steps, args.lr, args.lr * 0.1);
        for (auto& pg : optimizer.param_groups())
            static_cast<torch::optim::AdamWOptions&>(pg.options()).lr(lr_now);

        float loss = bitnet_a48_train_step(model, optimizer, inp, tgt);
        smooth_loss = (smooth_loss < 0) ? loss : (float)(alpha * smooth_loss + (1.0 - alpha) * loss);

        if (step % args.log_every == 0) {
            std::printf("step %6ld | lr=%.2e | loss=%.4f | smooth=%.4f\n",
                        (long)step, lr_now, loss, smooth_loss);
            std::fflush(stdout);
        }
    }

    std::cout << "Training complete.\n";
    return 0;
}
