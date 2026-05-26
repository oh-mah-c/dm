// ─────────────────────────────────────────────────────────────────────────────
// Llama 3 training entry point — Llama Team, AI @ Meta, arXiv:2407.21783v3
//
// Usage:
//   ./llama3_train [options]
//   --size   <tiny|8b|70b|405b>   model preset      (default: tiny)
//   --seqlen <int>                context window     (default: from preset)
//   --epochs <int>                                   (default: 5)
//   --batch  <int>                batch size         (default: 4)
//   --lr     <float>              peak LR            (default: 3e-4)
//   --minlr  <float>              cosine floor LR    (default: 1e-5)
//   --warmup <int>                warmup steps       (default: 100)
//   --maxiter<int>                total steps        (default: 2000)
//   --save   <path>               checkpoint path    (default: llama3_best.pt)
//   --cuda
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/llama3/llama3.h"

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <cmath>

using namespace dm::models::nlp;

struct Args {
    std::string size    = "tiny";
    int64_t     seqlen  = -1;    // -1 means use preset default
    int64_t     epochs  = 5;
    int64_t     batch   = 4;
    double      lr      = 3e-4;
    double      minlr   = 1e-5;
    int64_t     warmup  = 100;
    int64_t     maxiter = 2000;
    std::string save    = "llama3_best.pt";
    bool        cuda    = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--size"    && i+1<argc) a.size    = argv[++i];
        else if (k == "--seqlen"  && i+1<argc) a.seqlen  = std::stoll(argv[++i]);
        else if (k == "--epochs"  && i+1<argc) a.epochs  = std::stoll(argv[++i]);
        else if (k == "--batch"   && i+1<argc) a.batch   = std::stoll(argv[++i]);
        else if (k == "--lr"      && i+1<argc) a.lr      = std::stod(argv[++i]);
        else if (k == "--minlr"   && i+1<argc) a.minlr   = std::stod(argv[++i]);
        else if (k == "--warmup"  && i+1<argc) a.warmup  = std::stoll(argv[++i]);
        else if (k == "--maxiter" && i+1<argc) a.maxiter = std::stoll(argv[++i]);
        else if (k == "--save"    && i+1<argc) a.save    = argv[++i];
        else if (k == "--cuda")                a.cuda    = true;
    }
    return a;
}

int main(int argc, char* argv[]) {
    auto a = parse_args(argc, argv);

    torch::Device device = (a.cuda && torch::cuda::is_available())
                         ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << device << "\n";

    // ── Build model ──────────────────────────────────────────────────────────
    Llama3Config cfg;
    if      (a.size == "8b"  ) cfg = Llama3Config::llama3_8b();
    else if (a.size == "70b" ) cfg = Llama3Config::llama3_70b();
    else if (a.size == "405b") cfg = Llama3Config::llama3_405b();
    else                        cfg = Llama3Config::tiny();

    if (a.seqlen > 0) cfg.seq_len = a.seqlen;

    auto model = Llama3Model(cfg);
    model->to(device);

    // Parameter count
    int64_t total = 0;
    for (auto& p : model->parameters()) total += p.numel();
    std::printf("Model: Llama 3 %-5s  |  params: %.3fM\n",
                a.size.c_str(), total / 1e6);
    std::printf("Architecture: layers=%lld  dim=%lld  n_heads=%lld  n_kv_heads=%lld  ffn=%lld\n",
                (long long)cfg.n_layers, (long long)cfg.dim,
                (long long)cfg.n_heads,  (long long)cfg.n_kv_heads,
                (long long)cfg.ffn_dim);
    std::printf("Vocab: %lld  SeqLen: %lld  RoPE theta: %.0f\n",
                (long long)cfg.vocab_size, (long long)cfg.seq_len,
                (double)cfg.rope_theta);

    // ── Optimizer ────────────────────────────────────────────────────────────
    Llama3TrainConfig tcfg;
    tcfg.lr          = a.lr;
    tcfg.min_lr      = a.minlr;
    tcfg.warmup_iters= a.warmup;
    tcfg.max_iters   = a.maxiter;
    tcfg.device      = device;

    auto opt = make_llama3_optimizer(model, tcfg);

    // ── Training loop ────────────────────────────────────────────────────────
    double best_loss = 1e9;
    int64_t step     = 0;

    model->train();

    for (int64_t epoch = 0; epoch < a.epochs && step < a.maxiter; ++epoch) {
        for (int64_t bi = 0; bi < 100 && step < a.maxiter; ++bi, ++step) {
            // Synthetic random tokens [B, T+1]
            auto tokens = torch::randint(
                0, cfg.vocab_size,
                {a.batch, cfg.seq_len + 1},
                torch::TensorOptions().dtype(torch::kLong).device(device));

            double lr_now = llama3_lr_schedule(step, tcfg);
            for (auto& pg : opt.param_groups())
                static_cast<torch::optim::AdamWOptions&>(pg.options()).lr(lr_now);

            opt.zero_grad();

            auto inp    = tokens.slice(1, 0, cfg.seq_len);
            auto tgt    = tokens.slice(1, 1, cfg.seq_len + 1);
            auto logits = model->forward(inp, tgt);
            auto loss   = model->last_loss;
            loss.backward();

            torch::nn::utils::clip_grad_norm_(model->parameters(), tcfg.grad_clip);
            opt.step();

            double loss_val = loss.item<double>();

            if (step % 20 == 0) {
                std::printf("epoch %lld  step %lld  loss %.4f  bpc %.4f  lr %.2e\n",
                    (long long)epoch, (long long)step,
                    loss_val, loss_val / std::log(2.0), lr_now);
                std::fflush(stdout);
            }

            if (loss_val < best_loss) {
                best_loss = loss_val;
                torch::save(model, a.save);
            }
        }
    }

    std::printf("Training complete.  Best loss: %.4f  Saved → %s\n",
                best_loss, a.save.c_str());
    return 0;
}
