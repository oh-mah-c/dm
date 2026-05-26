// ─────────────────────────────────────────────────────────────────────────────
// RWKV training entry point — Peng et al., arXiv:2305.13048v2, 2023
//
// Usage:
//   ./rwkv_train [options]
//   --size    <tiny|169m|430m|1p5b|3b|7b|14b>  (default: tiny)
//   --layers  <int>     override n_layers
//   --dim     <int>     override d_model
//   --ctx     <int>     context / sequence length   (default: 1024)
//   --seqlen  <int>     training segment length     (default: 128)
//   --epochs  <int>                                 (default: 10)
//   --batch   <int>     batch size                  (default: 32)
//   --lr      <float>   initial learning rate       (default: 6e-4)
//   --endlr   <float>   final LR after exp decay    (default: 1e-5)
//   --warmup  <int>     warmup steps                (default: 0)
//   --maxiter <int>     total training steps        (default: 4800)
//   --data    <path>    dataset root                (default: ./data)
//   --save    <path>    checkpoint path             (default: rwkv_best.pt)
//   --cuda
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/rwkv/rwkv.h"

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <cmath>

using namespace dm::models::nlp;

struct Args {
    std::string size    = "tiny";
    int64_t     layers  = 0;
    int64_t     dim     = 0;
    int64_t     ctx     = 0;
    int64_t     seqlen  = 128;
    int64_t     epochs  = 10;
    int64_t     batch   = 32;
    double      lr      = 6e-4;
    double      endlr   = 1e-5;
    int64_t     warmup  = 0;
    int64_t     maxiter = 4800;
    std::string data    = "./data";
    std::string save    = "rwkv_best.pt";
    bool        cuda    = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--size"    && i+1<argc) a.size    = argv[++i];
        else if (k == "--layers"  && i+1<argc) a.layers  = std::stoll(argv[++i]);
        else if (k == "--dim"     && i+1<argc) a.dim     = std::stoll(argv[++i]);
        else if (k == "--ctx"     && i+1<argc) a.ctx     = std::stoll(argv[++i]);
        else if (k == "--seqlen"  && i+1<argc) a.seqlen  = std::stoll(argv[++i]);
        else if (k == "--epochs"  && i+1<argc) a.epochs  = std::stoll(argv[++i]);
        else if (k == "--batch"   && i+1<argc) a.batch   = std::stoll(argv[++i]);
        else if (k == "--lr"      && i+1<argc) a.lr      = std::stod(argv[++i]);
        else if (k == "--endlr"   && i+1<argc) a.endlr   = std::stod(argv[++i]);
        else if (k == "--warmup"  && i+1<argc) a.warmup  = std::stoll(argv[++i]);
        else if (k == "--maxiter" && i+1<argc) a.maxiter = std::stoll(argv[++i]);
        else if (k == "--data"    && i+1<argc) a.data    = argv[++i];
        else if (k == "--save"    && i+1<argc) a.save    = argv[++i];
        else if (k == "--cuda")                a.cuda    = true;
    }
    return a;
}

static RWKVConfig build_config(const Args& a) {
    RWKVConfig cfg;
    if      (a.size == "169m")  cfg = RWKVConfig::r169m();
    else if (a.size == "430m")  cfg = RWKVConfig::r430m();
    else if (a.size == "1p5b")  cfg = RWKVConfig::r1p5b();
    else if (a.size == "3b")    cfg = RWKVConfig::r3b();
    else if (a.size == "7b")    cfg = RWKVConfig::r7b();
    else if (a.size == "14b")   cfg = RWKVConfig::r14b();
    else                        cfg = RWKVConfig::tiny();

    if (a.layers > 0) cfg.n_layers = a.layers;
    if (a.dim    > 0) cfg.d_model  = a.dim;
    if (a.ctx    > 0) cfg.ctx_len  = a.ctx;
    return cfg;
}

int main(int argc, char* argv[]) {
    auto a   = parse_args(argc, argv);
    auto cfg = build_config(a);

    torch::Device device = a.cuda && torch::cuda::is_available()
                         ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << device << "\n";

    auto model = RWKVModel(cfg);
    model->to(device);

    int64_t n_params = 0;
    for (auto& item : model->named_parameters())
        n_params += item.value().numel();
    std::cout << "Params:  " << n_params / 1e6 << "M\n";
    std::cout << "n_layers:" << cfg.n_layers
              << "  d_model:" << cfg.d_model
              << "  d_ff:"    << cfg.d_ff()
              << "  ctx_len:" << cfg.ctx_len << "\n";

    RWKVTrainConfig tcfg;
    tcfg.lr          = a.lr;
    tcfg.end_lr      = a.endlr;
    tcfg.warmup      = a.warmup;
    tcfg.total_steps = a.maxiter;

    auto opt = make_rwkv_optimizer(model, tcfg);

    double  best_loss = 1e9;
    int64_t step      = 0;

    for (int64_t epoch = 0; epoch < a.epochs && step < a.maxiter; ++epoch) {
        // Simulate batches; replace with real data loader for real training
        for (int64_t bi = 0; bi < 100 && step < a.maxiter; ++bi, ++step) {
            auto tokens = torch::randint(
                0, cfg.vocab_size,
                {a.batch, a.seqlen + 1},
                torch::TensorOptions().dtype(torch::kLong).device(device));

            auto result = rwkv_train_step(model, opt, tokens, tcfg, step);
            double loss = result.second;

            if (step % 20 == 0) {
                std::printf("epoch %lld  step %lld  loss %.4f  bpc %.4f  lr %.2e\n",
                    static_cast<long long>(epoch),
                    static_cast<long long>(step),
                    loss,
                    loss / std::log(2.0),
                    rwkv_lr_schedule(step, tcfg));
                std::fflush(stdout);
            }

            if (loss < best_loss) {
                best_loss = loss;
                torch::save(model, a.save);
            }
        }
    }

    std::printf("Training complete. Best loss: %.4f  Saved: %s\n",
                best_loss, a.save.c_str());
    return 0;
}
