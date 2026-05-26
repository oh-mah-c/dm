// ─────────────────────────────────────────────────────────────────────────────
// Mamba training entry point — Gu & Dao, arXiv:2312.00752v2, 2023
//
// Usage:
//   ./mamba_train [options]
//   --size    <tiny|130m|370m|790m|1p4b>  (default: tiny)
//   --layers  <int>     override n_layers
//   --dim     <int>     override d_model
//   --dstate  <int>     SSM state dim N           (default: 16)
//   --dconv   <int>     depthwise conv kernel     (default: 4)
//   --expand  <int>     channel expansion factor  (default: 2)
//   --seqlen  <int>     sequence length           (default: 128)
//   --epochs  <int>                               (default: 10)
//   --batch   <int>     batch size                (default: 32)
//   --lr      <float>   peak learning rate        (default: 6e-4)
//   --minlr   <float>   cosine decay floor        (default: 1e-5)
//   --clip    <float>   gradient clip norm        (default: 1.0)
//   --warmup  <int>     warmup steps              (default: 0)
//   --maxiter <int>     total training steps      (default: 4800)
//   --data    <path>    dataset root              (default: ./data)
//   --save    <path>    checkpoint path           (default: mamba_best.pt)
//   --cuda
// ─────────────────────────────────────────────────────────────────────────────

#include "models/nlp/mamba/mamba.h"

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <cmath>

using namespace dm::models::nlp;

struct Args {
    std::string size     = "tiny";
    int64_t     layers   = 0;
    int64_t     dim      = 0;
    int64_t     dstate   = 0;
    int64_t     dconv    = 0;
    int64_t     expand   = 0;
    int64_t     seqlen   = 128;
    int64_t     epochs   = 10;
    int64_t     batch    = 32;
    double      lr       = 6e-4;
    double      minlr    = 1e-5;
    double      clip     = 1.0;
    int64_t     warmup   = 0;
    int64_t     maxiter  = 4800;
    std::string data     = "./data";
    std::string save     = "mamba_best.pt";
    bool        cuda     = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--size"    && i+1<argc) a.size    = argv[++i];
        else if (k == "--layers"  && i+1<argc) a.layers  = std::stoll(argv[++i]);
        else if (k == "--dim"     && i+1<argc) a.dim     = std::stoll(argv[++i]);
        else if (k == "--dstate"  && i+1<argc) a.dstate  = std::stoll(argv[++i]);
        else if (k == "--dconv"   && i+1<argc) a.dconv   = std::stoll(argv[++i]);
        else if (k == "--expand"  && i+1<argc) a.expand  = std::stoll(argv[++i]);
        else if (k == "--seqlen"  && i+1<argc) a.seqlen  = std::stoll(argv[++i]);
        else if (k == "--epochs"  && i+1<argc) a.epochs  = std::stoll(argv[++i]);
        else if (k == "--batch"   && i+1<argc) a.batch   = std::stoll(argv[++i]);
        else if (k == "--lr"      && i+1<argc) a.lr      = std::stod(argv[++i]);
        else if (k == "--minlr"   && i+1<argc) a.minlr   = std::stod(argv[++i]);
        else if (k == "--clip"    && i+1<argc) a.clip    = std::stod(argv[++i]);
        else if (k == "--warmup"  && i+1<argc) a.warmup  = std::stoll(argv[++i]);
        else if (k == "--maxiter" && i+1<argc) a.maxiter = std::stoll(argv[++i]);
        else if (k == "--data"    && i+1<argc) a.data    = argv[++i];
        else if (k == "--save"    && i+1<argc) a.save    = argv[++i];
        else if (k == "--cuda")                a.cuda    = true;
    }
    return a;
}

static MambaConfig build_config(const Args& a) {
    MambaConfig cfg;
    if      (a.size == "130m")  cfg = MambaConfig::m130();
    else if (a.size == "370m")  cfg = MambaConfig::m370();
    else if (a.size == "790m")  cfg = MambaConfig::m790();
    else if (a.size == "1p4b")  cfg = MambaConfig::m1p4b();
    else                        cfg = MambaConfig::tiny();

    if (a.layers > 0) cfg.n_layers = a.layers;
    if (a.dim    > 0) cfg.d_model  = a.dim;
    if (a.dstate > 0) cfg.d_state  = a.dstate;
    if (a.dconv  > 0) cfg.d_conv   = a.dconv;
    if (a.expand > 0) cfg.expand   = a.expand;
    return cfg;
}

int main(int argc, char* argv[]) {
    auto a   = parse_args(argc, argv);
    auto cfg = build_config(a);

    torch::Device device = a.cuda && torch::cuda::is_available()
                         ? torch::kCUDA : torch::kCPU;
    std::cout << "Device: " << device << "\n";

    auto model = MambaModel(cfg);
    model->to(device);

    int64_t n_params = 0;
    for (auto& item : model->named_parameters())
        n_params += item.value().numel();
    std::cout << "Params:  " << n_params / 1e6 << "M\n";
    std::cout << "n_layers:" << cfg.n_layers
              << "  d_model:" << cfg.d_model
              << "  d_state:" << cfg.d_state
              << "  d_conv:"  << cfg.d_conv
              << "  expand:"  << cfg.expand
              << "  d_inner:" << cfg.d_inner()
              << "  dt_rank:" << cfg.dt_rank() << "\n";

    MambaTrainConfig tcfg;
    tcfg.lr          = a.lr;
    tcfg.min_lr      = a.minlr;
    tcfg.clip        = a.clip;
    tcfg.warmup      = a.warmup;
    tcfg.total_steps = a.maxiter;

    auto opt = make_mamba_optimizer(model, tcfg);

    double best_loss = 1e9;
    int64_t step     = 0;

    for (int64_t epoch = 0; epoch < a.epochs && step < a.maxiter; ++epoch) {
        // Simulate batches; replace with real data loader for real training
        for (int64_t bi = 0; bi < 100 && step < a.maxiter; ++bi, ++step) {
            auto tokens = torch::randint(
                0, cfg.vocab_size,
                {a.batch, a.seqlen + 1},
                torch::TensorOptions().dtype(torch::kLong).device(device));

            auto result = mamba_train_step(model, opt, tokens, tcfg, step);
            double loss = result.second;

            if (step % 20 == 0) {
                std::printf("epoch %lld  step %lld  loss %.4f  bpc %.4f  lr %.2e\n",
                    static_cast<long long>(epoch),
                    static_cast<long long>(step),
                    loss,
                    loss / std::log(2.0),
                    mamba_lr_schedule(step, tcfg));
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
