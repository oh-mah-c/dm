// ─────────────────────────────────────────────────────────────────────────────
// TRM training entry point
//
// Usage:
//   ./trm_train [options]
//   --variant  att|mlp        (default: att)
//   --epochs   <int>          (default: 60000 — paper Section 6)
//   --lr       <float>        (default: 1e-4)
//   --batch    <int>          (default: 64 — reduced from paper's 768)
//   --vocab    <int>          (default: 16  — digits 0-9 + pad + special)
//   --seqlen   <int>          (default: 81  — 9×9 Sudoku grid)
//   --dim      <int>          (default: 512 — paper Section 6)
//   --layers   <int>          (default: 2   — paper Section 4.4)
//   --heads    <int>          (default: 8)
//   --n        <int>          (default: 6   — recursions per deep step)
//   --T        <int>          (default: 3   — supervision depth)
//   --nsup     <int>          (default: 16  — max supervision steps)
//   --save     <path>         (default: trm_best.pt)
//   --cuda
// ─────────────────────────────────────────────────────────────────────────────

#include "models/reasoning/trm/trm.h"

#include <torch/torch.h>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace dm::models::reasoning;

struct Args {
    std::string variant   = "att";
    int64_t  epochs       = 60000;
    double   lr           = 1e-4;
    int64_t  batch_size   = 64;
    int64_t  vocab        = 16;
    int64_t  seqlen       = 81;
    int64_t  dim          = 512;
    int64_t  layers       = 2;
    int64_t  heads        = 8;
    int64_t  n            = 6;
    int64_t  T            = 3;
    int64_t  nsup         = 16;
    std::string save_path = "trm_best.pt";
    bool     cuda         = false;
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--variant" && i+1<argc) a.variant    = argv[++i];
        else if (k == "--epochs"  && i+1<argc) a.epochs     = std::stoll(argv[++i]);
        else if (k == "--lr"      && i+1<argc) a.lr         = std::stod(argv[++i]);
        else if (k == "--batch"   && i+1<argc) a.batch_size = std::stoll(argv[++i]);
        else if (k == "--vocab"   && i+1<argc) a.vocab      = std::stoll(argv[++i]);
        else if (k == "--seqlen"  && i+1<argc) a.seqlen     = std::stoll(argv[++i]);
        else if (k == "--dim"     && i+1<argc) a.dim        = std::stoll(argv[++i]);
        else if (k == "--layers"  && i+1<argc) a.layers     = std::stoll(argv[++i]);
        else if (k == "--heads"   && i+1<argc) a.heads      = std::stoll(argv[++i]);
        else if (k == "--n"       && i+1<argc) a.n          = std::stoll(argv[++i]);
        else if (k == "--T"       && i+1<argc) a.T          = std::stoll(argv[++i]);
        else if (k == "--nsup"    && i+1<argc) a.nsup       = std::stoll(argv[++i]);
        else if (k == "--save"    && i+1<argc) a.save_path  = argv[++i];
        else if (k == "--cuda")                a.cuda       = true;
    }
    return a;
}

int main(int argc, char* argv[]) {
    auto args   = parse_args(argc, argv);
    auto device = (args.cuda && torch::cuda::is_available())
                  ? torch::kCUDA : torch::kCPU;
    std::cout << "[TRM] device: " << device << "\n";

    bool use_att = (args.variant == "att");
    if (args.variant != "att" && args.variant != "mlp")
        throw std::invalid_argument("Unknown variant: " + args.variant);

    auto model = TRM(args.vocab, args.seqlen, args.dim, args.layers, args.heads,
                     use_att, args.n, args.T);
    model->to(device);

    int64_t n_params = 0;
    for (auto& p : model->parameters()) n_params += p.numel();
    std::cout << "[TRM] variant=" << args.variant
              << "  vocab=" << args.vocab
              << "  seqlen=" << args.seqlen
              << "  params=" << n_params / 1'000'000 << "M\n";

    // Synthetic demonstration dataset: random token sequences
    // Replace with real puzzle loader (Sudoku, Maze, ARC-AGI) for actual training
    std::cout << "[TRM] generating synthetic demo data...\n";
    auto make_batch = [&](int64_t n) {
        auto x = torch::randint(0, args.vocab, {n, args.seqlen},
                     torch::TensorOptions().dtype(torch::kLong));
        auto y = torch::randint(0, args.vocab, {n, args.seqlen},
                     torch::TensorOptions().dtype(torch::kLong));
        return std::make_pair(x, y);
    };

    std::vector<std::pair<torch::Tensor,torch::Tensor>> train_batches, val_batches;
    for (int i = 0; i < 4; ++i) train_batches.push_back(make_batch(args.batch_size));
    for (int i = 0; i < 2; ++i) val_batches.push_back(make_batch(args.batch_size));

    TRMTrainConfig cfg;
    cfg.lr          = args.lr;
    cfg.batch_size  = args.batch_size;
    cfg.max_epochs  = args.epochs;
    cfg.n_sup       = args.nsup;
    cfg.device      = device;

    trm_train(model, cfg, train_batches, val_batches, args.save_path);
    return 0;
}
