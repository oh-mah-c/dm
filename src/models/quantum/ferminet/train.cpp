// ─────────────────────────────────────────────────────────────────────────────
// FermiNet VMC training entry point
//
// Usage:
//   ./ferminet_train [options]
//
//   --system   h|he|h2|lih|c          (default: h2)
//   --bond     <float>                 bond length in bohr for H₂ (default: 1.401)
//   --n-det    <int>                   number of Slater determinants (default: 16)
//   --dim-1e   <int>                   single-electron stream dim (default: 256)
//   --dim-2e   <int>                   two-electron stream dim (default: 32)
//   --layers   <int>                   interaction layers L (default: 4)
//   --walkers  <int>                   MCMC walkers (default: 2000)
//   --steps    <int>                   VMC optimisation steps (default: 200000)
//   --batch    <int>                   walkers per gradient step (default: 4096)
//   --pretrain <int>                   pretraining steps (default: 0)
//   --cuda                             use CUDA if available
//   --save     <path>                  checkpoint path (default: ferminet_ckpt.pt)
//
// Reference: Pfau, Spencer, Matthews & Foulkes, arXiv:1909.02487v3
// Hyperparameters from Table V.
// ─────────────────────────────────────────────────────────────────────────────

#include "models/quantum/ferminet/ferminet.h"

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <stdexcept>

using namespace dm::models::quantum;

// ── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    std::string system  = "h2";
    double  bond        = 1.401;
    int64_t n_det       = 16;
    int64_t dim_1e      = 256;
    int64_t dim_2e      = 32;
    int64_t n_layers    = 4;
    int64_t n_walkers   = 2000;
    int64_t n_steps     = 200000;
    int64_t batch_size  = 4096;
    int64_t pretrain    = 0;
    bool    cuda        = false;
    std::string save    = "ferminet_ckpt.pt";
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--system"   && i+1 < argc) a.system    = argv[++i];
        else if (s == "--bond"     && i+1 < argc) a.bond      = std::stod(argv[++i]);
        else if (s == "--n-det"    && i+1 < argc) a.n_det     = std::stoll(argv[++i]);
        else if (s == "--dim-1e"   && i+1 < argc) a.dim_1e    = std::stoll(argv[++i]);
        else if (s == "--dim-2e"   && i+1 < argc) a.dim_2e    = std::stoll(argv[++i]);
        else if (s == "--layers"   && i+1 < argc) a.n_layers  = std::stoll(argv[++i]);
        else if (s == "--walkers"  && i+1 < argc) a.n_walkers = std::stoll(argv[++i]);
        else if (s == "--steps"    && i+1 < argc) a.n_steps   = std::stoll(argv[++i]);
        else if (s == "--batch"    && i+1 < argc) a.batch_size= std::stoll(argv[++i]);
        else if (s == "--pretrain" && i+1 < argc) a.pretrain  = std::stoll(argv[++i]);
        else if (s == "--save"     && i+1 < argc) a.save      = argv[++i];
        else if (s == "--cuda")                   a.cuda      = true;
    }
    return a;
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    Args args = parse_args(argc, argv);

    torch::Device device = torch::kCPU;
    if (args.cuda && torch::cuda::is_available()) {
        device = torch::kCUDA;
        std::cout << "Using CUDA.\n";
    }

    // ── Build system ─────────────────────────────────────────────────────────
    FermiNetConfig cfg;
    cfg.n_det      = args.n_det;
    cfg.dim_1e     = args.dim_1e;
    cfg.dim_2e     = args.dim_2e;
    cfg.n_layers   = args.n_layers;
    cfg.n_walkers  = args.n_walkers;
    cfg.batch_size = args.batch_size;

    torch::Tensor npos, ncharge;

    if (args.system == "h") {
        cfg.n_up = 1; cfg.n_down = 0; cfg.n_nuclei = 1;
        npos    = torch::zeros({1, 3});
        ncharge = torch::tensor({1.0f});
    } else if (args.system == "he") {
        cfg.n_up = 1; cfg.n_down = 1; cfg.n_nuclei = 1;
        npos    = torch::zeros({1, 3});
        ncharge = torch::tensor({2.0f});
    } else if (args.system == "h2") {
        cfg.n_up = 1; cfg.n_down = 1; cfg.n_nuclei = 2;
        float d = (float)args.bond / 2;
        npos    = torch::tensor({{-d, 0.f, 0.f}, {d, 0.f, 0.f}});
        ncharge = torch::tensor({1.0f, 1.0f});
    } else if (args.system == "lih") {
        cfg = FermiNetConfig::lih();
        cfg.n_det = args.n_det; cfg.dim_1e = args.dim_1e;
        cfg.dim_2e = args.dim_2e; cfg.n_layers = args.n_layers;
        cfg.n_walkers = args.n_walkers; cfg.batch_size = args.batch_size;
        npos    = torch::tensor({{0.f, 0.f, 0.f}, {3.015f, 0.f, 0.f}});
        ncharge = torch::tensor({3.0f, 1.0f});
    } else if (args.system == "c") {
        cfg = FermiNetConfig::carbon();
        cfg.n_det = args.n_det; cfg.dim_1e = args.dim_1e;
        cfg.dim_2e = args.dim_2e; cfg.n_layers = args.n_layers;
        npos    = torch::zeros({1, 3});
        ncharge = torch::tensor({6.0f});
    } else {
        throw std::invalid_argument("Unknown system: " + args.system);
    }

    std::cout << "FermiNet  system=" << args.system
              << "  n_up="    << cfg.n_up
              << "  n_down="  << cfg.n_down
              << "  n_det="   << cfg.n_det
              << "  dim_1e="  << cfg.dim_1e
              << "  dim_2e="  << cfg.dim_2e
              << "  layers="  << cfg.n_layers << "\n";

    // ── Build trainer ────────────────────────────────────────────────────────
    FermiNetTrainer trainer(cfg, npos, ncharge);
    trainer.model->to(device);

    int64_t n_params = 0;
    for (auto& p : trainer.model->parameters()) n_params += p.numel();
    std::cout << "Trainable parameters: " << n_params << "\n";
    std::cout << "VMC walkers: " << cfg.n_walkers
              << "  batch: "    << cfg.batch_size
              << "  steps: "    << args.n_steps << "\n\n";

    // ── Pretraining ──────────────────────────────────────────────────────────
    if (args.pretrain > 0) {
        trainer.pretrain(args.pretrain);
    }

    // ── VMC training loop ────────────────────────────────────────────────────
    trainer.train(args.n_steps, /*verbose=*/true);

    // Save model
    torch::save(trainer.model, args.save);
    std::cout << "\nCheckpoint saved: " << args.save << "\n";

    return 0;
}
