// ─────────────────────────────────────────────────────────────────────────────
// PauliNet VMC training entry point
//
// Usage:
//   ./paulinet_train [options]
//
//   --system   h|h2|he|lih          (default: h2)
//   --bond     <float>              bond length in bohr for H₂ (default: 1.401)
//   --n-det    <int>                number of Slater determinants (default: 1)
//   --dim-x    <int>                electron feature dim (default: 128)
//   --dim-z    <int>                message dim (default: 64)
//   --layers   <int>                SchNet interaction layers (default: 3)
//   --walkers  <int>                MCMC walkers (default: 2000)
//   --steps    <int>                VMC optimisation steps (default: 7000)
//   --batch    <int>                batch of walkers per gradient step (default: 1000)
//   --lr       <float>              learning rate (default: 0.01)
//   --cuda                          use CUDA if available
//   --save     <path>               checkpoint path (default: paulinet_ckpt.pt)
//
// Reference: Hermann, Schätzle & Noé, arXiv:1909.08423v5, Nature Chemistry 2020
// Hyperparameters from Table 2.
// ─────────────────────────────────────────────────────────────────────────────

#include "models/quantum/paulinet/paulinet.h"

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <stdexcept>

using namespace dm::models::quantum;

// ── CLI ──────────────────────────────────────────────────────────────────────

struct Args {
    std::string system   = "h2";
    double  bond         = 1.401;
    int64_t n_det        = 1;
    int64_t dim_x        = 128;
    int64_t dim_z        = 64;
    int64_t n_layers     = 3;
    int64_t n_walkers    = 2000;
    int64_t n_steps      = 7000;
    int64_t batch_size   = 1000;
    double  lr           = 0.01;
    bool    cuda         = false;
    std::string save     = "paulinet_ckpt.pt";
};

static Args parse_args(int argc, char* argv[]) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if      (s == "--system"  && i+1 < argc) a.system    = argv[++i];
        else if (s == "--bond"    && i+1 < argc) a.bond      = std::stod(argv[++i]);
        else if (s == "--n-det"   && i+1 < argc) a.n_det     = std::stoll(argv[++i]);
        else if (s == "--dim-x"   && i+1 < argc) a.dim_x     = std::stoll(argv[++i]);
        else if (s == "--dim-z"   && i+1 < argc) a.dim_z     = std::stoll(argv[++i]);
        else if (s == "--layers"  && i+1 < argc) a.n_layers  = std::stoll(argv[++i]);
        else if (s == "--walkers" && i+1 < argc) a.n_walkers = std::stoll(argv[++i]);
        else if (s == "--steps"   && i+1 < argc) a.n_steps   = std::stoll(argv[++i]);
        else if (s == "--batch"   && i+1 < argc) a.batch_size= std::stoll(argv[++i]);
        else if (s == "--lr"      && i+1 < argc) a.lr        = std::stod(argv[++i]);
        else if (s == "--save"    && i+1 < argc) a.save      = argv[++i];
        else if (s == "--cuda")                  a.cuda      = true;
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
    PauliNetConfig cfg;
    cfg.n_det        = args.n_det;
    cfg.dim_x        = args.dim_x;
    cfg.dim_z        = args.dim_z;
    cfg.n_layers     = args.n_layers;
    cfg.n_walkers    = args.n_walkers;
    cfg.batch_size   = args.batch_size;
    cfg.lr           = args.lr;

    torch::Tensor npos, ncharge;

    if (args.system == "h") {
        cfg.n_up = 1; cfg.n_down = 0; cfg.n_nuclei = 1;
        npos    = torch::zeros({1, 3});
        ncharge = torch::tensor({1.0f});
    } else if (args.system == "h2") {
        cfg.n_up = 1; cfg.n_down = 1; cfg.n_nuclei = 2;
        float d = (float)args.bond / 2;
        npos    = torch::tensor({{-d, 0.f, 0.f}, {d, 0.f, 0.f}});
        ncharge = torch::tensor({1.0f, 1.0f});
    } else if (args.system == "he") {
        cfg.n_up = 1; cfg.n_down = 1; cfg.n_nuclei = 1;
        npos    = torch::zeros({1, 3});
        ncharge = torch::tensor({2.0f});
    } else if (args.system == "lih") {
        cfg = PauliNetConfig::lih();
        cfg.n_det = args.n_det; cfg.dim_x = args.dim_x; cfg.dim_z = args.dim_z;
        cfg.n_walkers = args.n_walkers; cfg.batch_size = args.batch_size;
        // LiH equilibrium: Li at origin, H at 3.015 a₀
        npos    = torch::tensor({{0.f, 0.f, 0.f}, {3.015f, 0.f, 0.f}});
        ncharge = torch::tensor({3.0f, 1.0f});
    } else {
        throw std::invalid_argument("Unknown system: " + args.system);
    }

    std::cout << "PauliNet  system=" << args.system
              << "  n_up=" << cfg.n_up
              << "  n_down=" << cfg.n_down
              << "  n_det=" << cfg.n_det
              << "  dim_x=" << cfg.dim_x
              << "  layers=" << cfg.n_layers << "\n";

    // ── Build trainer ────────────────────────────────────────────────────────
    VMCTrainer trainer(cfg, npos, ncharge);
    trainer.model->to(device);

    int64_t n_params = 0;
    for (auto& p : trainer.model->parameters()) n_params += p.numel();
    std::cout << "Trainable parameters: " << n_params << "\n";
    std::cout << "VMC walkers: " << cfg.n_walkers
              << "  batch: " << cfg.batch_size
              << "  steps: " << args.n_steps << "\n\n";

    // ── Training loop ────────────────────────────────────────────────────────
    std::cout << "Burn-in MCMC (" << cfg.n_discard << " steps)...\n";

    trainer.train(args.n_steps, /*verbose=*/true);

    // Save model
    torch::save(trainer.model, args.save);
    std::cout << "\nCheckpoint saved: " << args.save << "\n";

    return 0;
}
