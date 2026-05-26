// ─────────────────────────────────────────────────────────────────────────────
// PhysNet training smoke/entry point.
//
// The default path trains on a tiny synthetic H2-like set so the binary can be
// built and smoke-tested without external QM9/MD17/SN2 files. Real loaders can
// feed PhysNetBatch records into physnet_train_epoch.
//
// Usage: ./physnet_train [--steps N] [--features F] [--modules M]
//                        [--rbf K] [--cutoff C] [--lr LR] [--save PATH]
//                        [--energy-only] [--cuda]
// ─────────────────────────────────────────────────────────────────────────────

#include "models/quantum/physnet/physnet.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace dm::models::quantum;

struct Args {
    int64_t steps = 5;
    int64_t features = 32;
    int64_t modules = 2;
    int64_t rbf = 16;
    double cutoff = 5.0;
    double lr = 1e-3;
    bool charges = true;
    bool cuda = false;
    std::string save = "physnet_ckpt.pt";
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--steps"   && i + 1 < argc) a.steps = std::stoll(argv[++i]);
        else if (s == "--features"&& i + 1 < argc) a.features = std::stoll(argv[++i]);
        else if (s == "--modules" && i + 1 < argc) a.modules = std::stoll(argv[++i]);
        else if (s == "--rbf"     && i + 1 < argc) a.rbf = std::stoll(argv[++i]);
        else if (s == "--cutoff"  && i + 1 < argc) a.cutoff = std::stod(argv[++i]);
        else if (s == "--lr"      && i + 1 < argc) a.lr = std::stod(argv[++i]);
        else if (s == "--save"    && i + 1 < argc) a.save = argv[++i];
        else if (s == "--energy-only") a.charges = false;
        else if (s == "--cuda") a.cuda = true;
    }
    return a;
}

static PhysNetBatch h2(float bond, float energy, torch::Device device) {
    PhysNetBatch b;
    b.atomic_numbers = torch::tensor({1, 1},
        torch::TensorOptions().dtype(torch::kLong).device(device));
    b.positions = torch::tensor({{-0.5f * bond, 0.0f, 0.0f},
                                 { 0.5f * bond, 0.0f, 0.0f}},
                                torch::TensorOptions().device(device));
    b.energy = torch::tensor(energy, torch::TensorOptions().device(device));
    b.forces = torch::zeros({2, 3}, torch::TensorOptions().device(device));
    b.dipole = torch::zeros({3}, torch::TensorOptions().device(device));
    b.total_charge = 0.0;
    return b;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    torch::Device device = (args.cuda && torch::cuda::is_available())
        ? torch::kCUDA : torch::kCPU;
    if (args.cuda && !torch::cuda::is_available())
        std::cout << "CUDA not available, falling back to CPU.\n";

    PhysNetConfig cfg = args.charges
        ? PhysNetConfig::default_config()
        : PhysNetConfig::energy_only();
    cfg.feature_dim = args.features;
    cfg.n_modules = args.modules;
    cfg.n_rbf = args.rbf;
    cfg.cutoff = args.cutoff;
    cfg.lr = args.lr;
    cfg.n_atomic_residual = 1;
    cfg.n_interaction_residual = 1;
    cfg.n_output_residual = 1;
    cfg.w_force = 1.0;

    PhysNet model(cfg);
    model->to(device);
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(args.lr));
    std::vector<PhysNetBatch> batches = {
        h2(0.70f, -1.0f, device),
        h2(0.80f, -0.9f, device),
        h2(1.00f, -0.7f, device),
    };

    std::cout << "PhysNet  features=" << cfg.feature_dim
              << "  modules=" << cfg.n_modules
              << "  rbf=" << cfg.n_rbf
              << "  charges=" << (cfg.predict_charges ? "yes" : "no")
              << "  device=" << device << "\n";

    for (int64_t step = 0; step < args.steps; ++step) {
        float loss = physnet_train_epoch(model, opt, batches,
                                         /*use_forces=*/true,
                                         /*use_dipole=*/cfg.predict_charges);
        if (step % 10 == 0 || step + 1 == args.steps)
            std::cout << "  step " << step << "  loss = " << loss << "\n";
        if (!std::isfinite(loss)) {
            std::cerr << "PhysNet training produced non-finite loss\n";
            return 1;
        }
    }

    torch::save(model, args.save);
    std::cout << "Checkpoint saved: " << args.save << "\n";
    return 0;
}
