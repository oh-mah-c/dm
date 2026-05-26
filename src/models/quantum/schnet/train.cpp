// ─────────────────────────────────────────────────────────────────────────────
// SchNet training smoke/entry point
//
// Usage:
//   ./schnet_train [--steps N] [--hidden F] [--interactions K]
//                 [--gaussians K] [--lr LR] [--save PATH] [--cuda]
//
// The CLI accepts synthetic batches by default so the target can be built and
// smoke-tested without requiring QM9/MD17 files. Real dataset loaders can feed
// SchNetBatch records into schnet_train_epoch.
// ─────────────────────────────────────────────────────────────────────────────

#include "models/quantum/schnet/schnet.h"

#include <torch/torch.h>
#include <iostream>
#include <string>
#include <vector>

using namespace dm::models::quantum;

struct Args {
    int64_t steps = 5;
    int64_t hidden = 64;
    int64_t interactions = 3;
    int64_t gaussians = 301;
    double cutoff = 30.0;
    double lr = 1e-3;
    bool forces = true;
    bool cuda = false;
    std::string save = "schnet_ckpt.pt";
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--steps"       && i + 1 < argc) a.steps = std::stoll(argv[++i]);
        else if (s == "--hidden"      && i + 1 < argc) a.hidden = std::stoll(argv[++i]);
        else if (s == "--interactions"&& i + 1 < argc) a.interactions = std::stoll(argv[++i]);
        else if (s == "--gaussians"   && i + 1 < argc) a.gaussians = std::stoll(argv[++i]);
        else if (s == "--cutoff"      && i + 1 < argc) a.cutoff = std::stod(argv[++i]);
        else if (s == "--lr"          && i + 1 < argc) a.lr = std::stod(argv[++i]);
        else if (s == "--save"        && i + 1 < argc) a.save = argv[++i];
        else if (s == "--no-forces") a.forces = false;
        else if (s == "--cuda") a.cuda = true;
    }
    return a;
}

static SchNetBatch make_h2_batch(float bond, float energy,
                                  torch::Device device) {
    SchNetBatch b;
    b.atomic_numbers = torch::tensor({1, 1}, torch::TensorOptions().dtype(torch::kLong).device(device));
    b.positions = torch::tensor({{-0.5f * bond, 0.0f, 0.0f},
                                 { 0.5f * bond, 0.0f, 0.0f}},
                                torch::TensorOptions().device(device));
    b.energy = torch::tensor(energy, torch::TensorOptions().device(device));
    b.forces = torch::zeros({2, 3}, torch::TensorOptions().device(device));
    return b;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);

    torch::Device device = (args.cuda && torch::cuda::is_available())
        ? torch::kCUDA : torch::kCPU;
    if (args.cuda && !torch::cuda::is_available())
        std::cout << "CUDA not available, falling back to CPU.\n";

    SchNetConfig cfg = SchNetConfig::qm9();
    cfg.hidden_dim = args.hidden;
    cfg.n_interactions = args.interactions;
    cfg.n_gaussians = args.gaussians;
    cfg.cutoff = args.cutoff;
    cfg.lr = args.lr;

    SchNet model(cfg);
    model->to(device);
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(args.lr));

    std::vector<SchNetBatch> batches = {
        make_h2_batch(0.70f, -1.0f, device),
        make_h2_batch(0.80f, -0.9f, device),
        make_h2_batch(1.00f, -0.7f, device),
    };

    std::cout << "SchNet  hidden=" << cfg.hidden_dim
              << "  interactions=" << cfg.n_interactions
              << "  gaussians=" << cfg.n_gaussians
              << "  forces=" << (args.forces ? "yes" : "no")
              << "  device=" << device << "\n";

    for (int64_t step = 0; step < args.steps; ++step) {
        float loss = schnet_train_epoch(model, opt, batches, args.forces);
        if (step % 10 == 0 || step + 1 == args.steps) {
            std::cout << "  step " << step << "  loss = " << loss << "\n";
        }
        if (!std::isfinite(loss)) {
            std::cerr << "SchNet training produced non-finite loss\n";
            return 1;
        }
    }

    torch::save(model, args.save);
    std::cout << "Checkpoint saved: " << args.save << "\n";
    return 0;
}
