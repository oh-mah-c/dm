// -----------------------------------------------------------------------------
// Neural Quantum States training smoke/entry point
//
// Usage:
//   ./nqs_train [--spins N] [--hidden H] [--steps S] [--lr LR]
//               [--J J] [--h H] [--open] [--save PATH] [--cuda]
//
// This CLI trains the RBM neural quantum state against the exact small-system
// transverse-field Ising energy. It is intended as a deterministic integration
// path and smoke test; large-system VMC training can use the same local-energy
// and Metropolis sampler APIs.
// -----------------------------------------------------------------------------

#include "models/quantum/nqs/nqs.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <string>

using namespace dm::models::quantum;

struct Args {
    int64_t spins = 8;
    int64_t hidden = 16;
    int64_t steps = 10;
    double lr = 1e-2;
    double j = 1.0;
    double h = 1.0;
    bool periodic = true;
    bool cuda = false;
    std::string save = "nqs_rbm_ckpt.pt";
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--spins"  && i + 1 < argc) a.spins = std::stoll(argv[++i]);
        else if (s == "--hidden" && i + 1 < argc) a.hidden = std::stoll(argv[++i]);
        else if (s == "--steps"  && i + 1 < argc) a.steps = std::stoll(argv[++i]);
        else if (s == "--lr"     && i + 1 < argc) a.lr = std::stod(argv[++i]);
        else if (s == "--J"      && i + 1 < argc) a.j = std::stod(argv[++i]);
        else if (s == "--h"      && i + 1 < argc) a.h = std::stod(argv[++i]);
        else if (s == "--save"   && i + 1 < argc) a.save = argv[++i];
        else if (s == "--open") a.periodic = false;
        else if (s == "--cuda") a.cuda = true;
    }
    return a;
}

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    if (args.spins <= 0 || args.spins > 20) {
        std::cerr << "nqs_train exact mode supports --spins in 1..20\n";
        return 1;
    }
    if (args.hidden <= 0 || args.steps < 0 || args.lr <= 0.0) {
        std::cerr << "Invalid NQS training arguments\n";
        return 1;
    }

    torch::Device device = (args.cuda && torch::cuda::is_available())
        ? torch::kCUDA : torch::kCPU;
    if (args.cuda && !torch::cuda::is_available())
        std::cout << "CUDA not available, falling back to CPU.\n";

    torch::manual_seed(11);
    NQSRBMConfig cfg;
    cfg.n_visible = args.spins;
    cfg.n_hidden = args.hidden;
    cfg.init_std = 0.01;
    auto model = NQSRBM(cfg);
    model->to(device);
    torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(args.lr));

    NQSTFIM ham;
    ham.j = args.j;
    ham.h = args.h;
    ham.periodic = args.periodic;

    std::cout << "NQS RBM  spins=" << args.spins
              << "  hidden=" << args.hidden
              << "  steps=" << args.steps
              << "  lr=" << args.lr
              << "  periodic=" << (args.periodic ? "yes" : "no")
              << "  device=" << device << "\n";

    for (int64_t step = 0; step < args.steps; ++step) {
        float energy = nqs_train_exact_step(model, opt, ham);
        if (step % 10 == 0 || step + 1 == args.steps)
            std::cout << "  step " << step << "  exact_energy = " << energy << "\n";
        if (!std::isfinite(energy)) {
            std::cerr << "NQS training produced non-finite energy\n";
            return 1;
        }
    }

    torch::save(model, args.save);
    std::cout << "Checkpoint saved: " << args.save << "\n";
    return 0;
}
