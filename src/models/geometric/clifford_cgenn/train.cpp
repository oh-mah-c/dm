// -----------------------------------------------------------------------------
// Clifford CGENN training smoke/entry point
//
// Usage:
//   ./clifford_cgenn_train [--dim N] [--channels C] [--hidden H]
//                          [--depth D] [--steps S] [--lr LR] [--cuda]
// -----------------------------------------------------------------------------

#include "models/geometric/clifford_cgenn/clifford_cgenn.h"

#include <torch/torch.h>
#include <cmath>
#include <iostream>
#include <string>

using namespace dm::models::geometric;

struct Args {
    int64_t dim = 3;
    int64_t channels = 2;
    int64_t hidden = 4;
    int64_t depth = 2;
    int64_t steps = 5;
    double lr = 1e-3;
    bool cuda = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--dim"      && i + 1 < argc) a.dim = std::stoll(argv[++i]);
        else if (s == "--channels" && i + 1 < argc) a.channels = std::stoll(argv[++i]);
        else if (s == "--hidden"   && i + 1 < argc) a.hidden = std::stoll(argv[++i]);
        else if (s == "--depth"    && i + 1 < argc) a.depth = std::stoll(argv[++i]);
        else if (s == "--steps"    && i + 1 < argc) a.steps = std::stoll(argv[++i]);
        else if (s == "--lr"       && i + 1 < argc) a.lr = std::stod(argv[++i]);
        else if (s == "--cuda") a.cuda = true;
    }
    return a;
}

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);

        torch::Device device = (args.cuda && torch::cuda::is_available())
            ? torch::kCUDA : torch::kCPU;
        if (args.cuda && !torch::cuda::is_available())
            std::cout << "CUDA not available, falling back to CPU.\n";

        auto table = clifford_build_product_table(CliffordAlgebraConfig::euclidean(args.dim));
        CGENN model(table, args.channels, args.hidden, args.channels, args.depth);
        model->to(device);
        torch::optim::Adam opt(model->parameters(), torch::optim::AdamOptions(args.lr));

        std::cout << "Clifford CGENN  dim=" << args.dim
                  << "  blades=" << table.n_blades
                  << "  channels=" << args.channels
                  << "  hidden=" << args.hidden
                  << "  depth=" << args.depth
                  << "  device=" << device << "\n";

        for (int64_t step = 0; step < args.steps; ++step) {
            auto x = torch::randn({8, args.channels, table.n_blades},
                                  torch::TensorOptions().device(device));
            auto target = torch::zeros_like(x);
            auto y = model->forward(x);
            auto loss = torch::mse_loss(y, target);
            opt.zero_grad();
            loss.backward();
            opt.step();
            const float v = loss.item<float>();
            if (step % 10 == 0 || step + 1 == args.steps)
                std::cout << "  step " << step << "  loss = " << v << "\n";
            if (!std::isfinite(v)) {
                std::cerr << "Clifford CGENN training produced non-finite loss\n";
                return 1;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "clifford_cgenn_train: " << e.what() << "\n";
        return 1;
    }
}
