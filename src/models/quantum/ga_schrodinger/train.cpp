// -----------------------------------------------------------------------------
// GA Schrödinger solver CLI
//
// Usage:
//   ./ga_schrodinger_train [--system box|oscillator|hydrogen] [--grid N]
//                          [--population N] [--generations N] [--energy E]
//                          [--seed S] [--threshold F]
// -----------------------------------------------------------------------------

#include "models/quantum/ga_schrodinger/ga_schrodinger.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace dm::models::quantum;

struct Args {
    std::string system = "box";
    int64_t grid = -1;
    int64_t population = -1;
    int64_t generations = -1;
    double energy = std::numeric_limits<double>::quiet_NaN();
    double threshold = std::numeric_limits<double>::quiet_NaN();
    uint64_t seed = 1234;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if      (s == "--system"      && i + 1 < argc) a.system = argv[++i];
        else if (s == "--grid"        && i + 1 < argc) a.grid = std::stoll(argv[++i]);
        else if (s == "--population"  && i + 1 < argc) a.population = std::stoll(argv[++i]);
        else if (s == "--generations" && i + 1 < argc) a.generations = std::stoll(argv[++i]);
        else if (s == "--energy"      && i + 1 < argc) a.energy = std::stod(argv[++i]);
        else if (s == "--threshold"   && i + 1 < argc) a.threshold = std::stod(argv[++i]);
        else if (s == "--seed"        && i + 1 < argc) a.seed = std::stoull(argv[++i]);
    }
    return a;
}

static GASchrodingerConfig config_for(const Args& args,
                                      GASchrodingerPotential& potential) {
    GASchrodingerConfig cfg;
    if (args.system == "box") {
        cfg = GASchrodingerConfig::particle_box();
        potential = GASchrodingerPotential::ParticleBox;
    } else if (args.system == "oscillator") {
        cfg = GASchrodingerConfig::harmonic_oscillator();
        potential = GASchrodingerPotential::HarmonicOscillator;
    } else if (args.system == "hydrogen") {
        cfg = GASchrodingerConfig::hydrogen_radial();
        potential = GASchrodingerPotential::HydrogenRadial;
    } else {
        throw std::invalid_argument("Unknown --system; use box, oscillator, or hydrogen");
    }

    if (args.grid > 0) cfg.grid_points = args.grid;
    if (args.population > 0) cfg.population_size = args.population;
    if (args.generations >= 0) cfg.max_generations = args.generations;
    if (std::isfinite(args.energy)) cfg.energy = args.energy;
    if (std::isfinite(args.threshold)) cfg.fitness_threshold = args.threshold;
    cfg.seed = args.seed;
    return cfg;
}

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);
        GASchrodingerPotential potential = GASchrodingerPotential::ParticleBox;
        auto cfg = config_for(args, potential);

        std::cout << "GA Schrödinger  system=" << args.system
                  << "  grid=" << cfg.grid_points
                  << "  population=" << cfg.population_size
                  << "  generations=" << cfg.max_generations
                  << "  energy=" << cfg.energy << "\n";

        auto result = ga_schrodinger_solve(cfg, potential);
        std::cout << "best_fitness=" << result.best_fitness
                  << "  residual=" << result.best_residual
                  << "  generations=" << result.generations << "\n";
        std::cout << "sample";
        const size_t stride = std::max<size_t>(1, result.x.size() / 8);
        for (size_t i = 0; i < result.x.size(); i += stride)
            std::cout << " (" << result.x[i] << "," << result.psi[i] << ")";
        std::cout << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ga_schrodinger_train: " << e.what() << "\n";
        return 1;
    }
}
