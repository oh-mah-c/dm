#pragma once
// -----------------------------------------------------------------------------
// Genetic-algorithm Schrödinger solver
// Lahoz-Beltra, "Solving the Schrödinger Equation with Genetic Algorithms:
// A Practical Approach", Computers 2022, 11, 169.
//
// Implements the concrete 1D stationary Schrödinger GA described in Section 2.1:
// chromosomes are sampled wave-function values, fitness is exp(-Z), selection is
// roulette-wheel, crossover is one-point, and mutation perturbs wave-function
// genes. Presets cover the paper's particle-in-box, harmonic oscillator, and
// simplified hydrogen radial potentials.
// -----------------------------------------------------------------------------

#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace dm {
namespace models {
namespace quantum {

enum class GASchrodingerPotential {
    ParticleBox,
    HarmonicOscillator,
    HydrogenRadial,
};

struct GASchrodingerConfig {
    int64_t grid_points = 64;
    int64_t population_size = 44;
    int64_t max_generations = 3200;
    double recombination_rate = 0.65;
    double mutation_rate = 0.2;
    double site_mutation_rate = 0.1;
    double fitness_threshold = 0.87;
    double x_min = 0.0;
    double x_max = 1.0;
    double gene_min = -1.0;
    double gene_max = 1.0;
    double mutation_sigma = 0.1;
    double energy = 0.02;
    int64_t hydrogen_l = 1;
    uint64_t seed = 1234;

    static GASchrodingerConfig particle_box();
    static GASchrodingerConfig harmonic_oscillator();
    static GASchrodingerConfig hydrogen_radial();
};

struct GASchrodingerResult {
    std::vector<double> x;
    std::vector<double> psi;
    double best_fitness = 0.0;
    double best_residual = 0.0;
    int64_t generations = 0;
};

using GASchrodingerPotentialFn = std::function<double(double)>;

std::vector<double> ga_schrodinger_grid(const GASchrodingerConfig& cfg);
double ga_schrodinger_potential(GASchrodingerPotential type,
                                double x,
                                int64_t hydrogen_l = 1);
std::vector<double> ga_schrodinger_normalize(const std::vector<double>& psi,
                                             double dx);
double ga_schrodinger_residual_z(const std::vector<double>& psi,
                                 const std::vector<double>& x,
                                 const GASchrodingerPotentialFn& potential,
                                 double energy);
double ga_schrodinger_fitness(const std::vector<double>& psi,
                              const std::vector<double>& x,
                              const GASchrodingerPotentialFn& potential,
                              double energy);

class GASchrodingerSolver {
public:
    GASchrodingerSolver(GASchrodingerConfig cfg,
                        GASchrodingerPotential potential);
    GASchrodingerSolver(GASchrodingerConfig cfg,
                        GASchrodingerPotentialFn potential);

    GASchrodingerResult solve();

private:
    GASchrodingerConfig cfg_;
    GASchrodingerPotentialFn potential_;
    std::mt19937_64 rng_;

    void validate_config() const;
    std::vector<double> random_chromosome();
    int64_t roulette_select(const std::vector<double>& fitness);
    int64_t tournament_select(const std::vector<double>& fitness);
    void mutate(std::vector<double>& chromosome);
};

GASchrodingerResult ga_schrodinger_solve(GASchrodingerConfig cfg,
                                         GASchrodingerPotential potential);

}  // namespace quantum
}  // namespace models
}  // namespace dm
