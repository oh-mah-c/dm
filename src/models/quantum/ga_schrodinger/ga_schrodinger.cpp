// -----------------------------------------------------------------------------
// Genetic-algorithm Schrödinger solver implementation
// -----------------------------------------------------------------------------

#include "models/quantum/ga_schrodinger/ga_schrodinger.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace dm {
namespace models {
namespace quantum {

namespace {

double clamp(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

}  // namespace

GASchrodingerConfig GASchrodingerConfig::particle_box() {
    GASchrodingerConfig cfg;
    cfg.energy = 0.02;
    cfg.x_min = 0.0;
    cfg.x_max = 1.0;
    return cfg;
}

GASchrodingerConfig GASchrodingerConfig::harmonic_oscillator() {
    GASchrodingerConfig cfg;
    cfg.energy = 0.5;
    cfg.x_min = -4.0;
    cfg.x_max = 4.0;
    return cfg;
}

GASchrodingerConfig GASchrodingerConfig::hydrogen_radial() {
    GASchrodingerConfig cfg;
    cfg.energy = -0.5;
    cfg.x_min = 0.05;
    cfg.x_max = 12.0;
    cfg.hydrogen_l = 1;
    return cfg;
}

std::vector<double> ga_schrodinger_grid(const GASchrodingerConfig& cfg) {
    if (cfg.grid_points < 3)
        throw std::invalid_argument("GA Schrödinger grid_points must be >= 3");
    if (!(cfg.x_max > cfg.x_min))
        throw std::invalid_argument("GA Schrödinger x_max must be greater than x_min");

    std::vector<double> x(static_cast<size_t>(cfg.grid_points));
    const double dx = (cfg.x_max - cfg.x_min) / static_cast<double>(cfg.grid_points - 1);
    for (int64_t i = 0; i < cfg.grid_points; ++i)
        x[static_cast<size_t>(i)] = cfg.x_min + static_cast<double>(i) * dx;
    return x;
}

double ga_schrodinger_potential(GASchrodingerPotential type,
                                double x,
                                int64_t hydrogen_l) {
    switch (type) {
        case GASchrodingerPotential::ParticleBox:
            return 0.0;
        case GASchrodingerPotential::HarmonicOscillator:
            return 0.5 * x * x;
        case GASchrodingerPotential::HydrogenRadial: {
            const double r = std::max(std::abs(x), 1e-12);
            const double l = static_cast<double>(hydrogen_l);
            return l * (l + 1.0) / (r * r) - 2.0 / r;
        }
    }
    throw std::invalid_argument("Unknown GA Schrödinger potential");
}

std::vector<double> ga_schrodinger_normalize(const std::vector<double>& psi,
                                             double dx) {
    if (psi.empty() || dx <= 0.0)
        throw std::invalid_argument("Cannot normalize empty wavefunction or nonpositive dx");
    double norm2 = 0.0;
    for (double v : psi)
        norm2 += v * v * dx;
    if (norm2 <= 0.0 || !std::isfinite(norm2))
        return psi;
    const double inv = 1.0 / std::sqrt(norm2);
    std::vector<double> out = psi;
    for (double& v : out)
        v *= inv;
    return out;
}

double ga_schrodinger_residual_z(const std::vector<double>& psi,
                                 const std::vector<double>& x,
                                 const GASchrodingerPotentialFn& potential,
                                 double energy) {
    if (psi.size() != x.size() || psi.size() < 3)
        throw std::invalid_argument("Residual requires matching psi/x vectors with >= 3 values");
    if (!potential)
        throw std::invalid_argument("Residual requires a potential function");

    const double dx = x[1] - x[0];
    if (dx <= 0.0)
        throw std::invalid_argument("Residual requires increasing grid");

    double residual2 = 0.0;
    double amp2 = 0.0;
    for (size_t j = 1; j + 1 < psi.size(); ++j) {
        const double second = (psi[j - 1] + psi[j + 1] - 2.0 * psi[j]) / (dx * dx);
        const double d = -0.5 * second + (potential(x[j]) - energy) * psi[j];
        residual2 += d * d;
        amp2 += psi[j] * psi[j];
    }
    if (amp2 <= 0.0)
        return std::numeric_limits<double>::infinity();
    return residual2 / amp2;
}

double ga_schrodinger_fitness(const std::vector<double>& psi,
                              const std::vector<double>& x,
                              const GASchrodingerPotentialFn& potential,
                              double energy) {
    const double z = ga_schrodinger_residual_z(psi, x, potential, energy);
    if (!std::isfinite(z))
        return 0.0;
    return std::exp(-z);
}

GASchrodingerSolver::GASchrodingerSolver(GASchrodingerConfig cfg,
                                         GASchrodingerPotential potential)
    : cfg_(cfg),
      potential_([potential, l = cfg.hydrogen_l](double x) {
          return ga_schrodinger_potential(potential, x, l);
      }),
      rng_(cfg.seed) {
    validate_config();
}

GASchrodingerSolver::GASchrodingerSolver(GASchrodingerConfig cfg,
                                         GASchrodingerPotentialFn potential)
    : cfg_(cfg), potential_(std::move(potential)), rng_(cfg.seed) {
    validate_config();
}

void GASchrodingerSolver::validate_config() const {
    if (cfg_.grid_points < 3)
        throw std::invalid_argument("GA Schrödinger grid_points must be >= 3");
    if (cfg_.population_size < 2)
        throw std::invalid_argument("GA Schrödinger population_size must be >= 2");
    if (cfg_.max_generations < 0)
        throw std::invalid_argument("GA Schrödinger max_generations must be >= 0");
    if (cfg_.recombination_rate < 0.0 || cfg_.recombination_rate > 1.0 ||
        cfg_.mutation_rate < 0.0 || cfg_.mutation_rate > 1.0 ||
        cfg_.site_mutation_rate < 0.0 || cfg_.site_mutation_rate > 1.0)
        throw std::invalid_argument("GA Schrödinger rates must be in [0,1]");
    if (!(cfg_.x_max > cfg_.x_min))
        throw std::invalid_argument("GA Schrödinger x range is invalid");
    if (!(cfg_.gene_max > cfg_.gene_min))
        throw std::invalid_argument("GA Schrödinger gene range is invalid");
    if (cfg_.mutation_sigma < 0.0)
        throw std::invalid_argument("GA Schrödinger mutation_sigma must be nonnegative");
    if (!potential_)
        throw std::invalid_argument("GA Schrödinger potential must be defined");
}

std::vector<double> GASchrodingerSolver::random_chromosome() {
    std::uniform_real_distribution<double> uniform(cfg_.gene_min, cfg_.gene_max);
    std::vector<double> chromosome(static_cast<size_t>(cfg_.grid_points));
    for (double& v : chromosome)
        v = uniform(rng_);
    chromosome.front() = 0.0;
    chromosome.back() = 0.0;
    return chromosome;
}

int64_t GASchrodingerSolver::roulette_select(const std::vector<double>& fitness) {
    const double total = std::accumulate(fitness.begin(), fitness.end(), 0.0);
    if (total <= 0.0 || !std::isfinite(total)) {
        std::uniform_int_distribution<int64_t> pick(0, static_cast<int64_t>(fitness.size()) - 1);
        return pick(rng_);
    }
    std::uniform_real_distribution<double> uniform(0.0, total);
    double r = uniform(rng_);
    for (size_t i = 0; i < fitness.size(); ++i) {
        r -= fitness[i];
        if (r <= 0.0)
            return static_cast<int64_t>(i);
    }
    return static_cast<int64_t>(fitness.size()) - 1;
}

int64_t GASchrodingerSolver::tournament_select(const std::vector<double>& fitness) {
    std::uniform_int_distribution<int64_t> pick(0, static_cast<int64_t>(fitness.size()) - 1);
    const int64_t a = pick(rng_);
    const int64_t b = pick(rng_);
    return fitness[static_cast<size_t>(a)] >= fitness[static_cast<size_t>(b)] ? a : b;
}

void GASchrodingerSolver::mutate(std::vector<double>& chromosome) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    if (unit(rng_) > cfg_.mutation_rate)
        return;
    std::normal_distribution<double> noise(0.0, cfg_.mutation_sigma);
    for (size_t j = 1; j + 1 < chromosome.size(); ++j) {
        if (unit(rng_) <= cfg_.site_mutation_rate) {
            chromosome[j] = clamp(chromosome[j] + noise(rng_),
                                  cfg_.gene_min, cfg_.gene_max);
        }
    }
}

GASchrodingerResult GASchrodingerSolver::solve() {
    auto x = ga_schrodinger_grid(cfg_);
    const double dx = x[1] - x[0];
    std::vector<std::vector<double>> population;
    population.reserve(static_cast<size_t>(cfg_.population_size));
    for (int64_t i = 0; i < cfg_.population_size; ++i)
        population.push_back(random_chromosome());

    std::vector<double> fitness(static_cast<size_t>(cfg_.population_size), 0.0);
    int64_t best_idx = 0;
    double best_fit = -1.0;
    double best_z = std::numeric_limits<double>::infinity();

    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (int64_t generation = 0; generation <= cfg_.max_generations; ++generation) {
        for (int64_t i = 0; i < cfg_.population_size; ++i) {
            fitness[static_cast<size_t>(i)] =
                ga_schrodinger_fitness(population[static_cast<size_t>(i)],
                                       x, potential_, cfg_.energy);
            if (fitness[static_cast<size_t>(i)] > best_fit) {
                best_fit = fitness[static_cast<size_t>(i)];
                best_idx = i;
                best_z = ga_schrodinger_residual_z(population[static_cast<size_t>(i)],
                                                   x, potential_, cfg_.energy);
            }
        }

        if (best_fit >= cfg_.fitness_threshold || generation == cfg_.max_generations) {
            GASchrodingerResult result;
            result.x = x;
            result.psi = ga_schrodinger_normalize(population[static_cast<size_t>(best_idx)], dx);
            result.best_fitness = best_fit;
            result.best_residual = best_z;
            result.generations = generation;
            return result;
        }

        std::vector<std::vector<double>> next;
        next.reserve(population.size());
        next.push_back(population[static_cast<size_t>(best_idx)]);
        while (next.size() < population.size()) {
            auto parent_a = population[static_cast<size_t>(roulette_select(fitness))];
            auto parent_b = population[static_cast<size_t>(roulette_select(fitness))];
            if (unit(rng_) <= cfg_.recombination_rate) {
                const int64_t ia = tournament_select(fitness);
                const int64_t ib = tournament_select(fitness);
                parent_a = population[static_cast<size_t>(ia)];
                parent_b = population[static_cast<size_t>(ib)];
                std::uniform_int_distribution<int64_t> cut_pick(1, cfg_.grid_points - 2);
                const int64_t cut = cut_pick(rng_);
                for (int64_t j = cut; j < cfg_.grid_points; ++j)
                    std::swap(parent_a[static_cast<size_t>(j)],
                              parent_b[static_cast<size_t>(j)]);
            }
            mutate(parent_a);
            mutate(parent_b);
            parent_a.front() = parent_a.back() = 0.0;
            parent_b.front() = parent_b.back() = 0.0;
            next.push_back(std::move(parent_a));
            if (next.size() < population.size())
                next.push_back(std::move(parent_b));
        }
        population = std::move(next);
    }

    throw std::runtime_error("GA Schrödinger solver reached unreachable state");
}

GASchrodingerResult ga_schrodinger_solve(GASchrodingerConfig cfg,
                                         GASchrodingerPotential potential) {
    GASchrodingerSolver solver(cfg, potential);
    return solver.solve();
}

}  // namespace quantum
}  // namespace models
}  // namespace dm
