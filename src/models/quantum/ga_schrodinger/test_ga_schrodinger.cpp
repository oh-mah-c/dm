// -----------------------------------------------------------------------------
// GA Schrödinger solver tests
// -----------------------------------------------------------------------------

#include "models/quantum/ga_schrodinger/ga_schrodinger.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace dm::models::quantum;

static int passed = 0;
static int failed = 0;

#define GAQ_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "  [FAIL] " << (msg) << "\n"; \
            failed++; \
        } else { \
            std::cout << "  [PASS] " << (msg) << "\n"; \
            passed++; \
        } \
    } while (0)

static void test_presets() {
    auto box = GASchrodingerConfig::particle_box();
    auto osc = GASchrodingerConfig::harmonic_oscillator();
    auto hyd = GASchrodingerConfig::hydrogen_radial();
    GAQ_CHECK(box.population_size == 44, "paper default population size N=44");
    GAQ_CHECK(std::abs(box.recombination_rate - 0.65) < 1e-12,
              "paper default recombination rate");
    GAQ_CHECK(std::abs(box.mutation_rate - 0.2) < 1e-12,
              "paper default mutation rate");
    GAQ_CHECK(std::abs(osc.energy - 0.5) < 1e-12,
              "harmonic oscillator preset E=0.5");
    GAQ_CHECK(std::abs(hyd.energy + 0.5) < 1e-12 && hyd.hydrogen_l == 1,
              "hydrogen radial preset E=-0.5,l=1");
}

static void test_grid_and_potentials() {
    GASchrodingerConfig cfg;
    cfg.grid_points = 5;
    cfg.x_min = 0.0;
    cfg.x_max = 1.0;
    auto x = ga_schrodinger_grid(cfg);
    GAQ_CHECK(x.size() == 5 && std::abs(x[2] - 0.5) < 1e-12,
              "uniform grid includes midpoint");
    GAQ_CHECK(ga_schrodinger_potential(GASchrodingerPotential::ParticleBox, 0.5) == 0.0,
              "particle-box potential is zero inside box");
    GAQ_CHECK(std::abs(ga_schrodinger_potential(
                  GASchrodingerPotential::HarmonicOscillator, 2.0) - 2.0) < 1e-12,
              "harmonic oscillator potential is x^2/2");
    GAQ_CHECK(std::isfinite(ga_schrodinger_potential(
                  GASchrodingerPotential::HydrogenRadial, 1.0, 1)),
              "hydrogen radial potential finite away from origin");
}

static void test_residual_and_normalization() {
    GASchrodingerConfig cfg;
    cfg.grid_points = 101;
    cfg.x_min = 0.0;
    cfg.x_max = 1.0;
    cfg.energy = 0.5 * M_PI * M_PI;
    auto x = ga_schrodinger_grid(cfg);
    std::vector<double> psi(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        psi[i] = std::sin(M_PI * x[i]);
    auto v0 = [](double) { return 0.0; };
    const double z = ga_schrodinger_residual_z(psi, x, v0, cfg.energy);
    GAQ_CHECK(z < 1e-2, "finite-difference residual small for box eigenfunction");

    const double dx = x[1] - x[0];
    auto normed = ga_schrodinger_normalize(psi, dx);
    double norm2 = 0.0;
    for (double v : normed)
        norm2 += v * v * dx;
    GAQ_CHECK(std::abs(norm2 - 1.0) < 5e-3, "wavefunction normalization");
}

static void test_solver_smoke() {
    auto cfg = GASchrodingerConfig::particle_box();
    cfg.grid_points = 24;
    cfg.population_size = 16;
    cfg.max_generations = 8;
    cfg.fitness_threshold = 2.0;
    cfg.seed = 99;
    auto result = ga_schrodinger_solve(cfg, GASchrodingerPotential::ParticleBox);
    GAQ_CHECK(result.x.size() == 24 && result.psi.size() == 24,
              "solver returns grid and wavefunction");
    GAQ_CHECK(result.generations == 8, "solver respects max generations");
    GAQ_CHECK(std::isfinite(result.best_fitness) && result.best_fitness >= 0.0,
              "solver fitness finite/nonnegative");
    GAQ_CHECK(std::isfinite(result.best_residual), "solver residual finite");
    GAQ_CHECK(std::abs(result.psi.front()) < 1e-12 &&
              std::abs(result.psi.back()) < 1e-12,
              "solver enforces zero boundary values");
}

int main() {
    std::cout << "=== GA Schrödinger Tests ===\n\n";

    std::cout << "-- Config --\n";
    test_presets();

    std::cout << "\n-- Operators --\n";
    test_grid_and_potentials();
    test_residual_and_normalization();

    std::cout << "\n-- Solver --\n";
    test_solver_smoke();

    std::cout << "\n=== Results: " << passed << " passed, "
              << failed << " failed ===\n";
    return failed == 0 ? 0 : 1;
}
