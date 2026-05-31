#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include "modules/ohm_quantum.h"

using namespace dm::modules;

int main() {
    std::cout << "--- Testing Ohm-QUANTUM (Time-Dependent Schrödinger Equation) ---\n";

    // Simulation Parameters
    size_t width = 1000;
    float dx = 0.1f;
    float dt = 0.001f;

    OhmQUANTUM quantum(width, dx, dt);

    // Initial Conditions: Gaussian Wave Packet
    // Represents a particle located at x0, moving to the right with momentum p0
    float x0 = 300.0f; // Initial position (index)
    float p0 = 1.0f;   // Initial momentum
    float sigma = 20.0f; // Width of the wave packet

    for (size_t x = 1; x < width - 1; ++x) {
        float x_real = (float)x;
        // Envelope
        float envelope = std::exp(-0.5f * std::pow((x_real - x0) / sigma, 2.0f));
        // Plane wave: e^{i * p0 * x} = cos(p0 * x) + i * sin(p0 * x)
        quantum->R_[x] = envelope * std::cos(p0 * x_real);
        quantum->I_[x] = envelope * std::sin(p0 * x_real);
    }

    // Normalize the wave function (sum of probabilities = 1)
    auto prob = quantum->prob_density();
    float sum_prob = prob.sum().item<float>();
    quantum->R_ /= std::sqrt(sum_prob);
    quantum->I_ /= std::sqrt(sum_prob);

    // Setup Potential Barrier (The "Wall")
    // Let's place a barrier from index 600 to 650
    float V0 = 2.0f; // Height of the barrier
    for (size_t x = 600; x < 650; ++x) {
        quantum->V_[x] = V0;
    }

    // Run the simulation for a number of time steps
    int steps = 5000;
    std::cout << "[*] Running " << steps << " Leapfrog integration steps...\n";

    // Track total probability to ensure Unitarity (should remain 1.0)
    for (int i = 0; i < steps; ++i) {
        quantum->forward();
    }

    // Measure Probabilities
    auto final_prob = quantum->prob_density();
    float final_sum_prob = final_prob.sum().item<float>();

    // Probability before the barrier (Reflected)
    auto prob_reflected = final_prob.slice(0, 0, 600).sum().item<float>();
    
    // Probability after the barrier (Tunneled)
    auto prob_tunneled = final_prob.slice(0, 650, width).sum().item<float>();

    std::cout << "\n--- Results ---\n";
    std::cout << "Total Probability (Unitarity Check): " << final_sum_prob << " (Should be ~1.0)\n";
    std::cout << "Reflected Probability (Bounced off): " << prob_reflected * 100.0f << "%\n";
    std::cout << "Tunneled Probability (Passed through): " << prob_tunneled * 100.0f << "%\n";

    if (prob_tunneled > 0.0001f) {
        std::cout << "\nSUCCESS: Quantum Tunneling Phenomenon observed!\n";
    } else {
        std::cout << "\nFAILURE: No Tunneling detected.\n";
    }

    return 0;
}
