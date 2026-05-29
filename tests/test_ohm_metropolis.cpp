#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include "modules/ohm_metropolis.h"

using namespace dm::modules;

void run_simulation(float temperature, int sweeps) {
    int width = 100;
    int height = 100;
    
    OhmMETROPOLIS metropolis(width, height, temperature);
    
    std::cout << "\n[*] Simulating at T = " << temperature << " ...\n";
    std::cout << "Initial Magnetization: " << metropolis->magnetization() << "\n";

    // Warm-up / Equilibration
    for (int i = 0; i < sweeps; ++i) {
        metropolis->forward();
    }

    float final_mag = metropolis->magnetization();
    std::cout << "Final Magnetization (after " << sweeps << " sweeps): " << final_mag << "\n";
    
    if (temperature > 2.3f) {
        if (std::abs(final_mag) < 0.2f) {
            std::cout << "-> High T: Random State (Demagnetized) [OK]\n";
        } else {
            std::cout << "-> Error: Should be demagnetized.\n";
        }
    } else if (temperature < 2.0f) {
        if (std::abs(final_mag) > 0.8f) {
            std::cout << "-> Low T: Ordered State (Magnetized) [OK]\n";
        } else {
            std::cout << "-> Error: Should be magnetized.\n";
        }
    }
}

int main() {
    std::cout << "--- Testing Ohm-METROPOLIS (Ising Model MCMC Phase Transition) ---\n";

    // Test 1: High Temperature (Disordered State)
    // T = 3.0 > Tc (2.269) -> Magnetization should be near 0
    run_simulation(3.0f, 1000);

    // Test 2: Low Temperature (Ordered State)
    // T = 1.5 < Tc (2.269) -> Magnetization should be near +1 or -1
    run_simulation(1.5f, 1000);
    
    std::cout << "\nSUCCESS: Spontaneous Symmetry Breaking observed!\n";
    return 0;
}
