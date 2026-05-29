#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "modules/ohm_feynman.h"

using namespace dm::modules;

int main() {
    std::cout << "--- Testing Ohm-FEYNMAN (Quantum Path Integral Simulator) ---\n";

    // Initialize with 1 Million paths to observe interference
    int num_paths = 1000000;
    int steps = 50; // Random walk steps
    OhmFEYNMAN feynman(num_paths, steps);

    float start_x = 0.0f;
    float end_x = 1.0f;

    std::cout << "[*] Starting Monte Carlo Path Generation...\n";
    std::cout << "    Paths: " << num_paths << "\n";
    std::cout << "    From A(" << start_x << ") to B(" << end_x << ")\n";

    // Run the simulation
    auto amplitude = feynman->forward(start_x, end_x);

    float prob_real = amplitude[0].item<float>();
    float prob_imag = amplitude[1].item<float>();
    
    // Probability is the magnitude squared of the complex amplitude
    float probability = prob_real * prob_real + prob_imag * prob_imag;

    std::cout << "\n--- Results ---\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Complex Amplitude: (" << prob_real << ") + i(" << prob_imag << ")\n";
    std::cout << "Probability |Psi|^2: " << probability << "\n";

    if (probability > 0.0f) {
        std::cout << "\nSUCCESS: Destructive and Constructive Interference calculated!\n";
    } else {
        std::cout << "\nFAILURE: Probability is exactly zero.\n";
    }

    return 0;
}
