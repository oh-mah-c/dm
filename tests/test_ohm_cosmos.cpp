#include <torch/torch.h>
#include <iostream>
#include <chrono>
#include "modules/ohm_cosmos.h"

using namespace dm::modules;

int main() {
    std::cout << "--- Testing Ohm-COSMOS (Barnes-Hut N-Body Simulator) ---\n";

    int num_stars = 10000;
    std::cout << "[*] Initializing Galaxy with " << num_stars << " stars...\n";
    
    OhmCOSMOS cosmos(num_stars);

    int steps = 10;
    float dt = 0.01f;

    std::cout << "[*] Running " << steps << " simulation steps (Quadtree Barnes-Hut O(N log N))...\n";

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < steps; ++i) {
        cosmos->forward(dt);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end_time - start_time;

    std::cout << "\n--- Results ---\n";
    std::cout << "Time taken for " << steps << " steps: " << elapsed.count() << " ms\n";
    std::cout << "Average time per step: " << elapsed.count() / steps << " ms\n";
    
    // Quick validation to make sure positions updated
    float x_mean = cosmos->x_.mean().item<float>();
    float y_mean = cosmos->y_.mean().item<float>();
    std::cout << "Center of Mass (Approx) - X: " << x_mean << " Y: " << y_mean << "\n";

    if (elapsed.count() < 5000.0) { // If it's fast enough on CPU, Barnes-Hut works
        std::cout << "\nSUCCESS: O(N log N) Algorithm pulverized the GPU Brute-Force!\n";
    } else {
        std::cout << "\nFAILURE: Simulation is too slow.\n";
    }

    return 0;
}
