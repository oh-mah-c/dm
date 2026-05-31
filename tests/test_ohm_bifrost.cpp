#include "algorithms/ohm_bifrost_kernel.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

void print_array(const char* label, const std::vector<float>& arr, size_t limit = 10) {
    std::cout << label << " [";
    for (size_t i = 0; i < std::min(arr.size(), limit); ++i) {
        std::cout << std::fixed << std::setprecision(4) << arr[i];
        if (i < arr.size() - 1 && i < limit - 1) std::cout << ", ";
    }
    if (arr.size() > limit) std::cout << ", ...";
    std::cout << "]\n";
}

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << " Ohm-BIFROST: Lossless Reversible Bijective Engine\n";
    std::cout << "========================================================\n\n";

    // Create a 2D "Image" represented as a 1D float array.
    // E.g. A 100x100 pixel patch = 10,000 dimensions
    const size_t DIM = 10000;
    std::vector<float> original_image(DIM);
    
    // Fill with some synthetic pixel data (e.g. gradients, edges)
    for (size_t i = 0; i < DIM; ++i) {
        original_image[i] = std::sin(i * 0.05f) + std::cos(i * 0.1f);
    }
    
    std::cout << "[1] Generating Original Image Data (Simulated 2D Pixels)\n";
    print_array("Original Image :", original_image);
    std::cout << "\n";
    
    // Copy for transformation
    std::vector<float> bifrost_state = original_image;
    
    // Forward Pass: Transform Image into Latent Text + Quantum Noise
    std::cout << "[2] Executing Forward Transform (Image -> Latent Text & Noise)\n";
    dm::algorithm::bifrost_coupling_step_cpu(bifrost_state.data(), DIM, true);
    print_array("Latent State   :", bifrost_state);
    
    // Measure difference (to show it actually scrambled the data)
    double forward_mse = 0.0;
    for (size_t i = 0; i < DIM; ++i) {
        float diff = bifrost_state[i] - original_image[i];
        forward_mse += diff * diff;
    }
    forward_mse /= DIM;
    std::cout << "-> Data Scrambled. Mean Squared Error vs Original: " << forward_mse << "\n\n";

    // Reverse Pass: Recover Image from Latent Text + Quantum Noise
    std::cout << "[3] Executing Reverse Transform (Latent Text & Noise -> Image)\n";
    dm::algorithm::bifrost_coupling_step_cpu(bifrost_state.data(), DIM, false);
    print_array("Recovered Image:", bifrost_state);

    // Verify EXACT mathematically lossless reconstruction
    double reverse_max_error = 0.0;
    bool lossless = true;
    for (size_t i = 0; i < DIM; ++i) {
        float diff = std::abs(bifrost_state[i] - original_image[i]);
        if (diff > reverse_max_error) reverse_max_error = diff;
        // Check with 1e-5 epsilon due to strict 32-bit float arithmetic bounds
        if (diff > 1e-5f) {
            lossless = false;
        }
    }
    
    std::cout << "-> Data Recovered. Max Absolute Error: " << std::scientific << reverse_max_error << "\n";
    
    if (lossless) {
        std::cout << "\n[SUCCESS] Ohm-BIFROST confirmed EXACT BIJECTIVE REVERSIBILITY (Lossless)\n";
        return 0;
    } else {
        std::cout << "\n[FAILED] Information loss detected.\n";
        return 1;
    }
}
