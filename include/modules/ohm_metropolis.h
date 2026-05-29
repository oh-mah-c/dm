#pragma once

#include <torch/torch.h>
#include "algorithms/ohm_metropolis_kernel.h"
#include <cmath>

namespace dm {
namespace modules {

struct OhmMETROPOLISImpl : torch::nn::Module {
    torch::Tensor lattice_; // Grid of spins (+1, -1), type kInt8
    float prob_lut_[3];     // LUT for delta_E_half / 2 = 0, 1, 2
    int width_, height_;
    float temperature_;
    uint32_t rng_state_;

    OhmMETROPOLISImpl(int width, int height, float temperature = 2.269f) 
        : width_(width), height_(height), rng_state_(12345) {
        
        // Initialize lattice with random spins (+1 or -1)
        auto rand_spins = torch::randint(0, 2, {height, width}, torch::kInt32) * 2 - 1;
        lattice_ = register_buffer("lattice", rand_spins.to(torch::kInt8));

        update_temperature(temperature);
    }

    void update_temperature(float T) {
        temperature_ = T;
        // delta_E_half / 2 = 0 -> delta_E_half = 0 -> Delta E = 0
        // delta_E_half / 2 = 1 -> delta_E_half = 2 -> Delta E = 4
        // delta_E_half / 2 = 2 -> delta_E_half = 4 -> Delta E = 8
        prob_lut_[0] = 1.0f; 
        prob_lut_[1] = std::exp(-4.0f / T); 
        prob_lut_[2] = std::exp(-8.0f / T); 
    }

    // Run one Metropolis sweep (N spin updates where N = width*height)
    void forward() {
        int8_t* lattice_ptr = lattice_.data_ptr<int8_t>();
        
        dm::algorithm::metropolis_sweep_cpu(
            lattice_ptr, width_, height_, prob_lut_, rng_state_
        );
    }

    // Calculate Magnetization (average spin)
    float magnetization() {
        return lattice_.to(torch::kFloat32).mean().item<float>();
    }
};

TORCH_MODULE(OhmMETROPOLIS);

} // namespace modules
} // namespace dm
