#pragma once

#include <torch/torch.h>
#include "algorithms/ohm_quantum_kernel.h"

namespace dm {
namespace modules {

struct OhmQUANTUMImpl : torch::nn::Module {
    // Quantum State Tensors
    torch::Tensor R_; // Real part [width]
    torch::Tensor I_; // Imaginary part [width]
    torch::Tensor V_; // Potential Barrier [width]
    
    // Double Buffers
    torch::Tensor R_new_;
    torch::Tensor I_new_;

    size_t width_;
    float dx_;
    float dt_;
    float dx_sq_;

    // Initialize with size, dx, dt
    OhmQUANTUMImpl(size_t width, float dx = 0.1f, float dt = 0.001f) 
        : width_(width), dx_(dx), dt_(dt) {
        
        dx_sq_ = dx * dx;

        // Initialize wavefunction to zeros
        R_ = register_buffer("R", torch::zeros({(long)width}, torch::kFloat32));
        I_ = register_buffer("I", torch::zeros({(long)width}, torch::kFloat32));
        
        // Initialize potential barrier to zeros
        V_ = register_buffer("V", torch::zeros({(long)width}, torch::kFloat32));

        // Initialize buffers
        R_new_ = torch::zeros({(long)width}, torch::kFloat32);
        I_new_ = torch::zeros({(long)width}, torch::kFloat32);
    }

    // Forward pass simulates exactly ONE time step
    void forward() {
        // Zero-FLOP PyTorch Overhead! Raw pointer access.
        const float* r_old = R_.data_ptr<float>();
        const float* i_old = I_.data_ptr<float>();
        const float* v_ptr = V_.data_ptr<float>();
        
        float* r_new = R_new_.data_ptr<float>();
        float* i_new = I_new_.data_ptr<float>();

        dm::algorithm::schrodinger_step_cpu_1d(
            r_old, i_old, v_ptr,
            r_new, i_new,
            width_, dt_, dx_sq_
        );

        // Ping-pong or copy back
        R_.copy_(R_new_);
        I_.copy_(I_new_);
    }

    // Convenience function to get Probability Density |Psi|^2
    torch::Tensor prob_density() {
        return R_.pow(2) + I_.pow(2);
    }
};

TORCH_MODULE(OhmQUANTUM);

} // namespace modules
} // namespace dm
