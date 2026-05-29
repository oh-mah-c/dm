#pragma once

#include <torch/torch.h>
#include "algorithms/ohm_cosmos_kernel.h"
#include <memory>

namespace dm {
namespace modules {

struct OhmCOSMOSImpl : torch::nn::Module {
    torch::Tensor x_;
    torch::Tensor y_;
    torch::Tensor vx_;
    torch::Tensor vy_;
    torch::Tensor m_;
    
    int num_stars_;
    std::unique_ptr<dm::algorithm::OhmCosmosArena> arena_;

    OhmCOSMOSImpl(int num_stars) : num_stars_(num_stars) {
        // Initialize random positions, velocities, and mass
        x_ = register_buffer("x", torch::rand({num_stars}, torch::kFloat32) * 1000.0f);
        y_ = register_buffer("y", torch::rand({num_stars}, torch::kFloat32) * 1000.0f);
        vx_ = register_buffer("vx", torch::zeros({num_stars}, torch::kFloat32));
        vy_ = register_buffer("vy", torch::zeros({num_stars}, torch::kFloat32));
        m_ = register_buffer("m", torch::ones({num_stars}, torch::kFloat32));

        // Create the pre-allocated Memory Arena
        arena_ = std::make_unique<dm::algorithm::OhmCosmosArena>(num_stars);
    }

    void forward(float dt = 0.1f) {
        float* x_ptr = x_.data_ptr<float>();
        float* y_ptr = y_.data_ptr<float>();
        float* vx_ptr = vx_.data_ptr<float>();
        float* vy_ptr = vy_.data_ptr<float>();
        const float* m_ptr = m_.data_ptr<float>();

        // Zero-FLOP overhead call to the C++ core
        dm::algorithm::cosmos_step_cpu(
            *arena_, x_ptr, y_ptr, vx_ptr, vy_ptr, m_ptr, num_stars_, dt
        );
    }
};

TORCH_MODULE(OhmCOSMOS);

} // namespace modules
} // namespace dm
