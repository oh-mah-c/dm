#pragma once

#include <torch/torch.h>
#include "algorithms/ohm_relativity_kernel.h"
#include <vector>

namespace dm {
namespace modules {

struct OhmRELATIVITYImpl : torch::nn::Module {
    float mass_;

    OhmRELATIVITYImpl(float mass = 1.0f) : mass_(mass) {}

    // Mô phỏng quỹ đạo Photon từ vị trí ban đầu u0, v0
    // Trả về Tuple: (Tensor r, Tensor phi, Tensor dphi)
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        float r0, float dr_dphi0, int max_steps = 1000, float tolerance = 1e-6f) {
        
        dm::algorithm::GeoState state;
        state.u = 1.0f / r0;
        
        // du/dphi = d(1/r)/dphi = -1/r^2 * dr/dphi
        state.v = -(1.0f / (r0 * r0)) * dr_dphi0;
        state.phi = 0.0f;

        std::vector<float> r_hist;
        std::vector<float> phi_hist;
        std::vector<float> dphi_hist;

        dm::algorithm::simulate_photon_path(state, mass_, max_steps, tolerance, r_hist, phi_hist, dphi_hist);

        auto r_tensor = torch::from_blob(r_hist.data(), {(long)r_hist.size()}, torch::kFloat32).clone();
        auto phi_tensor = torch::from_blob(phi_hist.data(), {(long)phi_hist.size()}, torch::kFloat32).clone();
        auto dphi_tensor = torch::from_blob(dphi_hist.data(), {(long)dphi_hist.size()}, torch::kFloat32).clone();

        return std::make_tuple(r_tensor, phi_tensor, dphi_tensor);
    }
};

TORCH_MODULE(OhmRELATIVITY);

} // namespace modules
} // namespace dm
