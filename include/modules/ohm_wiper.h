#pragma once

#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>
#include "algorithms/ohm_wiper_kernel.h"

namespace dm {
namespace modules {

struct TORCH_API OhmWIPERImpl : torch::nn::Module {
    torch::Tensor arena_; // Static lookup array for learned physics rules

    // The state_capacity should match our 20-bit hashing scheme (2^20 = 1048576)
    OhmWIPERImpl(int64_t state_capacity = 1048576) {
        // Initialize RAM lookup table for the weightless logic
        arena_ = register_parameter("arena", torch::zeros({state_capacity}, torch::kInt32), /*requires_grad=*/false);
    }

    torch::Tensor forward(const torch::Tensor& grid_in) {
        // Ensure the input tensor is physically contiguous and Int32 on CPU
        auto grid_contiguous = grid_in.contiguous().to(torch::kInt32).cpu();
        
        // 1. Initialize an empty tensor for the next state
        auto grid_out = torch::empty_like(grid_contiguous);
        
        // 2. Obtain raw pointers (bypassing PyTorch overhead)
        auto* arena_ptr = arena_.data_ptr<int32_t>();
        auto* in_ptr = grid_contiguous.data_ptr<int32_t>();
        auto* out_ptr = grid_out.data_ptr<int32_t>();
        
        // Get dimensions (Assuming grid is 2D: [height, width])
        // If it's a batch of grids, we'd need to loop over the batch dimension.
        // For simplicity, we assume a single [height, width] layout.
        int64_t height = grid_contiguous.size(0);
        int64_t width = grid_contiguous.size(1);
        
        // 3. Dispatch to the raw C++ core logic
        dm::algorithm::wiper_simulate_step_cpu(
            arena_ptr, in_ptr, out_ptr, 
            width, height
        );
        
        return grid_out;
    }
};

// Generates the OhmWIPER value-semantics PyTorch C++ API
TORCH_MODULE(OhmWIPER);

} // namespace modules
} // namespace dm
