#include <iostream>
#include <torch/torch.h>
#include "modules/ohm_wiper.h"

int main() {
    std::cout << "--- Testing Ohm-WIPER (Zero-FLOP Physics Engine) ---\n";

    // 1. Initialize the module (defaults to 2^20 capacity)
    dm::modules::OhmWIPER wiper;

    // 2. Set up a dummy arena logic 
    // For demonstration, let's say if address == 0, next state is 1 (North)
    // If address == 1, next state is 2 (South), etc.
    {
        torch::NoGradGuard no_grad;
        auto arena_acc = wiper->arena_.accessor<int32_t, 1>();
        arena_acc[0] = 1; // 0000 -> 0001
        arena_acc[1] = 2; // 0001 -> 0010
        // The rest are 0
    }

    // 3. Create a 10x10 input grid filled with 0s
    auto grid_in = torch::zeros({10, 10}, torch::kInt32);

    // Let's set a single pixel at (5, 5) to state 1
    grid_in[5][5] = 1;

    // 4. Run the forward pass (the actual core loop)
    std::cout << "[*] Executing Zero-FLOP forward pass..." << std::endl;
    auto grid_out = wiper->forward(grid_in);

    // 5. Verify the outputs
    // All 0-cells should have address 0 => become state 1
    // The cell (5,5) has state 1. Its neighbors are 0.
    // So its center bit is 1. Its neighbors (N,S,E,W) are 0.
    // Address = 1 | (0<<4) | (0<<8) | (0<<12) | (0<<16) = 1.
    // Thus grid_out[5][5] should be arena_acc[1] = 2.

    std::cout << "grid_out[0][0] (expected 1): " << grid_out[0][0].item<int32_t>() << std::endl;
    std::cout << "grid_out[5][5] (expected 2): " << grid_out[5][5].item<int32_t>() << std::endl;

    if (grid_out[0][0].item<int32_t>() == 1 && grid_out[5][5].item<int32_t>() == 2) {
        std::cout << "\nSUCCESS: Ohm-WIPER Logic is perfectly intact!" << std::endl;
        return 0;
    } else {
        std::cerr << "\nERROR: Output does not match expected logic!" << std::endl;
        return 1;
    }
}
