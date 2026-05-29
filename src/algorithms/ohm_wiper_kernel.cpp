#include "algorithms/ohm_wiper_kernel.h"

namespace dm {
namespace algorithm {

void wiper_simulate_step_cpu(
    const int32_t* arena_ram,
    const int32_t* grid_in,
    int32_t* grid_out,
    size_t width, size_t height
) {
    // Iterate over the Bit-Spatial Arena
    // We use a Von Neumann neighborhood (Center, North, South, East, West)
    // 5 cells * 4 bits = 20 bits integer address
    
    for (size_t y = 0; y < height; ++y) {
        // Toroidal boundary wrap for Y
        size_t n_y = (y > 0) ? (y - 1) : (height - 1);
        size_t s_y = (y < height - 1) ? (y + 1) : 0;
        
        for (size_t x = 0; x < width; ++x) {
            // Toroidal boundary wrap for X
            size_t w_x = (x > 0) ? (x - 1) : (width - 1);
            size_t e_x = (x < width - 1) ? (x + 1) : 0;
            
            // Fetch the 4-bit states (masking to ensure strict 4-bit bounds)
            int32_t c_state = grid_in[y * width + x] & 0xF;
            int32_t n_state = grid_in[n_y * width + x] & 0xF;
            int32_t s_state = grid_in[s_y * width + x] & 0xF;
            int32_t e_state = grid_in[y * width + e_x] & 0xF;
            int32_t w_state = grid_in[y * width + w_x] & 0xF;
            
            // Spatial Hashing: Pack into a 20-bit RAM address
            int32_t address = c_state 
                            | (n_state << 4) 
                            | (s_state << 8) 
                            | (e_state << 12) 
                            | (w_state << 16);
                            
            // Zero-FLOP Lookup: Retrieve the next state from the learned physical rules
            grid_out[y * width + x] = arena_ram[address] & 0xF;
        }
    }
}

} // namespace algorithm
} // namespace dm
