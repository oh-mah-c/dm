#pragma once

#include <cstdint>
#include <cstddef>

namespace dm {
namespace algorithm {

    // Ohm-WIPER (Weightless Implicit Physics Equation Resolver)
    // Zero-FLOP lattice-gas cellular automata engine using Ohm-WNN
    void wiper_simulate_step_cpu(
        const int32_t* arena_ram, // Lookup table (Learned physics rules)
        const int32_t* grid_in,   // Current Bit-Spatial Arena (Width x Height)
        int32_t* grid_out,        // Next Bit-Spatial Arena
        size_t width, size_t height
    );

} // namespace algorithm
} // namespace dm
