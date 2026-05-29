#pragma once

#include <cstdint>

namespace dm {
namespace algorithm {

    // Lõi mô phỏng Mô hình Ising 2D (Ohm-METROPOLIS)
    // Chạy thuật toán Markov Chain Monte Carlo với Bảng tra cứu tĩnh O(1)
    void metropolis_sweep_cpu(
        int8_t* lattice,         // Mảng lưới 2D chứa các spin (+1, -1)
        int width,               // Chiều rộng của lưới
        int height,              // Chiều cao của lưới
        const float* prob_lut,   // Bảng tra cứu xác suất cho 5 mức Delta E
        uint32_t& rng_state      // Trạng thái của bộ sinh số ngẫu nhiên Xorshift
    );

    // Hàm tiện ích sinh số ngẫu nhiên Bitwise (Xorshift32)
    inline uint32_t xorshift32_metro(uint32_t& state) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

} // namespace algorithm
} // namespace dm
