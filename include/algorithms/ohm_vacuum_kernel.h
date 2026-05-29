#pragma once

#include <cstdint>

namespace dm {
namespace algorithm {

    // Hàm băm giả ngẫu nhiên siêu tốc (Xorshift32)
    inline uint32_t xorshift32(uint32_t& state) {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        return state = x;
    }

    // C++ Kernel: Giải tích Lưới Chân Không Lượng Tử
    // Tính toán liên tục MCMC trên mảng Float phẳng
    void quantum_vacuum_fluctuation_step(
        float* phi_field, int width, int height,
        float alpha, float beta, float quantum_jitter, uint32_t& rng_state
    );

} // namespace algorithm
} // namespace dm
