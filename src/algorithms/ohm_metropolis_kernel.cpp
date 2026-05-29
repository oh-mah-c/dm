#include "algorithms/ohm_metropolis_kernel.h"

namespace dm {
namespace algorithm {

void metropolis_sweep_cpu(
    int8_t* lattice, 
    int width, int height, 
    const float* prob_lut, 
    uint32_t& rng_state
) {
    int total_spins = width * height;

    // Chạy 1 Sweep (cập nhật bằng đúng số lượng spin trong mạng)
    for (int i = 0; i < total_spins; ++i) {
        // Lấy ngẫu nhiên 1 tọa độ 1D
        int idx = xorshift32_metro(rng_state) % total_spins;
        
        // Tính toán tọa độ hàng xóm (Periodic Boundary Conditions)
        int up = (idx >= width) ? (idx - width) : (idx - width + total_spins);
        int down = (idx < total_spins - width) ? (idx + width) : (idx + width - total_spins);
        int left = (idx % width > 0) ? (idx - 1) : (idx + width - 1);
        int right = ((idx + 1) % width > 0) ? (idx + 1) : (idx - width + 1);

        int sum_neighbors = lattice[up] + lattice[down] + lattice[left] + lattice[right];
        
        // Tính delta E_half.
        // Năng lượng cục bộ hiện tại E_current = -J * S_i * sum_neighbors (với J=1)
        // Năng lượng cục bộ nếu lật E_flipped = J * S_i * sum_neighbors
        // Sự thay đổi năng lượng Delta E = E_flipped - E_current = 2 * S_i * sum_neighbors
        // Gọi delta_E_half = S_i * sum_neighbors = Delta E / 2
        // delta_E_half nhận giá trị trong { -4, -2, 0, 2, 4 }.
        int delta_E_half = lattice[idx] * sum_neighbors;

        // Nếu năng lượng giảm hoặc bằng không, Delta E <= 0 <=> delta_E_half <= 0
        if (delta_E_half <= 0) {
            lattice[idx] = -lattice[idx]; // Lật 100%
        } else {
            // Tra cứu xác suất:
            // Vì delta_E_half > 0, nó chỉ có thể là 2 hoặc 4.
            // Bảng LUT lưu: prob_lut[i] = exp(-4.0 * i / T) (nếu tính Delta_E = 4*i)
            // Wait, Delta E = 2 * delta_E_half.
            // Nếu delta_E_half = 2 -> Delta E = 4
            // Nếu delta_E_half = 4 -> Delta E = 8
            // Vậy bảng prob_lut[delta_E_half / 2] = prob_lut[1] hoặc prob_lut[2].
            float p = prob_lut[delta_E_half / 2]; 

            // Sinh số ngẫu nhiên r trong [0, 1)
            float r = (xorshift32_metro(rng_state) & 0xFFFFFF) / (float)0x1000000;
            if (r < p) {
                lattice[idx] = -lattice[idx]; // Lật do dao động nhiệt
            }
        }
    }
}

} // namespace algorithm
} // namespace dm
