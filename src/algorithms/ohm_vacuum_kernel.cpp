#include "algorithms/ohm_vacuum_kernel.h"
#include <cmath>

namespace dm {
namespace algorithm {

    void quantum_vacuum_fluctuation_step(
        float* phi_field, int width, int height,
        float alpha, float beta, float quantum_jitter, uint32_t& rng_state
    ) {
        int total_sites = width * height;

        // Vòng lặp Metropolis chạy tuyến tính
        for (int i = 0; i < total_sites; ++i) {
            
            // 1. Lấy tọa độ 2D từ Index 1D
            int x = i % width;
            int y = i / width;

            float current_phi = phi_field[i];

            // 2. Bơm Nhiễu Lượng Tử (Random Walk)
            // Sinh số ngẫu nhiên liên tục [-1.0, 1.0]
            float noise = ((xorshift32(rng_state) % 2000) - 1000) / 1000.0f;
            float proposed_phi = current_phi + noise * 0.5f;

            // 3. Tính Hamiltonian O(1) với Điều kiện biên tuần hoàn (Periodic Boundary Conditions)
            int up    = (y == 0) ? i + (height - 1) * width : i - width;
            int down  = (y == height - 1) ? i - (height - 1) * width : i + width;
            int left  = (x == 0) ? i + (width - 1) : i - 1;
            int right = (x == width - 1) ? i - (width - 1) : i + 1;

            float sum_neighbors = phi_field[up] + phi_field[down] + phi_field[left] + phi_field[right];

            // Lambda tính tổng năng lượng cục bộ (Động năng + Thế năng)
            auto calc_energy = [&](float phi) {
                // Động năng không gian (Gradient/Tension)
                // Năng lượng cục bộ của phi_i: 2 * phi^2 - phi * sum(neighbors)
                float gradient_energy = 2.0f * phi * phi - phi * sum_neighbors;
                
                // Thế năng phi^4 (Mexican Hat Potential)
                float potential_energy = alpha * phi * phi + beta * phi * phi * phi * phi;
                
                return gradient_energy + potential_energy;
            };

            float E_old = calc_energy(current_phi);
            float E_new = calc_energy(proposed_phi);
            float delta_E = E_new - E_old;

            // 4. Metropolis Acceptance (Xác suất chui qua hầm)
            if (delta_E <= 0.0f) {
                // Trạng thái năng lượng thấp hơn -> Chấp nhận ngay
                phi_field[i] = proposed_phi;
            } else {
                // Trạng thái năng lượng cao hơn -> Chấp nhận dựa trên phân bố Boltzmann
                float accept_prob = std::exp(-delta_E / quantum_jitter);
                float rand_val = (xorshift32(rng_state) % 10000) / 10000.0f; // [0.0, 1.0]
                
                if (rand_val < accept_prob) {
                    phi_field[i] = proposed_phi; 
                }
            }
        }
    }

} // namespace algorithm
} // namespace dm
