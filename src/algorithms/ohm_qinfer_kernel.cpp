#include "algorithms/ohm_qinfer_kernel.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace dm {
namespace algorithm {

    OhmQInferArena::OhmQInferArena(int n_universes, int g_size) 
        : num_universes(n_universes), grid_size(g_size), rng_state(123456789) {
        
        // Cấp phát mảng tĩnh khổng lồ (Zero-Allocation during runtime)
        V_arena.resize(num_universes * grid_size, 0.0f);
        psi_arena.resize(num_universes * grid_size, 0.0f);
        
        universes.resize(num_universes);
        
        for (int i = 0; i < num_universes; ++i) {
            universes[i].weight = 1.0f / num_universes;
            universes[i].V_barrier = &V_arena[i * grid_size];
            universes[i].psi_prob = &psi_arena[i * grid_size];
        }
    }

    float OhmQInferArena::xorshift32_float() {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 17;
        rng_state ^= rng_state << 5;
        // Trả về số thực [0.0, 1.0)
        return (rng_state & 0xFFFFFF) / (float)0x1000000;
    }

    // Mô phỏng 1 bước Schrodinger đơn giản hóa (Hoặc tính xác suất dựa trên V)
    // Để bài test nhanh, ta giả định xác suất tại màn hình tỷ lệ nghịch với V_barrier
    // Hạt sẽ có xu hướng đi qua các "khe" (nơi V_barrier thấp).
    void OhmQInferArena::compute_psi_prob(Universe& u) {
        float sum = 0.0f;
        for (int x = 0; x < grid_size; ++x) {
            // Giả định hạt đi từ nguồn tỏa ra. Nơi nào V cao thì sóng bị chặn (xác suất thấp).
            // Xác suất = exp(-V[x])
            u.psi_prob[x] = std::exp(-u.V_barrier[x]);
            sum += u.psi_prob[x];
        }
        
        // Chuẩn hóa tổng xác suất = 1.0
        if (sum > 0.0f) {
            for (int x = 0; x < grid_size; ++x) {
                u.psi_prob[x] /= sum;
            }
        }
    }

    void OhmQInferArena::init_random_universes() {
        for (int i = 0; i < num_universes; ++i) {
            universes[i].weight = 1.0f / num_universes;
            for (int x = 0; x < grid_size; ++x) {
                // Khởi tạo Bức tường ngẫu nhiên từ 0.0 đến 5.0
                universes[i].V_barrier[x] = xorshift32_float() * 5.0f;
            }
            compute_psi_prob(universes[i]);
        }
    }

    void OhmQInferArena::deduce_physics(int X_measured) {
        float sum_weights = 0.0f;

        // BƯỚC 1: TRỪNG PHẠT BẰNG ĐỊNH LÝ BAYES
        for (int i = 0; i < num_universes; ++i) {
            // Likelihood: Xác suất hạt rơi vào X_measured theo giả thuyết của vũ trụ này
            float predicted_prob = universes[i].psi_prob[X_measured];
            
            // Cập nhật niềm tin
            universes[i].weight *= (predicted_prob + 1e-6f); // Tránh weight = 0
            sum_weights += universes[i].weight;
        }

        // BƯỚC 2: CHUẨN HÓA
        if (sum_weights > 0.0f) {
            for (int i = 0; i < num_universes; ++i) {
                universes[i].weight /= sum_weights;
            }
        } else {
            // Nếu hỏng, reset
            for (int i = 0; i < num_universes; ++i) {
                universes[i].weight = 1.0f / num_universes;
            }
        }

        // BƯỚC 3: RESAMPLING (VÒNG QUAY ROULETTE) & MUTATION
        // Sao chép dữ liệu V_barrier sang một bộ nhớ đệm
        std::vector<float> V_next(num_universes * grid_size);

        for (int i = 0; i < num_universes; ++i) {
            float rand_spin = xorshift32_float(); 
            float cumulative = 0.0f;
            int chosen_idx = num_universes - 1; // Default fallback

            for (int j = 0; j < num_universes; ++j) {
                cumulative += universes[j].weight;
                if (cumulative >= rand_spin) {
                    chosen_idx = j;
                    break;
                }
            }

            // Clone vũ trụ được chọn (Memcpy siêu tốc)
            std::memcpy(&V_next[i * grid_size], universes[chosen_idx].V_barrier, grid_size * sizeof(float));

            // Đột biến (Mutation): Cộng nhiễu Gaussian để tiến hóa
            // Cứ 5 điểm thì đột biến 1 điểm để khám phá
            for (int x = 0; x < grid_size; ++x) {
                if (xorshift32_float() < 0.1f) {
                    float noise = (xorshift32_float() * 2.0f - 1.0f) * 0.5f; // [-0.5, 0.5]
                    V_next[i * grid_size + x] += noise;
                    if (V_next[i * grid_size + x] < 0.0f) V_next[i * grid_size + x] = 0.0f;
                }
            }
        }

        // Cập nhật lại mảng gốc và tính lại xác suất
        std::memcpy(V_arena.data(), V_next.data(), V_next.size() * sizeof(float));

        for (int i = 0; i < num_universes; ++i) {
            universes[i].weight = 1.0f / num_universes; // Reset trọng số sau resampling
            compute_psi_prob(universes[i]);
        }
    }

    const float* OhmQInferArena::get_best_V_barrier() const {
        // Tìm vũ trụ có V_barrier sát với thực tế nhất (Hoặc lấy mảng trung bình)
        // Vì tất cả đã được resampling, ta có thể lấy luôn Vũ trụ đầu tiên hoặc Trung bình.
        return universes[0].V_barrier;
    }

} // namespace algorithm
} // namespace dm
