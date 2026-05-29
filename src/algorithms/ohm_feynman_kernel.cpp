#include "algorithms/ohm_feynman_kernel.h"
#include <cmath>
#include <cstdint>

namespace dm {
namespace algorithm {

// Bộ sinh số ngẫu nhiên bạo lực bằng thao tác Bit (Zero-overhead)
inline uint32_t xorshift32(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

// Lõi mô phỏng Tích phân đường Lượng tử
void feynman_path_integral_cpu(
    int num_paths, int steps_per_path, 
    float start_x, float end_x, 
    float* prob_real, float* prob_imag
) {
    uint32_t rng_state = 1337; // Seed
    float total_R = 0.0f;
    float total_I = 0.0f;
    const float hbar = 1.0f; // Đơn vị tự nhiên

    // Vòng lặp bắn hàng triệu con đường (Path Generation)
    // CPU đa nhân có thể chia nhỏ vòng for này (OpenMP/SIMD) chạy cực cháy!
    // TODO: Thêm #pragma omp parallel for reduction(+:total_R, total_I) để CPU gầm lên
    for (int p = 0; p < num_paths; ++p) {
        float current_x = start_x;
        float action_S = 0.0f;
        
        // Mô phỏng từng bước đi của 1 con đường (Random Walk)
        for (int s = 0; s < steps_per_path; ++s) {
            // Ép bit ngẫu nhiên thành float [-1.0, 1.0]
            float noise = ((xorshift32(rng_state) % 2000) - 1000) / 1000.0f;
            float next_x = current_x + noise;
            
            // Tính Động năng và Thế năng (Ví dụ: Thế năng dao động điều hòa V = 0.5 * x^2)
            float velocity = (next_x - current_x);
            float kinetic = 0.5f * velocity * velocity;
            float potential = 0.5f * current_x * current_x;
            
            action_S += (kinetic - potential); // Tích phân Tác dụng (Action)
            current_x = next_x;
        }
        
        // Ép điểm cuối phải về đúng target B
        action_S += 0.5f * (end_x - current_x) * (end_x - current_x);

        // Cộng dồn Biên độ phức (Euler's formula)
        // Đây là chỗ quyết định giao thoa lượng tử!
        total_R += std::cos(action_S / hbar);
        total_I += std::sin(action_S / hbar);
    }

    *prob_real = total_R / (float)num_paths; // Chuẩn hóa
    *prob_imag = total_I / (float)num_paths; // Chuẩn hóa
}

} // namespace algorithm
} // namespace dm
