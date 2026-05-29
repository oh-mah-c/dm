#include "algorithms/ohm_genesis_kernel.h"

namespace dm {
namespace algorithm {

    void reaction_diffusion_step_cpu(
        const float* __restrict__ A_old, const float* __restrict__ B_old,
        float* __restrict__ A_new, float* __restrict__ B_new,
        int width, int height,
        float D_a, float D_b, float feed, float kill, float dt
    ) {
        // Vòng lặp bỏ qua viền (Tránh câu lệnh if rẽ nhánh bên trong vòng lặp)
        // CPU Branch Predictor sẽ nuốt trọn vòng lặp này!
        
        #pragma omp simd
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                
                int idx = y * width + x;
                
                float a = A_old[idx];
                float b = B_old[idx];

                // Tính Laplacian cho A
                float laplace_A = 
                    (A_old[idx - width] + A_old[idx + width] + 
                     A_old[idx - 1] + A_old[idx + 1]) * 0.2f 
                    + (A_old[idx - width - 1] + A_old[idx - width + 1] + 
                       A_old[idx + width - 1] + A_old[idx + width + 1]) * 0.05f 
                    - a;

                // Tính Laplacian cho B
                float laplace_B = 
                    (B_old[idx - width] + B_old[idx + width] + 
                     B_old[idx - 1] + B_old[idx + 1]) * 0.2f 
                    + (B_old[idx - width - 1] + B_old[idx - width + 1] + 
                       B_old[idx + width - 1] + B_old[idx + width + 1]) * 0.05f 
                    - b;

                // Phản ứng hóa học
                float reaction = a * b * b;

                // Euler Integration
                A_new[idx] = a + (D_a * laplace_A - reaction + feed * (1.0f - a)) * dt;
                B_new[idx] = b + (D_b * laplace_B + reaction - (kill + feed) * b) * dt;
            }
        }
    }

} // namespace algorithm
} // namespace dm
