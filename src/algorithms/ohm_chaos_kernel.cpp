#include "algorithms/ohm_chaos_kernel.h"

namespace dm {
namespace algorithm {

    void lorenz_chaos_step_cpu(
        double* __restrict__ x, 
        double* __restrict__ y, 
        double* __restrict__ z, 
        int num_twins, 
        double sigma, double rho, double beta, double dt
    ) {
        // Pragma ép trình biên dịch (GCC/Clang) tự động Vector hóa (Auto-Vectorization)
        // Nó sẽ tự động cuộn vòng lặp và dùng tập lệnh AVX2 (hoặc AVX-512 nếu có flag)
        // Đây là cách an toàn và tối ưu nhất để dùng SIMD mà không phụ thuộc cứng vào phần cứng.
        
        #pragma omp simd
        for (int i = 0; i < num_twins; ++i) {
            double xi = x[i];
            double yi = y[i];
            double zi = z[i];

            // Phương trình Lorenz (Dùng Runge-Kutta bậc 4 để đảm bảo sai số tích phân tối thiểu)
            // Tuy nhiên, để biểu diễn sức mạnh SIMD bạo lực, ta tính RK4 cho N hạt cùng lúc
            
            // k1
            double k1_dx = sigma * (yi - xi);
            double k1_dy = xi * (rho - zi) - yi;
            double k1_dz = xi * yi - beta * zi;

            // k2
            double x2 = xi + 0.5 * dt * k1_dx;
            double y2 = yi + 0.5 * dt * k1_dy;
            double z2 = zi + 0.5 * dt * k1_dz;
            double k2_dx = sigma * (y2 - x2);
            double k2_dy = x2 * (rho - z2) - y2;
            double k2_dz = x2 * y2 - beta * z2;

            // k3
            double x3 = xi + 0.5 * dt * k2_dx;
            double y3 = yi + 0.5 * dt * k2_dy;
            double z3 = zi + 0.5 * dt * k2_dz;
            double k3_dx = sigma * (y3 - x3);
            double k3_dy = x3 * (rho - z3) - y3;
            double k3_dz = x3 * y3 - beta * z3;

            // k4
            double x4 = xi + dt * k3_dx;
            double y4 = yi + dt * k3_dy;
            double z4 = zi + dt * k3_dz;
            double k4_dx = sigma * (y4 - x4);
            double k4_dy = x4 * (rho - z4) - y4;
            double k4_dz = x4 * y4 - beta * z4;

            // Cập nhật tọa độ
            x[i] = xi + (dt / 6.0) * (k1_dx + 2.0*k2_dx + 2.0*k3_dx + k4_dx);
            y[i] = yi + (dt / 6.0) * (k1_dy + 2.0*k2_dy + 2.0*k3_dy + k4_dy);
            z[i] = zi + (dt / 6.0) * (k1_dz + 2.0*k2_dz + 2.0*k3_dz + k4_dz);
        }
    }

} // namespace algorithm
} // namespace dm
