#pragma once

namespace dm {
namespace algorithm {

    // Chạy N hạt song sinh (Vũ trụ kề nhau) trong 1 mảng tĩnh (Vectorized)
    // Sử dụng double (Float64) để tránh sai số làm tròn số sớm
    void lorenz_chaos_step_cpu(
        double* __restrict__ x, 
        double* __restrict__ y, 
        double* __restrict__ z, 
        int num_twins, 
        double sigma, double rho, double beta, double dt
    );

} // namespace algorithm
} // namespace dm
