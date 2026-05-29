#pragma once

#include <torch/torch.h>
#include "algorithms/ohm_chaos_kernel.h"
#include <vector>

namespace dm {
namespace modules {

struct OhmCHAOSImpl : torch::nn::Module {
    int num_twins_;
    double sigma_, rho_, beta_;

    torch::Tensor x_;
    torch::Tensor y_;
    torch::Tensor z_;

    // Khởi tạo hệ thống Hỗn độn với N "vũ trụ song sinh"
    // Mỗi vũ trụ chỉ lệch nhau 1e-14
    OhmCHAOSImpl(int num_twins = 2, double initial_x = 1.0, double initial_y = 1.0, double initial_z = 1.0) 
        : num_twins_(num_twins), sigma_(10.0), rho_(28.0), beta_(8.0/3.0) {
        
        // Cấp phát Tensor Float64 (Double)
        x_ = register_buffer("x", torch::full({num_twins}, initial_x, torch::kFloat64));
        y_ = register_buffer("y", torch::full({num_twins}, initial_y, torch::kFloat64));
        z_ = register_buffer("z", torch::full({num_twins}, initial_z, torch::kFloat64));

        // Cộng nhiễu Cánh Bướm (1e-14) vào hạt thứ 2 trở đi
        if (num_twins > 1) {
            auto x_acc = x_.accessor<double, 1>();
            for (int i = 1; i < num_twins; ++i) {
                x_acc[i] += i * 1e-14;
            }
        }
    }

    void forward(double dt = 0.01) {
        double* x_ptr = x_.data_ptr<double>();
        double* y_ptr = y_.data_ptr<double>();
        double* z_ptr = z_.data_ptr<double>();

        dm::algorithm::lorenz_chaos_step_cpu(
            x_ptr, y_ptr, z_ptr, num_twins_, sigma_, rho_, beta_, dt
        );
    }
};

TORCH_MODULE(OhmCHAOS);

} // namespace modules
} // namespace dm
