#pragma once

#include <torch/torch.h>
#include "algorithms/ohm_vacuum_kernel.h"

namespace dm {
namespace modules {

struct OhmVACUUMImpl : torch::nn::Module {
    int width_, height_;
    torch::Tensor phi_field_;
    uint32_t rng_state_;

    OhmVACUUMImpl(int width = 50, int height = 50, uint32_t seed = 42) 
        : width_(width), height_(height), rng_state_(seed) {
        
        // Cấp phát lưới Chân Không Lượng Tử
        // Khởi tạo trạng thái dao động ngẫu nhiên quanh 0
        phi_field_ = register_buffer("phi_field", torch::randn({height, width}, torch::kFloat32) * 0.1f);
    }

    void forward(float alpha, float beta, float quantum_jitter = 0.5f, int steps = 100) {
        float* phi_ptr = phi_field_.data_ptr<float>();

        for (int i = 0; i < steps; ++i) {
            dm::algorithm::quantum_vacuum_fluctuation_step(
                phi_ptr, width_, height_,
                alpha, beta, quantum_jitter, rng_state_
            );
        }
    }
    
    // Lấy Giá trị Kỳ vọng Chân không (Vacuum Expectation Value)
    float get_vacuum_expectation_value() {
        return phi_field_.mean().item<float>();
    }
    
    // Lấy Độ lớn tuyệt đối trung bình (để kiểm tra xem nó có rớt khỏi số 0 hay không)
    float get_absolute_expectation() {
        return phi_field_.abs().mean().item<float>();
    }
};

TORCH_MODULE(OhmVACUUM);

} // namespace modules
} // namespace dm
