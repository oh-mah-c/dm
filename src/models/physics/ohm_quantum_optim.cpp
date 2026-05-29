#include "models/physics/ohm_quantum_optim.h"
#include <torch/torch.h>

namespace dm {
namespace optim {

    torch::Tensor OhmQuantumOptimizer::step(torch::optim::Optimizer::LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure != nullptr) {
            {
                torch::AutoGradMode enable_grad(true);
                loss = closure();
            }
        }

        for (auto& group : param_groups_) {
            auto& options = static_cast<OhmQuantumOptimizerOptions&>(group.options());
            double jitter = options.jitter();
            double temperature = options.temperature();

            for (auto& p : group.params()) {
                if (!p.defined() || !p.grad().defined()) {
                    continue;
                }

                auto grad = p.grad();
                
                // 1. Sinh Nhiễu Lượng Tử (Noise Matrix N) từ phân bố Uniform [-jitter, jitter]
                auto noise = (torch::rand_like(p) * 2.0 - 1.0) * jitter;
                
                // 2. Xấp xỉ Năng lượng biến thiên (Taylor Expansion: Delta E = Gradient * Noise)
                // (Chuyển đổi bài toán chạy lại mô hình thành bài toán nhân ma trận cực nhanh)
                auto delta_e = grad * noise;

                // 3. Metropolis-Hastings Acceptance
                // Tính xác suất Boltzmann cho các hạt có delta_e > 0
                // Nếu delta_e <= 0, xác suất > 1 -> chắc chắn chấp nhận
                auto accept_prob = torch::exp(-delta_e / temperature);
                
                // Sinh ma trận tung đồng xu Uniform [0, 1]
                auto rand_val = torch::rand_like(p);
                
                // 4. Màng lọc Hầm lượng tử (Quantum Tunneling Mask)
                // True nếu hạt được phép chui qua hầm (hoặc trượt xuống dốc)
                auto accept_mask = rand_val < accept_prob;

                // 5. Cập nhật Weight: Chỉ cộng noise vào những hạt được chấp nhận
                p.add_(noise * accept_mask.to(p.dtype()));
            }
        }

        return loss;
    }

} // namespace optim
} // namespace dm
