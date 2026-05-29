#pragma once

#include <torch/torch.h>

namespace dm {
namespace optim {

    // Tham số khởi tạo cho Quantum Optimizer
    struct OhmQuantumOptimizerOptions : public torch::optim::OptimizerCloneableOptions<OhmQuantumOptimizerOptions> {
        OhmQuantumOptimizerOptions(double jitter = 0.01, double temperature = 0.1) 
            : jitter_(jitter), temperature_(temperature) {}
        
        TORCH_ARG(double, jitter) = 0.01;
        TORCH_ARG(double, temperature) = 0.1;
    };

    // Lớp Optimizer tùy chỉnh thay thế Adam/SGD
    class OhmQuantumOptimizer : public torch::optim::Optimizer {
    public:
        explicit OhmQuantumOptimizer(std::vector<torch::optim::OptimizerParamGroup> param_groups,
                                     OhmQuantumOptimizerOptions defaults = {})
            : Optimizer(std::move(param_groups), std::make_unique<OhmQuantumOptimizerOptions>(defaults)) {}

        explicit OhmQuantumOptimizer(std::vector<torch::Tensor> params,
                                     OhmQuantumOptimizerOptions defaults = {})
            : OhmQuantumOptimizer({torch::optim::OptimizerParamGroup(std::move(params))}, defaults) {}

        // Hàm step() sẽ chạy sau khi gọi loss.backward()
        torch::Tensor step(torch::optim::Optimizer::LossClosure closure = nullptr) override;
    };

} // namespace optim
} // namespace dm
