#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "models/physics/ohm_quantum_optim.h"

using namespace dm::optim;

// Hàm mục tiêu: Có một Local Minimum tại x=2 và Global Minimum tại x=-2
// f(x) = (x-2)^2 * (x+2)^2 + x
// Hàm này là một ví dụ kinh điển nơi Gradient Descent truyền thống bị kẹt.
torch::Tensor loss_function(torch::Tensor x) {
    return torch::pow(x - 2.0, 2) * torch::pow(x + 2.0, 2) + x;
}

int main() {
    std::cout << "--- Testing Ohm-QuantumOptimizer (MCMC Taylor Approximation) ---\n";
    
    // Khởi tạo trọng số (Weight) ban đầu tại x = 3.0 (Rất gần Local Minima tại x = 2)
    torch::Tensor w_sgd = torch::tensor({3.0}, torch::requires_grad(true));
    torch::Tensor w_quantum = torch::tensor({3.0}, torch::requires_grad(true));

    // SGD truyền thống
    torch::optim::SGD optim_sgd({w_sgd}, torch::optim::SGDOptions(0.01));
    
    // Quantum Optimizer (Jitter mạnh để vượt hố, Nhiệt độ cao để dễ chui hầm)
    OhmQuantumOptimizer optim_quantum({w_quantum}, OhmQuantumOptimizerOptions(0.5, 5.0));

    std::cout << "\n[1] Running Traditional SGD (Gradient Descent)...\n";
    for (int i = 0; i < 500; ++i) {
        optim_sgd.zero_grad();
        auto loss = loss_function(w_sgd);
        loss.backward();
        optim_sgd.step();
    }
    std::cout << "SGD Final Weight (Trapped in Local Minimum): " << w_sgd.item<float>() << "\n";

    std::cout << "\n[2] Running Ohm-QuantumOptimizer (MCMC Lọc Lượng Tử)...\n";
    for (int i = 0; i < 500; ++i) {
        optim_quantum.zero_grad();
        auto loss = loss_function(w_quantum);
        loss.backward();
        optim_quantum.step();
    }
    std::cout << "Quantum Final Weight (Escaped to Global Minimum): " << w_quantum.item<float>() << "\n";

    if (w_sgd.item<float>() > 0.0 && w_quantum.item<float>() < 0.0) {
        std::cout << "\nSUCCESS: Quantum Optimizer successfully tunneled through the energy barrier!\n";
        std::cout << "Trong khi SGD khóc thét dưới đáy giếng (x ~ 1.96),\n";
        std::cout << "Thuật toán Lượng tử đã 'sôi sục' bay qua đỉnh đồi để tìm thấy thung lũng tuyệt đối (x ~ -2.0)!\n";
    } else {
        std::cout << "\nFAILURE: Quantum Optimizer failed to escape.\n";
    }

    return 0;
}
