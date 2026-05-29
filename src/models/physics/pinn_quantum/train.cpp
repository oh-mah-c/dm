#include <torch/torch.h>
#include <iostream>
#include <vector>
#include "models/nlp/kan/kan.h"
#include "models/physics/ohm_quantum_optim.h"

using namespace dm::models::nlp;
using namespace dm::optim;

// Tính đạo hàm bằng Autograd
torch::Tensor compute_gradient(torch::Tensor y, torch::Tensor x) {
    auto grad_outputs = torch::ones_like(y);
    auto grad = torch::autograd::grad({y}, {x}, {grad_outputs}, 
                                      /*retain_graph=*/true, 
                                      /*create_graph=*/true)[0];
    return grad;
}

int main() {
    std::cout << "--- Ohm-PINN: Luyện Kim Lượng Tử bằng OhmQuantumOptimizer ---\n";
    std::cout << "[*] Mục tiêu: Giải Phương trình Dao động Điều hòa 1D (Harmonic Oscillator)\n";
    std::cout << "    d^2x/dt^2 + w^2 x = 0  với w = 2.0\n";
    
    // Mạng KAN [1, 10, 1] nhận t, trả về x(t)
    std::vector<int64_t> widths = {1, 10, 1};
    KAN model(widths);

    // Kích hoạt Ohm-QuantumOptimizer! (Jitter = 0.05, Temperature = 0.1)
    OhmQuantumOptimizer optim(model->parameters(), OhmQuantumOptimizerOptions(0.05, 0.1));

    float omega = 2.0f;
    int epochs = 1000;

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        optim.zero_grad();

        // 1. SINH "DỮ LIỆU" TỪ CHÂN KHÔNG (Collocation Points)
        // Tạo 100 điểm thời gian ngẫu nhiên t trong khoảng [0, 2pi]
        auto t = torch::rand({100, 1}, torch::requires_grad(true)) * 2 * M_PI;
        
        // 2. DỰ ĐOÁN
        auto x_pred = model->forward(t);

        // 3. TÍNH ĐẠO HÀM VẬT LÝ (Autograd)
        auto dx_dt = compute_gradient(x_pred, t);
        auto d2x_dt2 = compute_gradient(dx_dt, t);

        // 4. HÀM MẤT MÁT VẬT LÝ (Physics Loss)
        // Phương trình vi phân: d^2x/dt^2 + w^2 x = 0
        auto ode_loss = torch::mse_loss(d2x_dt2 + omega * omega * x_pred, torch::zeros_like(x_pred));

        // Điều kiện biên/Ban đầu (Boundary Conditions)
        // Tại t = 0, x(0) = 1.0 (Kéo lò xo ra vị trí 1.0)
        // Tại t = 0, v(0) = dx/dt(0) = 0.0 (Thả nhẹ)
        auto t0 = torch::zeros({1, 1}, torch::requires_grad(true));
        auto x0_pred = model->forward(t0);
        auto dx0_dt_pred = compute_gradient(x0_pred, t0);
        
        auto ic_loss = torch::mse_loss(x0_pred, torch::ones_like(x0_pred)) + 
                       torch::mse_loss(dx0_dt_pred, torch::zeros_like(dx0_dt_pred));

        auto total_loss = ode_loss + ic_loss;

        // 5. TRUYỀN NGƯỢC (Backward) - Lấy Force
        total_loss.backward();

        // 6. XUYÊN HẦM LƯỢNG TỬ (Quantum MCMC Update)
        optim.step();

        if (epoch % 100 == 0) {
            std::cout << "Epoch [" << epoch << "/" << epochs << "] "
                      << "Total Loss: " << total_loss.item<float>() 
                      << " (ODE: " << ode_loss.item<float>() << ", IC: " << ic_loss.item<float>() << ")\n";
        }
    }

    std::cout << "\n[+] Quá trình Luyện Kim Lượng Tử Hoàn Tất!\n";
    std::cout << "AI đã bẻ khóa Phương trình Vi phân thuần túy bằng Nhiễu Lượng Tử mà KHÔNG cần dữ liệu nhãn!\n";

    return 0;
}
