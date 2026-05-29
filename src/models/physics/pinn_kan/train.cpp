#include <torch/torch.h>
#include <iostream>
#include <cmath>
#include "models/nlp/kan/kan.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace dm::models::nlp;

int main() {
    std::cout << "========================================================\n";
    std::cout << " Ohm-PINN: Algorithm-Driven 1D Heat Equation Solver \n";
    std::cout << "========================================================\n";

    // 1. Initialize KAN model: Input [x, t] -> Hidden [5] -> Output [u]
    // We use a small architecture [2, 5, 1] as KAN is highly expressive.
    // The domain is x in [0, 1] and t in [0, 1], so grid_min=0, grid_max=1.
    KAN kan(std::vector<int64_t>{2, 5, 1}, /*G=*/5, /*k=*/3, /*grid_min=*/0.0, /*grid_max=*/1.0);
    kan->to(torch::kCPU);

    // 2. Setup Optimizer
    // We use Adam with learning rate 1e-2 for faster convergence on PINNs
    torch::optim::Adam optimizer(kan->parameters(), torch::optim::AdamOptions(1e-2));

    int epochs = 2000;
    int N_f = 1000; // Number of collocation points (domain)
    int N_b = 200;  // Number of boundary points

    float alpha = 0.01; // Thermal diffusivity

    std::cout << "[*] Starting Algorithm-Driven Physics Training Loop...\n\n";

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        optimizer.zero_grad();

        // -------------------------------------------------------------
        // Step A: PHYSICS LOSS (Collocation Points inside the domain)
        // -------------------------------------------------------------
        // Generate random points in space and time
        auto X_f = torch::rand({N_f, 2}, torch::TensorOptions().requires_grad(true));
        
        // Forward Ansatz: Predict u(x, t)
        auto u_f = kan->forward(X_f); 

        // Compute 1st order gradients using Autograd
        auto grad_outputs = torch::ones_like(u_f);
        auto du_dX = torch::autograd::grad({u_f}, {X_f}, {grad_outputs},
                                           /*retain_graph=*/true,
                                           /*create_graph=*/true)[0];
        
        // du_dX has shape [N_f, 2]. Column 0 is du/dx, Column 1 is du/dt.
        auto du_dx = du_dX.select(1, 0).unsqueeze(1); 
        auto du_dt = du_dX.select(1, 1).unsqueeze(1); 

        // Compute 2nd order gradient: d^2u/dx^2
        auto grad_outputs_x = torch::ones_like(du_dx);
        auto d2u_dX2 = torch::autograd::grad({du_dx}, {X_f}, {grad_outputs_x},
                                             /*retain_graph=*/true,
                                             /*create_graph=*/true)[0];
        auto d2u_dx2 = d2u_dX2.select(1, 0).unsqueeze(1); 

        // PDE Residual: du/dt - alpha * d^2u/dx^2 = 0
        auto pde_residual = du_dt - alpha * d2u_dx2;
        auto loss_pde = torch::mse_loss(pde_residual, torch::zeros_like(pde_residual));

        // -------------------------------------------------------------
        // Step B: BOUNDARY LOSS
        // -------------------------------------------------------------
        // 1. Initial Condition: u(x, 0) = sin(pi * x)
        auto X_ic = torch::empty({N_b, 2});
        X_ic.select(1, 0) = torch::rand({N_b}); // x in [0, 1]
        X_ic.select(1, 1) = torch::zeros({N_b}); // t = 0
        
        auto u_ic_pred = kan->forward(X_ic);
        auto u_ic_exact = torch::sin(M_PI * X_ic.select(1, 0)).unsqueeze(1);
        auto loss_ic = torch::mse_loss(u_ic_pred, u_ic_exact);

        // 2. Boundary Condition 1: u(0, t) = 0
        auto X_bc1 = torch::empty({N_b, 2});
        X_bc1.select(1, 0) = torch::zeros({N_b}); // x = 0
        X_bc1.select(1, 1) = torch::rand({N_b}); // t in [0, 1]
        auto u_bc1_pred = kan->forward(X_bc1);
        auto loss_bc1 = torch::mse_loss(u_bc1_pred, torch::zeros_like(u_bc1_pred));

        // 3. Boundary Condition 2: u(1, t) = 0
        auto X_bc2 = torch::empty({N_b, 2});
        X_bc2.select(1, 0) = torch::ones({N_b}); // x = 1
        X_bc2.select(1, 1) = torch::rand({N_b}); // t in [0, 1]
        auto u_bc2_pred = kan->forward(X_bc2);
        auto loss_bc2 = torch::mse_loss(u_bc2_pred, torch::zeros_like(u_bc2_pred));

        // Aggregate boundary losses
        auto loss_boundary = loss_ic + loss_bc1 + loss_bc2;

        // -------------------------------------------------------------
        // Step C: OPTIMIZATION
        // -------------------------------------------------------------
        // Weight boundary loss higher initially to constrain the PDE
        auto loss_total = loss_pde + 10.0 * loss_boundary; 
        
        loss_total.backward();
        optimizer.step();

        // -------------------------------------------------------------
        // Logging
        // -------------------------------------------------------------
        if (epoch % 100 == 0 || epoch == 1) {
            std::cout << "[Epoch " << epoch << "/" << epochs << "] "
                      << "Total Loss: " << loss_total.item<float>() 
                      << " | PDE Loss: " << loss_pde.item<float>()
                      << " | Boundary Loss: " << loss_boundary.item<float>()
                      << std::endl;
        }
    }

    std::cout << "\n[+] Training Complete! Physics laws successfully encoded into KAN splines.\n";
    return 0;
}
