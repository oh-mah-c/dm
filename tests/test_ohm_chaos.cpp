#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "modules/ohm_chaos.h"

using namespace dm::modules;

int main() {
    std::cout << "--- Testing Ohm-CHAOS (Butterfly Effect Simulator) ---\n";

    // Khởi tạo 2 hạt song sinh (twin particles) lệch nhau 1e-14
    OhmCHAOS chaos(2, 1.0, 1.0, 1.0);

    std::cout << "[*] Twin Particles initialized with 1e-14 perturbation in X.\n";
    std::cout << "[*] Simulating Lorenz Attractor (Float64 SIMD RK4)...\n\n";

    double dt = 0.005;
    int max_steps = 10000;
    
    std::cout << "Step\tDistance(P1, P2)\tP1_X\t\tP2_X\n";

    bool chaos_triggered = false;
    
    for (int i = 0; i <= max_steps; ++i) {
        if (i % 1000 == 0) {
            auto x_acc = chaos->x_.accessor<double, 1>();
            auto y_acc = chaos->y_.accessor<double, 1>();
            auto z_acc = chaos->z_.accessor<double, 1>();

            double dx = x_acc[0] - x_acc[1];
            double dy = y_acc[0] - y_acc[1];
            double dz = z_acc[0] - z_acc[1];
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz);

            std::cout << i << "\t" << std::scientific << std::setprecision(6) 
                      << dist << "\t\t" << std::fixed << x_acc[0] << "\t" << x_acc[1] << "\n";
            
            // Nếu khoảng cách bị xé toạc lên mức vĩ mô (> 1.0)
            if (dist > 1.0) {
                chaos_triggered = true;
            }
        }
        
        chaos->forward(dt);
    }

    if (chaos_triggered) {
        std::cout << "\nSUCCESS: Exponential Divergence observed! The Butterfly Effect tore the universes apart!\n";
    } else {
        std::cout << "\nFAILURE: The particles stayed too close. Chaos not achieved.\n";
    }

    return 0;
}
