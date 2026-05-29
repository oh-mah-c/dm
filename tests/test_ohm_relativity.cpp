#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include "modules/ohm_relativity.h"

using namespace dm::modules;

int main() {
    std::cout << "--- Testing Ohm-RELATIVITY (General Relativity Geodesic Simulator) ---\n";

    float mass = 1.0f; // Khối lượng Lỗ Đen (G=1, c=1)
    
    // Khởi tạo Lỗ Đen
    OhmRELATIVITY relativity(mass);

    // Chân trời Sự kiện (Event Horizon): r = 2M = 2.0
    // Vòng Ánh sáng (Photon Sphere): r = 3M = 3.0
    
    // Bắn 1 tia sáng sượt sát qua Vòng Ánh Sáng
    float r0 = 3.1f;       // Xuất phát sát vòng ánh sáng
    float dr_dphi0 = 0.0f; // Bắn tiếp tuyến

    std::cout << "[*] Photon fired at r = " << r0 << " with dr/dphi = " << dr_dphi0 << "\n";
    std::cout << "[*] Simulating RK45 with Adaptive Step-size...\n\n";

    auto result = relativity->forward(r0, dr_dphi0, 10000, 1e-11f);
    
    torch::Tensor r_hist = std::get<0>(result);
    torch::Tensor phi_hist = std::get<1>(result);
    torch::Tensor dphi_hist = std::get<2>(result);

    int steps = r_hist.size(0);
    std::cout << "Step\tRadius(r)\tAngle(phi)\tStep-Size(dphi)\n";
    
    bool adaptive_works = false;
    float min_dphi = 1.0f;
    float max_dphi = 0.0f;

    for (int i = 0; i < steps; i += std::max(1, steps / 10)) {
        float r = r_hist[i].item<float>();
        float phi = phi_hist[i].item<float>();
        float dphi = dphi_hist[i].item<float>();
        
        std::cout << i << "\t" << std::fixed << std::setprecision(4) 
                  << r << "\t\t" << phi << "\t\t" << dphi << "\n";
                  
        if (dphi < min_dphi) min_dphi = dphi;
        if (dphi > max_dphi) max_dphi = dphi;
    }

    std::cout << "\n--- Adaptive Step Analysis ---\n";
    std::cout << "Max Step-Size (Flat Space): " << max_dphi << "\n";
    std::cout << "Min Step-Size (Curved Space): " << min_dphi << "\n";

    if (max_dphi / min_dphi > 2.0f) {
        adaptive_works = true;
    }

    if (adaptive_works) {
        std::cout << "\nSUCCESS: Branch Predictor dynamically adjusted resolution based on spacetime curvature!\n";
    } else {
        std::cout << "\nFAILURE: Adaptive Step-size did not trigger effectively.\n";
    }

    return 0;
}
