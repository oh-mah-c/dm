#include <torch/torch.h>
#include <iostream>
#include "modules/ohm_vacuum.h"

using namespace dm::modules;

int main() {
    std::cout << "--- Testing Ohm-VACUUM (Lattice Quantum Field Theory) ---\n";

    // Khởi tạo Lưới lượng tử
    OhmVACUUM vacuum(50, 50, 42);

    std::cout << "\n[PHA 1] Unbroken Symmetry (Alpha > 0)\n";
    std::cout << "Hàm thế năng hình cái Bát. Không gian trống rỗng.\n";
    
    float alpha = 1.0f;
    float beta = 1.0f;
    float jitter = 0.1f;

    vacuum->forward(alpha, beta, jitter, 1000);
    
    float abs_vev_1 = vacuum->get_absolute_expectation();
    std::cout << "Absolute Vacuum Expectation Value: " << abs_vev_1 << "\n";
    
    if (abs_vev_1 < 0.2f) {
        std::cout << "-> Chân không dao động ổn định quanh mức 0.\n";
    }

    std::cout << "\n[PHA 2] Spontaneous Symmetry Breaking (Alpha < 0)\n";
    std::cout << "Toán học ép Alpha xuống Âm! Hàm thế năng biến thành Mũ Mexican Hat.\n";
    
    alpha = -2.0f; 
    
    std::cout << "Bơm Nhiễu Lượng Tử và chờ Không gian Sụp đổ...\n";
    vacuum->forward(alpha, beta, jitter, 1000);

    float abs_vev_2 = vacuum->get_absolute_expectation();
    float vev_2 = vacuum->get_vacuum_expectation_value();
    std::cout << "Absolute Vacuum Expectation Value: " << abs_vev_2 << "\n";
    std::cout << "Signed VEV (Polarization): " << vev_2 << "\n";

    if (abs_vev_2 > 0.8f) {
        std::cout << "\nSUCCESS: Spontaneous Symmetry Breaking achieved!\n";
        std::cout << "Chân không đã kết tủa giá trị liên tục! Trường Higgs đã ban phát Khối lượng cho Vũ trụ!\n";
    } else {
        std::cout << "\nFAILURE: Vacuum did not break symmetry.\n";
    }

    return 0;
}
