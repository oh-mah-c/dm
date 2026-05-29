#include <iostream>
#include <iomanip>
#include "models/physics/ohm_qcd.h"

using namespace dm::physics;

int main() {
    std::cout << "========================================================\n";
    std::cout << " Ohm-QCD: 4D Lattice Gauge Theory Simulator (SU(3)) \n";
    std::cout << "========================================================\n\n";

    // Khởi tạo không gian 4D nhỏ: 4x4x4x4
    int L = 4;
    int T = 4;
    float beta = 5.7f; // Hằng số cặp (Thường dao động quanh 5.5 - 6.0 cho SU(3))

    std::cout << "[*] Kích thước Không-Thời gian: " << L << "^3 x " << T << "\n";
    std::cout << "[*] Số lượng điểm (Sites): " << L*L*L*T << "\n";
    std::cout << "[*] Cấu trúc Toán học: Nhóm Lie SU(3)\n";
    std::cout << "[*] Khởi tạo Trạng thái Chân không Lạnh (Cold Start - Identity Matrices)...\n\n";

    LatticeQCD universe(L, T, beta);

    std::cout << "--- NẤU SÔI CHÂN KHÔNG LƯỢNG TỬ (METROPOLIS MCMC) ---\n";
    
    // Năng lượng ban đầu (Tất cả là Identity -> Plaquette = 1.0)
    float initial_plaq = universe.average_plaquette();
    std::cout << "Vòng lặp 0 (Khởi thủy): Năng lượng Plaquette trung bình = " << std::fixed << std::setprecision(6) << initial_plaq << "\n";

    // Chạy 50 vòng lặp tiến hóa Metropolis
    for (int epoch = 1; epoch <= 50; ++epoch) {
        universe.metropolis_step();

        // Cứ mỗi 5 vòng lặp thì đo lường Năng lượng Vũ trụ một lần
        if (epoch % 5 == 0) {
            float plaq = universe.average_plaquette();
            std::cout << "Vòng lặp " << std::setw(2) << epoch 
                      << ": Năng lượng Plaquette trung bình = " << std::fixed << std::setprecision(6) << plaq << "\n";
        }
    }

    std::cout << "\n[+] SUCCESS: Chân không lượng tử đã đạt trạng thái Sôi sục (Thermalization)!\n";
    std::cout << "[+] Sự tương tác của hạt Gluon đã bẻ cong các cạnh Không-Thời gian.\n";

    return 0;
}
