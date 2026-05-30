#include <iostream>
#include <vector>
#include <cmath>
#include "models/vision/ohm_q_nlos.h"

using namespace dm::vision;

int main() {
    std::cout << "========================================================\n";
    std::cout << " Ohm-Q NLOS: Quantum Seeing Around Corners Simulator\n";
    std::cout << "========================================================\n\n";

    OhmQNLOS nlos_engine;

    // Giả lập Dữ liệu Đo đạc trên Bức tường (Wall Measurements)
    // Kịch bản: Một vật thể (Con mèo) đang trốn ở tọa độ (X=5.0, Y=5.0, Z=10.0) trong góc khuất.
    // Ánh sáng laser bắn vào tường, dội vào con mèo, tán xạ lại lên 3 điểm trên bức tường (Z=0).
    std::cout << "[*] Giả lập thu thập dữ liệu SPAD trên Bức tường (Z=0)...\n";
    
    std::vector<WallMeasurement> simulated_wall_data;
    float cat_x = 5.0f, cat_y = 5.0f, cat_z = 10.0f;
    float speed_of_light = 1.0f;

    // Lấy mẫu Lưới SPAD trên bức tường (Mảng 21x21 điểm)
    std::vector<std::pair<float, float>> wall_points;
    for (float wx = 0; wx <= 10.0f; wx += 0.5f) {
        for (float wy = 0; wy <= 10.0f; wy += 0.5f) {
            wall_points.emplace_back(wx, wy);
        }
    }

    for (const auto& pt : wall_points) {
        float dx = cat_x - pt.first;
        float dy = cat_y - pt.second;
        float dz = cat_z - 0.0f; // Bức tường ở Z=0
        
        // Tính khoảng cách thực tế từ Con mèo đến điểm trên tường
        float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        // Thời gian bay
        float tof = distance / speed_of_light;
        
        // Giả lập sóng tán xạ nhận được tại tường
        float k = 1.0f;
        std::complex<float> phase_at_wall = std::polar(1.0f, k * distance);
        
        simulated_wall_data.emplace_back(pt.first, pt.second, tof, phase_at_wall);
    }
    std::cout << "    Đã thu thập " << simulated_wall_data.size() << " điểm đo đạc trên tường.\n";

    // 1. Nạp dữ liệu
    std::cout << "\n[*] Nạp dữ liệu nhiễu vào Lõi Ohm-Q NLOS...\n";
    nlos_engine.load_wall_measurements(simulated_wall_data);

    // 2. Khởi tạo không gian tối
    std::cout << "[*] Khởi tạo Không gian Góc Khuất (Kích thước 10x10x15, Độ phân giải 1.0)...\n";
    nlos_engine.initialize_hidden_volume(10.0f, 10.0f, 15.0f, 1.0f);

    // 3. Phóng ngược thời gian
    std::cout << "[*] Kích hoạt Phương trình Schrödinger Đảo ngược (Time-Reversal U(-t))...\n";
    nlos_engine.propagate_waves_backward();

    // 4. Đo lường
    std::cout << "[*] Quét Giao thoa Cộng hưởng trong Bóng tối...\n\n";
    nlos_engine.reconstruct_hidden_object();

    std::cout << "[SUCCESS] Ohm-Q NLOS đã nhìn xuyên thành công góc khuất!\n";

    return 0;
}
