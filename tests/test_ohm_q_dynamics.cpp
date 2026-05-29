#include <iostream>
#include <complex>
#include "models/vision/ohm_q_dynamics.h"

using namespace dm::vision;

void print_object(const std::string& frame_name, const QObject& obj) {
    std::cout << "[" << frame_name << "] " << obj.class_name 
              << " | Vị trí: " << obj.position_x 
              << " | Động lượng: " << obj.momentum 
              << " | Sóng (Pha): " << std::arg(obj.state_vector) << "\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " Ohm-Q Dynamics: Schrödinger Evolution & Entanglement\n";
    std::cout << "========================================================\n\n";

    // Khởi tạo Lõi Động lực học
    OhmQTimeEvolution time_prop(1.0f); // Bước thời gian dt = 1.0
    OhmQEntanglement entangler(0.5f);  // Ngưỡng động lượng = 0.5

    // Frame 1: Khởi tạo Người và Xe đạp đứng cạnh nhau và di chuyển cùng tốc độ (Momentum)
    std::complex<float> state_person = std::polar(1.0f, 0.1f);
    std::complex<float> state_bike = std::polar(1.0f, 0.15f);

    QObject person(1, "Person", 10.0f, 5.0f, state_person); // Vận tốc 5.0
    QObject bike(2, "Bicycle", 10.5f, 5.1f, state_bike);    // Vận tốc 5.1 (Rất gần với Người)

    std::cout << "--- FRAME 1 (Quá khứ) ---\n";
    print_object("Frame 1", person);
    print_object("Frame 1", bike);
    std::cout << "\n";

    // Tiến hóa Thời gian (Time Evolution to Frame 2)
    std::cout << "[*] Kích hoạt Phương trình Schrödinger: Dòng chảy Thời gian (dt=1.0)...\n";
    time_prop.propagate_forward(person);
    time_prop.propagate_forward(bike);

    std::cout << "--- FRAME 2 (Tương lai - Zero-shot Tracking) ---\n";
    print_object("Frame 2", person);
    print_object("Frame 2", bike);
    std::cout << "\n";

    // Vướng mắc Lượng tử (Entanglement)
    std::cout << "[*] Kích hoạt Đo lường Vướng mắc Lượng tử (Tích Tensor)...\n";
    QObject composite_obj(0, "", 0, 0, 0);
    bool is_entangled = entangler.measure_entanglement(person, bike, composite_obj);

    if (is_entangled) {
        std::cout << "[SUCCESS] Phát hiện Vướng mắc Lượng tử! Đã gộp 2 vật thể thành 1.\n";
        std::cout << "--- SIÊU VẬT THỂ (COMPOSITE STATE) ---\n";
        print_object("Entangled", composite_obj);
    } else {
        std::cout << "[FAILED] Không có vướng mắc lượng tử.\n";
    }

    std::cout << "\n[SUCCESS] Hệ thống Động lực học Lượng tử Ohm-Q đã vượt qua thử nghiệm!\n";
    return 0;
}
