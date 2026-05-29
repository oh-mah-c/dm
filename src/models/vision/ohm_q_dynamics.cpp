#include "models/vision/ohm_q_dynamics.h"
#include <cmath>

namespace dm {
namespace vision {

void OhmQTimeEvolution::propagate_forward(QObject& obj) {
    // Phương trình Schrödinger: |psi(t+dt)> = exp(-i * H * dt) * |psi(t)>
    // Giả lập Hamiltonian H tỷ lệ thuận với Động lượng (Momentum)
    float H = obj.momentum; 
    
    // Tính pha xoay (Phase rotation) exp(-i * H * dt)
    std::complex<float> evolution_operator = std::polar(1.0f, -H * dt);
    
    // Cập nhật Trạng thái Sóng
    obj.state_vector *= evolution_operator;
    
    // Cập nhật Vị trí Không gian (Zero-shot Tracking)
    // Vận tốc v = H (trong hệ thống lượng tử chuẩn hóa c = 1)
    obj.position_x += obj.momentum * dt;
}

bool OhmQEntanglement::measure_entanglement(const QObject& obj_A, const QObject& obj_B, QObject& composite_obj) {
    // Đo độ lệch động lượng giữa 2 vật thể
    float momentum_diff = std::abs(obj_A.momentum - obj_B.momentum);
    
    // Nếu động lượng gần bằng nhau (di chuyển cùng nhau) -> Vướng mắc Lượng tử (Entanglement)
    if (momentum_diff < momentum_threshold) {
        // Tích Tensor (Kronecker Product) của 2 hàm sóng
        // Trong mô hình thu gọn này, tích tensor của 2 số phức là tích đại số của chúng 
        // (đại diện cho Amplitude nhân nhau, Phase cộng nhau)
        std::complex<float> entangled_state = obj_A.state_vector * obj_B.state_vector;
        
        // Vị trí trung tâm khối lượng (Center of Mass)
        float center_pos = (obj_A.position_x + obj_B.position_x) / 2.0f;
        
        // Khởi tạo Siêu Vật Thể (Composite Object)
        composite_obj = QObject(
            obj_A.object_id * 100 + obj_B.object_id, // ID kết hợp
            obj_A.class_name + "-AND-" + obj_B.class_name, // Class kết hợp
            center_pos,
            (obj_A.momentum + obj_B.momentum) / 2.0f, // Động lượng kết hợp
            entangled_state
        );
        return true;
    }
    return false;
}

} // namespace vision
} // namespace dm
