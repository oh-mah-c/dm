#include "models/vision/ohm_q_nlos.h"
#include <cmath>

namespace dm {
namespace vision {

OhmQNLOS::OhmQNLOS() : speed_of_light(1.0f) {}

void OhmQNLOS::load_wall_measurements(const std::vector<WallMeasurement>& measurements) {
    wall_data = measurements;
}

void OhmQNLOS::initialize_hidden_volume(float size_x, float size_y, float size_z, float resolution) {
    // Khởi tạo không gian lưới Voxel 3D phía sau bức tường
    hidden_space.clear();
    for (float z = 0; z <= size_z; z += resolution) {
        for (float y = 0; y <= size_y; y += resolution) {
            for (float x = 0; x <= size_x; x += resolution) {
                hidden_space.emplace_back(x, y, z);
            }
        }
    }
}

void OhmQNLOS::propagate_waves_backward() {
    // Đảo nghịch Thời gian (Time-Reversal)
    // Với mỗi điểm đo trên tường, chúng ta phát ngược sóng cầu (spherical wave) vào không gian
    
    for (auto& voxel : hidden_space) {
        std::complex<float> total_wave(0, 0);
        
        for (const auto& wall_pt : wall_data) {
            // Tính khoảng cách từ Điểm trên tường đến Voxel trong góc khuất
            float dx = voxel.x - wall_pt.x;
            float dy = voxel.y - wall_pt.y;
            float dz = voxel.z; // Tường nằm ở z=0, vật thể ở z > 0
            float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
            
            // Thời gian ánh sáng đi (backward time)
            float t_travel = distance / speed_of_light;
            
            // Toán tử Time-Reversal: U(-t) = exp(-i * H * dt)
            // Trong đó H ~ tần số k = 2 * PI / lambda (Giả sử k=1.0)
            float k = 1.0f;
            
            // Hàm sóng đảo ngược (Time Reversal Conjugate)
            // Để có giao thoa tăng cường tại vị trí vật thể, pha truyền ngược phải triệt tiêu pha tới.
            std::complex<float> backward_wave = wall_pt.phase_state * std::polar(1.0f, -k * distance);
            
            // Chồng chất lượng tử (Superposition)
            // Trong mô hình Time-Reversal lý tưởng, ta bỏ qua suy giảm cường độ để tránh nhiễu do khoảng cách
            total_wave += backward_wave;
        }
        
        voxel.wave_amplitude = total_wave;
    }
}

void OhmQNLOS::reconstruct_hidden_object() const {
    // Tìm điểm Voxel có Giao thoa Tăng cường cực đại (Constructive Interference)
    float max_intensity = 0;
    HiddenVoxel best_voxel(0, 0, 0);
    
    for (const auto& voxel : hidden_space) {
        float intensity = std::norm(voxel.wave_amplitude); // Năng lượng = |Amplitude|^2
        if (intensity > max_intensity) {
            max_intensity = intensity;
            best_voxel = voxel;
        }
    }
    
    std::cout << "--- Ohm-Q NLOS: KẾT QUẢ QUÉT GÓC KHUẤT ---\n";
    if (max_intensity > 0) {
        std::cout << "Phát hiện Giao thoa Cộng hưởng cực đại!\n";
        std::cout << "Tọa độ Vật thể bị giấu: (X: " << best_voxel.x 
                  << ", Y: " << best_voxel.y 
                  << ", Z: " << best_voxel.z << ")\n";
        std::cout << "Năng lượng hội tụ: " << max_intensity << "\n";
    } else {
        std::cout << "Không tìm thấy vật thể nào (Chân không tuyệt đối).\n";
    }
    std::cout << "------------------------------------------\n";
}

} // namespace vision
} // namespace dm
