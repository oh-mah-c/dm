#pragma once

#include <vector>
#include <complex>
#include <iostream>

namespace dm {
namespace vision {

// Điểm đo đạc trên Bức tường (Wall Measurement Point)
struct WallMeasurement {
    float x, y; // Tọa độ trên tường
    float time_of_flight; // Thời gian bay của photon từ lúc chớp Laser đến khi dội về
    std::complex<float> phase_state; // Trạng thái pha tán xạ đo được bằng SPAD
    
    WallMeasurement(float x_, float y_, float tof, std::complex<float> phase) 
        : x(x_), y(y_), time_of_flight(tof), phase_state(phase) {}
};

// Điểm Voxel trong không gian Khuất (Hidden Voxel Space)
struct HiddenVoxel {
    float x, y, z;
    std::complex<float> wave_amplitude; // Tổng biên độ sóng giao thoa tại điểm này
    
    HiddenVoxel(float x_, float y_, float z_) 
        : x(x_), y(y_), z(z_), wave_amplitude(0, 0) {}
};

// Ohm-Q NLOS: Hệ thống Nhìn Xuyên Góc Khuất
class OhmQNLOS {
private:
    std::vector<WallMeasurement> wall_data;
    std::vector<HiddenVoxel> hidden_space;
    float speed_of_light; // Tốc độ truyền sóng (Chuẩn hóa = 1.0 trong hệ tính toán này)

public:
    OhmQNLOS();

    // 1. Nạp dữ liệu các đốm sáng hình Elip đo được trên tường
    void load_wall_measurements(const std::vector<WallMeasurement>& measurements);

    // 2. Thiết lập Không gian lưới Voxel 3D cho vùng góc khuất
    void initialize_hidden_volume(float size_x, float size_y, float size_z, float resolution);

    // 3. Sử dụng Toán tử Time-Reversal U(-t) bắn ngược sóng về vùng khuất
    void propagate_waves_backward();

    // 4. Tìm kiếm điểm có Giao thoa cộng hưởng mạnh nhất (Vị trí thực sự của vật thể)
    void reconstruct_hidden_object() const;
};

} // namespace vision
} // namespace dm
