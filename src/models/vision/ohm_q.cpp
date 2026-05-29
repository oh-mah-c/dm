#include "models/vision/ohm_q.h"
#include <cmath>

namespace dm {
namespace vision {

OhmQVisionModel::OhmQVisionModel(int w, int h) 
    : width(w), height(h), 
      mla_engine({1, 16, 8, 4}), // Cấu hình MLA nhỏ gọn cho Pixel Attention
      qho_engine() {
    pixel_grid.resize(w * h);
}

void OhmQVisionModel::load_image(const std::vector<float>& image_data) {
    for (size_t i = 0; i < image_data.size() && i < pixel_grid.size(); ++i) {
        pixel_grid[i].intensity = image_data[i];
        pixel_grid[i].object_id = 0; // 0 = Background
    }
}

void OhmQVisionModel::extract_quantum_edges() {
    // Ohm-QRender: Biến cường độ sáng thành Trạng thái Sóng (Wave State)
    // Giả lập Giao thoa pha (Phase Interference)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int idx = y * width + x;
            float center = pixel_grid[idx].intensity;
            
            // Tính Gradient để tạo Pha
            float dx = pixel_grid[y * width + (x + 1)].intensity - center;
            float dy = pixel_grid[(y + 1) * width + x].intensity - center;
            
            float phase = std::atan2(dy, dx);
            float amplitude = center;
            
            // Gán Sóng Lượng tử
            pixel_grid[idx].wave_state = std::polar(amplitude, phase);
        }
    }
}

void OhmQVisionModel::apply_global_attention() {
    // Ohm-VideoMLA: Cho phép Pixel nhìn thấy toàn bộ lưới ảnh thông qua Vector Tiềm ẩn (Latent Vector)
    // Trong môi trường giả lập, chúng ta làm mịn biên độ dựa trên Attention giả lập
    for (auto& p : pixel_grid) {
        if (std::abs(p.wave_state) > 0.1f) {
            p.wave_state *= 1.1f; // Tăng cường tín hiệu nhờ Attention
        }
    }
}

void OhmQVisionModel::apply_birkhoff_noise_filter() {
    // Ohm-QBNF: Chuẩn hóa Birkhoff Lượng tử
    // Giả sử các pixel nhiễu có pha không đồng đều và biên độ thấp
    for (auto& p : pixel_grid) {
        float amp = std::abs(p.wave_state);
        // Ngưỡng nhiễu (Noise Threshold)
        if (amp < 0.5f) {
            // Triệt tiêu nhiễu phi tuyến bằng Biến đổi Unitary (Giả lập triệt tiêu)
            p.wave_state = 0.0f; 
        }
    }
}

void OhmQVisionModel::segment_objects() {
    // Ohm-Discrete-QHO: Nhóm các pixel vào các Hố Thế Năng (Energy Wells)
    // Dựa vào Pha (Phase) để phân biệt vật thể (Mỗi vật thể cộng hưởng ở một pha khác nhau)
    
    for (auto& p : pixel_grid) {
        float amp = std::abs(p.wave_state);
        if (amp > 0) {
            float phase = std::arg(p.wave_state);
            
            // Đưa pha về khoảng [0, 2PI]
            if (phase < 0) phase += 2 * M_PI;
            
            // Lượng tử hóa Pha thành các mức Năng lượng (Eigenvalues) 1, 2, 3...
            if (phase >= 0.0f && phase < 1.0f) {
                p.object_id = 1; // Vật thể 1
            } else if (phase >= 1.0f && phase < 3.0f) {
                p.object_id = 2; // Vật thể 2
            } else {
                p.object_id = 3; // Vật thể 3
            }
        } else {
            p.object_id = 0; // Background
        }
    }
}

void OhmQVisionModel::print_segmentation_map() const {
    std::cout << "--- Ohm-Q Segmentation Map (Pixel-Level) ---\n";
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int id = pixel_grid[y * width + x].object_id;
            if (id == 0) {
                std::cout << ". ";
            } else {
                std::cout << id << " ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "--------------------------------------------\n";
}

} // namespace vision
} // namespace dm
