#pragma once

#include <vector>
#include <complex>
#include <iostream>
#include "models/nlp/ohm_mla.h"
#include "models/physics/ohm_discrete_qho.h"
#include "models/physics/ohm_qbnf.h"

namespace dm {
namespace vision {

using namespace dm::nlp::mla;
using namespace dm::physics::qbnf;
using namespace dm::physics::symbolic;

// Điểm ảnh Lượng tử (Quantum Pixel)
struct QPixel {
    float intensity; // Độ sáng gốc (Từ RGB)
    std::complex<float> wave_state; // Sóng lượng tử sau QRender (Biên độ + Pha)
    int object_id; // Nhãn đối tượng sau khi phân vùng (Segmentation)
};

// Ohm-Q: The Quantum Vision Model
class OhmQVisionModel {
private:
    int width;
    int height;
    std::vector<QPixel> pixel_grid;
    
    // Modules
    VideoMLAEngine mla_engine;
    dm::physics::DiscreteQHO5D qho_engine;

public:
    OhmQVisionModel(int w, int h);

    // 1. Khởi tạo Ma trận Điểm ảnh Lượng tử từ dữ liệu ảnh thô
    void load_image(const std::vector<float>& image_data);

    // 2. Trích xuất Pha (QRender QSobel Simulation)
    void extract_quantum_edges();

    // 3. Giao tiếp Toàn cục (VideoMLA Simulation)
    void apply_global_attention();

    // 4. Lọc Nhiễu bằng Chuẩn hóa Birkhoff (QBNF Simulation)
    void apply_birkhoff_noise_filter();

    // 5. Phân vùng Vật thể bằng Hố thế năng (QHO Segmentation)
    void segment_objects();
    
    // Hỗ trợ in ra Bản đồ Phân vùng (Segmentation Map)
    void print_segmentation_map() const;
};

} // namespace vision
} // namespace dm
