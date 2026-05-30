#include "models/vision/ohm_q_hybrid.h"

namespace dm {
namespace vision {

OhmQHybrid::OhmQHybrid() : full_width(0), full_height(0) {}

void OhmQHybrid::run_classical_coarse_pass(const std::vector<float>& full_image, int w, int h) {
    full_width = w;
    full_height = h;
    coarse_map = full_image; // Giữ lại ảnh gốc
    
    // Giả lập Classical Coarse Pass (VD: Ngưỡng sáng đơn giản)
    // Các vùng tối sẽ bị mờ đi (nhiễu), vùng sáng sẽ giữ nguyên. Đây là cách làm rẻ tiền, độ chính xác thấp.
    for (auto& val : coarse_map) {
        if (val < 0.3f) val = 0.0f; 
        else if (val > 0.8f) val = 1.0f;
        else val = 0.5f; // Vùng mập mờ (Coarse Uncertainty)
    }
}

void OhmQHybrid::run_quantum_roi_refinement(const std::vector<float>& full_image, const BoundingBox& roi) {
    // 1. Trích xuất Dữ liệu ảnh thô (Raw Data) CHỈ NẰM TRONG ROI
    std::vector<float> roi_data;
    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            int idx = y * full_width + x;
            roi_data.push_back(full_image[idx]);
        }
    }

    // 2. Khởi tạo Ma trận Lượng tử (QUBO Size reduction)
    // Thay vì chạy Quantum cho full_width * full_height (Rất đắt đỏ)
    // Chúng ta chỉ chạy cho roi.width * roi.height!
    OhmQVisionModel quantum_roi_engine(roi.width, roi.height);
    quantum_roi_engine.load_image(roi_data);

    // 3. Thực thi Pipeline Lượng tử tối thượng (QRender -> QBNF -> QHO)
    quantum_roi_engine.extract_quantum_edges();
    quantum_roi_engine.apply_global_attention();
    quantum_roi_engine.apply_birkhoff_noise_filter();
    quantum_roi_engine.segment_objects();

    // 4. Ghép nối (Merge) Bản vá Lượng tử (Quantum Patch) đè lên Bản đồ Cổ điển
    // Lưu ý: Đoạn mã này chỉ mang tính chất mô phỏng trong môi trường C++ terminal.
    // Trong thực tế, quantum_roi_engine sẽ nhả ra một Quantum Segmentation Map.
    // Giả lập việc đắp bản vá bằng cách gán giá trị đặc biệt (vd: 9.9f) cho vùng ROI.
    for (int y = roi.y; y < roi.y + roi.height; ++y) {
        for (int x = roi.x; x < roi.x + roi.width; ++x) {
            int idx = y * full_width + x;
            // Đánh dấu đây là vùng đã được Refined bằng Lượng tử
            coarse_map[idx] = 9.9f; 
        }
    }
}

void OhmQHybrid::print_hybrid_map() const {
    std::cout << "\n--- OHM-Q HYBRID RECONSTRUCTION MAP ---\n";
    for (int y = 0; y < full_height; ++y) {
        for (int x = 0; x < full_width; ++x) {
            int idx = y * full_width + x;
            float val = coarse_map[idx];
            if (val == 9.9f) {
                std::cout << "Q"; // Quantum Refined Voxel
            } else if (val == 0.0f) {
                std::cout << "."; // Classical Background
            } else if (val == 1.0f) {
                std::cout << "#"; // Classical Edge
            } else {
                std::cout << "~"; // Classical Uncertainty (Nhiễu Cổ điển)
            }
            std::cout << " ";
        }
        std::cout << "\n";
    }
    std::cout << "Chú thích: [.] Nền cổ điển  [#] Biên cổ điển  [~] Nhiễu mập mờ  [Q] Lượng tử siêu nét\n";
}

} // namespace vision
} // namespace dm
