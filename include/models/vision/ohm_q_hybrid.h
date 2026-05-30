#pragma once

#include <vector>
#include <iostream>
#include "models/vision/ohm_q.h"

namespace dm {
namespace vision {

// Cấu trúc Vùng cần quan tâm (Region of Interest)
struct BoundingBox {
    int x, y;
    int width, height;
    
    BoundingBox(int x_, int y_, int w, int h) : x(x_), y(y_), width(w), height(h) {}
};

// Ohm-Q Hybrid: Tinh chỉnh Lượng tử cục bộ dựa trên Bản đồ Cổ điển
class OhmQHybrid {
private:
    std::vector<float> coarse_map; // Bản đồ thô (Classical)
    int full_width;
    int full_height;

public:
    OhmQHybrid();

    // 1. Quét toàn bộ ảnh bằng thuật toán Cổ điển (Mô phỏng Classical Pass)
    void run_classical_coarse_pass(const std::vector<float>& full_image, int w, int h);

    // 2. Trích xuất ROI và Khởi chạy Lượng tử (Quantum Refinement)
    void run_quantum_roi_refinement(const std::vector<float>& full_image, const BoundingBox& roi);

    // 3. Hỗ trợ in kết quả Hybrid
    void print_hybrid_map() const;
};

} // namespace vision
} // namespace dm
