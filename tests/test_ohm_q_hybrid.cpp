#include <iostream>
#include <vector>
#include <random>
#include "models/vision/ohm_q_hybrid.h"

using namespace dm::vision;

int main() {
    std::cout << "========================================================\n";
    std::cout << " Ohm-Q Hybrid ROI Refinement Simulator\n";
    std::cout << "========================================================\n\n";

    const int FULL_WIDTH = 40;
    const int FULL_HEIGHT = 20;
    std::vector<float> full_image(FULL_WIDTH * FULL_HEIGHT);

    // 1. Tạo Ảnh giả lập khổng lồ (Nhiễu hạt và độ sáng mập mờ)
    std::cout << "[*] Khởi tạo Bức ảnh lớn (" << FULL_WIDTH << "x" << FULL_HEIGHT << " pixels)...\n";
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.3f, 0.7f); // Nhiễu mập mờ

    for (int i = 0; i < FULL_WIDTH * FULL_HEIGHT; ++i) {
        full_image[i] = dist(gen);
    }

    // Tạo một vùng có độ sáng cao (Vật thể cần quan tâm) ở giữa
    for (int y = 8; y < 15; ++y) {
        for (int x = 15; x < 25; ++x) {
            full_image[y * FULL_WIDTH + x] = 0.9f;
        }
    }

    OhmQHybrid hybrid_engine;

    // 2. Chạy Classical Pass
    std::cout << "[*] Quét toàn bộ ảnh bằng Thuật toán Cổ điển (Classical Coarse Pass)...\n";
    std::cout << "    => Tốc độ: Rất nhanh (Độ phân giải thô, nhiễu cao)\n";
    hybrid_engine.run_classical_coarse_pass(full_image, FULL_WIDTH, FULL_HEIGHT);

    // 3. Khai báo ROI
    BoundingBox roi(14, 7, 12, 9); // Bao trọn vật thể sáng ở giữa
    std::cout << "\n[*] Trích xuất Vùng cần quan tâm (ROI) tại: X=" << roi.x << ", Y=" << roi.y 
              << ", W=" << roi.width << ", H=" << roi.height << "\n";
    std::cout << "    => Tiết kiệm " << (1.0f - float(roi.width * roi.height) / (FULL_WIDTH * FULL_HEIGHT)) * 100 
              << "% Tài nguyên Lượng tử!\n";

    // 4. Chạy Quantum Refinement
    std::cout << "\n[*] Bắt đầu Tinh chỉnh Lượng tử (Quantum Refinement) ĐỘC QUYỀN trên vùng ROI...\n";
    hybrid_engine.run_quantum_roi_refinement(full_image, roi);

    // 5. In kết quả Hybrid
    hybrid_engine.print_hybrid_map();

    std::cout << "\n[SUCCESS] Pipeline Ohm-Q Hybrid ROI đã hoạt động thành công!\n";

    return 0;
}
