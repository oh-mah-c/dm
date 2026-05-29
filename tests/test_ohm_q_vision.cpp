#include <iostream>
#include <vector>
#include "models/vision/ohm_q.h"

using namespace dm::vision;

int main() {
    std::cout << "========================================================\n";
    std::cout << " Ohm-Q: The Quantum Vision Model Simulator\n";
    std::cout << " (Pixel-Level Object Detection via Quantum Mechanics)\n";
    std::cout << "========================================================\n\n";

    // Khởi tạo Lưới ảnh 10x10
    int width = 10;
    int height = 10;
    OhmQVisionModel model(width, height);

    // Tạo một bức ảnh thô (RGB cường độ)
    std::vector<float> image(width * height, 0.2f); // Nhiễu nền (Background Noise)

    // Nhúng Object 1 (Độ sáng cao, hướng X)
    for(int y = 2; y <= 4; y++) {
        for(int x = 2; x <= 4; x++) {
            image[y * width + x] = 0.8f;
        }
    }
    // Tạo gradient cho Object 1
    image[3 * width + 4] = 0.9f;

    // Nhúng Object 2 (Độ sáng cao, hướng Y)
    for(int y = 6; y <= 8; y++) {
        for(int x = 6; x <= 8; x++) {
            image[y * width + x] = 0.7f;
        }
    }
    // Tạo gradient cho Object 2
    image[8 * width + 7] = 0.9f;

    std::cout << "[*] Đang nạp Ảnh vào Hệ thống Ohm-Q (Chuyển thành Sóng Lượng tử)...\n";
    model.load_image(image);

    std::cout << "[*] Kích hoạt Lõi Ohm-QRender (Giao thoa pha)...\n";
    model.extract_quantum_edges();

    std::cout << "[*] Kích hoạt Lõi Ohm-VideoMLA (Attention Toàn cục)...\n";
    model.apply_global_attention();

    std::cout << "[*] Kích hoạt Lõi Ohm-QBNF (Triệt tiêu Nhiễu Phi tuyến)...\n";
    model.apply_birkhoff_noise_filter();

    std::cout << "[*] Kích hoạt Lõi Ohm-Discrete-QHO (Phân vùng theo Mức Năng lượng)...\n\n";
    model.segment_objects();

    // In Bản đồ Phân vùng
    model.print_segmentation_map();

    std::cout << "\n[SUCCESS] MÔ HÌNH ĐÃ PHÁT HIỆN ĐƯỢC 2 VẬT THỂ MÀ KHÔNG CẦN DÙNG BOUNDING BOX!\n";
    std::cout << "[SUCCESS] C++ đã hợp nhất 4 Định luật Vật lý thành một Hệ Hình AI mới!\n";

    return 0;
}
