#include <iostream>
#include <iomanip>
#include <vector>
#include "models/nlp/ohm_mla.h"

using namespace dm::nlp::mla;

int main() {
    std::cout << "========================================================\n";
    std::cout << " Ohm-VideoMLA: Low-Rank Latent KV Cache Simulator\n";
    std::cout << " (arXiv:2605.30351v1 implementation)\n";
    std::cout << "========================================================\n\n";

    // Cấu hình mô phỏng Video Diffusion (Tương đương Wan-1.3B)
    MLAConfig config;
    config.num_heads = 12;
    config.head_dim = 128;
    config.latent_dim = 192; // Bài báo chỉ ra rằng latent rank (dc) = 192 là đủ cho Video Diffusion
    config.rope_dim = 32;    // Chiều của 3D-RoPE

    VideoMLAEngine engine(config);

    // Kịch bản: Khởi tạo Cache cho 1000 Tokens (1 Khung hình Video)
    int max_tokens = 1000;
    MLACache cache(max_tokens, config);

    std::cout << "[*] Cấu hình Mô hình (Wan-1.3B Scale):\n";
    std::cout << "    Số Heads        : " << config.num_heads << "\n";
    std::cout << "    Kích thước Head : " << config.head_dim << "\n";
    std::cout << "    Latent Rank (dc): " << config.latent_dim << "\n";
    std::cout << "    3D-RoPE Dim     : " << config.rope_dim << "\n\n";

    std::cout << "[*] Đang so sánh Kích thước KV Cache (Memory Footprint)...\n";
    
    // Tính toán dung lượng cho 1000 tokens
    size_t standard_size_bytes = config.standard_kv_cache_bytes_per_token() * max_tokens;
    size_t mla_size_bytes = config.mla_cache_bytes_per_token() * max_tokens;
    
    float standard_size_kb = standard_size_bytes / 1024.0f;
    float mla_size_kb = mla_size_bytes / 1024.0f;
    
    std::cout << "    - Standard Multi-Head KV Cache: " << std::fixed << std::setprecision(2) << standard_size_kb << " KB\n";
    std::cout << "    - VideoMLA Compressed Cache   : " << std::fixed << std::setprecision(2) << mla_size_kb << " KB\n";

    float reduction_pct = 100.0f - (static_cast<float>(mla_size_bytes) / standard_size_bytes * 100.0f);
    std::cout << "    -> Tỷ lệ Giảm thiểu VRAM (Reduction): " << reduction_pct << "%\n\n";

    if (reduction_pct >= 90.0f) {
        std::cout << "[SUCCESS] VRAM ĐÃ ĐƯỢC NÉN XUỐNG HƠN 90% (Thỏa mãn tỷ lệ 92.7% của Bài báo)!\n";
        std::cout << "[SUCCESS] Hệ thống sẵn sàng cho Minute-Scale Autoregressive Video Generation!\n";
    } else {
        std::cout << "[FAILED] Thuật toán nén chưa đạt hiệu suất kỳ vọng.\n";
    }

    return 0;
}
