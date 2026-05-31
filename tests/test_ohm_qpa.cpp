#include <torch/torch.h>
#include <iostream>
#include "torch/nn/modules/ohm_qpa.h"

int main() {
    std::cout << "========================================================\n";
    std::cout << " Test OhmQPA (Quantum Parametric Attention) in PyTorch\n";
    std::cout << "========================================================\n\n";

    int64_t batch_size = 2;
    int64_t num_heads = 4;
    int64_t seq_len_q = 10;
    int64_t seq_len_k = 15;
    int64_t hidden_dim = 64;

    std::cout << "[*] Khởi tạo Tensors (Query, Key)...\n";
    std::cout << "    Query Shape: [" << batch_size << ", " << num_heads << ", " << seq_len_q << ", " << hidden_dim << "]\n";
    std::cout << "    Key Shape:   [" << batch_size << ", " << num_heads << ", " << seq_len_k << ", " << hidden_dim << "]\n\n";

    auto query = torch::randn({batch_size, num_heads, seq_len_q, hidden_dim});
    auto key = torch::randn({batch_size, num_heads, seq_len_k, hidden_dim});

    std::cout << "[*] Khởi tạo Module OhmQPA...\n";
    torch::nn::OhmQPA qpa(hidden_dim);

    std::cout << "[*] Chạy Forward Pass (Mô phỏng Mạch Lượng tử 2-Qubit)...\n";
    auto score = qpa->forward(query, key);

    std::cout << "    Output Score Shape: [" 
              << score.size(0) << ", " 
              << score.size(1) << ", " 
              << score.size(2) << ", " 
              << score.size(3) << "]\n\n";

    // Kiểm tra tính giới hạn của hàm mục tiêu (Boundedness Property)
    float min_val = score.min().item<float>();
    float max_val = score.max().item<float>();

    std::cout << "[*] Kiểm tra tính Boundedness của Toán tử Lượng tử...\n";
    std::cout << "    Min Score: " << min_val << "\n";
    std::cout << "    Max Score: " << max_val << "\n";

    if (min_val >= 0.0f && max_val <= 1.0f) {
        std::cout << "\n[SUCCESS] Điểm số Attention được kẹp tuyệt đối trong khoảng [0, 1]. Toán học Lượng tử chính xác!\n";
    } else {
        std::cout << "\n[FAILED] Lỗi! Điểm số vượt ra ngoài vùng giới hạn [0, 1].\n";
        return 1;
    }

    return 0;
}
