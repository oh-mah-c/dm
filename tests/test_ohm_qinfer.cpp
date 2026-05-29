#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "modules/ohm_qinfer.h"

using namespace dm::modules;

int main() {
    std::cout << "--- Testing Ohm-QINFER (Quantum Inference via Particle Filter) ---\n";

    int grid_size = 50;
    int num_universes = 5000;
    
    // Khởi tạo Bộ Suy Diễn
    OhmQINFER qinfer(num_universes, grid_size);

    std::cout << "[*] Multiverse Arena initialized with " << num_universes << " random universes.\n";

    // 1. TẠO RA BỨC TƯỜNG SỰ THẬT (DOUBLE-SLIT BARRIER)
    // Giả sử có một bức tường vô hình, chỉ hở ra ở 2 khe: index 15 và 35.
    // Nơi nào có khe, V = 0. Nơi có tường, V = 10.
    std::vector<float> true_V(grid_size, 10.0f);
    true_V[15] = 0.0f;
    true_V[35] = 0.0f;

    // 2. SINH DỮ LIỆU ĐO LƯỜNG NGẪU NHIÊN TỪ BỨC TƯỜNG SỰ THẬT
    // Xác suất hạt đi qua: P(x) = exp(-true_V[x]) / sum
    std::vector<float> true_prob(grid_size, 0.0f);
    float sum_p = 0.0f;
    for (int x = 0; x < grid_size; ++x) {
        true_prob[x] = std::exp(-true_V[x]);
        sum_p += true_prob[x];
    }
    for (int x = 0; x < grid_size; ++x) {
        true_prob[x] /= sum_p;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(true_prob.begin(), true_prob.end());

    int num_measurements = 1000;
    std::cout << "[*] Bombarding screen with " << num_measurements << " random electrons...\n";

    // 3. SUY DIỄN (INFERENCE)
    for (int i = 0; i < num_measurements; ++i) {
        int hit_x = d(gen); // Điểm hạt rơi ngẫu nhiên
        qinfer->forward(hit_x);
    }

    // 4. KẾT QUẢ SUY DIỄN
    torch::Tensor deduced_V = qinfer->get_deduced_barrier();
    
    std::cout << "\n--- Truth vs Deduced Barrier (Lower V means Slit) ---\n";
    std::cout << "Index\tTruth\tDeduced\n";
    
    bool double_slit_found = false;
    float v15 = deduced_V[15].item<float>();
    float v35 = deduced_V[35].item<float>();
    float v25 = deduced_V[25].item<float>(); // Tường ở giữa

    for (int i = 10; i <= 40; i+=5) {
        std::cout << i << "\t" << true_V[i] << "\t" << deduced_V[i].item<float>() << "\n";
    }

    // Kiểm tra xem máy có đoán được khe ở 15 và 35 (V thấp), và tường ở giữa 25 (V cao) không.
    if (v15 < v25 && v35 < v25) {
        double_slit_found = true;
    }

    if (double_slit_found) {
        std::cout << "\nSUCCESS: Quantum Tomography recovered the Double-Slit from random hits!\n";
    } else {
        std::cout << "\nFAILURE: AI could not find the Double-Slit.\n";
    }

    return 0;
}
