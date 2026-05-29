#include <torch/torch.h>
#include <iostream>
#include "modules/ohm_genesis.h"

using namespace dm::modules;

int main() {
    std::cout << "--- Testing Ohm-GENESIS (Reaction-Diffusion Morphogenesis) ---\n";

    int grid_size = 50;
    OhmGENESIS genesis(grid_size, grid_size);

    std::cout << "[*] Genesis Engine Initialized (Corals Preset).\n";
    std::cout << "[*] Planting Seed in the center of the grid...\n";
    
    genesis->plant_seed(3);

    int steps = 5000;
    std::cout << "[*] Simulating " << steps << " steps of Morphogenesis (Sliding Window Laplace)...\n";

    genesis->forward(steps);

    torch::Tensor pattern = genesis->get_biology_pattern();

    // Tính toán tổng lượng hóa chất B còn sống sót (Nó phải lây lan ra ngoài seed ban đầu)
    float total_life = pattern.sum().item<float>();
    
    std::cout << "\n--- Results ---\n";
    std::cout << "Total Bio-Mass (Chemical B concentration): " << total_life << "\n";

    // Kích thước seed ban đầu là 6x6 (size=3) = 36 điểm = khối lượng 36.0
    // Nếu nó lớn hơn 36 nhiều lần (vd 100), nghĩa là nó đã lan rộng thành hình san hô.
    // Nếu nó rớt xuống 0, nghĩa là sự sống đã bị tuyệt diệt.
    if (total_life > 50.0f) {
        std::cout << "SUCCESS: Morphogenesis achieved! The chemical reactions grew into a complex biological structure!\n";
    } else if (total_life < 1.0f) {
        std::cout << "FAILURE: The seed died out.\n";
    } else {
        std::cout << "FAILURE: The seed did not grow significantly.\n";
    }

    return 0;
}
