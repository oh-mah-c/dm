#pragma once

#include <vector>
#include <cstdint>

namespace dm {
namespace algorithm {

    // Cấu trúc của MỘT Vũ trụ giả thuyết
    struct Universe {
        float weight;         // Trọng số niềm tin
        float* V_barrier;     // Mảng hình dáng bức tường (Cần suy diễn)
        float* psi_prob;      // Mảng xác suất |psi|^2 dự đoán trên màn hình
    };

    class OhmQInferArena {
    private:
        int num_universes;
        int grid_size;
        
        // Mảng tĩnh 1D khổng lồ để tránh cấp phát động cho V và psi
        std::vector<float> V_arena;
        std::vector<float> psi_arena;
        
        // Danh sách các vũ trụ trỏ vào Arena
        std::vector<Universe> universes;

        // Sinh số ngẫu nhiên Xorshift32 Float [0, 1)
        uint32_t rng_state;
        float xorshift32_float();

        // Helper để chạy phương trình Schrodinger đơn giản tính psi_prob từ V_barrier
        void compute_psi_prob(Universe& u);

    public:
        OhmQInferArena(int n_universes, int g_size);

        // Khởi tạo các vũ trụ ngẫu nhiên ban đầu
        void init_random_universes();

        // Cập nhật niềm tin (Bayes) và Chọn lọc tự nhiên (Roulette Resampling)
        void deduce_physics(int X_measured);

        // Lấy vũ trụ tốt nhất hiện tại (để hiển thị)
        const float* get_best_V_barrier() const;
    };

} // namespace algorithm
} // namespace dm
