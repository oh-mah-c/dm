#pragma once

#include <vector>

namespace dm {
namespace algorithm {

    // Node của Quadtree lưu trong mảng tĩnh phẳng (Memory Arena)
    struct QuadNode {
        float mass_total;       // Tổng khối lượng của các hạt trong ô này
        float center_x;         // Khối tâm X
        float center_y;         // Khối tâm Y
        float bounds_x, bounds_y; // Tọa độ góc dưới trái của ô
        float size;             // Chiều dài cạnh của ô vuông
        
        int first_child_idx;    // Chỉ số của đứa con đầu tiên (4 con nằm liên tiếp). -1 nếu là Node lá.
        int star_idx;           // Nếu là Node lá (1 hạt), lưu chỉ số của hạt đó. -1 nếu trống hoặc có con.
    };

    class OhmCosmosArena {
    private:
        std::vector<QuadNode> tree_arena;
        int node_count;

        // Recursive helpers
        void insert(int node_idx, int star_idx, const float* x, const float* y, const float* m);
        void compute_mass_distribution(int node_idx);

    public:
        // Cấp phát trước bộ nhớ để tránh malloc/new trong runtime
        OhmCosmosArena(int max_stars);

        // Xóa cây (Logic) và xây dựng lại từ đầu cho khung hình mới
        void build_tree(const float* x, const float* y, const float* m, int num_stars);

        // Tính lực hấp dẫn lên 1 hạt sử dụng Barnes-Hut O(N log N)
        void compute_force_cpu(int node_idx, float star_x, float star_y, float& fx, float& fy) const;
    };

    // Hàm gọi từ PyTorch Wrapper để chạy toàn bộ quá trình (Xây cây -> Tính lực -> Cập nhật vị trí)
    void cosmos_step_cpu(
        OhmCosmosArena& arena,
        float* x, float* y, 
        float* vx, float* vy, 
        const float* m, 
        int num_stars, 
        float dt
    );

} // namespace algorithm
} // namespace dm
