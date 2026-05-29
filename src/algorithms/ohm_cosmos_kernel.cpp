#include "algorithms/ohm_cosmos_kernel.h"
#include <cmath>
#include <algorithm>

namespace dm {
namespace algorithm {

    OhmCosmosArena::OhmCosmosArena(int max_stars) {
        // Cấp phát trước bộ nhớ. 1 Quadtree cho N điểm có tối đa ~4N nodes.
        // Ta chọn 4N + dự phòng. Zero-allocation trong runtime!
        tree_arena.resize(max_stars * 8); 
        node_count = 0;
    }

    void OhmCosmosArena::insert(int node_idx, int star_idx, const float* x, const float* y, const float* m) {
        QuadNode& node = tree_arena[node_idx];

        // Nếu node không có con và chưa chứa hạt nào, gán luôn hạt này vào node.
        if (node.first_child_idx == -1 && node.star_idx == -1) {
            node.star_idx = star_idx;
            return;
        }

        // Nếu node này đang chứa 1 hạt, chúng ta cần chẻ nó ra thành 4 ô con,
        // đẩy hạt cũ xuống ô con, và đẩy hạt mới (star_idx) xuống tiếp.
        if (node.first_child_idx == -1 && node.star_idx != -1) {
            // Chẻ làm 4 node con
            int child_base = node_count;
            node_count += 4;
            
            // Xử lý tràn bộ nhớ tĩnh (rất hiếm nếu mảng đủ to)
            if (node_count > tree_arena.size()) return;

            node.first_child_idx = child_base;
            float half_size = node.size / 2.0f;

            for (int i = 0; i < 4; ++i) {
                QuadNode& child = tree_arena[child_base + i];
                child.size = half_size;
                child.first_child_idx = -1;
                child.star_idx = -1;
                child.mass_total = 0.0f;
                
                // Xác định tọa độ góc của ô con
                float bx = node.bounds_x + (i % 2) * half_size;
                float by = node.bounds_y + (i / 2) * half_size;
                child.bounds_x = bx;
                child.bounds_y = by;
            }

            // Đẩy hạt cũ xuống
            int old_star = node.star_idx;
            node.star_idx = -1;
            
            int old_quad = ((x[old_star] >= node.bounds_x + half_size) ? 1 : 0) + 
                           ((y[old_star] >= node.bounds_y + half_size) ? 2 : 0);
            insert(child_base + old_quad, old_star, x, y, m);
        }

        // Đẩy hạt mới xuống node con tương ứng
        float half_size = node.size / 2.0f;
        int new_quad = ((x[star_idx] >= node.bounds_x + half_size) ? 1 : 0) + 
                       ((y[star_idx] >= node.bounds_y + half_size) ? 2 : 0);
        
        insert(node.first_child_idx + new_quad, star_idx, x, y, m);
    }

    void OhmCosmosArena::compute_mass_distribution(int node_idx) {
        QuadNode& node = tree_arena[node_idx];
        
        if (node.first_child_idx == -1) {
            // Node lá rỗng
            if (node.star_idx == -1) {
                node.mass_total = 0.0f;
                node.center_x = 0.0f;
                node.center_y = 0.0f;
            } 
            // Node lá có 1 hạt
            else {
                // Sẽ không truy cập mảng m, x, y ở đây để giữ hàm đơn giản,
                // Nhưng thực tế ta cần. Cách tốt nhất: Tính luôn lúc insert, hoặc truyền mảng vào.
                // Để đơn giản, ta sẽ cập nhật lúc xây cây hoặc truyền vào.
                // Ở đây ta để logic ngoài, vì hàm này gọi đệ quy.
            }
            return;
        }

        // Là node cha, đệ quy xuống các con
        node.mass_total = 0.0f;
        node.center_x = 0.0f;
        node.center_y = 0.0f;

        for (int i = 0; i < 4; ++i) {
            compute_mass_distribution(node.first_child_idx + i);
            QuadNode& child = tree_arena[node.first_child_idx + i];
            
            if (child.mass_total > 0.0f) {
                node.mass_total += child.mass_total;
                node.center_x += child.center_x * child.mass_total;
                node.center_y += child.center_y * child.mass_total;
            }
        }

        if (node.mass_total > 0.0f) {
            node.center_x /= node.mass_total;
            node.center_y /= node.mass_total;
        }
    }

    // Viết lại hàm tính phân bố khối lượng có truyền mảng để lấy giá trị cho lá
    void compute_mass_distribution_with_arrays(OhmCosmosArena& arena, std::vector<QuadNode>& tree, int node_idx, const float* x, const float* y, const float* m) {
        QuadNode& node = tree[node_idx];
        
        if (node.first_child_idx == -1) {
            if (node.star_idx != -1) {
                node.mass_total = m[node.star_idx];
                node.center_x = x[node.star_idx];
                node.center_y = y[node.star_idx];
            } else {
                node.mass_total = 0.0f;
                node.center_x = 0.0f;
                node.center_y = 0.0f;
            }
            return;
        }

        node.mass_total = 0.0f;
        node.center_x = 0.0f;
        node.center_y = 0.0f;

        for (int i = 0; i < 4; ++i) {
            compute_mass_distribution_with_arrays(arena, tree, node.first_child_idx + i, x, y, m);
            QuadNode& child = tree[node.first_child_idx + i];
            if (child.mass_total > 0.0f) {
                node.mass_total += child.mass_total;
                node.center_x += child.center_x * child.mass_total;
                node.center_y += child.center_y * child.mass_total;
            }
        }

        if (node.mass_total > 0.0f) {
            node.center_x /= node.mass_total;
            node.center_y /= node.mass_total;
        }
    }

    void OhmCosmosArena::build_tree(const float* x, const float* y, const float* m, int num_stars) {
        if (num_stars <= 0) return;

        // Xóa cây logic (O(1))
        node_count = 0;

        // Tìm khung bao (Bounding Box) của toàn bộ thiên hà
        float min_x = x[0], max_x = x[0];
        float min_y = y[0], max_y = y[0];
        for (int i = 1; i < num_stars; ++i) {
            if (x[i] < min_x) min_x = x[i];
            if (x[i] > max_x) max_x = x[i];
            if (y[i] < min_y) min_y = y[i];
            if (y[i] > max_y) max_y = y[i];
        }
        
        float size = std::max(max_x - min_x, max_y - min_y) + 0.1f; // + padding

        // Khởi tạo Node Gốc (Root)
        QuadNode& root = tree_arena[0];
        root.bounds_x = min_x;
        root.bounds_y = min_y;
        root.size = size;
        root.first_child_idx = -1;
        root.star_idx = -1;
        root.mass_total = 0.0f;
        node_count = 1;

        // Bơm từng ngôi sao vào cây (Xây cấu trúc)
        for (int i = 0; i < num_stars; ++i) {
            insert(0, i, x, y, m);
        }

        // Duyệt từ dưới lên để tính Khối tâm (Center of Mass)
        compute_mass_distribution_with_arrays(*this, tree_arena, 0, x, y, m);
    }

    void OhmCosmosArena::compute_force_cpu(int node_idx, float star_x, float star_y, float& fx, float& fy) const {
        if (node_idx >= node_count) return;
        const QuadNode& node = tree_arena[node_idx];

        if (node.mass_total == 0.0f) return; // Ô trống

        // Bỏ qua nếu là chính nó (trường hợp node lá có chứa chính hạt đang xét)
        // (Xử lý xấp xỉ: node lá với distance rất bé sẽ bị softening)

        float dx = node.center_x - star_x;
        float dy = node.center_y - star_y;
        float dist_sq = dx*dx + dy*dy + 1.0f; // Softening parameter = 1.0 (tránh lực tiến tới vô cùng)
        float dist = std::sqrt(dist_sq);

        // Điều kiện Barnes-Hut
        float theta = 0.5f; 
        if (node.first_child_idx == -1 || (node.size / dist) < theta) {
            // Lực hấp dẫn G = 1.0 để demo
            float F = node.mass_total / (dist_sq * dist);
            fx += F * dx;
            fy += F * dy;
            return;
        }

        // Rẽ nhánh đi sâu
        for (int i = 0; i < 4; ++i) {
            compute_force_cpu(node.first_child_idx + i, star_x, star_y, fx, fy);
        }
    }

    void cosmos_step_cpu(
        OhmCosmosArena& arena,
        float* x, float* y, 
        float* vx, float* vy, 
        const float* m, 
        int num_stars, 
        float dt
    ) {
        // 1. Dựng cây Tứ phân (Quadtree)
        arena.build_tree(x, y, m, num_stars);

        // 2. Tính lực và Cập nhật Vận tốc (Euler integration)
        // CPU cực đỉnh ở vòng lặp độc lập này (Có thể OpenMP song song)
        for (int i = 0; i < num_stars; ++i) {
            float fx = 0.0f;
            float fy = 0.0f;
            
            // Tính lực O(log N) cho hạt thứ i
            arena.compute_force_cpu(0, x[i], y[i], fx, fy);
            
            // F = m*a => a = F/m (Nhưng F ở đây đang tính là a nếu mass=1. Thực tế F_g = G*m*M/r^2. 
            // Công thức ở trên trả về a = F_g / m_i = G*M/r^2)
            // Cập nhật vận tốc
            vx[i] += fx * dt;
            vy[i] += fy * dt;
        }

        // 3. Cập nhật Vị trí
        for (int i = 0; i < num_stars; ++i) {
            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
        }
    }

} // namespace algorithm
} // namespace dm
