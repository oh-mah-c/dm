#pragma once

#include <vector>

namespace dm {
namespace algorithm {

    // Trạng thái của một Photon trong hệ tọa độ cực r, phi
    // Phương trình: d^2u/dphi^2 + u = 3M u^2  (với u = 1/r)
    // Chuyển thành hệ bậc 1:
    // du/dphi = v
    // dv/dphi = -u + 3M u^2
    struct GeoState {
        float u;   // u = 1/r
        float v;   // v = du/dphi
        float phi; // góc phi
    };

    // Hàm tính đạo hàm cho phương trình quỹ đạo Schwarzschild
    // Trả về {du, dv}
    void compute_derivatives(float u, float v, float M, float& du, float& dv);

    // Chạy 1 bước RK45 Thích ứng (Adaptive Step-size)
    // Trả về true nếu bước nhảy thành công (sai số nhỏ), false nếu cần băm nhỏ dt và tính lại
    bool compute_geodesic_step_cpu(GeoState& state, float& dphi, float mass, float tolerance, int& dt_shrinks);

    // Bắn một photon và lưu lại quỹ đạo
    void simulate_photon_path(GeoState initial_state, float mass, int max_steps, float tolerance, 
                              std::vector<float>& out_r, std::vector<float>& out_phi, std::vector<float>& out_dphi);

} // namespace algorithm
} // namespace dm
