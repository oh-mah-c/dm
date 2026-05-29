#include "algorithms/ohm_relativity_kernel.h"
#include <cmath>
#include <algorithm>

namespace dm {
namespace algorithm {

    void compute_derivatives(float u, float v, float M, float& du, float& dv) {
        du = v;
        dv = -u + 3.0f * M * u * u;
    }

    bool compute_geodesic_step_cpu(GeoState& state, float& dphi, float mass, float tolerance, int& dt_shrinks) {
        // Runge-Kutta-Fehlberg (RK45) Coefficients
        // Using Cash-Karp parameters for RK45
        float A2 = 1.0f/5.0f, A3 = 3.0f/10.0f, A4 = 3.0f/5.0f, A5 = 1.0f, A6 = 7.0f/8.0f;
        
        float k1_u, k1_v, k2_u, k2_v, k3_u, k3_v, k4_u, k4_v, k5_u, k5_v, k6_u, k6_v;
        
        compute_derivatives(state.u, state.v, mass, k1_u, k1_v);
        
        compute_derivatives(state.u + dphi*(1.0f/5.0f)*k1_u, 
                            state.v + dphi*(1.0f/5.0f)*k1_v, mass, k2_u, k2_v);
                            
        compute_derivatives(state.u + dphi*(3.0f/40.0f*k1_u + 9.0f/40.0f*k2_u), 
                            state.v + dphi*(3.0f/40.0f*k1_v + 9.0f/40.0f*k2_v), mass, k3_u, k3_v);
                            
        compute_derivatives(state.u + dphi*(3.0f/10.0f*k1_u - 9.0f/10.0f*k2_u + 6.0f/5.0f*k3_u), 
                            state.v + dphi*(3.0f/10.0f*k1_v - 9.0f/10.0f*k2_v + 6.0f/5.0f*k3_v), mass, k4_u, k4_v);
                            
        compute_derivatives(state.u + dphi*(-11.0f/54.0f*k1_u + 5.0f/2.0f*k2_u - 70.0f/27.0f*k3_u + 35.0f/27.0f*k4_u), 
                            state.v + dphi*(-11.0f/54.0f*k1_v + 5.0f/2.0f*k2_v - 70.0f/27.0f*k3_v + 35.0f/27.0f*k4_v), mass, k5_u, k5_v);
                            
        compute_derivatives(state.u + dphi*(1631.0f/55296.0f*k1_u + 175.0f/512.0f*k2_u + 575.0f/13824.0f*k3_u + 44275.0f/110592.0f*k4_u + 253.0f/4096.0f*k5_u), 
                            state.v + dphi*(1631.0f/55296.0f*k1_v + 175.0f/512.0f*k2_v + 575.0f/13824.0f*k3_v + 44275.0f/110592.0f*k4_v + 253.0f/4096.0f*k5_v), mass, k6_u, k6_v);

        // RK5 Estimate
        float u5 = state.u + dphi*(37.0f/378.0f*k1_u + 250.0f/621.0f*k3_u + 125.0f/594.0f*k4_u + 512.0f/1771.0f*k6_u);
        float v5 = state.v + dphi*(37.0f/378.0f*k1_v + 250.0f/621.0f*k3_v + 125.0f/594.0f*k4_v + 512.0f/1771.0f*k6_v);

        // RK4 Estimate
        float u4 = state.u + dphi*(2825.0f/27648.0f*k1_u + 18575.0f/48384.0f*k3_u + 13525.0f/55296.0f*k4_u + 277.0f/14336.0f*k5_u + 0.25f*k6_u);
        float v4 = state.v + dphi*(2825.0f/27648.0f*k1_v + 18575.0f/48384.0f*k3_v + 13525.0f/55296.0f*k4_v + 277.0f/14336.0f*k5_v + 0.25f*k6_v);

        // Tính sai số
        float error_u = std::abs(u5 - u4);
        float error_v = std::abs(v5 - v4);
        float error = std::max(error_u, error_v);

        // LỆNH RẼ NHÁNH TÀN KHỐC ĐÁNH BẠI GPU
        if (error > tolerance) {
            // Bước nhảy quá xa tạo sai số lớn (Sượt sát Lỗ đen)
            // Băm nhỏ dt và tính lại
            dphi *= 0.5f;
            dt_shrinks++;
            return false;
        } else {
            // Cập nhật State
            state.u = u5;
            state.v = v5;
            state.phi += dphi;
            
            // Nếu không gian phẳng (sai số cực thấp), tăng tốc!
            if (error < tolerance * 0.1f && dphi < 10.0f) {
                dphi *= 1.2f;
            }
            return true;
        }
    }

    void simulate_photon_path(GeoState initial_state, float mass, int max_steps, float tolerance, 
                              std::vector<float>& out_r, std::vector<float>& out_phi, std::vector<float>& out_dphi) {
        
        GeoState state = initial_state;
        float dphi = 0.1f; // Bước nhảy dPhi ban đầu

        out_r.push_back(1.0f / state.u);
        out_phi.push_back(state.phi);
        out_dphi.push_back(dphi);

        for (int i = 0; i < max_steps; ++i) {
            int shrinks = 0;
            // Vòng lặp thích ứng: Ép CPU tính lại nếu sai số cao
            while (!compute_geodesic_step_cpu(state, dphi, mass, tolerance, shrinks)) {
                // Nếu dphi quá nhỏ thì thoát để tránh loop vô hạn (đã rơi vào Singularity)
                if (dphi < 1e-6f) break; 
            }
            
            out_r.push_back(1.0f / state.u);
            out_phi.push_back(state.phi);
            out_dphi.push_back(dphi);

            // Rơi vào Event Horizon (r < 2M -> u > 1/2M)
            if (state.u > 1.0f / (2.0f * mass)) {
                break;
            }
            
            // Bay ra vô tận (r > 1000)
            if (state.u < 1e-4f) {
                break;
            }
        }
    }

} // namespace algorithm
} // namespace dm
