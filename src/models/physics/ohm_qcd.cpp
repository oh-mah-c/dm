#include "models/physics/ohm_qcd.h"
#include <cstdlib>
#include <cmath>

namespace dm {
namespace physics {

SU3Matrix SU3Matrix::identity() {
    SU3Matrix I;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            I.M[i][j] = (i == j) ? std::complex<float>(1.0f, 0.0f) : std::complex<float>(0.0f, 0.0f);
        }
    }
    return I;
}

// Hàm hỗ trợ tạo nhiễu, thực tế phải trả về một ma trận thuộc SU(3) (Unitary và det=1)
// Để đơn giản hóa trong mô phỏng này, ta mô phỏng bằng cách tạo ma trận I + i * epsilon * H
// với H là ma trận Hermitian sinh ra bởi các ma trận Gell-Mann, sau đó Gram-Schmidt hóa.
// Vì C++ thô bạo, ta giả lập một ma trận SU(3) nhiễu nhẹ như sau:
SU3Matrix SU3Matrix::random_su3(float epsilon) {
    SU3Matrix R = identity();
    // Một phép quay đơn giản trong SU(3) để làm nhiễu
    float r1 = ((float)std::rand() / RAND_MAX - 0.5f) * epsilon;
    float r2 = ((float)std::rand() / RAND_MAX - 0.5f) * epsilon;
    float r3 = ((float)std::rand() / RAND_MAX - 0.5f) * epsilon;
    
    // Ma trận quay nhỏ (gần Unitary)
    std::complex<float> c1(std::cos(r1), 0), s1(std::sin(r1), 0);
    R.M[0][0] = c1; R.M[0][1] = s1;
    R.M[1][0] = -s1; R.M[1][1] = c1;
    // (Bỏ qua phép Gram-Schmidt chuẩn xác cho bản Demo tốc độ này)
    return R;
}

SU3Matrix multiply_SU3(const SU3Matrix& A, const SU3Matrix& B) {
    SU3Matrix C;
    // Bạo lực: Hardcode hoàn toàn vòng lặp (Unrolled 100%) để tránh overhead
    C.M[0][0] = A.M[0][0]*B.M[0][0] + A.M[0][1]*B.M[1][0] + A.M[0][2]*B.M[2][0];
    C.M[0][1] = A.M[0][0]*B.M[0][1] + A.M[0][1]*B.M[1][1] + A.M[0][2]*B.M[2][1];
    C.M[0][2] = A.M[0][0]*B.M[0][2] + A.M[0][1]*B.M[1][2] + A.M[0][2]*B.M[2][2];
    
    C.M[1][0] = A.M[1][0]*B.M[0][0] + A.M[1][1]*B.M[1][0] + A.M[1][2]*B.M[2][0];
    C.M[1][1] = A.M[1][0]*B.M[0][1] + A.M[1][1]*B.M[1][1] + A.M[1][2]*B.M[2][1];
    C.M[1][2] = A.M[1][0]*B.M[0][2] + A.M[1][1]*B.M[1][2] + A.M[1][2]*B.M[2][2];
    
    C.M[2][0] = A.M[2][0]*B.M[0][0] + A.M[2][1]*B.M[1][0] + A.M[2][2]*B.M[2][0];
    C.M[2][1] = A.M[2][0]*B.M[0][1] + A.M[2][1]*B.M[1][1] + A.M[2][2]*B.M[2][1];
    C.M[2][2] = A.M[2][0]*B.M[0][2] + A.M[2][1]*B.M[1][2] + A.M[2][2]*B.M[2][2];
    return C;
}

SU3Matrix adjoint_SU3(const SU3Matrix& A) {
    SU3Matrix C;
    // Ma trận liên hợp phức (Chuyển vị + Liên hợp)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            C.M[i][j] = std::conj(A.M[j][i]);
        }
    }
    return C;
}

LatticeQCD::LatticeQCD(int L, int T, float beta) : L(L), T(T), beta(beta) {
    int volume = L * L * L * T * 4;
    links.resize(volume, SU3Matrix::identity()); // Khởi tạo Cold Start (Cold Vacuum)
}

inline int LatticeQCD::get_index(int x, int y, int z, int t, int mu) const {
    // Ép mảng 4D về 1D. Điều kiện biên tuần hoàn (Periodic Boundary Conditions)
    x = (x + L) % L;
    y = (y + L) % L;
    z = (z + L) % L;
    t = (t + T) % T;
    return (((t * L + z) * L + y) * L + x) * 4 + mu;
}

SU3Matrix LatticeQCD::get_link(int x, int y, int z, int t, int mu) const {
    return links[get_index(x, y, z, t, mu)];
}

float LatticeQCD::calculate_local_action(int x, int y, int z, int t, int mu) const {
    float action = 0.0f;
    SU3Matrix U_mu = get_link(x, y, z, t, mu);

    // Duyệt qua 3 hướng vuông góc với mu để tạo các vòng Plaquette (Plaquette Loop)
    for (int nu = 0; nu < 4; ++nu) {
        if (mu == nu) continue;

        // Tính các cạnh tiến
        int dx_mu = (mu == 0) ? 1 : 0, dy_mu = (mu == 1) ? 1 : 0, dz_mu = (mu == 2) ? 1 : 0, dt_mu = (mu == 3) ? 1 : 0;
        int dx_nu = (nu == 0) ? 1 : 0, dy_nu = (nu == 1) ? 1 : 0, dz_nu = (nu == 2) ? 1 : 0, dt_nu = (nu == 3) ? 1 : 0;

        // "Staple" forward (U_nu(x+mu) * U_mu(x+nu)^dagger * U_nu(x)^dagger)
        SU3Matrix U_nu_up = get_link(x + dx_mu, y + dy_mu, z + dz_mu, t + dt_mu, nu);
        SU3Matrix U_mu_up = adjoint_SU3(get_link(x + dx_nu, y + dy_nu, z + dz_nu, t + dt_nu, mu));
        SU3Matrix U_nu_down = adjoint_SU3(get_link(x, y, z, t, nu));

        SU3Matrix staple = multiply_SU3(multiply_SU3(U_nu_up, U_mu_up), U_nu_down);
        SU3Matrix plaquette = multiply_SU3(U_mu, staple);

        // Năng lượng cục bộ ~ Re(Trace(Plaquette))
        float trace_real = plaquette.M[0][0].real() + plaquette.M[1][1].real() + plaquette.M[2][2].real();
        action += (1.0f - trace_real / 3.0f);
    }
    return action;
}

void LatticeQCD::metropolis_step() {
    // Monte Carlo Cập nhật Mạng Tinh Thể
    int total_sites = L * L * L * T;
    
    #pragma omp parallel for
    for (int site = 0; site < total_sites; ++site) {
        // Tái tạo lại tọa độ 4D từ index 1D
        int temp = site;
        int x = temp % L; temp /= L;
        int y = temp % L; temp /= L;
        int z = temp % L; temp /= L;
        int t = temp;

        for (int mu = 0; mu < 4; ++mu) {
            int idx = get_index(x, y, z, t, mu);
            float old_action = calculate_local_action(x, y, z, t, mu);

            // Đề xuất thay đổi
            SU3Matrix old_link = links[idx];
            SU3Matrix delta = SU3Matrix::random_su3(0.1f);
            links[idx] = multiply_SU3(delta, old_link);

            float new_action = calculate_local_action(x, y, z, t, mu);
            float dS = new_action - old_action;

            // Chấp nhận/Từ chối theo Metropolis-Hastings
            if (dS > 0) {
                float r = (float)std::rand() / RAND_MAX;
                if (r > std::exp(-beta * dS)) {
                    // Từ chối (Revert)
                    links[idx] = old_link;
                }
            }
        }
    }
}

float LatticeQCD::average_plaquette() const {
    float total_action = 0.0f;
    int plaquettes_count = 0;

    #pragma omp parallel for reduction(+:total_action, plaquettes_count)
    for (int t = 0; t < T; ++t) {
        for (int z = 0; z < L; ++z) {
            for (int y = 0; y < L; ++y) {
                for (int x = 0; x < L; ++x) {
                    for (int mu = 0; mu < 4; ++mu) {
                        for (int nu = mu + 1; nu < 4; ++nu) { // Mỗi mặt phẳng tính 1 lần
                            int dx_mu = (mu == 0) ? 1 : 0, dy_mu = (mu == 1) ? 1 : 0, dz_mu = (mu == 2) ? 1 : 0, dt_mu = (mu == 3) ? 1 : 0;
                            
                            SU3Matrix U1 = get_link(x, y, z, t, mu);
                            SU3Matrix U2 = get_link(x + dx_mu, y + dy_mu, z + dz_mu, t + dt_mu, nu);
                            SU3Matrix U3 = adjoint_SU3(get_link(x + (nu==0?1:0), y + (nu==1?1:0), z + (nu==2?1:0), t + (nu==3?1:0), mu));
                            SU3Matrix U4 = adjoint_SU3(get_link(x, y, z, t, nu));

                            SU3Matrix plaq = multiply_SU3(multiply_SU3(U1, U2), multiply_SU3(U3, U4));
                            float trace_real = plaq.M[0][0].real() + plaq.M[1][1].real() + plaq.M[2][2].real();
                            
                            total_action += (trace_real / 3.0f);
                            plaquettes_count++;
                        }
                    }
                }
            }
        }
    }
    return total_action / plaquettes_count;
}

} // namespace physics
} // namespace dm
