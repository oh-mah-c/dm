#pragma once

#include <complex>
#include <vector>

namespace dm {
namespace physics {

// Cấu trúc Ma trận Nhóm Lie SU(3) cho hạt Gluon
// Chứa 9 số phức
struct alignas(32) SU3Matrix {
    std::complex<float> M[3][3];

    // Tạo Ma trận Đơn vị (Identity)
    static SU3Matrix identity();
    
    // Tạo Ma trận ngẫu nhiên gần với Đơn vị (để sinh nhiễu Metropolis)
    static SU3Matrix random_su3(float epsilon);
};

// Phép nhân và lấy liên hợp phức của SU(3) (Tối ưu Unrolled)
SU3Matrix multiply_SU3(const SU3Matrix& A, const SU3Matrix& B);
SU3Matrix adjoint_SU3(const SU3Matrix& A);

// Lò phản ứng Sắc động lực học lượng tử trên lưới
class LatticeQCD {
public:
    LatticeQCD(int L, int T, float beta);
    
    // Cập nhật toàn bộ lưới bằng MCMC Metropolis-Hastings
    void metropolis_step();
    
    // Tính trung bình Năng lượng Plaquette của không gian
    float average_plaquette() const;

private:
    int L, T;
    float beta;
    // Mảng phẳng 1D [T * L * L * L * 4] đại diện cho lưới Không-Thời gian 4D
    // 4 là 4 hướng Không-thời gian (x, y, z, t)
    std::vector<SU3Matrix> links;
    
    // Ánh xạ tọa độ 4D xuống địa chỉ 1D
    inline int get_index(int x, int y, int z, int t, int mu) const;
    
    // Tính toán năng lượng cục bộ (Action) quanh một điểm x theo hướng mu
    float calculate_local_action(int x, int y, int z, int t, int mu) const;
    
    // Lấy link có hỗ trợ Boundary Conditions (Periodic)
    SU3Matrix get_link(int x, int y, int z, int t, int mu) const;
};

} // namespace physics
} // namespace dm
