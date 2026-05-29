#pragma once

#include <complex>

namespace dm {
namespace physics {

// Cấu trúc Ma trận 5x5 Số Phức (Trạng thái Lượng tử Rời rạc 5D)
struct Matrix5x5 {
    std::complex<float> M[5][5];

    // Khởi tạo Ma trận 0
    Matrix5x5();

    // Phép Cộng
    Matrix5x5 operator+(const Matrix5x5& other) const;
    // Phép Trừ
    Matrix5x5 operator-(const Matrix5x5& other) const;
    // Phép Nhân 2 Ma trận
    Matrix5x5 operator*(const Matrix5x5& other) const;
    // Phép Nhân vô hướng (Complex Scalar)
    Matrix5x5 operator*(std::complex<float> scalar) const;
};

// Hàm tiện ích: Trả về chuẩn Frobenius (Norm) của Ma trận để đo lường Sai số
float frobenius_norm(const Matrix5x5& A);

// Lõi Mô phỏng: Dao động tử Điều hòa Lượng tử Rời rạc (5D)
class DiscreteQHO5D {
public:
    // Toán tử Vị trí X (Coordinate Operator)
    static Matrix5x5 X5();
    
    // Toán tử Hoán vị Tuần hoàn (Circulant Permutation)
    static Matrix5x5 C5();
    
    // Toán tử Động lượng Y (Momentum Operator)
    static Matrix5x5 Y5();
    
    // Toán tử Đạo hàm Rời rạc D_5
    static Matrix5x5 D5();
    
    // Biến đổi Fourier Rời rạc 5D (DFT)
    static Matrix5x5 Phi5();
    
    // Toán tử Hạ lượng tử A (Lowering Operator)
    static Matrix5x5 A5();
    
    // Toán tử Thăng lượng tử A^T (Raising Operator)
    static Matrix5x5 A5_T();
    
    // Toán tử Số hạt (Number Operator N = A^T * A)
    static Matrix5x5 N5();
};

} // namespace physics
} // namespace dm
