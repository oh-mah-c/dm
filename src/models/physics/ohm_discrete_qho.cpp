#include "models/physics/ohm_discrete_qho.h"
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace dm {
namespace physics {

Matrix5x5::Matrix5x5() {
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            M[i][j] = std::complex<float>(0.0f, 0.0f);
        }
    }
}

Matrix5x5 Matrix5x5::operator+(const Matrix5x5& other) const {
    Matrix5x5 C;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            C.M[i][j] = M[i][j] + other.M[i][j];
    return C;
}

Matrix5x5 Matrix5x5::operator-(const Matrix5x5& other) const {
    Matrix5x5 C;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            C.M[i][j] = M[i][j] - other.M[i][j];
    return C;
}

Matrix5x5 Matrix5x5::operator*(const Matrix5x5& other) const {
    Matrix5x5 C;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::complex<float> sum(0.0f, 0.0f);
            for (int k = 0; k < 5; ++k) {
                sum += M[i][k] * other.M[k][j];
            }
            C.M[i][j] = sum;
        }
    }
    return C;
}

Matrix5x5 Matrix5x5::operator*(std::complex<float> scalar) const {
    Matrix5x5 C;
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            C.M[i][j] = M[i][j] * scalar;
    return C;
}

float frobenius_norm(const Matrix5x5& A) {
    float norm_sq = 0.0f;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            norm_sq += std::norm(A.M[i][j]); // std::norm trả về bình phương độ lớn (magnitude squared)
        }
    }
    return std::sqrt(norm_sq);
}

Matrix5x5 DiscreteQHO5D::X5() {
    Matrix5x5 X;
    // X5 = diag(s0, s1, s2, s3, s4) với sn = 2 sin(2 * pi * n / 5)
    for (int n = 0; n < 5; ++n) {
        float sn = 2.0f * std::sin(2.0f * M_PI * n / 5.0f);
        X.M[n][n] = std::complex<float>(sn, 0.0f);
    }
    return X;
}

Matrix5x5 DiscreteQHO5D::C5() {
    Matrix5x5 C;
    // C5_kl = delta_{k, l-1} (l modulo 5)
    for (int k = 0; k < 5; ++k) {
        int l = (k + 1) % 5;
        C.M[k][l] = std::complex<float>(1.0f, 0.0f);
    }
    return C;
}

Matrix5x5 DiscreteQHO5D::D5() {
    Matrix5x5 D;
    Matrix5x5 C = C5();
    // D5 = -(C5_T - C5)
    // C5_T là chuyển vị của C5
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            D.M[i][j] = -(C.M[j][i] - C.M[i][j]);
        }
    }
    return D;
}

Matrix5x5 DiscreteQHO5D::Y5() {
    // Y5 = -i * D5
    Matrix5x5 D = D5();
    std::complex<float> neg_i(0.0f, -1.0f);
    return D * neg_i;
}

Matrix5x5 DiscreteQHO5D::Phi5() {
    Matrix5x5 Phi;
    // Phi5_kl = 5^{-1/2} q^{kl}, với q = exp(2 * pi * i / 5)
    float inv_sqrt5 = 1.0f / std::sqrt(5.0f);
    for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 5; ++l) {
            float phase = 2.0f * M_PI * (k * l) / 5.0f;
            Phi.M[k][l] = std::complex<float>(inv_sqrt5 * std::cos(phase), inv_sqrt5 * std::sin(phase));
        }
    }
    return Phi;
}

Matrix5x5 DiscreteQHO5D::A5() {
    // A5 = (1/sqrt(2)) * (X5 + iY5) = (1/sqrt(2)) * (X5 + D5)
    Matrix5x5 X = X5();
    Matrix5x5 D = D5();
    Matrix5x5 A = X + D;
    return A * std::complex<float>(1.0f / std::sqrt(2.0f), 0.0f);
}

Matrix5x5 DiscreteQHO5D::A5_T() {
    // A5_T = (1/sqrt(2)) * (X5 - iY5) = (1/sqrt(2)) * (X5 - D5)
    Matrix5x5 X = X5();
    Matrix5x5 D = D5();
    Matrix5x5 A_T = X - D;
    return A_T * std::complex<float>(1.0f / std::sqrt(2.0f), 0.0f);
}

Matrix5x5 DiscreteQHO5D::N5() {
    // N5 = A5^T * A5
    Matrix5x5 A = A5();
    Matrix5x5 A_T = A5_T();
    return A_T * A;
}

} // namespace physics
} // namespace dm
