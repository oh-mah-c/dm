#pragma once

#include "models/physics/ohm_symbolic.h"

namespace dm {
namespace physics {
namespace qbnf {

using namespace dm::physics::symbolic;

// Mô-đun Quantum Birkhoff Normal Form (Chuẩn hóa Lượng tử)
class QBNFEngine {
private:
    LieAlgebra algebra;

public:
    QBNFEngine(const LieAlgebra& alg) : algebra(alg) {}

    // Thuật toán cốt lõi QBNF bậc 1 (First-order Quantum Birkhoff Normal Form)
    // Đầu vào:
    //  - H0: Hamiltonian Gốc (Chưa bị nhiễu, ví dụ: a^dagger * a)
    //  - H1: Nhiễu loạn lượng tử (Ví dụ: a + a^dagger)
    // Đầu ra:
    //  - N1: Phần dư Chuẩn hóa (Resonant terms - giao hoán với H0)
    //  - W1: Toán tử sinh (Generator của Unitary Transform)
    void compute_first_order_qbnf(const Expression& H0, const Expression& H1,
                                  Expression& out_N1, Expression& out_W1) const;

    // Phân rã Phương trình Đồng điều (Homological Equation):
    // Phân tích H1 thành phần Kernel (giao hoán với H0) và phần Range (có thể triệt tiêu)
    // - Kernel (N1): Tổng các terms có số lượng a và a^dagger bằng nhau
    // - Range (R1): Tổng các terms bị lệch số lượng (có thể tìm được W1)
    void homological_decomposition(const Expression& H1, 
                                   Expression& out_Kernel, 
                                   Expression& out_Range) const;
};

} // namespace qbnf
} // namespace physics
} // namespace dm
