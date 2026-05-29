#include <iostream>
#include "models/physics/ohm_qbnf.h"

using namespace dm::physics::symbolic;
using namespace dm::physics::qbnf;

int main() {
    std::cout << "========================================================\n";
    std::cout << " Ohm-QBNF: Quantum Birkhoff Normal Form Extractor\n";
    std::cout << " (arXiv:2501.05041v2 [math-ph] implementation)\n";
    std::cout << "========================================================\n\n";

    // Khởi tạo Đại số Heisenberg (Quantum Harmonic Oscillator)
    // a: Toán tử Hủy hạt
    // adag (a^dagger): Toán tử Tạo hạt
    Symbol a("a", 1);
    Symbol adag("adag", 2);

    LieAlgebra heisenberg;
    // [a, a^dagger] = 1
    heisenberg.define_commutator(a, adag, Expression(Term(1.0f, {}))); // Term không có Symbol là hằng số

    QBNFEngine qbnf(heisenberg);

    // Hamiltonian Lượng tử Không nhiễu (Harmonic Oscillator)
    // H0 = a^dagger * a
    Expression H0(Term(1.0f, {adag, a}));
    
    // Yếu tố nhiễu loạn phi tuyến (Nonlinear Perturbation)
    // H1 = a^3 + (a^dagger)^3 + a^dagger * a
    Expression H1;
    H1.terms.push_back(Term(1.0f, {a, a, a}));
    H1.terms.push_back(Term(1.0f, {adag, adag, adag}));
    H1.terms.push_back(Term(2.5f, {adag, a})); // Term này giữ nguyên số hạt (Resonant)

    std::cout << "[*] Cấu trúc Lượng tử Ban đầu:\n";
    std::cout << "    H0 (Gốc) : " << H0.to_string() << "\n";
    std::cout << "    H1 (Nhiễu): " << H1.to_string() << "\n\n";

    std::cout << "[*] Chạy Thuật toán QBNF (Trích xuất Dạng chuẩn Birkhoff Lượng tử)...\n";
    
    Expression N1; // Phần dư (Resonant)
    Expression W1; // Toán tử sinh (Generator)
    qbnf.compute_first_order_qbnf(H0, H1, N1, W1);

    std::cout << "\n=> [KẾT QUẢ] PHÂN RÃ THEO CHUẨN BIRKHOFF:\n";
    std::cout << "   N1 (Quantum Normal Form): " << N1.to_string() << "\n";
    std::cout << "   W1 (Unitary Generator)  : " << W1.to_string() << "\n\n";

    // Kiểm tra tính chính xác: N1 chỉ được chứa các terms giao hoán với H0 (số 'a' == số 'adag')
    // Nghĩa là các hạt sinh ra và bị hủy phải bằng nhau (Bảo toàn số hạt)
    if (N1.to_string() == "2.5 * adag * a") {
        std::cout << "[SUCCESS] C++ ĐÃ TỰ LỌC BỎ NHIỄU LOẠN PHI TUYẾN!\n";
        std::cout << "[SUCCESS] Hamiltonian Lượng tử đã được trả về trạng thái Tinh khiết!\n";
    } else {
        std::cout << "[FAILED] Kết quả QBNF không khớp!\n";
    }

    return 0;
}
