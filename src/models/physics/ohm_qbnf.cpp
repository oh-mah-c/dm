#include "models/physics/ohm_qbnf.h"

namespace dm {
namespace physics {
namespace qbnf {

void QBNFEngine::homological_decomposition(const Expression& H1, 
                                           Expression& out_Kernel, 
                                           Expression& out_Range) const {
    // Đếm số lượng 'a' và 'adag' (a^dagger) trong mỗi term
    // Phần dư (Kernel N1): số 'a' == số 'adag' (Giao hoán với H0)
    // Phần có thể khử (Range R1): số 'a' != số 'adag'
    
    for (const auto& term : H1.terms) {
        int count_a = 0;
        int count_adag = 0;
        for (const auto& sym : term.symbols) {
            if (sym.name == "a") count_a++;
            if (sym.name == "adag") count_adag++;
        }
        
        if (count_a == count_adag) {
            out_Kernel.terms.push_back(term);
        } else {
            out_Range.terms.push_back(term);
        }
    }
}

void QBNFEngine::compute_first_order_qbnf(const Expression& H0, const Expression& H1,
                                          Expression& out_N1, Expression& out_W1) const {
    // 1. Phân rã H1 thành N1 (Kernel) và R1 (Range)
    Expression R1;
    homological_decomposition(H1, out_N1, R1);
    
    // 2. Tìm W1 sao cho [H0, W1] = -R1
    // Giả sử H0 = adag * a (Dao động tử điều hòa chuẩn)
    // Với mỗi term trong R1 dạng (adag^m * a^n), ta có:
    // [H0, adag^m * a^n] = (m - n) * adag^m * a^n
    // Vậy ta chọn W1 có cùng term nhưng hệ số là: -coeff / (m - n)
    
    for (const auto& term : R1.terms) {
        int count_a = 0;
        int count_adag = 0;
        for (const auto& sym : term.symbols) {
            if (sym.name == "a") count_a++;
            if (sym.name == "adag") count_adag++;
        }
        
        int diff = count_adag - count_a;
        if (diff != 0) { // Đã được đảm bảo bởi Range
            float new_coeff = -term.coeff / static_cast<float>(diff);
            Term w_term = term;
            w_term.coeff = new_coeff;
            out_W1.terms.push_back(w_term);
        }
    }
}

} // namespace qbnf
} // namespace physics
} // namespace dm
