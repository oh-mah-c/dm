#include <iostream>
#include "models/physics/ohm_symbolic.h"

using namespace dm::physics::symbolic;

int main() {
    std::cout << "========================================================\n";
    std::cout << " Ohm-STRING: Symbolic Lie Algebra Engine (CAS) \n";
    std::cout << "========================================================\n\n";

    // Khởi tạo các ký hiệu (Symbols) của đại số Lie sl_2
    // Normal Ordering Order: f < h < e (Tức là f luôn bị đẩy ra trước, e bị đẩy về sau)
    Symbol f("f", 1);
    Symbol h("h", 2);
    Symbol e("e", 3);

    LieAlgebra sl2;
    
    // Định nghĩa các hệ thức giao hoán (Commutators) của sl_2
    // [e, f] = h
    sl2.define_commutator(e, f, Expression(Term(1.0f, {h})));
    // [h, e] = 2e
    sl2.define_commutator(h, e, Expression(Term(2.0f, {e})));
    // [h, f] = -2f
    sl2.define_commutator(h, f, Expression(Term(-2.0f, {f})));

    std::cout << "[*] Đại số Lie sl_2 đã được khởi tạo.\n";
    std::cout << "    [e, f] = h\n";
    std::cout << "    [h, e] = 2e\n";
    std::cout << "    [h, f] = -2f\n\n";

    // BÀI TOÁN: Tính [e, f^2]
    // Nghĩa là: e * f * f - f * f * e
    Expression term1(Term(1.0f, {e, f, f}));
    Expression term2(Term(-1.0f, {f, f, e}));
    Expression expr = term1 + term2;

    std::cout << "Bài toán: Rút gọn biểu thức [e, f^2]\n";
    std::cout << "Dạng ban đầu: " << expr.to_string() << "\n\n";

    // Yêu cầu C++ tự động tính toán Đại số Trừu tượng (Normal Ordering)
    std::cout << "[*] C++ đang tiến hành Sắp xếp Chuẩn (Normal Ordering - PBW Basis)...\n";
    Expression result = sl2.normal_order(expr);

    // KẾT QUẢ KỲ VỌNG (Giải bằng tay):
    // [e, f^2] = [e, f]f + f[e, f] = h f + f h
    // Nhưng do Normal Ordering: f < h, nên 'h f' phải đổi thành 'f h + [h, f]'
    // => (f h - 2f) + f h = 2 f h - 2 f
    std::cout << "=> Kết quả cuối cùng (Thuần túy Ký hiệu Đại số): \n";
    std::cout << "   " << result.to_string() << "\n\n";

    if (result.to_string() == "2 * f * h + (-2 * f)") {
        std::cout << "[SUCCESS] C++ ĐÃ GIẢI TOÁN ĐẠI SỐ TRỪU TƯỢNG HOÀN TOÀN TỰ ĐỘNG!\n";
        std::cout << "[SUCCESS] Ohm-STRING đã sẵn sàng để tiến vào Vertex Algebras và String Theory!\n";
    } else {
        std::cout << "[FAILED] Kết quả tính toán Symbolic không khớp!\n";
    }

    return 0;
}
