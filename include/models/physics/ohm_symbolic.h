#pragma once

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <memory>
#include <utility>

namespace dm {
namespace physics {
namespace symbolic {

// Đại diện cho một toán tử sinh (Generator) cơ bản (Ví dụ: e, f, h)
class Symbol {
public:
    std::string name;
    int order; // Dùng để xác định thứ tự Normal Ordering (Ví dụ: f < h < e)

    Symbol(std::string n, int o) : name(n), order(o) {}

    bool operator==(const Symbol& other) const { return name == other.name; }
    bool operator<(const Symbol& other) const { return order < other.order; }
};

// Đại diện cho một đơn thức (Ví dụ: 3.5 * f * h * e)
class Term {
public:
    float coeff;
    std::vector<Symbol> symbols;

    Term(float c, std::vector<Symbol> syms = {}) : coeff(c), symbols(syms) {}

    // Nhân hai đơn thức (Không giao hoán!)
    Term operator*(const Term& other) const;
    
    // In ra chuỗi
    std::string to_string() const;
};

// Đại diện cho một Đa thức Không Giao hoán (Tổng của các Terms)
class Expression {
public:
    std::vector<Term> terms;

    Expression() {}
    Expression(const Term& t) { terms.push_back(t); }

    Expression operator+(const Expression& other) const;
    Expression operator-(const Expression& other) const;
    Expression operator*(const Expression& other) const; // Nhân phân phối đa thức
    Expression operator*(float scalar) const;

    // Gộp các Term có cùng chuỗi Symbols lại (Rút gọn)
    void simplify();
    
    std::string to_string() const;
};

// Định nghĩa Cấu trúc Đại số Lie (Commutator relations)
class LieAlgebra {
private:
    // Lưu trữ [A, B] = Expression
    // Khóa là pair<tên_A, tên_B>
    std::map<std::pair<std::string, std::string>, Expression> commutators;

public:
    // Định nghĩa [sym1, sym2] = expr
    void define_commutator(const Symbol& sym1, const Symbol& sym2, const Expression& expr);

    // Lấy hệ thức giao hoán [A, B]
    // Nếu [A, B] không được định nghĩa, trả về 0 (Giao hoán).
    // Tự động suy luận [B, A] = -[A, B].
    Expression get_commutator(const Symbol& a, const Symbol& b) const;

    // Tính Commutator của 2 đa thức bất kỳ: [ExprA, ExprB] = A*B - B*A
    // và áp dụng Normal Ordering
    Expression compute_commutator(Expression A, Expression B) const;

    // Thuật toán cốt lõi: Sắp xếp đa thức về PBW Basis (Normal Ordering)
    // Nếu gặp A * B mà order(A) > order(B), đổi thành B * A + [A, B]
    Expression normal_order(const Expression& expr) const;
};

} // namespace symbolic
} // namespace physics
} // namespace dm
