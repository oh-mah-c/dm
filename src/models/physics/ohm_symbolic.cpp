#include "models/physics/ohm_symbolic.h"
#include <sstream>
#include <cmath>
#include <algorithm>

namespace dm {
namespace physics {
namespace symbolic {

// --- Term ---
Term Term::operator*(const Term& other) const {
    Term result(this->coeff * other.coeff);
    result.symbols = this->symbols;
    result.symbols.insert(result.symbols.end(), other.symbols.begin(), other.symbols.end());
    return result;
}

std::string Term::to_string() const {
    if (std::abs(coeff) < 1e-6) return "0";
    
    std::ostringstream oss;
    oss << coeff;
    for (const auto& sym : symbols) {
        oss << " * " << sym.name;
    }
    return oss.str();
}

// --- Expression ---
Expression Expression::operator+(const Expression& other) const {
    Expression result = *this;
    result.terms.insert(result.terms.end(), other.terms.begin(), other.terms.end());
    return result;
}

Expression Expression::operator-(const Expression& other) const {
    Expression result = *this;
    for (auto t : other.terms) {
        t.coeff = -t.coeff;
        result.terms.push_back(t);
    }
    return result;
}

Expression Expression::operator*(const Expression& other) const {
    Expression result;
    for (const auto& t1 : this->terms) {
        for (const auto& t2 : other.terms) {
            result.terms.push_back(t1 * t2);
        }
    }
    return result;
}

Expression Expression::operator*(float scalar) const {
    Expression result = *this;
    for (auto& t : result.terms) {
        t.coeff *= scalar;
    }
    return result;
}

void Expression::simplify() {
    std::vector<Term> new_terms;
    for (auto& t : terms) {
        if (std::abs(t.coeff) < 1e-6) continue;
        
        bool merged = false;
        for (auto& nt : new_terms) {
            // Kiểm tra xem hai list symbol có giống nhau hoàn toàn không
            if (t.symbols.size() == nt.symbols.size()) {
                bool match = true;
                for (size_t i = 0; i < t.symbols.size(); ++i) {
                    if (!(t.symbols[i] == nt.symbols[i])) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    nt.coeff += t.coeff;
                    merged = true;
                    break;
                }
            }
        }
        if (!merged) {
            new_terms.push_back(t);
        }
    }

    // Xóa các term có coeff == 0
    terms.clear();
    for (const auto& nt : new_terms) {
        if (std::abs(nt.coeff) > 1e-6) {
            terms.push_back(nt);
        }
    }
}

std::string Expression::to_string() const {
    if (terms.empty()) return "0";
    
    std::string s = terms[0].to_string();
    for (size_t i = 1; i < terms.size(); ++i) {
        if (terms[i].coeff > 0) {
            s += " + " + terms[i].to_string();
        } else {
            // coeff âm, tự động in dấu trừ vì to_string() in ra số âm
            s += " + (" + terms[i].to_string() + ")";
        }
    }
    return s;
}

// --- LieAlgebra ---
void LieAlgebra::define_commutator(const Symbol& sym1, const Symbol& sym2, const Expression& expr) {
    commutators[{sym1.name, sym2.name}] = expr;
    commutators[{sym2.name, sym1.name}] = expr * (-1.0f); // Tính phản giao hoán
}

Expression LieAlgebra::get_commutator(const Symbol& a, const Symbol& b) const {
    auto it = commutators.find({a.name, b.name});
    if (it != commutators.end()) {
        return it->second;
    }
    return Expression(); // Trả về 0 nếu giao hoán (commuting)
}

Expression LieAlgebra::normal_order(const Expression& expr) const {
    Expression current = expr;
    bool swapped = true;
    
    // Bubble sort-like algorithm cho chuỗi Ký hiệu, 
    // nhưng mỗi lần swap phải sinh ra phần dư (Commutator)
    while (swapped) {
        swapped = false;
        Expression next_expr;
        
        for (const auto& t : current.terms) {
            bool term_swapped = false;
            for (size_t i = 0; i + 1 < t.symbols.size(); ++i) {
                if (t.symbols[i].order > t.symbols[i+1].order) {
                    // Cần hoán đổi: A * B = B * A + [A, B]
                    Symbol A = t.symbols[i];
                    Symbol B = t.symbols[i+1];
                    
                    // Term 1: B * A
                    Term t_ba = t;
                    t_ba.symbols[i] = B;
                    t_ba.symbols[i+1] = A;
                    next_expr.terms.push_back(t_ba);
                    
                    // Term 2: [A, B]
                    Expression comm = get_commutator(A, B);
                    if (!comm.terms.empty()) {
                        // Tích vô hướng phần đầu, commutator, phần đuôi
                        Expression head(Term(t.coeff, std::vector<Symbol>(t.symbols.begin(), t.symbols.begin() + i)));
                        Expression tail(Term(1.0f, std::vector<Symbol>(t.symbols.begin() + i + 2, t.symbols.end())));
                        
                        Expression comm_expanded = head * comm * tail;
                        next_expr = next_expr + comm_expanded;
                    }
                    
                    term_swapped = true;
                    swapped = true;
                    break; // Phá vòng lặp để xử lý lại từ đầu biểu thức mới (tránh rối)
                }
            }
            if (!term_swapped) {
                next_expr.terms.push_back(t);
            }
        }
        
        if (swapped) {
            next_expr.simplify();
            current = next_expr;
        }
    }
    
    current.simplify();
    return current;
}

Expression LieAlgebra::compute_commutator(Expression A, Expression B) const {
    // [A, B] = A*B - B*A
    Expression AB = A * B;
    Expression BA = B * A;
    Expression comm = AB - BA;
    return normal_order(comm); // Tự động đưa về dạng chuẩn PBW
}

} // namespace symbolic
} // namespace physics
} // namespace dm
