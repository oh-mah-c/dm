#pragma once

#include <vector>
#include <complex>
#include <iostream>
#include <string>

namespace dm {
namespace vision {

// Một Vật thể Lượng tử trong hệ thống (Energy Well / Object)
struct QObject {
    int object_id;
    std::string class_name;
    std::complex<float> state_vector; // Trạng thái Sóng $|\psi\rangle$
    float momentum; // Động lượng (Mức thay đổi không gian theo thời gian)
    float position_x;
    
    QObject(int id, std::string name, float pos, float mom, std::complex<float> state) 
        : object_id(id), class_name(name), position_x(pos), momentum(mom), state_vector(state) {}
};

// Module 1: OhmQTimeEvolution (Schrödinger Time Propagator)
class OhmQTimeEvolution {
private:
    float dt; // Bước thời gian (Delta t)

public:
    OhmQTimeEvolution(float delta_time = 1.0f) : dt(delta_time) {}

    // Dự đoán Trạng thái và Vị trí của vật thể ở Frame tiếp theo
    // Sử dụng Toán tử Tiến hóa: U(t) = exp(-i * H * dt)
    void propagate_forward(QObject& obj);
};

// Module 2: OhmQEntanglement (Quantum Tensor Products)
class OhmQEntanglement {
private:
    float momentum_threshold; // Ngưỡng đồng điệu động lượng để tạo vướng mắc

public:
    OhmQEntanglement(float threshold = 0.5f) : momentum_threshold(threshold) {}

    // Đo lường sự vướng mắc giữa 2 vật thể.
    // Nếu chúng di chuyển cùng nhau, gộp chúng thành một Trạng thái Vướng mắc (Entangled State).
    bool measure_entanglement(const QObject& obj_A, const QObject& obj_B, QObject& composite_obj);
};

} // namespace vision
} // namespace dm
