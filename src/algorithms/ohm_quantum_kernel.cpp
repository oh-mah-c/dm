#include "algorithms/ohm_quantum_kernel.h"

namespace dm {
namespace algorithm {

// C++ Kernel giải phương trình Schrödinger Phụ thuộc thời gian
// Chạy hoàn toàn bằng con trỏ thô để vắt kiệt L1/L2 Cache của CPU
void schrodinger_step_cpu_1d(
    const float* R_old,  
    const float* I_old,  
    const float* V,      
    float* R_new,        
    float* I_new,        
    size_t length,       
    float dt,            
    float dx_sq          
) {
    // Hằng số tính toán được dịch ra ngoài vòng lặp (Loop Hoisting)
    const float coef = 0.5f / dx_sq; 

    // Bỏ qua 2 điểm biên (i=0 và i=length-1) để tránh tràn bộ nhớ (Segfault)
    // CPU sẽ chạy vòng lặp này mượt mà, branch-prediction đạt 100%!
    for (size_t i = 1; i < length - 1; ++i) {
        
        // 1. Tính Đạo hàm bậc 2 bằng con trỏ liền kề (Spatial Locality)
        // CPU nạp R_old[i-1], R_old[i], R_old[i+1] cùng lúc vào 1 Cache Line!
        float d2R_dx2 = R_old[i - 1] - 2.0f * R_old[i] + R_old[i + 1];
        float d2I_dx2 = I_old[i - 1] - 2.0f * I_old[i] + I_old[i + 1];
        
        d2R_dx2 *= coef;
        d2I_dx2 *= coef;

        // 2. Cập nhật hàm sóng chéo nhau (Leapfrog Integration)
        // Toán học: I_new = I_old + dt * (-0.5*d2R + V*R)
        //          R_new = R_old - dt * (-0.5*d2I + V*I)
        
        I_new[i] = I_old[i] + dt * (-d2R_dx2 + V[i] * R_old[i]);
        R_new[i] = R_old[i] - dt * (-d2I_dx2 + V[i] * I_old[i]);
    }
    
    // Xử lý điều kiện biên Dirichlet (Đóng băng 2 đầu mút bằng 0)
    R_new[0] = 0.0f; I_new[0] = 0.0f;
    R_new[length - 1] = 0.0f; I_new[length - 1] = 0.0f;
}

} // namespace algorithm
} // namespace dm
