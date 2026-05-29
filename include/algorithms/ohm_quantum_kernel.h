#pragma once

#include <cstddef>

namespace dm {
namespace algorithm {

    // C++ Kernel giải phương trình Schrödinger Phụ thuộc thời gian
    // Chạy hoàn toàn bằng con trỏ thô để vắt kiệt L1/L2 Cache của CPU
    void schrodinger_step_cpu_1d(
        const float* R_old,  // Mảng Phần Thực cũ
        const float* I_old,  // Mảng Phần Ảo cũ
        const float* V,      // Bức tường Thế năng (Potential Barrier)
        float* R_new,        // Mảng Phần Thực mới (Đầu ra)
        float* I_new,        // Mảng Phần Ảo mới (Đầu ra)
        size_t length,       // Độ dài không gian lưới
        float dt,            // Bước thời gian
        float dx_sq          // dx bình phương
    );

} // namespace algorithm
} // namespace dm
