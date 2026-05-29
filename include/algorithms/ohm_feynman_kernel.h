#pragma once

namespace dm {
namespace algorithm {

    // Ohm-FEYNMAN: Feynman Path Integral Simulator
    // Uses Monte Carlo path generation with xorshift32 PRNG.
    void feynman_path_integral_cpu(
        int num_paths,        // Số lượng quỹ đạo sinh ngẫu nhiên
        int steps_per_path,   // Số bước thời gian của mỗi quỹ đạo
        float start_x,        // Điểm bắt đầu A
        float end_x,          // Điểm kết thúc B
        float* prob_real,     // Trả về phần thực của Biên độ xác suất
        float* prob_imag      // Trả về phần ảo của Biên độ xác suất
    );

} // namespace algorithm
} // namespace dm
