#pragma once

namespace dm {
namespace algorithm {

    // C++ Kernel: Giải PDE Gray-Scott bằng Sai phân hữu hạn (Finite Difference)
    // Tính toán song song trên mảng tĩnh để ép CPU đạt cực đại băng thông RAM
    void reaction_diffusion_step_cpu(
        const float* __restrict__ A_old, const float* __restrict__ B_old,
        float* __restrict__ A_new, float* __restrict__ B_new,
        int width, int height,
        float D_a, float D_b, float feed, float kill, float dt
    );

} // namespace algorithm
} // namespace dm
