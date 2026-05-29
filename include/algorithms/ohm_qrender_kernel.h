#pragma once

#include <complex>
#include <SDL.h>

namespace dm {
namespace algorithm {

    // Lượng tử hóa: Ép SDL Surface (Grayscale/RGB) thành Hàm sóng số phức (Biên độ biểu diễn ánh sáng)
    void image_to_wavefunction(const SDL_Surface* surface, std::complex<float>* wavefunction);

    // Sụp đổ Hàm sóng: Đo lường xác suất |psi|^2 để tạo lại SDL Surface
    void wavefunction_collapse(const std::complex<float>* wavefunction, SDL_Surface* surface);

    // Cổng Giao Thoa QSobel (Trích xuất viền lượng tử O(1) vectorized)
    // Tịnh tiến và tạo giao thoa triệt tiêu
    void quantum_interference_edge_detect(
        const std::complex<float>* psi_in, std::complex<float>* psi_out, 
        int width, int height
    );

} // namespace algorithm
} // namespace dm
