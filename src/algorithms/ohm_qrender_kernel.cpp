#include "algorithms/ohm_qrender_kernel.h"
#include <cmath>
#include <algorithm>

namespace dm {
namespace algorithm {

    void image_to_wavefunction(const SDL_Surface* surface, std::complex<float>* wavefunction) {
        int width = surface->w;
        int height = surface->h;
        int pitch = surface->pitch;
        const uint8_t* pixels = static_cast<const uint8_t*>(surface->pixels);
        int bpp = surface->format->BytesPerPixel;

        // Vòng lặp SIMD quét qua toàn bộ Pixel, chuyển hóa thành Hàm sóng
        #pragma omp simd
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int pixel_idx = y * pitch + x * bpp;
                // Lấy cường độ xám (Grayscale) đơn giản từ kênh R (giả sử grayscale hoặc RGB)
                float intensity = static_cast<float>(pixels[pixel_idx]) / 255.0f;
                
                // FRQI: Cường độ sáng được lưu vào phần thực (Amplitude)
                // Pha ban đầu bằng 0
                wavefunction[y * width + x] = std::complex<float>(intensity, 0.0f);
            }
        }
    }

    void wavefunction_collapse(const std::complex<float>* wavefunction, SDL_Surface* surface) {
        int width = surface->w;
        int height = surface->h;
        int pitch = surface->pitch;
        uint8_t* pixels = static_cast<uint8_t*>(surface->pixels);
        int bpp = surface->format->BytesPerPixel;

        #pragma omp simd
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Đo lường: Xác suất P = |psi|^2
                float prob = std::norm(wavefunction[y * width + x]); // norm() trả về thực^2 + ảo^2
                
                // Khuếch đại hiển thị
                float intensity = std::min(prob * 255.0f, 255.0f);
                uint8_t color = static_cast<uint8_t>(intensity);

                int pixel_idx = y * pitch + x * bpp;
                pixels[pixel_idx] = color;       // R
                if (bpp > 1) pixels[pixel_idx+1] = color; // G
                if (bpp > 2) pixels[pixel_idx+2] = color; // B
                if (bpp > 3) pixels[pixel_idx+3] = 255;   // A
            }
        }
    }

    void quantum_interference_edge_detect(
        const std::complex<float>* psi_in, std::complex<float>* psi_out, 
        int width, int height
    ) {
        // QSobel: Giao thoa triệt tiêu O(1) vectorized
        // Dịch chuyển pha và tạo sự chồng chập giữa hàm sóng hiện tại và hàm sóng bị tịnh tiến
        
        #pragma omp simd
        for (int i = 0; i < width * height; ++i) {
            int x = i % width;
            int y = i / width;

            std::complex<float> center = psi_in[i];

            // Biên
            if (x == width - 1 || y == height - 1) {
                psi_out[i] = std::complex<float>(0.0f, 0.0f);
                continue;
            }

            std::complex<float> right = psi_in[i + 1];
            std::complex<float> bottom = psi_in[i + width];

            // Phép trừ số phức -> Giao thoa triệt tiêu (Destructive Interference)
            // Nếu nền trơn (center == right == bottom), gradient = 0
            // Nếu có viền, gradient sinh ra biên độ khác 0
            std::complex<float> grad_x = center - right;
            std::complex<float> grad_y = center - bottom;

            // Chồng chập 2 hướng viền (Superposition of gradients)
            psi_out[i] = grad_x + std::complex<float>(0.0f, 1.0f) * grad_y;
        }
    }

} // namespace algorithm
} // namespace dm
