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

    // Một hàm tạo số ngẫu nhiên Xorshift32 siêu tốc
    static inline uint32_t xorshift32(uint32_t* state) {
        uint32_t x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        return x;
    }

    void feynman_path_integral_aa(
        const std::complex<float>* psi_in, std::complex<float>* aa_out, 
        int width, int height, int num_paths
    ) {
        // Tích phân đường Feynman:
        // Ánh sáng từ mỗi pixel có cơ hội "nhảy" sang các pixel lân cận.
        // Hạt di chuyển ngẫu nhiên. Mỗi bước nhảy tiêu tốn Tác dụng (Action S).
        // Tổng biên độ xác suất = sum(e^(i * S))
        
        #pragma omp parallel for
        for (int y = 0; y < height; ++y) {
            uint32_t seed = 123456789 + y * 987654321; // RNG state
            for (int x = 0; x < width; ++x) {
                int i = y * width + x;
                std::complex<float> sum_amplitude(0.0f, 0.0f);

                // Nếu pixel hiện tại hoàn toàn tối, bỏ qua (để tăng tốc)
                // Nhưng trong lý thuyết lượng tử, mọi điểm đều có thể nhận photon.
                // Để tối ưu, ta bắn photon từ chính nó và tích lũy.

                for (int p = 0; p < num_paths; ++p) {
                    int cx = x;
                    int cy = y;
                    float action = 0.0f;
                    
                    // Thực hiện một vài bước nhảy (Random Walk)
                    int max_steps = 3; 
                    for (int step = 0; step < max_steps; ++step) {
                        uint32_t rand_val = xorshift32(&seed);
                        int dir = rand_val % 4; // 0: L, 1: R, 2: U, 3: D
                        if (dir == 0 && cx > 0) cx--;
                        else if (dir == 1 && cx < width - 1) cx++;
                        else if (dir == 2 && cy > 0) cy--;
                        else if (dir == 3 && cy < height - 1) cy++;

                        // Tác dụng tăng lên theo số bước
                        action += 1.0f; // S = integral(L dt)
                    }

                    // Tới điểm đích, lấy biên độ gốc tại đó
                    std::complex<float> origin_amp = psi_in[cy * width + cx];
                    
                    // e^{i S / hbar}, giả sử hbar = 1
                    std::complex<float> phase_factor(std::cos(action), std::sin(action));
                    
                    sum_amplitude += origin_amp * phase_factor;
                }

                // Trung bình hóa (Normalization)
                aa_out[i] = sum_amplitude / static_cast<float>(num_paths);
            }
        }
    }

} // namespace algorithm
} // namespace dm
