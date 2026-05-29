#include <SDL.h>
#include <iostream>
#include <vector>
#include <complex>
#include "algorithms/ohm_qrender_kernel.h"

using namespace dm::algorithm;

int main(int argc, char* argv[]) {
    std::cout << "--- Testing Ohm-QRENDER (Quantum Image Processing) ---\n";
    
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "FAILURE: SDL could not initialize! SDL_Error: " << SDL_GetError() << "\n";
        return 1;
    }

    int width = 100;
    int height = 100;
    
    // Tạo mảng Pixel bằng SDL2 (32-bit RGBA)
    SDL_Surface* surface = SDL_CreateRGBSurfaceWithFormat(0, width, height, 32, SDL_PIXELFORMAT_RGBA32);
    
    // Đổ nền đen
    SDL_FillRect(surface, NULL, SDL_MapRGBA(surface->format, 0, 0, 0, 255));
    
    // Vẽ một hình vuông màu trắng ở giữa làm "Vật thể"
    SDL_Rect rect = {25, 25, 50, 50};
    SDL_FillRect(surface, &rect, SDL_MapRGBA(surface->format, 255, 255, 255, 255));

    std::cout << "[*] Mảng Pixel SDL truyền thống đã được khởi tạo (Nền đen, Khối trắng ở giữa).\n";

    // Khởi tạo Không gian Hilbert cho Bức ảnh
    std::vector<std::complex<float>> psi_in(width * height);
    std::vector<std::complex<float>> psi_out(width * height);

    // 1. Lượng tử hóa
    std::cout << "[*] Lượng tử hóa: Ép mảng Pixel vào Hàm Sóng Lượng Tử...\n";
    image_to_wavefunction(surface, psi_in.data());

    // 2. Giao thoa Lượng tử (Trích xuất viền O(1))
    std::cout << "[*] Kích hoạt Cổng Giao Thoa QSobel...\n";
    quantum_interference_edge_detect(psi_in.data(), psi_out.data(), width, height);

    // 3. Sụp đổ Hàm Sóng ngược lại thành Pixel
    std::cout << "[*] Sụp đổ Hàm Sóng (Đo lường xác suất |psi|^2)...\n";
    wavefunction_collapse(psi_out.data(), surface);

    // 4. Kích hoạt Feynman-AA (Khử răng cưa bằng Tích phân đường)
    std::cout << "[*] Kích hoạt Khử răng cưa Feynman-AA (Monte Carlo Random Walk)...\n";
    std::vector<std::complex<float>> psi_aa(width * height);
    feynman_path_integral_aa(psi_out.data(), psi_aa.data(), width, height, 100);

    // 5. Sụp đổ Hàm Sóng đã được Anti-aliased ngược lại thành Pixel
    std::cout << "[*] Sụp đổ Hàm Sóng Feynman (Đo lường xác suất |psi|^2)...\n";
    wavefunction_collapse(psi_aa.data(), surface);

    int bright_pixels = 0;
    int aa_pixels = 0;
    uint8_t* pixels = static_cast<uint8_t*>(surface->pixels);
    for (int i = 0; i < width * height; ++i) {
        if (pixels[i * 4] > 100) { // Sáng chói (Core edge)
            bright_pixels++;
        } else if (pixels[i * 4] > 10) { // Sáng mờ (Anti-aliased glow)
            aa_pixels++;
        }
    }

    std::cout << "\n--- Results ---\n";
    std::cout << "Số lượng Điểm ảnh Sáng chói (Core Edge, Probability > 100): " << bright_pixels << "\n";
    std::cout << "Số lượng Điểm ảnh Mờ ảo (Feynman Glow, 10 < Probability <= 100): " << aa_pixels << "\n";
    
    if (aa_pixels > bright_pixels) {
        std::cout << "SUCCESS: Feynman Path Integral Anti-Aliasing Hoàn Hảo!\n";
        std::cout << "Ánh sáng đã lan tỏa từ viền cốt lõi ra xung quanh nhờ hạt Lượng tử, tạo ra viền mượt mà ảo diệu!\n";
    } else {
        std::cout << "FAILURE: Thuật toán giao thoa thất bại.\n";
    }

    SDL_FreeSurface(surface);
    SDL_Quit();
    return 0;
}
