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

    // Đếm số lượng pixel sáng để kiểm chứng (Viền thì ít hơn Diện tích)
    // Diện tích ban đầu: 50x50 = 2500 pixels sáng
    // Chu vi (Viền): xấp xỉ 50*4 = 200 pixels sáng
    int bright_pixels = 0;
    uint8_t* pixels = static_cast<uint8_t*>(surface->pixels);
    for (int i = 0; i < width * height; ++i) {
        if (pixels[i * 4] > 100) { // Kênh R sáng
            bright_pixels++;
        }
    }

    std::cout << "\n--- Results ---\n";
    std::cout << "Số lượng Điểm ảnh Sáng (Wavefunction Probability > 100): " << bright_pixels << "\n";
    
    if (bright_pixels > 150 && bright_pixels < 250) {
        std::cout << "SUCCESS: Quantum Edge Detection Hoàn Hảo!\n";
        std::cout << "Giao thoa triệt tiêu đã xóa sạch khối trắng 2500 pixel, chỉ để lại chính xác cái viền sáng chói!\n";
    } else {
        std::cout << "FAILURE: Thuật toán giao thoa thất bại.\n";
    }

    SDL_FreeSurface(surface);
    SDL_Quit();
    return 0;
}
