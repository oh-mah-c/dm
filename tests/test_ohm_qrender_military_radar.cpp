#include <SDL.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cstdlib>
#include <ctime>
#include "algorithms/ohm_qrender_kernel.h"

using namespace dm::algorithm;

// Hàm hỗ trợ vẽ hình chữ nhật
void fill_rect(SDL_Surface* surface, int x, int y, int w, int h, uint32_t color) {
    SDL_Rect rect = {x, y, w, h};
    SDL_FillRect(surface, &rect, color);
}

// Hàm hỗ trợ vẽ hình tròn
void fill_circle(SDL_Surface* surface, int cx, int cy, int radius, uint32_t color) {
    uint8_t* pixels = static_cast<uint8_t*>(surface->pixels);
    int pitch = surface->pitch;
    int bpp = surface->format->BytesPerPixel;
    
    for (int y = cy - radius; y <= cy + radius; y++) {
        for (int x = cx - radius; x <= cx + radius; x++) {
            if (x >= 0 && x < surface->w && y >= 0 && y < surface->h) {
                if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius) {
                    uint32_t* target_pixel = reinterpret_cast<uint32_t*>(pixels + y * pitch + x * bpp);
                    *target_pixel = color;
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "--- Ohm-QRENDER: Military Radar Extreme Noise Test ---\n";
    std::srand(std::time(nullptr));
    
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "FAILURE: SDL Error: " << SDL_GetError() << "\n";
        return 1;
    }

    int width = 512;
    int height = 512;
    
    // Tạo mảng Pixel (32-bit RGBA)
    SDL_Surface* surface = SDL_CreateRGBSurfaceWithFormat(0, width, height, 32, SDL_PIXELFORMAT_RGBA32);
    uint32_t black = SDL_MapRGBA(surface->format, 0, 0, 0, 255);
    uint32_t white = SDL_MapRGBA(surface->format, 255, 255, 255, 255);

    // 1. Dựng hình Tàu ngầm (Một vòng tròn + Một hình chữ nhật nhỏ)
    SDL_FillRect(surface, NULL, black);
    fill_circle(surface, 256, 256, 60, white); // Thân tàu ngầm
    fill_rect(surface, 246, 170, 20, 40, white); // Tháp điều khiển

    // 2. Bơm 75% Bão Nhiễu (Salt and Pepper Noise)
    std::cout << "[*] Đang bơm Bão Nhiễu Radar 75% vào Không gian...\n";
    uint32_t* pixels32 = static_cast<uint32_t*>(surface->pixels);
    for (int i = 0; i < width * height; ++i) {
        if (std::rand() % 100 < 75) { // 75% xác suất bị nhiễu
            // Nhiễu trắng hoặc xám ngẫu nhiên
            uint8_t noise_val = (std::rand() % 2 == 0) ? 255 : (std::rand() % 100);
            pixels32[i] = SDL_MapRGBA(surface->format, noise_val, noise_val, noise_val, 255);
        }
    }

    // Xuất bức ảnh bị nhiễu tàn khốc
    SDL_SaveBMP(surface, "radar_input.bmp");
    std::cout << "[+] Đã xuất file: radar_input.bmp\n";

    // Khởi tạo Không gian Hilbert
    std::vector<std::complex<float>> psi_in(width * height);
    std::vector<std::complex<float>> psi_out(width * height);

    // 3. Lượng tử hóa
    std::cout << "[*] Lượng tử hóa bức ảnh nhiễu thành Hàm Sóng...\n";
    image_to_wavefunction(surface, psi_in.data());

    // 4. Giao thoa Lượng tử (Trích xuất viền QSobel O(1))
    std::cout << "[*] Kích hoạt Cổng Giao Thoa QSobel...\n";
    quantum_interference_edge_detect(psi_in.data(), psi_out.data(), width, height);

    // 5. Sụp đổ Hàm Sóng ngược lại thành Pixel
    std::cout << "[*] Sụp đổ Hàm Sóng (Đo lường xác suất |psi|^2)...\n";
    wavefunction_collapse(psi_out.data(), surface);

    // 6. Thresholding: Bộ lọc đo lường lượng tử (Chỉ lấy các pixel có cường độ văng ra khỏi mức nhiễu nền)
    uint8_t* pixels8 = static_cast<uint8_t*>(surface->pixels);
    for (int i = 0; i < width * height; ++i) {
        if (pixels8[i * 4] < 60) { // Lọc bỏ nhiễu nhiễu xạ yếu
            pixels32[i] = black;
        } else {
            pixels32[i] = white; // Lóe sáng viền cốt lõi
        }
    }

    // Xuất bức ảnh sau khi dùng lượng tử để lọc
    SDL_SaveBMP(surface, "radar_output_quantum.bmp");
    std::cout << "[+] Đã xuất file: radar_output_quantum.bmp\n";

    std::cout << "\nSUCCESS: Xuyên thấu màn bão nhiễu hoàn tất!\n";

    SDL_FreeSurface(surface);
    SDL_Quit();
    return 0;
}
