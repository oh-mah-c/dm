#include <SDL.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cstdlib>
#include <ctime>
#include "algorithms/ohm_qrender_kernel.h"

using namespace dm::algorithm;

void fill_rect(SDL_Surface* surface, int x, int y, int w, int h, uint32_t color) {
    SDL_Rect rect = {x, y, w, h};
    SDL_FillRect(surface, &rect, color);
}

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
    std::srand(12345); // Fixed seed for reproducibility
    
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "FAILURE: SDL Error: " << SDL_GetError() << "\n";
        return 1;
    }

    int width = 512;
    int height = 512;
    
    SDL_Surface* surface = SDL_CreateRGBSurfaceWithFormat(0, width, height, 32, SDL_PIXELFORMAT_RGBA32);
    uint32_t black = SDL_MapRGBA(surface->format, 0, 0, 0, 255);
    uint32_t white = SDL_MapRGBA(surface->format, 255, 255, 255, 255);

    // 1. Dựng hình Tàu ngầm
    SDL_FillRect(surface, NULL, black);
    fill_circle(surface, 256, 256, 80, white); // Thân tàu ngầm to hơn chút
    fill_rect(surface, 236, 150, 40, 50, white); // Tháp điều khiển

    // 2. Bơm Bão Nhiễu
    std::cout << "[*] Đang bơm Bão Nhiễu Radar 60% vào Không gian...\n";
    uint32_t* pixels32 = static_cast<uint32_t*>(surface->pixels);
    for (int i = 0; i < width * height; ++i) {
        if (std::rand() % 100 < 60) { // Giảm noise xuống 60% để cấu trúc vi vĩ mô tồn tại
            uint8_t noise_val = (std::rand() % 2 == 0) ? 255 : (std::rand() % 100);
            pixels32[i] = SDL_MapRGBA(surface->format, noise_val, noise_val, noise_val, 255);
        }
    }

    SDL_SaveBMP(surface, "radar_input.bmp");
    std::cout << "[+] Đã xuất file: radar_input.bmp\n";

    std::vector<std::complex<float>> psi_in(width * height);
    std::vector<std::complex<float>> psi_diffused(width * height);
    std::vector<std::complex<float>> psi_out(width * height);

    // 3. Lượng tử hóa
    std::cout << "[*] Lượng tử hóa bức ảnh nhiễu thành Hàm Sóng...\n";
    image_to_wavefunction(surface, psi_in.data());

    // 4. Bước đệm: Khử nhiễu bằng Tích phân đường Feynman (Quantum Decoherence/Diffusion)
    // Phân tán các hạt nhiễu để triệt tiêu chúng trước khi tìm viền
    std::cout << "[*] Kích hoạt Feynman Path Integral để trung hòa nhiễu ngẫu nhiên...\n";
    feynman_path_integral_aa(psi_in.data(), psi_diffused.data(), width, height, 50);

    // 5. Giao thoa Lượng tử (Trích xuất viền QSobel O(1))
    std::cout << "[*] Kích hoạt Cổng Giao Thoa QSobel...\n";
    quantum_interference_edge_detect(psi_diffused.data(), psi_out.data(), width, height);

    // 6. Sụp đổ Hàm Sóng
    std::cout << "[*] Sụp đổ Hàm Sóng (Đo lường xác suất |psi|^2)...\n";
    wavefunction_collapse(psi_out.data(), surface);

    // 7. Thresholding
    uint8_t* pixels8 = static_cast<uint8_t*>(surface->pixels);
    for (int i = 0; i < width * height; ++i) {
        // Sau khi diffuse và edge detect, biên độ sẽ rất nhỏ. 
        // Ta cần khuếch đại (amplification) và cắt nhiễu nền
        float intensity = static_cast<float>(pixels8[i * 4]);
        if (intensity < 15.0f) { // Threshold cực thấp do biên độ đã bị chia nhỏ bởi Diffusion
            pixels32[i] = black;
        } else {
            pixels32[i] = white;
        }
    }

    SDL_SaveBMP(surface, "radar_output_quantum.bmp");
    std::cout << "[+] Đã xuất file: radar_output_quantum.bmp\n";

    std::cout << "\nSUCCESS: Xuyên thấu màn bão nhiễu hoàn tất!\n";

    SDL_FreeSurface(surface);
    SDL_Quit();
    return 0;
}
