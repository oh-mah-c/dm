#include <SDL.h>
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "--- Testing Ohm-QRENDER (SDL2 Native Core Initialization) ---\n";
    
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "FAILURE: SDL could not initialize! SDL_Error: " << SDL_GetError() << "\n";
        return 1;
    }
    
    std::cout << "[+] SUCCESS: SDL2 Initialized Successfully as a Native Core!\n";
    std::cout << "[+] The Quantum Framebuffer is ready to be collapsed.\n";
    
    SDL_Quit();
    return 0;
}
