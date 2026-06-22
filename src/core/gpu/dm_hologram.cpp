#include "dm_hologram.h"
#include <iostream>

namespace ohm {
namespace gpu {

HologramRenderer::HologramRenderer(mjModel* m, mjData* d) : m_(m), d_(d) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "[Hologram] SDL Init Failed: " << SDL_GetError() << "\n";
        return;
    }

    // Configure OpenGL context parameters required by MuJoCo
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

    window_ = SDL_CreateWindow("Ohm Bifrost Holography",
                               SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                               1280, 720,
                               SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
    
    if (!window_) {
        std::cerr << "[Hologram] Window Creation Failed: " << SDL_GetError() << "\n";
        return;
    }

    gl_context_ = SDL_GL_CreateContext(window_);
    SDL_GL_MakeCurrent(window_, gl_context_);
    
    // Initialize MuJoCo visualization objects
    mjv_defaultCamera(&cam_);
    mjv_defaultOption(&opt_);
    mjv_defaultScene(&scn_);
    mjr_defaultContext(&con_);

    mjv_makeScene(m_, &scn_, 2000); // Buffer for 2000 geometry objects
    mjr_makeContext(m_, &con_, mjFONTSCALE_150); // Initialize custom OpenGL context
    
    std::cout << "[Hologram] Visual Interface Online. (SDL2 + OpenGL Hybrid)\n";
}

HologramRenderer::~HologramRenderer() {
    mjr_freeContext(&con_);
    mjv_freeScene(&scn_);
    if (gl_context_) SDL_GL_DeleteContext(gl_context_);
    if (window_) SDL_DestroyWindow(window_);
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
}

bool HologramRenderer::should_close() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) {
            is_running_ = false;
        }
    }
    return !is_running_;
}

void HologramRenderer::render() {
    if (!window_ || !is_running_) return;

    int w, h;
    SDL_GetWindowSize(window_, &w, &h);
    
    // Sync the visualization scene with the physics state
    mjv_updateScene(m_, d_, &opt_, NULL, &cam_, mjCAT_ALL, &scn_);
    
    // Render the scene to the viewport
    mjrRect viewport = {0, 0, w, h};
    mjr_render(viewport, &scn_, &con_);
    
    // Swap OpenGL buffers to present the frame
    SDL_GL_SwapWindow(window_);
}

} // namespace gpu
} // namespace ohm
