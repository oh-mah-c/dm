#pragma once
#include <mujoco/mujoco.h>
#include <SDL.h>

namespace ohm {
namespace gpu {

class HologramRenderer {
public:
    HologramRenderer(mjModel* m, mjData* d);
    ~HologramRenderer();

    // Renders the current MuJoCo state to the SDL window
    void render();
    
    // Check if the user closed the window and process SDL events
    bool should_close();

private:
    mjModel* m_;
    mjData*  d_;

    mjvScene scn_;
    mjvCamera cam_;
    mjvOption opt_;
    mjrContext con_;

    SDL_Window* window_ = nullptr;
    SDL_GLContext gl_context_ = nullptr;
    
    bool is_running_ = true;
};

} // namespace gpu
} // namespace ohm
