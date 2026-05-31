#include "algorithms/ohm_bifrost_kernel.h"
#include <cmath>

namespace dm {
namespace algorithm {

// Reversible Affine Coupling Layer Step
void bifrost_coupling_step_cpu(float* state, size_t total_dim, bool is_forward) {
    if (!state || total_dim < 2) return;
    
    // Split the state array into two halves
    size_t half_dim = total_dim / 2;
    float* A = state;                // Half A: Stays unchanged
    float* B = state + half_dim;     // Half B: Gets transformed

    // Non-linear transformation function f(A)
    // To ensure exact mathematical reversibility, this function must be deterministic.
    for (size_t i = 0; i < half_dim; ++i) {
        // A simple non-linear scalar shift based on A
        // Using sin() and a mixing scalar. 
        // Note: For real Neural Nets, f(A) could be an arbitrary complex MLP.
        float shift = std::sin(A[i]) * 0.5f + std::cos(A[i] * 2.1f) * 0.3f;
        
        if (is_forward) {
            // Forward: Image -> [Latent + Noise]
            B[i] = B[i] + shift;
        } else {
            // Reverse: [Latent + Noise] -> Image
            // Exact cancellation of the shift guarantees 100% data recovery
            B[i] = B[i] - shift;
        }
    }
}

} // namespace algorithm
} // namespace dm

extern "C" {

void dm_ohm_bifrost_coupling_step(float* state, size_t total_dim, int is_forward) {
    dm::algorithm::bifrost_coupling_step_cpu(state, total_dim, is_forward != 0);
}

}
