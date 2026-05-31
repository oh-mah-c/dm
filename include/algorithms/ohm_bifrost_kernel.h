#ifndef DM_ALGORITHM_OHM_BIFROST_KERNEL_H
#define DM_ALGORITHM_OHM_BIFROST_KERNEL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
namespace dm {
namespace algorithm {

/**
 * @brief Ohm-BIFROST: Bijective Forward/Reverse Operations via Spatio-Temporal Transformations
 * 
 * Reversible Affine Coupling Layer step for unified coordinate mapping.
 * Performs strictly in-place, zero-allocation transformations.
 *
 * @param state The state vector (e.g. Pixel space or Latent space)
 * @param total_dim Total dimension size of the state vector (must be even)
 * @param is_forward If true, performs the forward transform (A -> A, B -> B + f(A)). 
 *                   If false, performs reverse (A -> A, B -> B - f(A)).
 */
void bifrost_coupling_step_cpu(float* state, size_t total_dim, bool is_forward);

} // namespace algorithm
} // namespace dm

extern "C" {
#endif

// C-compatible wrapper for integration with dm_core framework if needed
void dm_ohm_bifrost_coupling_step(float* state, size_t total_dim, int is_forward);

#ifdef __cplusplus
}
#endif

#endif // DM_ALGORITHM_OHM_BIFROST_KERNEL_H
