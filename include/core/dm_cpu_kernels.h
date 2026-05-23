/**
 * include/core/dm_cpu_kernels.h — Tier-0 pure-C CPU kernel prototypes
 *
 * These are always available and are called as the final fallback when
 * neither TFE nor Vulkan backends are initialised.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef DM_CPU_KERNELS_H
#define DM_CPU_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * C = A × B   (A[M×K], B[K×N] → C[M×N])
 * Uses 8×8 blocking for cache efficiency.
 */
void dm_cpu_matmul(const float *A, const float *B, float *C, int M, int K, int N);

/**
 * C = A × Bᵀ  (A[M×K], B[N×K] → C[M×N])
 * B is stored as [N×K] row-major (transposed).
 */
void dm_cpu_matmul_nt(const float *A, const float *B, float *C, int M, int N, int K);

/**
 * 2-D convolution with SAME padding (naive, always correct).
 * in/out layout: NCHW.  Weights layout: OIHW [out_c][in_c][kernel][kernel].
 * @param w_   input width (named w_ to avoid shadowing 'w' weight arg)
 * Returns 0 on success.
 */
int dm_cpu_conv2d_same(const float *in, float *out,
                        const float *w,  const float *b,
                        int n, int in_c, int out_c,
                        int h, int w_, int kernel, int stride);

/**
 * Global average pool.
 * in[n, c, h, w] → out[n, c, 1, 1]  (out must hold n×c floats)
 */
void dm_cpu_global_avg_pool(const float *in, float *out, int n, int c, int h, int w);

/** ReLU in-place on a flat float buffer of length n. */
void dm_cpu_relu(float *x, int n);

/**
 * Numerically-stable row-wise softmax.
 * x[rows × cols] updated in-place; each row becomes a probability distribution.
 */
void dm_cpu_softmax(float *x, int rows, int cols);

/**
 * Layer normalisation in-place.
 * x[rows × cols];  gamma and beta are per-column scale/shift (may be NULL).
 */
void dm_cpu_layer_norm(float *x, int rows, int cols,
                        const float *gamma, const float *beta, float eps);

#ifdef __cplusplus
}
#endif

#endif /* DM_CPU_KERNELS_H */
