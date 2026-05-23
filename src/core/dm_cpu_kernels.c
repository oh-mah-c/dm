/**
 * src/core/dm_cpu_kernels.c — Tier-0 pure-C CPU fallback kernels
 *
 * These are called when no GPU backend (Vulkan / TFE) is available.
 * They are also the authoritative reference implementations.
 *
 * Performance notes:
 *   - matmul uses an 8×8 register tile to improve cache reuse
 *   - All other ops are cache-friendly passes over contiguous arrays
 *   - No SIMD intrinsics are used — the compiler is expected to
 *     auto-vectorise with -O2 on all supported targets
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/dm_cpu_kernels.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <float.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * Matrix multiplication — tiled 8×8
 * ═══════════════════════════════════════════════════════════════════════════ */

#define TILE 8

/**
 * C = A × B   (A[M×K], B[K×N] → C[M×N])
 */
void dm_cpu_matmul(const float *A, const float *B, float *C,
                   int M, int K, int N)
{
    memset(C, 0, (size_t)M * N * sizeof(float));

    for (int i0 = 0; i0 < M; i0 += TILE) {
        int imax = i0 + TILE < M ? i0 + TILE : M;
        for (int j0 = 0; j0 < N; j0 += TILE) {
            int jmax = j0 + TILE < N ? j0 + TILE : N;
            for (int k0 = 0; k0 < K; k0 += TILE) {
                int kmax = k0 + TILE < K ? k0 + TILE : K;
                /* inner 8×8 tile */
                for (int i = i0; i < imax; i++) {
                    for (int k = k0; k < kmax; k++) {
                        float a = A[i * K + k];
                        for (int j = j0; j < jmax; j++) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

/**
 * C = A × Bᵀ  (A[M×K], B[N×K] → C[M×N])
 * B is stored row-major as [N×K] — equivalent to transposing before multiply.
 */
void dm_cpu_matmul_nt(const float *A, const float *B, float *C,
                      int M, int N, int K)
{
    memset(C, 0, (size_t)M * N * sizeof(float));

    for (int i0 = 0; i0 < M; i0 += TILE) {
        int imax = i0 + TILE < M ? i0 + TILE : M;
        for (int j0 = 0; j0 < N; j0 += TILE) {
            int jmax = j0 + TILE < N ? j0 + TILE : N;
            for (int k0 = 0; k0 < K; k0 += TILE) {
                int kmax = k0 + TILE < K ? k0 + TILE : K;
                for (int i = i0; i < imax; i++) {
                    for (int j = j0; j < jmax; j++) {
                        float acc = 0.0f;
                        for (int k = k0; k < kmax; k++) {
                            acc += A[i * K + k] * B[j * K + k];
                        }
                        C[i * N + j] += acc;
                    }
                }
            }
        }
    }
}

#undef TILE

/* ═══════════════════════════════════════════════════════════════════════════
 * Conv2d SAME padding — naive but correct
 * Layout: in/out NCHW, weights OIHW [out_c][in_c][kernel][kernel]
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_cpu_conv2d_same(const float *in, float *out,
                        const float *w,  const float *b,
                        int n, int in_c, int out_c,
                        int h, int w_, int kernel, int stride)
{
    int out_h = (h + stride - 1) / stride;
    int out_w = (w_ + stride - 1) / stride;
    int pad_h = ((out_h - 1) * stride + kernel - h);
    int pad_w = ((out_w - 1) * stride + kernel - w_);
    int pad_top  = pad_h / 2;
    int pad_left = pad_w / 2;

    memset(out, 0, (size_t)n * out_c * out_h * out_w * sizeof(float));

    for (int ni = 0; ni < n; ni++) {
        for (int oc = 0; oc < out_c; oc++) {
            float bias = b ? b[oc] : 0.0f;
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow_ = 0; ow_ < out_w; ow_++) {
                    float acc = bias;
                    for (int ic = 0; ic < in_c; ic++) {
                        for (int kh = 0; kh < kernel; kh++) {
                            for (int kw = 0; kw < kernel; kw++) {
                                int ih = oh * stride + kh - pad_top;
                                int iw = ow_ * stride + kw - pad_left;
                                if (ih < 0 || ih >= h || iw < 0 || iw >= w_)
                                    continue;
                                int in_idx  = ni * (in_c * h * w_)
                                            + ic * (h * w_)
                                            + ih * w_ + iw;
                                int w_idx   = oc * (in_c * kernel * kernel)
                                            + ic * (kernel * kernel)
                                            + kh * kernel + kw;
                                acc += in[in_idx] * w[w_idx];
                            }
                        }
                    }
                    int out_idx = ni * (out_c * out_h * out_w)
                                + oc * (out_h * out_w)
                                + oh * out_w + ow_;
                    out[out_idx] = acc;
                }
            }
        }
    }
    return 0; /* DM_OK */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Global average pool — NCHW layout
 * in[n, c, h, w] → out[n, c, 1, 1]
 * ═══════════════════════════════════════════════════════════════════════════ */

void dm_cpu_global_avg_pool(const float *in, float *out,
                             int n, int c, int h, int w)
{
    int hw = h * w;
    float inv_hw = 1.0f / (float)hw;
    for (int ni = 0; ni < n; ni++) {
        for (int ci = 0; ci < c; ci++) {
            float acc = 0.0f;
            const float *src = in + (ni * c + ci) * hw;
            for (int i = 0; i < hw; i++) acc += src[i];
            out[ni * c + ci] = acc * inv_hw;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ReLU — in-place on flat buffer
 * ═══════════════════════════════════════════════════════════════════════════ */

void dm_cpu_relu(float *x, int n)
{
    for (int i = 0; i < n; i++)
        if (x[i] < 0.0f) x[i] = 0.0f;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Softmax — in-place, numerically stable, over cols dimension
 * x[rows × cols], each row is an independent distribution
 * ═══════════════════════════════════════════════════════════════════════════ */

void dm_cpu_softmax(float *x, int rows, int cols)
{
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        /* find max for numerical stability */
        float mx = -FLT_MAX;
        for (int j = 0; j < cols; j++) if (row[j] > mx) mx = row[j];
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row[j] = expf(row[j] - mx);
            sum += row[j];
        }
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < cols; j++) row[j] *= inv_sum;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Layer normalisation — in-place
 * x[rows × cols];  gamma[cols], beta[cols]
 * ═══════════════════════════════════════════════════════════════════════════ */

void dm_cpu_layer_norm(float *x, int rows, int cols,
                        const float *gamma, const float *beta, float eps)
{
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;
        /* mean */
        float mean = 0.0f;
        for (int j = 0; j < cols; j++) mean += row[j];
        mean /= (float)cols;
        /* variance */
        float var = 0.0f;
        for (int j = 0; j < cols; j++) {
            float d = row[j] - mean;
            var += d * d;
        }
        var /= (float)cols;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < cols; j++) {
            float norm = (row[j] - mean) * inv_std;
            row[j] = gamma ? (norm * gamma[j] + beta[j]) : norm;
        }
    }
}
