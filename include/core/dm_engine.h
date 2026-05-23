/*
 * dm_engine.h — DM tensor type and neural-op primitives
 *
 * All compute dispatches through the TensorFlow Eager C API (TFE_*) so every
 * model automatically inherits the optimised TF C++ backend (XLA, cuDNN,
 * oneDNN, …).
 *
 * DM_Tensor layout: NCHW (n, c, h, w) — row-major, contiguous float32.
 *
 * Optimiser step functions (adam/adagrad/sgd_momentum) and all analytical
 * backward passes are pure-C; ResourceApply* ops would require TF Variable
 * handles which add no practical benefit for parameter-array updates.
 */

#ifndef DM_ENGINE_H
#define DM_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include "core/dm_block.h"

/* ── Tensor data structure ───────────────────────────────────────────────── */
/*
 * DM_Tensor is also declared in include/dm.h (§ 8).  When dm.h has been
 * included first, skip the re-declaration to avoid "conflicting types"
 * errors in translation units that include both headers (e.g. dm_lib.c).
 */
#ifndef DM_H   /* dm.h defines DM_H */

typedef struct {
    int    n, c, h, w;
    float *data;
} DM_Tensor;

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

int    dm_tensor_alloc(DM_Block *t, int n, int c, int h, int w);
void   dm_tensor_free (DM_Block *t);
void   dm_tensor_fill (DM_Block *t, float value);
float  dm_tensor_get  (const DM_Block *t, int n, int c, int y, int x);
void   dm_tensor_set  (DM_Block *t, int n, int c, int y, int x, float v);
size_t dm_tensor_count(const DM_Block *t);

#else  /* dm.h was included first — lifecycle decls come from there */

/* Re-export the internal (non-DM_API) versions so internal code can call
 * them without the DM_API visibility attribute. */
int    dm_tensor_alloc(DM_Block *t, int n, int c, int h, int w);
void   dm_tensor_free (DM_Block *t);
void   dm_tensor_fill (DM_Block *t, float value);
float  dm_tensor_get  (const DM_Block *t, int n, int c, int y, int x);
void   dm_tensor_set  (DM_Block *t, int n, int c, int y, int x, float v);
size_t dm_tensor_count(const DM_Block *t);

#endif /* DM_H */

/* ── Convolutions (TFE Conv2D / DepthwiseConv2dNative, NCHW, SAME) ────────
 * Weight layout for dm_conv2d_same:  w[out_c][in_c][ky][kx]  (OIHW)
 * Weight layout for dm_depthwise:    w[c][ky][kx]            (transposed internally)
 */
int dm_conv2d_same(const DM_Block *in, DM_Block *out,
                   const float *w, const float *b,
                   int out_c, int kernel, int stride);

int dm_depthwise_conv2d_same(const DM_Block *in, DM_Block *out,
                              const float *w, const float *b,
                              int kernel, int stride);

int dm_pointwise_conv2d(const DM_Block *in, DM_Block *out,
                         const float *w, const float *b, int out_c);

/* ── Activations (TFE) ────────────────────────────────────────────────────── */
void dm_relu6          (DM_Block *t);
void dm_relu           (DM_Block *t);
void dm_tanh_inplace   (DM_Block *t);
void dm_sigmoid_inplace(DM_Block *t);
void dm_gelu_inplace   (float *x, int n);   /* raw float buffer */

/* ── Elementwise (TFE AddV2) ─────────────────────────────────────────────── */
int dm_tensor_add(DM_Block *out, const DM_Block *in);

/* ── Pooling (TFE MaxPool / Mean, NCHW, SAME) ───────────────────────────── */
int dm_max_pool2d_same(const DM_Block *in, DM_Block *out,
                        int kernel, int stride);
int dm_global_avg_pool(const DM_Block *in, DM_Block *out);

/* ── Normalisation (TFE FusedBatchNorm / Mean pipeline) ─────────────────── */
int dm_batch_norm(DM_Block *t,
                  const float *gamma, const float *beta,
                  const float *mean,  const float *var, float eps);

/* Layer-norm over sequence rows: x[seq_len × d_model] (TFE pipeline) */
int dm_layer_norm_seq(float *x, int seq_len, int d_model,
                      const float *gamma, const float *beta, float eps);

/* ── Linear / Fully-connected (TFE MatMul + AddV2) ─────────────────────── */
/* in:  [n, in_c, 1, 1]  →  out: [n, out_c, 1, 1] */
int dm_linear(const DM_Block *in, DM_Block *out,
              const float *w, const float *b, int out_c);

/* ── Softmax (TFE) ───────────────────────────────────────────────────────── */
void dm_softmax     (DM_Block *t);               /* [n,c,1,1] over channel dim */
void dm_softmax_rows(float *x, int rows, int cols); /* raw [rows×cols] in-place  */

/* ── Matrix multiplications (TFE MatMul) ────────────────────────────────── */
/* C = A @ B^T   A[M×K], B[N×K] → C[M×N] */
void dm_matmul_nt(const float *A, const float *B, float *C, int M, int N, int K);
/* C = A @ B     A[M×K], B[K×N] → C[M×N] */
void dm_matmul_nn(const float *A, const float *B, float *C, int M, int K, int N);

/* ── Training primitives (pure-C — no TFE needed) ───────────────────────── */

/* Backward passes */
int  dm_linear_backward(const DM_Block *in, const DM_Block *grad_out,
                         DM_Block *grad_in,
                         float *grad_w, float *grad_b,
                         const float *w, int out_c);
void dm_tanh_backward  (const DM_Block *out, const DM_Block *grad_out,
                         DM_Block *grad_in);
void dm_relu_backward  (const DM_Block *in,  const DM_Block *grad_out,
                         DM_Block *grad_in);

/* Maxout */
int dm_maxout         (const DM_Block *in,  DM_Block *out, int k, int *argmax);
int dm_maxout_backward(const DM_Block *grad_out, DM_Block *grad_in,
                        int k, const int *argmax);

/* Dropout */
void dm_dropout         (const DM_Block *in,  DM_Block *out,
                          float drop_prob, int *mask);
void dm_dropout_backward(const DM_Block *grad_out, DM_Block *grad_in,
                          float drop_prob, const int *mask);

/* Optimiser steps */
void dm_adagrad_step(float *param, float *grad, float *g_sum,
                     int n, float lr, float eps, float weight_decay);

void dm_adam_step(float *param, float *grad, float *m, float *v,
                  int n, float lr,
                  float beta1, float beta2, float eps,
                  float weight_decay, int t);

void dm_sgd_momentum_step(float *param, float *grad, float *velocity,
                           int n, float lr, float momentum,
                           float weight_decay, int nesterov);

#endif /* DM_ENGINE_H */
