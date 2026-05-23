#ifndef DM_TENSOR_H
#define DM_TENSOR_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int n;
    int c;
    int h;
    int w;
    float *data;
} DM_Tensor;

int dm_tensor_alloc(DM_Tensor *t, int n, int c, int h, int w);
void dm_tensor_free(DM_Tensor *t);
void dm_tensor_fill(DM_Tensor *t, float value);
float dm_tensor_get(const DM_Tensor *t, int n, int c, int y, int x);
void dm_tensor_set(DM_Tensor *t, int n, int c, int y, int x, float v);
size_t dm_tensor_count(const DM_Tensor *t);

int dm_conv2d_same(const DM_Tensor *in, DM_Tensor *out, const float *w, const float *b,
                   int out_c, int kernel, int stride);
int dm_depthwise_conv2d_same(const DM_Tensor *in, DM_Tensor *out, const float *w, const float *b,
                             int kernel, int stride);
int dm_pointwise_conv2d(const DM_Tensor *in, DM_Tensor *out, const float *w, const float *b, int out_c);
void dm_relu6(DM_Tensor *t);
void dm_relu(DM_Tensor *t);
int dm_tensor_add(DM_Tensor *out, const DM_Tensor *in);
int dm_max_pool2d_same(const DM_Tensor *in, DM_Tensor *out, int kernel, int stride);
int dm_batch_norm(DM_Tensor *t, const float *gamma, const float *beta, const float *mean, const float *var, float eps);
int dm_global_avg_pool(const DM_Tensor *in, DM_Tensor *out);
int dm_linear(const DM_Tensor *in, DM_Tensor *out, const float *w, const float *b, int out_c);
void dm_softmax(DM_Tensor *t);

/* ViT primitives */
/* Layer Norm over last dim (channel dim c) of shape (n,c,1,1) or (seq,1,c,1) */
int  dm_layer_norm_seq(float *x, int seq_len, int d_model, const float *gamma, const float *beta, float eps);
void dm_gelu_inplace(float *x, int n);
/* C = A @ B^T  shapes: A[M×K], B[N×K] → C[M×N]  (no tensor wrapper, raw float) */
void dm_matmul_nt(const float *A, const float *B, float *C, int M, int N, int K);
/* C = A @ B    shapes: A[M×K], B[K×N] → C[M×N] */
void dm_matmul_nn(const float *A, const float *B, float *C, int M, int K, int N);
void dm_softmax_rows(float *x, int rows, int cols);

#endif /* DM_TENSOR_H */
