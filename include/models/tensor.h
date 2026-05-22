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
int dm_global_avg_pool(const DM_Tensor *in, DM_Tensor *out);
int dm_linear(const DM_Tensor *in, DM_Tensor *out, const float *w, const float *b, int out_c);
void dm_softmax(DM_Tensor *t);

#endif /* DM_TENSOR_H */
