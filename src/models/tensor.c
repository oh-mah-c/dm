#include "models/tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

size_t dm_tensor_count(const DM_Tensor *t) {
    if (!t || t->n <= 0 || t->c <= 0 || t->h <= 0 || t->w <= 0) return 0;
    return (size_t)t->n * (size_t)t->c * (size_t)t->h * (size_t)t->w;
}

int dm_tensor_alloc(DM_Tensor *t, int n, int c, int h, int w) {
    size_t count;
    if (!t || n <= 0 || c <= 0 || h <= 0 || w <= 0) return -1;
    memset(t, 0, sizeof(*t));
    t->n = n; t->c = c; t->h = h; t->w = w;
    count = dm_tensor_count(t);
    t->data = (float *)calloc(count, sizeof(float));
    return t->data ? 0 : -1;
}

void dm_tensor_free(DM_Tensor *t) {
    if (!t) return;
    free(t->data);
    memset(t, 0, sizeof(*t));
}

void dm_tensor_fill(DM_Tensor *t, float value) {
    size_t i, n = dm_tensor_count(t);
    for (i = 0; i < n; i++) t->data[i] = value;
}

static size_t idx4(const DM_Tensor *t, int n, int c, int y, int x) {
    return (((size_t)n * (size_t)t->c + (size_t)c) * (size_t)t->h + (size_t)y) * (size_t)t->w + (size_t)x;
}

float dm_tensor_get(const DM_Tensor *t, int n, int c, int y, int x) {
    return t->data[idx4(t, n, c, y, x)];
}

void dm_tensor_set(DM_Tensor *t, int n, int c, int y, int x, float v) {
    t->data[idx4(t, n, c, y, x)] = v;
}

static int out_size_same(int in, int stride) {
    return (in + stride - 1) / stride;
}

int dm_conv2d_same(const DM_Tensor *in, DM_Tensor *out, const float *w, const float *b,
                   int out_c, int kernel, int stride) {
    int oh, ow, pad, n, oc, oy, ox, ic, ky, kx;
    if (!in || !out || !w || out_c <= 0 || kernel <= 0 || stride <= 0) return -1;
    oh = out_size_same(in->h, stride);
    ow = out_size_same(in->w, stride);
    if (dm_tensor_alloc(out, in->n, out_c, oh, ow) != 0) return -1;
    pad = kernel / 2;
    for (n = 0; n < in->n; n++) for (oc = 0; oc < out_c; oc++) for (oy = 0; oy < oh; oy++) for (ox = 0; ox < ow; ox++) {
        float sum = b ? b[oc] : 0.0f;
        for (ic = 0; ic < in->c; ic++) for (ky = 0; ky < kernel; ky++) for (kx = 0; kx < kernel; kx++) {
            int iy = oy * stride + ky - pad;
            int ix = ox * stride + kx - pad;
            if (iy >= 0 && iy < in->h && ix >= 0 && ix < in->w) {
                size_t wi = (((size_t)oc * (size_t)in->c + (size_t)ic) * (size_t)kernel + (size_t)ky) * (size_t)kernel + (size_t)kx;
                sum += dm_tensor_get(in, n, ic, iy, ix) * w[wi];
            }
        }
        dm_tensor_set(out, n, oc, oy, ox, sum);
    }
    return 0;
}

int dm_depthwise_conv2d_same(const DM_Tensor *in, DM_Tensor *out, const float *w, const float *b,
                             int kernel, int stride) {
    int oh, ow, pad, n, c, oy, ox, ky, kx;
    if (!in || !out || !w || kernel <= 0 || stride <= 0) return -1;
    oh = out_size_same(in->h, stride);
    ow = out_size_same(in->w, stride);
    if (dm_tensor_alloc(out, in->n, in->c, oh, ow) != 0) return -1;
    pad = kernel / 2;
    for (n = 0; n < in->n; n++) for (c = 0; c < in->c; c++) for (oy = 0; oy < oh; oy++) for (ox = 0; ox < ow; ox++) {
        float sum = b ? b[c] : 0.0f;
        for (ky = 0; ky < kernel; ky++) for (kx = 0; kx < kernel; kx++) {
            int iy = oy * stride + ky - pad;
            int ix = ox * stride + kx - pad;
            if (iy >= 0 && iy < in->h && ix >= 0 && ix < in->w) {
                sum += dm_tensor_get(in, n, c, iy, ix) * w[((size_t)c * (size_t)kernel + (size_t)ky) * (size_t)kernel + (size_t)kx];
            }
        }
        dm_tensor_set(out, n, c, oy, ox, sum);
    }
    return 0;
}

int dm_pointwise_conv2d(const DM_Tensor *in, DM_Tensor *out, const float *w, const float *b, int out_c) {
    int n, oc, y, x, ic;
    if (!in || !out || !w || out_c <= 0) return -1;
    if (dm_tensor_alloc(out, in->n, out_c, in->h, in->w) != 0) return -1;
    for (n = 0; n < in->n; n++) for (oc = 0; oc < out_c; oc++) for (y = 0; y < in->h; y++) for (x = 0; x < in->w; x++) {
        float sum = b ? b[oc] : 0.0f;
        for (ic = 0; ic < in->c; ic++) sum += dm_tensor_get(in, n, ic, y, x) * w[(size_t)oc * (size_t)in->c + (size_t)ic];
        dm_tensor_set(out, n, oc, y, x, sum);
    }
    return 0;
}

void dm_relu6(DM_Tensor *t) {
    size_t i, n = dm_tensor_count(t);
    for (i = 0; i < n; i++) {
        if (t->data[i] < 0.0f) t->data[i] = 0.0f;
        if (t->data[i] > 6.0f) t->data[i] = 6.0f;
    }
}

int dm_global_avg_pool(const DM_Tensor *in, DM_Tensor *out) {
    int n, c, y, x;
    float scale;
    if (!in || !out) return -1;
    if (dm_tensor_alloc(out, in->n, in->c, 1, 1) != 0) return -1;
    scale = 1.0f / (float)(in->h * in->w);
    for (n = 0; n < in->n; n++) for (c = 0; c < in->c; c++) {
        float sum = 0.0f;
        for (y = 0; y < in->h; y++) for (x = 0; x < in->w; x++) sum += dm_tensor_get(in, n, c, y, x);
        dm_tensor_set(out, n, c, 0, 0, sum * scale);
    }
    return 0;
}

int dm_linear(const DM_Tensor *in, DM_Tensor *out, const float *w, const float *b, int out_c) {
    int n, oc, ic;
    if (!in || !out || !w || in->h != 1 || in->w != 1 || out_c <= 0) return -1;
    if (dm_tensor_alloc(out, in->n, out_c, 1, 1) != 0) return -1;
    for (n = 0; n < in->n; n++) for (oc = 0; oc < out_c; oc++) {
        float sum = b ? b[oc] : 0.0f;
        for (ic = 0; ic < in->c; ic++) sum += dm_tensor_get(in, n, ic, 0, 0) * w[(size_t)oc * (size_t)in->c + (size_t)ic];
        dm_tensor_set(out, n, oc, 0, 0, sum);
    }
    return 0;
}

void dm_softmax(DM_Tensor *t) {
    int n, c;
    for (n = 0; n < t->n; n++) {
        float mx = -1.0e30f, sum = 0.0f;
        for (c = 0; c < t->c; c++) {
            float v = dm_tensor_get(t, n, c, 0, 0);
            if (v > mx) mx = v;
        }
        for (c = 0; c < t->c; c++) {
            float e = expf(dm_tensor_get(t, n, c, 0, 0) - mx);
            dm_tensor_set(t, n, c, 0, 0, e);
            sum += e;
        }
        if (sum <= 0.0f) sum = 1.0f;
        for (c = 0; c < t->c; c++) dm_tensor_set(t, n, c, 0, 0, dm_tensor_get(t, n, c, 0, 0) / sum);
    }
}
