#include "core/dm_engine.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * dm_engine — TensorFlow Eager C API backend
 *
 * All compute-intensive primitives dispatch through TFE_Execute so every
 * DM model automatically inherits the optimised TF C++ backend (XLA, cuDNN,
 * oneDNN, …).  Optimizer step functions and analytical backward passes are
 * kept as pure-C because the ResourceApply* ops require Variable handles.
 *
 * DM_Tensor layout: NCHW  (n, c, h, w) — row-major, contiguous.
 * ═══════════════════════════════════════════════════════════════════════════ */

static TFE_Context *tf_ctx = NULL;

static void dm_tf_init(void) {
    if (tf_ctx != NULL) return;
    TF_Status *s = TF_NewStatus();
    TFE_ContextOptions *opts = TFE_NewContextOptions();
    tf_ctx = TFE_NewContext(opts, s);
    TFE_DeleteContextOptions(opts);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] TFE init failed: %s\n", TF_Message(s));
    TF_DeleteStatus(s);
}

/* No-op deallocator: the DM_Tensor / caller buffer owns the memory. */
static void noop_dealloc(void *data, size_t len, void *arg) {
    (void)data; (void)len; (void)arg;
}

/* Free-on-done deallocator: used when tensor.c owns the buffer (e.g. transposed
 * weight copies).  Signature matches TF_NewTensor's deallocator contract. */
static void free_dealloc(void *data, size_t len, void *arg) {
    (void)len; (void)arg;
    free(data);
}

/* ── Handle helpers ─────────────────────────────────────────────────────── */

/* Wrap a DM_Tensor's data buffer as a [n,c,h,w] TFE handle (zero-copy). */
static TFE_TensorHandle *dm_to_tf(const DM_Tensor *t) {
    if (!t || !t->data) return NULL;
    dm_tf_init();
    int64_t dims[4] = {t->n, t->c, t->h, t->w};
    size_t sz = (size_t)t->n * t->c * t->h * t->w * sizeof(float);
    TF_Tensor *tft = TF_NewTensor(TF_FLOAT, dims, 4, t->data, sz, noop_dealloc, NULL);
    TF_Status *s = TF_NewStatus();
    TFE_TensorHandle *h = TFE_NewTensorHandle(tft, s);
    TF_DeleteTensor(tft);
    TF_DeleteStatus(s);
    return h;
}

/* Copy resolved TFE handle data back into a pre-allocated DM_Tensor. */
static void tf_to_dm(TFE_TensorHandle *h, DM_Tensor *out) {
    if (!h || !out || !out->data) return;
    TF_Status *s = TF_NewStatus();
    TF_Tensor *tft = TFE_TensorHandleResolve(h, s);
    if (TF_GetCode(s) == TF_OK)
        memcpy(out->data, TF_TensorData(tft), TF_TensorByteSize(tft));
    else
        fprintf(stderr, "[dm_engine] resolve failed: %s\n", TF_Message(s));
    TF_DeleteTensor(tft);
    TF_DeleteStatus(s);
}

/* Wrap an arbitrary float buffer as an N-dimensional TFE handle (zero-copy). */
static TFE_TensorHandle *raw_to_tf(const float *data, const int64_t *dims, int ndim) {
    dm_tf_init();
    size_t count = 1;
    for (int i = 0; i < ndim; i++) count *= (size_t)dims[i];
    TF_Tensor *tft = TF_NewTensor(TF_FLOAT, dims, ndim,
                                  (void *)data, count * sizeof(float),
                                  noop_dealloc, NULL);
    TF_Status *s = TF_NewStatus();
    TFE_TensorHandle *h = TFE_NewTensorHandle(tft, s);
    TF_DeleteTensor(tft);
    TF_DeleteStatus(s);
    return h;
}

/* Copy resolved TFE handle into a flat float buffer (caller provides). */
static void tf_to_raw(TFE_TensorHandle *h, float *out) {
    if (!h || !out) return;
    TF_Status *s = TF_NewStatus();
    TF_Tensor *tft = TFE_TensorHandleResolve(h, s);
    if (TF_GetCode(s) == TF_OK)
        memcpy(out, TF_TensorData(tft), TF_TensorByteSize(tft));
    TF_DeleteTensor(tft);
    TF_DeleteStatus(s);
}

/* Wrap a scalar int32 as a TFE handle (used for reduction axes). */
static TFE_TensorHandle *scalar_int32_to_tf(int32_t v) {
    dm_tf_init();
    int64_t dims[1] = {1};
    int32_t *buf = (int32_t *)malloc(sizeof(int32_t));
    *buf = v;
    TF_Tensor *tft = TF_NewTensor(TF_INT32, dims, 1, buf, sizeof(int32_t), free_dealloc, NULL);
    TF_Status *s = TF_NewStatus();
    TFE_TensorHandle *h = TFE_NewTensorHandle(tft, s);
    TF_DeleteTensor(tft);
    TF_DeleteStatus(s);
    return h;
}

/* Wrap an int32 array as a TFE handle (used for reduction axes lists). */
static TFE_TensorHandle *int32_array_to_tf(const int32_t *arr, int n) {
    dm_tf_init();
    int64_t dims[1] = {n};
    size_t sz = (size_t)n * sizeof(int32_t);
    int32_t *buf = (int32_t *)malloc(sz);
    memcpy(buf, arr, sz);
    TF_Tensor *tft = TF_NewTensor(TF_INT32, dims, 1, buf, sz, free_dealloc, NULL);
    TF_Status *s = TF_NewStatus();
    TFE_TensorHandle *h = TFE_NewTensorHandle(tft, s);
    TF_DeleteTensor(tft);
    TF_DeleteStatus(s);
    return h;
}

/* ── Op execution helpers ───────────────────────────────────────────────── */

/* Execute a 1-input op with no extra attributes. */
static TFE_TensorHandle *op1(const char *name, TFE_TensorHandle *in) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, name, s);
    TFE_OpAddInput(op, in, s);
    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] %s failed: %s\n", name, TF_Message(s));
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);
    return ret[0];
}

/* Execute a 2-input op with no extra attributes. */
static TFE_TensorHandle *op2(const char *name,
                              TFE_TensorHandle *a, TFE_TensorHandle *b) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, name, s);
    TFE_OpAddInput(op, a, s);
    TFE_OpAddInput(op, b, s);
    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] %s failed: %s\n", name, TF_Message(s));
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);
    return ret[0];
}

/* MatMul with transpose flags. */
static TFE_TensorHandle *execute_tf_matmul(TFE_TensorHandle *a, TFE_TensorHandle *b,
                                            bool ta, bool tb) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "MatMul", s);
    TFE_OpAddInput(op, a, s);
    TFE_OpAddInput(op, b, s);
    TFE_OpSetAttrBool(op, "transpose_a", (unsigned char)(ta ? 1 : 0));
    TFE_OpSetAttrBool(op, "transpose_b", (unsigned char)(tb ? 1 : 0));
    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] MatMul failed: %s\n", TF_Message(s));
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);
    return ret[0];
}

/* ── Conv2D helper: builds the OIHW kernel handle ──────────────────────────
 * DM weight layout for dm_conv2d_same:  w[oc][ic][ky][kx]  (OIHW)
 * TF Conv2D with data_format="NCHW" expects filter shape [oc, ic, kH, kW]
 * which is exactly OIHW — no transpose needed.
 */
static TFE_TensorHandle *conv_weight_to_tf(const float *w,
                                            int oc, int ic, int k) {
    int64_t dims[4] = {oc, ic, k, k};
    return raw_to_tf(w, dims, 4);
}

/* Conv2D with SAME padding, NCHW data format.
 * strides4 = {1, 1, stride_h, stride_w}
 * dilations4 = {1, 1, 1, 1}
 */
static TFE_TensorHandle *execute_conv2d(TFE_TensorHandle *input,
                                         TFE_TensorHandle *filter,
                                         int stride) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "Conv2D", s);
    TFE_OpAddInput(op, input, s);
    TFE_OpAddInput(op, filter, s);
    int64_t strides[4]   = {1, 1, stride, stride};
    int64_t dilations[4] = {1, 1, 1, 1};
    TFE_OpSetAttrIntList(op, "strides",   strides,   4);
    TFE_OpSetAttrIntList(op, "dilations", dilations, 4);
    TFE_OpSetAttrString(op, "padding",     "SAME",  4);
    TFE_OpSetAttrString(op, "data_format", "NCHW",  4);
    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] Conv2D failed: %s\n", TF_Message(s));
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);
    return ret[0];
}

/* Add a bias vector (shape [oc]) to a [n,oc,h,w] tensor via BiasAdd NCHW. */
static TFE_TensorHandle *execute_bias_add(TFE_TensorHandle *input,
                                           TFE_TensorHandle *bias) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "BiasAdd", s);
    TFE_OpAddInput(op, input, s);
    TFE_OpAddInput(op, bias,  s);
    TFE_OpSetAttrString(op, "data_format", "NCHW", 4);
    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] BiasAdd failed: %s\n", TF_Message(s));
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);
    return ret[0];
}

/* DepthwiseConv2dNative NCHW.
 * DM weight layout: w[c][ky][kx] → TF depthwise expects [kH,kW,in_c,channel_mult].
 * We use channel_mult=1 (standard depthwise) so we must transpose from
 * [c, k, k] → [k, k, c, 1].
 */
static TFE_TensorHandle *depthwise_weight_to_tf(const float *w, int c, int k) {
    /* Transpose: src[ic][ky][kx] → dst[ky][kx][ic][1] */
    int total = c * k * k;
    float *buf = (float *)malloc((size_t)total * sizeof(float));
    for (int ic = 0; ic < c; ic++)
        for (int ky = 0; ky < k; ky++)
            for (int kx = 0; kx < k; kx++) {
                int src = (ic * k + ky) * k + kx;
                int dst = (ky * k + kx) * c + ic; /* [ky][kx][ic] (channel_mult=1) */
                buf[dst] = w[src];
            }
    int64_t dims[4] = {k, k, c, 1};
    size_t sz = (size_t)total * sizeof(float);
    TF_Tensor *tft = TF_NewTensor(TF_FLOAT, dims, 4, buf, sz, free_dealloc, NULL);
    TF_Status *s = TF_NewStatus();
    TFE_TensorHandle *h = TFE_NewTensorHandle(tft, s);
    TF_DeleteTensor(tft);
    TF_DeleteStatus(s);
    return h;
}

static TFE_TensorHandle *execute_depthwise_conv2d(TFE_TensorHandle *input,
                                                    TFE_TensorHandle *filter,
                                                    int stride) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "DepthwiseConv2dNative", s);
    TFE_OpAddInput(op, input,  s);
    TFE_OpAddInput(op, filter, s);
    int64_t strides[4]   = {1, 1, stride, stride};
    int64_t dilations[4] = {1, 1, 1, 1};
    TFE_OpSetAttrIntList(op, "strides",   strides,   4);
    TFE_OpSetAttrIntList(op, "dilations", dilations, 4);
    TFE_OpSetAttrString(op, "padding",     "SAME", 4);
    TFE_OpSetAttrString(op, "data_format", "NCHW", 4);
    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] DepthwiseConv2dNative failed: %s\n", TF_Message(s));
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);
    return ret[0];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Tensor lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

size_t dm_tensor_count(const DM_Tensor *t) {
    if (!t || t->n <= 0 || t->c <= 0 || t->h <= 0 || t->w <= 0) return 0;
    return (size_t)t->n * (size_t)t->c * (size_t)t->h * (size_t)t->w;
}

int dm_tensor_alloc(DM_Tensor *t, int n, int c, int h, int w) {
    if (!t || n <= 0 || c <= 0 || h <= 0 || w <= 0) return -1;
    memset(t, 0, sizeof(*t));
    t->n = n; t->c = c; t->h = h; t->w = w;
    size_t count = dm_tensor_count(t);
    t->data = (float *)calloc(count, sizeof(float));
    return t->data ? 0 : -1;
}

void dm_tensor_free(DM_Tensor *t) {
    if (!t) return;
    free(t->data);
    memset(t, 0, sizeof(*t));
}

void dm_tensor_fill(DM_Tensor *t, float value) {
    size_t n = dm_tensor_count(t);
    for (size_t i = 0; i < n; i++) t->data[i] = value;
}

static size_t idx4(const DM_Tensor *t, int n, int c, int y, int x) {
    return (((size_t)n * (size_t)t->c + (size_t)c) * (size_t)t->h + (size_t)y)
           * (size_t)t->w + (size_t)x;
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

/* ═══════════════════════════════════════════════════════════════════════════
 * Convolutions  (dm_engine / TFE backend)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* dm_conv2d_same — Standard 2-D convolution, SAME padding.
 * weight layout: w[out_c][in_c][ky][kx]  (OIHW)
 */
int dm_conv2d_same(const DM_Tensor *in, DM_Tensor *out,
                   const float *w, const float *b,
                   int out_c, int kernel, int stride) {
    if (!in || !out || !w || out_c <= 0 || kernel <= 0 || stride <= 0) return -1;
    int oh = out_size_same(in->h, stride);
    int ow = out_size_same(in->w, stride);
    if (dm_tensor_alloc(out, in->n, out_c, oh, ow) != 0) return -1;

    TFE_TensorHandle *h_in  = dm_to_tf(in);
    TFE_TensorHandle *h_w   = conv_weight_to_tf(w, out_c, in->c, kernel);
    if (!h_in || !h_w) goto conv2d_err;

    TFE_TensorHandle *h_out = execute_conv2d(h_in, h_w, stride);
    if (!h_out) goto conv2d_err;

    if (b) {
        /* BiasAdd: bias shape [out_c] */
        int64_t bdims[1] = {out_c};
        TFE_TensorHandle *h_b   = raw_to_tf(b, bdims, 1);
        TFE_TensorHandle *h_out2 = execute_bias_add(h_out, h_b);
        TFE_DeleteTensorHandle(h_b);
        TFE_DeleteTensorHandle(h_out);
        h_out = h_out2;
    }
    if (h_out) {
        tf_to_dm(h_out, out);
        TFE_DeleteTensorHandle(h_out);
    }
    TFE_DeleteTensorHandle(h_in);
    TFE_DeleteTensorHandle(h_w);
    return 0;

conv2d_err:
    if (h_in) TFE_DeleteTensorHandle(h_in);
    if (h_w)  TFE_DeleteTensorHandle(h_w);
    return -1;
}

/* dm_depthwise_conv2d_same — Depthwise separable conv, SAME padding. */
int dm_depthwise_conv2d_same(const DM_Tensor *in, DM_Tensor *out,
                              const float *w, const float *b,
                              int kernel, int stride) {
    if (!in || !out || !w || kernel <= 0 || stride <= 0) return -1;
    int oh = out_size_same(in->h, stride);
    int ow = out_size_same(in->w, stride);
    if (dm_tensor_alloc(out, in->n, in->c, oh, ow) != 0) return -1;

    TFE_TensorHandle *h_in  = dm_to_tf(in);
    TFE_TensorHandle *h_w   = depthwise_weight_to_tf(w, in->c, kernel);
    if (!h_in || !h_w) goto dw_err;

    TFE_TensorHandle *h_out = execute_depthwise_conv2d(h_in, h_w, stride);
    if (!h_out) goto dw_err;

    if (b) {
        int64_t bdims[1] = {in->c};
        TFE_TensorHandle *h_b    = raw_to_tf(b, bdims, 1);
        TFE_TensorHandle *h_out2 = execute_bias_add(h_out, h_b);
        TFE_DeleteTensorHandle(h_b);
        TFE_DeleteTensorHandle(h_out);
        h_out = h_out2;
    }
    if (h_out) {
        tf_to_dm(h_out, out);
        TFE_DeleteTensorHandle(h_out);
    }
    TFE_DeleteTensorHandle(h_in);
    TFE_DeleteTensorHandle(h_w);
    return 0;

dw_err:
    if (h_in) TFE_DeleteTensorHandle(h_in);
    if (h_w)  TFE_DeleteTensorHandle(h_w);
    return -1;
}

/* dm_pointwise_conv2d — 1×1 convolution (channel mixing). */
int dm_pointwise_conv2d(const DM_Tensor *in, DM_Tensor *out,
                         const float *w, const float *b, int out_c) {
    if (!in || !out || !w || out_c <= 0) return -1;
    if (dm_tensor_alloc(out, in->n, out_c, in->h, in->w) != 0) return -1;

    TFE_TensorHandle *h_in = dm_to_tf(in);
    /* 1×1 conv: filter shape [out_c, in_c, 1, 1] */
    int64_t wdims[4] = {out_c, in->c, 1, 1};
    TFE_TensorHandle *h_w  = raw_to_tf(w, wdims, 4);
    if (!h_in || !h_w) goto pw_err;

    TFE_TensorHandle *h_out = execute_conv2d(h_in, h_w, 1);
    if (!h_out) goto pw_err;

    if (b) {
        int64_t bdims[1] = {out_c};
        TFE_TensorHandle *h_b    = raw_to_tf(b, bdims, 1);
        TFE_TensorHandle *h_out2 = execute_bias_add(h_out, h_b);
        TFE_DeleteTensorHandle(h_b);
        TFE_DeleteTensorHandle(h_out);
        h_out = h_out2;
    }
    if (h_out) {
        tf_to_dm(h_out, out);
        TFE_DeleteTensorHandle(h_out);
    }
    TFE_DeleteTensorHandle(h_in);
    TFE_DeleteTensorHandle(h_w);
    return 0;

pw_err:
    if (h_in) TFE_DeleteTensorHandle(h_in);
    if (h_w)  TFE_DeleteTensorHandle(h_w);
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Activations
 * ═══════════════════════════════════════════════════════════════════════════ */

void dm_relu6(DM_Tensor *t) {
    if (!t) return;
    TFE_TensorHandle *in_h  = dm_to_tf(t);
    if (!in_h) return;
    TFE_TensorHandle *out_h = op1("Relu6", in_h);
    if (out_h) { tf_to_dm(out_h, t); TFE_DeleteTensorHandle(out_h); }
    TFE_DeleteTensorHandle(in_h);
}

void dm_relu(DM_Tensor *t) {
    if (!t) return;
    TFE_TensorHandle *in_h  = dm_to_tf(t);
    if (!in_h) return;
    TFE_TensorHandle *out_h = op1("Relu", in_h);
    if (out_h) { tf_to_dm(out_h, t); TFE_DeleteTensorHandle(out_h); }
    TFE_DeleteTensorHandle(in_h);
}

/* GELU: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x³))) */
void dm_gelu_inplace(float *x, int n) {
    if (!x || n <= 0) return;
    int64_t dims[1] = {n};
    TFE_TensorHandle *h = raw_to_tf(x, dims, 1);
    if (!h) return;
    TFE_TensorHandle *res = op1("Gelu", h);
    if (res) { tf_to_raw(res, x); TFE_DeleteTensorHandle(res); }
    TFE_DeleteTensorHandle(h);
}

void dm_tanh_inplace(DM_Tensor *t) {
    if (!t) return;
    TFE_TensorHandle *in_h  = dm_to_tf(t);
    if (!in_h) return;
    TFE_TensorHandle *out_h = op1("Tanh", in_h);
    if (out_h) { tf_to_dm(out_h, t); TFE_DeleteTensorHandle(out_h); }
    TFE_DeleteTensorHandle(in_h);
}

void dm_sigmoid_inplace(DM_Tensor *t) {
    if (!t) return;
    TFE_TensorHandle *in_h  = dm_to_tf(t);
    if (!in_h) return;
    TFE_TensorHandle *out_h = op1("Sigmoid", in_h);
    if (out_h) { tf_to_dm(out_h, t); TFE_DeleteTensorHandle(out_h); }
    TFE_DeleteTensorHandle(in_h);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Elementwise add
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_tensor_add(DM_Tensor *out, const DM_Tensor *in) {
    if (!out || !in ||
        out->n != in->n || out->c != in->c ||
        out->h != in->h || out->w != in->w) return -1;
    TFE_TensorHandle *h_out = dm_to_tf(out);
    TFE_TensorHandle *h_in  = dm_to_tf(in);
    if (!h_out || !h_in) {
        if (h_out) TFE_DeleteTensorHandle(h_out);
        if (h_in)  TFE_DeleteTensorHandle(h_in);
        return -1;
    }
    TFE_TensorHandle *res = op2("AddV2", h_out, h_in);
    if (res) { tf_to_dm(res, out); TFE_DeleteTensorHandle(res); }
    TFE_DeleteTensorHandle(h_out);
    TFE_DeleteTensorHandle(h_in);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Pooling
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_max_pool2d_same(const DM_Tensor *in, DM_Tensor *out, int kernel, int stride) {
    if (!in || !out || kernel <= 0 || stride <= 0) return -1;
    int oh = out_size_same(in->h, stride);
    int ow = out_size_same(in->w, stride);
    if (dm_tensor_alloc(out, in->n, in->c, oh, ow) != 0) return -1;

    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "MaxPool", s);

    TFE_TensorHandle *h_in = dm_to_tf(in);
    TFE_OpAddInput(op, h_in, s);

    int64_t ksize[4]   = {1, 1, kernel, kernel};
    int64_t strides[4] = {1, 1, stride, stride};
    TFE_OpSetAttrIntList(op, "ksize",       ksize,   4);
    TFE_OpSetAttrIntList(op, "strides",     strides, 4);
    TFE_OpSetAttrString(op,  "padding",     "SAME",  4);
    TFE_OpSetAttrString(op,  "data_format", "NCHW",  4);

    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] MaxPool failed: %s\n", TF_Message(s));
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);

    if (ret[0]) { tf_to_dm(ret[0], out); TFE_DeleteTensorHandle(ret[0]); }
    TFE_DeleteTensorHandle(h_in);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Batch Normalization — FusedBatchNorm (inference mode, is_training=false)
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_batch_norm(DM_Tensor *t,
                  const float *gamma, const float *beta,
                  const float *mean,  const float *var,
                  float eps) {
    if (!t || !gamma || !beta || !mean || !var) return -1;
    int C = t->c;

    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "FusedBatchNorm", s);

    TFE_TensorHandle *h_x     = dm_to_tf(t);
    int64_t cdims[1] = {C};
    TFE_TensorHandle *h_scale = raw_to_tf(gamma, cdims, 1);
    TFE_TensorHandle *h_offset= raw_to_tf(beta,  cdims, 1);
    TFE_TensorHandle *h_mean  = raw_to_tf(mean,  cdims, 1);
    TFE_TensorHandle *h_var   = raw_to_tf(var,   cdims, 1);

    TFE_OpAddInput(op, h_x,      s);
    TFE_OpAddInput(op, h_scale,  s);
    TFE_OpAddInput(op, h_offset, s);
    TFE_OpAddInput(op, h_mean,   s);
    TFE_OpAddInput(op, h_var,    s);

    TFE_OpSetAttrFloat(op,  "epsilon",     eps);
    TFE_OpSetAttrBool(op,   "is_training", 0);
    TFE_OpSetAttrString(op, "data_format", "NCHW", 4);

    /* FusedBatchNorm returns 5 outputs: y, batch_mean, batch_var,
     * reserved_space_1, reserved_space_2.  We only need y. */
    TFE_TensorHandle *ret[5] = {NULL, NULL, NULL, NULL, NULL};
    int nret = 5;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] FusedBatchNorm failed: %s\n", TF_Message(s));
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);

    if (ret[0]) { tf_to_dm(ret[0], t); }
    for (int i = 0; i < 5; i++) if (ret[i]) TFE_DeleteTensorHandle(ret[i]);
    TFE_DeleteTensorHandle(h_x);
    TFE_DeleteTensorHandle(h_scale);
    TFE_DeleteTensorHandle(h_offset);
    TFE_DeleteTensorHandle(h_mean);
    TFE_DeleteTensorHandle(h_var);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Global Average Pooling — Mean over spatial dims (h=2, w=3 in NCHW)
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_global_avg_pool(const DM_Tensor *in, DM_Tensor *out) {
    if (!in || !out) return -1;
    if (dm_tensor_alloc(out, in->n, in->c, 1, 1) != 0) return -1;

    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "Mean", s);

    TFE_TensorHandle *h_in = dm_to_tf(in);
    /* reduction indices: dims 2 and 3 (h and w in NCHW) */
    int32_t axes[2] = {2, 3};
    TFE_TensorHandle *h_axes = int32_array_to_tf(axes, 2);

    TFE_OpAddInput(op, h_in,   s);
    TFE_OpAddInput(op, h_axes, s);
    TFE_OpSetAttrBool(op, "keep_dims", 1);

    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] Mean (GlobalAvgPool) failed: %s\n", TF_Message(s));
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);

    if (ret[0]) { tf_to_dm(ret[0], out); TFE_DeleteTensorHandle(ret[0]); }
    TFE_DeleteTensorHandle(h_in);
    TFE_DeleteTensorHandle(h_axes);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Linear (fully-connected) layer
 * in:  DM_Tensor [n, in_c, 1, 1]
 * out: DM_Tensor [n, out_c, 1, 1]
 * w:   float[out_c × in_c]   (row = output neuron)
 * b:   float[out_c]  or NULL
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_linear(const DM_Tensor *in, DM_Tensor *out,
              const float *w, const float *b, int out_c) {
    if (!in || !out || !w || in->h != 1 || in->w != 1 || out_c <= 0) return -1;
    if (dm_tensor_alloc(out, in->n, out_c, 1, 1) != 0) return -1;

    /* Reshape to 2-D: [n, in_c] and [out_c, in_c] */
    int64_t adims[2] = {in->n, in->c};
    int64_t bdims[2] = {out_c, in->c};
    TFE_TensorHandle *h_a = raw_to_tf(in->data, adims, 2);
    TFE_TensorHandle *h_b = raw_to_tf(w,        bdims, 2);
    if (!h_a || !h_b) {
        if (h_a) TFE_DeleteTensorHandle(h_a);
        if (h_b) TFE_DeleteTensorHandle(h_b);
        return -1;
    }

    /* out = in @ W^T  → [n, out_c] */
    TFE_TensorHandle *h_out = execute_tf_matmul(h_a, h_b, false, true);
    TFE_DeleteTensorHandle(h_a);
    TFE_DeleteTensorHandle(h_b);
    if (!h_out) return -1;

    if (b) {
        int64_t biasdims[1] = {out_c};
        TFE_TensorHandle *h_bias = raw_to_tf(b, biasdims, 1);
        TFE_TensorHandle *h_out2 = op2("AddV2", h_out, h_bias);
        TFE_DeleteTensorHandle(h_bias);
        TFE_DeleteTensorHandle(h_out);
        h_out = h_out2;
    }

    if (h_out) {
        /* Result is [n, out_c]; copy flat into out->data [n, out_c, 1, 1] */
        tf_to_raw(h_out, out->data);
        TFE_DeleteTensorHandle(h_out);
    }
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Softmax
 * ═══════════════════════════════════════════════════════════════════════════ */

/* dm_softmax — operates on t with shape [n, c, 1, 1].
 * Softmax over the class (channel) axis (axis=1 in NCHW).
 */
void dm_softmax(DM_Tensor *t) {
    if (!t) return;

    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "Softmax", s);

    /* Reshape to [n, c] for the Softmax op (axis=-1 = axis 1) */
    int64_t dims2[2] = {t->n, t->c};
    TFE_TensorHandle *h_in = raw_to_tf(t->data, dims2, 2);
    TFE_OpAddInput(op, h_in, s);

    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] Softmax failed: %s\n", TF_Message(s));
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);

    if (ret[0]) { tf_to_raw(ret[0], t->data); TFE_DeleteTensorHandle(ret[0]); }
    TFE_DeleteTensorHandle(h_in);
}

/* dm_softmax_rows — row-wise softmax over a [rows × cols] flat float buffer. */
void dm_softmax_rows(float *x, int rows, int cols) {
    if (!x || rows <= 0 || cols <= 0) return;

    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "Softmax", s);

    int64_t dims[2] = {rows, cols};
    TFE_TensorHandle *h_in = raw_to_tf(x, dims, 2);
    TFE_OpAddInput(op, h_in, s);

    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] Softmax (rows) failed: %s\n", TF_Message(s));
    TFE_DeleteOp(op);
    TF_DeleteStatus(s);

    if (ret[0]) { tf_to_raw(ret[0], x); TFE_DeleteTensorHandle(ret[0]); }
    TFE_DeleteTensorHandle(h_in);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Layer Normalisation over sequence rows
 *
 * x : float[seq_len × d_model]  row-major
 * Normalises each row independently: y = (x − μ) / √(σ²+ε) * γ + β
 *
 * Pipeline (all via TFE):
 *   μ   = Mean(x, axis=1, keepdims)              → [seq, 1]
 *   x0  = x − μ                                  → [seq, d_model]
 *   σ²  = Mean(x0², axis=1, keepdims)            → [seq, 1]
 *   inv = Rsqrt(σ² + ε)                          → [seq, 1]
 *   y   = x0 * inv * γ + β
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_layer_norm_seq(float *x, int seq_len, int d_model,
                      const float *gamma, const float *beta, float eps) {
    if (!x || !gamma || !beta || seq_len <= 0 || d_model <= 0) return -1;

    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    int64_t xdims[2]   = {seq_len, d_model};
    int64_t gdims[1]   = {d_model};

    TFE_TensorHandle *h_x  = raw_to_tf(x,     xdims, 2);
    TFE_TensorHandle *h_g  = raw_to_tf(gamma, gdims, 1);
    TFE_TensorHandle *h_b  = raw_to_tf(beta,  gdims, 1);

    /* reduction axis = 1 (d_model dim) */
    int32_t ax = 1;
    TFE_TensorHandle *h_ax = int32_array_to_tf(&ax, 1);

    /* μ = Mean(x, axis=1, keepdims=true)  → [seq, 1] */
    TFE_Op *op_mean = TFE_NewOp(tf_ctx, "Mean", s);
    TFE_OpAddInput(op_mean, h_x,  s);
    TFE_OpAddInput(op_mean, h_ax, s);
    TFE_OpSetAttrBool(op_mean, "keep_dims", 1);
    TFE_TensorHandle *h_mu[1] = {NULL}; int nr = 1;
    TFE_Execute(op_mean, h_mu, &nr, s);
    TFE_DeleteOp(op_mean);

    /* x0 = x − μ  (broadcasts [seq,1] → [seq,d_model]) */
    TFE_TensorHandle *h_x0 = op2("Sub", h_x, h_mu[0]);

    /* x0² = x0 * x0 */
    TFE_TensorHandle *h_sq = op2("Mul", h_x0, h_x0);

    /* σ² = Mean(x0², axis=1, keepdims=true) */
    TFE_Op *op_var = TFE_NewOp(tf_ctx, "Mean", s);
    TFE_OpAddInput(op_var, h_sq,  s);
    TFE_OpAddInput(op_var, h_ax,  s);
    TFE_OpSetAttrBool(op_var, "keep_dims", 1);
    TFE_TensorHandle *h_var[1] = {NULL}; nr = 1;
    TFE_Execute(op_var, h_var, &nr, s);
    TFE_DeleteOp(op_var);

    /* add epsilon scalar: eps_handle = scalar [seq,1] filled with eps */
    /* Use AddV2 with a scalar tensor for eps */
    float eps_val = eps;
    int64_t eps_dims[1] = {1};
    TFE_TensorHandle *h_eps  = raw_to_tf(&eps_val, eps_dims, 1);
    TFE_TensorHandle *h_veps = op2("AddV2", h_var[0], h_eps);

    /* inv = Rsqrt(σ² + ε) */
    TFE_TensorHandle *h_inv  = op1("Rsqrt", h_veps);

    /* normed = x0 * inv  → [seq, d_model] */
    TFE_TensorHandle *h_norm = op2("Mul", h_x0, h_inv);

    /* scaled = normed * γ  (γ shape [d_model] broadcasts) */
    TFE_TensorHandle *h_scaled = op2("Mul", h_norm, h_g);

    /* out = scaled + β */
    TFE_TensorHandle *h_out = op2("AddV2", h_scaled, h_b);

    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] LayerNorm failed: %s\n", TF_Message(s));

    if (h_out) { tf_to_raw(h_out, x); TFE_DeleteTensorHandle(h_out); }

    TFE_DeleteTensorHandle(h_x);
    TFE_DeleteTensorHandle(h_g);
    TFE_DeleteTensorHandle(h_b);
    TFE_DeleteTensorHandle(h_ax);
    if (h_mu[0])  TFE_DeleteTensorHandle(h_mu[0]);
    TFE_DeleteTensorHandle(h_x0);
    TFE_DeleteTensorHandle(h_sq);
    if (h_var[0]) TFE_DeleteTensorHandle(h_var[0]);
    TFE_DeleteTensorHandle(h_eps);
    TFE_DeleteTensorHandle(h_veps);
    TFE_DeleteTensorHandle(h_inv);
    TFE_DeleteTensorHandle(h_norm);
    TFE_DeleteTensorHandle(h_scaled);
    TF_DeleteStatus(s);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Matrix multiplications
 * ═══════════════════════════════════════════════════════════════════════════ */

/* C = A @ B^T   A[M×K], B[N×K] → C[M×N] */
void dm_matmul_nt(const float *A, const float *B, float *C, int M, int N, int K) {
    int64_t da[2] = {M, K};
    int64_t db[2] = {N, K};
    TFE_TensorHandle *hA = raw_to_tf(A, da, 2);
    TFE_TensorHandle *hB = raw_to_tf(B, db, 2);
    if (!hA || !hB) {
        if (hA) TFE_DeleteTensorHandle(hA);
        if (hB) TFE_DeleteTensorHandle(hB);
        return;
    }
    TFE_TensorHandle *res = execute_tf_matmul(hA, hB, false, true);
    if (res) { tf_to_raw(res, C); TFE_DeleteTensorHandle(res); }
    TFE_DeleteTensorHandle(hA);
    TFE_DeleteTensorHandle(hB);
}

/* C = A @ B    A[M×K], B[K×N] → C[M×N] */
void dm_matmul_nn(const float *A, const float *B, float *C, int M, int K, int N) {
    int64_t da[2] = {M, K};
    int64_t db[2] = {K, N};
    TFE_TensorHandle *hA = raw_to_tf(A, da, 2);
    TFE_TensorHandle *hB = raw_to_tf(B, db, 2);
    if (!hA || !hB) {
        if (hA) TFE_DeleteTensorHandle(hA);
        if (hB) TFE_DeleteTensorHandle(hB);
        return;
    }
    TFE_TensorHandle *res = execute_tf_matmul(hA, hB, false, false);
    if (res) { tf_to_raw(res, C); TFE_DeleteTensorHandle(res); }
    TFE_DeleteTensorHandle(hA);
    TFE_DeleteTensorHandle(hB);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Training Primitives
 *
 * Optimiser step functions operate element-wise on flat float arrays and are
 * kept as pure-C. Migrating them to TFE ResourceApply* ops would require
 * TF_Variable handles and a graph-mode setup that adds no practical benefit
 * for the small parameter tensors typical in training steps.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── Backward passes ────────────────────────────────────────────────────── */

int dm_linear_backward(const DM_Tensor *in, const DM_Tensor *grad_out,
                        DM_Tensor *grad_in,
                        float *grad_w, float *grad_b,
                        const float *w, int out_c) {
    if (!in || !grad_out) return -1;
    if (grad_in && dm_tensor_alloc(grad_in, in->n, in->c, 1, 1) != 0) return -1;
    if (grad_in) dm_tensor_fill(grad_in, 0.0f);

    for (int n = 0; n < in->n; n++) {
        for (int oc = 0; oc < out_c; oc++) {
            float go = dm_tensor_get(grad_out, n, oc, 0, 0);
            if (grad_b) grad_b[oc] += go;
            for (int ic = 0; ic < in->c; ic++) {
                float iv = dm_tensor_get(in, n, ic, 0, 0);
                if (grad_w) grad_w[(size_t)oc * (size_t)in->c + ic] += go * iv;
                if (grad_in && w) {
                    float cur = dm_tensor_get(grad_in, n, ic, 0, 0);
                    dm_tensor_set(grad_in, n, ic, 0, 0,
                                  cur + go * w[(size_t)oc * (size_t)in->c + ic]);
                }
            }
        }
    }
    return 0;
}

void dm_tanh_backward(const DM_Tensor *out, const DM_Tensor *grad_out,
                       DM_Tensor *grad_in) {
    if (dm_tensor_alloc(grad_in, out->n, out->c, out->h, out->w) != 0) return;
    size_t count = dm_tensor_count(out);
    for (size_t i = 0; i < count; i++) {
        float o = out->data[i];
        grad_in->data[i] = grad_out->data[i] * (1.0f - o * o);
    }
}

void dm_relu_backward(const DM_Tensor *in, const DM_Tensor *grad_out,
                       DM_Tensor *grad_in) {
    size_t count = dm_tensor_count(in);
    for (size_t i = 0; i < count; i++)
        grad_in->data[i] = in->data[i] > 0.0f ? grad_out->data[i] : 0.0f;
}

/* ── Maxout ─────────────────────────────────────────────────────────────── */

int dm_maxout(const DM_Tensor *in, DM_Tensor *out, int k, int *argmax) {
    if (k <= 0 || in->c % k != 0) return -1;
    int out_c = in->c / k;
    if (out->n != in->n || out->c != out_c ||
        out->h != in->h || out->w != in->w) return -1;

    int spatial = in->h * in->w;
    for (int n = 0; n < in->n; n++)
        for (int c = 0; c < out_c; c++)
            for (int s = 0; s < spatial; s++) {
                float mx = -1e30f; int mi = -1;
                for (int j = 0; j < k; j++) {
                    int ic  = c * k + j;
                    int idx = (n * in->c + ic) * spatial + s;
                    if (in->data[idx] > mx || mi == -1) { mx = in->data[idx]; mi = idx; }
                }
                int oi = (n * out_c + c) * spatial + s;
                out->data[oi] = mx;
                if (argmax) argmax[oi] = mi;
            }
    return 0;
}

int dm_maxout_backward(const DM_Tensor *grad_out, DM_Tensor *grad_in,
                        int k, const int *argmax) {
    if (grad_in->c % k != 0 || grad_in->c / k != grad_out->c) return -1;
    if (!argmax) return -1;
    size_t in_count  = dm_tensor_count(grad_in);
    size_t out_count = dm_tensor_count(grad_out);
    for (size_t i = 0; i < in_count; i++) grad_in->data[i] = 0.0f;
    for (size_t i = 0; i < out_count; i++) {
        int mi = argmax[i];
        if (mi >= 0 && (size_t)mi < in_count)
            grad_in->data[mi] += grad_out->data[i];
    }
    return 0;
}

/* ── Dropout ────────────────────────────────────────────────────────────── */

void dm_dropout(const DM_Tensor *in, DM_Tensor *out, float drop_prob, int *mask) {
    size_t count = dm_tensor_count(in);
    float scale = 1.0f / (1.0f - drop_prob);
    for (size_t i = 0; i < count; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        if (r < drop_prob) {
            out->data[i] = 0.0f;
            if (mask) mask[i] = 0;
        } else {
            out->data[i] = in->data[i] * scale;
            if (mask) mask[i] = 1;
        }
    }
}

void dm_dropout_backward(const DM_Tensor *grad_out, DM_Tensor *grad_in,
                          float drop_prob, const int *mask) {
    size_t count = dm_tensor_count(grad_out);
    float scale = 1.0f / (1.0f - drop_prob);
    for (size_t i = 0; i < count; i++)
        grad_in->data[i] = (mask && mask[i]) ? grad_out->data[i] * scale : 0.0f;
}

/* ── Optimisers ─────────────────────────────────────────────────────────── */

void dm_adagrad_step(float *param, float *grad, float *g_sum,
                     int n, float lr, float eps, float weight_decay) {
    for (int i = 0; i < n; i++) {
        float g = grad[i];
        if (weight_decay > 0.0f) g += weight_decay * param[i];
        g_sum[i] += g * g;
        param[i] -= (lr / (sqrtf(g_sum[i]) + eps)) * g;
        grad[i] = 0.0f;
    }
}

void dm_adam_step(float *param, float *grad, float *m, float *v,
                  int n, float lr,
                  float beta1, float beta2, float eps,
                  float weight_decay, int t) {
    float corr1 = 1.0f - powf(beta1, (float)t);
    float corr2 = 1.0f - powf(beta2, (float)t);
    for (int i = 0; i < n; i++) {
        float g = grad[i];
        if (weight_decay > 0.0f) g += weight_decay * param[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        param[i] -= lr * (m[i] / corr1) / (sqrtf(v[i] / corr2) + eps);
        grad[i] = 0.0f;
    }
}

void dm_sgd_momentum_step(float *param, float *grad, float *velocity,
                           int n, float lr, float momentum,
                           float weight_decay, int nesterov) {
    for (int i = 0; i < n; i++) {
        float g = grad[i];
        if (weight_decay > 0.0f) g += weight_decay * param[i];
        velocity[i] = momentum * velocity[i] - lr * g;
        if (nesterov)
            param[i] += momentum * velocity[i] - lr * g;
        else
            param[i] += velocity[i];
        grad[i] = 0.0f;
    }
}
