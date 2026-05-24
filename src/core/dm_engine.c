#include "core/dm_engine.h"
#include "lowering/dm_lowering.h"
#include "lowering/dm_lower_tf.h"

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


static DM_LayoutPolicy g_dm_layout_policy = DM_LAYOUT_POLICY_STRICT;

void dm_set_layout_policy(DM_LayoutPolicy policy) {
    g_dm_layout_policy = policy;
}

DM_LayoutPolicy dm_get_layout_policy(void) {
    return g_dm_layout_policy;
}

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

static TFE_TensorHandle *dm_to_tf(const DM_Block *t) {
    if (!t || !t->data) return NULL;
    dm_tf_init();
    DM_Block tf_b;
    // Note: CPU -> TF is a zero-copy wrap using TF_NewTensor with noop_dealloc.
    if (dm_lower_block(t, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return NULL;
    // dm_engine.c manually manages and frees TFE handles via TFE_DeleteTensorHandle.
    // We intentionally clear owns_handle so it doesn't try to free it if dm_block_free were called.
    tf_b.owns_handle = 0;
    return (TFE_TensorHandle *)tf_b.handle;
}

static void tf_to_dm(TFE_TensorHandle *h, DM_Block *out) {
    if (!h || !out || !out->data) return;
    DM_Block tf_src;
    memset(&tf_src, 0, sizeof(DM_Block));
    tf_src.backend = DM_BACKEND_TENSORFLOW;
    tf_src.kind = DM_KIND_EXTERNAL;
    tf_src.layout = DM_LAYOUT_TF_HANDLE;
    tf_src.handle = h;
    tf_src.owns_handle = 0; // The caller retains ownership of h
    
    // dm_raise_block triggers TFE_TensorHandleResolve and memcpy internally
    dm_raise_block(&tf_src, out, DM_BACKEND_CPU, DM_LOWER_COPY);
    out->dirty = 1;
    out->version++;
}

static TFE_TensorHandle *raw_to_tf(const float *data, const int64_t *dims, int ndim) {
    dm_tf_init();
    DM_Block b;
    if (dm_block_view(&b, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, ndim, dims, (void*)data) != 0) return NULL;
    
    DM_Block tf_b;
    if (dm_lower_block(&b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return NULL;
    tf_b.owns_handle = 0;
    return (TFE_TensorHandle *)tf_b.handle;
}

static void tf_to_raw(TFE_TensorHandle *h, float *out) {
    if (!h || !out) return;
    TF_Status *s = TF_NewStatus();
    TF_Tensor *tft = TFE_TensorHandleResolve(h, s);
    if (TF_GetCode(s) == TF_OK)
        memcpy(out, TF_TensorData(tft), TF_TensorByteSize(tft));
    TF_DeleteTensor(tft);
    TF_DeleteStatus(s);
}

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
                                         int stride, const char *data_format) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "Conv2D", s);
    TFE_OpAddInput(op, input, s);
    TFE_OpAddInput(op, filter, s);
    int64_t strides[4]   = {1, 1, stride, stride};
    if (strcmp(data_format, "NHWC") == 0) {
        strides[1] = stride; strides[2] = stride; strides[3] = 1;
    }
    int64_t dilations[4] = {1, 1, 1, 1};
    TFE_OpSetAttrIntList(op, "strides",   strides,   4);
    TFE_OpSetAttrIntList(op, "dilations", dilations, 4);
    TFE_OpSetAttrString(op, "padding",     "SAME",  4);
    TFE_OpSetAttrString(op, "data_format", data_format, strlen(data_format));
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
                                           TFE_TensorHandle *bias, const char *data_format) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "BiasAdd", s);
    TFE_OpAddInput(op, input, s);
    TFE_OpAddInput(op, bias,  s);
    TFE_OpSetAttrString(op, "data_format", data_format, strlen(data_format));
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



/* CPU native transpose fallback: NCHW -> NHWC */
int dm_transpose_nchw_to_nhwc_cpu(const DM_Block *src, DM_Block *dst) {
    if (!dm_block_is_nchw4(src) || !dst) return -1;
    int n = DM_NCHW_N(src), c = DM_NCHW_C(src), h = DM_NCHW_H(src), w = DM_NCHW_W(src);
    if (dm_block_create(dst, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){n, h, w, c}) != 0) return -1;
    const float *sdata = (const float *)src->data;
    float *ddata = (float *)dst->data;
    for (int in = 0; in < n; in++) {
        for (int ic = 0; ic < c; ic++) {
            for (int ih = 0; ih < h; ih++) {
                for (int iw = 0; iw < w; iw++) {
                    ddata[in * h * w * c + ih * w * c + iw * c + ic] = sdata[in * c * h * w + ic * h * w + ih * w + iw];
                }
            }
        }
    }
    return 0;
}

/* CPU native transpose fallback: NHWC -> NCHW */
int dm_transpose_nhwc_to_nchw_cpu(const DM_Block *src, DM_Block *dst) {
    if (!src || src->ndim != 4 || !dst) return -1;
    int n = src->shape[0], h = src->shape[1], w = src->shape[2], c = src->shape[3];
    if (dm_block_create(dst, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){n, c, h, w}) != 0) return -1;
    const float *sdata = (const float *)src->data;
    float *ddata = (float *)dst->data;
    for (int in = 0; in < n; in++) {
        for (int ih = 0; ih < h; ih++) {
            for (int iw = 0; iw < w; iw++) {
                for (int ic = 0; ic < c; ic++) {
                    ddata[in * c * h * w + ic * h * w + ih * w + iw] = sdata[in * h * w * c + ih * w * c + iw * c + ic];
                }
            }
        }
    }
    return 0;
}

static TFE_TensorHandle *execute_tf_transpose(TFE_TensorHandle *input, int *perm, int perm_len) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    
    // Create perm tensor
    int64_t perm_dims[1] = {perm_len};
    TF_Tensor *tf_perm = TF_NewTensor(TF_INT32, perm_dims, 1, perm, perm_len * sizeof(int), noop_dealloc, NULL);
    TFE_TensorHandle *h_perm = TFE_NewTensorHandle(tf_perm, s);
    TF_DeleteTensor(tf_perm);

    TFE_Op *op = TFE_NewOp(tf_ctx, "Transpose", s);
    TFE_OpAddInput(op, input, s);
    TFE_OpAddInput(op, h_perm, s);

    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    if (TF_GetCode(s) != TF_OK)
        fprintf(stderr, "[dm_engine] Transpose failed: %s\n", TF_Message(s));
    
    TFE_DeleteOp(op);
    TFE_DeleteTensorHandle(h_perm);
    TF_DeleteStatus(s);
    return ret[0];
}

static TFE_TensorHandle *execute_depthwise_conv2d(TFE_TensorHandle *input,
                                                    TFE_TensorHandle *filter,
                                                    int stride, const char *data_format) {
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "DepthwiseConv2dNative", s);
    TFE_OpAddInput(op, input,  s);
    TFE_OpAddInput(op, filter, s);
    int64_t strides[4]   = {1, 1, stride, stride};
    if (strcmp(data_format, "NHWC") == 0) {
        strides[1] = stride; strides[2] = stride; strides[3] = 1;
    }
    int64_t dilations[4] = {1, 1, 1, 1};
    TFE_OpSetAttrIntList(op, "strides",   strides,   4);
    TFE_OpSetAttrIntList(op, "dilations", dilations, 4);
    TFE_OpSetAttrString(op, "padding",     "SAME", 4);
    TFE_OpSetAttrString(op, "data_format", data_format, 4);
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

size_t dm_tensor_count(const DM_Block *t) {
    if (!t) return 0;
    return (size_t)DM_NCHW_N(t) * DM_NCHW_C(t) * DM_NCHW_H(t) * DM_NCHW_W(t);
}

int dm_tensor_alloc(DM_Block *t, int n, int c, int h, int w) {
    return dm_block_create(t, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){n, c, h, w});
}

void dm_tensor_free(DM_Block *t) {
    dm_block_free(t);
}

void dm_tensor_fill(DM_Block *t, float value) {
    if (!t || !t->data) return;
    size_t n = dm_tensor_count(t);
    for (size_t i = 0; i < n; i++) ((float*)t->data)[i] = value;
}

static size_t idx4(const DM_Block *t, int n, int c, int y, int x) {
    return (((size_t)n * (size_t)DM_NCHW_C(t) + (size_t)c) * (size_t)DM_NCHW_H(t) + (size_t)y) * (size_t)DM_NCHW_W(t) + (size_t)x;
}

float dm_tensor_get(const DM_Block *t, int n, int c, int y, int x) {
    if (!t || !t->data) return 0.0f;
    size_t idx = (((size_t)n * DM_NCHW_C(t) + c) * DM_NCHW_H(t) + y) * DM_NCHW_W(t) + x;
    return ((float*)t->data)[idx];
}

static inline int out_size_same(int in_size, int stride) {
    return (in_size + stride - 1) / stride;
}

void dm_tensor_set(DM_Block *t, int n, int c, int y, int x, float v) {
    if (!t || !t->data) return;
    size_t idx = (((size_t)n * DM_NCHW_C(t) + c) * DM_NCHW_H(t) + y) * DM_NCHW_W(t) + x;
    ((float*)t->data)[idx] = v;
}

int dm_conv2d_same(const DM_Block *in, DM_Block *out,
                   const DM_Block *w, const DM_Block *b,
                   int out_c, int kernel, int stride) {
    if (!dm_block_is_nchw4(in)) return -1;

    if (!in || !out || !w || out_c <= 0 || kernel <= 0 || stride <= 0) return -1;
    int oh = out_size_same(DM_NCHW_H(in), stride);
    int ow = out_size_same(DM_NCHW_W(in), stride);
    if (dm_block_create(out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(in), out_c, oh, ow}) != 0) return -1;

    TFE_TensorHandle *h_in  = dm_to_tf(in);
    DM_Block tf_w;
    if (dm_lower_block(w, &tf_w, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) goto conv2d_err;
    TFE_TensorHandle *h_w = (TFE_TensorHandle *)tf_w.handle;
    if (!h_in || !h_w) goto conv2d_err;

    TFE_TensorHandle *h_out = NULL;
    if (dm_get_layout_policy() == DM_LAYOUT_POLICY_AUTO_TRANSPOSE) {
        int perm_to_nhwc[4] = {0, 2, 3, 1};
        TFE_TensorHandle *h_nhwc = execute_tf_transpose(h_in, perm_to_nhwc, 4);
        if (!h_nhwc) goto conv2d_err;
        TFE_TensorHandle *h_y_nhwc = execute_conv2d(h_nhwc, h_w, stride, "NHWC");
        TFE_DeleteTensorHandle(h_nhwc);
        if (!h_y_nhwc) goto conv2d_err;
        
        if (b) {
            DM_Block tf_b;
            if (dm_lower_block(b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) { TFE_DeleteTensorHandle(h_y_nhwc); goto conv2d_err; }
            TFE_TensorHandle *h_b = (TFE_TensorHandle *)tf_b.handle;
            TFE_TensorHandle *h_y2_nhwc = execute_bias_add(h_y_nhwc, h_b, "NHWC");
            TFE_DeleteTensorHandle(h_y_nhwc);
            h_y_nhwc = h_y2_nhwc;
        }
        
        int perm_to_nchw[4] = {0, 3, 1, 2};
        h_out = execute_tf_transpose(h_y_nhwc, perm_to_nchw, 4);
        TFE_DeleteTensorHandle(h_y_nhwc);
    } else {
        h_out = execute_conv2d(h_in, h_w, stride, "NCHW");
        if (!h_out) goto conv2d_err;
        if (b) {
            DM_Block tf_b;
            if (dm_lower_block(b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) { /* TFE_DeleteTensorHandle(h_out); */ goto conv2d_err; }
            TFE_TensorHandle *h_b = (TFE_TensorHandle *)tf_b.handle;
            TFE_TensorHandle *h_out2 = execute_bias_add(h_out, h_b, "NCHW");
            TFE_DeleteTensorHandle(h_out);
            h_out = h_out2;
        }
    }
    if (h_out) {
        tf_to_dm(h_out, out);
        TFE_DeleteTensorHandle(h_out);
    }
    return 0;

conv2d_err:
    return -1;
}

/* dm_depthwise_conv2d_same — Depthwise separable conv, SAME padding. */
int dm_depthwise_conv2d_same(const DM_Block *in, DM_Block *out,
                              const DM_Block *w, const DM_Block *b,
                              int kernel, int stride) {
    if (!dm_block_is_nchw4(in)) return -1;

    if (!in || !out || !w || kernel <= 0 || stride <= 0) return -1;
    int oh = out_size_same(DM_NCHW_H(in), stride);
    int ow = out_size_same(DM_NCHW_W(in), stride);
    if (dm_block_create(out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(in), DM_NCHW_C(in), oh, ow}) != 0) return -1;

    TFE_TensorHandle *h_in  = dm_to_tf(in);
    DM_Block tf_w;
    if (dm_lower_block(w, &tf_w, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) goto dw_err;
    TFE_TensorHandle *h_w = (TFE_TensorHandle *)tf_w.handle;
    if (!h_in || !h_w) goto dw_err;

    TFE_TensorHandle *h_out = NULL;
    if (dm_get_layout_policy() == DM_LAYOUT_POLICY_AUTO_TRANSPOSE) {
        int perm_to_nhwc[4] = {0, 2, 3, 1};
        TFE_TensorHandle *h_nhwc = execute_tf_transpose(h_in, perm_to_nhwc, 4);
        if (!h_nhwc) goto dw_err;
        TFE_TensorHandle *h_y_nhwc = execute_depthwise_conv2d(h_nhwc, h_w, stride, "NHWC");
        TFE_DeleteTensorHandle(h_nhwc);
        if (!h_y_nhwc) goto dw_err;
        
        if (b) {
            DM_Block tf_b;
            if (dm_lower_block(b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) { TFE_DeleteTensorHandle(h_y_nhwc); goto dw_err; }
            TFE_TensorHandle *h_b = (TFE_TensorHandle *)tf_b.handle;
            TFE_TensorHandle *h_y2_nhwc = execute_bias_add(h_y_nhwc, h_b, "NHWC");
            TFE_DeleteTensorHandle(h_y_nhwc);
            h_y_nhwc = h_y2_nhwc;
        }
        
        int perm_to_nchw[4] = {0, 3, 1, 2};
        h_out = execute_tf_transpose(h_y_nhwc, perm_to_nchw, 4);
        TFE_DeleteTensorHandle(h_y_nhwc);
    } else {
        h_out = execute_depthwise_conv2d(h_in, h_w, stride, "NCHW");
        if (!h_out) goto dw_err;
        
        if (b) {
            DM_Block tf_b;
            if (dm_lower_block(b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) { TFE_DeleteTensorHandle(h_out); goto dw_err; }
            TFE_TensorHandle *h_b = (TFE_TensorHandle *)tf_b.handle;
            TFE_TensorHandle *h_out2 = execute_bias_add(h_out, h_b, "NCHW");
            TFE_DeleteTensorHandle(h_out);
            h_out = h_out2;
        }
    }
    if (h_out) {
        tf_to_dm(h_out, out);
        TFE_DeleteTensorHandle(h_out);
    }
    return 0;

dw_err:
    return -1;
}

/* dm_pointwise_conv2d — 1×1 convolution (channel mixing). */
int dm_pointwise_conv2d(const DM_Block *in, DM_Block *out,
                         const DM_Block *w, const DM_Block *b, int out_c) {
    if (!in || !out || !w || out_c <= 0) return -1;
    if (dm_block_create(out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(in), out_c, DM_NCHW_H(in), DM_NCHW_W(in)}) != 0) return -1;

    TFE_TensorHandle *h_in = dm_to_tf(in);
    /* 1×1 conv: filter shape [out_c, in_c, 1, 1] */
    DM_Block tf_w;
    if (dm_lower_block(w, &tf_w, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) goto pw_err;
    TFE_TensorHandle *h_w = (TFE_TensorHandle *)tf_w.handle;
    if (!h_in || !h_w) goto pw_err;

    TFE_TensorHandle *h_out = NULL;
    if (dm_get_layout_policy() == DM_LAYOUT_POLICY_AUTO_TRANSPOSE) {
        int perm_to_nhwc[4] = {0, 2, 3, 1};
        TFE_TensorHandle *h_nhwc = execute_tf_transpose(h_in, perm_to_nhwc, 4);
        if (!h_nhwc) goto pw_err;
        TFE_TensorHandle *h_y_nhwc = execute_conv2d(h_nhwc, h_w, 1, "NHWC");
        TFE_DeleteTensorHandle(h_nhwc);
        if (!h_y_nhwc) goto pw_err;
        
        if (b) {
            DM_Block tf_b;
            if (dm_lower_block(b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) { TFE_DeleteTensorHandle(h_y_nhwc); goto pw_err; }
            TFE_TensorHandle *h_b = (TFE_TensorHandle *)tf_b.handle;
            TFE_TensorHandle *h_y2_nhwc = execute_bias_add(h_y_nhwc, h_b, "NHWC");
            TFE_DeleteTensorHandle(h_y_nhwc);
            h_y_nhwc = h_y2_nhwc;
        }
        
        int perm_to_nchw[4] = {0, 3, 1, 2};
        h_out = execute_tf_transpose(h_y_nhwc, perm_to_nchw, 4);
        TFE_DeleteTensorHandle(h_y_nhwc);
    } else {
        h_out = execute_conv2d(h_in, h_w, 1, "NCHW");
        if (!h_out) goto pw_err;
        if (b) {
            DM_Block tf_b;
            if (dm_lower_block(b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) { TFE_DeleteTensorHandle(h_out); goto pw_err; }
            TFE_TensorHandle *h_b = (TFE_TensorHandle *)tf_b.handle;
            TFE_TensorHandle *h_out2 = execute_bias_add(h_out, h_b, "NCHW");
            TFE_DeleteTensorHandle(h_out);
            h_out = h_out2;
        }
    }
    if (h_out) {
        tf_to_dm(h_out, out);
        TFE_DeleteTensorHandle(h_out);
    }
    return 0;

pw_err:
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Activations
 * ═══════════════════════════════════════════════════════════════════════════ */

void dm_relu6(DM_Block *t) {
    if (!t) return;
    TFE_TensorHandle *in_h  = dm_to_tf(t);
    if (!in_h) return;
    TFE_TensorHandle *out_h = op1("Relu6", in_h);
    if (out_h) { tf_to_dm(out_h, t); TFE_DeleteTensorHandle(out_h); }
    // /* TFE_DeleteTensorHandle(in_h); */
}

void dm_relu(DM_Block *t) {
    if (!t) return;
    TFE_TensorHandle *in_h  = dm_to_tf(t);
    if (!in_h) return;
    TFE_TensorHandle *out_h = op1("Relu", in_h);
    if (out_h) { tf_to_dm(out_h, t); TFE_DeleteTensorHandle(out_h); }
    // /* TFE_DeleteTensorHandle(in_h); */
}

/* GELU: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x³))) */
void dm_gelu_inplace(float *x, int n) {
    if (!x || n <= 0) return;
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.79788456f * (v + 0.044715f * v * v * v)));
    }
}

void dm_tanh_inplace(DM_Block *t) {
    if (!t) return;
    TFE_TensorHandle *in_h  = dm_to_tf(t);
    if (!in_h) return;
    TFE_TensorHandle *out_h = op1("Tanh", in_h);
    if (out_h) { tf_to_dm(out_h, t); TFE_DeleteTensorHandle(out_h); }
    // /* TFE_DeleteTensorHandle(in_h); */
}

void dm_sigmoid_inplace(DM_Block *t) {
    if (!t) return;
    TFE_TensorHandle *in_h  = dm_to_tf(t);
    if (!in_h) return;
    TFE_TensorHandle *out_h = op1("Sigmoid", in_h);
    if (out_h) { tf_to_dm(out_h, t); TFE_DeleteTensorHandle(out_h); }
    // /* TFE_DeleteTensorHandle(in_h); */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Elementwise add
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_tensor_add(DM_Block *out, const DM_Block *in) {
    if (!out || !in ||
        DM_NCHW_N(out) != DM_NCHW_N(in) || DM_NCHW_C(out) != DM_NCHW_C(in) ||
        DM_NCHW_H(out) != DM_NCHW_H(in) || DM_NCHW_W(out) != DM_NCHW_W(in)) return -1;
    TFE_TensorHandle *h_out = dm_to_tf(out);
    TFE_TensorHandle *h_in  = dm_to_tf(in);
    if (!h_out || !h_in) {
        if (h_out) // TFE_DeleteTensorHandle(h_out);
        if (h_in)  // /* TFE_DeleteTensorHandle(h_in); */
        return -1;
    }
    TFE_TensorHandle *res = op2("AddV2", h_out, h_in);
    if (res) { tf_to_dm(res, out); TFE_DeleteTensorHandle(res); }
    // TFE_DeleteTensorHandle(h_out);
    // /* TFE_DeleteTensorHandle(h_in); */
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Pooling
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_max_pool2d_same(const DM_Block *in, DM_Block *out, int kernel, int stride) {
    if (!in || !out || kernel <= 0 || stride <= 0) return -1;
    int oh = out_size_same(DM_NCHW_H(in), stride);
    int ow = out_size_same(DM_NCHW_W(in), stride);
    if (dm_block_create(out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(in), DM_NCHW_C(in), oh, ow}) != 0) return -1;

    dm_tf_init();
    TF_Status *s = TF_NewStatus();

    TFE_TensorHandle *h_in = dm_to_tf(in);
    if (!h_in) { TF_DeleteStatus(s); return -1; }

    TFE_TensorHandle *h_out = NULL;
    if (dm_get_layout_policy() == DM_LAYOUT_POLICY_AUTO_TRANSPOSE) {
        int perm_to_nhwc[4] = {0, 2, 3, 1};
        TFE_TensorHandle *h_nhwc = execute_tf_transpose(h_in, perm_to_nhwc, 4);
        if (!h_nhwc) { TF_DeleteStatus(s); return -1; }

        TFE_Op *op = TFE_NewOp(tf_ctx, "MaxPool", s);
        TFE_OpAddInput(op, h_nhwc, s);
        int64_t ksize[4]   = {1, kernel, kernel, 1};
        int64_t strides[4] = {1, stride, stride, 1};
        TFE_OpSetAttrIntList(op, "ksize",       ksize,   4);
        TFE_OpSetAttrIntList(op, "strides",     strides, 4);
        TFE_OpSetAttrString(op,  "padding",     "SAME",  4);
        TFE_OpSetAttrString(op,  "data_format", "NHWC",  4);

        TFE_TensorHandle *ret[1] = {NULL};
        int nret = 1;
        TFE_Execute(op, ret, &nret, s);
        TFE_DeleteOp(op);
        TFE_DeleteTensorHandle(h_nhwc);

        if (TF_GetCode(s) != TF_OK) {
            fprintf(stderr, "[dm_engine] MaxPool failed: %s\\n", TF_Message(s));
            TF_DeleteStatus(s); return -1;
        }

        int perm_to_nchw[4] = {0, 3, 1, 2};
        h_out = execute_tf_transpose(ret[0], perm_to_nchw, 4);
        TFE_DeleteTensorHandle(ret[0]);
    } else {
        TFE_Op *op = TFE_NewOp(tf_ctx, "MaxPool", s);
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
        TFE_DeleteOp(op);

        if (TF_GetCode(s) != TF_OK) {
            fprintf(stderr, "[dm_engine] MaxPool failed: %s\\n", TF_Message(s));
            TF_DeleteStatus(s); return -1;
        }
        h_out = ret[0];
    }

    if (h_out) { tf_to_raw(h_out, out->data); TFE_DeleteTensorHandle(h_out); }
    // /* TFE_DeleteTensorHandle(h_in); */
    TF_DeleteStatus(s);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Batch Normalization — FusedBatchNorm (inference mode, is_training=false)
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_batch_norm(DM_Block *t,
                  const float *gamma, const float *beta,
                  const float *mean,  const float *var,
                  float eps) {
    if (!t || !gamma || !beta || !mean || !var) return -1;
    int C = DM_NCHW_C(t);

    dm_tf_init();
    TF_Status *s = TF_NewStatus();

    TFE_TensorHandle *h_x     = dm_to_tf(t);
    int64_t cdims[1] = {C};
    TFE_TensorHandle *h_scale = raw_to_tf(gamma, cdims, 1);
    TFE_TensorHandle *h_offset= raw_to_tf(beta,  cdims, 1);
    TFE_TensorHandle *h_mean  = raw_to_tf(mean,  cdims, 1);
    TFE_TensorHandle *h_var   = raw_to_tf(var,   cdims, 1);

    TFE_TensorHandle *ret[5] = {NULL, NULL, NULL, NULL, NULL};
    int nret = 5;

    if (dm_get_layout_policy() == DM_LAYOUT_POLICY_AUTO_TRANSPOSE) {
        int perm_to_nhwc[4] = {0, 2, 3, 1};
        TFE_TensorHandle *h_nhwc = execute_tf_transpose(h_x, perm_to_nhwc, 4);
        
        TFE_Op *op = TFE_NewOp(tf_ctx, "FusedBatchNorm", s);
        TFE_OpAddInput(op, h_nhwc,      s);
        TFE_OpAddInput(op, h_scale,  s);
        TFE_OpAddInput(op, h_offset, s);
        TFE_OpAddInput(op, h_mean,   s);
        TFE_OpAddInput(op, h_var,    s);

        TFE_OpSetAttrFloat(op,  "epsilon",     eps);
        TFE_OpSetAttrBool(op,   "is_training", 0);
        TFE_OpSetAttrString(op, "data_format", "NHWC", 4);

        TFE_Execute(op, ret, &nret, s);
        TFE_DeleteOp(op);
        TFE_DeleteTensorHandle(h_nhwc);
        
        if (TF_GetCode(s) != TF_OK) {
            fprintf(stderr, "[dm_engine] FusedBatchNorm failed: %s\\n", TF_Message(s));
        } else {
            int perm_to_nchw[4] = {0, 3, 1, 2};
            TFE_TensorHandle *h_out = execute_tf_transpose(ret[0], perm_to_nchw, 4);
            TFE_DeleteTensorHandle(ret[0]);
            ret[0] = h_out;
        }
    } else {
        TFE_Op *op = TFE_NewOp(tf_ctx, "FusedBatchNorm", s);
        TFE_OpAddInput(op, h_x,      s);
        TFE_OpAddInput(op, h_scale,  s);
        TFE_OpAddInput(op, h_offset, s);
        TFE_OpAddInput(op, h_mean,   s);
        TFE_OpAddInput(op, h_var,    s);

        TFE_OpSetAttrFloat(op,  "epsilon",     eps);
        TFE_OpSetAttrBool(op,   "is_training", 0);
        TFE_OpSetAttrString(op, "data_format", "NCHW", 4);

        TFE_Execute(op, ret, &nret, s);
        if (TF_GetCode(s) != TF_OK)
            fprintf(stderr, "[dm_engine] FusedBatchNorm failed: %s\\n", TF_Message(s));
        TFE_DeleteOp(op);
    }

    TF_DeleteStatus(s);

    if (ret[0]) { tf_to_dm(ret[0], t); }
    for (int i = 0; i < 5; i++) if (ret[i]) TFE_DeleteTensorHandle(ret[i]);
    // TFE_DeleteTensorHandle(h_x);
    TFE_DeleteTensorHandle(h_scale);
    TFE_DeleteTensorHandle(h_offset);
    TFE_DeleteTensorHandle(h_mean);
    TFE_DeleteTensorHandle(h_var);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Global Average Pooling — Mean over spatial dims (h=2, w=3 in NCHW)
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_global_avg_pool(const DM_Block *in, DM_Block *out) {
    if (!in || !out) return -1;
    if (dm_block_create(out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(in), DM_NCHW_C(in), 1, 1}) != 0) return -1;

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
    // /* TFE_DeleteTensorHandle(h_in); */
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

int dm_linear(const DM_Block *in, DM_Block *out,
              const DM_Block *w, const DM_Block *b, int out_c) {
    if (!in || !out || !w || DM_NCHW_H(in) != 1 || DM_NCHW_W(in) != 1 || out_c <= 0) return -1;
    if (dm_block_create(out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(in), out_c, 1, 1}) != 0) return -1;

    /* Reshape to 2-D: [n, in_c] */
    int64_t adims[2] = {DM_NCHW_N(in), DM_NCHW_C(in)};
    TFE_TensorHandle *h_a = raw_to_tf(in->data, adims, 2);
    
    DM_Block tf_w;
    if (dm_lower_block(w, &tf_w, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) {
        if (h_a) TFE_DeleteTensorHandle(h_a);
        return -1;
    }
    TFE_TensorHandle *h_b = (TFE_TensorHandle *)tf_w.handle;
    
    /* Ensure h_b is 2D if needed. If w was created as 2D [out_c, in_c], no reshape is needed.
       We assume the caller provided a 2D block or 4D [out_c, in_c, 1, 1].
       For safety, we could reshape h_b to [out_c, in_c], but execute_tf_matmul requires 2D. 
       Let's assume w is passed with correct shape. */

    if (!h_a || !h_b) {
        if (h_a) TFE_DeleteTensorHandle(h_a);
        return -1;
    }

    /* out = in @ W^T  → [n, out_c] */
    TFE_TensorHandle *h_out = execute_tf_matmul(h_a, h_b, false, true);
    TFE_DeleteTensorHandle(h_a);
    if (!h_out) return -1;

    if (b) {
        DM_Block tf_b;
        if (dm_lower_block(b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) { TFE_DeleteTensorHandle(h_out); return -1; }
        TFE_TensorHandle *h_bias = (TFE_TensorHandle *)tf_b.handle;
        TFE_TensorHandle *h_out2 = op2("AddV2", h_out, h_bias);
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
void dm_softmax(DM_Block *t) {
    if (!t) return;

    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "Softmax", s);

    /* Reshape to [n, c] for the Softmax op (axis=-1 = axis 1) */
    int64_t dims2[2] = {DM_NCHW_N(t), DM_NCHW_C(t)};
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
    // /* TFE_DeleteTensorHandle(h_in); */
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
    // /* TFE_DeleteTensorHandle(h_in); */
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

int dm_layer_norm_seq(DM_Block *x, const DM_Block *gamma, const DM_Block *beta, float eps) {
    if (!x || !gamma || !beta) return -1;

    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    
    DM_Block tf_x, tf_g, tf_b;
    if (dm_lower_block(x, &tf_x, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0 ||
        dm_lower_block(gamma, &tf_g, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0 ||
        dm_lower_block(beta, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return -1;

    TFE_TensorHandle *h_x = (TFE_TensorHandle*)tf_x.handle;
    TFE_TensorHandle *h_g = (TFE_TensorHandle*)tf_g.handle;
    TFE_TensorHandle *h_b = (TFE_TensorHandle*)tf_b.handle;

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

    if (h_out) { tf_to_dm(h_out, x); TFE_DeleteTensorHandle(h_out); }

    // TFE_DeleteTensorHandle(h_x);
    // TFE_DeleteTensorHandle(h_g);
    // TFE_DeleteTensorHandle(h_b);
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
void dm_matmul_nt(const DM_Block *A, const DM_Block *B, DM_Block *C) {
    if (!A || !B || !C) return;
    DM_Block tf_A, tf_B;
    if (dm_lower_block(A, &tf_A, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0 ||
        dm_lower_block(B, &tf_B, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return;
    TFE_TensorHandle *hA = (TFE_TensorHandle *)tf_A.handle;
    TFE_TensorHandle *hB = (TFE_TensorHandle *)tf_B.handle;
    if (!hA || !hB) return;
    TFE_TensorHandle *res = execute_tf_matmul(hA, hB, false, true);
    if (res) { tf_to_dm(res, C); TFE_DeleteTensorHandle(res); }
}

/* C = A @ B    A[M×K], B[K×N] → C[M×N] */
void dm_matmul_nn(const DM_Block *A, const DM_Block *B, DM_Block *C) {
    if (!A || !B || !C) return;
    DM_Block tf_A, tf_B;
    if (dm_lower_block(A, &tf_A, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0 ||
        dm_lower_block(B, &tf_B, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) return;
    TFE_TensorHandle *hA = (TFE_TensorHandle *)tf_A.handle;
    TFE_TensorHandle *hB = (TFE_TensorHandle *)tf_B.handle;
    if (!hA || !hB) return;
    TFE_TensorHandle *res = execute_tf_matmul(hA, hB, false, false);
    if (res) { tf_to_dm(res, C); TFE_DeleteTensorHandle(res); }
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

int dm_linear_backward(const DM_Block *in, const DM_Block *grad_out,
                         DM_Block *grad_in,
                        float *grad_w, float *grad_b,
                        const DM_Block *w, int out_c) {
    if (!in || !grad_out) return -1;
    if (grad_in && dm_block_create(grad_in, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(in), DM_NCHW_C(in), 1, 1}) != 0) return -1;
    if (grad_in) memset(((float*)(grad_in)->data), 0, (grad_in)->count * sizeof(float));

    for (int n = 0; n < DM_NCHW_N(in); n++) {
        for (int oc = 0; oc < out_c; oc++) {
            float go = ((float*)(grad_out)->data)[(((size_t)n * DM_NCHW_C(grad_out) + oc) * DM_NCHW_H(grad_out) + 0) * DM_NCHW_W(grad_out) + 0];
            if (grad_b) grad_b[oc] += go;
            for (int ic = 0; ic < DM_NCHW_C(in); ic++) {
                float iv = ((float*)(in)->data)[(((size_t)n * DM_NCHW_C(in) + ic) * DM_NCHW_H(in) + 0) * DM_NCHW_W(in) + 0];
                if (grad_w) grad_w[(size_t)oc * (size_t)DM_NCHW_C(in) + ic] += go * iv;
                if (grad_in && w) {
                    float cur = ((float*)(grad_in)->data)[(((size_t)n * DM_NCHW_C(grad_in) + ic) * DM_NCHW_H(grad_in) + 0) * DM_NCHW_W(grad_in) + 0];
                    ((float*)(grad_in)->data)[(((size_t)n * DM_NCHW_C(grad_in) + ic) * DM_NCHW_H(grad_in) + 0) * DM_NCHW_W(grad_in) + 0] = cur + go * ((float*)w->data)[(size_t)oc * DM_NCHW_C(in) + ic];
                }
            }
        }
    }
    return 0;
}

void dm_tanh_backward(const DM_Block *out, const DM_Block *grad_out,
                         DM_Block *grad_in) {
    if (dm_block_create(grad_in, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(out), DM_NCHW_C(out), DM_NCHW_H(out), DM_NCHW_W(out)}) != 0) return;
    size_t count = (out)->count;
    for (size_t i = 0; i < count; i++) {
        float o = ((float*)out->data)[i];
        ((float*)grad_in->data)[i] = ((float*)grad_out->data)[i] * (1.0f - o * o);
    }
}

void dm_relu_backward(const DM_Block *in, const DM_Block *grad_out,
                         DM_Block *grad_in) {
    size_t count = (in)->count;
    for (size_t i = 0; i < count; i++)
        ((float*)grad_in->data)[i] = ((float*)in->data)[i] > 0.0f ? ((float*)grad_out->data)[i] : 0.0f;
}

void dm_matmul_nt_backward(const DM_Block *A, const DM_Block *B, const DM_Block *grad_out, 
                           DM_Block *grad_A, DM_Block *grad_B) {
    if (!A || !B || !grad_out) return;
    if (grad_A) dm_matmul_nn(grad_out, B, grad_A);
    if (grad_B) {
        DM_Block tf_g, tf_A;
        if (dm_lower_block(grad_out, &tf_g, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) == 0 &&
            dm_lower_block(A, &tf_A, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) == 0) {
            TFE_TensorHandle *hg = (TFE_TensorHandle *)tf_g.handle;
            TFE_TensorHandle *hA = (TFE_TensorHandle *)tf_A.handle;
            TFE_TensorHandle *res = execute_tf_matmul(hg, hA, true, false);
            if (res) { tf_to_dm(res, grad_B); TFE_DeleteTensorHandle(res); }
        }
    }
}

void dm_matmul_nn_backward(const DM_Block *A, const DM_Block *B, const DM_Block *grad_out, 
                           DM_Block *grad_A, DM_Block *grad_B) {
    if (!A || !B || !grad_out) return;
    if (grad_A) dm_matmul_nt(grad_out, B, grad_A);
    if (grad_B) {
        DM_Block tf_g, tf_A;
        if (dm_lower_block(grad_out, &tf_g, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) == 0 &&
            dm_lower_block(A, &tf_A, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) == 0) {
            TFE_TensorHandle *hg = (TFE_TensorHandle *)tf_g.handle;
            TFE_TensorHandle *hA = (TFE_TensorHandle *)tf_A.handle;
            TFE_TensorHandle *res = execute_tf_matmul(hA, hg, true, false);
            if (res) { tf_to_dm(res, grad_B); TFE_DeleteTensorHandle(res); }
        }
    }
}

int dm_layer_norm_seq_backward(const DM_Block *x, const DM_Block *gamma, 
                               const DM_Block *grad_out, float eps,
                               DM_Block *grad_x, DM_Block *grad_gamma, DM_Block *grad_beta) {
    if (!x || !gamma || !grad_out) return -1;
    int64_t dim = gamma->shape[0];
    int64_t rows = x->count / dim;
    const float *xp = (const float *)x->data;
    const float *gp = (const float *)gamma->data;
    const float *gop = (const float *)grad_out->data;
    
    float *gxp = grad_x ? (float *)grad_x->data : NULL;
    float *ggp = grad_gamma ? (float *)grad_gamma->data : NULL;
    float *gbp = grad_beta ? (float *)grad_beta->data : NULL;
    
    if (ggp) memset(ggp, 0, dim * sizeof(float));
    if (gbp) memset(gbp, 0, dim * sizeof(float));
    
    for (int64_t r = 0; r < rows; r++) {
        const float *xr = xp + r * dim;
        const float *gor = gop + r * dim;
        float *gxr = gxp ? gxp + r * dim : NULL;
        
        double mean = 0.0, var = 0.0;
        for (int64_t c = 0; c < dim; c++) mean += xr[c];
        mean /= (double)dim;
        for (int64_t c = 0; c < dim; c++) {
            double d = (double)xr[c] - mean;
            var += d * d;
        }
        var /= (double)dim;
        float inv = 1.0f / sqrtf((float)var + eps);
        
        double sum_gor_x_gp = 0.0, sum_gor_x_gp_x_xhat = 0.0;
        for (int64_t c = 0; c < dim; c++) {
            float xhat = (xr[c] - (float)mean) * inv;
            float g = gor[c] * gp[c];
            sum_gor_x_gp += g;
            sum_gor_x_gp_x_xhat += g * xhat;
            
            if (ggp) ggp[c] += gor[c] * xhat;
            if (gbp) gbp[c] += gor[c];
        }
        
        if (gxr) {
            for (int64_t c = 0; c < dim; c++) {
                float xhat = (xr[c] - (float)mean) * inv;
                float grad_xhat = gor[c] * gp[c];
                gxr[c] = inv * (grad_xhat - (float)(sum_gor_x_gp / dim) - xhat * (float)(sum_gor_x_gp_x_xhat / dim));
            }
        }
    }
    return 0;
}

void dm_gelu_backward(const DM_Block *x, const DM_Block *grad_out, DM_Block *grad_in) {
    if (!x || !grad_out || !grad_in) return;
    const float *xp = (const float *)x->data;
    const float *gop = (const float *)grad_out->data;
    float *gip = (float *)grad_in->data;
    for (size_t i = 0; i < x->count; i++) {
        float v = xp[i];
        float t = tanhf(0.79788456f * (v + 0.044715f * v * v * v));
        float cdf = 0.5f * (1.0f + t);
        float pdf = 0.5f * 0.79788456f * (1.0f - t * t) * (1.0f + 0.134145f * v * v);
        gip[i] = gop[i] * (cdf + v * pdf);
    }
}

void dm_softmax_backward(const DM_Block *y, const DM_Block *grad_out, DM_Block *grad_in) {
    if (!y || !grad_out || !grad_in) return;
    int64_t dim = y->shape[y->ndim - 1];
    int64_t rows = y->count / dim;
    const float *yp = (const float *)y->data;
    const float *gop = (const float *)grad_out->data;
    float *gip = (float *)grad_in->data;
    
    for (int64_t r = 0; r < rows; r++) {
        const float *yr = yp + r * dim;
        const float *gor = gop + r * dim;
        float *gir = gip + r * dim;
        
        double sum = 0.0;
        for (int64_t c = 0; c < dim; c++) sum += yr[c] * gor[c];
        
        for (int64_t c = 0; c < dim; c++) {
            gir[c] = yr[c] * (gor[c] - (float)sum);
        }
    }
}

/* ── Maxout ─────────────────────────────────────────────────────────────── */

int dm_maxout(const DM_Block *in,  DM_Block *out, int k, int *argmax) {
    if (k <= 0 || DM_NCHW_C(in) % k != 0) return -1;
    int out_c = DM_NCHW_C(in) / k;
    if (DM_NCHW_N(out) != DM_NCHW_N(in) || DM_NCHW_C(out) != out_c ||
        DM_NCHW_H(out) != DM_NCHW_H(in) || DM_NCHW_W(out) != DM_NCHW_W(in)) return -1;

    int spatial = DM_NCHW_H(in) * DM_NCHW_W(in);
    for (int n = 0; n < DM_NCHW_N(in); n++)
        for (int c = 0; c < out_c; c++)
            for (int s = 0; s < spatial; s++) {
                float mx = -1e30f; int mi = -1;
                for (int j = 0; j < k; j++) {
                    int ic  = c * k + j;
                    int idx = (n * DM_NCHW_C(in) + ic) * spatial + s;
                    if (((float*)in->data)[idx] > mx || mi == -1) { mx = ((float*)in->data)[idx]; mi = idx; }
                }
                int oi = (n * out_c + c) * spatial + s;
                ((float*)out->data)[oi] = mx;
                if (argmax) argmax[oi] = mi;
            }
    return 0;
}

int dm_maxout_backward(const DM_Block *grad_out, DM_Block *grad_in,
                        int k, const int *argmax) {
    if (DM_NCHW_C(grad_in) % k != 0 || DM_NCHW_C(grad_in) / k != DM_NCHW_C(grad_out)) return -1;
    if (!argmax) return -1;
    size_t in_count  = (grad_in)->count;
    size_t out_count = (grad_out)->count;
    for (size_t i = 0; i < in_count; i++) ((float*)grad_in->data)[i] = 0.0f;
    for (size_t i = 0; i < out_count; i++) {
        int mi = argmax[i];
        if (mi >= 0 && (size_t)mi < in_count)
            ((float*)grad_in->data)[mi] += ((float*)grad_out->data)[i];
    }
    return 0;
}

/* ── Dropout ────────────────────────────────────────────────────────────── */

void dm_dropout(const DM_Block *in,  DM_Block *out, float drop_prob, int *mask) {
    size_t count = (in)->count;
    float scale = 1.0f / (1.0f - drop_prob);
    for (size_t i = 0; i < count; i++) {
        float r = (float)rand() / (float)RAND_MAX;
        if (r < drop_prob) {
            ((float*)out->data)[i] = 0.0f;
            if (mask) mask[i] = 0;
        } else {
            ((float*)out->data)[i] = ((float*)in->data)[i] * scale;
            if (mask) mask[i] = 1;
        }
    }
}

void dm_dropout_backward(const DM_Block *grad_out, DM_Block *grad_in,
                          float drop_prob, const int *mask) {
    size_t count = (grad_out)->count;
    float scale = 1.0f / (1.0f - drop_prob);
    for (size_t i = 0; i < count; i++)
        ((float*)grad_in->data)[i] = (mask && mask[i]) ? ((float*)grad_out->data)[i] * scale : 0.0f;
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

/* ── DM_WeightCache Implementation ──────────────────────────────────────── */

DM_WeightCache *dm_weight_cache_new(void) {
    DM_WeightCache *c = malloc(sizeof(DM_WeightCache));
    if (!c) return NULL;
    c->blocks = NULL;
    c->seeds = NULL;
    c->count = 0;
    c->capacity = 0;
    return c;
}

DM_Block *dm_weight_cache_get(DM_WeightCache *cache, int ndim, const int64_t *shape, unsigned int seed, float scale) {
    for (int i = 0; i < cache->count; i++) {
        if (cache->seeds[i] == seed) return cache->blocks[i];
    }
    if (cache->count >= cache->capacity) {
        cache->capacity = cache->capacity ? cache->capacity * 2 : 16;
        cache->blocks = realloc(cache->blocks, cache->capacity * sizeof(DM_Block *));
        cache->seeds = realloc(cache->seeds, cache->capacity * sizeof(uint32_t));
    }
    DM_Block *b = malloc(sizeof(DM_Block));
    dm_block_create(b, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, ndim, shape);
    
    // Generate weights
    uint32_t s = seed;
    size_t n = b->count;
    for (size_t i = 0; i < n; i++) {
        uint32_t x = s ? s : 2463534242u;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        s = x;
        float val = (((s >> 8) * (1.0f / 16777216.0f)) * 2.0f - 1.0f) * scale;
        // Ensure variance/scale-like 1D vectors are positive to avoid NaN in BatchNorm
        if (ndim == 1 && scale == 1.0f && val < 0.01f) val = 0.01f - val; 
        ((float*)b->data)[i] = val;
    }
    
    // Lower block once (Phase 5 persistent caching)
    DM_Block tf_b;
    dm_lower_block(b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW);
    
    cache->blocks[cache->count] = b;
    cache->seeds[cache->count] = seed;
    cache->count++;
    return b;
}

void dm_weight_cache_free(DM_WeightCache *cache) {
    if (!cache) return;
    for (int i = 0; i < cache->count; i++) {
        dm_block_free(cache->blocks[i]);
        free(cache->blocks[i]);
    }
    free(cache->blocks);
    free(cache->seeds);
    free(cache);
}

void dm_softmax_last_dim(DM_Block *t) {
    if (!t) return;
    dm_tf_init();
    TF_Status *s = TF_NewStatus();
    TFE_Op *op = TFE_NewOp(tf_ctx, "Softmax", s);
    
    DM_Block tf_in;
    if (dm_lower_block(t, &tf_in, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW) != 0) { TF_DeleteStatus(s); return; }
    TFE_TensorHandle *h_in = (TFE_TensorHandle*)tf_in.handle;
    
    TFE_OpAddInput(op, h_in, s);
    TFE_TensorHandle *ret[1] = {NULL};
    int nret = 1;
    TFE_Execute(op, ret, &nret, s);
    TFE_DeleteOp(op);
    if (TF_GetCode(s) != TF_OK) {
        fprintf(stderr, "[dm_engine] Softmax failed: %s\n", TF_Message(s));
    }
    if (ret[0]) { tf_to_dm(ret[0], t); TFE_DeleteTensorHandle(ret[0]); }
    TF_DeleteStatus(s);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Training Runtime Substrate Lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_activation_cache_init(DM_ActivationCache *cache, size_t initial_capacity) {
    if (!cache) return -1;
    cache->count = 0;
    cache->capacity = initial_capacity > 0 ? initial_capacity : 64;
    cache->activations = (DM_Block *)calloc(cache->capacity, sizeof(DM_Block));
    if (!cache->activations) return -1;
    return 0;
}

void dm_activation_cache_free(DM_ActivationCache *cache) {
    if (!cache || !cache->activations) return;
    for (size_t i = 0; i < cache->count; i++) {
        dm_block_free(&cache->activations[i]);
    }
    free(cache->activations);
    cache->activations = NULL;
    cache->count = 0;
    cache->capacity = 0;
}

int dm_activation_cache_push(DM_ActivationCache *cache, const DM_Block *block) {
    if (!cache || !block) return -1;
    if (cache->count >= cache->capacity) {
        size_t new_cap = cache->capacity * 2;
        DM_Block *new_acts = (DM_Block *)realloc(cache->activations, new_cap * sizeof(DM_Block));
        if (!new_acts) return -1;
        memset(new_acts + cache->capacity, 0, (new_cap - cache->capacity) * sizeof(DM_Block));
        cache->activations = new_acts;
        cache->capacity = new_cap;
    }
    int rc = dm_block_create(&cache->activations[cache->count], block->kind, block->dtype, block->layout, block->backend, block->ndim, block->shape);
    if (rc == 0) {
        memcpy(cache->activations[cache->count].data, block->data, block->bytes);
        cache->count++;
    }
    return rc;
}

int dm_training_context_init(DM_TrainingContext *ctx, size_t num_params) {
    if (!ctx) return -1;
    memset(ctx, 0, sizeof(*ctx));
    ctx->num_params = num_params;
    if (num_params > 0) {
        ctx->params = (DM_TrainableParam *)calloc(num_params, sizeof(DM_TrainableParam));
        ctx->opt_states = (DM_OptimizerState *)calloc(num_params, sizeof(DM_OptimizerState));
        if (!ctx->params || !ctx->opt_states) {
            free(ctx->params);
            free(ctx->opt_states);
            return -1;
        }
    }
    ctx->learning_rate = 0.001f;
    ctx->beta1 = 0.9f;
    ctx->beta2 = 0.999f;
    ctx->eps = 1e-8f;
    ctx->weight_decay = 0.0f;
    ctx->step = 0;
    return 0;
}

void dm_training_context_free(DM_TrainingContext *ctx) {
    if (!ctx) return;
    for (size_t i = 0; i < ctx->num_params; i++) {
        if (ctx->params) {
            if (ctx->params[i].param) dm_block_free(&ctx->params[i].param->weight);
            if (ctx->params[i].grad) dm_block_free(&ctx->params[i].grad->grad);
        }
        if (ctx->opt_states) {
            dm_block_free(&ctx->opt_states[i].m);
            dm_block_free(&ctx->opt_states[i].v);
        }
    }
    if (ctx->params) { free(ctx->params); ctx->params = NULL; }
    if (ctx->opt_states) { free(ctx->opt_states); ctx->opt_states = NULL; }
    ctx->num_params = 0;
}

