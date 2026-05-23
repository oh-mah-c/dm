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

void dm_relu(DM_Tensor *t) {
    if (!t) return;
    size_t i, n = dm_tensor_count(t);
    for (i = 0; i < n; i++) {
        if (t->data[i] < 0.0f) t->data[i] = 0.0f;
    }
}

int dm_tensor_add(DM_Tensor *out, const DM_Tensor *in) {
    if (!out || !in || out->n != in->n || out->c != in->c || out->h != in->h || out->w != in->w) return -1;
    size_t i, n = dm_tensor_count(out);
    for (i = 0; i < n; i++) {
        out->data[i] += in->data[i];
    }
    return 0;
}

int dm_max_pool2d_same(const DM_Tensor *in, DM_Tensor *out, int kernel, int stride) {
    int oh, ow, n, c, oy, ox, ky, kx;
    if (!in || !out || kernel <= 0 || stride <= 0) return -1;
    oh = out_size_same(in->h, stride);
    ow = out_size_same(in->w, stride);
    if (dm_tensor_alloc(out, in->n, in->c, oh, ow) != 0) return -1;
    int pad_h = (oh - 1) * stride + kernel - in->h;
    int pad_w = (ow - 1) * stride + kernel - in->w;
    int pad_top = pad_h > 0 ? pad_h / 2 : 0;
    int pad_left = pad_w > 0 ? pad_w / 2 : 0;
    for (n = 0; n < in->n; n++) {
        for (c = 0; c < in->c; c++) {
            for (oy = 0; oy < oh; oy++) {
                for (ox = 0; ox < ow; ox++) {
                    float max_val = -1e30f;
                    for (ky = 0; ky < kernel; ky++) {
                        for (kx = 0; kx < kernel; kx++) {
                            int iy = oy * stride + ky - pad_top;
                            int ix = ox * stride + kx - pad_left;
                            if (iy >= 0 && iy < in->h && ix >= 0 && ix < in->w) {
                                float val = dm_tensor_get(in, n, c, iy, ix);
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    dm_tensor_set(out, n, c, oy, ox, max_val);
                }
            }
        }
    }
    return 0;
}

int dm_batch_norm(DM_Tensor *t, const float *gamma, const float *beta, const float *mean, const float *var, float eps) {
    int n, c, y, x;
    if (!t || !gamma || !beta || !mean || !var) return -1;
    for (n = 0; n < t->n; n++) {
        for (c = 0; c < t->c; c++) {
            float s = gamma[c] / sqrtf(fabsf(var[c]) + eps);
            float offset = beta[c] - mean[c] * s;
            for (y = 0; y < t->h; y++) {
                for (x = 0; x < t->w; x++) {
                    float val = dm_tensor_get(t, n, c, y, x);
                    dm_tensor_set(t, n, c, y, x, val * s + offset);
                }
            }
        }
    }
    return 0;
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

/* ─── LayerNorm over rows ──────────────────────────────────────────────────
 * x : [seq_len × d_model] row-major
 * Normalises each row independently: y = (x - μ) / √(σ²+ε) * γ + β
 */
int dm_layer_norm_seq(float *x, int seq_len, int d_model,
                      const float *gamma, const float *beta, float eps)
{
    int s, d;
    if (!x || !gamma || !beta || seq_len <= 0 || d_model <= 0) return -1;
    for (s = 0; s < seq_len; s++) {
        float *row = x + (size_t)s * d_model;
        float mean = 0.0f, var = 0.0f;
        for (d = 0; d < d_model; d++) mean += row[d];
        mean /= (float)d_model;
        for (d = 0; d < d_model; d++) { float v = row[d] - mean; var += v * v; }
        var /= (float)d_model;
        float inv_std = 1.0f / sqrtf(var + eps);
        for (d = 0; d < d_model; d++)
            row[d] = (row[d] - mean) * inv_std * gamma[d] + beta[d];
    }
    return 0;
}

/* ─── GELU (paper §3.1: MLP uses GELU) ────────────────────────────────────
 * Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x³)))
 */
void dm_gelu_inplace(float *x, int n)
{
    static const float K = 0.7978845608f; /* sqrt(2/pi) */
    int i;
    for (i = 0; i < n; i++) {
        float v = x[i];
        float t = tanhf(K * (v + 0.044715f * v * v * v));
        x[i] = 0.5f * v * (1.0f + t);
    }
}

/* ─── Matrix multiply: C = A @ B^T ─────────────────────────────────────────
 * A[M×K], B[N×K] → C[M×N]
 * Used for attention score: Q @ K^T
 */
void dm_matmul_nt(const float *A, const float *B, float *C, int M, int N, int K)
{
    int m, n, k;
    for (m = 0; m < M; m++) {
        for (n = 0; n < N; n++) {
            float s = 0.0f;
            const float *ra = A + (size_t)m * K;
            const float *rb = B + (size_t)n * K;
            for (k = 0; k < K; k++) s += ra[k] * rb[k];
            C[(size_t)m * N + n] = s;
        }
    }
}

/* ─── Matrix multiply: C = A @ B ───────────────────────────────────────────
 * A[M×K], B[K×N] → C[M×N]
 * Used for projection: attn @ V and linear projections
 */
void dm_matmul_nn(const float *A, const float *B, float *C, int M, int K, int N)
{
    int m, k, n;
    memset(C, 0, sizeof(float) * (size_t)M * N);
    for (m = 0; m < M; m++) {
        const float *ra = A + (size_t)m * K;
        float *rc = C + (size_t)m * N;
        for (k = 0; k < K; k++) {
            float a = ra[k];
            const float *rb = B + (size_t)k * N;
            for (n = 0; n < N; n++) rc[n] += a * rb[n];
        }
    }
}

/* ─── Softmax over each row ─────────────────────────────────────────────────
 * x[rows × cols] → softmax applied row-wise in-place
 */
void dm_softmax_rows(float *x, int rows, int cols)
{
    int r, c;
    for (r = 0; r < rows; r++) {
        float *row = x + (size_t)r * cols;
        float mx = row[0];
        for (c = 1; c < cols; c++) if (row[c] > mx) mx = row[c];
        float sum = 0.0f;
        for (c = 0; c < cols; c++) { row[c] = expf(row[c] - mx); sum += row[c]; }
        if (sum <= 0.0f) sum = 1.0f;
        for (c = 0; c < cols; c++) row[c] /= sum;
    }
}

/* ─── Training Primitives ────────────────────────────────────────────────── */
int dm_linear_backward(const DM_Tensor *in, const DM_Tensor *grad_out, DM_Tensor *grad_in, 
                       float *grad_w, float *grad_b, const float *w, int out_c) {
    if (!in || !grad_out) return -1;
    int n, oc, ic;
    if (grad_in && dm_tensor_alloc(grad_in, in->n, in->c, 1, 1) != 0) return -1;
    if (grad_in) dm_tensor_fill(grad_in, 0.0f);
    
    for (n = 0; n < in->n; n++) {
        for (oc = 0; oc < out_c; oc++) {
            float go = dm_tensor_get(grad_out, n, oc, 0, 0);
            if (grad_b) grad_b[oc] += go;
            for (ic = 0; ic < in->c; ic++) {
                float in_val = dm_tensor_get(in, n, ic, 0, 0);
                if (grad_w) grad_w[(size_t)oc * (size_t)in->c + (size_t)ic] += go * in_val;
                if (grad_in && w) {
                    float current_gi = dm_tensor_get(grad_in, n, ic, 0, 0);
                    dm_tensor_set(grad_in, n, ic, 0, 0, current_gi + go * w[(size_t)oc * (size_t)in->c + (size_t)ic]);
                }
            }
        }
    }
    return 0;
}

void dm_tanh_inplace(DM_Tensor *t) {
    size_t count = dm_tensor_count(t);
    for (size_t i = 0; i < count; i++) {
        t->data[i] = tanhf(t->data[i]);
    }
}

void dm_tanh_backward(const DM_Tensor *out, const DM_Tensor *grad_out, DM_Tensor *grad_in) {
    if (dm_tensor_alloc(grad_in, out->n, out->c, out->h, out->w) != 0) return;
    size_t count = dm_tensor_count(out);
    for (size_t i = 0; i < count; i++) {
        float o = out->data[i];
        grad_in->data[i] = grad_out->data[i] * (1.0f - o * o);
    }
}

void dm_sigmoid_inplace(DM_Tensor *t) {
    size_t count = dm_tensor_count(t);
    for (size_t i = 0; i < count; i++) {
        t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
    }
}

void dm_adagrad_step(float *param, float *grad, float *g_sum, int n, float lr, float eps, float weight_decay) {
    for (int i = 0; i < n; i++) {
        float g = grad[i];
        if (weight_decay > 0.0f) g += weight_decay * param[i];
        g_sum[i] += g * g;
        param[i] -= (lr / (sqrtf(g_sum[i]) + eps)) * g;
        grad[i] = 0.0f;
    }
}

void dm_adam_step(float *param, float *grad, float *m, float *v, int n, 
                  float lr, float beta1, float beta2, float eps, float weight_decay, int t) {
    float beta1_t = 1.0f - powf(beta1, (float)t);
    float beta2_t = 1.0f - powf(beta2, (float)t);
    
    for (int i = 0; i < n; i++) {
        float g = grad[i];
        if (weight_decay > 0.0f) g += weight_decay * param[i];
        
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        
        float m_hat = m[i] / beta1_t;
        float v_hat = v[i] / beta2_t;
        
        param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
        grad[i] = 0.0f; // reset grad
    }
}

void dm_sgd_momentum_step(float *param, float *grad, float *velocity, int n, float lr, float momentum, float weight_decay, int nesterov) {
    for (int i = 0; i < n; i++) {
        float g = grad[i];
        if (weight_decay > 0.0f) g += weight_decay * param[i];
        velocity[i] = momentum * velocity[i] - lr * g;
        if (nesterov) {
            param[i] += momentum * velocity[i] - lr * g;
        } else {
            param[i] += velocity[i];
        }
        grad[i] = 0.0f;
    }
}

void dm_relu_backward(const DM_Tensor *in, const DM_Tensor *grad_out, DM_Tensor *grad_in) {
    size_t count = dm_tensor_count(in);
    for (size_t i = 0; i < count; i++) {
        grad_in->data[i] = in->data[i] > 0.0f ? grad_out->data[i] : 0.0f;
    }
}

int dm_maxout(const DM_Tensor *in, DM_Tensor *out, int k, int *argmax) {
    if (k <= 0 || in->c % k != 0) return -1;
    int out_c = in->c / k;
    if (out->n != in->n || out->c != out_c || out->h != in->h || out->w != in->w) return -1;
    
    int spatial = in->h * in->w;
    for (int n = 0; n < in->n; n++) {
        for (int c = 0; c < out_c; c++) {
            for (int s = 0; s < spatial; s++) {
                float max_val = -1e30f;
                int max_idx = -1;
                for (int j = 0; j < k; j++) {
                    int in_c = c * k + j;
                    int in_idx = (n * in->c + in_c) * spatial + s;
                    float val = in->data[in_idx];
                    if (val > max_val || max_idx == -1) {
                        max_val = val;
                        max_idx = in_idx;
                    }
                }
                int out_idx = (n * out_c + c) * spatial + s;
                out->data[out_idx] = max_val;
                if (argmax) argmax[out_idx] = max_idx;
            }
        }
    }
    return 0;
}

int dm_maxout_backward(const DM_Tensor *grad_out, DM_Tensor *grad_in, int k, const int *argmax) {
    if (grad_in->c % k != 0 || grad_in->c / k != grad_out->c) return -1;
    if (!argmax) return -1;
    
    size_t in_count = dm_tensor_count(grad_in);
    for (size_t i = 0; i < in_count; i++) {
        grad_in->data[i] = 0.0f;
    }
    
    size_t out_count = dm_tensor_count(grad_out);
    for (size_t i = 0; i < out_count; i++) {
        int max_idx = argmax[i];
        if (max_idx >= 0 && (size_t)max_idx < in_count) {
            grad_in->data[max_idx] += grad_out->data[i];
        }
    }
    return 0;
}

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

void dm_dropout_backward(const DM_Tensor *grad_out, DM_Tensor *grad_in, float drop_prob, const int *mask) {
    size_t count = dm_tensor_count(grad_out);
    float scale = 1.0f / (1.0f - drop_prob);
    for (size_t i = 0; i < count; i++) {
        grad_in->data[i] = mask && mask[i] ? grad_out->data[i] * scale : 0.0f;
    }
}
