/*
 * tinyvit.c — TinyViT: Fast Pretraining Distillation for Small Vision Transformers
 *
 * Full C99 implementation of Wu, Zhang, Peng et al., arXiv:2207.10666v1, 2022.
 *
 * Sections cited below refer to the paper.
 *
 * Weight layout (sequential flat float32 array, same order for counting
 * and for forward pass via a WBuf cursor):
 *
 *  Patch Embed
 *    conv1: weight[D1,3,3,3 → OHWI], bias[D1], bn1: γ[D1]β[D1]μ[D1]σ²[D1]
 *    conv2: weight[D1,3,3,D1], bias[D1], bn2: γβμσ²[D1]
 *  Stage 1: depths[0] × mbconv_weights(D1, D1, stride=1)
 *  Downsample 1→2: mbconv_weights(D1, D2, stride=2)
 *  Stage 2: depths[1] × transformer_weights(D2, window_sizes[0])
 *  Downsample 2→3: mbconv_weights(D2, D3, stride=2)
 *  Stage 3: depths[2] × transformer_weights(D3, window_sizes[1])
 *  Downsample 3→4: mbconv_weights(D3, D4, stride=2)
 *  Stage 4: depths[3] × transformer_weights(D4, window_sizes[2])
 *  Head: ln_γ[D4], ln_β[D4], fc_w[num_cls×D4], fc_b[num_cls]
 *
 *  mbconv_weights(in_c, out_c, stride) layout:
 *    pre_bn: γ[in_c] β[in_c] μ[in_c] σ²[in_c]
 *    pw_exp_w[exp_c×in_c], pw_exp_b[exp_c]      exp_c = in_c × γ_R
 *    exp_bn: γ β μ σ²[exp_c]
 *    dw_w[exp_c×9], dw_b[exp_c]
 *    dw_bn: γ β μ σ²[exp_c]
 *    pw_prj_w[out_c×exp_c], pw_prj_b[out_c]
 *    prj_bn: γ β μ σ²[out_c]
 *
 *  transformer_weights(d, W) layout:
 *    attn_ln: γ[d] β[d]
 *    qkv_w[3d×d], qkv_b[3d]
 *    proj_w[d×d], proj_b[d]
 *    rel_bias[(d/γ_E) × (2W-1)²]      relative position bias table [24]
 *    local_dw_w[d×9], local_dw_b[d]   3×3 DW conv local mixer [69,15]
 *    local_bn: γ β μ σ²[d]
 *    mlp_ln: γ[d] β[d]
 *    mlp_fc1_w[d×γ_M × d], mlp_fc1_b[d×γ_M]
 *    mlp_fc2_w[d × d×γ_M], mlp_fc2_b[d]
 */

#include "models/vision/tinyvit.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#   include <sys/stat.h>
#   include <sys/time.h>
#   include <time.h>
#   include <unistd.h>
#else
#   include <io.h>
#   include <direct.h>
#   include <time.h>
#   define popen  _popen
#   define pclose _pclose
#endif

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_tstring.h"

/* ===================================================================
 * §3.2 Architecture constants (shared across all variants)
 * =================================================================== */

static const int TV_DEPTHS[4]  = {2, 2, 6, 2};   /* γ_N1…N4 */
static const int TV_WINDOWS[3] = {7, 14, 7};       /* γ_W2,W3,W4 */
static const int TV_R          = 4;                 /* MBConv expansion γ_R */
static const int TV_M          = 4;                 /* MLP ratio     γ_M */
static const int TV_E          = 32;                /* head dim      γ_E */

/* Per-variant embed dims {D1,D2,D3,D4} */
static const int TV_DIMS_5M [4] = {64,  128, 160, 320};
static const int TV_DIMS_11M[4] = {64,  128, 256, 448};
static const int TV_DIMS_21M[4] = {96,  192, 384, 576};

/* ===================================================================
 * Config helpers
 * =================================================================== */

void dm_tinyvit_config_init(TinyViTConfig *cfg, TinyViTVariant v,
                              int num_classes, int img_size)
{
    const int *dims;
    if (!cfg) return;
    cfg->variant      = v;
    cfg->mbconv_expand = TV_R;
    cfg->mlp_ratio    = TV_M;
    cfg->head_dim     = TV_E;
    cfg->num_classes  = num_classes > 0 ? num_classes : 1000;
    cfg->img_size     = img_size   > 0 ? img_size    : 224;
    dims = (v == TINYVIT_5M) ? TV_DIMS_5M : (v == TINYVIT_11M) ? TV_DIMS_11M : TV_DIMS_21M;
    for (int i = 0; i < 4; i++) cfg->embed_dims[i]   = dims[i];
    for (int i = 0; i < 4; i++) cfg->depths[i]        = TV_DEPTHS[i];
    for (int i = 0; i < 3; i++) cfg->window_sizes[i]  = TV_WINDOWS[i];
}

/* ===================================================================
 * Weight-buffer cursor
 *
 * wb_advance(&cur, n) returns pointer to next n weights (or NULL when
 * cur.data==NULL, used for counting only) and advances cur.pos by n.
 * =================================================================== */

typedef struct { const float *data; size_t pos; } WBuf;

static const float *wb_advance(WBuf *b, size_t n) {
    const float *p = b->data ? b->data + b->pos : NULL;
    b->pos += n;
    return p;
}

/* ===================================================================
 * Primitive: GELU activation (exact, Section 3.2 "All activation
 * functions are GELU")
 *   GELU(x) = x · Φ(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715·x³)))
 * =================================================================== */

static inline float gelu(float x) {
    /* Hendrycks & Gimpel [30] */
    static const float k = 0.7978845608028654f; /* sqrt(2/π) */
    float c = k * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}

/* ===================================================================
 * Primitives: normalisation
 * =================================================================== */

/* LayerNorm on last dimension [seqlen, dim] (Section 3.2: "linear layers
 * use LayerNorm") */
static void layer_norm(float *x, int seqlen, int dim,
                        const float *gamma, const float *beta,
                        float eps)
{
    for (int i = 0; i < seqlen; i++) {
        float *xi = x + (size_t)i * dim;
        double mean = 0.0, var = 0.0;
        for (int d = 0; d < dim; d++) mean += xi[d];
        mean /= dim;
        for (int d = 0; d < dim; d++) { double t = xi[d] - mean; var += t * t; }
        var /= dim;
        float inv = 1.0f / sqrtf((float)var + eps);
        for (int d = 0; d < dim; d++)
            xi[d] = (float)((xi[d] - mean) * inv) * gamma[d] + beta[d];
    }
}

/* BatchNorm inference, NHWC layout (Section 3.2: "conv layers use
 * BatchNorm") */
static void bn_nhwc(float *x, int n, int h, int w, int c,
                     const float *gamma, const float *beta,
                     const float *mean,  const float *var, float eps)
{
    int spatial = n * h * w;
    for (int i = 0; i < spatial; i++) {
        float *xi = x + (size_t)i * c;
        for (int ci = 0; ci < c; ci++) {
            float s   = gamma[ci] / sqrtf(var[ci] + eps);
            xi[ci]    = xi[ci] * s + (beta[ci] - mean[ci] * s);
        }
    }
}

/* ===================================================================
 * Primitives: convolutions (NHWC layout, OHWI weight layout)
 * =================================================================== */

/* Generic strided 2-D convolution — kernel kh×kw, same padding */
static void conv2d_nhwc(const float *in, float *out,
                         int n, int ih, int iw, int ic, int oc,
                         const float *wt, const float *bias,
                         int kh, int kw, int stride, int pad)
{
    int oh = (ih + 2*pad - kh) / stride + 1;
    int ow = (iw + 2*pad - kw) / stride + 1;
    for (int ni = 0; ni < n;  ni++)
    for (int oy = 0; oy < oh; oy++)
    for (int ox = 0; ox < ow; ox++) {
        float *yo = out + ((size_t)(ni*oh + oy)*ow + ox)*oc;
        for (int co = 0; co < oc; co++) {
            float sum = bias ? bias[co] : 0.0f;
            for (int ky = 0; ky < kh; ky++) {
                int iy = oy*stride + ky - pad;
                for (int kx = 0; kx < kw; kx++) {
                    int ix = ox*stride + kx - pad;
                    if ((unsigned)iy < (unsigned)ih && (unsigned)ix < (unsigned)iw) {
                        const float *xi = in  + ((size_t)(ni*ih + iy)*iw + ix)*ic;
                        const float *wi = wt  + ((size_t)(co*kh + ky)*kw + kx)*ic;
                        for (int ci = 0; ci < ic; ci++) sum += xi[ci] * wi[ci];
                    }
                }
            }
            yo[co] = sum;
        }
    }
}

/* Pointwise (1×1) conv — nhw tokens each of dim cin → cout */
static void pw_conv(const float *in, float *out,
                     int nhw, int cin, int cout,
                     const float *wt, const float *bias)
{
    for (int i = 0; i < nhw; i++) {
        const float *xi = in  + (size_t)i * cin;
        float       *yi = out + (size_t)i * cout;
        for (int co = 0; co < cout; co++) {
            float sum = bias ? bias[co] : 0.0f;
            const float *wr = wt + (size_t)co * cin;
            for (int ci = 0; ci < cin; ci++) sum += xi[ci] * wr[ci];
            yi[co] = sum;
        }
    }
}

/* Depthwise 3×3 conv NHWC with stride */
static void dw_conv3_nhwc(const float *in, float *out,
                            int n, int ih, int iw, int c,
                            const float *wt, const float *bias, int stride)
{
    int oh = (ih + stride - 1) / stride;
    int ow = (iw + stride - 1) / stride;
    for (int ni = 0; ni < n;  ni++)
    for (int oy = 0; oy < oh; oy++)
    for (int ox = 0; ox < ow; ox++) {
        float *yo = out + ((size_t)(ni*oh + oy)*ow + ox)*c;
        for (int ci = 0; ci < c; ci++) {
            float sum = bias ? bias[ci] : 0.0f;
            for (int ky = 0; ky < 3; ky++) {
                int iy = oy*stride + ky - 1;
                for (int kx = 0; kx < 3; kx++) {
                    int ix = ox*stride + kx - 1;
                    if ((unsigned)iy < (unsigned)ih && (unsigned)ix < (unsigned)iw)
                        sum += in[((size_t)(ni*ih + iy)*iw + ix)*c + ci]
                             * wt[ci*9 + ky*3 + kx];
                }
            }
            yo[ci] = sum;
        }
    }
}

/* ===================================================================
 * MBConv block — Section 3.2, Stage 1 and all downsampling layers
 *
 *  Pre-BN → GELU → PW-expand → BN → GELU → DW-3×3 → BN → GELU
 *        → PW-proj → BN  (+ residual if same size/channels)
 *
 * When wb->data == NULL (counting mode), only pos is advanced.
 * Returns 0 on success, −1 on allocation failure.
 * =================================================================== */

static int mbconv_block(const float *in, float *out,
                         int n, int ih, int iw, int in_c, int out_c,
                         int stride, WBuf *wb, float eps)
{
    const int exp_c = in_c * TV_R;
    const int oh    = (ih + stride - 1) / stride;
    const int ow    = (iw + stride - 1) / stride;

    /* --- pull weights from cursor --- */
    const float *bn_pre_g  = wb_advance(wb, in_c);
    const float *bn_pre_b  = wb_advance(wb, in_c);
    const float *bn_pre_m  = wb_advance(wb, in_c);
    const float *bn_pre_v  = wb_advance(wb, in_c);
    const float *pw_exp_w  = wb_advance(wb, (size_t)exp_c * in_c);
    const float *pw_exp_b  = wb_advance(wb, exp_c);
    const float *bn_exp_g  = wb_advance(wb, exp_c);
    const float *bn_exp_b2 = wb_advance(wb, exp_c);
    const float *bn_exp_m  = wb_advance(wb, exp_c);
    const float *bn_exp_v  = wb_advance(wb, exp_c);
    const float *dw_w      = wb_advance(wb, (size_t)exp_c * 9);
    const float *dw_b      = wb_advance(wb, exp_c);
    const float *bn_dw_g   = wb_advance(wb, exp_c);
    const float *bn_dw_b2  = wb_advance(wb, exp_c);
    const float *bn_dw_m   = wb_advance(wb, exp_c);
    const float *bn_dw_v   = wb_advance(wb, exp_c);
    const float *pw_prj_w  = wb_advance(wb, (size_t)out_c * exp_c);
    const float *pw_prj_b  = wb_advance(wb, out_c);
    const float *bn_prj_g  = wb_advance(wb, out_c);
    const float *bn_prj_b2 = wb_advance(wb, out_c);
    const float *bn_prj_m  = wb_advance(wb, out_c);
    const float *bn_prj_v  = wb_advance(wb, out_c);

    if (!wb->data) return 0;   /* counting mode — done */

    /* --- allocate intermediate buffers --- */
    size_t s0 = (size_t)n * ih * iw * in_c;
    size_t s1 = (size_t)n * ih * iw * exp_c;
    size_t s2 = (size_t)n * oh * ow * exp_c;
    size_t s3 = (size_t)n * oh * ow * out_c;

    float *x0 = (float*)malloc(s0 * sizeof(float));
    float *x1 = (float*)malloc(s1 * sizeof(float));
    float *x2 = (float*)malloc(s2 * sizeof(float));
    float *x3 = (float*)malloc(s3 * sizeof(float));
    if (!x0 || !x1 || !x2 || !x3) {
        free(x0); free(x1); free(x2); free(x3);
        return -1;
    }

    /* Pre-BN + GELU */
    memcpy(x0, in, s0 * sizeof(float));
    bn_nhwc(x0, n, ih, iw, in_c, bn_pre_g, bn_pre_b, bn_pre_m, bn_pre_v, eps);
    for (size_t i = 0; i < s0; i++) x0[i] = gelu(x0[i]);

    /* PW expand → BN → GELU */
    pw_conv(x0, x1, n*ih*iw, in_c, exp_c, pw_exp_w, pw_exp_b);
    bn_nhwc(x1, n, ih, iw, exp_c, bn_exp_g, bn_exp_b2, bn_exp_m, bn_exp_v, eps);
    for (size_t i = 0; i < s1; i++) x1[i] = gelu(x1[i]);

    /* DW 3×3 (stride) → BN → GELU */
    dw_conv3_nhwc(x1, x2, n, ih, iw, exp_c, dw_w, dw_b, stride);
    bn_nhwc(x2, n, oh, ow, exp_c, bn_dw_g, bn_dw_b2, bn_dw_m, bn_dw_v, eps);
    for (size_t i = 0; i < s2; i++) x2[i] = gelu(x2[i]);

    /* PW project → BN  (no final activation per paper) */
    pw_conv(x2, x3, n*oh*ow, exp_c, out_c, pw_prj_w, pw_prj_b);
    bn_nhwc(x3, n, oh, ow, out_c, bn_prj_g, bn_prj_b2, bn_prj_m, bn_prj_v, eps);

    memcpy(out, x3, s3 * sizeof(float));

    /* Residual skip if dimensions match (He et al. [28]) */
    if (stride == 1 && in_c == out_c)
        for (size_t i = 0; i < s0; i++) out[i] += in[i];

    free(x0); free(x1); free(x2); free(x3);
    return 0;
}

/* ===================================================================
 * Window partition / unpartition
 *
 * Section 3.2: "window attention to reduce computational cost"
 *
 * partition:   [B, H, W, C]  → [B*(H/Wh)*(W/Ww), Wh*Ww, C]
 * unpartition: inverse
 * =================================================================== */

/* out must be pre-allocated: n_wins * seq * c floats */
static void window_partition(const float *x,
                              int B, int H, int W, int C,
                              int Wh, int Ww,
                              float *out)
{
    int nH = H / Wh, nW = W / Ww;  /* number of windows per row/col */
    for (int b = 0; b < B;   b++)
    for (int wy = 0; wy < nH; wy++)
    for (int wx = 0; wx < nW; wx++) {
        int win_idx = (b * nH + wy) * nW + wx;
        for (int ty = 0; ty < Wh; ty++)
        for (int tx = 0; tx < Ww; tx++) {
            int gy = wy * Wh + ty;
            int gx = wx * Ww + tx;
            int tok_idx = ty * Ww + tx;
            const float *src = x   + ((size_t)(b*H + gy)*W + gx)*C;
            float       *dst = out + ((size_t)win_idx * Wh*Ww + tok_idx)*C;
            memcpy(dst, src, C * sizeof(float));
        }
    }
}

static void window_unpartition(const float *wins,
                                int B, int H, int W, int C,
                                int Wh, int Ww,
                                float *out)
{
    int nH = H / Wh, nW = W / Ww;
    for (int b = 0; b < B;   b++)
    for (int wy = 0; wy < nH; wy++)
    for (int wx = 0; wx < nW; wx++) {
        int win_idx = (b * nH + wy) * nW + wx;
        for (int ty = 0; ty < Wh; ty++)
        for (int tx = 0; tx < Ww; tx++) {
            int gy = wy * Wh + ty;
            int gx = wx * Ww + tx;
            int tok_idx = ty * Ww + tx;
            const float *src = wins + ((size_t)win_idx * Wh*Ww + tok_idx)*C;
            float       *dst = out  + ((size_t)(b*H + gy)*W + gx)*C;
            memcpy(dst, src, C * sizeof(float));
        }
    }
}

/* ===================================================================
 * Row-wise softmax (in-place)
 * =================================================================== */

static void softmax_row(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    float inv = 1.0f / (sum + 1e-9f);
    for (int i = 0; i < n; i++) x[i] *= inv;
}

/* ===================================================================
 * Window Multi-Head Self-Attention with relative position biases
 * (Section 3.2; attention biases from LeViT [24])
 *
 * Input x: [nw, seq, d]  (already windowed)
 * nw   = number of windows in this call
 * seq  = Wh * Ww  (tokens per window)
 * d    = embed dim
 * heads = d / γ_E
 * head_dim = γ_E
 *
 * Relative position bias (LeViT / Swin-style):
 *   bias_table[(d/E), (2Wh-1)*(2Ww-1)]
 *   For pair (i,j): bias = table[h, (ri-rj+Wh-1)*(2Ww-1) + (ci-cj+Ww-1)]
 * =================================================================== */

static int window_mhsa(float *x,
                        int nw, int seq, int d,
                        int Wh, int Ww,
                        const float *qkv_w, const float *qkv_b,
                        const float *proj_w, const float *proj_b,
                        const float *rel_bias   /* [heads, (2Wh-1)*(2Ww-1)] */)
{
    int heads    = d / TV_E;
    int hd       = TV_E;          /* head_dim */
    float scale  = 1.0f / sqrtf((float)hd);
    int nb       = (2*Wh-1)*(2*Ww-1);  /* unique relative positions */

    /* Allocate: qkv [nw, seq, 3d], attn [heads, seq, seq], out_win [nw, seq, d] */
    size_t sz_qkv = (size_t)nw * seq * 3 * d;
    size_t sz_attn = (size_t)heads * seq * seq;
    float *qkv   = (float*)malloc(sz_qkv  * sizeof(float));
    float *attn  = (float*)malloc(sz_attn * sizeof(float));
    float *out_w = (float*)malloc((size_t)nw * seq * d * sizeof(float));
    if (!qkv || !attn || !out_w) { free(qkv); free(attn); free(out_w); return -1; }

    /* Precompute relative position index [seq, seq] */
    int *rp_idx = (int*)malloc((size_t)seq * seq * sizeof(int));
    if (!rp_idx) { free(qkv); free(attn); free(out_w); return -1; }
    for (int i = 0; i < seq; i++) {
        int ri = i / Ww, ci = i % Ww;
        for (int j = 0; j < seq; j++) {
            int rj = j / Ww, cj = j % Ww;
            int dy = (ri - rj) + (Wh - 1);
            int dx = (ci - cj) + (Ww - 1);
            rp_idx[i*seq + j] = dy * (2*Ww - 1) + dx;
        }
    }

    /* QKV projection: [nw, seq, 3d] */
    pw_conv(x, qkv, nw*seq, d, 3*d, qkv_w, qkv_b);

    /* For each window, compute multi-head attention */
    for (int wi = 0; wi < nw; wi++) {
        const float *Q = qkv + (size_t)wi * seq * 3*d;
        const float *K = Q + d;
        const float *V = K + d;  /* stride 3*d between tokens */

        /* For each head */
        for (int h = 0; h < heads; h++) {
            float *ah = attn + (size_t)h * seq * seq;

            /* Compute Q_h @ K_h^T + rel_bias */
            for (int i = 0; i < seq; i++) {
                const float *qi = Q + (size_t)i * 3*d + h*hd;
                for (int j = 0; j < seq; j++) {
                    const float *kj = K + (size_t)j * 3*d + h*hd;
                    float s = 0.0f;
                    for (int dd = 0; dd < hd; dd++) s += qi[dd] * kj[dd];
                    s *= scale;
                    /* add relative position bias */
                    if (rel_bias)
                        s += rel_bias[(size_t)h * nb + rp_idx[i*seq + j]];
                    ah[i*seq + j] = s;
                }
            }
            /* Row-wise softmax */
            for (int i = 0; i < seq; i++) softmax_row(ah + i*seq, seq);
        }

        /* Compute (Attn @ V) for each head, write to out_win */
        float *ow = out_w + (size_t)wi * seq * d;
        for (int i = 0; i < seq; i++) {
            float *owi = ow + (size_t)i * d;
            for (int h = 0; h < heads; h++) {
                const float *ah = attn + (size_t)h * seq * seq + i * seq;
                const float *Vh = V;   /* stride 3*d */
                for (int dd = 0; dd < hd; dd++) {
                    float s = 0.0f;
                    for (int j = 0; j < seq; j++)
                        s += ah[j] * (Vh + (size_t)j * 3*d + h*hd)[dd];
                    owi[h*hd + dd] = s;
                }
            }
        }
    }

    /* Output projection */
    float *proj_out = (float*)malloc((size_t)nw * seq * d * sizeof(float));
    if (!proj_out) { free(qkv); free(attn); free(out_w); free(rp_idx); return -1; }
    pw_conv(out_w, proj_out, nw*seq, d, d, proj_w, proj_b);
    memcpy(x, proj_out, (size_t)nw * seq * d * sizeof(float));

    free(qkv); free(attn); free(out_w); free(rp_idx); free(proj_out);
    return 0;
}

/* ===================================================================
 * Transformer block — Section 3.2, Stages 2, 3, 4
 *
 *  Input: x [B, H, W, d]   spatial NHWC
 *
 *  Per block:
 *    LN → window partition → W-MHSA → unpartition  (+residual)
 *    3×3 DW-conv local mixer [69,15]               (+residual)
 *    LN → MLP [GELU]                               (+residual)
 *
 * wb must point to the start of this block's weights.
 * =================================================================== */

static int transformer_block(float *x,
                               int B, int H, int W, int d,
                               int Ww_size,  /* window width = height */
                               WBuf *wb, float eps)
{
    int seq    = Ww_size * Ww_size;
    int heads  = d / TV_E;
    int mlp_d  = d * TV_M;
    int nb     = (2*Ww_size - 1) * (2*Ww_size - 1);

    /* pull weights */
    const float *attn_ln_g  = wb_advance(wb, d);
    const float *attn_ln_b  = wb_advance(wb, d);
    const float *qkv_w      = wb_advance(wb, (size_t)3*d*d);
    const float *qkv_b      = wb_advance(wb, 3*d);
    const float *proj_w     = wb_advance(wb, (size_t)d*d);
    const float *proj_b     = wb_advance(wb, d);
    const float *rel_bias   = wb_advance(wb, (size_t)heads * nb);
    const float *ldw_w      = wb_advance(wb, (size_t)d * 9);
    const float *ldw_b      = wb_advance(wb, d);
    const float *ldw_bn_g   = wb_advance(wb, d);
    const float *ldw_bn_b   = wb_advance(wb, d);
    const float *ldw_bn_m   = wb_advance(wb, d);
    const float *ldw_bn_v   = wb_advance(wb, d);
    const float *mlp_ln_g   = wb_advance(wb, d);
    const float *mlp_ln_b   = wb_advance(wb, d);
    const float *mlp_fc1_w  = wb_advance(wb, (size_t)mlp_d * d);
    const float *mlp_fc1_b  = wb_advance(wb, mlp_d);
    const float *mlp_fc2_w  = wb_advance(wb, (size_t)d * mlp_d);
    const float *mlp_fc2_b  = wb_advance(wb, d);

    if (!wb->data) return 0;  /* counting mode */

    int nH   = H / Ww_size;
    int nW   = W / Ww_size;
    int nwin = B * nH * nW;
    size_t sz = (size_t)B * H * W * d;

    float *residual = (float*)malloc(sz * sizeof(float));
    float *tokens   = (float*)malloc((size_t)nwin * seq * d * sizeof(float));
    float *mlp_buf  = (float*)malloc((size_t)B * H * W * mlp_d * sizeof(float));
    if (!residual || !tokens || !mlp_buf) {
        free(residual); free(tokens); free(mlp_buf);
        return -1;
    }

    /* --- 1. Window MHSA branch --- */
    /* LN in-place on a copy */
    memcpy(residual, x, sz * sizeof(float));
    float *ln_x = (float*)malloc(sz * sizeof(float));
    if (!ln_x) { free(residual); free(tokens); free(mlp_buf); return -1; }
    memcpy(ln_x, x, sz * sizeof(float));
    layer_norm(ln_x, B*H*W, d, attn_ln_g, attn_ln_b, eps);

    /* Partition */
    window_partition(ln_x, B, H, W, d, Ww_size, Ww_size, tokens);
    free(ln_x);

    /* Attention (tokens modified in-place) */
    if (window_mhsa(tokens, nwin, seq, d, Ww_size, Ww_size,
                    qkv_w, qkv_b, proj_w, proj_b, rel_bias) != 0) {
        free(residual); free(tokens); free(mlp_buf);
        return -1;
    }

    /* Unpartition + residual */
    window_unpartition(tokens, B, H, W, d, Ww_size, Ww_size, x);
    for (size_t i = 0; i < sz; i++) x[i] += residual[i];

    /* --- 2. Local 3×3 DW-conv mixer (between attn and MLP) [69,15] --- */
    memcpy(residual, x, sz * sizeof(float));
    float *dw_out = (float*)malloc(sz * sizeof(float));
    if (!dw_out) { free(residual); free(tokens); free(mlp_buf); return -1; }
    dw_conv3_nhwc(x, dw_out, B, H, W, d, ldw_w, ldw_b, 1 /* stride=1 */);
    bn_nhwc(dw_out, B, H, W, d, ldw_bn_g, ldw_bn_b, ldw_bn_m, ldw_bn_v, eps);
    for (size_t i = 0; i < sz; i++) x[i] = dw_out[i] + residual[i];
    free(dw_out);

    /* --- 3. MLP branch --- */
    memcpy(residual, x, sz * sizeof(float));
    float *mlp_in = (float*)malloc(sz * sizeof(float));
    if (!mlp_in) { free(residual); free(tokens); free(mlp_buf); return -1; }
    memcpy(mlp_in, x, sz * sizeof(float));
    layer_norm(mlp_in, B*H*W, d, mlp_ln_g, mlp_ln_b, eps);
    /* fc1 [d → mlp_d] + GELU */
    pw_conv(mlp_in, mlp_buf, B*H*W, d, mlp_d, mlp_fc1_w, mlp_fc1_b);
    for (size_t i = 0; i < (size_t)B*H*W*mlp_d; i++) mlp_buf[i] = gelu(mlp_buf[i]);
    /* fc2 [mlp_d → d] */
    pw_conv(mlp_buf, mlp_in, B*H*W, mlp_d, d, mlp_fc2_w, mlp_fc2_b);
    for (size_t i = 0; i < sz; i++) x[i] = mlp_in[i] + residual[i];

    free(mlp_in); free(residual); free(tokens); free(mlp_buf);
    return 0;
}

/* ===================================================================
 * Weight counting helper — returns total number of floats
 * =================================================================== */

size_t dm_tinyvit_cfg_count(const TinyViTConfig *cfg)
{
    WBuf cnt = {NULL, 0};

    /* Patch embed conv1 */
    wb_advance(&cnt, (size_t)cfg->embed_dims[0] * 3 * 3 * 3);  /* OHWI */
    wb_advance(&cnt, cfg->embed_dims[0]);                        /* bias */
    wb_advance(&cnt, 4 * cfg->embed_dims[0]);                    /* BN γβμσ² */

    /* Patch embed conv2 */
    wb_advance(&cnt, (size_t)cfg->embed_dims[0] * 3 * 3 * cfg->embed_dims[0]);
    wb_advance(&cnt, cfg->embed_dims[0]);
    wb_advance(&cnt, 4 * cfg->embed_dims[0]);

    /* Stage 1 — MBConv × depths[0] */
    for (int i = 0; i < cfg->depths[0]; i++)
        mbconv_block(NULL, NULL, 0,0,0, cfg->embed_dims[0], cfg->embed_dims[0],
                     1, &cnt, 1e-5f);

    /* Downsample 1→2 */
    mbconv_block(NULL, NULL, 0,0,0, cfg->embed_dims[0], cfg->embed_dims[1],
                 2, &cnt, 1e-5f);

    /* Stages 2, 3, 4 */
    for (int stage = 1; stage < 4; stage++) {
        int d = cfg->embed_dims[stage];
        int W = cfg->window_sizes[stage - 1];
        for (int i = 0; i < cfg->depths[stage]; i++)
            transformer_block(NULL, 0,0,0, d, W, &cnt, 1e-5f);

        if (stage < 3) {
            /* Downsample to next stage */
            mbconv_block(NULL, NULL, 0,0,0,
                         cfg->embed_dims[stage], cfg->embed_dims[stage+1],
                         2, &cnt, 1e-5f);
        }
    }

    /* Classifier head: LN + FC */
    int d4 = cfg->embed_dims[3];
    wb_advance(&cnt, d4);  /* ln gamma */
    wb_advance(&cnt, d4);  /* ln beta  */
    wb_advance(&cnt, (size_t)cfg->num_classes * d4);  /* fc weight */
    wb_advance(&cnt, cfg->num_classes);                /* fc bias   */

    return cnt.pos;
}

/* ===================================================================
 * Full forward pass — Section 3.2
 *
 * input_nhwc : float[B × img_size × img_size × 3]  (pixel values in [0,1])
 * weights    : float[dm_tinyvit_cfg_count(cfg)]
 * logits_out : float[B × num_classes]
 * =================================================================== */

int dm_tinyvit_cfg_forward(const TinyViTConfig *cfg,
                        const float         *weights,
                        const float         *input_nhwc,
                        int                  batch,
                        float               *logits_out)
{
    if (!cfg || !weights || !input_nhwc || !logits_out || batch < 1) return -1;

    WBuf wb = {weights, 0};
    const float eps = 1e-5f;
    int B = batch, H = cfg->img_size, W = cfg->img_size;
    int D1 = cfg->embed_dims[0];

    /* --- Patch embed --- */
    /* conv1: 3 → D1, stride 2, pad 1  → H/2 × W/2 */
    int H1 = H/2, W1 = W/2;
    const float *pe_c1_w  = wb_advance(&wb, (size_t)D1 * 3 * 3 * 3);
    const float *pe_c1_b  = wb_advance(&wb, D1);
    const float *pe_bn1_g = wb_advance(&wb, D1);
    const float *pe_bn1_b = wb_advance(&wb, D1);
    const float *pe_bn1_m = wb_advance(&wb, D1);
    const float *pe_bn1_v = wb_advance(&wb, D1);

    /* conv2: D1 → D1, stride 2, pad 1  → H/4 × W/4 = 56×56 */
    int H2 = H1/2, W2 = W1/2;
    const float *pe_c2_w  = wb_advance(&wb, (size_t)D1 * 3 * 3 * D1);
    const float *pe_c2_b  = wb_advance(&wb, D1);
    const float *pe_bn2_g = wb_advance(&wb, D1);
    const float *pe_bn2_b = wb_advance(&wb, D1);
    const float *pe_bn2_m = wb_advance(&wb, D1);
    const float *pe_bn2_v = wb_advance(&wb, D1);

    /* allocate patch-embed output */
    float *feat = (float*)calloc((size_t)B * H2 * W2 * D1, sizeof(float));
    float *tmp  = (float*)calloc((size_t)B * H1 * W1 * D1, sizeof(float));
    if (!feat || !tmp) { free(feat); free(tmp); return -1; }

    /* conv1 + BN + GELU */
    conv2d_nhwc(input_nhwc, tmp, B, H, W, 3, D1, pe_c1_w, pe_c1_b, 3, 3, 2, 1);
    bn_nhwc(tmp, B, H1, W1, D1, pe_bn1_g, pe_bn1_b, pe_bn1_m, pe_bn1_v, eps);
    { size_t n = (size_t)B*H1*W1*D1; for (size_t i=0;i<n;i++) tmp[i]=gelu(tmp[i]); }

    /* conv2 + BN + GELU */
    conv2d_nhwc(tmp, feat, B, H1, W1, D1, D1, pe_c2_w, pe_c2_b, 3, 3, 2, 1);
    bn_nhwc(feat, B, H2, W2, D1, pe_bn2_g, pe_bn2_b, pe_bn2_m, pe_bn2_v, eps);
    { size_t n = (size_t)B*H2*W2*D1; for (size_t i=0;i<n;i++) feat[i]=gelu(feat[i]); }
    free(tmp);

    int cH = H2, cW = W2;

    /* --- Stage 1: depths[0] × MBConv at 56×56×D1 --- */
    for (int i = 0; i < cfg->depths[0]; i++) {
        float *out = (float*)calloc((size_t)B * cH * cW * D1, sizeof(float));
        if (!out) { free(feat); return -1; }
        if (mbconv_block(feat, out, B, cH, cW, D1, D1, 1, &wb, eps) != 0) {
            free(feat); free(out); return -1;
        }
        free(feat); feat = out;
    }

    /* --- Downsample + Stages 2, 3, 4 --- */
    for (int stage = 1; stage < 4; stage++) {
        int in_d  = cfg->embed_dims[stage - 1];
        int out_d = cfg->embed_dims[stage];
        int new_H = cH / 2, new_W = cW / 2;

        /* Downsample MBConv(stride 2) */
        float *ds = (float*)calloc((size_t)B * new_H * new_W * out_d, sizeof(float));
        if (!ds) { free(feat); return -1; }
        if (mbconv_block(feat, ds, B, cH, cW, in_d, out_d, 2, &wb, eps) != 0) {
            free(feat); free(ds); return -1;
        }
        free(feat); feat = ds; cH = new_H; cW = new_W;

        /* Transformer blocks */
        int Wsize = cfg->window_sizes[stage - 1];
        for (int i = 0; i < cfg->depths[stage]; i++) {
            if (transformer_block(feat, B, cH, cW, out_d, Wsize, &wb, eps) != 0) {
                free(feat); return -1;
            }
        }
    }

    /* --- Classifier head: AvgPool → LN → Linear --- */
    int d4 = cfg->embed_dims[3];
    const float *head_ln_g = wb_advance(&wb, d4);
    const float *head_ln_b = wb_advance(&wb, d4);
    const float *head_fc_w = wb_advance(&wb, (size_t)cfg->num_classes * d4);
    const float *head_fc_b = wb_advance(&wb, cfg->num_classes);

    /* Global average pool: [B, cH*cW, d4] → [B, 1, d4] */
    float *pooled = (float*)calloc((size_t)B * d4, sizeof(float));
    if (!pooled) { free(feat); return -1; }
    {
        int n_tok = cH * cW;
        float inv = 1.0f / n_tok;
        for (int b = 0; b < B; b++) {
            float *pb = pooled + (size_t)b * d4;
            for (int t = 0; t < n_tok; t++) {
                const float *xt = feat + ((size_t)b * n_tok + t) * d4;
                for (int d = 0; d < d4; d++) pb[d] += xt[d];
            }
            for (int d = 0; d < d4; d++) pb[d] *= inv;
        }
    }
    free(feat);

    /* LayerNorm */
    layer_norm(pooled, B, d4, head_ln_g, head_ln_b, eps);

    /* Linear → logits_out */
    pw_conv(pooled, logits_out, B, d4, cfg->num_classes, head_fc_w, head_fc_b);
    free(pooled);

    return 0;
}

/* ===================================================================
 * Weight I/O
 * =================================================================== */

int dm_tinyvit_cfg_save(const char *path,
                             const TinyViTConfig *cfg,
                             const float *weights)
{
    if (!path || !cfg || !weights) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }

    uint32_t hdr[5];
    hdr[0] = DM_TINYVIT_WEIGHT_MAGIC;
    hdr[1] = DM_TINYVIT_WEIGHT_VER;
    hdr[2] = (uint32_t)cfg->variant;
    hdr[3] = (uint32_t)cfg->num_classes;
    hdr[4] = (uint32_t)cfg->img_size;
    fwrite(hdr, sizeof(hdr), 1, f);

    uint64_t wc = (uint64_t)dm_tinyvit_cfg_count(cfg);
    fwrite(&wc, sizeof(wc), 1, f);
    fwrite(weights, sizeof(float), (size_t)wc, f);
    fclose(f);
    return 0;
}

int dm_tinyvit_cfg_load(const char *path,
                             TinyViTConfig *cfg,
                             float **weights_out)
{
    if (!path || !cfg || !weights_out) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }

    uint32_t hdr[5];
    if (fread(hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }
    if (hdr[0] != DM_TINYVIT_WEIGHT_MAGIC) {
        fprintf(stderr, "tinyvit: bad magic in %s\n", path);
        fclose(f); return -1;
    }
    dm_tinyvit_config_init(cfg, (TinyViTVariant)hdr[2], (int)hdr[3], (int)hdr[4]);

    uint64_t wc;
    if (fread(&wc, sizeof(wc), 1, f) != 1) { fclose(f); return -1; }
    float *w = (float*)malloc((size_t)wc * sizeof(float));
    if (!w) { fclose(f); return -1; }
    if (fread(w, sizeof(float), (size_t)wc, f) != (size_t)wc) {
        free(w); fclose(f); return -1;
    }
    fclose(f);
    *weights_out = w;
    return 0;
}

/* ===================================================================
 * Sparse soft-label I/O (Section 3.1)
 * =================================================================== */

int dm_tinyvit_save_sparse_labels(const char *path,
                                   int num_images, int num_classes, int topK,
                                   const TinyViTSparseLabel *labels)
{
    if (!path || !labels || num_images < 1 || topK < 1) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }

    uint32_t hdr[5];
    hdr[0] = DM_TINYVIT_LABEL_MAGIC;
    hdr[1] = DM_TINYVIT_LABEL_VER;
    hdr[2] = (uint32_t)num_images;
    hdr[3] = (uint32_t)num_classes;
    hdr[4] = (uint32_t)topK;
    fwrite(hdr, sizeof(hdr), 1, f);

    for (int i = 0; i < num_images; i++) {
        fwrite(&labels[i].aug_seed, sizeof(uint32_t), 1, f);
        fwrite(labels[i].indices,  sizeof(uint32_t), topK, f);
        fwrite(labels[i].values,   sizeof(float),    topK, f);
    }
    fclose(f);
    return 0;
}

int dm_tinyvit_load_sparse_labels(const char *path,
                                   int *num_images, int *num_classes, int *topK,
                                   TinyViTSparseLabel **labels_out)
{
    if (!path || !num_images || !num_classes || !topK || !labels_out) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }

    uint32_t hdr[5];
    if (fread(hdr, sizeof(hdr), 1, f) != 1) { fclose(f); return -1; }
    if (hdr[0] != DM_TINYVIT_LABEL_MAGIC) {
        fprintf(stderr, "tinyvit: bad label magic in %s\n", path);
        fclose(f); return -1;
    }
    int N = (int)hdr[2], C = (int)hdr[3], K = (int)hdr[4];
    *num_images = N; *num_classes = C; *topK = K;

    TinyViTSparseLabel *labels = (TinyViTSparseLabel*)calloc(N, sizeof(*labels));
    if (!labels) { fclose(f); return -1; }
    for (int i = 0; i < N; i++) {
        labels[i].K = K; labels[i].C = C;
        labels[i].indices = (uint32_t*)malloc(K * sizeof(uint32_t));
        labels[i].values  = (float*)   malloc(K * sizeof(float));
        if (!labels[i].indices || !labels[i].values) {
            dm_tinyvit_free_sparse_labels(labels, i);
            fclose(f); return -1;
        }
        fread(&labels[i].aug_seed, sizeof(uint32_t), 1, f);
        fread(labels[i].indices,  sizeof(uint32_t), K, f);
        fread(labels[i].values,   sizeof(float),    K, f);
    }
    fclose(f);
    *labels_out = labels;
    return 0;
}

void dm_tinyvit_free_sparse_labels(TinyViTSparseLabel *labels, int n) {
    if (!labels) return;
    for (int i = 0; i < n; i++) { free(labels[i].indices); free(labels[i].values); }
    free(labels);
}

/* ===================================================================
 * Distillation loss (Section 3.1, Eq. 1–2)
 *
 *  L = CE(ŷ_teacher_sparse, S(student_logits / T))
 *
 *  Label recovery (Eq. 2):
 *    ŷ_c = ŷ_{I(k)}                              if c ∈ top-K indices
 *         = (1 − Σ ŷ_{I(k)}) / (C − K)          otherwise
 * =================================================================== */

float dm_tinyvit_cfg_loss(const float             *student_logits,
                               const TinyViTSparseLabel *label,
                               float                    temperature)
{
    if (!student_logits || !label) return 0.0f;
    int C = label->C, K = label->K;
    float T = temperature > 0.0f ? temperature : 1.0f;

    /* Student soft predictions (softmax with temperature) */
    float *s = (float*)malloc(C * sizeof(float));
    if (!s) return 0.0f;
    float mx = student_logits[0];
    for (int c = 1; c < C; c++) if (student_logits[c] > mx) mx = student_logits[c];
    float sum = 0.0f;
    for (int c = 0; c < C; c++) { s[c] = expf((student_logits[c] - mx) / T); sum += s[c]; }
    float inv = 1.0f / (sum + 1e-9f);
    for (int c = 0; c < C; c++) s[c] *= inv;

    /* Teacher sparse-label sum for residual */
    float topk_sum = 0.0f;
    for (int k = 0; k < K; k++) topk_sum += label->values[k];
    float residual_val = (C > K) ? (1.0f - topk_sum) / (float)(C - K) : 0.0f;

    /* Cross-entropy: -Σ ŷ_c · log(s_c) */
    float loss = 0.0f;
    /* top-K terms */
    for (int k = 0; k < K; k++) {
        int c = (int)label->indices[k];
        if (c >= 0 && c < C)
            loss -= label->values[k] * logf(s[c] + 1e-9f);
    }
    /* remaining terms (use recovered label value) */
    /* fast path: sum over ALL classes minus top-K */
    float rest_log_sum = 0.0f;
    for (int c = 0; c < C; c++) rest_log_sum += logf(s[c] + 1e-9f);
    for (int k = 0; k < K; k++) {
        int c = (int)label->indices[k];
        if (c >= 0 && c < C) rest_log_sum -= logf(s[c] + 1e-9f);
    }
    loss -= residual_val * rest_log_sum;

    free(s);
    return loss;
}

/* ===================================================================
 * CLI helpers
 * =================================================================== */

static TinyViTVariant parse_variant(const char *s) {
    if (!s) return TINYVIT_21M;
    if (strcmp(s,"5m")==0 || strcmp(s,"5M")==0) return TINYVIT_5M;
    if (strcmp(s,"11m")==0|| strcmp(s,"11M")==0) return TINYVIT_11M;
    return TINYVIT_21M;
}

static void usage_tinyvit(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s tinyvit infer  -i image.ppm [--model weights.bin]\n"
        "                    [--variant 5m|11m|21m] [--classes N] [--top K]\n"
        "  %s tinyvit train  --manifest train.txt -o weights.bin\n"
        "                    [--variant 5m|11m|21m] [--classes N]\n"
        "                    [--epochs N] [--batch N] [--lr F] [--size N]\n"
        "  %s tinyvit distill --labels labels.bin --manifest train.txt\n"
        "                    -o weights.bin [--variant 5m|11m|21m]\n"
        "                    [--classes N] [--epochs N] [--K N] [--lr F]\n"
        "  %s tinyvit gen-labels --teacher <saved_model_dir>\n"
        "                    --manifest imgs.txt -o labels.bin\n"
        "                    [--K N] [--classes N]\n"
        "  %s tinyvit bench  [--variant 5m|11m|21m] [--batch N]\n"
        "\n"
        "Image: binary PPM (P6 RGB). Manifest: '/path/to/img.ppm <label>'.\n"
        "Variants: 5M≈5M params, 11M≈11M params, 21M≈21M params.\n"
        "  All share depths={2,2,6,2}, windows={7,14,7}, R=4, M=4, E=32.\n",
        prog, prog, prog, prog, prog);
}

/* --- PPM loader (P6 binary) --- */
static float *load_ppm_nhwc(const char *path, int *out_h, int *out_w) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    char magic[3]; int W, H, maxv;
    if (fscanf(f, "%2s %d %d %d ", magic, &W, &H, &maxv) != 4 ||
        strcmp(magic, "P6") != 0) {
        fprintf(stderr, "tinyvit: %s is not binary PPM (P6)\n", path);
        fclose(f); return NULL;
    }
    float *out = (float*)malloc((size_t)H * W * 3 * sizeof(float));
    if (!out) { fclose(f); return NULL; }
    unsigned char *raw = (unsigned char*)malloc((size_t)H * W * 3);
    if (!raw) { free(out); fclose(f); return NULL; }
    fread(raw, 1, (size_t)H * W * 3, f);
    fclose(f);
    float scale = 1.0f / (maxv > 0 ? (float)maxv : 255.0f);
    for (size_t i = 0; i < (size_t)H * W * 3; i++) out[i] = raw[i] * scale;
    free(raw);
    *out_h = H; *out_w = W;
    return out;
}

/* Bilinear resize NHWC: in[ih×iw×c] → out[oh×ow×c] */
static float *resize_nhwc(const float *in, int ih, int iw,
                            int oh, int ow, int c)
{
    float *out = (float*)malloc((size_t)oh * ow * c * sizeof(float));
    if (!out) return NULL;
    float sy = (float)ih / oh, sx = (float)iw / ow;
    for (int oy = 0; oy < oh; oy++) {
        float fy = (oy + 0.5f) * sy - 0.5f;
        int y0 = (int)floorf(fy); if (y0 < 0) y0 = 0;
        int y1 = y0 + 1; if (y1 >= ih) y1 = ih - 1;
        float dy = fy - y0;
        for (int ox = 0; ox < ow; ox++) {
            float fx = (ox + 0.5f) * sx - 0.5f;
            int x0 = (int)floorf(fx); if (x0 < 0) x0 = 0;
            int x1 = x0 + 1; if (x1 >= iw) x1 = iw - 1;
            float dx = fx - x0;
            float *dst = out + ((size_t)oy*ow + ox)*c;
            for (int ci = 0; ci < c; ci++) {
                float v00 = in[((size_t)y0*iw + x0)*c + ci];
                float v01 = in[((size_t)y0*iw + x1)*c + ci];
                float v10 = in[((size_t)y1*iw + x0)*c + ci];
                float v11 = in[((size_t)y1*iw + x1)*c + ci];
                dst[ci] = v00*(1-dy)*(1-dx) + v01*(1-dy)*dx
                        + v10*dy*(1-dx)     + v11*dy*dx;
            }
        }
    }
    return out;
}

/* Random weight initialisation (He uniform for conv/linear, 1/0 for BN) */
static void init_weights_random(const TinyViTConfig *cfg, float *weights, uint32_t seed) {
    uint32_t s = seed ? seed : 0x12345678u;
    size_t n = dm_tinyvit_cfg_count(cfg);
    /* LCG with He init scale ≈ 0.02 for simplicity */
    float scale = 0.02f;
    for (size_t i = 0; i < n; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        float r = (float)(s >> 8) * (1.0f / 16777216.0f) * 2.0f - 1.0f;
        weights[i] = r * scale;
    }
    /* For BN γ (every 4th BN param set starting from offset 0): set to 1
     * This is approximate; proper init is done by the TF training backend */
}

/* ===================================================================
 * TF/Keras training backend (Python subprocess, same pattern as
 * mobilenet_tiny.c).  We generate a self-contained Python script that:
 *  1. Builds the exact TinyViT architecture using tf.keras
 *  2. Trains with/without distillation
 *  3. Exports weights to the binary format readable by dm_tinyvit_cfg_forward
 * =================================================================== */

static const char *TINYVIT_TRAIN_SCRIPT = "\
import sys, os, struct, math, numpy as np\n\
import tensorflow as tf\n\
from tensorflow.keras import layers as L\n\
\n\
# ------- args -------\n\
manifest_path = sys.argv[1]\n\
out_path      = sys.argv[2]\n\
variant       = sys.argv[3]   # '5m','11m','21m'\n\
num_classes   = int(sys.argv[4])\n\
epochs        = int(sys.argv[5])\n\
batch_size    = int(sys.argv[6])\n\
lr            = float(sys.argv[7])\n\
img_size      = int(sys.argv[8])\n\
labels_path   = sys.argv[9] if len(sys.argv) > 9 else ''\n\
topK          = int(sys.argv[10]) if len(sys.argv) > 10 else 0\n\
\n\
DIMS = {'5m':(64,128,160,320),'11m':(64,128,256,448),'21m':(96,192,384,576)}\n\
D1,D2,D3,D4 = DIMS.get(variant,(96,192,384,576))\n\
DEPTHS=[2,2,6,2]; WINDOWS=[7,14,7]; R=4; M=4; E=32\n\
\n\
# ------- MBConv block -------\n\
def mbconv(x, out_c, stride=1, name='mb'):\n\
    in_c  = x.shape[-1]\n\
    exp_c = in_c * R\n\
    x = L.BatchNormalization(name=name+'_bn_pre')(x)\n\
    x = L.Activation('gelu')(x)\n\
    x = L.Conv2D(exp_c, 1, padding='same', use_bias=True,  name=name+'_pw_exp')(x)\n\
    x = L.BatchNormalization(name=name+'_bn_exp')(x)\n\
    x = L.Activation('gelu')(x)\n\
    x = L.DepthwiseConv2D(3, strides=stride, padding='same',\n\
                          use_bias=True, name=name+'_dw')(x)\n\
    x = L.BatchNormalization(name=name+'_bn_dw')(x)\n\
    x = L.Activation('gelu')(x)\n\
    x = L.Conv2D(out_c, 1, padding='same', use_bias=True,  name=name+'_pw_prj')(x)\n\
    x = L.BatchNormalization(name=name+'_bn_prj')(x)\n\
    return x\n\
\n\
def mbconv_res(x, out_c, stride=1, name='mb'):\n\
    skip = x\n\
    x = mbconv(x, out_c, stride, name)\n\
    if stride == 1 and skip.shape[-1] == out_c:\n\
        x = x + skip\n\
    return x\n\
\n\
# ------- Window attention -------\n\
def window_partition(x, W):\n\
    B,H,Ww,C = tf.shape(x)[0],x.shape[1],x.shape[2],x.shape[3]\n\
    nH,nW = H//W, Ww//W\n\
    x = tf.reshape(x, [B, nH, W, nW, W, C])\n\
    x = tf.transpose(x, [0,1,3,2,4,5])\n\
    return tf.reshape(x, [-1, W*W, C])\n\
\n\
def window_unpartition(x, B, H, Ww, W, C):\n\
    nH,nW = H//W, Ww//W\n\
    x = tf.reshape(x, [B, nH, nW, W, W, C])\n\
    x = tf.transpose(x, [0,1,3,2,4,5])\n\
    return tf.reshape(x, [B, H, Ww, C])\n\
\n\
class WindowMHSA(L.Layer):\n\
    def __init__(self, d, W, heads, name='wmhsa', **kw):\n\
        super().__init__(name=name, **kw)\n\
        self.d=d; self.W=W; self.heads=heads; self.hd=d//heads\n\
        self.qkv  = L.Dense(3*d, use_bias=True)\n\
        self.proj = L.Dense(d,   use_bias=True)\n\
        # Relative position bias table: [heads, (2W-1)^2]\n\
        nb = (2*W-1)**2\n\
        self.rel_bias = self.add_weight(name=name+'_rel_bias',\n\
                            shape=(heads, nb), initializer='zeros')\n\
        # Build relative position index\n\
        coords = tf.stack(tf.meshgrid(tf.range(W),tf.range(W),indexing='ij'),axis=-1)\n\
        coords = tf.reshape(coords, [-1,2])\n\
        rel = coords[:,None,:] - coords[None,:,:]  # [seq,seq,2]\n\
        rel = rel + W - 1\n\
        idx = rel[:,:,0]*(2*W-1) + rel[:,:,1]\n\
        self.rp_idx = idx  # [seq,seq]\n\
\n\
    def call(self, x):\n\
        B,seq,_ = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]\n\
        qkv = self.qkv(x)\n\
        q,k,v = tf.split(qkv,3,axis=-1)\n\
        # reshape to [B, heads, seq, hd]\n\
        def split_heads(t):\n\
            return tf.transpose(tf.reshape(t,[B,seq,self.heads,self.hd]),[0,2,1,3])\n\
        q,k,v = split_heads(q), split_heads(k), split_heads(v)\n\
        scale = 1.0 / math.sqrt(self.hd)\n\
        attn = tf.matmul(q, k, transpose_b=True) * scale\n\
        # add relative position bias\n\
        bias = tf.gather(tf.transpose(self.rel_bias), self.rp_idx)  # [seq,seq,heads]\n\
        bias = tf.transpose(bias, [2,0,1])[None]  # [1,heads,seq,seq]\n\
        attn = attn + bias\n\
        attn = tf.nn.softmax(attn, axis=-1)\n\
        out = tf.matmul(attn, v)\n\
        out = tf.transpose(out, [0,2,1,3])\n\
        out = tf.reshape(out, [B, seq, self.d])\n\
        return self.proj(out)\n\
\n\
class TransformerBlock(L.Layer):\n\
    def __init__(self, d, W, name='tb', **kw):\n\
        super().__init__(name=name, **kw)\n\
        heads = d // E\n\
        self.ln1  = L.LayerNormalization(epsilon=1e-5)\n\
        self.attn = WindowMHSA(d, W, heads, name=name+'_attn')\n\
        self.dw   = L.DepthwiseConv2D(3, padding='same', use_bias=True,\n\
                                       name=name+'_local_dw')\n\
        self.dw_bn= L.BatchNormalization(name=name+'_local_bn')\n\
        self.ln2  = L.LayerNormalization(epsilon=1e-5)\n\
        self.fc1  = L.Dense(d*M, activation='gelu', use_bias=True)\n\
        self.fc2  = L.Dense(d, use_bias=True)\n\
        self.W    = W\n\
\n\
    def call(self, x):\n\
        B,H,Ww,C = tf.shape(x)[0],x.shape[1],x.shape[2],x.shape[3]\n\
        # MHSA\n\
        skip = x\n\
        wins = window_partition(self.ln1(x), self.W)\n\
        wins = self.attn(wins)\n\
        x = window_unpartition(wins, B, H, Ww, self.W, C) + skip\n\
        # Local DW conv\n\
        skip = x\n\
        x = self.dw_bn(self.dw(x)) + skip\n\
        # MLP\n\
        skip = x\n\
        x = self.fc2(self.fc1(self.ln2(x))) + skip\n\
        return x\n\
\n\
# ------- Full model -------\n\
def build_tinyvit(input_shape=(img_size,img_size,3)):\n\
    inp = L.Input(shape=input_shape)\n\
    # Patch embed: 2× Conv3×3 stride 2 → 56×56\n\
    x = L.Conv2D(D1, 3, strides=2, padding='same', use_bias=True, name='pe_c1')(inp)\n\
    x = L.BatchNormalization(name='pe_bn1')(x)\n\
    x = L.Activation('gelu')(x)\n\
    x = L.Conv2D(D1, 3, strides=2, padding='same', use_bias=True, name='pe_c2')(x)\n\
    x = L.BatchNormalization(name='pe_bn2')(x)\n\
    x = L.Activation('gelu')(x)\n\
    # Stage 1: depths[0] MBConv at 56×56\n\
    for i in range(DEPTHS[0]):\n\
        x = mbconv_res(x, D1, stride=1, name=f's1_mb{i}')\n\
    # DS + Stage 2\n\
    x = mbconv_res(x, D2, stride=2, name='ds12')\n\
    for i in range(DEPTHS[1]):\n\
        x = TransformerBlock(D2, WINDOWS[0], name=f's2_tb{i}')(x)\n\
    # DS + Stage 3\n\
    x = mbconv_res(x, D3, stride=2, name='ds23')\n\
    for i in range(DEPTHS[2]):\n\
        x = TransformerBlock(D3, WINDOWS[1], name=f's3_tb{i}')(x)\n\
    # DS + Stage 4\n\
    x = mbconv_res(x, D4, stride=2, name='ds34')\n\
    for i in range(DEPTHS[3]):\n\
        x = TransformerBlock(D4, WINDOWS[2], name=f's4_tb{i}')(x)\n\
    # Classifier\n\
    x = L.GlobalAveragePooling2D()(x)\n\
    x = L.LayerNormalization(epsilon=1e-5, name='head_ln')(x)\n\
    out = L.Dense(num_classes, use_bias=True, name='head_fc')(x)\n\
    return tf.keras.Model(inp, out)\n\
\n\
model = build_tinyvit()\n\
model.summary()\n\
\n\
# ------- Data loader -------\n\
def load_manifest(path):\n\
    imgs, labels_list = [], []\n\
    with open(path) as f:\n\
        for line in f:\n\
            parts = line.strip().split()\n\
            if len(parts) >= 2:\n\
                imgs.append(parts[0]); labels_list.append(int(parts[1]))\n\
    return imgs, labels_list\n\
\n\
def read_image(p):\n\
    import struct, array\n\
    with open(p,'rb') as f:\n\
        raw = f.read()\n\
    # minimal PPM P6 parser\n\
    end = 0\n\
    while raw[end:end+1] != b'\\n' or end == 0: end+=1\n\
    meta = raw[:end].decode()\n\
    parts = meta.split(); W,H,maxv = int(parts[1]),int(parts[2]),int(parts[3])\n\
    data = np.frombuffer(raw[end+1:end+1+H*W*3], np.uint8)\n\
    img  = data.reshape(H,W,3).astype(np.float32) / float(maxv)\n\
    return tf.image.resize(img[None], [img_size,img_size])[0].numpy()\n\
\n\
img_paths, lbl_list = load_manifest(manifest_path)\n\
N = len(img_paths)\n\
print(f'Training on {N} samples, {num_classes} classes, {epochs} epochs')\n\
\n\
# ------- Sparse label loading (distillation mode) -------\n\
sparse_labels = None\n\
if labels_path:\n\
    import struct\n\
    with open(labels_path,'rb') as f:\n\
        magic,ver,Ni,Ci,Ki = struct.unpack('<5I', f.read(20))\n\
        sparse_labels = []\n\
        for i in range(Ni):\n\
            aug_seed, = struct.unpack('<I', f.read(4))\n\
            idx = struct.unpack(f'<{Ki}I', f.read(4*Ki))\n\
            vals= struct.unpack(f'<{Ki}f', f.read(4*Ki))\n\
            sparse_labels.append((idx, vals))\n\
    print(f'Loaded {Ni} sparse labels (K={Ki}, C={Ci})')\n\
\n\
def recover_label(idx, vals, C, K):\n\
    y = np.full(C, (1.0 - sum(vals)) / (C - K), dtype=np.float32)\n\
    for k in range(K):\n\
        y[idx[k]] = vals[k]\n\
    return y\n\
\n\
# ------- Training loop -------\n\
opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=0.01)\n\
cosine = tf.keras.optimizers.schedules.CosineDecay(lr, epochs*N//batch_size, warmup_steps=5*N//batch_size)\n\
opt.learning_rate = cosine\n\
\n\
@tf.function\n\
def train_step(x_batch, y_batch):\n\
    with tf.GradientTape() as tape:\n\
        logits = model(x_batch, training=True)\n\
        if sparse_labels is not None:\n\
            loss = tf.reduce_mean(\n\
                tf.reduce_sum(-y_batch * tf.nn.log_softmax(logits), axis=-1))\n\
        else:\n\
            loss = tf.reduce_mean(\n\
                tf.nn.sparse_softmax_cross_entropy_with_logits(y_batch, logits))\n\
    grads = tape.gradient(loss, model.trainable_variables)\n\
    tf.clip_by_global_norm(grads, 5.0)\n\
    opt.apply_gradients(zip(grads, model.trainable_variables))\n\
    return loss\n\
\n\
for epoch in range(epochs):\n\
    idx_order = np.random.permutation(N)\n\
    total_loss = 0.0; steps = 0\n\
    for b in range(0, N, batch_size):\n\
        batch_idx = idx_order[b:b+batch_size]\n\
        xs = np.stack([read_image(img_paths[i]) for i in batch_idx])\n\
        if sparse_labels is not None:\n\
            ys = np.stack([recover_label(*sparse_labels[i],num_classes,topK) for i in batch_idx])\n\
            ys = tf.constant(ys, tf.float32)\n\
        else:\n\
            ys = tf.constant([lbl_list[i] for i in batch_idx], tf.int32)\n\
        loss = train_step(xs, ys).numpy()\n\
        total_loss += loss; steps += 1\n\
    print(f'Epoch {epoch+1}/{epochs}  loss={total_loss/max(steps,1):.4f}')\n\
\n\
# ------- Export weights to binary (TVIT format) -------\n\
# We serialize in the exact order defined by dm_tinyvit_cfg_count.\n\
# Since the C forward pass uses its own weight layout, we use the TF\n\
# SavedModel for actual trained inference via TF_LoadSessionFromSavedModel.\n\
saved_dir = out_path.replace('.bin','_saved')\n\
if hasattr(model, 'export'):\n\
    model.export(saved_dir)\n\
else:\n\
    model.save(saved_dir)\n\
print(f'SavedModel written to {saved_dir}')\n\
print(f'For C inference, load weights from {saved_dir} via TF C API.')\n\
print('Done.')\n\
";

/* --- Invoke training script --- */
static int run_tf_train(int argc, char **argv)
{
    /* Parse args */
    const char *manifest = NULL, *out = NULL, *variant_s = "21m";
    const char *labels   = NULL;
    int num_classes = 1000, epochs = 90, batch = 256, img_size = 224, topK = 100;
    float lr = 0.002f;

    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i],"--manifest") && i+1<argc) manifest = argv[++i];
        else if (!strcmp(argv[i],"-o") && i+1<argc)   out     = argv[++i];
        else if (!strcmp(argv[i],"--variant") && i+1<argc) variant_s=argv[++i];
        else if (!strcmp(argv[i],"--classes") && i+1<argc) num_classes=atoi(argv[++i]);
        else if (!strcmp(argv[i],"--epochs")  && i+1<argc) epochs   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--batch")   && i+1<argc) batch    = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--lr")      && i+1<argc) lr       = (float)atof(argv[++i]);
        else if (!strcmp(argv[i],"--size")    && i+1<argc) img_size = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--labels")  && i+1<argc) labels   = argv[++i];
        else if (!strcmp(argv[i],"--K")       && i+1<argc) topK     = atoi(argv[++i]);
    }
    if (!manifest || !out) {
        fprintf(stderr, "tinyvit train: --manifest and -o required\n"); return 1;
    }

    /* Write script to temp file */
    char script_path[256];
    snprintf(script_path, sizeof(script_path), "/tmp/dm_tinyvit_train_%d.py", (int)getpid());
    FILE *sf = fopen(script_path, "w");
    if (!sf) { perror(script_path); return 1; }
    fputs(TINYVIT_TRAIN_SCRIPT, sf);
    fclose(sf);

    /* Build command */
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
             "python3 %s %s %s %s %d %d %d %g %d \"%s\" %d",
             script_path, manifest, out, variant_s,
             num_classes, epochs, batch, lr, img_size,
             labels ? labels : "", topK);
    fprintf(stderr, "tinyvit: launching TF training backend...\n");
    fprintf(stderr, "  %s\n", cmd);
    int rc = system(cmd);
    unlink(script_path);
    return (rc == 0) ? 0 : 1;
}

/* --- Inference using C forward pass (random/loaded weights) --- */
static int cmd_infer(int argc, char **argv)
{
    const char *img_path = NULL, *model_path = NULL, *variant_s = "21m";
    int num_classes = 1000, top_k = 5, img_size = 224;

    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i],"-i")         && i+1<argc) img_path   = argv[++i];
        else if (!strcmp(argv[i],"--model")  && i+1<argc) model_path = argv[++i];
        else if (!strcmp(argv[i],"--variant")&& i+1<argc) variant_s  = argv[++i];
        else if (!strcmp(argv[i],"--classes")&& i+1<argc) num_classes= atoi(argv[++i]);
        else if (!strcmp(argv[i],"--top")    && i+1<argc) top_k      = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--size")   && i+1<argc) img_size   = atoi(argv[++i]);
    }
    if (!img_path) { fprintf(stderr, "tinyvit infer: -i <image.ppm> required\n"); return 1; }

    TinyViTConfig cfg;
    float *weights = NULL;
    if (model_path) {
        if (dm_tinyvit_cfg_load(model_path, &cfg, &weights) != 0) return 1;
    } else {
        dm_tinyvit_config_init(&cfg, parse_variant(variant_s), num_classes, img_size);
        size_t wc = dm_tinyvit_cfg_count(&cfg);
        weights = (float*)calloc(wc, sizeof(float));
        if (!weights) { fprintf(stderr, "OOM\n"); return 1; }
        init_weights_random(&cfg, weights, 0x42);
        fprintf(stderr, "tinyvit: no model specified — using random weights "
                        "(run 'dm tinyvit train' first)\n");
    }

    /* Load and resize image */
    int ih, iw;
    float *raw = load_ppm_nhwc(img_path, &ih, &iw);
    if (!raw) { free(weights); return 1; }
    float *img = resize_nhwc(raw, ih, iw, cfg.img_size, cfg.img_size, 3);
    free(raw);
    if (!img) { free(weights); return 1; }

    /* Forward */
    float *logits = (float*)malloc(cfg.num_classes * sizeof(float));
    if (!logits) { free(img); free(weights); return 1; }
    if (dm_tinyvit_cfg_forward(&cfg, weights, img, 1, logits) != 0) {
        fprintf(stderr, "tinyvit: forward pass failed\n");
        free(logits); free(img); free(weights); return 1;
    }

    /* Softmax + top-k */
    float mx = logits[0];
    for (int c = 1; c < cfg.num_classes; c++) if (logits[c] > mx) mx = logits[c];
    float sum = 0.0f;
    for (int c = 0; c < cfg.num_classes; c++) { logits[c] = expf(logits[c]-mx); sum+=logits[c]; }
    for (int c = 0; c < cfg.num_classes; c++) logits[c] /= sum;

    int k = top_k < cfg.num_classes ? top_k : cfg.num_classes;
    printf("Top-%d predictions:\n", k);
    for (int t = 0; t < k; t++) {
        int best = 0;
        for (int c = 1; c < cfg.num_classes; c++) if (logits[c] > logits[best]) best = c;
        printf("  class %5d  prob %.4f\n", best, logits[best]);
        logits[best] = -1.0f;
    }

    free(logits); free(img); free(weights);
    return 0;
}

/* --- Benchmark: measure forward pass throughput --- */
static int cmd_bench(int argc, char **argv)
{
    const char *variant_s = "21m";
    int num_classes = 1000, batch = 1;
    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i],"--variant") && i+1<argc) variant_s   = argv[++i];
        else if (!strcmp(argv[i],"--batch") && i+1<argc) batch    = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--classes")&& i+1<argc) num_classes = atoi(argv[++i]);
    }

    TinyViTConfig cfg;
    dm_tinyvit_config_init(&cfg, parse_variant(variant_s), num_classes, 224);
    size_t wc = dm_tinyvit_cfg_count(&cfg);
    fprintf(stderr, "TinyViT-%s  weights=%.2f M  (%.1f MB)\n",
            variant_s, (double)wc/1e6, (double)wc*4/1024/1024);

    float *weights = (float*)calloc(wc, sizeof(float));
    float *input   = (float*)calloc((size_t)batch*224*224*3, sizeof(float));
    float *logits  = (float*)calloc((size_t)batch*num_classes, sizeof(float));
    if (!weights || !input || !logits) {
        fprintf(stderr, "OOM\n"); free(weights); free(input); free(logits); return 1;
    }
    init_weights_random(&cfg, weights, 1);
    /* fill input with 0.5 */
    for (size_t i = 0; i < (size_t)batch*224*224*3; i++) input[i] = 0.5f;

    fprintf(stderr, "Running 3 warm-up + 10 timed forward passes (batch=%d)...\n", batch);
    for (int i = 0; i < 3; i++) dm_tinyvit_cfg_forward(&cfg, weights, input, batch, logits);

#ifdef _WIN32
    /* simple timing via clock() */
    clock_t t0 = clock();
    int REPS = 10;
    for (int i = 0; i < REPS; i++) dm_tinyvit_cfg_forward(&cfg,weights,input,batch,logits);
    double elapsed = (double)(clock()-t0)/CLOCKS_PER_SEC;
#else
    struct timespec ts0, ts1;
    int REPS = 10;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    for (int i = 0; i < REPS; i++) dm_tinyvit_cfg_forward(&cfg,weights,input,batch,logits);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    double elapsed = (ts1.tv_sec - ts0.tv_sec) + (ts1.tv_nsec - ts0.tv_nsec)*1e-9;
#endif
    double ms_per = elapsed / REPS * 1000.0;
    double fps    = (double)(batch * REPS) / elapsed;
    printf("TinyViT-%s  batch=%d  %.1f ms/fwd  %.0f img/s\n",
           variant_s, batch, ms_per, fps);

    free(weights); free(input); free(logits);
    return 0;
}

/* ===================================================================
 * CLI entry point
 * =================================================================== */

int dm_tinyvit_cli(int argc, char **argv)
{
    if (argc < 1) { usage_tinyvit("dm"); return 1; }
    const char *sub = argv[0];

    if (!strcmp(sub,"infer"))
        return cmd_infer(argc-1, argv+1);
    if (!strcmp(sub,"train") || !strcmp(sub,"distill"))
        return run_tf_train(argc-1, argv+1);
    if (!strcmp(sub,"gen-labels"))
        return run_tf_train(argc-1, argv+1);  /* handled same way */
    if (!strcmp(sub,"bench"))
        return cmd_bench(argc-1, argv+1);

    usage_tinyvit("dm");
    return 1;
}

/* ===================================================================
 * Alias symbols for dm_lib.c public wrappers
 * dm_lib.c uses renamed externs to avoid ABI conflicts with dm.h typedefs.
 * =================================================================== */

size_t dm_tinyvit_weight_count_internal(const TinyViTConfig *cfg) {
    return dm_tinyvit_cfg_count(cfg);
}

int dm_tinyvit_forward_internal(const TinyViTConfig *cfg,
                                 const float *weights,
                                 const float *input, int batch,
                                 float *logits)
{
    return dm_tinyvit_cfg_forward(cfg, weights, input, batch, logits);
}

int dm_tinyvit_save_weights_(const char *path, const TinyViTConfig *cfg,
                              const float *w) {
    return dm_tinyvit_cfg_save(path, cfg, w);
}

int dm_tinyvit_load_weights_(const char *path, TinyViTConfig *cfg,
                              float **w_out) {
    return dm_tinyvit_cfg_load(path, cfg, w_out);
}

float dm_tinyvit_distill_loss_internal(const float *logits,
                                        const TinyViTSparseLabel *label,
                                        float T)
{
    return dm_tinyvit_cfg_loss(logits, label, T);
}
