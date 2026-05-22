/*
 * transformer.c — "Attention Is All You Need" (Vaswani et al., NeurIPS 2017)
 *
 * Pure C99 implementation of the full Transformer encoder-decoder.
 * Each primitive is a standalone function: sdp_attention, mha, ffn,
 * layer_norm, positional_encoding, encoder_layer, decoder_layer.
 *
 * Training is delegated to a TF/Keras Python subprocess.
 *
 * Weight layout mirrors the order in transformer.h (sections A → D).
 *
 * Internal helpers follow the WBuf cursor pattern:
 *   • When wb.data == NULL  →  only accumulate wb.pos (weight counting mode)
 *   • When wb.data != NULL  →  advance through the weight array (forward mode)
 * This guarantees counting and forward always stay in sync.
 */

/* The Makefile sets CFLAGS with -I$(PROJECT_ROOT)/include so this resolves
 * to include/models/lm/transformer.h from the project root.              */
#include "models/lm/transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>

#ifndef _WIN32
#  include <sys/time.h>
#  include <time.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Internal: weight-buffer cursor
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct {
    const float *data; /* NULL in counting mode */
    size_t       pos;  /* current offset (floats) */
} WBuf;

/* Advance by n floats and return pointer to the first.
 * In counting mode (data == NULL) just increments pos and returns NULL. */
static const float *wb_next(WBuf *b, size_t n) {
    const float *p = b->data ? b->data + b->pos : NULL;
    b->pos += n;
    return p;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Math helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ReLU: max(0, x) — used in FFN */
static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

/* In-place row-wise softmax over float[n] */
static void softmax_inplace(float *x, int n) {
    float mx = -FLT_MAX;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    float inv_s = 1.0f / s;
    for (int i = 0; i < n; i++) x[i] *= inv_s;
}

/* Matrix multiply C = A * B  (row-major)
 *   A: [m × k],  B: [k × n],  C: [m × n]  */
static void matmul(const float *A, const float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int p = 0; p < k; p++) acc += (double)A[i*k+p] * B[p*n+j];
            C[i*n+j] = (float)acc;
        }
    }
}

/* Linear transform:  out[i] = x[i] * W^T + b
 *   x  : [rows × in_dim]
 *   W  : [out_dim × in_dim]   (standard weight layout: each row is one neuron)
 *   b  : [out_dim]  or NULL
 *   out: [rows × out_dim]   */
static void linear(const float *x, const float *W, const float *b,
                   int rows, int in_dim, int out_dim, float *out) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + r * in_dim;
        float       *or_ = out + r * out_dim;
        for (int j = 0; j < out_dim; j++) {
            double acc = b ? (double)b[j] : 0.0;
            for (int i = 0; i < in_dim; i++) acc += (double)xr[i] * W[j*in_dim+i];
            or_[j] = (float)acc;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.5  Positional Encoding  (sinusoidal)
 *
 *   PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

void dm_transformer_positional_encoding(int max_len, int d_model, float *pe_out) {
    for (int pos = 0; pos < max_len; pos++) {
        float *row = pe_out + pos * d_model;
        for (int i = 0; i < d_model / 2; i++) {
            /* exponent: 2i / d_model */
            double denom = pow(10000.0, (2.0 * i) / d_model);
            row[2*i]   = (float)sin(pos / denom);   /* even dimension */
            row[2*i+1] = (float)cos(pos / denom);   /* odd  dimension */
        }
        /* If d_model is odd, fill the last dimension with sin */
        if (d_model & 1) {
            int i = d_model / 2;
            double denom = pow(10000.0, (2.0 * i) / d_model);
            row[d_model-1] = (float)sin(pos / denom);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.1  Layer Normalization  (Ba et al. 2016)
 * ═══════════════════════════════════════════════════════════════════════════ */

void dm_transformer_layer_norm(float       *x,
                                const float *gamma,
                                const float *beta,
                                int          n_rows,
                                int          d,
                                float        eps) {
    for (int r = 0; r < n_rows; r++) {
        float *row = x + r * d;

        /* mean */
        double mean = 0.0;
        for (int i = 0; i < d; i++) mean += row[i];
        mean /= d;

        /* variance */
        double var = 0.0;
        for (int i = 0; i < d; i++) {
            double diff = row[i] - mean;
            var += diff * diff;
        }
        var /= d;

        float inv_std = 1.0f / sqrtf((float)var + eps);

        /* normalise + scale + shift */
        for (int i = 0; i < d; i++) {
            float norm = ((float)(row[i] - mean)) * inv_std;
            row[i] = norm * (gamma ? gamma[i] : 1.0f)
                          + (beta  ? beta[i]  : 0.0f);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.2.1  Scaled Dot-Product Attention  (Eq. 1)
 *
 *   Attention(Q, K, V) = softmax( QK^T / √d_k ) V
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

void dm_transformer_causal_mask(int seq, float *mask_out) {
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < seq; j++)
            mask_out[i*seq + j] = (j <= i) ? 0.0f : -1e9f;
}

int dm_transformer_sdp_attention(const float *Q,
                                  const float *K,
                                  const float *V,
                                  const float *mask,
                                  int          seq_q,
                                  int          seq_k,
                                  int          d_k,
                                  int          d_v,
                                  float       *out,
                                  float       *scratch) {
    /* scratch: float[seq_q × seq_k] — reused as the scores/weights matrix */
    float scale = 1.0f / sqrtf((float)d_k);

    /* Step 1: scores = Q K^T / √d_k */
    matmul(Q, K, scratch, seq_q, d_k, seq_k);   /* scratch = Q * K^T wrong */
    /*
     * matmul expects B to be [k × n].  K is [seq_k × d_k], we want Q*K^T.
     * That means B = K^T, shape [d_k × seq_k].
     * Easier: compute manually with the transpose.
     */
    for (int i = 0; i < seq_q; i++) {
        for (int j = 0; j < seq_k; j++) {
            double s = 0.0;
            for (int p = 0; p < d_k; p++)
                s += (double)Q[i*d_k+p] * K[j*d_k+p];
            scratch[i*seq_k+j] = (float)(s * scale);
        }
    }

    /* Step 2: apply optional mask */
    if (mask) {
        for (int i = 0; i < seq_q * seq_k; i++)
            scratch[i] += mask[i];
    }

    /* Step 3: softmax over keys dimension (row-wise) */
    for (int i = 0; i < seq_q; i++)
        softmax_inplace(scratch + i * seq_k, seq_k);

    /* Step 4: out = weights * V  [seq_q × d_v] */
    matmul(scratch, V, out, seq_q, seq_k, d_v);

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.2.2  Multi-Head Attention  (Eq. 2)
 *
 *   MultiHead(Q, K, V) = Concat(head_1,...,head_h) W^O
 *   head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
 *
 *   d_k = d_v = d_model / h  (enforced in config)
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_transformer_mha(const float *W_Q, const float *b_Q,
                        const float *W_K, const float *b_K,
                        const float *W_V, const float *b_V,
                        const float *W_O, const float *b_O,
                        const float *Q_in,
                        const float *K_in,
                        const float *V_in,
                        const float *mask,
                        int          seq_q,
                        int          seq_k,
                        int          d_model,
                        int          h,
                        float       *out) {
    int d_k = d_model / h;  /* = d_v (from paper: d_k = d_v = d_model/h) */
    int d_v = d_k;

    /* Allocate scratch buffers */
    float *Q_proj    = malloc((size_t)seq_q * d_model * sizeof(float));
    float *K_proj    = malloc((size_t)seq_k * d_model * sizeof(float));
    float *V_proj    = malloc((size_t)seq_k * d_model * sizeof(float));
    float *head_out  = malloc((size_t)seq_q * d_model * sizeof(float));
    float *attn_out  = malloc((size_t)seq_q * d_v     * sizeof(float));
    float *attn_scr  = malloc((size_t)seq_q * seq_k   * sizeof(float));
    if (!Q_proj || !K_proj || !V_proj || !head_out || !attn_out || !attn_scr) {
        free(Q_proj); free(K_proj); free(V_proj);
        free(head_out); free(attn_out); free(attn_scr);
        return -1;
    }

    /* Project: Q_proj = Q_in * W_Q^T + b_Q  [seq_q × d_model]
     * W_Q layout: [d_model × d_model]  (stacks all h heads)             */
    linear(Q_in, W_Q, b_Q, seq_q, d_model, d_model, Q_proj);
    linear(K_in, W_K, b_K, seq_k, d_model, d_model, K_proj);
    linear(V_in, W_V, b_V, seq_k, d_model, d_model, V_proj);

    /* Process each head independently */
    for (int i = 0; i < h; i++) {
        int q_off = i * d_k;  /* column offset in Q_proj for head i */
        int k_off = i * d_k;
        int v_off = i * d_v;

        /* Extract head-i slices into contiguous arrays */
        float *Qi = malloc((size_t)seq_q * d_k * sizeof(float));
        float *Ki = malloc((size_t)seq_k * d_k * sizeof(float));
        float *Vi = malloc((size_t)seq_k * d_v * sizeof(float));
        if (!Qi || !Ki || !Vi) {
            free(Qi); free(Ki); free(Vi);
            free(Q_proj); free(K_proj); free(V_proj);
            free(head_out); free(attn_out); free(attn_scr);
            return -1;
        }
        for (int t = 0; t < seq_q; t++)
            memcpy(Qi + t*d_k, Q_proj + t*d_model + q_off, d_k * sizeof(float));
        for (int t = 0; t < seq_k; t++) {
            memcpy(Ki + t*d_k, K_proj + t*d_model + k_off, d_k * sizeof(float));
            memcpy(Vi + t*d_v, V_proj + t*d_model + v_off, d_v * sizeof(float));
        }

        /* Scaled dot-product attention for head i */
        dm_transformer_sdp_attention(Qi, Ki, Vi, mask, seq_q, seq_k,
                                      d_k, d_v, attn_out, attn_scr);

        /* Write attn_out into head_out at the correct column offset */
        for (int t = 0; t < seq_q; t++)
            memcpy(head_out + t*d_model + i*d_v, attn_out + t*d_v,
                   d_v * sizeof(float));

        free(Qi); free(Ki); free(Vi);
    }

    /* Output projection: out = head_out * W_O^T + b_O  [seq_q × d_model] */
    linear(head_out, W_O, b_O, seq_q, d_model, d_model, out);

    free(Q_proj); free(K_proj); free(V_proj);
    free(head_out); free(attn_out); free(attn_scr);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.3  Position-wise Feed-Forward Network  (Eq. 2)
 *
 *   FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_transformer_ffn(const float *x,
                        const float *W_1, const float *b_1,
                        const float *W_2, const float *b_2,
                        int          seq,
                        int          d_model,
                        int          d_ff,
                        float       *out) {
    float *hidden = malloc((size_t)seq * d_ff * sizeof(float));
    if (!hidden) return -1;

    /* hidden = ReLU(x W_1 + b_1)  [seq × d_ff] */
    linear(x, W_1, b_1, seq, d_model, d_ff, hidden);
    for (int i = 0; i < seq * d_ff; i++) hidden[i] = relu(hidden[i]);

    /* out = hidden W_2 + b_2  [seq × d_model] */
    linear(hidden, W_2, b_2, seq, d_ff, d_model, out);

    free(hidden);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Weight layout helpers
 *
 * These functions read weights out of a WBuf in the documented order.
 * In counting mode (wb.data == NULL) they just advance the position.
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Weight block for one MHA layer (4 weight matrices + 4 bias vectors). */
typedef struct {
    const float *W_Q, *b_Q;   /* [d_model × d_model], [d_model] */
    const float *W_K, *b_K;
    const float *W_V, *b_V;
    const float *W_O, *b_O;
} MHAWeights;

/* Weight block for one FFN layer (2 weight matrices + 2 bias vectors). */
typedef struct {
    const float *W_1, *b_1;   /* [d_model × d_ff],  [d_ff]    */
    const float *W_2, *b_2;   /* [d_ff    × d_model],[d_model] */
} FFNWeights;

/* Weight block for one LayerNorm (gamma + beta). */
typedef struct {
    const float *gamma;  /* [d_model] */
    const float *beta;   /* [d_model] */
} LNWeights;

static MHAWeights read_mha_weights(WBuf *wb, int d_model) {
    MHAWeights w;
    w.W_Q = wb_next(wb, (size_t)d_model * d_model);
    w.b_Q = wb_next(wb, (size_t)d_model);
    w.W_K = wb_next(wb, (size_t)d_model * d_model);
    w.b_K = wb_next(wb, (size_t)d_model);
    w.W_V = wb_next(wb, (size_t)d_model * d_model);
    w.b_V = wb_next(wb, (size_t)d_model);
    w.W_O = wb_next(wb, (size_t)d_model * d_model);
    w.b_O = wb_next(wb, (size_t)d_model);
    return w;
}

static FFNWeights read_ffn_weights(WBuf *wb, int d_model, int d_ff) {
    FFNWeights w;
    w.W_1 = wb_next(wb, (size_t)d_model * d_ff);
    w.b_1 = wb_next(wb, (size_t)d_ff);
    w.W_2 = wb_next(wb, (size_t)d_ff    * d_model);
    w.b_2 = wb_next(wb, (size_t)d_model);
    return w;
}

static LNWeights read_ln_weights(WBuf *wb, int d_model) {
    LNWeights w;
    w.gamma = wb_next(wb, (size_t)d_model);
    w.beta  = wb_next(wb, (size_t)d_model);
    return w;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */

void dm_transformer_config_init(TransformerConfig *cfg,
                                 TransformerVariant  variant,
                                 int                 vocab_size,
                                 int                 max_seq_len) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->variant     = variant;
    cfg->vocab_size  = vocab_size;
    cfg->max_seq_len = max_seq_len > 0 ? max_seq_len : 512;

    if (variant == TRANSFORMER_BIG) {
        /* Table 3, row "big" */
        cfg->num_layers   = 6;
        cfg->d_model      = 1024;
        cfg->d_ff         = 4096;
        cfg->num_heads    = 16;
        cfg->d_k          = 64;    /* = d_model / h */
        cfg->dropout      = 0.3f;
        cfg->label_smooth = 0.1f;
        cfg->warmup_steps = 4000;
    } else {
        /* Table 3, row "base" (default) */
        cfg->num_layers   = 6;
        cfg->d_model      = 512;
        cfg->d_ff         = 2048;
        cfg->num_heads    = 8;
        cfg->d_k          = 64;    /* = d_model / h */
        cfg->dropout      = 0.1f;
        cfg->label_smooth = 0.1f;
        cfg->warmup_steps = 4000;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Weight counting
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Count weights for one encoder layer via WBuf. */
static void count_encoder_layer(WBuf *wb, const TransformerConfig *cfg) {
    read_mha_weights(wb, cfg->d_model);   /* self-attention */
    read_ln_weights (wb, cfg->d_model);   /* LN_1 */
    read_ffn_weights(wb, cfg->d_model, cfg->d_ff);
    read_ln_weights (wb, cfg->d_model);   /* LN_2 */
}

/* Count weights for one decoder layer via WBuf. */
static void count_decoder_layer(WBuf *wb, const TransformerConfig *cfg) {
    read_mha_weights(wb, cfg->d_model);   /* masked self-attention */
    read_ln_weights (wb, cfg->d_model);   /* LN_1 */
    read_mha_weights(wb, cfg->d_model);   /* cross-attention */
    read_ln_weights (wb, cfg->d_model);   /* LN_2 */
    read_ffn_weights(wb, cfg->d_model, cfg->d_ff);
    read_ln_weights (wb, cfg->d_model);   /* LN_3 */
}

size_t dm_transformer_weight_count(const TransformerConfig *cfg) {
    WBuf wb = { .data = NULL, .pos = 0 };

    /* [A] Shared token embedding */
    wb_next(&wb, (size_t)cfg->vocab_size * cfg->d_model);

    /* [B] Encoder layers */
    for (int i = 0; i < cfg->num_layers; i++)
        count_encoder_layer(&wb, cfg);

    /* [C] Decoder layers */
    for (int i = 0; i < cfg->num_layers; i++)
        count_decoder_layer(&wb, cfg);

    /* [D] Final encoder LayerNorm */
    read_ln_weights(&wb, cfg->d_model);

    return wb.pos;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §5.3  Learning Rate Schedule  (Eq. 3)
 *
 *   lrate = d_model^{-0.5} · min(step^{-0.5}, step · warmup_steps^{-1.5})
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

float dm_transformer_lr_schedule(int d_model, int step, int warmup_steps) {
    if (step <= 0) step = 1;
    double s  = (double)step;
    double ws = (double)warmup_steps;
    double lrate = pow((double)d_model, -0.5)
                 * fmin(pow(s, -0.5), s * pow(ws, -1.5));
    return (float)lrate;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.1  Encoder Layer
 *
 *   sub1 = LayerNorm(x   + MHA_self(x, x, x))
 *   out  = LayerNorm(sub1 + FFN(sub1))
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_transformer_encoder_layer(const float             *x_in,
                                  const TransformerConfig *cfg,
                                  const float             *layer_weights,
                                  int                      seq,
                                  float                   *x_out) {
    int dm = cfg->d_model;
    int df = cfg->d_ff;
    int h  = cfg->num_heads;

    /* Read weights from the layer block */
    WBuf wb = { .data = layer_weights, .pos = 0 };
    MHAWeights mha_w = read_mha_weights(&wb, dm);
    LNWeights  ln1_w = read_ln_weights (&wb, dm);
    FFNWeights ffn_w = read_ffn_weights(&wb, dm, df);
    LNWeights  ln2_w = read_ln_weights (&wb, dm);

    float *mha_out = malloc((size_t)seq * dm * sizeof(float));
    float *ffn_out = malloc((size_t)seq * dm * sizeof(float));
    float *buf     = malloc((size_t)seq * dm * sizeof(float));  /* residual copy */
    if (!mha_out || !ffn_out || !buf) {
        free(mha_out); free(ffn_out); free(buf); return -1;
    }

    /* ── Sub-layer 1: Self-Attention + Add & Norm ─────────────────────── */
    if (dm_transformer_mha(mha_w.W_Q, mha_w.b_Q,
                            mha_w.W_K, mha_w.b_K,
                            mha_w.W_V, mha_w.b_V,
                            mha_w.W_O, mha_w.b_O,
                            x_in, x_in, x_in,  /* self: Q=K=V=x */
                            NULL,               /* no mask in encoder */
                            seq, seq, dm, h, mha_out) != 0) {
        free(mha_out); free(ffn_out); free(buf); return -1;
    }
    /* buf = x_in + mha_out  (residual connection) */
    for (int i = 0; i < seq * dm; i++) buf[i] = x_in[i] + mha_out[i];
    /* LayerNorm(buf) — in-place */
    dm_transformer_layer_norm(buf, ln1_w.gamma, ln1_w.beta, seq, dm, 1e-6f);

    /* ── Sub-layer 2: FFN + Add & Norm ──────────────────────────────────── */
    if (dm_transformer_ffn(buf, ffn_w.W_1, ffn_w.b_1,
                            ffn_w.W_2, ffn_w.b_2,
                            seq, dm, df, ffn_out) != 0) {
        free(mha_out); free(ffn_out); free(buf); return -1;
    }
    /* x_out = buf + ffn_out  (residual) + LayerNorm */
    for (int i = 0; i < seq * dm; i++) x_out[i] = buf[i] + ffn_out[i];
    dm_transformer_layer_norm(x_out, ln2_w.gamma, ln2_w.beta, seq, dm, 1e-6f);

    free(mha_out); free(ffn_out); free(buf);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.1  Decoder Layer
 *
 *   sub1 = LayerNorm(x    + MaskedMHA_self(x, x, x))
 *   sub2 = LayerNorm(sub1 + CrossMHA(sub1, enc_out, enc_out))
 *   out  = LayerNorm(sub2 + FFN(sub2))
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_transformer_decoder_layer(const float             *x_in,
                                  const float             *enc_out,
                                  const TransformerConfig *cfg,
                                  const float             *layer_weights,
                                  int                      tgt_seq,
                                  int                      src_seq,
                                  float                   *x_out) {
    int dm = cfg->d_model;
    int df = cfg->d_ff;
    int h  = cfg->num_heads;

    WBuf wb = { .data = layer_weights, .pos = 0 };
    MHAWeights self_w  = read_mha_weights(&wb, dm);
    LNWeights  ln1_w   = read_ln_weights (&wb, dm);
    MHAWeights cross_w = read_mha_weights(&wb, dm);
    LNWeights  ln2_w   = read_ln_weights (&wb, dm);
    FFNWeights ffn_w   = read_ffn_weights(&wb, dm, df);
    LNWeights  ln3_w   = read_ln_weights (&wb, dm);

    float *causal_mask = malloc((size_t)tgt_seq * tgt_seq * sizeof(float));
    float *mha_out     = malloc((size_t)tgt_seq * dm      * sizeof(float));
    float *sub1        = malloc((size_t)tgt_seq * dm      * sizeof(float));
    float *cross_out   = malloc((size_t)tgt_seq * dm      * sizeof(float));
    float *sub2        = malloc((size_t)tgt_seq * dm      * sizeof(float));
    float *ffn_out     = malloc((size_t)tgt_seq * dm      * sizeof(float));
    if (!causal_mask || !mha_out || !sub1 || !cross_out || !sub2 || !ffn_out) {
        free(causal_mask); free(mha_out); free(sub1);
        free(cross_out); free(sub2); free(ffn_out);
        return -1;
    }

    /* Build causal mask for decoder self-attention (§3.2.3) */
    dm_transformer_causal_mask(tgt_seq, causal_mask);

    /* ── Sub-layer 1: Masked Self-Attention + Add & Norm ─────────────── */
    if (dm_transformer_mha(self_w.W_Q, self_w.b_Q,
                            self_w.W_K, self_w.b_K,
                            self_w.W_V, self_w.b_V,
                            self_w.W_O, self_w.b_O,
                            x_in, x_in, x_in,
                            causal_mask,
                            tgt_seq, tgt_seq, dm, h, mha_out) != 0) goto fail;

    for (int i = 0; i < tgt_seq * dm; i++) sub1[i] = x_in[i] + mha_out[i];
    dm_transformer_layer_norm(sub1, ln1_w.gamma, ln1_w.beta, tgt_seq, dm, 1e-6f);

    /* ── Sub-layer 2: Cross-Attention + Add & Norm ──────────────────── */
    /*   Q from sub1 (decoder), K and V from enc_out (encoder)           */
    if (dm_transformer_mha(cross_w.W_Q, cross_w.b_Q,
                            cross_w.W_K, cross_w.b_K,
                            cross_w.W_V, cross_w.b_V,
                            cross_w.W_O, cross_w.b_O,
                            sub1, enc_out, enc_out,   /* Q=decoder, K=V=encoder */
                            NULL,                      /* no mask in cross-attn */
                            tgt_seq, src_seq, dm, h, cross_out) != 0) goto fail;

    for (int i = 0; i < tgt_seq * dm; i++) sub2[i] = sub1[i] + cross_out[i];
    dm_transformer_layer_norm(sub2, ln2_w.gamma, ln2_w.beta, tgt_seq, dm, 1e-6f);

    /* ── Sub-layer 3: FFN + Add & Norm ──────────────────────────────── */
    if (dm_transformer_ffn(sub2, ffn_w.W_1, ffn_w.b_1,
                            ffn_w.W_2, ffn_w.b_2,
                            tgt_seq, dm, df, ffn_out) != 0) goto fail;

    for (int i = 0; i < tgt_seq * dm; i++) x_out[i] = sub2[i] + ffn_out[i];
    dm_transformer_layer_norm(x_out, ln3_w.gamma, ln3_w.beta, tgt_seq, dm, 1e-6f);

    free(causal_mask); free(mha_out); free(sub1);
    free(cross_out); free(sub2); free(ffn_out);
    return 0;

fail:
    free(causal_mask); free(mha_out); free(sub1);
    free(cross_out); free(sub2); free(ffn_out);
    return -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Helper: encoder-layer weight block size (in floats)
 * ═══════════════════════════════════════════════════════════════════════════ */
static size_t encoder_layer_weight_size(const TransformerConfig *cfg) {
    WBuf wb = { .data = NULL, .pos = 0 };
    count_encoder_layer(&wb, cfg);
    return wb.pos;
}

static size_t decoder_layer_weight_size(const TransformerConfig *cfg) {
    WBuf wb = { .data = NULL, .pos = 0 };
    count_decoder_layer(&wb, cfg);
    return wb.pos;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * §3.4  Embeddings and Softmax
 *
 * Shared weight matrix between:
 *   (1) source embedding  (2) target embedding  (3) pre-softmax projection
 *
 * Weights are multiplied by √d_model when applied to tokens (§3.4).
 * The pre-softmax linear projection uses the same matrix transposed.
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Embed token indices and add positional encoding.
 * embed_w: [vocab_size × d_model]  (section [A])
 * out: [seq × d_model]             */
static void embed_and_pe(const int   *tokens,
                          int          seq,
                          int          d_model,
                          int          vocab_size,
                          const float *embed_w,
                          float       *out) {
    float scale = sqrtf((float)d_model);  /* multiply by √d_model (§3.4) */

    for (int t = 0; t < seq; t++) {
        int tok = tokens[t];
        if (tok < 0 || tok >= vocab_size) tok = 0;  /* clamp OOV */
        const float *row = embed_w + tok * d_model;
        float       *dst = out     + t   * d_model;
        for (int d = 0; d < d_model; d++) dst[d] = row[d] * scale;
    }

    /* Add positional encodings (sinusoidal, computed on the fly) */
    for (int pos = 0; pos < seq; pos++) {
        float *dst = out + pos * d_model;
        for (int i = 0; i < d_model / 2; i++) {
            double denom = pow(10000.0, (2.0 * i) / d_model);
            dst[2*i]   += (float)sin(pos / denom);
            dst[2*i+1] += (float)cos(pos / denom);
        }
        if (d_model & 1) {
            int i = d_model / 2;
            double denom = pow(10000.0, (2.0 * i) / d_model);
            dst[d_model-1] += (float)sin(pos / denom);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Full Encoder Stack
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_transformer_encode(const TransformerConfig *cfg,
                           const float             *weights,
                           const int               *src_tokens,
                           int                      src_seq,
                           float                   *enc_out) {
    int dm = cfg->d_model;
    int N  = cfg->num_layers;

    /* Weight cursor starts at offset 0 */
    WBuf wb = { .data = weights, .pos = 0 };

    /* [A] Token embedding */
    const float *embed_w = wb_next(&wb, (size_t)cfg->vocab_size * dm);

    /* Embed source tokens + positional encoding → enc_out */
    embed_and_pe(src_tokens, src_seq, dm, cfg->vocab_size, embed_w, enc_out);

    /* [B] Encoder layers */
    size_t layer_sz = encoder_layer_weight_size(cfg);
    float *tmp = malloc((size_t)src_seq * dm * sizeof(float));
    if (!tmp) return -1;

    for (int i = 0; i < N; i++) {
        const float *lw = wb_next(&wb, layer_sz);
        /* Alternate enc_out ↔ tmp to avoid extra copies */
        float *in  = (i % 2 == 0) ? enc_out : tmp;
        float *out = (i % 2 == 0) ? tmp      : enc_out;
        if (i == 0) {
            /* First layer reads enc_out (just filled by embed_and_pe) */
            in  = enc_out;
            out = tmp;
        }
        if (dm_transformer_encoder_layer(in, cfg, lw, src_seq, out) != 0) {
            free(tmp); return -1;
        }
        /* After last layer, make sure result is in enc_out */
        if (i == N - 1 && (N % 2 == 1)) {
            /* Result is in tmp; copy to enc_out */
            memcpy(enc_out, tmp, (size_t)src_seq * dm * sizeof(float));
        }
    }

    /* [D] Final encoder LayerNorm */
    /* Skip [C] decoder layers in the weight stream */
    size_t dec_layer_sz = decoder_layer_weight_size(cfg);
    for (int i = 0; i < N; i++) wb_next(&wb, dec_layer_sz);

    LNWeights enc_final_ln = read_ln_weights(&wb, dm);
    dm_transformer_layer_norm(enc_out, enc_final_ln.gamma, enc_final_ln.beta,
                               src_seq, dm, 1e-6f);

    free(tmp);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Full Decoder Stack
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_transformer_decode(const TransformerConfig *cfg,
                           const float             *weights,
                           const int               *tgt_tokens,
                           int                      tgt_seq,
                           const float             *enc_out,
                           int                      src_seq,
                           float                   *logits_out) {
    int dm = cfg->d_model;
    int N  = cfg->num_layers;

    WBuf wb = { .data = weights, .pos = 0 };

    /* [A] Shared token embedding */
    const float *embed_w = wb_next(&wb, (size_t)cfg->vocab_size * dm);

    /* Embed target tokens + positional encoding */
    float *dec = malloc((size_t)tgt_seq * dm * sizeof(float));
    float *tmp = malloc((size_t)tgt_seq * dm * sizeof(float));
    if (!dec || !tmp) { free(dec); free(tmp); return -1; }

    embed_and_pe(tgt_tokens, tgt_seq, dm, cfg->vocab_size, embed_w, dec);

    /* [B] Skip encoder layers */
    size_t enc_layer_sz = encoder_layer_weight_size(cfg);
    for (int i = 0; i < N; i++) wb_next(&wb, enc_layer_sz);

    /* [C] Decoder layers */
    size_t dec_layer_sz = decoder_layer_weight_size(cfg);
    for (int i = 0; i < N; i++) {
        const float *lw = wb_next(&wb, dec_layer_sz);
        float *in  = (i % 2 == 0) ? dec : tmp;
        float *out = (i % 2 == 0) ? tmp : dec;
        if (i == 0) { in = dec; out = tmp; }

        if (dm_transformer_decoder_layer(in, enc_out, cfg, lw,
                                          tgt_seq, src_seq, out) != 0) {
            free(dec); free(tmp); return -1;
        }
        if (i == N-1 && (N % 2 == 1))
            memcpy(dec, tmp, (size_t)tgt_seq * dm * sizeof(float));
    }

    /* Output projection: logits = dec * embed_w^T
     * Shape: [tgt_seq × d_model] * [d_model × vocab_size] → [tgt_seq × vocab_size]
     * embed_w is [vocab_size × d_model], so embed_w^T is [d_model × vocab_size].
     * We want: logits[t][v] = Σ_d dec[t][d] * embed_w[v][d]             */
    for (int t = 0; t < tgt_seq; t++) {
        float *logit_row = logits_out + t * cfg->vocab_size;
        for (int v = 0; v < cfg->vocab_size; v++) {
            double acc = 0.0;
            const float *drow = dec     + t * dm;
            const float *erow = embed_w + v * dm;
            for (int d = 0; d < dm; d++) acc += (double)drow[d] * erow[d];
            logit_row[v] = (float)acc;
        }
    }

    free(dec); free(tmp);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Full Forward Pass
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_transformer_forward(const TransformerConfig *cfg,
                            const float             *weights,
                            const int               *src_tokens,
                            int                      src_seq,
                            const int               *tgt_tokens,
                            int                      tgt_seq,
                            float                   *logits_out) {
    int dm = cfg->d_model;

    float *enc_out = malloc((size_t)src_seq * dm * sizeof(float));
    if (!enc_out) return -1;

    if (dm_transformer_encode(cfg, weights, src_tokens, src_seq, enc_out) != 0) {
        free(enc_out); return -1;
    }
    if (dm_transformer_decode(cfg, weights, tgt_tokens, tgt_seq,
                               enc_out, src_seq, logits_out) != 0) {
        free(enc_out); return -1;
    }

    free(enc_out);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Weight file I/O
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_transformer_save(const char              *path,
                         const TransformerConfig *cfg,
                         const float             *weights) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    size_t wc = dm_transformer_weight_count(cfg);

    uint32_t magic   = DM_TRANSFORMER_WEIGHT_MAGIC;
    uint32_t version = DM_TRANSFORMER_WEIGHT_VER;
    uint32_t variant = (uint32_t)cfg->variant;
    uint32_t vocab   = (uint32_t)cfg->vocab_size;
    uint32_t maxlen  = (uint32_t)cfg->max_seq_len;
    uint64_t wc64    = (uint64_t)wc;

    fwrite(&magic,   sizeof(uint32_t), 1, f);
    fwrite(&version, sizeof(uint32_t), 1, f);
    fwrite(&variant, sizeof(uint32_t), 1, f);
    fwrite(&vocab,   sizeof(uint32_t), 1, f);
    fwrite(&maxlen,  sizeof(uint32_t), 1, f);
    fwrite(&wc64,    sizeof(uint64_t), 1, f);
    fwrite(weights,  sizeof(float), wc, f);

    fclose(f);
    return 0;
}

int dm_transformer_load(const char        *path,
                         TransformerConfig *cfg,
                         float            **weights_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t magic, version, variant, vocab, maxlen;
    uint64_t wc;

    if (fread(&magic,   sizeof(uint32_t), 1, f) != 1 ||
        fread(&version, sizeof(uint32_t), 1, f) != 1 ||
        fread(&variant, sizeof(uint32_t), 1, f) != 1 ||
        fread(&vocab,   sizeof(uint32_t), 1, f) != 1 ||
        fread(&maxlen,  sizeof(uint32_t), 1, f) != 1 ||
        fread(&wc,      sizeof(uint64_t), 1, f) != 1) {
        fclose(f); return -1;
    }

    if (magic != DM_TRANSFORMER_WEIGHT_MAGIC || version != DM_TRANSFORMER_WEIGHT_VER) {
        fclose(f); return -1;
    }

    dm_transformer_config_init(cfg, (TransformerVariant)variant, (int)vocab, (int)maxlen);

    float *w = malloc((size_t)wc * sizeof(float));
    if (!w) { fclose(f); return -1; }
    if (fread(w, sizeof(float), (size_t)wc, f) != (size_t)wc) {
        free(w); fclose(f); return -1;
    }
    fclose(f);

    *weights_out = w;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TF/Keras training script (embedded Python)
 *
 * Implements the exact paper training setup:
 *   • Adam with β1=0.9, β2=0.98, ε=1e-9
 *   • Warmup learning rate schedule (Eq. 3), warmup_steps=4000
 *   • Residual dropout P_drop (applied after each sub-layer)
 *   • Label smoothing ε_ls=0.1
 *   • Shared weight matrix for src/tgt embeddings + output projection
 *
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char TRANSFORMER_TRAIN_SCRIPT[] =
"import sys, os, math, numpy as np\n"
"import tensorflow as tf\n"
"\n"
"# ── Args ────────────────────────────────────────────────────────────────\n"
"src_path    = sys.argv[1]\n"
"tgt_path    = sys.argv[2]\n"
"out_path    = sys.argv[3]\n"
"variant     = sys.argv[4]        # 'base' or 'big'\n"
"vocab_size  = int(sys.argv[5])\n"
"max_len     = int(sys.argv[6])\n"
"epochs      = int(sys.argv[7])\n"
"batch_size  = int(sys.argv[8])\n"
"warmup_steps= int(sys.argv[9])\n"
"\n"
"# ── Hyperparameters (Table 3) ────────────────────────────────────────────\n"
"if variant == 'big':\n"
"    N, d_model, d_ff, h, d_k = 6, 1024, 4096, 16, 64\n"
"    dropout, eps_ls           = 0.3, 0.1\n"
"else:  # base\n"
"    N, d_model, d_ff, h, d_k = 6, 512, 2048, 8, 64\n"
"    dropout, eps_ls           = 0.1, 0.1\n"
"\n"
"# ── §5.3 Learning rate schedule (Eq. 3) ──────────────────────────────────\n"
"class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):\n"
"    def __init__(self, d_model, warmup_steps):\n"
"        self.d_model      = float(d_model)\n"
"        self.warmup_steps = float(warmup_steps)\n"
"    def __call__(self, step):\n"
"        step  = tf.cast(step + 1, tf.float32)\n"
"        arg1  = tf.math.rsqrt(step)\n"
"        arg2  = step * (self.warmup_steps ** -1.5)\n"
"        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)\n"
"\n"
"# ── §3.5 Positional encoding ─────────────────────────────────────────────\n"
"def positional_encoding(max_len, d_model):\n"
"    positions = np.arange(max_len)[:, None]          # [max_len, 1]\n"
"    dims      = np.arange(d_model)[None, :]           # [1, d_model]\n"
"    angles    = positions / np.power(10000, (2*(dims//2)) / d_model)\n"
"    angles[:, 0::2] = np.sin(angles[:, 0::2])         # even: sin\n"
"    angles[:, 1::2] = np.cos(angles[:, 1::2])         # odd:  cos\n"
"    return tf.cast(angles[None, :, :], tf.float32)    # [1, max_len, d_model]\n"
"\n"
"# ── §3.2.1 Scaled Dot-Product Attention (Eq. 1) ──────────────────────────\n"
"def scaled_dot_product_attention(Q, K, V, mask=None):\n"
"    d_k    = tf.cast(tf.shape(K)[-1], tf.float32)\n"
"    scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(d_k)\n"
"    if mask is not None:\n"
"        scores += (mask * -1e9)\n"
"    weights = tf.nn.softmax(scores, axis=-1)\n"
"    return tf.matmul(weights, V)\n"
"\n"
"# ── §3.2.2 Multi-Head Attention (Eq. 2) ──────────────────────────────────\n"
"class MultiHeadAttention(tf.keras.layers.Layer):\n"
"    def __init__(self, d_model, h):\n"
"        super().__init__()\n"
"        assert d_model % h == 0\n"
"        self.h       = h\n"
"        self.d_model = d_model\n"
"        self.d_k     = d_model // h  # = d_v\n"
"        # Single weight matrix stacking all h projections (§3.2.2)\n"
"        self.W_Q = tf.keras.layers.Dense(d_model, use_bias=True)\n"
"        self.W_K = tf.keras.layers.Dense(d_model, use_bias=True)\n"
"        self.W_V = tf.keras.layers.Dense(d_model, use_bias=True)\n"
"        self.W_O = tf.keras.layers.Dense(d_model, use_bias=True)  # W^O\n"
"\n"
"    def split_heads(self, x, batch):\n"
"        # x: [batch, seq, d_model] → [batch, h, seq, d_k]\n"
"        x = tf.reshape(x, (batch, -1, self.h, self.d_k))\n"
"        return tf.transpose(x, perm=[0, 2, 1, 3])\n"
"\n"
"    def call(self, Q, K, V, mask=None):\n"
"        batch = tf.shape(Q)[0]\n"
"        Q = self.split_heads(self.W_Q(Q), batch)  # [B, h, seq_q, d_k]\n"
"        K = self.split_heads(self.W_K(K), batch)\n"
"        V = self.split_heads(self.W_V(V), batch)\n"
"        attn = scaled_dot_product_attention(Q, K, V, mask)  # [B, h, seq_q, d_k]\n"
"        # Concat heads: [B, seq_q, h*d_k] = [B, seq_q, d_model]\n"
"        attn = tf.transpose(attn, perm=[0, 2, 1, 3])\n"
"        attn = tf.reshape(attn, (batch, -1, self.d_model))\n"
"        return self.W_O(attn)\n"
"\n"
"# ── §3.3 Position-wise FFN (Eq. 2): FFN(x) = max(0, xW_1+b_1)W_2+b_2 ──\n"
"class PositionwiseFFN(tf.keras.layers.Layer):\n"
"    def __init__(self, d_model, d_ff):\n"
"        super().__init__()\n"
"        self.W_1 = tf.keras.layers.Dense(d_ff,    activation='relu')\n"
"        self.W_2 = tf.keras.layers.Dense(d_model, activation=None)\n"
"    def call(self, x):\n"
"        return self.W_2(self.W_1(x))\n"
"\n"
"# ── §3.1 Encoder Layer ───────────────────────────────────────────────────\n"
"class EncoderLayer(tf.keras.layers.Layer):\n"
"    def __init__(self, d_model, d_ff, h, dropout):\n"
"        super().__init__()\n"
"        self.self_attn = MultiHeadAttention(d_model, h)\n"
"        self.ffn       = PositionwiseFFN(d_model, d_ff)\n"
"        self.ln1       = tf.keras.layers.LayerNormalization(epsilon=1e-6)\n"
"        self.ln2       = tf.keras.layers.LayerNormalization(epsilon=1e-6)\n"
"        self.drop1     = tf.keras.layers.Dropout(dropout)\n"
"        self.drop2     = tf.keras.layers.Dropout(dropout)\n"
"\n"
"    def call(self, x, training, mask=None):\n"
"        # Sub-layer 1: self-attention + Add & Norm\n"
"        attn = self.drop1(self.self_attn(x, x, x, mask), training=training)\n"
"        x    = self.ln1(x + attn)\n"
"        # Sub-layer 2: FFN + Add & Norm\n"
"        ff   = self.drop2(self.ffn(x), training=training)\n"
"        return self.ln2(x + ff)\n"
"\n"
"# ── §3.1 Decoder Layer ───────────────────────────────────────────────────\n"
"class DecoderLayer(tf.keras.layers.Layer):\n"
"    def __init__(self, d_model, d_ff, h, dropout):\n"
"        super().__init__()\n"
"        self.self_attn  = MultiHeadAttention(d_model, h)  # masked\n"
"        self.cross_attn = MultiHeadAttention(d_model, h)  # encoder-decoder\n"
"        self.ffn        = PositionwiseFFN(d_model, d_ff)\n"
"        self.ln1        = tf.keras.layers.LayerNormalization(epsilon=1e-6)\n"
"        self.ln2        = tf.keras.layers.LayerNormalization(epsilon=1e-6)\n"
"        self.ln3        = tf.keras.layers.LayerNormalization(epsilon=1e-6)\n"
"        self.drop1      = tf.keras.layers.Dropout(dropout)\n"
"        self.drop2      = tf.keras.layers.Dropout(dropout)\n"
"        self.drop3      = tf.keras.layers.Dropout(dropout)\n"
"\n"
"    def call(self, x, enc_out, training, causal_mask=None, src_mask=None):\n"
"        # Sub-layer 1: masked self-attention + Add & Norm\n"
"        s1 = self.drop1(self.self_attn(x, x, x, causal_mask), training=training)\n"
"        x  = self.ln1(x + s1)\n"
"        # Sub-layer 2: cross-attention (Q=decoder, K=V=encoder) + Add & Norm\n"
"        s2 = self.drop2(self.cross_attn(x, enc_out, enc_out, src_mask), training=training)\n"
"        x  = self.ln2(x + s2)\n"
"        # Sub-layer 3: FFN + Add & Norm\n"
"        s3 = self.drop3(self.ffn(x), training=training)\n"
"        return self.ln3(x + s3)\n"
"\n"
"# ── Full Transformer (§3.1) ───────────────────────────────────────────────\n"
"class Transformer(tf.keras.Model):\n"
"    def __init__(self, N, d_model, d_ff, h, vocab_size, max_len, dropout):\n"
"        super().__init__()\n"
"        # §3.4: shared embedding (src embed = tgt embed = output projection)\n"
"        self.embedding  = tf.keras.layers.Embedding(vocab_size, d_model)\n"
"        self.d_model    = d_model\n"
"        self.pe         = positional_encoding(max_len, d_model)\n"
"        self.enc_layers = [EncoderLayer(d_model, d_ff, h, dropout) for _ in range(N)]\n"
"        self.dec_layers = [DecoderLayer(d_model, d_ff, h, dropout) for _ in range(N)]\n"
"        self.enc_ln     = tf.keras.layers.LayerNormalization(epsilon=1e-6)\n"
"        self.emb_drop   = tf.keras.layers.Dropout(dropout)\n"
"\n"
"    def encode(self, src, training, src_mask=None):\n"
"        seq = tf.shape(src)[1]\n"
"        # §3.4: multiply embeddings by sqrt(d_model)\n"
"        x = self.embedding(src) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))\n"
"        x = self.emb_drop(x + self.pe[:, :seq, :], training=training)\n"
"        for layer in self.enc_layers:\n"
"            x = layer(x, training, src_mask)\n"
"        return self.enc_ln(x)\n"
"\n"
"    def decode(self, tgt, enc_out, training, causal_mask=None, src_mask=None):\n"
"        seq = tf.shape(tgt)[1]\n"
"        x = self.embedding(tgt) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))\n"
"        x = self.emb_drop(x + self.pe[:, :seq, :], training=training)\n"
"        for layer in self.dec_layers:\n"
"            x = layer(x, enc_out, training, causal_mask, src_mask)\n"
"        # §3.4: shared weight output projection = embedding^T\n"
"        return tf.matmul(x, self.embedding.embeddings, transpose_b=True)\n"
"\n"
"    def call(self, inputs, training=False):\n"
"        src, tgt = inputs\n"
"        # Build causal mask for decoder\n"
"        tgt_len = tf.shape(tgt)[1]\n"
"        causal  = 1 - tf.linalg.band_part(tf.ones((tgt_len, tgt_len)), -1, 0)\n"
"        causal  = causal[tf.newaxis, tf.newaxis, :, :]  # [1,1,tgt,tgt]\n"
"        enc_out = self.encode(src, training)\n"
"        return self.decode(tgt, enc_out, training, causal)\n"
"\n"
"# ── §5.4 Label smoothing (ε_ls = 0.1) ────────────────────────────────────\n"
"def label_smoothed_ce(logits, labels, vocab_size, eps_ls):\n"
"    one_hot  = tf.one_hot(labels, vocab_size)\n"
"    smooth   = one_hot * (1.0 - eps_ls) + (eps_ls / vocab_size)\n"
"    log_prob = tf.nn.log_softmax(logits, axis=-1)\n"
"    loss     = -tf.reduce_sum(smooth * log_prob, axis=-1)\n"
"    return tf.reduce_mean(loss)\n"
"\n"
"# ── Data loading (line-by-line token IDs) ────────────────────────────────\n"
"def load_token_lines(path, max_len):\n"
"    seqs = []\n"
"    with open(path) as f:\n"
"        for line in f:\n"
"            ids = [int(x) for x in line.strip().split()]\n"
"            seqs.append(ids[:max_len])\n"
"    return seqs\n"
"\n"
"def pad_batch(seqs, pad_id=0):\n"
"    L = max(len(s) for s in seqs)\n"
"    return np.array([s + [pad_id]*(L-len(s)) for s in seqs], dtype=np.int32)\n"
"\n"
"print('Loading data ...')\n"
"src_seqs = load_token_lines(src_path, max_len)\n"
"tgt_seqs = load_token_lines(tgt_path, max_len)\n"
"assert len(src_seqs) == len(tgt_seqs), 'src/tgt line count mismatch'\n"
"n = len(src_seqs)\n"
"print(f'  {n} sentence pairs loaded.')\n"
"\n"
"# Build model\n"
"model = Transformer(N, d_model, d_ff, h, vocab_size, max_len, dropout)\n"
"\n"
"# §5.3 Adam with warmup schedule\n"
"schedule   = WarmupSchedule(d_model, warmup_steps)\n"
"optimizer  = tf.keras.optimizers.Adam(schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9)\n"
"\n"
"@tf.function\n"
"def train_step(src_batch, tgt_batch):\n"
"    # Teacher-forcing: decoder input = tgt[:-1], label = tgt[1:]\n"
"    tgt_in  = tgt_batch[:, :-1]\n"
"    tgt_out = tgt_batch[:, 1:]\n"
"    with tf.GradientTape() as tape:\n"
"        logits = model((src_batch, tgt_in), training=True)\n"
"        loss   = label_smoothed_ce(logits, tgt_out, vocab_size, eps_ls)\n"
"    grads = tape.gradient(loss, model.trainable_variables)\n"
"    optimizer.apply_gradients(zip(grads, model.trainable_variables))\n"
"    return loss\n"
"\n"
"indices = np.arange(n)\n"
"for epoch in range(epochs):\n"
"    np.random.shuffle(indices)\n"
"    total_loss = 0.0\n"
"    steps = 0\n"
"    for start in range(0, n, batch_size):\n"
"        batch_idx = indices[start:start+batch_size]\n"
"        src_b = pad_batch([src_seqs[i] for i in batch_idx])\n"
"        tgt_b = pad_batch([tgt_seqs[i] for i in batch_idx])\n"
"        loss  = train_step(\n"
"            tf.constant(src_b, dtype=tf.int32),\n"
"            tf.constant(tgt_b, dtype=tf.int32))\n"
"        total_loss += float(loss)\n"
"        steps += 1\n"
"    print(f'Epoch {epoch+1}/{epochs}  loss={total_loss/steps:.4f}')\n"
"\n"
"# Extract and save weights in the C layout (sections A → D)\n"
"print('Saving weights ...')\n"
"ws = []\n"
"emb_w = model.embedding.embeddings.numpy()  # [vocab, d_model]\n"
"ws.append(emb_w.flatten())\n"
"\n"
"for layer in model.enc_layers:\n"
"    for proj in [layer.self_attn.W_Q, layer.self_attn.W_K,\n"
"                 layer.self_attn.W_V, layer.self_attn.W_O]:\n"
"        ws.append(proj.kernel.numpy().flatten())\n"
"        ws.append(proj.bias.numpy().flatten())\n"
"    ws.append(layer.ln1.gamma.numpy().flatten())\n"
"    ws.append(layer.ln1.beta.numpy().flatten())\n"
"    ws.append(layer.ffn.W_1.kernel.numpy().flatten())\n"
"    ws.append(layer.ffn.W_1.bias.numpy().flatten())\n"
"    ws.append(layer.ffn.W_2.kernel.numpy().flatten())\n"
"    ws.append(layer.ffn.W_2.bias.numpy().flatten())\n"
"    ws.append(layer.ln2.gamma.numpy().flatten())\n"
"    ws.append(layer.ln2.beta.numpy().flatten())\n"
"\n"
"for layer in model.dec_layers:\n"
"    for proj in [layer.self_attn.W_Q,  layer.self_attn.W_K,\n"
"                 layer.self_attn.W_V,  layer.self_attn.W_O]:\n"
"        ws.append(proj.kernel.numpy().flatten())\n"
"        ws.append(proj.bias.numpy().flatten())\n"
"    ws.append(layer.ln1.gamma.numpy().flatten())\n"
"    ws.append(layer.ln1.beta.numpy().flatten())\n"
"    for proj in [layer.cross_attn.W_Q, layer.cross_attn.W_K,\n"
"                 layer.cross_attn.W_V, layer.cross_attn.W_O]:\n"
"        ws.append(proj.kernel.numpy().flatten())\n"
"        ws.append(proj.bias.numpy().flatten())\n"
"    ws.append(layer.ln2.gamma.numpy().flatten())\n"
"    ws.append(layer.ln2.beta.numpy().flatten())\n"
"    ws.append(layer.ffn.W_1.kernel.numpy().flatten())\n"
"    ws.append(layer.ffn.W_1.bias.numpy().flatten())\n"
"    ws.append(layer.ffn.W_2.kernel.numpy().flatten())\n"
"    ws.append(layer.ffn.W_2.bias.numpy().flatten())\n"
"    ws.append(layer.ln3.gamma.numpy().flatten())\n"
"    ws.append(layer.ln3.beta.numpy().flatten())\n"
"\n"
"ws.append(model.enc_ln.gamma.numpy().flatten())\n"
"ws.append(model.enc_ln.beta.numpy().flatten())\n"
"\n"
"flat = np.concatenate(ws).astype(np.float32)\n"
"import struct\n"
"with open(out_path, 'wb') as f:\n"
"    f.write(struct.pack('<IIIIIq', 0x54524E53, 1,\n"
"                        0 if variant=='base' else 1,\n"
"                        vocab_size, max_len, len(flat)))\n"
"    flat.tofile(f)\n"
"print(f'Saved {len(flat)} weights to {out_path}')\n";

/* ═══════════════════════════════════════════════════════════════════════════
 * CLI helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static void transformer_usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s transformer train  --src <src.txt> --tgt <tgt.txt> -o <weights.bin>\n"
        "                        [--variant base|big] [--vocab-size N] [--max-len N]\n"
        "                        [--epochs N] [--batch N] [--warmup N]\n"
        "\n"
        "  %s transformer infer  -m <weights.bin> --src \"token ids...\"\n"
        "                        [--max-tokens N]\n"
        "\n"
        "  %s transformer encode --src <tokens.txt> -m <weights.bin> -o <enc.bin>\n"
        "\n"
        "  %s transformer bench  [--variant base|big] [--src-len N] [--tgt-len N]\n"
        "\n"
        "  %s transformer info   -m <weights.bin>\n"
        "\n"
        "Token files: one sequence per line, space-separated integer token IDs.\n"
        "Train uses TensorFlow/Keras (Python 3 + tensorflow must be installed).\n"
        "Infer/encode/bench use the pure-C forward pass.\n",
        prog, prog, prog, prog, prog);
}

/* Run training via Python subprocess */
static int run_tf_train(const char *src_path, const char *tgt_path,
                         const char *out_path,  const char *variant,
                         int vocab_size, int max_len, int epochs,
                         int batch_size, int warmup_steps) {
    /* Write the embedded Python script to a temp file */
    const char *tmpscript = "/tmp/_dm_transformer_train.py";
    FILE *f = fopen(tmpscript, "w");
    if (!f) { fprintf(stderr, "Cannot write temp script\n"); return -1; }
    fputs(TRANSFORMER_TRAIN_SCRIPT, f);
    fclose(f);

    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "python3 %s '%s' '%s' '%s' '%s' %d %d %d %d %d",
        tmpscript, src_path, tgt_path, out_path, variant,
        vocab_size, max_len, epochs, batch_size, warmup_steps);

    int rc = system(cmd);
    return (rc == 0) ? 0 : -1;
}

/* ── cmd: bench ─────────────────────────────────────────────────────────── */
static int cmd_bench(const char *variant_str, int src_len, int tgt_len) {
    TransformerVariant variant =
        (strcmp(variant_str, "big") == 0) ? TRANSFORMER_BIG : TRANSFORMER_BASE;

    TransformerConfig cfg;
    dm_transformer_config_init(&cfg, variant, 32000, 512);

    size_t wc = dm_transformer_weight_count(&cfg);
    printf("Transformer-%s weight count: %zu (%.2f M params)\n",
           variant_str, wc, (double)wc / 1e6);

    float *weights = calloc(wc, sizeof(float));
    if (!weights) { fprintf(stderr, "OOM\n"); return -1; }

    /* Build dummy sequences */
    int *src = calloc(src_len, sizeof(int));
    int *tgt = calloc(tgt_len, sizeof(int));
    if (!src || !tgt) { free(weights); free(src); free(tgt); return -1; }
    for (int i = 0; i < src_len; i++) src[i] = i % cfg.vocab_size;
    for (int i = 0; i < tgt_len; i++) tgt[i] = i % cfg.vocab_size;

    float *logits = malloc((size_t)tgt_len * cfg.vocab_size * sizeof(float));
    if (!logits) { free(weights); free(src); free(tgt); return -1; }

#ifndef _WIN32
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
#endif

    int rc = dm_transformer_forward(&cfg, weights, src, src_len, tgt, tgt_len, logits);

#ifndef _WIN32
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (rc == 0)
        printf("Forward pass (src=%d, tgt=%d): %.3f s\n", src_len, tgt_len, elapsed);
#endif

    if (rc != 0)
        fprintf(stderr, "Forward pass FAILED (rc=%d)\n", rc);
    else
        printf("  logits[0][0] = %.6f (random weights)\n", logits[0]);

    free(weights); free(src); free(tgt); free(logits);
    return rc;
}

/* ── cmd: info ──────────────────────────────────────────────────────────── */
static int cmd_info(const char *model_path) {
    TransformerConfig cfg;
    float *weights = NULL;
    if (dm_transformer_load(model_path, &cfg, &weights) != 0) {
        fprintf(stderr, "Failed to load: %s\n", model_path);
        return -1;
    }
    size_t wc = dm_transformer_weight_count(&cfg);
    printf("Transformer weight file: %s\n", model_path);
    printf("  Variant:    %s\n",
           cfg.variant == TRANSFORMER_BIG ? "big" : "base");
    printf("  vocab_size: %d\n", cfg.vocab_size);
    printf("  max_seq_len:%d\n", cfg.max_seq_len);
    printf("  N (layers): %d\n", cfg.num_layers);
    printf("  d_model:    %d\n", cfg.d_model);
    printf("  d_ff:       %d\n", cfg.d_ff);
    printf("  h (heads):  %d\n", cfg.num_heads);
    printf("  P_drop:     %.2f\n", cfg.dropout);
    printf("  eps_ls:     %.2f\n", cfg.label_smooth);
    printf("  weights:    %zu (%.2f M)\n", wc, (double)wc / 1e6);
    free(weights);
    return 0;
}

/* ── cmd: infer ─────────────────────────────────────────────────────────── */
static int cmd_infer(const char *model_path, const char *src_str,
                      int max_tokens) {
    TransformerConfig cfg;
    float *weights = NULL;
    if (dm_transformer_load(model_path, &cfg, &weights) != 0) {
        fprintf(stderr, "Failed to load: %s\n", model_path);
        return -1;
    }

    /* Parse space-separated token IDs from src_str */
    int src_tokens[4096];
    int src_len = 0;
    {
        char *buf = strdup(src_str);
        char *tok = strtok(buf, " \t,");
        while (tok && src_len < 4096) {
            src_tokens[src_len++] = atoi(tok);
            tok = strtok(NULL, " \t,");
        }
        free(buf);
    }
    if (src_len == 0) { fprintf(stderr, "No source tokens provided\n"); free(weights); return -1; }

    /* Encode source */
    float *enc_out = malloc((size_t)src_len * cfg.d_model * sizeof(float));
    if (!enc_out) { free(weights); return -1; }
    if (dm_transformer_encode(&cfg, weights, src_tokens, src_len, enc_out) != 0) {
        fprintf(stderr, "Encode failed\n"); free(enc_out); free(weights); return -1;
    }

    /* Greedy decode (start with token 1 = BOS) */
    int tgt[4096] = {1};
    int tgt_len   = 1;
    float *logits = malloc((size_t)cfg.vocab_size * sizeof(float));
    if (!logits) { free(enc_out); free(weights); return -1; }

    printf("Generated tokens: ");
    while (tgt_len < max_tokens) {
        if (dm_transformer_decode(&cfg, weights, tgt, tgt_len,
                                   enc_out, src_len, logits) != 0) break;
        /* Logits for the last position only */
        float *last_logits = logits + (tgt_len - 1) * cfg.vocab_size;
        /* Argmax */
        int best = 0;
        float best_v = last_logits[0];
        for (int v = 1; v < cfg.vocab_size; v++) {
            if (last_logits[v] > best_v) { best_v = last_logits[v]; best = v; }
        }
        printf("%d ", best);
        fflush(stdout);
        tgt[tgt_len++] = best;
        if (best == 2) break;  /* EOS token = 2 */
    }
    printf("\n");

    free(logits); free(enc_out); free(weights);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * CLI entry point
 * ═══════════════════════════════════════════════════════════════════════════ */

int dm_transformer_cli(int argc, char **argv) {
    if (argc < 2) { transformer_usage(argv[0]); return 1; }

    const char *cmd = argv[1];

    /* ── train ─────────────────────────────────────────────────────────── */
    if (strcmp(cmd, "train") == 0) {
        const char *src_path  = NULL;
        const char *tgt_path  = NULL;
        const char *out_path  = NULL;
        const char *variant   = "base";
        int  vocab_size       = 32000;
        int  max_len          = 512;
        int  epochs           = 10;
        int  batch_size       = 32;
        int  warmup_steps     = 4000;

        for (int i = 2; i < argc; i++) {
            if      (!strcmp(argv[i],"--src")        && i+1<argc) src_path   = argv[++i];
            else if (!strcmp(argv[i],"--tgt")        && i+1<argc) tgt_path   = argv[++i];
            else if (!strcmp(argv[i],"-o")           && i+1<argc) out_path   = argv[++i];
            else if (!strcmp(argv[i],"--variant")    && i+1<argc) variant    = argv[++i];
            else if (!strcmp(argv[i],"--vocab-size") && i+1<argc) vocab_size = atoi(argv[++i]);
            else if (!strcmp(argv[i],"--max-len")    && i+1<argc) max_len    = atoi(argv[++i]);
            else if (!strcmp(argv[i],"--epochs")     && i+1<argc) epochs     = atoi(argv[++i]);
            else if (!strcmp(argv[i],"--batch")      && i+1<argc) batch_size = atoi(argv[++i]);
            else if (!strcmp(argv[i],"--warmup")     && i+1<argc) warmup_steps=atoi(argv[++i]);
        }
        if (!src_path || !tgt_path || !out_path) {
            fprintf(stderr, "Error: --src, --tgt, -o are required\n");
            transformer_usage(argv[0]); return 1;
        }
        return run_tf_train(src_path, tgt_path, out_path, variant,
                            vocab_size, max_len, epochs, batch_size, warmup_steps);
    }

    /* ── infer ─────────────────────────────────────────────────────────── */
    if (strcmp(cmd, "infer") == 0) {
        const char *model_path = NULL;
        const char *src_str    = NULL;
        int max_tokens         = 64;
        for (int i = 2; i < argc; i++) {
            if      (!strcmp(argv[i],"-m")           && i+1<argc) model_path = argv[++i];
            else if (!strcmp(argv[i],"--src")        && i+1<argc) src_str    = argv[++i];
            else if (!strcmp(argv[i],"--max-tokens") && i+1<argc) max_tokens = atoi(argv[++i]);
        }
        if (!model_path || !src_str) {
            fprintf(stderr, "Error: -m and --src are required\n"); return 1;
        }
        return cmd_infer(model_path, src_str, max_tokens);
    }

    /* ── bench ─────────────────────────────────────────────────────────── */
    if (strcmp(cmd, "bench") == 0) {
        const char *variant_str = "base";
        int src_len = 32, tgt_len = 32;
        for (int i = 2; i < argc; i++) {
            if      (!strcmp(argv[i],"--variant") && i+1<argc) variant_str = argv[++i];
            else if (!strcmp(argv[i],"--src-len") && i+1<argc) src_len     = atoi(argv[++i]);
            else if (!strcmp(argv[i],"--tgt-len") && i+1<argc) tgt_len     = atoi(argv[++i]);
        }
        return cmd_bench(variant_str, src_len, tgt_len);
    }

    /* ── info ──────────────────────────────────────────────────────────── */
    if (strcmp(cmd, "info") == 0) {
        const char *model_path = NULL;
        for (int i = 2; i < argc; i++)
            if (!strcmp(argv[i],"-m") && i+1<argc) model_path = argv[++i];
        if (!model_path) { fprintf(stderr, "Error: -m is required\n"); return 1; }
        return cmd_info(model_path);
    }

    fprintf(stderr, "Unknown subcommand: %s\n", cmd);
    transformer_usage(argv[0]);
    return 1;
}
