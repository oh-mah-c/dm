/*
 * vit.c — Vision Transformer (ViT), Dosovitskiy et al., ICLR 2021
 * arXiv:2010.11929v2
 *
 * Architecture (§3.1, Eq.1-4):
 *   z0  = [x_cls ; x1_E ; ... ; xN_E] + E_pos          (1)
 *   z'l = MSA(LN(z{l-1})) + z{l-1}                      (2)
 *   zl  = MLP(LN(z'l))    + z'l                          (3)
 *   y   = LN(z0^L)                                       (4)
 *
 * Weight flat-array layout (sequential, same order as forward pass):
 *   patch_proj_w  [d_model × (P²·C)]
 *   patch_proj_b  [d_model]
 *   cls_token     [d_model]
 *   pos_embed     [(N+1) × d_model]   N = (img/patch)²
 *   for each of L encoder blocks:
 *     ln1_gamma   [d_model]
 *     ln1_beta    [d_model]
 *     qkv_w       [3·d_model × d_model]
 *     qkv_b       [3·d_model]
 *     proj_w      [d_model × d_model]
 *     proj_b      [d_model]
 *     ln2_gamma   [d_model]
 *     ln2_beta    [d_model]
 *     mlp_fc1_w   [mlp_dim × d_model]
 *     mlp_fc1_b   [mlp_dim]
 *     mlp_fc2_w   [d_model × mlp_dim]
 *     mlp_fc2_b   [d_model]
 *   head_ln_gamma [d_model]
 *   head_ln_beta  [d_model]
 *   head_w        [num_classes × d_model]
 *   head_b        [num_classes]
 */

#include "models/vision/vit.h"
#include "core/dm_benchmark.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Variant table (Table 1 of paper + Tiny/Small extras) ─────────────────
 * {num_layers, d_model, mlp_dim, num_heads, patch_size}               */
static const int VIT_CFG[5][5] = {
    /* TINY  */ { 5,  192,  768,  3, 16},
    /* SMALL */ { 6,  384, 1536,  6, 16},
    /* BASE  */ {12,  768, 3072, 12, 16},
    /* LARGE */ {24, 1024, 4096, 16, 16},
    /* HUGE  */ {32, 1280, 5120, 16, 14},
};

void dm_vit_config_init(ViTConfig *cfg, ViTVariant v,
                        int num_classes, int img_size)
{
    if (!cfg) return;
    cfg->variant     = v;
    cfg->img_size    = img_size   > 0 ? img_size    : 224;
    cfg->patch_size  = VIT_CFG[v][4];
    cfg->num_layers  = VIT_CFG[v][0];
    cfg->d_model     = VIT_CFG[v][1];
    cfg->mlp_dim     = VIT_CFG[v][2];
    cfg->num_heads   = VIT_CFG[v][3];
    cfg->num_classes = num_classes > 0 ? num_classes : 1000;
}

/* ── Parameter count ────────────────────────────────────────────────────── */
size_t dm_vit_param_count(const ViTConfig *cfg)
{
    int P = cfg->patch_size, C = 3;
    int D = cfg->d_model, M = cfg->mlp_dim, L = cfg->num_layers;
    int N = (cfg->img_size / P) * (cfg->img_size / P);
    int K = cfg->num_classes;
    size_t n = 0;
    /* patch proj + bias */
    n += (size_t)D * (P * P * C) + D;
    /* cls token + pos embed */
    n += D + (size_t)(N + 1) * D;
    /* L encoder blocks */
    n += (size_t)L * (
        2*D +            /* ln1 */
        3*D*D + 3*D +    /* qkv */
        D*D + D +        /* out proj */
        2*D +            /* ln2 */
        M*D + M +        /* mlp fc1 */
        D*M + D          /* mlp fc2 */
    );
    /* head ln + linear */
    n += 2*D + (size_t)K*D + K;
    return n;
}

/* ── Weight cursor helper ───────────────────────────────────────────────── */
typedef struct { const float *ptr; } WCur;
static const float *wcur_take(WCur *c, size_t n) {
    const float *p = c->ptr; c->ptr += n; return p;
}

/* ── RNG for random init ────────────────────────────────────────────────── */
static uint32_t rng_state;
static float rng_normal(void) {
    /* Box-Muller; returns one sample */
    uint32_t u1, u2;
    u1 = rng_state ^ (rng_state << 13); u1 ^= u1 >> 17; u1 ^= u1 << 5; rng_state = u1;
    u2 = rng_state ^ (rng_state << 13); u2 ^= u2 >> 17; u2 ^= u2 << 5; rng_state = u2;
    float f1 = (float)((u1 >> 8) + 1) * (1.0f / 16777216.0f);
    float f2 = (float)((u2 >> 8) + 1) * (1.0f / 16777216.0f);
    return sqrtf(-2.0f * logf(f1)) * cosf(6.28318530718f * f2);
}

static float *make_rand_weights(size_t n, float std) {
    float *w = (float*)malloc(n * sizeof(float));
    if (!w) return NULL;
    for (size_t i = 0; i < n; i++) w[i] = rng_normal() * std;
    return w;
}

/* ── Patch extraction: image NCHW → sequence [N × (P²·C)] ─────────────── */
static float *extract_patches(const DM_Tensor *img, int P, int *out_N, int *out_patch_dim)
{
    int H = img->h, W = img->w, C = img->c;
    int gh = H / P, gw = W / P;
    int N = gh * gw;
    int pd = P * P * C;
    *out_N = N; *out_patch_dim = pd;
    float *patches = (float*)malloc((size_t)N * pd * sizeof(float));
    if (!patches) return NULL;
    for (int r = 0; r < gh; r++) {
        for (int c = 0; c < gw; c++) {
            float *dst = patches + (size_t)(r * gw + c) * pd;
            int idx = 0;
            for (int ch = 0; ch < C; ch++)
                for (int py = 0; py < P; py++)
                    for (int px = 0; px < P; px++)
                        dst[idx++] = dm_tensor_get(img, 0, ch, r*P+py, c*P+px);
        }
    }
    return patches;
}

/* ── Single MHSA block (§3.1, Eq.2, Appendix A) ────────────────────────── */
static int mhsa_forward(float *x,           /* [seq × D] in-place out    */
                        float *buf,          /* scratch [seq × 3D + ...]  */
                        int seq, int D, int h,
                        const float *qkv_w, const float *qkv_b,
                        const float *proj_w, const float *proj_b)
{
    int head_dim = D / h;
    float scale = 1.0f / sqrtf((float)head_dim);
    /* QKV = x @ qkv_w^T + qkv_b   → [seq × 3D] */
    float *qkv = buf;
    dm_matmul_nt(x, qkv_w, qkv, seq, 3*D, D);
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < 3*D; j++)
            qkv[(size_t)i*3*D + j] += qkv_b[j];

    /* attn_out = concat over heads of softmax(Qh @ Kh^T / √d) @ Vh */
    float *attn_out = buf + (size_t)seq * 3*D;
    memset(attn_out, 0, (size_t)seq * D * sizeof(float));

    /* scratch for one head's scores [seq × seq] */
    float *scores = attn_out + (size_t)seq * D;

    for (int hi = 0; hi < h; hi++) {
        int off = hi * head_dim;
        /* extract Q,K,V for this head */
        float *Q = scores + (size_t)seq * seq;   /* [seq × head_dim] */
        float *K = Q + (size_t)seq * head_dim;
        float *V = K + (size_t)seq * head_dim;
        for (int s = 0; s < seq; s++) {
            const float *qkvrow = qkv + (size_t)s * 3*D;
            memcpy(Q + (size_t)s*head_dim, qkvrow + off,          head_dim*sizeof(float));
            memcpy(K + (size_t)s*head_dim, qkvrow + D   + off,    head_dim*sizeof(float));
            memcpy(V + (size_t)s*head_dim, qkvrow + 2*D + off,    head_dim*sizeof(float));
        }
        /* scores = Q @ K^T * scale  [seq × seq] */
        dm_matmul_nt(Q, K, scores, seq, seq, head_dim);
        for (int i = 0; i < seq*seq; i++) scores[i] *= scale;
        dm_softmax_rows(scores, seq, seq);
        /* head_out = scores @ V  [seq × head_dim] → accumulate into attn_out */
        float *ho = K; /* reuse K scratch */
        dm_matmul_nn(scores, V, ho, seq, seq, head_dim);
        for (int s = 0; s < seq; s++)
            memcpy(attn_out + (size_t)s*D + off, ho + (size_t)s*head_dim, head_dim*sizeof(float));
    }
    /* output projection: x = attn_out @ proj_w^T + proj_b  [seq × D] */
    dm_matmul_nt(attn_out, proj_w, x, seq, D, D);
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < D; j++)
            x[(size_t)i*D + j] += proj_b[j];
    return 0;
}

/* ── MLP block (§3.1, Eq.3): two linear + GELU ─────────────────────────── */
static int mlp_forward(float *x, float *hidden,
                       int seq, int D, int M,
                       const float *fc1_w, const float *fc1_b,
                       const float *fc2_w, const float *fc2_b)
{
    /* hidden = x @ fc1_w^T + fc1_b  [seq × M] */
    dm_matmul_nt(x, fc1_w, hidden, seq, M, D);
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < M; j++)
            hidden[(size_t)i*M + j] += fc1_b[j];
    dm_gelu_inplace(hidden, seq * M);
    /* x = hidden @ fc2_w^T + fc2_b  [seq × D] */
    dm_matmul_nt(hidden, fc2_w, x, seq, D, M);
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < D; j++)
            x[(size_t)i*D + j] += fc2_b[j];
    return 0;
}

/* ── Full forward pass ──────────────────────────────────────────────────── */
int dm_vit_forward(const DM_Tensor *input, DM_Tensor *logits,
                   const ViTConfig *cfg, const float *weights,
                   unsigned int seed)
{
    if (!input || !logits || !cfg) return -1;

    const int P  = cfg->patch_size;
    const int D  = cfg->d_model;
    const int M  = cfg->mlp_dim;
    const int L  = cfg->num_layers;
    const int h  = cfg->num_heads;
    const int K  = cfg->num_classes;
    const int C  = 3;
    const int pd = P * P * C;

    int N, patch_dim;
    float *patches = extract_patches(input, P, &N, &patch_dim);
    if (!patches) return -1;
    int seq = N + 1; /* +1 for [CLS] */

    /* ── Possibly generate random weights ─────────────────────────────── */
    float *rw = NULL;
    if (!weights) {
        rng_state = seed ? seed : 2463534242u;
        size_t total = dm_vit_param_count(cfg);
        float std = 0.02f; /* ViT paper uses 0.02 init std */
        rw = make_rand_weights(total, std);
        if (!rw) { free(patches); return -1; }
        weights = rw;
    }
    WCur wc = {weights};

    /* ── Load static weights ─────────────────────────────────────────── */
    const float *patch_proj_w = wcur_take(&wc, (size_t)D * pd);
    const float *patch_proj_b = wcur_take(&wc, D);
    const float *cls_token    = wcur_take(&wc, D);
    const float *pos_embed    = wcur_take(&wc, (size_t)(N+1) * D);

    /* ── Allocate sequence buffer z[seq × D] ─────────────────────────── */
    float *z = (float*)malloc((size_t)seq * D * sizeof(float));
    if (!z) { free(patches); free(rw); return -1; }

    /* ── §3.1 Eq.1: Patch embedding + [CLS] + position embed ────────── */
    /* z[0] = cls_token */
    memcpy(z, cls_token, D * sizeof(float));
    /* z[1..N] = patches @ patch_proj_w^T + patch_proj_b */
    dm_matmul_nt(patches, patch_proj_w, z + D, N, D, pd);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            z[(size_t)(i+1)*D + j] += patch_proj_b[j];
    free(patches);
    /* add position embeddings */
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < D; j++)
            z[(size_t)i*D + j] += pos_embed[(size_t)i*D + j];

    /* ── Scratch buffers ─────────────────────────────────────────────── */
    /* mhsa needs: [seq×3D] qkv + [seq×D] attn_out + [seq×seq] scores
     *             + [seq×hd] Q,K,V  per head  (worst case 3*seq*hd = seq*D extra)
     * We allocate one large block: seq*(3D + D + seq + 3*D) */
    int head_dim = D / h;
    size_t mhsa_scratch = (size_t)seq * (3*D + D + seq + 3*head_dim);
    float *mhsa_buf = (float*)malloc(mhsa_scratch * sizeof(float));
    float *mlp_hidden = (float*)malloc((size_t)seq * M * sizeof(float));
    float *z_res = (float*)malloc((size_t)seq * D * sizeof(float));
    if (!mhsa_buf || !mlp_hidden || !z_res) {
        free(z); free(mhsa_buf); free(mlp_hidden); free(z_res);
        free(rw); return -1;
    }

    /* ── §3.1 Eq.2-3: L Transformer Encoder blocks ──────────────────── */
    for (int l = 0; l < L; l++) {
        const float *ln1_g  = wcur_take(&wc, D);
        const float *ln1_b  = wcur_take(&wc, D);
        const float *qkv_w  = wcur_take(&wc, (size_t)3*D*D);
        const float *qkv_b  = wcur_take(&wc, 3*D);
        const float *prj_w  = wcur_take(&wc, (size_t)D*D);
        const float *prj_b  = wcur_take(&wc, D);
        const float *ln2_g  = wcur_take(&wc, D);
        const float *ln2_b  = wcur_take(&wc, D);
        const float *fc1_w  = wcur_take(&wc, (size_t)M*D);
        const float *fc1_b  = wcur_take(&wc, M);
        const float *fc2_w  = wcur_take(&wc, (size_t)D*M);
        const float *fc2_b  = wcur_take(&wc, D);

        /* save residual */
        memcpy(z_res, z, (size_t)seq * D * sizeof(float));
        /* LN1 */
        dm_layer_norm_seq(z, seq, D, ln1_g, ln1_b, 1e-6f);
        /* MHSA */
        mhsa_forward(z, mhsa_buf, seq, D, h, qkv_w, qkv_b, prj_w, prj_b);
        /* residual (Eq.2) */
        for (size_t i = 0; i < (size_t)seq*D; i++) z[i] += z_res[i];

        /* save residual */
        memcpy(z_res, z, (size_t)seq * D * sizeof(float));
        /* LN2 */
        dm_layer_norm_seq(z, seq, D, ln2_g, ln2_b, 1e-6f);
        /* MLP */
        mlp_forward(z, mlp_hidden, seq, D, M, fc1_w, fc1_b, fc2_w, fc2_b);
        /* residual (Eq.3) */
        for (size_t i = 0; i < (size_t)seq*D; i++) z[i] += z_res[i];
    }

    /* ── §3.1 Eq.4: head LN + linear classifier on z[0] = [CLS] ─────── */
    const float *hln_g = wcur_take(&wc, D);
    const float *hln_b = wcur_take(&wc, D);
    const float *hw    = wcur_take(&wc, (size_t)K*D);
    const float *hb    = wcur_take(&wc, K);

    /* LN on CLS token only */
    dm_layer_norm_seq(z, 1, D, hln_g, hln_b, 1e-6f);

    /* logits = z[0:1] @ hw^T + hb  → [1 × K] */
    if (dm_tensor_alloc(logits, 1, K, 1, 1) != 0) {
        free(z); free(mhsa_buf); free(mlp_hidden); free(z_res); free(rw);
        return -1;
    }
    dm_matmul_nt(z, hw, logits->data, 1, K, D);
    for (int j = 0; j < K; j++) logits->data[j] += hb[j];

    free(z); free(mhsa_buf); free(mlp_hidden); free(z_res); free(rw);
    return 0;
}

int dm_vit_forward_variant(const DM_Tensor *input, DM_Tensor *logits,
                            ViTVariant v, int num_classes,
                            const float *weights, unsigned int seed)
{
    ViTConfig cfg;
    dm_vit_config_init(&cfg, v, num_classes,
                       input ? input->h : 224);
    return dm_vit_forward(input, logits, &cfg, weights, seed);
}

/* ── CLI ─────────────────────────────────────────────────────────────────── */
static void vit_usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s vit bench [--variant tiny|small|base|large|huge]\n"
        "               [--size N] [--classes N] [--seed N]\n\n",
        prog);
}

int dm_vit_cli(int argc, char **argv)
{
    if (argc < 3) { vit_usage(argv[0]); return 1; }
    if (strcmp(argv[2], "bench") != 0) { vit_usage(argv[0]); return 1; }

    ViTVariant var = VIT_SMALL;
    int size = 224, classes = 1000;
    unsigned int seed = 42;

    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--variant") == 0 && i+1 < argc) {
            const char *s = argv[++i];
            if      (!strcmp(s,"tiny"))  var = VIT_TINY;
            else if (!strcmp(s,"small")) var = VIT_SMALL;
            else if (!strcmp(s,"base"))  var = VIT_BASE;
            else if (!strcmp(s,"large")) var = VIT_LARGE;
            else if (!strcmp(s,"huge"))  var = VIT_HUGE;
        } else if (!strcmp(argv[i],"--size")    && i+1<argc) size    = atoi(argv[++i]);
        else if   (!strcmp(argv[i],"--classes") && i+1<argc) classes = atoi(argv[++i]);
        else if   (!strcmp(argv[i],"--seed")    && i+1<argc) seed    = (unsigned)atoi(argv[++i]);
    }

    ViTConfig cfg;
    dm_vit_config_init(&cfg, var, classes, size);

    static const char *vnames[] = {"tiny","small","base","large","huge"};
    printf("ViT-%s/%d  size=%dx%d  D=%d  L=%d  heads=%d  classes=%d  seed=%u\n",
           vnames[var], cfg.patch_size, size, size,
           cfg.d_model, cfg.num_layers, cfg.num_heads, classes, seed);
    printf("Total params: %zu\n", dm_vit_param_count(&cfg));

    DM_Tensor in = {0}, out = {0};
    if (dm_tensor_alloc(&in, 1, 3, size, size) != 0) {
        fprintf(stderr, "alloc failed\n"); return 1;
    }
    dm_tensor_fill(&in, 1.0f);

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    dm_bench_start(DM_PHASE_ALGO);
    int rc = dm_vit_forward(&in, &out, &cfg, NULL, seed);
    dm_bench_stop(DM_PHASE_ALGO);
    dm_bench_stop(DM_PHASE_TOTAL);

    if (rc == 0) {
        printf("Raw logit range: [%.4f, %.4f]\n", out.data[0], out.data[classes-1]);
        dm_softmax(&out);
        /* top-5 */
        printf("Top-5 predictions:\n");
        for (int r = 0; r < 5 && r < classes; r++) {
            int best = -1; float bv = -1e30f;
            for (int j = 0; j < classes; j++)
                if (out.data[j] > bv) { bv = out.data[j]; best = j; }
            printf("  rank %d  class %4d  prob %.6f\n", r+1, best, bv);
            out.data[best] = -1e30f;
        }
        dm_bench_print_report("vit", "synthetic");
    } else {
        fprintf(stderr, "Forward pass failed\n");
    }

    dm_tensor_free(&in);
    dm_tensor_free(&out);
    return rc == 0 ? 0 : 1;
}
