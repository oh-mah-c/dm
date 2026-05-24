code = """/*
 * DM_NCHW_C(&vit) — Vision Transformer (ViT), Dosovitskiy et al., ICLR 2021
 * arXiv:2010.11929v2
 *
 * Modified for Phase 5.5.2 to natively use DM_Block and DM_WeightCache.
 */

#include "models/vision/vit.h"
#include "core/dm_benchmark.h"
#include "core/dm_engine.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int VIT_CFG[5][5] = {
    { 5,  192,  768,  3, 16},
    { 6,  384, 1536,  6, 16},
    {12,  768, 3072, 12, 16},
    {24, 1024, 4096, 16, 16},
    {32, 1280, 5120, 16, 14},
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

size_t dm_vit_param_count(const ViTConfig *cfg)
{
    int P = cfg->patch_size, C = 3;
    int D = cfg->d_model, M = cfg->mlp_dim, L = cfg->num_layers;
    int N = (cfg->img_size / P) * (cfg->img_size / P);
    int K = cfg->num_classes;
    size_t n = 0;
    n += (size_t)D * (P * P * C) + D;
    n += D + (size_t)(N + 1) * D;
    n += (size_t)L * (2*D + 3*D*D + 3*D + D*D + D + 2*D + M*D + M + D*M + D);
    n += 2*D + (size_t)K*D + K;
    return n;
}

static int mhsa_forward(DM_WeightCache *cache, DM_Block *x, unsigned int seed, int l, int seq, int D, int h)
{
    int head_dim = D / h;
    float scale = 1.0f / sqrtf((float)head_dim);

    DM_Block *qkv_w = dm_weight_cache_get(cache, 2, (int64_t[]){3*D, D}, seed + l*100 + 1, 0.05f);
    DM_Block *qkv_b = dm_weight_cache_get(cache, 1, (int64_t[]){3*D}, seed + l*100 + 2, 0.00f);
    DM_Block *proj_w = dm_weight_cache_get(cache, 2, (int64_t[]){D, D}, seed + l*100 + 3, 0.05f);
    DM_Block *proj_b = dm_weight_cache_get(cache, 1, (int64_t[]){D}, seed + l*100 + 4, 0.00f);

    DM_Block qkv = {0};
    dm_block_create(&qkv, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){seq, 3*D});
    
    dm_matmul_nt(x, qkv_w, &qkv);
    float *q_ptr = (float*)qkv.data;
    float *qb_ptr = (float*)qkv_b->data;
    for (int i=0; i<seq; i++)
        for (int j=0; j<3*D; j++)
            q_ptr[i*3*D + j] += qb_ptr[j];

    DM_Block attn_out = {0};
    dm_block_create(&attn_out, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){seq, D});
    memset(attn_out.data, 0, (size_t)seq * D * sizeof(float));

    DM_Block blk_Q = {0}, blk_K = {0}, blk_V = {0}, blk_scores = {0}, blk_ho = {0};
    dm_block_create(&blk_Q, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){seq, head_dim});
    dm_block_create(&blk_K, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){seq, head_dim});
    dm_block_create(&blk_V, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){seq, head_dim});
    dm_block_create(&blk_scores, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){seq, seq});
    dm_block_create(&blk_ho, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){seq, head_dim});

    for (int hi = 0; hi < h; hi++) {
        int off = hi * head_dim;
        float *Q = (float*)blk_Q.data;
        float *K = (float*)blk_K.data;
        float *V = (float*)blk_V.data;
        
        for (int s = 0; s < seq; s++) {
            const float *qkvrow = q_ptr + (size_t)s * 3*D;
            memcpy(Q + (size_t)s*head_dim, qkvrow + off,          head_dim*sizeof(float));
            memcpy(K + (size_t)s*head_dim, qkvrow + D   + off,    head_dim*sizeof(float));
            memcpy(V + (size_t)s*head_dim, qkvrow + 2*D + off,    head_dim*sizeof(float));
        }

        dm_matmul_nt(&blk_Q, &blk_K, &blk_scores);
        float *scores = (float*)blk_scores.data;
        for (int i = 0; i < seq*seq; i++) scores[i] *= scale;
        
        dm_softmax_last_dim(&blk_scores);

        dm_matmul_nn(&blk_scores, &blk_V, &blk_ho);
        float *ho = (float*)blk_ho.data;
        
        float *ao_ptr = (float*)attn_out.data;
        for (int s = 0; s < seq; s++)
            memcpy(ao_ptr + (size_t)s*D + off, ho + (size_t)s*head_dim, head_dim*sizeof(float));
    }

    dm_matmul_nt(&attn_out, proj_w, x);
    float *x_ptr = (float*)x->data;
    float *pb_ptr = (float*)proj_b->data;
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < D; j++)
            x_ptr[(size_t)i*D + j] += pb_ptr[j];

    dm_block_free(&qkv);
    dm_block_free(&attn_out);
    dm_block_free(&blk_Q);
    dm_block_free(&blk_K);
    dm_block_free(&blk_V);
    dm_block_free(&blk_scores);
    dm_block_free(&blk_ho);
    return 0;
}

static int mlp_forward(DM_WeightCache *cache, DM_Block *x, unsigned int seed, int l, int seq, int D, int M)
{
    DM_Block *fc1_w = dm_weight_cache_get(cache, 2, (int64_t[]){M, D}, seed + l*100 + 5, 0.05f);
    DM_Block *fc1_b = dm_weight_cache_get(cache, 1, (int64_t[]){M}, seed + l*100 + 6, 0.00f);
    DM_Block *fc2_w = dm_weight_cache_get(cache, 2, (int64_t[]){D, M}, seed + l*100 + 7, 0.05f);
    DM_Block *fc2_b = dm_weight_cache_get(cache, 1, (int64_t[]){D}, seed + l*100 + 8, 0.00f);

    DM_Block hidden = {0};
    dm_block_create(&hidden, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){seq, M});

    dm_matmul_nt(x, fc1_w, &hidden);
    float *h_ptr = (float*)hidden.data;
    float *b1_ptr = (float*)fc1_b->data;
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < M; j++)
            h_ptr[(size_t)i*M + j] += b1_ptr[j];

    dm_gelu_inplace(h_ptr, seq * M);

    dm_matmul_nt(&hidden, fc2_w, x);
    float *x_ptr = (float*)x->data;
    float *b2_ptr = (float*)fc2_b->data;
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < D; j++)
            x_ptr[(size_t)i*D + j] += b2_ptr[j];

    dm_block_free(&hidden);
    return 0;
}

int dm_vit_forward(const DM_Block *input, DM_Block *logits,
                   const ViTConfig *cfg, const float *weights,
                   unsigned int seed)
{
    if (!input || !logits || !cfg) return -1;

    DM_WeightCache *cache = dm_weight_cache_new();

    const int P  = cfg->patch_size;
    const int D  = cfg->d_model;
    const int M  = cfg->mlp_dim;
    const int L  = cfg->num_layers;
    const int h  = cfg->num_heads;
    const int K  = cfg->num_classes;
    const int C  = 3;
    const int pd = P * P * C;

    int H = DM_NCHW_H(input), W = DM_NCHW_W(input);
    int gh = H / P, gw = W / P;
    int N = gh * gw;
    int seq = N + 1; /* +1 for [CLS] */

    DM_Block *patch_proj_w = dm_weight_cache_get(cache, 2, (int64_t[]){D, pd}, seed + 10, 0.05f);
    DM_Block *patch_proj_b = dm_weight_cache_get(cache, 1, (int64_t[]){D}, seed + 11, 0.00f);
    DM_Block *cls_token    = dm_weight_cache_get(cache, 1, (int64_t[]){D}, seed + 12, 0.05f);
    DM_Block *pos_embed    = dm_weight_cache_get(cache, 2, (int64_t[]){seq, D}, seed + 13, 0.02f);

    /* Allocate sequence buffer z[seq × D] */
    DM_Block z = {0}, z_res = {0};
    dm_block_create(&z, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){seq, D});
    dm_block_create(&z_res, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){seq, D});
    float *z_ptr = (float*)z.data;

    /* z[0] = cls_token */
    memcpy(z_ptr, cls_token->data, D * sizeof(float));

    /* Extract patches using DM_Block */
    DM_Block patches = {0};
    dm_block_create(&patches, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){N, pd});
    float *p_ptr = (float*)patches.data;
    for (int r = 0; r < gh; r++) {
        for (int c = 0; c < gw; c++) {
            float *dst = p_ptr + (size_t)(r * gw + c) * pd;
            int idx = 0;
            for (int ch = 0; ch < C; ch++)
                for (int py = 0; py < P; py++)
                    for (int px = 0; px < P; px++)
                        dst[idx++] = dm_tensor_get(input, 0, ch, r*P+py, c*P+px);
        }
    }

    DM_Block patches_proj = {0};
    dm_block_create(&patches_proj, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){N, D});
    dm_matmul_nt(&patches, patch_proj_w, &patches_proj);
    
    float *pp_ptr = (float*)patches_proj.data;
    float *pb_ptr = (float*)patch_proj_b->data;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            z_ptr[(size_t)(i+1)*D + j] = pp_ptr[(size_t)i*D + j] + pb_ptr[j];
            
    dm_block_free(&patches);
    dm_block_free(&patches_proj);

    float *pe_ptr = (float*)pos_embed->data;
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < D; j++)
            z_ptr[(size_t)i*D + j] += pe_ptr[(size_t)i*D + j];

    for (int l = 0; l < L; l++) {
        DM_Block *ln1_g = dm_weight_cache_get(cache, 1, (int64_t[]){D}, seed + l*100 + 20, 1.0f);
        DM_Block *ln1_b = dm_weight_cache_get(cache, 1, (int64_t[]){D}, seed + l*100 + 21, 0.0f);
        DM_Block *ln2_g = dm_weight_cache_get(cache, 1, (int64_t[]){D}, seed + l*100 + 22, 1.0f);
        DM_Block *ln2_b = dm_weight_cache_get(cache, 1, (int64_t[]){D}, seed + l*100 + 23, 0.0f);

        memcpy(z_res.data, z.data, (size_t)seq * D * sizeof(float));
        dm_layer_norm_seq(&z, ln1_g, ln1_b, 1e-6f);
        mhsa_forward(cache, &z, seed, l, seq, D, h);
        for (size_t i = 0; i < (size_t)seq*D; i++) z_ptr[i] += ((float*)z_res.data)[i];

        memcpy(z_res.data, z.data, (size_t)seq * D * sizeof(float));
        dm_layer_norm_seq(&z, ln2_g, ln2_b, 1e-6f);
        mlp_forward(cache, &z, seed, l, seq, D, M);
        for (size_t i = 0; i < (size_t)seq*D; i++) z_ptr[i] += ((float*)z_res.data)[i];
    }

    DM_Block *hln_g = dm_weight_cache_get(cache, 1, (int64_t[]){D}, seed + 999, 1.0f);
    DM_Block *hln_b = dm_weight_cache_get(cache, 1, (int64_t[]){D}, seed + 1000, 0.0f);
    DM_Block *hw    = dm_weight_cache_get(cache, 2, (int64_t[]){K, D}, seed + 1001, 0.05f);
    DM_Block *hb    = dm_weight_cache_get(cache, 1, (int64_t[]){K}, seed + 1002, 0.0f);

    /* LN on CLS token only */
    DM_Block z_cls = {0};
    dm_block_create(&z_cls, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 2, (int64_t[]){1, D});
    memcpy(z_cls.data, z.data, D * sizeof(float));
    dm_layer_norm_seq(&z_cls, hln_g, hln_b, 1e-6f);

    if (dm_block_create(logits, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){1, K, 1, 1}) != 0) {
        dm_block_free(&z); dm_block_free(&z_res); dm_block_free(&z_cls);
        dm_weight_cache_free(cache);
        return -1;
    }
    
    dm_matmul_nt(&z_cls, hw, logits);
    float *logits_ptr = (float*)logits->data;
    float *hb_ptr = (float*)hb->data;
    for (int j = 0; j < K; j++) logits_ptr[j] += hb_ptr[j];

    dm_block_free(&z);
    dm_block_free(&z_res);
    dm_block_free(&z_cls);
    dm_weight_cache_free(cache);
    return 0;
}

int dm_vit_forward_variant(const DM_Block *input, DM_Block *logits,
                            ViTVariant v, int num_classes,
                            const float *weights, unsigned int seed)
{
    ViTConfig cfg;
    dm_vit_config_init(&cfg, v, num_classes,
                       input ? DM_NCHW_H(input) : 224);
    return dm_vit_forward(input, logits, &cfg, weights, seed);
}

static void vit_usage(const char *prog) {
    fprintf(stderr,
        "Usage:\\n"
        "  %s vit bench [--variant tiny|small|base|large|huge]\\n"
        "               [--size N] [--classes N] [--seed N]\\n\\n",
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
    printf("ViT-%s/%d  size=%dx%d  D=%d  L=%d  heads=%d  classes=%d  seed=%u\\n",
           vnames[var], cfg.patch_size, size, size,
           cfg.d_model, cfg.num_layers, cfg.num_heads, classes, seed);
    printf("Total params: %zu\\n", dm_vit_param_count(&cfg));

    DM_Block in = {0}, out = {0};
    if (dm_block_create(&in, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){1, 3, size, size}) != 0) {
        fprintf(stderr, "alloc failed\\n"); return 1;
    }
    { size_t __n = (&in)->count; float *__d = (float*)(&in)->data; for(size_t __i=0; __i<__n; __i++) __d[__i] = 1.0f; }

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    dm_bench_start(DM_PHASE_ALGO);
    int rc = dm_vit_forward(&in, &out, &cfg, NULL, seed);
    dm_bench_stop(DM_PHASE_ALGO);
    dm_bench_stop(DM_PHASE_TOTAL);

    if (rc == 0) {
        printf("Raw logit range: [%.4f, %.4f]\\n", ((float*)((float*)out.data))[0], ((float*)((float*)out.data))[classes-1]);
        dm_softmax(&out);
        printf("Top-5 predictions:\\n");
        for (int r = 0; r < 5 && r < classes; r++) {
            int best = -1; float bv = -1e30f;
            for (int j = 0; j < classes; j++)
                if (((float*)((float*)out.data))[j] > bv) { bv = ((float*)((float*)out.data))[j]; best = j; }
            printf("  rank %d  class %4d  prob %.6f\\n", r+1, best, bv);
            ((float*)((float*)out.data))[best] = -1e30f;
        }
        dm_bench_print_report("vit", "synthetic");
    } else {
        fprintf(stderr, "Forward pass failed\\n");
    }

    dm_block_free(&in);
    dm_block_free(&out);
    return rc == 0 ? 0 : 1;
}
"""

with open("src/models/vision/vit.c", "w") as f:
    f.write(code)

