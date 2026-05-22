/*
 * bert.c - BERT encoder implementation from Devlin et al., NAACL-HLT 2019.
 *
 * This file implements the architecture described in N19-1423:
 * learned token/segment/position embeddings, bidirectional Transformer
 * encoder layers with GELU feed-forward blocks, and the tanh [CLS] pooler.
 */

#include "models/lm/bert.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BERT_LN_EPS 1.0e-12f

typedef struct {
    const float *data;
    size_t pos;
} BertWBuf;

typedef struct {
    const float *W_Q, *b_Q;
    const float *W_K, *b_K;
    const float *W_V, *b_V;
    const float *W_O, *b_O;
} BertMHAWeights;

typedef struct {
    const float *gamma;
    const float *beta;
} BertLNWeights;

typedef struct {
    const float *W_1, *b_1;
    const float *W_2, *b_2;
} BertFFNWeights;

static const float *bw_next(BertWBuf *b, size_t n)
{
    const float *p = b->data ? b->data + b->pos : NULL;
    b->pos += n;
    return p;
}

static BertMHAWeights read_mha(BertWBuf *b, int h)
{
    BertMHAWeights w;
    w.W_Q = bw_next(b, (size_t)h * h); w.b_Q = bw_next(b, (size_t)h);
    w.W_K = bw_next(b, (size_t)h * h); w.b_K = bw_next(b, (size_t)h);
    w.W_V = bw_next(b, (size_t)h * h); w.b_V = bw_next(b, (size_t)h);
    w.W_O = bw_next(b, (size_t)h * h); w.b_O = bw_next(b, (size_t)h);
    return w;
}

static BertLNWeights read_ln(BertWBuf *b, int h)
{
    BertLNWeights w;
    w.gamma = bw_next(b, (size_t)h);
    w.beta = bw_next(b, (size_t)h);
    return w;
}

static BertFFNWeights read_ffn(BertWBuf *b, int h, int f)
{
    BertFFNWeights w;
    w.W_1 = bw_next(b, (size_t)h * f); w.b_1 = bw_next(b, (size_t)f);
    w.W_2 = bw_next(b, (size_t)f * h); w.b_2 = bw_next(b, (size_t)h);
    return w;
}

static void read_layer(BertWBuf *b, const BertConfig *cfg)
{
    read_mha(b, cfg->hidden_size);
    read_ln(b, cfg->hidden_size);
    read_ffn(b, cfg->hidden_size, cfg->intermediate);
    read_ln(b, cfg->hidden_size);
}

static int valid_cfg(const BertConfig *cfg)
{
    return cfg &&
           cfg->num_layers > 0 &&
           cfg->hidden_size > 0 &&
           cfg->num_heads > 0 &&
           cfg->intermediate > 0 &&
           cfg->vocab_size > 0 &&
           cfg->max_seq_len > 0 &&
           cfg->num_seg_types > 0 &&
           (cfg->hidden_size % cfg->num_heads) == 0;
}

static inline float gelu(float x)
{
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

static void linear(const float *x, const float *W, const float *b,
                   int rows, int in_dim, int out_dim, float *out)
{
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * in_dim;
        float *yr = out + (size_t)r * out_dim;
        for (int o = 0; o < out_dim; o++) {
            double acc = b ? (double)b[o] : 0.0;
            const float *wr = W + (size_t)o * in_dim;
            for (int i = 0; i < in_dim; i++) acc += (double)xr[i] * wr[i];
            yr[o] = (float)acc;
        }
    }
}

static void layer_norm(float *x, const float *gamma, const float *beta,
                       int rows, int d)
{
    for (int r = 0; r < rows; r++) {
        float *row = x + (size_t)r * d;
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < d; i++) mean += row[i];
        mean /= (double)d;
        for (int i = 0; i < d; i++) {
            double z = (double)row[i] - mean;
            var += z * z;
        }
        var /= (double)d;
        float inv = 1.0f / sqrtf((float)var + BERT_LN_EPS);
        for (int i = 0; i < d; i++) {
            float z = ((float)((double)row[i] - mean)) * inv;
            row[i] = z * (gamma ? gamma[i] : 1.0f) + (beta ? beta[i] : 0.0f);
        }
    }
}

static void softmax(float *x, int n)
{
    float mx = -FLT_MAX;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) if (x[i] > mx) mx = x[i];
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    if (sum <= 0.0f) sum = 1.0f;
    for (int i = 0; i < n; i++) x[i] /= sum;
}

static int bert_mha(const BertMHAWeights *w, const float *x,
                    const int *attention_mask, int seq,
                    int hidden, int heads, float *out)
{
    int head_dim = hidden / heads;
    size_t sh = (size_t)seq * hidden;
    float *Q = (float *)malloc(sh * sizeof(float));
    float *K = (float *)malloc(sh * sizeof(float));
    float *V = (float *)malloc(sh * sizeof(float));
    float *ctx = (float *)calloc(sh, sizeof(float));
    float *scores = (float *)malloc((size_t)seq * seq * sizeof(float));
    if (!Q || !K || !V || !ctx || !scores) {
        free(Q); free(K); free(V); free(ctx); free(scores);
        return -1;
    }

    linear(x, w->W_Q, w->b_Q, seq, hidden, hidden, Q);
    linear(x, w->W_K, w->b_K, seq, hidden, hidden, K);
    linear(x, w->W_V, w->b_V, seq, hidden, hidden, V);

    float scale = 1.0f / sqrtf((float)head_dim);
    for (int head = 0; head < heads; head++) {
        int off = head * head_dim;
        for (int qi = 0; qi < seq; qi++) {
            float *row = scores + (size_t)qi * seq;
            for (int kj = 0; kj < seq; kj++) {
                if (attention_mask && attention_mask[kj] == 0) {
                    row[kj] = -1.0e9f;
                    continue;
                }
                double acc = 0.0;
                const float *qv = Q + (size_t)qi * hidden + off;
                const float *kv = K + (size_t)kj * hidden + off;
                for (int d = 0; d < head_dim; d++) acc += (double)qv[d] * kv[d];
                row[kj] = (float)acc * scale;
            }
            softmax(row, seq);
        }
        for (int qi = 0; qi < seq; qi++) {
            float *dst = ctx + (size_t)qi * hidden + off;
            for (int d = 0; d < head_dim; d++) {
                double acc = 0.0;
                for (int kj = 0; kj < seq; kj++) {
                    acc += (double)scores[(size_t)qi * seq + kj] *
                           V[(size_t)kj * hidden + off + d];
                }
                dst[d] = (float)acc;
            }
        }
    }

    linear(ctx, w->W_O, w->b_O, seq, hidden, hidden, out);

    free(Q); free(K); free(V); free(ctx); free(scores);
    return 0;
}

static int bert_ffn(const BertFFNWeights *w, const float *x, int seq,
                    int hidden, int intermediate, float *out)
{
    float *mid = (float *)malloc((size_t)seq * intermediate * sizeof(float));
    if (!mid) return -1;
    linear(x, w->W_1, w->b_1, seq, hidden, intermediate, mid);
    for (int i = 0; i < seq * intermediate; i++) mid[i] = gelu(mid[i]);
    linear(mid, w->W_2, w->b_2, seq, intermediate, hidden, out);
    free(mid);
    return 0;
}

static int bert_layer(const BertConfig *cfg, BertWBuf *wb,
                      const int *attention_mask,
                      const float *x, int seq, float *out)
{
    int h = cfg->hidden_size;
    size_t n = (size_t)seq * h;
    BertMHAWeights mha = read_mha(wb, h);
    BertLNWeights ln1 = read_ln(wb, h);
    BertFFNWeights ffn = read_ffn(wb, h, cfg->intermediate);
    BertLNWeights ln2 = read_ln(wb, h);
    float *attn = (float *)malloc(n * sizeof(float));
    float *sub = (float *)malloc(n * sizeof(float));
    float *ff = (float *)malloc(n * sizeof(float));
    if (!attn || !sub || !ff) {
        free(attn); free(sub); free(ff);
        return -1;
    }
    if (bert_mha(&mha, x, attention_mask, seq, h, cfg->num_heads, attn) != 0) {
        free(attn); free(sub); free(ff);
        return -1;
    }
    for (size_t i = 0; i < n; i++) sub[i] = x[i] + attn[i];
    layer_norm(sub, ln1.gamma, ln1.beta, seq, h);
    if (bert_ffn(&ffn, sub, seq, h, cfg->intermediate, ff) != 0) {
        free(attn); free(sub); free(ff);
        return -1;
    }
    for (size_t i = 0; i < n; i++) out[i] = sub[i] + ff[i];
    layer_norm(out, ln2.gamma, ln2.beta, seq, h);
    free(attn); free(sub); free(ff);
    return 0;
}

void dm_bert_config_init(BertConfig *cfg, BertVariant variant,
                         int vocab_size, int max_seq_len)
{
    if (!cfg) return;
    memset(cfg, 0, sizeof(*cfg));
    cfg->variant = variant;
    cfg->vocab_size = vocab_size > 0 ? vocab_size : 30522;
    cfg->max_seq_len = max_seq_len > 0 ? max_seq_len : 512;
    cfg->num_seg_types = 2;
    cfg->dropout = 0.1f;
    if (variant == BERT_LARGE) {
        cfg->num_layers = 24;
        cfg->hidden_size = 1024;
        cfg->num_heads = 16;
        cfg->intermediate = 4096;
    } else {
        cfg->variant = BERT_BASE;
        cfg->num_layers = 12;
        cfg->hidden_size = 768;
        cfg->num_heads = 12;
        cfg->intermediate = 3072;
    }
}

size_t dm_bert_weight_count(const BertConfig *cfg)
{
    if (!valid_cfg(cfg)) return 0;
    BertWBuf wb;
    wb.data = NULL;
    wb.pos = 0;
    bw_next(&wb, (size_t)cfg->vocab_size * cfg->hidden_size);
    bw_next(&wb, (size_t)cfg->num_seg_types * cfg->hidden_size);
    bw_next(&wb, (size_t)cfg->max_seq_len * cfg->hidden_size);
    read_ln(&wb, cfg->hidden_size);
    for (int i = 0; i < cfg->num_layers; i++) read_layer(&wb, cfg);
    bw_next(&wb, (size_t)cfg->hidden_size * cfg->hidden_size);
    bw_next(&wb, (size_t)cfg->hidden_size);
    return wb.pos;
}

int dm_bert_save(const char *path, const BertConfig *cfg, const float *weights)
{
    if (!path || !valid_cfg(cfg) || !weights) return -1;
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    uint32_t magic = DM_BERT_WEIGHT_MAGIC;
    uint32_t ver = DM_BERT_WEIGHT_VER;
    uint32_t variant = (uint32_t)cfg->variant;
    uint32_t vocab = (uint32_t)cfg->vocab_size;
    uint32_t max_len = (uint32_t)cfg->max_seq_len;
    uint64_t wc = (uint64_t)dm_bert_weight_count(cfg);
    int ok = fwrite(&magic, sizeof(magic), 1, fp) == 1 &&
             fwrite(&ver, sizeof(ver), 1, fp) == 1 &&
             fwrite(&variant, sizeof(variant), 1, fp) == 1 &&
             fwrite(&vocab, sizeof(vocab), 1, fp) == 1 &&
             fwrite(&max_len, sizeof(max_len), 1, fp) == 1 &&
             fwrite(&wc, sizeof(wc), 1, fp) == 1 &&
             fwrite(weights, sizeof(float), (size_t)wc, fp) == (size_t)wc;
    fclose(fp);
    return ok ? 0 : -1;
}

int dm_bert_load(const char *path, BertConfig *cfg, float **weights_out)
{
    if (!path || !cfg || !weights_out) return -1;
    *weights_out = NULL;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    uint32_t magic, ver, variant, vocab, max_len;
    uint64_t wc;
    if (fread(&magic, sizeof(magic), 1, fp) != 1 ||
        fread(&ver, sizeof(ver), 1, fp) != 1 ||
        fread(&variant, sizeof(variant), 1, fp) != 1 ||
        fread(&vocab, sizeof(vocab), 1, fp) != 1 ||
        fread(&max_len, sizeof(max_len), 1, fp) != 1 ||
        fread(&wc, sizeof(wc), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    if (magic != DM_BERT_WEIGHT_MAGIC || ver != DM_BERT_WEIGHT_VER ||
        (variant != (uint32_t)BERT_BASE && variant != (uint32_t)BERT_LARGE)) {
        fclose(fp);
        return -1;
    }
    dm_bert_config_init(cfg, (BertVariant)variant, (int)vocab, (int)max_len);
    if ((uint64_t)dm_bert_weight_count(cfg) != wc) {
        fclose(fp);
        return -1;
    }
    float *w = (float *)malloc((size_t)wc * sizeof(float));
    if (!w) {
        fclose(fp);
        return -1;
    }
    if (fread(w, sizeof(float), (size_t)wc, fp) != (size_t)wc) {
        free(w);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    *weights_out = w;
    return 0;
}

int dm_bert_forward_all_layers_masked(const BertConfig *cfg, const float *weights,
                                      const int *token_ids, const int *segment_ids,
                                      const int *attention_mask, int seq,
                                      float *all_hidden_out, float *cls_out)
{
    if (!valid_cfg(cfg) || !weights || !token_ids || !all_hidden_out ||
        seq <= 0 || seq > cfg->max_seq_len) return -1;
    int h = cfg->hidden_size;
    size_t sh = (size_t)seq * h;
    BertWBuf wb;
    wb.data = weights;
    wb.pos = 0;
    const float *tok_emb = bw_next(&wb, (size_t)cfg->vocab_size * h);
    const float *seg_emb = bw_next(&wb, (size_t)cfg->num_seg_types * h);
    const float *pos_emb = bw_next(&wb, (size_t)cfg->max_seq_len * h);
    BertLNWeights emb_ln = read_ln(&wb, h);

    float *cur = (float *)malloc(sh * sizeof(float));
    float *next = (float *)malloc(sh * sizeof(float));
    if (!cur || !next) {
        free(cur); free(next);
        return -1;
    }
    for (int t = 0; t < seq; t++) {
        int tok = token_ids[t];
        int seg = segment_ids ? segment_ids[t] : 0;
        if (tok < 0 || tok >= cfg->vocab_size || seg < 0 || seg >= cfg->num_seg_types) {
            free(cur); free(next);
            return -1;
        }
        const float *te = tok_emb + (size_t)tok * h;
        const float *se = seg_emb + (size_t)seg * h;
        const float *pe = pos_emb + (size_t)t * h;
        float *dst = cur + (size_t)t * h;
        for (int d = 0; d < h; d++) dst[d] = te[d] + se[d] + pe[d];
    }
    layer_norm(cur, emb_ln.gamma, emb_ln.beta, seq, h);

    for (int layer = 0; layer < cfg->num_layers; layer++) {
        if (bert_layer(cfg, &wb, attention_mask, cur, seq, next) != 0) {
            free(cur); free(next);
            return -1;
        }
        memcpy(all_hidden_out + (size_t)layer * sh, next, sh * sizeof(float));
        float *tmp = cur;
        cur = next;
        next = tmp;
    }

    const float *W_pool = bw_next(&wb, (size_t)h * h);
    const float *b_pool = bw_next(&wb, (size_t)h);
    if (cls_out) {
        linear(cur, W_pool, b_pool, 1, h, h, cls_out);
        for (int d = 0; d < h; d++) cls_out[d] = tanhf(cls_out[d]);
    }
    free(cur);
    free(next);
    return 0;
}

int dm_bert_forward_all_layers(const BertConfig *cfg, const float *weights,
                               const int *token_ids, const int *segment_ids,
                               int seq, float *all_hidden_out, float *cls_out)
{
    return dm_bert_forward_all_layers_masked(cfg, weights, token_ids,
                                             segment_ids, NULL, seq,
                                             all_hidden_out, cls_out);
}

int dm_bert_forward_masked(const BertConfig *cfg, const float *weights,
                           const int *token_ids, const int *segment_ids,
                           const int *attention_mask, int seq,
                           float *hidden_out, float *cls_out)
{
    if (!valid_cfg(cfg) || !hidden_out) return -1;
    size_t sh = (size_t)seq * cfg->hidden_size;
    float *all = (float *)malloc((size_t)cfg->num_layers * sh * sizeof(float));
    if (!all) return -1;
    int rc = dm_bert_forward_all_layers_masked(cfg, weights, token_ids,
                                               segment_ids, attention_mask,
                                               seq, all, cls_out);
    if (rc == 0)
        memcpy(hidden_out, all + (size_t)(cfg->num_layers - 1) * sh,
               sh * sizeof(float));
    free(all);
    return rc;
}

int dm_bert_forward(const BertConfig *cfg, const float *weights,
                    const int *token_ids, const int *segment_ids, int seq,
                    float *hidden_out, float *cls_out)
{
    return dm_bert_forward_masked(cfg, weights, token_ids, segment_ids, NULL,
                                  seq, hidden_out, cls_out);
}

static int parse_variant(const char *s, BertVariant *v)
{
    if (!s || strcmp(s, "base") == 0 || strcmp(s, "BERT_BASE") == 0) {
        *v = BERT_BASE;
        return 0;
    }
    if (strcmp(s, "large") == 0 || strcmp(s, "BERT_LARGE") == 0) {
        *v = BERT_LARGE;
        return 0;
    }
    return -1;
}

static int parse_ids(const char *s, int **ids_out, int *n_out)
{
    int cap = 16, n = 0;
    int *ids = (int *)malloc((size_t)cap * sizeof(int));
    if (!ids) return -1;
    const char *p = s;
    while (*p) {
        char *end = NULL;
        errno = 0;
        long v = strtol(p, &end, 10);
        if (p == end) {
            p++;
            continue;
        }
        if (errno || v < 0 || v > INT32_MAX) {
            free(ids);
            return -1;
        }
        if (n == cap) {
            cap *= 2;
            int *tmp = (int *)realloc(ids, (size_t)cap * sizeof(int));
            if (!tmp) {
                free(ids);
                return -1;
            }
            ids = tmp;
        }
        ids[n++] = (int)v;
        p = end;
    }
    if (n == 0) {
        free(ids);
        return -1;
    }
    *ids_out = ids;
    *n_out = n;
    return 0;
}

static void usage(void)
{
    fprintf(stderr,
        "Usage:\n"
        "  bert info -m weights.bin\n"
        "  bert encode -m weights.bin --text \"101 7592 102\" [--all-layers]\n"
        "  bert bench [--variant base|large] [--seq-len N] [--vocab-size N]\n");
}

int dm_bert_cli(int argc, char **argv)
{
    if (argc < 2) {
        usage();
        return 1;
    }
    const char *cmd = argv[1];
    if (strcmp(cmd, "info") == 0) {
        const char *model = NULL;
        for (int i = 2; i < argc; i++)
            if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc)
                model = argv[++i];
        BertConfig cfg;
        float *w = NULL;
        if (!model || dm_bert_load(model, &cfg, &w) != 0) return 1;
        printf("BERT %s L=%d H=%d A=%d d_ff=%d vocab=%d max_len=%d weights=%zu\n",
               cfg.variant == BERT_LARGE ? "large" : "base",
               cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.intermediate,
               cfg.vocab_size, cfg.max_seq_len, dm_bert_weight_count(&cfg));
        free(w);
        return 0;
    }
    if (strcmp(cmd, "encode") == 0) {
        const char *model = NULL, *text = NULL;
        int all_layers = 0;
        for (int i = 2; i < argc; i++) {
            if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) model = argv[++i];
            else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) text = argv[++i];
            else if (strcmp(argv[i], "--all-layers") == 0) all_layers = 1;
        }
        BertConfig cfg;
        float *w = NULL;
        int *ids = NULL;
        int seq = 0;
        if (!model || !text || dm_bert_load(model, &cfg, &w) != 0 ||
            parse_ids(text, &ids, &seq) != 0 || seq > cfg.max_seq_len) {
            free(w);
            free(ids);
            return 1;
        }
        size_t sh = (size_t)seq * cfg.hidden_size;
        float *hidden = (float *)malloc((all_layers ? (size_t)cfg.num_layers : 1) * sh * sizeof(float));
        float *cls = (float *)malloc((size_t)cfg.hidden_size * sizeof(float));
        int *seg = (int *)calloc((size_t)seq, sizeof(int));
        if (!hidden || !cls || !seg) {
            free(w); free(ids); free(hidden); free(cls); free(seg);
            return 1;
        }
        int rc = all_layers
            ? dm_bert_forward_all_layers(&cfg, w, ids, seg, seq, hidden, cls)
            : dm_bert_forward(&cfg, w, ids, seg, seq, hidden, cls);
        if (rc != 0) {
            free(w); free(ids); free(hidden); free(cls); free(seg);
            return 1;
        }
        printf("cls");
        int limit = cfg.hidden_size < 16 ? cfg.hidden_size : 16;
        for (int i = 0; i < limit; i++) printf(" %.7g", cls[i]);
        printf("\n");
        free(w); free(ids); free(hidden); free(cls); free(seg);
        return 0;
    }
    if (strcmp(cmd, "bench") == 0) {
        BertVariant variant = BERT_BASE;
        int seq = 16, vocab = 30522;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) parse_variant(argv[++i], &variant);
            else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) seq = atoi(argv[++i]);
            else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) vocab = atoi(argv[++i]);
        }
        BertConfig cfg;
        dm_bert_config_init(&cfg, variant, vocab, seq > 512 ? seq : 512);
        if (seq <= 0 || seq > cfg.max_seq_len) return 1;
        printf("BERT %s seq=%d weights=%zu\n",
               cfg.variant == BERT_LARGE ? "large" : "base",
               seq, dm_bert_weight_count(&cfg));
        return 0;
    }
    usage();
    return 1;
}
