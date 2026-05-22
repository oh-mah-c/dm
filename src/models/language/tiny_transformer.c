#include "models/language/tiny_transformer.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TT_MAGIC "DMTTLM1"
#define TT_VOCAB 256

typedef struct {
    uint32_t context;
    uint32_t dim;
    uint32_t seed;
    float *tok_emb;
    float *pos_emb;
    float *out_w;
    float *out_b;
} TT_Model;

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s tiny_lm train -i corpus.txt -o model.bin [--context N] [--dim N]\n"
        "       [--epochs N] [--lr F] [--seed N] [--max-bytes N]\n"
        "  %s tiny_lm eval -i corpus.txt -m model.bin [--max-bytes N]\n"
        "  %s tiny_lm generate -m model.bin --prompt TEXT [--tokens N] [--temperature F] [--seed N]\n",
        prog, prog, prog);
}

static uint32_t rng_next(uint32_t *s) {
    uint32_t x = *s ? *s : 2463534242u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static float rng_uniform(uint32_t *s) {
    return ((rng_next(s) >> 8) * (1.0f / 16777216.0f));
}

static float rng_normalish(uint32_t *s) {
    float a = rng_uniform(s), b = rng_uniform(s), c = rng_uniform(s), d = rng_uniform(s);
    return (a + b + c + d - 2.0f) * 0.5f;
}

static int tt_alloc(TT_Model *m, uint32_t context, uint32_t dim, uint32_t seed) {
    size_t td = (size_t)TT_VOCAB * dim;
    size_t cd = (size_t)context * dim;
    size_t od = (size_t)dim * TT_VOCAB;
    memset(m, 0, sizeof(*m));
    if (!context || !dim) return -1;
    m->context = context;
    m->dim = dim;
    m->seed = seed;
    m->tok_emb = (float *)calloc(td, sizeof(float));
    m->pos_emb = (float *)calloc(cd, sizeof(float));
    m->out_w = (float *)calloc(od, sizeof(float));
    m->out_b = (float *)calloc(TT_VOCAB, sizeof(float));
    if (!m->tok_emb || !m->pos_emb || !m->out_w || !m->out_b) return -1;
    return 0;
}

static void tt_free(TT_Model *m) {
    if (!m) return;
    free(m->tok_emb);
    free(m->pos_emb);
    free(m->out_w);
    free(m->out_b);
    memset(m, 0, sizeof(*m));
}

static void tt_init(TT_Model *m) {
    uint32_t s = m->seed;
    size_t i, n;
    float scale = 0.02f;
    n = (size_t)TT_VOCAB * m->dim;
    for (i = 0; i < n; i++) m->tok_emb[i] = scale * rng_normalish(&s);
    n = (size_t)m->context * m->dim;
    for (i = 0; i < n; i++) m->pos_emb[i] = scale * rng_normalish(&s);
    n = (size_t)m->dim * TT_VOCAB;
    for (i = 0; i < n; i++) m->out_w[i] = scale * rng_normalish(&s);
    m->seed = s;
}

static unsigned char *read_file_bytes(const char *path, size_t *n_out, size_t max_bytes) {
    FILE *fp = fopen(path, "rb");
    unsigned char *buf;
    size_t n, cap;
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    cap = (size_t)ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (max_bytes && cap > max_bytes) cap = max_bytes;
    buf = (unsigned char *)malloc(cap ? cap : 1);
    if (!buf) { fclose(fp); return NULL; }
    n = fread(buf, 1, cap, fp);
    fclose(fp);
    *n_out = n;
    return buf;
}

static void build_features(const TT_Model *m, const unsigned char *ctx, size_t ctx_len, float *h) {
    uint32_t d, i;
    float norm = ctx_len ? 1.0f / (float)ctx_len : 1.0f;
    for (d = 0; d < m->dim; d++) h[d] = 0.0f;
    for (i = 0; i < ctx_len; i++) {
        unsigned char token = ctx[ctx_len - 1 - i];
        uint32_t pos = i < m->context ? i : m->context - 1;
        float recency = 1.0f / sqrtf((float)i + 1.0f);
        for (d = 0; d < m->dim; d++) {
            float x = m->tok_emb[(size_t)token * m->dim + d] + m->pos_emb[(size_t)pos * m->dim + d];
            h[d] += recency * x;
        }
    }
    for (d = 0; d < m->dim; d++) h[d] = tanhf(h[d] * norm);
}

static float softmax_loss(const TT_Model *m, const float *h, unsigned char target, float *prob) {
    uint32_t v, d;
    float max_logit = -1.0e30f, sum = 0.0f, loss;
    for (v = 0; v < TT_VOCAB; v++) {
        float z = m->out_b[v];
        for (d = 0; d < m->dim; d++) z += h[d] * m->out_w[(size_t)d * TT_VOCAB + v];
        prob[v] = z;
        if (z > max_logit) max_logit = z;
    }
    for (v = 0; v < TT_VOCAB; v++) {
        prob[v] = expf(prob[v] - max_logit);
        sum += prob[v];
    }
    if (sum <= 0.0f) sum = 1.0f;
    for (v = 0; v < TT_VOCAB; v++) prob[v] /= sum;
    loss = -logf(prob[target] > 1.0e-20f ? prob[target] : 1.0e-20f);
    return loss;
}

static float train_one(TT_Model *m, const unsigned char *ctx, size_t ctx_len,
                       unsigned char target, float lr, float *h, float *prob, float *dh) {
    uint32_t v, d, i;
    float loss = softmax_loss(m, h, target, prob);
    float norm = ctx_len ? 1.0f / (float)ctx_len : 1.0f;
    for (d = 0; d < m->dim; d++) dh[d] = 0.0f;
    prob[target] -= 1.0f;
    for (d = 0; d < m->dim; d++) {
        float hd = h[d];
        for (v = 0; v < TT_VOCAB; v++) {
            float g = prob[v];
            dh[d] += g * m->out_w[(size_t)d * TT_VOCAB + v];
            m->out_w[(size_t)d * TT_VOCAB + v] -= lr * g * hd;
        }
    }
    for (v = 0; v < TT_VOCAB; v++) m->out_b[v] -= lr * prob[v];
    for (d = 0; d < m->dim; d++) dh[d] *= (1.0f - h[d] * h[d]) * norm;
    for (i = 0; i < ctx_len; i++) {
        unsigned char token = ctx[ctx_len - 1 - i];
        uint32_t pos = i < m->context ? i : m->context - 1;
        float recency = 1.0f / sqrtf((float)i + 1.0f);
        for (d = 0; d < m->dim; d++) {
            float g = lr * recency * dh[d];
            m->tok_emb[(size_t)token * m->dim + d] -= g;
            m->pos_emb[(size_t)pos * m->dim + d] -= g;
        }
    }
    return loss;
}

static int tt_save(const TT_Model *m, const char *path) {
    FILE *fp = fopen(path, "wb");
    uint32_t vocab = TT_VOCAB;
    if (!fp) return -1;
    fwrite(TT_MAGIC, 1, 7, fp);
    fwrite(&vocab, sizeof(vocab), 1, fp);
    fwrite(&m->context, sizeof(m->context), 1, fp);
    fwrite(&m->dim, sizeof(m->dim), 1, fp);
    fwrite(&m->seed, sizeof(m->seed), 1, fp);
    fwrite(m->tok_emb, sizeof(float), (size_t)TT_VOCAB * m->dim, fp);
    fwrite(m->pos_emb, sizeof(float), (size_t)m->context * m->dim, fp);
    fwrite(m->out_w, sizeof(float), (size_t)m->dim * TT_VOCAB, fp);
    fwrite(m->out_b, sizeof(float), TT_VOCAB, fp);
    fclose(fp);
    return 0;
}

static int tt_load(TT_Model *m, const char *path) {
    FILE *fp = fopen(path, "rb");
    char magic[8] = {0};
    uint32_t vocab, context, dim, seed;
    if (!fp) return -1;
    if (fread(magic, 1, 7, fp) != 7 || memcmp(magic, TT_MAGIC, 7) != 0) { fclose(fp); return -1; }
    if (fread(&vocab, sizeof(vocab), 1, fp) != 1 || vocab != TT_VOCAB) { fclose(fp); return -1; }
    if (fread(&context, sizeof(context), 1, fp) != 1) { fclose(fp); return -1; }
    if (fread(&dim, sizeof(dim), 1, fp) != 1) { fclose(fp); return -1; }
    if (fread(&seed, sizeof(seed), 1, fp) != 1) { fclose(fp); return -1; }
    if (tt_alloc(m, context, dim, seed) != 0) { fclose(fp); return -1; }
    if (fread(m->tok_emb, sizeof(float), (size_t)TT_VOCAB * m->dim, fp) != (size_t)TT_VOCAB * m->dim) { fclose(fp); tt_free(m); return -1; }
    if (fread(m->pos_emb, sizeof(float), (size_t)m->context * m->dim, fp) != (size_t)m->context * m->dim) { fclose(fp); tt_free(m); return -1; }
    if (fread(m->out_w, sizeof(float), (size_t)m->dim * TT_VOCAB, fp) != (size_t)m->dim * TT_VOCAB) { fclose(fp); tt_free(m); return -1; }
    if (fread(m->out_b, sizeof(float), TT_VOCAB, fp) != TT_VOCAB) { fclose(fp); tt_free(m); return -1; }
    fclose(fp);
    return 0;
}

static int cmd_train(int argc, char **argv) {
    const char *input = NULL, *output = NULL;
    uint32_t context = 64, dim = 64, seed = 1, epochs = 3;
    float lr = 0.05f;
    size_t max_bytes = 0, n = 0, step, epoch;
    unsigned char *data;
    TT_Model m;
    float *h, *prob, *dh;
    int i;
    for (i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) input = argv[++i];
        else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) output = argv[++i];
        else if (strcmp(argv[i], "--context") == 0 && i + 1 < argc) context = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) dim = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) epochs = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) lr = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--max-bytes") == 0 && i + 1 < argc) max_bytes = (size_t)strtoull(argv[++i], NULL, 10);
        else return 2;
    }
    if (!input || !output || context < 1 || dim < 1 || epochs < 1) return 2;
    data = read_file_bytes(input, &n, max_bytes);
    if (!data || n < 2) { free(data); return 1; }
    if (tt_alloc(&m, context, dim, seed) != 0) { free(data); return 1; }
    tt_init(&m);
    h = (float *)malloc(sizeof(float) * dim);
    prob = (float *)malloc(sizeof(float) * TT_VOCAB);
    dh = (float *)malloc(sizeof(float) * dim);
    if (!h || !prob || !dh) { free(h); free(prob); free(dh); tt_free(&m); free(data); return 1; }
    for (epoch = 0; epoch < epochs; epoch++) {
        double total = 0.0;
        size_t count = 0;
        for (step = 1; step < n; step++) {
            size_t start = step > context ? step - context : 0;
            build_features(&m, data + start, step - start, h);
            total += train_one(&m, data + start, step - start, data[step], lr, h, prob, dh);
            count++;
        }
        printf("epoch=%zu loss=%.6f ppl=%.6f tokens=%zu\n", epoch + 1, total / (double)count, exp(total / (double)count), count);
    }
    if (tt_save(&m, output) != 0) { free(h); free(prob); free(dh); tt_free(&m); free(data); return 1; }
    printf("saved=%s\n", output);
    printf("context=%u\n", context);
    printf("dim=%u\n", dim);
    free(h); free(prob); free(dh); tt_free(&m); free(data);
    return 0;
}

static double eval_loss(TT_Model *m, const unsigned char *data, size_t n) {
    float *h = (float *)malloc(sizeof(float) * m->dim);
    float *prob = (float *)malloc(sizeof(float) * TT_VOCAB);
    double total = 0.0;
    size_t step, count = 0;
    if (!h || !prob || n < 2) { free(h); free(prob); return 0.0; }
    for (step = 1; step < n; step++) {
        size_t start = step > m->context ? step - m->context : 0;
        build_features(m, data + start, step - start, h);
        total += softmax_loss(m, h, data[step], prob);
        count++;
    }
    free(h); free(prob);
    return count ? total / (double)count : 0.0;
}

static int cmd_eval(int argc, char **argv) {
    const char *input = NULL, *model_path = NULL;
    size_t max_bytes = 0, n = 0;
    unsigned char *data;
    TT_Model m;
    double loss;
    int i;
    for (i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) input = argv[++i];
        else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--max-bytes") == 0 && i + 1 < argc) max_bytes = (size_t)strtoull(argv[++i], NULL, 10);
        else return 2;
    }
    if (!input || !model_path || tt_load(&m, model_path) != 0) return 2;
    data = read_file_bytes(input, &n, max_bytes);
    if (!data || n < 2) { tt_free(&m); free(data); return 1; }
    loss = eval_loss(&m, data, n);
    printf("loss=%.6f\n", loss);
    printf("ppl=%.6f\n", exp(loss));
    printf("tokens=%zu\n", n - 1);
    tt_free(&m); free(data);
    return 0;
}

static int sample_next(float *prob, float temperature, uint32_t *seed) {
    float sum = 0.0f, r, acc = 0.0f;
    int v;
    for (v = 0; v < TT_VOCAB; v++) {
        if (!(v == '\n' || (v >= 32 && v <= 126))) prob[v] = 0.0f;
    }
    if (temperature <= 0.0f) {
        int best = 0;
        for (v = 1; v < TT_VOCAB; v++) if (prob[v] > prob[best]) best = v;
        if (!(best == '\n' || (best >= 32 && best <= 126))) best = ' ';
        return best;
    }
    for (v = 0; v < TT_VOCAB; v++) {
        prob[v] = powf(prob[v], 1.0f / temperature);
        sum += prob[v];
    }
    r = rng_uniform(seed) * sum;
    for (v = 0; v < TT_VOCAB; v++) {
        acc += prob[v];
        if (acc >= r) return v;
    }
    return TT_VOCAB - 1;
}

static int cmd_generate(int argc, char **argv) {
    const char *model_path = NULL, *prompt = "";
    uint32_t seed = 1;
    size_t tokens = 200, len, cap, t;
    float temperature = 0.8f;
    TT_Model m;
    unsigned char *buf;
    float *h, *prob;
    int i;
    for (i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) tokens = (size_t)strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else return 2;
    }
    if (!model_path || tt_load(&m, model_path) != 0) return 2;
    len = strlen(prompt);
    cap = len + tokens + 1;
    buf = (unsigned char *)malloc(cap);
    h = (float *)malloc(sizeof(float) * m.dim);
    prob = (float *)malloc(sizeof(float) * TT_VOCAB);
    if (!buf || !h || !prob) { free(buf); free(h); free(prob); tt_free(&m); return 1; }
    memcpy(buf, prompt, len);
    for (t = 0; t < tokens; t++) {
        size_t start = len > m.context ? len - m.context : 0;
        int next;
        build_features(&m, buf + start, len - start, h);
        (void)softmax_loss(&m, h, 0, prob);
        next = sample_next(prob, temperature, &seed);
        buf[len++] = (unsigned char)next;
    }
    fwrite(buf, 1, len, stdout);
    fputc('\n', stdout);
    free(buf); free(h); free(prob); tt_free(&m);
    return 0;
}

int dm_tiny_transformer_cli(int argc, char **argv) {
    const char *cmd;
    if (argc < 2) { usage(argv[0]); return 2; }
    cmd = argv[1];
    if (strcmp(cmd, "tiny_lm") == 0 || strcmp(cmd, "tiny_transformer") == 0 || strcmp(cmd, "dm_tiny_transformer") == 0) {
        if (argc < 3) { usage(argv[0]); return 2; }
        cmd = argv[2];
        argv++;
        argc--;
    }
    if (strcmp(cmd, "train") == 0) return cmd_train(argc, argv);
    if (strcmp(cmd, "eval") == 0) return cmd_eval(argc, argv);
    if (strcmp(cmd, "generate") == 0) return cmd_generate(argc, argv);
    usage(argv[0]);
    return 2;
}
