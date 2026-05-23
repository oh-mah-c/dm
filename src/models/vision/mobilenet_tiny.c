#include "models/vision/mobilenet_tiny.h"

#include "encoding/image_patchify.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <sys/wait.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/tf_tstring.h"

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s mobilenet_tiny infer -i image.ppm [--classes N] [--seed N] [--size N]\n"
        "       [--model trained_head.bin]\n"
        "  %s mobilenet_tiny train --manifest train.txt -o trained_head.bin\n"
        "       [--classes N] [--epochs N] [--lr F] [--seed N] [--size N]\n"
        "  %s mobilenet_tiny tf-train --manifest train.txt -o model.keras --classes N\n"
        "       [--epochs N] [--batch N] [--lr F] [--size N] [--width F]\n"
        "  %s mobilenet_tiny tf-eval --manifest test.txt --model model.keras [--size N]\n"
        "  %s mobilenet_tiny tf-predict --model model.keras -i image.ppm [--top N]\n"
        "  %s mobilenet_tiny tf-export --model model.keras -o model.tflite [--format tflite]\n"
        "  %s mobilenet_tiny bench-block [--h N] [--w N] [--c N]\n\n"
        "Manifest format: one sample per line: /path/to/image.ppm <integer_label>\n"
        "PPM input must be binary P6 RGB. This module implements MobileNetV4-style\n"
        "UIB/FusedIB/Mobile-MQA operators for C99 inference plus a TensorFlow\n"
        "end-to-end training backend for full CNN backpropagation.\n",
        prog, prog, prog, prog, prog, prog, prog);
}

static uint32_t rng_next(uint32_t *s) {
    uint32_t x = *s ? *s : 2463534242u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}

static float rng_weight(uint32_t *s, float scale) {
    return (((rng_next(s) >> 8) * (1.0f / 16777216.0f)) * 2.0f - 1.0f) * scale;
}

static float *make_weights(size_t n, unsigned int seed, float scale) {
    size_t i;
    uint32_t s = seed;
    float *w = (float *)malloc(sizeof(float) * n);
    if (!w) return NULL;
    for (i = 0; i < n; i++) w[i] = rng_weight(&s, scale);
    return w;
}

static int conv_relu(const DM_Tensor *in, DM_Tensor *out, int out_c, int kernel, int stride, unsigned int seed) {
    float *w = make_weights((size_t)out_c * in->c * kernel * kernel, seed, 0.08f);
    int rc;
    if (!w) return -1;
    rc = dm_conv2d_same(in, out, w, NULL, out_c, kernel, stride);
    free(w);
    if (rc == 0) dm_relu6(out);
    return rc;
}

static int depthwise_relu(const DM_Tensor *in, DM_Tensor *out, int kernel, int stride, unsigned int seed) {
    float *w = make_weights((size_t)in->c * kernel * kernel, seed, 0.08f);
    int rc;
    if (!w) return -1;
    rc = dm_depthwise_conv2d_same(in, out, w, NULL, kernel, stride);
    free(w);
    if (rc == 0) dm_relu6(out);
    return rc;
}

static int pointwise_relu(const DM_Tensor *in, DM_Tensor *out, int out_c, unsigned int seed, int activate) {
    float *w = make_weights((size_t)out_c * in->c, seed, 0.08f);
    int rc;
    if (!w) return -1;
    rc = dm_pointwise_conv2d(in, out, w, NULL, out_c);
    free(w);
    if (rc == 0 && activate) dm_relu6(out);
    return rc;
}

static int fused_ib(const DM_Tensor *in, DM_Tensor *out, int expanded_c, int out_c, int kernel, int stride, unsigned int seed) {
    DM_Tensor x = {0};
    if (conv_relu(in, &x, expanded_c, kernel, stride, seed + 1) != 0) return -1;
    if (pointwise_relu(&x, out, out_c, seed + 2, 1) != 0) { dm_tensor_free(&x); return -1; }
    dm_tensor_free(&x);
    return 0;
}

static int add_residual_if_same(const DM_Tensor *in, DM_Tensor *out) {
    size_t i, n;
    if (!in || !out || in->n != out->n || in->c != out->c || in->h != out->h || in->w != out->w) return 0;
    n = dm_tensor_count(out);
    for (i = 0; i < n; i++) out->data[i] += in->data[i];
    return 1;
}

int dm_uib_block(const DM_Tensor *in, DM_Tensor *out, DM_UIBKind kind,
                 int expanded_c, int out_c, int kernel1, int kernel2, int stride, unsigned int seed) {
    DM_Tensor a = {0}, b = {0}, c = {0};
    const DM_Tensor *cur = in;
    int rc = -1;
    int use_dw1 = (kind == DM_UIB_CONVNEXT || kind == DM_UIB_EXTRADW);
    int use_dw2 = (kind == DM_UIB_IB || kind == DM_UIB_EXTRADW);

    if (use_dw1) {
        if (depthwise_relu(cur, &a, kernel1, stride, seed + 11) != 0) goto done;
        cur = &a;
    }
    if (pointwise_relu(cur, &b, expanded_c, seed + 12, 1) != 0) goto done;
    cur = &b;
    if (use_dw2) {
        int dw_stride = use_dw1 ? 1 : stride;
        if (depthwise_relu(cur, &c, kernel2, dw_stride, seed + 13) != 0) goto done;
        cur = &c;
    }
    if (pointwise_relu(cur, out, out_c, seed + 14, 0) != 0) goto done;
    add_residual_if_same(in, out);
    rc = 0;
done:
    dm_tensor_free(&a); dm_tensor_free(&b); dm_tensor_free(&c);
    return rc;
}

static int spatial_reduce(const DM_Tensor *in, DM_Tensor *out, int reduce, unsigned int seed) {
    if (!reduce) {
        size_t bytes = dm_tensor_count(in) * sizeof(float);
        if (dm_tensor_alloc(out, in->n, in->c, in->h, in->w) != 0) return -1;
        memcpy(out->data, in->data, bytes);
        return 0;
    }
    return depthwise_relu(in, out, 3, 2, seed);
}

int dm_mobile_mqa_block(const DM_Tensor *in, DM_Tensor *out, int heads, int key_dim,
                        int spatial_reduction, unsigned int seed) {
    DM_Tensor kv_in = {0};
    float *wq = NULL, *wk = NULL, *wv = NULL, *wo = NULL;
    float *q = NULL, *k = NULL, *v = NULL, *cat = NULL;
    int tokens_q, tokens_kv, y, x, t, s, h, d, c, oc;
    float scale;
    if (!in || !out || heads <= 0 || key_dim <= 0) return -1;
    if (spatial_reduce(in, &kv_in, spatial_reduction, seed + 21) != 0) return -1;
    tokens_q = in->h * in->w;
    tokens_kv = kv_in.h * kv_in.w;
    wq = make_weights((size_t)heads * key_dim * in->c, seed + 22, 0.05f);
    wk = make_weights((size_t)key_dim * in->c, seed + 23, 0.05f);
    wv = make_weights((size_t)key_dim * in->c, seed + 24, 0.05f);
    wo = make_weights((size_t)in->c * heads * key_dim, seed + 25, 0.05f);
    q = (float *)calloc((size_t)heads * tokens_q * key_dim, sizeof(float));
    k = (float *)calloc((size_t)tokens_kv * key_dim, sizeof(float));
    v = (float *)calloc((size_t)tokens_kv * key_dim, sizeof(float));
    cat = (float *)calloc((size_t)tokens_q * heads * key_dim, sizeof(float));
    if (!wq || !wk || !wv || !wo || !q || !k || !v || !cat) goto fail;

    for (h = 0; h < heads; h++) for (t = 0; t < tokens_q; t++) {
        y = t / in->w; x = t % in->w;
        for (d = 0; d < key_dim; d++) for (c = 0; c < in->c; c++)
            q[((size_t)h * tokens_q + t) * key_dim + d] += dm_tensor_get(in, 0, c, y, x) * wq[((size_t)h * key_dim + d) * in->c + c];
    }
    for (t = 0; t < tokens_kv; t++) {
        y = t / kv_in.w; x = t % kv_in.w;
        for (d = 0; d < key_dim; d++) for (c = 0; c < kv_in.c; c++) {
            float val = dm_tensor_get(&kv_in, 0, c, y, x);
            k[(size_t)t * key_dim + d] += val * wk[(size_t)d * kv_in.c + c];
            v[(size_t)t * key_dim + d] += val * wv[(size_t)d * kv_in.c + c];
        }
    }
    scale = 1.0f / sqrtf((float)key_dim);
    for (h = 0; h < heads; h++) for (t = 0; t < tokens_q; t++) {
        float max_logit = -1.0e30f, sum = 0.0f;
        float *scores = (float *)malloc(sizeof(float) * (size_t)tokens_kv);
        if (!scores) goto fail;
        for (s = 0; s < tokens_kv; s++) {
            float z = 0.0f;
            for (d = 0; d < key_dim; d++) z += q[((size_t)h * tokens_q + t) * key_dim + d] * k[(size_t)s * key_dim + d];
            scores[s] = z * scale;
            if (scores[s] > max_logit) max_logit = scores[s];
        }
        for (s = 0; s < tokens_kv; s++) { scores[s] = expf(scores[s] - max_logit); sum += scores[s]; }
        if (sum <= 0.0f) sum = 1.0f;
        for (d = 0; d < key_dim; d++) {
            float acc = 0.0f;
            for (s = 0; s < tokens_kv; s++) acc += (scores[s] / sum) * v[(size_t)s * key_dim + d];
            cat[((size_t)t * heads + h) * key_dim + d] = acc;
        }
        free(scores);
    }
    if (dm_tensor_alloc(out, in->n, in->c, in->h, in->w) != 0) goto fail;
    for (t = 0; t < tokens_q; t++) {
        y = t / in->w; x = t % in->w;
        for (oc = 0; oc < in->c; oc++) {
            float z = 0.0f;
            for (c = 0; c < heads * key_dim; c++) z += cat[(size_t)t * heads * key_dim + c] * wo[(size_t)oc * heads * key_dim + c];
            dm_tensor_set(out, 0, oc, y, x, z);
        }
    }
    add_residual_if_same(in, out);
    dm_tensor_free(&kv_in); free(wq); free(wk); free(wv); free(wo); free(q); free(k); free(v); free(cat);
    return 0;
fail:
    dm_tensor_free(&kv_in); free(wq); free(wk); free(wv); free(wo); free(q); free(k); free(v); free(cat);
    return -1;
}

static int mobilenet_features(const DM_Tensor *input, DM_Tensor *features, unsigned int seed) {
    DM_Tensor x1 = {0}, x2 = {0}, x3 = {0}, x4 = {0}, x5 = {0}, x6 = {0}, pool = {0};
    int rc = -1;
    if (fused_ib(input, &x1, 16, 16, 3, 2, seed + 100) != 0) goto done;
    if (dm_uib_block(&x1, &x2, DM_UIB_EXTRADW, 64, 24, 3, 3, 2, seed + 200) != 0) goto done;
    if (dm_uib_block(&x2, &x3, DM_UIB_IB, 96, 24, 3, 3, 1, seed + 300) != 0) goto done;
    if (dm_uib_block(&x3, &x4, DM_UIB_CONVNEXT, 96, 32, 5, 3, 2, seed + 400) != 0) goto done;
    if (dm_mobile_mqa_block(&x4, &x5, 4, 8, 1, seed + 500) != 0) goto done;
    if (pointwise_relu(&x5, &x6, 64, seed + 600, 1) != 0) goto done;
    if (dm_global_avg_pool(&x6, &pool) != 0) goto done;
    *features = pool;
    memset(&pool, 0, sizeof(pool));
    if (features->c >= 6 && input && input->data && input->c >= 3) {
        int ch, y, x;
        float inv = 1.0f / (float)(input->h * input->w);
        for (ch = 0; ch < 3; ch++) {
            float mean = 0.0f, sq = 0.0f;
            for (y = 0; y < input->h; y++) {
                for (x = 0; x < input->w; x++) {
                    float v = dm_tensor_get(input, 0, ch, y, x);
                    mean += v;
                    sq += v * v;
                }
            }
            mean *= inv;
            sq *= inv;
            features->data[ch] += mean;
            features->data[ch + 3] += sqrtf(fmaxf(0.0f, sq - mean * mean));
        }
    }
    rc = 0;
done:
    dm_tensor_free(&x1); dm_tensor_free(&x2); dm_tensor_free(&x3); dm_tensor_free(&x4);
    dm_tensor_free(&x5); dm_tensor_free(&x6); dm_tensor_free(&pool);
    return rc;
}

int dm_mobilenet_tiny_forward(const DM_Tensor *input, DM_Tensor *logits, int classes, unsigned int seed) {
    DM_Tensor features = {0};
    float *w = NULL;
    int rc = -1;
    if (mobilenet_features(input, &features, seed) != 0) goto done;
    w = make_weights((size_t)classes * features.c, seed + 700, 0.05f);
    if (!w) goto done;
    if (dm_linear(&features, logits, w, NULL, classes) != 0) goto done;
    dm_softmax(logits);
    rc = 0;
done:
    free(w);
    dm_tensor_free(&features);
    return rc;
}

typedef struct {
    int classes;
    int feature_dim;
    int image_size;
    unsigned int seed;
    float *w;
    float *b;
} TinyHead;

static void head_free(TinyHead *h) {
    if (!h) return;
    free(h->w);
    free(h->b);
    memset(h, 0, sizeof(*h));
}

static int head_alloc(TinyHead *h, int classes, int feature_dim, int image_size, unsigned int seed) {
    memset(h, 0, sizeof(*h));
    h->classes = classes;
    h->feature_dim = feature_dim;
    h->image_size = image_size;
    h->seed = seed;
    h->w = (float *)calloc((size_t)classes * (size_t)feature_dim, sizeof(float));
    h->b = (float *)calloc((size_t)classes, sizeof(float));
    return (h->w && h->b) ? 0 : -1;
}

static int head_save(const TinyHead *h, const char *path) {
    FILE *fp = fopen(path, "wb");
    char magic[8] = "DMMNH1";
    if (!fp) return -1;
    fwrite(magic, 1, 8, fp);
    fwrite(&h->classes, sizeof(h->classes), 1, fp);
    fwrite(&h->feature_dim, sizeof(h->feature_dim), 1, fp);
    fwrite(&h->image_size, sizeof(h->image_size), 1, fp);
    fwrite(&h->seed, sizeof(h->seed), 1, fp);
    fwrite(h->w, sizeof(float), (size_t)h->classes * (size_t)h->feature_dim, fp);
    fwrite(h->b, sizeof(float), (size_t)h->classes, fp);
    fclose(fp);
    return 0;
}

static int head_load(TinyHead *h, const char *path) {
    FILE *fp = fopen(path, "rb");
    char magic[8];
    int classes, feature_dim, image_size;
    unsigned int seed;
    if (!fp) return -1;
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "DMMNH1", 6) != 0) { fclose(fp); return -1; }
    if (fread(&classes, sizeof(classes), 1, fp) != 1) { fclose(fp); return -1; }
    if (fread(&feature_dim, sizeof(feature_dim), 1, fp) != 1) { fclose(fp); return -1; }
    if (fread(&image_size, sizeof(image_size), 1, fp) != 1) { fclose(fp); return -1; }
    if (fread(&seed, sizeof(seed), 1, fp) != 1) { fclose(fp); return -1; }
    if (head_alloc(h, classes, feature_dim, image_size, seed) != 0) { fclose(fp); return -1; }
    if (fread(h->w, sizeof(float), (size_t)classes * (size_t)feature_dim, fp) != (size_t)classes * (size_t)feature_dim) { fclose(fp); head_free(h); return -1; }
    if (fread(h->b, sizeof(float), (size_t)classes, fp) != (size_t)classes) { fclose(fp); head_free(h); return -1; }
    fclose(fp);
    return 0;
}

static int load_resized_features(const char *path, int size, unsigned int seed, DM_Tensor *features) {
    DM_Tensor img = {0}, resized = {0};
    int rc = -1;
    if (dm_image_load_ppm_rgb_f32(path, &img) != 0) goto done;
    if (dm_image_resize_nearest(&img, &resized, size, size) != 0) goto done;
    if (mobilenet_features(&resized, features, seed) != 0) goto done;
    rc = 0;
done:
    dm_tensor_free(&img);
    dm_tensor_free(&resized);
    return rc;
}

static void head_logits(const TinyHead *h, const DM_Tensor *features, float *logits) {
    int c, d;
    for (c = 0; c < h->classes; c++) {
        float z = h->b[c];
        for (d = 0; d < h->feature_dim; d++) z += h->w[(size_t)c * (size_t)h->feature_dim + (size_t)d] * features->data[d];
        logits[c] = z;
    }
}

static float head_softmax_loss(float *logits, int classes, int label) {
    int c;
    float mx = logits[0], sum = 0.0f, loss;
    for (c = 1; c < classes; c++) if (logits[c] > mx) mx = logits[c];
    for (c = 0; c < classes; c++) {
        logits[c] = expf(logits[c] - mx);
        sum += logits[c];
    }
    if (sum <= 0.0f) sum = 1.0f;
    for (c = 0; c < classes; c++) logits[c] /= sum;
    loss = -logf(logits[label] > 1.0e-20f ? logits[label] : 1.0e-20f);
    logits[label] -= 1.0f;
    return loss;
}

static int cmd_train(int argc, char **argv) {
    const char *manifest = NULL, *output = NULL;
    int classes = 2, epochs = 10, size = 32, i, epoch;
    unsigned int seed = 1;
    float lr = 0.2f;
    TinyHead head = {0};
    for (i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--manifest") == 0 && i + 1 < argc) manifest = argv[++i];
        else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) output = argv[++i];
        else if (strcmp(argv[i], "--classes") == 0 && i + 1 < argc) classes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) lr = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) size = atoi(argv[++i]);
        else return 2;
    }
    if (!manifest || !output || classes <= 1 || epochs <= 0 || size <= 0) return 2;
    if (head_alloc(&head, classes, 64, size, seed) != 0) return 1;
    for (epoch = 0; epoch < epochs; epoch++) {
        FILE *fp = fopen(manifest, "r");
        char path[4096];
        int label, correct = 0;
        size_t samples = 0;
        double total = 0.0;
        float *grad_w;
        float *grad_b;
        if (!fp) { head_free(&head); return 1; }
        grad_w = (float *)calloc((size_t)head.classes * (size_t)head.feature_dim, sizeof(float));
        grad_b = (float *)calloc((size_t)head.classes, sizeof(float));
        if (!grad_w || !grad_b) { free(grad_w); free(grad_b); fclose(fp); head_free(&head); return 1; }
        while (fscanf(fp, "%4095s %d", path, &label) == 2) {
            DM_Tensor feat = {0};
            float *logits;
            int c, d, pred = 0;
            if (label < 0 || label >= classes) continue;
            if (load_resized_features(path, size, seed, &feat) != 0) continue;
            logits = (float *)malloc(sizeof(float) * (size_t)classes);
            if (!logits) { dm_tensor_free(&feat); fclose(fp); head_free(&head); return 1; }
            head_logits(&head, &feat, logits);
            for (c = 1; c < classes; c++) if (logits[c] > logits[pred]) pred = c;
            if (pred == label) correct++;
            total += head_softmax_loss(logits, classes, label);
            for (c = 0; c < classes; c++) {
                float g = logits[c];
                for (d = 0; d < head.feature_dim; d++) grad_w[(size_t)c * (size_t)head.feature_dim + (size_t)d] += g * feat.data[d];
                grad_b[c] += g;
            }
            samples++;
            free(logits);
            dm_tensor_free(&feat);
        }
        fclose(fp);
        if (samples) {
            float rate = lr / (float)samples;
            int c, d;
            for (c = 0; c < classes; c++) {
                for (d = 0; d < head.feature_dim; d++) {
                    size_t idx = (size_t)c * (size_t)head.feature_dim + (size_t)d;
                    head.w[idx] -= rate * grad_w[idx];
                }
                head.b[c] -= rate * grad_b[c];
            }
        }
        free(grad_w);
        free(grad_b);
        printf("epoch=%d loss=%.6f accuracy=%.6f samples=%zu\n", epoch + 1, samples ? total / (double)samples : 0.0, samples ? (double)correct / (double)samples : 0.0, samples);
    }
    if (head_save(&head, output) != 0) { head_free(&head); return 1; }
    printf("saved=%s\nclasses=%d\nfeature_dim=%d\nimage_size=%d\n", output, head.classes, head.feature_dim, head.image_size);
    head_free(&head);
    return 0;
}

static int cmd_infer(int argc, char **argv) {
    const char *input = NULL, *model_path = NULL;
    int classes = 10, size = 64, top = 5, i;
    unsigned int seed = 1;
    DM_Tensor img = {0}, resized = {0}, logits = {0};
    for (i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) input = argv[++i];
        else if (strcmp(argv[i], "--classes") == 0 && i + 1 < argc) classes = atoi(argv[++i]);
        else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--top") == 0 && i + 1 < argc) top = atoi(argv[++i]);
        else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) model_path = argv[++i];
        else return 2;
    }
    if (!input || classes <= 0 || size <= 0) return 2;
    if (model_path) {
        TinyHead head = {0};
        DM_Tensor feat = {0};
        float *raw;
        int c;
        if (head_load(&head, model_path) != 0) return 1;
        classes = head.classes;
        size = head.image_size;
        seed = head.seed;
        if (load_resized_features(input, size, seed, &feat) != 0) { head_free(&head); return 1; }
        if (dm_tensor_alloc(&logits, 1, classes, 1, 1) != 0) { dm_tensor_free(&feat); head_free(&head); return 1; }
        raw = (float *)malloc(sizeof(float) * (size_t)classes);
        if (!raw) { dm_tensor_free(&feat); head_free(&head); dm_tensor_free(&logits); return 1; }
        head_logits(&head, &feat, raw);
        for (c = 0; c < classes; c++) dm_tensor_set(&logits, 0, c, 0, 0, raw[c]);
        dm_softmax(&logits);
        free(raw);
        dm_tensor_free(&feat);
        head_free(&head);
    } else {
        if (dm_image_load_ppm_rgb_f32(input, &img) != 0) return 1;
        if (dm_image_resize_nearest(&img, &resized, size, size) != 0) { dm_tensor_free(&img); return 1; }
        if (dm_mobilenet_tiny_forward(&resized, &logits, classes, seed) != 0) { dm_tensor_free(&img); dm_tensor_free(&resized); return 1; }
    }
    printf("model=mobilenet_tiny_mnv4_style\n");
    printf("input=%s\nclasses=%d\nimage_size=%d\n", input, classes, size);
    for (i = 0; i < top && i < classes; i++) {
        int c, best = -1;
        float bv = -1.0f;
        for (c = 0; c < classes; c++) {
            float v = dm_tensor_get(&logits, 0, c, 0, 0);
            if (v > bv) { bv = v; best = c; }
        }
        printf("rank%d_class=%d prob=%.8f\n", i + 1, best, bv);
        dm_tensor_set(&logits, 0, best, 0, 0, -1.0f);
    }
    dm_tensor_free(&img); dm_tensor_free(&resized); dm_tensor_free(&logits);
    return 0;
}

static int cmd_bench_block(int argc, char **argv) {
    int h = 16, w = 16, c = 32, i;
    DM_Tensor x = {0}, y = {0}, z = {0};
    for (i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--h") == 0 && i + 1 < argc) h = atoi(argv[++i]);
        else if (strcmp(argv[i], "--w") == 0 && i + 1 < argc) w = atoi(argv[++i]);
        else if (strcmp(argv[i], "--c") == 0 && i + 1 < argc) c = atoi(argv[++i]);
        else return 2;
    }
    if (dm_tensor_alloc(&x, 1, c, h, w) != 0) return 1;
    for (i = 0; i < (int)dm_tensor_count(&x); i++) x.data[i] = (float)(i % 17) / 17.0f;
    if (dm_uib_block(&x, &y, DM_UIB_EXTRADW, c * 4, c, 3, 3, 1, 1) != 0) { dm_tensor_free(&x); return 1; }
    if (dm_mobile_mqa_block(&y, &z, 4, 8, 1, 2) != 0) { dm_tensor_free(&x); dm_tensor_free(&y); return 1; }
    printf("uib_out=%dx%dx%d\n", y.c, y.h, y.w);
    printf("mobile_mqa_out=%dx%dx%d\n", z.c, z.h, z.w);
    printf("operators=FusedIB,UIB_FFN,UIB_IB,UIB_ConvNext,UIB_ExtraDW,Mobile_MQA(shared_KV,SRA_stride2_DW)\n");
    dm_tensor_free(&x); dm_tensor_free(&y); dm_tensor_free(&z);
    return 0;
}

static int parse_metadata_key(const char *json, const char *sec, const char *key, char *out_val, size_t max_len) {
    const char *p = strstr(json, sec);
    if (!p) return -1;
    p = strstr(p, key);
    if (!p) return -1;
    p = strchr(p, ':');
    if (!p) return -1;
    p = strchr(p, '"');
    if (!p) return -1;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i < max_len - 1) {
        out_val[i++] = *p++;
    }
    out_val[i] = '\0';
    return 0;
}

static void get_op_name(const char *tensor_name, char *op_name) {
    const char *colon = strchr(tensor_name, ':');
    if (colon) {
        size_t len = (size_t)(colon - tensor_name);
        strncpy(op_name, tensor_name, len);
        op_name[len] = '\0';
    } else {
        strcpy(op_name, tensor_name);
    }
}

static char* read_entire_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)size + 1);
    if (buf) {
        size_t n = fread(buf, 1, (size_t)size, f);
        buf[n] = '\0';
    }
    fclose(f);
    return buf;
}

static void tensor_deallocator(void* data, size_t len, void* arg) {
    (void)len; (void)arg;
    free(data);
}

static int run_save_or_load_signature(TF_Session *session, TF_Graph *graph, TF_Status *status,
                                      const char *path_tensor_name, const char *status_tensor_name,
                                      const char *ckpt_prefix) {
    TF_Tensor *path_tensor = TF_AllocateTensor(TF_STRING, NULL, 0, sizeof(TF_TString));
    if (!path_tensor) return -1;
    TF_TString *tstr = (TF_TString*)TF_TensorData(path_tensor);
    TF_StringInit(tstr);
    TF_StringCopy(tstr, ckpt_prefix, strlen(ckpt_prefix));

    char path_op_name[128];
    char status_op_name[128];
    get_op_name(path_tensor_name, path_op_name);
    get_op_name(status_tensor_name, status_op_name);

    TF_Output input = {TF_GraphOperationByName(graph, path_op_name), 0};
    if (!input.oper) {
        fprintf(stderr, "Error: cannot find operation %s\n", path_op_name);
        TF_DeleteTensor(path_tensor);
        return -1;
    }
    TF_Tensor *input_tensors[1] = {path_tensor};

    TF_Output output = {TF_GraphOperationByName(graph, status_op_name), 0};
    if (!output.oper) {
        fprintf(stderr, "Error: cannot find operation %s\n", status_op_name);
        TF_DeleteTensor(path_tensor);
        return -1;
    }
    TF_Tensor *output_tensors[1] = {NULL};

    TF_SessionRun(session, NULL, &input, input_tensors, 1, &output, output_tensors, 1, NULL, 0, NULL, status);
    TF_DeleteTensor(path_tensor);

    if (TF_GetCode(status) != TF_OK) {
        fprintf(stderr, "Signature execution failed: %s\n", TF_Message(status));
        if (output_tensors[0]) TF_DeleteTensor(output_tensors[0]);
        return -1;
    }

    if (output_tensors[0]) {
        TF_DeleteTensor(output_tensors[0]);
    }
    return 0;
}

static int run_tf_backend(int argc, char **argv, const char *tf_cmd) {
    const char *manifest = NULL;
    const char *output = NULL;
    const char *model_path = NULL;
    const char *input_image = NULL;
    int classes = 2;
    int epochs = 10;
    int batch_size = 16;
    float lr = 0.001f;
    int size = 128;
    float width = 0.5f;
    float dropout = 0.0f;
    int top = 5;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--manifest") == 0 && i + 1 < argc) { manifest = argv[++i]; }
        else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) { output = argv[++i]; }
        else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) { model_path = argv[++i]; }
        else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) { input_image = argv[++i]; }
        else if (strcmp(argv[i], "--classes") == 0 && i + 1 < argc) { classes = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) { epochs = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) { batch_size = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) { lr = (float)atof(argv[++i]); }
        else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) { size = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) { width = (float)atof(argv[++i]); }
        else if (strcmp(argv[i], "--dropout") == 0 && i + 1 < argc) { dropout = (float)atof(argv[++i]); }
        else if (strcmp(argv[i], "--top") == 0 && i + 1 < argc) { top = atoi(argv[++i]); }
    }

    if (strcmp(tf_cmd, "export") == 0) {
#ifdef _WIN32
        fprintf(stderr, "TensorFlow export subprocess is implemented for POSIX shells in this build.\n");
        return 127;
#else
        const char *python = getenv("DM_PYTHON");
        const char *script = getenv("DM_TF_MOBILENET_SCRIPT");
        char **args;
        pid_t pid;
        int status;
        int i;
        if (!python || !python[0]) python = access(".venv/bin/python", X_OK) == 0 ? ".venv/bin/python" : "python3";
        if (!script || !script[0]) script = "src/models/vision/tf_mobilenetv4.py";
        args = (char **)calloc((size_t)argc + 4u, sizeof(char *));
        if (!args) return 1;
        args[0] = (char *)python;
        args[1] = (char *)script;
        args[2] = (char *)tf_cmd;
        for (i = 2; i < argc; i++) args[i + 1] = argv[i];
        args[argc + 1] = NULL;
        pid = fork();
        if (pid < 0) {
            free(args);
            return 1;
        }
        if (pid == 0) {
            execvp(python, args);
            fprintf(stderr, "error: failed to launch Python for TensorFlow backend\n");
            _exit(127);
        }
        free(args);
        if (waitpid(pid, &status, 0) < 0) return 1;
        if (WIFEXITED(status)) return WEXITSTATUS(status);
        if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
        return 1;
#endif
    }

    if (strcmp(tf_cmd, "train") == 0) {
        if (!manifest || !output) {
            fprintf(stderr, "error: --manifest and -o/--output are required for training\n");
            return 1;
        }

        char metadata_path[1024];
        snprintf(metadata_path, sizeof(metadata_path), "%s/metadata.json", output);

        struct stat st;
        if (stat(output, &st) != 0 || stat(metadata_path, &st) != 0) {
            const char *python = getenv("DM_PYTHON");
            const char *script = getenv("DM_TF_MOBILENET_SCRIPT");
            if (!python || !python[0]) python = access(".venv/bin/python", X_OK) == 0 ? ".venv/bin/python" : "python3";
            if (!script || !script[0]) script = "src/models/vision/tf_mobilenetv4.py";

            char init_cmd[2048];
            snprintf(init_cmd, sizeof(init_cmd), "%s %s init-savedmodel -o %s --classes %d --size %d --width %f --dropout %f --lr %f",
                     python, script, output, classes, size, width, dropout, lr);
            printf("Initializing trainable SavedModel using command: %s\n", init_cmd);
            int rc = system(init_cmd);
            if (rc != 0) {
                fprintf(stderr, "error: failed to initialize SavedModel\n");
                return 1;
            }
        }

        char *json = read_entire_file(metadata_path);
        if (!json) {
            fprintf(stderr, "error: cannot read metadata.json\n");
            return 1;
        }

        char train_image_name[128], train_label_name[128];
        char train_loss_name[128], train_acc_name[128];
        char save_path_name[128], save_status_name[128];
        char load_path_name[128], load_status_name[128];

        if (parse_metadata_key(json, "\"train\"", "\"image\"", train_image_name, sizeof(train_image_name)) != 0 ||
            parse_metadata_key(json, "\"train\"", "\"label\"", train_label_name, sizeof(train_label_name)) != 0 ||
            parse_metadata_key(json, "\"train\"", "\"loss\"", train_loss_name, sizeof(train_loss_name)) != 0 ||
            parse_metadata_key(json, "\"train\"", "\"accuracy\"", train_acc_name, sizeof(train_acc_name)) != 0 ||
            parse_metadata_key(json, "\"save\"", "\"path\"", save_path_name, sizeof(save_path_name)) != 0 ||
            parse_metadata_key(json, "\"save\"", "\"status\"", save_status_name, sizeof(save_status_name)) != 0 ||
            parse_metadata_key(json, "\"load\"", "\"path\"", load_path_name, sizeof(load_path_name)) != 0 ||
            parse_metadata_key(json, "\"load\"", "\"status\"", load_status_name, sizeof(load_status_name)) != 0) {
            fprintf(stderr, "error: invalid metadata.json format\n");
            free(json);
            return 1;
        }
        free(json);

        TF_Status* status = TF_NewStatus();
        TF_Graph* graph = TF_NewGraph();
        TF_SessionOptions* opts = TF_NewSessionOptions();
        const char* tags[] = {"serve"};
        TF_Session* session = TF_LoadSessionFromSavedModel(opts, NULL, output, tags, 1, graph, NULL, status);
        TF_DeleteSessionOptions(opts);
        if (TF_GetCode(status) != TF_OK) {
            fprintf(stderr, "error: TF_LoadSessionFromSavedModel failed: %s\n", TF_Message(status));
            TF_DeleteGraph(graph);
            TF_DeleteStatus(status);
            return 1;
        }

        char ckpt_index_path[1024];
        snprintf(ckpt_index_path, sizeof(ckpt_index_path), "%s/variables/variables_train.index", output);
        if (access(ckpt_index_path, F_OK) == 0) {
            char ckpt_prefix[1024];
            snprintf(ckpt_prefix, sizeof(ckpt_prefix), "%s/variables/variables_train", output);
            printf("Restoring weights from custom checkpoint: %s\n", ckpt_prefix);
            if (run_save_or_load_signature(session, graph, status, load_path_name, load_status_name, ckpt_prefix) != 0) {
                fprintf(stderr, "error: failed to restore weights from custom checkpoint\n");
                TF_DeleteSession(session, status);
                TF_DeleteGraph(graph);
                TF_DeleteStatus(status);
                return 1;
            }
        } else {
            printf("No custom checkpoint found. Using base SavedModel weights.\n");
        }

        printf("Starting training: epochs=%d batch_size=%d lr=%f size=%d\n", epochs, batch_size, lr, size);
        for (int epoch = 0; epoch < epochs; epoch++) {
            FILE *fp = fopen(manifest, "r");
            if (!fp) {
                fprintf(stderr, "error: cannot open manifest %s\n", manifest);
                TF_DeleteSession(session, status);
                TF_DeleteGraph(graph);
                TF_DeleteStatus(status);
                return 1;
            }

            char path[4096];
            int label;
            int batch_count = 0;
            char **batch_paths = malloc(sizeof(char*) * batch_size);
            int *batch_labels = malloc(sizeof(int) * batch_size);

            double epoch_loss = 0.0;
            int epoch_correct = 0;
            int epoch_total = 0;

            while (1) {
                int read_ok = (fscanf(fp, "%4095s %d", path, &label) == 2);
                if (read_ok) {
                    batch_paths[batch_count] = strdup(path);
                    batch_labels[batch_count] = label;
                    batch_count++;
                }

                if (batch_count == batch_size || (!read_ok && batch_count > 0)) {
                    int N = batch_count;
                    float *img_data = malloc(sizeof(float) * N * size * size * 3);
                    int32_t *lbl_data = malloc(sizeof(int32_t) * N);
                    int loaded_samples = 0;

                    for (int b = 0; b < N; b++) {
                        DM_Tensor img = {0}, resized = {0};
                        if (dm_image_load_ppm_rgb_f32(batch_paths[b], &img) != 0 ||
                            dm_image_resize_nearest(&img, &resized, size, size) != 0) {
                            dm_tensor_free(&img);
                            dm_tensor_free(&resized);
                            continue;
                        }

                        lbl_data[loaded_samples] = batch_labels[b];
                        for (int y = 0; y < size; y++) {
                            for (int x = 0; x < size; x++) {
                                for (int c = 0; c < 3; c++) {
                                    float val = dm_tensor_get(&resized, 0, c, y, x);
                                    float norm = val * 0.5f + 0.5f;
                                    size_t idx = ((size_t)loaded_samples * size * size * 3) + ((size_t)y * size * 3) + ((size_t)x * 3) + c;
                                    img_data[idx] = norm;
                                }
                            }
                        }
                        loaded_samples++;
                        dm_tensor_free(&img);
                        dm_tensor_free(&resized);
                    }

                    if (loaded_samples > 0) {
                        int64_t img_dims[4] = {loaded_samples, size, size, 3};
                        TF_Tensor* img_tensor = TF_NewTensor(TF_FLOAT, img_dims, 4, img_data, sizeof(float) * loaded_samples * size * size * 3, tensor_deallocator, NULL);

                        int64_t lbl_dims[1] = {loaded_samples};
                        TF_Tensor* lbl_tensor = TF_NewTensor(TF_INT32, lbl_dims, 1, lbl_data, sizeof(int32_t) * loaded_samples, tensor_deallocator, NULL);

                        char image_op_name[128], label_op_name[128];
                        char loss_op_name[128], acc_op_name[128];
                        get_op_name(train_image_name, image_op_name);
                        get_op_name(train_label_name, label_op_name);
                        get_op_name(train_loss_name, loss_op_name);
                        get_op_name(train_acc_name, acc_op_name);

                        TF_Output inputs[2] = {
                            {TF_GraphOperationByName(graph, image_op_name), 0},
                            {TF_GraphOperationByName(graph, label_op_name), 0}
                        };
                        TF_Tensor* input_tensors[2] = {img_tensor, lbl_tensor};

                        TF_Output outputs[2] = {
                            {TF_GraphOperationByName(graph, loss_op_name), 0},
                            {TF_GraphOperationByName(graph, acc_op_name), 0}
                        };
                        TF_Tensor* output_tensors[2] = {NULL, NULL};

                        TF_SessionRun(session, NULL, inputs, input_tensors, 2, outputs, output_tensors, 2, NULL, 0, NULL, status);

                        TF_DeleteTensor(img_tensor);
                        TF_DeleteTensor(lbl_tensor);

                        if (TF_GetCode(status) != TF_OK) {
                            fprintf(stderr, "Session run failed: %s\n", TF_Message(status));
                            fclose(fp);
                            free(batch_paths);
                            free(batch_labels);
                            TF_DeleteSession(session, status);
                            TF_DeleteGraph(graph);
                            TF_DeleteStatus(status);
                            return 1;
                        }

                        float loss_val = *(float*)TF_TensorData(output_tensors[0]);
                        float acc_val = *(float*)TF_TensorData(output_tensors[1]);

                        epoch_loss += (double)loss_val * loaded_samples;
                        epoch_correct += (int)(acc_val * loaded_samples);
                        epoch_total += loaded_samples;

                        TF_DeleteTensor(output_tensors[0]);
                        TF_DeleteTensor(output_tensors[1]);
                    } else {
                        free(img_data);
                        free(lbl_data);
                    }

                    for (int b = 0; b < N; b++) free(batch_paths[b]);
                    batch_count = 0;
                }

                if (!read_ok) break;
            }
            fclose(fp);
            free(batch_paths);
            free(batch_labels);

            printf("epoch=%d loss=%.6f accuracy=%.6f samples=%d\n",
                   epoch + 1,
                   epoch_total ? epoch_loss / epoch_total : 0.0,
                   epoch_total ? (double)epoch_correct / epoch_total : 0.0,
                   epoch_total);
        }

        char ckpt_prefix[1024];
        snprintf(ckpt_prefix, sizeof(ckpt_prefix), "%s/variables/variables_train", output);
        printf("Saving trained weights to: %s\n", ckpt_prefix);
        if (run_save_or_load_signature(session, graph, status, save_path_name, save_status_name, ckpt_prefix) != 0) {
            fprintf(stderr, "error: failed to save checkpoint\n");
            TF_DeleteSession(session, status);
            TF_DeleteGraph(graph);
            TF_DeleteStatus(status);
            return 1;
        }

        TF_DeleteSession(session, status);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);
        return 0;
    }

    if (strcmp(tf_cmd, "eval") == 0) {
        if (!manifest || !model_path) {
            fprintf(stderr, "error: --manifest and -m/--model are required for evaluation\n");
            return 1;
        }

        char metadata_path[1024];
        snprintf(metadata_path, sizeof(metadata_path), "%s/metadata.json", model_path);

        char *json = read_entire_file(metadata_path);
        if (!json) {
            fprintf(stderr, "error: cannot read metadata.json\n");
            return 1;
        }

        char eval_image_name[128], eval_label_name[128];
        char eval_loss_name[128], eval_acc_name[128];
        char load_path_name[128], load_status_name[128];

        if (parse_metadata_key(json, "\"eval\"", "\"image\"", eval_image_name, sizeof(eval_image_name)) != 0 ||
            parse_metadata_key(json, "\"eval\"", "\"label\"", eval_label_name, sizeof(eval_label_name)) != 0 ||
            parse_metadata_key(json, "\"eval\"", "\"loss\"", eval_loss_name, sizeof(eval_loss_name)) != 0 ||
            parse_metadata_key(json, "\"eval\"", "\"accuracy\"", eval_acc_name, sizeof(eval_acc_name)) != 0 ||
            parse_metadata_key(json, "\"load\"", "\"path\"", load_path_name, sizeof(load_path_name)) != 0 ||
            parse_metadata_key(json, "\"load\"", "\"status\"", load_status_name, sizeof(load_status_name)) != 0) {
            fprintf(stderr, "error: invalid metadata.json format\n");
            free(json);
            return 1;
        }
        free(json);

        TF_Status* status = TF_NewStatus();
        TF_Graph* graph = TF_NewGraph();
        TF_SessionOptions* opts = TF_NewSessionOptions();
        const char* tags[] = {"serve"};
        TF_Session* session = TF_LoadSessionFromSavedModel(opts, NULL, model_path, tags, 1, graph, NULL, status);
        TF_DeleteSessionOptions(opts);
        if (TF_GetCode(status) != TF_OK) {
            fprintf(stderr, "error: TF_LoadSessionFromSavedModel failed: %s\n", TF_Message(status));
            TF_DeleteGraph(graph);
            TF_DeleteStatus(status);
            return 1;
        }

        char ckpt_prefix[1024];
        snprintf(ckpt_prefix, sizeof(ckpt_prefix), "%s/variables/variables", model_path);
        if (run_save_or_load_signature(session, graph, status, load_path_name, load_status_name, ckpt_prefix) != 0) {
            printf("warning: custom checkpoint restore failed (normal on the first run). Re-initializing session to restore SavedModel base weights...\n");
            TF_DeleteSession(session, status);
            TF_DeleteGraph(graph);
            graph = TF_NewGraph();
            TF_SessionOptions* opts = TF_NewSessionOptions();
            session = TF_LoadSessionFromSavedModel(opts, NULL, model_path, tags, 1, graph, NULL, status);
            TF_DeleteSessionOptions(opts);
            if (TF_GetCode(status) != TF_OK) {
                fprintf(stderr, "error: failed to re-initialize session: %s\n", TF_Message(status));
                TF_DeleteGraph(graph);
                TF_DeleteStatus(status);
                return 1;
            }
        }

        FILE *fp = fopen(manifest, "r");
        if (!fp) {
            fprintf(stderr, "error: cannot open manifest %s\n", manifest);
            TF_DeleteSession(session, status);
            TF_DeleteGraph(graph);
            TF_DeleteStatus(status);
            return 1;
        }

        char path[4096];
        int label;
        int batch_count = 0;
        char **batch_paths = malloc(sizeof(char*) * batch_size);
        int *batch_labels = malloc(sizeof(int) * batch_size);

        double total_loss = 0.0;
        int total_correct = 0;
        int total_samples = 0;

        while (1) {
            int read_ok = (fscanf(fp, "%4095s %d", path, &label) == 2);
            if (read_ok) {
                batch_paths[batch_count] = strdup(path);
                batch_labels[batch_count] = label;
                batch_count++;
            }

            if (batch_count == batch_size || (!read_ok && batch_count > 0)) {
                int N = batch_count;
                float *img_data = malloc(sizeof(float) * N * size * size * 3);
                int32_t *lbl_data = malloc(sizeof(int32_t) * N);
                int loaded_samples = 0;

                for (int b = 0; b < N; b++) {
                    DM_Tensor img = {0}, resized = {0};
                    if (dm_image_load_ppm_rgb_f32(batch_paths[b], &img) != 0 ||
                        dm_image_resize_nearest(&img, &resized, size, size) != 0) {
                        dm_tensor_free(&img);
                        dm_tensor_free(&resized);
                        continue;
                    }

                    lbl_data[loaded_samples] = batch_labels[b];
                    for (int y = 0; y < size; y++) {
                        for (int x = 0; x < size; x++) {
                            for (int c = 0; c < 3; c++) {
                                float val = dm_tensor_get(&resized, 0, c, y, x);
                                float norm = val * 0.5f + 0.5f;
                                size_t idx = ((size_t)loaded_samples * size * size * 3) + ((size_t)y * size * 3) + ((size_t)x * 3) + c;
                                img_data[idx] = norm;
                            }
                        }
                    }
                    loaded_samples++;
                    dm_tensor_free(&img);
                    dm_tensor_free(&resized);
                }

                if (loaded_samples > 0) {
                    int64_t img_dims[4] = {loaded_samples, size, size, 3};
                    TF_Tensor* img_tensor = TF_NewTensor(TF_FLOAT, img_dims, 4, img_data, sizeof(float) * loaded_samples * size * size * 3, tensor_deallocator, NULL);

                    int64_t lbl_dims[1] = {loaded_samples};
                    TF_Tensor* lbl_tensor = TF_NewTensor(TF_INT32, lbl_dims, 1, lbl_data, sizeof(int32_t) * loaded_samples, tensor_deallocator, NULL);

                    char image_op_name[128], label_op_name[128];
                    char loss_op_name[128], acc_op_name[128];
                    get_op_name(eval_image_name, image_op_name);
                    get_op_name(eval_label_name, label_op_name);
                    get_op_name(eval_loss_name, loss_op_name);
                    get_op_name(eval_acc_name, acc_op_name);

                    TF_Output inputs[2] = {
                        {TF_GraphOperationByName(graph, image_op_name), 0},
                        {TF_GraphOperationByName(graph, label_op_name), 0}
                    };
                    TF_Tensor* input_tensors[2] = {img_tensor, lbl_tensor};

                    TF_Output outputs[2] = {
                        {TF_GraphOperationByName(graph, loss_op_name), 0},
                        {TF_GraphOperationByName(graph, acc_op_name), 0}
                    };
                    TF_Tensor* output_tensors[2] = {NULL, NULL};

                    TF_SessionRun(session, NULL, inputs, input_tensors, 2, outputs, output_tensors, 2, NULL, 0, NULL, status);

                    TF_DeleteTensor(img_tensor);
                    TF_DeleteTensor(lbl_tensor);

                    if (TF_GetCode(status) != TF_OK) {
                        fprintf(stderr, "Session run failed: %s\n", TF_Message(status));
                        fclose(fp);
                        free(batch_paths);
                        free(batch_labels);
                        TF_DeleteSession(session, status);
                        TF_DeleteGraph(graph);
                        TF_DeleteStatus(status);
                        return 1;
                    }

                    float loss_val = *(float*)TF_TensorData(output_tensors[0]);
                    float acc_val = *(float*)TF_TensorData(output_tensors[1]);

                    total_loss += (double)loss_val * loaded_samples;
                    total_correct += (int)(acc_val * loaded_samples);
                    total_samples += loaded_samples;

                    TF_DeleteTensor(output_tensors[0]);
                    TF_DeleteTensor(output_tensors[1]);
                } else {
                    free(img_data);
                    free(lbl_data);
                }

                for (int b = 0; b < N; b++) free(batch_paths[b]);
                batch_count = 0;
            }

            if (!read_ok) break;
        }
        fclose(fp);
        free(batch_paths);
        free(batch_labels);

        printf("loss=%.6f accuracy=%.6f samples=%d\n",
               total_samples ? total_loss / total_samples : 0.0,
               total_samples ? (double)total_correct / total_samples : 0.0,
               total_samples);

        TF_DeleteSession(session, status);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);
        return 0;
    }

    if (strcmp(tf_cmd, "predict") == 0) {
        if (!model_path || !input_image) {
            fprintf(stderr, "error: --model and --input are required for prediction\n");
            return 1;
        }

        char metadata_path[1024];
        snprintf(metadata_path, sizeof(metadata_path), "%s/metadata.json", model_path);

        char *json = read_entire_file(metadata_path);
        if (!json) {
            fprintf(stderr, "error: cannot read metadata.json\n");
            return 1;
        }

        char pred_image_name[128], pred_probs_name[128];
        char load_path_name[128], load_status_name[128];

        if (parse_metadata_key(json, "\"predict\"", "\"image\"", pred_image_name, sizeof(pred_image_name)) != 0 ||
            parse_metadata_key(json, "\"predict\"", "\"probs\"", pred_probs_name, sizeof(pred_probs_name)) != 0 ||
            parse_metadata_key(json, "\"load\"", "\"path\"", load_path_name, sizeof(load_path_name)) != 0 ||
            parse_metadata_key(json, "\"load\"", "\"status\"", load_status_name, sizeof(load_status_name)) != 0) {
            fprintf(stderr, "error: invalid metadata.json format\n");
            free(json);
            return 1;
        }
        free(json);

        TF_Status* status = TF_NewStatus();
        TF_Graph* graph = TF_NewGraph();
        TF_SessionOptions* opts = TF_NewSessionOptions();
        const char* tags[] = {"serve"};
        TF_Session* session = TF_LoadSessionFromSavedModel(opts, NULL, model_path, tags, 1, graph, NULL, status);
        TF_DeleteSessionOptions(opts);
        if (TF_GetCode(status) != TF_OK) {
            fprintf(stderr, "error: TF_LoadSessionFromSavedModel failed: %s\n", TF_Message(status));
            TF_DeleteGraph(graph);
            TF_DeleteStatus(status);
            return 1;
        }

        char ckpt_prefix[1024];
        snprintf(ckpt_prefix, sizeof(ckpt_prefix), "%s/variables/variables", model_path);
        if (run_save_or_load_signature(session, graph, status, load_path_name, load_status_name, ckpt_prefix) != 0) {
            printf("warning: custom checkpoint restore failed (normal on the first run). Re-initializing session to restore SavedModel base weights...\n");
            TF_DeleteSession(session, status);
            TF_DeleteGraph(graph);
            graph = TF_NewGraph();
            TF_SessionOptions* opts = TF_NewSessionOptions();
            session = TF_LoadSessionFromSavedModel(opts, NULL, model_path, tags, 1, graph, NULL, status);
            TF_DeleteSessionOptions(opts);
            if (TF_GetCode(status) != TF_OK) {
                fprintf(stderr, "error: failed to re-initialize session: %s\n", TF_Message(status));
                TF_DeleteGraph(graph);
                TF_DeleteStatus(status);
                return 1;
            }
        }

        DM_Tensor img = {0}, resized = {0};
        if (dm_image_load_ppm_rgb_f32(input_image, &img) != 0 ||
            dm_image_resize_nearest(&img, &resized, size, size) != 0) {
            fprintf(stderr, "error: failed to load/resize image %s\n", input_image);
            dm_tensor_free(&img);
            dm_tensor_free(&resized);
            TF_DeleteSession(session, status);
            TF_DeleteGraph(graph);
            TF_DeleteStatus(status);
            return 1;
        }

        float *img_data = malloc(sizeof(float) * 1 * size * size * 3);
        for (int y = 0; y < size; y++) {
            for (int x = 0; x < size; x++) {
                for (int c = 0; c < 3; c++) {
                    float val = dm_tensor_get(&resized, 0, c, y, x);
                    float norm = val * 0.5f + 0.5f;
                    size_t idx = ((size_t)y * size * 3) + ((size_t)x * 3) + c;
                    img_data[idx] = norm;
                }
            }
        }
        dm_tensor_free(&img);
        dm_tensor_free(&resized);

        int64_t img_dims[4] = {1, size, size, 3};
        TF_Tensor* img_tensor = TF_NewTensor(TF_FLOAT, img_dims, 4, img_data, sizeof(float) * 1 * size * size * 3, tensor_deallocator, NULL);

        char image_op_name[128], probs_op_name[128];
        get_op_name(pred_image_name, image_op_name);
        get_op_name(pred_probs_name, probs_op_name);

        TF_Output input = {TF_GraphOperationByName(graph, image_op_name), 0};
        TF_Tensor* input_tensors[1] = {img_tensor};

        TF_Output output = {TF_GraphOperationByName(graph, probs_op_name), 0};
        TF_Tensor* output_tensors[1] = {NULL};

        TF_SessionRun(session, NULL, &input, input_tensors, 1, &output, output_tensors, 1, NULL, 0, NULL, status);
        TF_DeleteTensor(img_tensor);

        if (TF_GetCode(status) != TF_OK) {
            fprintf(stderr, "Session run failed: %s\n", TF_Message(status));
            TF_DeleteSession(session, status);
            TF_DeleteGraph(graph);
            TF_DeleteStatus(status);
            return 1;
        }

        float *probs_data = (float*)TF_TensorData(output_tensors[0]);
        int64_t num_classes = TF_Dim(output_tensors[0], 1);

        printf("model=mobilenet_tiny_mnv4_style\n");
        printf("input=%s\nclasses=%d\nimage_size=%d\n", input_image, (int)num_classes, size);

        int *visited = calloc((size_t)num_classes, sizeof(int));
        for (int rank = 0; rank < top && rank < num_classes; rank++) {
            int best = -1;
            float best_val = -1.0f;
            for (int c = 0; c < num_classes; c++) {
                if (!visited[c] && probs_data[c] > best_val) {
                    best_val = probs_data[c];
                    best = c;
                }
            }
            if (best != -1) {
                visited[best] = 1;
                printf("rank%d_class=%d prob=%.8f\n", rank + 1, best, best_val);
            }
        }
        free(visited);
        TF_DeleteTensor(output_tensors[0]);

        TF_DeleteSession(session, status);
        TF_DeleteGraph(graph);
        TF_DeleteStatus(status);
        return 0;
    }

    fprintf(stderr, "error: unknown tf_cmd %s\n", tf_cmd);
    return 1;
}

int dm_mobilenet_tiny_cli(int argc, char **argv) {
    const char *cmd;
    if (argc < 2) { usage(argv[0]); return 2; }
    cmd = argv[1];
    if (strcmp(cmd, "mobilenet_tiny") == 0 || strcmp(cmd, "mobilenetv4_tiny") == 0 || strcmp(cmd, "dm_mobilenet_tiny") == 0) {
        if (argc < 3) { usage(argv[0]); return 2; }
        cmd = argv[2];
        argv++; argc--;
    }
    if (strcmp(cmd, "infer") == 0) return cmd_infer(argc, argv);
    if (strcmp(cmd, "train") == 0) return cmd_train(argc, argv);
    if (strcmp(cmd, "tf-train") == 0 || strcmp(cmd, "e2e-train") == 0) return run_tf_backend(argc, argv, "train");
    if (strcmp(cmd, "tf-eval") == 0 || strcmp(cmd, "e2e-eval") == 0) return run_tf_backend(argc, argv, "eval");
    if (strcmp(cmd, "tf-predict") == 0 || strcmp(cmd, "e2e-predict") == 0) return run_tf_backend(argc, argv, "predict");
    if (strcmp(cmd, "tf-export") == 0 || strcmp(cmd, "e2e-export") == 0) return run_tf_backend(argc, argv, "export");
    if (strcmp(cmd, "bench-block") == 0) return cmd_bench_block(argc, argv);
    usage(argv[0]);
    return 2;
}

#ifndef DM_API
#if defined(_WIN32) && defined(DM_BUILDING_LIB)
#  define DM_API __declspec(dllexport)
#elif defined(__GNUC__) && defined(DM_BUILDING_LIB)
#  define DM_API __attribute__((visibility("default")))
#else
#  define DM_API
#endif
#endif

typedef enum {
    DM_OK                = 0,
    DM_ERR_GENERIC       = -1,
    DM_ERR_IO            = -2,
    DM_ERR_MEMORY        = -3,
    DM_ERR_INVALID_PARAM = -4
} DM_Status;

DM_API DM_Status dm_mobilenet_tiny_forward_raw(const float *img_nchw,
                                               int          img_size,
                                               int          classes,
                                               unsigned int seed,
                                               const float *head_w,
                                               const float *head_b,
                                               float       *logits_out)
{
    if (!img_nchw || !logits_out || img_size <= 0 || classes <= 0) return DM_ERR_INVALID_PARAM;

    DM_Tensor input = {0};
    if (dm_tensor_alloc(&input, 1, 3, img_size, img_size) != 0) return DM_ERR_MEMORY;
    memcpy(input.data, img_nchw, 3 * img_size * img_size * sizeof(float));

    DM_Tensor features = {0};
    int rc = mobilenet_features(&input, &features, seed);
    dm_tensor_free(&input);
    if (rc != 0) {
        dm_tensor_free(&features);
        return DM_ERR_GENERIC;
    }

    DM_Tensor logits = {0};
    if (dm_tensor_alloc(&logits, 1, classes, 1, 1) != 0) {
        dm_tensor_free(&features);
        return DM_ERR_MEMORY;
    }

    float *w_to_use = (float *)head_w;
    float *w_alloc = NULL;
    if (!w_to_use) {
        w_alloc = make_weights((size_t)classes * features.c, seed + 700, 0.05f);
        if (!w_alloc) {
            dm_tensor_free(&features);
            dm_tensor_free(&logits);
            return DM_ERR_MEMORY;
        }
        w_to_use = w_alloc;
    }

    if (dm_linear(&features, &logits, w_to_use, head_b, classes) != 0) {
        free(w_alloc);
        dm_tensor_free(&features);
        dm_tensor_free(&logits);
        return DM_ERR_GENERIC;
    }
    dm_softmax(&logits);

    memcpy(logits_out, logits.data, classes * sizeof(float));

    free(w_alloc);
    dm_tensor_free(&features);
    dm_tensor_free(&logits);
    return DM_OK;
}

DM_API DM_Status dm_mobilenet_tiny_head_load(const char    *path,
                                             int           *classes_out,
                                             int           *feature_dim_out,
                                             int           *image_size_out,
                                             unsigned int  *seed_out,
                                             float        **w_out,
                                             float        **b_out)
{
    if (!path || !w_out || !b_out) return DM_ERR_INVALID_PARAM;
    TinyHead h;
    memset(&h, 0, sizeof(h));
    if (head_load(&h, path) != 0) return DM_ERR_GENERIC;
    if (classes_out)     *classes_out     = h.classes;
    if (feature_dim_out) *feature_dim_out = h.feature_dim;
    if (image_size_out)  *image_size_out  = h.image_size;
    if (seed_out)        *seed_out        = h.seed;
    *w_out = h.w;
    *b_out = h.b;
    return DM_OK;
}

DM_API DM_Status dm_mobilenet_tiny_head_save(const char  *path,
                                             int          classes,
                                             int          feature_dim,
                                             int          image_size,
                                             unsigned int seed,
                                             const float *w,
                                             const float *b)
{
    if (!path || !w) return DM_ERR_INVALID_PARAM;
    TinyHead h;
    h.classes = classes;
    h.feature_dim = feature_dim;
    h.image_size = image_size;
    h.seed = seed;
    h.w = (float *)w;
    h.b = (float *)b;
    if (head_save(&h, path) != 0) return DM_ERR_GENERIC;
    return DM_OK;
}

DM_API void dm_mobilenet_tiny_head_free(float *w, float *b)
{
    free(w);
    free(b);
}

