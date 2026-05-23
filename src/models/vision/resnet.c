#include "models/vision/resnet.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "core/dm_benchmark.h"

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s resnet18 bench [--size N] [--classes N] [--seed N]\n\n",
        prog);
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

int dm_resnet_basic_block(const DM_Tensor *in, DM_Tensor *out, int out_c, int stride, unsigned int seed) {
    DM_Tensor x1 = {0};
    DM_Tensor x2 = {0};
    DM_Tensor shortcut = {0};

    // Main path: Conv1 (3x3)
    float *w1 = make_weights((size_t)out_c * in->c * 3 * 3, seed + 1, 0.08f);
    if (!w1) return -1;
    int rc = dm_conv2d_same(in, &x1, w1, NULL, out_c, 3, stride);
    free(w1);
    if (rc != 0) return -1;

    // BN1
    float *gamma1 = make_weights(out_c, seed + 2, 1.0f);
    float *beta1 = make_weights(out_c, seed + 3, 0.0f);
    float *mean1 = make_weights(out_c, seed + 4, 0.0f);
    float *var1 = make_weights(out_c, seed + 5, 1.0f);
    if (!gamma1 || !beta1 || !mean1 || !var1) {
        free(gamma1); free(beta1); free(mean1); free(var1);
        dm_tensor_free(&x1); return -1;
    }
    rc = dm_batch_norm(&x1, gamma1, beta1, mean1, var1, 1e-5f);
    free(gamma1); free(beta1); free(mean1); free(var1);
    if (rc != 0) { dm_tensor_free(&x1); return -1; }

    // ReLU
    dm_relu(&x1);

    // Conv2 (3x3)
    float *w2 = make_weights((size_t)out_c * out_c * 3 * 3, seed + 6, 0.08f);
    if (!w2) { dm_tensor_free(&x1); return -1; }
    rc = dm_conv2d_same(&x1, &x2, w2, NULL, out_c, 3, 1);
    free(w2);
    dm_tensor_free(&x1);
    if (rc != 0) return -1;

    // BN2
    float *gamma2 = make_weights(out_c, seed + 7, 1.0f);
    float *beta2 = make_weights(out_c, seed + 8, 0.0f);
    float *mean2 = make_weights(out_c, seed + 9, 0.0f);
    float *var2 = make_weights(out_c, seed + 10, 1.0f);
    if (!gamma2 || !beta2 || !mean2 || !var2) {
        free(gamma2); free(beta2); free(mean2); free(var2);
        dm_tensor_free(&x2); return -1;
    }
    rc = dm_batch_norm(&x2, gamma2, beta2, mean2, var2, 1e-5f);
    free(gamma2); free(beta2); free(mean2); free(var2);
    if (rc != 0) { dm_tensor_free(&x2); return -1; }

    // Shortcut path
    if (stride != 1 || in->c != out_c) {
        // Conv1x1 shortcut
        float *w_short = make_weights((size_t)out_c * in->c * 1 * 1, seed + 11, 0.08f);
        if (!w_short) { dm_tensor_free(&x2); return -1; }
        rc = dm_conv2d_same(in, &shortcut, w_short, NULL, out_c, 1, stride);
        free(w_short);
        if (rc != 0) { dm_tensor_free(&x2); return -1; }

        // BN shortcut
        float *gamma_s = make_weights(out_c, seed + 12, 1.0f);
        float *beta_s = make_weights(out_c, seed + 13, 0.0f);
        float *mean_s = make_weights(out_c, seed + 14, 0.0f);
        float *var_s = make_weights(out_c, seed + 15, 1.0f);
        if (!gamma_s || !beta_s || !mean_s || !var_s) {
            free(gamma_s); free(beta_s); free(mean_s); free(var_s);
            dm_tensor_free(&x2); dm_tensor_free(&shortcut); return -1;
        }
        rc = dm_batch_norm(&shortcut, gamma_s, beta_s, mean_s, var_s, 1e-5f);
        free(gamma_s); free(beta_s); free(mean_s); free(var_s);
        if (rc != 0) { dm_tensor_free(&x2); dm_tensor_free(&shortcut); return -1; }
    } else {
        // Identity: just copy input
        if (dm_tensor_alloc(&shortcut, in->n, in->c, in->h, in->w) != 0) {
            dm_tensor_free(&x2); return -1;
        }
        size_t cnt = dm_tensor_count(in);
        memcpy(shortcut.data, in->data, cnt * sizeof(float));
    }

    // Add main path + shortcut
    rc = dm_tensor_add(&x2, &shortcut);
    dm_tensor_free(&shortcut);
    if (rc != 0) { dm_tensor_free(&x2); return -1; }

    // Final ReLU
    dm_relu(&x2);

    *out = x2;
    return 0;
}

int dm_resnet18_forward(const DM_Tensor *input, DM_Tensor *logits, int classes, unsigned int seed) {
    DM_Tensor t1 = {0}, t2 = {0};
    unsigned int s = seed;

    // Conv1: 7x7, stride 2, output 64 channels
    float *w_conv1 = make_weights((size_t)64 * input->c * 7 * 7, s++, 0.08f);
    if (!w_conv1) return -1;
    int rc = dm_conv2d_same(input, &t1, w_conv1, NULL, 64, 7, 2);
    free(w_conv1);
    if (rc != 0) return -1;

    // BN1
    float *g1 = make_weights(64, s++, 1.0f);
    float *b1 = make_weights(64, s++, 0.0f);
    float *m1 = make_weights(64, s++, 0.0f);
    float *v1 = make_weights(64, s++, 1.0f);
    if (!g1 || !b1 || !m1 || !v1) {
        free(g1); free(b1); free(m1); free(v1);
        dm_tensor_free(&t1); return -1;
    }
    rc = dm_batch_norm(&t1, g1, b1, m1, v1, 1e-5f);
    free(g1); free(b1); free(m1); free(v1);
    if (rc != 0) { dm_tensor_free(&t1); return -1; }

    // ReLU1
    dm_relu(&t1);

    // MaxPool
    rc = dm_max_pool2d_same(&t1, &t2, 3, 2);
    dm_tensor_free(&t1);
    if (rc != 0) return -1;

    // Stage 1 Basic Block 1: 64 channels, stride 1
    rc = dm_resnet_basic_block(&t2, &t1, 64, 1, s); s += 20;
    dm_tensor_free(&t2);
    if (rc != 0) return -1;

    // Stage 1 Basic Block 2: 64 channels, stride 1
    rc = dm_resnet_basic_block(&t1, &t2, 64, 1, s); s += 20;
    dm_tensor_free(&t1);
    if (rc != 0) return -1;

    // Stage 2 Basic Block 1: 128 channels, stride 2
    rc = dm_resnet_basic_block(&t2, &t1, 128, 2, s); s += 20;
    dm_tensor_free(&t2);
    if (rc != 0) return -1;

    // Stage 2 Basic Block 2: 128 channels, stride 1
    rc = dm_resnet_basic_block(&t1, &t2, 128, 1, s); s += 20;
    dm_tensor_free(&t1);
    if (rc != 0) return -1;

    // Stage 3 Basic Block 1: 256 channels, stride 2
    rc = dm_resnet_basic_block(&t2, &t1, 256, 2, s); s += 20;
    dm_tensor_free(&t2);
    if (rc != 0) return -1;

    // Stage 3 Basic Block 2: 256 channels, stride 1
    rc = dm_resnet_basic_block(&t1, &t2, 256, 1, s); s += 20;
    dm_tensor_free(&t1);
    if (rc != 0) return -1;

    // Stage 4 Basic Block 1: 512 channels, stride 2
    rc = dm_resnet_basic_block(&t2, &t1, 512, 2, s); s += 20;
    dm_tensor_free(&t2);
    if (rc != 0) return -1;

    // Stage 4 Basic Block 2: 512 channels, stride 1
    rc = dm_resnet_basic_block(&t1, &t2, 512, 1, s); s += 20;
    dm_tensor_free(&t1);
    if (rc != 0) return -1;

    // Global Average Pool
    rc = dm_global_avg_pool(&t2, &t1);
    dm_tensor_free(&t2);
    if (rc != 0) return -1;

    // FC/Linear: input channels = 512, output channels = classes
    float *w_fc = make_weights((size_t)classes * 512, s++, 0.08f);
    float *b_fc = make_weights(classes, s++, 0.0f);
    if (!w_fc || !b_fc) {
        free(w_fc); free(b_fc);
        dm_tensor_free(&t1); return -1;
    }
    rc = dm_linear(&t1, logits, w_fc, b_fc, classes);
    free(w_fc); free(b_fc);
    dm_tensor_free(&t1);
    if (rc != 0) return -1;

    return 0;
}

int dm_resnet18_cli(int argc, char **argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }
    const char *subcmd = argv[2];
    if (strcmp(subcmd, "bench") == 0) {
        int size = 224;
        int classes = 1000;
        unsigned int seed = 42;
        for (int i = 3; i < argc; i++) {
            if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) size = atoi(argv[++i]);
            else if (strcmp(argv[i], "--classes") == 0 && i + 1 < argc) classes = atoi(argv[++i]);
            else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (unsigned int)atoi(argv[++i]);
        }

        printf("ResNet-18 Benchmark: size=%dx%d, classes=%d, seed=%u\n", size, size, classes, seed);
        DM_Tensor in, logits;
        if (dm_tensor_alloc(&in, 1, 3, size, size) != 0) {
            fprintf(stderr, "Failed to allocate input tensor\n");
            return 1;
        }
        if (dm_tensor_alloc(&logits, 1, classes, 1, 1) != 0) {
            dm_tensor_free(&in);
            fprintf(stderr, "Failed to allocate logits tensor\n");
            return 1;
        }
        dm_tensor_fill(&in, 1.0f);

        dm_bench_reset();
        dm_bench_start(DM_PHASE_TOTAL);
        dm_bench_start(DM_PHASE_ALGO);
        int rc = dm_resnet18_forward(&in, &logits, classes, seed);
        dm_bench_stop(DM_PHASE_ALGO);
        dm_bench_stop(DM_PHASE_TOTAL);

        if (rc == 0) {
            printf("ResNet-18 raw logits:\n");
            for (int i = 0; i < 5 && i < classes; i++) {
                printf("  class %d raw logit: %f\n", i, logits.data[i]);
            }
            dm_softmax(&logits);
            printf("ResNet-18 prediction top index values:\n");
            for (int i = 0; i < 5 && i < classes; i++) {
                int best_class = -1;
                float best_val = -1e30f;
                for (int c = 0; c < classes; c++) {
                    float v = logits.data[c];
                    if (v > best_val) {
                        best_val = v;
                        best_class = c;
                    }
                }
                if (best_class != -1) {
                    printf("  rank %d class: %d prob: %f\n", i + 1, best_class, best_val);
                    logits.data[best_class] = -1e30f; // mark as visited
                }
            }
            dm_bench_print_report("resnet18", "synthetic");
        } else {
            fprintf(stderr, "Forward pass failed\n");
        }

        dm_tensor_free(&in);
        dm_tensor_free(&logits);
        return rc == 0 ? 0 : 1;
    } else {
        usage(argv[0]);
        return 1;
    }
}
