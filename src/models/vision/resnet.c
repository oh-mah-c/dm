#include "models/vision/resnet.h"
#include "core/dm_engine.h"


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


int dm_resnet_basic_block(DM_WeightCache *cache, const DM_Block *in, DM_Block *out, int out_c, int stride, unsigned int seed) {
    DM_Block x1 = {0};
    DM_Block x2 = {0};
    DM_Block shortcut = {0};

    // Main path: Conv1 (3x3)
    DM_Block *w1 = dm_weight_cache_get(cache, 4, (int64_t[]){3, 3, DM_NCHW_C(in), out_c}, seed + 1, 0.08f);
    int rc = dm_conv2d_same(in, &x1, w1, NULL, out_c, 3, stride);
    if (rc != 0) return -1;

    // BN1
    DM_Block *gamma1 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 2, 1.0f);
    DM_Block *beta1 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 3, 0.0f);
    DM_Block *mean1 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 4, 0.0f);
    DM_Block *var1 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 5, 1.0f);
    rc = dm_batch_norm(&x1, (float*)gamma1->data, (float*)beta1->data, (float*)mean1->data, (float*)var1->data, 1e-5f);
    if (rc != 0) { dm_block_free(&x1); return -1; }

    // ReLU
    dm_relu(&x1);

    // Conv2 (3x3)
    DM_Block *w2 = dm_weight_cache_get(cache, 4, (int64_t[]){3, 3, out_c, out_c}, seed + 6, 0.08f);
    rc = dm_conv2d_same(&x1, &x2, w2, NULL, out_c, 3, 1);
    dm_block_free(&x1);
    if (rc != 0) return -1;

    // BN2
    DM_Block *gamma2 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 7, 1.0f);
    DM_Block *beta2 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 8, 0.0f);
    DM_Block *mean2 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 9, 0.0f);
    DM_Block *var2 = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 10, 1.0f);
    rc = dm_batch_norm(&x2, (float*)gamma2->data, (float*)beta2->data, (float*)mean2->data, (float*)var2->data, 1e-5f);
    if (rc != 0) { dm_block_free(&x2); return -1; }

    // Shortcut path
    if (stride != 1 || DM_NCHW_C(in) != out_c) {
        // Conv1x1 shortcut
        DM_Block *w_short = dm_weight_cache_get(cache, 4, (int64_t[]){1, 1, DM_NCHW_C(in), out_c}, seed + 11, 0.08f);
        rc = dm_conv2d_same(in, &shortcut, w_short, NULL, out_c, 1, stride);
        if (rc != 0) { dm_block_free(&x2); return -1; }

        // BN shortcut
        DM_Block *gamma_s = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 12, 1.0f);
        DM_Block *beta_s = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 13, 0.0f);
        DM_Block *mean_s = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 14, 0.0f);
        DM_Block *var_s = dm_weight_cache_get(cache, 1, (int64_t[]){out_c}, seed + 15, 1.0f);
        rc = dm_batch_norm(&shortcut, (float*)gamma_s->data, (float*)beta_s->data, (float*)mean_s->data, (float*)var_s->data, 1e-5f);
        if (rc != 0) { dm_block_free(&x2); dm_block_free(&shortcut); return -1; }
    } else {
        // Identity: just copy input
        if (dm_block_create(&shortcut, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){DM_NCHW_N(in), DM_NCHW_C(in), DM_NCHW_H(in), DM_NCHW_W(in)}) != 0) {
            dm_block_free(&x2); return -1;
        }
        size_t cnt = (in)->count;
        memcpy(((float*)shortcut.data), ((float*)in->data), cnt * sizeof(float));
    }

    // Add main path + shortcut
    rc = dm_tensor_add(&x2, &shortcut);
    dm_block_free(&shortcut);
    if (rc != 0) { dm_block_free(&x2); return -1; }

    // Final ReLU
    dm_relu(&x2);

    *out = x2;
    return 0;
}

int dm_resnet18_forward(const DM_Block *input, DM_Block *logits, int classes, unsigned int seed) {
    DM_WeightCache *cache = dm_weight_cache_new();
    if (!cache) return -1;
    DM_Block t1 = {0}, t2 = {0};
    unsigned int s = seed;

    // Conv1: 7x7, stride 2, output 64 channels
    DM_Block *w_conv1 = dm_weight_cache_get(cache, 4, (int64_t[]){7, 7, DM_NCHW_C(input), 64}, s++, 0.08f);
    int rc = dm_conv2d_same(input, &t1, w_conv1, NULL, 64, 7, 2);
    if (rc != 0) return -1;

    // BN1
    DM_Block *g1 = dm_weight_cache_get(cache, 1, (int64_t[]){64}, s++, 1.0f);
    DM_Block *b1 = dm_weight_cache_get(cache, 1, (int64_t[]){64}, s++, 0.0f);
    DM_Block *m1 = dm_weight_cache_get(cache, 1, (int64_t[]){64}, s++, 0.0f);
    DM_Block *v1 = dm_weight_cache_get(cache, 1, (int64_t[]){64}, s++, 1.0f);
    rc = dm_batch_norm(&t1, (float*)g1->data, (float*)b1->data, (float*)m1->data, (float*)v1->data, 1e-5f);
    if (rc != 0) { dm_weight_cache_free(cache); dm_block_free(&t1); return -1; }

    // ReLU1
    dm_relu(&t1);

    // MaxPool
    rc = dm_max_pool2d_same(&t1, &t2, 3, 2);
    dm_block_free(&t1);
    if (rc != 0) return -1;

    // Stage 1 Basic Block 1: 64 channels, stride 1
    rc = dm_resnet_basic_block(cache, &t2, &t1, 64, 1, s); s += 20;
    dm_block_free(&t2);
    if (rc != 0) return -1;

    // Stage 1 Basic Block 2: 64 channels, stride 1
    rc = dm_resnet_basic_block(cache, &t1, &t2, 64, 1, s); s += 20;
    dm_block_free(&t1);
    if (rc != 0) return -1;

    // Stage 2 Basic Block 1: 128 channels, stride 2
    rc = dm_resnet_basic_block(cache, &t2, &t1, 128, 2, s); s += 20;
    dm_block_free(&t2);
    if (rc != 0) return -1;

    // Stage 2 Basic Block 2: 128 channels, stride 1
    rc = dm_resnet_basic_block(cache, &t1, &t2, 128, 1, s); s += 20;
    dm_block_free(&t1);
    if (rc != 0) return -1;

    // Stage 3 Basic Block 1: 256 channels, stride 2
    rc = dm_resnet_basic_block(cache, &t2, &t1, 256, 2, s); s += 20;
    dm_block_free(&t2);
    if (rc != 0) return -1;

    // Stage 3 Basic Block 2: 256 channels, stride 1
    rc = dm_resnet_basic_block(cache, &t1, &t2, 256, 1, s); s += 20;
    dm_block_free(&t1);
    if (rc != 0) return -1;

    // Stage 4 Basic Block 1: 512 channels, stride 2
    rc = dm_resnet_basic_block(cache, &t2, &t1, 512, 2, s); s += 20;
    dm_block_free(&t2);
    if (rc != 0) return -1;

    // Stage 4 Basic Block 2: 512 channels, stride 1
    rc = dm_resnet_basic_block(cache, &t1, &t2, 512, 1, s); s += 20;
    dm_block_free(&t1);
    if (rc != 0) return -1;

    // Global Average Pool
    rc = dm_global_avg_pool(&t2, &t1);
    dm_block_free(&t2);
    if (rc != 0) return -1;

    // FC/Linear: input channels = 512, output channels = classes
    DM_Block *w_fc = dm_weight_cache_get(cache, 2, (int64_t[]){classes, 512}, s++, 0.08f);
    DM_Block *b_fc = dm_weight_cache_get(cache, 1, (int64_t[]){classes}, s++, 0.0f);
    rc = dm_linear(&t1, logits, w_fc, b_fc, classes);
    dm_block_free(&t1);
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
        DM_Block in, logits;
        if (dm_block_create(&in, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){1, 3, size, size}) != 0) {
            fprintf(stderr, "Failed to allocate input tensor\n");
            return 1;
        }
        if (dm_block_create(&logits, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, 4, (int64_t[]){1, classes, 1, 1}) != 0) {
            dm_block_free(&in);
            fprintf(stderr, "Failed to allocate logits tensor\n");
            return 1;
        }
        { size_t __n = (&in)->count; float *__d = (float*)(&in)->data; for(size_t __i=0; __i<__n; __i++) __d[__i] = 1.0f; }

        dm_bench_reset();
        dm_set_layout_policy(DM_LAYOUT_POLICY_AUTO_TRANSPOSE);
        dm_bench_start(DM_PHASE_TOTAL);
        dm_bench_start(DM_PHASE_ALGO);
        int rc = dm_resnet18_forward(&in, &logits, classes, seed);
        dm_bench_stop(DM_PHASE_ALGO);
        dm_bench_stop(DM_PHASE_TOTAL);

        if (rc == 0) {
            printf("ResNet-18 raw logits:\n");
            for (int i = 0; i < 5 && i < classes; i++) {
                printf("  class %d raw logit: %f\n", i, ((float*)((float*)logits.data))[i]);
            }
            dm_softmax(&logits);
            printf("ResNet-18 prediction top index values:\n");
            for (int i = 0; i < 5 && i < classes; i++) {
                int best_class = -1;
                float best_val = -1e30f;
                for (int c = 0; c < classes; c++) {
                    float v = ((float*)((float*)logits.data))[c];
                    if (v > best_val) {
                        best_val = v;
                        best_class = c;
                    }
                }
                if (best_class != -1) {
                    printf("  rank %d class: %d prob: %f\n", i + 1, best_class, best_val);
                    ((float*)((float*)logits.data))[best_class] = -1e30f; // mark as visited
                }
            }
            dm_bench_print_report("resnet18", "synthetic");
        } else {
            fprintf(stderr, "Forward pass failed\n");
        }

        dm_block_free(&in);
        dm_block_free(&logits);
        return rc == 0 ? 0 : 1;
    } else {
        usage(argv[0]);
        return 1;
    }
}
