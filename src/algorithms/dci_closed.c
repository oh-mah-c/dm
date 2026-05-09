#include "algorithms/dci_closed.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    uint64_t *words;
    size_t num_words;
} BitSet;

typedef struct {
    BitSet *db;
    uint64_t *depth_tidlists;
    size_t num_words;
    uint32_t min_sup;
    size_t total_fci;
    size_t total_footprint;
} DCIContext;

static inline size_t intersect_bitsets(BitSet *out, const BitSet *a, const BitSet *b) {
    size_t count = 0;
    for (size_t i = 0; i < a->num_words; i++) {
        out->words[i] = a->words[i] & b->words[i];
        count += (size_t)__builtin_popcountll(out->words[i]);
    }
    return count;
}

static inline bool is_subset(const BitSet *a, const BitSet *b) {
    for (size_t i = 0; i < a->num_words; i++) {
        if ((a->words[i] & ~b->words[i]) != 0) return false;
    }
    return true;
}

static uint32_t *g_counts = NULL;
static int cmp_freq_asc(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    if (g_counts[ia] < g_counts[ib]) return -1;
    if (g_counts[ia] > g_counts[ib]) return 1;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

static void dci_closed_d(uint32_t *closed_set, size_t closed_len, BitSet *closed_tidlist,
                         uint32_t *pre_set, size_t pre_len,
                         uint32_t *post_set, size_t post_len,
                         DCIContext *ctx, int depth) {
    
    uint32_t *current_pre = NULL;
    if (pre_len + post_len > 0) {
        current_pre = malloc((pre_len + post_len) * sizeof(uint32_t));
        if (pre_len > 0) {
            memcpy(current_pre, pre_set, pre_len * sizeof(uint32_t));
        }
    }
    size_t current_pre_len = pre_len;

    BitSet gen_tidlist;
    gen_tidlist.num_words = ctx->num_words;
    gen_tidlist.words = &ctx->depth_tidlists[depth * ctx->num_words];

    for (size_t i_idx = 0; i_idx < post_len; i_idx++) {
        uint32_t i = post_set[i_idx];
        
        size_t support = intersect_bitsets(&gen_tidlist, closed_tidlist, &ctx->db[i]);
        if (support < ctx->min_sup) continue;
        
        bool is_dup = false;
        for (size_t p = 0; p < current_pre_len; p++) {
            uint32_t j = current_pre[p];
            if (is_subset(&gen_tidlist, &ctx->db[j])) {
                is_dup = true;
                break;
            }
        }
        
        if (is_dup) continue;
        
        current_pre[current_pre_len++] = i;
        
        uint32_t *new_closed = malloc((closed_len + post_len) * sizeof(uint32_t));
        if (closed_len > 0) {
            memcpy(new_closed, closed_set, closed_len * sizeof(uint32_t));
        }
        size_t new_len = closed_len;
        new_closed[new_len++] = i;
        
        uint32_t *new_post = NULL;
        if (post_len > 0) {
            new_post = malloc(post_len * sizeof(uint32_t));
        }
        size_t new_post_len = 0;
        
        for (size_t j_idx = i_idx + 1; j_idx < post_len; j_idx++) {
            uint32_t j = post_set[j_idx];
            if (is_subset(&gen_tidlist, &ctx->db[j])) {
                new_closed[new_len++] = j;
            } else {
                new_post[new_post_len++] = j;
            }
        }
        
        ctx->total_fci++;
        ctx->total_footprint += new_len;
        
        if (new_post_len > 0) {
            dci_closed_d(new_closed, new_len, &gen_tidlist,
                         current_pre, current_pre_len - 1,
                         new_post, new_post_len, ctx, depth + 1);
        }
        
        free(new_closed);
        if (new_post) free(new_post);
    }
    
    if (current_pre) free(current_pre);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_DCI_CLOSED_Params *dci_params = (DM_DCI_CLOSED_Params *)params;
    double min_sup_param = dci_params ? dci_params->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[DCI_CLOSED] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }

    uint32_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) freq_count++;
    }

    if (freq_count == 0) {
        printf("[DCI_CLOSED] Complete. Total frequent closed itemsets found: 0\n");
        free(counts);
        return DM_SUCCESS;
    }

    size_t num_words = (ds->count + 63) / 64;

    BitSet *db = malloc((ds->max_id + 1) * sizeof(BitSet));
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            db[i].words = calloc(num_words, sizeof(uint64_t));
            db[i].num_words = num_words;
        } else {
            db[i].words = NULL;
            db[i].num_words = 0;
        }
    }

    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            if (counts[item] >= min_sup) {
                db[item].words[i / 64] |= (1ULL << (i % 64));
            }
        }
    }

    uint32_t *F1 = malloc(freq_count * sizeof(uint32_t));
    size_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) F1[idx++] = i;
    }
    
    g_counts = counts;
    qsort(F1, freq_count, sizeof(uint32_t), cmp_freq_asc);

    uint32_t *c_empty = malloc(freq_count * sizeof(uint32_t));
    size_t c_empty_len = 0;
    uint32_t *post_set = malloc(freq_count * sizeof(uint32_t));
    size_t post_len = 0;

    for (size_t i = 0; i < freq_count; i++) {
        if (counts[F1[i]] == ds->count) {
            c_empty[c_empty_len++] = F1[i];
        } else {
            post_set[post_len++] = F1[i];
        }
    }

    DCIContext ctx;
    ctx.db = db;
    ctx.min_sup = min_sup;
    ctx.total_fci = (c_empty_len > 0) ? 1 : 0;
    ctx.total_footprint = c_empty_len;
    ctx.num_words = num_words;
    
    uint64_t *depth_tidlists = malloc(freq_count * num_words * sizeof(uint64_t));
    ctx.depth_tidlists = depth_tidlists;

    BitSet c_empty_tidlist;
    c_empty_tidlist.num_words = num_words;
    c_empty_tidlist.words = malloc(num_words * sizeof(uint64_t));
    for (size_t i = 0; i < num_words; i++) c_empty_tidlist.words[i] = ~0ULL;
    if (ds->count % 64 != 0) {
        c_empty_tidlist.words[num_words - 1] &= (1ULL << (ds->count % 64)) - 1;
    }

    if (post_len > 0) {
        dci_closed_d(c_empty, c_empty_len, &c_empty_tidlist, NULL, 0, post_set, post_len, &ctx, 0);
    } else {
        if (c_empty_len == 0) {
            ctx.total_fci = 0;
        }
    }

    printf("[DCI_CLOSED] Complete. Total frequent closed itemsets found: %zu\n", ctx.total_fci);
    dm_bench_record_results(ctx.total_fci, ctx.total_footprint);

    free(c_empty_tidlist.words);
    free(depth_tidlists);
    free(c_empty);
    free(post_set);
    free(F1);
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (db[i].words) free(db[i].words);
    }
    free(db);
    free(counts);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "dci_closed",
    .name = "DCI_CLOSED Algorithm",
    .description = "Fast and Memory Efficient Mining of Frequent Closed Itemsets using Bitmaps.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
