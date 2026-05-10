#include "algorithms/sam.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * SaM (Split and Merge) Algorithm.
 * Reference: Christian Borgelt and Xiaomeng Wang, 
 * "SaM: A Split and Merge Algorithm for Fuzzy Frequent Item Set Mining", 2009.
 */

// Global context for qsort (non-thread-safe but okay for this framework)
static uint32_t *g_counts = NULL;

typedef struct {
    uint32_t *items;
    uint32_t length;
    size_t weight;
} SAM_Trans;

typedef struct {
    uint32_t *counts;
    size_t total_frequent;
    size_t total_footprint;
    uint32_t min_sup;
} SAM_Context;

static int cmp_items_asc(const void *a, const void *b) {
    uint32_t i1 = *(uint32_t*)a;
    uint32_t i2 = *(uint32_t*)b;
    if (g_counts[i1] != g_counts[i2]) 
        return (int)g_counts[i1] - (int)g_counts[i2];
    return (i1 < i2) ? -1 : (i1 > i2 ? 1 : 0);
}

static int cmp_trans_lex(const void *a, const void *b) {
    const SAM_Trans *t1 = (const SAM_Trans*)a;
    const SAM_Trans *t2 = (const SAM_Trans*)b;
    uint32_t len = (t1->length < t2->length) ? t1->length : t2->length;
    for (uint32_t i = 0; i < len; i++) {
        int c = cmp_items_asc(&t1->items[i], &t2->items[i]);
        if (c != 0) return c;
    }
    return (int)t1->length - (int)t2->length;
}

static SAM_Trans* sam_merge_impl(SAM_Trans *a, size_t a_len, SAM_Trans *b, size_t b_len, size_t *out_len) {
    if (a_len == 0 && b_len == 0) { *out_len = 0; return NULL; }
    SAM_Trans *res = malloc((a_len + b_len) * sizeof(SAM_Trans));
    size_t i = 0, j = 0, k = 0;
    while (i < a_len && j < b_len) {
        int c = cmp_trans_lex(&a[i], &b[j]);
        if (c < 0) { res[k++] = a[i++]; }
        else if (c > 0) { res[k++] = b[j++]; }
        else {
            res[k] = a[i++];
            res[k++].weight += b[j++].weight;
        }
    }
    while (i < a_len) res[k++] = a[i++];
    while (j < b_len) res[k++] = b[j++];
    *out_len = k;
    return res;
}

static void sam_recursive(SAM_Trans *a, size_t a_len, uint32_t *prefix, uint32_t p_len, SAM_Context *ctx) {
    if (a_len == 0) return;

    SAM_Trans *curr_a = a;
    size_t curr_a_len = a_len;
    bool curr_a_owned = false; // Initial 'a' is owned by caller

    while (curr_a_len > 0) {
        uint32_t split_item = curr_a[0].items[0];
        size_t s = 0;
        size_t split_count = 0;
        
        // Find all transactions starting with the same leading item
        while (split_count < curr_a_len && curr_a[split_count].items[0] == split_item) {
            s += curr_a[split_count].weight;
            split_count++;
        }

        // Conditional database 'b' (transactions starting with split_item, with item removed)
        SAM_Trans *b = malloc(split_count * sizeof(SAM_Trans));
        size_t b_len = 0;
        for (size_t i = 0; i < split_count; i++) {
            if (curr_a[i].length > 1) {
                b[b_len].items = curr_a[i].items + 1;
                b[b_len].length = curr_a[i].length - 1;
                b[b_len].weight = curr_a[i].weight;
                b_len++;
            }
        }

        // Transactions that didn't start with split_item
        SAM_Trans *a_rem = curr_a + split_count;
        size_t a_rem_len = curr_a_len - split_count;

        // Merge a_rem and b into next_a for the next iteration of this level
        size_t next_a_len;
        SAM_Trans *next_a = sam_merge_impl(a_rem, a_rem_len, b, b_len, &next_a_len);

        if (s >= ctx->min_sup) {
            ctx->total_frequent++;
            ctx->total_footprint += (p_len + 1);
            
            if (b_len > 0) {
                prefix[p_len] = split_item;
                sam_recursive(b, b_len, prefix, p_len + 1, ctx);
            }
        }

        free(b);
        if (curr_a_owned) free(curr_a);
        
        curr_a = next_a;
        curr_a_len = next_a_len;
        curr_a_owned = true;
    }
    if (curr_a_owned) free(curr_a);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_SAM_Params *p = (DM_SAM_Params*)params;
    double ms_val = p ? p->min_support : 0.05;
    uint32_t min_sup = (ms_val < 1.0) ? (uint32_t)ceil(ms_val * ds->count) : (uint32_t)ms_val;
    if (min_sup == 0) min_sup = 1;

    printf("[SaM] Starting. Min Support: %u\n", min_sup);
    fflush(stdout);

    // 1. Preprocessing: Count frequencies
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple*)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }

    // Set global counts for sorting
    g_counts = counts;

    // 2. Filter, Sort items within transactions, and prepare initial database
    SAM_Trans *initial_db = malloc(ds->count * sizeof(SAM_Trans));
    size_t db_size = 0;
    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *t_items = malloc(data[i].count * sizeof(uint32_t));
        uint32_t t_len = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (counts[data[i].items[j]] >= min_sup) {
                t_items[t_len++] = data[i].items[j];
            }
        }
        if (t_len > 0) {
            qsort(t_items, t_len, sizeof(uint32_t), cmp_items_asc);
            initial_db[db_size].items = t_items;
            initial_db[db_size].length = t_len;
            initial_db[db_size].weight = 1;
            db_size++;
        } else {
            free(t_items);
        }
    }

    // Sort transactions lexicographically
    qsort(initial_db, db_size, sizeof(SAM_Trans), cmp_trans_lex);

    // Combine identical transactions
    size_t unique_count = 0;
    if (db_size > 0) {
        unique_count = 1;
        for (size_t i = 1; i < db_size; i++) {
            if (cmp_trans_lex(&initial_db[i], &initial_db[unique_count-1]) == 0) {
                initial_db[unique_count-1].weight++;
                free(initial_db[i].items);
            } else {
                initial_db[unique_count++] = initial_db[i];
            }
        }
    }

    SAM_Context ctx = { .counts = counts, .min_sup = min_sup, .total_frequent = 0, .total_footprint = 0 };
    uint32_t *prefix = malloc((ds->max_id + 1) * sizeof(uint32_t));
    
    sam_recursive(initial_db, unique_count, prefix, 0, &ctx);

    printf("[SaM] Complete. FIs found: %zu\n", ctx.total_frequent);
    dm_bench_record_results(ctx.total_frequent, ctx.total_footprint);

    // Cleanup
    for (size_t i = 0; i < unique_count; i++) free(initial_db[i].items);
    free(initial_db);
    free(prefix);
    free(counts);
    g_counts = NULL;

    return DM_SUCCESS;
}

static DM_Algorithm algo_sam = {
    .id = "sam", .name = "Split and Merge Algorithm",
    .description = "A purely horizontal frequent itemset mining algorithm based on recursive split and merge operations.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL), .run = run
};
DM_REGISTER_ALGORITHM(algo_sam)
