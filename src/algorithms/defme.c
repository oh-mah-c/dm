#include "algorithms/defme.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * dEFME Algorithm for depth-first minimal pattern mining (free itemsets).
 * Reference: Arnaud Soulet and François Rioult, "Efficiently Depth-First Minimal Pattern Mining", PAKDD 2014.
 */

// Bitset utilities
static inline bool bs_is_empty(const uint64_t *bs, size_t num_words) {
    for (size_t i = 0; i < num_words; i++) {
        if (bs[i] != 0) return false;
    }
    return true;
}

typedef struct {
    uint32_t item_id;
    uint64_t *bits;
} DEFME_Item;

typedef struct {
    uint32_t *items;
    uint64_t *cov_bits;
    uint64_t **crit_bits; // crit_bits[i] corresponds to items[i]
    size_t len;
    uint32_t support;
} DEFME_Node;

typedef struct {
    uint32_t min_sup;
    size_t num_rows;
    size_t num_words;
    DEFME_Item *vdb;
    size_t vdb_size;
    size_t total_minimal;
    size_t total_footprint;
} DEFME_Context;

static void mine_defme(DEFME_Node *current, uint32_t *tail, size_t tail_len, DEFME_Context *ctx) {
    // Step 1: Minimality Check
    // X is minimal if ∀e ∈ X, cov(X,e) != ∅
    for (size_t i = 0; i < current->len; i++) {
        if (bs_is_empty(current->crit_bits[i], ctx->num_words)) return;
    }

    // Step 2: Output minimal pattern
    if (current->len > 0) {
        ctx->total_minimal++;
        ctx->total_footprint += current->len;
    }

    // Step 3-14: DFS Enumeration
    for (size_t i = 0; i < tail_len; i++) {
        uint32_t item_idx = tail[i];
        const uint64_t *item_bits = ctx->vdb[item_idx].bits;

        // Step 7: cov(Y) = cov(X) ∩ cov(e)
        uint32_t new_sup = 0;
        uint64_t *new_cov = malloc(ctx->num_words * sizeof(uint64_t));
        for (size_t w = 0; w < ctx->num_words; w++) {
            new_cov[w] = current->cov_bits[w] & item_bits[w];
            new_sup += __builtin_popcountll(new_cov[w]);
        }

        // Step 4: if Xe ∈ F (Frequent)
        if (new_sup >= ctx->min_sup) {
            DEFME_Node next;
            next.len = current->len + 1;
            next.items = malloc(next.len * sizeof(uint32_t));
            memcpy(next.items, current->items, current->len * sizeof(uint32_t));
            next.items[current->len] = ctx->vdb[item_idx].item_id;
            next.support = new_sup;
            next.cov_bits = new_cov;
            next.crit_bits = malloc(next.len * sizeof(uint64_t *));
            
            // Step 10: cov(Y, e') = cov(X, e') ∩ cov(e) for all e' in X
            for (size_t j = 0; j < current->len; j++) {
                next.crit_bits[j] = malloc(ctx->num_words * sizeof(uint64_t));
                for (size_t w = 0; w < ctx->num_words; w++) {
                    next.crit_bits[j][w] = current->crit_bits[j][w] & item_bits[w];
                }
            }
            
            // Step 8: cov(Y, e) = cov(X) \ cov(e) (i.e. cov(X) ∩ ~cov(e))
            next.crit_bits[current->len] = malloc(ctx->num_words * sizeof(uint64_t));
            for (size_t w = 0; w < ctx->num_words; w++) {
                next.crit_bits[current->len][w] = current->cov_bits[w] & ~item_bits[w];
            }

            // Step 12: Recursive call
            mine_defme(&next, &tail[i + 1], tail_len - (i + 1), ctx);

            // Cleanup
            for (size_t j = 0; j < next.len; j++) free(next.crit_bits[j]);
            free(next.crit_bits);
            free(next.items);
            free(next.cov_bits);
        } else {
            free(new_cov);
        }
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_DEFME_Params *p = (DM_DEFME_Params *)params;
    double min_sup_param = p ? p->min_support : 0.05;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[dEFME] Starting on %zu rows. Min Support: %u\n", ds->count, min_sup);

    // 1. Scan for frequent items
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }

    size_t num_words = (ds->count + 63) / 64;
    size_t num_frequent = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) num_frequent++;
    }

    if (num_frequent == 0) {
        printf("[dEFME] No frequent items found.\n");
        free(counts);
        return DM_SUCCESS;
    }

    // 2. Build Vertical Database (VDB)
    DEFME_Context ctx;
    ctx.min_sup = min_sup;
    ctx.num_rows = ds->count;
    ctx.num_words = num_words;
    ctx.vdb_size = num_frequent;
    ctx.vdb = malloc(num_frequent * sizeof(DEFME_Item));
    ctx.total_minimal = 0;
    ctx.total_footprint = 0;

    size_t f_idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            ctx.vdb[f_idx].item_id = i;
            ctx.vdb[f_idx].bits = calloc(num_words, sizeof(uint64_t));
            f_idx++;
        }
    }

    // Map and fill bits
    uint32_t *item_map = malloc((ds->max_id + 1) * sizeof(uint32_t));
    memset(item_map, 0xFF, (ds->max_id + 1) * sizeof(uint32_t));
    for (size_t i = 0; i < num_frequent; i++) item_map[ctx.vdb[i].item_id] = (uint32_t)i;

    for (size_t r = 0; r < ds->count; r++) {
        for (size_t j = 0; j < data[r].count; j++) {
            uint32_t m = item_map[data[r].items[j]];
            if (m != 0xFFFFFFFF) {
                ctx.vdb[m].bits[r >> 6] |= (1ULL << (r & 0x3F));
            }
        }
    }
    free(item_map);

    // 3. Root Node Initialization
    DEFME_Node root;
    root.len = 0;
    root.items = NULL;
    root.support = (uint32_t)ds->count;
    root.cov_bits = malloc(num_words * sizeof(uint64_t));
    memset(root.cov_bits, 0xFF, num_words * sizeof(uint64_t));
    // Mask extra bits in the last word
    if (ds->count % 64 != 0) {
        root.cov_bits[num_words - 1] &= (1ULL << (ds->count % 64)) - 1;
    }
    root.crit_bits = NULL;

    uint32_t *tail = malloc(num_frequent * sizeof(uint32_t));
    for (size_t i = 0; i < num_frequent; i++) tail[i] = (uint32_t)i;

    // 4. Start Mining
    mine_defme(&root, tail, num_frequent, &ctx);

    printf("[dEFME] Complete. Total minimal patterns (generators) found: %zu\n", ctx.total_minimal);
    dm_bench_record_results(ctx.total_minimal, ctx.total_footprint);

    // 5. Cleanup
    free(tail);
    free(root.cov_bits);
    for (size_t i = 0; i < num_frequent; i++) free(ctx.vdb[i].bits);
    free(ctx.vdb);
    free(counts);

    return DM_SUCCESS;
}

static DM_Algorithm algo_defme = {
    .id = "defme",
    .name = "dEFME Algorithm",
    .description = "Depth-First Minimal Pattern Mining for enumerating free itemsets (generators).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo_defme)
