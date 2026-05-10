#include "algorithms/cori.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * CORI Algorithm for mining rare correlated patterns using the bond measure.
 * Reference: S. Bouasker and S. Ben Yahia, "Key correlation mining by simultaneous 
 * monotone and anti-monotone constraints checking", SAC 2015.
 */

typedef struct {
    uint64_t *bits;
    size_t size;
} Bitset;

typedef struct {
    uint32_t id;
    Bitset bs;
    uint32_t sup;
} CoriNode;

typedef struct {
    double min_bond;
    uint32_t min_sup;
    size_t total_rcp;
    size_t total_footprint;
    size_t bitset_words;
} CoriContext;

static size_t bitset_count(const Bitset *b) {
    size_t count = 0;
    for (size_t i = 0; i < b->size; i++) count += __builtin_popcountll(b->bits[i]);
    return count;
}

static void bitset_and(Bitset *res, const Bitset *a, const Bitset *b) {
    for (size_t i = 0; i < a->size; i++) res->bits[i] = a->bits[i] & b->bits[i];
}

static void bitset_or(Bitset *res, const Bitset *a, const Bitset *b) {
    for (size_t i = 0; i < a->size; i++) res->bits[i] = a->bits[i] | b->bits[i];
}

static void mine(CoriContext *ctx, CoriNode *items, size_t num_items, Bitset *curr_conj, Bitset *curr_disj, size_t depth) {
    Bitset new_conj = { malloc(ctx->bitset_words * sizeof(uint64_t)), ctx->bitset_words };
    Bitset new_disj = { malloc(ctx->bitset_words * sizeof(uint64_t)), ctx->bitset_words };

    for (size_t i = 0; i < num_items; i++) {
        bitset_and(&new_conj, curr_conj, &items[i].bs);
        bitset_or(&new_disj, curr_disj, &items[i].bs);

        uint32_t sup = (uint32_t)bitset_count(&new_conj);
        if (sup == 0) continue;

        uint32_t disj_sup = (uint32_t)bitset_count(&new_disj);
        double bond = (double)sup / disj_sup;

        if (bond >= ctx->min_bond) {
            if (sup < ctx->min_sup) {
                ctx->total_rcp++;
                ctx->total_footprint += depth + 1;
            }
            
            if (i + 1 < num_items) {
                mine(ctx, &items[i+1], num_items - (i + 1), &new_conj, &new_disj, depth + 1);
            }
        }
        // else: bond < min_bond, prune sub-tree (anti-monotone property of bond)
    }

    free(new_conj.bits);
    free(new_disj.bits);
}

static int compare_nodes(const void *a, const void *b) {
    return (int)((CoriNode *)a)->sup - (int)((CoriNode *)b)->sup;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_CORI_Params *p = (DM_CORI_Params *)params;
    double min_sup_param = p ? p->min_support : 0.4;
    double min_bond = p ? p->min_bond : 0.2;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[CORI] Starting on %zu transactions. Min Sup: %u, Min Bond: %.2f\n", ds->count, min_sup, min_bond);

    CoriContext ctx = {
        .min_bond = min_bond,
        .min_sup = min_sup,
        .total_rcp = 0,
        .total_footprint = 0,
        .bitset_words = (ds->count + 63) / 64
    };

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    CoriNode *nodes = calloc(ds->max_id + 1, sizeof(CoriNode));
    size_t num_nodes = 0;

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        nodes[i].id = i;
        nodes[i].bs.bits = calloc(ctx.bitset_words, sizeof(uint64_t));
        nodes[i].bs.size = ctx.bitset_words;
    }

    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            nodes[item].bs.bits[i / 64] |= (1ULL << (i % 64));
        }
    }

    CoriNode *frequent_items = malloc((ds->max_id + 1) * sizeof(CoriNode));
    size_t num_frequent = 0;

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        nodes[i].sup = (uint32_t)bitset_count(&nodes[i].bs);
        if (nodes[i].sup > 0) {
            // Rarity check for 1-itemsets (depth 0)
            if (nodes[i].sup < min_sup) {
                // Rarity monotone constraint fulfilled. 
                // Correlation for 1-itemset is always 1.0 (sup/sup)
                ctx.total_rcp++;
                ctx.total_footprint++;
            }
            frequent_items[num_frequent++] = nodes[i];
        } else {
            free(nodes[i].bs.bits);
        }
    }

    // Sort items by support ascending
    qsort(frequent_items, num_frequent, sizeof(CoriNode), compare_nodes);

    Bitset curr_conj = { malloc(ctx.bitset_words * sizeof(uint64_t)), ctx.bitset_words };
    Bitset curr_disj = { malloc(ctx.bitset_words * sizeof(uint64_t)), ctx.bitset_words };

    for (size_t i = 0; i < num_frequent; i++) {
        memcpy(curr_conj.bits, frequent_items[i].bs.bits, ctx.bitset_words * sizeof(uint64_t));
        memcpy(curr_disj.bits, frequent_items[i].bs.bits, ctx.bitset_words * sizeof(uint64_t));
        
        if (i + 1 < num_frequent) {
            mine(&ctx, &frequent_items[i+1], num_frequent - (i + 1), &curr_conj, &curr_disj, 1);
        }
    }

    printf("[CORI] Complete. Total Rare Correlated Patterns (RCP): %zu\n", ctx.total_rcp);
    dm_bench_record_results(ctx.total_rcp, ctx.total_footprint);

    // Cleanup
    free(curr_conj.bits);
    free(curr_disj.bits);
    for (size_t i = 0; i < num_frequent; i++) free(frequent_items[i].bs.bits);
    free(frequent_items);
    free(nodes);

    return DM_SUCCESS;
}

static DM_Algorithm algo_cori = {
    .id = "cori",
    .name = "CORI Algorithm",
    .description = "Rare correlated pattern mining using simultaneous monotone and anti-monotone constraints.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo_cori)
