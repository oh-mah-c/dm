#include "algorithms/cfi_stream.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * CFI-Stream: Mining Closed Frequent Itemsets in Data Streams.
 * Implementation based on Jiang & Gruenwald (2006).
 * ACM Citation: 1150402.1150473
 * 
 * Note: This implementation adaptively handles batch datasets 
 * using the bit-sequence (vertical bitset) methodology of CFI-Stream.
 */

/* --- Data Structures --- */

typedef struct {
    uint32_t item;
    uint64_t *bits;
    uint32_t support;
} ItemBitset;

typedef struct {
    uint32_t min_sup;
    uint32_t num_trans;
    uint32_t num_words;
    size_t total_cfi;
    size_t total_footprint;
    ItemBitset *all_freq_items;
    uint32_t total_freq_items;
} Context;

/* --- Bitset Operations --- */

static uint32_t count_bits(uint64_t *bits, uint32_t num_words) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < num_words; i++) {
        #ifdef _MSC_VER
        count += (uint32_t)__popcnt64(bits[i]);
        #else
        count += (uint32_t)__builtin_popcountll(bits[i]);
        #endif
    }
    return count;
}

static bool is_equal(uint64_t *a, uint64_t *b, uint32_t num_words) {
    for (uint32_t i = 0; i < num_words; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

static bool is_subset(uint64_t *a, uint64_t *b, uint32_t num_words) {
    // Check if a is a subset of b (a & b == a)
    for (uint32_t i = 0; i < num_words; i++) {
        if ((a[i] & b[i]) != a[i]) return false;
    }
    return true;
}

/* --- CFI-Tree (Lexicographical Tree) --- */

typedef struct CFINode {
    uint32_t *items;
    uint32_t count;
    uint64_t *bits;
    uint32_t support;
    struct CFINode *children;
    struct CFINode *sibling;
} CFINode;

static CFINode* create_node(uint32_t *items, uint32_t count, uint64_t *bits, uint32_t support, uint32_t num_words) {
    CFINode *node = malloc(sizeof(CFINode));
    node->items = malloc(count * sizeof(uint32_t));
    memcpy(node->items, items, count * sizeof(uint32_t));
    node->count = count;
    node->bits = malloc(num_words * sizeof(uint64_t));
    memcpy(node->bits, bits, num_words * sizeof(uint64_t));
    node->support = support;
    node->children = NULL;
    node->sibling = NULL;
    return node;
}

/* --- Mining Logic --- */

static void mine_recursive(uint32_t *prefix, uint32_t prefix_len, uint64_t *prefix_bits, uint32_t prefix_supp, 
                           ItemBitset *candidates, uint32_t cand_count, Context *ctx) {
    
    for (uint32_t i = 0; i < cand_count; i++) {
        // Intersect prefix_bits with candidates[i].bits
        uint64_t *new_bits = malloc(ctx->num_words * sizeof(uint64_t));
        for (uint32_t w = 0; w < ctx->num_words; w++) {
            new_bits[w] = prefix_bits[w] & candidates[i].bits[w];
        }
        
        uint32_t new_supp = count_bits(new_bits, ctx->num_words);
        
        if (new_supp >= ctx->min_sup) {
            uint32_t *new_prefix = malloc((prefix_len + 1) * sizeof(uint32_t));
            memcpy(new_prefix, prefix, prefix_len * sizeof(uint32_t));
            new_prefix[prefix_len] = candidates[i].item;

            // Closure check:
            // X is closed if for all j NOT in S, B(X) is not a subset of B(j).
            bool is_closed = true;
            for (uint32_t j = 0; j < ctx->total_freq_items; j++) {
                // Skip if j is already in the new prefix
                bool in_prefix = false;
                for (uint32_t k = 0; k <= prefix_len; k++) {
                    if (new_prefix[k] == ctx->all_freq_items[j].item) {
                        in_prefix = true;
                        break;
                    }
                }
                if (in_prefix) continue;

                if (is_subset(new_bits, ctx->all_freq_items[j].bits, ctx->num_words)) {
                    is_closed = false;
                    break;
                }
            }

            if (is_closed) {
                ctx->total_cfi++;
                ctx->total_footprint += (prefix_len + 1);
            }
            
            // Recurse even if not closed, because subsets of a non-closed set can be closed?
            // Actually, if B(X) subset of B(j), then X is not closed. 
            // But X U {k} might be closed.
            if (i + 1 < cand_count) {
                mine_recursive(new_prefix, prefix_len + 1, new_bits, new_supp, candidates + i + 1, cand_count - i - 1, ctx);
            }
            
            free(new_prefix);
        }
        free(new_bits);
    }
}

static int cmp_item_bitset(const void *a, const void *b) {
    const ItemBitset *ia = (const ItemBitset *)a;
    const ItemBitset *ib = (const ItemBitset *)b;
    if (ia->support > ib->support) return -1;
    if (ia->support < ib->support) return 1;
    return (ia->item < ib->item) ? -1 : 1;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_CFI_STREAM_Params *cs_params = (DM_CFI_STREAM_Params *)params;
    double min_sup_param = cs_params ? cs_params->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[CFI-Stream] Starting. Min Support: %u\n", min_sup);

    uint32_t num_trans = (uint32_t)ds->count;
    uint32_t num_words = (num_trans + 63) / 64;
    
    ItemBitset *items = calloc(ds->max_id + 1, sizeof(ItemBitset));
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        items[i].item = i;
        items[i].bits = calloc(num_words, sizeof(uint64_t));
    }

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (uint32_t i = 0; i < num_trans; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            items[item].bits[i / 64] |= (1ULL << (i % 64));
        }
    }

    ItemBitset *freq_items = malloc((ds->max_id + 1) * sizeof(ItemBitset));
    uint32_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        items[i].support = count_bits(items[i].bits, num_words);
        if (items[i].support >= min_sup) {
            freq_items[freq_count++] = items[i];
        } else {
            free(items[i].bits);
        }
    }
    free(items);

    qsort(freq_items, freq_count, sizeof(ItemBitset), cmp_item_bitset);

    Context ctx = { 
        .min_sup = min_sup, 
        .num_trans = num_trans, 
        .num_words = num_words, 
        .total_cfi = 0, 
        .total_footprint = 0,
        .all_freq_items = freq_items,
        .total_freq_items = freq_count
    };

    uint64_t *all_ones = malloc(num_words * sizeof(uint64_t));
    memset(all_ones, 0xFF, num_words * sizeof(uint64_t));

    mine_recursive(NULL, 0, all_ones, num_trans, freq_items, freq_count, &ctx);

    printf("[CFI-Stream] Complete. CFIs found: %zu\n", ctx.total_cfi);
    dm_bench_record_results(ctx.total_cfi, ctx.total_footprint);

    for (uint32_t i = 0; i < freq_count; i++) free(freq_items[i].bits);
    free(freq_items);
    free(all_ones);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "cfi_stream",
    .name = "CFI-Stream Algorithm",
    .description = "Mining Closed Frequent Itemsets using Bit-sequences (Jiang & Gruenwald).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
