#include "algorithms/dbv_miner.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * DBV-Miner Algorithm for fast mining frequent closed itemsets.
 * Reference: Bay Vo, Tzung-Pei Hong, Bac Le, "DBV-Miner: A Dynamic Bit-Vector approach for fast mining frequent closed itemsets", 
 * Expert Systems with Applications, 2012.
 */

// Lookup table for bit counting
static const uint8_t bit_counts[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

typedef struct {
    uint32_t pos;
    uint32_t len;
    uint8_t *bits;
} DBV;

typedef struct {
    uint32_t *items;
    size_t items_len;
    DBV dbv;
    uint32_t support;
    bool deleted;
} DBVNode;

typedef struct FCPNode {
    uint32_t *features;
    size_t count;
    struct FCPNode *next;
} FCPNode;

#define HASH_SIZE 65536

typedef struct {
    FCPNode **buckets;
    size_t total_count;
} FCP_Registry;

static uint64_t hash_features(const uint32_t *features, size_t count) {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < count; i++) {
        h ^= (uint64_t)features[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static bool fcp_check_and_add(FCP_Registry *reg, const uint32_t *features, size_t count) {
    if (count == 0) return true;
    uint32_t *sorted = malloc(count * sizeof(uint32_t));
    memcpy(sorted, features, count * sizeof(uint32_t));
    // Sort items for consistent hashing/comparison
    for (size_t i = 0; i < count; i++) {
        for (size_t j = i + 1; j < count; j++) {
            if (sorted[i] > sorted[j]) {
                uint32_t tmp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = tmp;
            }
        }
    }

    uint64_t h = hash_features(sorted, count);
    size_t idx = h % HASH_SIZE;
    FCPNode *curr = reg->buckets[idx];
    while (curr) {
        if (curr->count == count) {
            if (memcmp(curr->features, sorted, count * sizeof(uint32_t)) == 0) {
                free(sorted);
                return true;
            }
        }
        curr = curr->next;
    }
    FCPNode *node = malloc(sizeof(FCPNode));
    node->count = count;
    node->features = sorted;
    node->next = reg->buckets[idx];
    reg->buckets[idx] = node;
    reg->total_count++;
    return false;
}

typedef struct {
    uint32_t min_sup;
    FCP_Registry *reg;
    size_t total_closed;
    size_t total_footprint;
} DBVContext;

static void dbv_free(DBV *d) {
    if (d->bits) free(d->bits);
    d->bits = NULL;
    d->len = 0;
}

static DBV dbv_copy(const DBV *d) {
    DBV res = {d->pos, d->len, NULL};
    if (d->len > 0) {
        res.bits = malloc(d->len);
        memcpy(res.bits, d->bits, d->len);
    }
    return res;
}

static DBV dbv_intersect(const DBV *a, const DBV *b, uint32_t *support_out) {
    DBV res = {0, 0, NULL};
    uint32_t start = (a->pos > b->pos) ? a->pos : b->pos;
    uint32_t end_a = a->pos + a->len;
    uint32_t end_b = b->pos + b->len;
    uint32_t end = (end_a < end_b) ? end_a : end_b;

    if (start >= end) {
        *support_out = 0;
        return res;
    }

    uint32_t temp_len = end - start;
    uint8_t *temp_bits = malloc(temp_len);
    uint32_t sup = 0;
    uint32_t first_nz = 0xFFFFFFFF;
    uint32_t last_nz = 0;

    for (uint32_t i = 0; i < temp_len; i++) {
        uint8_t ba = a->bits[start - a->pos + i];
        uint8_t bb = b->bits[start - b->pos + i];
        uint8_t res_b = ba & bb;
        temp_bits[i] = res_b;
        if (res_b > 0) {
            sup += bit_counts[res_b];
            if (first_nz == 0xFFFFFFFF) first_nz = i;
            last_nz = i;
        }
    }

    if (sup == 0) {
        free(temp_bits);
        *support_out = 0;
        return res;
    }

    res.pos = start + first_nz;
    res.len = last_nz - first_nz + 1;
    res.bits = malloc(res.len);
    memcpy(res.bits, temp_bits + first_nz, res.len);
    free(temp_bits);
    *support_out = sup;
    return res;
}

static bool dbv_is_subset(const DBV *a, const DBV *b) {
    if (a->pos < b->pos || a->pos + a->len > b->pos + b->len) return false;
    for (uint32_t i = 0; i < a->len; i++) {
        uint8_t ba = a->bits[i];
        uint8_t bb = b->bits[a->pos - b->pos + i];
        if ((ba & bb) != ba) return false;
    }
    return true;
}

static bool dbv_is_equal(const DBV *a, const DBV *b) {
    if (a->pos != b->pos || a->len != b->len) return false;
    return memcmp(a->bits, b->bits, a->len) == 0;
}

static void dbv_extend(DBVNode *L, size_t L_count, DBVContext *ctx) {
    for (size_t i = 0; i < L_count; i++) {
        if (L[i].deleted) continue;
        
        uint32_t *pe_items = malloc(L[i].items_len * sizeof(uint32_t));
        memcpy(pe_items, L[i].items, L[i].items_len * sizeof(uint32_t));
        size_t pe_len = L[i].items_len;

        DBVNode *new_L = malloc(L_count * sizeof(DBVNode));
        size_t new_L_count = 0;
        
        for (size_t j = i + 1; j < L_count; j++) {
            if (L[j].deleted) continue;
            
            uint32_t sup_z;
            DBV dbv_z = dbv_intersect(&L[i].dbv, &L[j].dbv, &sup_z);
            
            if (sup_z >= ctx->min_sup) {
                // Lemma 4.1: Check if Z is subsumed by any node Xk (k < i)
                bool subsumed_by_prev = false;
                for (size_t k = 0; k < i; k++) {
                    if (dbv_is_subset(&dbv_z, &L[k].dbv)) {
                        subsumed_by_prev = true;
                        break;
                    }
                }
                
                if (subsumed_by_prev) {
                    dbv_free(&dbv_z);
                    continue;
                }

                if (sup_z == L[i].support) {
                    // Xi is subsumed by Xj -> Xj is a perfect extension
                    pe_items = realloc(pe_items, (pe_len + L[j].items_len) * sizeof(uint32_t));
                    memcpy(pe_items + pe_len, L[j].items, L[j].items_len * sizeof(uint32_t));
                    pe_len += L[j].items_len;
                    
                    if (sup_z == L[j].support) L[j].deleted = true;
                    dbv_free(&dbv_z);
                } else {
                    // Potential child
                    DBVNode node;
                    node.items_len = L[i].items_len + L[j].items_len;
                    node.items = malloc(node.items_len * sizeof(uint32_t));
                    memcpy(node.items, L[i].items, L[i].items_len * sizeof(uint32_t));
                    memcpy(node.items + L[i].items_len, L[j].items, L[j].items_len * sizeof(uint32_t));
                    node.dbv = dbv_z;
                    node.support = sup_z;
                    node.deleted = false;
                    new_L[new_L_count++] = node;
                }
            } else {
                dbv_free(&dbv_z);
            }
        }
        
        // Finalize L[i] with perfect extensions
        if (!fcp_check_and_add(ctx->reg, pe_items, pe_len)) {
            ctx->total_closed++;
            ctx->total_footprint += pe_len;
        }
        free(pe_items);

        if (new_L_count > 0) {
            dbv_extend(new_L, new_L_count, ctx);
        }
        
        // Cleanup new_L
        for (size_t k = 0; k < new_L_count; k++) {
            free(new_L[k].items);
            dbv_free(&new_L[k].dbv);
        }
        free(new_L);
    }
}

static int cmp_nodes_sup(const void *a, const void *b) {
    uint32_t sa = ((const DBVNode *)a)->support;
    uint32_t sb = ((const DBVNode *)b)->support;
    if (sa < sb) return -1;
    if (sa > sb) return 1;
    return 0;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_DBV_MINER_Params *p = (DM_DBV_MINER_Params *)params;
    double min_sup_param = p ? p->min_support : 0.05;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[DBV-Miner] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    // 1. Initial Scan and Filtering
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }

    size_t num_frequent = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) num_frequent++;
    }

    if (num_frequent == 0) {
        printf("[DBV-Miner] No frequent items found.\n");
        free(counts);
        return DM_SUCCESS;
    }

    // 2. Build Level 1 Nodes with DBV
    DBVNode *L1 = malloc(num_frequent * sizeof(DBVNode));
    size_t f_idx = 0;
    uint32_t num_bytes = (ds->count + 7) / 8;

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            uint8_t *full_bits = calloc(num_bytes, 1);
            for (size_t r = 0; r < ds->count; r++) {
                for (size_t j = 0; j < data[r].count; j++) {
                    if (data[r].items[j] == i) {
                        full_bits[r >> 3] |= (1 << (r & 7));
                        break;
                    }
                }
            }
            
            // Find first and last non-zero bytes
            uint32_t first = 0xFFFFFFFF, last = 0;
            for (uint32_t b = 0; b < num_bytes; b++) {
                if (full_bits[b] > 0) {
                    if (first == 0xFFFFFFFF) first = b;
                    last = b;
                }
            }
            
            L1[f_idx].items_len = 1;
            L1[f_idx].items = malloc(sizeof(uint32_t));
            L1[f_idx].items[0] = i;
            L1[f_idx].support = counts[i];
            L1[f_idx].deleted = false;
            L1[f_idx].dbv.pos = first;
            L1[f_idx].dbv.len = last - first + 1;
            L1[f_idx].dbv.bits = malloc(L1[f_idx].dbv.len);
            memcpy(L1[f_idx].dbv.bits, full_bits + first, L1[f_idx].dbv.len);
            
            free(full_bits);
            f_idx++;
        }
    }

    // Sort Level 1 by support increasing
    qsort(L1, num_frequent, sizeof(DBVNode), cmp_nodes_sup);

    // Initial Subsumption Check (Optional but recommended in paper)
    for (size_t i = 0; i < num_frequent; i++) {
        if (L1[i].deleted) continue;
        for (size_t j = i + 1; j < num_frequent; j++) {
            if (L1[j].deleted) continue;
            if (L1[i].support == L1[j].support && dbv_is_equal(&L1[i].dbv, &L1[j].dbv)) {
                // Merge j into i
                L1[i].items = realloc(L1[i].items, (L1[i].items_len + 1) * sizeof(uint32_t));
                L1[i].items[L1[i].items_len++] = L1[j].items[0];
                L1[j].deleted = true;
            }
        }
    }

    FCP_Registry reg;
    reg.buckets = calloc(HASH_SIZE, sizeof(FCPNode *));
    reg.total_count = 0;

    DBVContext ctx = {min_sup, &reg, 0, 0};
    dbv_extend(L1, num_frequent, &ctx);

    printf("[DBV-Miner] Complete. Total frequent closed itemsets found: %zu\n", ctx.total_closed);
    dm_bench_record_results(ctx.total_closed, ctx.total_footprint);

    // Cleanup
    for (size_t i = 0; i < num_frequent; i++) {
        free(L1[i].items);
        dbv_free(&L1[i].dbv);
    }
    free(L1);
    free(counts);

    for (size_t i = 0; i < HASH_SIZE; i++) {
        FCPNode *curr = reg.buckets[i];
        while (curr) {
            FCPNode *next = curr->next;
            free(curr->features);
            free(curr);
            curr = next;
        }
    }
    free(reg.buckets);

    return DM_SUCCESS;
}

static DM_Algorithm algo_dbv_miner = {
    .id = "dbv_miner",
    .name = "DBV-Miner Algorithm",
    .description = "A Dynamic Bit-Vector approach for fast mining frequent closed itemsets.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo_dbv_miner)
