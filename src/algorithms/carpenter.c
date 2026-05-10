#include "algorithms/carpenter.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * CARPENTER Algorithm for finding closed patterns in long datasets.
 * Reference: Feng Pan, Gao Cong, Anthony K. H. Tung, Jiong Yang, Mohammed J. Zaki, 
 * "CARPENTER: Finding Closed Patterns in Long Biological Datasets", KDD 2003.
 */

// Bitset utilities
#define BS_WORD(bit) ((bit) >> 6)
#define BS_MASK(bit) (1ULL << ((bit) & 0x3F))

static inline void bs_set(uint64_t *bs, uint32_t bit) {
    bs[BS_WORD(bit)] |= BS_MASK(bit);
}

static inline bool bs_test(const uint64_t *bs, uint32_t bit) {
    return (bs[BS_WORD(bit)] & BS_MASK(bit)) != 0;
}

typedef struct {
    uint32_t feature_id;
    uint64_t *row_bits;
} CARPENTER_Tuple;

typedef struct {
    CARPENTER_Tuple *tuples;
    size_t num_tuples;
    size_t num_rows;
    size_t num_words;
} CARPENTER_TT;

// Hash Table for FCP (Frequent Closed Patterns)
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
    uint64_t h = hash_features(features, count);
    size_t idx = h % HASH_SIZE;
    FCPNode *curr = reg->buckets[idx];
    while (curr) {
        if (curr->count == count) {
            if (memcmp(curr->features, features, count * sizeof(uint32_t)) == 0) return true;
        }
        curr = curr->next;
    }
    FCPNode *node = malloc(sizeof(FCPNode));
    node->count = count;
    node->features = malloc(count * sizeof(uint32_t));
    memcpy(node->features, features, count * sizeof(uint32_t));
    node->next = reg->buckets[idx];
    reg->buckets[idx] = node;
    reg->total_count++;
    return false;
}

typedef struct {
    CARPENTER_TT *tt;
    FCP_Registry *reg;
    uint32_t min_sup;
    size_t total_closed;
    size_t total_footprint;
} CARPENTER_Context;

static void mine_pattern(uint32_t *active_indices, size_t num_active,
                         uint32_t x_len,
                         uint32_t *r_prime, size_t r_prime_len,
                         CARPENTER_Context *ctx) {
    
    if (num_active == 0) return;

    // Step 1: Scan TT'|X and count the frequency of occurrences for each row ri in R'
    uint32_t *counts = calloc(ctx->tt->num_rows, sizeof(uint32_t));
    for (size_t i = 0; i < num_active; i++) {
        uint32_t f_idx = active_indices[i];
        const uint64_t *bits = ctx->tt->tuples[f_idx].row_bits;
        for (size_t r = 0; r < r_prime_len; r++) {
            uint32_t rid = r_prime[r];
            if (bs_test(bits, rid)) counts[rid]++;
        }
    }

    // Step 2 (Pruning 1): Let U subset R' be the set of rows which occur in at least one tuple
    uint32_t *u = malloc(r_prime_len * sizeof(uint32_t));
    size_t u_len = 0;
    for (size_t r = 0; r < r_prime_len; r++) {
        uint32_t rid = r_prime[r];
        if (counts[rid] > 0) u[u_len++] = rid;
    }
    
    if (u_len + x_len < ctx->min_sup) {
        free(u); free(counts); return;
    }

    // Step 3 (Pruning 2): Let Y be the set of rows which are found in every tuple of TT'|X
    uint32_t *y = malloc(u_len * sizeof(uint32_t));
    size_t y_len = 0;
    uint32_t *r_prime_new = malloc(u_len * sizeof(uint32_t));
    size_t r_prime_new_len = 0;

    for (size_t i = 0; i < u_len; i++) {
        uint32_t rid = u[i];
        if (counts[rid] == (uint32_t)num_active) {
            y[y_len++] = rid;
        } else {
            r_prime_new[r_prime_new_len++] = rid;
        }
    }

    // Preparation of feature set F(X)
    uint32_t *f_x = malloc(num_active * sizeof(uint32_t));
    for (size_t i = 0; i < num_active; i++) {
        f_x[i] = ctx->tt->tuples[active_indices[i]].feature_id;
    }

    // Step 4 (Pruning 3): If F(X) in FCP, then return
    if (fcp_check_and_add(ctx->reg, f_x, num_active)) {
        free(f_x); free(r_prime_new); free(y); free(u); free(counts);
        return;
    }

    // Step 5: If |X| + |Y| >= minsup, add F(X) into FCP
    if (x_len + y_len >= ctx->min_sup) {
        ctx->total_closed++;
        ctx->total_footprint += num_active;
    }

    // Step 6: DFS enumeration
    for (size_t i = 0; i < r_prime_new_len; i++) {
        uint32_t ri = r_prime_new[i];
        
        // Form TT'|X|ri
        uint32_t *next_active = malloc(num_active * sizeof(uint32_t));
        size_t next_num_active = 0;
        for (size_t j = 0; j < num_active; j++) {
            if (bs_test(ctx->tt->tuples[active_indices[j]].row_bits, ri)) {
                next_active[next_num_active++] = active_indices[j];
            }
        }

        if (next_num_active > 0) {
            // New R' is the set of rows in R'_new occurring after ri
            size_t next_r_prime_len = r_prime_new_len - (i + 1);
            uint32_t *next_r_prime = (next_r_prime_len > 0) ? &r_prime_new[i+1] : NULL;
            
            // X_next = X U Y U {ri}
            mine_pattern(next_active, next_num_active, x_len + (uint32_t)y_len + 1, next_r_prime, next_r_prime_len, ctx);
        }
        free(next_active);
    }

    free(f_x); free(r_prime_new); free(y); free(u); free(counts);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_CARPENTER_Params *p = (DM_CARPENTER_Params *)params;
    double min_sup_param = p ? p->min_support : 0.05;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[CARPENTER] Starting on %zu rows. Min Support: %u\n", ds->count, min_sup);

    // 1. Build Transposed Table (TT)
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
        printf("[CARPENTER] No frequent features found.\n");
        free(counts);
        return DM_SUCCESS;
    }

    CARPENTER_TT tt;
    tt.num_tuples = num_frequent;
    tt.num_rows = ds->count;
    tt.num_words = (ds->count + 63) / 64;
    tt.tuples = malloc(num_frequent * sizeof(CARPENTER_Tuple));
    
    uint32_t *temp_map = malloc((ds->max_id + 1) * sizeof(uint32_t));
    memset(temp_map, 0xFF, (ds->max_id + 1) * sizeof(uint32_t));

    size_t f_idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            tt.tuples[f_idx].feature_id = i;
            tt.tuples[f_idx].row_bits = calloc(tt.num_words, sizeof(uint64_t));
            temp_map[i] = (uint32_t)f_idx;
            f_idx++;
        }
    }

    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            uint32_t mapped = temp_map[item];
            if (mapped != 0xFFFFFFFF) bs_set(tt.tuples[mapped].row_bits, (uint32_t)i);
        }
    }
    free(temp_map);

    // 2. Initialize Registry and Context
    FCP_Registry reg;
    reg.buckets = calloc(HASH_SIZE, sizeof(FCPNode *));
    reg.total_count = 0;

    CARPENTER_Context ctx;
    ctx.tt = &tt;
    ctx.reg = &reg;
    ctx.min_sup = min_sup;
    ctx.total_closed = 0;
    ctx.total_footprint = 0;

    // 3. Initial RowSet R' and Active Features
    uint32_t *active_indices = malloc(num_frequent * sizeof(uint32_t));
    for (size_t i = 0; i < num_frequent; i++) active_indices[i] = (uint32_t)i;

    uint32_t *r_prime = malloc(ds->count * sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) r_prime[i] = (uint32_t)i;

    // 4. Run Mining
    mine_pattern(active_indices, (uint32_t)num_frequent, 0, r_prime, ds->count, &ctx);

    printf("[CARPENTER] Complete. Total frequent closed patterns found: %zu\n", ctx.total_closed);
    dm_bench_record_results(ctx.total_closed, ctx.total_footprint);

    // 5. Cleanup
    for (size_t i = 0; i < num_frequent; i++) free(tt.tuples[i].row_bits);
    free(tt.tuples);
    free(active_indices);
    free(r_prime);
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

static DM_Algorithm algo_carpenter = {
    .id = "carpenter",
    .name = "CARPENTER Algorithm",
    .description = "A row-wise enumeration algorithm for finding frequent closed patterns in long datasets.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo_carpenter)
