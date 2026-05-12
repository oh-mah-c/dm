#include "algorithms/tshoun.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

typedef struct {
    uint32_t u, v;
    double *puu_per_period; // Periodical Utility Upper-bound
} TSHOUN_HPUU2_Entry;

typedef struct {
    TSHOUN_HPUU2_Entry *entries;
    size_t count;
    size_t capacity;
} TSHOUN_HPUU2;

typedef struct TSHOUN_Node {
    uint32_t *items;
    size_t item_count;
    double *pu_per_period;
    struct TSHOUN_Node *next;
} TSHOUN_Node;

#define HASH_SIZE 65536
static TSHOUN_Node *hash_table[HASH_SIZE];

static uint32_t hash_itemset(uint32_t *items, size_t count) {
    uint32_t hash = 0;
    for (size_t i = 0; i < count; i++) {
        hash = hash * 31 + items[i];
    }
    return hash % HASH_SIZE;
}

static TSHOUN_Node* hash_find(uint32_t *items, size_t count) {
    uint32_t h = hash_itemset(items, count);
    TSHOUN_Node *node = hash_table[h];
    while (node) {
        if (node->item_count == count) {
            bool match = true;
            for (size_t i = 0; i < count; i++) {
                if (node->items[i] != items[i]) { match = false; break; }
            }
            if (match) return node;
        }
        node = node->next;
    }
    return NULL;
}

static void hash_add(uint32_t *items, size_t count, int num_periods, int period, double utility) {
    TSHOUN_Node *node = hash_find(items, count);
    if (node) {
        node->pu_per_period[period] += utility;
    } else {
        uint32_t h = hash_itemset(items, count);
        node = malloc(sizeof(TSHOUN_Node));
        node->items = malloc(sizeof(uint32_t) * count);
        memcpy(node->items, items, sizeof(uint32_t) * count);
        node->item_count = count;
        node->pu_per_period = calloc(num_periods, sizeof(double));
        node->pu_per_period[period] = utility;
        node->next = hash_table[h];
        hash_table[h] = node;
    }
}

static void hpuu2_add(TSHOUN_HPUU2 *table, uint32_t u, uint32_t v, int period, double tu, int num_periods) {
    if (u > v) { uint32_t t = u; u = v; v = t; }
    for (size_t i = 0; i < table->count; i++) {
        if (table->entries[i].u == u && table->entries[i].v == v) {
            table->entries[i].puu_per_period[period] += tu;
            return;
        }
    }
    if (table->count >= table->capacity) {
        table->capacity *= 2;
        table->entries = realloc(table->entries, sizeof(TSHOUN_HPUU2_Entry) * table->capacity);
    }
    table->entries[table->count].u = u;
    table->entries[table->count].v = v;
    table->entries[table->count].puu_per_period = calloc(num_periods, sizeof(double));
    table->entries[table->count].puu_per_period[period] = tu;
    table->count++;
}

static bool hpuu2_check(TSHOUN_HPUU2 *table, uint32_t u, uint32_t v, int period, double min_util, double *pttu) {
    if (u > v) { uint32_t t = u; u = v; v = t; }
    for (size_t i = 0; i < table->count; i++) {
        if (table->entries[i].u == u && table->entries[i].v == v) {
            return (table->entries[i].puu_per_period[period] / pttu[period] >= min_util);
        }
    }
    return false;
}

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

typedef struct {
    uint32_t *t_items;
    double *t_utils;
    size_t t_count;
    int period;
    int num_periods;
    TSHOUN_HPUU2 *hpuu2;
    double min_util;
    double *pttu;
} TSHOUN_GenCtx;

static void find_cliques(TSHOUN_GenCtx *ctx, uint32_t *current, size_t current_count, size_t start_idx) {
    for (size_t i = start_idx; i < ctx->t_count; i++) {
        uint32_t item = ctx->t_items[i];
        
        bool ok = true;
        for (size_t j = 0; j < current_count; j++) {
            if (!hpuu2_check(ctx->hpuu2, item, current[j], ctx->period, ctx->min_util, ctx->pttu)) {
                ok = false;
                break;
            }
        }
        
        if (ok) {
            current[current_count] = item;
            double util = 0;
            for (size_t j = 0; j <= current_count; j++) {
                for (size_t k = 0; k < ctx->t_count; k++) {
                    if (ctx->t_items[k] == current[j]) { util += ctx->t_utils[k]; break; }
                }
            }
            
            hash_add(current, current_count + 1, ctx->num_periods, ctx->period, util);
            
            if (current_count + 1 < 32) {
                find_cliques(ctx, current, current_count + 1, i + 1);
            }
        }
    }
}

static void generate_itemsets(uint32_t *t_items, double *t_utils, size_t t_count, int period, int num_periods, TSHOUN_HPUU2 *hpuu2, double min_util, double *pttu) {
    TSHOUN_GenCtx ctx = { t_items, t_utils, t_count, period, num_periods, hpuu2, min_util, pttu };
    uint32_t current[32];
    
    // Add 1-itemsets
    for (size_t i = 0; i < t_count; i++) {
        hash_add(&t_items[i], 1, num_periods, period, t_utils[i]);
    }
    
    find_cliques(&ctx, current, 0, 0);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_TSHOUN_Params *p = (DM_TSHOUN_Params *)params;
    double min_util = p ? p->min_utility : 0.3;
    int num_periods = p ? p->num_periods : 5;

    printf("Executing TS-HOUN Algorithm (minutil=%.4f, periods=%d)...\n", min_util, num_periods);

    memset(hash_table, 0, sizeof(hash_table));

    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    double *pttu = calloc(num_periods, sizeof(double));
    int *tid_period = malloc(sizeof(int) * ds->count);

    // Scan 1: PTTU and HPUU2
    printf("[TS-HOUN] Scan 1: PTTU and HPUU2...\n");
    TSHOUN_HPUU2 hpuu2 = { malloc(sizeof(TSHOUN_HPUU2_Entry) * 1024), 0, 1024 };

    for (size_t i = 0; i < ds->count; i++) {
        int h = (int)(i * num_periods / ds->count);
        if (h >= num_periods) h = num_periods - 1;
        tid_period[i] = h;

        double tu_pos = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (data[i].items[j].utility > 0) tu_pos += data[i].items[j].utility;
        }
        pttu[h] += tu_pos;

        for (size_t j = 0; j < data[i].count; j++) {
            for (size_t k = j + 1; k < data[i].count; k++) {
                hpuu2_add(&hpuu2, data[i].items[j].id, data[i].items[k].id, h, tu_pos, num_periods);
            }
        }
    }

    // Scan 2: Promising Itemsets
    printf("[TS-HOUN] Scan 2: Generating candidates...\n");
    for (size_t i = 0; i < ds->count; i++) {
        int h = tid_period[i];
        
        uint32_t t_items[data[i].count];
        double t_utils[data[i].count];
        size_t t_count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            t_items[t_count] = data[i].items[j].id;
            t_utils[t_count] = data[i].items[j].utility;
            t_count++;
        }
        // Sort items for consistent hashing
        for(size_t j=0; j<t_count; j++) {
            for(size_t k=j+1; k<t_count; k++) {
                if(t_items[j] > t_items[k]) {
                    uint32_t tmp_i = t_items[j]; t_items[j] = t_items[k]; t_items[k] = tmp_i;
                    double tmp_u = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = tmp_u;
                }
            }
        }

        generate_itemsets(t_items, t_utils, t_count, h, num_periods, &hpuu2, min_util, pttu);
    }

    // Scan 3: Filter HOUN
    printf("[TS-HOUN] Filtering High On-Shelf Utility Itemsets...\n");
    size_t hui_count = 0;
    size_t items_sum = 0;

    // First, determine op(i) - periods where item i appears
    bool **item_periods = malloc(sizeof(bool*) * (ds->max_id + 1));
    for (uint32_t i = 0; i <= ds->max_id; i++) item_periods[i] = calloc(num_periods, sizeof(bool));
    for (size_t i = 0; i < ds->count; i++) {
        int h = tid_period[i];
        for (size_t j = 0; j < data[i].count; j++) {
            item_periods[data[i].items[j].id][h] = true;
        }
    }

    for (int i = 0; i < HASH_SIZE; i++) {
        TSHOUN_Node *node = hash_table[i];
        while (node) {
            // Determine op(X) = Intersection of op(i) for all i in X
            bool op_x[num_periods];
            double total_pttu_opx = 0;
            double actual_utility = 0;
            
            for (int h = 0; h < num_periods; h++) {
                op_x[h] = true;
                for (size_t j = 0; j < node->item_count; j++) {
                    if (!item_periods[node->items[j]][h]) { op_x[h] = false; break; }
                }
                if (op_x[h]) {
                    total_pttu_opx += pttu[h];
                    actual_utility += node->pu_per_period[h];
                }
            }

            if (total_pttu_opx > 0 && actual_utility / total_pttu_opx >= min_util) {
                hui_count++;
                items_sum += node->item_count;
            }

            TSHOUN_Node *tmp = node;
            node = node->next;
            free(tmp->items);
            free(tmp->pu_per_period);
            free(tmp);
        }
    }

    printf("[TS-HOUN] Found %zu HOUN itemsets.\n", hui_count);

    // Cleanup
    for (uint32_t i = 0; i <= ds->max_id; i++) free(item_periods[i]);
    free(item_periods);
    for (size_t i = 0; i < hpuu2.count; i++) free(hpuu2.entries[i].puu_per_period);
    free(hpuu2.entries);
    free(pttu);
    free(tid_period);

    dm_bench_record_results(hui_count, items_sum);
    return DM_SUCCESS;
}

static DM_Algorithm tshoun_algo = {
    .id = "tshoun",
    .name = "TS-HOUN Algorithm",
    .description = "Three-Scan High On-Shelf Utility miner with Negative profit.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(tshoun_algo)
