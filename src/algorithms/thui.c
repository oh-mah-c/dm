#include "algorithms/thui.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    int tid;
    double iutil;
    double rutil;
} UL_Entry;

typedef struct {
    uint32_t item;
    UL_Entry *entries;
    size_t count;
    double sum_iutil;
    double sum_rutil;
} UtilityList;

typedef struct {
    uint32_t *items;
    size_t len;
    double utility;
} HUI_Entry;

typedef struct {
    uint64_t key; // (start_item << 32) | end_item
    double utility;
} LIU_Entry;

typedef struct {
    LIU_Entry *table;
    size_t size;
    size_t count;
} LIU_Table;

typedef struct {
    HUI_Entry *buffer;
    size_t count;
    size_t k;
    double threshold;
} TopK_Buffer;

typedef struct {
    TopK_Buffer topk;
    LIU_Table liu;
    uint32_t *rank;
    double *item_utilities;
    size_t item_count;
} THUI_Context;

/* --- LIU HASH TABLE --- */

static void liu_init(LIU_Table *t, size_t size) {
    t->size = size;
    t->count = 0;
    t->table = calloc(size, sizeof(LIU_Entry));
}

static uint32_t hash_64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return (uint32_t)x;
}

static void liu_update(LIU_Table *t, uint32_t start, uint32_t end, double util) {
    uint64_t key = ((uint64_t)start << 32) | end;
    uint32_t h = hash_64(key) % t->size;
    while (t->table[h].utility > 0) {
        if (t->table[h].key == key) {
            t->table[h].utility += util;
            return;
        }
        h = (h + 1) % t->size;
    }
    t->table[h].key = key;
    t->table[h].utility = util;
    t->count++;
    // Simple resize omitted for brevity, assume initial size is sufficient or use a better map
}

/* --- TOP-K BUFFER --- */

static void topk_init(TopK_Buffer *b, size_t k) {
    b->k = k;
    b->count = 0;
    b->buffer = malloc(sizeof(HUI_Entry) * (k + 1));
    b->threshold = 0;
}

static void topk_update(TopK_Buffer *b, uint32_t *items, size_t len, double util) {
    if (util < b->threshold && b->count == b->k) return;

    // Check if itemset already exists (optional for THUI but good for correctness)
    for (size_t i = 0; i < b->count; i++) {
        if (b->buffer[i].len == len) {
            bool match = true;
            for (size_t j = 0; j < len; j++) if (b->buffer[i].items[j] != items[j]) { match = false; break; }
            if (match) return;
        }
    }

    // Insert and sort
    size_t i = b->count;
    while (i > 0 && b->buffer[i - 1].utility < util) {
        b->buffer[i] = b->buffer[i - 1];
        i--;
    }
    b->buffer[i].items = malloc(sizeof(uint32_t) * len);
    memcpy(b->buffer[i].items, items, sizeof(uint32_t) * len);
    b->buffer[i].len = len;
    b->buffer[i].utility = util;

    if (b->count < b->k) b->count++;
    else {
        free(b->buffer[b->k].items);
    }

    if (b->count == b->k) {
        b->threshold = b->buffer[b->k - 1].utility;
    }
}

/* --- LOGIC --- */

static UtilityList *construct(UtilityList *p, UtilityList *px, UtilityList *py) {
    UtilityList *pxy = malloc(sizeof(UtilityList));
    pxy->entries = malloc(sizeof(UL_Entry) * (px->count < py->count ? px->count : py->count));
    pxy->count = 0;
    pxy->sum_iutil = 0;
    pxy->sum_rutil = 0;

    size_t ix = 0, iy = 0;
    while (ix < px->count && iy < py->count) {
        if (px->entries[ix].tid < py->entries[iy].tid) ix++;
        else if (px->entries[ix].tid > py->entries[iy].tid) iy++;
        else {
            if (p) {
                // Simplified join - in a real HUI-Miner we'd need to find the matching TID in p
                // but since tids are sorted, we can search or use a pointer.
                // For this implementation, we search.
                double rutil_p = 0;
                for (size_t k = 0; k < p->count; k++) {
                    if (p->entries[k].tid == px->entries[ix].tid) {
                        rutil_p = p->entries[k].rutil;
                        break;
                    }
                }
                pxy->entries[pxy->count].tid = px->entries[ix].tid;
                pxy->entries[pxy->count].iutil = px->entries[ix].iutil + py->entries[iy].iutil - rutil_p;
                pxy->entries[pxy->count].rutil = py->entries[iy].rutil;
            } else {
                pxy->entries[pxy->count].tid = px->entries[ix].tid;
                pxy->entries[pxy->count].iutil = px->entries[ix].iutil + py->entries[iy].iutil;
                pxy->entries[pxy->count].rutil = py->entries[iy].rutil;
            }
            pxy->sum_iutil += pxy->entries[pxy->count].iutil;
            pxy->sum_rutil += pxy->entries[pxy->count].rutil;
            pxy->count++;
            ix++; iy++;
        }
    }
    return pxy;
}

static void explore(THUI_Context *ctx, uint32_t *prefix, size_t prefix_len, UtilityList **uls, size_t ul_count) {
    for (size_t i = 0; i < ul_count; i++) {
        UtilityList *x = uls[i];
        
        // RUC strategy: update threshold
        if (x->sum_iutil >= ctx->topk.threshold) {
            uint32_t *new_p = malloc(sizeof(uint32_t) * (prefix_len + 1));
            memcpy(new_p, prefix, sizeof(uint32_t) * prefix_len);
            new_p[prefix_len] = x->item;
            topk_update(&ctx->topk, new_p, prefix_len + 1, x->sum_iutil);
            free(new_p);
        }

        // Pruning
        if (x->sum_iutil + x->sum_rutil >= ctx->topk.threshold) {
            UtilityList **extensions = malloc(sizeof(UtilityList *) * (ul_count - i - 1));
            size_t ext_count = 0;
            for (size_t j = i + 1; j < ul_count; j++) {
                UtilityList *y = uls[j];
                UtilityList *xy = construct(uls[i], x, y);
                if (xy->count > 0) {
                    xy->item = y->item;
                    extensions[ext_count++] = xy;
                } else {
                    free(xy->entries);
                    free(xy);
                }
            }

            if (ext_count > 0) {
                uint32_t *next_p = malloc(sizeof(uint32_t) * (prefix_len + 1));
                memcpy(next_p, prefix, sizeof(uint32_t) * prefix_len);
                next_p[prefix_len] = x->item;
                explore(ctx, next_p, prefix_len + 1, extensions, ext_count);
                free(next_p);
            }

            for (size_t j = 0; j < ext_count; j++) {
                free(extensions[j]->entries);
                free(extensions[j]);
            }
            free(extensions);
        }
    }
}

static int cmp_items(const void *a, const void *b, void *arg) {
    uint32_t *rank = (uint32_t *)arg;
    return (rank[*(uint32_t *)a] < rank[*(uint32_t *)b]) ? -1 : 1;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_THUI_Params *p = (DM_THUI_Params *)params;
    size_t k = p ? p->k : 10;

    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    double *twus = calloc(ds->max_id + 1, sizeof(double));
    double *item_utils = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) {
            twus[src[i].items[j].id] += src[i].total_utility;
            item_utils[src[i].items[j].id] += src[i].items[j].utility;
        }
    }

    THUI_Context ctx;
    topk_init(&ctx.topk, k);
    liu_init(&ctx.liu, 1000003); // Large prime for hash table
    ctx.rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    ctx.item_utilities = item_utils;

    // Sort items by TWU
    typedef struct { uint32_t id; double twu; } ItemTWU;
    ItemTWU *sorted_items = malloc(sizeof(ItemTWU) * (ds->max_id + 1));
    size_t distinct_items = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twus[i] > 0) {
            sorted_items[distinct_items].id = i;
            sorted_items[distinct_items].twu = twus[i];
            distinct_items++;
        }
    }
    // TWU Sort (Definition 11: usually ascending TWU for HUI-Miner)
    for (size_t i = 0; i < distinct_items; i++) {
        for (size_t j = i + 1; j < distinct_items; j++) {
            if (sorted_items[i].twu > sorted_items[j].twu) {
                ItemTWU tmp = sorted_items[i]; sorted_items[i] = sorted_items[j]; sorted_items[j] = tmp;
            }
        }
    }
    for (size_t i = 0; i < distinct_items; i++) ctx.rank[sorted_items[i].id] = (uint32_t)i;

    // RIU Strategy
    for (size_t i = 0; i < distinct_items; i++) {
        topk_update(&ctx.topk, &sorted_items[i].id, 1, item_utils[sorted_items[i].id]);
    }

    // Second Scan: Build LIU and Utility Lists
    UtilityList **uls = malloc(sizeof(UtilityList *) * distinct_items);
    for (size_t i = 0; i < distinct_items; i++) {
        uls[i] = malloc(sizeof(UtilityList));
        uls[i]->item = sorted_items[i].id;
        uls[i]->entries = malloc(sizeof(UL_Entry) * ds->count);
        uls[i]->count = 0;
        uls[i]->sum_iutil = 0;
        uls[i]->sum_rutil = 0;
    }

    for (size_t t = 0; t < ds->count; t++) {
        uint32_t *t_items = malloc(sizeof(uint32_t) * src[t].count);
        double *t_utils = malloc(sizeof(double) * src[t].count);
        size_t t_len = src[t].count;
        for (size_t j = 0; j < t_len; j++) {
            t_items[j] = src[t].items[j].id;
            t_utils[j] = src[t].items[j].utility;
        }

        // Sort by Rank
        for (size_t i = 0; i < t_len; i++) {
            for (size_t j = i + 1; j < t_len; j++) {
                if (ctx.rank[t_items[i]] > ctx.rank[t_items[j]]) {
                    uint32_t tmp_i = t_items[i]; t_items[i] = t_items[j]; t_items[j] = tmp_i;
                    double tmp_u = t_utils[i]; t_utils[i] = t_utils[j]; t_utils[j] = tmp_u;
                }
            }
        }

        // Update LIU
        for (size_t i = 0; i < t_len; i++) {
            double cum = t_utils[i];
            for (size_t j = i + 1; j < t_len; j++) {
                cum += t_utils[j];
                liu_update(&ctx.liu, t_items[i], t_items[j], cum);
            }
        }

        // Build UL
        double remaining = 0;
        for (int j = (int)t_len - 1; j >= 0; j--) {
            uint32_t r = ctx.rank[t_items[j]];
            uls[r]->entries[uls[r]->count].tid = (int)t;
            uls[r]->entries[uls[r]->count].iutil = t_utils[j];
            uls[r]->entries[uls[r]->count].rutil = remaining;
            uls[r]->sum_iutil += t_utils[j];
            uls[r]->sum_rutil += remaining;
            uls[r]->count++;
            remaining += t_utils[j];
        }
        free(t_items);
        free(t_utils);
    }

    // LIU-E Strategy
    for (size_t i = 0; i < ctx.liu.size; i++) {
        if (ctx.liu.table[i].utility > 0) {
            uint32_t start = (uint32_t)(ctx.liu.table[i].key >> 32);
            uint32_t end = (uint32_t)ctx.liu.table[i].key;
            // Since we don't store the full itemset, this only works for contiguous itemsets
            // but the paper says LIU(x,y) is a contiguous sequence.
            // For LIU-E we can just use the utility value if we knew the sequence length
            // or just assume it represents 'some' itemset.
            // Correct approach: generate the itemset.
            uint32_t start_rank = ctx.rank[start];
            uint32_t end_rank = ctx.rank[end];
            size_t len = end_rank - start_rank + 1;
            uint32_t *items = malloc(sizeof(uint32_t) * len);
            for (uint32_t r = start_rank; r <= end_rank; r++) items[r - start_rank] = sorted_items[r].id;
            topk_update(&ctx.topk, items, len, ctx.liu.table[i].utility);
            free(items);
        }
    }

    // LIU-LB Strategy (Algorithm 2)
    for (size_t i = 0; i < ctx.liu.size; i++) {
        if (ctx.liu.table[i].utility > 0) {
            uint32_t start = (uint32_t)(ctx.liu.table[i].key >> 32);
            uint32_t end = (uint32_t)ctx.liu.table[i].key;
            uint32_t start_rank = ctx.rank[start];
            uint32_t end_rank = ctx.rank[end];
            
            if (end_rank - start_rank > 1) {
                // 1 item exclusion
                for (uint32_t r = start_rank + 1; r < end_rank; r++) {
                    double lb = ctx.liu.table[i].utility - item_utils[sorted_items[r].id];
                    topk_update(&ctx.topk, NULL, 0, lb); // In THUI we can update threshold with just LB
                }
                // 2 item exclusion
                for (uint32_t r1 = start_rank + 1; r1 < end_rank; r1++) {
                    for (uint32_t r2 = r1 + 1; r2 < end_rank; r2++) {
                        double lb = ctx.liu.table[i].utility - item_utils[sorted_items[r1].id] - item_utils[sorted_items[r2].id];
                        topk_update(&ctx.topk, NULL, 0, lb);
                    }
                }
            }
        }
    }

    // Final Growth
    explore(&ctx, NULL, 0, uls, distinct_items);

    dm_bench_record_results(ctx.topk.count, 0);

    // Cleanup
    for (size_t i = 0; i < ctx.topk.count; i++) free(ctx.topk.buffer[i].items);
    free(ctx.topk.buffer);
    free(ctx.liu.table);
    free(ctx.rank);
    free(item_utils);
    free(twus);
    free(sorted_items);
    for (size_t i = 0; i < distinct_items; i++) { free(uls[i]->entries); free(uls[i]); }
    free(uls);

    return DM_SUCCESS;
}

DM_Algorithm thui_algo = {
    .id = "thui",
    .name = "THUI",
    .description = "Mining top-k high utility itemsets with effective threshold raising (Krishnamoorthy 2019).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(thui_algo)
