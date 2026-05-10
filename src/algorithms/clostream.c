#include "algorithms/clostream.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * CloStream Algorithm for maintaining frequent closed itemsets over data streams.
 * Reference: S.-J. Yen et al., "An Efficient Algorithm for Maintaining 
 * Frequent Closed Itemsets over Data Stream", IEA/AIE 2009.
 */

typedef struct {
    uint32_t *items;
    uint32_t len;
    uint32_t sc;
    uint32_t cid;
} ClosedRecord;

typedef struct {
    uint32_t *cids;
    uint32_t count;
    uint32_t cap;
} CidSet;

static bool itemset_equal(uint32_t *a, uint32_t alen, uint32_t *b, uint32_t blen) {
    if (alen != blen) return false;
    for (uint32_t i = 0; i < alen; i++) if (a[i] != b[i]) return false;
    return true;
}

static uint32_t intersect(uint32_t *res, uint32_t *a, uint32_t alen, uint32_t *b, uint32_t blen) {
    uint32_t i = 0, j = 0, k = 0;
    while (i < alen && j < blen) {
        if (a[i] == b[j]) { res[k++] = a[i]; i++; j++; }
        else if (a[i] < b[j]) i++;
        else j++;
    }
    return k;
}

typedef struct TempRecord {
    uint32_t *items;
    uint32_t len;
    uint32_t closure_id;
    uint32_t original_sc; // Support count in DB
    struct TempRecord *next;
} TempRecord;

typedef struct {
    TempRecord **buckets;
    size_t size;
    size_t count;
} TempTable;

static uint32_t hash_itemset(uint32_t *items, uint32_t len) {
    uint32_t hash = 5381;
    for (uint32_t i = 0; i < len; i++) hash = ((hash << 5) + hash) + items[i];
    return hash;
}

static void add_to_temp(TempTable *temp, uint32_t *items, uint32_t len, uint32_t cid, ClosedRecord *ct) {
    uint32_t h = hash_itemset(items, len) % temp->size;
    TempRecord *curr = temp->buckets[h];
    while (curr) {
        if (itemset_equal(curr->items, curr->len, items, len)) {
            if (ct[cid].sc > ct[curr->closure_id].sc) {
                curr->closure_id = cid;
                curr->original_sc = ct[cid].sc;
            }
            return;
        }
        curr = curr->next;
    }
    TempRecord *node = malloc(sizeof(TempRecord));
    node->items = malloc(len * sizeof(uint32_t));
    memcpy(node->items, items, len * sizeof(uint32_t));
    node->len = len;
    node->closure_id = cid;
    node->original_sc = ct[cid].sc;
    node->next = temp->buckets[h];
    temp->buckets[h] = node;
    temp->count++;
}

static void add_to_cidset(CidSet *cs, uint32_t cid) {
    for (uint32_t i = 0; i < cs->count; i++) if (cs->cids[i] == cid) return;
    if (cs->count >= cs->cap) {
        cs->cap = (cs->cap == 0) ? 4 : cs->cap * 2;
        cs->cids = realloc(cs->cids, cs->cap * sizeof(uint32_t));
    }
    cs->cids[cs->count++] = cid;
}

static int compare_uint32(const void *a, const void *b) {
    uint32_t v1 = *(uint32_t *)a;
    uint32_t v2 = *(uint32_t *)b;
    if (v1 < v2) return -1;
    if (v1 > v2) return 1;
    return 0;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_CLOSTREAM_Params *p = (DM_CLOSTREAM_Params *)params;
    double min_sup_param = p ? p->min_support : 0.4;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;

    printf("[CloStream] Starting on %zu transactions. Min Sup: %u\n", ds->count, min_sup);

    // Initial Closed Table with record 0 (empty set)
    ClosedRecord *ct = malloc(1 * sizeof(ClosedRecord));
    ct[0].cid = 0;
    ct[0].items = NULL;
    ct[0].len = 0;
    ct[0].sc = 0;
    size_t ct_count = 1;
    size_t ct_cap = 1;

    CidSet *cl = calloc(ds->max_id + 1, sizeof(CidSet));

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    uint32_t *inter_buf = malloc((ds->max_id + 1) * sizeof(uint32_t));

    uint8_t *active_flags = calloc(1000000, 1);
    uint32_t active_flags_size = 1000000;

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *t_items = data[i].items;
        uint32_t t_len = (uint32_t)data[i].count;
        qsort(t_items, t_len, sizeof(uint32_t), compare_uint32);

        TempTable temp = { .buckets = calloc(65536, sizeof(TempRecord*)), .size = 65536, .count = 0 };
        
        // Phase 1: Intersections
        uint32_t *active_cids = malloc(16 * sizeof(uint32_t));
        size_t active_count = 0;
        size_t active_cap = 16;

        // Resize active_flags if needed
        if (ct_count >= active_flags_size) {
            uint32_t new_size = active_flags_size * 2;
            while (ct_count >= new_size) new_size *= 2;
            active_flags = realloc(active_flags, new_size);
            memset(active_flags + active_flags_size, 0, new_size - active_flags_size);
            active_flags_size = new_size;
        }

        for (uint32_t j = 0; j < t_len; j++) {
            uint32_t item = t_items[j];
            for (uint32_t k = 0; k < cl[item].count; k++) {
                uint32_t cid = cl[item].cids[k];
                if (!active_flags[cid]) {
                    active_flags[cid] = 1;
                    if (active_count >= active_cap) {
                        active_cap *= 2;
                        active_cids = realloc(active_cids, active_cap * sizeof(uint32_t));
                    }
                    active_cids[active_count++] = cid;
                }
            }
        }

        // Add t_A itself to Temp_A
        add_to_temp(&temp, t_items, t_len, 0, ct);

        for (size_t j = 0; j < active_count; j++) {
            uint32_t cid = active_cids[j];
            active_flags[cid] = 0; // Reset flag for next transaction
            uint32_t s_len = intersect(inter_buf, t_items, t_len, ct[cid].items, ct[cid].len);
            if (s_len > 0) {
                add_to_temp(&temp, inter_buf, s_len, cid, ct);
            }
        }
        free(active_cids);

        // Phase 2: Update Closed Table and Cid List
        for (size_t j = 0; j < temp.size; j++) {
            TempRecord *curr = temp.buckets[j];
            while (curr) {
                uint32_t *X = curr->items;
                uint32_t Xlen = curr->len;
                uint32_t c = curr->closure_id;
                uint32_t old_sc = curr->original_sc;

                if (itemset_equal(X, Xlen, ct[c].items, ct[c].len)) {
                    ct[c].sc++;
                } else {
                    // New closed itemset
                    if (ct_count >= ct_cap) {
                        ct_cap = (ct_cap == 0) ? 2 : ct_cap * 2;
                        ct = realloc(ct, ct_cap * sizeof(ClosedRecord));
                    }
                    uint32_t new_cid = (uint32_t)ct_count;
                    ct[new_cid].cid = new_cid;
                    ct[new_cid].items = malloc(Xlen * sizeof(uint32_t));
                    memcpy(ct[new_cid].items, X, Xlen * sizeof(uint32_t));
                    ct[new_cid].len = Xlen;
                    ct[new_cid].sc = old_sc + 1; // Accurate support from DB + 1
                    ct_count++;

                    for (uint32_t k = 0; k < Xlen; k++) {
                        add_to_cidset(&cl[X[k]], new_cid);
                    }
                }
                
                TempRecord *next = curr->next;
                free(X);
                free(curr);
                curr = next;
            }
        }
        free(temp.buckets);
    }

    size_t total_fci = 0;
    size_t total_footprint = 0;
    for (size_t i = 1; i < ct_count; i++) {
        if (ct[i].sc >= min_sup) {
            total_fci++;
            total_footprint += ct[i].len;
        }
    }

    printf("[CloStream] Complete. Total Frequent Closed Itemsets: %zu\n", total_fci);
    dm_bench_record_results(total_fci, total_footprint);

    // Cleanup
    for (size_t i = 1; i < ct_count; i++) free(ct[i].items);
    free(ct);
    for (uint32_t i = 0; i <= ds->max_id; i++) free(cl[i].cids);
    free(cl);
    free(inter_buf);
    free(active_flags);

    return DM_SUCCESS;
}

static DM_Algorithm algo_clostream = {
    .id = "clostream",
    .name = "CloStream Algorithm",
    .description = "Incremental maintenance of frequent closed itemsets over data streams.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo_clostream)
