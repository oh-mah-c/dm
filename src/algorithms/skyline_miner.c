#include "algorithms/skyline_miner.h"
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
    int f;
    double u;
} SkylineEntry;

typedef struct {
    uint32_t id;
    double twu;
} ItemTWU;

/* --- CONTEXT --- */

static SkylineEntry *skyline = NULL;
static size_t skyline_count = 0;
static size_t skyline_capacity = 0;
static double *umax = NULL;
static size_t max_f = 0;

/* --- UTILS --- */

static int cmp_item_twu(const void *a, const void *b) {
    double diff = ((ItemTWU *)a)->twu - ((ItemTWU *)b)->twu;
    return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
}

static void add_to_skyline(uint32_t *items, size_t len, int f, double u) {
    // 1. Check if (f, u) is dominated
    for (size_t i = 0; i < skyline_count; i++) {
        if (skyline[i].f >= f && skyline[i].u >= u && (skyline[i].f > f || skyline[i].u > u)) {
            return;
        }
    }

    // 2. Remove items dominated by (f, u)
    size_t write_idx = 0;
    for (size_t i = 0; i < skyline_count; i++) {
        if (f >= skyline[i].f && u >= skyline[i].u && (f > skyline[i].f || u > skyline[i].u)) {
            free(skyline[i].items);
        } else {
            skyline[write_idx++] = skyline[i];
        }
    }
    skyline_count = write_idx;

    // 3. Add (f, u)
    if (skyline_count >= skyline_capacity) {
        skyline_capacity = skyline_capacity == 0 ? 100 : skyline_capacity * 2;
        skyline = realloc(skyline, sizeof(SkylineEntry) * skyline_capacity);
    }
    skyline[skyline_count].items = malloc(sizeof(uint32_t) * len);
    memcpy(skyline[skyline_count].items, items, sizeof(uint32_t) * len);
    skyline[skyline_count].len = len;
    skyline[skyline_count].f = f;
    skyline[skyline_count].u = u;
    skyline_count++;

    // 4. Update umax
    for (int i = 1; i <= f; i++) {
        if (u > umax[i]) umax[i] = u;
    }
}

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
                // Search for tid in p
                // Optimization: since tids are sorted, we can keep track of index in p
                // But for simplicity, we just look for it or assume it's there
                // HUI-Miner construct logic:
                double rutil_p = 0;
                // find tid in p
                static size_t ip = 0; // Not thread safe but okay for this context
                // Reset ip if needed? No, caller should manage or we search.
                // Re-searching for simplicity:
                for(size_t k = 0; k < p->count; k++) {
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

static void search(uint32_t *prefix, size_t prefix_len, UtilityList *p, UtilityList **extensions, size_t ext_count) {
    for (size_t i = 0; i < ext_count; i++) {
        UtilityList *x = extensions[i];
        
        // Skyline Check & Pruning
        if (x->sum_iutil >= umax[x->count]) {
            uint32_t *new_prefix = malloc(sizeof(uint32_t) * (prefix_len + 1));
            memcpy(new_prefix, prefix, sizeof(uint32_t) * prefix_len);
            new_prefix[prefix_len] = x->item;
            add_to_skyline(new_prefix, prefix_len + 1, (int)x->count, x->sum_iutil);
            free(new_prefix);
        }

        // Pruning extensions
        if (x->sum_iutil + x->sum_rutil >= umax[x->count]) {
            UtilityList **exes = malloc(sizeof(UtilityList *) * (ext_count - i - 1));
            size_t exes_count = 0;
            for (size_t j = i + 1; j < ext_count; j++) {
                UtilityList *y = extensions[j];
                UtilityList *xy = construct(p, x, y);
                if (xy->count > 0) {
                    xy->item = y->item;
                    exes[exes_count++] = xy;
                } else {
                    free(xy->entries);
                    free(xy);
                }
            }

            if (exes_count > 0) {
                uint32_t *next_prefix = malloc(sizeof(uint32_t) * (prefix_len + 1));
                memcpy(next_prefix, prefix, sizeof(uint32_t) * prefix_len);
                next_prefix[prefix_len] = x->item;
                search(next_prefix, prefix_len + 1, x, exes, exes_count);
                free(next_prefix);
            }

            for (size_t j = 0; j < exes_count; j++) {
                free(exes[j]->entries);
                free(exes[j]);
            }
            free(exes);
        }
    }
}

/* --- MAIN LOGIC --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_Skyline_Miner_Params *p = (DM_Skyline_Miner_Params *)params;
    double min_utility = p ? p->min_utility : 0;
    int min_support = p ? p->min_support : 0;

    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;

    // 1. Calculate TWU
    double *twus = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) twus[src[i].items[j].id] += src[i].total_utility;
    }

    ItemTWU *items = malloc(sizeof(ItemTWU) * (ds->max_id + 1));
    size_t item_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twus[i] >= min_utility) {
            items[item_count].id = i;
            items[item_count].twu = twus[i];
            item_count++;
        }
    }
    qsort(items, item_count, sizeof(ItemTWU), cmp_item_twu);

    // Map IDs to their rank
    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < item_count; i++) rank[items[i].id] = (uint32_t)i;

    // 2. Build Utility Lists for 1-items
    UtilityList **uls = malloc(sizeof(UtilityList *) * item_count);
    for (size_t i = 0; i < item_count; i++) {
        uls[i] = malloc(sizeof(UtilityList));
        uls[i]->item = items[i].id;
        uls[i]->entries = malloc(sizeof(UL_Entry) * ds->count);
        uls[i]->count = 0;
        uls[i]->sum_iutil = 0;
        uls[i]->sum_rutil = 0;
    }

    for (size_t t = 0; t < ds->count; t++) {
        // Sort items in transaction by rank
        uint32_t *t_items = malloc(sizeof(uint32_t) * src[t].count);
        double *t_utils = malloc(sizeof(double) * src[t].count);
        size_t t_count = 0;
        for (size_t j = 0; j < src[t].count; j++) {
            if (twus[src[t].items[j].id] >= min_utility) {
                t_items[t_count] = src[t].items[j].id;
                t_utils[t_count] = src[t].items[j].utility;
                t_count++;
            }
        }
        // Selection sort for simplicity (transactions are small)
        for (size_t i = 0; i < t_count; i++) {
            for (size_t j = i + 1; j < t_count; j++) {
                if (rank[t_items[i]] > rank[t_items[j]]) {
                    uint32_t temp_id = t_items[i]; t_items[i] = t_items[j]; t_items[j] = temp_id;
                    double temp_u = t_utils[i]; t_utils[i] = t_utils[j]; t_utils[j] = temp_u;
                }
            }
        }

        double remaining = 0;
        for (int j = (int)t_count - 1; j >= 0; j--) {
            uint32_t r = rank[t_items[j]];
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

    // Initialize Skyline Context
    max_f = ds->count;
    umax = calloc(max_f + 1, sizeof(double));
    skyline_count = 0;
    skyline_capacity = 100;
    skyline = malloc(sizeof(SkylineEntry) * skyline_capacity);

    // Initial umax update with 1-items
    for (size_t i = 0; i < item_count; i++) {
        if (uls[i]->count >= (size_t)min_support) {
            add_to_skyline(&uls[i]->item, 1, (int)uls[i]->count, uls[i]->sum_iutil);
        }
    }

    // 3. Search
    search(NULL, 0, NULL, uls, item_count);

    dm_bench_record_results(skyline_count, 0);

    // Cleanup
    for (size_t i = 0; i < item_count; i++) {
        free(uls[i]->entries);
        free(uls[i]);
    }
    free(uls);
    free(twus);
    free(items);
    free(rank);
    free(umax);
    for (size_t i = 0; i < skyline_count; i++) free(skyline[i].items);
    free(skyline);

    return DM_SUCCESS;
}

DM_Algorithm skyline_miner_algo = {
    .id = "skyline_miner",
    .name = "Skyline-Miner",
    .description = "Mining Skyline Frequent-Utility Patterns (Lin et al. 2016).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(skyline_miner_algo)
