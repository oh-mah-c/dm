#include "algorithms/hui_list_ins.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t tid;
    double iutil;
    double rutil;
} HLI_Tuple;

typedef struct {
    uint32_t item;
    HLI_Tuple *tuples;
    size_t count;
    double sum_iutil;
    double sum_rutil;
} HLI_UtilityList;

typedef struct {
    uint32_t u, v;
    double twu;
} HLI_EUCS_Entry;

typedef struct {
    HLI_EUCS_Entry *entries;
    size_t count;
    size_t capacity;
} HLI_EUCS;

/* --- UTILS --- */

static HLI_UtilityList* construct(HLI_UtilityList *p, HLI_UtilityList *px, HLI_UtilityList *py) {
    HLI_UtilityList *pxy = calloc(1, sizeof(HLI_UtilityList));
    pxy->item = py->item;
    pxy->tuples = malloc(sizeof(HLI_Tuple) * (px->count < py->count ? px->count : py->count));

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < px->count && iy < py->count) {
        if (px->tuples[ix].tid == py->tuples[iy].tid) {
            uint32_t tid = px->tuples[ix].tid;
            double iutil = px->tuples[ix].iutil + py->tuples[iy].iutil;
            
            if (p != NULL) {
                while (ip < p->count && p->tuples[ip].tid < tid) ip++;
                if (ip < p->count && p->tuples[ip].tid == tid) {
                    iutil -= p->tuples[ip].iutil;
                }
            }
            
            pxy->tuples[pxy->count].tid = tid;
            pxy->tuples[pxy->count].iutil = iutil;
            pxy->tuples[pxy->count].rutil = py->tuples[iy].rutil;
            pxy->sum_iutil += iutil;
            pxy->sum_rutil += py->tuples[iy].rutil;
            pxy->count++;
            
            ix++; iy++;
        } else if (px->tuples[ix].tid < py->tuples[iy].tid) ix++;
        else iy++;
    }
    return pxy;
}

static void free_ul(HLI_UtilityList *ul) {
    if (!ul) return;
    free(ul->tuples);
    free(ul);
}

static size_t hui_count = 0;

static void search(HLI_UtilityList *p, HLI_UtilityList **extensions, size_t ext_count, double min_util, HLI_EUCS *eucs) {
    for (size_t i = 0; i < ext_count; i++) {
        HLI_UtilityList *px = extensions[i];
        if (px->sum_iutil >= min_util) {
            hui_count++;
        }

        if (px->sum_iutil + px->sum_rutil >= min_util) {
            HLI_UtilityList **ext_px = malloc(sizeof(HLI_UtilityList*) * (ext_count - i - 1));
            size_t ext_px_count = 0;

            for (size_t j = i + 1; j < ext_count; j++) {
                HLI_UtilityList *py = extensions[j];
                
                // EUCS Pruning
                double twu_pair = 0;
                uint32_t u = px->item, v = py->item;
                if (u > v) { uint32_t t = u; u = v; v = t; }
                for (size_t k = 0; k < eucs->count; k++) {
                    if (eucs->entries[k].u == u && eucs->entries[k].v == v) {
                        twu_pair = eucs->entries[k].twu;
                        break;
                    }
                }

                if (twu_pair >= min_util) {
                    HLI_UtilityList *pxy = construct(p, px, py);
                    if (pxy->sum_iutil + pxy->sum_rutil >= min_util) {
                        ext_px[ext_px_count++] = pxy;
                    } else {
                        free_ul(pxy);
                    }
                }
            }

            if (ext_px_count > 0) {
                search(px, ext_px, ext_px_count, min_util, eucs);
            }

            for (size_t k = 0; k < ext_px_count; k++) free_ul(ext_px[k]);
            free(ext_px);
        }
    }
}

static int cmp_item_twu(void *twu_arr, const void *a, const void *b) {
    uint32_t i1 = *(const uint32_t *)a;
    uint32_t i2 = *(const uint32_t *)b;
    double *twu = (double *)twu_arr;
    if (twu[i1] < twu[i2]) return -1;
    if (twu[i1] > twu[i2]) return 1;
    return (int)i1 - (int)i2;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_HUI_LIST_INS_Params *p = (DM_HUI_LIST_INS_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    
    // Split for incremental simulation (D and d)
    size_t total_count = ds->count;
    size_t d_count = (size_t)(total_count * 0.8);
    if (d_count == 0 && total_count > 0) d_count = 1;
    
    printf("[HUI-LIST-INS] Total: %zu, D: %zu, d: %zu, minutil: %.2f\n", total_count, d_count, total_count - d_count, min_util);

    hui_count = 0;
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;

    // 1. Calculate TWU for all items in D+d
    double *twu = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < total_count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            twu[data[i].items[j].id] += data[i].total_utility;
        }
    }

    // 2. Identify I* (TWU >= minutil) and Establish Order
    uint32_t *i_star = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    size_t i_star_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu[i] >= min_util) i_star[i_star_count++] = i;
    }
    qsort_s(i_star, i_star_count, sizeof(uint32_t), cmp_item_twu, twu);

    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < i_star_count; i++) rank[i_star[i]] = (uint32_t)i;

    // 3. Build Merged Utility Lists and EUCS
    HLI_UtilityList **initial_ext = malloc(sizeof(HLI_UtilityList*) * i_star_count);
    for (size_t i = 0; i < i_star_count; i++) {
        initial_ext[i] = calloc(1, sizeof(HLI_UtilityList));
        initial_ext[i]->item = i_star[i];
        initial_ext[i]->tuples = malloc(sizeof(HLI_Tuple) * 16);
    }

    HLI_EUCS eucs = { malloc(sizeof(HLI_EUCS_Entry) * 1024), 0, 1024 };

    for (size_t i = 0; i < total_count; i++) {
        uint32_t *t_items = malloc(sizeof(uint32_t) * data[i].count);
        double *t_utils = malloc(sizeof(double) * data[i].count);
        if (!t_items || !t_utils) {
            free(t_items);
            free(t_utils);
            continue;
        }
        size_t t_count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (rank[data[i].items[j].id] != 0xFFFFFFFF) {
                t_items[t_count] = data[i].items[j].id;
                t_utils[t_count] = data[i].items[j].utility;
                t_count++;
            }
        }
        // Sort by rank
        for (size_t j = 0; j < t_count; j++) {
            for (size_t k = j + 1; k < t_count; k++) {
                if (rank[t_items[j]] > rank[t_items[k]]) {
                    uint32_t tmp_i = t_items[j]; t_items[j] = t_items[k]; t_items[k] = tmp_i;
                    double tmp_u = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = tmp_u;
                }
            }
        }

        double remaining_utility = 0;
        for (size_t j = t_count; j-- > 0; ) {
            uint32_t item_id = t_items[j];
            HLI_UtilityList *ul = initial_ext[rank[item_id]];
            
            if (ul->count % 16 == 0 && ul->count > 0) {
                ul->tuples = realloc(ul->tuples, sizeof(HLI_Tuple) * (ul->count + 16));
            }
            ul->tuples[ul->count].tid = (uint32_t)i;
            ul->tuples[ul->count].iutil = t_utils[j];
            ul->tuples[ul->count].rutil = remaining_utility;
            ul->sum_iutil += t_utils[j];
            ul->sum_rutil += remaining_utility;
            ul->count++;

            for (size_t k = 0; k < j; k++) {
                uint32_t u = item_id, v = t_items[k];
                if (u > v) { uint32_t t = u; u = v; v = t; }
                bool found = false;
                for (size_t m = 0; m < eucs.count; m++) {
                    if (eucs.entries[m].u == u && eucs.entries[m].v == v) {
                        eucs.entries[m].twu += data[i].total_utility;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    if (eucs.count >= eucs.capacity) {
                        eucs.capacity *= 2;
                        eucs.entries = realloc(eucs.entries, sizeof(HLI_EUCS_Entry) * eucs.capacity);
                    }
                    eucs.entries[eucs.count].u = u;
                    eucs.entries[eucs.count].v = v;
                    eucs.entries[eucs.count].twu = data[i].total_utility;
                    eucs.count++;
                }
            }
            remaining_utility += t_utils[j];
        }
        free(t_items);
        free(t_utils);
    }

    // 4. Mining on Merged Lists
    printf("[HUI-LIST-INS] Starting Merged Mining...\n");
    search(NULL, initial_ext, i_star_count, min_util, &eucs);

    printf("[HUI-LIST-INS] Found %zu HUIs.\n", hui_count);

    // Cleanup
    for (size_t i = 0; i < i_star_count; i++) free_ul(initial_ext[i]);
    free(initial_ext);
    free(i_star);
    free(rank);
    free(twu);
    free(eucs.entries);

    dm_bench_record_results(hui_count, 0);
    return DM_SUCCESS;
}

static DM_Algorithm hli_algo = {
    .id = "hui_list_ins",
    .name = "HUI-list-INS Algorithm",
    .description = "Incremental HUI miner using merged utility-list structures.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(hli_algo)
