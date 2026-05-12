#include "algorithms/foshu.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    uint32_t tid;
    double iputil;
    double inutil;
    double rutil;
    double rtu; // Redefined Transaction Utility
} FOSHU_Tuple;

typedef struct {
    uint32_t *items;
    size_t count;
    FOSHU_Tuple *tuples;
    size_t tuple_count;
    size_t *period_start; // Array of size num_periods
    size_t *period_end;   // Array of size num_periods
    double sum_iputil;
    double sum_inutil;
    double sum_rutil;
} FOSHU_UtilityList;

typedef struct {
    uint32_t id;
    double global_twu;
    double *twu_per_period;
    bool is_positive;
} FOSHU_ItemInfo;

static uint64_t total_hui_count = 0;
static uint64_t total_items_sum = 0;

static FOSHU_UtilityList* create_utility_list(uint32_t *items, size_t count, size_t initial_cap, int num_periods) {
    FOSHU_UtilityList *ul = malloc(sizeof(FOSHU_UtilityList));
    ul->items = malloc(sizeof(uint32_t) * count);
    memcpy(ul->items, items, sizeof(uint32_t) * count);
    ul->count = count;
    ul->tuples = malloc(sizeof(FOSHU_Tuple) * initial_cap);
    ul->tuple_count = 0;
    ul->period_start = calloc(num_periods, sizeof(size_t));
    ul->period_end = calloc(num_periods, sizeof(size_t));
    ul->sum_iputil = 0;
    ul->sum_inutil = 0;
    ul->sum_rutil = 0;
    return ul;
}

static void free_utility_list(FOSHU_UtilityList *ul) {
    if (!ul) return;
    free(ul->items);
    free(ul->tuples);
    free(ul->period_start);
    free(ul->period_end);
    free(ul);
}

static FOSHU_UtilityList* construct(FOSHU_UtilityList *p, FOSHU_UtilityList *px, FOSHU_UtilityList *py, int num_periods) {
    size_t items_count = px->count + 1;
    uint32_t *new_items = malloc(sizeof(uint32_t) * items_count);
    memcpy(new_items, px->items, sizeof(uint32_t) * px->count);
    new_items[items_count - 1] = py->items[py->count - 1];

    size_t cap = (px->tuple_count < py->tuple_count) ? px->tuple_count : py->tuple_count;
    FOSHU_UtilityList *pxy = create_utility_list(new_items, items_count, cap, num_periods);
    free(new_items);

    for (int h = 0; h < num_periods; h++) {
        size_t ix = px->period_start[h];
        size_t iy = py->period_start[h];
        size_t ip = (p != NULL) ? p->period_start[h] : 0;

        pxy->period_start[h] = pxy->tuple_count;

        while (ix < px->period_end[h] && iy < py->period_end[h]) {
            if (px->tuples[ix].tid == py->tuples[iy].tid) {
                uint32_t tid = px->tuples[ix].tid;
                double iputil = px->tuples[ix].iputil + py->tuples[iy].iputil;
                double inutil = px->tuples[ix].inutil + py->tuples[iy].inutil;
                double rutil = py->tuples[iy].rutil;
                double rtu = py->tuples[iy].rtu;

                if (p != NULL) {
                    while (ip < p->period_end[h] && p->tuples[ip].tid < tid) ip++;
                    if (ip < p->period_end[h] && p->tuples[ip].tid == tid) {
                        iputil -= p->tuples[ip].iputil;
                        inutil -= p->tuples[ip].inutil;
                    }
                }

                pxy->tuples[pxy->tuple_count].tid = tid;
                pxy->tuples[pxy->tuple_count].iputil = iputil;
                pxy->tuples[pxy->tuple_count].inutil = inutil;
                pxy->tuples[pxy->tuple_count].rutil = rutil;
                pxy->tuples[pxy->tuple_count].rtu = rtu;

                pxy->sum_iputil += iputil;
                pxy->sum_inutil += inutil;
                pxy->sum_rutil += rutil;
                pxy->tuple_count++;
                ix++; iy++;
            } else if (px->tuples[ix].tid < py->tuples[iy].tid) {
                ix++;
            } else {
                iy++;
            }
        }
        pxy->period_end[h] = pxy->tuple_count;
    }

    return pxy;
}

static void search(FOSHU_UtilityList *p, FOSHU_UtilityList **extensions, size_t ext_count, double min_util, int num_periods, double *pto, bool *is_positive) {
    for (size_t i = 0; i < ext_count; i++) {
        FOSHU_UtilityList *px = extensions[i];

        // 1. Calculate to(Px) and ru(Px)
        double to_px = 0;
        double sum_ip_h[num_periods];
        double sum_r_h[num_periods];
        memset(sum_ip_h, 0, sizeof(sum_ip_h));
        memset(sum_r_h, 0, sizeof(sum_r_h));

        for (int h = 0; h < num_periods; h++) {
            if (px->period_end[h] > px->period_start[h]) {
                to_px += pto[h];
                for (size_t k = px->period_start[h]; k < px->period_end[h]; k++) {
                    sum_ip_h[h] += px->tuples[k].iputil;
                    sum_r_h[h] += px->tuples[k].rutil;
                }
            }
        }

        double util_px = px->sum_iputil + px->sum_inutil;
        if (util_px / to_px >= min_util) {
            total_hui_count++;
            total_items_sum += px->count;
        }

        // 2. Pruning Condition (Property 17)
        bool can_extend = false;
        for (int h = 0; h < num_periods; h++) {
            if (px->period_end[h] > px->period_start[h]) {
                if ((sum_ip_h[h] + sum_r_h[h]) / pto[h] >= min_util) {
                    can_extend = true;
                    break;
                }
            }
        }

        if (can_extend) {
            FOSHU_UtilityList **ext_px = malloc(sizeof(FOSHU_UtilityList*) * (ext_count - i - 1));
            size_t ext_px_count = 0;

            for (size_t j = i + 1; j < ext_count; j++) {
                FOSHU_UtilityList *py = extensions[j];
                FOSHU_UtilityList *pxy = construct(p, px, py, num_periods);

                // 3. TWU Pruning (Algorithm 2 line 12)
                bool potential_hui = false;
                for (int h = 0; h < num_periods; h++) {
                    if (pxy->period_end[h] > pxy->period_start[h]) {
                        double twu_pxy_h = 0;
                        for (size_t k = pxy->period_start[h]; k < pxy->period_end[h]; k++) {
                            twu_pxy_h += pxy->tuples[k].rtu;
                        }
                        if (twu_pxy_h / pto[h] >= min_util) {
                            potential_hui = true;
                            break;
                        }
                    }
                }

                if (potential_hui) {
                    ext_px[ext_px_count++] = pxy;
                } else {
                    free_utility_list(pxy);
                }
            }

            if (ext_px_count > 0) {
                search(px, ext_px, ext_px_count, min_util, num_periods, pto, is_positive);
            }

            for (size_t k = 0; k < ext_px_count; k++) free_utility_list(ext_px[k]);
            free(ext_px);
        }
    }
}

static int cmp_item_info(const void *a, const void *b) {
    const FOSHU_ItemInfo *ia = (const FOSHU_ItemInfo *)a;
    const FOSHU_ItemInfo *ib = (const FOSHU_ItemInfo *)b;
    if (ia->is_positive && !ib->is_positive) return -1;
    if (!ia->is_positive && ib->is_positive) return 1;
    if (ia->global_twu < ib->global_twu) return -1;
    if (ia->global_twu > ib->global_twu) return 1;
    return (int)ia->id - (int)ib->id;
}

static DM_Status run_foshu(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_FOSHU_Params *p = (DM_FOSHU_Params *)params;
    double min_util = p ? p->min_utility : 0.43;
    int num_periods = (p && p->num_periods > 0) ? p->num_periods : 1;

    printf("Executing FOSHU Algorithm (minutil=%.4f, periods=%d)...\n", min_util, num_periods);

    total_hui_count = 0;
    total_items_sum = 0;

    // Phase 1: Global TWU, Period TWU, and pto(h)
    printf("[FOSHU] Starting Phase 1 (TWU and Ordering)...\n");
    double *pto = calloc(num_periods, sizeof(double));
    double **twu_per_period = malloc(sizeof(double*) * (ds->max_id + 1));
    for (uint32_t i = 0; i <= ds->max_id; i++) twu_per_period[i] = calloc(num_periods, sizeof(double));

    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    int *tid_period = malloc(sizeof(int) * ds->count);

    for (size_t i = 0; i < ds->count; i++) {
        int h = (int)(i * num_periods / ds->count);
        if (h >= num_periods) h = num_periods - 1;
        tid_period[i] = h;

        double rtu = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (data[i].items[j].utility > 0) rtu += data[i].items[j].utility;
        }
        pto[h] += rtu;

        for (size_t j = 0; j < data[i].count; j++) {
            twu_per_period[data[i].items[j].id][h] += rtu;
        }
    }

    bool *is_positive = calloc(ds->max_id + 1, sizeof(bool));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            if (data[i].items[j].utility > 0) is_positive[data[i].items[j].id] = true;
        }
    }

    FOSHU_ItemInfo *items = malloc(sizeof(FOSHU_ItemInfo) * (ds->max_id + 1));
    size_t item_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        double global_twu = 0;
        bool potential = false;
        for (int h = 0; h < num_periods; h++) {
            global_twu += twu_per_period[i][h];
            if (pto[h] > 0 && twu_per_period[i][h] / pto[h] >= min_util) potential = true;
        }

        if (potential) {
            items[item_count].id = i;
            items[item_count].global_twu = global_twu;
            items[item_count].twu_per_period = twu_per_period[i];
            items[item_count].is_positive = is_positive[i];
            item_count++;
        } else {
            free(twu_per_period[i]);
        }
    }
    free(twu_per_period);

    qsort(items, item_count, sizeof(FOSHU_ItemInfo), cmp_item_info);

    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < item_count; i++) rank[items[i].id] = i;

    // Phase 2: Building Utility-Lists
    printf("[FOSHU] Starting Phase 2 (Building Structures)...\n");
    FOSHU_UtilityList **initial_ext = malloc(sizeof(FOSHU_UtilityList*) * item_count);
    for (size_t i = 0; i < item_count; i++) {
        initial_ext[i] = create_utility_list(&items[i].id, 1, 10, num_periods);
    }

    size_t *ul_caps = calloc(item_count, sizeof(size_t));
    for (size_t i = 0; i < item_count; i++) ul_caps[i] = 10;

    for (size_t i = 0; i < ds->count; i++) {
        int h = tid_period[i];
        
        // Redefined TU (RTU)
        double rtu = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (data[i].items[j].utility > 0) rtu += data[i].items[j].utility;
        }

        // Filter and sort items in transaction
        uint32_t t_items[data[i].count];
        double t_utils[data[i].count];
        size_t t_count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (rank[data[i].items[j].id] != 0xFFFFFFFF) {
                t_items[t_count] = data[i].items[j].id;
                t_utils[t_count] = data[i].items[j].utility;
                t_count++;
            }
        }

        for (size_t j = 0; j < t_count; j++) {
            for (size_t k = j + 1; k < t_count; k++) {
                if (rank[t_items[j]] > rank[t_items[k]]) {
                    uint32_t temp_i = t_items[j]; t_items[j] = t_items[k]; t_items[k] = temp_i;
                    double temp_u = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = temp_u;
                }
            }
        }

        double remaining_positive_utility = 0;
        for (size_t j = t_count; j-- > 0; ) {
            uint32_t item_id = t_items[j];
            uint32_t r = rank[item_id];
            FOSHU_UtilityList *ul = initial_ext[r];

            if (ul->tuple_count >= ul_caps[r]) {
                ul_caps[r] *= 2;
                ul->tuples = realloc(ul->tuples, sizeof(FOSHU_Tuple) * ul_caps[r]);
            }

            ul->tuples[ul->tuple_count].tid = (uint32_t)i;
            if (t_utils[j] > 0) {
                ul->tuples[ul->tuple_count].iputil = t_utils[j];
                ul->tuples[ul->tuple_count].inutil = 0;
                ul->sum_iputil += t_utils[j];
            } else {
                ul->tuples[ul->tuple_count].iputil = 0;
                ul->tuples[ul->tuple_count].inutil = t_utils[j];
                ul->sum_inutil += t_utils[j];
            }
            ul->tuples[ul->tuple_count].rutil = remaining_positive_utility;
            ul->tuples[ul->tuple_count].rtu = rtu;
            ul->sum_rutil += remaining_positive_utility;
            
            // Period grouping
            if (ul->tuple_count == 0) {
                 ul->period_start[h] = 0;
            } else if (tid_period[ul->tuples[ul->tuple_count-1].tid] != h) {
                 ul->period_start[h] = ul->tuple_count;
            }
            
            ul->tuple_count++;
            ul->period_end[h] = ul->tuple_count;

            if (t_utils[j] > 0) remaining_positive_utility += t_utils[j];
        }
    }
    free(ul_caps);
    free(tid_period);

    // Phase 3: Search
    printf("[FOSHU] Starting Recursive Search...\n");
    search(NULL, initial_ext, item_count, min_util, num_periods, pto, is_positive);

    printf("[FOSHU] Found %llu High Utility On-Shelf Itemsets.\n", total_hui_count);

    // Cleanup
    for (size_t i = 0; i < item_count; i++) {
        free(items[i].twu_per_period);
        free_utility_list(initial_ext[i]);
    }
    free(items);
    free(initial_ext);
    free(rank);
    free(is_positive);
    free(pto);

    dm_bench_record_results(total_hui_count, total_items_sum);
    return DM_SUCCESS;
}

static DM_Algorithm foshu_algo = {
    .id = "foshu",
    .name = "FOSHU Algorithm",
    .description = "Faster On-Shelf High Utility Itemset Mining.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run_foshu
};

DM_REGISTER_ALGORITHM(foshu_algo)
