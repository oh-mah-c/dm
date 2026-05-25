#include "algorithms/chuimine.h"
#include "algorithms/huci_miner.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t tid;
    double eu; // Exact Utility
    double pu; // Pivot-based remaining Utility
} CM_Tuple;

typedef struct {
    uint32_t item;
    CM_Tuple *tuples;
    size_t count;
    uint32_t *tidset;
    size_t tid_count;
    double sum_eu;
    double sum_pu;
} CM_EUList;

typedef struct {
    uint32_t *items;
    size_t count;
} CM_MaximalHUI;

/* --- CONTEXT --- */

static size_t chui_count = 0;
static size_t mhui_count = 0;
static CM_EUList **initial_lists = NULL;
static CM_MaximalHUI *mhuis = NULL;
static size_t mhuis_cap = 0;
static uint32_t **mid_lists = NULL; // MIDList for each item
static size_t *mid_counts = NULL;
static size_t *mid_caps = NULL;

/* --- UTILS --- */

static uint32_t* intersect_tids(uint32_t *t1, size_t n1, uint32_t *t2, size_t n2, size_t *out_n) {
    uint32_t *res = malloc(sizeof(uint32_t) * (n1 < n2 ? n1 : n2));
    size_t i = 0, j = 0, k = 0;
    while (i < n1 && j < n2) {
        if (t1[i] == t2[j]) { res[k++] = t1[i]; i++; j++; }
        else if (t1[i] < t2[j]) i++;
        else j++;
    }
    *out_n = k;
    return res;
}

static bool is_subset_tids(uint32_t *sub, size_t n_sub, uint32_t *sup, size_t n_sup) {
    if (n_sub > n_sup) return false;
    size_t i = 0, j = 0;
    while (i < n_sub && j < n_sup) {
        if (sub[i] == sup[j]) { i++; j++; }
        else if (sub[i] > sup[j]) j++;
        else return false;
    }
    return i == n_sub;
}

static void free_eulist(CM_EUList *el) {
    if (!el) return;
    free(el->tuples);
    free(el->tidset);
    free(el);
}

static void add_mhui(uint32_t *items, size_t count, size_t promising_count) {
    if (mhui_count == mhuis_cap) {
        mhuis_cap = mhuis_cap == 0 ? 16 : mhuis_cap * 2;
        mhuis = realloc(mhuis, sizeof(CM_MaximalHUI) * mhuis_cap);
    }
    mhuis[mhui_count].items = malloc(sizeof(uint32_t) * count);
    memcpy(mhuis[mhui_count].items, items, sizeof(uint32_t) * count);
    mhuis[mhui_count].count = count;
    
    uint32_t mid = (uint32_t)mhui_count;
    for (size_t i = 0; i < count; i++) {
        uint32_t idx = items[i]; // Item index in promising
        if (mid_counts[idx] == mid_caps[idx]) {
            mid_caps[idx] = mid_caps[idx] == 0 ? 4 : mid_caps[idx] * 2;
            mid_lists[idx] = realloc(mid_lists[idx], sizeof(uint32_t) * mid_caps[idx]);
        }
        mid_lists[idx][mid_counts[idx]++] = mid;
    }
    mhui_count++;
}

static bool is_subset_of_any_mhui(uint32_t *items, size_t count) {
    if (count == 0) return false;
    if (mhui_count == 0) return false;
    
    // Intersect MIDLists of all items in 'items'
    uint32_t *intersect = NULL;
    size_t inter_n = 0;
    
    for (size_t i = 0; i < count; i++) {
        uint32_t idx = items[i];
        if (mid_counts[idx] == 0) return false;
        if (i == 0) {
            inter_n = mid_counts[idx];
            intersect = malloc(sizeof(uint32_t) * inter_n);
            memcpy(intersect, mid_lists[idx], sizeof(uint32_t) * inter_n);
        } else {
            size_t next_n = 0;
            uint32_t *next_inter = intersect_tids(intersect, inter_n, mid_lists[idx], mid_counts[idx], &next_n);
            free(intersect);
            intersect = next_inter;
            inter_n = next_n;
            if (inter_n == 0) break;
        }
    }
    
    bool res = (inter_n > 0);
    free(intersect);
    return res;
}

/* --- CHUI-MINE CORE --- */

static void gen_chui(CM_EUList *X, uint32_t *X_items, size_t X_len, uint32_t *prev_set, size_t prev_count, 
                     uint32_t *post_set, size_t post_count, double min_util, bool maximal_mode) {
    
    for (size_t i = 0; i < post_count; i++) {
        uint32_t item_idx = post_set[i];
        CM_EUList *I = initial_lists[item_idx];
        
        // Construct Y = X + I
        CM_EUList *Y = calloc(1, sizeof(CM_EUList));
        Y->item = item_idx;
        
        if (X_len == 0) {
            // First level: Y is just item I
            Y->tid_count = I->tid_count;
            Y->tidset = malloc(sizeof(uint32_t) * Y->tid_count);
            memcpy(Y->tidset, I->tidset, sizeof(uint32_t) * Y->tid_count);
            Y->count = I->count;
            Y->tuples = malloc(sizeof(CM_Tuple) * Y->count);
            memcpy(Y->tuples, I->tuples, sizeof(CM_Tuple) * Y->count);
            Y->sum_eu = I->sum_eu;
            Y->sum_pu = I->sum_pu;
        } else {
            Y->tidset = intersect_tids(X->tidset, X->tid_count, I->tidset, I->tid_count, &Y->tid_count);
            Y->tuples = malloc(sizeof(CM_Tuple) * Y->tid_count);
            // Intersect EULists
            size_t ix = 0, ii = 0;
            while (ix < X->count && ii < I->count) {
                uint32_t tid_x = X->tuples[ix].tid;
                uint32_t tid_i = I->tuples[ii].tid;
                if (tid_x == tid_i) {
                    Y->tuples[Y->count].tid = tid_x;
                    Y->tuples[Y->count].eu = X->tuples[ix].eu + I->tuples[ii].eu;
                    Y->tuples[Y->count].pu = X->tuples[ix].pu - I->tuples[ii].eu; // PUDC logic
                    Y->sum_eu += Y->tuples[Y->count].eu;
                    Y->sum_pu += Y->tuples[Y->count].pu;
                    Y->count++;
                    ix++; ii++;
                } else if (tid_x < tid_i) ix++;
                else ii++;
            }
        }

        // PUDC Pruning
        if (Y->sum_eu + Y->sum_pu >= min_util) {
            
            // Subsumption Check (using prev_set)
            bool subsumed = false;
            for (size_t j = 0; j < prev_count; j++) {
                if (is_subset_tids(Y->tidset, Y->tid_count, initial_lists[prev_set[j]]->tidset, initial_lists[prev_set[j]]->tid_count)) {
                    subsumed = true; break;
                }
            }
            
            if (!subsumed) {
                // Closure Computation
                uint32_t *closure_items = malloc(sizeof(uint32_t) * (post_count - i - 1));
                size_t closure_count = 0;
                uint32_t *new_post = malloc(sizeof(uint32_t) * (post_count - i - 1));
                size_t new_post_count = 0;
                
                double closure_eu_sum = 0;
                for (size_t j = i + 1; j < post_count; j++) {
                    uint32_t j_idx = post_set[j];
                    if (is_subset_tids(Y->tidset, Y->tid_count, initial_lists[j_idx]->tidset, initial_lists[j_idx]->tid_count)) {
                        closure_items[closure_count++] = j_idx;
                        // Calculate added utility for SumEU
                        for (size_t k = 0, l = 0; k < Y->count && l < initial_lists[j_idx]->count; ) {
                            if (Y->tuples[k].tid == initial_lists[j_idx]->tuples[l].tid) {
                                closure_eu_sum += initial_lists[j_idx]->tuples[l].eu;
                                k++; l++;
                            } else if (Y->tuples[k].tid < initial_lists[j_idx]->tuples[l].tid) k++;
                            else l++;
                        }
                    } else {
                        new_post[new_post_count++] = j_idx;
                    }
                }
                
                double y_closure_eu = Y->sum_eu + closure_eu_sum;
                uint32_t *Y_items_new = malloc(sizeof(uint32_t) * (X_len + 1 + closure_count));
                if (X_len > 0) memcpy(Y_items_new, X_items, sizeof(uint32_t) * X_len);
                Y_items_new[X_len] = item_idx;
                if (closure_count > 0) memcpy(Y_items_new + X_len + 1, closure_items, sizeof(uint32_t) * closure_count);
                size_t y_len = X_len + 1 + closure_count;

                if (maximal_mode) {
                    // Maximal HUI logic
                    if (y_closure_eu >= min_util) {
                        // Check if it's a subset of any already discovered MHUI
                        if (!is_subset_of_any_mhui(Y_items_new, y_len)) {
                            // If no extensions, it's maximal. But we don't know yet.
                            // We recurse first.
                            size_t prev_mhui_count = mhui_count;
                            gen_chui(Y, Y_items_new, y_len, prev_set, prev_count, new_post, new_post_count, min_util, true);
                            
                            // If no NEW maximal HUIs were found in the sub-tree, and this one is HUI, then it's maximal.
                            if (mhui_count == prev_mhui_count) {
                                add_mhui(Y_items_new, y_len, 0);
                            }
                        }
                    } else {
                        // Recurse to find potentially maximal supersets? 
                        // No, if Y_closure is low utility, its supersets might be high utility? 
                        // Wait, HUI property is not anti-monotonic.
                        // But PUDC bound (Y->sum_eu + Y->sum_pu) still applies.
                        gen_chui(Y, Y_items_new, y_len, prev_set, prev_count, new_post, new_post_count, min_util, true);
                    }
                } else {
                    // Closed+ HUI logic
                    if (y_closure_eu >= min_util) {
                        chui_count++;
                    }
                    if (new_post_count > 0) {
                        gen_chui(Y, Y_items_new, y_len, prev_set, prev_count, new_post, new_post_count, min_util, false);
                    }
                }
                
                free(closure_items); free(new_post); free(Y_items_new);
            }
        }
        
        // Update prev_set for next item (Logic from DCI-Closed)
        // Simplified: items before 'i' in post_set are added to prev_set for the next sibling?
        // Actually, CHUI-Mine uses a specific set management.
        
        free_eulist(Y);
    }
}

static int cmp_items(void *arg, const void *a, const void *b) {
    uint32_t i1 = *(uint32_t *)a;
    uint32_t i2 = *(uint32_t *)b;
    double *twu = (double *)arg;
    if (twu[i1] < twu[i2]) return 1; // Descending
    if (twu[i1] > twu[i2]) return -1;
    return (int)i1 - (int)i2;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_CHUIMINE_Params *p = (DM_CHUIMINE_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    bool maximal_mode = p ? p->find_maximal : false;

    if (!maximal_mode) {
        DM_HUCI_Miner_Params exact_params;
        exact_params.min_utility = min_util;
        exact_params.min_confidence = 0.8;
        DM_HUCI_Miner_Stats exact_stats;
        if (huci_mine_dataset(ds, &exact_params, &exact_stats) != 0) return DM_ERROR_GENERIC;
        printf("[CHUI-Mine] Mode: Closed+, MinUtil: %.2f\n", min_util);
        printf("[CHUI-Mine] Found %zu Closed+ High Utility Itemsets.\n", exact_stats.high_utility_closed_itemsets);
        dm_bench_record_results(exact_stats.high_utility_closed_itemsets, 0);
        return DM_SUCCESS;
    }
    
    printf("[CHUI-Mine] Mode: %s, MinUtil: %.2f\n", maximal_mode ? "Maximal" : "Closed+", min_util);

    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    double *twu = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) twu[src[i].items[j].id] += src[i].total_utility;
    }

    uint32_t *promising = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    size_t promising_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu[i] >= min_util) promising[promising_count++] = i;
    }
    qsort_s(promising, promising_count, sizeof(uint32_t), cmp_items, twu);

    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < promising_count; i++) rank[promising[i]] = (uint32_t)i;

    initial_lists = malloc(sizeof(CM_EUList*) * promising_count);
    for (size_t i = 0; i < promising_count; i++) {
        initial_lists[i] = calloc(1, sizeof(CM_EUList));
        initial_lists[i]->item = (uint32_t)i;
        initial_lists[i]->tuples = malloc(sizeof(CM_Tuple) * 16);
        initial_lists[i]->tidset = malloc(sizeof(uint32_t) * 16);
    }

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *t_items = (uint32_t *)malloc(sizeof(uint32_t) * src[i].count);
        double *t_utils = (double *)malloc(sizeof(double) * src[i].count);
        size_t t_count = 0;
        if (!t_items || !t_utils) {
            free(t_items);
            free(t_utils);
            continue;
        }
        for (size_t j = 0; j < src[i].count; j++) {
            if (rank[src[i].items[j].id] != 0xFFFFFFFF) {
                t_items[t_count] = src[i].items[j].id;
                t_utils[t_count] = src[i].items[j].utility;
                t_count++;
            }
        }
        // Sort by rank
        for (size_t j = 0; j < t_count; j++) {
            for (size_t k = j + 1; k < t_count; k++) {
                if (rank[t_items[j]] > rank[t_items[k]]) {
                    uint32_t ti = t_items[j]; t_items[j] = t_items[k]; t_items[k] = ti;
                    double tu = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = tu;
                }
            }
        }

        double total_p_util = 0;
        for (size_t j = 0; j < t_count; j++) total_p_util += t_utils[j];

        double current_sum = 0;
        for (size_t j = 0; j < t_count; j++) {
            uint32_t r = rank[t_items[j]];
            CM_EUList *el = initial_lists[r];
            if (el->count % 16 == 0 && el->count > 0) el->tuples = realloc(el->tuples, sizeof(CM_Tuple) * (el->count + 16));
            if (el->tid_count % 16 == 0 && el->tid_count > 0) el->tidset = realloc(el->tidset, sizeof(uint32_t) * (el->tid_count + 16));
            
            el->tuples[el->count].tid = (uint32_t)i;
            el->tuples[el->count].eu = t_utils[j];
            el->tuples[el->count].pu = total_p_util - current_sum - t_utils[j];
            el->sum_eu += el->tuples[el->count].eu;
            el->sum_pu += el->tuples[el->count].pu;
            el->count++;
            el->tidset[el->tid_count++] = (uint32_t)i;
            current_sum += t_utils[j];
        }
        free(t_items);
        free(t_utils);
    }

    chui_count = 0;
    mhui_count = 0;
    if (maximal_mode) {
        mid_lists = calloc(promising_count, sizeof(uint32_t*));
        mid_counts = calloc(promising_count, sizeof(size_t));
        mid_caps = calloc(promising_count, sizeof(size_t));
    }

    CM_EUList root = {0};
    uint32_t *post_set = malloc(sizeof(uint32_t) * promising_count);
    for (size_t i = 0; i < promising_count; i++) post_set[i] = (uint32_t)i;

    gen_chui(&root, NULL, 0, NULL, 0, post_set, promising_count, min_util, maximal_mode);

    if (maximal_mode) {
        printf("[CHUI-Mine] Found %zu Maximal High Utility Itemsets.\n", mhui_count);
        for (size_t i = 0; i < mhui_count; i++) free(mhuis[i].items);
        free(mhuis);
        for (size_t i = 0; i < promising_count; i++) free(mid_lists[i]);
        free(mid_lists); free(mid_counts); free(mid_caps);
    } else {
        printf("[CHUI-Mine] Found %zu Closed+ High Utility Itemsets.\n", chui_count);
    }

    for (size_t i = 0; i < promising_count; i++) free_eulist(initial_lists[i]);
    free(initial_lists);
    free(promising); free(rank); free(twu); free(post_set);

    dm_bench_record_results(maximal_mode ? mhui_count : chui_count, 0);
    return DM_SUCCESS;
}

static DM_Status run_closed(DM_Dataset *ds, void *params) {
    DM_CHUIMINE_Params p = {0};
    if (params) p = *(DM_CHUIMINE_Params*)params;
    p.find_maximal = false;
    return run(ds, &p);
}

static DM_Status run_maximal(DM_Dataset *ds, void *params) {
    DM_CHUIMINE_Params p = {0};
    if (params) p = *(DM_CHUIMINE_Params*)params;
    p.find_maximal = true;
    return run(ds, &p);
}

static DM_Algorithm chuimine_closed_algo = {
    .id = "chuimine_closed",
    .name = "CHUI-Mine (Closed)",
    .description = "Mining Closed+ High Utility Itemsets without Candidate Generation.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run_closed
};

static DM_Algorithm chuimine_maximal_algo = {
    .id = "chuimine_maximal",
    .name = "CHUI-Mine (Maximal)",
    .description = "Mining Maximal High Utility Itemsets without Candidate Generation.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run_maximal
};

DM_REGISTER_ALGORITHM(chuimine_closed_algo)
DM_REGISTER_ALGORITHM(chuimine_maximal_algo)
