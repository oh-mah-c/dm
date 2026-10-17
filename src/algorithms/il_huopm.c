/**
 * @file il_huopm.c
 * @brief IL-HUOPM / HUOMIL: High Utility Occupancy Pattern Mining with Indexed List Structure
 *
 * Reference:
 *   Hyeonmo Kim, Taewoong Ryu, Chanhee Lee, Sinyoung Kim, Bay Vo, Jerry Chun-Wei Lin, and Unil Yun,
 *   "Efficient Method for Mining High Utility Occupancy Patterns based on Indexed List Structure,"
 *   IEEE Access, vol. 11, pp. 45053-45071, May 2023.
 *   DOI: 10.1109/ACCESS.2023.3271864
 */

#include "algorithms/il_huopm.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

/* --- DATA STRUCTURES --- */

/**
 * @brief Quadruple entry with next pointer and prefix tracking
 */
typedef struct {
    uint32_t tid;
    uint32_t next_item; /**< Index of target sibling list in current layer, or 0xFFFFFFFF */
    uint32_t next_idx;  /**< Index of entry in that target list, or 0xFFFFFFFF */
    double uo;          /**< Relative utility occupancy */
    double ruo;         /**< Remaining utility occupancy */
    double prefix_uo;   /**< Utility occupancy of the common prefix in this transaction */
} IL_Entry;

/**
 * @brief Indexed List (GUO-IL / CUO-IL)
 */
typedef struct {
    uint32_t *items;
    size_t count;
    IL_Entry *entries;
    size_t size;
    size_t capacity;
    double sum_uo;
    double sum_ruo;
} IL_List;

typedef struct {
    uint32_t id;
    size_t support;
} ItemSupport;

typedef struct {
    uint32_t child_id;
    uint32_t entry_idx;
} EntryRef;

/* --- HELPERS --- */

static uint32_t *rank_map = NULL;

static int cmp_item_supp_asc(const void *a, const void *b) {
    const ItemSupport *ia = (const ItemSupport *)a;
    const ItemSupport *ib = (const ItemSupport *)b;
    if (ia->support < ib->support) return -1;
    if (ia->support > ib->support) return 1;
    return (ia->id < ib->id) ? -1 : ((ia->id > ib->id) ? 1 : 0);
}

static int cmp_item_rank(const void *a, const void *b) {
    uint32_t id1 = ((const DM_Item *)a)->id;
    uint32_t id2 = ((const DM_Item *)b)->id;
    uint32_t r1 = rank_map[id1];
    uint32_t r2 = rank_map[id2];
    return (r1 < r2) ? -1 : ((r1 > r2) ? 1 : 0);
}

static int cmp_desc_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) ? -1 : ((da < db) ? 1 : 0);
}

static IL_List* create_il_list(const uint32_t *items, size_t count, size_t capacity) {
    IL_List *l = (IL_List *)malloc(sizeof(IL_List));
    if (!l) return NULL;
    l->count = count;
    l->items = (uint32_t *)malloc(sizeof(uint32_t) * (count ? count : 1));
    if (items && count > 0) {
        memcpy(l->items, items, sizeof(uint32_t) * count);
    }
    l->size = 0;
    l->capacity = capacity > 0 ? capacity : 4;
    l->entries = (IL_Entry *)malloc(sizeof(IL_Entry) * l->capacity);
    l->sum_uo = 0.0;
    l->sum_ruo = 0.0;
    return l;
}

static void free_il_list(IL_List *l) {
    if (l) {
        if (l->items) free(l->items);
        if (l->entries) free(l->entries);
        free(l);
    }
}

static inline void il_list_add(IL_List *l, const IL_Entry *e) {
    if (l->size >= l->capacity) {
        size_t new_cap = l->capacity == 0 ? 4 : l->capacity * 2;
        IL_Entry *new_entries = (IL_Entry *)realloc(l->entries, sizeof(IL_Entry) * new_cap);
        if (!new_entries) return;
        l->entries = new_entries;
        l->capacity = new_cap;
    }
    l->entries[l->size++] = *e;
    l->sum_uo += e->uo;
    l->sum_ruo += e->ruo;
}

/* --- UPPER BOUND (Definition 9 & Algorithm 4) --- */

static double upper_bound(const IL_List *l, size_t min_supp_count) {
    if (l->size < min_supp_count || min_supp_count == 0) return 0.0;

    if (l->size == min_supp_count) {
        double sum = 0.0;
        for (size_t i = 0; i < l->size; i++) {
            sum += (l->entries[i].uo + l->entries[i].ruo);
        }
        return sum / (double)min_supp_count;
    }

    double *v = (double *)malloc(sizeof(double) * l->size);
    if (!v) return 0.0;

    for (size_t i = 0; i < l->size; i++) {
        v[i] = l->entries[i].uo + l->entries[i].ruo;
    }

    qsort(v, l->size, sizeof(double), cmp_desc_double);

    double sum = 0.0;
    for (size_t i = 0; i < min_supp_count; i++) {
        sum += v[i];
    }
    free(v);
    return sum / (double)min_supp_count;
}

/* --- RECURSIVE MINING (Algorithm 3: Mine) --- */

static size_t total_huop_count = 0;
static size_t total_items_sum = 0;
static DM_PatternCallback g_il_huopm_cb = NULL;
static void *g_il_huopm_cb_data = NULL;
static const double *g_il_huopm_trans_utility = NULL;

static size_t il_huopm_visited_nodes = 0;

size_t il_huopm_get_visited_nodes(void) {
    return il_huopm_visited_nodes;
}

static void il_mine(IL_List **cuo_k, size_t cuo_k_count, double min_occupancy,
                    size_t min_supp_count, uint32_t max_id) {
    /* Temporary buffers for child generation */
    uint32_t *child_map = (uint32_t *)malloc(sizeof(uint32_t) * (max_id + 1));
    memset(child_map, 0xFF, sizeof(uint32_t) * (max_id + 1));

    size_t tx_cap = 256;
    EntryRef *tx_entries = (EntryRef *)malloc(sizeof(EntryRef) * tx_cap);

    for (size_t p = 0; p < cuo_k_count; p++) {
        il_huopm_visited_nodes++;
        IL_List *cp = cuo_k[p];

        /* Line 03: Support condition */
        if (cp->size < min_supp_count) continue;

        /* Line 04-06: Check if P is a HUOP */
        double occ = cp->sum_uo / (double)cp->size;
        if (occ >= min_occupancy) {
            total_huop_count++;
            total_items_sum += cp->count;
            if (g_il_huopm_cb) {
                double util = 0.0;
                if (g_il_huopm_trans_utility) {
                    for (size_t t = 0; t < cp->size; t++) {
                        util += cp->entries[t].uo * g_il_huopm_trans_utility[cp->entries[t].tid];
                    }
                }
                g_il_huopm_cb(cp->items, cp->count, cp->size, util, occ, g_il_huopm_cb_data);
            }
        }

        /* Line 07: Lemma 1 - Loose Upper Bound (lubuo) Pruning in O(1) */
        double lubuo = (cp->sum_uo + cp->sum_ruo) / (double)min_supp_count;
        if (lubuo < min_occupancy) continue;

        /* Line 08-09: Upper Bound (ubuo) Pruning */
        double ubuo = upper_bound(cp, min_supp_count);
        if (ubuo < min_occupancy) continue;

        /* Allocate dynamic child list array for CUOk+1 */
        size_t next_cuo_cap = 16;
        size_t next_cuo_count = 0;
        IL_List **next_cuo = (IL_List **)malloc(sizeof(IL_List *) * next_cuo_cap);

        /* Lines 11-24: Traverse entries of CUO(P) and follow index links */
        for (size_t e_idx = 0; e_idx < cp->size; e_idx++) {
            const IL_Entry *e = &cp->entries[e_idx];

            size_t tx_ext_count = 0;
            uint32_t curr_item = e->next_item;
            uint32_t curr_idx = e->next_idx;

            while (curr_item != 0xFFFFFFFF) {
                if (curr_item >= cuo_k_count) break;
                IL_List *curr_list = cuo_k[curr_item];
                if (curr_idx >= curr_list->size) break;
                const IL_Entry *ce = &curr_list->entries[curr_idx];
                uint32_t ext_item = curr_list->items[curr_list->count - 1];

                uint32_t child_id = child_map[ext_item];
                if (child_id == 0xFFFFFFFF) {
                    if (next_cuo_count >= next_cuo_cap) {
                        next_cuo_cap *= 2;
                        next_cuo = (IL_List **)realloc(next_cuo, sizeof(IL_List *) * next_cuo_cap);
                    }
                    child_id = (uint32_t)next_cuo_count++;
                    child_map[ext_item] = child_id;

                    /* Create new CUO list for (P U {ext_item}) */
                    uint32_t *new_items = (uint32_t *)malloc(sizeof(uint32_t) * (cp->count + 1));
                    memcpy(new_items, cp->items, sizeof(uint32_t) * cp->count);
                    new_items[cp->count] = ext_item;
                    next_cuo[child_id] = create_il_list(new_items, cp->count + 1, 8);
                    free(new_items);
                }

                IL_List *child_l = next_cuo[child_id];
                uint32_t new_entry_idx = (uint32_t)child_l->size;

                IL_Entry ne;
                ne.tid = e->tid;
                /* Line 19: ne.uo <- e.uo + ce.uo - pe.uo */
                ne.uo = e->uo + ce->uo - e->prefix_uo;
                /* Line 20: ne.ruo <- ce.ruo */
                ne.ruo = ce->ruo;
                /* Store prefix utility for next level combinations */
                ne.prefix_uo = e->uo;
                ne.next_item = 0xFFFFFFFF;
                ne.next_idx = 0xFFFFFFFF;
                il_list_add(child_l, &ne);

                if (tx_ext_count >= tx_cap) {
                    tx_cap *= 2;
                    tx_entries = (EntryRef *)realloc(tx_entries, sizeof(EntryRef) * tx_cap);
                }
                tx_entries[tx_ext_count].child_id = child_id;
                tx_entries[tx_ext_count].entry_idx = new_entry_idx;
                tx_ext_count++;

                curr_item = ce->next_item;
                curr_idx = ce->next_idx;
            }

            /* Line 22: Connect generated entries in this transaction */
            for (size_t c = 0; c + 1 < tx_ext_count; c++) {
                uint32_t cid = tx_entries[c].child_id;
                uint32_t cidx = tx_entries[c].entry_idx;
                next_cuo[cid]->entries[cidx].next_item = tx_entries[c + 1].child_id;
                next_cuo[cid]->entries[cidx].next_idx = tx_entries[c + 1].entry_idx;
            }
        }

        /* Reset child_map entries for reused lookup */
        for (size_t c = 0; c < next_cuo_count; c++) {
            uint32_t ext_item = next_cuo[c]->items[next_cuo[c]->count - 1];
            child_map[ext_item] = 0xFFFFFFFF;
        }

        /* Line 28-30: Recursively mine CUOk+1 */
        if (next_cuo_count > 0) {
            il_mine(next_cuo, next_cuo_count, min_occupancy, min_supp_count, max_id);
        }

        /* Free CUOk+1 lists */
        for (size_t c = 0; c < next_cuo_count; c++) {
            free_il_list(next_cuo[c]);
        }
        free(next_cuo);
    }

    free(tx_entries);
    free(child_map);
}

/* --- TOP-LEVEL EXECUTION (Algorithm 1: HUOMIL) --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;

    DM_IL_HUOPM_Params *p = (DM_IL_HUOPM_Params *)params;
    double min_occu = p ? p->min_occupancy : 0.5;
    double min_supp_arg = p ? p->min_support : 0.01;
    g_il_huopm_cb = p ? p->on_pattern : NULL;
    g_il_huopm_cb_data = p ? p->user_data : NULL;

    size_t min_supp_count = 1;
    if (min_supp_arg > 0.0 && min_supp_arg < 1.0) {
        min_supp_count = (size_t)ceil(min_supp_arg * (double)ds->count);
    } else if (min_supp_arg >= 1.0) {
        min_supp_count = (size_t)min_supp_arg;
    }
    if (min_supp_count < 1) min_supp_count = 1;

    total_huop_count = 0;
    total_items_sum = 0;

    DM_Trans_Utility *src_data = (DM_Trans_Utility *)ds->payload;
    uint32_t max_id = ds->max_id;

    /* Scan 1: Calculate item supports and transaction utilities */
    size_t *supp_counts = (size_t *)calloc(max_id + 1, sizeof(size_t));
    double *trans_utility = (double *)calloc(ds->count, sizeof(double));
    if (!supp_counts || !trans_utility) {
        free(supp_counts);
        free(trans_utility);
        return DM_ERROR_MEMORY;
    }
    g_il_huopm_trans_utility = trans_utility;

    for (size_t i = 0; i < ds->count; i++) {
        double tu = src_data[i].total_utility;
        if (tu <= 0.0) {
            for (size_t j = 0; j < src_data[i].count; j++) tu += src_data[i].items[j].utility;
        }
        trans_utility[i] = tu;

        for (size_t j = 0; j < src_data[i].count; j++) {
            supp_counts[src_data[i].items[j].id]++;
        }
    }

    /* Filter frequent items */
    size_t frequent_count = 0;
    for (uint32_t i = 0; i <= max_id; i++) {
        if (supp_counts[i] >= min_supp_count) {
            frequent_count++;
        }
    }

    if (frequent_count == 0) {
        free(supp_counts);
        free(trans_utility);
        dm_bench_record_results(0, 0);
        return DM_SUCCESS;
    }

    ItemSupport *items = (ItemSupport *)malloc(sizeof(ItemSupport) * frequent_count);
    size_t idx = 0;
    for (uint32_t i = 0; i <= max_id; i++) {
        if (supp_counts[i] >= min_supp_count) {
            items[idx].id = i;
            items[idx].support = supp_counts[i];
            idx++;
        }
    }

    /* Algorithm 1 Line 10: Sort items in support ascending order */
    qsort(items, frequent_count, sizeof(ItemSupport), cmp_item_supp_asc);
    il_huopm_visited_nodes = frequent_count;

    rank_map = (uint32_t *)malloc(sizeof(uint32_t) * (max_id + 1));
    memset(rank_map, 0xFF, sizeof(uint32_t) * (max_id + 1));
    for (size_t i = 0; i < frequent_count; i++) {
        rank_map[items[i].id] = (uint32_t)i;
    }

    /* Algorithm 2: Construct GUO-IL for all frequent 1-items */
    IL_List **guo = (IL_List **)malloc(sizeof(IL_List *) * frequent_count);
    for (size_t i = 0; i < frequent_count; i++) {
        uint32_t it = items[i].id;
        guo[i] = create_il_list(&it, 1, items[i].support);
    }

    DM_Item *reordered = (DM_Item *)malloc(sizeof(DM_Item) * (max_id + 1));

    for (size_t t = 0; t < ds->count; t++) {
        double tu = trans_utility[t];
        if (tu <= 0.0) continue;

        size_t m = 0;
        for (size_t j = 0; j < src_data[t].count; j++) {
            uint32_t id = src_data[t].items[j].id;
            if (rank_map[id] != 0xFFFFFFFF) {
                reordered[m++] = src_data[t].items[j];
            }
        }
        if (m == 0) continue;

        /* Sort in support ascending order */
        qsort(reordered, m, sizeof(DM_Item), cmp_item_rank);

        /* Traverse in reverse order (Algorithm 2 Lines 06-17) */
        double ruo = 0.0;
        uint32_t next_item = 0xFFFFFFFF;
        uint32_t next_idx = 0xFFFFFFFF;

        for (size_t j = m; j-- > 0; ) {
            uint32_t id = reordered[j].id;
            uint32_t r = rank_map[id];

            double uo = reordered[j].utility / tu;

            IL_Entry e;
            e.tid = (uint32_t)t;
            e.uo = uo;
            e.ruo = ruo;
            e.prefix_uo = 0.0;
            e.next_item = next_item;
            e.next_idx = next_idx;

            il_list_add(guo[r], &e);
            uint32_t cur_idx = (uint32_t)(guo[r]->size - 1);

            ruo += uo;
            next_item = r;
            next_idx = cur_idx;
        }
    }
    free(reordered);

    /* Algorithm 3: Mine */
    il_mine(guo, frequent_count, min_occu, min_supp_count, max_id);

    printf("[IL-HUOPM] Found %zu High-Utility Occupancy Patterns.\n", total_huop_count);

    /* Free memory */
    for (size_t i = 0; i < frequent_count; i++) {
        free_il_list(guo[i]);
    }
    free(guo);
    free(items);
    free(rank_map);
    rank_map = NULL;
    free(supp_counts);
    free(trans_utility);
    g_il_huopm_cb = NULL;
    g_il_huopm_cb_data = NULL;
    g_il_huopm_trans_utility = NULL;

    dm_bench_record_results(total_huop_count, total_items_sum);
    return DM_SUCCESS;
}

DM_Algorithm il_huopm_algo = {
    .id = "il_huopm",
    .name = "IL-HUOPM / HUOMIL Algorithm",
    .description = "High Utility Occupancy Pattern Mining with Indexed List Structure (Kim et al. IEEE Access 2023).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(il_huopm_algo)
