#include "algorithms/fchm.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include "core/dm_bitset.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

/* 
 * FCHM (Fast Correlated High-Utility Itemset Miner)
 * Reference: Fournier-Viger, P., et al. "Mining Correlated High-Utility Itemsets using the Bond Measure", HAIS 2016.
 */

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t tid;
    double iutil;
    double rutil;
} UtilityTuple;

typedef struct {
    uint32_t *items;
    size_t count;
    UtilityTuple *tuples;
    size_t tuple_count;
    double sum_iutil;
    double sum_rutil;
    DM_BitSet *bv; // Disjunctive Bit Vector
} UtilityList;

typedef struct {
    double *matrix;
    size_t size;
} Matrix; // Used for EUCS and Bond Matrix

typedef struct {
    uint32_t id;
    double twu;
    uint32_t support;
} ItemInfo;

/* --- UTILS --- */

static int cmp_item_twu(const void *a, const void *b) {
    double twu1 = ((ItemInfo*)a)->twu;
    double twu2 = ((ItemInfo*)b)->twu;
    if (twu1 < twu2) return -1;
    if (twu1 > twu2) return 1;
    return (int)(((ItemInfo*)a)->id - ((ItemInfo*)b)->id);
}

static void matrix_set(Matrix *m, uint32_t u, uint32_t v, double val) {
    if (u >= m->size || v >= m->size) return;
    m->matrix[u * m->size + v] = val;
    m->matrix[v * m->size + u] = val;
}

static double matrix_get(Matrix *m, uint32_t u, uint32_t v) {
    if (u >= m->size || v >= m->size) return 0;
    return m->matrix[u * m->size + v];
}

/* --- UTILITY LIST JOIN (Algorithm 3 with AUL Strategy 4) --- */

static UtilityList* construct(UtilityList *p, UtilityList *px, UtilityList *py, double min_bond) {
    // Strategy 4: AUL (Abandoning Utility-List construction early)
    // pxy.bv = px.bv | py.bv
    DM_BitSet *pxy_bv = dm_bitset_copy(px->bv);
    dm_bitset_or(pxy_bv, py->bv);
    size_t dissup_pxy = dm_bitset_count(pxy_bv);
    
    // Property 6: Required conjunctive support
    size_t lower_bound = (size_t)ceil(dissup_pxy * min_bond);
    size_t max_support = px->tuple_count;

    if (max_support < lower_bound) {
        dm_bitset_free(pxy_bv);
        return NULL;
    }

    UtilityList *pxy = malloc(sizeof(UtilityList));
    pxy->count = px->count + 1;
    pxy->items = malloc(sizeof(uint32_t) * pxy->count);
    memcpy(pxy->items, px->items, sizeof(uint32_t) * px->count);
    pxy->items[px->count] = py->items[py->count - 1];
    
    size_t capacity = px->tuple_count < py->tuple_count ? px->tuple_count : py->tuple_count;
    pxy->tuples = malloc(sizeof(UtilityTuple) * capacity);
    pxy->tuple_count = 0;
    pxy->sum_iutil = 0;
    pxy->sum_rutil = 0;
    pxy->bv = pxy_bv;

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < px->tuple_count && iy < py->tuple_count) {
        if (px->tuples[ix].tid == py->tuples[iy].tid) {
            uint32_t tid = px->tuples[ix].tid;
            double iutil = px->tuples[ix].iutil + py->tuples[iy].iutil;
            if (p != NULL) {
                while (ip < p->tuple_count && p->tuples[ip].tid < tid) ip++;
                if (ip < p->tuple_count && p->tuples[ip].tid == tid) {
                    iutil -= p->tuples[ip].iutil;
                }
            }
            pxy->tuples[pxy->tuple_count].tid = tid;
            pxy->tuples[pxy->tuple_count].iutil = iutil;
            pxy->tuples[pxy->tuple_count].rutil = py->tuples[iy].rutil;
            pxy->sum_iutil += iutil;
            pxy->sum_rutil += py->tuples[iy].rutil;
            pxy->tuple_count++;
            ix++; iy++;
        } else {
            if (px->tuples[ix].tid < py->tuples[iy].tid) {
                ix++;
                max_support--;
            } else {
                iy++;
            }
            // Check Strategy 4 early abandonment
            if (max_support < lower_bound) {
                // Not enough potential support to meet bond requirement
                free(pxy->items);
                free(pxy->tuples);
                dm_bitset_free(pxy->bv);
                free(pxy);
                return NULL;
            }
        }
    }

    // Final check for bond
    if (pxy->tuple_count < lower_bound) {
        free(pxy->items);
        free(pxy->tuples);
        dm_bitset_free(pxy->bv);
        free(pxy);
        return NULL;
    }

    return pxy;
}

static void free_utility_list(UtilityList *ul) {
    if (!ul) return;
    free(ul->items);
    free(ul->tuples);
    if (ul->bv) dm_bitset_free(ul->bv);
    free(ul);
}

/* --- RECURSIVE SEARCH (Algorithm 2) --- */

static size_t total_chi_count = 0;
static size_t total_items_sum = 0;

static void search(UtilityList *p, UtilityList **extensions, size_t ext_count, double min_util, double min_bond, Matrix *eucs, Matrix *bond_matrix, uint32_t *rank) {
    for (size_t i = 0; i < ext_count; i++) {
        UtilityList *px = extensions[i];
        
        // Strategy 2 (PSN): if we are here, px is already known to be correlated (or is a single item)
        // Check utility
        if (px->sum_iutil >= min_util) {
            total_chi_count++;
            total_items_sum += px->count;
        }

        if (px->sum_iutil + px->sum_rutil >= min_util) {
            UtilityList **ext_px = malloc(sizeof(UtilityList*) * (ext_count - i - 1));
            size_t ext_px_count = 0;

            uint32_t x_id = px->items[px->count - 1];
            uint32_t r_x = rank[x_id];
            
            for (size_t j = i + 1; j < ext_count; j++) {
                UtilityList *py = extensions[j];
                uint32_t y_id = py->items[py->count - 1];
                uint32_t r_y = rank[y_id];
                
                // Strategy 3: PBM (Pruning using the Bond Matrix)
                if (matrix_get(bond_matrix, r_x, r_y) < min_bond) continue;
                
                // EUCS check
                if (matrix_get(eucs, r_x, r_y) < min_util) continue;
                
                // Construct Pxy (includes Strategy 4: AUL)
                UtilityList *pxy = construct(p, px, py, min_bond);
                if (pxy != NULL) {
                    // Property 5: Calculate bond and Strategy 2: PSN
                    size_t consup = pxy->tuple_count;
                    size_t dissup = dm_bitset_count(pxy->bv);
                    double bond = (double)consup / dissup;
                    
                    if (bond >= min_bond) {
                        ext_px[ext_px_count++] = pxy;
                    } else {
                        free_utility_list(pxy);
                    }
                }
            }

            if (ext_px_count > 0) {
                search(px, ext_px, ext_px_count, min_util, min_bond, eucs, bond_matrix, rank);
                for (size_t j = 0; j < ext_px_count; j++) free_utility_list(ext_px[j]);
            }
            free(ext_px);
        }
    }
}

/* --- MAIN RUN (Algorithm 1) --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_FCHM_Params *p = (DM_FCHM_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    double min_bond = p ? p->min_bond : 0.2;
    
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    total_chi_count = 0;
    total_items_sum = 0;

    printf("[FCHM] Starting Phase 1 (TWU and Ordering)...\n");

    double *twu_counts = calloc(ds->max_id + 1, sizeof(double));
    uint32_t *supports = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t id = data[i].items[j].id;
            twu_counts[id] += data[i].total_utility;
            supports[id]++;
        }
    }

    ItemInfo *items = malloc(sizeof(ItemInfo) * (ds->max_id + 1));
    size_t item_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu_counts[i] >= min_util) {
            items[item_count].id = i;
            items[item_count].twu = twu_counts[i];
            items[item_count].support = supports[i];
            item_count++;
        }
    }
    // Step 3: Establishment of total order (TWU ascending)
    qsort(items, item_count, sizeof(ItemInfo), cmp_item_twu);

    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < item_count; i++) rank[items[i].id] = (uint32_t)i;

    printf("[FCHM] Starting Phase 2 (Building Structures)...\n");

    UtilityList **initial_ext = malloc(sizeof(UtilityList*) * item_count);
    for (size_t i = 0; i < item_count; i++) {
        initial_ext[i] = malloc(sizeof(UtilityList));
        initial_ext[i]->items = malloc(sizeof(uint32_t));
        initial_ext[i]->items[0] = items[i].id;
        initial_ext[i]->count = 1;
        initial_ext[i]->tuples = malloc(sizeof(UtilityTuple) * 8);
        initial_ext[i]->tuple_count = 0;
        initial_ext[i]->sum_iutil = 0;
        initial_ext[i]->sum_rutil = 0;
        initial_ext[i]->bv = dm_bitset_create(ds->count);
    }

    Matrix eucs;
    eucs.size = item_count;
    eucs.matrix = calloc(eucs.size * eucs.size, sizeof(double));

    // For Bond Matrix pre-calculation
    uint32_t *pair_consup = calloc(eucs.size * eucs.size, sizeof(uint32_t));

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *t_items = malloc(sizeof(uint32_t) * data[i].count);
        double *t_utils = malloc(sizeof(double) * data[i].count);
        size_t t_count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t id = data[i].items[j].id;
            uint32_t r = rank[id];
            if (r != 0xFFFFFFFF) {
                t_items[t_count] = id;
                t_utils[t_count] = data[i].items[j].utility;
                dm_bitset_set(initial_ext[r]->bv, i);
                t_count++;
            }
        }
        
        // Re-order by rank
        for (size_t j = 0; j < t_count; j++) {
            for (size_t k = j + 1; k < t_count; k++) {
                if (rank[t_items[j]] > rank[t_items[k]]) {
                    uint32_t temp_i = t_items[j]; t_items[j] = t_items[k]; t_items[k] = temp_i;
                    double temp_u = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = temp_u;
                }
            }
        }

        double remaining_utility = 0;
        for (size_t j = t_count; j-- > 0; ) {
            uint32_t item_id = t_items[j];
            uint32_t r = rank[item_id];
            UtilityList *ul = initial_ext[r];
            if (ul->tuple_count > 0 && ul->tuple_count % 8 == 0) {
                ul->tuples = realloc(ul->tuples, sizeof(UtilityTuple) * (ul->tuple_count + 8));
            }
            ul->tuples[ul->tuple_count].tid = (uint32_t)i;
            ul->tuples[ul->tuple_count].iutil = t_utils[j];
            ul->tuples[ul->tuple_count].rutil = remaining_utility;
            ul->sum_iutil += t_utils[j];
            ul->sum_rutil += remaining_utility;
            ul->tuple_count++;

            for (size_t k = 0; k < j; k++) {
                uint32_t r_k = rank[t_items[k]];
                eucs.matrix[r * eucs.size + r_k] += data[i].total_utility;
                eucs.matrix[r_k * eucs.size + r] += data[i].total_utility;
                pair_consup[r * eucs.size + r_k]++;
                pair_consup[r_k * eucs.size + r]++;
            }
            remaining_utility += t_utils[j];
        }
        free(t_items); free(t_utils);
    }

    // Build Bond Matrix (Strategy 3: PBM)
    Matrix bond_matrix;
    bond_matrix.size = item_count;
    bond_matrix.matrix = calloc(bond_matrix.size * bond_matrix.size, sizeof(double));
    for (uint32_t r_u = 0; r_u < item_count; r_u++) {
        for (uint32_t r_v = r_u + 1; r_v < item_count; r_v++) {
            uint32_t consup = pair_consup[r_u * bond_matrix.size + r_v];
            if (consup > 0) {
                uint32_t dissup = items[r_u].support + items[r_v].support - consup;
                double bond = (double)consup / dissup;
                bond_matrix.matrix[r_u * bond_matrix.size + r_v] = bond;
                bond_matrix.matrix[r_v * bond_matrix.size + r_u] = bond;
            }
        }
    }
    free(pair_consup);

    printf("[FCHM] Starting Recursive Search...\n");
    // Step 5 of Algorithm 1 (Output single items) is handled inside search (Strategy 1: DOS)
    // Actually, Strategy 1 says single items are always correlated (bond=1).
    search(NULL, initial_ext, item_count, min_util, min_bond, &eucs, &bond_matrix, rank);

    printf("[FCHM] Found %zu Correlated High Utility Itemsets.\n", total_chi_count);

    // Cleanup
    for (size_t i = 0; i < item_count; i++) free_utility_list(initial_ext[i]);
    free(initial_ext);
    free(items); free(rank); free(twu_counts); free(supports);
    free(eucs.matrix);
    free(bond_matrix.matrix);

    dm_bench_record_results(total_chi_count, total_items_sum);
    return DM_SUCCESS;
}

static DM_Algorithm fchm_algo = {
    .id = "fchm",
    .name = "FCHM Algorithm",
    .description = "Fast Correlated High-Utility itemset Miner using the bond measure (HAIS 2016).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(fchm_algo)
