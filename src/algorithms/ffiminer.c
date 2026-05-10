#include "algorithms/ffiminer.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

/**
 * FFI-Miner: A fast Algorithm for mining fuzzy frequent itemsets.
 * Reference: Lin et al., "A fast Algorithm for mining fuzzy frequent itemsets", 
 * Journal of Intelligent & Fuzzy Systems 29 (2015) 2373–2379.
 */

/* Membership Functions (Fig 1) */
typedef enum { FUZZY_LOW = 0, FUZZY_MID = 1, FUZZY_HIGH = 2 } FuzzyTerm;

static double get_membership(double val, FuzzyTerm term) {
    if (term == FUZZY_LOW) {
        if (val <= 1.0) return 1.0;
        if (val >= 6.0) return 0.0;
        return (6.0 - val) / (6.0 - 1.0);
    } else if (term == FUZZY_MID) {
        if (val <= 1.0 || val >= 11.0) return 0.0;
        if (val <= 6.0) return (val - 1.0) / (6.0 - 1.0);
        return (11.0 - val) / (11.0 - 6.0);
    } else if (term == FUZZY_HIGH) {
        if (val <= 6.0) return 0.0;
        if (val >= 11.0) return 1.0;
        return (val - 6.0) / (11.0 - 6.0);
    }
    return 0.0;
}

/* Fuzzy List Element */
typedef struct {
    uint32_t tid;
    double if_val; // internal fuzzy value
    double rf_val; // resting fuzzy value
} FFI_Element;

/* Fuzzy List */
typedef struct {
    uint32_t *items;
    size_t k;
    FFI_Element *elements;
    size_t count;
    double sum_if;
    double sum_rf;
} FFI_List;

/* --- Internal Helpers --- */

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

static FFI_List* create_list(uint32_t *items, size_t k) {
    FFI_List *fl = malloc(sizeof(FFI_List));
    fl->items = malloc(sizeof(uint32_t) * k);
    memcpy(fl->items, items, sizeof(uint32_t) * k);
    fl->k = k;
    fl->elements = NULL;
    fl->count = 0;
    fl->sum_if = 0;
    fl->sum_rf = 0;
    return fl;
}

static void free_list(FFI_List *fl) {
    if (!fl) return;
    free(fl->items);
    free(fl->elements);
    free(fl);
}

/* Algorithm 1: Fuzzy-list Construction */
static FFI_List* construct(FFI_List *Px, FFI_List *Py) {
    uint32_t *new_items = malloc(sizeof(uint32_t) * (Px->k + 1));
    memcpy(new_items, Px->items, sizeof(uint32_t) * Px->k);
    new_items[Px->k] = Py->items[Py->k - 1];
    
    FFI_List *Pxy = create_list(new_items, Px->k + 1);
    free(new_items);
    
    Pxy->elements = malloc(sizeof(FFI_Element) * Px->count);
    size_t i = 0, j = 0;
    while (i < Px->count && j < Py->count) {
        if (Px->elements[i].tid == Py->elements[j].tid) {
            FFI_Element e;
            e.tid = Px->elements[i].tid;
            e.if_val = (Px->elements[i].if_val < Py->elements[j].if_val) ? Px->elements[i].if_val : Py->elements[j].if_val;
            e.rf_val = Py->elements[j].rf_val;
            
            Pxy->elements[Pxy->count++] = e;
            Pxy->sum_if += e.if_val;
            Pxy->sum_rf += e.rf_val;
            i++; j++;
        } else if (Px->elements[i].tid < Py->elements[j].tid) {
            i++;
        } else {
            j++;
        }
    }
    return Pxy;
}

/* --- Global State --- */
static size_t total_ffi = 0;
static size_t total_items_footprint = 0;

/* Algorithm 2: FFI-Miner */
static void ffi_miner_recursive(FFI_List **FLs, size_t fl_count, double threshold) {
    for (size_t i = 0; i < fl_count; i++) {
        FFI_List *X = FLs[i];
        
        // Strategy 3: Check SUM.X.if
        if (X->sum_if >= threshold - 1e-9) {
            total_ffi++;
            total_items_footprint += X->k;
        }
        
        // Strategy 3: Check SUM.X.rf (Pruning extensions)
        if (X->sum_rf >= threshold - 1e-9) {
            FFI_List **exFLs = malloc(sizeof(FFI_List*) * (fl_count - i - 1));
            size_t ex_count = 0;
            for (size_t j = i + 1; j < fl_count; j++) {
                FFI_List *Y = FLs[j];
                exFLs[ex_count++] = construct(X, Y);
            }
            
            if (ex_count > 0) {
                ffi_miner_recursive(exFLs, ex_count, threshold);
                for (size_t j = 0; j < ex_count; j++) free_list(exFLs[j]);
            }
            free(exFLs);
        }
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_FFIMINER_Params *p = (DM_FFIMINER_Params *)params;
    double delta = p ? p->min_support : 0.1;
    double threshold = delta * ds->count;

    printf("[FFI-Miner] Starting on %zu transactions. Delta: %.2f (Threshold: %.2f)\n", ds->count, delta, threshold);

    total_ffi = 0;
    total_items_footprint = 0;

    // 1. Transform to fuzzy terms and apply Strategy 1
    // We need to find for each item which fuzzy term has max support.
    double *term_supports = calloc((ds->max_id + 1) * 3, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        if (ds->type == DM_TYPE_TRANSACTIONAL) {
            DM_Trans_Simple *tr = &((DM_Trans_Simple *)ds->payload)[i];
            for (size_t j = 0; j < tr->count; j++) {
                term_supports[tr->items[j] * 3 + FUZZY_LOW] += get_membership(1.0, FUZZY_LOW);
                term_supports[tr->items[j] * 3 + FUZZY_MID] += get_membership(1.0, FUZZY_MID);
                term_supports[tr->items[j] * 3 + FUZZY_HIGH] += get_membership(1.0, FUZZY_HIGH);
            }
        } else if (ds->type == DM_TYPE_UTILITY) {
            DM_Trans_Utility *tr = &((DM_Trans_Utility *)ds->payload)[i];
            for (size_t j = 0; j < tr->count; j++) {
                term_supports[tr->items[j].id * 3 + FUZZY_LOW] += get_membership(tr->items[j].utility, FUZZY_LOW);
                term_supports[tr->items[j].id * 3 + FUZZY_MID] += get_membership(tr->items[j].utility, FUZZY_MID);
                term_supports[tr->items[j].id * 3 + FUZZY_HIGH] += get_membership(tr->items[j].utility, FUZZY_HIGH);
            }
        }
    }

    // Item term map (Strategy 1)
    FuzzyTerm *item_best_term = malloc(sizeof(FuzzyTerm) * (ds->max_id + 1));
    bool *item_valid = calloc(ds->max_id + 1, sizeof(bool));
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        double max_sup = -1.0;
        int best_t = -1;
        for (int t = 0; t < 3; t++) {
            if (term_supports[i * 3 + t] > max_sup) {
                max_sup = term_supports[i * 3 + t];
                best_t = t;
            }
        }
        if (max_sup >= threshold - 1e-9) {
            item_best_term[i] = (FuzzyTerm)best_t;
            item_valid[i] = true;
        }
    }

    // 2. Sorting by support (Strategy 2)
    typedef struct { uint32_t id; double sup; } ItemSup;
    ItemSup *sorted_items = malloc(sizeof(ItemSup) * (ds->max_id + 1));
    size_t valid_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_valid[i]) {
            sorted_items[valid_count].id = i;
            sorted_items[valid_count].sup = term_supports[i * 3 + item_best_term[i]];
            valid_count++;
        }
    }
    // Sort ascending by support
    for (size_t i = 0; i < valid_count; i++) {
        for (size_t j = i + 1; j < valid_count; j++) {
            if (sorted_items[i].sup > sorted_items[j].sup) {
                ItemSup tmp = sorted_items[i];
                sorted_items[i] = sorted_items[j];
                sorted_items[j] = tmp;
            }
        }
    }

    // 3. Build Initial Fuzzy Lists
    FFI_List **L1 = malloc(sizeof(FFI_List*) * valid_count);
    for (size_t i = 0; i < valid_count; i++) {
        L1[i] = create_list(&sorted_items[i].id, 1);
        L1[i]->elements = malloc(sizeof(FFI_Element) * ds->count);
    }

    for (size_t i = 0; i < ds->count; i++) {
        // Map transaction items to their best fuzzy term and membership
        typedef struct { size_t order_idx; double membership; } TransFuzzy;
        TransFuzzy *tf = malloc(sizeof(TransFuzzy) * valid_count);
        size_t tf_count = 0;
        
        if (ds->type == DM_TYPE_TRANSACTIONAL) {
            DM_Trans_Simple *tr = &((DM_Trans_Simple *)ds->payload)[i];
            for (size_t j = 0; j < tr->count; j++) {
                uint32_t id = tr->items[j];
                if (item_valid[id]) {
                    for (size_t o = 0; o < valid_count; o++) {
                        if (sorted_items[o].id == id) {
                            tf[tf_count].order_idx = o;
                            tf[tf_count].membership = get_membership(1.0, item_best_term[id]);
                            tf_count++; break;
                        }
                    }
                }
            }
        } else {
            DM_Trans_Utility *tr = &((DM_Trans_Utility *)ds->payload)[i];
            for (size_t j = 0; j < tr->count; j++) {
                uint32_t id = tr->items[j].id;
                if (item_valid[id]) {
                    for (size_t o = 0; o < valid_count; o++) {
                        if (sorted_items[o].id == id) {
                            tf[tf_count].order_idx = o;
                            tf[tf_count].membership = get_membership(tr->items[j].utility, item_best_term[id]);
                            tf_count++; break;
                        }
                    }
                }
            }
        }
        
        // Sort tf by order_idx
        for (size_t a = 0; a < tf_count; a++) {
            for (size_t b = a + 1; b < tf_count; b++) {
                if (tf[a].order_idx > tf[b].order_idx) {
                    TransFuzzy tmp = tf[a]; tf[a] = tf[b]; tf[b] = tmp;
                }
            }
        }

        // Fill lists
        for (size_t a = 0; a < tf_count; a++) {
            size_t idx = tf[a].order_idx;
            FFI_Element e;
            e.tid = (uint32_t)i;
            e.if_val = tf[a].membership;
            
            // Calc rf (max membership of items after this)
            double max_after = 0.0;
            for (size_t b = a + 1; b < tf_count; b++) {
                if (tf[b].membership > max_after) max_after = tf[b].membership;
            }
            e.rf_val = max_after;
            
            L1[idx]->elements[L1[idx]->count++] = e;
            L1[idx]->sum_if += e.if_val;
            L1[idx]->sum_rf += e.rf_val;
        }
        free(tf);
    }

    // 4. Recursive Mining
    ffi_miner_recursive(L1, valid_count, threshold);

    // Cleanup
    for (size_t i = 0; i < valid_count; i++) free_list(L1[i]);
    free(L1); free(sorted_items); free(item_valid); free(item_best_term); free(term_supports);

    printf("[FFI-Miner] Complete. Total FFIs found: %zu\n", total_ffi);
    dm_bench_record_results(total_ffi, total_items_footprint);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "ffiminer",
    .name = "FFI-Miner Algorithm",
    .description = "Fuzzy Frequent Itemset Mining with Strategy 1 (Max Scalar Cardinality).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL) | (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
