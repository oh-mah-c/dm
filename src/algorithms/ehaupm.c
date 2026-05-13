#include "algorithms/ehaupm.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t tid;
    double iutil;
    double rmu;
    double remu;
} MAUElement;

typedef struct {
    uint32_t *items;
    size_t len;
    MAUElement *elements;
    size_t element_count;
    double sum_iutil;
    double sum_rmu;
    double sum_remu;
} MAUList;

typedef struct {
    double *data;
    size_t size;
} EAUCM;

/* --- UTILS --- */

static MAUList* create_mau_list(uint32_t *items, size_t len) {
    MAUList *ml = malloc(sizeof(MAUList));
    ml->items = malloc(sizeof(uint32_t) * len);
    memcpy(ml->items, items, sizeof(uint32_t) * len);
    ml->len = len;
    ml->elements = NULL;
    ml->element_count = 0;
    ml->sum_iutil = 0;
    ml->sum_rmu = 0;
    ml->sum_remu = 0;
    return ml;
}

static void free_mau_list(MAUList *ml) {
    if (!ml) return;
    free(ml->items);
    free(ml->elements);
    free(ml);
}

static double calculate_lub(MAUList *ml) {
    // lub(X) = sum over all Tq in ml: (u(X, Tq) + |X| * remu(X, Tq)) / |X|
    // lub(X) = (sum_iutil / |X|) + sum_remu
    return (ml->sum_iutil / ml->len) + ml->sum_remu;
}

static MAUList* construct(MAUList *p, MAUList *px, MAUList *py, double threshold) {
    // PS3: Stop Unpromising Join
    double lub_px = calculate_lub(px);
    double rtub_px = px->sum_rmu;

    uint32_t *new_items = malloc(sizeof(uint32_t) * (px->len + 1));
    memcpy(new_items, px->items, sizeof(uint32_t) * px->len);
    new_items[px->len] = py->items[py->len - 1];

    MAUList *pxy = create_mau_list(new_items, px->len + 1);
    free(new_items);

    size_t capacity = (px->element_count < py->element_count) ? px->element_count : py->element_count;
    pxy->elements = malloc(sizeof(MAUElement) * capacity);

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < px->element_count) {
        MAUElement *ea = &px->elements[ix];
        // Search for tid in py
        while (iy < py->element_count && py->elements[iy].tid < ea->tid) iy++;
        
        if (iy < py->element_count && py->elements[iy].tid == ea->tid) {
            MAUElement *eb = &py->elements[iy];
            double iutil = ea->iutil + eb->iutil;

            if (p != NULL) {
                while (ip < p->element_count && p->elements[ip].tid < ea->tid) ip++;
                if (ip < p->element_count && p->elements[ip].tid == ea->tid) {
                    iutil -= p->elements[ip].iutil;
                }
            }

            pxy->elements[pxy->element_count].tid = ea->tid;
            pxy->elements[pxy->element_count].iutil = iutil;
            pxy->elements[pxy->element_count].rmu = ea->rmu;
            pxy->elements[pxy->element_count].remu = eb->remu;
            pxy->sum_iutil += iutil;
            pxy->sum_rmu += ea->rmu;
            pxy->sum_remu += eb->remu;
            pxy->element_count++;
            iy++;
        } else {
            // PS3: Pruning Strategy 3
            lub_px -= (ea->iutil / px->len) + ea->remu;
            rtub_px -= ea->rmu;
            double min_bound = (lub_px < rtub_px) ? lub_px : rtub_px;
            if (min_bound < threshold) {
                free_mau_list(pxy);
                return NULL;
            }
        }
        ix++;
    }

    if (pxy->element_count == 0) {
        free_mau_list(pxy);
        return NULL;
    } else {
        pxy->elements = realloc(pxy->elements, sizeof(MAUElement) * pxy->element_count);
    }
    return pxy;
}

/* --- SEARCH --- */

static size_t haui_count = 0;
static size_t total_items = 0;

static void search(MAUList *p, MAUList **extensions, size_t ext_count, EAUCM *eaucm, uint32_t *rank, double threshold) {
    for (size_t i = 0; i < ext_count; i++) {
        MAUList *px = extensions[i];

        // Check if HAUI
        if (px->sum_iutil / px->len >= threshold) {
            haui_count++;
            total_items += px->len;
        }

        // Pruning: if min(lub, rtub) >= threshold
        double lub = calculate_lub(px);
        double rtub = px->sum_rmu;
        double min_bound = (lub < rtub) ? lub : rtub;

        if (min_bound >= threshold) {
            MAUList **ex_px = malloc(sizeof(MAUList*) * (ext_count - i - 1));
            size_t ex_px_count = 0;

            for (size_t j = i + 1; j < ext_count; j++) {
                MAUList *py = extensions[j];
                
                // PS2: ECAUPS
                uint32_t id1 = px->items[px->len - 1];
                uint32_t id2 = py->items[py->len - 1];
                uint32_t r1 = rank[id1];
                uint32_t r2 = rank[id2];
                if (eaucm->data[r1 * eaucm->size + r2] < threshold) continue;

                MAUList *pxy = construct(px, px, py, threshold); // Wait, first arg should be px or p?
                // Actually, if we are extending px with item from py, px is the prefix.
                // But wait, the recursive call in HAUI-Miner was Search(Y.AUL, exAULs, ...).
                // My construct(p, px, py) follows the logic u(XY) = u(X) + u(Y) - u(P).
                // So if we extend X with a, then prefix is X.
                // But the items in px already include the prefix.
                // The correct call is construct(p, px, py) where p is the current prefix's list.
                // Wait, px IS the extension of p.
                
                // Let's re-verify: Search(P, ExtensionsOfP).
                // Loop over Xa in ExtensionsOfP.
                // Xa is extension of P.
                // We want to create Xab (extension of Xa).
                // u(Xab) = u(Xa) + u(Xb) - u(P).
                // So construct(p, px, py) is correct.
            }
            // Wait, I need to fix the construct call below.
        }
    }
}

/* --- RE-IMPLEMENTING SEARCH WITH CORRECT ARGS --- */

static void search_fixed(MAUList *p, MAUList **extensions, size_t ext_count, EAUCM *eaucm, uint32_t *rank, double threshold) {
    for (size_t i = 0; i < ext_count; i++) {
        MAUList *px = extensions[i];

        if (px->sum_iutil / px->len >= threshold) {
            haui_count++;
            total_items += px->len;
        }

        double lub = calculate_lub(px);
        double rtub = px->sum_rmu;
        double min_bound = (lub < rtub) ? lub : rtub;

        if (min_bound >= threshold) {
            MAUList **ex_px = malloc(sizeof(MAUList*) * (ext_count - i - 1));
            size_t ex_px_count = 0;

            for (size_t j = i + 1; j < ext_count; j++) {
                MAUList *py = extensions[j];
                
                uint32_t id1 = px->items[px->len - 1];
                uint32_t id2 = py->items[py->len - 1];
                uint32_t r1 = rank[id1];
                uint32_t r2 = rank[id2];
                if (eaucm->data[r1 * eaucm->size + r2] < threshold) continue;

                MAUList *pxy = construct(p, px, py, threshold);
                if (pxy) {
                    ex_px[ex_px_count++] = pxy;
                }
            }

            if (ex_px_count > 0) {
                search_fixed(px, ex_px, ex_px_count, eaucm, rank, threshold);
                for (size_t j = 0; j < ex_px_count; j++) free_mau_list(ex_px[j]);
            }
            free(ex_px);
        }
    }
}

/* --- MAIN RUN --- */

typedef struct {
    uint32_t id;
    double auub;
} ItemAUUB;

static int cmp_item_auub(const void *a, const void *b) {
    double v1 = ((ItemAUUB*)a)->auub;
    double v2 = ((ItemAUUB*)b)->auub;
    if (v1 < v2) return -1;
    if (v1 > v2) return 1;
    return (int)(((ItemAUUB*)a)->id - ((ItemAUUB*)b)->id);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;

    DM_EHAUPM_Params *p = (DM_EHAUPM_Params *)params;
    double ratio = p ? p->min_utility_ratio : 0.01;
    
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    
    // 1. Calculate TU and 1st pass auub
    double total_utility = 0;
    double *item_auub = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        total_utility += data[i].total_utility;
        double max_u = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (data[i].items[j].utility > max_u) max_u = data[i].items[j].utility;
        }
        for (size_t j = 0; j < data[i].count; j++) {
            item_auub[data[i].items[j].id] += max_u;
        }
    }

    double threshold = ratio * total_utility;
    printf("[EHAUPM] Ratio: %.4f, Total Utility: %.2f, Threshold: %.2f\n", ratio, total_utility, threshold);

    // 2. Sort items by auub (PS1: RUI)
    ItemAUUB *items_raw = malloc(sizeof(ItemAUUB) * (ds->max_id + 1));
    size_t promising_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_auub[i] >= threshold) {
            items_raw[promising_count].id = i;
            items_raw[promising_count].auub = item_auub[i];
            promising_count++;
        }
    }
    qsort(items_raw, promising_count, sizeof(ItemAUUB), cmp_item_auub);

    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < promising_count; i++) rank[items_raw[i].id] = (uint32_t)i;

    // 3. Build EAUCM and initial MAU-lists
    EAUCM eaucm;
    eaucm.size = promising_count;
    eaucm.data = calloc(promising_count * promising_count, sizeof(double));

    MAUList **initial_ext = malloc(sizeof(MAUList*) * promising_count);
    for (size_t i = 0; i < promising_count; i++) {
        uint32_t id = items_raw[i].id;
        initial_ext[i] = create_mau_list(&id, 1);
        initial_ext[i]->elements = malloc(sizeof(MAUElement) * 8);
    }

    for (size_t i = 0; i < ds->count; i++) {
        // Filter and sort items in transaction
        uint32_t *t_items = malloc(sizeof(uint32_t) * data[i].count);
        double *t_utils = malloc(sizeof(double) * data[i].count);
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

        if (t_count > 0) {
            // Recalculate TMU for this transaction after removing unpromising items
            double tmu = 0;
            for (size_t j = 0; j < t_count; j++) if (t_utils[j] > tmu) tmu = t_utils[j];

            // Build initial lists and update EAUCM
            for (size_t j = 0; j < t_count; j++) {
                uint32_t r1 = rank[t_items[j]];
                MAUList *ml = initial_ext[r1];
                if (ml->element_count > 0 && (ml->element_count % 8 == 0)) {
                    ml->elements = realloc(ml->elements, sizeof(MAUElement) * (ml->element_count + 8));
                }
                
                // Calculate rmu and remu
                double rmu = 0; // max utility of items after ml (wait, items in t_items are sorted)
                for (size_t k = 0; k < t_count; k++) if (k != j && t_utils[k] > rmu) rmu = t_utils[k];
                
                double remu = 0; // max utility of items FOLLOWING ml
                for (size_t k = j + 1; k < t_count; k++) if (t_utils[k] > remu) remu = t_utils[k];

                ml->elements[ml->element_count].tid = (uint32_t)i;
                ml->elements[ml->element_count].iutil = t_utils[j];
                ml->elements[ml->element_count].rmu = rmu;
                ml->elements[ml->element_count].remu = remu;
                ml->sum_iutil += t_utils[j];
                ml->sum_rmu += rmu;
                ml->sum_remu += remu;
                ml->element_count++;

                // EAUCM
                for (size_t k = j + 1; k < t_count; k++) {
                    uint32_t r2 = rank[t_items[k]];
                    eaucm.data[r1 * promising_count + r2] += tmu;
                }
            }
        }
        free(t_items); free(t_utils);
    }

    for (size_t i = 0; i < promising_count; i++) {
        if (initial_ext[i]->element_count > 0) {
            initial_ext[i]->elements = realloc(initial_ext[i]->elements, sizeof(MAUElement) * initial_ext[i]->element_count);
        } else {
            free(initial_ext[i]->elements);
            initial_ext[i]->elements = NULL;
        }
    }

    // 4. Search
    haui_count = 0;
    total_items = 0;
    search_fixed(NULL, initial_ext, promising_count, &eaucm, rank, threshold);

    printf("[EHAUPM] Found %zu HAUIs\n", haui_count);
    dm_bench_record_results(haui_count, total_items);

    // Cleanup
    for (size_t i = 0; i < promising_count; i++) free_mau_list(initial_ext[i]);
    free(initial_ext);
    free(eaucm.data); free(items_raw); free(rank); free(item_auub);

    return DM_SUCCESS;
}

DM_Algorithm ehaupm_algo = {
    .id = "ehaupm",
    .name = "EHAUPM",
    .description = "Efficient High Average-Utility Pattern Mining with Tighter Upper Bounds (Lin et al. 2017).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(ehaupm_algo)
