#include "algorithms/memu.h"
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
} CAUElement;

typedef struct {
    uint32_t *items;
    size_t len;
    CAUElement *elements;
    size_t element_count;
    double sum_iutil;
    double sum_rmu;
    double sum_mau; // Sum of mau values for items in this itemset
} CAUList;

typedef struct {
    double *data;
    size_t size;
} EAUM;

/* --- UTILS --- */

static CAUList* create_cau_list(uint32_t *items, size_t len, double sum_mau) {
    CAUList *cl = malloc(sizeof(CAUList));
    cl->items = malloc(sizeof(uint32_t) * len);
    memcpy(cl->items, items, sizeof(uint32_t) * len);
    cl->len = len;
    cl->elements = NULL;
    cl->element_count = 0;
    cl->sum_iutil = 0;
    cl->sum_rmu = 0;
    cl->sum_mau = sum_mau;
    return cl;
}

static void free_cau_list(CAUList *cl) {
    if (!cl) return;
    free(cl->items);
    free(cl->elements);
    free(cl);
}

static CAUList* construct(CAUList *p, CAUList *px, CAUList *py, double lmau) {
    // PS3: LA-Prune
    double rtub_px = px->sum_rmu;

    uint32_t *new_items = malloc(sizeof(uint32_t) * (px->len + 1));
    memcpy(new_items, px->items, sizeof(uint32_t) * px->len);
    new_items[px->len] = py->items[py->len - 1];

    double new_sum_mau = px->sum_mau + (py->sum_mau - (p ? p->sum_mau : 0));
    CAUList *pxy = create_cau_list(new_items, px->len + 1, new_sum_mau);
    free(new_items);

    size_t capacity = (px->element_count < py->element_count) ? px->element_count : py->element_count;
    pxy->elements = malloc(sizeof(CAUElement) * capacity);

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < px->element_count) {
        CAUElement *ea = &px->elements[ix];
        while (iy < py->element_count && py->elements[iy].tid < ea->tid) iy++;
        
        if (iy < py->element_count && py->elements[iy].tid == ea->tid) {
            CAUElement *eb = &py->elements[iy];
            double iutil = ea->iutil + eb->iutil;

            if (p != NULL) {
                while (ip < p->element_count && p->elements[ip].tid < ea->tid) ip++;
                if (ip < p->element_count && p->elements[ip].tid == ea->tid) {
                    iutil -= p->elements[ip].iutil;
                }
            }

            pxy->elements[pxy->element_count].tid = ea->tid;
            pxy->elements[pxy->element_count].iutil = iutil;
            pxy->elements[pxy->element_count].rmu = eb->rmu;
            pxy->sum_iutil += iutil;
            pxy->sum_rmu += eb->rmu;
            pxy->element_count++;
            iy++;
        } else {
            // PS3: LA-Prune
            rtub_px -= ea->rmu;
            if (rtub_px < lmau) {
                free_cau_list(pxy);
                return NULL;
            }
        }
        ix++;
    }

    if (pxy->element_count == 0) {
        free_cau_list(pxy);
        return NULL;
    } else {
        pxy->elements = realloc(pxy->elements, sizeof(CAUElement) * pxy->element_count);
    }
    return pxy;
}

/* --- SEARCH --- */

static size_t haui_count = 0;
static size_t total_items = 0;

static void search(CAUList *p, CAUList **extensions, size_t ext_count, EAUM *eaum, uint32_t *rank, double lmau) {
    for (size_t i = 0; i < ext_count; i++) {
        CAUList *px = extensions[i];

        // Check if HAUI: sum_iutil / len >= mau(X)
        if (px->sum_iutil / px->len >= (px->sum_mau / px->len)) {
            haui_count++;
            total_items += px->len;
        }

        // Pruning: if sum_rmu >= mau(X)
        // Wait, Theorem 5 says Sorted Downward Closure of rtub.
        // rtub(X) = sum_rmu.
        if (px->sum_rmu >= (px->sum_mau / px->len)) {
            CAUList **ex_px = malloc(sizeof(CAUList*) * (ext_count - i - 1));
            size_t ex_px_count = 0;

            for (size_t j = i + 1; j < ext_count; j++) {
                CAUList *py = extensions[j];
                
                // PS2: EAUMS
                uint32_t id1 = px->items[px->len - 1];
                uint32_t id2 = py->items[py->len - 1];
                if (eaum->data[rank[id1] * eaum->size + rank[id2]] < lmau) continue;

                CAUList *pxy = construct(px, px, py, lmau);
                if (pxy) {
                    ex_px[ex_px_count++] = pxy;
                }
            }

            if (ex_px_count > 0) {
                search(px, ex_px, ex_px_count, eaum, rank, lmau);
                for (size_t j = 0; j < ex_px_count; j++) free_cau_list(ex_px[j]);
            }
            free(ex_px);
        }
    }
}

/* --- MAIN RUN --- */

typedef struct {
    uint32_t id;
    double mau;
} ItemMAU;

static int cmp_item_mau(const void *a, const void *b) {
    double v1 = ((ItemMAU*)a)->mau;
    double v2 = ((ItemMAU*)b)->mau;
    if (v1 < v2) return -1;
    if (v1 > v2) return 1;
    return (int)(((ItemMAU*)a)->id - ((ItemMAU*)b)->id);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;

    DM_MEMU_Params *p = (DM_MEMU_Params *)params;
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;

    // 0. Setup MAU-Table if not provided (for benchmarking, use a heuristic like in paper)
    double *mau_table;
    if (!p) {
        mau_table = malloc(sizeof(double) * (ds->max_id + 1));
        for (uint32_t i = 0; i <= ds->max_id; i++) mau_table[i] = 1000.0; // dummy
    } else {
        mau_table = p->mau_table;
    }

    double lmau = 1e18;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (mau_table[i] < lmau) lmau = mau_table[i];
    }

    // 1. First Scan: Calculate auub(i)
    double *item_auub = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        double max_u = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (data[i].items[j].utility > max_u) max_u = data[i].items[j].utility;
        }
        for (size_t j = 0; j < data[i].count; j++) {
            item_auub[data[i].items[j].id] += max_u;
        }
    }

    // 2. Sorting items by mau ascending
    ItemMAU *items_raw = malloc(sizeof(ItemMAU) * (ds->max_id + 1));
    size_t promising_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_auub[i] >= lmau) {
            items_raw[promising_count].id = i;
            items_raw[promising_count].mau = mau_table[i];
            promising_count++;
        }
    }
    qsort(items_raw, promising_count, sizeof(ItemMAU), cmp_item_mau);

    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < promising_count; i++) rank[items_raw[i].id] = (uint32_t)i;

    // 3. Build EAUM and initial CAU-lists
    EAUM eaum;
    eaum.size = promising_count;
    eaum.data = calloc(promising_count * promising_count, sizeof(double));

    CAUList **initial_ext = malloc(sizeof(CAUList*) * promising_count);
    for (size_t i = 0; i < promising_count; i++) {
        uint32_t id = items_raw[i].id;
        initial_ext[i] = create_cau_list(&id, 1, mau_table[id]);
        initial_ext[i]->elements = malloc(sizeof(CAUElement) * 8);
    }

    for (size_t i = 0; i < ds->count; i++) {
        // Filter and sort items in transaction by rank
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
        for (size_t j = 0; j < t_count; j++) {
            for (size_t k = j + 1; k < t_count; k++) {
                if (rank[t_items[j]] > rank[t_items[k]]) {
                    uint32_t tmp_i = t_items[j]; t_items[j] = t_items[k]; t_items[k] = tmp_i;
                    double tmp_u = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = tmp_u;
                }
            }
        }

        if (t_count > 0) {
            double tmu = 0;
            for (size_t j = 0; j < t_count; j++) if (t_utils[j] > tmu) tmu = t_utils[j];

            for (size_t j = 0; j < t_count; j++) {
                uint32_t r1 = rank[t_items[j]];
                CAUList *cl = initial_ext[r1];
                if (cl->element_count > 0 && (cl->element_count % 8 == 0)) {
                    cl->elements = realloc(cl->elements, sizeof(CAUElement) * (cl->element_count + 8));
                }
                
                double rmu = 0;
                for (size_t k = j + 1; k < t_count; k++) if (t_utils[k] > rmu) rmu = t_utils[k];

                cl->elements[cl->element_count].tid = (uint32_t)i;
                cl->elements[cl->element_count].iutil = t_utils[j];
                cl->elements[cl->element_count].rmu = rmu;
                cl->sum_iutil += t_utils[j];
                cl->sum_rmu += rmu;
                cl->element_count++;

                for (size_t k = j + 1; k < t_count; k++) {
                    uint32_t r2 = rank[t_items[k]];
                    eaum.data[r1 * promising_count + r2] += tmu;
                }
            }
        }
        free(t_items); free(t_utils);
    }

    for (size_t i = 0; i < promising_count; i++) {
        if (initial_ext[i]->element_count > 0) {
            initial_ext[i]->elements = realloc(initial_ext[i]->elements, sizeof(CAUElement) * initial_ext[i]->element_count);
        } else {
            free(initial_ext[i]->elements);
            initial_ext[i]->elements = NULL;
        }
    }

    // 4. Search
    haui_count = 0;
    total_items = 0;
    search(NULL, initial_ext, promising_count, &eaum, rank, lmau);

    printf("[MEMU] Found %zu HAUIs\n", haui_count);
    dm_bench_record_results(haui_count, total_items);

    // Cleanup
    for (size_t i = 0; i < promising_count; i++) free_cau_list(initial_ext[i]);
    free(initial_ext);
    free(eaum.data); free(items_raw); free(rank); free(item_auub);
    if (!p) free(mau_table);

    return DM_SUCCESS;
}

DM_Algorithm memu_algo = {
    .id = "memu",
    .name = "MEMU",
    .description = "Mining High Average-Utility Patterns with Multiple Thresholds (Lin et al. 2018).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(memu_algo)
