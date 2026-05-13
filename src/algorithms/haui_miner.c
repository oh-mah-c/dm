#include "algorithms/haui_miner.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t tid;
    double iu;   // Sum of utilities of items in X in this transaction
    double tmu;  // Transaction maximum utility
} AUElement;

typedef struct {
    uint32_t *items;
    size_t len;
    AUElement *elements;
    size_t element_count;
    double sum_iu;
    double sum_tmu;
} AUList;

/* --- UTILS --- */

static AUList* create_au_list(uint32_t *items, size_t len) {
    AUList *aul = malloc(sizeof(AUList));
    aul->items = malloc(sizeof(uint32_t) * len);
    memcpy(aul->items, items, sizeof(uint32_t) * len);
    aul->len = len;
    aul->elements = NULL;
    aul->element_count = 0;
    aul->sum_iu = 0;
    aul->sum_tmu = 0;
    return aul;
}

static void free_au_list(AUList *aul) {
    if (!aul) return;
    free(aul->items);
    free(aul->elements);
    free(aul);
}

static AUList* construct(AUList *p, AUList *px, AUList *py) {
    uint32_t *new_items = malloc(sizeof(uint32_t) * (px->len + 1));
    memcpy(new_items, px->items, sizeof(uint32_t) * px->len);
    new_items[px->len] = py->items[py->len - 1];

    AUList *pxy = create_au_list(new_items, px->len + 1);
    free(new_items);

    size_t capacity = (px->element_count < py->element_count) ? px->element_count : py->element_count;
    pxy->elements = malloc(sizeof(AUElement) * capacity);

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < px->element_count && iy < py->element_count) {
        if (px->elements[ix].tid == py->elements[iy].tid) {
            uint32_t tid = px->elements[ix].tid;
            double iu = px->elements[ix].iu + py->elements[iy].iu;

            if (p != NULL) {
                while (ip < p->element_count && p->elements[ip].tid < tid) ip++;
                if (ip < p->element_count && p->elements[ip].tid == tid) {
                    iu -= p->elements[ip].iu;
                }
            }

            pxy->elements[pxy->element_count].tid = tid;
            pxy->elements[pxy->element_count].iu = iu;
            pxy->elements[pxy->element_count].tmu = py->elements[iy].tmu;
            pxy->sum_iu += iu;
            pxy->sum_tmu += py->elements[iy].tmu;
            pxy->element_count++;
            ix++; iy++;
        } else if (px->elements[ix].tid < py->elements[iy].tid) ix++;
        else iy++;
    }

    if (pxy->element_count == 0) {
        free(pxy->elements);
        pxy->elements = NULL;
    } else {
        pxy->elements = realloc(pxy->elements, sizeof(AUElement) * pxy->element_count);
    }
    return pxy;
}

/* --- SEARCH --- */

static size_t haui_count = 0;
static size_t total_items = 0;

static void search(AUList *p, AUList **extensions, size_t ext_count, double threshold) {
    for (size_t i = 0; i < ext_count; i++) {
        AUList *px = extensions[i];

        // Check if HAUI: sum_iu / len >= threshold
        if (px->sum_iu / px->len >= threshold) {
            haui_count++;
            total_items += px->len;
        }

        // Pruning: if sum_tmu >= threshold
        if (px->sum_tmu >= threshold) {
            AUList **ex_px = malloc(sizeof(AUList*) * (ext_count - i - 1));
            size_t ex_px_count = 0;

            for (size_t j = i + 1; j < ext_count; j++) {
                AUList *py = extensions[j];
                AUList *pxy = construct(p, px, py);
                if (pxy->sum_tmu >= threshold) {
                    ex_px[ex_px_count++] = pxy;
                } else {
                    free_au_list(pxy);
                }
            }

            if (ex_px_count > 0) {
                search(px, ex_px, ex_px_count, threshold);
                for (size_t j = 0; j < ex_px_count; j++) free_au_list(ex_px[j]);
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

    DM_HAUI_Miner_Params *p = (DM_HAUI_Miner_Params *)params;
    double ratio = p ? p->min_utility_ratio : 0.01;
    
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    
    // 1. Calculate TU and TMU for each transaction
    double total_utility = 0;
    double *tmus = malloc(sizeof(double) * ds->count);
    for (size_t i = 0; i < ds->count; i++) {
        total_utility += data[i].total_utility;
        double max_u = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (data[i].items[j].utility > max_u) max_u = data[i].items[j].utility;
        }
        tmus[i] = max_u;
    }

    double threshold = ratio * total_utility;
    printf("[HAUI-Miner] Ratio: %.4f, Total Utility: %.2f, Threshold: %.2f\n", ratio, total_utility, threshold);

    // 2. Calculate AUUB for each item
    double *item_auub = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            item_auub[data[i].items[j].id] += tmus[i];
        }
    }

    // 3. Find 1-HAUUBIs and sort
    ItemAUUB *items = malloc(sizeof(ItemAUUB) * (ds->max_id + 1));
    size_t item_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_auub[i] >= threshold) {
            items[item_count].id = i;
            items[item_count].auub = item_auub[i];
            item_count++;
        }
    }
    qsort(items, item_count, sizeof(ItemAUUB), cmp_item_auub);

    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < item_count; i++) rank[items[i].id] = (uint32_t)i;

    // 4. Build AU-lists for 1-items
    AUList **initial_ext = malloc(sizeof(AUList*) * item_count);
    for (size_t i = 0; i < item_count; i++) {
        uint32_t id = items[i].id;
        initial_ext[i] = create_au_list(&id, 1);
        initial_ext[i]->elements = malloc(sizeof(AUElement) * 8); // initial capacity
    }

    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t id = data[i].items[j].id;
            uint32_t r = rank[id];
            if (r != 0xFFFFFFFF) {
                AUList *aul = initial_ext[r];
                if (aul->element_count > 0 && (aul->element_count % 8 == 0)) {
                    aul->elements = realloc(aul->elements, sizeof(AUElement) * (aul->element_count + 8));
                }
                aul->elements[aul->element_count].tid = (uint32_t)i;
                aul->elements[aul->element_count].iu = data[i].items[j].utility;
                aul->elements[aul->element_count].tmu = tmus[i];
                aul->sum_iu += data[i].items[j].utility;
                aul->sum_tmu += tmus[i];
                aul->element_count++;
            }
        }
    }

    for (size_t i = 0; i < item_count; i++) {
        if (initial_ext[i]->element_count > 0) {
            initial_ext[i]->elements = realloc(initial_ext[i]->elements, sizeof(AUElement) * initial_ext[i]->element_count);
        } else {
            free(initial_ext[i]->elements);
            initial_ext[i]->elements = NULL;
        }
    }

    // 5. Search
    haui_count = 0;
    total_items = 0;
    search(NULL, initial_ext, item_count, threshold);

    printf("[HAUI-Miner] Found %zu HAUIs\n", haui_count);
    dm_bench_record_results(haui_count, total_items);

    // Cleanup
    for (size_t i = 0; i < item_count; i++) free_au_list(initial_ext[i]);
    free(initial_ext);
    free(items); free(rank); free(item_auub); free(tmus);

    return DM_SUCCESS;
}

DM_Algorithm haui_miner_algo = {
    .id = "haui_miner",
    .name = "HAUI-Miner",
    .description = "Mining High Average-Utility Itemsets using AU-lists (Lin et al. 2016).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(haui_miner_algo)
