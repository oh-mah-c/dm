#include "algorithms/fhmds.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

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
} UtilityList;

typedef struct {
    uint32_t *items;
    size_t len;
    double utility;
} HUI_Result;

typedef struct {
    HUI_Result *results;
    size_t count;
    size_t k;
} TopK_Buffer;

/* --- TOP-K BUFFER UTILS --- */

static void topk_init(TopK_Buffer *buf, size_t k) {
    buf->k = k;
    buf->count = 0;
    buf->results = malloc(sizeof(HUI_Result) * (k + 1));
}

static void topk_update(TopK_Buffer *buf, uint32_t *items, size_t len, double util, double *threshold) {
    if (util < *threshold) return;

    // Check for duplicates
    for (size_t i = 0; i < buf->count; i++) {
        if (buf->results[i].len == len) {
            bool match = true;
            for (size_t j = 0; j < len; j++) {
                if (buf->results[i].items[j] != items[j]) { match = false; break; }
            }
            if (match) return;
        }
    }

    if (buf->count < buf->k) {
        buf->results[buf->count].items = malloc(sizeof(uint32_t) * len);
        memcpy(buf->results[buf->count].items, items, sizeof(uint32_t) * len);
        buf->results[buf->count].len = len;
        buf->results[buf->count].utility = util;
        buf->count++;
    } else if (util > buf->results[buf->count - 1].utility) {
        free(buf->results[buf->count - 1].items);
        buf->results[buf->count - 1].items = malloc(sizeof(uint32_t) * len);
        memcpy(buf->results[buf->count - 1].items, items, sizeof(uint32_t) * len);
        buf->results[buf->count - 1].len = len;
        buf->results[buf->count - 1].utility = util;
    } else {
        return;
    }

    // Sort buffer descending
    for (size_t i = 0; i < buf->count; i++) {
        for (size_t j = i + 1; j < buf->count; j++) {
            if (buf->results[i].utility < buf->results[j].utility) {
                HUI_Result tmp = buf->results[i];
                buf->results[i] = buf->results[j];
                buf->results[j] = tmp;
            }
        }
    }

    if (buf->count == buf->k) {
        *threshold = buf->results[buf->k - 1].utility;
    }
}

/* --- UTILITY LIST UTILS --- */

static UtilityList* construct(UtilityList *p, UtilityList *px, UtilityList *py) {
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
        } else if (px->tuples[ix].tid < py->tuples[iy].tid) ix++;
        else iy++;
    }
    return pxy;
}

static void free_utility_list(UtilityList *ul) {
    if (!ul) return;
    free(ul->items);
    free(ul->tuples);
    free(ul);
}

/* --- SEARCH --- */

static void search(UtilityList *p, UtilityList **extensions, size_t ext_count, TopK_Buffer *buf, double *threshold) {
    for (size_t i = 0; i < ext_count; i++) {
        UtilityList *px = extensions[i];
        
        topk_update(buf, px->items, px->count, px->sum_iutil, threshold);

        if (px->sum_iutil + px->sum_rutil >= *threshold) {
            UtilityList **ex_px = malloc(sizeof(UtilityList*) * (ext_count - i - 1));
            size_t ex_px_count = 0;

            for (size_t j = i + 1; j < ext_count; j++) {
                UtilityList *py = extensions[j];
                UtilityList *pxy = construct(p, px, py);
                if (pxy->sum_iutil + pxy->sum_rutil >= *threshold) {
                    ex_px[ex_px_count++] = pxy;
                } else {
                    free_utility_list(pxy);
                }
            }

            if (ex_px_count > 0) {
                search(px, ex_px, ex_px_count, buf, threshold);
                for (size_t j = 0; j < ex_px_count; j++) free_utility_list(ex_px[j]);
            }
            free(ex_px);
        }
    }
}

/* --- MAIN RUN --- */

typedef struct {
    uint32_t id;
    double twu;
} ItemTWU;

static int cmp_item_twu(const void *a, const void *b) {
    double twu1 = ((ItemTWU*)a)->twu;
    double twu2 = ((ItemTWU*)b)->twu;
    if (twu1 < twu2) return -1;
    if (twu1 > twu2) return 1;
    return (int)(((ItemTWU*)a)->id - ((ItemTWU*)b)->id);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_FHMDS_Params *p = (DM_FHMDS_Params *)params;
    int k = p ? p->k : 10;
    int batch_size = p ? p->batch_size : 1000;
    int window_size = p ? p->window_size : 5; // 5 batches in window

    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    size_t total_trans = ds->count;
    size_t batch_count = (total_trans + batch_size - 1) / batch_size;

    TopK_Buffer topk_buf;
    topk_init(&topk_buf, k);
    double threshold = 0;

    printf("[FHMDS] Total transactions: %zu, Batch size: %d, Window size: %d batches\n", total_trans, batch_size, window_size);

    for (size_t b = 0; b < batch_count; b++) {
        size_t win_start_batch = (b >= (size_t)window_size) ? b - window_size + 1 : 0;
        size_t win_start_trans = win_start_batch * batch_size;
        size_t win_end_trans = (b + 1) * batch_size;
        if (win_end_trans > total_trans) win_end_trans = total_trans;

        printf("[FHMDS] Processing Batch %zu (Transactions %zu to %zu). Window: Batch %zu to %zu\n", 
               b, b * batch_size, win_end_trans - 1, win_start_batch, b);

        // 1. Calculate TWU for items in the current window
        double *twu_counts = calloc(ds->max_id + 1, sizeof(double));
        for (size_t i = win_start_trans; i < win_end_trans; i++) {
            for (size_t j = 0; j < data[i].count; j++) twu_counts[data[i].items[j].id] += data[i].total_utility;
        }

        ItemTWU *items = malloc(sizeof(ItemTWU) * (ds->max_id + 1));
        size_t item_count = 0;
        for (uint32_t i = 0; i <= ds->max_id; i++) {
            if (twu_counts[i] > 0) {
                items[item_count].id = i;
                items[item_count].twu = twu_counts[i];
                item_count++;
            }
        }
        qsort(items, item_count, sizeof(ItemTWU), cmp_item_twu);

        uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
        memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
        for (size_t i = 0; i < item_count; i++) rank[items[i].id] = (uint32_t)i;

        // 2. Build initial iLists for items in window
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
        }

        for (size_t i = win_start_trans; i < win_end_trans; i++) {
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
                for (size_t l = j + 1; l < t_count; l++) {
                    if (rank[t_items[j]] > rank[t_items[l]]) {
                        uint32_t tmp_i = t_items[j]; t_items[j] = t_items[l]; t_items[l] = tmp_i;
                        double tmp_u = t_utils[j]; t_utils[j] = t_utils[l]; t_utils[l] = tmp_u;
                    }
                }
            }

            double remaining_utility = 0;
            for (int j = (int)t_count - 1; j >= 0; j--) {
                uint32_t r = rank[t_items[j]];
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
                remaining_utility += t_utils[j];
            }
            free(t_items); free(t_utils);
        }

        // 3. Raise threshold using individual items
        for (size_t i = 0; i < item_count; i++) {
            topk_update(&topk_buf, initial_ext[i]->items, 1, initial_ext[i]->sum_iutil, &threshold);
        }

        // 4. Recursive search
        search(NULL, initial_ext, item_count, &topk_buf, &threshold);

        // Cleanup for this window
        for (size_t i = 0; i < item_count; i++) free_utility_list(initial_ext[i]);
        free(initial_ext);
        free(items); free(rank); free(twu_counts);
        
        printf("[FHMDS] Batch %zu done. Current threshold: %.2f\n", b, threshold);
    }

    printf("[FHMDS] Final Top-K HUIs found: %zu\n", topk_buf.count);
    size_t total_items_sum = 0;
    for (size_t i = 0; i < topk_buf.count; i++) {
        total_items_sum += topk_buf.results[i].len;
    }

    dm_bench_record_results(topk_buf.count, total_items_sum);
    
    for (size_t i = 0; i < topk_buf.count; i++) free(topk_buf.results[i].items);
    free(topk_buf.results);

    return DM_SUCCESS;
}

DM_Algorithm fhmds_algo = {
    .id = "fhmds",
    .name = "FHMDS",
    .description = "Fast Top-K High Utility Itemset Mining from Data Streams (Sliding Window).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(fhmds_algo)
