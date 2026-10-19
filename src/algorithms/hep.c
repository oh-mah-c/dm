/*
 * HEP (Deng, "Mining high occupancy itemsets", FGCS 102 (2020) 222-229).
 *
 * Level-wise search: candidate k-itemsets are joined from (k-1)-itemsets that
 * share their first k-2 items, their occupancy lists (transaction ids sorted by
 * increasing transaction length) are intersected, and a candidate is kept for
 * the next level when its support reaches xi and its upper bound UBO (Deng,
 * Definition 7) reaches xi. Items are ordered by increasing support and
 * transactions by increasing length, as described in Section 3.2.3 of the paper.
 *
 * The test occ(X) >= xi that decides whether an itemset is reported is exact
 * (core/dm_exact_occ.h); pruning tests compare bounds with xi minus a small
 * tolerance, which can only keep more candidates.
 *
 * If the environment variable DM_HOI_OUT is set, the HOIs are written to that
 * file, one per line, with the original item identifiers in increasing order.
 */
#include "algorithms/hep.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include "core/dm_exact_occ.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    uint32_t *items;      /* item ranks, increasing */
    size_t length;
    uint32_t *tids;
    size_t num_tids;
} HEPItemset;

typedef struct {
    HEPItemset *itemsets;
    size_t count;
    size_t capacity;
} HEPItemsetList;

static void list_init(HEPItemsetList *list) {
    list->count = 0;
    list->capacity = 1024;
    list->itemsets = malloc(list->capacity * sizeof(HEPItemset));
}

static void list_add(HEPItemsetList *list, HEPItemset itemset) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->itemsets = realloc(list->itemsets, list->capacity * sizeof(HEPItemset));
    }
    list->itemsets[list->count++] = itemset;
}

static void list_free(HEPItemsetList *list) {
    for (size_t i = 0; i < list->count; i++) {
        free(list->itemsets[i].items);
        free(list->itemsets[i].tids);
    }
    free(list->itemsets);
}

typedef struct {
    uint32_t orig_tid;
    uint32_t len;
} TransMeta;

static int cmp_trans_meta(const void *a, const void *b) {
    const TransMeta *ta = (const TransMeta *)a;
    const TransMeta *tb = (const TransMeta *)b;
    if (ta->len != tb->len) return ta->len < tb->len ? -1 : 1;
    return (ta->orig_tid > tb->orig_tid) - (ta->orig_tid < tb->orig_tid);
}

typedef struct { uint32_t id; uint32_t count; } ItemCount;
static int cmp_item_count(const void *a, const void *b) {
    const ItemCount *x = (const ItemCount *)a, *y = (const ItemCount *)b;
    if (x->count != y->count) return x->count < y->count ? -1 : 1;
    return (x->id > y->id) - (x->id < y->id);
}

typedef struct {
    DmThreshold xi;
    size_t sigma;
    uint32_t *g_tsize;
    uint32_t *len_scratch;
    FILE *out;
    uint32_t *rank_to_id;
    uint32_t *out_scratch;
} HEPCtx;

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

/* exact test occ(X) >= xi for |X| = k and the supporting transactions tids */
static int occ_is_ho(HEPCtx *c, const uint32_t *tids, size_t size, size_t k, double recip_sum) {
    int res;
    if (dm_occ_decided((double)k * recip_sum, c->xi, &res)) return res;
    for (size_t i = 0; i < size; i++) c->len_scratch[i] = c->g_tsize[tids[i]];
    return dm_occ_ge_exact(k, c->len_scratch, NULL, size, c->xi);
}

/* UBO test of Deng (Definition 7) on a TID list sorted by transaction length;
 * returns whether X is kept, and sets *is_ho to whether X is an HOI */
static bool check_ubo_and_o(HEPCtx *c, const uint32_t *tids, size_t size, size_t k, int *is_ho) {
    if (size < c->sigma) return false;
    double current_sum = 0.0;
    double max_ubo = 0.0;
    for (int i = (int)size - 1; i >= 0; i--) {
        uint32_t tsize = c->g_tsize[tids[i]];
        current_sum += 1.0 / (double)tsize;
        if (i == 0 || c->g_tsize[tids[i - 1]] < tsize) {
            double current_ubo = tsize * current_sum;
            if (current_ubo > max_ubo) max_ubo = current_ubo;
        }
    }
    if (!dm_bound_ge(max_ubo, c->xi)) return false;
    *is_ho = occ_is_ho(c, tids, size, k, current_sum);
    return true;
}

static void emit(HEPCtx *c, const uint32_t *items, size_t k) {
    if (!c->out) return;
    for (size_t i = 0; i < k; i++) c->out_scratch[i] = c->rank_to_id[items[i]];
    qsort(c->out_scratch, k, sizeof(uint32_t), cmp_u32);
    for (size_t i = 0; i < k; i++) fprintf(c->out, i ? " %u" : "%u", c->out_scratch[i]);
    fputc('\n', c->out);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_HEP_Params *hep_params = (DM_HEP_Params *)params;
    double min_occ_param = hep_params ? hep_params->min_occupancy : 0.01;
    HEPCtx c;
    memset(&c, 0, sizeof c);
    c.xi = dm_threshold_make(min_occ_param, ds->count, min_occ_param < 1.0);
    c.sigma = dm_threshold_ceil(c.xi);

    printf("[HEP] Starting on %zu transactions. Min Occupancy Threshold: %.6f (min support %zu)\n",
           ds->count, c.xi.value, c.sigma);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;

    TransMeta *meta = malloc(ds->count * sizeof(TransMeta));
    for (size_t i = 0; i < ds->count; i++) {
        meta[i].orig_tid = i;
        meta[i].len = data[i].count;
    }
    qsort(meta, ds->count, sizeof(TransMeta), cmp_trans_meta);

    c.g_tsize = malloc(ds->count * sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) c.g_tsize[i] = meta[i].len;
    c.len_scratch = malloc((ds->count + 1) * sizeof(uint32_t));

    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        uint32_t orig = meta[i].orig_tid;
        for (size_t j = 0; j < data[orig].count; j++) counts[data[orig].items[j]]++;
    }

    /* items with support >= sigma, ranked by increasing support */
    ItemCount *ic = malloc((ds->max_id + 1) * sizeof(ItemCount));
    size_t m = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++)
        if (counts[i] >= c.sigma) { ic[m].id = i; ic[m].count = counts[i]; m++; }
    qsort(ic, m, sizeof(ItemCount), cmp_item_count);
    uint32_t *id_to_rank = malloc((ds->max_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= ds->max_id; i++) id_to_rank[i] = UINT32_MAX;
    c.rank_to_id = malloc((m + 1) * sizeof(uint32_t));
    for (size_t r = 0; r < m; r++) { id_to_rank[ic[r].id] = (uint32_t)r; c.rank_to_id[r] = ic[r].id; }

    uint32_t **item_tids = malloc((m + 1) * sizeof(uint32_t *));
    uint32_t *item_idx = calloc(m + 1, sizeof(uint32_t));
    for (size_t r = 0; r < m; r++) item_tids[r] = malloc(ic[r].count * sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        uint32_t orig = meta[i].orig_tid;
        for (size_t j = 0; j < data[orig].count; j++) {
            uint32_t r = id_to_rank[data[orig].items[j]];
            if (r != UINT32_MAX) item_tids[r][item_idx[r]++] = (uint32_t)i;
        }
    }
    free(meta);
    free(item_idx);
    free(id_to_rank);

    const char *outp = getenv("DM_HOI_OUT");
    if (outp) c.out = fopen(outp, "w");
    c.out_scratch = malloc((m + 1) * sizeof(uint32_t));

    HEPItemsetList C_prev;
    list_init(&C_prev);

    size_t total_ho_count = 0;
    size_t total_ho_footprint = 0;
    size_t total_visited_nodes = 0;

    for (size_t r = 0; r < m; r++) {
        int is_ho = 0;
        if (check_ubo_and_o(&c, item_tids[r], ic[r].count, 1, &is_ho)) {
            HEPItemset it;
            it.length = 1;
            it.items = malloc(sizeof(uint32_t));
            it.items[0] = (uint32_t)r;
            it.num_tids = ic[r].count;
            it.tids = malloc(ic[r].count * sizeof(uint32_t));
            memcpy(it.tids, item_tids[r], ic[r].count * sizeof(uint32_t));
            list_add(&C_prev, it);
            if (is_ho) {
                total_ho_count++;
                total_ho_footprint += 1;
                emit(&c, it.items, 1);
            }
        }
        free(item_tids[r]);
    }
    free(item_tids);
    free(counts);
    free(ic);

    size_t k = 2;
    while (C_prev.count > 0) {
        HEPItemsetList C_curr;
        list_init(&C_curr);

        for (size_t i = 0; i < C_prev.count; i++) {
            for (size_t j = i + 1; j < C_prev.count; j++) {
                HEPItemset *P1 = &C_prev.itemsets[i];
                HEPItemset *P2 = &C_prev.itemsets[j];

                bool match = true;
                for (size_t idx = 0; idx < k - 2; idx++) {
                    if (P1->items[idx] != P2->items[idx]) { match = false; break; }
                }
                if (!match) break;
                total_visited_nodes++;

                uint32_t *new_tids = malloc((P1->num_tids < P2->num_tids ? P1->num_tids : P2->num_tids) * sizeof(uint32_t) + 1);
                size_t p1_idx = 0, p2_idx = 0, new_idx = 0;
                while (p1_idx < P1->num_tids && p2_idx < P2->num_tids) {
                    if (P1->tids[p1_idx] < P2->tids[p2_idx]) p1_idx++;
                    else if (P1->tids[p1_idx] > P2->tids[p2_idx]) p2_idx++;
                    else { new_tids[new_idx++] = P1->tids[p1_idx]; p1_idx++; p2_idx++; }
                }

                int is_ho = 0;
                if (check_ubo_and_o(&c, new_tids, new_idx, k, &is_ho)) {
                    HEPItemset P;
                    P.length = k;
                    P.items = malloc(k * sizeof(uint32_t));
                    memcpy(P.items, P1->items, (k - 1) * sizeof(uint32_t));
                    P.items[k - 1] = P2->items[k - 2];
                    P.num_tids = new_idx;
                    P.tids = malloc(new_idx * sizeof(uint32_t) + 1);
                    memcpy(P.tids, new_tids, new_idx * sizeof(uint32_t));
                    list_add(&C_curr, P);
                    if (is_ho) {
                        total_ho_count++;
                        total_ho_footprint += k;
                        emit(&c, P.items, k);
                    }
                }
                free(new_tids);
            }
        }

        list_free(&C_prev);
        C_prev = C_curr;
        k++;
    }
    list_free(&C_prev);

    if (c.out) fclose(c.out);
    free(c.out_scratch);
    free(c.rank_to_id);
    free(c.len_scratch);
    free(c.g_tsize);
    printf("[HEP] Complete. Total high occupancy itemsets found: %zu (visited_nodes=%zu)\n", total_ho_count, total_visited_nodes);
    dm_bench_record_results(total_ho_count, total_ho_footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "hep",
    .name = "HEP Algorithm",
    .description = "High Efficient algorithm for mining high occupancy itemsets using UBO pruning (Deng 2020).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
