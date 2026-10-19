/*
 * DFHOI (Nguyen et al., "An efficient method for mining high occupancy itemsets
 * based on equivalence class and early pruning", Knowledge-Based Systems 2023).
 *
 * Depth-first search over prefix equivalence classes of TID lists. A child of a
 * class member is kept when its support reaches xi and its upper bound UBO
 * (Deng 2020, Definition 7) reaches xi. When all transactions have the same
 * length L, occ(X) = |X| supp(X) / L is computed directly and UBO equals the
 * support. Items are ordered by increasing support and transactions by
 * increasing length.
 *
 * The test occ(X) >= xi that decides whether an itemset is reported is exact
 * (core/dm_exact_occ.h); pruning tests compare bounds with xi minus a small
 * tolerance, which can only keep more candidates.
 *
 * If the environment variable DM_HOI_OUT is set, the HOIs are written to that
 * file, one per line, with the original item identifiers in increasing order.
 */
#include "algorithms/dfhoi.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include "core/dm_exact_occ.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    uint32_t last;        /* rank of the last item */
    size_t length;
    uint32_t *tids;
    size_t num_tids;
} DFHOIItemset;

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

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

typedef struct {
    DmThreshold xi;
    size_t sigma;
    uint32_t *g_tsize;
    bool same_length;
    uint32_t first_len;
    uint32_t *len_scratch;
    uint32_t *buffer;
    uint32_t *prefix;      /* ranks of the items on the current path */
    uint32_t *rank_to_id;
    uint32_t *out_scratch;
    FILE *out;
    size_t ho_count, ho_footprint, visited;
} DFHOICtx;

static void emit(DFHOICtx *c, size_t k) {
    if (!c->out) return;
    for (size_t i = 0; i < k; i++) c->out_scratch[i] = c->rank_to_id[c->prefix[i]];
    qsort(c->out_scratch, k, sizeof(uint32_t), cmp_u32);
    for (size_t i = 0; i < k; i++) fprintf(c->out, i ? " %u" : "%u", c->out_scratch[i]);
    fputc('\n', c->out);
}

/* returns whether X (k items, TID list tids) is kept; sets *is_ho exactly */
static bool check_ubo_and_o(DFHOICtx *c, const uint32_t *tids, size_t size, size_t k, int *is_ho) {
    if (size < c->sigma) return false;
    if (c->same_length) {
        double o_val = (double)(k * size) / (double)c->first_len;
        int res;
        *is_ho = dm_occ_decided(o_val, c->xi, &res) ? res : dm_occ_ge_uniform(k, size, c->first_len, c->xi);
        return true;
    }
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
    int res;
    if (dm_occ_decided((double)k * current_sum, c->xi, &res)) {
        *is_ho = res;
    } else {
        for (size_t i = 0; i < size; i++) c->len_scratch[i] = c->g_tsize[tids[i]];
        *is_ho = dm_occ_ge_exact(k, c->len_scratch, NULL, size, c->xi);
    }
    return true;
}

static void mine_depth_hois(DFHOICtx *c, DFHOIItemset *classes, size_t class_size) {
    if (class_size < 2) return;
    for (size_t i = 0; i < class_size; i++) {
        DFHOIItemset *P1 = &classes[i];
        c->prefix[P1->length - 1] = P1->last;
        DFHOIItemset *children = NULL;
        if (class_size - i - 1 > 0) children = malloc((class_size - i - 1) * sizeof(DFHOIItemset));
        size_t child_count = 0;

        for (size_t j = i + 1; j < class_size; j++) {
            DFHOIItemset *P2 = &classes[j];
            c->visited++;
            size_t p1_idx = 0, p2_idx = 0, new_idx = 0;
            while (p1_idx < P1->num_tids && p2_idx < P2->num_tids) {
                if (P1->tids[p1_idx] < P2->tids[p2_idx]) p1_idx++;
                else if (P1->tids[p1_idx] > P2->tids[p2_idx]) p2_idx++;
                else { c->buffer[new_idx++] = P1->tids[p1_idx]; p1_idx++; p2_idx++; }
            }
            int is_ho = 0;
            size_t new_k = P1->length + 1;
            if (check_ubo_and_o(c, c->buffer, new_idx, new_k, &is_ho)) {
                DFHOIItemset *P = &children[child_count++];
                P->last = P2->last;
                P->length = new_k;
                P->num_tids = new_idx;
                P->tids = malloc(new_idx * sizeof(uint32_t) + 1);
                memcpy(P->tids, c->buffer, new_idx * sizeof(uint32_t));
                if (is_ho) {
                    c->ho_count++;
                    c->ho_footprint += new_k;
                    c->prefix[new_k - 1] = P->last;
                    emit(c, new_k);
                }
            }
        }

        if (child_count > 0) {
            mine_depth_hois(c, children, child_count);
            for (size_t ch = 0; ch < child_count; ch++) free(children[ch].tids);
        }
        if (children) free(children);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_DFHOI_Params *dfhoi_params = (DM_DFHOI_Params *)params;
    double min_occ_param = dfhoi_params ? dfhoi_params->min_occupancy : 0.01;
    DFHOICtx c;
    memset(&c, 0, sizeof c);
    c.xi = dm_threshold_make(min_occ_param, ds->count, min_occ_param < 1.0);
    c.sigma = dm_threshold_ceil(c.xi);

    printf("[DFHOI] Starting on %zu transactions. Min Occupancy Threshold: %.6f (min support %zu)\n",
           ds->count, c.xi.value, c.sigma);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;

    TransMeta *meta = malloc(ds->count * sizeof(TransMeta));
    c.same_length = true;
    c.first_len = data[0].count;
    for (size_t i = 0; i < ds->count; i++) {
        meta[i].orig_tid = i;
        meta[i].len = data[i].count;
        if (data[i].count != c.first_len) c.same_length = false;
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
    free(counts);

    const char *outp = getenv("DM_HOI_OUT");
    if (outp) c.out = fopen(outp, "w");
    c.out_scratch = malloc((m + 1) * sizeof(uint32_t));
    c.prefix = malloc((m + 1) * sizeof(uint32_t));

    DFHOIItemset *C1 = malloc((m + 1) * sizeof(DFHOIItemset));
    size_t c1_count = 0;
    for (size_t r = 0; r < m; r++) {
        int is_ho = 0;
        if (check_ubo_and_o(&c, item_tids[r], ic[r].count, 1, &is_ho)) {
            DFHOIItemset *it = &C1[c1_count++];
            it->last = (uint32_t)r;
            it->length = 1;
            it->num_tids = ic[r].count;
            it->tids = malloc(ic[r].count * sizeof(uint32_t));
            memcpy(it->tids, item_tids[r], ic[r].count * sizeof(uint32_t));
            if (is_ho) {
                c.ho_count++;
                c.ho_footprint += 1;
                c.prefix[0] = (uint32_t)r;
                emit(&c, 1);
            }
        }
        free(item_tids[r]);
    }
    free(item_tids);
    free(ic);

    c.buffer = malloc((ds->count + 1) * sizeof(uint32_t));
    mine_depth_hois(&c, C1, c1_count);
    free(c.buffer);

    for (size_t i = 0; i < c1_count; i++) free(C1[i].tids);
    free(C1);

    if (c.out) fclose(c.out);
    free(c.out_scratch);
    free(c.prefix);
    free(c.rank_to_id);
    free(c.len_scratch);
    free(c.g_tsize);
    printf("[DFHOI] Complete. Total high occupancy itemsets found: %zu (visited_nodes=%zu)\n", c.ho_count, c.visited);
    dm_bench_record_results(c.ho_count, c.ho_footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "dfhoi",
    .name = "DFHOI Algorithm",
    .description = "Depth First Search for High Occupancy Itemset Mining (Nguyen et al. 2023).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
