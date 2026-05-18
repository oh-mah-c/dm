#include "algorithms/closed_fhuim_kinana.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t tid;
    double iutil;
    double rutil;
} KFHTuple;

typedef struct {
    uint32_t *items;
    size_t count;
    KFHTuple *tuples;
    size_t tuple_count;
    size_t tuple_capacity;
    double sum_iutil;
    double sum_rutil;
} KFHList;

typedef struct {
    uint32_t *items;
    size_t count;
    size_t support;
    double utility;
} KFHPattern;

typedef struct {
    KFHPattern *patterns;
    size_t count;
    size_t capacity;
} KFHPatternList;

typedef struct {
    uint32_t id;
    double twu;
    size_t support;
    double utility;
    double owl;
    double osr;
} KFHItemInfo;

typedef struct {
    double min_utility;
    int min_support;
    double min_owl;
    KFHPatternList all_hfui;
    size_t candidates;
    size_t pruned_osr;
    size_t pruned_owl;
    size_t pruned_msu;
} KFHContext;

static uint32_t *g_rank = NULL;

static int cmp_item_info(const void *a, const void *b) {
    const KFHItemInfo *ia = (const KFHItemInfo *)a;
    const KFHItemInfo *ib = (const KFHItemInfo *)b;
    if (ia->twu < ib->twu) return -1;
    if (ia->twu > ib->twu) return 1;
    return (ia->id < ib->id) ? -1 : ((ia->id > ib->id) ? 1 : 0);
}

static int cmp_trans_item_rank(const void *a, const void *b) {
    const DM_Item *ia = (const DM_Item *)a;
    const DM_Item *ib = (const DM_Item *)b;
    uint32_t ra = g_rank[ia->id];
    uint32_t rb = g_rank[ib->id];
    if (ra < rb) return -1;
    if (ra > rb) return 1;
    return 0;
}

static void list_tuple_add(KFHList *list, uint32_t tid, double iutil, double rutil) {
    if (list->tuple_count >= list->tuple_capacity) {
        list->tuple_capacity = list->tuple_capacity ? list->tuple_capacity * 2 : 8;
        list->tuples = realloc(list->tuples, list->tuple_capacity * sizeof(KFHTuple));
    }
    list->tuples[list->tuple_count].tid = tid;
    list->tuples[list->tuple_count].iutil = iutil;
    list->tuples[list->tuple_count].rutil = rutil;
    list->tuple_count++;
    list->sum_iutil += iutil;
    list->sum_rutil += rutil;
}

static KFHList *list_new_single(uint32_t item) {
    KFHList *list = calloc(1, sizeof(KFHList));
    if (!list) return NULL;
    list->items = malloc(sizeof(uint32_t));
    if (!list->items) {
        free(list);
        return NULL;
    }
    list->items[0] = item;
    list->count = 1;
    return list;
}

static void list_free(KFHList *list) {
    if (!list) return;
    free(list->items);
    free(list->tuples);
    free(list);
}

static KFHList *construct_list(KFHList *p, KFHList *px, KFHList *py) {
    KFHList *pxy = calloc(1, sizeof(KFHList));
    if (!pxy) return NULL;
    pxy->count = px->count + 1;
    pxy->items = malloc(pxy->count * sizeof(uint32_t));
    if (!pxy->items) {
        free(pxy);
        return NULL;
    }
    memcpy(pxy->items, px->items, px->count * sizeof(uint32_t));
    pxy->items[px->count] = py->items[py->count - 1];
    pxy->tuple_capacity = px->tuple_count < py->tuple_count ? px->tuple_count : py->tuple_count;
    pxy->tuples = pxy->tuple_capacity ? malloc(pxy->tuple_capacity * sizeof(KFHTuple)) : NULL;

    size_t ix = 0;
    size_t iy = 0;
    size_t ip = 0;
    while (ix < px->tuple_count && iy < py->tuple_count) {
        if (px->tuples[ix].tid == py->tuples[iy].tid) {
            uint32_t tid = px->tuples[ix].tid;
            double iutil = px->tuples[ix].iutil + py->tuples[iy].iutil;
            if (p) {
                while (ip < p->tuple_count && p->tuples[ip].tid < tid) ip++;
                if (ip < p->tuple_count && p->tuples[ip].tid == tid) {
                    iutil -= p->tuples[ip].iutil;
                }
            }
            list_tuple_add(pxy, tid, iutil, py->tuples[iy].rutil);
            ix++;
            iy++;
        } else if (px->tuples[ix].tid < py->tuples[iy].tid) {
            ix++;
        } else {
            iy++;
        }
    }
    return pxy;
}

static void patterns_add(KFHPatternList *out, const KFHList *list) {
    if (out->count >= out->capacity) {
        out->capacity = out->capacity ? out->capacity * 2 : 1024;
        out->patterns = realloc(out->patterns, out->capacity * sizeof(KFHPattern));
    }
    KFHPattern *p = &out->patterns[out->count++];
    p->items = malloc(list->count * sizeof(uint32_t));
    memcpy(p->items, list->items, list->count * sizeof(uint32_t));
    p->count = list->count;
    p->support = list->tuple_count;
    p->utility = list->sum_iutil;
}

static void patterns_free(KFHPatternList *list) {
    for (size_t i = 0; i < list->count; i++) free(list->patterns[i].items);
    free(list->patterns);
}

static int pattern_contains_all(const KFHPattern *y, const KFHPattern *x) {
    size_t i = 0;
    size_t j = 0;
    while (i < x->count && j < y->count) {
        if (x->items[i] == y->items[j]) {
            i++;
            j++;
        } else if (x->items[i] > y->items[j]) {
            j++;
        } else {
            return 0;
        }
    }
    return i == x->count;
}

static int is_closed_pattern(const KFHPatternList *all, size_t idx) {
    const KFHPattern *x = &all->patterns[idx];
    for (size_t i = 0; i < all->count; i++) {
        const KFHPattern *y = &all->patterns[i];
        if (i == idx || y->count <= x->count) continue;
        if (y->support == x->support && fabs(y->utility - x->utility) <= 1e-9 && pattern_contains_all(y, x)) {
            return 0;
        }
    }
    return 1;
}

static void search(KFHContext *ctx, KFHList *p, KFHList **extensions, size_t ext_count) {
    for (size_t i = 0; i < ext_count; i++) {
        KFHList *px = extensions[i];
        ctx->candidates++;

        if ((int)px->tuple_count >= ctx->min_support && px->sum_iutil >= ctx->min_utility) {
            patterns_add(&ctx->all_hfui, px);
        }

        if ((int)px->tuple_count >= ctx->min_support && px->sum_iutil + px->sum_rutil >= ctx->min_utility) {
            size_t child_cap = ext_count - i - 1;
            KFHList **children = child_cap ? malloc(child_cap * sizeof(KFHList *)) : NULL;
            size_t child_count = 0;
            for (size_t j = i + 1; j < ext_count; j++) {
                KFHList *child = construct_list(p, px, extensions[j]);
                if (child && child->tuple_count > 0) {
                    double msu = child->sum_iutil + child->sum_rutil;
                    if ((int)child->tuple_count >= ctx->min_support && msu >= ctx->min_utility) {
                        children[child_count++] = child;
                    } else {
                        ctx->pruned_msu++;
                        list_free(child);
                    }
                } else {
                    list_free(child);
                }
            }
            if (child_count > 0) search(ctx, px, children, child_count);
            for (size_t j = 0; j < child_count; j++) list_free(children[j]);
            free(children);
        } else {
            ctx->pruned_msu++;
        }
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_UTILITY || !ds->payload) return DM_ERROR_INCOMPATIBLE;

    DM_ClosedFHUIMKinana_Params *p = (DM_ClosedFHUIMKinana_Params *)params;
    double min_util = p ? p->min_utility : 10000.0;
    int min_sup = p && p->min_support > 0 ? p->min_support : 20;
    double min_owl = p ? p->min_owl : 0.0;
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;

    size_t item_slots = (size_t)ds->max_id + 1;
    double *twu = calloc(item_slots, sizeof(double));
    double *util = calloc(item_slots, sizeof(double));
    double *tidset_tu = calloc(item_slots, sizeof(double));
    size_t *support = calloc(item_slots, sizeof(size_t));
    if (!twu || !util || !tidset_tu || !support) return DM_ERROR_MEMORY;

    for (size_t tid = 0; tid < ds->count; tid++) {
        for (size_t j = 0; j < data[tid].count; j++) {
            uint32_t item = data[tid].items[j].id;
            twu[item] += data[tid].total_utility;
            util[item] += data[tid].items[j].utility;
            tidset_tu[item] += data[tid].total_utility;
            support[item]++;
        }
    }

    KFHItemInfo *items = malloc(item_slots * sizeof(KFHItemInfo));
    size_t item_count = 0;
    KFHContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.min_utility = min_util;
    ctx.min_support = min_sup;
    ctx.min_owl = min_owl;

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (support[i] == 0) continue;
        double osr = (double)support[i] / (double)ds->count;
        double owl = tidset_tu[i] != 0.0 ? util[i] / tidset_tu[i] : 0.0;
        if ((int)support[i] < min_sup || twu[i] < min_util) {
            ctx.pruned_osr++;
            continue;
        }
        if (owl < min_owl) {
            ctx.pruned_owl++;
            continue;
        }
        items[item_count].id = i;
        items[item_count].twu = twu[i];
        items[item_count].support = support[i];
        items[item_count].utility = util[i];
        items[item_count].owl = owl;
        items[item_count].osr = osr;
        item_count++;
    }
    qsort(items, item_count, sizeof(KFHItemInfo), cmp_item_info);

    g_rank = malloc(item_slots * sizeof(uint32_t));
    memset(g_rank, 0xFF, item_slots * sizeof(uint32_t));
    KFHList **initial = malloc(item_count * sizeof(KFHList *));
    for (size_t i = 0; i < item_count; i++) {
        g_rank[items[i].id] = (uint32_t)i;
        initial[i] = list_new_single(items[i].id);
    }

    for (size_t tid = 0; tid < ds->count; tid++) {
        DM_Item *filtered = malloc(data[tid].count * sizeof(DM_Item));
        size_t fcount = 0;
        for (size_t j = 0; j < data[tid].count; j++) {
            if (g_rank[data[tid].items[j].id] != 0xFFFFFFFFu) filtered[fcount++] = data[tid].items[j];
        }
        qsort(filtered, fcount, sizeof(DM_Item), cmp_trans_item_rank);
        double remaining = 0.0;
        for (size_t j = fcount; j-- > 0;) {
            uint32_t idx = g_rank[filtered[j].id];
            list_tuple_add(initial[idx], (uint32_t)tid, filtered[j].utility, remaining);
            remaining += filtered[j].utility;
        }
        free(filtered);
    }

    printf("[Closed-FHUIM-Kinana] Starting on %zu transactions. minutil=%.6g minsup=%d min_owl=%.6g promising_items=%zu\n",
           ds->count, min_util, min_sup, min_owl, item_count);

    search(&ctx, NULL, initial, item_count);

    size_t closed_count = 0;
    size_t total_items = 0;
    for (size_t i = 0; i < ctx.all_hfui.count; i++) {
        if (is_closed_pattern(&ctx.all_hfui, i)) {
            closed_count++;
            total_items += ctx.all_hfui.patterns[i].count;
        }
    }

    printf("[Closed-FHUIM-Kinana] Complete. HFUIs=%zu Closed-HFUIs=%zu candidates=%zu pruned_osr=%zu pruned_owl=%zu pruned_msu=%zu\n",
           ctx.all_hfui.count, closed_count, ctx.candidates, ctx.pruned_osr, ctx.pruned_owl, ctx.pruned_msu);

    dm_bench_record_results(closed_count, total_items);

    for (size_t i = 0; i < item_count; i++) list_free(initial[i]);
    patterns_free(&ctx.all_hfui);
    free(initial);
    free(items);
    free(g_rank);
    g_rank = NULL;
    free(twu);
    free(util);
    free(tidset_tu);
    free(support);
    return DM_SUCCESS;
}

DM_Algorithm closed_fhuim_kinana_algo = {
    .id = "closed_fhuim_kinana",
    .name = "Closed-FHUIM-Kinana",
    .description = "Closed frequent high-utility itemset mining with OSR, OWL, and MSU pruning.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(closed_fhuim_kinana_algo)
