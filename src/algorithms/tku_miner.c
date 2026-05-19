#include "algorithms/tku_miner.h"
#include "core/dm_dataset_types.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int tid;
    double iutil;
    double rutil;
} TKUEntry;

typedef struct {
    uint32_t item;
    TKUEntry *entries;
    size_t count;
    double sum_iutil;
    double sum_rutil;
} TKUList;

typedef struct {
    uint32_t *items;
    size_t len;
    double utility;
} TKUPattern;

typedef struct {
    TKUPattern *data;
    size_t count;
    size_t cap;
    size_t k;
    double threshold;
    size_t threshold_raises;
} TopK;

typedef struct {
    DM_Trans_Utility *trans;
    size_t transactions;
    uint32_t max_id;
    uint32_t *rank;
    double *twu;
    double *item_util;
    TopK topk;
    DM_TKU_Stats *stats;
    DM_TKU_Params params;
    clock_t start_clock;
} TKUCtx;

static double elapsed_sec(TKUCtx *ctx) {
    return (double)(clock() - ctx->start_clock) / (double)CLOCKS_PER_SEC;
}

static int limited(TKUCtx *ctx) {
    if (ctx->params.max_seconds > 0.0 && elapsed_sec(ctx) >= ctx->params.max_seconds) {
        ctx->stats->limited = 1;
        return 1;
    }
    return 0;
}

static int cmp_pattern_desc(const void *a, const void *b) {
    const TKUPattern *x = (const TKUPattern *)a;
    const TKUPattern *y = (const TKUPattern *)b;
    if (x->utility < y->utility) return 1;
    if (x->utility > y->utility) return -1;
    if (x->len < y->len) return -1;
    if (x->len > y->len) return 1;
    return 0;
}

static int same_items(const TKUPattern *p, const uint32_t *items, size_t len) {
    if (p->len != len) return 0;
    for (size_t i = 0; i < len; i++) {
        if (p->items[i] != items[i]) return 0;
    }
    return 1;
}

static void topk_free(TopK *t) {
    for (size_t i = 0; i < t->count; i++) free(t->data[i].items);
    free(t->data);
    memset(t, 0, sizeof(*t));
}

static int topk_add(TopK *t, const uint32_t *items, size_t len, double utility) {
    if (t->k == 0) return 0;
    if (t->count >= t->k && utility < t->threshold) return 0;
    for (size_t i = 0; i < t->count; i++) {
        if (same_items(&t->data[i], items, len)) return 0;
    }
    if (t->count == t->cap) {
        size_t nc = t->cap ? t->cap * 2 : 64;
        TKUPattern *nd = (TKUPattern *)realloc(t->data, nc * sizeof(*nd));
        if (!nd) return -1;
        t->data = nd;
        t->cap = nc;
    }
    t->data[t->count].items = (uint32_t *)malloc(len * sizeof(uint32_t));
    if (!t->data[t->count].items) return -1;
    memcpy(t->data[t->count].items, items, len * sizeof(uint32_t));
    t->data[t->count].len = len;
    t->data[t->count].utility = utility;
    t->count++;
    qsort(t->data, t->count, sizeof(*t->data), cmp_pattern_desc);
    if (t->count >= t->k) {
        double old = t->threshold;
        t->threshold = t->data[t->k - 1].utility;
        if (t->threshold > old) t->threshold_raises++;
        size_t keep = t->count;
        while (keep > 0 && keep > t->k && t->data[keep - 1].utility < t->threshold) {
            free(t->data[keep - 1].items);
            keep--;
        }
        t->count = keep;
    }
    return 0;
}

typedef struct {
    uint32_t id;
    double twu;
} ItemTWU;

static int cmp_item_twu_desc(const void *a, const void *b) {
    const ItemTWU *x = (const ItemTWU *)a;
    const ItemTWU *y = (const ItemTWU *)b;
    if (x->twu < y->twu) return 1;
    if (x->twu > y->twu) return -1;
    if (x->id > y->id) return 1;
    if (x->id < y->id) return -1;
    return 0;
}

static double kth_value_desc(double *vals, size_t n, size_t k) {
    if (k == 0 || n < k) return 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (vals[j] > vals[i]) {
                double tmp = vals[i];
                vals[i] = vals[j];
                vals[j] = tmp;
            }
        }
    }
    return vals[k - 1];
}

static double pre_evaluation_threshold(DM_Trans_Utility *tr, size_t n, size_t k) {
    double *vals = NULL;
    size_t count = 0, cap = 0;
    for (size_t t = 0; t < n; t++) {
        if (tr[t].count < 2) continue;
        double first = tr[t].items[0].utility;
        for (size_t j = 1; j < tr[t].count; j++) {
            if (count == cap) {
                size_t nc = cap ? cap * 2 : 1024;
                double *nv = (double *)realloc(vals, nc * sizeof(double));
                if (!nv) {
                    free(vals);
                    return 0.0;
                }
                vals = nv;
                cap = nc;
            }
            vals[count++] = first + tr[t].items[j].utility;
        }
    }
    double kth = kth_value_desc(vals, count, k);
    free(vals);
    return kth;
}

static TKUList *construct_list(TKUCtx *ctx, TKUList *p, TKUList *px, TKUList *py) {
    TKUList *out = (TKUList *)calloc(1, sizeof(*out));
    if (!out) return NULL;
    out->item = py->item;
    size_t cap = px->count < py->count ? px->count : py->count;
    out->entries = (TKUEntry *)malloc(cap * sizeof(TKUEntry));
    if (!out->entries) {
        free(out);
        return NULL;
    }
    size_t ix = 0, iy = 0, ip = 0;
    while (ix < px->count && iy < py->count) {
        if (px->entries[ix].tid < py->entries[iy].tid) {
            ix++;
        } else if (px->entries[ix].tid > py->entries[iy].tid) {
            iy++;
        } else {
            double putil = 0.0;
            if (p) {
                while (ip < p->count && p->entries[ip].tid < px->entries[ix].tid) ip++;
                if (ip < p->count && p->entries[ip].tid == px->entries[ix].tid) putil = p->entries[ip].iutil;
            }
            TKUEntry e;
            e.tid = px->entries[ix].tid;
            e.iutil = px->entries[ix].iutil + py->entries[iy].iutil - putil;
            e.rutil = py->entries[iy].rutil;
            out->entries[out->count++] = e;
            out->sum_iutil += e.iutil;
            out->sum_rutil += e.rutil;
            ix++;
            iy++;
            ctx->stats->joined_entries++;
        }
    }
    ctx->stats->joins++;
    return out;
}

static void free_lists(TKUList **lists, size_t n) {
    if (!lists) return;
    for (size_t i = 0; i < n; i++) {
        if (lists[i]) {
            free(lists[i]->entries);
            free(lists[i]);
        }
    }
    free(lists);
}

static int explore(TKUCtx *ctx, TKUList *prefix_list, uint32_t *prefix, size_t plen,
                   TKUList **lists, size_t nlists) {
    if (limited(ctx)) return 0;
    if (ctx->params.max_depth > 0 && plen >= ctx->params.max_depth) return 0;
    for (size_t i = 0; i < nlists; i++) {
        TKUList *x = lists[i];
        ctx->stats->visited_nodes++;
        uint32_t *next = (uint32_t *)malloc((plen + 1) * sizeof(uint32_t));
        if (!next) return -1;
        if (plen) memcpy(next, prefix, plen * sizeof(uint32_t));
        next[plen] = x->item;
        ctx->stats->candidates++;
        if (x->sum_iutil >= ctx->topk.threshold) {
            if (topk_add(&ctx->topk, next, plen + 1, x->sum_iutil) != 0) {
                free(next);
                return -1;
            }
        }
        if (x->sum_iutil + x->sum_rutil >= ctx->topk.threshold) {
            TKUList **ext = NULL;
            size_t ec = 0, ecap = 0;
            for (size_t j = i + 1; j < nlists; j++) {
                TKUList *xy = construct_list(ctx, prefix_list, x, lists[j]);
                if (!xy) {
                    free(next);
                    free_lists(ext, ec);
                    return -1;
                }
                if (xy->count > 0 && xy->sum_iutil + xy->sum_rutil >= ctx->topk.threshold) {
                    if (ec == ecap) {
                        size_t nc = ecap ? ecap * 2 : 16;
                        TKUList **ne = (TKUList **)realloc(ext, nc * sizeof(*ext));
                        if (!ne) {
                            free(xy->entries);
                            free(xy);
                            free(next);
                            free_lists(ext, ec);
                            return -1;
                        }
                        ext = ne;
                        ecap = nc;
                    }
                    ext[ec++] = xy;
                } else {
                    ctx->stats->pruned_subtree_utility++;
                    free(xy->entries);
                    free(xy);
                }
            }
            if (ec > 0 && explore(ctx, x, next, plen + 1, ext, ec) != 0) {
                free(next);
                free_lists(ext, ec);
                return -1;
            }
            free_lists(ext, ec);
        } else {
            ctx->stats->pruned_subtree_utility++;
        }
        free(next);
    }
    return 0;
}

int tku_mine_dataset(DM_Dataset *ds, const DM_TKU_Params *params, DM_TKU_Stats *stats) {
    if (!ds || ds->type != DM_TYPE_UTILITY || !stats) return -1;
    memset(stats, 0, sizeof(*stats));
    DM_TKU_Params p = {0};
    p.k = params && params->k ? params->k : 10;
    p.max_depth = params ? params->max_depth : 0;
    p.max_seconds = params ? params->max_seconds : 0.0;
    DM_Trans_Utility *tr = (DM_Trans_Utility *)ds->payload;
    TKUCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.trans = tr;
    ctx.transactions = ds->count;
    ctx.max_id = ds->max_id;
    ctx.params = p;
    ctx.stats = stats;
    ctx.start_clock = clock();
    ctx.twu = (double *)calloc(ds->max_id + 1, sizeof(double));
    ctx.item_util = (double *)calloc(ds->max_id + 1, sizeof(double));
    ctx.rank = (uint32_t *)malloc((ds->max_id + 1) * sizeof(uint32_t));
    ItemTWU *items = (ItemTWU *)malloc((ds->max_id + 1) * sizeof(ItemTWU));
    if (!ctx.twu || !ctx.item_util || !ctx.rank || !items) {
        free(ctx.twu); free(ctx.item_util); free(ctx.rank); free(items);
        return -1;
    }
    for (size_t i = 0; i <= ds->max_id; i++) ctx.rank[i] = UINT32_MAX;
    for (size_t t = 0; t < ds->count; t++) {
        for (size_t i = 0; i < tr[t].count; i++) {
            uint32_t id = tr[t].items[i].id;
            ctx.twu[id] += tr[t].total_utility;
            ctx.item_util[id] += tr[t].items[i].utility;
        }
    }
    size_t nitems = 0;
    for (uint32_t id = 0; id <= ds->max_id; id++) {
        if (ctx.twu[id] > 0.0) {
            items[nitems].id = id;
            items[nitems].twu = ctx.twu[id];
            nitems++;
        }
    }
    qsort(items, nitems, sizeof(*items), cmp_item_twu_desc);
    for (size_t i = 0; i < nitems; i++) ctx.rank[items[i].id] = (uint32_t)i;
    ctx.topk.k = p.k;
    stats->pe_threshold = pre_evaluation_threshold(tr, ds->count, p.k);
    double *singletons = (double *)malloc(nitems * sizeof(double));
    if (!singletons) {
        free(ctx.twu); free(ctx.item_util); free(ctx.rank); free(items);
        return -1;
    }
    for (size_t i = 0; i < nitems; i++) singletons[i] = ctx.item_util[items[i].id];
    stats->singleton_threshold = kth_value_desc(singletons, nitems, p.k);
    ctx.topk.threshold = fmax(stats->pe_threshold, stats->singleton_threshold);
    for (size_t i = 0; i < nitems; i++) {
        uint32_t id = items[i].id;
        if (ctx.item_util[id] >= ctx.topk.threshold) topk_add(&ctx.topk, &id, 1, ctx.item_util[id]);
    }
    TKUList **uls = (TKUList **)calloc(nitems, sizeof(*uls));
    size_t kept = 0;
    for (size_t i = 0; i < nitems; i++) {
        if (ctx.twu[items[i].id] < ctx.topk.threshold) {
            stats->pruned_twu++;
            continue;
        }
        TKUList *ul = (TKUList *)calloc(1, sizeof(*ul));
        if (!ul) {
            free_lists(uls, kept); free(singletons); topk_free(&ctx.topk);
            free(ctx.twu); free(ctx.item_util); free(ctx.rank); free(items);
            return -1;
        }
        ul->item = items[i].id;
        ul->entries = (TKUEntry *)malloc(ds->count * sizeof(TKUEntry));
        if (!ul->entries) {
            free(ul); free_lists(uls, kept); free(singletons); topk_free(&ctx.topk);
            free(ctx.twu); free(ctx.item_util); free(ctx.rank); free(items);
            return -1;
        }
        uls[kept++] = ul;
    }
    for (size_t t = 0; t < ds->count; t++) {
        size_t len = tr[t].count;
        uint32_t *ids = (uint32_t *)malloc(len * sizeof(uint32_t));
        double *utils = (double *)malloc(len * sizeof(double));
        if (!ids || !utils) {
            free(ids); free(utils); free_lists(uls, kept); free(singletons); topk_free(&ctx.topk);
            free(ctx.twu); free(ctx.item_util); free(ctx.rank); free(items);
            return -1;
        }
        size_t m = 0;
        for (size_t i = 0; i < len; i++) {
            uint32_t id = tr[t].items[i].id;
            if (ctx.rank[id] != UINT32_MAX && ctx.twu[id] >= ctx.topk.threshold) {
                ids[m] = id;
                utils[m] = tr[t].items[i].utility;
                m++;
            }
        }
        for (size_t i = 0; i < m; i++) {
            for (size_t j = i + 1; j < m; j++) {
                if (ctx.rank[ids[i]] > ctx.rank[ids[j]]) {
                    uint32_t ti = ids[i]; ids[i] = ids[j]; ids[j] = ti;
                    double tu = utils[i]; utils[i] = utils[j]; utils[j] = tu;
                }
            }
        }
        double remaining = 0.0;
        for (size_t r = m; r > 0; r--) {
            size_t i = r - 1;
            uint32_t rank = ctx.rank[ids[i]];
            TKUList *ul = NULL;
            for (size_t u = 0; u < kept; u++) {
                if (ctx.rank[uls[u]->item] == rank) {
                    ul = uls[u];
                    break;
                }
            }
            if (ul) {
                TKUEntry e;
                e.tid = (int)t;
                e.iutil = utils[i];
                e.rutil = remaining;
                ul->entries[ul->count++] = e;
                ul->sum_iutil += e.iutil;
                ul->sum_rutil += e.rutil;
            }
            remaining += utils[i];
        }
        free(ids);
        free(utils);
    }
    explore(&ctx, NULL, NULL, 0, uls, kept);
    qsort(ctx.topk.data, ctx.topk.count, sizeof(*ctx.topk.data), cmp_pattern_desc);
    stats->transactions = ds->count;
    stats->distinct_items = nitems;
    stats->k = p.k;
    stats->final_threshold = ctx.topk.threshold;
    stats->output_count = ctx.topk.count;
    for (size_t i = 0; i < ctx.topk.count; i++) {
        stats->total_output_items += ctx.topk.data[i].len;
        stats->avg_utility += ctx.topk.data[i].utility;
        if (i == 0) stats->best_utility = ctx.topk.data[i].utility;
    }
    if (ctx.topk.count) {
        stats->avg_utility /= (double)ctx.topk.count;
        stats->avg_length = (double)stats->total_output_items / (double)ctx.topk.count;
    }
    stats->threshold_raises = ctx.topk.threshold_raises;
    stats->phase2_checked = stats->candidates;
    stats->result_ram_bytes = ctx.topk.count * sizeof(TKUPattern) + stats->total_output_items * sizeof(uint32_t);
    stats->result_disk_est_bytes = ctx.topk.count * 32 + stats->total_output_items * 12;
    free_lists(uls, kept);
    free(singletons);
    topk_free(&ctx.topk);
    free(ctx.twu);
    free(ctx.item_util);
    free(ctx.rank);
    free(items);
    return 0;
}
