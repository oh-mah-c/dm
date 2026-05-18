#include "algorithms/vifp.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VIFP_NO_PRED UINT32_MAX

typedef struct {
    uint32_t tid;
    uint32_t pos;
    uint32_t item;
    uint32_t pred;
    uint8_t live;
} VIFPOccurrence;

typedef struct {
    uint32_t item;
    uint32_t support;
    uint32_t rank;
    size_t start;
    size_t end;
    uint8_t frequent;
} VIFPItemInfo;

typedef struct {
    uint32_t id;
    uint32_t depth;
    size_t start;
    size_t end;
    size_t cap;
} VIFPPID;

typedef struct {
    uint32_t *idx;
    size_t count;
    size_t cap;
} VIFPIndexBuffer;

typedef struct {
    VIFPOccurrence *ott;
    size_t occ_count;
    uint32_t *hpa_idx;
    VIFPItemInfo *items;
    uint32_t max_item;
    uint32_t minsup;
    VIFPMode mode;
    size_t max_itemsets;
    double max_seconds;
    clock_t started;
    VIFPStats *stats;
} VIFPContext;

static const VIFPOccurrence *g_sort_ott = NULL;

static void buf_init(VIFPIndexBuffer *b) {
    b->idx = NULL;
    b->count = 0;
    b->cap = 0;
}

static int buf_push(VIFPIndexBuffer *b, uint32_t v) {
    if (b->count >= b->cap) {
        size_t next = b->cap ? b->cap * 2 : 64;
        uint32_t *tmp = (uint32_t *)realloc(b->idx, next * sizeof(uint32_t));
        if (!tmp) return -1;
        b->idx = tmp;
        b->cap = next;
    }
    b->idx[b->count++] = v;
    return 0;
}

static void buf_free(VIFPIndexBuffer *b) {
    free(b->idx);
    b->idx = NULL;
    b->count = 0;
    b->cap = 0;
}

static int mode_multiplier(VIFPMode mode) {
    if (mode == VIFP_MODE_FHE) return 4096;
    if (mode == VIFP_MODE_SMPC) return 24;
    return 1;
}

int vifp_parse_mode(const char *name, VIFPMode *mode) {
    if (!name || strcmp(name, "plaintext") == 0 || strcmp(name, "plain") == 0) {
        *mode = VIFP_MODE_PLAINTEXT;
        return 0;
    }
    if (strcmp(name, "smpc") == 0 || strcmp(name, "secret_sharing") == 0) {
        *mode = VIFP_MODE_SMPC;
        return 0;
    }
    if (strcmp(name, "fhe") == 0) {
        *mode = VIFP_MODE_FHE;
        return 0;
    }
    return -1;
}

const char *vifp_mode_name(VIFPMode mode) {
    switch (mode) {
        case VIFP_MODE_SMPC: return "smpc";
        case VIFP_MODE_FHE: return "fhe";
        default: return "plaintext";
    }
}

static int item_info_cmp(const void *a, const void *b) {
    const VIFPItemInfo *ia = (const VIFPItemInfo *)a;
    const VIFPItemInfo *ib = (const VIFPItemInfo *)b;
    if (ia->support > ib->support) return -1;
    if (ia->support < ib->support) return 1;
    if (ia->item < ib->item) return -1;
    if (ia->item > ib->item) return 1;
    return 0;
}

static int hpa_idx_cmp(const void *a, const void *b) {
    const VIFPOccurrence *ra = &g_sort_ott[*(const uint32_t *)a];
    const VIFPOccurrence *rb = &g_sort_ott[*(const uint32_t *)b];
    if (ra->item < rb->item) return -1;
    if (ra->item > rb->item) return 1;
    if (ra->tid < rb->tid) return -1;
    if (ra->tid > rb->tid) return 1;
    if (ra->pos < rb->pos) return -1;
    if (ra->pos > rb->pos) return 1;
    return 0;
}

static int idx_item_cmp(const void *a, const void *b) {
    const VIFPOccurrence *ra = &g_sort_ott[*(const uint32_t *)a];
    const VIFPOccurrence *rb = &g_sort_ott[*(const uint32_t *)b];
    if (ra->item < rb->item) return -1;
    if (ra->item > rb->item) return 1;
    if (ra->tid < rb->tid) return -1;
    if (ra->tid > rb->tid) return 1;
    if (ra->pos < rb->pos) return -1;
    if (ra->pos > rb->pos) return 1;
    return 0;
}

static int sort_items_by_rank(uint32_t *items, size_t count, VIFPItemInfo *info) {
    for (size_t i = 1; i < count; i++) {
        uint32_t key = items[i];
        size_t j = i;
        while (j > 0 && info[items[j - 1]].rank > info[key].rank) {
            items[j] = items[j - 1];
            j--;
        }
        items[j] = key;
    }
    return 0;
}

static int time_limited(VIFPContext *ctx) {
    if (ctx->stats->limited) return 1;
    if (ctx->max_itemsets && ctx->stats->emitted_itemsets >= ctx->max_itemsets) {
        ctx->stats->limited = 1;
        return 1;
    }
    if (ctx->max_seconds > 0.0) {
        double elapsed = (double)(clock() - ctx->started) / (double)CLOCKS_PER_SEC;
        if (elapsed >= ctx->max_seconds) {
            ctx->stats->limited = 1;
            return 1;
        }
    }
    return 0;
}

static void estimate_crypto_costs(VIFPContext *ctx) {
    int mul = mode_multiplier(ctx->mode);
    VIFPStats *s = ctx->stats;
    size_t logical_records = s->occurrence_records + s->hpa_records + s->cpb_records;
    s->estimated_comm_bytes = 0;
    s->estimated_ciphertext_bytes = 0;
    s->estimated_bootstraps = 0;
    if (ctx->mode == VIFP_MODE_SMPC) {
        s->estimated_comm_bytes = (logical_records * 32 + s->secure_comparisons * 16 + s->oblivious_sorts * s->occurrence_records * 8) * (size_t)mul;
    } else if (ctx->mode == VIFP_MODE_FHE) {
        s->estimated_ciphertext_bytes = (logical_records * 32 + s->secure_comparisons * 8) * (size_t)mul;
        s->estimated_bootstraps = s->secure_comparisons;
    } else {
        s->estimated_comm_bytes = logical_records * 32;
    }
}

static int build_cpb(VIFPContext *ctx, const VIFPIndexBuffer *seed, VIFPIndexBuffer *cpb) {
    for (size_t i = 0; i < seed->count; i++) {
        uint32_t p = ctx->ott[seed->idx[i]].pred;
        while (p != VIFP_NO_PRED) {
            if (buf_push(cpb, p) != 0) return -1;
            ctx->stats->predecessor_fetches++;
            p = ctx->ott[p].pred;
        }
    }
    ctx->stats->cpb_records += cpb->count;
    if (cpb->count > ctx->stats->max_cpb_records) ctx->stats->max_cpb_records = cpb->count;
    return 0;
}

static int mine_interval(VIFPContext *ctx, VIFPPID pid, const VIFPIndexBuffer *seed) {
    if (time_limited(ctx) || seed->count == 0) return 0;
    ctx->stats->projected_states++;
    ctx->stats->pid_descriptors++;
    ctx->stats->public_capacity_slots += pid.cap;
    ctx->stats->active_capacity_slots += seed->count;
    if (pid.depth > ctx->stats->max_depth) ctx->stats->max_depth = pid.depth;

    VIFPIndexBuffer cpb;
    buf_init(&cpb);
    if (build_cpb(ctx, seed, &cpb) != 0) {
        buf_free(&cpb);
        return -1;
    }
    if (cpb.count == 0) {
        buf_free(&cpb);
        return 0;
    }

    uint32_t *hist = (uint32_t *)calloc(ctx->max_item + 1, sizeof(uint32_t));
    if (!hist) {
        buf_free(&cpb);
        return -1;
    }
    for (size_t i = 0; i < cpb.count; i++) {
        hist[ctx->ott[cpb.idx[i]].item]++;
        ctx->stats->histogram_updates++;
    }

    VIFPIndexBuffer alive_items;
    buf_init(&alive_items);
    for (uint32_t item = 0; item <= ctx->max_item; item++) {
        if (hist[item] > 0) {
            ctx->stats->secure_comparisons++;
            if (hist[item] >= ctx->minsup && ctx->items[item].frequent) {
                if (buf_push(&alive_items, item) != 0) {
                    free(hist);
                    buf_free(&cpb);
                    buf_free(&alive_items);
                    return -1;
                }
                ctx->stats->emitted_itemsets++;
                ctx->stats->total_output_items += pid.depth + 1;
            }
        }
    }
    free(hist);

    if (alive_items.count == 0 || time_limited(ctx)) {
        buf_free(&alive_items);
        buf_free(&cpb);
        return 0;
    }

    g_sort_ott = ctx->ott;
    qsort(cpb.idx, cpb.count, sizeof(uint32_t), idx_item_cmp);
    ctx->stats->stable_partitions++;

    size_t start = 0;
    while (start < cpb.count) {
        uint32_t item = ctx->ott[cpb.idx[start]].item;
        size_t end = start + 1;
        while (end < cpb.count && ctx->ott[cpb.idx[end]].item == item) end++;
        int alive = 0;
        for (size_t a = 0; a < alive_items.count; a++) {
            if (alive_items.idx[a] == item) {
                alive = 1;
                break;
            }
        }
        if (alive) {
            VIFPIndexBuffer child;
            child.idx = &cpb.idx[start];
            child.count = end - start;
            child.cap = end - start;
            VIFPPID child_pid = { item, pid.depth + 1, start, end, child.count };
            mine_interval(ctx, child_pid, &child);
        }
        start = end;
        if (time_limited(ctx)) break;
    }

    buf_free(&alive_items);
    buf_free(&cpb);
    return 0;
}

static int build_structures(DM_Dataset *ds, uint32_t minsup, VIFPContext *ctx) {
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    ctx->max_item = ds->max_id;
    ctx->items = (VIFPItemInfo *)calloc((size_t)ctx->max_item + 1, sizeof(VIFPItemInfo));
    if (!ctx->items) return -1;

    for (uint32_t i = 0; i <= ctx->max_item; i++) {
        ctx->items[i].item = i;
        ctx->items[i].rank = UINT32_MAX;
    }

    size_t occ_cap = 0;
    for (size_t t = 0; t < ds->count; t++) {
        occ_cap += data[t].count;
        for (size_t j = 0; j < data[t].count; j++) {
            uint32_t item = data[t].items[j];
            if (item <= ctx->max_item) ctx->items[item].support++;
        }
    }

    VIFPItemInfo *order = (VIFPItemInfo *)malloc(((size_t)ctx->max_item + 1) * sizeof(VIFPItemInfo));
    if (!order) return -1;
    size_t f = 0;
    for (uint32_t item = 0; item <= ctx->max_item; item++) {
        if (ctx->items[item].support >= minsup) {
            ctx->items[item].frequent = 1;
            order[f++] = ctx->items[item];
        }
    }
    qsort(order, f, sizeof(VIFPItemInfo), item_info_cmp);
    ctx->stats->oblivious_sorts++;
    for (size_t r = 0; r < f; r++) {
        ctx->items[order[r].item].rank = (uint32_t)r;
    }
    ctx->stats->frequent_singletons = f;
    free(order);

    ctx->ott = (VIFPOccurrence *)malloc(sizeof(VIFPOccurrence) * (occ_cap ? occ_cap : 1));
    if (!ctx->ott) return -1;

    uint32_t *tmp = (uint32_t *)malloc(sizeof(uint32_t) * ((size_t)ctx->max_item + 1));
    if (!tmp) return -1;
    for (size_t t = 0; t < ds->count; t++) {
        size_t n = 0;
        for (size_t j = 0; j < data[t].count; j++) {
            uint32_t item = data[t].items[j];
            if (item <= ctx->max_item && ctx->items[item].frequent) tmp[n++] = item;
        }
        if (n == 0) continue;
        sort_items_by_rank(tmp, n, ctx->items);
        uint32_t pred = VIFP_NO_PRED;
        for (size_t pos = 0; pos < n; pos++) {
            uint32_t idx = (uint32_t)ctx->occ_count;
            ctx->ott[ctx->occ_count].tid = (uint32_t)t;
            ctx->ott[ctx->occ_count].pos = (uint32_t)pos;
            ctx->ott[ctx->occ_count].item = tmp[pos];
            ctx->ott[ctx->occ_count].pred = pred;
            ctx->ott[ctx->occ_count].live = 1;
            pred = idx;
            ctx->occ_count++;
        }
    }
    free(tmp);
    ctx->stats->occurrence_records = ctx->occ_count;

    ctx->hpa_idx = (uint32_t *)malloc(sizeof(uint32_t) * (ctx->occ_count ? ctx->occ_count : 1));
    if (!ctx->hpa_idx) return -1;
    for (size_t i = 0; i < ctx->occ_count; i++) ctx->hpa_idx[i] = (uint32_t)i;
    g_sort_ott = ctx->ott;
    qsort(ctx->hpa_idx, ctx->occ_count, sizeof(uint32_t), hpa_idx_cmp);
    ctx->stats->oblivious_sorts++;
    ctx->stats->hpa_records = ctx->occ_count;

    for (uint32_t item = 0; item <= ctx->max_item; item++) {
        ctx->items[item].start = ctx->items[item].end = 0;
    }
    size_t p = 0;
    while (p < ctx->occ_count) {
        uint32_t item = ctx->ott[ctx->hpa_idx[p]].item;
        size_t start = p++;
        while (p < ctx->occ_count && ctx->ott[ctx->hpa_idx[p]].item == item) p++;
        ctx->items[item].start = start;
        ctx->items[item].end = p;
    }
    return 0;
}

int vifp_mine_dataset(DM_Dataset *ds, uint32_t minsup_count, VIFPMode mode, size_t max_itemsets, double max_seconds, VIFPStats *stats) {
    if (!ds || ds->type != DM_TYPE_TRANSACTIONAL || !stats) return -1;
    memset(stats, 0, sizeof(*stats));
    stats->transactions = ds->count;
    stats->max_item_id = ds->max_id;
    stats->minsup_count = minsup_count ? minsup_count : 1;

    VIFPContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.minsup = stats->minsup_count;
    ctx.mode = mode;
    ctx.max_itemsets = max_itemsets;
    ctx.max_seconds = max_seconds;
    ctx.started = clock();
    ctx.stats = stats;

    if (build_structures(ds, stats->minsup_count, &ctx) != 0) {
        free(ctx.items);
        free(ctx.ott);
        free(ctx.hpa_idx);
        return -1;
    }

    for (uint32_t rank = (uint32_t)ctx.stats->frequent_singletons; rank > 0; rank--) {
        if (time_limited(&ctx)) break;
        uint32_t item = UINT32_MAX;
        for (uint32_t id = 0; id <= ctx.max_item; id++) {
            if (ctx.items[id].frequent && ctx.items[id].rank == rank - 1) {
                item = id;
                break;
            }
        }
        if (item == UINT32_MAX) continue;
        VIFPIndexBuffer seed;
        buf_init(&seed);
        for (size_t p = ctx.items[item].start; p < ctx.items[item].end; p++) {
            if (buf_push(&seed, ctx.hpa_idx[p]) != 0) {
                buf_free(&seed);
                free(ctx.items);
                free(ctx.ott);
                free(ctx.hpa_idx);
                return -1;
            }
        }
        stats->emitted_itemsets++;
        stats->total_output_items += 1;
        VIFPPID pid = { item, 1, ctx.items[item].start, ctx.items[item].end, seed.count };
        mine_interval(&ctx, pid, &seed);
        buf_free(&seed);
    }

    estimate_crypto_costs(&ctx);
    dm_bench_record_results(stats->emitted_itemsets, stats->total_output_items);
    free(ctx.items);
    free(ctx.ott);
    free(ctx.hpa_idx);
    return stats->limited ? 2 : 0;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_VIFP_Params *p = (DM_VIFP_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01;
    uint32_t minsup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * (double)ds->count) : (uint32_t)min_sup_param;
    if (minsup == 0) minsup = 1;
    VIFPMode mode = p ? p->mode : VIFP_MODE_PLAINTEXT;
    size_t max_itemsets = p ? p->max_itemsets : 0;
    double max_seconds = p ? p->max_seconds : 0.0;
    VIFPStats stats;
    int rc = vifp_mine_dataset(ds, minsup, mode, max_itemsets, max_seconds, &stats);
    printf("[VIFP] Mode: %s\n", vifp_mode_name(mode));
    printf("[VIFP] Min Support: %u\n", minsup);
    printf("[VIFP] Occurrence records: %zu\n", stats.occurrence_records);
    printf("[VIFP] Projected states: %zu\n", stats.projected_states);
    printf("[VIFP] Frequent itemsets: %zu\n", stats.emitted_itemsets);
    printf("[VIFP] Secure comparisons: %zu\n", stats.secure_comparisons);
    dm_bench_record_results(stats.emitted_itemsets, stats.total_output_items);
    return rc == 0 ? DM_SUCCESS : DM_ERROR_GENERIC;
}

DM_Algorithm vifp_algo = {
    .id = "vifp",
    .name = "Virtual-Interval FP-Growth",
    .description = "Shape-hiding FP-Growth semantics using occurrence tapes, header segments, and projected intervals.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};
