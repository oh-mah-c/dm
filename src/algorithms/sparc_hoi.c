#include "algorithms/sparc_hoi.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    unsigned char *base;
    size_t cap;
    size_t top;
    size_t high_water;
} SPARCArena;

typedef struct {
    uint32_t begin;
    uint32_t len;
    double inv_len;
} SPARCTrans;

typedef struct {
    uint32_t tid;
    uint32_t pos;
    uint32_t rem;
    double inv_len;
} SPARCEntry;

typedef struct {
    uint32_t off;
    uint32_t count;
    double sum_inv;
    double rrob;
} SPARCList;

typedef struct {
    uint32_t item;
    uint32_t support;
    uint32_t rank;
} SPARCItemInfo;

typedef struct {
    DM_Trans_Simple *src;
    size_t ntrans;
    uint32_t max_id;
    SPARCTrans *trans;
    uint32_t *ranked_items;
    size_t ranked_count;
    SPARCItemInfo *items;
    size_t active_count;
    uint32_t *id_to_rank;
    uint32_t *rank_to_item;
    size_t min_support;
    double min_occupancy;
    int summed_occupancy_mode;
    double threshold_value;
    size_t max_patterns;
    double max_seconds;
    clock_t start_clock;
    SPARCArena arena;
    uint32_t *support_scratch;
    uint32_t *seen_scratch;
    uint32_t stamp;
    size_t raw_count;
    size_t total_output_items;
    size_t visited_nodes;
    size_t extension_tests;
    size_t projected_entries_scanned;
    size_t pruned_support;
    size_t pruned_rrob;
    size_t pruned_tail;
    int limited;
} SPARCCtx;

static double elapsed_sec(const SPARCCtx *ctx) {
    return (double)(clock() - ctx->start_clock) / (double)CLOCKS_PER_SEC;
}

static int should_stop(SPARCCtx *ctx) {
    if (ctx->max_patterns && ctx->raw_count >= ctx->max_patterns) {
        ctx->limited = 1;
        return 1;
    }
    if (ctx->max_seconds > 0.0 && elapsed_sec(ctx) >= ctx->max_seconds) {
        ctx->limited = 1;
        return 1;
    }
    return 0;
}

static int arena_init(SPARCArena *a, size_t cap) {
    a->base = (unsigned char *)malloc(cap ? cap : 1);
    if (!a->base) return -1;
    a->cap = cap ? cap : 1;
    a->top = 0;
    a->high_water = 0;
    return 0;
}

static void arena_free(SPARCArena *a) {
    free(a->base);
    memset(a, 0, sizeof(*a));
}

static size_t arena_mark(const SPARCArena *a) {
    return a->top;
}

static void arena_rewind(SPARCArena *a, size_t mark) {
    if (mark <= a->top) a->top = mark;
}

static void *arena_alloc(SPARCArena *a, size_t bytes, size_t align) {
    size_t mask = align ? align - 1 : 0;
    size_t top = align ? ((a->top + mask) & ~mask) : a->top;
    if (top + bytes > a->cap) return NULL;
    void *p = a->base + top;
    a->top = top + bytes;
    if (a->top > a->high_water) a->high_water = a->top;
    return p;
}

static SPARCEntry *list_entries(SPARCCtx *ctx, SPARCList l) {
    return (SPARCEntry *)(void *)(ctx->arena.base + l.off);
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int cmp_item_info(const void *a, const void *b) {
    const SPARCItemInfo *x = (const SPARCItemInfo *)a;
    const SPARCItemInfo *y = (const SPARCItemInfo *)b;
    if (x->support < y->support) return -1;
    if (x->support > y->support) return 1;
    return (x->item > y->item) - (x->item < y->item);
}

static int cmp_double_desc(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    if (x < y) return 1;
    if (x > y) return -1;
    return 0;
}

static double compute_rrob(SPARCCtx *ctx, SPARCList l, size_t prefix_len) {
    if (l.count < ctx->min_support) return 0.0;
    SPARCEntry *e = list_entries(ctx, l);
    size_t mark = arena_mark(&ctx->arena);
    double *vals = (double *)arena_alloc(&ctx->arena, (size_t)l.count * sizeof(double), 64);
    if (!vals) return 1.0;
    for (uint32_t i = 0; i < l.count; i++) {
        vals[i] = ((double)prefix_len + (double)e[i].rem) * e[i].inv_len;
    }
    qsort(vals, l.count, sizeof(double), cmp_double_desc);
    double best = 0.0;
    double sum = 0.0;
    for (uint32_t u = 1; u <= l.count; u++) {
        sum += vals[u - 1];
        if (u >= ctx->min_support) {
            double b = sum / (double)u;
            if (b > best) best = b;
        }
    }
    arena_rewind(&ctx->arena, mark);
    return best > 1.0 ? 1.0 : best;
}

static double compute_sum_bound(SPARCCtx *ctx, SPARCList l, size_t prefix_len) {
    if (l.count < ctx->min_support) return 0.0;
    SPARCEntry *e = list_entries(ctx, l);
    double sum = 0.0;
    for (uint32_t i = 0; i < l.count; i++) {
        sum += ((double)prefix_len + (double)e[i].rem) * e[i].inv_len;
    }
    return sum;
}

static uint32_t count_active_suffix(SPARCCtx *ctx, uint32_t tid, uint32_t pos) {
    SPARCTrans tr = ctx->trans[tid];
    if (pos == UINT32_MAX) return tr.len;
    return (pos + 1u < tr.len) ? (tr.len - pos - 1u) : 0u;
}

static int project_extend(SPARCCtx *ctx, SPARCList parent, uint32_t rank,
                          size_t child_prefix_len, SPARCList *child) {
    SPARCEntry *pe = list_entries(ctx, parent);
    size_t bytes = (size_t)parent.count * sizeof(SPARCEntry);
    SPARCEntry *ce = (SPARCEntry *)arena_alloc(&ctx->arena, bytes, 64);
    if (!ce) return -1;
    child->off = (uint32_t)((unsigned char *)(void *)ce - ctx->arena.base);
    child->count = 0;
    child->sum_inv = 0.0;
    child->rrob = 0.0;
    ctx->projected_entries_scanned += parent.count;

    for (uint32_t i = 0; i < parent.count; i++) {
        SPARCTrans tr = ctx->trans[pe[i].tid];
        uint32_t start = (pe[i].pos == UINT32_MAX) ? 0u : pe[i].pos + 1u;
        uint32_t found = UINT32_MAX;
        for (uint32_t p = start; p < tr.len; p++) {
            uint32_t r = ctx->ranked_items[tr.begin + p];
            if (r == rank) {
                found = p;
                break;
            }
            if (r > rank) break;
        }
        if (found != UINT32_MAX) {
            SPARCEntry *dst = &ce[child->count++];
            dst->tid = pe[i].tid;
            dst->pos = found;
            dst->rem = count_active_suffix(ctx, pe[i].tid, found);
            dst->inv_len = pe[i].inv_len;
            child->sum_inv += pe[i].inv_len;
        }
    }
    if (child->count >= ctx->min_support) {
        if (ctx->summed_occupancy_mode) {
            child->rrob = compute_sum_bound(ctx, *child, child_prefix_len);
        } else {
            child->rrob = compute_rrob(ctx, *child, child_prefix_len);
        }
    }
    return 0;
}

static int tail_contains_after(const uint32_t *tail, size_t tail_count, size_t start, uint32_t rank) {
    for (size_t i = start; i < tail_count; i++) {
        if (tail[i] == rank) return 1;
        if (tail[i] > rank) return 0;
    }
    return 0;
}

static uint32_t *squeeze_tail(SPARCCtx *ctx, SPARCList child,
                             const uint32_t *tail, size_t tail_count, size_t tail_start,
                             size_t *out_count) {
    *out_count = 0;
    if (tail_start >= tail_count || child.count < ctx->min_support) return NULL;
    if (++ctx->stamp == 0) {
        memset(ctx->seen_scratch, 0, ctx->active_count * sizeof(uint32_t));
        ctx->stamp = 1;
    }
    for (size_t i = tail_start; i < tail_count; i++) ctx->support_scratch[tail[i]] = 0;
    SPARCEntry *e = list_entries(ctx, child);
    for (uint32_t k = 0; k < child.count; k++) {
        SPARCTrans tr = ctx->trans[e[k].tid];
        uint32_t start = e[k].pos + 1u;
        for (uint32_t p = start; p < tr.len; p++) {
            uint32_t r = ctx->ranked_items[tr.begin + p];
            if (!tail_contains_after(tail, tail_count, tail_start, r)) continue;
            if (ctx->seen_scratch[r] != ctx->stamp) {
                ctx->seen_scratch[r] = ctx->stamp;
                ctx->support_scratch[r]++;
            }
        }
        ctx->stamp++;
        if (ctx->stamp == 0) {
            memset(ctx->seen_scratch, 0, ctx->active_count * sizeof(uint32_t));
            ctx->stamp = 1;
        }
    }

    uint32_t *next = (uint32_t *)arena_alloc(&ctx->arena, (tail_count - tail_start) * sizeof(uint32_t), 64);
    if (!next) return NULL;
    for (size_t i = tail_start; i < tail_count; i++) {
        uint32_t r = tail[i];
        if (ctx->support_scratch[r] >= ctx->min_support) {
            next[(*out_count)++] = r;
        } else {
            ctx->pruned_tail++;
        }
    }
    return next;
}

static void mine_raw(SPARCCtx *ctx, uint32_t *prefix, size_t prefix_len,
                     SPARCList list, const uint32_t *tail, size_t tail_count, double bound) {
    if (should_stop(ctx)) return;
    if (list.count < ctx->min_support) return;
    if (bound + 1e-12 < ctx->threshold_value) {
        ctx->pruned_rrob++;
        return;
    }

    for (size_t i = 0; i < tail_count; i++) {
        if (should_stop(ctx)) return;
        size_t mark = arena_mark(&ctx->arena);
        SPARCList child;
        ctx->extension_tests++;
        if (project_extend(ctx, list, tail[i], prefix_len + 1, &child) != 0) {
            ctx->limited = 1;
            return;
        }
        ctx->visited_nodes++;
        if (child.count < ctx->min_support) {
            ctx->pruned_support++;
            arena_rewind(&ctx->arena, mark);
            continue;
        }

        double occ;
        if (ctx->summed_occupancy_mode) {
            occ = (double)(prefix_len + 1) * child.sum_inv;
        } else {
            occ = ((double)(prefix_len + 1) * child.sum_inv) / (double)child.count;
        }
        prefix[prefix_len] = ctx->rank_to_item[tail[i]];
        if (occ + 1e-12 >= ctx->threshold_value) {
            ctx->raw_count++;
            ctx->total_output_items += prefix_len + 1;
        }

        double child_bound = child.rrob < bound ? child.rrob : bound;
        if (child_bound + 1e-12 >= ctx->threshold_value && i + 1 < tail_count) {
            size_t next_count = 0;
            uint32_t *next_tail = squeeze_tail(ctx, child, tail, tail_count, i + 1, &next_count);
            if (next_tail && next_count > 0) {
                mine_raw(ctx, prefix, prefix_len + 1, child, next_tail, next_count, child_bound);
            }
        } else if (i + 1 < tail_count) {
            ctx->pruned_rrob++;
        }
        arena_rewind(&ctx->arena, mark);
    }
}

static int build_flat_db(SPARCCtx *ctx) {
    uint32_t *counts = (uint32_t *)calloc((size_t)ctx->max_id + 1, sizeof(uint32_t));
    if (!counts) return -1;
    for (size_t t = 0; t < ctx->ntrans; t++) {
        for (size_t j = 0; j < ctx->src[t].count; j++) counts[ctx->src[t].items[j]]++;
    }
    size_t active = 0;
    for (uint32_t id = 0; id <= ctx->max_id; id++) if (counts[id] >= ctx->min_support) active++;
    ctx->items = (SPARCItemInfo *)malloc(active * sizeof(SPARCItemInfo));
    ctx->id_to_rank = (uint32_t *)malloc(((size_t)ctx->max_id + 1) * sizeof(uint32_t));
    ctx->rank_to_item = (uint32_t *)malloc(active * sizeof(uint32_t));
    if (!ctx->items || !ctx->id_to_rank || !ctx->rank_to_item) {
        free(counts);
        return -1;
    }
    for (uint32_t id = 0; id <= ctx->max_id; id++) ctx->id_to_rank[id] = UINT32_MAX;
    for (uint32_t id = 0; id <= ctx->max_id; id++) {
        if (counts[id] >= ctx->min_support) {
            ctx->items[ctx->active_count].item = id;
            ctx->items[ctx->active_count].support = counts[id];
            ctx->items[ctx->active_count].rank = 0;
            ctx->active_count++;
        }
    }
    free(counts);
    qsort(ctx->items, ctx->active_count, sizeof(SPARCItemInfo), cmp_item_info);
    for (size_t r = 0; r < ctx->active_count; r++) {
        ctx->items[r].rank = (uint32_t)r;
        ctx->id_to_rank[ctx->items[r].item] = (uint32_t)r;
        ctx->rank_to_item[r] = ctx->items[r].item;
    }

    ctx->trans = (SPARCTrans *)calloc(ctx->ntrans, sizeof(SPARCTrans));
    if (!ctx->trans) return -1;
    size_t total = 0;
    for (size_t t = 0; t < ctx->ntrans; t++) {
        for (size_t j = 0; j < ctx->src[t].count; j++) {
            uint32_t id = ctx->src[t].items[j];
            if (id <= ctx->max_id && ctx->id_to_rank[id] != UINT32_MAX) total++;
        }
    }
    ctx->ranked_items = (uint32_t *)malloc(total ? total * sizeof(uint32_t) : sizeof(uint32_t));
    if (!ctx->ranked_items) return -1;
    ctx->ranked_count = 0;
    for (size_t t = 0; t < ctx->ntrans; t++) {
        ctx->trans[t].begin = (uint32_t)ctx->ranked_count;
        ctx->trans[t].inv_len = ctx->src[t].count ? 1.0 / (double)ctx->src[t].count : 0.0;
        size_t before = ctx->ranked_count;
        for (size_t j = 0; j < ctx->src[t].count; j++) {
            uint32_t id = ctx->src[t].items[j];
            if (id <= ctx->max_id && ctx->id_to_rank[id] != UINT32_MAX) {
                ctx->ranked_items[ctx->ranked_count++] = ctx->id_to_rank[id];
            }
        }
        size_t len = ctx->ranked_count - before;
        qsort(ctx->ranked_items + before, len, sizeof(uint32_t), cmp_u32);
        size_t out = before;
        for (size_t in = before; in < ctx->ranked_count; in++) {
            if (out == before || ctx->ranked_items[in] != ctx->ranked_items[out - 1]) {
                ctx->ranked_items[out++] = ctx->ranked_items[in];
            }
        }
        ctx->ranked_count = out;
        ctx->trans[t].len = (uint32_t)(ctx->ranked_count - before);
    }
    ctx->support_scratch = (uint32_t *)calloc(ctx->active_count ? ctx->active_count : 1, sizeof(uint32_t));
    ctx->seen_scratch = (uint32_t *)calloc(ctx->active_count ? ctx->active_count : 1, sizeof(uint32_t));
    ctx->stamp = 1;
    return (ctx->support_scratch && ctx->seen_scratch) ? 0 : -1;
}

static void free_ctx(SPARCCtx *ctx) {
    arena_free(&ctx->arena);
    free(ctx->trans);
    free(ctx->ranked_items);
    free(ctx->items);
    free(ctx->id_to_rank);
    free(ctx->rank_to_item);
    free(ctx->support_scratch);
    free(ctx->seen_scratch);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_TRANSACTIONAL || !ds->payload) return DM_ERROR_INCOMPATIBLE;
    DM_SPARC_HOI_Params *p = (DM_SPARC_HOI_Params *)params;
    SPARCCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.src = (DM_Trans_Simple *)ds->payload;
    ctx.ntrans = ds->count;
    ctx.max_id = ds->max_id;
    ctx.min_occupancy = p ? p->min_occupancy : 0.5;
    ctx.summed_occupancy_mode = p ? p->summed_occupancy_mode : 0;
    ctx.threshold_value = ctx.summed_occupancy_mode ? ((ctx.min_occupancy < 1.0) ? ctx.min_occupancy * (double)ctx.ntrans : ctx.min_occupancy) : ctx.min_occupancy;
    ctx.min_support = (p && p->min_support) ? p->min_support : (size_t)ceil(ctx.min_occupancy * (double)ctx.ntrans);
    if (ctx.min_support < 1) ctx.min_support = 1;
    ctx.max_patterns = p ? p->max_patterns : 0;
    ctx.max_seconds = p ? p->max_seconds : 0.0;
    ctx.start_clock = clock();

    if (build_flat_db(&ctx) != 0) {
        free_ctx(&ctx);
        return DM_ERROR_MEMORY;
    }
    size_t entry_budget = (ctx.ranked_count + ctx.ntrans + 1) * (ctx.active_count < 64 ? ctx.active_count + 2 : 66);
    if (entry_budget < ctx.ntrans + 1) entry_budget = ctx.ntrans + 1;
    size_t arena_bytes = entry_budget * sizeof(SPARCEntry)
                       + (ctx.active_count + 1) * (ctx.active_count + 1) * sizeof(uint32_t)
                       + 4096;
    if (arena_init(&ctx.arena, arena_bytes) != 0) {
        free_ctx(&ctx);
        return DM_ERROR_MEMORY;
    }

    SPARCEntry *root_entries = (SPARCEntry *)arena_alloc(&ctx.arena, ctx.ntrans * sizeof(SPARCEntry), 64);
    uint32_t *tail = (uint32_t *)arena_alloc(&ctx.arena, ctx.active_count * sizeof(uint32_t), 64);
    uint32_t *prefix = (uint32_t *)malloc((ctx.active_count ? ctx.active_count : 1) * sizeof(uint32_t));
    if (!root_entries || !tail || !prefix) {
        free(prefix);
        free_ctx(&ctx);
        return DM_ERROR_MEMORY;
    }
    SPARCList root;
    root.off = (uint32_t)((unsigned char *)(void *)root_entries - ctx.arena.base);
    root.count = 0;
    root.sum_inv = 0.0;
    for (size_t t = 0; t < ctx.ntrans; t++) {
        if (ctx.trans[t].len == 0) continue;
        root_entries[root.count].tid = (uint32_t)t;
        root_entries[root.count].pos = UINT32_MAX;
        root_entries[root.count].rem = ctx.trans[t].len;
        root_entries[root.count].inv_len = ctx.trans[t].inv_len;
        root.sum_inv += ctx.trans[t].inv_len;
        root.count++;
    }
    for (size_t i = 0; i < ctx.active_count; i++) tail[i] = (uint32_t)i;
    if (ctx.summed_occupancy_mode) {
        root.rrob = compute_sum_bound(&ctx, root, 0);
    } else {
        root.rrob = compute_rrob(&ctx, root, 0);
    }

    printf("[SPARC-HOI] transactions=%zu active_items=%zu minsup=%zu minocc=%.6f threshold=%.6f mode=%s root_rrob=%.6f arena_bytes=%zu\n",
           ctx.ntrans, ctx.active_count, ctx.min_support, ctx.min_occupancy, ctx.threshold_value,
           ctx.summed_occupancy_mode ? "summed-compatible" : "average", root.rrob, ctx.arena.cap);
    mine_raw(&ctx, prefix, 0, root, tail, ctx.active_count, root.rrob);

    double seconds = elapsed_sec(&ctx);
    double bandwidth = seconds > 0.0
        ? ((double)ctx.projected_entries_scanned * (double)sizeof(SPARCEntry)) / (seconds * 1048576.0)
        : 0.0;
    printf("[SPARC-HOI] Complete. Raw fullset HO itemsets found: %zu\n", ctx.raw_count);
    printf("[SPARC-HOI] visited_nodes=%zu extension_tests=%zu pruned_support=%zu pruned_rrob=%zu pruned_tail=%zu arena_high_water_bytes=%zu limited=%s\n",
           ctx.visited_nodes, ctx.extension_tests, ctx.pruned_support, ctx.pruned_rrob, ctx.pruned_tail,
           ctx.arena.high_water, ctx.limited ? "yes" : "no");
    printf("[SPARC-HOI] projected_entries_scanned=%zu evaluation_bandwidth_mb_s=%.6f\n",
           ctx.projected_entries_scanned, bandwidth);
    dm_bench_record_results(ctx.raw_count, ctx.total_output_items);

    free(prefix);
    free_ctx(&ctx);
    return DM_SUCCESS;
}

DM_Algorithm sparc_hoi_algo = {
    .id = "sparc_hoi",
    .name = "SPARC-HOI",
    .description = "Candidate-free raw high-occupancy itemset mining with flat suffix projections and RROB pruning.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(sparc_hoi_algo)
