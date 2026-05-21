#include "algorithms/cloe_hoi.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint64_t *pool;
    size_t words_capacity;
    size_t words_used;
    size_t high_water;
} BitArena;

typedef struct {
    uint32_t *items;
    size_t len;
} EmittedPattern;

typedef struct {
    EmittedPattern *data;
    size_t count;
    size_t cap;
} EmittedSet;

typedef struct {
    DM_Trans_Simple *trans;
    size_t ntrans;
    uint32_t max_id;
    size_t words;
    uint32_t *active_items;
    size_t active_count;
    uint64_t *item_bits;
    uint32_t *transaction_len;
    double *reciprocal_len;
    size_t min_support;
    double min_occupancy;
    size_t max_patterns;
    double max_seconds;
    clock_t start_clock;
    BitArena arena;
    EmittedSet emitted;
    size_t visited_nodes;
    size_t pruned_support;
    size_t pruned_backward;
    size_t pruned_envelope;
    size_t closure_jumps;
    size_t total_output_items;
    int limited;
} CLOECtx;

static double elapsed_sec(const CLOECtx *ctx) {
    return (double)(clock() - ctx->start_clock) / (double)CLOCKS_PER_SEC;
}

static int should_stop(CLOECtx *ctx) {
    if (ctx->max_patterns && ctx->emitted.count >= ctx->max_patterns) {
        ctx->limited = 1;
        return 1;
    }
    if (ctx->max_seconds > 0.0 && elapsed_sec(ctx) >= ctx->max_seconds) {
        ctx->limited = 1;
        return 1;
    }
    return 0;
}

static int arena_init(BitArena *a, size_t words) {
    a->pool = (uint64_t *)calloc(words ? words : 1, sizeof(uint64_t));
    if (!a->pool) return -1;
    a->words_capacity = words;
    a->words_used = 0;
    a->high_water = 0;
    return 0;
}

static void arena_free(BitArena *a) {
    free(a->pool);
    memset(a, 0, sizeof(*a));
}

static size_t arena_mark(BitArena *a) {
    return a->words_used;
}

static uint64_t *arena_alloc_bitset(BitArena *a, size_t words) {
    if (a->words_used + words > a->words_capacity) return NULL;
    uint64_t *p = a->pool + a->words_used;
    a->words_used += words;
    if (a->words_used > a->high_water) a->high_water = a->words_used;
    memset(p, 0, words * sizeof(uint64_t));
    return p;
}

static void arena_rewind(BitArena *a, size_t mark) {
    if (mark <= a->words_used) a->words_used = mark;
}

static inline uint64_t *item_bitset(CLOECtx *ctx, size_t active_idx) {
    return ctx->item_bits + active_idx * ctx->words;
}

static inline int bit_test(const uint64_t *bits, size_t idx) {
    return (int)((bits[idx >> 6] >> (idx & 63u)) & 1ull);
}

static inline void bit_set(uint64_t *bits, size_t idx) {
    bits[idx >> 6] |= 1ull << (idx & 63u);
}

static size_t bitset_and_count(uint64_t *out, const uint64_t *a, const uint64_t *b, size_t words) {
    size_t count = 0;
    for (size_t w = 0; w < words; w++) {
        out[w] = a[w] & b[w];
        count += (size_t)__builtin_popcountll(out[w]);
    }
    return count;
}

static size_t bitset_count(const uint64_t *bits, size_t words) {
    size_t count = 0;
    for (size_t w = 0; w < words; w++) count += (size_t)__builtin_popcountll(bits[w]);
    return count;
}

static int support_subset_item(CLOECtx *ctx, const uint64_t *support, size_t item_idx) {
    const uint64_t *ib = item_bitset(ctx, item_idx);
    for (size_t w = 0; w < ctx->words; w++) {
        if (support[w] & ~ib[w]) return 0;
    }
    return 1;
}

static double exact_average_occupancy(CLOECtx *ctx, const uint64_t *support, size_t supp, size_t len) {
    if (supp == 0) return 0.0;
    double sum = 0.0;
    for (size_t w = 0; w < ctx->words; w++) {
        uint64_t x = support[w];
        while (x) {
            unsigned bit = (unsigned)__builtin_ctzll(x);
            size_t tid = (w << 6) + bit;
            if (tid < ctx->ntrans) sum += ctx->reciprocal_len[tid];
            x &= x - 1;
        }
    }
    return ((double)len * sum) / (double)supp;
}

static int cmp_double_desc(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    if (x < y) return 1;
    if (x > y) return -1;
    return 0;
}

static double occupancy_envelope_local(CLOECtx *ctx, size_t prefix_len, const uint64_t *support,
                                       const size_t *tail, size_t tail_count) {
    size_t supp = bitset_count(support, ctx->words);
    if (supp < ctx->min_support) return 0.0;
    if (tail_count == 0) return exact_average_occupancy(ctx, support, supp, prefix_len);

    uint16_t *rem = (uint16_t *)calloc(ctx->ntrans, sizeof(uint16_t));
    if (!rem) return 1.0;
    size_t bmax = 0;
    for (size_t t = 0; t < tail_count; t++) {
        const uint64_t *ib = item_bitset(ctx, tail[t]);
        for (size_t w = 0; w < ctx->words; w++) {
            uint64_t x = support[w] & ib[w];
            while (x) {
                unsigned bit = (unsigned)__builtin_ctzll(x);
                size_t tid = (w << 6) + bit;
                if (tid < ctx->ntrans) {
                    if (rem[tid] < UINT16_MAX) rem[tid]++;
                    if (rem[tid] > bmax) bmax = rem[tid];
                }
                x &= x - 1;
            }
        }
    }

    double *vals = (double *)malloc(supp * sizeof(double));
    if (!vals) {
        free(rem);
        return 1.0;
    }
    double best = 0.0;
    for (size_t b = 0; b <= bmax; b++) {
        size_t nvals = 0;
        for (size_t w = 0; w < ctx->words; w++) {
            uint64_t x = support[w];
            while (x) {
                unsigned bit = (unsigned)__builtin_ctzll(x);
                size_t tid = (w << 6) + bit;
                if (tid < ctx->ntrans && rem[tid] >= b) vals[nvals++] = ctx->reciprocal_len[tid];
                x &= x - 1;
            }
        }
        if (nvals >= ctx->min_support) {
            qsort(vals, nvals, sizeof(double), cmp_double_desc);
            double psum = 0.0;
            for (size_t u = 1; u <= nvals; u++) {
                psum += vals[u - 1];
                if (u >= ctx->min_support) {
                    double bound = ((double)(prefix_len + b) * psum) / (double)u;
                    if (bound > best) best = bound;
                }
            }
        }
    }
    free(vals);
    free(rem);
    return best > 1.0 ? 1.0 : best;
}

static int pattern_equals(const EmittedPattern *p, const uint32_t *items, size_t len) {
    if (p->len != len) return 0;
    for (size_t i = 0; i < len; i++) if (p->items[i] != items[i]) return 0;
    return 1;
}

static int emitted_add(CLOECtx *ctx, const uint32_t *items, size_t len) {
    for (size_t i = 0; i < ctx->emitted.count; i++) {
        if (pattern_equals(&ctx->emitted.data[i], items, len)) return 0;
    }
    if (ctx->emitted.count == ctx->emitted.cap) {
        size_t nc = ctx->emitted.cap ? ctx->emitted.cap * 2 : 256;
        EmittedPattern *nd = (EmittedPattern *)realloc(ctx->emitted.data, nc * sizeof(*nd));
        if (!nd) return -1;
        ctx->emitted.data = nd;
        ctx->emitted.cap = nc;
    }
    ctx->emitted.data[ctx->emitted.count].items = (uint32_t *)malloc(len * sizeof(uint32_t));
    if (!ctx->emitted.data[ctx->emitted.count].items) return -1;
    memcpy(ctx->emitted.data[ctx->emitted.count].items, items, len * sizeof(uint32_t));
    ctx->emitted.data[ctx->emitted.count].len = len;
    ctx->emitted.count++;
    ctx->total_output_items += len;
    return 1;
}

static void emitted_free(EmittedSet *e) {
    for (size_t i = 0; i < e->count; i++) free(e->data[i].items);
    free(e->data);
    memset(e, 0, sizeof(*e));
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static void mine_closed(CLOECtx *ctx, const uint32_t *prefix, size_t prefix_len,
                        const uint64_t *support, const size_t *tail, size_t tail_count,
                        double path_bound) {
    if (should_stop(ctx) || path_bound + 1e-12 < ctx->min_occupancy) {
        if (path_bound + 1e-12 < ctx->min_occupancy) ctx->pruned_envelope++;
        return;
    }

    for (size_t pos = 0; pos < tail_count; pos++) {
        if (should_stop(ctx)) return;
        size_t mark = arena_mark(&ctx->arena);
        uint64_t *child = arena_alloc_bitset(&ctx->arena, ctx->words);
        if (!child) {
            ctx->limited = 1;
            return;
        }
        size_t supp = bitset_and_count(child, support, item_bitset(ctx, tail[pos]), ctx->words);
        ctx->visited_nodes++;
        if (supp < ctx->min_support) {
            ctx->pruned_support++;
            arena_rewind(&ctx->arena, mark);
            continue;
        }

        int backward = 0;
        for (size_t j = 0; j < pos; j++) {
            if (support_subset_item(ctx, child, tail[j])) {
                backward = 1;
                break;
            }
        }
        if (backward) {
            ctx->pruned_backward++;
            arena_rewind(&ctx->arena, mark);
            continue;
        }

        uint32_t *closed = (uint32_t *)malloc((prefix_len + tail_count - pos) * sizeof(uint32_t));
        size_t closed_len = 0;
        if (prefix_len) {
            memcpy(closed, prefix, prefix_len * sizeof(uint32_t));
            closed_len = prefix_len;
        }
        closed[closed_len++] = ctx->active_items[tail[pos]];

        size_t *next_tail = (size_t *)malloc((tail_count - pos - 1) * sizeof(size_t));
        size_t next_count = 0;
        for (size_t j = pos + 1; j < tail_count; j++) {
            if (support_subset_item(ctx, child, tail[j])) {
                closed[closed_len++] = ctx->active_items[tail[j]];
                ctx->closure_jumps++;
            } else {
                next_tail[next_count++] = tail[j];
            }
        }
        qsort(closed, closed_len, sizeof(uint32_t), cmp_u32);

        double avg_occ = exact_average_occupancy(ctx, child, supp, closed_len);
        if (avg_occ + 1e-12 >= ctx->min_occupancy) emitted_add(ctx, closed, closed_len);

        double local = occupancy_envelope_local(ctx, closed_len, child, next_tail, next_count);
        double next_bound = local < path_bound ? local : path_bound;
        if (next_bound + 1e-12 >= ctx->min_occupancy && next_count > 0) {
            mine_closed(ctx, closed, closed_len, child, next_tail, next_count, next_bound);
        } else if (next_count > 0) {
            ctx->pruned_envelope++;
        }

        free(next_tail);
        free(closed);
        arena_rewind(&ctx->arena, mark);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_TRANSACTIONAL) return DM_ERROR_INCOMPATIBLE;
    DM_CLOE_HOI_Params *p = (DM_CLOE_HOI_Params *)params;
    CLOECtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.trans = (DM_Trans_Simple *)ds->payload;
    ctx.ntrans = ds->count;
    ctx.max_id = ds->max_id;
    ctx.words = (ctx.ntrans + 63) / 64;
    ctx.min_occupancy = p ? p->min_occupancy : 0.5;
    ctx.min_support = (p && p->min_support) ? p->min_support : (size_t)ceil(ctx.min_occupancy * (double)ctx.ntrans);
    if (ctx.min_support < 1) ctx.min_support = 1;
    ctx.max_patterns = p ? p->max_patterns : 0;
    ctx.max_seconds = p ? p->max_seconds : 0.0;
    ctx.start_clock = clock();

    ctx.transaction_len = (uint32_t *)malloc(ctx.ntrans * sizeof(uint32_t));
    ctx.reciprocal_len = (double *)malloc(ctx.ntrans * sizeof(double));
    uint32_t *support_count = (uint32_t *)calloc((size_t)ctx.max_id + 1, sizeof(uint32_t));
    if (!ctx.transaction_len || !ctx.reciprocal_len || !support_count) return DM_ERROR_MEMORY;

    for (size_t t = 0; t < ctx.ntrans; t++) {
        ctx.transaction_len[t] = (uint32_t)ctx.trans[t].count;
        ctx.reciprocal_len[t] = ctx.trans[t].count ? 1.0 / (double)ctx.trans[t].count : 0.0;
        for (size_t j = 0; j < ctx.trans[t].count; j++) support_count[ctx.trans[t].items[j]]++;
    }
    for (uint32_t i = 0; i <= ctx.max_id; i++) {
        if (support_count[i] >= ctx.min_support) ctx.active_count++;
    }
    ctx.active_items = (uint32_t *)malloc(ctx.active_count * sizeof(uint32_t));
    ctx.item_bits = (uint64_t *)calloc(ctx.active_count * ctx.words, sizeof(uint64_t));
    if (!ctx.active_items || !ctx.item_bits) return DM_ERROR_MEMORY;
    size_t idx = 0;
    uint32_t *id_to_active = (uint32_t *)malloc(((size_t)ctx.max_id + 1) * sizeof(uint32_t));
    if (!id_to_active) return DM_ERROR_MEMORY;
    for (uint32_t i = 0; i <= ctx.max_id; i++) id_to_active[i] = UINT32_MAX;
    for (uint32_t i = 0; i <= ctx.max_id; i++) {
        if (support_count[i] >= ctx.min_support) {
            id_to_active[i] = (uint32_t)idx;
            ctx.active_items[idx++] = i;
        }
    }
    for (size_t t = 0; t < ctx.ntrans; t++) {
        for (size_t j = 0; j < ctx.trans[t].count; j++) {
            uint32_t item = ctx.trans[t].items[j];
            if (id_to_active[item] != UINT32_MAX) bit_set(item_bitset(&ctx, id_to_active[item]), t);
        }
    }
    free(id_to_active);
    free(support_count);

    size_t arena_words = ctx.words * (ctx.active_count + 8);
    if (arena_init(&ctx.arena, arena_words) != 0) return DM_ERROR_MEMORY;
    uint64_t *root = arena_alloc_bitset(&ctx.arena, ctx.words);
    for (size_t w = 0; w < ctx.words; w++) root[w] = UINT64_MAX;
    if (ctx.ntrans & 63u) root[ctx.words - 1] &= ((1ull << (ctx.ntrans & 63u)) - 1ull);
    size_t *tail = (size_t *)malloc(ctx.active_count * sizeof(size_t));
    for (size_t i = 0; i < ctx.active_count; i++) tail[i] = i;

    double root_bound = occupancy_envelope_local(&ctx, 0, root, tail, ctx.active_count);
    printf("[CLOE-HOI] transactions=%zu active_items=%zu minsup=%zu minocc=%.6f root_envelope=%.6f\n",
           ctx.ntrans, ctx.active_count, ctx.min_support, ctx.min_occupancy, root_bound);
    mine_closed(&ctx, NULL, 0, root, tail, ctx.active_count, root_bound);

    printf("[CLOE-HOI] Complete. Support-closed HO itemsets found: %zu\n", ctx.emitted.count);
    printf("[CLOE-HOI] visited_nodes=%zu pruned_support=%zu pruned_backward=%zu pruned_envelope=%zu closure_jumps=%zu arena_high_water_bytes=%zu limited=%s\n",
           ctx.visited_nodes, ctx.pruned_support, ctx.pruned_backward, ctx.pruned_envelope,
           ctx.closure_jumps, ctx.arena.high_water * sizeof(uint64_t), ctx.limited ? "yes" : "no");
    dm_bench_record_results(ctx.emitted.count, ctx.total_output_items);

    free(tail);
    emitted_free(&ctx.emitted);
    arena_free(&ctx.arena);
    free(ctx.active_items);
    free(ctx.item_bits);
    free(ctx.transaction_len);
    free(ctx.reciprocal_len);
    return DM_SUCCESS;
}

DM_Algorithm cloe_hoi_algo = {
    .id = "cloe_hoi",
    .name = "CLOE-HOI",
    .description = "Candidate-free support-closed high-occupancy itemset mining with cache-local bitsets and Occupancy Envelope pruning.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(cloe_hoi_algo)
