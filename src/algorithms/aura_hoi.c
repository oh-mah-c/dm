#include "algorithms/aura_hoi.h"
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
} AURAArena;

typedef struct {
    uint64_t h1;
    uint64_t h2;
    size_t support;
    double occupancy;
    uint64_t *support_bits;
    uint32_t *items;
    size_t len;
} AURALedgerEntry;

typedef struct {
    AURALedgerEntry *data;
    size_t count;
    size_t cap;
} AURALedger;

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
    uint32_t max_transaction_len;
    double *len_bucket_sum;
    uint32_t *len_bucket_stamp;
    uint32_t len_bucket_epoch;
    size_t min_support;
    double min_occupancy;
    double threshold_value;
    int summed_occupancy_mode;
    size_t max_patterns;
    double max_seconds;
    int emit_raw_view;
    size_t top_k;
    clock_t start_clock;
    AURAArena arena;
    AURALedger ledger;
    size_t raw_hoi_count;
    size_t raw_total_output_items;
    size_t total_output_items;
    size_t visited_nodes;
    size_t pruned_support;
    size_t pruned_backward;
    size_t pruned_envelope;
    size_t closure_jumps;
    size_t ledger_duplicates;
    size_t topk_updates;
    int limited;
} AURACtx;

static double elapsed_sec(const AURACtx *ctx) {
    return (double)(clock() - ctx->start_clock) / (double)CLOCKS_PER_SEC;
}

static int should_stop(AURACtx *ctx) {
    if (ctx->max_patterns && ctx->ledger.count >= ctx->max_patterns) {
        ctx->limited = 1;
        return 1;
    }
    if (ctx->max_seconds > 0.0 && elapsed_sec(ctx) >= ctx->max_seconds) {
        ctx->limited = 1;
        return 1;
    }
    return 0;
}

static int arena_init(AURAArena *a, size_t words) {
    a->pool = (uint64_t *)calloc(words ? words : 1, sizeof(uint64_t));
    if (!a->pool) return -1;
    a->words_capacity = words;
    a->words_used = 0;
    a->high_water = 0;
    return 0;
}

static void arena_free(AURAArena *a) {
    free(a->pool);
    memset(a, 0, sizeof(*a));
}

static size_t arena_mark(AURAArena *a) {
    return a->words_used;
}

static uint64_t *arena_alloc_bitset(AURAArena *a, size_t words) {
    if (a->words_used + words > a->words_capacity) return NULL;
    uint64_t *p = a->pool + a->words_used;
    a->words_used += words;
    if (a->words_used > a->high_water) a->high_water = a->words_used;
    memset(p, 0, words * sizeof(uint64_t));
    return p;
}

static void arena_rewind(AURAArena *a, size_t mark) {
    if (mark <= a->words_used) a->words_used = mark;
}

static inline uint64_t *item_bitset(AURACtx *ctx, size_t active_idx) {
    return ctx->item_bits + active_idx * ctx->words;
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

static int support_subset_item(AURACtx *ctx, const uint64_t *support, size_t item_idx) {
    const uint64_t *ib = item_bitset(ctx, item_idx);
    for (size_t w = 0; w < ctx->words; w++) {
        if (support[w] & ~ib[w]) return 0;
    }
    return 1;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int cmp_double_desc(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    if (x < y) return 1;
    if (x > y) return -1;
    return 0;
}

static double exact_average_occupancy(AURACtx *ctx, const uint64_t *support, size_t supp, size_t len) {
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

static double exact_summed_occupancy(AURACtx *ctx, const uint64_t *support, size_t len) {
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
    return (double)len * sum;
}

static void support_summed_stats(AURACtx *ctx, const uint64_t *support,
                                 double *sum_recip_out, double *ubo_out) {
    double sum = 0.0;
    double max_ubo = 0.0;

    if (++ctx->len_bucket_epoch == 0) {
        memset(ctx->len_bucket_stamp, 0,
               ((size_t)ctx->max_transaction_len + 1) * sizeof(uint32_t));
        ctx->len_bucket_epoch = 1;
    }

    for (size_t w = 0; w < ctx->words; w++) {
        uint64_t x = support[w];
        while (x) {
            unsigned bit = (unsigned)__builtin_ctzll(x);
            size_t tid = (w << 6) + bit;
            if (tid < ctx->ntrans) {
                uint32_t len = ctx->transaction_len[tid];
                double recip = ctx->reciprocal_len[tid];
                sum += recip;
                if (len > 0) {
                    if (ctx->len_bucket_stamp[len] != ctx->len_bucket_epoch) {
                        ctx->len_bucket_stamp[len] = ctx->len_bucket_epoch;
                        ctx->len_bucket_sum[len] = 0.0;
                    }
                    ctx->len_bucket_sum[len] += recip;
                }
            }
            x &= x - 1;
        }
    }

    double suffix_sum = 0.0;
    for (uint32_t len = ctx->max_transaction_len; len > 0; len--) {
        if (ctx->len_bucket_stamp[len] == ctx->len_bucket_epoch) {
            suffix_sum += ctx->len_bucket_sum[len];
            double ubo = (double)len * suffix_sum;
            if (ubo > max_ubo) max_ubo = ubo;
        }
    }

    *sum_recip_out = sum;
    *ubo_out = max_ubo;
}

static void support_hash(const uint64_t *bits, size_t words, uint64_t *h1, uint64_t *h2) {
    uint64_t a = 1469598103934665603ull;
    uint64_t b = 1099511628211ull ^ (uint64_t)words;
    for (size_t i = 0; i < words; i++) {
        uint64_t x = bits[i];
        a ^= x;
        a *= 1099511628211ull;
        b ^= x + 0x9e3779b97f4a7c15ull + (b << 6) + (b >> 2);
    }
    *h1 = a;
    *h2 = b;
}

static int ledger_has(AURACtx *ctx, const uint64_t *support, size_t supp, uint64_t h1, uint64_t h2) {
    for (size_t i = 0; i < ctx->ledger.count; i++) {
        AURALedgerEntry *e = &ctx->ledger.data[i];
        if (e->h1 == h1 && e->h2 == h2 && e->support == supp) {
            int same = 1;
            for (size_t w = 0; w < ctx->words; w++) {
                if (e->support_bits[w] != support[w]) {
                    same = 0;
                    break;
                }
            }
            if (same) return 1;
        }
    }
    return 0;
}

static int ledger_add(AURACtx *ctx, const uint32_t *items, size_t len,
                      const uint64_t *support, size_t supp, double occupancy) {
    uint64_t h1, h2;
    support_hash(support, ctx->words, &h1, &h2);
    if (ledger_has(ctx, support, supp, h1, h2)) {
        ctx->ledger_duplicates++;
        return 0;
    }
    if (ctx->ledger.count == ctx->ledger.cap) {
        size_t nc = ctx->ledger.cap ? ctx->ledger.cap * 2 : 256;
        AURALedgerEntry *nd = (AURALedgerEntry *)realloc(ctx->ledger.data, nc * sizeof(*nd));
        if (!nd) return -1;
        ctx->ledger.data = nd;
        ctx->ledger.cap = nc;
    }
    AURALedgerEntry *e = &ctx->ledger.data[ctx->ledger.count++];
    e->h1 = h1;
    e->h2 = h2;
    e->support = supp;
    e->occupancy = occupancy;
    e->support_bits = (uint64_t *)malloc(ctx->words * sizeof(uint64_t));
    e->items = (uint32_t *)malloc(len * sizeof(uint32_t));
    if (!e->support_bits || !e->items) {
        free(e->support_bits);
        free(e->items);
        return -1;
    }
    memcpy(e->support_bits, support, ctx->words * sizeof(uint64_t));
    memcpy(e->items, items, len * sizeof(uint32_t));
    e->len = len;
    ctx->total_output_items += len;
    if (ctx->top_k && ctx->ledger.count <= ctx->top_k) ctx->topk_updates++;
    return 1;
}

static void ledger_free(AURALedger *l) {
    for (size_t i = 0; i < l->count; i++) {
        free(l->data[i].support_bits);
        free(l->data[i].items);
    }
    free(l->data);
    memset(l, 0, sizeof(*l));
}

static double residual_envelope(AURACtx *ctx, size_t prefix_len, const uint64_t *support,
                                const size_t *tail, size_t tail_count) {
    size_t supp = bitset_count(support, ctx->words);
    if (supp < ctx->min_support) return 0.0;
    if (tail_count == 0) return exact_average_occupancy(ctx, support, supp, prefix_len);

    uint16_t *rem = (uint16_t *)calloc(ctx->ntrans, sizeof(uint16_t));
    double *vals = (double *)malloc(supp * sizeof(double));
    if (!rem || !vals) {
        free(rem);
        free(vals);
        return 1.0;
    }

    for (size_t t = 0; t < tail_count; t++) {
        const uint64_t *ib = item_bitset(ctx, tail[t]);
        for (size_t w = 0; w < ctx->words; w++) {
            uint64_t x = support[w] & ib[w];
            while (x) {
                unsigned bit = (unsigned)__builtin_ctzll(x);
                size_t tid = (w << 6) + bit;
                if (tid < ctx->ntrans && rem[tid] < UINT16_MAX) rem[tid]++;
                x &= x - 1;
            }
        }
    }

    size_t nvals = 0;
    for (size_t w = 0; w < ctx->words; w++) {
        uint64_t x = support[w];
        while (x) {
            unsigned bit = (unsigned)__builtin_ctzll(x);
            size_t tid = (w << 6) + bit;
            if (tid < ctx->ntrans) {
                vals[nvals++] = ((double)prefix_len + (double)rem[tid]) * ctx->reciprocal_len[tid];
            }
            x &= x - 1;
        }
    }
    qsort(vals, nvals, sizeof(double), cmp_double_desc);

    double best = 0.0;
    double psum = 0.0;
    for (size_t u = 1; u <= nvals; u++) {
        psum += vals[u - 1];
        if (u >= ctx->min_support) {
            double bound = psum / (double)u;
            if (bound > best) best = bound;
        }
    }
    free(vals);
    free(rem);
    return best > 1.0 ? 1.0 : best;
}

static double residual_sum_envelope(AURACtx *ctx, size_t prefix_len, const uint64_t *support,
                                    const size_t *tail, size_t tail_count) {
    size_t supp = bitset_count(support, ctx->words);
    if (supp < ctx->min_support) return 0.0;
    if (tail_count == 0) return exact_summed_occupancy(ctx, support, prefix_len);

    uint16_t *rem = (uint16_t *)calloc(ctx->ntrans, sizeof(uint16_t));
    if (!rem) return ctx->threshold_value;
    for (size_t t = 0; t < tail_count; t++) {
        const uint64_t *ib = item_bitset(ctx, tail[t]);
        for (size_t w = 0; w < ctx->words; w++) {
            uint64_t x = support[w] & ib[w];
            while (x) {
                unsigned bit = (unsigned)__builtin_ctzll(x);
                size_t tid = (w << 6) + bit;
                if (tid < ctx->ntrans && rem[tid] < UINT16_MAX) rem[tid]++;
                x &= x - 1;
            }
        }
    }

    double bound = 0.0;
    for (size_t w = 0; w < ctx->words; w++) {
        uint64_t x = support[w];
        while (x) {
            unsigned bit = (unsigned)__builtin_ctzll(x);
            size_t tid = (w << 6) + bit;
            if (tid < ctx->ntrans) {
                bound += ((double)prefix_len + (double)rem[tid]) * ctx->reciprocal_len[tid];
            }
            x &= x - 1;
        }
    }
    free(rem);
    return bound;
}

static void aura_search(AURACtx *ctx, const uint32_t *prefix, size_t prefix_len,
                        const uint64_t *support, const size_t *tail, size_t tail_count,
                        double path_bound) {
    if (should_stop(ctx) || path_bound + 1e-12 < ctx->threshold_value) {
        if (path_bound + 1e-12 < ctx->threshold_value) ctx->pruned_envelope++;
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
        size_t *next_tail = (size_t *)malloc((tail_count - pos - 1) * sizeof(size_t));
        if (!closed || (!next_tail && tail_count > pos + 1)) {
            free(closed);
            free(next_tail);
            ctx->limited = 1;
            arena_rewind(&ctx->arena, mark);
            return;
        }

        size_t closed_len = 0;
        if (prefix_len) {
            memcpy(closed, prefix, prefix_len * sizeof(uint32_t));
            closed_len = prefix_len;
        }
        closed[closed_len++] = ctx->active_items[tail[pos]];

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
        double score = ctx->summed_occupancy_mode ? exact_summed_occupancy(ctx, child, closed_len) : avg_occ;
        if (score + 1e-12 >= ctx->threshold_value) {
            ctx->raw_hoi_count++;
            ledger_add(ctx, closed, closed_len, child, supp, avg_occ);
        }

        double local = ctx->summed_occupancy_mode
            ? residual_sum_envelope(ctx, closed_len, child, next_tail, next_count)
            : residual_envelope(ctx, closed_len, child, next_tail, next_count);
        double next_bound = local < path_bound ? local : path_bound;
        if (next_bound + 1e-12 >= ctx->threshold_value && next_count > 0) {
            aura_search(ctx, closed, closed_len, child, next_tail, next_count, next_bound);
        } else if (next_count > 0) {
            ctx->pruned_envelope++;
        }

        free(next_tail);
        free(closed);
        arena_rewind(&ctx->arena, mark);
    }
}

static void aura_search_raw(AURACtx *ctx, const uint32_t *prefix, size_t prefix_len,
                            const uint64_t *support, const size_t *tail, size_t tail_count,
                            double path_bound) {
    if (should_stop(ctx) || path_bound + 1e-12 < ctx->threshold_value) {
        if (path_bound + 1e-12 < ctx->threshold_value) ctx->pruned_envelope++;
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

        double sum_recip = 0.0;
        double ubo_bound = path_bound;
        if (ctx->summed_occupancy_mode) {
            support_summed_stats(ctx, child, &sum_recip, &ubo_bound);
            if (ubo_bound + 1e-12 < ctx->threshold_value) {
                ctx->pruned_envelope++;
                arena_rewind(&ctx->arena, mark);
                continue;
            }
        }

        uint32_t *next_prefix = (uint32_t *)malloc((prefix_len + 1) * sizeof(uint32_t));
        size_t *next_tail = (size_t *)malloc((tail_count - pos - 1) * sizeof(size_t));
        if (!next_prefix || (!next_tail && tail_count > pos + 1)) {
            free(next_prefix);
            free(next_tail);
            ctx->limited = 1;
            arena_rewind(&ctx->arena, mark);
            return;
        }
        if (prefix_len) memcpy(next_prefix, prefix, prefix_len * sizeof(uint32_t));
        next_prefix[prefix_len] = ctx->active_items[tail[pos]];
        size_t next_len = prefix_len + 1;

        size_t next_count = 0;
        for (size_t j = pos + 1; j < tail_count; j++) next_tail[next_count++] = tail[j];

        double score = ctx->summed_occupancy_mode
            ? (double)next_len * sum_recip
            : exact_average_occupancy(ctx, child, supp, next_len);
        if (score + 1e-12 >= ctx->threshold_value) {
            ctx->raw_hoi_count++;
            ctx->raw_total_output_items += next_len;
        }

        double local = ctx->summed_occupancy_mode
            ? ubo_bound
            : residual_envelope(ctx, next_len, child, next_tail, next_count);
        double next_bound = local < path_bound ? local : path_bound;
        if (next_bound + 1e-12 >= ctx->threshold_value && next_count > 0) {
            aura_search_raw(ctx, next_prefix, next_len, child, next_tail, next_count, next_bound);
        } else if (next_count > 0) {
            ctx->pruned_envelope++;
        }

        free(next_tail);
        free(next_prefix);
        arena_rewind(&ctx->arena, mark);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_TRANSACTIONAL) return DM_ERROR_INCOMPATIBLE;
    DM_AURA_HOI_Params *p = (DM_AURA_HOI_Params *)params;
    AURACtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.trans = (DM_Trans_Simple *)ds->payload;
    ctx.ntrans = ds->count;
    ctx.max_id = ds->max_id;
    ctx.words = (ctx.ntrans + 63) / 64;
    ctx.min_occupancy = p ? p->min_occupancy : 0.5;
    ctx.summed_occupancy_mode = p ? p->summed_occupancy_mode : 0;
    ctx.threshold_value = ctx.summed_occupancy_mode
        ? ((ctx.min_occupancy < 1.0) ? ctx.min_occupancy * (double)ctx.ntrans : ctx.min_occupancy)
        : ctx.min_occupancy;
    ctx.min_support = (p && p->min_support) ? p->min_support : (size_t)ceil(ctx.min_occupancy * (double)ctx.ntrans);
    if (ctx.min_support < 1) ctx.min_support = 1;
    ctx.max_patterns = p ? p->max_patterns : 0;
    ctx.max_seconds = p ? p->max_seconds : 0.0;
    ctx.emit_raw_view = p ? p->emit_raw_view : 0;
    ctx.top_k = p ? p->top_k : 0;
    ctx.start_clock = clock();

    ctx.transaction_len = (uint32_t *)malloc(ctx.ntrans * sizeof(uint32_t));
    ctx.reciprocal_len = (double *)malloc(ctx.ntrans * sizeof(double));
    uint32_t *support_count = (uint32_t *)calloc((size_t)ctx.max_id + 1, sizeof(uint32_t));
    if (!ctx.transaction_len || !ctx.reciprocal_len || !support_count) return DM_ERROR_MEMORY;

    for (size_t t = 0; t < ctx.ntrans; t++) {
        ctx.transaction_len[t] = (uint32_t)ctx.trans[t].count;
        if (ctx.transaction_len[t] > ctx.max_transaction_len) {
            ctx.max_transaction_len = ctx.transaction_len[t];
        }
        ctx.reciprocal_len[t] = ctx.trans[t].count ? 1.0 / (double)ctx.trans[t].count : 0.0;
        for (size_t j = 0; j < ctx.trans[t].count; j++) support_count[ctx.trans[t].items[j]]++;
    }
    ctx.len_bucket_sum = (double *)calloc((size_t)ctx.max_transaction_len + 1, sizeof(double));
    ctx.len_bucket_stamp = (uint32_t *)calloc((size_t)ctx.max_transaction_len + 1, sizeof(uint32_t));
    ctx.len_bucket_epoch = 1;
    if (!ctx.len_bucket_sum || !ctx.len_bucket_stamp) return DM_ERROR_MEMORY;

    for (uint32_t i = 0; i <= ctx.max_id; i++) {
        if (support_count[i] >= ctx.min_support) ctx.active_count++;
    }
    ctx.active_items = (uint32_t *)malloc(ctx.active_count * sizeof(uint32_t));
    ctx.item_bits = (uint64_t *)calloc(ctx.active_count * ctx.words, sizeof(uint64_t));
    uint32_t *id_to_active = (uint32_t *)malloc(((size_t)ctx.max_id + 1) * sizeof(uint32_t));
    if (!ctx.active_items || !ctx.item_bits || !id_to_active) return DM_ERROR_MEMORY;
    for (uint32_t i = 0; i <= ctx.max_id; i++) id_to_active[i] = UINT32_MAX;
    size_t idx = 0;
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
    if (!tail) return DM_ERROR_MEMORY;
    for (size_t i = 0; i < ctx.active_count; i++) tail[i] = i;

    double root_bound;
    if (ctx.summed_occupancy_mode) {
        double root_sum = 0.0;
        support_summed_stats(&ctx, root, &root_sum, &root_bound);
    } else {
        root_bound = residual_envelope(&ctx, 0, root, tail, ctx.active_count);
    }
    printf("[AURA-HOI] transactions=%zu active_items=%zu minsup=%zu minocc=%.6f threshold=%.6f mode=%s root_residual_envelope=%.6f view=%s\n",
           ctx.ntrans, ctx.active_count, ctx.min_support, ctx.min_occupancy, ctx.threshold_value,
           ctx.summed_occupancy_mode ? "summed-compatible" : "average", root_bound,
           ctx.emit_raw_view ? "raw-fullset" : "closed-ledger");
    if (ctx.emit_raw_view) {
        aura_search_raw(&ctx, NULL, 0, root, tail, ctx.active_count, root_bound);
    } else {
        aura_search(&ctx, NULL, 0, root, tail, ctx.active_count, root_bound);
    }

    if (ctx.emit_raw_view) {
        printf("[AURA-HOI] Complete. Raw fullset HO itemsets found: %zu\n", ctx.raw_hoi_count);
    } else {
        printf("[AURA-HOI] Complete. Auditable closed HOI representatives found: %zu\n", ctx.ledger.count);
    }
    printf("[AURA-HOI] raw_accepts=%zu support_classes=%zu ledger_duplicates=%zu topk_updates=%zu\n",
           ctx.raw_hoi_count, ctx.ledger.count, ctx.ledger_duplicates, ctx.topk_updates);
    printf("[AURA-HOI] visited_nodes=%zu pruned_support=%zu pruned_backward=%zu pruned_envelope=%zu closure_jumps=%zu arena_high_water_bytes=%zu limited=%s\n",
           ctx.visited_nodes, ctx.pruned_support, ctx.pruned_backward, ctx.pruned_envelope,
           ctx.closure_jumps, ctx.arena.high_water * sizeof(uint64_t), ctx.limited ? "yes" : "no");

    if (ctx.emit_raw_view) {
        dm_bench_record_results(ctx.raw_hoi_count, ctx.raw_total_output_items);
    } else {
        dm_bench_record_results(ctx.ledger.count, ctx.total_output_items);
    }

    free(tail);
    ledger_free(&ctx.ledger);
    arena_free(&ctx.arena);
    free(ctx.active_items);
    free(ctx.item_bits);
    free(ctx.transaction_len);
    free(ctx.reciprocal_len);
    free(ctx.len_bucket_sum);
    free(ctx.len_bucket_stamp);
    return DM_SUCCESS;
}

DM_Algorithm aura_hoi_algo = {
    .id = "aura_hoi",
    .name = "AURA-HOI",
    .description = "Auditable support-class representative high-occupancy itemset mining with residual occupancy envelope pruning.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(aura_hoi_algo)
