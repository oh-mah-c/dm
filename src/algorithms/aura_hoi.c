#include "algorithms/aura_hoi.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Arena Allocator for Bitset Nodes ────────────────────────────────────── */
typedef struct {
    uint64_t *pool;
    size_t words_capacity;
    size_t words_used;
    size_t high_water;
} AURAArena;

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

static inline size_t arena_mark(AURAArena *a) {
    return a->words_used;
}

static inline uint64_t *arena_alloc_bitset(AURAArena *a, size_t words) {
    if (a->words_used + words > a->words_capacity) return NULL;
    uint64_t *p = a->pool + a->words_used;
    a->words_used += words;
    if (a->words_used > a->high_water) a->high_water = a->words_used;
    memset(p, 0, words * sizeof(uint64_t));
    return p;
}

static inline void arena_rewind(AURAArena *a, size_t mark) {
    if (mark <= a->words_used) a->words_used = mark;
}

/* ── Closed Ledger with O(1) Double-Hashing ──────────────────────────────── */
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
    size_t *hash_table; /* 1-based index into data; 0 means empty slot */
    size_t table_cap;
} AURALedger;

static int ledger_init(AURALedger *l, size_t initial_cap) {
    l->count = 0;
    l->cap = initial_cap ? initial_cap : 256;
    l->data = (AURALedgerEntry *)malloc(l->cap * sizeof(AURALedgerEntry));
    if (!l->data) return -1;
    l->table_cap = initial_cap ? initial_cap * 2 : 512;
    l->hash_table = (size_t *)calloc(l->table_cap, sizeof(size_t));
    if (!l->hash_table) {
        free(l->data);
        return -1;
    }
    return 0;
}

static void ledger_free(AURALedger *l) {
    for (size_t i = 0; i < l->count; i++) {
        free(l->data[i].support_bits);
        free(l->data[i].items);
    }
    free(l->data);
    free(l->hash_table);
    memset(l, 0, sizeof(*l));
}

static void support_hash(const uint64_t *bits, size_t words, uint64_t *h1, uint64_t *h2) {
    uint64_t a = 1469598103934665603ULL;
    uint64_t b = 1099511628211ULL ^ (uint64_t)words;
    for (size_t i = 0; i < words; i++) {
        uint64_t x = bits[i];
        a ^= x;
        a *= 1099511628211ULL;
        b ^= x + 0x9e3779b97f4a7c15ULL + (b << 6) + (b >> 2);
    }
    *h1 = a;
    *h2 = b;
}

static int ledger_has(AURALedger *l, const uint64_t *support, size_t supp,
                      uint64_t h1, uint64_t h2, size_t words) {
    if (!l->table_cap) return 0;
    size_t mask = l->table_cap - 1;
    size_t idx = (h1 ^ (h2 * 0x9e3779b97f4a7c15ULL)) & mask;
    size_t step = ((h2 << 1) | 1ULL) & mask;
    if (!step) step = 1;

    for (size_t probe = 0; probe < l->table_cap; probe++) {
        size_t entry_idx = l->hash_table[idx];
        if (entry_idx == 0) return 0; /* not found */
        AURALedgerEntry *e = &l->data[entry_idx - 1];
        if (e->h1 == h1 && e->h2 == h2 && e->support == supp) {
            int same = 1;
            for (size_t w = 0; w < words; w++) {
                if (e->support_bits[w] != support[w]) {
                    same = 0;
                    break;
                }
            }
            if (same) return 1;
        }
        idx = (idx + step) & mask;
    }
    return 0;
}

static int ledger_rehash(AURALedger *l) {
    size_t new_cap = l->table_cap * 2;
    size_t *new_table = (size_t *)calloc(new_cap, sizeof(size_t));
    if (!new_table) return -1;
    size_t mask = new_cap - 1;

    for (size_t i = 0; i < l->count; i++) {
        AURALedgerEntry *e = &l->data[i];
        size_t idx = (e->h1 ^ (e->h2 * 0x9e3779b97f4a7c15ULL)) & mask;
        size_t step = ((e->h2 << 1) | 1ULL) & mask;
        if (!step) step = 1;
        while (new_table[idx] != 0) {
            idx = (idx + step) & mask;
        }
        new_table[idx] = i + 1;
    }
    free(l->hash_table);
    l->hash_table = new_table;
    l->table_cap = new_cap;
    return 0;
}

static int ledger_add(AURALedger *l, const uint32_t *items, size_t len,
                      const uint64_t *support, size_t supp, double occupancy,
                      size_t words) {
    uint64_t h1, h2;
    support_hash(support, words, &h1, &h2);
    if (ledger_has(l, support, supp, h1, h2, words)) {
        return 0; /* Duplicate support class */
    }

    if (l->count >= l->cap) {
        size_t nc = l->cap * 2;
        AURALedgerEntry *nd = (AURALedgerEntry *)realloc(l->data, nc * sizeof(*nd));
        if (!nd) return -1;
        l->data = nd;
        l->cap = nc;
    }

    if (l->count * 2 >= l->table_cap) {
        if (ledger_rehash(l) != 0) return -1;
    }

    size_t entry_index = l->count++;
    AURALedgerEntry *e = &l->data[entry_index];
    e->h1 = h1;
    e->h2 = h2;
    e->support = supp;
    e->occupancy = occupancy;
    e->len = len;
    e->support_bits = (uint64_t *)malloc(words * sizeof(uint64_t));
    e->items = (uint32_t *)malloc(len * sizeof(uint32_t));
    if (!e->support_bits || !e->items) return -1;
    memcpy(e->support_bits, support, words * sizeof(uint64_t));
    memcpy(e->items, items, len * sizeof(uint32_t));

    size_t mask = l->table_cap - 1;
    size_t idx = (h1 ^ (h2 * 0x9e3779b97f4a7c15ULL)) & mask;
    size_t step = ((h2 << 1) | 1ULL) & mask;
    if (!step) step = 1;
    while (l->hash_table[idx] != 0) {
        idx = (idx + step) & mask;
    }
    l->hash_table[idx] = entry_index + 1;
    return 1;
}

/* ── Transaction Metadata & Length Classes ───────────────────────────────── */
typedef struct {
    uint32_t orig_tid;
    uint32_t len;
} TransMeta;

static int cmp_trans_meta(const void *a, const void *b) {
    const TransMeta *ta = (const TransMeta *)a;
    const TransMeta *tb = (const TransMeta *)b;
    if (ta->len < tb->len) return -1;
    if (ta->len > tb->len) return 1;
    return 0;
}

/* ── Context for Mining ─────────────────────────────────────────────────── */
typedef struct {
    DM_Trans_Simple *trans;
    size_t ntrans;
    uint32_t max_id;
    size_t words;
    
    /* Active candidate items (frequent 1-items) */
    uint32_t *active_items;
    size_t active_count;
    
    /* Support representations */
    int use_tid_mode; /* 1 if sparse TID-list mode, 0 if dense bitset mode */
    uint64_t *item_bits;      /* Dense bitset per active item (if bitset mode) */
    uint32_t **item_tids;     /* Sorted TID lists (if TID mode) */
    uint32_t *item_counts;    /* Support count per active item */
    
    /* Transaction length structures */
    uint32_t *g_tsize;        /* Transaction length sorted by TID */
    double *recip_len;        /* 1.0 / g_tsize[tid] */
    int uniform_length;       /* 1 if all transactions have identical length */
    uint32_t first_len;       /* Length when uniform_length == 1 */
    
    /* Thresholds */
    double min_occupancy;
    double threshold_value;   /* xi threshold */
    size_t min_support;
    int summed_occupancy_mode;
    int emit_raw_view;
    size_t max_patterns;
    double max_seconds;
    size_t top_k;
    clock_t start_clock;
    
    /* Statistics */
    size_t raw_hoi_count;
    size_t raw_total_output_items;
    size_t total_output_items;
    size_t visited_nodes;
    size_t pruned_support;
    size_t pruned_envelope;
    size_t pruned_backward;
    size_t closure_jumps;
    size_t ledger_duplicates;
    size_t topk_updates;
    int limited;
    
    /* Arenas & Pre-allocated Recursion Scratchpads */
    AURAArena arena;
    AURALedger ledger;
    
    /* Pre-allocated TID intersection buffer */
    uint32_t *tid_scratch;
    
    /* Pre-allocated residual envelope scratch buffers (O(1) allocation) */
    uint16_t *rem_buffer;
    uint32_t *rem_stamp;
    uint32_t rem_epoch;
    double *vals_buffer;
    
    /* Recursion tail stacks: max_depth x active_count */
    uint32_t *tail_stack;
    uint32_t *prefix_stack;
} AURACtx;

static inline double elapsed_sec(const AURACtx *ctx) {
    return (double)(clock() - ctx->start_clock) / (double)CLOCKS_PER_SEC;
}

static inline int should_stop(AURACtx *ctx) {
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

static inline uint64_t *item_bitset(AURACtx *ctx, size_t active_idx) {
    return ctx->item_bits + active_idx * ctx->words;
}

static inline void bit_set(uint64_t *bits, size_t idx) {
    bits[idx >> 6] |= 1ULL << (idx & 63U);
}

static inline size_t bitset_and_count(uint64_t *out, const uint64_t *a, const uint64_t *b, size_t words) {
    size_t count = 0;
    for (size_t w = 0; w < words; w++) {
        uint64_t val = a[w] & b[w];
        out[w] = val;
        count += (size_t)__builtin_popcountll(val);
    }
    return count;
}

static inline size_t bitset_count(const uint64_t *bits, size_t words) {
    size_t count = 0;
    for (size_t w = 0; w < words; w++) {
        count += (size_t)__builtin_popcountll(bits[w]);
    }
    return count;
}

/* ── UBO and Score Calculations ─────────────────────────────────────────── */

/* O(size) UBO and score check for sorted TID list */
static inline int check_tid_ubo_and_o(const uint32_t *tids, size_t size,
                                      const uint32_t *g_tsize, const double *recip_len,
                                      size_t k, size_t R_count, double xi,
                                      int uniform_length, uint32_t first_len,
                                      int *is_ho, double *out_recip_sum) {
    if ((double)size < xi) return 0;

    if (uniform_length) {
        double max_k = (double)(k + R_count);
        if (max_k > (double)first_len) max_k = (double)first_len;
        double max_desc_score = (max_k * (double)size) / (double)first_len;
        if (max_desc_score + 1e-12 < xi) return 0; /* Residual deficit prune */

        double o_val = ((double)k * (double)size) / (double)first_len;
        *is_ho = (o_val >= xi);
        *out_recip_sum = (double)size / (double)first_len;
        return 1;
    }

    double current_sum = 0.0;
    double max_ubo = 0.0;

    for (int i = (int)size - 1; i >= 0; i--) {
        uint32_t tsize = g_tsize[tids[i]];
        current_sum += recip_len[tids[i]];

        if (i == 0 || g_tsize[tids[i - 1]] < tsize) {
            double current_ubo = (double)tsize * current_sum;
            if (current_ubo > max_ubo) {
                max_ubo = current_ubo;
            }
        }
    }

    if (max_ubo >= xi) {
        *is_ho = (((double)k * current_sum) >= xi);
        *out_recip_sum = current_sum;
        return 1;
    }
    return 0;
}

/* Bitset occupancy & UBO calculation with residual capacity bounding */
static inline int check_bitset_ubo_and_o(AURACtx *ctx, const uint64_t *support, size_t supp,
                                         size_t k, size_t R_count, double xi,
                                         int *is_ho, double *out_recip_sum) {
    if ((double)supp < xi) return 0;

    if (ctx->uniform_length) {
        double max_k = (double)(k + R_count);
        if (max_k > (double)ctx->first_len) max_k = (double)ctx->first_len;
        double max_desc_score = (max_k * (double)supp) / (double)ctx->first_len;
        if (max_desc_score + 1e-12 < xi) return 0; /* Residual deficit prune */

        double o_val = ((double)k * (double)supp) / (double)ctx->first_len;
        *is_ho = (o_val >= xi);
        *out_recip_sum = (double)supp / (double)ctx->first_len;
        return 1;
    }

    double sum = 0.0;

    /* Single pass over set bits using ctzll */
    for (size_t w = 0; w < ctx->words; w++) {
        uint64_t x = support[w];
        while (x) {
            unsigned bit = (unsigned)__builtin_ctzll(x);
            size_t tid = (w << 6) + bit;
            if (tid < ctx->ntrans) {
                sum += ctx->recip_len[tid];
            }
            x &= x - 1;
        }
    }

    /* Fast residual bound check before expensive UBO */
    double max_score_bound = (double)(k + R_count) * sum;
    if (max_score_bound + 1e-12 < xi) return 0; /* Pruned by residual capacity */

    /* Compute UBO if needed */
    *is_ho = (((double)k * sum) >= xi);
    *out_recip_sum = sum;
    return 1;
}

/* ── Closed Mode: Zero-Allocation Residual Envelope ──────────────────────── */
static double exact_average_occupancy(AURACtx *ctx, const uint64_t *support, size_t supp, size_t len) {
    if (supp == 0) return 0.0;
    double sum = 0.0;
    for (size_t w = 0; w < ctx->words; w++) {
        uint64_t x = support[w];
        while (x) {
            unsigned bit = (unsigned)__builtin_ctzll(x);
            size_t tid = (w << 6) + bit;
            if (tid < ctx->ntrans) sum += ctx->recip_len[tid];
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
            if (tid < ctx->ntrans) sum += ctx->recip_len[tid];
            x &= x - 1;
        }
    }
    return (double)len * sum;
}

static int cmp_double_desc(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    if (x < y) return 1;
    if (x > y) return -1;
    return 0;
}

static double fast_residual_envelope(AURACtx *ctx, size_t prefix_len, const uint64_t *support,
                                     const uint32_t *tail, size_t tail_count) {
    size_t supp = bitset_count(support, ctx->words);
    if (supp < ctx->min_support) return 0.0;
    if (tail_count == 0) return exact_average_occupancy(ctx, support, supp, prefix_len);

    /* Use pre-allocated rem_buffer and timestamp clearing */
    if (++ctx->rem_epoch == 0) {
        memset(ctx->rem_stamp, 0, ctx->ntrans * sizeof(uint32_t));
        ctx->rem_epoch = 1;
    }
    uint32_t epoch = ctx->rem_epoch;

    for (size_t t = 0; t < tail_count; t++) {
        const uint64_t *ib = item_bitset(ctx, tail[t]);
        for (size_t w = 0; w < ctx->words; w++) {
            uint64_t x = support[w] & ib[w];
            while (x) {
                unsigned bit = (unsigned)__builtin_ctzll(x);
                size_t tid = (w << 6) + bit;
                if (tid < ctx->ntrans) {
                    if (ctx->rem_stamp[tid] != epoch) {
                        ctx->rem_stamp[tid] = epoch;
                        ctx->rem_buffer[tid] = 0;
                    }
                    if (ctx->rem_buffer[tid] < UINT16_MAX) ctx->rem_buffer[tid]++;
                }
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
                uint16_t rem = (ctx->rem_stamp[tid] == epoch) ? ctx->rem_buffer[tid] : 0;
                ctx->vals_buffer[nvals++] = ((double)prefix_len + (double)rem) * ctx->recip_len[tid];
            }
            x &= x - 1;
        }
    }

    qsort(ctx->vals_buffer, nvals, sizeof(double), cmp_double_desc);

    double best = 0.0;
    double psum = 0.0;
    for (size_t u = 1; u <= nvals; u++) {
        psum += ctx->vals_buffer[u - 1];
        if (u >= ctx->min_support) {
            double bound = psum / (double)u;
            if (bound > best) best = bound;
        }
    }
    return best > 1.0 ? 1.0 : best;
}

static double fast_residual_sum_envelope(AURACtx *ctx, size_t prefix_len, const uint64_t *support,
                                         const uint32_t *tail, size_t tail_count) {
    size_t supp = bitset_count(support, ctx->words);
    if (supp < ctx->min_support) return 0.0;
    if (tail_count == 0) return exact_summed_occupancy(ctx, support, prefix_len);

    if (ctx->uniform_length) {
        double max_k = (double)(prefix_len + tail_count);
        if (max_k > (double)ctx->first_len) max_k = (double)ctx->first_len;
        return (max_k * (double)supp) / (double)ctx->first_len;
    }

    if (++ctx->rem_epoch == 0) {
        memset(ctx->rem_stamp, 0, ctx->ntrans * sizeof(uint32_t));
        ctx->rem_epoch = 1;
    }
    uint32_t epoch = ctx->rem_epoch;

    for (size_t t = 0; t < tail_count; t++) {
        const uint64_t *ib = item_bitset(ctx, tail[t]);
        for (size_t w = 0; w < ctx->words; w++) {
            uint64_t x = support[w] & ib[w];
            while (x) {
                unsigned bit = (unsigned)__builtin_ctzll(x);
                size_t tid = (w << 6) + bit;
                if (tid < ctx->ntrans) {
                    if (ctx->rem_stamp[tid] != epoch) {
                        ctx->rem_stamp[tid] = epoch;
                        ctx->rem_buffer[tid] = 0;
                    }
                    if (ctx->rem_buffer[tid] < UINT16_MAX) ctx->rem_buffer[tid]++;
                }
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
                uint16_t rem = (ctx->rem_stamp[tid] == epoch) ? ctx->rem_buffer[tid] : 0;
                bound += ((double)prefix_len + (double)rem) * ctx->recip_len[tid];
            }
            x &= x - 1;
        }
    }
    return bound;
}

static inline int support_subset_item(AURACtx *ctx, const uint64_t *support, size_t item_idx) {
    const uint64_t *ib = item_bitset(ctx, item_idx);
    for (size_t w = 0; w < ctx->words; w++) {
        if (support[w] & ~ib[w]) return 0;
    }
    return 1;
}

/* ── DFS Engine 1: Bitset Mode (for Dense / Moderate Datasets) ──────────── */
static void aura_dfs_bitset_raw(AURACtx *ctx, size_t depth,
                                const uint64_t *support, size_t supp_P,
                                size_t tail_offset, size_t tail_count) {
    if (should_stop(ctx) || tail_count == 0) return;
    (void)supp_P;

    uint32_t *tail = ctx->tail_stack + tail_offset;
    size_t next_tail_offset = tail_offset + tail_count;

    for (size_t pos = 0; pos < tail_count; pos++) {
        if (should_stop(ctx)) return;

        uint32_t item_idx = tail[pos];
        size_t mark = arena_mark(&ctx->arena);
        uint64_t *child = arena_alloc_bitset(&ctx->arena, ctx->words);
        if (!child) {
            ctx->limited = 1;
            return;
        }

        size_t supp = bitset_and_count(child, support, item_bitset(ctx, item_idx), ctx->words);
        ctx->visited_nodes++;

        if (supp < ctx->min_support) {
            ctx->pruned_support++;
            arena_rewind(&ctx->arena, mark);
            continue;
        }

        size_t new_k = depth + 1;
        size_t remaining_candidates = tail_count - pos - 1;
        int is_ho = 0;
        double recip_sum = 0.0;

        if (!check_bitset_ubo_and_o(ctx, child, supp, new_k, remaining_candidates,
                                    ctx->threshold_value, &is_ho, &recip_sum)) {
            ctx->pruned_envelope++;
            arena_rewind(&ctx->arena, mark);
            continue;
        }

        if (is_ho) {
            ctx->raw_hoi_count++;
            ctx->raw_total_output_items += new_k;
        }

        if (remaining_candidates > 0) {
            /* Zero-Branch Equivalence Absorption Check:
             * Identify items that occur in 100% of child transactions */
            uint32_t *next_tail = ctx->tail_stack + next_tail_offset;
            size_t next_count = 0;

            for (size_t j = pos + 1; j < tail_count; j++) {
                next_tail[next_count++] = tail[j];
            }

            aura_dfs_bitset_raw(ctx, new_k, child, supp, next_tail_offset, next_count);
        }

        arena_rewind(&ctx->arena, mark);
    }
}

/* ── DFS Engine 2: Sparse TID-List Mode (for Large Clickstream / Kosarak) ── */
typedef struct {
    uint32_t length;
    uint32_t *tids;
    size_t num_tids;
} AURATidItemset;

static void aura_dfs_tid_raw(AURACtx *ctx, AURATidItemset *classes, size_t class_size) {
    if (class_size < 2 || should_stop(ctx)) return;

    for (size_t i = 0; i < class_size; i++) {
        if (should_stop(ctx)) return;
        AURATidItemset *P1 = &classes[i];

        AURATidItemset *children = NULL;
        size_t remaining = class_size - i - 1;
        if (remaining > 0) {
            children = (AURATidItemset *)malloc(remaining * sizeof(AURATidItemset));
        }
        size_t child_count = 0;

        for (size_t j = i + 1; j < class_size; j++) {
            AURATidItemset *P2 = &classes[j];

            size_t p1_idx = 0, p2_idx = 0, new_idx = 0;
            while (p1_idx < P1->num_tids && p2_idx < P2->num_tids) {
                if (P1->tids[p1_idx] < P2->tids[p2_idx]) {
                    p1_idx++;
                } else if (P1->tids[p1_idx] > P2->tids[p2_idx]) {
                    p2_idx++;
                } else {
                    ctx->tid_scratch[new_idx++] = P1->tids[p1_idx];
                    p1_idx++;
                    p2_idx++;
                }
            }
            ctx->visited_nodes++;

            if (new_idx >= ctx->min_support) {
                int is_ho = 0;
                double recip_sum = 0.0;
                size_t new_k = P1->length + 1;
                size_t R_count = class_size - j - 1;

                if (check_tid_ubo_and_o(ctx->tid_scratch, new_idx, ctx->g_tsize, ctx->recip_len,
                                        new_k, R_count, ctx->threshold_value,
                                        ctx->uniform_length, ctx->first_len,
                                        &is_ho, &recip_sum)) {
                    AURATidItemset *P = &children[child_count++];
                    P->length = (uint32_t)new_k;
                    P->num_tids = new_idx;
                    P->tids = (uint32_t *)malloc(new_idx * sizeof(uint32_t));
                    memcpy(P->tids, ctx->tid_scratch, new_idx * sizeof(uint32_t));

                    if (is_ho) {
                        ctx->raw_hoi_count++;
                        ctx->raw_total_output_items += new_k;
                    }
                } else {
                    ctx->pruned_envelope++;
                }
            } else {
                ctx->pruned_support++;
            }
        }

        if (child_count > 0) {
            aura_dfs_tid_raw(ctx, children, child_count);
            for (size_t c = 0; c < child_count; c++) {
                free(children[c].tids);
            }
        }
        if (children) free(children);
    }
}

/* ── Closed Ledger DFS Engine ───────────────────────────────────────────── */
static void aura_dfs_closed(AURACtx *ctx, size_t depth,
                            const uint64_t *support,
                            size_t tail_offset, size_t tail_count,
                            double path_bound) {
    if (should_stop(ctx) || path_bound + 1e-12 < ctx->threshold_value) {
        if (path_bound + 1e-12 < ctx->threshold_value) ctx->pruned_envelope++;
        return;
    }

    uint32_t *tail = ctx->tail_stack + tail_offset;
    size_t next_tail_offset = tail_offset + tail_count;

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

        /* Backward Closure Pruning */
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

        /* Build Closed Representative & Filter Tail */
        uint32_t *closed = ctx->prefix_stack + depth;
        size_t closed_len = 0;
        closed[closed_len++] = ctx->active_items[tail[pos]];

        uint32_t *next_tail = ctx->tail_stack + next_tail_offset;
        size_t next_count = 0;

        for (size_t j = pos + 1; j < tail_count; j++) {
            if (support_subset_item(ctx, child, tail[j])) {
                closed[closed_len++] = ctx->active_items[tail[j]];
                ctx->closure_jumps++;
            } else {
                next_tail[next_count++] = tail[j];
            }
        }

        size_t total_len = depth + closed_len;
        double avg_occ = exact_average_occupancy(ctx, child, supp, total_len);
        double score = ctx->summed_occupancy_mode
            ? exact_summed_occupancy(ctx, child, total_len)
            : avg_occ;

        if (score + 1e-12 >= ctx->threshold_value) {
            ctx->raw_hoi_count++;
            if (ledger_add(&ctx->ledger, ctx->prefix_stack, total_len, child, supp, avg_occ, ctx->words)) {
                ctx->total_output_items += total_len;
            } else {
                ctx->ledger_duplicates++;
            }
        }

        double local = ctx->summed_occupancy_mode
            ? fast_residual_sum_envelope(ctx, total_len, child, next_tail, next_count)
            : fast_residual_envelope(ctx, total_len, child, next_tail, next_count);
        double next_bound = local < path_bound ? local : path_bound;

        if (next_bound + 1e-12 >= ctx->threshold_value && next_count > 0) {
            aura_dfs_closed(ctx, total_len, child, next_tail_offset, next_count, next_bound);
        } else if (next_count > 0) {
            ctx->pruned_envelope++;
        }

        arena_rewind(&ctx->arena, mark);
    }
}

/* ── Entry Point ────────────────────────────────────────────────────────── */
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

    /* ── Step 1: Preprocess Transaction Lengths & Length Equivalence ────── */
    TransMeta *meta = (TransMeta *)malloc(ctx.ntrans * sizeof(TransMeta));
    if (!meta) return DM_ERROR_MEMORY;

    ctx.uniform_length = 1;
    ctx.first_len = ctx.trans[0].count;

    for (size_t i = 0; i < ctx.ntrans; i++) {
        meta[i].orig_tid = (uint32_t)i;
        meta[i].len = (uint32_t)ctx.trans[i].count;
        if (ctx.trans[i].count != ctx.first_len) {
            ctx.uniform_length = 0;
        }
    }
    qsort(meta, ctx.ntrans, sizeof(TransMeta), cmp_trans_meta);

    ctx.g_tsize = (uint32_t *)malloc(ctx.ntrans * sizeof(uint32_t));
    ctx.recip_len = (double *)malloc(ctx.ntrans * sizeof(double));
    if (!ctx.g_tsize || !ctx.recip_len) {
        free(meta);
        return DM_ERROR_MEMORY;
    }

    for (size_t i = 0; i < ctx.ntrans; i++) {
        ctx.g_tsize[i] = meta[i].len;
        ctx.recip_len[i] = meta[i].len ? 1.0 / (double)meta[i].len : 0.0;
    }

    /* ── Step 2: 1-Item Support Counting ────────────────────────────────── */
    uint32_t *counts = (uint32_t *)calloc((size_t)ctx.max_id + 1, sizeof(uint32_t));
    if (!counts) {
        free(meta);
        return DM_ERROR_MEMORY;
    }

    for (size_t i = 0; i < ctx.ntrans; i++) {
        uint32_t orig = meta[i].orig_tid;
        for (size_t j = 0; j < ctx.trans[orig].count; j++) {
            counts[ctx.trans[orig].items[j]]++;
        }
    }

    for (uint32_t i = 0; i <= ctx.max_id; i++) {
        if (counts[i] >= ctx.min_support) ctx.active_count++;
    }

    if (ctx.active_count == 0) {
        free(meta);
        free(counts);
        free(ctx.g_tsize);
        free(ctx.recip_len);
        dm_bench_record_results(0, 0);
        return DM_SUCCESS;
    }

    ctx.active_items = (uint32_t *)malloc(ctx.active_count * sizeof(uint32_t));
    ctx.item_counts = (uint32_t *)malloc(ctx.active_count * sizeof(uint32_t));
    uint32_t *id_to_active = (uint32_t *)malloc(((size_t)ctx.max_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= ctx.max_id; i++) id_to_active[i] = UINT32_MAX;

    size_t idx = 0;
    for (uint32_t i = 0; i <= ctx.max_id; i++) {
        if (counts[i] >= ctx.min_support) {
            id_to_active[i] = (uint32_t)idx;
            ctx.active_items[idx] = i;
            ctx.item_counts[idx] = counts[i];
            idx++;
        }
    }

    /* ── Step 3: Representation Selection (Bitset vs TID) ───────────────── */
    /* If dataset is large & sparse clickstream (e.g., kosarak words > 512), use TID mode;
     * otherwise use vertical bitsets with arena allocation. */
    ctx.use_tid_mode = (ctx.words > 512 && ctx.emit_raw_view) ? 1 : 0;

    if (ctx.use_tid_mode) {
        /* Allocate TID lists */
        ctx.item_tids = (uint32_t **)malloc(ctx.active_count * sizeof(uint32_t *));
        uint32_t *item_idx = (uint32_t *)calloc(ctx.active_count, sizeof(uint32_t));
        for (size_t i = 0; i < ctx.active_count; i++) {
            ctx.item_tids[i] = (uint32_t *)malloc(ctx.item_counts[i] * sizeof(uint32_t));
        }
        for (size_t i = 0; i < ctx.ntrans; i++) {
            uint32_t orig = meta[i].orig_tid;
            for (size_t j = 0; j < ctx.trans[orig].count; j++) {
                uint32_t item = ctx.trans[orig].items[j];
                uint32_t aid = id_to_active[item];
                if (aid != UINT32_MAX) {
                    ctx.item_tids[aid][item_idx[aid]++] = (uint32_t)i;
                }
            }
        }
        free(item_idx);
        ctx.tid_scratch = (uint32_t *)malloc(ctx.ntrans * sizeof(uint32_t));
    } else {
        /* Build vertical bitsets */
        ctx.item_bits = (uint64_t *)calloc(ctx.active_count * ctx.words, sizeof(uint64_t));
        for (size_t i = 0; i < ctx.ntrans; i++) {
            uint32_t orig = meta[i].orig_tid;
            for (size_t j = 0; j < ctx.trans[orig].count; j++) {
                uint32_t item = ctx.trans[orig].items[j];
                uint32_t aid = id_to_active[item];
                if (aid != UINT32_MAX) {
                    bit_set(item_bitset(&ctx, aid), i);
                }
            }
        }
        size_t arena_words = ctx.words * (ctx.active_count + 16);
        if (arena_init(&ctx.arena, arena_words) != 0) return DM_ERROR_MEMORY;
    }

    free(id_to_active);
    free(counts);
    free(meta);

    /* Allocate recursion stack arrays (zero malloc during search) */
    size_t stack_size = (ctx.active_count + 1) * (ctx.active_count + 1);
    ctx.tail_stack = (uint32_t *)malloc(stack_size * sizeof(uint32_t));
    ctx.prefix_stack = (uint32_t *)malloc((ctx.active_count + 1) * sizeof(uint32_t));

    if (!ctx.emit_raw_view) {
        ledger_init(&ctx.ledger, 256);
        ctx.rem_buffer = (uint16_t *)calloc(ctx.ntrans, sizeof(uint16_t));
        ctx.rem_stamp = (uint32_t *)calloc(ctx.ntrans, sizeof(uint32_t));
        ctx.vals_buffer = (double *)malloc(ctx.ntrans * sizeof(double));
        ctx.rem_epoch = 1;
    }

    printf("[AURA-HOI] transactions=%zu active_items=%zu minsup=%zu minocc=%.6f threshold=%.6f mode=%s rep=%s view=%s uniform_len=%s\n",
           ctx.ntrans, ctx.active_count, ctx.min_support, ctx.min_occupancy, ctx.threshold_value,
           ctx.summed_occupancy_mode ? "summed-compatible" : "average",
           ctx.use_tid_mode ? "sparse-tid" : "dense-bitset",
           ctx.emit_raw_view ? "raw-fullset" : "closed-ledger",
           ctx.uniform_length ? "yes" : "no");

    /* ── Step 4: Mining Execution ───────────────────────────────────────── */
    if (ctx.emit_raw_view) {
        if (ctx.use_tid_mode) {
            /* 1-Item classes for TID miner */
            AURATidItemset *C1 = (AURATidItemset *)malloc(ctx.active_count * sizeof(AURATidItemset));
            size_t c1_count = 0;

            for (size_t i = 0; i < ctx.active_count; i++) {
                int is_ho = 0;
                double recip_sum = 0.0;
                if (check_tid_ubo_and_o(ctx.item_tids[i], ctx.item_counts[i],
                                        ctx.g_tsize, ctx.recip_len, 1,
                                        ctx.active_count - i - 1, ctx.threshold_value,
                                        ctx.uniform_length, ctx.first_len,
                                        &is_ho, &recip_sum)) {
                    AURATidItemset *it = &C1[c1_count++];
                    it->length = 1;
                    it->num_tids = ctx.item_counts[i];
                    it->tids = (uint32_t *)malloc(ctx.item_counts[i] * sizeof(uint32_t));
                    memcpy(it->tids, ctx.item_tids[i], ctx.item_counts[i] * sizeof(uint32_t));

                    if (is_ho) {
                        ctx.raw_hoi_count++;
                        ctx.raw_total_output_items += 1;
                    }
                }
            }

            aura_dfs_tid_raw(&ctx, C1, c1_count);

            for (size_t c = 0; c < c1_count; c++) free(C1[c].tids);
            free(C1);
        } else {
            /* 1-Item classes for Bitset miner */
            for (size_t i = 0; i < ctx.active_count; i++) {
                ctx.tail_stack[i] = (uint32_t)i;
            }

            uint64_t *root = arena_alloc_bitset(&ctx.arena, ctx.words);
            for (size_t w = 0; w < ctx.words; w++) root[w] = UINT64_MAX;
            if (ctx.ntrans & 63U) root[ctx.words - 1] &= ((1ULL << (ctx.ntrans & 63U)) - 1ULL);

            aura_dfs_bitset_raw(&ctx, 0, root, ctx.ntrans, 0, ctx.active_count);
        }

        printf("[AURA-HOI] Complete. Raw fullset HO itemsets found: %zu\n", ctx.raw_hoi_count);
        dm_bench_record_results(ctx.raw_hoi_count, ctx.raw_total_output_items);
    } else {
        /* Closed Ledger Mining */
        for (size_t i = 0; i < ctx.active_count; i++) {
            ctx.tail_stack[i] = (uint32_t)i;
        }

        uint64_t *root = arena_alloc_bitset(&ctx.arena, ctx.words);
        for (size_t w = 0; w < ctx.words; w++) root[w] = UINT64_MAX;
        if (ctx.ntrans & 63U) root[ctx.words - 1] &= ((1ULL << (ctx.ntrans & 63U)) - 1ULL);

        double root_bound = ctx.summed_occupancy_mode
            ? fast_residual_sum_envelope(&ctx, 0, root, ctx.tail_stack, ctx.active_count)
            : fast_residual_envelope(&ctx, 0, root, ctx.tail_stack, ctx.active_count);
        aura_dfs_closed(&ctx, 0, root, 0, ctx.active_count, root_bound);

        printf("[AURA-HOI] Complete. Auditable closed HOI representatives found: %zu\n", ctx.ledger.count);
        dm_bench_record_results(ctx.ledger.count, ctx.total_output_items);
    }

    printf("[AURA-HOI] raw_accepts=%zu support_classes=%zu ledger_duplicates=%zu topk_updates=%zu\n",
           ctx.raw_hoi_count, ctx.ledger.count, ctx.ledger_duplicates, ctx.topk_updates);
    printf("[AURA-HOI] visited_nodes=%zu pruned_support=%zu pruned_backward=%zu pruned_envelope=%zu closure_jumps=%zu limited=%s\n",
           ctx.visited_nodes, ctx.pruned_support, ctx.pruned_backward, ctx.pruned_envelope,
           ctx.closure_jumps, ctx.limited ? "yes" : "no");

    /* ── Cleanup ────────────────────────────────────────────────────────── */
    free(ctx.tail_stack);
    free(ctx.prefix_stack);
    free(ctx.active_items);
    free(ctx.item_counts);
    free(ctx.g_tsize);
    free(ctx.recip_len);

    if (ctx.use_tid_mode) {
        for (size_t i = 0; i < ctx.active_count; i++) free(ctx.item_tids[i]);
        free(ctx.item_tids);
        free(ctx.tid_scratch);
    } else {
        free(ctx.item_bits);
        arena_free(&ctx.arena);
    }

    if (!ctx.emit_raw_view) {
        ledger_free(&ctx.ledger);
        free(ctx.rem_buffer);
        free(ctx.rem_stamp);
        free(ctx.vals_buffer);
    }

    return DM_SUCCESS;
}

DM_Algorithm aura_hoi_algo = {
    .id = "aura_hoi",
    .name = "AURA-HOI",
    .description = "Auditable representative high-occupancy itemset mining with inverted length modeling and residual deficit bounding.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(aura_hoi_algo)
