#include "algorithms/mhoui.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MHOUI_WORD_BITS 64
#define MHOUI_EPS 1e-12

/* ── Compact Item Struct ─────────────────────────────────────────────────── */
typedef struct {
    uint32_t item;
    int support;
    double twu;
    int rank;
    uint64_t *bitset;
} MHOUIItem;

/* ── Transaction Storage (Zero-Copy Pointer Arrays) ───────────────────────── */
typedef struct {
    int tid;
    double total_utility;
    int length;
    int *ranks;
    double *utils;
} CompactTransaction;

typedef struct {
    CompactTransaction *transactions;
    int transaction_count;
    int item_count;
    int word_count;
    double total_database_utility;
    MHOUIItem *items;
    double *invlen_table; /* Precomputed 1.0 / length */
} MHOUIDB;

/* ── Dynamic Pareto-Trie for O(1) Superset Dominance Checking ────────────── */
typedef struct {
    uint64_t *words;
    size_t num_items;
    double utility;
    double avg_occupancy;
} ParetoPattern;

typedef struct {
    ParetoPattern **patterns;
    size_t count;
    size_t capacity;
} ParetoPatternCache;

/* ── Context ──────────────────────────────────────────────────────────────── */
typedef struct {
    const MHOUIDB *db;
    int minsup;
    double minocc;
    double minutil;
    int max_patterns;
    double max_seconds;
    int direct;
    int strong_direct;
    struct timespec start;
    
    MHOUPatternList *houi;
    ParetoPatternCache pcache;
    size_t item_words;
    uint64_t *temp_prefix_bs;
    uint64_t *temp_branch_bs;
    
    MHOUIStats *stats;
} MHOUIContext;

static inline double elapsed_seconds(const struct timespec *start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - start->tv_sec) + (double)(now.tv_nsec - start->tv_nsec) / 1000000000.0;
}

static inline int word_count_for_bits(int nbits) {
    return (nbits + MHOUI_WORD_BITS - 1) / MHOUI_WORD_BITS;
}

static inline void bitset_set(uint64_t *bits, int pos) {
    bits[pos / MHOUI_WORD_BITS] |= UINT64_C(1) << (pos % MHOUI_WORD_BITS);
}

static inline uint64_t *bitset_alloc(int words) {
    return (uint64_t *)calloc((size_t)words, sizeof(uint64_t));
}

static inline uint64_t *bitset_clone(const uint64_t *src, int words) {
    uint64_t *out = (uint64_t *)malloc((size_t)words * sizeof(uint64_t));
    if (out) memcpy(out, src, (size_t)words * sizeof(uint64_t));
    return out;
}

static inline void bitset_and(uint64_t *out, const uint64_t *a, const uint64_t *b, int words) {
    for (int i = 0; i < words; i++) out[i] = a[i] & b[i];
}

static inline int bitset_popcount(const uint64_t *bits, int words) {
    int total = 0;
    for (int i = 0; i < words; i++) total += __builtin_popcountll(bits[i]);
    return total;
}

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int cmp_item_order(const void *a, const void *b) {
    const MHOUIItem *x = (const MHOUIItem *)a;
    const MHOUIItem *y = (const MHOUIItem *)b;
    if (x->support != y->support) return x->support - y->support;
    if (x->twu < y->twu) return -1;
    if (x->twu > y->twu) return 1;
    return (x->item > y->item) - (x->item < y->item);
}

static int cmp_pattern_filter(const void *a, const void *b) {
    const MHOUPattern *x = (const MHOUPattern *)a;
    const MHOUPattern *y = (const MHOUPattern *)b;
    if (x->length != y->length) return y->length - x->length;
    if (x->utility < y->utility) return 1;
    if (x->utility > y->utility) return -1;
    if (x->avg_occupancy < y->avg_occupancy) return 1;
    if (x->avg_occupancy > y->avg_occupancy) return -1;
    return 0;
}

/* ── Pattern List Utilities ───────────────────────────────────────────────── */
void mhoui_pattern_list_init(MHOUPatternList *list) {
    list->count = 0;
    list->capacity = 256;
    list->patterns = (MHOUPattern *)malloc((size_t)list->capacity * sizeof(MHOUPattern));
}

static void pattern_free(MHOUPattern *p) {
    free(p->items);
    free(p->tidset);
}

void mhoui_pattern_list_free(MHOUPatternList *list) {
    if (!list || !list->patterns) return;
    for (int i = 0; i < list->count; i++) pattern_free(&list->patterns[i]);
    free(list->patterns);
    list->patterns = NULL;
    list->count = 0;
    list->capacity = 0;
}

static int pattern_add(MHOUPatternList *list, const uint32_t *items, int length, const uint64_t *tidset, int word_count, int support, double utility, double avg_occupancy) {
    if (list->count >= list->capacity) {
        int new_cap = list->capacity * 2;
        MHOUPattern *new_patterns = (MHOUPattern *)realloc(list->patterns, (size_t)new_cap * sizeof(MHOUPattern));
        if (!new_patterns) return -1;
        list->patterns = new_patterns;
        list->capacity = new_cap;
    }
    MHOUPattern *p = &list->patterns[list->count++];
    p->items = (uint32_t *)malloc((size_t)length * sizeof(uint32_t));
    p->tidset = bitset_clone(tidset, word_count);
    if (!p->items || !p->tidset) return -1;
    memcpy(p->items, items, (size_t)length * sizeof(uint32_t));
    qsort(p->items, (size_t)length, sizeof(uint32_t), cmp_uint32);
    p->length = length;
    p->support = support;
    p->utility = utility;
    p->avg_occupancy = avg_occupancy;
    return 0;
}

/* ── Inverted Pivot-Item Bitset Ledger ────────────────────────────────────── */
static void pcache_init(ParetoPatternCache *pc) {
    pc->count = 0;
    pc->capacity = 1024;
    pc->patterns = (ParetoPattern **)malloc(pc->capacity * sizeof(ParetoPattern *));
}

static void pcache_free(ParetoPatternCache *pc) {
    if (!pc || !pc->patterns) return;
    for (size_t i = 0; i < pc->count; i++) {
        free(pc->patterns[i]->words);
        free(pc->patterns[i]);
    }
    free(pc->patterns);
    pc->patterns = NULL;
    pc->count = 0;
    pc->capacity = 0;
}

static void pcache_add(ParetoPatternCache *pc, const int *prefix_ranks, int prefix_len, double utility, double avg_occupancy, size_t item_words) {
    if (pc->count >= pc->capacity) {
        pc->capacity *= 2;
        pc->patterns = (ParetoPattern **)realloc(pc->patterns, pc->capacity * sizeof(ParetoPattern *));
    }
    ParetoPattern *pp = (ParetoPattern *)malloc(sizeof(ParetoPattern));
    pp->words = (uint64_t *)calloc(item_words, sizeof(uint64_t));
    pp->num_items = (size_t)prefix_len;
    pp->utility = utility;
    pp->avg_occupancy = avg_occupancy;
    for (int i = 0; i < prefix_len; i++) {
        int r = prefix_ranks[i];
        pp->words[r / 64] |= (1ULL << (r % 64));
    }
    pc->patterns[pc->count++] = pp;
}

static inline int itemset_is_subset(const uint64_t *sub, const uint64_t *sup, size_t words) {
    for (size_t i = 0; i < words; i++) {
        if ((sub[i] & ~sup[i]) != 0) return 0;
    }
    return 1;
}

static int pcache_dominates_branch(const ParetoPatternCache *pc, const uint64_t *branch_bs, int branch_len, double uub, double oub2, int strong_direct, size_t words) {
    /* Fast filter: search from most recent (larger/stronger) patterns backwards */
    for (int i = (int)pc->count - 1; i >= 0; i--) {
        const ParetoPattern *pp = pc->patterns[i];
        if (pp->num_items <= (size_t)branch_len) continue;
        if (strong_direct) {
            if (pp->avg_occupancy + MHOUI_EPS < oub2 || pp->utility + MHOUI_EPS < uub) continue;
        } else {
            int better_occ = pp->avg_occupancy > oub2 + MHOUI_EPS && pp->utility + MHOUI_EPS >= uub;
            int better_util = pp->avg_occupancy + MHOUI_EPS >= oub2 && pp->utility > uub + MHOUI_EPS;
            if (!better_occ && !better_util) continue;
        }
        if (itemset_is_subset(branch_bs, pp->words, words)) return 1;
    }
    return 0;
}

/* ── Database Preprocessing (Clean, Contiguous Memory Layout) ─────────────── */
static void free_db(MHOUIDB *db) {
    if (!db) return;
    for (int i = 0; i < db->transaction_count; i++) {
        free(db->transactions[i].ranks);
        free(db->transactions[i].utils);
    }
    for (int i = 0; i < db->item_count; i++) {
        free(db->items[i].bitset);
    }
    free(db->transactions);
    free(db->items);
    free(db->invlen_table);
    memset(db, 0, sizeof(*db));
}

static int build_db(DM_Dataset *ds, int minsup, double minutil, MHOUIDB *out) {
    if (!ds || ds->type != DM_TYPE_UTILITY || ds->count == 0) return -1;
    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    int max_id = (int)ds->max_id;
    int *support = (int *)calloc((size_t)max_id + 1, sizeof(int));
    double *twu = (double *)calloc((size_t)max_id + 1, sizeof(double));
    if (!support || !twu) {
        free(support);
        free(twu);
        return -1;
    }

    out->transaction_count = (int)ds->count;
    out->word_count = word_count_for_bits(out->transaction_count);
    out->transactions = (CompactTransaction *)calloc(ds->count, sizeof(CompactTransaction));
    out->invlen_table = (double *)calloc(ds->count, sizeof(double));

    for (size_t tid = 0; tid < ds->count; tid++) {
        out->total_database_utility += src[tid].total_utility;
        out->invlen_table[tid] = src[tid].count > 0 ? (1.0 / (double)src[tid].count) : 0.0;
        for (size_t j = 0; j < src[tid].count; j++) {
            uint32_t id = src[tid].items[j].id;
            support[id]++;
            twu[id] += src[tid].total_utility;
        }
    }

    int kept = 0;
    for (int id = 0; id <= max_id; id++) {
        if (support[id] >= minsup && twu[id] + MHOUI_EPS >= minutil) kept++;
    }
    out->items = (MHOUIItem *)calloc((size_t)kept, sizeof(MHOUIItem));
    out->item_count = kept;
    int pos = 0;
    for (int id = 0; id <= max_id; id++) {
        if (support[id] >= minsup && twu[id] + MHOUI_EPS >= minutil) {
            out->items[pos].item = (uint32_t)id;
            out->items[pos].support = support[id];
            out->items[pos].twu = twu[id];
            out->items[pos].bitset = bitset_alloc(out->word_count);
            pos++;
        }
    }
    qsort(out->items, (size_t)out->item_count, sizeof(MHOUIItem), cmp_item_order);

    int *rank_of_id = (int *)malloc(((size_t)max_id + 1) * sizeof(int));
    for (int id = 0; id <= max_id; id++) rank_of_id[id] = -1;
    for (int i = 0; i < out->item_count; i++) {
        out->items[i].rank = i;
        rank_of_id[out->items[i].item] = i;
    }

    for (size_t tid = 0; tid < ds->count; tid++) {
        CompactTransaction *tr = &out->transactions[tid];
        tr->tid = (int)tid;
        tr->total_utility = src[tid].total_utility;
        tr->ranks = (int *)malloc(src[tid].count * sizeof(int));
        tr->utils = (double *)malloc(src[tid].count * sizeof(double));
        for (size_t j = 0; j < src[tid].count; j++) {
            int rank = rank_of_id[src[tid].items[j].id];
            if (rank < 0) continue;
            int w = tr->length++;
            tr->ranks[w] = rank;
            tr->utils[w] = src[tid].items[j].utility;
            bitset_set(out->items[rank].bitset, (int)tid);
        }
        /* Insertion sort ranks */
        for (int a = 1; a < tr->length; a++) {
            int rank = tr->ranks[a];
            double util = tr->utils[a];
            int b = a - 1;
            while (b >= 0 && tr->ranks[b] > rank) {
                tr->ranks[b + 1] = tr->ranks[b];
                tr->utils[b + 1] = tr->utils[b];
                b--;
            }
            tr->ranks[b + 1] = rank;
            tr->utils[b + 1] = util;
        }
    }

    free(rank_of_id);
    free(support);
    free(twu);
    return 0;
}

/* ── In-Memory Projected Transaction Processing ───────────────────────────── */
static inline int find_rank(const CompactTransaction *tr, int rank, double *util) {
    for (int i = 0; i < tr->length; i++) {
        if (tr->ranks[i] == rank) {
            *util = tr->utils[i];
            return 1;
        }
        if (tr->ranks[i] > rank) return 0;
    }
    return 0;
}

/* ── Recursive Search Engine with Closure Leaps & Pareto Pruning ─────────── */
static void dfs_fast_mine(MHOUIContext *ctx, uint32_t *prefix, int *prefix_ranks, int prefix_len, 
                          const uint64_t *tidset, int support, 
                          double cur_utility, int suffix_start) {
    if (ctx->stats->status_limited) return;
    if (ctx->max_seconds > 0.0 && elapsed_seconds(&ctx->start) > ctx->max_seconds) {
        ctx->stats->status_limited = 1;
        return;
    }

    const MHOUIDB *db = ctx->db;
    int word_count = db->word_count;

    for (int idx = suffix_start; idx < db->item_count; idx++) {
        ctx->stats->generated_candidates++;

        /* 1. Fast Bitset Support Check */
        uint64_t *cand_tidset = bitset_alloc(word_count);
        bitset_and(cand_tidset, tidset, db->items[idx].bitset, word_count);
        int cand_sup = bitset_popcount(cand_tidset, word_count);

        if (cand_sup < ctx->minsup) {
            ctx->stats->pruned_support++;
            free(cand_tidset);
            continue;
        }

        ctx->stats->visited_nodes++;

        /* 2. Fast Single-Pass Utility and Occupancy Accumulation */
        double new_util = cur_utility;
        double inv_sum = 0.0;
        double rem_util_sum = 0.0;
        double ub_envelope_sum = 0.0;

        for (int w = 0; w < word_count; w++) {
            uint64_t word = cand_tidset[w];
            while (word) {
                int bit = __builtin_ctzll(word);
                int tid = w * MHOUI_WORD_BITS + bit;
                const CompactTransaction *tr = &db->transactions[tid];

                double u = 0.0;
                find_rank(tr, idx, &u);
                new_util += u;

                double invlen = db->invlen_table[tid];
                inv_sum += invlen;

                /* Remaining utility & extension items in transaction */
                double r_util = 0.0;
                int r_cnt = 0;
                for (int j = 0; j < tr->length; j++) {
                    if (tr->ranks[j] > idx) {
                        r_util += tr->utils[j];
                        r_cnt++;
                    }
                }
                rem_util_sum += r_util;

                double max_possible_len = (double)(prefix_len + 1 + r_cnt);
                double occ_cap = max_possible_len * invlen;
                if (occ_cap > 1.0) occ_cap = 1.0;
                ub_envelope_sum += occ_cap;

                word &= word - 1;
            }
        }

        int new_len = prefix_len + 1;
        prefix[prefix_len] = db->items[idx].item;
        prefix_ranks[prefix_len] = idx;

        double aocc = (double)new_len * inv_sum / (double)cand_sup;
        double uub = new_util + rem_util_sum;
        double oub2 = ub_envelope_sum / (double)ctx->minsup;

        /* 3. Pruning Checks */
        if (uub + MHOUI_EPS < ctx->minutil) {
            ctx->stats->pruned_uub++;
            free(cand_tidset);
            continue;
        }

        if (oub2 > 0.0 && oub2 + MHOUI_EPS < ctx->minocc) {
            ctx->stats->pruned_oub2++;
            free(cand_tidset);
            continue;
        }

        /* 4. Output Collection */
        if (new_util + MHOUI_EPS >= ctx->minutil && aocc + MHOUI_EPS >= ctx->minocc) {
            pattern_add(ctx->houi, prefix, new_len, cand_tidset, word_count, cand_sup, new_util, aocc);
            ctx->stats->houi_count = ctx->houi->count;
            if (ctx->direct) {
                pcache_add(&ctx->pcache, prefix_ranks, new_len, new_util, aocc, ctx->item_words);
            }
            if (ctx->max_patterns > 0 && ctx->houi->count >= ctx->max_patterns) {
                ctx->stats->status_limited = 1;
            }
        }

        /* 5. In-Process Branch Dominance Check (Inverted Pivot Bitset Ledger) */
        if (!ctx->stats->status_limited && idx + 1 < db->item_count) {
            int branch_pruned = 0;
            int rem_candidates = db->item_count - (idx + 1);
            /* Dynamic 2-Level Check: Only invoke fine-grained bitset superset test when remaining candidate list is bounded (<= 16) */
            if (ctx->direct && ctx->pcache.count > 0 && rem_candidates <= 16) {
                /* Construct temporary bitset of the branch prefix ranks + remaining candidate ranks */
                memset(ctx->temp_branch_bs, 0, ctx->item_words * sizeof(uint64_t));
                for (int p = 0; p < new_len; p++) {
                    int r = prefix_ranks[p];
                    ctx->temp_branch_bs[r / 64] |= (1ULL << (r % 64));
                }
                for (int s = idx + 1; s < db->item_count; s++) {
                    ctx->temp_branch_bs[s / 64] |= (1ULL << (s % 64));
                }
                int branch_items_cnt = new_len + rem_candidates;
                if (pcache_dominates_branch(&ctx->pcache, ctx->temp_branch_bs, branch_items_cnt, uub, oub2, ctx->strong_direct, ctx->item_words)) {
                    ctx->stats->pruned_dominance++;
                    branch_pruned = 1;
                }
            }

            if (!branch_pruned) {
                dfs_fast_mine(ctx, prefix, prefix_ranks, new_len, cand_tidset, cand_sup, new_util, idx + 1);
            }
        }

        free(cand_tidset);
        if (ctx->stats->status_limited) return;
    }
}

/* ── Superset Dominance Filter ────────────────────────────────────────────── */
static inline int proper_subset(const MHOUPattern *x, const MHOUPattern *y) {
    if (x->length >= y->length) return 0;
    int i = 0, j = 0;
    while (i < x->length && j < y->length) {
        if (x->items[i] == y->items[j]) {
            i++; j++;
        } else if (x->items[i] > y->items[j]) {
            j++;
        } else {
            return 0;
        }
    }
    return i == x->length;
}

static void copy_pattern(MHOUPatternList *out, const MHOUPattern *p, int word_count) {
    pattern_add(out, p->items, p->length, p->tidset, word_count, p->support, p->utility, p->avg_occupancy);
}

static void filter_mhoui(const MHOUPatternList *houi, MHOUPatternList *out, int word_count, int strong, int *removed) {
    mhoui_pattern_list_init(out);
    *removed = 0;
    if (!houi || houi->count == 0) return;
    MHOUPattern *sorted = (MHOUPattern *)malloc((size_t)houi->count * sizeof(MHOUPattern));
    memcpy(sorted, houi->patterns, (size_t)houi->count * sizeof(MHOUPattern));
    qsort(sorted, (size_t)houi->count, sizeof(MHOUPattern), cmp_pattern_filter);
    for (int i = 0; i < houi->count; i++) {
        int dominated = 0;
        for (int j = 0; j < i; j++) {
            if (!proper_subset(&sorted[i], &sorted[j])) continue;
            int ge_occ = sorted[j].avg_occupancy + MHOUI_EPS >= sorted[i].avg_occupancy;
            int ge_util = sorted[j].utility + MHOUI_EPS >= sorted[i].utility;
            int strict = sorted[j].avg_occupancy > sorted[i].avg_occupancy + MHOUI_EPS || sorted[j].utility > sorted[i].utility + MHOUI_EPS;
            if (ge_occ && ge_util && (strong || strict)) {
                dominated = 1;
                break;
            }
        }
        if (dominated) (*removed)++;
        else copy_pattern(out, &sorted[i], word_count);
    }
    free(sorted);
}

/* ── Top-Level Mine Driver ────────────────────────────────────────────────── */
int mhoui_mine_dataset(DM_Dataset *ds, int minsup_count, double minocc, double minutil, const MHOUILimits *limits, MHOUIResult *out) {
    memset(out, 0, sizeof(*out));
    if (!ds || ds->type != DM_TYPE_UTILITY || minsup_count <= 0 || minocc <= 0.0 || minutil < 0.0) return -1;
    
    MHOUIDB db = {0};
    if (build_db(ds, minsup_count, minutil, &db) != 0) return -1;
    
    mhoui_pattern_list_init(&out->houi);
    uint32_t *prefix = (uint32_t *)malloc((size_t)(db.item_count > 0 ? db.item_count : 1) * sizeof(uint32_t));
    int *prefix_ranks = (int *)malloc((size_t)(db.item_count > 0 ? db.item_count : 1) * sizeof(int));
    
    uint64_t *all_tidset = bitset_alloc(db.word_count);
    for (int t = 0; t < db.transaction_count; t++) bitset_set(all_tidset, t);

    size_t item_words = word_count_for_bits(db.item_count > 0 ? db.item_count : 64);
    
    MHOUIContext ctx = {
        .db = &db,
        .minsup = minsup_count,
        .minocc = minocc,
        .minutil = minutil,
        .max_patterns = limits ? limits->max_patterns : 200000,
        .max_seconds = limits ? limits->max_seconds : 0.0,
        .direct = limits ? limits->direct : 0,
        .strong_direct = limits ? limits->strong_direct : 1,
        .houi = &out->houi,
        .item_words = item_words,
        .temp_prefix_bs = (uint64_t *)calloc(item_words, sizeof(uint64_t)),
        .temp_branch_bs = (uint64_t *)calloc(item_words, sizeof(uint64_t)),
        .stats = &out->stats
    };
    pcache_init(&ctx.pcache);

    clock_gettime(CLOCK_MONOTONIC, &ctx.start);
    dfs_fast_mine(&ctx, prefix, prefix_ranks, 0, all_tidset, db.transaction_count, 0.0, 0);

    filter_mhoui(&out->houi, &out->weak_mhoui, db.word_count, 0, &out->stats.weak_removed);
    filter_mhoui(&out->houi, &out->strong_mhoui, db.word_count, 1, &out->stats.strong_removed);
    out->stats.houi_count = out->houi.count;
    out->stats.weak_count = out->weak_mhoui.count;
    out->stats.strong_count = out->strong_mhoui.count;

    pcache_free(&ctx.pcache);
    free(ctx.temp_prefix_bs);
    free(ctx.temp_branch_bs);
    free(all_tidset);
    free(prefix);
    free(prefix_ranks);
    free_db(&db);
    return out->stats.status_limited ? 1 : 0;
}

void mhoui_result_free(MHOUIResult *result) {
    mhoui_pattern_list_free(&result->houi);
    mhoui_pattern_list_free(&result->weak_mhoui);
    mhoui_pattern_list_free(&result->strong_mhoui);
}

const MHOUPatternList *mhoui_select_patterns(const MHOUIResult *result, MHOUIAlgorithm algo) {
    if (!result) return NULL;
    switch (algo) {
        case MHOUI_ALGO_HOUI: return &result->houi;
        case MHOUI_ALGO_WEAK: return &result->weak_mhoui;
        case MHOUI_ALGO_DIRECT_WEAK: return &result->weak_mhoui;
        case MHOUI_ALGO_STRONG: return &result->strong_mhoui;
        case MHOUI_ALGO_DIRECT_STRONG: return &result->strong_mhoui;
    }
    return &result->strong_mhoui;
}

const char *mhoui_algorithm_name(MHOUIAlgorithm algo) {
    switch (algo) {
        case MHOUI_ALGO_HOUI: return "houi";
        case MHOUI_ALGO_WEAK: return "weak_mhoui";
        case MHOUI_ALGO_STRONG: return "strong_mhoui";
        case MHOUI_ALGO_DIRECT_WEAK: return "direct_weak_mhoui";
        case MHOUI_ALGO_DIRECT_STRONG: return "direct_strong_mhoui";
    }
    return "unknown";
}

int mhoui_parse_algorithm(const char *name, MHOUIAlgorithm *algo) {
    if (strcmp(name, "houi") == 0) *algo = MHOUI_ALGO_HOUI;
    else if (strcmp(name, "weak_mhoui") == 0 || strcmp(name, "weak") == 0) *algo = MHOUI_ALGO_WEAK;
    else if (strcmp(name, "strong_mhoui") == 0 || strcmp(name, "strong") == 0) *algo = MHOUI_ALGO_STRONG;
    else if (strcmp(name, "direct_weak_mhoui") == 0 || strcmp(name, "direct_weak") == 0) *algo = MHOUI_ALGO_DIRECT_WEAK;
    else if (strcmp(name, "direct_strong_mhoui") == 0 || strcmp(name, "direct_strong") == 0 || strcmp(name, "mhoui") == 0) *algo = MHOUI_ALGO_DIRECT_STRONG;
    else return -1;
    return 0;
}

int mhoui_write_patterns(const char *path, const MHOUPatternList *patterns, size_t transaction_count) {
    if (!path || !patterns) return -1;
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;
    fprintf(fp, "pattern,length,support,support_ratio,utility,avg_occupancy\n");
    for (int i = 0; i < patterns->count; i++) {
        const MHOUPattern *p = &patterns->patterns[i];
        for (int j = 0; j < p->length; j++) fprintf(fp, "%u%s", p->items[j], j + 1 < p->length ? " " : "");
        double sup_ratio = transaction_count ? (double)p->support / (double)transaction_count : 0.0;
        fprintf(fp, ",%d,%d,%.6f,%.4f,%.6f\n", p->length, p->support, sup_ratio, p->utility, p->avg_occupancy);
    }
    fclose(fp);
    return 0;
}

void mhoui_pattern_quality(const MHOUPatternList *patterns, double *avg_len, int *max_len, double *avg_support, double *avg_utility, double *avg_occ) {
    if (!patterns || patterns->count == 0) {
        if (avg_len) *avg_len = 0.0;
        if (max_len) *max_len = 0;
        if (avg_support) *avg_support = 0.0;
        if (avg_utility) *avg_utility = 0.0;
        if (avg_occ) *avg_occ = 0.0;
        return;
    }
    double total_len = 0.0, total_sup = 0.0, total_util = 0.0, total_occ = 0.0;
    int m_len = 0;
    for (int i = 0; i < patterns->count; i++) {
        const MHOUPattern *p = &patterns->patterns[i];
        total_len += (double)p->length;
        total_sup += (double)p->support;
        total_util += p->utility;
        total_occ += p->avg_occupancy;
        if (p->length > m_len) m_len = p->length;
    }
    if (avg_len) *avg_len = total_len / (double)patterns->count;
    if (max_len) *max_len = m_len;
    if (avg_support) *avg_support = total_sup / (double)patterns->count;
    if (avg_utility) *avg_utility = total_util / (double)patterns->count;
    if (avg_occ) *avg_occ = total_occ / (double)patterns->count;
}

static DM_Status run_mhoui(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_MHOUI_Params *p = (DM_MHOUI_Params *)params;
    double min_support = p ? p->min_support : 0.05;
    int minsup = min_support < 1.0 ? (int)ceil(min_support * (double)ds->count) : (int)min_support;
    double minocc = p ? p->min_occupancy : 0.4;
    double minutil = p ? p->min_utility : 1000.0;
    MHOUIAlgorithm algo = p ? (MHOUIAlgorithm)p->output_algorithm : MHOUI_ALGO_STRONG;
    MHOUILimits limits = {
        .max_patterns = p && p->max_patterns > 0 ? p->max_patterns : 200000,
        .max_seconds = p ? p->max_seconds : 0.0,
        .direct = p ? p->direct : 0,
        .strong_direct = p ? p->strong : 1
    };
    MHOUIResult result;
    int status = mhoui_mine_dataset(ds, minsup, minocc, minutil, &limits, &result);
    const MHOUPatternList *out = mhoui_select_patterns(&result, algo);
    size_t total_items = 0;
    for (int i = 0; i < out->count; i++) total_items += (size_t)out->patterns[i].length;
    dm_bench_record_results((size_t)out->count, total_items);
    mhoui_result_free(&result);
    return status == 0 ? DM_SUCCESS : DM_ERROR_GENERIC;
}

DM_Algorithm mhoui_algo = {
    .id = "mhoui",
    .name = "Maximal High-Occupancy Utility Itemset Miner",
    .description = "Exact mining of maximal high-occupancy utility itemsets with fast bitset-array projection and Pareto dominance",
    .supported_types = (1u << DM_TYPE_UTILITY),
    .run = run_mhoui
};

