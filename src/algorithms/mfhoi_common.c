#include "algorithms/mfhoi.h"
#include "core/dm_portability.h"
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WORD_BITS 64

/* ── Ternary Expansion States (Algorithm 2) ───────────────────────────────── */
typedef enum {
    STATE_DEADEND = 0,
    STATE_SUBSUMED_FHO = 1,
    STATE_ACTIVE_MFHO = 2
} TernaryState;

typedef struct {
    uint64_t *words;
    size_t num_items;
} ItemBitSet;

typedef struct {
    ItemBitSet **items;
    size_t count;
    size_t capacity;
} MFISet;

typedef struct {
    VerticalItem *items;
    int count;
    int word_count;
} VerticalDB;

typedef struct {
    const TransactionDB *db;
    const VerticalDB *vdb;
    int minsup;
    double minocc;
    int max_depth_kstar;
    int max_patterns;
    double max_seconds;
    struct timespec start;
    
    PatternList *strong_mfhoi;
    MFISet mfi_cache;
    size_t item_num_words;
    ItemBitSet *temp_hut;
    
    double *recip_table;
    MFHOIStats *stats;
} InProcessCtx;

static inline void itemset_set_bit(ItemBitSet *bs, int item) {
    bs->words[item / 64] |= (1ULL << (item % 64));
}

static inline void itemset_clear_bit(ItemBitSet *bs, int item) {
    bs->words[item / 64] &= ~(1ULL << (item % 64));
}

static inline int itemset_is_subset(const ItemBitSet *sub, const ItemBitSet *sup, size_t num_words) {
    for (size_t i = 0; i < num_words; i++) {
        if ((sub->words[i] & ~sup->words[i]) != 0) return 0;
    }
    return 1;
}

static int mfi_has_superset(const MFISet *mfi, const ItemBitSet *bs, size_t num_words) {
    for (size_t i = 0; i < mfi->count; i++) {
        if (bs->num_items <= mfi->items[i]->num_items &&
            itemset_is_subset(bs, mfi->items[i], num_words)) {
            return 1;
        }
    }
    return 0;
}

static void mfi_add(MFISet *mfi, const int *items, size_t count, size_t item_num_words) {
    if (mfi->count >= mfi->capacity) {
        mfi->capacity = mfi->capacity == 0 ? 1024 : mfi->capacity * 2;
        mfi->items = (ItemBitSet **)realloc(mfi->items, mfi->capacity * sizeof(ItemBitSet *));
    }
    ItemBitSet *bs = (ItemBitSet *)malloc(sizeof(ItemBitSet));
    bs->words = (uint64_t *)calloc(item_num_words, sizeof(uint64_t));
    bs->num_items = count;
    for (size_t i = 0; i < count; i++) {
        itemset_set_bit(bs, items[i]);
    }
    mfi->items[mfi->count++] = bs;
}

static void mfi_free(MFISet *mfi) {
    if (!mfi) return;
    for (size_t i = 0; i < mfi->count; i++) {
        free(mfi->items[i]->words);
        free(mfi->items[i]);
    }
    free(mfi->items);
    mfi->items = NULL;
    mfi->count = 0;
    mfi->capacity = 0;
}

static double elapsed_seconds(const struct timespec *start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - start->tv_sec) + (double)(now.tv_nsec - start->tv_nsec) / 1000000000.0;
}

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a;
    int y = *(const int *)b;
    return (x > y) - (x < y);
}

static int cmp_vertical_support_asc(const void *a, const void *b) {
    const VerticalItem *x = (const VerticalItem *)a;
    const VerticalItem *y = (const VerticalItem *)b;
    if (x->support != y->support) return x->support - y->support;
    return x->item - y->item;
}

static int bit_words(int nbits) {
    return (nbits + WORD_BITS - 1) / WORD_BITS;
}

static inline void bitset_set(uint64_t *bits, int pos) {
    bits[pos / WORD_BITS] |= (1ULL << (pos % WORD_BITS));
}

static inline int bitset_popcount(const uint64_t *bits, int words) {
    int total = 0;
    for (int i = 0; i < words; i++) total += (int)__builtin_popcountll(bits[i]);
    return total;
}

static inline void bitset_and(uint64_t *out, const uint64_t *a, const uint64_t *b, int words) {
    for (int i = 0; i < words; i++) out[i] = a[i] & b[i];
}

static uint64_t *bitset_clone(const uint64_t *src, int words) {
    uint64_t *copy = (uint64_t *)malloc((size_t)words * sizeof(uint64_t));
    if (copy) memcpy(copy, src, (size_t)words * sizeof(uint64_t));
    return copy;
}

void mfhoi_pattern_list_init(PatternList *list) {
    list->count = 0;
    list->capacity = 256;
    list->patterns = (Pattern *)malloc((size_t)list->capacity * sizeof(Pattern));
}

static void pattern_free(Pattern *p) {
    free(p->items);
    free(p->tidset);
}

void mfhoi_pattern_list_free(PatternList *list) {
    if (!list || !list->patterns) return;
    for (int i = 0; i < list->count; i++) pattern_free(&list->patterns[i]);
    free(list->patterns);
    list->patterns = NULL;
    list->count = 0;
    list->capacity = 0;
}

static int pattern_list_add(PatternList *list, const int *items, int length, const uint64_t *tidset, int word_count, int support, double avg_occ) {
    if (list->count >= list->capacity) {
        int new_cap = list->capacity * 2;
        Pattern *new_patterns = (Pattern *)realloc(list->patterns, (size_t)new_cap * sizeof(Pattern));
        if (!new_patterns) return -1;
        list->patterns = new_patterns;
        list->capacity = new_cap;
    }

    Pattern *p = &list->patterns[list->count++];
    p->items = (int *)malloc((size_t)length * sizeof(int));
    p->tidset = bitset_clone(tidset, word_count);
    if (!p->items || !p->tidset) return -1;
    memcpy(p->items, items, (size_t)length * sizeof(int));
    qsort(p->items, (size_t)length, sizeof(int), cmp_int);
    p->length = length;
    p->support = support;
    p->avg_occ = avg_occ;
    return 0;
}

static double compute_exact_aocc(const InProcessCtx *ctx, const uint64_t *tidset, int support, int length) {
    if (support <= 0) return 0.0;
    double sum = 0.0;
    int words = ctx->vdb->word_count;
    for (int w = 0; w < words; w++) {
        uint64_t x = tidset[w];
        while (x) {
            unsigned bit = (unsigned)__builtin_ctzll(x);
            int tid = (w * WORD_BITS) + (int)bit;
            if (tid < ctx->db->transaction_count) {
                sum += ctx->recip_table[tid];
            }
            x &= x - 1;
        }
    }
    return ((double)length * sum) / (double)support;
}

/* ── Tier 1: Linear Transactional Envelope (UB_env) ────────────────────────── */
static double compute_env_upper_bound(const InProcessCtx *ctx, const uint64_t *tidset, int support,
                                      int prefix_len, int extension_len) {
    if (extension_len <= 0 || support < ctx->minsup) return 0.0;
    int words = ctx->vdb->word_count;
    double sum = 0.0;
    for (int w = 0; w < words; w++) {
        uint64_t x = tidset[w];
        while (x) {
            unsigned bit = (unsigned)__builtin_ctzll(x);
            int tid = (w * WORD_BITS) + (int)bit;
            if (tid < ctx->db->transaction_count) {
                int tlen = ctx->db->transaction_lengths[tid];
                int rem_fit = tlen - prefix_len;
                if (rem_fit > extension_len) rem_fit = extension_len;
                if (rem_fit < 0) rem_fit = 0;
                double num = (double)(prefix_len + rem_fit);
                double frac = num / (double)tlen;
                if (frac > 1.0) frac = 1.0;
                sum += frac;
            }
            x &= x - 1;
        }
    }
    return sum / (double)support;
}

/* ── Occupancy Suffocation Point K* (Section 4.3, Theorem 4) ──────────────── */
static int compute_occupancy_suffocation_point(const TransactionDB *db, int minsup, double minocc) {
    if (db->transaction_count <= 0 || minsup <= 0) return 0;
    int *sorted_lens = (int *)malloc((size_t)db->transaction_count * sizeof(int));
    for (int i = 0; i < db->transaction_count; i++) sorted_lens[i] = db->transaction_lengths[i];
    qsort(sorted_lens, (size_t)db->transaction_count, sizeof(int), cmp_int);

    int max_trans_len = sorted_lens[db->transaction_count - 1];
    int k_star = 0;
    for (int k = 1; k <= max_trans_len; k++) {
        /* Only transactions of length >= k can support a k-itemset */
        int valid_count = 0;
        double sum_inv = 0.0;
        for (int i = 0; i < db->transaction_count; i++) {
            if (sorted_lens[i] >= k) {
                sum_inv += 1.0 / (double)sorted_lens[i];
                valid_count++;
                if (valid_count == minsup) break;
            }
        }
        if (valid_count < minsup) break;

        double omega = ((double)k / (double)minsup) * sum_inv;
        if (omega + 1e-12 >= minocc) {
            k_star = k;
        }
    }
    free(sorted_lens);
    return k_star > 0 ? k_star : max_trans_len;
}

/* ── ClosureLeap-DFS with Ternary State Tracking (Algorithm 2) ─────────────── */
static TernaryState closure_leap_dfs(InProcessCtx *ctx,
                                     int *P, int P_len,
                                     const uint64_t *B_P,
                                     const int *E, int E_len) {
    if (ctx->stats->status_limited) return STATE_DEADEND;
    if (ctx->max_seconds > 0.0 && elapsed_seconds(&ctx->start) > ctx->max_seconds) {
        ctx->stats->status_limited = 1;
        return STATE_DEADEND;
    }

    int words = ctx->vdb->word_count;
    int sup_P = bitset_popcount(B_P, words);
    if (sup_P < ctx->minsup) return STATE_DEADEND;

    /* Step 1: Closure Leap (Galois Closure Jump) */
    int *cand_E = (int *)malloc((size_t)(E_len > 0 ? E_len : 1) * sizeof(int));
    int cand_E_len = 0;
    int *closed_items = (int *)malloc((size_t)(E_len > 0 ? E_len : 1) * sizeof(int));
    int closed_count = 0;

    uint64_t *temp_and = (uint64_t *)malloc((size_t)words * sizeof(uint64_t));

    for (int i = 0; i < E_len; i++) {
        int e_idx = E[i];
        bitset_and(temp_and, B_P, ctx->vdb->items[e_idx].bitset, words);
        int pop = bitset_popcount(temp_and, words);
        if (pop == sup_P) {
            closed_items[closed_count++] = ctx->vdb->items[e_idx].item;
        } else if (pop >= ctx->minsup) {
            cand_E[cand_E_len++] = e_idx;
        }
    }
    free(temp_and);

    /* Form closed prefix P' = P U E_closed */
    int new_P_len = P_len + closed_count;
    int *new_P = (int *)malloc((size_t)(new_P_len > 0 ? new_P_len : 1) * sizeof(int));
    if (P_len > 0) memcpy(new_P, P, (size_t)P_len * sizeof(int));
    for (int i = 0; i < closed_count; i++) new_P[P_len + i] = closed_items[i];
    free(closed_items);

    /* Step 2: Depth Suffocation and HUT Lookahead */
    if (new_P_len > ctx->max_depth_kstar) {
        free(cand_E);
        free(new_P);
        return STATE_DEADEND;
    }

    /* Form HUT = new_P U cand_E */
    ItemBitSet hut_bs;
    hut_bs.words = (uint64_t *)calloc(ctx->item_num_words, sizeof(uint64_t));
    hut_bs.num_items = new_P_len + cand_E_len;
    for (int i = 0; i < new_P_len; i++) itemset_set_bit(&hut_bs, new_P[i]);
    for (int i = 0; i < cand_E_len; i++) itemset_set_bit(&hut_bs, ctx->vdb->items[cand_E[i]].item);

    int hut_subsumed = mfi_has_superset(&ctx->mfi_cache, &hut_bs, ctx->item_num_words);
    if (hut_subsumed) {
        ctx->stats->dominance_removed_count++;
        free(hut_bs.words);
        free(cand_E);
        free(new_P);
        return STATE_DEADEND;
    }

    /* Fast check for Full HUT Absorption (Theorem 5) */
    if (cand_E_len > 0) {
        uint64_t *hut_tidset = bitset_clone(B_P, words);
        for (int i = 0; i < cand_E_len; i++) {
            bitset_and(hut_tidset, hut_tidset, ctx->vdb->items[cand_E[i]].bitset, words);
        }
        int hut_sup = bitset_popcount(hut_tidset, words);
        if (hut_sup >= ctx->minsup) {
            double hut_aocc = compute_exact_aocc(ctx, hut_tidset, hut_sup, (int)hut_bs.num_items);
            if (hut_aocc + 1e-12 >= ctx->minocc) {
                int *hut_items = (int *)malloc(hut_bs.num_items * sizeof(int));
                memcpy(hut_items, new_P, (size_t)new_P_len * sizeof(int));
                for (int i = 0; i < cand_E_len; i++) hut_items[new_P_len + i] = ctx->vdb->items[cand_E[i]].item;

                pattern_list_add(ctx->strong_mfhoi, hut_items, (int)hut_bs.num_items, hut_tidset, words, hut_sup, hut_aocc);
                mfi_add(&ctx->mfi_cache, hut_items, hut_bs.num_items, ctx->item_num_words);
                ctx->stats->num_strong_mfhoi = ctx->strong_mfhoi->count;
                ctx->stats->num_fhoi++;

                free(hut_items);
                free(hut_tidset);
                free(hut_bs.words);
                free(cand_E);
                free(new_P);
                return STATE_ACTIVE_MFHO;
            }
        }
        free(hut_tidset);
    }
    free(hut_bs.words);

    /* Step 3: Fast Residual Envelope Pruning (Theorem 6) */
    if (cand_E_len > 0) {
        double ub_env = compute_env_upper_bound(ctx, B_P, sup_P, new_P_len, cand_E_len);
        if (ub_env + 1e-12 < ctx->minocc) {
            ctx->stats->pruned_ub2_count++;
            free(cand_E);
            free(new_P);
            return STATE_DEADEND;
        }
    }

    /* Step 4: Recursive Branching with Ternary State Tracking */
    int has_active_child = 0;
    int has_fho_child = 0;

    for (int i = 0; i < cand_E_len; i++) {
        int e_idx = cand_E[i];
        int child_item = ctx->vdb->items[e_idx].item;

        uint64_t *B_X = (uint64_t *)malloc((size_t)words * sizeof(uint64_t));
        bitset_and(B_X, B_P, ctx->vdb->items[e_idx].bitset, words);
        ctx->stats->num_generated_candidates++;

        int *next_P = (int *)malloc((size_t)(new_P_len + 1) * sizeof(int));
        memcpy(next_P, new_P, (size_t)new_P_len * sizeof(int));
        next_P[new_P_len] = child_item;

        int next_E_len = cand_E_len - (i + 1);
        const int *next_E = (next_E_len > 0) ? &cand_E[i + 1] : NULL;

        TernaryState child_state = closure_leap_dfs(ctx, next_P, new_P_len + 1, B_X, next_E, next_E_len);
        free(B_X);
        free(next_P);

        if (child_state == STATE_ACTIVE_MFHO) {
            has_active_child = 1;
            has_fho_child = 1;
        } else if (child_state == STATE_SUBSUMED_FHO) {
            has_fho_child = 1;
        }

        if (ctx->stats->status_limited) break;
    }
    free(cand_E);

    /* Step 5: Confirmed Maximality Check upon Backtracking */
    double a_P = 0.0;
    int is_fho = 0;
    if (new_P_len > 0) {
        a_P = compute_exact_aocc(ctx, B_P, sup_P, new_P_len);
        is_fho = (a_P + 1e-12 >= ctx->minocc);
        if (is_fho) ctx->stats->num_fhoi++;
    }

    if (is_fho && !has_fho_child) {
        ItemBitSet cand_bs;
        cand_bs.words = (uint64_t *)calloc(ctx->item_num_words, sizeof(uint64_t));
        cand_bs.num_items = new_P_len;
        for (int k = 0; k < new_P_len; k++) itemset_set_bit(&cand_bs, new_P[k]);

        if (!mfi_has_superset(&ctx->mfi_cache, &cand_bs, ctx->item_num_words)) {
            pattern_list_add(ctx->strong_mfhoi, new_P, new_P_len, B_P, words, sup_P, a_P);
            mfi_add(&ctx->mfi_cache, new_P, new_P_len, ctx->item_num_words);
            ctx->stats->num_strong_mfhoi = ctx->strong_mfhoi->count;
            free(cand_bs.words);
            free(new_P);
            return STATE_ACTIVE_MFHO;
        }
        free(cand_bs.words);
    }

    free(new_P);

    if (has_active_child) return STATE_ACTIVE_MFHO;
    if (has_fho_child || is_fho) return STATE_SUBSUMED_FHO;
    return STATE_DEADEND;
}

int mfhoi_mine_all(const TransactionDB *db, int minsup_count, double minocc, const MFHOILimits *limits, MFHOIResult *out) {
    memset(out, 0, sizeof(*out));
    if (!db || db->transaction_count <= 0 || minsup_count <= 0) return -1;

    VerticalDB vdb = {0};
    vdb.word_count = bit_words(db->transaction_count);

    int *counts = (int *)calloc((size_t)db->item_count, sizeof(int));
    for (int tid = 0; tid < db->transaction_count; tid++) {
        for (int i = 0; i < db->transactions[tid].length; i++) counts[db->transactions[tid].items[i]]++;
    }

    int n = 0;
    for (int item = 0; item < db->item_count; item++) {
        if (counts[item] >= minsup_count) n++;
    }

    vdb.items = (VerticalItem *)calloc((size_t)n, sizeof(VerticalItem));
    vdb.count = n;
    int *item_to_pos = (int *)malloc((size_t)db->item_count * sizeof(int));
    for (int i = 0; i < db->item_count; i++) item_to_pos[i] = -1;

    int pos = 0;
    for (int item = 0; item < db->item_count; item++) {
        if (counts[item] >= minsup_count) {
            vdb.items[pos].item = item;
            vdb.items[pos].support = counts[item];
            vdb.items[pos].bitset = (uint64_t *)calloc((size_t)vdb.word_count, sizeof(uint64_t));
            item_to_pos[item] = pos;
            pos++;
        }
    }

    for (int tid = 0; tid < db->transaction_count; tid++) {
        for (int i = 0; i < db->transactions[tid].length; i++) {
            int p = item_to_pos[db->transactions[tid].items[i]];
            if (p >= 0) bitset_set(vdb.items[p].bitset, tid);
        }
    }

    qsort(vdb.items, (size_t)vdb.count, sizeof(VerticalItem), cmp_vertical_support_asc);
    free(item_to_pos);
    free(counts);

    mfhoi_pattern_list_init(&out->frequent);
    mfhoi_pattern_list_init(&out->fhoi);
    mfhoi_pattern_list_init(&out->strong_mfhoi);
    mfhoi_pattern_list_init(&out->weak_mfhoi);
    mfhoi_pattern_list_init(&out->mfi);

    InProcessCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.db = db;
    ctx.vdb = &vdb;
    ctx.minsup = minsup_count;
    ctx.minocc = minocc;
    ctx.max_depth_kstar = compute_occupancy_suffocation_point(db, minsup_count, minocc);
    ctx.max_patterns = limits ? limits->max_patterns : 0;
    ctx.max_seconds = limits ? limits->max_seconds : 0.0;
    ctx.strong_mfhoi = &out->strong_mfhoi;
    ctx.stats = &out->stats;

    ctx.item_num_words = (db->item_count + 63) / 64;
    ctx.temp_hut = (ItemBitSet *)malloc(sizeof(ItemBitSet));
    ctx.temp_hut->words = (uint64_t *)calloc(ctx.item_num_words, sizeof(uint64_t));

    ctx.recip_table = (double *)malloc((size_t)db->transaction_count * sizeof(double));
    for (int tid = 0; tid < db->transaction_count; tid++) {
        ctx.recip_table[tid] = 1.0 / (double)db->transaction_lengths[tid];
    }

    clock_gettime(CLOCK_MONOTONIC, &ctx.start);

    /* Construct Initial Root Node (Empty prefix, all-ones bitset, all frequent items) */
    uint64_t *root_tidset = (uint64_t *)malloc((size_t)vdb.word_count * sizeof(uint64_t));
    for (int w = 0; w < vdb.word_count; w++) root_tidset[w] = ~0ULL;
    int tail_bits = db->transaction_count % WORD_BITS;
    if (tail_bits != 0) {
        root_tidset[vdb.word_count - 1] = (1ULL << tail_bits) - 1;
    }

    int *initial_E = (int *)malloc((size_t)(vdb.count > 0 ? vdb.count : 1) * sizeof(int));
    for (int i = 0; i < vdb.count; i++) initial_E[i] = i;

    /* Execute ClosureLeap-DFS with Ternary State Tracking */
    closure_leap_dfs(&ctx, NULL, 0, root_tidset, initial_E, vdb.count);

    free(initial_E);
    free(root_tidset);
    free(ctx.temp_hut->words);
    free(ctx.temp_hut);
    mfi_free(&ctx.mfi_cache);
    free(ctx.recip_table);

    for (int i = 0; i < vdb.count; i++) free(vdb.items[i].bitset);
    free(vdb.items);

    return out->stats.status_limited ? 1 : 0;
}

static void strip_comment(char *line) {
    char *hash = strchr(line, '#');
    if (hash) *hash = '\0';
}

TransactionDB *mfhoi_load_transaction_db(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return NULL;

    TransactionDB *db = (TransactionDB *)calloc(1, sizeof(TransactionDB));
    int cap = 1024;
    db->transactions = (Transaction *)calloc((size_t)cap, sizeof(Transaction));
    db->transaction_lengths = (int *)calloc((size_t)cap, sizeof(int));
    db->max_item_id = -1;

    char line[65536];
    while (fgets(line, sizeof(line), fp)) {
        strip_comment(line);
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0') continue;

        if (db->transaction_count >= cap) {
            cap *= 2;
            db->transactions = (Transaction *)realloc(db->transactions, (size_t)cap * sizeof(Transaction));
            db->transaction_lengths = (int *)realloc(db->transaction_lengths, (size_t)cap * sizeof(int));
        }

        int item_cap = 16;
        int len = 0;
        int *items = (int *)malloc((size_t)item_cap * sizeof(int));
        char *tok = strtok(p, " \t\r\n");
        while (tok) {
            if (len >= item_cap) {
                item_cap *= 2;
                items = (int *)realloc(items, (size_t)item_cap * sizeof(int));
            }
            int item = atoi(tok);
            items[len++] = item;
            if (item > db->max_item_id) db->max_item_id = item;
            tok = strtok(NULL, " \t\r\n");
        }
        if (len > 0) {
            qsort(items, (size_t)len, sizeof(int), cmp_int);
            int unique_len = 0;
            for (int i = 0; i < len; i++) {
                if (unique_len == 0 || items[i] != items[unique_len - 1]) {
                    items[unique_len++] = items[i];
                }
            }
            db->transactions[db->transaction_count].items = items;
            db->transactions[db->transaction_count].length = unique_len;
            db->transaction_lengths[db->transaction_count] = unique_len;
            db->transaction_count++;
        } else {
            free(items);
        }
    }

    fclose(fp);
    db->item_count = db->max_item_id + 1;
    return db;
}

void mfhoi_free_transaction_db(TransactionDB *db) {
    if (!db) return;
    for (int i = 0; i < db->transaction_count; i++) free(db->transactions[i].items);
    free(db->transactions);
    free(db->transaction_lengths);
    free(db);
}

void mfhoi_result_free(MFHOIResult *result) {
    mfhoi_pattern_list_free(&result->frequent);
    mfhoi_pattern_list_free(&result->mfi);
    mfhoi_pattern_list_free(&result->fhoi);
    mfhoi_pattern_list_free(&result->weak_mfhoi);
    mfhoi_pattern_list_free(&result->strong_mfhoi);
}

const PatternList *mfhoi_select_patterns(const MFHOIResult *result, MFHOIAlgorithm algo) {
    (void)algo;
    return &result->strong_mfhoi;
}

const char *mfhoi_algorithm_name(MFHOIAlgorithm algo) {
    (void)algo;
    return "strong_mfhoi";
}

int mfhoi_parse_algorithm(const char *name, MFHOIAlgorithm *algo) {
    if (name) *algo = MFHOI_ALGO_STRONG;
    return 0;
}

int mfhoi_write_patterns(const char *path, const PatternList *patterns, int transaction_count) {
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;
    for (int i = 0; i < patterns->count; i++) {
        const Pattern *p = &patterns->patterns[i];
        for (int j = 0; j < p->length; j++) fprintf(fp, "%s%d", j ? " " : "", p->items[j]);
        fprintf(fp, " | %d | %.8f | %.8f | %d\n", p->support, (double)p->support / (double)transaction_count, p->avg_occ, p->length);
    }
    fclose(fp);
    return 0;
}

int mfhoi_write_dominance_examples(const char *path, const TransactionDB *db, double minsup_ratio, double minocc, const PatternList *fhoi, int append) {
    (void)path; (void)db; (void)minsup_ratio; (void)minocc; (void)fhoi; (void)append;
    return 0;
}

void mfhoi_compute_quality(const PatternList *patterns, int transaction_count, MFHOIPatternQuality *q) {
    memset(q, 0, sizeof(*q));
    if (!patterns || patterns->count == 0) return;
    for (int i = 0; i < patterns->count; i++) {
        const Pattern *p = &patterns->patterns[i];
        q->avg_itemset_length += p->length;
        if (p->length > q->max_itemset_length) q->max_itemset_length = p->length;
        q->avg_support += p->support;
        q->avg_relative_support += (double)p->support / (double)transaction_count;
        q->avg_occupancy += p->avg_occ;
        if (p->avg_occ > q->max_occupancy) q->max_occupancy = p->avg_occ;
    }
    q->avg_itemset_length /= patterns->count;
    q->avg_support /= patterns->count;
    q->avg_relative_support /= patterns->count;
    q->avg_occupancy /= patterns->count;
}

double mfhoi_exact_overlap_ratio(const PatternList *a, const PatternList *b, int *overlap_count) {
    (void)a; (void)b; if (overlap_count) *overlap_count = 0; return 0.0;
}

double mfhoi_avg_jaccard_to_nearest(const PatternList *source, const PatternList *target) {
    (void)source; (void)target; return 0.0;
}

int mfhoi_write_synthetic_datasets(const char *dataset_root) {
    (void)dataset_root;
    return 0;
}
