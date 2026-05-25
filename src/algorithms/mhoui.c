#include "algorithms/mhoui.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <intrin.h>
#include <windows.h>
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
static int clock_gettime(int unused, struct timespec *ts) {
    static LARGE_INTEGER freq;
    static int initialized = 0;
    LARGE_INTEGER counter;
    (void)unused;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    QueryPerformanceCounter(&counter);
    ts->tv_sec = (time_t)(counter.QuadPart / freq.QuadPart);
    ts->tv_nsec = (long)(((counter.QuadPart % freq.QuadPart) * 1000000000LL) / freq.QuadPart);
    return 0;
}
#endif
#endif

#define MHOUI_WORD_BITS 64
#define MHOUI_EPS 1e-12

typedef struct {
    int tid;
    double iu;
    double ru;
    double invlen;
    int remcnt;
    int pos;
} BOUEntry;

typedef struct {
    uint32_t item;
    int support;
    double twu;
    int rank;
    uint64_t *bitset;
    BOUEntry *entries;
    int count;
    int capacity;
} MHOUIItem;

typedef struct {
    int tid;
    double total_utility;
    int length;
    uint32_t *items;
    int *ranks;
    double *utils;
} OrderedTransaction;

typedef struct {
    OrderedTransaction *transactions;
    int transaction_count;
    int item_count;
    int word_count;
    double total_database_utility;
    MHOUIItem *items;
} MHOUIDB;

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
    MHOUIStats *stats;
} MHOUIContext;

static double elapsed_seconds(const struct timespec *start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - start->tv_sec) + (double)(now.tv_nsec - start->tv_nsec) / 1000000000.0;
}

static int word_count_for_bits(int nbits) {
    return (nbits + MHOUI_WORD_BITS - 1) / MHOUI_WORD_BITS;
}

static void bitset_set(uint64_t *bits, int pos) {
    bits[pos / MHOUI_WORD_BITS] |= UINT64_C(1) << (pos % MHOUI_WORD_BITS);
}

static uint64_t *bitset_clone(const uint64_t *src, int words) {
    uint64_t *out = malloc((size_t)words * sizeof(uint64_t));
    if (out) memcpy(out, src, (size_t)words * sizeof(uint64_t));
    return out;
}

static void bitset_and(uint64_t *out, const uint64_t *a, const uint64_t *b, int words) {
    for (int i = 0; i < words; i++) out[i] = a[i] & b[i];
}

static int bitset_popcount(const uint64_t *bits, int words) {
    int total = 0;
    for (int i = 0; i < words; i++) {
#ifdef _MSC_VER
        total += (int)__popcnt64(bits[i]);
#else
        total += __builtin_popcountll(bits[i]);
#endif
    }
    return total;
}

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int cmp_double_desc(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (y > x) - (y < x);
}

static int cmp_int_desc(const void *a, const void *b) {
    int x = *(const int *)a;
    int y = *(const int *)b;
    return (y > x) - (y < x);
}

static int cmp_item_order(const void *a, const void *b) {
    const MHOUIItem *x = (const MHOUIItem *)a;
    const MHOUIItem *y = (const MHOUIItem *)b;
    if (x->twu < y->twu) return -1;
    if (x->twu > y->twu) return 1;
    if (x->support != y->support) return x->support - y->support;
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

static void bou_add(MHOUIItem *item, BOUEntry entry) {
    if (item->count >= item->capacity) {
        int new_cap = item->capacity ? item->capacity * 2 : 16;
        item->entries = realloc(item->entries, (size_t)new_cap * sizeof(BOUEntry));
        item->capacity = new_cap;
    }
    item->entries[item->count++] = entry;
}

void mhoui_pattern_list_init(MHOUPatternList *list) {
    list->count = 0;
    list->capacity = 256;
    list->patterns = malloc((size_t)list->capacity * sizeof(MHOUPattern));
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
        MHOUPattern *new_patterns = realloc(list->patterns, (size_t)new_cap * sizeof(MHOUPattern));
        if (!new_patterns) return -1;
        list->patterns = new_patterns;
        list->capacity = new_cap;
    }
    MHOUPattern *p = &list->patterns[list->count++];
    p->items = malloc((size_t)length * sizeof(uint32_t));
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

static int proper_subset(const MHOUPattern *x, const MHOUPattern *y) {
    if (x->length >= y->length) return 0;
    int i = 0;
    int j = 0;
    while (i < x->length && j < y->length) {
        if (x->items[i] == y->items[j]) {
            i++;
            j++;
        } else if (x->items[i] > y->items[j]) {
            j++;
        } else {
            return 0;
        }
    }
    return i == x->length;
}

static int pattern_has_item(const MHOUPattern *m, uint32_t item) {
    for (int j = 0; j < m->length; j++) {
        if (m->items[j] == item) return 1;
    }
    return 0;
}

static int pattern_contains_branch(const MHOUPattern *m, const MHOUIDB *db, const uint32_t *prefix, int prefix_len, int suffix_start) {
    int branch_len = prefix_len + (db->item_count - suffix_start);
    if (m->length <= branch_len) return 0;
    for (int i = 0; i < prefix_len; i++) {
        if (!pattern_has_item(m, prefix[i])) return 0;
    }
    for (int i = suffix_start; i < db->item_count; i++) {
        if (!pattern_has_item(m, db->items[i].item)) return 0;
    }
    return 1;
}

static void free_db(MHOUIDB *db) {
    if (!db) return;
    for (int i = 0; i < db->transaction_count; i++) {
        free(db->transactions[i].items);
        free(db->transactions[i].ranks);
        free(db->transactions[i].utils);
    }
    for (int i = 0; i < db->item_count; i++) {
        free(db->items[i].bitset);
        free(db->items[i].entries);
    }
    free(db->transactions);
    free(db->items);
    memset(db, 0, sizeof(*db));
}

static int find_rank_in_transaction(const OrderedTransaction *tr, int rank, double *util, int *pos) {
    for (int i = 0; i < tr->length; i++) {
        if (tr->ranks[i] == rank) {
            if (util) *util = tr->utils[i];
            if (pos) *pos = i;
            return 1;
        }
        if (tr->ranks[i] > rank) return 0;
    }
    return 0;
}

static int build_db(DM_Dataset *ds, int minsup, double minutil, MHOUIDB *out) {
    if (!ds || ds->type != DM_TYPE_UTILITY || ds->count == 0) return -1;
    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    int max_id = (int)ds->max_id;
    int *support = calloc((size_t)max_id + 1, sizeof(int));
    double *twu = calloc((size_t)max_id + 1, sizeof(double));
    if (!support || !twu) {
        free(support);
        free(twu);
        return -1;
    }

    out->transaction_count = (int)ds->count;
    out->word_count = word_count_for_bits(out->transaction_count);
    out->transactions = calloc(ds->count, sizeof(OrderedTransaction));

    for (size_t tid = 0; tid < ds->count; tid++) {
        out->total_database_utility += src[tid].total_utility;
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
    out->items = calloc((size_t)kept, sizeof(MHOUIItem));
    out->item_count = kept;
    int pos = 0;
    for (int id = 0; id <= max_id; id++) {
        if (support[id] >= minsup && twu[id] + MHOUI_EPS >= minutil) {
            out->items[pos].item = (uint32_t)id;
            out->items[pos].support = support[id];
            out->items[pos].twu = twu[id];
            out->items[pos].bitset = calloc((size_t)out->word_count, sizeof(uint64_t));
            pos++;
        }
    }
    qsort(out->items, (size_t)out->item_count, sizeof(MHOUIItem), cmp_item_order);

    int *rank_of_id = malloc(((size_t)max_id + 1) * sizeof(int));
    for (int id = 0; id <= max_id; id++) rank_of_id[id] = -1;
    for (int i = 0; i < out->item_count; i++) {
        out->items[i].rank = i;
        rank_of_id[out->items[i].item] = i;
    }

    for (size_t tid = 0; tid < ds->count; tid++) {
        OrderedTransaction *tr = &out->transactions[tid];
        tr->tid = (int)tid;
        tr->total_utility = src[tid].total_utility;
        tr->items = malloc(src[tid].count * sizeof(uint32_t));
        tr->ranks = malloc(src[tid].count * sizeof(int));
        tr->utils = malloc(src[tid].count * sizeof(double));
        for (size_t j = 0; j < src[tid].count; j++) {
            int rank = rank_of_id[src[tid].items[j].id];
            if (rank < 0) continue;
            int w = tr->length++;
            tr->items[w] = src[tid].items[j].id;
            tr->ranks[w] = rank;
            tr->utils[w] = src[tid].items[j].utility;
            bitset_set(out->items[rank].bitset, (int)tid);
        }
        for (int a = 1; a < tr->length; a++) {
            uint32_t item = tr->items[a];
            int rank = tr->ranks[a];
            double util = tr->utils[a];
            int b = a - 1;
            while (b >= 0 && tr->ranks[b] > rank) {
                tr->items[b + 1] = tr->items[b];
                tr->ranks[b + 1] = tr->ranks[b];
                tr->utils[b + 1] = tr->utils[b];
                b--;
            }
            tr->items[b + 1] = item;
            tr->ranks[b + 1] = rank;
            tr->utils[b + 1] = util;
        }
        double suffix_util = 0.0;
        int suffix_count = 0;
        for (int j = tr->length - 1; j >= 0; j--) {
            int rank = tr->ranks[j];
            BOUEntry e = {
                .tid = (int)tid,
                .iu = tr->utils[j],
                .ru = suffix_util,
                .invlen = tr->length > 0 ? 1.0 / (double)tr->length : 0.0,
                .remcnt = suffix_count,
                .pos = j
            };
            bou_add(&out->items[rank], e);
            suffix_util += tr->utils[j];
            suffix_count++;
        }
    }

    free(rank_of_id);
    free(support);
    free(twu);
    return 0;
}

static BOUEntry *join_bou(const MHOUIDB *db, const BOUEntry *parent, int parent_count, int item_rank, int *out_count, uint64_t *tidset) {
    BOUEntry *entries = malloc((size_t)(parent_count > 0 ? parent_count : 1) * sizeof(BOUEntry));
    int count = 0;
    for (int i = 0; i < parent_count; i++) {
        const OrderedTransaction *tr = &db->transactions[parent[i].tid];
        double util = 0.0;
        int pos = -1;
        if (!find_rank_in_transaction(tr, item_rank, &util, &pos)) continue;
        if (pos <= parent[i].pos) continue;
        double ru = 0.0;
        int remcnt = 0;
        for (int j = pos + 1; j < tr->length; j++) {
            ru += tr->utils[j];
            remcnt++;
        }
        entries[count++] = (BOUEntry){
            .tid = parent[i].tid,
            .iu = parent[i].iu + util,
            .ru = ru,
            .invlen = parent[i].invlen,
            .remcnt = remcnt,
            .pos = pos
        };
        bitset_set(tidset, parent[i].tid);
    }
    *out_count = count;
    return entries;
}

static void list_stats(const MHOUIDB *db, const BOUEntry *entries, int count, int length, double *utility, double *twu, double *aocc, double *uub, double *oub1, double *oub2, int minsup) {
    double util = 0.0;
    double twu_sum = 0.0;
    double invsum = 0.0;
    double bound = 0.0;
    int *caps = NULL;
    double *invs = NULL;
    double *env = NULL;
    if (count >= minsup && minsup > 0) {
        caps = malloc((size_t)count * sizeof(int));
        invs = malloc((size_t)count * sizeof(double));
        env = malloc((size_t)count * sizeof(double));
    }
    for (int i = 0; i < count; i++) {
        util += entries[i].iu;
        twu_sum += db->transactions[entries[i].tid].total_utility;
        invsum += entries[i].invlen;
        bound += entries[i].iu + entries[i].ru;
        if (caps) {
            caps[i] = entries[i].remcnt;
            invs[i] = entries[i].invlen;
            env[i] = entries[i].invlen * (double)(length + entries[i].remcnt);
        }
    }
    *utility = util;
    *twu = twu_sum;
    *aocc = count > 0 ? (double)length * invsum / (double)count : 0.0;
    *uub = bound;
    *oub1 = 0.0;
    *oub2 = 0.0;
    if (caps) {
        qsort(caps, (size_t)count, sizeof(int), cmp_int_desc);
        qsort(invs, (size_t)count, sizeof(double), cmp_double_desc);
        qsort(env, (size_t)count, sizeof(double), cmp_double_desc);
        double inv_top = 0.0;
        double env_top = 0.0;
        for (int i = 0; i < minsup; i++) {
            inv_top += invs[i];
            env_top += env[i];
        }
        *oub1 = (double)(length + caps[minsup - 1]) * inv_top / (double)minsup;
        *oub2 = env_top / (double)minsup;
    }
    free(caps);
    free(invs);
    free(env);
}

static int branch_dominated(const MHOUIContext *ctx, const uint32_t *prefix, int prefix_len, int suffix_start, double uub, double oub2) {
    for (int i = 0; i < ctx->houi->count; i++) {
        const MHOUPattern *m = &ctx->houi->patterns[i];
        if (!pattern_contains_branch(m, ctx->db, prefix, prefix_len, suffix_start)) continue;
        if (ctx->strong_direct) {
            if (m->avg_occupancy + MHOUI_EPS >= oub2 && m->utility + MHOUI_EPS >= uub) return 1;
        } else {
            int better_occ = m->avg_occupancy > oub2 + MHOUI_EPS && m->utility + MHOUI_EPS >= uub;
            int better_util = m->avg_occupancy + MHOUI_EPS >= oub2 && m->utility > uub + MHOUI_EPS;
            if (better_occ || better_util) return 1;
        }
    }
    return 0;
}

static void dfs_mine(MHOUIContext *ctx, uint32_t *prefix, int prefix_len, const uint64_t *parent_tidset, const BOUEntry *parent_entries, int parent_count, int suffix_start) {
    if (ctx->stats->status_limited) return;
    if (ctx->max_seconds > 0.0 && elapsed_seconds(&ctx->start) > ctx->max_seconds) {
        ctx->stats->status_limited = 1;
        return;
    }
    for (int idx = suffix_start; idx < ctx->db->item_count; idx++) {
        ctx->stats->generated_candidates++;
        uint64_t *tidset = calloc((size_t)ctx->db->word_count, sizeof(uint64_t));
        BOUEntry *entries = NULL;
        int count = 0;
        if (parent_entries) {
            bitset_and(tidset, parent_tidset, ctx->db->items[idx].bitset, ctx->db->word_count);
            int support = bitset_popcount(tidset, ctx->db->word_count);
            if (support < ctx->minsup) {
                ctx->stats->pruned_support++;
                free(tidset);
                continue;
            }
            memset(tidset, 0, (size_t)ctx->db->word_count * sizeof(uint64_t));
            entries = join_bou(ctx->db, parent_entries, parent_count, idx, &count, tidset);
        } else {
            count = ctx->db->items[idx].count;
            if (count < ctx->minsup) {
                ctx->stats->pruned_support++;
                free(tidset);
                continue;
            }
            memcpy(tidset, ctx->db->items[idx].bitset, (size_t)ctx->db->word_count * sizeof(uint64_t));
            entries = malloc((size_t)count * sizeof(BOUEntry));
            memcpy(entries, ctx->db->items[idx].entries, (size_t)count * sizeof(BOUEntry));
        }
        if (count < ctx->minsup) {
            ctx->stats->pruned_support++;
            free(entries);
            free(tidset);
            continue;
        }

        prefix[prefix_len] = ctx->db->items[idx].item;
        int new_len = prefix_len + 1;
        double utility = 0.0;
        double twu = 0.0;
        double aocc = 0.0;
        double uub = 0.0;
        double oub1 = 0.0;
        double oub2 = 0.0;
        list_stats(ctx->db, entries, count, new_len, &utility, &twu, &aocc, &uub, &oub1, &oub2, ctx->minsup);
        ctx->stats->visited_nodes++;

        if (twu + MHOUI_EPS < ctx->minutil) {
            ctx->stats->pruned_twu++;
            free(entries);
            free(tidset);
            continue;
        }
        if (uub + MHOUI_EPS < ctx->minutil) {
            ctx->stats->pruned_uub++;
            free(entries);
            free(tidset);
            continue;
        }
        if (oub1 > 0.0 && oub1 + MHOUI_EPS < ctx->minocc) {
            ctx->stats->pruned_oub1++;
            free(entries);
            free(tidset);
            continue;
        }
        if (oub2 > 0.0 && oub2 + MHOUI_EPS < ctx->minocc) {
            ctx->stats->pruned_oub2++;
            free(entries);
            free(tidset);
            continue;
        }

        if (utility + MHOUI_EPS >= ctx->minutil && aocc + MHOUI_EPS >= ctx->minocc) {
            pattern_add(ctx->houi, prefix, new_len, tidset, ctx->db->word_count, count, utility, aocc);
            ctx->stats->houi_count = ctx->houi->count;
            if (ctx->max_patterns > 0 && ctx->houi->count >= ctx->max_patterns) ctx->stats->status_limited = 1;
        }
        if (!ctx->stats->status_limited && idx + 1 < ctx->db->item_count) {
            if (ctx->direct && branch_dominated(ctx, prefix, new_len, idx + 1, uub, oub2)) {
                ctx->stats->pruned_dominance++;
            } else {
                dfs_mine(ctx, prefix, new_len, tidset, entries, count, idx + 1);
            }
        }
        free(entries);
        free(tidset);
        if (ctx->stats->status_limited) return;
    }
}

static void copy_pattern(MHOUPatternList *out, const MHOUPattern *p, int word_count) {
    pattern_add(out, p->items, p->length, p->tidset, word_count, p->support, p->utility, p->avg_occupancy);
}

static void filter_mhoui(const MHOUPatternList *houi, MHOUPatternList *out, int word_count, int strong, int *removed) {
    mhoui_pattern_list_init(out);
    *removed = 0;
    if (!houi || houi->count == 0) return;
    MHOUPattern *sorted = malloc((size_t)houi->count * sizeof(MHOUPattern));
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

int mhoui_mine_dataset(DM_Dataset *ds, int minsup_count, double minocc, double minutil, const MHOUILimits *limits, MHOUIResult *out) {
    memset(out, 0, sizeof(*out));
    if (!ds || ds->type != DM_TYPE_UTILITY || minsup_count <= 0 || minocc <= 0.0 || minutil < 0.0) return -1;
    MHOUIDB db = {0};
    if (build_db(ds, minsup_count, minutil, &db) != 0) return -1;
    mhoui_pattern_list_init(&out->houi);
    uint32_t *prefix = malloc((size_t)(db.item_count > 0 ? db.item_count : 1) * sizeof(uint32_t));
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
        .stats = &out->stats
    };
    clock_gettime(CLOCK_MONOTONIC, &ctx.start);
    dfs_mine(&ctx, prefix, 0, NULL, NULL, 0, 0);
    filter_mhoui(&out->houi, &out->weak_mhoui, db.word_count, 0, &out->stats.weak_removed);
    filter_mhoui(&out->houi, &out->strong_mhoui, db.word_count, 1, &out->stats.strong_removed);
    out->stats.houi_count = out->houi.count;
    out->stats.weak_count = out->weak_mhoui.count;
    out->stats.strong_count = out->strong_mhoui.count;
    free(prefix);
    free_db(&db);
    return out->stats.status_limited ? 1 : 0;
}

void mhoui_result_free(MHOUIResult *result) {
    mhoui_pattern_list_free(&result->houi);
    mhoui_pattern_list_free(&result->weak_mhoui);
    mhoui_pattern_list_free(&result->strong_mhoui);
}

const MHOUPatternList *mhoui_select_patterns(const MHOUIResult *result, MHOUIAlgorithm algo) {
    switch (algo) {
        case MHOUI_ALGO_HOUI: return &result->houi;
        case MHOUI_ALGO_WEAK:
        case MHOUI_ALGO_DIRECT_WEAK: return &result->weak_mhoui;
        case MHOUI_ALGO_STRONG:
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
    if (strcmp(name, "houi") == 0 || strcmp(name, "houi_miner") == 0) *algo = MHOUI_ALGO_HOUI;
    else if (strcmp(name, "weak_mhoui") == 0 || strcmp(name, "weak") == 0) *algo = MHOUI_ALGO_WEAK;
    else if (strcmp(name, "strong_mhoui") == 0 || strcmp(name, "strong") == 0) *algo = MHOUI_ALGO_STRONG;
    else if (strcmp(name, "direct_weak_mhoui") == 0 || strcmp(name, "direct_weak") == 0) *algo = MHOUI_ALGO_DIRECT_WEAK;
    else if (strcmp(name, "direct_strong_mhoui") == 0 || strcmp(name, "direct_strong") == 0 || strcmp(name, "mhoui") == 0) *algo = MHOUI_ALGO_DIRECT_STRONG;
    else return -1;
    return 0;
}

int mhoui_write_patterns(const char *path, const MHOUPatternList *patterns, size_t transaction_count) {
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;
    fprintf(fp, "items,support,relative_support,utility,avg_occupancy,length\n");
    for (int i = 0; i < patterns->count; i++) {
        const MHOUPattern *p = &patterns->patterns[i];
        fprintf(fp, "\"");
        for (int j = 0; j < p->length; j++) fprintf(fp, "%s%u", j ? " " : "", p->items[j]);
        fprintf(fp, "\",%d,%.10f,%.10f,%.10f,%d\n", p->support, transaction_count ? (double)p->support / (double)transaction_count : 0.0, p->utility, p->avg_occupancy, p->length);
    }
    fclose(fp);
    return 0;
}

void mhoui_pattern_quality(const MHOUPatternList *patterns, double *avg_len, int *max_len, double *avg_support, double *avg_utility, double *avg_occ) {
    *avg_len = 0.0;
    *max_len = 0;
    *avg_support = 0.0;
    *avg_utility = 0.0;
    *avg_occ = 0.0;
    if (!patterns || patterns->count == 0) return;
    for (int i = 0; i < patterns->count; i++) {
        const MHOUPattern *p = &patterns->patterns[i];
        *avg_len += p->length;
        if (p->length > *max_len) *max_len = p->length;
        *avg_support += p->support;
        *avg_utility += p->utility;
        *avg_occ += p->avg_occupancy;
    }
    *avg_len /= patterns->count;
    *avg_support /= patterns->count;
    *avg_utility /= patterns->count;
    *avg_occ /= patterns->count;
}

static DM_Status run_mhoui(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
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
    printf("[MHOUI] Complete. HOUI=%d Weak=%d Strong=%d Output=%d%s\n",
           result.stats.houi_count, result.stats.weak_count, result.stats.strong_count,
           out->count, status == 1 ? " LIMITED" : "");
    printf("[MHOUI] visited=%d candidates=%d pruned_support=%d pruned_twu=%d pruned_uub=%d pruned_oub1=%d pruned_oub2=%d pruned_dom=%d\n",
           result.stats.visited_nodes, result.stats.generated_candidates, result.stats.pruned_support,
           result.stats.pruned_twu, result.stats.pruned_uub, result.stats.pruned_oub1,
           result.stats.pruned_oub2, result.stats.pruned_dominance);
    dm_bench_record_results((size_t)out->count, total_items);
    mhoui_result_free(&result);
    return status == 0 ? DM_SUCCESS : DM_ERROR_GENERIC;
}

DM_Algorithm mhoui_algo = {
    .id = "mhoui",
    .name = "MHOUI-Miner",
    .description = "Exact high-occupancy utility itemset miner with weak/strong MHOUI dominance filtering.",
    .supported_types = (1u << DM_TYPE_UTILITY),
    .run = run_mhoui
};
