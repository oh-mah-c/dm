#include "algorithms/mhoui.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MHOUI_EPS 1e-12
#define MAX_HUT_LEN 512

/* ── Compact Item Struct ─────────────────────────────────────────────────── */
typedef struct {
    uint32_t item;
    int support;
    double twu;
    int rank;
    int occ_offset; /* offset in db->all_occs */
} MHOUIItem;

/* ── Item Occurrence Record ──────────────────────────────────────────────── */
typedef struct {
    int tid;
    int offset;     /* position of item in transaction ranks */
} ItemOcc;

/* ── Transaction Storage (Zero-Copy Pointer Arrays) ───────────────────────── */
typedef struct {
    int tid;
    double total_utility;
    int length;
    int *ranks;        /* sorted ascending by rank */
    double *utils;     /* utilities of items in this transaction */
    double *suf_utils; /* suf_utils[i] = sum_{j >= i} utils[j] */
} CompactTransaction;

typedef struct {
    CompactTransaction *transactions;
    int transaction_count;
    int item_count;
    double total_database_utility;
    MHOUIItem *items;
    double *invlen_table; /* Precomputed 1.0 / length */
    int *all_ranks;
    double *all_utils;
    double *all_suf_utils;
    ItemOcc *all_occs;
    size_t total_trans_items;
} MHOUIDB;

/* ── Pure Projected Transaction (Zero Bitsets) ───────────────────────────── */
typedef struct {
    int tid;
    int offset;         /* offset in tr->ranks after the prefix item */
    double prefix_util; /* accumulated utility of prefix items in this transaction */
} ProjTrans;

/* ── Continuous Reusable Projection Arena ─────────────────────────────────── */
typedef struct {
    ProjTrans *pool;
    size_t capacity;
    size_t top;
} ProjArena;

static inline int arena_init(ProjArena *arena, size_t initial_cap) {
    arena->capacity = initial_cap > 0 ? initial_cap : 1048576;
    arena->pool = (ProjTrans *)malloc(arena->capacity * sizeof(ProjTrans));
    arena->top = 0;
    return arena->pool ? 0 : -1;
}

static inline void arena_free(ProjArena *arena) {
    if (arena && arena->pool) {
        free(arena->pool);
        arena->pool = NULL;
    }
    arena->capacity = 0;
    arena->top = 0;
}

static inline ProjTrans *arena_alloc(ProjArena *arena, size_t count) {
    if (arena->top + count > arena->capacity) {
        size_t new_cap = arena->capacity * 2;
        while (arena->top + count > new_cap) new_cap *= 2;
        ProjTrans *new_pool = (ProjTrans *)realloc(arena->pool, new_cap * sizeof(ProjTrans));
        if (!new_pool) return NULL;
        arena->pool = new_pool;
        arena->capacity = new_cap;
    }
    ProjTrans *ptr = &arena->pool[arena->top];
    arena->top += count;
    return ptr;
}

static inline void arena_reset_to(ProjArena *arena, size_t save_top) {
    arena->top = save_top;
}

/* ── Global DFS Workspace (Zero Malloc in Search Tree) ────────────────────── */
typedef struct {
    int *local_supp;
    double *local_util;
    double *local_suf_util;
    double *local_inv_sum;
    double *local_occ_env;
    int *touched;
    int *cand_stack;
    double *uub_stack;
    double *oub_stack;
} MHOUIWorkspace;

static int ws_init(MHOUIWorkspace *ws, int item_count, int max_depth) {
    size_t n = (size_t)(item_count > 0 ? item_count : 1);
    size_t stack_sz = (size_t)(max_depth > 0 ? max_depth : 64) * n;

    ws->local_supp = (int *)calloc(n, sizeof(int));
    ws->local_util = (double *)calloc(n, sizeof(double));
    ws->local_suf_util = (double *)calloc(n, sizeof(double));
    ws->local_inv_sum = (double *)calloc(n, sizeof(double));
    ws->local_occ_env = (double *)calloc(n, sizeof(double));
    ws->touched = (int *)malloc(n * sizeof(int));
    ws->cand_stack = (int *)malloc(stack_sz * sizeof(int));
    ws->uub_stack = (double *)malloc(stack_sz * sizeof(double));
    ws->oub_stack = (double *)malloc(stack_sz * sizeof(double));

    return (ws->local_supp && ws->local_util && ws->local_suf_util &&
            ws->local_inv_sum && ws->local_occ_env && ws->touched &&
            ws->cand_stack && ws->uub_stack && ws->oub_stack) ? 0 : -1;
}

static void ws_free(MHOUIWorkspace *ws) {
    free(ws->local_supp);
    free(ws->local_util);
    free(ws->local_suf_util);
    free(ws->local_inv_sum);
    free(ws->local_occ_env);
    free(ws->touched);
    free(ws->cand_stack);
    free(ws->uub_stack);
    free(ws->oub_stack);
    memset(ws, 0, sizeof(*ws));
}

/* ── Dynamic Pareto-Trie / Bloom Cache for Superset Dominance ────────────── */
typedef struct {
    uint32_t *items;
    int length;
    uint64_t mask;
    double utility;
    double avg_occupancy;
} ParetoPattern;

typedef struct {
    ParetoPattern *patterns;
    size_t count;
    size_t capacity;
} ParetoPatternCache;

/* ── Mining Context ──────────────────────────────────────────────────────── */
typedef struct {
    const MHOUIDB *db;
    int minsup;
    double minocc;
    double minutil;
    int max_patterns;
    double max_seconds;
    int direct;
    int strong_direct;
    int disable_uub;
    int disable_oub;
    int disable_dominance;
    int max_depth;
    struct timespec start;

    MHOUPatternList *houi;
    ParetoPatternCache pcache;
    ProjArena arena;
    MHOUIWorkspace ws;

    MHOUIStats *stats;
} MHOUIContext;

static inline double elapsed_seconds(const struct timespec *start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - start->tv_sec) + (double)(now.tv_nsec - start->tv_nsec) / 1000000000.0;
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
    if (p->tidset) free(p->tidset);
    p->items = NULL;
    p->tidset = NULL;
}

void mhoui_pattern_list_free(MHOUPatternList *list) {
    if (!list || !list->patterns) return;
    for (int i = 0; i < list->count; i++) pattern_free(&list->patterns[i]);
    free(list->patterns);
    list->patterns = NULL;
    list->count = 0;
    list->capacity = 0;
}

static int pattern_add(MHOUPatternList *list, const uint32_t *items, int length,
                       int support, double utility, double avg_occupancy) {
    if (list->count >= list->capacity) {
        int new_cap = list->capacity ? list->capacity * 2 : 256;
        MHOUPattern *new_patterns = (MHOUPattern *)realloc(list->patterns, (size_t)new_cap * sizeof(MHOUPattern));
        if (!new_patterns) return -1;
        list->patterns = new_patterns;
        list->capacity = new_cap;
    }
    MHOUPattern *p = &list->patterns[list->count++];
    p->items = (uint32_t *)malloc((size_t)length * sizeof(uint32_t));
    p->tidset = NULL;
    if (!p->items) return -1;
    memcpy(p->items, items, (size_t)length * sizeof(uint32_t));
    qsort(p->items, (size_t)length, sizeof(uint32_t), cmp_uint32);
    p->length = length;
    p->support = support;
    p->utility = utility;
    p->avg_occupancy = avg_occupancy;
    return 0;
}

/* ── Inverted Pareto Cache with Bloom Rejection ───────────────────────────── */
static void pcache_init(ParetoPatternCache *pc) {
    pc->count = 0;
    pc->capacity = 1024;
    pc->patterns = (ParetoPattern *)malloc(pc->capacity * sizeof(ParetoPattern));
}

static void pcache_free(ParetoPatternCache *pc) {
    if (!pc || !pc->patterns) return;
    for (size_t i = 0; i < pc->count; i++) {
        free(pc->patterns[i].items);
    }
    free(pc->patterns);
    pc->patterns = NULL;
    pc->count = 0;
    pc->capacity = 0;
}

static void pcache_add(ParetoPatternCache *pc, const uint32_t *items, int length, double utility, double avg_occupancy) {
    if (pc->count >= pc->capacity) {
        pc->capacity *= 2;
        ParetoPattern *np = (ParetoPattern *)realloc(pc->patterns, pc->capacity * sizeof(ParetoPattern));
        if (!np) return;
        pc->patterns = np;
    }
    ParetoPattern *pp = &pc->patterns[pc->count++];
    pp->items = (uint32_t *)malloc((size_t)length * sizeof(uint32_t));
    if (!pp->items) {
        pc->count--;
        return;
    }
    memcpy(pp->items, items, (size_t)length * sizeof(uint32_t));
    qsort(pp->items, (size_t)length, sizeof(uint32_t), cmp_uint32);
    pp->length = length;
    pp->utility = utility;
    pp->avg_occupancy = avg_occupancy;
    uint64_t m = 0;
    for (int i = 0; i < length; i++) {
        m |= UINT64_C(1) << (items[i] & 63);
    }
    pp->mask = m;
}

static inline int itemset_is_subset(const uint32_t *sub, int sub_len, const uint32_t *sup, int sup_len) {
    if (sub_len > sup_len) return 0;
    int i = 0, j = 0;
    while (i < sub_len && j < sup_len) {
        if (sub[i] == sup[j]) {
            i++; j++;
        } else if (sub[i] > sup[j]) {
            j++;
        } else {
            return 0;
        }
    }
    return i == sub_len;
}

static int pcache_dominates_branch(const ParetoPatternCache *pc, const uint32_t *branch_items, int branch_len,
                                   uint64_t branch_mask, double max_uub, double max_oub, int strong_direct) {
    for (int i = (int)pc->count - 1; i >= 0; i--) {
        const ParetoPattern *pp = &pc->patterns[i];
        if (pp->length <= branch_len) continue;
        if (strong_direct) {
            if (pp->avg_occupancy + MHOUI_EPS < max_oub || pp->utility + MHOUI_EPS < max_uub) continue;
        } else {
            int better_occ = pp->avg_occupancy > max_oub + MHOUI_EPS && pp->utility + MHOUI_EPS >= max_uub;
            int better_util = pp->avg_occupancy + MHOUI_EPS >= max_oub && pp->utility > max_uub + MHOUI_EPS;
            if (!better_occ && !better_util) continue;
        }
        if ((branch_mask & ~pp->mask) != 0) continue;
        if (itemset_is_subset(branch_items, branch_len, pp->items, pp->length)) return 1;
    }
    return 0;
}

/* ── Database Preprocessing (Zero Bitsets) ────────────────────────────────── */
static void free_db(MHOUIDB *db) {
    if (!db) return;
    free(db->all_ranks);
    free(db->all_utils);
    free(db->all_suf_utils);
    free(db->all_occs);
    free(db->transactions);
    free(db->items);
    free(db->invlen_table);
    memset(db, 0, sizeof(*db));
}

static int bsearch_u32(const uint32_t *arr, int n, uint32_t key) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (arr[mid] == key) return mid;
        if (arr[mid] < key) lo = mid + 1;
        else hi = mid - 1;
    }
    return -1;
}

static int build_db(DM_Dataset *ds, int minsup, double minutil, MHOUIDB *out) {
    if (!ds || ds->type != DM_TYPE_UTILITY || ds->count == 0) return -1;
    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    int max_id = (int)ds->max_id;

    out->transaction_count = (int)ds->count;
    out->transactions = (CompactTransaction *)calloc(ds->count, sizeof(CompactTransaction));
    out->invlen_table = (double *)calloc(ds->count, sizeof(double));
    if (!out->transactions || !out->invlen_table) return -1;

    size_t total_trans_items = 0;
    for (size_t tid = 0; tid < ds->count; tid++) {
        out->total_database_utility += src[tid].total_utility;
        out->invlen_table[tid] = src[tid].count > 0 ? (1.0 / (double)src[tid].count) : 0.0;
        total_trans_items += src[tid].count;
    }
    out->total_trans_items = total_trans_items;

    out->all_ranks = (int *)malloc(total_trans_items * sizeof(int));
    out->all_utils = (double *)malloc(total_trans_items * sizeof(double));
    out->all_suf_utils = (double *)malloc((total_trans_items + ds->count) * sizeof(double));
    if (total_trans_items > 0 && (!out->all_ranks || !out->all_utils || !out->all_suf_utils)) {
        free_db(out);
        return -1;
    }

    if (max_id <= 1000000) {
        int *support = (int *)calloc((size_t)max_id + 1, sizeof(int));
        double *twu = (double *)calloc((size_t)max_id + 1, sizeof(double));
        if (!support || !twu) {
            free(support);
            free(twu);
            free_db(out);
            return -1;
        }

        for (size_t tid = 0; tid < ds->count; tid++) {
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
                pos++;
            }
        }
        qsort(out->items, (size_t)out->item_count, sizeof(MHOUIItem), cmp_item_order);

        int *rank_of_id = (int *)malloc(((size_t)max_id + 1) * sizeof(int));
        if (!rank_of_id) { free(support); free(twu); free_db(out); return -1; }
        for (int id = 0; id <= max_id; id++) rank_of_id[id] = -1;
        for (int i = 0; i < out->item_count; i++) {
            out->items[i].rank = i;
            rank_of_id[out->items[i].item] = i;
        }

        size_t cur_offset = 0;
        size_t cur_suf_offset = 0;
        for (size_t tid = 0; tid < ds->count; tid++) {
            CompactTransaction *tr = &out->transactions[tid];
            tr->tid = (int)tid;
            tr->total_utility = src[tid].total_utility;
            tr->ranks = &out->all_ranks[cur_offset];
            tr->utils = &out->all_utils[cur_offset];
            tr->suf_utils = &out->all_suf_utils[cur_suf_offset];
            cur_offset += src[tid].count;
            cur_suf_offset += src[tid].count + 1;
            for (size_t j = 0; j < src[tid].count; j++) {
                int rank = rank_of_id[src[tid].items[j].id];
                if (rank < 0) continue;
                int w = tr->length++;
                tr->ranks[w] = rank;
                tr->utils[w] = src[tid].items[j].utility;
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
            /* Precompute suffix utilities */
            tr->suf_utils[tr->length] = 0.0;
            for (int a = tr->length - 1; a >= 0; a--) {
                tr->suf_utils[a] = tr->suf_utils[a + 1] + tr->utils[a];
            }
        }

        free(rank_of_id);
        free(support);
        free(twu);
    } else {
        /* Sparse item ID mode */
        size_t total_items = 0;
        for (size_t tid = 0; tid < ds->count; tid++) total_items += src[tid].count;

        uint32_t *all_ids = (uint32_t *)malloc(total_items * sizeof(uint32_t));
        if (!all_ids) { free_db(out); return -1; }
        size_t cur_pos = 0;
        for (size_t tid = 0; tid < ds->count; tid++) {
            for (size_t j = 0; j < src[tid].count; j++) {
                all_ids[cur_pos++] = src[tid].items[j].id;
            }
        }
        qsort(all_ids, total_items, sizeof(uint32_t), cmp_uint32);

        int ucount = 0;
        for (size_t i = 0; i < total_items; i++) {
            if (i == 0 || all_ids[i] != all_ids[i - 1]) {
                all_ids[ucount++] = all_ids[i];
            }
        }

        int *support = (int *)calloc((size_t)ucount, sizeof(int));
        double *twu = (double *)calloc((size_t)ucount, sizeof(double));
        if (!support || !twu) {
            free(all_ids); free(support); free(twu); free_db(out); return -1;
        }

        for (size_t tid = 0; tid < ds->count; tid++) {
            for (size_t j = 0; j < src[tid].count; j++) {
                int idx = bsearch_u32(all_ids, ucount, src[tid].items[j].id);
                if (idx >= 0) {
                    support[idx]++;
                    twu[idx] += src[tid].total_utility;
                }
            }
        }

        int kept = 0;
        for (int idx = 0; idx < ucount; idx++) {
            if (support[idx] >= minsup && twu[idx] + MHOUI_EPS >= minutil) kept++;
        }
        out->items = (MHOUIItem *)calloc((size_t)kept, sizeof(MHOUIItem));
        out->item_count = kept;
        int pos = 0;
        for (int idx = 0; idx < ucount; idx++) {
            if (support[idx] >= minsup && twu[idx] + MHOUI_EPS >= minutil) {
                out->items[pos].item = all_ids[idx];
                out->items[pos].support = support[idx];
                out->items[pos].twu = twu[idx];
                pos++;
            }
        }
        qsort(out->items, (size_t)out->item_count, sizeof(MHOUIItem), cmp_item_order);

        int *rank_of_uidx = (int *)malloc((size_t)ucount * sizeof(int));
        if (!rank_of_uidx) { free(all_ids); free(support); free(twu); free_db(out); return -1; }
        for (int idx = 0; idx < ucount; idx++) rank_of_uidx[idx] = -1;
        for (int i = 0; i < out->item_count; i++) {
            out->items[i].rank = i;
            int idx = bsearch_u32(all_ids, ucount, out->items[i].item);
            if (idx >= 0) rank_of_uidx[idx] = i;
        }

        size_t cur_offset = 0;
        size_t cur_suf_offset = 0;
        for (size_t tid = 0; tid < ds->count; tid++) {
            CompactTransaction *tr = &out->transactions[tid];
            tr->tid = (int)tid;
            tr->total_utility = src[tid].total_utility;
            tr->ranks = &out->all_ranks[cur_offset];
            tr->utils = &out->all_utils[cur_offset];
            tr->suf_utils = &out->all_suf_utils[cur_suf_offset];
            cur_offset += src[tid].count;
            cur_suf_offset += src[tid].count + 1;
            for (size_t j = 0; j < src[tid].count; j++) {
                int idx = bsearch_u32(all_ids, ucount, src[tid].items[j].id);
                if (idx < 0) continue;
                int rank = rank_of_uidx[idx];
                if (rank < 0) continue;
                int w = tr->length++;
                tr->ranks[w] = rank;
                tr->utils[w] = src[tid].items[j].utility;
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
            /* Precompute suffix utilities */
            tr->suf_utils[tr->length] = 0.0;
            for (int a = tr->length - 1; a >= 0; a--) {
                tr->suf_utils[a] = tr->suf_utils[a + 1] + tr->utils[a];
            }
        }

        free(rank_of_uidx);
        free(all_ids);
        free(support);
        free(twu);
    }

    /* Build Inverted Occurrence Pool (8 bytes per occurrence, completely replaces bitsets) */
    size_t cur_occ = 0;
    for (int i = 0; i < out->item_count; i++) {
        out->items[i].occ_offset = (int)cur_occ;
        cur_occ += (size_t)out->items[i].support;
    }
    out->all_occs = (ItemOcc *)malloc((cur_occ > 0 ? cur_occ : 1) * sizeof(ItemOcc));
    if (cur_occ > 0 && !out->all_occs) {
        free_db(out);
        return -1;
    }

    size_t itm_cnt = out->item_count > 0 ? (size_t)out->item_count : 1;
    int *item_cur = (int *)malloc(itm_cnt * sizeof(int));
    if (out->item_count > 0 && !item_cur) {
        free_db(out);
        return -1;
    }
    for (int i = 0; i < out->item_count; i++) {
        item_cur[i] = out->items[i].occ_offset;
    }

    for (size_t tid = 0; tid < ds->count; tid++) {
        CompactTransaction *tr = &out->transactions[tid];
        for (int w = 0; w < tr->length; w++) {
            int r = tr->ranks[w];
            int pos = item_cur[r]++;
            out->all_occs[pos].tid = (int)tid;
            out->all_occs[pos].offset = w;
        }
    }
    free(item_cur);

    return 0;
}

/* ── Recursive Search Engine with Pure Projection & HUT Pruning ───────────── */
static void proj_dfs(MHOUIContext *ctx, int depth,
                     uint32_t *prefix_items, int *prefix_ranks, int prefix_len,
                     size_t proj_offset, int proj_count, double cur_util) {
    if (ctx->stats->status_limited) return;
    if (ctx->max_seconds > 0.0 && elapsed_seconds(&ctx->start) > ctx->max_seconds) {
        ctx->stats->status_limited = 1;
        return;
    }
    if (depth >= ctx->max_depth) return;

    const MHOUIDB *db = ctx->db;
    MHOUIWorkspace *ws = &ctx->ws;
    const ProjTrans *proj = &ctx->arena.pool[proj_offset];

    /* Step 1: Single-pass local support, utility, and occupancy accumulation */
    int num_touched = 0;
    for (int i = 0; i < proj_count; i++) {
        int tid = proj[i].tid;
        int off = proj[i].offset;
        double p_u = proj[i].prefix_util;
        const CompactTransaction *tr = &db->transactions[tid];
        double invlen = db->invlen_table[tid];

        for (int j = off; j < tr->length; j++) {
            int r = tr->ranks[j];
            if (ws->local_supp[r] == 0) {
                ws->touched[num_touched++] = r;
            }
            ws->local_supp[r]++;
            ws->local_util[r] += p_u + tr->utils[j];
            ws->local_suf_util[r] += tr->suf_utils[j + 1];
            ws->local_inv_sum[r] += invlen;

            int rem_cnt = tr->length - 1 - j;
            double max_l = (double)(prefix_len + 1 + rem_cnt);
            double cap = max_l * invlen;
            if (cap > 1.0) cap = 1.0;
            ws->local_occ_env[r] += cap;
        }
    }

    if (num_touched == 0) return;

    /* Step 2: Candidate Filtering */
    int *cands = &ws->cand_stack[depth * db->item_count];
    double *cands_uub = &ws->uub_stack[depth * db->item_count];
    double *cands_oub = &ws->oub_stack[depth * db->item_count];
    int cand_count = 0;

    for (int i = 0; i < num_touched; i++) {
        int r = ws->touched[i];
        ctx->stats->generated_candidates++;

        if (ws->local_supp[r] < ctx->minsup) {
            ctx->stats->pruned_support++;
            continue;
        }

        double uub = ws->local_util[r] + ws->local_suf_util[r];
        if (!ctx->disable_uub && (uub + MHOUI_EPS < ctx->minutil)) {
            ctx->stats->pruned_uub++;
            continue;
        }

        double oub = ws->local_occ_env[r] / (double)ctx->minsup;
        if (!ctx->disable_oub && (oub > 0.0 && oub + MHOUI_EPS < ctx->minocc)) {
            ctx->stats->pruned_oub2++;
            continue;
        }

        cands[cand_count] = r;
        cands_uub[cand_count] = uub;
        cands_oub[cand_count] = oub;
        cand_count++;
    }

    /* Step 4: Reset global workspace arrays before recursion */
    for (int i = 0; i < num_touched; i++) {
        int r = ws->touched[i];
        ws->local_supp[r] = 0;
        ws->local_util[r] = 0.0;
        ws->local_suf_util[r] = 0.0;
        ws->local_inv_sum[r] = 0.0;
        ws->local_occ_env[r] = 0.0;
    }

    if (cand_count == 0) return;

    /* Sort candidate ranks ascending */
    for (int a = 1; a < cand_count; a++) {
        int cr = cands[a];
        double cu = cands_uub[a];
        double co = cands_oub[a];
        int b = a - 1;
        while (b >= 0 && cands[b] > cr) {
            cands[b + 1] = cands[b];
            cands_uub[b + 1] = cands_uub[b];
            cands_oub[b + 1] = cands_oub[b];
            b--;
        }
        cands[b + 1] = cr;
        cands_uub[b + 1] = cu;
        cands_oub[b + 1] = co;
    }

    /* Step 5: Head-Union-Tail (HUT) Branch Pareto Dominance Pruning */
    if (ctx->direct && !ctx->disable_dominance && ctx->pcache.count > 0 && cand_count <= 32) {
        int hut_len = prefix_len + cand_count;
        uint32_t hut_items[MAX_HUT_LEN];
        if (hut_len < MAX_HUT_LEN) {
            for (int p = 0; p < prefix_len; p++) hut_items[p] = prefix_items[p];
            for (int c = 0; c < cand_count; c++) hut_items[prefix_len + c] = db->items[cands[c]].item;
            qsort(hut_items, (size_t)hut_len, sizeof(uint32_t), cmp_uint32);

            uint64_t hut_mask = 0;
            for (int h = 0; h < hut_len; h++) hut_mask |= UINT64_C(1) << (hut_items[h] & 63);

            double max_uub = 0.0;
            double max_oub = 0.0;
            for (int c = 0; c < cand_count; c++) {
                if (cands_uub[c] > max_uub) max_uub = cands_uub[c];
                if (cands_oub[c] > max_oub) max_oub = cands_oub[c];
            }

            if (pcache_dominates_branch(&ctx->pcache, hut_items, hut_len, hut_mask, max_uub, max_oub, ctx->strong_direct)) {
                ctx->stats->pruned_dominance++;
                return;
            }
        }
    }

    /* Step 6: Explore children with pure projected databases and arena memory */
    for (int i = 0; i < cand_count; i++) {
        if (ctx->stats->status_limited) break;

        int r = cands[i];
        ctx->stats->visited_nodes++;

        size_t child_offset = ctx->arena.top;
        if (!arena_alloc(&ctx->arena, (size_t)proj_count)) {
            ctx->stats->status_limited = 1;
            break;
        }

        /* Re-fetch pointers after possible arena reallocation */
        const ProjTrans *src_proj = &ctx->arena.pool[proj_offset];
        ProjTrans *child_proj = &ctx->arena.pool[child_offset];

        int child_sup = 0;
        double child_util = 0.0;
        double child_inv_sum = 0.0;

        for (int p = 0; p < proj_count; p++) {
            int tid = src_proj[p].tid;
            int off = src_proj[p].offset;
            const CompactTransaction *tr = &db->transactions[tid];

            for (int j = off; j < tr->length; j++) {
                if (tr->ranks[j] == r) {
                    child_proj[child_sup].tid = tid;
                    child_proj[child_sup].offset = j + 1;
                    double u = src_proj[p].prefix_util + tr->utils[j];
                    child_proj[child_sup].prefix_util = u;
                    child_util += u;
                    child_inv_sum += db->invlen_table[tid];
                    child_sup++;
                    break;
                } else if (tr->ranks[j] > r) {
                    break;
                }
            }
        }

        ctx->arena.top = child_offset + (size_t)child_sup;

        prefix_ranks[prefix_len] = r;
        prefix_items[prefix_len] = db->items[r].item;

        double child_aocc = (double)(prefix_len + 1) * child_inv_sum / (double)child_sup;

        if (child_util + MHOUI_EPS >= ctx->minutil && child_aocc + MHOUI_EPS >= ctx->minocc) {
            pattern_add(ctx->houi, prefix_items, prefix_len + 1, child_sup, child_util, child_aocc);
            ctx->stats->houi_count = ctx->houi->count;
            if (ctx->max_patterns > 0 && ctx->houi->count >= ctx->max_patterns) {
                ctx->stats->status_limited = 1;
            }
            if (ctx->direct && !ctx->disable_dominance) {
                pcache_add(&ctx->pcache, prefix_items, prefix_len + 1, child_util, child_aocc);
            }
        }

        if (!ctx->stats->status_limited && i + 1 < cand_count) {
            proj_dfs(ctx, depth + 1, prefix_items, prefix_ranks, prefix_len + 1, child_offset, child_sup, child_util);
        }

        arena_reset_to(&ctx->arena, child_offset);
    }
}

/* ── Superset Dominance Filter (Bloom & Two-Pointer) ──────────────────────── */
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

typedef struct {
    MHOUPattern pat;
    uint64_t mask;
} FilterPatternEntry;

static void filter_mhoui(const MHOUPatternList *houi, MHOUPatternList *out, int strong, int *removed) {
    mhoui_pattern_list_init(out);
    *removed = 0;
    if (!houi || houi->count == 0) return;
    FilterPatternEntry *sorted = (FilterPatternEntry *)malloc((size_t)houi->count * sizeof(FilterPatternEntry));
    if (!sorted) return;
    for (int i = 0; i < houi->count; i++) {
        sorted[i].pat = houi->patterns[i];
        uint64_t m = 0;
        for (int k = 0; k < houi->patterns[i].length; k++) {
            m |= UINT64_C(1) << (houi->patterns[i].items[k] & 63);
        }
        sorted[i].mask = m;
    }
    qsort(sorted, (size_t)houi->count, sizeof(FilterPatternEntry), cmp_pattern_filter);
    for (int i = 0; i < houi->count; i++) {
        int dominated = 0;
        uint64_t mask_i = sorted[i].mask;
        const MHOUPattern *pi = &sorted[i].pat;
        for (int j = 0; j < i; j++) {
            if ((mask_i & ~sorted[j].mask) != 0) continue;
            if (!proper_subset(pi, &sorted[j].pat)) continue;
            int ge_occ = sorted[j].pat.avg_occupancy + MHOUI_EPS >= pi->avg_occupancy;
            int ge_util = sorted[j].pat.utility + MHOUI_EPS >= pi->utility;
            int strict = sorted[j].pat.avg_occupancy > pi->avg_occupancy + MHOUI_EPS || sorted[j].pat.utility > pi->utility + MHOUI_EPS;
            if (ge_occ && ge_util && (strong || strict)) {
                dominated = 1;
                break;
            }
        }
        if (dominated) (*removed)++;
        else pattern_add(out, pi->items, pi->length, pi->support, pi->utility, pi->avg_occupancy);
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

    int max_depth = 64;
    for (int i = 0; i < db.transaction_count; i++) {
        if (db.transactions[i].length + 2 > max_depth) {
            max_depth = db.transactions[i].length + 2;
        }
    }
    if (max_depth > db.item_count + 2) max_depth = db.item_count + 2;

    uint32_t *prefix_items = (uint32_t *)malloc((size_t)max_depth * sizeof(uint32_t));
    int *prefix_ranks = (int *)malloc((size_t)max_depth * sizeof(int));

    MHOUIContext ctx = {
        .db = &db,
        .minsup = minsup_count,
        .minocc = minocc,
        .minutil = minutil,
        .max_patterns = limits ? limits->max_patterns : 200000,
        .max_seconds = limits ? limits->max_seconds : 0.0,
        .direct = limits ? limits->direct : 0,
        .strong_direct = limits ? limits->strong_direct : 1,
        .disable_uub = limits ? limits->disable_uub : 0,
        .disable_oub = limits ? limits->disable_oub : 0,
        .disable_dominance = limits ? limits->disable_dominance : 0,
        .max_depth = max_depth,
        .houi = &out->houi,
        .stats = &out->stats
    };

    pcache_init(&ctx.pcache);
    arena_init(&ctx.arena, (size_t)db.transaction_count > 0 ? (size_t)db.transaction_count * 2 : 1048576);
    ws_init(&ctx.ws, db.item_count, max_depth);

    clock_gettime(CLOCK_MONOTONIC, &ctx.start);

    /* Explore from root single items */
    for (int r = 0; r < db.item_count; r++) {
        if (ctx.stats->status_limited) break;

        int sup = db.items[r].support;
        if (sup < minsup_count) {
            ctx.stats->pruned_support++;
            continue;
        }
        if (db.items[r].twu + MHOUI_EPS < minutil) {
            ctx.stats->pruned_twu++;
            continue;
        }

        ctx.stats->visited_nodes++;

        /* Build root projection for single item r */
        const ItemOcc *occs = &db.all_occs[db.items[r].occ_offset];
        size_t arena_save = ctx.arena.top;
        ProjTrans *proj = arena_alloc(&ctx.arena, (size_t)sup);
        if (!proj) {
            ctx.stats->status_limited = 1;
            break;
        }

        double single_util = 0.0;
        double inv_sum = 0.0;
        for (int i = 0; i < sup; i++) {
            int tid = occs[i].tid;
            int off = occs[i].offset;
            const CompactTransaction *tr = &db.transactions[tid];
            proj[i].tid = tid;
            proj[i].offset = off + 1;
            proj[i].prefix_util = tr->utils[off];
            single_util += tr->utils[off];
            inv_sum += db.invlen_table[tid];
        }

        double single_aocc = (1.0 * inv_sum) / (double)sup;

        prefix_ranks[0] = r;
        prefix_items[0] = db.items[r].item;

        if (single_util + MHOUI_EPS >= minutil && single_aocc + MHOUI_EPS >= minocc) {
            pattern_add(ctx.houi, prefix_items, 1, sup, single_util, single_aocc);
            ctx.stats->houi_count = ctx.houi->count;
            if (ctx.direct && !ctx.disable_dominance) {
                pcache_add(&ctx.pcache, prefix_items, 1, single_util, single_aocc);
            }
        }

        if (!ctx.stats->status_limited && r + 1 < db.item_count) {
            proj_dfs(&ctx, 1, prefix_items, prefix_ranks, 1, arena_save, sup, single_util);
        }

        arena_reset_to(&ctx.arena, arena_save);
    }

    if (limits && limits->direct) {
        if (limits->strong_direct) {
            filter_mhoui(&out->houi, &out->strong_mhoui, 1, &out->stats.strong_removed);
            out->stats.strong_count = out->strong_mhoui.count;
        } else {
            filter_mhoui(&out->houi, &out->weak_mhoui, 0, &out->stats.weak_removed);
            out->stats.weak_count = out->weak_mhoui.count;
        }
    } else {
        filter_mhoui(&out->houi, &out->weak_mhoui, 0, &out->stats.weak_removed);
        filter_mhoui(&out->houi, &out->strong_mhoui, 1, &out->stats.strong_removed);
        out->stats.weak_count = out->weak_mhoui.count;
        out->stats.strong_count = out->strong_mhoui.count;
    }
    out->stats.houi_count = out->houi.count;

    pcache_free(&ctx.pcache);
    arena_free(&ctx.arena);
    ws_free(&ctx.ws);
    free(prefix_items);
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
    .description = "Exact mining of maximal high-occupancy utility itemsets with pure projection and branch Pareto dominance",
    .supported_types = (1u << DM_TYPE_UTILITY),
    .run = run_mhoui
};
