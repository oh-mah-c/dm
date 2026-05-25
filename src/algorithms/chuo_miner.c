#include "algorithms/chuo_miner.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _MSC_VER
#include <intrin.h>
#define dm_popcount64(x) ((uint32_t)__popcnt64((unsigned __int64)(x)))
#else
#define dm_popcount64(x) ((uint32_t)__builtin_popcountll(x))
#endif
#include <time.h>

typedef struct {
    uint32_t tid;
    uint32_t pos;
    double iu;
    double ru;
    uint32_t rc;
    double invlen;
} CHUOEntry;

typedef struct {
    CHUOEntry *entries;
    size_t count;
    size_t cap;
} CHUOList;

typedef struct {
    uint32_t *items;
    double *utils;
    size_t count;
    double total_utility;
} CHUOTrans;

typedef struct {
    uint32_t id;
    uint32_t support;
    double twu;
    uint32_t rank;
    uint64_t *bitset;
} CHUOItemInfo;

typedef struct {
    CHUOTrans *tx;
    size_t tx_count;
    uint32_t max_item;
    CHUOItemInfo *info;
    uint32_t *rank_to_item;
    uint32_t *item_to_rank;
    size_t item_count;
    size_t words;
} CHUODB;

typedef struct HashNode {
    uint64_t h1;
    uint64_t h2;
    uint32_t support;
    struct HashNode *next;
} HashNode;

typedef struct {
    HashNode **buckets;
    size_t bucket_count;
} ClosureHash;

typedef struct {
    CHUODB *db;
    ClosureHash hcl;
    uint32_t sigma;
    double mu;
    double beta;
    size_t max_patterns;
    size_t max_depth;
    double max_seconds;
    clock_t started;
    DM_CHUO_Stats *stats;
} CHUOCtx;

static int list_push(CHUOList *l, CHUOEntry e) {
    if (l->count >= l->cap) {
        size_t next = l->cap ? l->cap * 2 : 64;
        CHUOEntry *tmp = (CHUOEntry *)realloc(l->entries, next * sizeof(CHUOEntry));
        if (!tmp) return -1;
        l->entries = tmp;
        l->cap = next;
    }
    l->entries[l->count++] = e;
    return 0;
}

static void list_free(CHUOList *l) {
    free(l->entries);
    l->entries = NULL;
    l->count = l->cap = 0;
}

static void bitset_set(uint64_t *b, uint32_t tid) {
    b[tid >> 6] |= 1ULL << (tid & 63);
}

static uint32_t bitset_intersection_count(const uint64_t *a, const uint64_t *b, size_t words) {
    uint32_t c = 0;
    for (size_t i = 0; i < words; i++) c += dm_popcount64(a[i] & b[i]);
    return c;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

static void bitset_fingerprint(const uint64_t *b, size_t words, uint64_t *h1, uint64_t *h2) {
    uint64_t a = 1469598103934665603ULL;
    uint64_t c = 1099511628211ULL;
    for (size_t i = 0; i < words; i++) {
        uint64_t x = b[i] + 0x9e3779b97f4a7c15ULL + (i << 6);
        a ^= x;
        a *= 1099511628211ULL;
        c ^= mix64(x);
        c *= 0x100000001b3ULL;
    }
    *h1 = a;
    *h2 = c;
}

static int hash_init(ClosureHash *h, size_t buckets) {
    h->bucket_count = buckets ? buckets : 4096;
    h->buckets = (HashNode **)calloc(h->bucket_count, sizeof(HashNode *));
    return h->buckets ? 0 : -1;
}

static void hash_free(ClosureHash *h) {
    if (!h || !h->buckets) return;
    for (size_t i = 0; i < h->bucket_count; i++) {
        HashNode *n = h->buckets[i];
        while (n) {
            HashNode *next = n->next;
            free(n);
            n = next;
        }
    }
    free(h->buckets);
    h->buckets = NULL;
}

static int hash_contains_or_add(ClosureHash *h, uint64_t h1, uint64_t h2, uint32_t support, int add) {
    size_t idx = (size_t)(h1 ^ (h2 << 1)) % h->bucket_count;
    for (HashNode *n = h->buckets[idx]; n; n = n->next) {
        if (n->h1 == h1 && n->h2 == h2 && n->support == support) return 1;
    }
    if (!add) return 0;
    HashNode *node = (HashNode *)malloc(sizeof(HashNode));
    if (!node) return -1;
    node->h1 = h1;
    node->h2 = h2;
    node->support = support;
    node->next = h->buckets[idx];
    h->buckets[idx] = node;
    return 0;
}

static int order_cmp(const void *a, const void *b) {
    const CHUOItemInfo *ia = (const CHUOItemInfo *)a;
    const CHUOItemInfo *ib = (const CHUOItemInfo *)b;
    if (ia->support < ib->support) return -1;
    if (ia->support > ib->support) return 1;
    if (ia->twu < ib->twu) return -1;
    if (ia->twu > ib->twu) return 1;
    return ia->id < ib->id ? -1 : (ia->id > ib->id);
}

static void sort_trans_by_rank(CHUODB *db, CHUOTrans *t) {
    for (size_t i = 1; i < t->count; i++) {
        uint32_t item = t->items[i];
        double util = t->utils[i];
        uint32_t rank = db->item_to_rank[item];
        size_t j = i;
        while (j > 0 && db->item_to_rank[t->items[j - 1]] > rank) {
            t->items[j] = t->items[j - 1];
            t->utils[j] = t->utils[j - 1];
            j--;
        }
        t->items[j] = item;
        t->utils[j] = util;
    }
}

static int build_db(DM_Dataset *ds, const DM_CHUO_Params *params, CHUODB *db, DM_CHUO_Stats *stats) {
    if (!ds || ds->type != DM_TYPE_UTILITY || !ds->payload) return -1;
    memset(db, 0, sizeof(*db));
    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    db->tx_count = ds->count;
    db->max_item = ds->max_id;
    db->words = (ds->count + 63) / 64;
    db->info = (CHUOItemInfo *)calloc((size_t)db->max_item + 1, sizeof(CHUOItemInfo));
    db->item_to_rank = (uint32_t *)malloc(((size_t)db->max_item + 1) * sizeof(uint32_t));
    db->tx = (CHUOTrans *)calloc(ds->count, sizeof(CHUOTrans));
    if (!db->info || !db->item_to_rank || !db->tx) return -1;
    for (uint32_t i = 0; i <= db->max_item; i++) {
        db->info[i].id = i;
        db->info[i].rank = UINT32_MAX;
        db->item_to_rank[i] = UINT32_MAX;
    }

    uint8_t *seen = (uint8_t *)calloc((size_t)db->max_item + 1, 1);
    if (!seen) return -1;
    for (size_t t = 0; t < ds->count; t++) {
        memset(seen, 0, (size_t)db->max_item + 1);
        double tu = src[t].total_utility > 0.0 ? src[t].total_utility : 0.0;
        if (tu <= 0.0) {
            for (size_t k = 0; k < src[t].count; k++) if (src[t].items[k].utility > 0.0) tu += src[t].items[k].utility;
        }
        for (size_t k = 0; k < src[t].count; k++) {
            uint32_t id = src[t].items[k].id;
            if (id > db->max_item || seen[id]) continue;
            seen[id] = 1;
            db->info[id].support++;
            db->info[id].twu += tu;
        }
    }
    free(seen);

    CHUOItemInfo *surv = (CHUOItemInfo *)malloc(((size_t)db->max_item + 1) * sizeof(CHUOItemInfo));
    if (!surv) return -1;
    for (uint32_t id = 0; id <= db->max_item; id++) {
        if (db->info[id].support >= params->min_support && db->info[id].twu >= params->min_utility) {
            surv[db->item_count++] = db->info[id];
        } else if (db->info[id].support > 0) {
            stats->pruned_twu++;
        }
    }
    qsort(surv, db->item_count, sizeof(CHUOItemInfo), order_cmp);
    db->rank_to_item = (uint32_t *)malloc(db->item_count * sizeof(uint32_t));
    if (!db->rank_to_item) {
        free(surv);
        return -1;
    }
    for (size_t r = 0; r < db->item_count; r++) {
        uint32_t id = surv[r].id;
        db->rank_to_item[r] = id;
        db->item_to_rank[id] = (uint32_t)r;
        db->info[id] = surv[r];
        db->info[id].rank = (uint32_t)r;
        db->info[id].bitset = (uint64_t *)calloc(db->words, sizeof(uint64_t));
        if (!db->info[id].bitset) {
            free(surv);
            return -1;
        }
    }
    free(surv);

    for (size_t t = 0; t < ds->count; t++) {
        CHUOTrans *dst = &db->tx[t];
        dst->items = (uint32_t *)malloc(src[t].count * sizeof(uint32_t));
        dst->utils = (double *)malloc(src[t].count * sizeof(double));
        if (!dst->items || !dst->utils) return -1;
        dst->total_utility = src[t].total_utility;
        for (size_t k = 0; k < src[t].count; k++) {
            uint32_t id = src[t].items[k].id;
            double u = src[t].items[k].utility;
            if (id <= db->max_item && db->item_to_rank[id] != UINT32_MAX && u > 0.0) {
                dst->items[dst->count] = id;
                dst->utils[dst->count] = u;
                dst->count++;
                bitset_set(db->info[id].bitset, (uint32_t)t);
            }
        }
        if (dst->count > 1) sort_trans_by_rank(db, dst);
    }

    stats->transactions = ds->count;
    stats->max_item_id = ds->max_id;
    stats->surviving_items = db->item_count;
    return 0;
}

static void free_db(CHUODB *db) {
    if (!db) return;
    for (size_t t = 0; t < db->tx_count; t++) {
        free(db->tx[t].items);
        free(db->tx[t].utils);
    }
    if (db->info) {
        for (uint32_t id = 0; id <= db->max_item; id++) free(db->info[id].bitset);
    }
    free(db->tx);
    free(db->info);
    free(db->rank_to_item);
    free(db->item_to_rank);
}

static int find_after(const CHUOTrans *t, uint32_t pos, uint32_t item, uint32_t *found) {
    for (uint32_t p = pos + 1; p < t->count; p++) {
        if (t->items[p] == item) {
            *found = p;
            return 1;
        }
    }
    return 0;
}

static void suffix_stats(const CHUOTrans *t, uint32_t pos, double *ru, uint32_t *rc) {
    double s = 0.0;
    uint32_t c = 0;
    for (uint32_t p = pos + 1; p < t->count; p++) {
        s += t->utils[p];
        c++;
    }
    *ru = s;
    *rc = c;
}

static int build_singleton_list(CHUODB *db, uint32_t item, CHUOList *out) {
    memset(out, 0, sizeof(*out));
    for (size_t tid = 0; tid < db->tx_count; tid++) {
        CHUOTrans *t = &db->tx[tid];
        for (uint32_t p = 0; p < t->count; p++) {
            if (t->items[p] != item) continue;
            CHUOEntry e;
            e.tid = (uint32_t)tid;
            e.pos = p;
            e.iu = t->utils[p];
            suffix_stats(t, p, &e.ru, &e.rc);
            e.invlen = t->count ? 1.0 / (double)t->count : 0.0;
            if (list_push(out, e) != 0) return -1;
            break;
        }
    }
    return 0;
}

static int join_list(CHUODB *db, const CHUOList *prefix, uint32_t item, CHUOList *out) {
    memset(out, 0, sizeof(*out));
    for (size_t i = 0; i < prefix->count; i++) {
        const CHUOEntry *e = &prefix->entries[i];
        CHUOTrans *t = &db->tx[e->tid];
        uint32_t pos;
        if (!find_after(t, e->pos, item, &pos)) continue;
        CHUOEntry ne;
        ne.tid = e->tid;
        ne.pos = pos;
        ne.iu = e->iu + t->utils[pos];
        suffix_stats(t, pos, &ne.ru, &ne.rc);
        ne.invlen = e->invlen;
        if (list_push(out, ne) != 0) return -1;
    }
    return 0;
}

static void list_stats(const CHUOList *l, size_t itemset_len, double *utility, double *ruu, double *sinv, double *aocc) {
    double u = 0.0, r = 0.0, si = 0.0;
    for (size_t i = 0; i < l->count; i++) {
        u += l->entries[i].iu;
        r += l->entries[i].iu + l->entries[i].ru;
        si += l->entries[i].invlen;
    }
    *utility = u;
    *ruu = r;
    *sinv = si;
    *aocc = l->count ? (double)itemset_len * si / (double)l->count : 0.0;
}

static int double_desc_cmp(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return x > y ? -1 : (x < y);
}

static double compute_oub(const CHUOList *l, size_t itemset_len, uint32_t sigma) {
    if (l->count < sigma || sigma == 0) return 0.0;
    double *vals = (double *)malloc(l->count * sizeof(double));
    if (!vals) return 0.0;
    for (size_t i = 0; i < l->count; i++) {
        vals[i] = ((double)itemset_len + (double)l->entries[i].rc) * l->entries[i].invlen;
    }
    qsort(vals, l->count, sizeof(double), double_desc_cmp);
    double sum = 0.0;
    for (uint32_t i = 0; i < sigma; i++) sum += vals[i];
    free(vals);
    return sum / (double)sigma;
}

static uint64_t *bitset_from_list(CHUODB *db, const CHUOList *l) {
    uint64_t *b = (uint64_t *)calloc(db->words, sizeof(uint64_t));
    if (!b) return NULL;
    for (size_t i = 0; i < l->count; i++) bitset_set(b, l->entries[i].tid);
    return b;
}

static int item_in_set(const uint32_t *items, size_t len, uint32_t item) {
    for (size_t i = 0; i < len; i++) if (items[i] == item) return 1;
    return 0;
}

static int has_backward_extension(CHUOCtx *ctx, const uint32_t *items, size_t len, const uint64_t *bitset, uint32_t support) {
    if (len == 0) return 0;
    uint32_t last_rank = ctx->db->item_to_rank[items[len - 1]];
    for (uint32_t r = 0; r < last_rank; r++) {
        uint32_t z = ctx->db->rank_to_item[r];
        if (item_in_set(items, len, z)) continue;
        if (bitset_intersection_count(bitset, ctx->db->info[z].bitset, ctx->db->words) == support) return 1;
    }
    return 0;
}

static int append_item(uint32_t **items, size_t *len, size_t *cap, uint32_t item) {
    if (item_in_set(*items, *len, item)) return 0;
    if (*len >= *cap) {
        size_t next = *cap ? *cap * 2 : 8;
        uint32_t *tmp = (uint32_t *)realloc(*items, next * sizeof(uint32_t));
        if (!tmp) return -1;
        *items = tmp;
        *cap = next;
    }
    (*items)[(*len)++] = item;
    return 0;
}

static int closure_jump(CHUOCtx *ctx, uint32_t **items, size_t *len, size_t *cap, CHUOList *list, const uint32_t *eq, size_t eq_count) {
    for (size_t e = 0; e < eq_count; e++) {
        if (append_item(items, len, cap, eq[e]) != 0) return -1;
    }
    for (size_t i = 0; i < list->count; i++) {
        CHUOEntry *entry = &list->entries[i];
        CHUOTrans *t = &ctx->db->tx[entry->tid];
        uint32_t max_pos = entry->pos;
        double add = 0.0;
        for (size_t e = 0; e < eq_count; e++) {
            uint32_t pos;
            if (find_after(t, entry->pos, eq[e], &pos)) {
                add += t->utils[pos];
                if (pos > max_pos) max_pos = pos;
            }
        }
        entry->iu += add;
        entry->pos = max_pos;
        suffix_stats(t, max_pos, &entry->ru, &entry->rc);
    }
    ctx->stats->closure_jumps++;
    return 0;
}

static int time_limited(CHUOCtx *ctx) {
    if (ctx->stats->limited) return 1;
    if (ctx->max_patterns && ctx->stats->emitted_chuois >= ctx->max_patterns) {
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

static void build_signature(CHUOCtx *ctx, const uint32_t *items, size_t len, const CHUOList *list) {
    ctx->stats->signatures_built++;
    ctx->stats->signature_value_records += len;
    double *vc = (double *)calloc(len ? len : 1, sizeof(double));
    if (!vc) return;
    double r_c = 0.0;
    for (size_t e = 0; e < list->count; e++) {
        const CHUOEntry *entry = &list->entries[e];
        const CHUOTrans *t = &ctx->db->tx[entry->tid];
        r_c += entry->invlen;
        for (size_t k = 0; k < len; k++) {
            for (size_t p = 0; p < t->count; p++) {
                if (t->items[p] == items[k]) {
                    vc[k] += t->utils[p];
                    break;
                }
            }
        }
    }
    if (!ctx->stats->reconstruction_checks) {
        double reconstructed_utility = 0.0;
        for (size_t k = 0; k < len; k++) reconstructed_utility += vc[k];
        double reconstructed_aocc = list->count ? (double)len * r_c / (double)list->count : 0.0;
        double exact_u, ruu, sinv, exact_a;
        list_stats(list, len, &exact_u, &ruu, &sinv, &exact_a);
        ctx->stats->reconstruction_checks++;
        if (fabs(reconstructed_utility - exact_u) > 1e-6 || fabs(reconstructed_aocc - exact_a) > 1e-9) {
            ctx->stats->reconstruction_failures++;
        }
        (void)ruu;
        (void)sinv;
    }
    free(vc);
}

static int emit_if_new(CHUOCtx *ctx, const uint32_t *items, size_t len, const CHUOList *list, const uint64_t *bitset, double utility, double aocc) {
    uint64_t h1, h2;
    bitset_fingerprint(bitset, ctx->db->words, &h1, &h2);
    int seen = hash_contains_or_add(&ctx->hcl, h1, h2, (uint32_t)list->count, 1);
    if (seen < 0) return -1;
    if (seen) {
        ctx->stats->closure_hash_hits++;
        return 0;
    }
    ctx->stats->emitted_chuois++;
    ctx->stats->total_output_items += len;
    ctx->stats->avg_support += (double)list->count;
    ctx->stats->avg_utility += utility;
    ctx->stats->avg_occupancy += aocc;
    if (utility > ctx->stats->best_utility) ctx->stats->best_utility = utility;
    if (aocc > ctx->stats->best_occupancy) ctx->stats->best_occupancy = aocc;
    build_signature(ctx, items, len, list);
    return 0;
}

static int dfs(CHUOCtx *ctx, const uint32_t *prefix, size_t prefix_len, const CHUOList *prefix_list, size_t start_rank) {
    if (time_limited(ctx)) return 0;
    if (ctx->max_depth && prefix_len >= ctx->max_depth) return 0;

    for (size_t r = start_rank; r < ctx->db->item_count; r++) {
        if (time_limited(ctx)) break;
        uint32_t item = ctx->db->rank_to_item[r];
        CHUOList y;
        int rc = prefix_len == 0 ? build_singleton_list(ctx->db, item, &y) : join_list(ctx->db, prefix_list, item, &y);
        if (rc != 0) return -1;
        ctx->stats->generated_children++;
        ctx->stats->joined_entries += y.count;
        if (y.count < ctx->sigma) {
            ctx->stats->pruned_support++;
            list_free(&y);
            continue;
        }

        size_t cap = prefix_len + 8;
        uint32_t *items = (uint32_t *)malloc(cap * sizeof(uint32_t));
        if (!items) {
            list_free(&y);
            return -1;
        }
        if (prefix_len) memcpy(items, prefix, prefix_len * sizeof(uint32_t));
        size_t len = prefix_len;
        if (append_item(&items, &len, &cap, item) != 0) {
            free(items);
            list_free(&y);
            return -1;
        }

        double utility, ruu, sinv, aocc;
        list_stats(&y, len, &utility, &ruu, &sinv, &aocc);
        if (ruu < ctx->mu) {
            ctx->stats->pruned_ruu++;
            free(items);
            list_free(&y);
            continue;
        }
        double oub = compute_oub(&y, len, ctx->sigma);
        if (oub < ctx->beta) {
            ctx->stats->pruned_oub++;
            free(items);
            list_free(&y);
            continue;
        }
        uint64_t *bitset = bitset_from_list(ctx->db, &y);
        if (!bitset) {
            free(items);
            list_free(&y);
            return -1;
        }
        if (has_backward_extension(ctx, items, len, bitset, (uint32_t)y.count)) {
            ctx->stats->pruned_backward++;
            free(bitset);
            free(items);
            list_free(&y);
            continue;
        }

        uint32_t *eq = NULL;
        size_t eq_count = 0, eq_cap = 0;
        size_t max_rank = ctx->db->item_to_rank[items[len - 1]];
        for (size_t rr = r + 1; rr < ctx->db->item_count; rr++) {
            uint32_t z = ctx->db->rank_to_item[rr];
            if (bitset_intersection_count(bitset, ctx->db->info[z].bitset, ctx->db->words) == y.count) {
                if (append_item(&eq, &eq_count, &eq_cap, z) != 0) {
                    free(eq); free(bitset); free(items); list_free(&y);
                    return -1;
                }
                max_rank = rr;
            }
        }
        if (eq_count) {
            if (closure_jump(ctx, &items, &len, &cap, &y, eq, eq_count) != 0) {
                free(eq); free(bitset); free(items); list_free(&y);
                return -1;
            }
            list_stats(&y, len, &utility, &ruu, &sinv, &aocc);
        }
        free(eq);

        ctx->stats->visited_nodes++;
        if (len > ctx->stats->max_depth_seen) ctx->stats->max_depth_seen = len;
        if (utility >= ctx->mu && aocc >= ctx->beta) {
            if (emit_if_new(ctx, items, len, &y, bitset, utility, aocc) != 0) {
                free(bitset); free(items); list_free(&y);
                return -1;
            }
        }

        if (ruu >= ctx->mu && compute_oub(&y, len, ctx->sigma) >= ctx->beta && max_rank + 1 < ctx->db->item_count) {
            if (dfs(ctx, items, len, &y, max_rank + 1) != 0) {
                free(bitset); free(items); list_free(&y);
                return -1;
            }
        }
        free(bitset);
        free(items);
        list_free(&y);
    }
    return 0;
}

int chuo_mine_dataset(DM_Dataset *ds, const DM_CHUO_Params *params, DM_CHUO_Stats *stats) {
    if (!ds || !params || !stats) return -1;
    memset(stats, 0, sizeof(*stats));
    if (params->min_support == 0 || params->min_utility <= 0.0 || params->min_occupancy <= 0.0) return -1;

    CHUODB db;
    if (build_db(ds, params, &db, stats) != 0) {
        free_db(&db);
        return -1;
    }

    CHUOCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.db = &db;
    ctx.sigma = params->min_support;
    ctx.mu = params->min_utility;
    ctx.beta = params->min_occupancy;
    ctx.max_patterns = params->max_patterns;
    ctx.max_depth = params->max_depth;
    ctx.max_seconds = params->max_seconds;
    ctx.started = clock();
    ctx.stats = stats;
    stats->min_support = params->min_support;
    stats->min_utility = params->min_utility;
    stats->min_occupancy = params->min_occupancy;

    if (hash_init(&ctx.hcl, 16384) != 0) {
        free_db(&db);
        return -1;
    }
    int rc = dfs(&ctx, NULL, 0, NULL, 0);

    if (stats->emitted_chuois) {
        stats->avg_support /= (double)stats->emitted_chuois;
        stats->avg_utility /= (double)stats->emitted_chuois;
        stats->avg_occupancy /= (double)stats->emitted_chuois;
    }
    stats->result_ram_bytes = stats->emitted_chuois * 56 + stats->total_output_items * 12 + stats->signature_value_records * 16;
    stats->result_disk_est_bytes = stats->emitted_chuois * 112 + stats->total_output_items * 16 + stats->signature_value_records * 24;

    hash_free(&ctx.hcl);
    free_db(&db);
    return rc;
}
