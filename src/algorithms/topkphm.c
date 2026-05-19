#include "algorithms/topkphm.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint32_t item;
    double util;
} ItemUtil;

typedef struct {
    ItemUtil *items;
    size_t count;
    double tu;
} Transaction;

typedef struct {
    size_t support;
    double twu;
    size_t last_tid;
    size_t max_period;
} ItemStat;

typedef struct {
    int tid;
    double iutil;
    double rutil;
} PUEntry;

typedef struct {
    uint32_t item;
    PUEntry *entries;
    size_t count;
    double sum_iutil;
    double sum_rutil;
    size_t max_period;
    double avg_period;
} PUList;

typedef struct {
    uint32_t *items;
    size_t len;
    double utility;
} Pattern;

typedef struct {
    Pattern *data;
    size_t count;
    size_t cap;
    size_t k;
    double threshold;
    size_t threshold_raises;
} TopK;

typedef struct {
    uint64_t key;
    double twu;
    size_t support;
    int used;
} PairSlot;

typedef struct {
    PairSlot *slots;
    size_t cap;
    size_t count;
} PairMap;

typedef struct {
    Transaction *transactions;
    size_t ntransactions;
    uint32_t max_item;
    ItemStat *stats_by_id;
    uint32_t *kept_ids;
    uint32_t *rank_by_id;
    size_t kept_count;
    PUList **lists;
    PairMap euscs;
} Database;

typedef struct {
    DM_TOPKPHM_Params params;
    DM_TOPKPHM_Stats *stats;
    TopK topk;
    PairMap *euscs;
    clock_t start_clock;
} Ctx;

static const ItemStat *g_sort_stats = NULL;
static const uint32_t *g_sort_rank = NULL;

static double elapsed_sec(Ctx *ctx) {
    return (double)(clock() - ctx->start_clock) / (double)CLOCKS_PER_SEC;
}

static int limited(Ctx *ctx) {
    if (ctx->params.max_seconds > 0.0 && elapsed_sec(ctx) >= ctx->params.max_seconds) {
        ctx->stats->limited = 1;
        return 1;
    }
    if (ctx->params.max_candidates > 0 && ctx->stats->candidates >= ctx->params.max_candidates) {
        ctx->stats->limited = 1;
        return 1;
    }
    return 0;
}

static uint64_t pair_key(uint32_t a, uint32_t b) {
    if (a > b) {
        uint32_t t = a;
        a = b;
        b = t;
    }
    return ((uint64_t)a << 32) | (uint64_t)b;
}

static int pairmap_init(PairMap *m, size_t want) {
    size_t cap = 1024;
    while (cap < want * 2) cap <<= 1;
    m->slots = (PairSlot *)calloc(cap, sizeof(*m->slots));
    if (!m->slots) return -1;
    m->cap = cap;
    return 0;
}

static void pairmap_free(PairMap *m) {
    free(m->slots);
    memset(m, 0, sizeof(*m));
}

static PairSlot *pairmap_get_slot(PairMap *m, uint64_t key) {
    size_t mask = m->cap - 1;
    size_t pos = (size_t)((key * 11400714819323198485ull) & mask);
    while (m->slots[pos].used && m->slots[pos].key != key) pos = (pos + 1) & mask;
    return &m->slots[pos];
}

static int pairmap_rehash(PairMap *m) {
    PairMap n;
    memset(&n, 0, sizeof(n));
    if (pairmap_init(&n, m->cap * 2) != 0) return -1;
    for (size_t i = 0; i < m->cap; i++) {
        if (!m->slots[i].used) continue;
        PairSlot *s = pairmap_get_slot(&n, m->slots[i].key);
        *s = m->slots[i];
        n.count++;
    }
    free(m->slots);
    *m = n;
    return 0;
}

static int pairmap_add(PairMap *m, uint32_t a, uint32_t b, double tu) {
    if ((m->count + 1) * 10 >= m->cap * 7 && pairmap_rehash(m) != 0) return -1;
    PairSlot *s = pairmap_get_slot(m, pair_key(a, b));
    if (!s->used) {
        s->used = 1;
        s->key = pair_key(a, b);
        m->count++;
    }
    s->twu += tu;
    s->support++;
    return 0;
}

static PairSlot *pairmap_find(PairMap *m, uint32_t a, uint32_t b) {
    if (!m->slots) return NULL;
    PairSlot *s = pairmap_get_slot(m, pair_key(a, b));
    return s->used ? s : NULL;
}

static int parse_utility_line(char *line, Transaction *tr, uint32_t *max_item) {
    char *first = strchr(line, ':');
    if (!first) return 0;
    char *second = strchr(first + 1, ':');
    if (!second) return 0;
    *first = '\0';
    *second = '\0';
    tr->tu = atof(first + 1);
    size_t item_cap = 0;
    for (char *p = line; *p;) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        char *end = NULL;
        unsigned long v = strtoul(p, &end, 10);
        if (end == p) break;
        if (tr->count == item_cap) {
            size_t nc = item_cap ? item_cap * 2 : 16;
            ItemUtil *ni = (ItemUtil *)realloc(tr->items, nc * sizeof(*ni));
            if (!ni) return -1;
            tr->items = ni;
            item_cap = nc;
        }
        tr->items[tr->count].item = (uint32_t)v;
        tr->items[tr->count].util = 0.0;
        if (v > *max_item) *max_item = (uint32_t)v;
        tr->count++;
        p = end;
    }
    size_t pos = 0;
    for (char *p = second + 1; *p && pos < tr->count;) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        char *end = NULL;
        double u = strtod(p, &end);
        if (end == p) break;
        tr->items[pos++].util = u;
        p = end;
    }
    return 0;
}

static void database_free(Database *db) {
    for (size_t i = 0; i < db->ntransactions; i++) free(db->transactions[i].items);
    if (db->lists) {
        for (size_t i = 0; i < db->kept_count; i++) {
            if (db->lists[i]) {
                free(db->lists[i]->entries);
                free(db->lists[i]);
            }
        }
    }
    free(db->transactions);
    free(db->stats_by_id);
    free(db->kept_ids);
    free(db->rank_by_id);
    free(db->lists);
    pairmap_free(&db->euscs);
    memset(db, 0, sizeof(*db));
}

static int load_transactions(const char *path, Database *db) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char line[1 << 16];
    while (fgets(line, sizeof(line), fp)) {
        Transaction tr;
        memset(&tr, 0, sizeof(tr));
        if (parse_utility_line(line, &tr, &db->max_item) != 0) {
            free(tr.items);
            fclose(fp);
            return -1;
        }
        if (tr.count == 0) {
            free(tr.items);
            continue;
        }
        Transaction *nt = (Transaction *)realloc(db->transactions, (db->ntransactions + 1) * sizeof(*nt));
        if (!nt) {
            free(tr.items);
            fclose(fp);
            return -1;
        }
        db->transactions = nt;
        db->transactions[db->ntransactions++] = tr;
    }
    fclose(fp);
    return 0;
}

static int cmp_kept_by_twu(const void *a, const void *b) {
    const ItemStat *stats = g_sort_stats;
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    if (stats[x].twu < stats[y].twu) return -1;
    if (stats[x].twu > stats[y].twu) return 1;
    return (x > y) - (x < y);
}

static int cmp_trans_rank(const void *a, const void *b) {
    const uint32_t *rank = g_sort_rank;
    const ItemUtil *x = (const ItemUtil *)a;
    const ItemUtil *y = (const ItemUtil *)b;
    return (rank[x->item] > rank[y->item]) - (rank[x->item] < rank[y->item]);
}

static int preprocess(Database *db, const DM_TOPKPHM_Params *params, DM_TOPKPHM_Stats *out) {
    db->stats_by_id = (ItemStat *)calloc((size_t)db->max_item + 1, sizeof(*db->stats_by_id));
    if (!db->stats_by_id && db->max_item > 0) return -1;
    for (size_t t = 0; t < db->ntransactions; t++) {
        size_t tid = t + 1;
        for (size_t i = 0; i < db->transactions[t].count; i++) {
            uint32_t item = db->transactions[t].items[i].item;
            ItemStat *s = &db->stats_by_id[item];
            size_t gap = s->last_tid ? tid - s->last_tid : tid;
            if (gap > s->max_period) s->max_period = gap;
            s->last_tid = tid;
            s->support++;
            s->twu += db->transactions[t].tu;
        }
    }
    for (uint32_t item = 0; item <= db->max_item; item++) {
        ItemStat *s = &db->stats_by_id[item];
        if (s->support == 0) continue;
        out->distinct_items++;
        size_t tail = (db->ntransactions + 1) - s->last_tid;
        if (tail > s->max_period) s->max_period = tail;
    }
    out->support_threshold = 0;
    if (params->max_avg_period > 0.0) {
        double raw = (double)db->ntransactions / params->max_avg_period - 1.0;
        out->support_threshold = raw > 0.0 ? (size_t)raw : 0;
    }
    for (uint32_t item = 0; item <= db->max_item; item++) {
        ItemStat *s = &db->stats_by_id[item];
        if (s->support == 0) continue;
        if (s->support >= out->support_threshold && s->max_period <= params->max_period) {
            uint32_t *ni = (uint32_t *)realloc(db->kept_ids, (db->kept_count + 1) * sizeof(*ni));
            if (!ni) return -1;
            db->kept_ids = ni;
            db->kept_ids[db->kept_count++] = item;
        } else {
            if (s->support < out->support_threshold) out->pruned_support++;
            else out->pruned_periodicity++;
        }
    }
    g_sort_stats = db->stats_by_id;
    qsort(db->kept_ids, db->kept_count, sizeof(*db->kept_ids), cmp_kept_by_twu);
    g_sort_stats = NULL;
    db->rank_by_id = (uint32_t *)malloc(((size_t)db->max_item + 1) * sizeof(*db->rank_by_id));
    if (!db->rank_by_id && db->max_item > 0) return -1;
    for (uint32_t item = 0; item <= db->max_item; item++) db->rank_by_id[item] = UINT32_MAX;
    for (size_t i = 0; i < db->kept_count; i++) db->rank_by_id[db->kept_ids[i]] = (uint32_t)i;
    return 0;
}

static int build_lists(Database *db) {
    db->lists = (PUList **)calloc(db->kept_count, sizeof(*db->lists));
    if (!db->lists && db->kept_count > 0) return -1;
    for (size_t i = 0; i < db->kept_count; i++) {
        db->lists[i] = (PUList *)calloc(1, sizeof(PUList));
        if (!db->lists[i]) return -1;
        db->lists[i]->item = db->kept_ids[i];
        db->lists[i]->entries = (PUEntry *)malloc(db->stats_by_id[db->kept_ids[i]].support * sizeof(PUEntry));
        if (!db->lists[i]->entries && db->stats_by_id[db->kept_ids[i]].support > 0) return -1;
    }
    if (pairmap_init(&db->euscs, db->kept_count * 8 + 1024) != 0) return -1;
    for (size_t t = 0; t < db->ntransactions; t++) {
        Transaction *tr = &db->transactions[t];
        size_t keep = 0;
        for (size_t i = 0; i < tr->count; i++) {
            uint32_t item = tr->items[i].item;
            if (item <= db->max_item && db->rank_by_id[item] != UINT32_MAX) tr->items[keep++] = tr->items[i];
        }
        tr->count = keep;
        g_sort_rank = db->rank_by_id;
        qsort(tr->items, tr->count, sizeof(*tr->items), cmp_trans_rank);
        g_sort_rank = NULL;
        double rutil = 0.0;
        for (size_t rev = tr->count; rev > 0; rev--) {
            size_t i = rev - 1;
            uint32_t rank = db->rank_by_id[tr->items[i].item];
            PUList *list = db->lists[rank];
            PUEntry e;
            e.tid = (int)(t + 1);
            e.iutil = tr->items[i].util;
            e.rutil = rutil;
            list->entries[list->count++] = e;
            list->sum_iutil += e.iutil;
            list->sum_rutil += e.rutil;
            rutil += tr->items[i].util;
        }
        for (size_t i = 0; i < tr->count; i++) {
            for (size_t j = i + 1; j < tr->count; j++) {
                if (pairmap_add(&db->euscs, tr->items[i].item, tr->items[j].item, tr->tu) != 0) return -1;
            }
        }
    }
    for (size_t i = 0; i < db->kept_count; i++) {
        PUList *l = db->lists[i];
        l->avg_period = db->ntransactions > 0 ? (double)db->ntransactions / (double)(l->count + 1) : 0.0;
        size_t prev = 0;
        for (size_t e = 0; e < l->count; e++) {
            size_t gap = (size_t)l->entries[e].tid - prev;
            if (gap > l->max_period) l->max_period = gap;
            prev = (size_t)l->entries[e].tid;
        }
        size_t tail = (db->ntransactions + 1) - prev;
        if (tail > l->max_period) l->max_period = tail;
    }
    return 0;
}

static int cmp_pattern_desc(const void *a, const void *b) {
    const Pattern *x = (const Pattern *)a;
    const Pattern *y = (const Pattern *)b;
    if (x->utility < y->utility) return 1;
    if (x->utility > y->utility) return -1;
    if (x->len > y->len) return 1;
    if (x->len < y->len) return -1;
    return 0;
}

static void topk_free(TopK *t) {
    for (size_t i = 0; i < t->count; i++) free(t->data[i].items);
    free(t->data);
    memset(t, 0, sizeof(*t));
}

static int topk_add(TopK *t, const uint32_t *items, size_t len, double utility) {
    if (t->k == 0) return 0;
    if (t->count >= t->k && utility < t->threshold) return 0;
    if (t->count == t->cap) {
        size_t nc = t->cap ? t->cap * 2 : 64;
        Pattern *nd = (Pattern *)realloc(t->data, nc * sizeof(*nd));
        if (!nd) return -1;
        t->data = nd;
        t->cap = nc;
    }
    t->data[t->count].items = (uint32_t *)malloc(len * sizeof(uint32_t));
    if (!t->data[t->count].items) return -1;
    memcpy(t->data[t->count].items, items, len * sizeof(uint32_t));
    t->data[t->count].len = len;
    t->data[t->count].utility = utility;
    t->count++;
    qsort(t->data, t->count, sizeof(*t->data), cmp_pattern_desc);
    if (t->count >= t->k) {
        double old = t->threshold;
        t->threshold = t->data[t->k - 1].utility;
        if (t->threshold > old) t->threshold_raises++;
        size_t keep = t->count;
        while (keep > t->k && t->data[keep - 1].utility < t->threshold) {
            free(t->data[keep - 1].items);
            keep--;
        }
        t->count = keep;
    }
    return 0;
}

static int periodic_valid(PUList *l, size_t max_period, double max_avg) {
    return l->count > 0 && l->max_period <= max_period && l->avg_period <= max_avg;
}

static PUList *construct_join(Ctx *ctx, PUList *p, PUList *x, PUList *y, size_t ntransactions) {
    PUList *out = (PUList *)calloc(1, sizeof(*out));
    if (!out) return NULL;
    out->item = y->item;
    size_t cap = x->count < y->count ? x->count : y->count;
    out->entries = (PUEntry *)malloc(cap * sizeof(PUEntry));
    if (!out->entries && cap > 0) {
        free(out);
        return NULL;
    }
    size_t ix = 0, iy = 0, ip = 0, prev_tid = 0;
    while (ix < x->count && iy < y->count) {
        if (x->entries[ix].tid < y->entries[iy].tid) {
            ix++;
        } else if (x->entries[ix].tid > y->entries[iy].tid) {
            iy++;
        } else {
            double putil = 0.0;
            if (p) {
                while (ip < p->count && p->entries[ip].tid < x->entries[ix].tid) ip++;
                if (ip < p->count && p->entries[ip].tid == x->entries[ix].tid) putil = p->entries[ip].iutil;
            }
            PUEntry e;
            e.tid = x->entries[ix].tid;
            e.iutil = x->entries[ix].iutil + y->entries[iy].iutil - putil;
            e.rutil = y->entries[iy].rutil;
            out->entries[out->count++] = e;
            out->sum_iutil += e.iutil;
            out->sum_rutil += e.rutil;
            size_t gap = (size_t)e.tid - prev_tid;
            if (gap > out->max_period) out->max_period = gap;
            prev_tid = (size_t)e.tid;
            ctx->stats->joined_entries++;
            ix++;
            iy++;
        }
    }
    size_t tail = (ntransactions + 1) - prev_tid;
    if (tail > out->max_period) out->max_period = tail;
    out->avg_period = ntransactions > 0 ? (double)ntransactions / (double)(out->count + 1) : 0.0;
    ctx->stats->joins++;
    return out;
}

static void pulist_free(PUList *l) {
    if (!l) return;
    free(l->entries);
    free(l);
}

static int recursive_mine(Ctx *ctx, PUList *prefix, uint32_t *prefix_items, size_t prefix_len, PUList **exts, size_t ext_count, size_t ntransactions) {
    for (size_t i = 0; i < ext_count; i++) {
        if (limited(ctx)) return 0;
        PUList *x = exts[i];
        uint32_t *items = (uint32_t *)malloc((prefix_len + 1) * sizeof(*items));
        if (!items) return -1;
        memcpy(items, prefix_items, prefix_len * sizeof(*items));
        items[prefix_len] = x->item;
        ctx->stats->visited_nodes++;
        if (periodic_valid(x, ctx->params.max_period, ctx->params.max_avg_period) && x->sum_iutil >= ctx->topk.threshold) {
            if (topk_add(&ctx->topk, items, prefix_len + 1, x->sum_iutil) != 0) {
                free(items);
                return -1;
            }
        } else if (!periodic_valid(x, ctx->params.max_period, ctx->params.max_avg_period)) {
            ctx->stats->pruned_periodicity++;
        }
        if (ctx->params.max_depth > 0 && prefix_len + 1 >= ctx->params.max_depth) {
            free(items);
            continue;
        }
        if (x->sum_iutil + x->sum_rutil < ctx->topk.threshold) {
            ctx->stats->pruned_subtree_utility++;
            free(items);
            continue;
        }
        PUList **new_exts = NULL;
        size_t new_count = 0;
        for (size_t j = i + 1; j < ext_count; j++) {
            ctx->stats->candidates++;
            PairSlot *pair = pairmap_find(ctx->euscs, x->item, exts[j]->item);
            if (!pair || pair->twu < ctx->topk.threshold) {
                ctx->stats->pruned_twu++;
                continue;
            }
            PUList *xy = construct_join(ctx, prefix, x, exts[j], ntransactions);
            if (!xy) {
                free(items);
                for (size_t z = 0; z < new_count; z++) pulist_free(new_exts[z]);
                free(new_exts);
                return -1;
            }
            if (xy->count == 0 || !periodic_valid(xy, ctx->params.max_period, ctx->params.max_avg_period)) {
                ctx->stats->pruned_periodicity++;
                pulist_free(xy);
                continue;
            }
            PUList **ne = (PUList **)realloc(new_exts, (new_count + 1) * sizeof(*ne));
            if (!ne) {
                pulist_free(xy);
                free(items);
                for (size_t z = 0; z < new_count; z++) pulist_free(new_exts[z]);
                free(new_exts);
                return -1;
            }
            new_exts = ne;
            new_exts[new_count++] = xy;
        }
        if (new_count > 0 && recursive_mine(ctx, x, items, prefix_len + 1, new_exts, new_count, ntransactions) != 0) {
            free(items);
            for (size_t z = 0; z < new_count; z++) pulist_free(new_exts[z]);
            free(new_exts);
            return -1;
        }
        for (size_t z = 0; z < new_count; z++) pulist_free(new_exts[z]);
        free(new_exts);
        free(items);
    }
    return 0;
}

int topkphm_mine_file(const char *path, const DM_TOPKPHM_Params *params, DM_TOPKPHM_Stats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (!path || !params || params->k == 0 || params->max_period == 0 || params->max_avg_period <= 0.0) return -1;
    Database db;
    memset(&db, 0, sizeof(db));
    if (load_transactions(path, &db) != 0) {
        database_free(&db);
        return -1;
    }
    stats->transactions = db.ntransactions;
    stats->k = params->k;
    stats->max_period = params->max_period;
    stats->max_avg_period = params->max_avg_period;
    if (preprocess(&db, params, stats) != 0 || build_lists(&db) != 0) {
        database_free(&db);
        return -1;
    }
    stats->kept_items = db.kept_count;
    stats->euscs_pairs = db.euscs.count;

    Ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.params = *params;
    ctx.stats = stats;
    ctx.topk.k = params->k;
    ctx.euscs = &db.euscs;
    ctx.start_clock = clock();
    uint32_t *empty = NULL;
    int rc = recursive_mine(&ctx, NULL, empty, 0, db.lists, db.kept_count, db.ntransactions);
    if (rc == 0) {
        stats->final_threshold = ctx.topk.threshold;
        stats->threshold_raises = ctx.topk.threshold_raises;
        stats->output_count = ctx.topk.count;
        for (size_t i = 0; i < ctx.topk.count; i++) {
            stats->total_output_items += ctx.topk.data[i].len;
            stats->avg_utility += ctx.topk.data[i].utility;
            if (i == 0 || ctx.topk.data[i].utility > stats->best_utility) stats->best_utility = ctx.topk.data[i].utility;
        }
        if (ctx.topk.count > 0) {
            stats->avg_utility /= (double)ctx.topk.count;
            stats->avg_output_length = (double)stats->total_output_items / (double)ctx.topk.count;
        }
        stats->result_ram_bytes = ctx.topk.count * sizeof(Pattern) + stats->total_output_items * sizeof(uint32_t);
        stats->result_disk_est_bytes = stats->total_output_items * 12 + ctx.topk.count * 32;
    }
    topk_free(&ctx.topk);
    database_free(&db);
    return rc;
}
