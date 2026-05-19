#include "algorithms/htk_miner.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint32_t *items;
    size_t count;
} Transaction;

typedef struct {
    uint32_t item;
    size_t support;
    uint64_t *bits;
} Singleton;

typedef struct {
    uint32_t *items;
    size_t len;
    size_t support;
    uint64_t *bits;
} Pattern;

typedef struct {
    Pattern *data;
    size_t count;
    size_t cap;
} Level;

typedef struct {
    Pattern *data;
    size_t count;
    size_t cap;
    size_t k;
    size_t threshold;
    size_t threshold_raises;
} QHeap;

typedef struct {
    Transaction *transactions;
    size_t transaction_count;
    size_t nonempty_transactions;
    uint32_t max_item;
    size_t *support_by_id;
    Singleton *singletons;
    size_t singleton_count;
    size_t bitset_words;
} Database;

typedef struct {
    DM_HTK_Params params;
    DM_HTK_Stats *stats;
    QHeap heap;
    clock_t start_clock;
} HTKCtx;

const char *htk_mode_name(DM_HTK_Mode mode) {
    switch (mode) {
        case DM_HTK_MODE_BSN: return "bsn";
    }
    return "unknown";
}

int htk_parse_mode(const char *name, DM_HTK_Mode *mode) {
    if (!name || strcmp(name, "bsn") == 0 || strcmp(name, "BSN") == 0) {
        *mode = DM_HTK_MODE_BSN;
        return 0;
    }
    return -1;
}

static double elapsed_sec(HTKCtx *ctx) {
    return (double)(clock() - ctx->start_clock) / (double)CLOCKS_PER_SEC;
}

static int is_limited(HTKCtx *ctx) {
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

static size_t popcount_words(const uint64_t *bits, size_t words) {
    size_t out = 0;
    for (size_t i = 0; i < words; i++) out += (size_t)__builtin_popcountll(bits[i]);
    return out;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int parse_transaction_line(char *line, Transaction *tr, uint32_t *max_item) {
    size_t cap = 0;
    char *p = line;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p || *p == '#') break;
        char *end = NULL;
        unsigned long v = strtoul(p, &end, 10);
        if (end == p) break;
        if (tr->count == cap) {
            size_t nc = cap ? cap * 2 : 16;
            uint32_t *ni = (uint32_t *)realloc(tr->items, nc * sizeof(*ni));
            if (!ni) return -1;
            tr->items = ni;
            cap = nc;
        }
        tr->items[tr->count++] = (uint32_t)v;
        if (v > *max_item) *max_item = (uint32_t)v;
        p = end;
    }
    if (tr->count > 1) {
        qsort(tr->items, tr->count, sizeof(*tr->items), cmp_u32);
        size_t keep = 0;
        for (size_t i = 0; i < tr->count; i++) {
            if (keep == 0 || tr->items[i] != tr->items[keep - 1]) tr->items[keep++] = tr->items[i];
        }
        tr->count = keep;
    }
    return 0;
}

static void database_free(Database *db) {
    if (!db) return;
    for (size_t i = 0; i < db->transaction_count; i++) free(db->transactions[i].items);
    for (size_t i = 0; i < db->singleton_count; i++) free(db->singletons[i].bits);
    free(db->transactions);
    free(db->support_by_id);
    free(db->singletons);
    memset(db, 0, sizeof(*db));
}

static int load_database(const char *path, Database *db) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char line[1 << 16];
    while (fgets(line, sizeof(line), fp)) {
        if (db->transaction_count == db->nonempty_transactions + 1024) {
            ;
        }
        Transaction tr;
        memset(&tr, 0, sizeof(tr));
        if (parse_transaction_line(line, &tr, &db->max_item) != 0) {
            free(tr.items);
            fclose(fp);
            return -1;
        }
        Transaction *nt = (Transaction *)realloc(db->transactions, (db->transaction_count + 1) * sizeof(*nt));
        if (!nt) {
            free(tr.items);
            fclose(fp);
            return -1;
        }
        db->transactions = nt;
        db->transactions[db->transaction_count++] = tr;
        if (tr.count > 0) db->nonempty_transactions++;
    }
    fclose(fp);

    db->support_by_id = (size_t *)calloc((size_t)db->max_item + 1, sizeof(*db->support_by_id));
    if (!db->support_by_id && db->max_item > 0) return -1;
    for (size_t t = 0; t < db->transaction_count; t++) {
        for (size_t i = 0; i < db->transactions[t].count; i++) {
            db->support_by_id[db->transactions[t].items[i]]++;
        }
    }
    for (uint32_t id = 0; id <= db->max_item; id++) {
        if (db->support_by_id[id] > 0) db->singleton_count++;
    }
    db->singletons = (Singleton *)calloc(db->singleton_count, sizeof(*db->singletons));
    if (!db->singletons && db->singleton_count > 0) return -1;
    db->bitset_words = (db->transaction_count + 63) / 64;
    size_t pos = 0;
    for (uint32_t id = 0; id <= db->max_item; id++) {
        if (db->support_by_id[id] == 0) continue;
        db->singletons[pos].item = id;
        db->singletons[pos].support = db->support_by_id[id];
        db->singletons[pos].bits = (uint64_t *)calloc(db->bitset_words, sizeof(uint64_t));
        if (!db->singletons[pos].bits && db->bitset_words > 0) return -1;
        pos++;
    }
    uint32_t *map = (uint32_t *)malloc(((size_t)db->max_item + 1) * sizeof(*map));
    if (!map && db->max_item > 0) return -1;
    for (uint32_t id = 0; id <= db->max_item; id++) map[id] = UINT32_MAX;
    for (size_t i = 0; i < db->singleton_count; i++) map[db->singletons[i].item] = (uint32_t)i;
    for (size_t t = 0; t < db->transaction_count; t++) {
        for (size_t i = 0; i < db->transactions[t].count; i++) {
            uint32_t idx = map[db->transactions[t].items[i]];
            db->singletons[idx].bits[t / 64] |= 1ULL << (t % 64);
        }
    }
    free(map);
    return 0;
}

static int cmp_singleton_desc(const void *a, const void *b) {
    const Singleton *x = (const Singleton *)a;
    const Singleton *y = (const Singleton *)b;
    if (x->support < y->support) return 1;
    if (x->support > y->support) return -1;
    if (x->item > y->item) return 1;
    if (x->item < y->item) return -1;
    return 0;
}

static int cmp_pattern_desc(const void *a, const void *b) {
    const Pattern *x = (const Pattern *)a;
    const Pattern *y = (const Pattern *)b;
    if (x->support < y->support) return 1;
    if (x->support > y->support) return -1;
    if (x->len > y->len) return 1;
    if (x->len < y->len) return -1;
    size_t n = x->len < y->len ? x->len : y->len;
    for (size_t i = 0; i < n; i++) {
        if (x->items[i] > y->items[i]) return 1;
        if (x->items[i] < y->items[i]) return -1;
    }
    return 0;
}

static void pattern_free(Pattern *p) {
    free(p->items);
    free(p->bits);
    memset(p, 0, sizeof(*p));
}

static int pattern_copy(Pattern *dst, const uint32_t *items, size_t len, const uint64_t *bits, size_t words, size_t support) {
    memset(dst, 0, sizeof(*dst));
    dst->items = (uint32_t *)malloc(len * sizeof(*dst->items));
    dst->bits = (uint64_t *)malloc(words * sizeof(*dst->bits));
    if (!dst->items || (!dst->bits && words > 0)) {
        pattern_free(dst);
        return -1;
    }
    memcpy(dst->items, items, len * sizeof(*dst->items));
    memcpy(dst->bits, bits, words * sizeof(*dst->bits));
    dst->len = len;
    dst->support = support;
    return 0;
}

static void level_free(Level *level) {
    for (size_t i = 0; i < level->count; i++) pattern_free(&level->data[i]);
    free(level->data);
    memset(level, 0, sizeof(*level));
}

static int level_add(Level *level, const Pattern *p) {
    if (level->count == level->cap) {
        size_t nc = level->cap ? level->cap * 2 : 64;
        Pattern *nd = (Pattern *)realloc(level->data, nc * sizeof(*nd));
        if (!nd) return -1;
        level->data = nd;
        level->cap = nc;
    }
    level->data[level->count++] = *p;
    return 0;
}

static void qheap_free(QHeap *heap) {
    for (size_t i = 0; i < heap->count; i++) pattern_free(&heap->data[i]);
    free(heap->data);
    memset(heap, 0, sizeof(*heap));
}

static int qheap_add(QHeap *heap, const uint32_t *items, size_t len, const uint64_t *bits, size_t words, size_t support) {
    if (heap->k == 0) return 0;
    if (heap->count >= heap->k && support < heap->threshold) return 0;
    if (heap->count == heap->cap) {
        size_t nc = heap->cap ? heap->cap * 2 : 128;
        Pattern *nd = (Pattern *)realloc(heap->data, nc * sizeof(*nd));
        if (!nd) return -1;
        heap->data = nd;
        heap->cap = nc;
    }
    if (pattern_copy(&heap->data[heap->count], items, len, bits, words, support) != 0) return -1;
    heap->count++;
    qsort(heap->data, heap->count, sizeof(*heap->data), cmp_pattern_desc);
    if (heap->count >= heap->k) {
        size_t old = heap->threshold;
        heap->threshold = heap->data[heap->k - 1].support;
        if (heap->threshold > old) heap->threshold_raises++;
        size_t keep = heap->count;
        while (keep > heap->k && heap->data[keep - 1].support < heap->threshold) {
            pattern_free(&heap->data[keep - 1]);
            keep--;
        }
        heap->count = keep;
    }
    return 0;
}

static int same_prefix(const Pattern *a, const Pattern *b) {
    if (a->len != b->len || a->len == 0) return 0;
    for (size_t i = 0; i + 1 < a->len; i++) {
        if (a->items[i] != b->items[i]) return 0;
    }
    return a->items[a->len - 1] < b->items[b->len - 1];
}

static void filter_level(Level *level, size_t threshold, DM_HTK_Stats *stats) {
    size_t keep = 0;
    for (size_t i = 0; i < level->count; i++) {
        if (level->data[i].support >= threshold) {
            if (keep != i) level->data[keep] = level->data[i];
            keep++;
        } else {
            pattern_free(&level->data[i]);
            stats->pruned_support++;
        }
    }
    level->count = keep;
}

static int init_level_from_singletons(Database *db, HTKCtx *ctx, Level *level) {
    qsort(db->singletons, db->singleton_count, sizeof(*db->singletons), cmp_singleton_desc);
    size_t keep = db->singleton_count;
    size_t initial_threshold = 0;
    if (ctx->params.k > 0 && db->singleton_count >= ctx->params.k) {
        initial_threshold = db->singletons[ctx->params.k - 1].support;
        keep = ctx->params.k;
        while (keep < db->singleton_count && db->singletons[keep].support == initial_threshold) keep++;
    }
    ctx->stats->singleton_kept = keep;
    ctx->stats->pruned_singletons = db->singleton_count > keep ? db->singleton_count - keep : 0;
    for (size_t i = 0; i < keep; i++) {
        uint32_t item = db->singletons[i].item;
        Pattern p;
        if (pattern_copy(&p, &item, 1, db->singletons[i].bits, db->bitset_words, db->singletons[i].support) != 0) return -1;
        if (level_add(level, &p) != 0) {
            pattern_free(&p);
            return -1;
        }
        if (qheap_add(&ctx->heap, &item, 1, db->singletons[i].bits, db->bitset_words, db->singletons[i].support) != 0) return -1;
    }
    return 0;
}

static int generate_next_level(HTKCtx *ctx, const Level *cur, size_t words, Level *next) {
    for (size_t i = 0; i < cur->count; i++) {
        for (size_t j = i + 1; j < cur->count; j++) {
            if (!same_prefix(&cur->data[i], &cur->data[j])) {
                if (cur->data[i].len > 1) break;
                continue;
            }
            if (is_limited(ctx)) return 0;
            ctx->stats->candidates++;
            ctx->stats->joins++;
            size_t len = cur->data[i].len + 1;
            uint32_t *items = (uint32_t *)malloc(len * sizeof(*items));
            uint64_t *bits = (uint64_t *)malloc(words * sizeof(*bits));
            if (!items || (!bits && words > 0)) {
                free(items);
                free(bits);
                return -1;
            }
            memcpy(items, cur->data[i].items, cur->data[i].len * sizeof(*items));
            items[len - 1] = cur->data[j].items[cur->data[j].len - 1];
            for (size_t w = 0; w < words; w++) bits[w] = cur->data[i].bits[w] & cur->data[j].bits[w];
            ctx->stats->intersections++;
            size_t support = popcount_words(bits, words);
            if (support > 0 && support >= ctx->heap.threshold) {
                Pattern p;
                p.items = items;
                p.len = len;
                p.support = support;
                p.bits = bits;
                if (qheap_add(&ctx->heap, items, len, bits, words, support) != 0) {
                    pattern_free(&p);
                    return -1;
                }
                if (level_add(next, &p) != 0) {
                    pattern_free(&p);
                    return -1;
                }
            } else {
                ctx->stats->pruned_support++;
                free(items);
                free(bits);
            }
        }
    }
    filter_level(next, ctx->heap.threshold, ctx->stats);
    return 0;
}

int htk_mine_file(const char *path, const DM_HTK_Params *params, DM_HTK_Stats *stats) {
    memset(stats, 0, sizeof(*stats));
    if (!path || !params || params->k == 0) return -1;
    Database db;
    memset(&db, 0, sizeof(db));
    if (load_database(path, &db) != 0) {
        database_free(&db);
        return -1;
    }

    HTKCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.params = *params;
    ctx.stats = stats;
    ctx.heap.k = params->k;
    ctx.start_clock = clock();

    stats->transactions = db.transaction_count;
    stats->nonempty_transactions = db.nonempty_transactions;
    stats->distinct_items = db.singleton_count;
    stats->k = params->k;
    stats->bitset_words = db.bitset_words;

    Level cur;
    memset(&cur, 0, sizeof(cur));
    if (init_level_from_singletons(&db, &ctx, &cur) != 0) {
        level_free(&cur);
        qheap_free(&ctx.heap);
        database_free(&db);
        return -1;
    }
    stats->levels = cur.count > 0 ? 1 : 0;

    while (cur.count > 1 && !is_limited(&ctx)) {
        if (params->max_depth > 0 && cur.data[0].len >= params->max_depth) break;
        Level next;
        memset(&next, 0, sizeof(next));
        if (generate_next_level(&ctx, &cur, db.bitset_words, &next) != 0) {
            level_free(&next);
            level_free(&cur);
            qheap_free(&ctx.heap);
            database_free(&db);
            return -1;
        }
        level_free(&cur);
        cur = next;
        if (cur.count > 0) stats->levels++;
    }

    stats->final_threshold = ctx.heap.threshold;
    stats->threshold_raises = ctx.heap.threshold_raises;
    stats->output_count = ctx.heap.count;
    for (size_t i = 0; i < ctx.heap.count; i++) {
        stats->total_output_items += ctx.heap.data[i].len;
        stats->avg_support += (double)ctx.heap.data[i].support;
        if (i == 0 || ctx.heap.data[i].support > stats->best_support) stats->best_support = (double)ctx.heap.data[i].support;
    }
    if (ctx.heap.count > 0) {
        stats->avg_output_length = (double)stats->total_output_items / (double)ctx.heap.count;
        stats->avg_support /= (double)ctx.heap.count;
    }
    stats->result_ram_bytes = ctx.heap.count * sizeof(Pattern) + stats->total_output_items * sizeof(uint32_t) + ctx.heap.count * db.bitset_words * sizeof(uint64_t);
    stats->result_disk_est_bytes = stats->total_output_items * 12 + ctx.heap.count * 24;

    level_free(&cur);
    qheap_free(&ctx.heap);
    database_free(&db);
    return 0;
}
