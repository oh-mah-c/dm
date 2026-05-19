#include "algorithms/kclotree_miner.h"
#include "core/dm_portability.h"

#include <ctype.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

typedef struct {
    uint32_t *items;
    size_t len;
} Seq;

typedef struct {
    Seq *seqs;
    size_t count;
    size_t cap;
    uint32_t max_item;
} SeqDB;

typedef struct {
    uint32_t *items;
    size_t len;
    uint32_t *positions;
    size_t support;
    int may_be_closed;
} Pattern;

typedef struct {
    Pattern **data;
    size_t count;
    size_t cap;
} Heap;

typedef struct {
    Pattern **data;
    size_t count;
    size_t cap;
} PatternVec;

typedef struct {
    SeqDB db;
    KCloParams params;
    KCloStats *stats;
    clock_t start_clock;
    size_t support_bound;
} Ctx;

static double elapsed_sec(Ctx *ctx) {
    return (double)(clock() - ctx->start_clock) / (double)CLOCKS_PER_SEC;
}

static int should_stop(Ctx *ctx) {
    if (ctx->params.max_seconds > 0.0 && elapsed_sec(ctx) >= ctx->params.max_seconds) {
        ctx->stats->limited = 1;
        return 1;
    }
    if (ctx->params.max_candidates > 0 && ctx->stats->candidates_processed >= ctx->params.max_candidates) {
        ctx->stats->limited = 1;
        return 1;
    }
    return 0;
}

static void pattern_free(Pattern *p) {
    if (!p) return;
    free(p->items);
    free(p->positions);
    free(p);
}

static Pattern *pattern_new(const uint32_t *items, size_t len, const uint32_t *positions, size_t support) {
    Pattern *p = (Pattern *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->items = (uint32_t *)malloc(len * sizeof(uint32_t));
    p->positions = (uint32_t *)malloc(support * 2 * sizeof(uint32_t));
    if (!p->items || !p->positions) {
        pattern_free(p);
        return NULL;
    }
    memcpy(p->items, items, len * sizeof(uint32_t));
    memcpy(p->positions, positions, support * 2 * sizeof(uint32_t));
    p->len = len;
    p->support = support;
    p->may_be_closed = 1;
    return p;
}

static int heap_less(Pattern *a, Pattern *b) {
    if (a->support != b->support) return a->support < b->support;
    if (a->len != b->len) return a->len > b->len;
    for (size_t i = 0; i < a->len && i < b->len; i++) {
        if (a->items[i] != b->items[i]) return a->items[i] > b->items[i];
    }
    return 0;
}

static int heap_push(Heap *h, Pattern *p) {
    if (h->count == h->cap) {
        size_t nc = h->cap ? h->cap * 2 : 256;
        Pattern **nd = (Pattern **)realloc(h->data, nc * sizeof(*nd));
        if (!nd) return -1;
        h->data = nd;
        h->cap = nc;
    }
    size_t i = h->count++;
    h->data[i] = p;
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        if (!heap_less(h->data[parent], h->data[i])) break;
        Pattern *tmp = h->data[parent];
        h->data[parent] = h->data[i];
        h->data[i] = tmp;
        i = parent;
    }
    return 0;
}

static Pattern *heap_pop(Heap *h) {
    if (h->count == 0) return NULL;
    Pattern *top = h->data[0];
    h->data[0] = h->data[--h->count];
    size_t i = 0;
    for (;;) {
        size_t l = 2 * i + 1, r = l + 1, best = i;
        if (l < h->count && heap_less(h->data[best], h->data[l])) best = l;
        if (r < h->count && heap_less(h->data[best], h->data[r])) best = r;
        if (best == i) break;
        Pattern *tmp = h->data[i];
        h->data[i] = h->data[best];
        h->data[best] = tmp;
        i = best;
    }
    return top;
}

static void heap_free(Heap *h) {
    for (size_t i = 0; i < h->count; i++) pattern_free(h->data[i]);
    free(h->data);
    memset(h, 0, sizeof(*h));
}

static int vec_add(PatternVec *v, Pattern *p) {
    if (v->count == v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 64;
        Pattern **nd = (Pattern **)realloc(v->data, nc * sizeof(*nd));
        if (!nd) return -1;
        v->data = nd;
        v->cap = nc;
    }
    v->data[v->count++] = p;
    return 0;
}

static void vec_free(PatternVec *v) {
    for (size_t i = 0; i < v->count; i++) pattern_free(v->data[i]);
    free(v->data);
    memset(v, 0, sizeof(*v));
}

static void db_free(SeqDB *db) {
    for (size_t i = 0; i < db->count; i++) free(db->seqs[i].items);
    free(db->seqs);
    memset(db, 0, sizeof(*db));
}

static int db_add_seq(SeqDB *db, uint32_t *items, size_t len) {
    if (len == 0) {
        free(items);
        return 0;
    }
    if (db->count == db->cap) {
        size_t nc = db->cap ? db->cap * 2 : 1024;
        Seq *ns = (Seq *)realloc(db->seqs, nc * sizeof(*ns));
        if (!ns) {
            free(items);
            return -1;
        }
        db->seqs = ns;
        db->cap = nc;
    }
    db->seqs[db->count].items = items;
    db->seqs[db->count].len = len;
    for (size_t i = 0; i < len; i++) if (items[i] > db->max_item) db->max_item = items[i];
    db->count++;
    return 0;
}

static int parse_file(const char *path, SeqDB *db) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char *line = NULL;
    size_t n = 0;
    while (dm_getline(&line, &n, fp) != -1) {
        if (line[0] == '@' || line[0] == '#' || line[0] == '%' || line[0] == '\n') continue;
        uint32_t *items = NULL;
        size_t len = 0, cap = 0;
        char *p = line;
        while (*p) {
            while (*p && isspace((unsigned char)*p)) p++;
            if (!*p) break;
            char *end = NULL;
            long v = strtol(p, &end, 10);
            if (end == p) break;
            if (v == -2) break;
            if (v > 0) {
                if (len == cap) {
                    size_t nc = cap ? cap * 2 : 64;
                    uint32_t *ni = (uint32_t *)realloc(items, nc * sizeof(*ni));
                    if (!ni) {
                        free(items);
                        free(line);
                        fclose(fp);
                        return -1;
                    }
                    items = ni;
                    cap = nc;
                }
                items[len++] = (uint32_t)v;
            }
            p = end;
        }
        if (db_add_seq(db, items, len) != 0) {
            free(line);
            fclose(fp);
            return -1;
        }
    }
    free(line);
    fclose(fp);
    return 0;
}

static int ends_with(const char *s, const char *suffix) {
    size_t n = strlen(s), m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static char *join_path(const char *a, const char *b) {
    size_t n = strlen(a), m = strlen(b);
    int slash = n && a[n - 1] == '/';
    char *r = (char *)malloc(n + m + (slash ? 1 : 2));
    if (!r) return NULL;
    sprintf(r, "%s%s%s", a, slash ? "" : "/", b);
    return r;
}

static int load_path(const char *path, SeqDB *db) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    if (S_ISREG(st.st_mode)) return parse_file(path, db);
    if (!S_ISDIR(st.st_mode)) return -1;
    DIR *dir = opendir(path);
    if (!dir) return -1;
    struct dirent *ent;
    int ok = 0;
    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        if (!ends_with(ent->d_name, ".txt")) continue;
        char *p = join_path(path, ent->d_name);
        if (!p) {
            closedir(dir);
            return -1;
        }
        if (parse_file(p, db) == 0) ok = 1;
        free(p);
    }
    closedir(dir);
    return ok ? 0 : -1;
}

static int pattern_contains(const Pattern *super, const Pattern *sub) {
    if (sub->len > super->len) return 0;
    size_t i = 0, j = 0;
    while (i < sub->len && j < super->len) {
        if (sub->items[i] == super->items[j]) i++;
        j++;
    }
    return i == sub->len;
}

static void closed_coverage(Ctx *ctx, PatternVec *closed, Pattern *p) {
    for (size_t i = 0; i < closed->count; i++) {
        Pattern *q = closed->data[i];
        if (q->support == p->support && pattern_contains(p, q)) {
            pattern_free(q);
            closed->data[i] = closed->data[--closed->count];
            i--;
            ctx->stats->absorbed_patterns++;
        }
    }
}

static int add_closed(Ctx *ctx, PatternVec *closed, Pattern *p) {
    if (ctx->params.type == KCLO_TYPE_REDUNDANCY_AWARE) {
        for (size_t i = 0; i < closed->count; i++) {
            if (closed->data[i]->support >= p->support && pattern_contains(closed->data[i], p)) {
                ctx->stats->absorbed_patterns++;
                return 0;
            }
        }
    }
    closed_coverage(ctx, closed, p);
    Pattern *copy = pattern_new(p->items, p->len, p->positions, p->support);
    if (!copy) return -1;
    if (vec_add(closed, copy) != 0) {
        pattern_free(copy);
        return -1;
    }
    ctx->stats->closed_candidates++;
    return 0;
}

static size_t kth_unique_support(PatternVec *closed, size_t k) {
    if (k == 0 || closed->count == 0) return 0;
    size_t *vals = (size_t *)malloc(closed->count * sizeof(size_t));
    if (!vals) return 0;
    size_t n = 0;
    for (size_t i = 0; i < closed->count; i++) {
        int seen = 0;
        for (size_t j = 0; j < n; j++) if (vals[j] == closed->data[i]->support) seen = 1;
        if (!seen) vals[n++] = closed->data[i]->support;
    }
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (vals[j] > vals[i]) {
                size_t tmp = vals[i];
                vals[i] = vals[j];
                vals[j] = tmp;
            }
        }
    }
    size_t out = n >= k ? vals[k - 1] : 0;
    free(vals);
    return out;
}

static int satisfies_break(Ctx *ctx, PatternVec *closed, Heap *heap) {
    if (closed->count < ctx->params.k) return 0;
    if (ctx->params.type == KCLO_TYPE_GROUP) {
        size_t kth = kth_unique_support(closed, ctx->params.k);
        if (kth == 0) return 0;
        return heap->count == 0 || heap->data[0]->support < kth;
    }
    size_t min_sup = closed->data[0]->support;
    for (size_t i = 1; i < closed->count; i++) {
        if (closed->data[i]->support < min_sup) min_sup = closed->data[i]->support;
    }
    return heap->count == 0 || heap->data[0]->support < min_sup;
}

static int extend_pattern(Ctx *ctx, Pattern *p, Heap *heap) {
    uint32_t max_item = ctx->db.max_item;
    uint32_t *support = (uint32_t *)calloc(max_item + 1, sizeof(uint32_t));
    uint32_t *last_seen = (uint32_t *)calloc(max_item + 1, sizeof(uint32_t));
    uint32_t *positions = (uint32_t *)malloc((max_item + 1) * p->support * 2 * sizeof(uint32_t));
    uint32_t *next_items = (uint32_t *)malloc((p->len + 1) * sizeof(uint32_t));
    if (!support || !last_seen || !positions || !next_items) {
        free(support); free(last_seen); free(positions); free(next_items);
        return -1;
    }
    memcpy(next_items, p->items, p->len * sizeof(uint32_t));
    for (size_t occ = 0; occ < p->support; occ++) {
        uint32_t sid = p->positions[2 * occ];
        uint32_t pos = p->positions[2 * occ + 1];
        Seq *s = &ctx->db.seqs[sid];
        for (size_t j = pos + 1; j < s->len; j++) {
            uint32_t item = s->items[j];
            if (last_seen[item] == sid + 1) continue;
            last_seen[item] = sid + 1;
            size_t idx = (size_t)item * p->support * 2 + support[item] * 2;
            positions[idx] = sid;
            positions[idx + 1] = (uint32_t)j;
            support[item]++;
        }
    }
    int has_same_support_extension = 0;
    for (uint32_t item = 1; item <= max_item; item++) {
        if (support[item] == 0) continue;
        ctx->stats->projected_extensions++;
        if (support[item] < ctx->support_bound) {
            ctx->stats->pruned_by_bound++;
            continue;
        }
        if (support[item] == p->support) {
            has_same_support_extension = 1;
            ctx->stats->same_support_extensions++;
        }
        next_items[p->len] = item;
        size_t off = (size_t)item * p->support * 2;
        Pattern *child = pattern_new(next_items, p->len + 1, positions + off, support[item]);
        if (!child) {
            free(support); free(last_seen); free(positions); free(next_items);
            return -1;
        }
        if (heap_push(heap, child) != 0) {
            pattern_free(child);
            free(support); free(last_seen); free(positions); free(next_items);
            return -1;
        }
        ctx->stats->candidates_created++;
        if (heap->count > ctx->stats->max_heap_size) ctx->stats->max_heap_size = heap->count;
    }
    if (has_same_support_extension) p->may_be_closed = 0;
    free(support); free(last_seen); free(positions); free(next_items);
    return 0;
}

KCloParams kclotree_default_params(void) {
    KCloParams p;
    p.k = 10;
    p.max_depth = 6;
    p.max_candidates = 250000;
    p.max_seconds = 60.0;
    p.type = KCLO_TYPE_GENERIC;
    return p;
}

int kclotree_parse_type(const char *name, KCloMiningType *type) {
    if (!name || strcmp(name, "generic") == 0) {
        *type = KCLO_TYPE_GENERIC;
        return 0;
    }
    if (strcmp(name, "group") == 0) {
        *type = KCLO_TYPE_GROUP;
        return 0;
    }
    if (strcmp(name, "redundancy") == 0 || strcmp(name, "redundancy_aware") == 0) {
        *type = KCLO_TYPE_REDUNDANCY_AWARE;
        return 0;
    }
    return -1;
}

const char *kclotree_type_name(KCloMiningType type) {
    switch (type) {
        case KCLO_TYPE_GROUP: return "group";
        case KCLO_TYPE_REDUNDANCY_AWARE: return "redundancy_aware";
        default: return "generic";
    }
}

int kclotree_mine_path(const char *path, const KCloParams *params, KCloStats *stats) {
    if (!path || !stats) return -1;
    Ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.params = params ? *params : kclotree_default_params();
    if (ctx.params.k == 0) ctx.params.k = 10;
    ctx.stats = stats;
    ctx.start_clock = clock();
    memset(stats, 0, sizeof(*stats));
    stats->type_name = kclotree_type_name(ctx.params.type);
    if (load_path(path, &ctx.db) != 0 || ctx.db.count == 0) {
        db_free(&ctx.db);
        return -1;
    }
    stats->sequences = ctx.db.count;
    stats->k = ctx.params.k;
    for (size_t i = 0; i < ctx.db.count; i++) {
        if (ctx.db.seqs[i].len > stats->max_sequence_length) stats->max_sequence_length = ctx.db.seqs[i].len;
    }
    uint32_t max_item = ctx.db.max_item;
    uint32_t *support = (uint32_t *)calloc(max_item + 1, sizeof(uint32_t));
    uint32_t *last_seen = (uint32_t *)calloc(max_item + 1, sizeof(uint32_t));
    if (!support || !last_seen) {
        free(support); free(last_seen); db_free(&ctx.db);
        return -1;
    }
    for (size_t sid = 0; sid < ctx.db.count; sid++) {
        Seq *s = &ctx.db.seqs[sid];
        for (size_t j = 0; j < s->len; j++) {
            uint32_t item = s->items[j];
            if (last_seen[item] == sid + 1) continue;
            last_seen[item] = sid + 1;
            support[item]++;
        }
    }
    for (uint32_t item = 1; item <= max_item; item++) if (support[item]) stats->distinct_items++;
    size_t *uniq = (size_t *)malloc(stats->distinct_items * sizeof(size_t));
    size_t un = 0;
    for (uint32_t item = 1; item <= max_item; item++) {
        if (!support[item]) continue;
        int seen = 0;
        for (size_t i = 0; i < un; i++) if (uniq[i] == support[item]) seen = 1;
        if (!seen) uniq[un++] = support[item];
    }
    for (size_t i = 0; i < un; i++) {
        for (size_t j = i + 1; j < un; j++) {
            if (uniq[j] > uniq[i]) {
                size_t tmp = uniq[i]; uniq[i] = uniq[j]; uniq[j] = tmp;
            }
        }
    }
    ctx.support_bound = (ctx.params.type == KCLO_TYPE_REDUNDANCY_AWARE || un < ctx.params.k) ? 1 : uniq[ctx.params.k - 1];
    free(uniq);
    Heap heap = {0};
    PatternVec closed = {0};
    for (uint32_t item = 1; item <= max_item; item++) {
        if (support[item] < ctx.support_bound) continue;
        uint32_t *pos = (uint32_t *)malloc(support[item] * 2 * sizeof(uint32_t));
        if (!pos) {
            free(support); free(last_seen); heap_free(&heap); vec_free(&closed); db_free(&ctx.db);
            return -1;
        }
        size_t c = 0;
        for (size_t sid = 0; sid < ctx.db.count; sid++) {
            Seq *s = &ctx.db.seqs[sid];
            for (size_t j = 0; j < s->len; j++) {
                if (s->items[j] == item) {
                    pos[2 * c] = (uint32_t)sid;
                    pos[2 * c + 1] = (uint32_t)j;
                    c++;
                    break;
                }
            }
        }
        Pattern *p = pattern_new(&item, 1, pos, c);
        free(pos);
        if (!p || heap_push(&heap, p) != 0) {
            pattern_free(p);
            free(support); free(last_seen); heap_free(&heap); vec_free(&closed); db_free(&ctx.db);
            return -1;
        }
        stats->candidates_created++;
        if (heap.count > stats->max_heap_size) stats->max_heap_size = heap.count;
    }
    free(support);
    free(last_seen);
    while (heap.count > 0 && !should_stop(&ctx) && !satisfies_break(&ctx, &closed, &heap)) {
        Pattern *p = heap_pop(&heap);
        stats->candidates_processed++;
        if (p->support < ctx.support_bound) {
            stats->pruned_by_bound++;
            pattern_free(p);
            continue;
        }
        if (ctx.params.max_depth == 0 || p->len < ctx.params.max_depth) {
            if (extend_pattern(&ctx, p, &heap) != 0) {
                pattern_free(p);
                heap_free(&heap); vec_free(&closed); db_free(&ctx.db);
                return -1;
            }
        }
        if (p->may_be_closed) {
            if (add_closed(&ctx, &closed, p) != 0) {
                pattern_free(p);
                heap_free(&heap); vec_free(&closed); db_free(&ctx.db);
                return -1;
            }
        }
        pattern_free(p);
    }
    size_t unique_supports = 0;
    for (size_t i = 0; i < closed.count; i++) {
        int seen = 0;
        for (size_t j = 0; j < i; j++) if (closed.data[j]->support == closed.data[i]->support) seen = 1;
        if (!seen) unique_supports++;
        if (i == 0 || closed.data[i]->support > stats->max_reported_support) stats->max_reported_support = closed.data[i]->support;
        if (i == 0 || closed.data[i]->support < stats->min_reported_support) stats->min_reported_support = closed.data[i]->support;
        stats->avg_reported_support += (double)closed.data[i]->support;
        stats->avg_pattern_length += (double)closed.data[i]->len;
        stats->result_ram_bytes += sizeof(Pattern) + closed.data[i]->len * sizeof(uint32_t) + closed.data[i]->support * 2 * sizeof(uint32_t);
        stats->result_disk_est_bytes += 24 + closed.data[i]->len * 12;
    }
    stats->output_count = closed.count;
    stats->unique_supports_reported = unique_supports;
    if (closed.count) {
        stats->avg_reported_support /= (double)closed.count;
        stats->avg_pattern_length /= (double)closed.count;
    }
    heap_free(&heap);
    vec_free(&closed);
    db_free(&ctx.db);
    return 0;
}
