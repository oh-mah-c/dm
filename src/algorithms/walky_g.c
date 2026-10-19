/*
 * Walky-G (Szathmary, Valtchev, Napoli, Godin, CLA 2012), implemented as described in
 * Algorithms 1-4 of the paper.
 *
 * Depth-first traversal of the IT-tree restricted to frequent generators (FGs), siblings
 * processed right-to-left so that every subset of an itemset is met before the itemset.
 * fgMap stores every FG found so far with its support; it is used for test 3 (does the
 * candidate subsume an FG?) and for the minimality test of rare candidates.
 *
 * Items are placed in the tree in increasing order of identifier (the paper does not fix an
 * order). If the environment variable DM_MRI_OUT is set, the minimal rare itemsets are written
 * to that file, one per line, items in increasing order.
 */
#include "algorithms/walky_g.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef struct {
    uint32_t *items;   /* itemset, increasing item identifiers */
    uint32_t len;
    uint32_t *tids;
    uint32_t sup;
} WNode;

/* ---- fgMap: open-addressing hash map, key = itemset, value = support ---- */
typedef struct {
    uint64_t *slot_off;    /* offset+1 into key pool, 0 = empty */
    uint32_t *slot_len;
    uint32_t *slot_sup;
    size_t cap, count;
    uint32_t *pool;
    size_t pool_len, pool_cap;
} FgMap;

static uint64_t hash_items(const uint32_t *a, uint32_t n) {
    uint64_t h = 1469598103934665603ULL ^ n;
    for (uint32_t i = 0; i < n; i++) { h ^= a[i] + 0x9e3779b97f4a7c15ULL; h *= 1099511628211ULL; h ^= h >> 29; }
    return h;
}

static void fg_init(FgMap *m) {
    m->cap = 1 << 16; m->count = 0;
    m->slot_off = calloc(m->cap, sizeof(uint64_t));
    m->slot_len = calloc(m->cap, sizeof(uint32_t));
    m->slot_sup = calloc(m->cap, sizeof(uint32_t));
    m->pool_cap = 1 << 20; m->pool_len = 0;
    m->pool = malloc(m->pool_cap * sizeof(uint32_t));
    if (!m->slot_off || !m->slot_len || !m->slot_sup || !m->pool) { fprintf(stderr, "[Walky-G] out of memory\n"); abort(); }
}

static void fg_free(FgMap *m) { free(m->slot_off); free(m->slot_len); free(m->slot_sup); free(m->pool); }

static long fg_find(const FgMap *m, const uint32_t *a, uint32_t n) {
    size_t i = hash_items(a, n) & (m->cap - 1);
    while (m->slot_off[i]) {
        if (m->slot_len[i] == n && memcmp(m->pool + (m->slot_off[i] - 1), a, n * sizeof(uint32_t)) == 0) return (long)i;
        i = (i + 1) & (m->cap - 1);
    }
    return -1;
}

static void fg_put(FgMap *m, const uint32_t *a, uint32_t n, uint32_t sup);

static void fg_grow(FgMap *m) {
    FgMap o = *m;
    m->cap *= 2; m->count = 0;
    m->slot_off = calloc(m->cap, sizeof(uint64_t));
    m->slot_len = calloc(m->cap, sizeof(uint32_t));
    m->slot_sup = calloc(m->cap, sizeof(uint32_t));
    if (!m->slot_off || !m->slot_len || !m->slot_sup) { fprintf(stderr, "[Walky-G] out of memory\n"); abort(); }
    for (size_t i = 0; i < o.cap; i++) {
        if (!o.slot_off[i]) continue;
        size_t j = hash_items(o.pool + (o.slot_off[i] - 1), o.slot_len[i]) & (m->cap - 1);
        while (m->slot_off[j]) j = (j + 1) & (m->cap - 1);
        m->slot_off[j] = o.slot_off[i]; m->slot_len[j] = o.slot_len[i]; m->slot_sup[j] = o.slot_sup[i];
        m->count++;
    }
    free(o.slot_off); free(o.slot_len); free(o.slot_sup);
}

static void fg_put(FgMap *m, const uint32_t *a, uint32_t n, uint32_t sup) {
    if (2 * (m->count + 1) > m->cap) fg_grow(m);
    if (m->pool_len + n + 1 > m->pool_cap) {
        while (m->pool_len + n + 1 > m->pool_cap) m->pool_cap *= 2;
        m->pool = realloc(m->pool, m->pool_cap * sizeof(uint32_t));
        if (!m->pool) { fprintf(stderr, "[Walky-G] out of memory\n"); abort(); }
    }
    memcpy(m->pool + m->pool_len, a, n * sizeof(uint32_t));
    size_t i = hash_items(a, n) & (m->cap - 1);
    while (m->slot_off[i]) i = (i + 1) & (m->cap - 1);
    m->slot_off[i] = m->pool_len + 1; m->slot_len[i] = n; m->slot_sup[i] = sup;
    m->pool_len += n;
    m->count++;
}

typedef struct {
    uint32_t lambda;
    FgMap fg;
    FILE *out;
    uint32_t *scratch;
    size_t mri, mri_footprint, candidates;
} WCtx;

static void save_mri(WCtx *c, const uint32_t *items, uint32_t len) {
    c->mri++;
    c->mri_footprint += len;
    if (!c->out) return;
    for (uint32_t i = 0; i < len; i++) fprintf(c->out, i ? " %u" : "%u", items[i]);
    fputc('\n', c->out);
}

/* one-size-smaller subset of items without position skip, into c->scratch */
static const uint32_t *drop(WCtx *c, const uint32_t *items, uint32_t len, uint32_t skip) {
    uint32_t k = 0;
    for (uint32_t i = 0; i < len; i++) if (i != skip) c->scratch[k++] = items[i];
    return c->scratch;
}

/* Algorithm 4: does cand subsume an FG (or have a non-FG subset)? parents are positions len-1, len-2 */
static int cand_subsumes_an_fg(WCtx *c, const uint32_t *items, uint32_t len, uint32_t sup) {
    for (uint32_t s = 0; s + 2 < len; s++) {
        long i = fg_find(&c->fg, drop(c, items, len, s), len - 1);
        if (i < 0) return 1;                       /* case 2: non-FG subset */
        if (c->fg.slot_sup[i] == sup) return 1;    /* case 1: same support */
    }
    return 0;
}

static uint32_t intersect(const uint32_t *a, uint32_t na, const uint32_t *b, uint32_t nb, uint32_t *out) {
    uint32_t i = 0, j = 0, k = 0;
    while (i < na && j < nb) {
        if (a[i] < b[j]) i++;
        else if (a[i] > b[j]) j++;
        else { out[k++] = a[i]; i++; j++; }
    }
    return k;
}

/* Algorithm 3. Returns 1 and fills *g if cand is a frequent generator. */
static int get_next_generator(WCtx *c, const WNode *curr, const WNode *other, WNode *g) {
    c->candidates++;
    uint32_t len = curr->len + 1;
    uint32_t *items = malloc(len * sizeof(uint32_t));
    memcpy(items, curr->items, curr->len * sizeof(uint32_t));
    items[len - 1] = other->items[other->len - 1];
    uint32_t cap = curr->sup < other->sup ? curr->sup : other->sup;
    uint32_t *tids = malloc((cap ? cap : 1) * sizeof(uint32_t));
    uint32_t sup = intersect(curr->tids, curr->sup, other->tids, other->sup, tids);
    if (sup < c->lambda) {                                   /* test 1 */
        int minimal = 1;
        for (uint32_t s = 0; s < len && minimal; s++)
            if (fg_find(&c->fg, drop(c, items, len, s), len - 1) < 0) minimal = 0;
        if (minimal) save_mri(c, items, len);   /* Algorithm 3 keeps support-0 candidates too */
        free(items); free(tids);
        return 0;
    }
    if (sup == curr->sup || sup == other->sup) { free(items); free(tids); return 0; }   /* test 2 */
    if (cand_subsumes_an_fg(c, items, len, sup)) { free(items); free(tids); return 0; } /* test 3 */
    g->items = items; g->len = len; g->tids = tids; g->sup = sup;
    return 1;
}

static void node_free(WNode *n) { free(n->items); free(n->tids); }

/* Algorithm 2 on sibling list sib[0..k-1], node index i */
static void extend(WCtx *c, WNode *sib, uint32_t k, uint32_t i) {
    WNode *curr = &sib[i];
    WNode *kids = malloc((k - i) * sizeof(WNode));
    uint32_t nk = 0;
    for (uint32_t j = i + 1; j < k; j++)
        if (get_next_generator(c, curr, &sib[j], &kids[nk])) nk++;
    for (uint32_t t = nk; t-- > 0;) {
        fg_put(&c->fg, kids[t].items, kids[t].len, kids[t].sup);   /* saveFg */
        extend(c, kids, nk, t);
    }
    for (uint32_t t = 0; t < nk; t++) node_free(&kids[t]);
    free(kids);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_WALKY_G_Params *p = (DM_WALKY_G_Params *)params;
    double ms = p ? p->min_support : 0.01;
    size_t n = ds->count;
    WCtx c;
    memset(&c, 0, sizeof c);
    c.lambda = ms < 1.0 ? (uint32_t)ceil(ms * (double)n) : (uint32_t)ms;
    if (c.lambda == 0) c.lambda = 1;
    printf("[Walky-G] %zu transactions, lambda = %u (rare: support < lambda)\n", n, c.lambda);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    uint32_t *cnt = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t t = 0; t < n; t++) for (size_t j = 0; j < data[t].count; j++) cnt[data[t].items[j]]++;

    const char *outp = getenv("DM_MRI_OUT");
    c.out = outp ? fopen(outp, "w") : NULL;
    c.scratch = malloc((ds->max_id + 2) * sizeof(uint32_t));
    fg_init(&c.fg);
    fg_put(&c.fg, NULL, 0, (uint32_t)n);                 /* the empty set */

    uint32_t nroot = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (cnt[i] >= c.lambda && cnt[i] < n) nroot++;
        else if (cnt[i] > 0 && cnt[i] < c.lambda) save_mri(&c, &i, 1);
    }
    WNode *root = malloc((nroot ? nroot : 1) * sizeof(WNode));
    uint32_t *fill = calloc(ds->max_id + 1, sizeof(uint32_t));
    int32_t *pos = malloc((ds->max_id + 1) * sizeof(int32_t));
    uint32_t r = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        pos[i] = -1;
        if (cnt[i] >= c.lambda && cnt[i] < n) {
            pos[i] = (int32_t)r;
            root[r].items = malloc(sizeof(uint32_t)); root[r].items[0] = i; root[r].len = 1;
            root[r].tids = malloc(cnt[i] * sizeof(uint32_t)); root[r].sup = cnt[i];
            r++;
        }
    }
    for (size_t t = 0; t < n; t++)
        for (size_t j = 0; j < data[t].count; j++) {
            int32_t q = pos[data[t].items[j]];
            if (q >= 0) root[q].tids[fill[q]++] = (uint32_t)t;
        }
    free(fill); free(pos); free(cnt);

    for (uint32_t t = nroot; t-- > 0;) {
        fg_put(&c.fg, root[t].items, 1, root[t].sup);
        extend(&c, root, nroot, t);
    }
    for (uint32_t t = 0; t < nroot; t++) node_free(&root[t]);
    free(root);

    printf("[Walky-G] Complete. Minimal rare itemsets: %zu (frequent generators=%zu, visited_nodes=%zu)\n",
           c.mri, c.fg.count - 1, c.candidates);
    dm_bench_record_results(c.mri, c.mri_footprint);
    if (c.out) fclose(c.out);
    fg_free(&c.fg);
    free(c.scratch);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "walky_g",
    .name = "Walky-G",
    .description = "Minimal rare itemsets by depth-first traversal of frequent generators (Szathmary et al., CLA 2012).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
