/*
 * mfhoi: maximal high occupancy itemsets (MHOIs).
 *
 * Problem (occupancy of Deng, FGCS 2020). For an itemset X with supporting
 * transactions G(X), occ(X) = |X| * sum_{T in G(X)} 1/|T|. X is a high
 * occupancy itemset (HOI) if occ(X) >= xi, and an MHOI if no proper superset
 * of X is an HOI. xi is absolute, or a fraction of the number of transactions
 * if it is below 1 (as in Deng 2020).
 *
 * Properties used:
 *   (1) occ(X) <= supp(X) (Deng 2020, Theorem 1), so every HOI has support
 *       >= ceil(xi); items and extensions below this support are discarded.
 *   (2) occ(cl X) = |cl X| / |X| * occ(X) >= occ(X), so every MHOI is closed;
 *       rule P2 below records closures. The search itself enumerates itemsets
 *       by prefix extension in increasing-support order (a set-enumeration
 *       tree), restricted to itemsets with support >= ceil(xi); computing
 *       closures at every node did not pay off in our experiments.
 *   (3) Size-coupled bound. For a node C with tail items R, let cap(T) be the
 *       number of tail items in T. A descendant C + Z with |Z| = k is supported
 *       only by transactions with cap(T) >= k, so
 *           occ(C + Z) <= B(C) = max_{k>=1} (|C| + k) * sum_{T in G(C), cap(T) >= k} 1/|T|.
 *       Since cap(T) >= k implies |T| >= |C| + k, B(C) <= UBO(C) of Deng 2020.
 *
 * Pruning rules at a node C:
 *   P1  B(C) < xi: no descendant is an HOI;
 *   P2  if C + R is an HOI, cl(C + R) is an HOI that contains every itemset of
 *       the subtree: record it and skip the subtree (head-union-tail);
 *   P3  if C + R is contained in a recorded HOI, the subtree has no MHOI;
 *   P4  before computing the closure of a child C + e, apply the bound to it
 *       (base |C| + 1, k >= 0, cap = tail items after e).
 * An HOI whose subtree contains no HOI is recorded; recorded itemsets that are
 * subsets of other recorded itemsets are removed at the end.
 *
 * Projected transactions with the same items and the same original length are
 * merged into one weighted row; occ and B depend only on these, so merging is
 * exact.
 *
 * The legacy two-threshold miner (support and average occupancy) used by
 * fhoi_miner, weak_mfhoi_miner and strong_mfhoi_miner is kept below as
 * mfhoi_adapter_run.
 */
#include "algorithms/mfhoi.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include "core/dm_exact_occ.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ══ MHOI miner ═══════════════════════════════════════════════════════════ */

typedef struct {
    int len;        /* length of the original transaction */
    int w;          /* number of merged transactions */
    int n;          /* number of items */
    int *it;        /* sorted items */
} MRow;

typedef struct { int len, w, cap; } MUnit;

typedef struct {
    double xi;
    DmThreshold xit;            /* xi as an exact fraction */
    int minsup;                 /* ceil(xi) */
    int m;                      /* items with support >= minsup */
    int *occ, *pos, *touched, *keep;
    int **sets; int *size; int ncand, capcand;
    long long nodes, hoi_nodes, pruned_bound, pruned_hut, pruned_sub, pruned_child;
} MCtx;

static int m_ge(double x, double t) { return x >= t - DM_OCC_TOL(t); }

/* exact test k * sum_r w_r / len_r >= xi over the rows (rows with cap == need_cap
 * if need_cap >= 0); approx is the double value */
static int m_rows_ge(const MCtx *c, double approx, int k, const MRow *rows, int nr, const MUnit *u, int need_cap) {
    int res;
    if (dm_occ_decided(approx, c->xit, &res)) return res;
    uint32_t *len = malloc((nr > 0 ? nr : 1) * sizeof(uint32_t));
    uint64_t *w = malloc((nr > 0 ? nr : 1) * sizeof(uint64_t));
    size_t m = 0;
    for (int r = 0; r < nr; r++) {
        if (need_cap >= 0 && u[r].cap != need_cap) continue;
        len[m] = (uint32_t)rows[r].len; w[m] = (uint64_t)rows[r].w; m++;
    }
    res = dm_occ_ge_exact((uint64_t)k, len, w, m, c->xit);
    free(len); free(w);
    return res;
}

static int m_cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static int m_cmp_row(const void *a, const void *b) {
    const MRow *x = a, *y = b;
    if (x->len != y->len) return (x->len > y->len) - (x->len < y->len);
    if (x->n != y->n) return (x->n > y->n) - (x->n < y->n);
    for (int i = 0; i < x->n; i++)
        if (x->it[i] != y->it[i]) return (x->it[i] > y->it[i]) - (x->it[i] < y->it[i]);
    return 0;
}

static int m_merge_rows(MRow *r, int nr) {
    if (nr <= 1) return nr;
    qsort(r, nr, sizeof(MRow), m_cmp_row);
    int out = 0;
    for (int i = 0; i < nr; i++) {
        if (out > 0 && m_cmp_row(&r[out - 1], &r[i]) == 0) { r[out - 1].w += r[i].w; continue; }
        r[out++] = r[i];
    }
    return out;
}

static int m_is_subset(const int *a, int na, const int *b, int nb) {
    if (na > nb) return 0;
    int i = 0, j = 0;
    while (i < na && j < nb) {
        if (a[i] == b[j]) { i++; j++; }
        else if (a[i] > b[j]) j++;
        else return 0;
    }
    return i == na;
}

static int m_has_superset(const MCtx *c, const int *set, int size) {
    for (int i = 0; i < c->ncand; i++)
        if (c->size[i] >= size && m_is_subset(set, size, c->sets[i], c->size[i])) return 1;
    return 0;
}

static void m_record(MCtx *c, const int *set, int size) {
    if (c->ncand == c->capcand) {
        c->capcand = c->capcand ? 2 * c->capcand : 256;
        c->sets = realloc(c->sets, c->capcand * sizeof(int *));
        c->size = realloc(c->size, c->capcand * sizeof(int));
    }
    int *s = malloc((size > 0 ? size : 1) * sizeof(int));
    memcpy(s, set, size * sizeof(int));
    qsort(s, size, sizeof(int), m_cmp_int);
    c->sets[c->ncand] = s;
    c->size[c->ncand] = size;
    c->ncand++;
}

/* max over k >= kmin of (base + k) * sum of w/len over units with cap >= k,
 * restricted to k with at least minsup such transactions */
static double m_bound(const MCtx *c, const MUnit *u, int nu, int base, int kmin) {
    int cmax = 0;
    for (int i = 0; i < nu; i++) if (u[i].cap > cmax) cmax = u[i].cap;
    if (cmax < kmin) return 0.0;
    double *s = calloc(cmax + 2, sizeof(double));
    long *w = calloc(cmax + 2, sizeof(long));
    for (int i = 0; i < nu; i++) { s[u[i].cap] += (double)u[i].w / u[i].len; w[u[i].cap] += u[i].w; }
    double best = 0.0, acc = 0.0;
    long accw = 0;
    for (int k = cmax; k >= kmin; k--) {
        acc += s[k]; accw += w[k];
        if (accw < c->minsup) continue;
        double v = (base + k) * acc;
        if (v > best) best = v;
    }
    free(s); free(w);
    return best;
}

/* C: itemset of the node; rows: its cover projected on the items after the
 * last item of C that have support >= minsup in the cover; core: last item added.
 * Returns 1 if the subtree rooted at C (C included) contains an HOI. */
static int m_mine(MCtx *c, const int *C, int csize, MRow *rows, int nr, int core) {
    c->nodes++;
    double rsum = 0.0;
    int ntouch = 0;
    for (int r = 0; r < nr; r++) {
        rsum += (double)rows[r].w / rows[r].len;
        for (int j = 0; j < rows[r].n; j++) {
            int it = rows[r].it[j];
            if (c->occ[it] == 0) c->touched[ntouch++] = it;
            c->occ[it] += rows[r].w;
        }
    }
    int is_hoi = csize > 0 && m_rows_ge(c, (double)csize * rsum, csize, rows, nr, NULL, -1);
    if (is_hoi) c->hoi_nodes++;

    int ntail = 0;
    int *tail = malloc((ntouch > 0 ? ntouch : 1) * sizeof(int));
    for (int i = 0; i < ntouch; i++) {
        int it = c->touched[i];
        if (it > core && c->occ[it] >= c->minsup) tail[ntail++] = it;
    }
    qsort(tail, ntail, sizeof(int), m_cmp_int);
    for (int k = 0; k < ntail; k++) c->pos[tail[k]] = k + 1;

    MUnit *u = malloc((nr > 0 ? nr : 1) * sizeof(MUnit));
    long full_w = 0;
    double full_r = 0.0;
    for (int r = 0; r < nr; r++) {
        int cp = 0;
        for (int j = 0; j < rows[r].n; j++) if (c->pos[rows[r].it[j]]) cp++;
        u[r].len = rows[r].len; u[r].w = rows[r].w; u[r].cap = cp;
        if (ntail > 0 && cp == ntail) { full_w += rows[r].w; full_r += (double)rows[r].w / rows[r].len; }
    }

    int result = 0, done = (ntail == 0);
    if (!done && !m_ge(m_bound(c, u, nr, csize, 1), c->xi)) { c->pruned_bound++; done = 1; }      /* P1 */
    if (!done && full_w >= c->minsup && m_rows_ge(c, (double)(csize + ntail) * full_r, csize + ntail, rows, nr, u, ntail)) {          /* P2 */
        int ncl = csize, nfull = 0;
        int *cl = malloc((csize + ntouch + 1) * sizeof(int));
        memcpy(cl, C, csize * sizeof(int));
        int *cnt = calloc(c->m + 1, sizeof(int));
        for (int r = 0; r < nr; r++) {
            if (u[r].cap != ntail) continue;
            nfull++;
            for (int j = 0; j < rows[r].n; j++) cnt[rows[r].it[j]]++;
        }
        for (int i = 0; i < ntouch; i++) if (cnt[c->touched[i]] == nfull) cl[ncl++] = c->touched[i];
        free(cnt);
        qsort(cl, ncl, sizeof(int), m_cmp_int);
        if (!m_has_superset(c, cl, ncl)) m_record(c, cl, ncl);
        free(cl);
        c->pruned_hut++;
        result = done = 1;
    }
    if (!done && c->ncand > 0) {                                                                   /* P3 */
        int hs = csize + ntail;
        int *hut = malloc(hs * sizeof(int));
        memcpy(hut, C, csize * sizeof(int));
        memcpy(hut + csize, tail, ntail * sizeof(int));
        qsort(hut, hs, sizeof(int), m_cmp_int);
        if (m_has_superset(c, hut, hs)) { c->pruned_sub++; result = done = 1; }
        free(hut);
    }

    /* deliver rows to tail items (and the number of tail items after e) */
    int **deliv = NULL, **dcap = NULL, *dn = NULL;
    if (!done) {
        deliv = malloc(ntail * sizeof(int *));
        dcap = malloc(ntail * sizeof(int *));
        dn = calloc(ntail, sizeof(int));
        for (int k = 0; k < ntail; k++) {
            deliv[k] = malloc(c->occ[tail[k]] * sizeof(int));
            dcap[k] = malloc(c->occ[tail[k]] * sizeof(int));
        }
        for (int r = 0; r < nr; r++) {
            int seen = 0;
            for (int j = 0; j < rows[r].n; j++) {
                int p = c->pos[rows[r].it[j]];
                if (!p) continue;
                seen++;
                deliv[p - 1][dn[p - 1]] = r;
                dcap[p - 1][dn[p - 1]] = u[r].cap - seen;
                dn[p - 1]++;
            }
        }
    }
    free(u);
    for (int k = 0; k < ntail; k++) c->pos[tail[k]] = 0;
    for (int i = 0; i < ntouch; i++) c->occ[c->touched[i]] = 0;

    if (!done) {
        int found = 0;
        for (int k = 0; k < ntail; k++) {
            int e = tail[k];
            MUnit *pu = malloc((dn[k] > 0 ? dn[k] : 1) * sizeof(MUnit));                         /* P4 */
            for (int q = 0; q < dn[k]; q++) {
                MRow *rw = &rows[deliv[k][q]];
                pu[q].len = rw->len; pu[q].w = rw->w; pu[q].cap = dcap[k][q];
            }
            double b = m_bound(c, pu, dn[k], csize + 1, 0);
            free(pu);
            if (!m_ge(b, c->xi)) { c->pruned_child++; continue; }

            long se = 0;
            int nt = 0;
            for (int q = 0; q < dn[k]; q++) {
                MRow *rw = &rows[deliv[k][q]];
                se += rw->w;
                for (int j = 0; j < rw->n; j++) {
                    int it = rw->it[j];
                    if (c->occ[it] == 0) c->touched[nt++] = it;
                    c->occ[it] += rw->w;
                }
            }
            /* child C + {e}: rows keep the items after e that are frequent in its cover */
            int nadd = 1;
            int *add = malloc(sizeof(int));
            add[0] = e;
            (void)se;
            for (int i = 0; i < nt; i++) {
                int it = c->touched[i];
                c->keep[it] = it > e && c->occ[it] >= c->minsup;
            }
            for (int i = 0; i < nt; i++) c->occ[c->touched[i]] = 0;
            MRow *cr = malloc((dn[k] > 0 ? dn[k] : 1) * sizeof(MRow));
            int **owned = malloc((dn[k] > 0 ? dn[k] : 1) * sizeof(int *));
            for (int q = 0; q < dn[k]; q++) {
                MRow *rw = &rows[deliv[k][q]];
                int *it2 = malloc((rw->n > 0 ? rw->n : 1) * sizeof(int)), cc = 0;
                for (int j = 0; j < rw->n; j++) if (c->keep[rw->it[j]]) it2[cc++] = rw->it[j];
                cr[q].len = rw->len; cr[q].w = rw->w; cr[q].n = cc; cr[q].it = it2;
                owned[q] = it2;
            }
            for (int i = 0; i < nt; i++) c->keep[c->touched[i]] = 0;
            int ncr = m_merge_rows(cr, dn[k]);

            int ysize = csize + nadd;
            int *Y = malloc((ysize > 0 ? ysize : 1) * sizeof(int));
            memcpy(Y, C, csize * sizeof(int));
            memcpy(Y + csize, add, nadd * sizeof(int));
            qsort(Y, ysize, sizeof(int), m_cmp_int);
            if (m_mine(c, Y, ysize, cr, ncr, e)) found = 1;

            for (int q = 0; q < dn[k]; q++) free(owned[q]);
            free(owned); free(cr); free(Y); free(add);
        }
        for (int k = 0; k < ntail; k++) { free(deliv[k]); free(dcap[k]); }
        free(deliv); free(dcap); free(dn);
        if (found) result = 1;
        else if (is_hoi) { m_record(c, C, csize); result = 1; }
    } else if (!result && is_hoi) {
        m_record(c, C, csize);
        result = 1;
    }
    free(tail);
    return result;
}

typedef struct { int id, sup; } MItemSup;
static int m_cmp_itemsup(const void *a, const void *b) {
    const MItemSup *x = a, *y = b;
    if (x->sup != y->sup) return (x->sup > y->sup) - (x->sup < y->sup);
    return (x->id > y->id) - (x->id < y->id);
}

static MCtx *m_size_ctx;
static int m_cmp_size_desc(const void *a, const void *b) {
    int x = m_size_ctx->size[*(const int *)a], y = m_size_ctx->size[*(const int *)b];
    return (y > x) - (y < x);
}

static DM_Status mhoi_run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_TRANSACTIONAL || !ds->payload || ds->count == 0) return DM_ERROR_INVALID_PARAM;
    DM_MFHOI_Params *p = (DM_MFHOI_Params *)params;
    double arg = p ? p->min_occupancy : 0.01;
    DM_Trans_Simple *src = (DM_Trans_Simple *)ds->payload;
    int n = (int)ds->count, maxid = (int)ds->max_id;

    MCtx c = {0};
    c.xit = dm_threshold_make(arg, (size_t)n, arg < 1.0);
    c.xi = c.xit.value;
    c.minsup = (int)dm_threshold_ceil(c.xit);
    if (c.minsup < 1) c.minsup = 1;
    printf("[MHOI] Starting on %d transactions. xi=%.6f (min support %d)\n", n, c.xi, c.minsup);

    /* distinct items of each transaction, its length, and item supports */
    int *len = malloc(n * sizeof(int));
    int **items = malloc(n * sizeof(int *));
    int *sup = calloc(maxid + 1, sizeof(int));
    for (int t = 0; t < n; t++) {
        int k = (int)src[t].count;
        int *a = malloc((k > 0 ? k : 1) * sizeof(int));
        for (int j = 0; j < k; j++) a[j] = (int)src[t].items[j];
        qsort(a, k, sizeof(int), m_cmp_int);
        int u = 0;
        for (int j = 0; j < k; j++) if (u == 0 || a[j] != a[u - 1]) a[u++] = a[j];
        items[t] = a; len[t] = u;
        for (int j = 0; j < u; j++) sup[a[j]]++;
    }
    /* frequent items, re-indexed by ascending support */
    MItemSup *fi = malloc((maxid + 1) * sizeof(MItemSup));
    int m = 0;
    for (int i = 0; i <= maxid; i++) if (sup[i] >= c.minsup) { fi[m].id = i; fi[m].sup = sup[i]; m++; }
    qsort(fi, m, sizeof(MItemSup), m_cmp_itemsup);
    int *newid = malloc((maxid + 1) * sizeof(int)), *orig = malloc((m > 0 ? m : 1) * sizeof(int));
    for (int i = 0; i <= maxid; i++) newid[i] = -1;
    for (int i = 0; i < m; i++) { newid[fi[i].id] = i; orig[i] = fi[i].id; }
    c.m = m;
    /* the root holds the empty itemset */
    int *root = malloc((m > 0 ? m : 1) * sizeof(int)), nroot = 0;
    char *inroot = calloc(m + 1, 1);

    MRow *rows = malloc(n * sizeof(MRow));
    int **owned = malloc(n * sizeof(int *));
    for (int t = 0; t < n; t++) {
        int *a = malloc((len[t] > 0 ? len[t] : 1) * sizeof(int)), cc = 0;
        for (int j = 0; j < len[t]; j++) {
            int id = newid[items[t][j]];
            if (id >= 0 && !inroot[id]) a[cc++] = id;
        }
        qsort(a, cc, sizeof(int), m_cmp_int);
        rows[t].len = len[t]; rows[t].w = 1; rows[t].n = cc; rows[t].it = a;
        owned[t] = a;
        free(items[t]);
    }
    free(items); free(len); free(sup); free(fi); free(newid); free(inroot);
    int nr = m_merge_rows(rows, n);

    c.occ = calloc(m + 1, sizeof(int));
    c.pos = calloc(m + 1, sizeof(int));
    c.touched = malloc((m + 1) * sizeof(int));
    c.keep = calloc(m + 1, sizeof(int));
    if (n >= c.minsup) m_mine(&c, root, nroot, rows, nr, -1);

    /* keep recorded itemsets that are not subsets of a kept larger one */
    int *order = malloc((c.ncand + 1) * sizeof(int));
    for (int i = 0; i < c.ncand; i++) order[i] = i;
    m_size_ctx = &c;
    qsort(order, c.ncand, sizeof(int), m_cmp_size_desc);
    char *kept = calloc(c.ncand + 1, 1);
    size_t nout = 0, footprint = 0;
    for (int a = 0; a < c.ncand; a++) {
        int i = order[a], dom = 0;
        for (int b = 0; b < a && !dom; b++) {
            int j = order[b];
            if (kept[j] && m_is_subset(c.sets[i], c.size[i], c.sets[j], c.size[j])) dom = 1;
        }
        if (!dom) { kept[i] = 1; nout++; footprint += c.size[i]; }
    }

    /* optional output (one MHOI per line, original item ids), for verification */
    const char *outp = getenv("DM_MHOI_OUT");
    if (outp) {
        FILE *o = fopen(outp, "w");
        for (int i = 0; o && i < c.ncand; i++) {
            if (!kept[i]) continue;
            int *v = malloc((c.size[i] > 0 ? c.size[i] : 1) * sizeof(int));
            for (int k = 0; k < c.size[i]; k++) v[k] = orig[c.sets[i][k]];
            qsort(v, c.size[i], sizeof(int), m_cmp_int);
            for (int k = 0; k < c.size[i]; k++) fprintf(o, k ? " %d" : "%d", v[k]);
            fprintf(o, "\n");
            free(v);
        }
        if (o) fclose(o);
    }

    printf("[MHOI] Complete. MHOIs=%zu recorded=%d nodes=%lld hoi_nodes=%lld pruned: bound=%lld hut=%lld sub=%lld child=%lld\n",
           nout, c.ncand, c.nodes, c.hoi_nodes, c.pruned_bound, c.pruned_hut, c.pruned_sub, c.pruned_child);
    dm_bench_record_results(nout, footprint);

    for (int i = 0; i < c.ncand; i++) free(c.sets[i]);
    free(c.sets); free(c.size); free(order); free(kept);
    for (int t = 0; t < n; t++) free(owned[t]);
    free(owned); free(rows); free(root); free(orig);
    free(c.occ); free(c.pos); free(c.touched); free(c.keep);
    return DM_SUCCESS;
}

/* ══ Legacy two-threshold miner (support and average occupancy) ══════════ */

static int cmp_int_local(const void *a, const void *b) {
    int x = *(const int *)a;
    int y = *(const int *)b;
    return (x > y) - (x < y);
}

static TransactionDB *db_from_dm_dataset(DM_Dataset *ds) {
    DM_Trans_Simple *src = (DM_Trans_Simple *)ds->payload;
    TransactionDB *db = calloc(1, sizeof(TransactionDB));
    db->transaction_count = (int)ds->count;
    db->item_count = (int)ds->max_id + 1;
    db->max_item_id = (int)ds->max_id;
    db->transactions = calloc(ds->count, sizeof(Transaction));
    db->transaction_lengths = calloc(ds->count, sizeof(int));

    for (size_t tid = 0; tid < ds->count; tid++) {
        db->transactions[tid].items = malloc(src[tid].count * sizeof(int));
        for (size_t j = 0; j < src[tid].count; j++) {
            db->transactions[tid].items[j] = (int)src[tid].items[j];
        }
        qsort(db->transactions[tid].items, src[tid].count, sizeof(int), cmp_int_local);
        int unique_len = 0;
        for (size_t j = 0; j < src[tid].count; j++) {
            if (unique_len == 0 || db->transactions[tid].items[j] != db->transactions[tid].items[unique_len - 1]) {
                db->transactions[tid].items[unique_len++] = db->transactions[tid].items[j];
            }
        }
        db->transactions[tid].length = unique_len;
        db->transaction_lengths[tid] = unique_len;
    }
    return db;
}

DM_Status mfhoi_adapter_run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_TRANSACTIONAL || !ds->payload || ds->count == 0) {
        return DM_ERROR_INVALID_PARAM;
    }

    DM_MFHOI_Params *p = (DM_MFHOI_Params *)params;
    double alpha = p ? p->min_support : 0.01;
    double minocc = p ? p->min_occupancy : 0.3;
    int minsup = alpha < 1.0 ? (int)ceil(alpha * (double)ds->count) : (int)ceil(alpha);
    if (minsup < 1) minsup = 1;

    printf("[MFHOI] Starting on %zu transactions. min_support=%d, min_occupancy=%.6g, variant=%s\n",
           ds->count, minsup, minocc, (p && p->strong) ? "strong" : "weak");

    TransactionDB *db = db_from_dm_dataset(ds);
    MFHOILimits limits = { .max_patterns = 2000000, .max_seconds = 0.0 };
    MFHOIResult result;
    int status = mfhoi_mine_all(db, minsup, minocc, &limits, &result);
    MFHOIAlgorithm output_algo = (p && p->strong) ? MFHOI_ALGO_STRONG : MFHOI_ALGO_WEAK;
    if (p && p->output_algorithm >= MFHOI_ALGO_APRIORI && p->output_algorithm <= MFHOI_ALGO_MFI) {
        output_algo = (MFHOIAlgorithm)p->output_algorithm;
    }
    const PatternList *out = mfhoi_select_patterns(&result, output_algo);

    printf("[MFHOI] Complete. Frequent=%d FHOI=%d Weak=%d Strong=%d Output=%d%s\n",
           result.stats.num_frequent_itemsets, result.stats.num_fhoi,
           result.stats.num_weak_mfhoi, result.stats.num_strong_mfhoi,
           out->count, status ? " (limited)" : "");
    printf("[MFHOI] Candidates=%d UB1-pruned=%d UB2-pruned=%d Dominance-removed=%d Weak-removed=%d\n",
           result.stats.num_generated_candidates, result.stats.pruned_ub1_count,
           result.stats.pruned_ub2_count, result.stats.dominance_removed_count,
           result.stats.weak_removed_count);

    int footprint = 0;
    for (int i = 0; i < out->count; i++) footprint += out->patterns[i].length;
    dm_bench_record_results((size_t)out->count, (size_t)footprint);

    mfhoi_result_free(&result);
    mfhoi_free_transaction_db(db);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "mfhoi",
    .name = "MHOI Algorithm",
    .description = "Maximal high occupancy itemsets (occupancy of Deng 2020, one threshold xi).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = mhoi_run
};

DM_REGISTER_ALGORITHM(algo)
