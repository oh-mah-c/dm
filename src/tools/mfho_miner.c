/*
 * mfho_miner: exact mining of Maximal Frequent High-Occupancy Itemsets (MFHOIs).
 *
 * Problem. Given a transaction database D, an absolute support threshold sigma
 * and an occupancy threshold beta in (0, 1], an itemset X is a frequent
 * high-occupancy itemset (FHO) if supp(X) >= sigma and
 *     aocc(X) = |X| / supp(X) * sum_{T in cover(X)} 1/|T|  >= beta,
 * where |T| is the full length of transaction T. An MFHOI is an FHO with no
 * FHO proper superset.
 *
 * Search space. Closure never decreases occupancy, so every MFHOI is a closed
 * itemset. Closed itemsets are enumerated by prefix-preserving closure
 * extension (as in LCM): each frequent closed itemset is generated at most
 * once, and nothing generated earlier has to be stored.
 *
 * Conditional databases. A node C keeps the transactions that contain C,
 * projected on the items that are frequent in that cover and not in C.
 * Projected transactions that have the same items AND the same full length
 * are merged into one weighted row; occupancy and every bound only depend on
 * the multiset of (items, full length), so merging is exact.
 *
 * Pruning (each rule can be switched off for the ablation study, see --mode):
 *   TAIL  size-coupled residual bound. A descendant Y = C + Z, Z within the tail
 *         R (items after the core of C), with |Z| = k >= 1 is supported only by
 *         transactions holding >= k tail items, and by at least sigma of them, so
 *         aocc(Y) <= max_k (|C|+k) * mean of the sigma largest 1/|T| among those
 *         transactions. If the bound is below beta, no descendant is an FHO.
 *         (--mode no_coupled replaces it by the weaker per-transaction bound
 *         mean of the sigma largest min(1, (|C| + cap(T)) / |T|).)
 *   HUT   if C plus all tail items is frequent and high-occupancy, its closure
 *         contains every itemset of the subtree and is an FHO.
 *   SUB   if C plus all tail items lies inside an FHO found earlier, the subtree
 *         contains no new MFHOI.
 * Every FHO node whose subtree contains no FHO becomes a candidate; candidates
 * that are subsets of other candidates are removed at the end.
 *
 * Usage: mfho_miner <db> <min_support (absolute, or relative if < 1)> <beta>
 *                   [--mode full|lcm|no_tail|no_coupled|no_hut|no_sub] [--out file]
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int len;        /* full length of the original transaction */
    int w;          /* number of merged transactions */
    int n;          /* number of items */
    int *it;        /* sorted items */
} Row;

typedef struct {
    int use_tail, use_coupled, use_hut, use_sub, use_pre, use_closure, use_ubo;
    int sigma;
    double beta;
    int sum_mode;   /* 1: high occupancy itemsets of Deng (2020), occ(X) >= xi */
    double xi;
} Config;

typedef struct {
    long long nodes, closure_evals, ppc_rejected;
    long long pruned_tail, pruned_hut, pruned_sub, pruned_pre;
    long long fho_nodes, rows_total, rows_merged_away;
} Stats;

typedef struct {
    int **sets;
    int *size;
    int count, cap;
} Cands;

static Config cfg;
static Stats st;
static Cands cands;
static int m_items;          /* frequent items, re-indexed by ascending support */
static int *orig_id;         /* new id -> original id */
static int *occ;             /* scratch: weighted occurrence count per item */
static int *pos;             /* scratch: item -> tail position + 1 */
static int *touched;
static int *keepflag;        /* scratch: item kept in a child row */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}
static int ge_tol(double x, double t) { return x >= t - 1e-9 * (1.0 + fabs(t)); }
static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

/* ── Rows ──────────────────────────────────────────────────────────────── */
static int cmp_row(const void *a, const void *b) {
    const Row *x = a, *y = b;
    if (x->len != y->len) return (x->len > y->len) - (x->len < y->len);
    if (x->n != y->n) return (x->n > y->n) - (x->n < y->n);
    for (int i = 0; i < x->n; i++)
        if (x->it[i] != y->it[i]) return (x->it[i] > y->it[i]) - (x->it[i] < y->it[i]);
    return 0;
}

/* Sort rows and merge rows with equal (length, items). Returns new count. */
static int merge_rows(Row *r, int nr) {
    if (nr <= 1) return nr;
    qsort(r, nr, sizeof(Row), cmp_row);
    int out = 0;
    for (int i = 0; i < nr; i++) {
        if (out > 0 && cmp_row(&r[out - 1], &r[i]) == 0) { r[out - 1].w += r[i].w; continue; }
        r[out++] = r[i];
    }
    st.rows_merged_away += nr - out;
    return out;
}

/* ── Candidates ────────────────────────────────────────────────────────── */
static int is_subset(const int *a, int na, const int *b, int nb) {
    if (na > nb) return 0;
    int i = 0, j = 0;
    while (i < na && j < nb) {
        if (a[i] == b[j]) { i++; j++; }
        else if (a[i] > b[j]) j++;
        else return 0;
    }
    return i == na;
}
static int cand_has_superset(const int *set, int size) {
    for (int c = 0; c < cands.count; c++)
        if (cands.size[c] >= size && is_subset(set, size, cands.sets[c], cands.size[c])) return 1;
    return 0;
}
static void cand_add(const int *set, int size) {
    if (cands.count == cands.cap) {
        cands.cap = cands.cap ? 2 * cands.cap : 256;
        cands.sets = realloc(cands.sets, cands.cap * sizeof(int *));
        cands.size = realloc(cands.size, cands.cap * sizeof(int));
    }
    int *s = malloc((size > 0 ? size : 1) * sizeof(int));
    memcpy(s, set, size * sizeof(int));
    qsort(s, size, sizeof(int), cmp_int);
    cands.sets[cands.count] = s;
    cands.size[cands.count] = size;
    cands.count++;
}

/* the occupancy test of the objective: average occupancy >= beta, or (sum
 * mode) summed occupancy |X| * sum_{T in cover} 1/|T| >= xi */
static double target(void) { return cfg.sum_mode ? cfg.xi : cfg.beta; }

/* ── Bounds ────────────────────────────────────────────────────────────── */
typedef struct { int len, w, cap; } Unit;
static int cmp_unit_len(const void *a, const void *b) {
    const Unit *x = a, *y = b;
    return (x->len > y->len) - (x->len < y->len);
}

/* Mean of 1/len over the sigma transactions of smallest length among units
 * with cap >= k (units sorted by len ascending); -1 if fewer than sigma. */
static double top_sigma_mean(const Unit *u, int nu, int k) {
    long need = cfg.sigma;
    double s = 0.0;
    for (int i = 0; i < nu && need > 0; i++) {
        if (u[i].cap < k) continue;
        long take = u[i].w < need ? u[i].w : need;
        s += (double)take / u[i].len;
        need -= take;
    }
    return need > 0 ? -1.0 : s / cfg.sigma;
}

/* Size-coupled residual bound: max over k >= kmin of (base + k) times the mean
 * of the sigma largest 1/len among units with cap >= k. With base = |C| and
 * kmin = 1 it bounds every proper descendant of C; with base = |C| + 1 and
 * kmin = 0 it bounds a child C + {e} together with its descendants, cap being
 * the number of tail items after e. Reorders u. */
static double coupled_bound_sum(const Unit *u, int nu, int base, int kmin) {
    /* sum mode: max over k >= kmin of (base + k) * sum of w/len over units with
     * cap >= k (terms with fewer than sigma transactions cannot reach xi) */
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
        if (accw < cfg.sigma) continue;
        double v = (base + k) * acc;
        if (v > best) best = v;
    }
    free(s); free(w);
    return best;
}

/* UBO of Deng (2020): max over lengths l in the cover of l * sum_{|T| >= l} 1/|T|. */
static double ubo_bound(const Unit *u, int nu) {
    int lmax = 0;
    for (int i = 0; i < nu; i++) if (u[i].len > lmax) lmax = u[i].len;
    double *s = calloc(lmax + 2, sizeof(double));
    for (int i = 0; i < nu; i++) s[u[i].len] += (double)u[i].w / u[i].len;
    double best = 0.0, acc = 0.0;
    for (int l = lmax; l >= 1; l--) {
        acc += s[l];
        if (s[l] > 0 && l * acc > best) best = l * acc;
    }
    free(s);
    return best;
}

static double coupled_bound(Unit *u, int nu, int base, int kmin) {
    if (cfg.sum_mode) return coupled_bound_sum(u, nu, base, kmin);
    int cmax = 0;
    long total = 0;
    for (int i = 0; i < nu; i++) { if (u[i].cap > cmax) cmax = u[i].cap; if (u[i].cap >= kmin) total += u[i].w; }
    if (total < cfg.sigma) return 0.0;
    qsort(u, nu, sizeof(Unit), cmp_unit_len);
    double best = 0.0;
    for (int k = cmax; k >= kmin; k--) {
        double mean = top_sigma_mean(u, nu, k);
        if (mean < 0) continue;
        double v = (base + k) * mean;
        if (v > best) { best = v; if (ge_tol(best, cfg.beta)) break; }  /* cannot prune: stop early */
    }
    return best;
}

/* Weaker per-transaction bound: mean of the sigma largest min(1, (|C|+cap)/len). */
typedef struct { double g; int w; } GUnit;
static int cmp_gunit_desc(const void *a, const void *b) {
    double x = ((const GUnit *)a)->g, y = ((const GUnit *)b)->g;
    return (y > x) - (y < x);
}
static double simple_bound(const Unit *u, int nu, int csize) {
    GUnit *g = malloc((nu > 0 ? nu : 1) * sizeof(GUnit));
    int k = 0;
    long total = 0;
    for (int i = 0; i < nu; i++) {
        if (u[i].cap <= 0) continue;
        double v = (double)(csize + u[i].cap) / u[i].len;
        g[k].g = v > 1.0 ? 1.0 : v;
        g[k].w = u[i].w;
        total += u[i].w;
        k++;
    }
    double res = 0.0;
    if (cfg.sum_mode) {
        /* sum mode: sum of the per-transaction values */
        if (total >= cfg.sigma) for (int i = 0; i < k; i++) res += g[i].w * g[i].g;
        free(g);
        return res;
    }
    if (total >= cfg.sigma) {
        qsort(g, k, sizeof(GUnit), cmp_gunit_desc);
        long need = cfg.sigma;
        double s = 0.0;
        for (int i = 0; i < k && need > 0; i++) {
            long take = g[i].w < need ? g[i].w : need;
            s += take * g[i].g;
            need -= take;
        }
        res = s / cfg.sigma;
    }
    free(g);
    return res;
}

/* ── Search ────────────────────────────────────────────────────────────── */
/* C: closed itemset (size csize); rows: cover of C projected on items not in C
 * that are frequent in the cover; core: last item added (prefix order).
 * Returns 1 if the subtree rooted at C (including C) contains an FHO. */
static int mine(const int *C, int csize, Row *rows, int nr, int core) {
    st.nodes++;
    st.rows_total += nr;

    long s = 0;
    double rsum = 0.0;
    int ntouch = 0;
    for (int r = 0; r < nr; r++) {
        s += rows[r].w;
        rsum += (double)rows[r].w / rows[r].len;
        for (int j = 0; j < rows[r].n; j++) {
            int it = rows[r].it[j];
            if (occ[it] == 0) touched[ntouch++] = it;
            occ[it] += rows[r].w;
        }
    }
    int is_fho = csize > 0 && (cfg.sum_mode ? ge_tol((double)csize * rsum, cfg.xi)
                                            : ge_tol((double)csize * rsum / s, cfg.beta));
    if (is_fho) st.fho_nodes++;

    /* tail: frequent items after the core */
    int ntail = 0;
    int *tail = malloc((ntouch > 0 ? ntouch : 1) * sizeof(int));
    for (int i = 0; i < ntouch; i++) {
        int it = touched[i];
        if (it > core && occ[it] >= cfg.sigma) tail[ntail++] = it;
    }
    qsort(tail, ntail, sizeof(int), cmp_int);
    for (int k = 0; k < ntail; k++) pos[tail[k]] = k + 1;

    /* per-row tail capacity */
    Unit *u = malloc((nr > 0 ? nr : 1) * sizeof(Unit));
    long full_w = 0;
    double full_r = 0.0;
    for (int r = 0; r < nr; r++) {
        int c = 0;
        for (int j = 0; j < rows[r].n; j++) if (pos[rows[r].it[j]]) c++;
        u[r].len = rows[r].len; u[r].w = rows[r].w; u[r].cap = c;
        if (ntail > 0 && c == ntail) { full_w += rows[r].w; full_r += (double)rows[r].w / rows[r].len; }
    }

    int result = 0, done = 0;
    if (ntail == 0) done = 1;

    if (!done && cfg.use_tail) {
        /* the bounds reorder their input, so give them a copy */
        Unit *uc = malloc((nr > 0 ? nr : 1) * sizeof(Unit));
        memcpy(uc, u, nr * sizeof(Unit));
        double b = cfg.use_ubo ? ubo_bound(uc, nr)
                 : cfg.use_coupled ? coupled_bound(uc, nr, csize, 1) : simple_bound(uc, nr, csize);
        free(uc);
        if (!ge_tol(b, target())) { st.pruned_tail++; done = 1; }
    }
    if (!done && cfg.use_hut && full_w >= cfg.sigma &&
        (cfg.sum_mode ? ge_tol((double)(csize + ntail) * full_r, cfg.xi)
                      : ge_tol((double)(csize + ntail) * full_r / full_w, cfg.beta))) {
        /* closure of HUT = C + items present in every row that holds all tail items */
        int ncl = 0, nfull = 0;
        int *cl = malloc((csize + ntouch + 1) * sizeof(int));
        memcpy(cl, C, csize * sizeof(int));
        ncl = csize;
        int *cnt = calloc(m_items + 1, sizeof(int));
        for (int r = 0; r < nr; r++) {
            if (u[r].cap != ntail) continue;
            nfull++;
            for (int j = 0; j < rows[r].n; j++) cnt[rows[r].it[j]]++;
        }
        for (int i = 0; i < ntouch; i++) if (cnt[touched[i]] == nfull) cl[ncl++] = touched[i];
        free(cnt);
        qsort(cl, ncl, sizeof(int), cmp_int);
        if (!cand_has_superset(cl, ncl)) cand_add(cl, ncl);
        free(cl);
        st.pruned_hut++;
        result = 1;
        done = 1;
    }
    if (!done && cfg.use_sub && cands.count > 0) {
        int hsize = csize + ntail;
        int *hut = malloc(hsize * sizeof(int));
        memcpy(hut, C, csize * sizeof(int));
        memcpy(hut + csize, tail, ntail * sizeof(int));
        qsort(hut, hsize, sizeof(int), cmp_int);
        if (cand_has_superset(hut, hsize)) { st.pruned_sub++; result = 1; done = 1; }
        free(hut);
    }
    /* Deliver rows to the tail items before the scratch arrays are reused; for
     * each delivered row also record how many tail items follow e in it. */
    int **deliv = NULL, **dcap = NULL, *dn = NULL;
    if (!done) {
        deliv = malloc(ntail * sizeof(int *));
        dcap = malloc(ntail * sizeof(int *));
        dn = calloc(ntail, sizeof(int));
        for (int k = 0; k < ntail; k++) {
            deliv[k] = malloc(occ[tail[k]] * sizeof(int));
            dcap[k] = malloc(occ[tail[k]] * sizeof(int));
        }
        for (int r = 0; r < nr; r++) {
            int seen = 0;
            for (int j = 0; j < rows[r].n; j++) {
                int p = pos[rows[r].it[j]];
                if (!p) continue;
                seen++;
                deliv[p - 1][dn[p - 1]] = r;
                dcap[p - 1][dn[p - 1]] = u[r].cap - seen;
                dn[p - 1]++;
            }
        }
    }
    free(u);
    for (int k = 0; k < ntail; k++) pos[tail[k]] = 0;
    for (int i = 0; i < ntouch; i++) occ[touched[i]] = 0;

    if (!done) {
        int found = 0;
        for (int k = 0; k < ntail; k++) {
            int e = tail[k];
            if (cfg.use_pre) {
                /* bound on C + {e} and all its descendants from the parent's rows */
                Unit *pu = malloc((dn[k] > 0 ? dn[k] : 1) * sizeof(Unit));
                for (int q = 0; q < dn[k]; q++) {
                    Row *rw = &rows[deliv[k][q]];
                    pu[q].len = rw->len; pu[q].w = rw->w; pu[q].cap = dcap[k][q];
                }
                double b = coupled_bound(pu, dn[k], csize + 1, 0);
                free(pu);
                if (!ge_tol(b, target())) { st.pruned_pre++; continue; }
            }
            st.closure_evals++;
            /* occurrences inside the rows holding e */
            long se = 0;
            int nt = 0;
            for (int q = 0; q < dn[k]; q++) {
                Row *rw = &rows[deliv[k][q]];
                se += rw->w;
                for (int j = 0; j < rw->n; j++) {
                    int it = rw->it[j];
                    if (occ[it] == 0) touched[nt++] = it;
                    occ[it] += rw->w;
                }
            }
            /* closure items (present in every row holding e) and prefix-preservation */
            int ppc_ok = 1, nadd = 0;
            int *add = malloc((nt > 0 ? nt : 1) * sizeof(int));
            if (cfg.use_closure) {
                for (int i = 0; i < nt; i++) {
                    int it = touched[i];
                    if (occ[it] == se) {
                        if (it < e) ppc_ok = 0;
                        add[nadd++] = it;
                    }
                }
            } else {
                add[nadd++] = e;   /* plain prefix extension (all itemsets, not only closed) */
            }
            if (!ppc_ok) {
                st.ppc_rejected++;
                for (int i = 0; i < nt; i++) occ[touched[i]] = 0;
                free(add);
                continue;
            }
            /* child rows: drop closure items and items infrequent in the new cover */
            int *keep = keepflag;
            for (int i = 0; i < nt; i++) {
                int it = touched[i];
                keep[it] = cfg.use_closure ? (occ[it] >= cfg.sigma && occ[it] < se)
                                           : (it > e && occ[it] >= cfg.sigma);
            }
            for (int i = 0; i < nt; i++) occ[touched[i]] = 0;
            Row *cr = malloc((dn[k] > 0 ? dn[k] : 1) * sizeof(Row));
            int ncr = 0;
            for (int q = 0; q < dn[k]; q++) {
                Row *rw = &rows[deliv[k][q]];
                int *it2 = malloc((rw->n > 0 ? rw->n : 1) * sizeof(int));
                int c = 0;
                for (int j = 0; j < rw->n; j++) if (keep[rw->it[j]]) it2[c++] = rw->it[j];
                cr[ncr].len = rw->len; cr[ncr].w = rw->w; cr[ncr].n = c; cr[ncr].it = it2;
                ncr++;
            }
            for (int i = 0; i < nt; i++) keep[touched[i]] = 0;
            /* merging may free item arrays of merged rows: keep a list to free */
            int **owned = malloc((ncr > 0 ? ncr : 1) * sizeof(int *));
            for (int q = 0; q < ncr; q++) owned[q] = cr[q].it;
            int ncr2 = merge_rows(cr, ncr);

            int ysize = csize + nadd;
            int *Y = malloc((ysize > 0 ? ysize : 1) * sizeof(int));
            memcpy(Y, C, csize * sizeof(int));
            memcpy(Y + csize, add, nadd * sizeof(int));
            qsort(Y, ysize, sizeof(int), cmp_int);
            if (mine(Y, ysize, cr, ncr2, e)) found = 1;

            for (int q = 0; q < ncr; q++) free(owned[q]);
            free(owned); free(cr); free(Y); free(add);
        }
        for (int k = 0; k < ntail; k++) { free(deliv[k]); free(dcap[k]); }
        free(deliv); free(dcap); free(dn);
        if (found) result = 1;
        else if (is_fho) { cand_add(C, csize); result = 1; }
    } else if (!result && is_fho) {
        cand_add(C, csize);
        result = 1;
    }
    free(tail);
    return result;
}

/* ── Loading ───────────────────────────────────────────────────────────── */
typedef struct { int id, sup; } ItemSup;
static int cmp_itemsup(const void *a, const void *b) {
    const ItemSup *x = a, *y = b;
    if (x->sup != y->sup) return (x->sup > y->sup) - (x->sup < y->sup);
    return (x->id > y->id) - (x->id < y->id);
}

static long peak_rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    char l[256];
    long v = -1;
    while (f && fgets(l, sizeof l, f)) if (!strncmp(l, "VmHWM:", 6)) { sscanf(l + 6, "%ld", &v); break; }
    if (f) fclose(f);
    return v;
}

static int cmp_size_desc(const void *a, const void *b) {
    int x = cands.size[*(const int *)a], y = cands.size[*(const int *)b];
    return (y > x) - (y < x);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s db minsup beta [--mode full|lcm|no_tail|no_coupled|no_hut|no_sub] [--out file]\n", argv[0]);
        return 1;
    }
    const char *mode = "full", *outp = NULL, *seedp = NULL;
    int open_search = 0;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else if (!strcmp(argv[i], "--sum")) cfg.sum_mode = 1;
        else if (!strcmp(argv[i], "--open")) open_search = 1;   /* all itemsets instead of closed ones */
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seedp = argv[++i];
    }
    cfg.use_tail = cfg.use_coupled = cfg.use_hut = cfg.use_sub = cfg.use_pre = cfg.use_closure = 1;
    if (!strcmp(mode, "eclat")) cfg.use_closure = 0;   /* all itemsets, same bounds */
    /* Baselines (same engine, so that visited nodes are comparable):
     *  wubocc : closed enumeration with the per-transaction residual bound, which is
     *           the weak upper bound wubocc of Duong et al. (2024) with unit utilities
     *           (restricted to locally frequent items), plus HUT and superset pruning;
     *  huopm  : enumeration of all FHO itemsets with the same per-transaction bound
     *           (the bound of HUOPM, Gan et al. 2020), followed by maximality filtering. */
    if (!strcmp(mode, "wubocc")) cfg.use_coupled = cfg.use_pre = 0;
    if (!strcmp(mode, "huopm")) cfg.use_coupled = cfg.use_pre = cfg.use_hut = cfg.use_sub = cfg.use_closure = 0;
    /* ubo: closed search, Deng's UBO in rule P1 instead of the size-coupled bound, no P4 */
    if (!strcmp(mode, "ubo")) { cfg.use_ubo = 1; cfg.use_pre = 0; }
    if (!strcmp(mode, "lcm")) cfg.use_tail = cfg.use_coupled = cfg.use_hut = cfg.use_sub = cfg.use_pre = 0;
    else if (!strcmp(mode, "no_pre")) cfg.use_pre = 0;
    else if (!strcmp(mode, "no_tail")) cfg.use_tail = 0;
    else if (!strcmp(mode, "no_coupled")) cfg.use_coupled = cfg.use_pre = 0;
    else if (!strcmp(mode, "no_hut")) cfg.use_hut = 0;
    else if (!strcmp(mode, "no_sub")) cfg.use_sub = 0;
    else if (strcmp(mode, "full") && strcmp(mode, "eclat") && strcmp(mode, "wubocc") && strcmp(mode, "huopm") && strcmp(mode, "ubo")) { fprintf(stderr, "unknown mode %s\n", mode); return 1; }
    if (open_search) cfg.use_closure = 0;
    cfg.beta = atof(argv[3]);
    double minsup_arg = atof(argv[2]);

    double t0 = now_sec();
    /* read */
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    int cap = 1024, n = 0, maxid = 0, bcap = 1 << 16;
    int **raw = malloc(cap * sizeof(int *)), *rawc = malloc(cap * sizeof(int));
    int *buf = malloc(bcap * sizeof(int));
    char *line = NULL;
    size_t lcap = 0;
    while (getline(&line, &lcap, f) > 0) {
        int k = 0;
        char *p = line, *e;
        for (;;) {
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '\n' || *p == '\r' || *p == 0) break;
            long v = strtol(p, &e, 10);
            if (e == p) break;
            if (k == bcap) { bcap *= 2; buf = realloc(buf, bcap * sizeof(int)); }
            buf[k++] = (int)v;
            p = e;
        }
        if (k == 0) continue;
        qsort(buf, k, sizeof(int), cmp_int);
        int u = 0;
        for (int i = 0; i < k; i++) if (i == 0 || buf[i] != buf[i - 1]) buf[u++] = buf[i];
        if (n == cap) { cap *= 2; raw = realloc(raw, cap * sizeof(int *)); rawc = realloc(rawc, cap * sizeof(int)); }
        raw[n] = malloc(u * sizeof(int));
        memcpy(raw[n], buf, u * sizeof(int));
        rawc[n] = u;
        if (buf[u - 1] > maxid) maxid = buf[u - 1];
        n++;
    }
    free(line); free(buf); fclose(f);

    if (cfg.sum_mode) {
        /* argv[2] is the occupancy threshold xi, absolute or (if < 1) a fraction of
         * the number of transactions as in Deng (2020). occ(X) <= supp(X), so every
         * high occupancy itemset has support >= ceil(xi). */
        cfg.xi = minsup_arg < 1.0 ? minsup_arg * n : minsup_arg;
        cfg.sigma = (int)ceil(cfg.xi - 1e-9 * (1.0 + cfg.xi));
    } else
        cfg.sigma = minsup_arg < 1.0 ? (int)ceil(minsup_arg * n - 1e-9) : (int)minsup_arg;
    if (cfg.sigma < 1) cfg.sigma = 1;

    int *sup = calloc(maxid + 1, sizeof(int));
    for (int t = 0; t < n; t++) for (int i = 0; i < rawc[t]; i++) sup[raw[t][i]]++;
    ItemSup *fi = malloc((maxid + 1) * sizeof(ItemSup));
    int m = 0;
    for (int i = 0; i <= maxid; i++) if (sup[i] >= cfg.sigma) { fi[m].id = i; fi[m].sup = sup[i]; m++; }
    qsort(fi, m, sizeof(ItemSup), cmp_itemsup);
    int *newid = malloc((maxid + 1) * sizeof(int));
    for (int i = 0; i <= maxid; i++) newid[i] = -1;
    orig_id = malloc((m > 0 ? m : 1) * sizeof(int));
    for (int i = 0; i < m; i++) { newid[fi[i].id] = i; orig_id[i] = fi[i].id; }
    m_items = m;

    /* root closure: frequent items in every transaction */
    int *root = malloc((m > 0 ? m : 1) * sizeof(int)), nroot = 0;
    if (cfg.use_closure) for (int i = 0; i < m; i++) if (fi[i].sup == n) root[nroot++] = i;
    char *inroot = calloc(m + 1, 1);
    for (int i = 0; i < nroot; i++) inroot[root[i]] = 1;

    /* experimental: itemsets (e.g. maximal frequent itemsets) read from a file;
     * those that pass the occupancy test are stored as candidates before the
     * search, so that rule P3 can use them (an MFI with occ >= xi is an MHOI) */
    long seeds_read = 0, seeds_used = 0;
    if (seedp) {
        FILE *sf = fopen(seedp, "r");
        char *sl = NULL;
        size_t scap = 0;
        int *sv = malloc((m + 1) * sizeof(int));
        char *mark = calloc(maxid + 1, 1);
        while (sf && getline(&sl, &scap, sf) > 0) {
            int k = 0, ok = 1;
            char *p = sl, *e;
            for (;;) {
                long v = strtol(p, &e, 10);
                if (e == p) break;
                p = e;
                if (v < 0 || v > maxid || newid[v] < 0 || k >= m) { ok = 0; break; }
                sv[k++] = (int)v;
                if (*p == '(') break;     /* LCM appends "(support)" */
            }
            if (!ok || k == 0) continue;
            seeds_read++;
            for (int i = 0; i < k; i++) mark[sv[i]] = 1;
            double s = 0; long sp = 0;
            for (int t = 0; t < n; t++) {
                int c = 0;
                for (int i = 0; i < rawc[t]; i++) c += mark[raw[t][i]];
                if (c == k) { s += 1.0 / rawc[t]; sp++; }
            }
            for (int i = 0; i < k; i++) mark[sv[i]] = 0;
            int pass = cfg.sum_mode ? ge_tol(k * s, cfg.xi) : (sp >= cfg.sigma && ge_tol(k * s / sp, cfg.beta));
            if (!pass) continue;
            int *mv = malloc((k + nroot) * sizeof(int)), q = 0;
            for (int i = 0; i < k; i++) if (!inroot[newid[sv[i]]]) mv[q++] = newid[sv[i]];
            for (int i = 0; i < nroot; i++) mv[q++] = root[i];
            qsort(mv, q, sizeof(int), cmp_int);
            cand_add(mv, q);
            free(mv);
            seeds_used++;
        }
        if (sf) fclose(sf);
        free(sl); free(sv); free(mark);
    }

    Row *rows = malloc((n > 0 ? n : 1) * sizeof(Row));
    for (int t = 0; t < n; t++) {
        int c = 0;
        int *arr = malloc((rawc[t] > 0 ? rawc[t] : 1) * sizeof(int));
        for (int i = 0; i < rawc[t]; i++) {
            int id = newid[raw[t][i]];
            if (id >= 0 && !inroot[id]) arr[c++] = id;
        }
        qsort(arr, c, sizeof(int), cmp_int);
        rows[t].len = rawc[t]; rows[t].w = 1; rows[t].n = c; rows[t].it = arr;
        free(raw[t]);
    }
    free(raw); free(rawc); free(sup); free(fi); free(newid); free(inroot);
    int **owned = malloc((n > 0 ? n : 1) * sizeof(int *));
    for (int t = 0; t < n; t++) owned[t] = rows[t].it;
    int nr = merge_rows(rows, n);
    long rss_loaded = peak_rss_kb();
    double t1 = now_sec();

    occ = calloc(m + 1, sizeof(int));
    pos = calloc(m + 1, sizeof(int));
    touched = malloc((m + 1) * sizeof(int));
    keepflag = calloc(m + 1, sizeof(int));
    if (n >= cfg.sigma) mine(root, nroot, rows, nr, -1);
    double t2 = now_sec();

    /* keep candidates that are not subsets of (or equal to) a kept larger one */
    int *order = malloc((cands.count + 1) * sizeof(int));
    for (int i = 0; i < cands.count; i++) order[i] = i;
    qsort(order, cands.count, sizeof(int), cmp_size_desc);
    char *keep = calloc(cands.count + 1, 1);
    int nout = 0;
    for (int a = 0; a < cands.count; a++) {
        int i = order[a], dominated = 0;
        for (int b = 0; b < a && !dominated; b++) {
            int j = order[b];
            if (keep[j] && is_subset(cands.sets[i], cands.size[i], cands.sets[j], cands.size[j])) dominated = 1;
        }
        if (!dominated) { keep[i] = 1; nout++; }
    }
    double t3 = now_sec();

    if (outp) {
        FILE *o = fopen(outp, "w");
        for (int i = 0; i < cands.count; i++) {
            if (!keep[i]) continue;
            int *v = malloc(cands.size[i] * sizeof(int));
            for (int k = 0; k < cands.size[i]; k++) v[k] = orig_id[cands.sets[i][k]];
            qsort(v, cands.size[i], sizeof(int), cmp_int);
            for (int k = 0; k < cands.size[i]; k++) fprintf(o, k ? " %d" : "%d", v[k]);
            fprintf(o, "\n");
            free(v);
        }
        fclose(o);
    }
    printf("mode=%s n=%d frequent_items=%d sigma=%d beta=%g sum=%d xi=%.6f\n", mode, n, m, cfg.sigma, cfg.beta, cfg.sum_mode, cfg.xi);
    printf("mfhoi=%d candidates=%d\n", nout, cands.count);
    printf("nodes=%lld closure_evals=%lld ppc_rejected=%lld fho_nodes=%lld pruned_tail=%lld pruned_hut=%lld pruned_sub=%lld pruned_pre=%lld\n",
           st.nodes, st.closure_evals, st.ppc_rejected, st.fho_nodes, st.pruned_tail, st.pruned_hut, st.pruned_sub, st.pruned_pre);
    if (seedp) printf("seeds_read=%ld seeds_used=%ld\n", seeds_read, seeds_used);
    printf("rows_total=%lld rows_merged_away=%lld\n", st.rows_total, st.rows_merged_away);
    printf("time_load=%.6f time_mine=%.6f time_filter=%.6f\n", t1 - t0, t2 - t1, t3 - t2);
    printf("rss_after_load_kb=%ld peak_rss_kb=%ld\n", rss_loaded, peak_rss_kb());
    for (int t = 0; t < n; t++) free(owned[t]);
    return 0;
}
