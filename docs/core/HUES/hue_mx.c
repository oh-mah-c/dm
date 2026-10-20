/*
 * hue_mx: sound and complete miner for the redefined HUEM problem of HUE-Span
 * (Fournier-Viger et al., ADMA 2019, Definitions 2-4 and 12-14), with incremental structures.
 *
 * Same search space, bounds and pruning as hue_exact.c (Lemma 1, Propositions 3 and 4 of
 * notes_bounds.md: MERU bound, event pruning to a fixpoint, events ranked by increasing event
 * bound); only the computation is incremental:
 *   for every minimal occurrence (MO) with start s of an episode alpha = <SE_1..SE_k>, the array
 *   endU[d] (d = t - s, 0 <= d < maxDur) holds U(s, t) = max over matchings s = t_1 < ... < t_k = t
 *   of sum_i u(SE_i, t_i) (-1 if there is none). Then, for an event x,
 *     serial child <SE_1..SE_k, {x}>:        endU'[t] = max_{t' < t} endU[t'] + u(x, t) if x in tSE_t
 *     simultaneous child (x after SE_k):     endU'[t] = endU[t] + u(x, t)               if x in tSE_t
 *   (a matching of the simultaneous child ending at t is a matching of alpha ending at t plus x
 *   at t). The earliest end of the child from s is the smallest t with endU'[t] >= 0, the MO
 *   utility is endU'[te - s] (Def. 12, reading I1), and the MERU term of s is
 *   max_t endU'[t] + r(t) + W(t + 1, s + maxDur - 1).
 * Minimality: by Lemma 1 the MO starts of a child are among the MO starts of its parent, and
 * earliest ends are non-decreasing over starts, so a child occurrence [s, te] is minimal iff the
 * next parent MO start that has a child occurrence ends later.
 *
 *   gcc -O2 -o hue_mx docs/core/HUES/hue_mx.c
 *   ./hue_mx <input> <minUtilRatio> <maxDur> <output|-> [meru|swu] [prune|noprune]
 *   (ablations: swu = Proposition 2 bound instead of MERU; noprune = no event deletion, events
 *   are still ranked by the event bound of the unpruned sequence)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { int32_t *ev; int64_t *u; int64_t *suf; int n; } TP;   /* ev sorted by rank; suf[j] = sum u[j..] */

static TP *tp;
static int n, D;                 /* D = maxDur */
static int64_t *pref;            /* prefix sums of time point utilities (remaining events) */
static double min_abs;
static FILE *out;
static long hue_count, evaluated, expanded;
static int32_t *orig_id;
static int use_meru = 1, use_prune = 1;

static void *xmalloc(size_t s) { void *p = malloc(s ? s : 1); if (!p) { fputs("out of memory\n", stderr); exit(1); } return p; }
static void *xrealloc(void *p, size_t s) { p = realloc(p, s ? s : 1); if (!p) { fputs("out of memory\n", stderr); exit(1); } return p; }

static inline int lim_of(int s) { int l = s + D - 1; return l > n ? n : l; }
static inline int64_t W(int a, int b) { return a > b ? 0 : pref[b] - pref[a - 1]; }

/* r(t): utility at t of the events ranked after `last` (last is present at t) */
static inline int64_t r_after(int t, int32_t last) {
    const TP *x = &tp[t];
    int lo = 0, hi = x->n - 1;
    while (lo < hi) { int mid = (lo + hi) / 2; if (x->ev[mid] < last) lo = mid + 1; else hi = mid; }
    return lo + 1 < x->n ? x->suf[lo + 1] : 0;
}

/* a node: MOs with starts ms[i] and arrays endU[i * D .. i * D + D) */
typedef struct { int m; int *ms; int64_t *endU; } Occ;
/* episode as a list of sets, sets stored consecutively (ranks) */
typedef struct { int32_t *ev; int *off; int k; } Ep;

static void emit(const Ep *ep, int64_t u) {
    hue_count++;
    if (!out) return;
    int32_t buf[256];
    for (int i = 0; i < ep->k; i++) {
        int m = ep->off[i + 1] - ep->off[i];
        if (m > 256) { fputs("set too large\n", stderr); exit(1); }
        for (int j = 0; j < m; j++) buf[j] = orig_id[ep->ev[ep->off[i] + j]];
        for (int a = 1; a < m; a++) for (int b = a; b > 0 && buf[b - 1] > buf[b]; b--) { int32_t t = buf[b]; buf[b] = buf[b - 1]; buf[b - 1] = t; }
        for (int j = 0; j < m; j++) fprintf(out, "%d ", buf[j]);
        fputs("-1 ", out);
    }
    fprintf(out, "#UTIL: %lld\n", (long long)u);
}

/* child accumulator, indexed by event rank */
typedef struct { int m, cap; int *ms; int64_t *endU; } Acc;

static void acc_open(Acc *a, int s) {               /* start a new MO row for start s if needed */
    if (a->m > 0 && a->ms[a->m - 1] == s) return;
    if (a->m == a->cap) {
        a->cap = a->cap ? 2 * a->cap : 4;
        a->ms = xrealloc(a->ms, sizeof(int) * (size_t)a->cap);
        a->endU = xrealloc(a->endU, sizeof(int64_t) * (size_t)a->cap * (size_t)D);
    }
    a->ms[a->m] = s;
    int64_t *row = a->endU + (size_t)a->m * D;
    for (int d = 0; d < D; d++) row[d] = -1;
    a->m++;
}

static void mine(const Ep *ep, Occ *oc);

/* keep only minimal occurrences in a child accumulator; returns utility and MERU */
static void finish_child(Acc *a, int32_t last, Occ *res, int64_t *u, int64_t *bound) {
    int m = 0;
    *u = 0; *bound = 0;
    int *te = xmalloc(sizeof(int) * (size_t)(a->m ? a->m : 1));
    for (int i = 0; i < a->m; i++) {
        const int64_t *row = a->endU + (size_t)i * D;
        int d = 0; while (d < D && row[d] < 0) d++;
        te[i] = a->ms[i] + d;                          /* rows are only opened when some entry is set */
    }
    for (int i = 0; i < a->m; i++) {
        if (i + 1 < a->m && te[i + 1] <= te[i]) continue;   /* not minimal */
        int s = a->ms[i], lim = lim_of(s);
        const int64_t *row = a->endU + (size_t)i * D;
        *u += row[te[i] - s];
        int64_t best = use_meru ? -1 : W(s, lim);
        for (int t = te[i]; use_meru && t <= lim; t++) {
            int64_t v = row[t - s];
            if (v < 0) continue;
            v += r_after(t, last) + W(t + 1, lim);
            if (v > best) best = v;
        }
        *bound += best;
        if (m != i) { a->ms[m] = a->ms[i]; memcpy(a->endU + (size_t)m * D, row, sizeof(int64_t) * (size_t)D); }
        m++;
    }
    free(te);
    res->m = m; res->ms = a->ms; res->endU = a->endU;
}

static void recurse_child(const Ep *ep, int32_t x, int serial, Acc *a) {
    Occ oc; int64_t u, bound;
    finish_child(a, x, &oc, &u, &bound);
    evaluated++;
    Ep c;
    int tot = ep->off[ep->k] + 1;
    c.ev = xmalloc(sizeof(int32_t) * (size_t)tot);
    c.off = xmalloc(sizeof(int) * (size_t)(ep->k + 2));
    memcpy(c.ev, ep->ev, sizeof(int32_t) * (size_t)ep->off[ep->k]);
    memcpy(c.off, ep->off, sizeof(int) * (size_t)(ep->k + 1));
    c.ev[tot - 1] = x;
    c.k = serial ? ep->k + 1 : ep->k;
    c.off[c.k] = tot;
    if ((double)u >= min_abs) emit(&c, u);
    if ((double)bound >= min_abs && oc.m > 0) mine(&c, &oc);
    free(c.ev); free(c.off);
}

static int n_ranks;
/* slot[2 * x + serial] = index of the accumulator of event x in the node being scanned, or -1.
   A node finishes scanning before any child is mined, so one global array is enough. */
static int *slot;

static void mine(const Ep *ep, Occ *oc) {
    expanded++;
    int32_t last = ep->ev[ep->off[ep->k] - 1];
    int na = 0, ca = 16;
    Acc *acc = xmalloc(sizeof(Acc) * (size_t)ca);
    int32_t *key = xmalloc(sizeof(int32_t) * (size_t)ca);     /* 2 * x + serial */
    int64_t *pm = xmalloc(sizeof(int64_t) * (size_t)(D + 1));
    for (int i = 0; i < oc->m; i++) {
        int s = oc->ms[i], lim = lim_of(s);
        const int64_t *row = oc->endU + (size_t)i * D;
        pm[0] = -1;                                            /* pm[d] = max_{d' < d} row[d'] */
        for (int d = 1; d <= D; d++) pm[d] = row[d - 1] > pm[d - 1] ? row[d - 1] : pm[d - 1];
        for (int t = s; t <= lim; t++) {
            int d = t - s;
            const TP *x = &tp[t];
            for (int serial = 0; serial < 2; serial++) {
                int64_t base = serial ? pm[d] : row[d];
                if (base < 0) continue;
                int lo = 0;
                if (!serial) {                                 /* simultaneous: events ranked after last */
                    int hi = x->n;
                    while (lo < hi) { int mid = (lo + hi) / 2; if (x->ev[mid] <= last) lo = mid + 1; else hi = mid; }
                }
                for (int j = lo; j < x->n; j++) {
                    int k = 2 * x->ev[j] + serial;
                    if (slot[k] < 0) {
                        if (na == ca) { ca *= 2; acc = xrealloc(acc, sizeof(Acc) * (size_t)ca); key = xrealloc(key, sizeof(int32_t) * (size_t)ca); }
                        memset(&acc[na], 0, sizeof(Acc));
                        key[na] = k; slot[k] = na++;
                    }
                    Acc *a = &acc[slot[k]];
                    acc_open(a, s);
                    int64_t v = base + x->u[j], *cell = &a->endU[(size_t)(a->m - 1) * D + d];
                    if (v > *cell) *cell = v;
                }
            }
        }
    }
    free(pm);
    for (int i = 0; i < na; i++) slot[key[i]] = -1;
    for (int i = 0; i < na; i++) {
        recurse_child(ep, key[i] >> 1, key[i] & 1, &acc[i]);
        free(acc[i].ms); free(acc[i].endU);
    }
    free(acc); free(key);
}

static int cmp_pair(const void *a, const void *b) { const int64_t *x = a, *y = b; return (x[0] > y[0]) - (x[0] < y[0]); }
static const double *g_eb;
static int cmp_eb(const void *a, const void *b) {
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    if (g_eb[x] != g_eb[y]) return g_eb[x] < g_eb[y] ? -1 : 1;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <input> <minUtilRatio> <maxDur> <output|->\n", argv[0]); return 2; }
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    D = atoi(argv[3]);
    if (argc > 5) use_meru = strcmp(argv[5], "swu") != 0;
    if (argc > 6) use_prune = strcmp(argv[6], "noprune") != 0;
    int cap = 1024;
    tp = xmalloc(sizeof(TP) * (size_t)cap);
    n = 0;
    int64_t total = 0; int32_t max_ev = 0;
    char *line = NULL; size_t lc = 0;
    while (getline(&line, &lc, f) >= 0) {
        if (++n >= cap) { cap *= 2; tp = xrealloc(tp, sizeof(TP) * (size_t)cap); }
        TP *x = &tp[n]; x->n = 0; x->ev = NULL; x->u = NULL; x->suf = NULL;
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '%' || *p == '@' || *p == '\n' || *p == '\r' || *p == 0) continue;
        char *c1 = strchr(p, ':'), *c2 = c1 ? strchr(c1 + 1, ':') : NULL;
        if (!c2) continue;
        *c1 = 0; *c2 = 0;
        int ne = 0; for (char *q = p; *q; q++) if ((q == p || q[-1] == ' ') && *q != ' ') ne++;
        int64_t *pairs = xmalloc(sizeof(int64_t) * 2 * (size_t)(ne ? ne : 1));
        char *q = p, *r = c2 + 1;
        for (int i = 0; i < ne; i++) { pairs[2 * i] = strtol(q, &q, 10); pairs[2 * i + 1] = strtoll(r, &r, 10); }
        qsort(pairs, (size_t)ne, sizeof(int64_t) * 2, cmp_pair);
        x->ev = xmalloc(sizeof(int32_t) * (size_t)(ne ? ne : 1));
        x->u = xmalloc(sizeof(int64_t) * (size_t)(ne ? ne : 1));
        for (int i = 0; i < ne; i++) {
            if (x->n > 0 && x->ev[x->n - 1] == (int32_t)pairs[2 * i]) { x->u[x->n - 1] += pairs[2 * i + 1]; continue; }
            x->ev[x->n] = (int32_t)pairs[2 * i]; x->u[x->n] = pairs[2 * i + 1]; x->n++;
            if ((int32_t)pairs[2 * i] > max_ev) max_ev = (int32_t)pairs[2 * i];
        }
        total += strtoll(c1 + 1, NULL, 10);
        free(pairs);
    }
    free(line); fclose(f);
    min_abs = (double)total * atof(argv[2]);
    out = strcmp(argv[4], "-") ? fopen(argv[4], "w") : NULL;
    pref = xmalloc(sizeof(int64_t) * (size_t)(n + 1));
    pref[0] = 0;
    for (int t = 1; t <= n; t++) { int64_t s = 0; for (int j = 0; j < tp[t].n; j++) s += tp[t].u[j]; pref[t] = pref[t - 1] + s; }

    /* event pruning (Proposition 4) to a fixpoint, then ranks by increasing event bound */
    int32_t n_ev = max_ev + 1;
    double *eb = calloc((size_t)n_ev, sizeof(double));
    uint8_t *alive = malloc((size_t)n_ev); memset(alive, 1, (size_t)n_ev);
    int *lastS = malloc(sizeof(int) * (size_t)n_ev);
    long removed = 0, rounds = 0;
    for (int changed = 1; changed; ) {
        changed = 0; rounds++;
        int64_t *wp = xmalloc(sizeof(int64_t) * (size_t)(n + 1));
        wp[0] = 0;
        for (int t = 1; t <= n; t++) wp[t] = wp[t - 1] + W(t, lim_of(t));
        for (int32_t e = 0; e < n_ev; e++) { eb[e] = 0; lastS[e] = 0; }
        for (int t = 1; t <= n; t++)
            for (int j = 0; j < tp[t].n; j++) {
                int32_t e = tp[t].ev[j];
                int a = t - D + 1; if (a < 1) a = 1;
                if (a <= lastS[e]) a = lastS[e] + 1;
                if (a <= t) eb[e] += (double)(wp[t] - wp[a - 1]);
                lastS[e] = t;
            }
        free(wp);
        for (int t = 1; t <= n; t++) {
            int w = 0;
            for (int j = 0; j < tp[t].n; j++) {
                int32_t e = tp[t].ev[j];
                if (use_prune && alive[e] && eb[e] < min_abs) { alive[e] = 0; removed++; changed = 1; }
                if (alive[e]) { tp[t].ev[w] = e; tp[t].u[w] = tp[t].u[j]; w++; }
            }
            tp[t].n = w;
        }
        for (int t = 1; t <= n; t++) { int64_t s = 0; for (int j = 0; j < tp[t].n; j++) s += tp[t].u[j]; pref[t] = pref[t - 1] + s; }
    }
    free(lastS);
    int32_t *order = malloc(sizeof(int32_t) * (size_t)n_ev), n_alive = 0;
    for (int32_t e = 0; e < n_ev; e++) if (alive[e]) order[n_alive++] = e;
    g_eb = eb;
    qsort(order, (size_t)n_alive, sizeof(int32_t), cmp_eb);
    int32_t *rank = malloc(sizeof(int32_t) * (size_t)n_ev);
    orig_id = malloc(sizeof(int32_t) * (size_t)(n_alive ? n_alive : 1));
    for (int32_t r = 0; r < n_alive; r++) { rank[order[r]] = r; orig_id[r] = order[r]; }
    for (int t = 1; t <= n; t++) {
        TP *x = &tp[t];
        for (int j = 0; j < x->n; j++) x->ev[j] = rank[x->ev[j]];
        for (int a = 1; a < x->n; a++)
            for (int b = a; b > 0 && x->ev[b - 1] > x->ev[b]; b--) {
                int32_t te = x->ev[b]; x->ev[b] = x->ev[b - 1]; x->ev[b - 1] = te;
                int64_t tu = x->u[b]; x->u[b] = x->u[b - 1]; x->u[b - 1] = tu;
            }
        x->suf = xmalloc(sizeof(int64_t) * (size_t)(x->n ? x->n : 1));
        int64_t s = 0;
        for (int j = x->n - 1; j >= 0; j--) { s += x->u[j]; x->suf[j] = s; }
    }
    n_ranks = n_alive;
    free(rank); free(order); free(alive); free(eb);

    /* single events: every occurrence [t, t] is minimal, endU[0] = u(x, t) */
    slot = malloc(sizeof(int) * 2 * (size_t)(n_alive > 0 ? n_alive : 1));
    for (int32_t i = 0; i < 2 * n_alive; i++) slot[i] = -1;
    Acc *single = calloc((size_t)(n_alive ? n_alive : 1), sizeof(Acc));
    for (int t = 1; t <= n; t++)
        for (int j = 0; j < tp[t].n; j++) {
            Acc *a = &single[tp[t].ev[j]];
            acc_open(a, t);
            a->endU[(size_t)(a->m - 1) * D] = tp[t].u[j];
        }
    for (int32_t x = 0; x < n_alive; x++) {
        Acc *a = &single[x];
        if (a->m == 0) continue;
        Ep ep; int32_t ev = x; int off[2] = {0, 1};
        ep.ev = &ev; ep.off = off; ep.k = 1;
        Occ oc; int64_t u, bound;
        finish_child(a, x, &oc, &u, &bound);
        evaluated++;
        if ((double)u >= min_abs) emit(&ep, u);
        if ((double)bound >= min_abs && oc.m > 0) mine(&ep, &oc);
        free(a->ms); free(a->endU);
    }
    free(single); free(slot);
    if (out) fclose(out);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("[hue_mx] bound=%s prune=%d time points=%d total=%lld minUtilAbs=%.3f maxDur=%d removed_events=%ld rounds=%ld "
           "HUEs=%ld evaluated=%ld expanded=%ld time_s=%.3f\n", use_meru ? "meru" : "swu", use_prune, n, (long long)total, min_abs, D, removed, rounds,
           hue_count, evaluated, expanded, (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9);
    return 0;
}
