/*
 * hue_exact: reference miner for the redefined HUEM problem of HUE-Span (Fournier-Viger et al.,
 * ADMA 2019, Definitions 2-4 and 12-14), written for auditing, not for speed.
 *
 * Correctness rests only on the statements proved in notes_bounds.md:
 *   - Lemma 1: every minimal occurrence (MO, length <= maxDur) of a descendant beta of alpha
 *     starts at the start of an MO of alpha;
 *   - Proposition 2: SWU(alpha) = sum over MOs [s, e] of alpha of the sequence utility of
 *     [s, s + maxDur - 1] bounds u(beta) for alpha and every descendant beta;
 *   - Proposition 3: MERU(alpha) (ERU maximised over every feasible end t' of alpha's matching)
 *     is a tighter bound of the same kind. Bound selected by the 5th argument: swu | meru.
 * No AWU single-event pruning, no co-occurrence pruning, no ERU.
 * Utility of an MO = max over matchings s = t_1 < ... < t_k = e of sum_i u(SE_i, t_i) (DP).
 *
 * Input: SPMF "events:total:utilities", one time point per physical line (lines starting with
 * '#', '%' or '@' and empty lines are empty time points, as in the reference implementation).
 * Output: one episode per line, sets in increasing identifier order: "a b -1 c -1 #UTIL: u".
 *
 *   gcc -O2 -o hue_exact docs/core/HUES/hue_exact.c
 *   ./hue_exact <input> <minUtilRatio> <maxDur> <output|-> [swu|meru] [prune|noprune]
 *   (defaults: meru, prune; Proposition 4 event pruning and ranking by increasing event bound)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int32_t *ev; int64_t *u; int n; } TP;   /* events sorted by id */

static TP *tp;            /* 1..n */
static int n;
static int64_t *pref;     /* pref[t] = sum of u(tSE_1..t) */
static int max_dur;
static double min_abs;
static FILE *out;
static long hue_count, nodes;
static int use_meru = 1, use_prune = 1;
static int32_t *orig_id;   /* internal rank -> input identifier (identity without pruning) */

static void *xmalloc(size_t s) { void *p = malloc(s ? s : 1); if (!p) { fputs("out of memory\n", stderr); exit(1); } return p; }
static void *xrealloc(void *p, size_t s) { p = realloc(p, s ? s : 1); if (!p) { fputs("out of memory\n", stderr); exit(1); } return p; }

/* u(set, t), -1 if set is not a subset of tSE_t */
static int64_t u_set(const int32_t *set, int k, int t) {
    const TP *x = &tp[t];
    int64_t s = 0; int j = 0;
    for (int i = 0; i < k; i++) {
        while (j < x->n && x->ev[j] < set[i]) j++;
        if (j == x->n || x->ev[j] != set[i]) return -1;
        s += x->u[j++];
    }
    return s;
}

/* episode: sets stored consecutively */
typedef struct { int32_t *ev; int *off; int k; } Ep;   /* set i = ev[off[i]..off[i+1]) */

static int64_t win(int s) { int e = s + max_dur - 1; if (e > n) e = n; return pref[e] - pref[s - 1]; }

/* earliest end of an occurrence of ep starting at s, within s + maxDur - 1; INT32_MAX if none */
static int earliest_end(const Ep *ep, int s) {
    if (u_set(ep->ev + ep->off[0], ep->off[1] - ep->off[0], s) < 0) return INT32_MAX;
    int t = s, lim = s + max_dur - 1;
    if (lim > n) lim = n;
    for (int i = 1; i < ep->k; i++) {
        do { t++; } while (t <= lim && u_set(ep->ev + ep->off[i], ep->off[i + 1] - ep->off[i], t) < 0);
        if (t > lim) return INT32_MAX;
    }
    return t;
}

static int64_t mo_utility(const Ep *ep, int s, int e) {
    int k = ep->k;
    int64_t best[64];
    if (k > 63) { fputs("episode too long\n", stderr); exit(1); }
    for (int i = 0; i <= k; i++) best[i] = -1;
    best[1] = u_set(ep->ev + ep->off[0], ep->off[1] - ep->off[0], s);
    if (k == 1) return best[1];
    for (int t = s + 1; t <= e; t++)
        for (int i = (t - s + 1 < k ? t - s + 1 : k); i >= 2; i--) {
            if (best[i - 1] < 0 || (i == k && t != e)) continue;
            int64_t u = u_set(ep->ev + ep->off[i - 1], ep->off[i] - ep->off[i - 1], t);
            if (u >= 0 && best[i - 1] + u > best[i]) best[i] = best[i - 1] + u;
        }
    return best[k];
}

/* term of Proposition 3 for the MO start s: max over ends t' of U(s, t') + r(t') + W(t' + 1, lim) */
static int64_t meru_term(const Ep *ep, int s) {
    int k = ep->k, lim = s + max_dur - 1;
    if (lim > n) lim = n;
    const int32_t *P = ep->ev + ep->off[k - 1];
    int np = ep->off[k] - ep->off[k - 1];
    int32_t pmax = P[np - 1];
    int64_t best[64], res = -1;
    for (int i = 0; i <= k; i++) best[i] = -1;
    best[1] = u_set(ep->ev + ep->off[0], ep->off[1] - ep->off[0], s);
    for (int t = s; t <= lim; t++) {
        int64_t endv = -1;
        if (t == s) endv = k == 1 ? best[1] : -1;
        else
            for (int i = (t - s + 1 < k ? t - s + 1 : k); i >= 2; i--) {
                if (best[i - 1] < 0) continue;
                int64_t u = u_set(ep->ev + ep->off[i - 1], ep->off[i] - ep->off[i - 1], t);
                if (u < 0) continue;
                if (i == k) { if (best[i - 1] + u > endv) endv = best[i - 1] + u; }
                else if (best[i - 1] + u > best[i]) best[i] = best[i - 1] + u;
            }
        if (endv < 0) continue;
        int64_t r = 0;
        for (int j = 0; j < tp[t].n; j++) if (tp[t].ev[j] > pmax) r += tp[t].u[j];
        int64_t v = endv + r + (pref[lim] - pref[t]);
        if (v > res) res = v;
    }
    return res;
}

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

/* MOs of ep among candidate starts cs[0..nc) (increasing); writes starts/ends, returns count */
static int compute_mos(const Ep *ep, const int *cs, int nc, int *ms, int *me) {
    int m = 0;
    int *ends = xmalloc(sizeof(int) * (size_t)(nc ? nc : 1));
    for (int i = 0; i < nc; i++) ends[i] = earliest_end(ep, cs[i]);
    for (int i = 0; i < nc; i++) {
        if (ends[i] == INT32_MAX) continue;
        /* earliest ends are non-decreasing over starts that have an occurrence, so only the next
           candidate start that has one can end no later (Lemma 1: candidates contain all MO starts) */
        int j = i + 1;
        while (j < nc && ends[j] == INT32_MAX) j++;
        if (j < nc && ends[j] <= ends[i]) continue;
        ms[m] = cs[i]; me[m] = ends[i]; m++;
    }
    free(ends);
    return m;
}

static int cmp_int(const void *a, const void *b) { int x = *(const int *)a, y = *(const int *)b; return (x > y) - (x < y); }

static void mine(const Ep *ep, const int *ms, const int *me, int m);

/* child = ep with event x added (simultaneous) or appended as a new set (serial) */
static void child(const Ep *ep, int32_t x, int serial, const int *ms, int m) {
    Ep c;
    int tot = ep->off[ep->k] + 1;
    c.ev = xmalloc(sizeof(int32_t) * (size_t)tot);
    c.off = xmalloc(sizeof(int) * (size_t)(ep->k + 2));
    memcpy(c.ev, ep->ev, sizeof(int32_t) * (size_t)ep->off[ep->k]);
    memcpy(c.off, ep->off, sizeof(int) * (size_t)(ep->k + 1));
    c.ev[tot - 1] = x;
    if (serial) { c.k = ep->k + 1; c.off[c.k] = tot; }
    else { c.k = ep->k; c.off[c.k] = tot; }
    int *cs = xmalloc(sizeof(int) * (size_t)m), *ce = xmalloc(sizeof(int) * (size_t)m);
    int cm = compute_mos(&c, ms, m, cs, ce);
    if (cm > 0) mine(&c, cs, ce, cm);
    free(cs); free(ce); free(c.ev); free(c.off);
}

static void mine(const Ep *ep, const int *ms, const int *me, int m) {
    nodes++;
    int64_t u = 0, swu = 0;
    for (int i = 0; i < m; i++) { u += mo_utility(ep, ms[i], me[i]); swu += use_meru ? meru_term(ep, ms[i]) : win(ms[i]); }
    if ((double)u >= min_abs) emit(ep, u);
    if ((double)swu < min_abs) return;
    /* candidate events: simultaneous x > last event of the last set, at a time point of
       [s, s + maxDur - 1]; serial x at a time point of (s, s + maxDur - 1] */
    int32_t last = ep->ev[ep->off[ep->k] - 1];
    int cap = 64, nsim = 0, nser = 0;
    int32_t *sim = xmalloc(sizeof(int32_t) * (size_t)cap), *ser = xmalloc(sizeof(int32_t) * (size_t)cap);
    int capr = 64;
    for (int i = 0; i < m; i++) {
        int lim = ms[i] + max_dur - 1; if (lim > n) lim = n;
        for (int t = ms[i]; t <= lim; t++)
            for (int j = 0; j < tp[t].n; j++) {
                int32_t x = tp[t].ev[j];
                if (x > last) { if (nsim == cap) { cap *= 2; sim = xrealloc(sim, sizeof(int32_t) * (size_t)cap); } sim[nsim++] = x; }
                if (t > ms[i]) { if (nser == capr) { capr *= 2; ser = xrealloc(ser, sizeof(int32_t) * (size_t)capr); } ser[nser++] = x; }
            }
    }
    qsort(sim, (size_t)nsim, sizeof(int32_t), cmp_int);
    qsort(ser, (size_t)nser, sizeof(int32_t), cmp_int);
    for (int i = 0; i < nsim; i++) if (i == 0 || sim[i] != sim[i - 1]) child(ep, sim[i], 0, ms, m);
    for (int i = 0; i < nser; i++) if (i == 0 || ser[i] != ser[i - 1]) child(ep, ser[i], 1, ms, m);
    free(sim); free(ser);
}

static const double *g_eb;
static int cmp_eb(const void *a, const void *b) {
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    if (g_eb[x] != g_eb[y]) return g_eb[x] < g_eb[y] ? -1 : 1;
    return (x > y) - (x < y);
}

static int cmp_pair(const void *a, const void *b) {
    const int64_t *x = a, *y = b;
    return (x[0] > y[0]) - (x[0] < y[0]);
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <input> <minUtilRatio> <maxDur> <output|->\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    max_dur = atoi(argv[3]);
    if (argc > 5) use_meru = strcmp(argv[5], "swu") != 0;
    if (argc > 6) use_prune = strcmp(argv[6], "noprune") != 0;
    int cap = 1024;
    tp = xmalloc(sizeof(TP) * (size_t)cap);
    n = 0;
    int64_t total = 0;
    int32_t max_ev = 0;
    char *line = NULL; size_t lc = 0;
    while (getline(&line, &lc, f) >= 0) {
        if (++n >= cap) { cap *= 2; tp = xrealloc(tp, sizeof(TP) * (size_t)cap); }
        TP *x = &tp[n]; x->n = 0; x->ev = NULL; x->u = NULL;
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '%' || *p == '@' || *p == '\n' || *p == '\r' || *p == 0) continue;
        char *c1 = strchr(p, ':'), *c2 = c1 ? strchr(c1 + 1, ':') : NULL;
        if (!c2) continue;
        *c1 = 0; *c2 = 0;
        int ne = 0; for (char *q = p; *q; q++) if ((q == p || q[-1] == ' ') && *q != ' ') ne++;
        int64_t *pairs = xmalloc(sizeof(int64_t) * 2 * (size_t)(ne ? ne : 1));
        char *q = p, *r = c2 + 1;
        for (int i = 0; i < ne; i++) {
            pairs[2 * i] = strtol(q, &q, 10);
            pairs[2 * i + 1] = strtoll(r, &r, 10);
        }
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
    pref = xmalloc(sizeof(int64_t) * (size_t)(n + 1));
    pref[0] = 0;
    for (int t = 1; t <= n; t++) {
        int64_t s = 0; for (int j = 0; j < tp[t].n; j++) s += tp[t].u[j];
        pref[t] = pref[t - 1] + s;
    }
    min_abs = (double)total * atof(argv[2]);
    out = strcmp(argv[4], "-") ? fopen(argv[4], "w") : NULL;

    /* Event pruning (Proposition 4, notes_bounds.md): every counted MO [s, e] of an episode that
       contains event x includes an occurrence of x in [s, s + maxDur - 1], and MOs have distinct
       starts, so u <= EB(x) = sum over the distinct s with x in [s, s + maxDur - 1] of
       W(s, s + maxDur - 1). Events with EB(x) < minUtil are in no HUE and are deleted; W only
       needs the remaining events, so this is repeated until nothing changes. The remaining
       events are then ranked by increasing EB (ties by identifier); the search order uses ranks. */
    int32_t n_ev = max_ev + 1;
    double *eb = calloc((size_t)n_ev, sizeof(double));
    uint8_t *alive = malloc((size_t)n_ev);
    memset(alive, 1, (size_t)n_ev);
    long removed_events = 0, rounds = 0;
    int *lastS = malloc(sizeof(int) * (size_t)n_ev);
    for (int changed = use_prune; changed; ) {
        changed = 0; rounds++;
        int64_t *wp = xmalloc(sizeof(int64_t) * (size_t)(n + 1));  /* prefix of window values */
        wp[0] = 0;
        for (int t = 1; t <= n; t++) wp[t] = wp[t - 1] + win(t);
        for (int32_t e = 0; e < n_ev; e++) { eb[e] = 0; lastS[e] = 0; }
        /* for each event, union of [t - maxDur + 1, t] over its occurrences, scanned in time order */
        for (int t = 1; t <= n; t++)
            for (int j = 0; j < tp[t].n; j++) {
                int32_t e = tp[t].ev[j];
                int a = t - max_dur + 1; if (a < 1) a = 1;
                if (a <= lastS[e]) a = lastS[e] + 1;
                if (a <= t) eb[e] += (double)(wp[t] - wp[a - 1]);
                lastS[e] = t;
            }
        free(wp);
        for (int t = 1; t <= n; t++) {
            int w = 0;
            for (int j = 0; j < tp[t].n; j++) {
                int32_t e = tp[t].ev[j];
                if (alive[e] && eb[e] < min_abs) { alive[e] = 0; removed_events++; changed = 1; }
                if (alive[e]) { tp[t].ev[w] = e; tp[t].u[w] = tp[t].u[j]; w++; }
            }
            tp[t].n = w;
        }
        for (int t = 1; t <= n; t++) {
            int64_t sum = 0; for (int j = 0; j < tp[t].n; j++) sum += tp[t].u[j];
            pref[t] = pref[t - 1] + sum;
        }
    }
    free(lastS);
    /* ranks */
    int32_t *order = malloc(sizeof(int32_t) * (size_t)n_ev), n_alive = 0;
    for (int32_t e = 0; e < n_ev; e++) if (alive[e]) order[n_alive++] = e;
    g_eb = eb;
    if (use_prune) qsort(order, (size_t)n_alive, sizeof(int32_t), cmp_eb);
    int32_t *rank = malloc(sizeof(int32_t) * (size_t)n_ev);
    orig_id = malloc(sizeof(int32_t) * (size_t)(n_alive ? n_alive : 1));
    for (int32_t r = 0; r < n_alive; r++) { rank[order[r]] = r; orig_id[r] = order[r]; }
    for (int t = 1; t <= n; t++) {
        for (int j = 0; j < tp[t].n; j++) tp[t].ev[j] = rank[tp[t].ev[j]];
        for (int a = 1; a < tp[t].n; a++)            /* re-sort by rank (insertion sort, short sets) */
            for (int b = a; b > 0 && tp[t].ev[b - 1] > tp[t].ev[b]; b--) {
                int32_t te = tp[t].ev[b]; tp[t].ev[b] = tp[t].ev[b - 1]; tp[t].ev[b - 1] = te;
                int64_t tu = tp[t].u[b]; tp[t].u[b] = tp[t].u[b - 1]; tp[t].u[b - 1] = tu;
            }
    }
    max_ev = n_alive - 1;
    free(rank); free(order); free(alive); free(eb);

    /* occurrence lists of single events (each occurrence [t, t] is minimal) */
    int *cnt = calloc((size_t)max_ev + 1, sizeof(int));
    for (int t = 1; t <= n; t++) for (int j = 0; j < tp[t].n; j++) cnt[tp[t].ev[j]]++;
    int **occ = calloc((size_t)max_ev + 1, sizeof(int *));
    int *fill = calloc((size_t)max_ev + 1, sizeof(int));
    for (int32_t e = 0; e <= max_ev; e++) if (cnt[e]) occ[e] = xmalloc(sizeof(int) * (size_t)cnt[e]);
    for (int t = 1; t <= n; t++) for (int j = 0; j < tp[t].n; j++) { int32_t e = tp[t].ev[j]; occ[e][fill[e]++] = t; }
    for (int32_t e = 0; e <= max_ev; e++) {
        if (!cnt[e]) continue;  /* e is a rank */
        Ep ep; int32_t ev = e; int off[2] = {0, 1};
        ep.ev = &ev; ep.off = off; ep.k = 1;
        mine(&ep, occ[e], occ[e], cnt[e]);
        free(occ[e]);
    }
    if (out) fclose(out);
    printf("[hue_exact] bound=%s prune=%d removed_events=%ld rounds=%ld time points=%d total=%lld minUtilAbs=%.3f maxDur=%d HUEs=%ld nodes=%ld\n",
           use_meru ? "meru" : "swu", use_prune, removed_events, rounds, n, (long long)total, min_abs, max_dur, hue_count, nodes);
    return 0;
}
