/*
 * hoi_filter: two-phase baselines for Maximal Frequent High-Occupancy Itemsets
 * built on published high-occupancy itemset (HOI) miners.
 *
 * An HOI miner reports every X with occ(X) = sum_{T in cover(X)} |X|/|T| >= xi.
 * Since occ(X) = supp(X) * aocc(X), every FHO (supp >= sigma, aocc >= beta)
 * satisfies occ(X) >= sigma * beta. Hence
 *   phase 1: run the HOI miner with xi = sigma * beta;
 *   phase 2: keep the HOIs with supp >= sigma and aocc >= beta (the FHOs) and
 *            remove those contained in another FHO.
 * The result is exactly the set of MFHOIs.
 *
 * Phase 1 follows src/algorithms/hep.c (HEP, Deng 2020: level-wise, UBO
 * pruning) and src/algorithms/dfhoi.c (DFHOI, Nguyen et al. 2023: depth-first
 * equivalence classes, UBO pruning, shortcut for uniform transaction length).
 * The search, the pruning tests and their order are unchanged; the itemsets
 * are only tracked so that phase 2 can be applied. Before the final
 * maximality check an FHO is discarded early when an FHO was generated from it
 * (a join parent in HEP, a DFS ancestor in DFHOI), which never discards an
 * MFHOI.
 *
 * nodes = number of itemsets whose tidset is computed (single items included).
 *
 * Usage: hoi_filter <db> <min_support (absolute, or relative if < 1)> <beta>
 *                   --mode hep|dfhoi [--out file]
 * Output keys match mfho_miner (mfhoi=, candidates=, nodes=, time_mine=, ...).
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* allocation failure (address-space limit) ends the run with SIGABRT, which
 * the benchmark script records as a memory failure */
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fputs("out of memory\n", stderr); abort(); } return p; }
static void *xcalloc(size_t n, size_t m) { void *p = calloc(n ? n : 1, m); if (!p) { fputs("out of memory\n", stderr); abort(); } return p; }
static void *xrealloc(void *q, size_t n) { void *p = realloc(q, n ? n : 1); if (!p) { fputs("out of memory\n", stderr); abort(); } return p; }

static int ge_tol(double x, double t) { return x >= t - 1e-9 * (1.0 + fabs(t)); }

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static long peak_rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    char l[256];
    long v = -1;
    while (f && fgets(l, sizeof l, f)) if (!strncmp(l, "VmHWM:", 6)) { sscanf(l + 6, "%ld", &v); break; }
    if (f) fclose(f);
    return v;
}

static size_t *g_counts;
static int cmp_by_count(const void *a, const void *b) {   /* support ascending, then id */
    int x = *(const int *)a, y = *(const int *)b;
    if (g_counts[x] != g_counts[y]) return (g_counts[x] > g_counts[y]) - (g_counts[x] < g_counts[y]);
    return (x > y) - (x < y);
}

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

/* ── global state ──────────────────────────────────────────────────────── */
static int sigma;
static double beta, xi;
static int sum_mode;               /* --sum: maximal HOIs of Deng (2020), no phase-2 test */
static uint32_t *g_tsize;          /* tid (ascending length order) -> |T| */
static long long n_nodes, n_hoi, n_fho;

typedef struct { int **sets; int *size; int count, cap; } Cands;
static Cands cands;

static void cand_add(const int *set, int size) {
    if (cands.count == cands.cap) {
        cands.cap = cands.cap ? 2 * cands.cap : 256;
        cands.sets = xrealloc(cands.sets, cands.cap * sizeof(int *));
        cands.size = xrealloc(cands.size, cands.cap * sizeof(int));
    }
    int *s = xmalloc((size > 0 ? size : 1) * sizeof(int));
    memcpy(s, set, size * sizeof(int));
    cands.sets[cands.count] = s;
    cands.size[cands.count] = size;
    cands.count++;
}

static int is_subset(const int *a, int na, const int *b, int nb) {
    int i = 0, j = 0;
    while (i < na && j < nb) {
        if (a[i] == b[j]) { i++; j++; }
        else if (a[i] > b[j]) j++;
        else return 0;
    }
    return i == na;
}

/* phase 2 test: X with k items and tidset tids (size >= xi already) is an FHO */
static int is_fho(const uint32_t *tids, size_t size, size_t k) {
    if (sum_mode) return 1;        /* called only for HOIs */
    if ((long long)size < sigma) return 0;
    double s = 0;
    for (size_t i = 0; i < size; i++) s += 1.0 / g_tsize[tids[i]];
    return ge_tol((double)k * s / (double)size, beta);
}

/* UBO test of HEP/DFHOI (unchanged): returns whether X may be extended, and
 * sets *is_ho to whether X itself is an HOI. */
static int check_ubo_and_o(const uint32_t *tids, size_t size, size_t k, int *is_ho, int same_len, uint32_t first_len) {
    if (!ge_tol((double)size, xi)) return 0;
    if (same_len) {
        *is_ho = ge_tol((double)(k * size) / (double)first_len, xi);
        return 1;
    }
    double cur = 0.0, max_ubo = 0.0;
    for (int i = (int)size - 1; i >= 0; i--) {
        uint32_t ts = g_tsize[tids[i]];
        cur += 1.0 / (double)ts;
        if (i == 0 || g_tsize[tids[i - 1]] < ts) {
            double u = ts * cur;
            if (u > max_ubo) max_ubo = u;
        }
    }
    if (ge_tol(max_ubo, xi)) { *is_ho = ge_tol(k * cur, xi); return 1; }
    return 0;
}

static size_t intersect(const uint32_t *a, size_t na, const uint32_t *b, size_t nb, uint32_t *out) {
    size_t i = 0, j = 0, k = 0;
    while (i < na && j < nb) {
        if (a[i] < b[j]) i++;
        else if (a[i] > b[j]) j++;
        else { out[k++] = a[i]; i++; j++; }
    }
    return k;
}

/* ── HEP (level-wise) ──────────────────────────────────────────────────── */
typedef struct { int *items; size_t length; uint32_t *tids; size_t num_tids; int fho, dominated; } HItem;
typedef struct { HItem *v; size_t count, cap; } HList;

static void hl_add(HList *l, HItem it) {
    if (l->count == l->cap) { l->cap = l->cap ? 2 * l->cap : 1024; l->v = xrealloc(l->v, l->cap * sizeof(HItem)); }
    l->v[l->count++] = it;
}

static void hep_flush(HList *l) {
    for (size_t i = 0; i < l->count; i++) {
        if (l->v[i].fho && !l->v[i].dominated) cand_add(l->v[i].items, (int)l->v[i].length);
        free(l->v[i].items);
        free(l->v[i].tids);
    }
    free(l->v);
    l->v = NULL; l->count = l->cap = 0;
}

static void run_hep(int nitems, uint32_t **item_tids, size_t *counts, size_t ntrans) {
    HList prev = {0};
    for (int i = 0; i < nitems; i++) {
        if (!ge_tol((double)counts[i], xi)) continue;
        n_nodes++;
        int is_ho = 0;
        if (check_ubo_and_o(item_tids[i], counts[i], 1, &is_ho, 0, 0)) {
            HItem it = {xmalloc(sizeof(int)), 1, xmalloc(counts[i] * sizeof(uint32_t)), counts[i], 0, 0};
            it.items[0] = i;
            memcpy(it.tids, item_tids[i], counts[i] * sizeof(uint32_t));
            if (is_ho) { n_hoi++; if (is_fho(it.tids, it.num_tids, 1)) { it.fho = 1; n_fho++; } }
            hl_add(&prev, it);
        }
    }
    uint32_t *buf = xmalloc((ntrans > 0 ? ntrans : 1) * sizeof(uint32_t));
    size_t k = 2;
    while (prev.count > 0) {
        HList cur = {0};
        for (size_t i = 0; i < prev.count; i++) {
            for (size_t j = i + 1; j < prev.count; j++) {
                HItem *P1 = &prev.v[i], *P2 = &prev.v[j];
                int match = 1;
                for (size_t idx = 0; idx + 2 < k; idx++)
                    if (P1->items[idx] != P2->items[idx]) { match = 0; break; }
                if (!match) break;
                n_nodes++;
                size_t nn = intersect(P1->tids, P1->num_tids, P2->tids, P2->num_tids, buf);
                if (!ge_tol((double)nn, xi)) continue;
                int is_ho = 0;
                if (!check_ubo_and_o(buf, nn, k, &is_ho, 0, 0)) continue;
                HItem P = {xmalloc(k * sizeof(int)), k, xmalloc(nn * sizeof(uint32_t)), nn, 0, 0};
                memcpy(P.items, P1->items, (k - 1) * sizeof(int));
                P.items[k - 1] = P2->items[k - 2];
                memcpy(P.tids, buf, nn * sizeof(uint32_t));
                if (is_ho) {
                    n_hoi++;
                    if (is_fho(P.tids, nn, k)) { P.fho = 1; n_fho++; P1->dominated = P2->dominated = 1; }
                }
                hl_add(&cur, P);
            }
        }
        hep_flush(&prev);
        prev = cur;
        k++;
    }
    hep_flush(&prev);
    free(buf);
}

/* ── DFHOI (depth-first equivalence classes) ───────────────────────────── */
typedef struct { int last; size_t length; uint32_t *tids; size_t num_tids; int fho; } DItem;

static int same_len_flag;
static uint32_t first_len;
static int *prefix;             /* items of the current DFS path */

/* returns whether an FHO was generated in the classes below */
static int dfhoi_mine(DItem *cls, size_t ncls, uint32_t *buffer) {
    int any = 0;
    for (size_t i = 0; i < ncls; i++) {
        DItem *P1 = &cls[i];
        prefix[P1->length - 1] = P1->last;
        DItem *ch = ncls - i - 1 > 0 ? xmalloc((ncls - i - 1) * sizeof(DItem)) : NULL;
        size_t nch = 0;
        for (size_t j = i + 1; j < ncls; j++) {
            DItem *P2 = &cls[j];
            n_nodes++;
            size_t nn = intersect(P1->tids, P1->num_tids, P2->tids, P2->num_tids, buffer);
            if (!ge_tol((double)nn, xi)) continue;
            size_t nk = P1->length + 1;
            int is_ho = 0;
            if (!check_ubo_and_o(buffer, nn, nk, &is_ho, same_len_flag, first_len)) continue;
            DItem *P = &ch[nch++];
            P->last = P2->last;
            P->length = nk;
            P->num_tids = nn;
            P->tids = xmalloc(nn * sizeof(uint32_t));
            memcpy(P->tids, buffer, nn * sizeof(uint32_t));
            P->fho = 0;
            if (is_ho) { n_hoi++; if (is_fho(P->tids, nn, nk)) { P->fho = 1; n_fho++; } }
        }
        int below = 0;
        if (nch > 1) below = dfhoi_mine(ch, nch, buffer);
        prefix[P1->length - 1] = P1->last;   /* restore after the recursion */
        /* leaves of this class: children are not expanded further when nch < 2 */
        for (size_t c = 0; c < nch; c++) {
            if (ch[c].fho) below = 1;
            if (nch < 2 && ch[c].fho) {
                prefix[ch[c].length - 1] = ch[c].last;
                cand_add(prefix, (int)ch[c].length);
            }
            free(ch[c].tids);
        }
        free(ch);
        if (P1->fho && !below) cand_add(prefix, (int)P1->length);
        any |= below | P1->fho;
    }
    return any;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s db minsup beta --mode hep|dfhoi [--out file]\n", argv[0]);
        return 1;
    }
    const char *mode = "hep", *outp = NULL;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
        else if (!strcmp(argv[i], "--sum")) sum_mode = 1;
    }
    if (strcmp(mode, "hep") && strcmp(mode, "dfhoi")) { fprintf(stderr, "unknown mode %s\n", mode); return 1; }
    beta = atof(argv[3]);
    double minsup_arg = atof(argv[2]);

    double t0 = now_sec();
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }
    int cap = 1024, n = 0, maxid = 0, bcap = 1 << 16;
    int **raw = xmalloc(cap * sizeof(int *)), *rawc = xmalloc(cap * sizeof(int));
    int *b = xmalloc(bcap * sizeof(int));
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
            if (k == bcap) { bcap *= 2; b = xrealloc(b, bcap * sizeof(int)); }
            b[k++] = (int)v;
            p = e;
        }
        if (k == 0) continue;
        qsort(b, k, sizeof(int), cmp_int);
        int u = 0;
        for (int i = 0; i < k; i++) if (i == 0 || b[i] != b[i - 1]) b[u++] = b[i];
        if (n == cap) { cap *= 2; raw = xrealloc(raw, cap * sizeof(int *)); rawc = xrealloc(rawc, cap * sizeof(int)); }
        raw[n] = xmalloc(u * sizeof(int));
        memcpy(raw[n], b, u * sizeof(int));
        rawc[n] = u;
        if (b[u - 1] > maxid) maxid = b[u - 1];
        n++;
    }
    free(line); free(b); fclose(f);
    if (sum_mode) {
        /* argv[2] is xi itself (absolute, or a fraction of n as in Deng 2020) */
        xi = minsup_arg < 1.0 ? minsup_arg * n : minsup_arg;
        sigma = 1;
    } else {
        sigma = minsup_arg < 1.0 ? (int)ceil(minsup_arg * n - 1e-9) : (int)minsup_arg;
        if (sigma < 1) sigma = 1;
        xi = sigma * beta;
    }

    /* initial database representation, as in hep.c/dfhoi.c: transactions
     * renumbered by ascending length, tidsets of the items with count >= xi */
    int *order = xmalloc((n > 0 ? n : 1) * sizeof(int));
    for (int t = 0; t < n; t++) order[t] = t;
    /* stable counting sort by length */
    int maxlen = 0;
    for (int t = 0; t < n; t++) if (rawc[t] > maxlen) maxlen = rawc[t];
    int *cnt = xcalloc(maxlen + 2, sizeof(int));
    for (int t = 0; t < n; t++) cnt[rawc[t] + 1]++;
    for (int l = 1; l <= maxlen + 1; l++) cnt[l] += cnt[l - 1];
    for (int t = 0; t < n; t++) order[cnt[rawc[t]]++] = t;
    free(cnt);
    same_len_flag = 1;
    first_len = n > 0 ? (uint32_t)rawc[0] : 0;
    g_tsize = xmalloc((n > 0 ? n : 1) * sizeof(uint32_t));
    for (int t = 0; t < n; t++) {
        g_tsize[t] = rawc[order[t]];
        if ((uint32_t)rawc[t] != first_len) same_len_flag = 0;
    }
    if (!strcmp(mode, "hep")) same_len_flag = 0;   /* HEP has no uniform-length shortcut */
    size_t *counts = xcalloc(maxid + 1, sizeof(size_t));
    for (int t = 0; t < n; t++) for (int i = 0; i < rawc[t]; i++) counts[raw[t][i]]++;
    uint32_t **item_tids = xcalloc(maxid + 1, sizeof(uint32_t *));
    size_t *fill = xcalloc(maxid + 1, sizeof(size_t));
    for (int i = 0; i <= maxid; i++) if (ge_tol((double)counts[i], xi)) item_tids[i] = xmalloc(counts[i] * sizeof(uint32_t));
    for (int t = 0; t < n; t++) {
        int o = order[t];
        for (int i = 0; i < rawc[o]; i++) {
            int it = raw[o][i];
            if (item_tids[it]) item_tids[it][fill[it]++] = t;
        }
    }
    for (int t = 0; t < n; t++) free(raw[t]);
    free(raw); free(rawc); free(order); free(fill);
    long rss_loaded = peak_rss_kb();
    double t1 = now_sec();

    int *id_of = NULL;   /* HEP: rank -> item id */
    if (!strcmp(mode, "hep")) {
        /* Deng (2020), Sec. 3.2.3: items sorted in support-ascending order */
        id_of = xmalloc((maxid + 1) * sizeof(int));
        for (int i = 0; i <= maxid; i++) id_of[i] = i;
        g_counts = counts;
        qsort(id_of, maxid + 1, sizeof(int), cmp_by_count);
        uint32_t **rt = xmalloc((maxid + 1) * sizeof(uint32_t *));
        size_t *rc = xmalloc((maxid + 1) * sizeof(size_t));
        for (int r = 0; r <= maxid; r++) { rt[r] = item_tids[id_of[r]]; rc[r] = counts[id_of[r]]; }
        free(item_tids); free(counts);
        item_tids = rt; counts = rc;
        run_hep(maxid + 1, item_tids, counts, n);
    } else {
        DItem *C1 = xmalloc((maxid + 1) * sizeof(DItem));
        size_t c1 = 0;
        for (int i = 0; i <= maxid; i++) {
            if (!ge_tol((double)counts[i], xi)) continue;
            n_nodes++;
            int is_ho = 0;
            if (check_ubo_and_o(item_tids[i], counts[i], 1, &is_ho, same_len_flag, first_len)) {
                DItem *it = &C1[c1++];
                it->last = i; it->length = 1; it->num_tids = counts[i];
                it->tids = xmalloc(counts[i] * sizeof(uint32_t));
                memcpy(it->tids, item_tids[i], counts[i] * sizeof(uint32_t));
                it->fho = 0;
                if (is_ho) { n_hoi++; if (is_fho(it->tids, it->num_tids, 1)) { it->fho = 1; n_fho++; } }
            }
        }
        prefix = xmalloc((maxid + 2) * sizeof(int));
        uint32_t *buffer = xmalloc((n > 0 ? n : 1) * sizeof(uint32_t));
        if (c1 > 1) dfhoi_mine(C1, c1, buffer);
        else if (c1 == 1 && C1[0].fho) cand_add(&C1[0].last, 1);
        free(buffer);
        for (size_t i = 0; i < c1; i++) free(C1[i].tids);
        free(C1);
    }
    double t2 = now_sec();

    /* phase 2: remove candidates contained in a kept larger one */
    int *ord = xmalloc((cands.count + 1) * sizeof(int));
    /* counting sort by size, descending */
    int maxs = 0;
    for (int i = 0; i < cands.count; i++) if (cands.size[i] > maxs) maxs = cands.size[i];
    int *sc = xcalloc(maxs + 2, sizeof(int));
    for (int i = 0; i < cands.count; i++) sc[maxs - cands.size[i] + 1]++;
    for (int s = 1; s <= maxs + 1; s++) sc[s] += sc[s - 1];
    for (int i = 0; i < cands.count; i++) ord[sc[maxs - cands.size[i]]++] = i;
    free(sc);
    char *keep = xcalloc(cands.count + 1, 1);
    int *kept = xmalloc((cands.count + 1) * sizeof(int)), nout = 0;
    for (int a = 0; a < cands.count; a++) {
        int i = ord[a], dom = 0;
        for (int q = 0; q < nout && !dom; q++) {
            int j = kept[q];
            if (cands.size[j] >= cands.size[i] && is_subset(cands.sets[i], cands.size[i], cands.sets[j], cands.size[j])) dom = 1;
        }
        if (!dom) { keep[i] = 1; kept[nout++] = i; }
    }
    double t3 = now_sec();

    if (outp) {
        FILE *o = fopen(outp, "w");
        for (int q = 0; q < nout; q++) {
            int i = kept[q];
            if (id_of) for (int k = 0; k < cands.size[i]; k++) cands.sets[i][k] = id_of[cands.sets[i][k]];
            qsort(cands.sets[i], cands.size[i], sizeof(int), cmp_int);
            for (int k = 0; k < cands.size[i]; k++) fprintf(o, k ? " %d" : "%d", cands.sets[i][k]);
            fprintf(o, "\n");
        }
        fclose(o);
    }
    printf("mode=%s n=%d sigma=%d beta=%g xi=%.6f\n", mode, n, sigma, beta, xi);
    printf("mfhoi=%d candidates=%d\n", nout, cands.count);
    printf("nodes=%lld hoi=%lld fho_nodes=%lld\n", n_nodes, n_hoi, n_fho);
    printf("time_load=%.6f time_mine=%.6f time_filter=%.6f\n", t1 - t0, t2 - t1, t3 - t2);
    printf("rss_after_load_kb=%ld peak_rss_kb=%ld\n", rss_loaded, peak_rss_kb());
    return 0;
}
