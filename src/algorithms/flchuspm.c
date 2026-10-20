/*
 * FLCHUSPM (Truong, Duong, Le, Fournier-Viger, Yun, Applied Intelligence 52 (2022) 7136-7157),
 * implemented from the paper; no reference implementation is available.
 *
 * Paper elements and where they are implemented:
 *  - Definition 4: u(alpha, Psi) and c(alpha, Psi) are maxima over the occurrences of alpha in
 *    Psi (taken independently: the running example uses c(d->c, Psi4) = max{5, 4});
 *    au = u / supp, ac = c / supp                                        -> node_measures()
 *  - Definition 5: AMUB = (sum of the m largest u(Psi) over rho(alpha)) / m,
 *    FUB = (sum of the m largest ub(alpha, Psi)) / m, ub = max (u_end + ru_end), m = mins
 *  - Definition 6: Flb = (sum of the m smallest cf) / m
 *  - Section 5.1: CUL(alpha) = list of (sid, cf, tuples (end, c_end, u_end, ru_end)); CULs of
 *    single items (5.1.1) and of i-/s-extensions (5.1.3), including the cf rules of 5.1.3
 *  - Fig. 3: S = {x | AMUB(x) >= minu and supp(x) >= mins}; RedS removes the other items and
 *    the ISU vector and the CULs are recomputed on the reduced database; iCMAP/sCMAP
 *  - Fig. 4: SearchFLCHUS with DPS (line 1), output (line 2), WPS + TS + CMAPS (lines 3-8)
 *    and the recursion (lines 9-10).
 * Choices where the paper leaves a detail open are listed in docs/core/FLCHUS/notes.md.
 *
 * If DM_FLCHUS_OUT is set, patterns are written there as
 *   "i1 i2 -1 i3 -1 #SUP: s #AU: au #AC: ac".
 */
#include "algorithms/flchuspm.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fputs("[FLCHUSPM] out of memory\n", stderr); abort(); } return p; }
static void *xrealloc(void *p, size_t n) { p = realloc(p, n ? n : 1); if (!p) { fputs("[FLCHUSPM] out of memory\n", stderr); abort(); } return p; }

/* ---------- database ---------- */
typedef struct { int32_t item; double c, u; } QItem;
typedef struct { int n; QItem *q; } QItemset;            /* items in increasing order */
typedef struct { int n; QItemset *s; } QSeq;

typedef struct {
    int nseq; QSeq *seq;
    int32_t max_item;
} DB;

static int cmp_qitem(const void *a, const void *b) {
    int32_t x = ((const QItem *)a)->item, y = ((const QItem *)b)->item;
    return (x > y) - (x < y);
}

static int read_db(const char *path, DB *db) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return 0; }
    memset(db, 0, sizeof *db);
    int cap = 256;
    db->seq = xmalloc(sizeof(QSeq) * (size_t)cap);
    char *line = NULL; size_t lc = 0;
    while (getline(&line, &lc, f) >= 0) {
        char *p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '%' || *p == '@' || *p == '\n' || *p == '\r' || *p == 0) continue;
        QSeq sq = {0, NULL}; int scap = 0;
        QItemset cur = {0, NULL}; int icap = 0;
        char *tok = strtok(p, " \t\r\n");
        while (tok) {
            if (strcmp(tok, "-1") == 0 || strcmp(tok, "-2") == 0) {
                if (cur.n > 0) {
                    qsort(cur.q, (size_t)cur.n, sizeof(QItem), cmp_qitem);
                    if (sq.n == scap) { scap = scap ? 2 * scap : 8; sq.s = xrealloc(sq.s, sizeof(QItemset) * (size_t)scap); }
                    sq.s[sq.n++] = cur;
                }
                cur.n = 0; cur.q = NULL; icap = 0;
                if (tok[1] == '2') break;
            } else {
                QItem q;
                char *c1 = strchr(tok, ':'), *c2 = c1 ? strchr(c1 + 1, ':') : NULL;
                if (!c2) { fprintf(stderr, "[FLCHUSPM] bad q-item '%s'\n", tok); free(line); fclose(f); return 0; }
                q.item = (int32_t)strtol(tok, NULL, 10);
                q.c = strtod(c1 + 1, NULL);
                q.u = strtod(c2 + 1, NULL);
                if (q.item > db->max_item) db->max_item = q.item;
                if (cur.n == icap) { icap = icap ? 2 * icap : 4; cur.q = xrealloc(cur.q, sizeof(QItem) * (size_t)icap); }
                cur.q[cur.n++] = q;
            }
            tok = strtok(NULL, " \t\r\n");
        }
        if (cur.n > 0) {                     /* sequence without a final -1 */
            qsort(cur.q, (size_t)cur.n, sizeof(QItem), cmp_qitem);
            if (sq.n == scap) { scap = scap ? 2 * scap : 8; sq.s = xrealloc(sq.s, sizeof(QItemset) * (size_t)scap); }
            sq.s[sq.n++] = cur;
        }
        if (db->nseq == cap) { cap *= 2; db->seq = xrealloc(db->seq, sizeof(QSeq) * (size_t)cap); }
        db->seq[db->nseq++] = sq;            /* empty sequences are kept (they count in |D|) */
    }
    free(line); fclose(f);
    return 1;
}

/* map item identifiers to 0..k-1 (in increasing order, so the item order is unchanged);
   orig[i] is the identifier written in the output */
static int cmp_i32(const void *a, const void *b) { int32_t x = *(const int32_t *)a, y = *(const int32_t *)b; return (x > y) - (x < y); }
static int32_t *densify(DB *db) {
    size_t n = 0, cap = 1024;
    int32_t *ids = xmalloc(sizeof(int32_t) * cap);
    for (int i = 0; i < db->nseq; i++)
        for (int k = 0; k < db->seq[i].n; k++)
            for (int j = 0; j < db->seq[i].s[k].n; j++) {
                if (n == cap) { cap *= 2; ids = xrealloc(ids, sizeof(int32_t) * cap); }
                ids[n++] = db->seq[i].s[k].q[j].item;
            }
    qsort(ids, n, sizeof(int32_t), cmp_i32);
    size_t w = 0;
    for (size_t i = 0; i < n; i++) if (w == 0 || ids[w - 1] != ids[i]) ids[w++] = ids[i];
    for (int i = 0; i < db->nseq; i++)
        for (int k = 0; k < db->seq[i].n; k++)
            for (int j = 0; j < db->seq[i].s[k].n; j++) {
                int32_t *f = bsearch(&db->seq[i].s[k].q[j].item, ids, w, sizeof(int32_t), cmp_i32);
                db->seq[i].s[k].q[j].item = (int32_t)(f - ids);
            }
    db->max_item = (int32_t)w - 1;
    return ids;
}

static void free_db(DB *db) {
    for (int i = 0; i < db->nseq; i++) {
        for (int k = 0; k < db->seq[i].n; k++) free(db->seq[i].s[k].q);
        free(db->seq[i].s);
    }
    free(db->seq);
}

/* ---------- CUL ---------- */
typedef struct { int32_t end; double c, u, ru; } Tup;   /* end is 1-based itemset index */
typedef struct { int32_t sid; double cf; int32_t off, n; } Ent;
typedef struct { int32_t n, cap; Ent *e; int32_t nt, tcap; Tup *t; } CUL;

static void cul_free(CUL *c) { free(c->e); free(c->t); memset(c, 0, sizeof *c); }
static Ent *cul_new_ent(CUL *c, int32_t sid) {
    if (c->n == c->cap) { c->cap = c->cap ? 2 * c->cap : 8; c->e = xrealloc(c->e, sizeof(Ent) * (size_t)c->cap); }
    Ent *e = &c->e[c->n++];
    e->sid = sid; e->cf = 0; e->off = c->nt; e->n = 0;
    return e;
}
static void cul_push_tup(CUL *c, Ent *e, Tup t) {
    if (c->nt == c->tcap) { c->tcap = c->tcap ? 2 * c->tcap : 16; c->t = xrealloc(c->t, sizeof(Tup) * (size_t)c->tcap); }
    c->t[c->nt++] = t; e->n++;
}

/* ---------- miner state ---------- */
typedef struct {
    DB db;
    double *isu;              /* u(Psi_i) on the (reduced) database */
    int m;                    /* mins as an integer count */
    double minu, maxc;
    int use_dps, use_wps, use_flb;
    CUL *item_cul;            /* CUL of single items (reduced database), index = item id */
    uint8_t *in_S;            /* item in S */
    uint8_t *icmap, *scmap;   /* [x * (max_item+1) + y] */
    int32_t nitems;           /* max_item + 1 */
    int32_t *pat; int *pat_set; int pat_len, pat_cap;
    int32_t *orig;            /* dense item id -> input identifier */   /* current pattern: items and set index */
    FILE *out;
    int trace;
    long nodes, extensions, patterns;
    double *scratch; size_t scratch_cap;
} FL;

static double *scratch(FL *s, size_t n) {
    if (n > s->scratch_cap) { s->scratch_cap = n * 2; s->scratch = xrealloc(s->scratch, sizeof(double) * s->scratch_cap); }
    return s->scratch;
}
static int cmp_desc(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x < y) - (x > y); }
static int cmp_asc(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

/* (sum of the m largest (desc=1) or smallest (desc=0) values) / m; requires n >= m */
static double mean_top_m(FL *s, double *v, int n, int desc) {
    qsort(v, (size_t)n, sizeof(double), desc ? cmp_desc : cmp_asc);
    double t = 0;
    for (int j = 0; j < s->m && j < n; j++) t += v[j];
    return t / s->m;
}

static double amub_of(FL *s, const CUL *c) {
    double *v = scratch(s, (size_t)c->n);
    for (int i = 0; i < c->n; i++) v[i] = s->isu[c->e[i].sid];
    return mean_top_m(s, v, c->n, 1);
}

typedef struct { double au, ac, amub, fub, flb; } Measures;

static Measures node_measures(FL *s, const CUL *c) {
    Measures r;
    double su = 0, sc = 0;
    double *v = scratch(s, (size_t)c->n);
    for (int i = 0; i < c->n; i++) {
        const Ent *e = &c->e[i];
        double mu = -INFINITY, mc = -INFINITY, mub = -INFINITY;
        for (int j = 0; j < e->n; j++) {
            const Tup *t = &c->t[e->off + j];
            if (t->u > mu) mu = t->u;
            if (t->c > mc) mc = t->c;
            if (t->u + t->ru > mub) mub = t->u + t->ru;
        }
        su += mu; sc += mc; v[i] = mub;
    }
    r.au = su / c->n; r.ac = sc / c->n;
    r.fub = mean_top_m(s, v, c->n, 1);
    for (int i = 0; i < c->n; i++) v[i] = c->e[i].cf;
    r.flb = mean_top_m(s, v, c->n, 0);
    r.amub = amub_of(s, c);
    return r;
}

/* Section 5.1.3: CUL of alpha <>_i y (serial = 0) or alpha <>_s y (serial = 1) */
static void extend(const CUL *a, const CUL *y, int serial, CUL *out) {
    memset(out, 0, sizeof *out);
    int i = 0, j = 0;
    while (i < a->n && j < y->n) {
        if (a->e[i].sid < y->e[j].sid) { i++; continue; }
        if (a->e[i].sid > y->e[j].sid) { j++; continue; }
        const Ent *ea = &a->e[i], *ey = &y->e[j];
        const Tup *ta = a->t + ea->off, *ty = y->t + ey->off;
        Ent *eb = NULL;
        if (!serial) {
            int p = 0;
            for (int q = 0; q < ey->n; q++) {
                while (p < ea->n && ta[p].end < ty[q].end) p++;
                if (p < ea->n && ta[p].end == ty[q].end) {
                    Tup t = { ty[q].end, ta[p].c + ty[q].c, ta[p].u + ty[q].u, ty[q].ru };
                    if (!eb) { eb = cul_new_ent(out, ea->sid); eb->cf = t.c; }   /* first matching pair */
                    cul_push_tup(out, eb, t);
                }
            }
        } else {
            int p = 0; double maxc = -INFINITY, maxu = -INFINITY;
            for (int q = 0; q < ey->n; q++) {
                while (p < ea->n && ta[p].end < ty[q].end) {
                    if (ta[p].c > maxc) maxc = ta[p].c;
                    if (ta[p].u > maxu) maxu = ta[p].u;
                    p++;
                }
                if (p == 0) continue;                     /* no t(alpha) with end < t(y).end */
                Tup t = { ty[q].end, maxc + ty[q].c, maxu + ty[q].u, ty[q].ru };
                if (!eb) { eb = cul_new_ent(out, ea->sid); eb->cf = ta[0].c + ty[q].c; }  /* first t(alpha), first t(y) */
                cul_push_tup(out, eb, t);
            }
        }
        i++; j++;
    }
}

/* CULs of all single items of the current database (Section 5.1.1): for item y in itemset k
   (end = k + 1) of sequence sid, the tuple is (end, c(y), u(y), ru) where ru is the utility of
   the items after y in the same itemset plus that of all later itemsets; cf is the cost of the
   first appearance of y. Each item appears at most once per itemset. */
static void build_item_culs(FL *s) {
    s->item_cul = calloc((size_t)s->nitems, sizeof(CUL));
    for (int sid = 0; sid < s->db.nseq; sid++) {
        const QSeq *sq = &s->db.seq[sid];
        double *after = xmalloc(sizeof(double) * (size_t)(sq->n + 1));
        after[sq->n] = 0;
        for (int k = sq->n - 1; k >= 0; k--) {
            double t = 0; for (int j = 0; j < sq->s[k].n; j++) t += sq->s[k].q[j].u;
            after[k] = after[k + 1] + t;
        }
        for (int k = 0; k < sq->n; k++) {
            const QItemset *is = &sq->s[k];
            double tail = after[k + 1];
            for (int j = is->n - 1; j >= 0; j--) {
                double ru = tail;
                tail += is->q[j].u;
                CUL *c = &s->item_cul[is->q[j].item];
                Ent *e;
                if (c->n > 0 && c->e[c->n - 1].sid == sid) e = &c->e[c->n - 1];
                else { e = cul_new_ent(c, sid); e->cf = is->q[j].c; }
                /* tuples of an item are appended in increasing end; each CUL owns its tuple
                   array and only the last entry is open, so tuples stay contiguous */
                Tup t = { k + 1, is->q[j].c, is->q[j].u, ru };
                cul_push_tup(c, e, t);
            }
        }
        free(after);
    }
}

static void compute_isu(FL *s) {
    free(s->isu);
    s->isu = xmalloc(sizeof(double) * (size_t)(s->db.nseq ? s->db.nseq : 1));
    for (int i = 0; i < s->db.nseq; i++) {
        double t = 0;
        for (int k = 0; k < s->db.seq[i].n; k++) for (int j = 0; j < s->db.seq[i].s[k].n; j++) t += s->db.seq[i].s[k].q[j].u;
        s->isu[i] = t;
    }
}

/* candidate test used in Fig. 3 line 1, Fig. 4 lines 6 and 8 and the CMAPs */
static int passes_width(FL *s, const CUL *c) {
    if (c->n < s->m) return 0;
    if (!s->use_wps) return 1;
    return amub_of(s, c) >= s->minu;
}

static void emit(FL *s, const Measures *mz, int supp) {
    s->patterns++;
    if (!s->out) return;
    for (int i = 0; i < s->pat_len; i++) {
        fprintf(s->out, "%d ", s->orig[s->pat[i]]);
        if (i + 1 == s->pat_len || s->pat_set[i + 1] != s->pat_set[i]) fputs("-1 ", s->out);
    }
    fprintf(s->out, "#SUP: %d #AU: %.6f #AC: %.6f\n", supp, mz->au, mz->ac);
}

static void push_item(FL *s, int32_t item, int new_set) {
    if (s->pat_len == s->pat_cap) {
        s->pat_cap = s->pat_cap ? 2 * s->pat_cap : 16;
        s->pat = xrealloc(s->pat, sizeof(int32_t) * (size_t)s->pat_cap);
        s->pat_set = xrealloc(s->pat_set, sizeof(int) * (size_t)s->pat_cap);
    }
    int set = s->pat_len == 0 ? 0 : s->pat_set[s->pat_len - 1] + (new_set ? 1 : 0);
    s->pat[s->pat_len] = item; s->pat_set[s->pat_len] = set; s->pat_len++;
}

/* Fig. 4, SearchFLCHUS(alpha, I, S); cul = CUL(alpha) */
static void search(FL *s, const CUL *cul, const int32_t *I, int nI, const int32_t *S, int nS) {
    s->nodes++;
    Measures mz = node_measures(s, cul);
    if (s->trace) {
        printf("[TRACE] node ");
        for (int i = 0; i < s->pat_len; i++) printf("%s%d", i == 0 ? "" : (s->pat_set[i] != s->pat_set[i - 1] ? "->" : ","), s->orig[s->pat[i]]);
        printf(" AMUB=%.4f FUB=%.4f Flb=%.4f au=%.4f ac=%.4f supp=%d\n", mz.amub, mz.fub, mz.flb, mz.au, mz.ac, cul->n);
    }
    if (s->use_dps && (mz.fub < s->minu || (s->use_flb && mz.flb > s->maxc))) return;   /* line 1 */
    if (mz.au >= s->minu && mz.ac <= s->maxc) emit(s, &mz, cul->n);             /* line 2 */
    int32_t x = s->pat[s->pat_len - 1];                                          /* line 4 */
    int32_t *Ia = xmalloc(sizeof(int32_t) * (size_t)(nI ? nI : 1)), *Sa = xmalloc(sizeof(int32_t) * (size_t)(nS ? nS : 1));
    CUL *Ic = xmalloc(sizeof(CUL) * (size_t)(nI ? nI : 1)), *Sc = xmalloc(sizeof(CUL) * (size_t)(nS ? nS : 1));
    int ni = 0, ns = 0;
    for (int k = 0; k < nI; k++) {                                               /* lines 5-6 */
        int32_t y = I[k];
        if (y <= x || !s->icmap[(size_t)x * s->nitems + y]) continue;
        CUL c; extend(cul, &s->item_cul[y], 0, &c); s->extensions++;
        if (passes_width(s, &c)) { Ia[ni] = y; Ic[ni++] = c; } else cul_free(&c);
    }
    for (int k = 0; k < nS; k++) {                                               /* lines 7-8 */
        int32_t y = S[k];
        if (!s->scmap[(size_t)x * s->nitems + y]) continue;
        CUL c; extend(cul, &s->item_cul[y], 1, &c); s->extensions++;
        if (passes_width(s, &c)) { Sa[ns] = y; Sc[ns++] = c; } else cul_free(&c);
    }
    for (int k = 0; k < ni; k++) {                                               /* line 9 */
        push_item(s, Ia[k], 0);
        search(s, &Ic[k], Ia, ni, Sa, ns);
        s->pat_len--;
        cul_free(&Ic[k]);
    }
    for (int k = 0; k < ns; k++) {                                               /* line 10 */
        push_item(s, Sa[k], 1);
        search(s, &Sc[k], Sa, ns, Sa, ns);
        s->pat_len--;
        cul_free(&Sc[k]);
    }
    free(Ia); free(Sa); free(Ic); free(Sc);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    (void)ds;
    DM_FLCHUSPM_Params *p = (DM_FLCHUSPM_Params *)params;
    if (!p || !p->input_path) return DM_ERROR_INVALID_PARAM;
    FL s; memset(&s, 0, sizeof s);
    if (!read_db(p->input_path, &s.db)) return DM_ERROR_INVALID_PARAM;
    s.use_dps = p->use_dps; s.use_wps = p->use_wps;
    s.use_flb = getenv("FLCHUS_FLB") ? atoi(getenv("FLCHUS_FLB")) : 1;   /* 0: correct baseline, Flb removed */
    s.orig = densify(&s.db);
    s.nitems = s.db.max_item + 1;

    /* thresholds (Section 3, percentage form uses the original database) */
    double maxU = 0, maxC = 0;
    for (int i = 0; i < s.db.nseq; i++) {
        double u = 0, c = 0;
        for (int k = 0; k < s.db.seq[i].n; k++) for (int j = 0; j < s.db.seq[i].s[k].n; j++) { u += s.db.seq[i].s[k].q[j].u; c += s.db.seq[i].s[k].q[j].c; }
        if (u > maxU) maxU = u;
        if (c > maxC) maxC = c;
    }
    double mins = p->percent ? p->mins / 100.0 * s.db.nseq : p->mins;
    s.minu = p->percent ? p->minu / 100.0 * maxU : p->minu;
    s.maxc = p->percent ? p->maxc / 100.0 * maxC : p->maxc;
    s.m = (int)ceil(mins - 1e-9);
    if (s.m < 1) s.m = 1;

    /* Fig. 3 line 1: S on the original database */
    compute_isu(&s);
    build_item_culs(&s);
    s.in_S = calloc((size_t)s.nitems, 1);
    for (int32_t x = 0; x < s.nitems; x++)
        if (s.item_cul[x].n > 0 && passes_width(&s, &s.item_cul[x])) s.in_S[x] = 1;
    /* lines 2-3: RedS, then ISU and CULs on the reduced database */
    for (int i = 0; i < s.db.nseq; i++) {
        QSeq *sq = &s.db.seq[i];
        int w = 0;
        for (int k = 0; k < sq->n; k++) {
            QItemset *is = &sq->s[k];
            int v = 0;
            for (int j = 0; j < is->n; j++) if (s.in_S[is->q[j].item]) is->q[v++] = is->q[j];
            is->n = v;
            if (v > 0) sq->s[w++] = *is; else free(is->q);
        }
        sq->n = w;
    }
    for (int32_t x = 0; x < s.nitems; x++) cul_free(&s.item_cul[x]);
    free(s.item_cul);
    compute_isu(&s);
    build_item_culs(&s);
    int32_t *S = xmalloc(sizeof(int32_t) * (size_t)s.nitems); int nS = 0;
    for (int32_t x = 0; x < s.nitems; x++) if (s.in_S[x]) S[nS++] = x;

    /* line 4: iCMAP and sCMAP of the items of S (on the reduced database) */
    s.icmap = calloc((size_t)s.nitems * (size_t)s.nitems, 1);
    s.scmap = calloc((size_t)s.nitems * (size_t)s.nitems, 1);
    for (int a = 0; a < nS; a++)
        for (int b = 0; b < nS; b++) {
            int32_t x = S[a], y = S[b];
            CUL c;
            if (y > x) {
                extend(&s.item_cul[x], &s.item_cul[y], 0, &c);
                if (passes_width(&s, &c)) s.icmap[(size_t)x * s.nitems + y] = 1;
                cul_free(&c);
            }
            extend(&s.item_cul[x], &s.item_cul[y], 1, &c);
            if (passes_width(&s, &c)) s.scmap[(size_t)x * s.nitems + y] = 1;
            cul_free(&c);
        }

    s.trace = getenv("FLCHUS_TRACE") != NULL;
    if (s.trace)
        for (int a = 0; a < nS; a++)
            for (int b = 0; b < nS; b++) {
                int32_t x = S[a], y = S[b];
                printf("[TRACE] cmap %d %d i=%d s=%d\n", s.orig[x], s.orig[y], s.icmap[(size_t)x * s.nitems + y], s.scmap[(size_t)x * s.nitems + y]);
            }
    if (s.trace) {
        printf("[TRACE] ISU");
        for (int i = 0; i < s.db.nseq; i++) printf(" %.4f", s.isu[i]);
        printf("\n");
        for (int a = 0; a < nS; a++) {
            const CUL *c = &s.item_cul[S[a]];
            printf("[TRACE] CUL %d:", s.orig[S[a]]);
            for (int i = 0; i < c->n; i++) {
                printf(" <%d, %g;", c->e[i].sid + 1, c->e[i].cf);
                for (int j = 0; j < c->e[i].n; j++) { const Tup *t = &c->t[c->e[i].off + j]; printf(" (%d,%g;%g,%g)", t->end, t->c, t->u, t->ru); }
                printf(">");
            }
            printf("\n");
        }
    }
    const char *outp = getenv("DM_FLCHUS_OUT");
    if (outp) s.out = fopen(outp, "w");
    printf("[FLCHUSPM] |D|=%d m=%d minu=%.6f maxc=%.6f maxUSeq=%.3f maxCSeq=%.3f |S|=%d DPS=%d WPS=%d Flb=%d\n",
           s.db.nseq, s.m, s.minu, s.maxc, maxU, maxC, nS, s.use_dps, s.use_wps, s.use_flb);
    for (int a = 0; a < nS; a++) {                                               /* line 5 */
        push_item(&s, S[a], 1);
        search(&s, &s.item_cul[S[a]], S, nS, S, nS);
        s.pat_len--;
    }
    if (s.out) fclose(s.out);
    printf("[FLCHUSPM] patterns=%ld nodes=%ld extensions=%ld\n", s.patterns, s.nodes, s.extensions);
    dm_bench_record_results((size_t)s.patterns, (size_t)s.nodes);

    for (int32_t x = 0; x < s.nitems; x++) cul_free(&s.item_cul[x]);
    free(s.item_cul); free(s.in_S); free(s.icmap); free(s.scmap); free(S);
    free(s.isu); free(s.pat); free(s.pat_set); free(s.scratch);
    free(s.orig);
    free_db(&s.db);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "flchuspm",
    .name = "FLCHUSPM",
    .description = "Frequent low-average-cost high-average-utility sequential patterns (Truong et al., Applied Intelligence 2022).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
