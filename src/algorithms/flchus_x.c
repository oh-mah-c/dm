/*
 * FLCHUS-X (proposed algorithm) for FLCHUSM, the problem of Truong et al., "Mining interesting
 * sequences with low average cost and high average utility", Applied Intelligence 52 (2022).
 *
 * Search space, CUL lists, utility bounds (AMUB, FUB) and the pseudo-code skeleton are those of
 * FLCHUSPM (src/algorithms/flchuspm.c). Differences (proofs in docs/core/FLCHUS/notes.md):
 *  1. MLB replaces Flb. For every occurrence end, a tuple also stores cmin_end, the minimum cost
 *     of the occurrences of alpha ending there; lbmin(alpha, Psi) = min over the tuples.
 *     MLB(alpha) = (sum of the m smallest lbmin) / m. Every occurrence of a super-sequence beta
 *     of alpha contains an occurrence of alpha, so c(beta, Psi) >= lbmin(alpha, Psi), and
 *     ac(beta) >= MLB(alpha) for every frequent super-sequence beta (not only forward
 *     extensions). Flb of FLCHUSPM lacks this property (Theorem 2b of the paper fails for
 *     i-extensions), which makes FLCHUSPM incomplete.
 *  2. Because MLB is monotone for all super-sequences, it can be used wherever AMUB is used:
 *     RedS removes items y with MLB(y) > maxc, the CMAPs exclude pairs with MLB > maxc, and the
 *     candidate sets I(alpha), S(alpha) exclude y with MLB(alpha <> y) > maxc   [cost_width].
 *  3. RedS is repeated until no item is removed (each removal lowers the ISU)  [reds_fixpoint].
 *  4. The m largest/smallest values are summed after an O(n) selection instead of a sort, and
 *     a join is abandoned as soon as the support threshold can no longer be reached
 *     [fast_select].
 * Output format: as flchuspm ("... #SUP: s #AU: au #AC: ac"), file given by DM_FLCHUS_OUT.
 */
#include "algorithms/flchus_x.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fputs("[FLCHUS-X] out of memory\n", stderr); abort(); } return p; }
static void *xrealloc(void *p, size_t n) { p = realloc(p, n ? n : 1); if (!p) { fputs("[FLCHUS-X] out of memory\n", stderr); abort(); } return p; }

/* ---------- database (same format as flchuspm) ---------- */
typedef struct { int32_t item; double c, u; } QItem;
typedef struct { int n; QItem *q; } QItemset;
typedef struct { int n; QItemset *s; } QSeq;
typedef struct { int nseq; QSeq *seq; int32_t max_item; } DB;

static int cmp_qitem(const void *a, const void *b) {
    int32_t x = ((const QItem *)a)->item, y = ((const QItem *)b)->item;
    return (x > y) - (x < y);
}

static void push_set(QSeq *sq, int *scap, QItemset *cur) {
    qsort(cur->q, (size_t)cur->n, sizeof(QItem), cmp_qitem);
    if (sq->n == *scap) { *scap = *scap ? 2 * *scap : 8; sq->s = xrealloc(sq->s, sizeof(QItemset) * (size_t)*scap); }
    sq->s[sq->n++] = *cur;
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
        for (char *tok = strtok(p, " \t\r\n"); tok; tok = strtok(NULL, " \t\r\n")) {
            if (strcmp(tok, "-1") == 0 || strcmp(tok, "-2") == 0) {
                if (cur.n > 0) push_set(&sq, &scap, &cur);
                cur.n = 0; cur.q = NULL; icap = 0;
                if (tok[1] == '2') break;
                continue;
            }
            char *c1 = strchr(tok, ':'), *c2 = c1 ? strchr(c1 + 1, ':') : NULL;
            if (!c2) { fprintf(stderr, "[FLCHUS-X] bad q-item '%s'\n", tok); free(line); fclose(f); return 0; }
            QItem q = { (int32_t)strtol(tok, NULL, 10), strtod(c1 + 1, NULL), strtod(c2 + 1, NULL) };
            if (q.item > db->max_item) db->max_item = q.item;
            if (cur.n == icap) { icap = icap ? 2 * icap : 4; cur.q = xrealloc(cur.q, sizeof(QItem) * (size_t)icap); }
            cur.q[cur.n++] = q;
        }
        if (cur.n > 0) push_set(&sq, &scap, &cur);
        if (db->nseq == cap) { cap *= 2; db->seq = xrealloc(db->seq, sizeof(QSeq) * (size_t)cap); }
        db->seq[db->nseq++] = sq;
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

/* ---------- CUL with cmin ---------- */
typedef struct { int32_t end; double c, cmin, u, ru; } Tup;
typedef struct { int32_t sid; int32_t off, n; } Ent;
typedef struct { int32_t n, cap; Ent *e; int32_t nt, tcap; Tup *t; } CUL;

static void cul_free(CUL *c) { free(c->e); free(c->t); memset(c, 0, sizeof *c); }
static Ent *cul_new_ent(CUL *c, int32_t sid) {
    if (c->n == c->cap) { c->cap = c->cap ? 2 * c->cap : 8; c->e = xrealloc(c->e, sizeof(Ent) * (size_t)c->cap); }
    Ent *e = &c->e[c->n++];
    e->sid = sid; e->off = c->nt; e->n = 0;
    return e;
}
static void cul_push(CUL *c, Ent *e, Tup t) {
    if (c->nt == c->tcap) { c->tcap = c->tcap ? 2 * c->tcap : 16; c->t = xrealloc(c->t, sizeof(Tup) * (size_t)c->tcap); }
    c->t[c->nt++] = t; e->n++;
}

typedef struct {
    DB db;
    double *isu;
    int m;
    double minu, maxc;
    int cost_width, reds_fixpoint, fast_select, desc_bound;
    CUL *item_cul;
    uint8_t *alive;
    uint8_t *icmap, *scmap;
    int32_t nitems;
    int32_t *pat; int *pat_set; int pat_len, pat_cap;
    int32_t *orig;            /* dense item id -> input identifier */
    FILE *out;
    int trace;
    long nodes, extensions, abandoned, patterns, removed_items, reds_rounds;
    double *scr; size_t scr_cap;
} FX;

static double *scr(FX *s, size_t n) {
    if (n > s->scr_cap) { s->scr_cap = 2 * n; s->scr = xrealloc(s->scr, sizeof(double) * s->scr_cap); }
    return s->scr;
}

static int cmp_desc(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x < y) - (x > y); }
static int cmp_asc(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

/* rearranges v so that v[0..k) are the k largest (desc) or smallest values (Hoare selection) */
static void select_k(double *v, int n, int k, int desc) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        double pivot = v[lo + (hi - lo) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            if (desc) { while (v[i] > pivot) i++; while (v[j] < pivot) j--; }
            else { while (v[i] < pivot) i++; while (v[j] > pivot) j--; }
            if (i <= j) { double t = v[i]; v[i] = v[j]; v[j] = t; i++; j--; }
        }
        if (k - 1 <= j) hi = j;
        else if (k - 1 >= i) lo = i;
        else break;
    }
}

/* (sum of the m largest (desc) or smallest values of v) / m; n >= m */
static double mean_m(FX *s, double *v, int n, int desc) {
    if (s->fast_select) select_k(v, n, s->m, desc);
    else qsort(v, (size_t)n, sizeof(double), desc ? cmp_desc : cmp_asc);
    double t = 0;
    for (int j = 0; j < s->m; j++) t += v[j];
    return t / s->m;
}

static double amub_of(FX *s, const CUL *c) {
    double *v = scr(s, (size_t)c->n);
    for (int i = 0; i < c->n; i++) v[i] = s->isu[c->e[i].sid];
    return mean_m(s, v, c->n, 1);
}

static double mlb_of(FX *s, const CUL *c) {
    double *v = scr(s, (size_t)c->n);
    for (int i = 0; i < c->n; i++) {
        const Ent *e = &c->e[i];
        double mn = INFINITY;
        for (int j = 0; j < e->n; j++) if (c->t[e->off + j].cmin < mn) mn = c->t[e->off + j].cmin;
        v[i] = mn;
    }
    return mean_m(s, v, c->n, 0);
}

typedef struct { double au, ac, fub, mlb, mdb; } Measures;

static Measures node_measures(FX *s, const CUL *c, int32_t last) {
    Measures r;
    double su = 0, sc = 0;
    double *v = scr(s, (size_t)c->n);
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
    r.fub = mean_m(s, v, c->n, 1);
    r.mlb = mlb_of(s, c);
    r.mdb = INFINITY;
    if (s->desc_bound) {
        /* descendant cost bound (notes.md, Lemma D): per sequence, the most expensive occurrence of
           alpha ending at the first end (if an s-extension can follow) or at an end whose itemset
           has an item after last (if an i-extension can follow); sequences where no extension can
           follow are skipped */
        int k = 0;
        for (int i = 0; i < c->n; i++) {
            const Ent *e = &c->e[i];
            const QSeq *sq = &s->db.seq[e->sid];
            const Tup *t0 = &c->t[e->off];
            double lb = INFINITY;
            if (t0->end < sq->n) lb = t0->c;
            for (int j = 0; j < e->n; j++) {
                const Tup *t = &c->t[e->off + j];
                const QItemset *is = &sq->s[t->end - 1];
                if (is->n > 0 && is->q[is->n - 1].item > last && t->c < lb) lb = t->c;
            }
            if (lb < INFINITY) v[k++] = lb;
        }
        if (k >= s->m) r.mdb = mean_m(s, v, k, 0);
    } else r.mdb = r.mlb;
    return r;
}

/* CUL of alpha <>_i y or alpha <>_s y. With fast_select, returns 0 (and an empty list) as soon
   as fewer than m sequences can still contain the extension. */
static int extend(FX *s, const CUL *a, const CUL *y, int serial, CUL *out) {
    memset(out, 0, sizeof *out);
    int i = 0, j = 0;
    while (i < a->n && j < y->n) {
        if (s->fast_select) {
            int left = (a->n - i) < (y->n - j) ? (a->n - i) : (y->n - j);
            if (out->n + left < s->m) { cul_free(out); return 0; }
        }
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
                    Tup t = { ty[q].end, ta[p].c + ty[q].c, ta[p].cmin + ty[q].c, ta[p].u + ty[q].u, ty[q].ru };
                    if (!eb) eb = cul_new_ent(out, ea->sid);
                    cul_push(out, eb, t);
                }
            }
        } else {
            int p = 0; double maxc = -INFINITY, minc = INFINITY, maxu = -INFINITY;
            for (int q = 0; q < ey->n; q++) {
                while (p < ea->n && ta[p].end < ty[q].end) {
                    if (ta[p].c > maxc) maxc = ta[p].c;
                    if (ta[p].cmin < minc) minc = ta[p].cmin;
                    if (ta[p].u > maxu) maxu = ta[p].u;
                    p++;
                }
                if (p == 0) continue;
                Tup t = { ty[q].end, maxc + ty[q].c, minc + ty[q].c, maxu + ty[q].u, ty[q].ru };
                if (!eb) eb = cul_new_ent(out, ea->sid);
                cul_push(out, eb, t);
            }
        }
        i++; j++;
    }
    if (s->fast_select && out->n < s->m) { cul_free(out); return 0; }
    return 1;
}

static void build_item_culs(FX *s) {
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
                Ent *e = (c->n > 0 && c->e[c->n - 1].sid == sid) ? &c->e[c->n - 1] : cul_new_ent(c, sid);
                Tup t = { k + 1, is->q[j].c, is->q[j].c, is->q[j].u, ru };
                cul_push(c, e, t);
            }
        }
        free(after);
    }
}

static void free_item_culs(FX *s) {
    if (!s->item_cul) return;
    for (int32_t x = 0; x < s->nitems; x++) cul_free(&s->item_cul[x]);
    free(s->item_cul); s->item_cul = NULL;
}

static void compute_isu(FX *s) {
    free(s->isu);
    s->isu = xmalloc(sizeof(double) * (size_t)(s->db.nseq ? s->db.nseq : 1));
    for (int i = 0; i < s->db.nseq; i++) {
        double t = 0;
        for (int k = 0; k < s->db.seq[i].n; k++) for (int j = 0; j < s->db.seq[i].s[k].n; j++) t += s->db.seq[i].s[k].q[j].u;
        s->isu[i] = t;
    }
}

/* width test: supp >= m, AMUB >= minu and, with cost_width, MLB <= maxc (all monotone for
   every super-sequence) */
static int passes_width(FX *s, const CUL *c) {
    if (c->n < s->m) return 0;
    if (amub_of(s, c) < s->minu) return 0;
    if (s->cost_width && mlb_of(s, c) > s->maxc) return 0;
    return 1;
}

static void emit(FX *s, const Measures *mz, int supp) {
    s->patterns++;
    if (!s->out) return;
    for (int i = 0; i < s->pat_len; i++) {
        fprintf(s->out, "%d ", s->orig[s->pat[i]]);
        if (i + 1 == s->pat_len || s->pat_set[i + 1] != s->pat_set[i]) fputs("-1 ", s->out);
    }
    fprintf(s->out, "#SUP: %d #AU: %.6f #AC: %.6f\n", supp, mz->au, mz->ac);
}

static void push_item(FX *s, int32_t item, int new_set) {
    if (s->pat_len == s->pat_cap) {
        s->pat_cap = s->pat_cap ? 2 * s->pat_cap : 16;
        s->pat = xrealloc(s->pat, sizeof(int32_t) * (size_t)s->pat_cap);
        s->pat_set = xrealloc(s->pat_set, sizeof(int) * (size_t)s->pat_cap);
    }
    s->pat_set[s->pat_len] = s->pat_len == 0 ? 0 : s->pat_set[s->pat_len - 1] + (new_set ? 1 : 0);
    s->pat[s->pat_len++] = item;
}

static void search(FX *s, const CUL *cul, const int32_t *I, int nI, const int32_t *S, int nS) {
    s->nodes++;
    Measures mz = node_measures(s, cul, s->pat[s->pat_len - 1]);
    if (s->trace) {
        printf("[TRACE] node ");
        for (int i = 0; i < s->pat_len; i++) printf("%s%d", i == 0 ? "" : (s->pat_set[i] != s->pat_set[i - 1] ? "->" : ","), s->orig[s->pat[i]]);
        printf(" AMUB=%.4f FUB=%.4f MLB=%.4f MDB=%.4f au=%.4f ac=%.4f supp=%d\n", amub_of(s, cul), mz.fub, mz.mlb, mz.mdb, mz.au, mz.ac, cul->n);
    }
    if (mz.fub < s->minu || mz.mlb > s->maxc) return;                       /* alpha and its subtree */
    if (mz.au >= s->minu && mz.ac <= s->maxc) emit(s, &mz, cul->n);
    if (mz.mdb > s->maxc) return;                                            /* proper descendants */
    int32_t x = s->pat[s->pat_len - 1];
    int32_t *Ia = xmalloc(sizeof(int32_t) * (size_t)(nI ? nI : 1)), *Sa = xmalloc(sizeof(int32_t) * (size_t)(nS ? nS : 1));
    CUL *Ic = xmalloc(sizeof(CUL) * (size_t)(nI ? nI : 1)), *Sc = xmalloc(sizeof(CUL) * (size_t)(nS ? nS : 1));
    int ni = 0, ns = 0;
    for (int k = 0; k < nI; k++) {
        int32_t y = I[k];
        if (y <= x || !s->icmap[(size_t)x * s->nitems + y]) continue;
        CUL c; s->extensions++;
        if (!extend(s, cul, &s->item_cul[y], 0, &c)) { s->abandoned++; continue; }
        if (passes_width(s, &c)) { Ia[ni] = y; Ic[ni++] = c; } else cul_free(&c);
    }
    for (int k = 0; k < nS; k++) {
        int32_t y = S[k];
        if (!s->scmap[(size_t)x * s->nitems + y]) continue;
        CUL c; s->extensions++;
        if (!extend(s, cul, &s->item_cul[y], 1, &c)) { s->abandoned++; continue; }
        if (passes_width(s, &c)) { Sa[ns] = y; Sc[ns++] = c; } else cul_free(&c);
    }
    for (int k = 0; k < ni; k++) {
        push_item(s, Ia[k], 0);
        search(s, &Ic[k], Ia, ni, Sa, ns);
        s->pat_len--;
        cul_free(&Ic[k]);
    }
    for (int k = 0; k < ns; k++) {
        push_item(s, Sa[k], 1);
        search(s, &Sc[k], Sa, ns, Sa, ns);
        s->pat_len--;
        cul_free(&Sc[k]);
    }
    free(Ia); free(Sa); free(Ic); free(Sc);
}

/* RedS: keep the items that pass the width test; returns the number of removed items */
static long reds_round(FX *s) {
    long removed = 0;
    for (int32_t x = 0; x < s->nitems; x++) {
        if (!s->alive[x]) continue;
        if (s->item_cul[x].n == 0 || !passes_width(s, &s->item_cul[x])) { s->alive[x] = 0; if (s->item_cul[x].n) removed++; }
    }
    if (!removed) return 0;
    for (int i = 0; i < s->db.nseq; i++) {
        QSeq *sq = &s->db.seq[i];
        int w = 0;
        for (int k = 0; k < sq->n; k++) {
            QItemset *is = &sq->s[k];
            int v = 0;
            for (int j = 0; j < is->n; j++) if (s->alive[is->q[j].item]) is->q[v++] = is->q[j];
            is->n = v;
            if (v > 0) sq->s[w++] = *is; else free(is->q);
        }
        sq->n = w;
    }
    free_item_culs(s);
    compute_isu(s);
    build_item_culs(s);
    return removed;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    (void)ds;
    DM_FLCHUS_X_Params *p = (DM_FLCHUS_X_Params *)params;
    if (!p || !p->input_path) return DM_ERROR_INVALID_PARAM;
    FX s; memset(&s, 0, sizeof s);
    if (!read_db(p->input_path, &s.db)) return DM_ERROR_INVALID_PARAM;
    s.cost_width = p->cost_width; s.reds_fixpoint = p->reds_fixpoint; s.fast_select = p->fast_select;
    s.desc_bound = p->desc_bound;
    s.orig = densify(&s.db);
    s.nitems = s.db.max_item + 1;
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

    compute_isu(&s);
    build_item_culs(&s);
    s.alive = xmalloc((size_t)s.nitems);
    memset(s.alive, 1, (size_t)s.nitems);
    s.trace = getenv("FLCHUS_TRACE") != NULL;
    if (s.trace)
        for (int32_t x = 0; x < s.nitems; x++)
            if (s.item_cul[x].n > 0)
                printf("[TRACE] item %d supp=%d AMUB=%.4f MLB=%.4f\n", s.orig[x], s.item_cul[x].n,
                       s.item_cul[x].n >= s.m ? amub_of(&s, &s.item_cul[x]) : -1.0, s.item_cul[x].n >= s.m ? mlb_of(&s, &s.item_cul[x]) : -1.0);
    long r;
    do { r = reds_round(&s); s.removed_items += r; s.reds_rounds++; } while (r > 0 && s.reds_fixpoint);
    int32_t *S = xmalloc(sizeof(int32_t) * (size_t)s.nitems); int nS = 0;
    for (int32_t x = 0; x < s.nitems; x++) if (s.alive[x] && s.item_cul[x].n > 0) S[nS++] = x;
    if (s.trace) { printf("[TRACE] S ="); for (int a = 0; a < nS; a++) printf(" %d", s.orig[S[a]]); printf("\n"); }

    s.icmap = calloc((size_t)s.nitems * (size_t)s.nitems, 1);
    s.scmap = calloc((size_t)s.nitems * (size_t)s.nitems, 1);
    for (int a = 0; a < nS; a++)
        for (int b = 0; b < nS; b++) {
            int32_t x = S[a], y = S[b];
            CUL c;
            if (y > x && extend(&s, &s.item_cul[x], &s.item_cul[y], 0, &c)) {
                if (passes_width(&s, &c)) s.icmap[(size_t)x * s.nitems + y] = 1;
                cul_free(&c);
            }
            if (extend(&s, &s.item_cul[x], &s.item_cul[y], 1, &c)) {
                if (passes_width(&s, &c)) s.scmap[(size_t)x * s.nitems + y] = 1;
                cul_free(&c);
            }
        }

    const char *outp = getenv("DM_FLCHUS_OUT");
    if (outp) s.out = fopen(outp, "w");
    printf("[FLCHUS-X] |D|=%d m=%d minu=%.6f maxc=%.6f |S|=%d removed=%ld rounds=%ld costW=%d fix=%d sel=%d desc=%d\n",
           s.db.nseq, s.m, s.minu, s.maxc, nS, s.removed_items, s.reds_rounds, s.cost_width, s.reds_fixpoint, s.fast_select, s.desc_bound);
    for (int a = 0; a < nS; a++) {
        push_item(&s, S[a], 1);
        search(&s, &s.item_cul[S[a]], S, nS, S, nS);
        s.pat_len--;
    }
    if (s.out) fclose(s.out);
    printf("[FLCHUS-X] patterns=%ld nodes=%ld extensions=%ld abandoned=%ld\n", s.patterns, s.nodes, s.extensions, s.abandoned);
    dm_bench_record_results((size_t)s.patterns, (size_t)s.nodes);

    free_item_culs(&s);
    free(s.alive); free(s.icmap); free(s.scmap); free(S);
    free(s.isu); free(s.pat); free(s.pat_set); free(s.scr);
    free(s.orig);
    free_db(&s.db);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "flchus_x",
    .name = "FLCHUS-X",
    .description = "Proposed sound and complete FLCHUSM miner (MLB cost bound, cost-aware width pruning).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
