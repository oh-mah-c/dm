/*
 * HUE-Span (Fournier-Viger, Yang, Lin, Yun, ADMA 2019), ported from the authors' reference
 * implementation in SPMF (AlgoHUESpan.java, ComplexSequence.java, FiniteStateMachine.java,
 * GPL-3.0, Copyright Peng Yang, Philippe Fournier-Viger 2019).
 *
 * Fidelity notes (the behaviour of the Java code is reproduced, including where it departs
 * from the text of the paper):
 *  - time points are physical line numbers of the input file, comment/metadata/empty lines
 *    included (AlgoHUESpan.scanDatabaseToFindHighUtilitySingleEpisodes);
 *  - all utility arithmetic is Java 32-bit int arithmetic (two's-complement wrap-around);
 *    minUtilityAbsolute is the double sequenceUtility * ratio;
 *  - events at each time point are ordered by increasing AWU with a stable sort
 *    (Collections.sort with comparator a - b), or by identifier when ERU is disabled;
 *  - the I-extension candidates, ERU, EEUCS tests and the serial extension bound follow the
 *    Java code line by line; FiniteStateMachine.scan keeps its side effect on the utility.
 * The order in which candidate episodes are expanded differs from Java's HashMap iteration
 * order; this changes only the order of the output lines, not the set of episodes.
 *
 * If the environment variable DM_HUE_OUT is set, the episodes are written to that file in
 * the SPMF output format ("e1 e2 -1 e3 -1 #UTIL: u").
 */
#include "algorithms/hue_span.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ---------- Java int arithmetic ---------- */
static inline int32_t jadd(int32_t a, int32_t b) { return (int32_t)((uint32_t)a + (uint32_t)b); }
static inline int32_t jsub(int32_t a, int32_t b) { return (int32_t)((uint32_t)a - (uint32_t)b); }

static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "[HUE-Span] out of memory\n"); abort(); } return p; }
static void *xrealloc(void *p, size_t n) { p = realloc(p, n ? n : 1); if (!p) { fprintf(stderr, "[HUE-Span] out of memory\n"); abort(); } return p; }

/* ---------- growable int list (java.util.List<Integer>) ---------- */
typedef struct { int32_t *a; size_t n, cap; } IList;
static void il_push(IList *l, int32_t v) {
    if (l->n == l->cap) { l->cap = l->cap ? 2 * l->cap : 8; l->a = xrealloc(l->a, l->cap * sizeof(int32_t)); }
    l->a[l->n++] = v;
}
static void il_free(IList *l) { free(l->a); l->a = NULL; l->n = l->cap = 0; }

/* ---------- hash map int -> int (java.util.HashMap<Integer,Integer>) ---------- */
typedef struct { int64_t *key; int32_t *val; uint8_t *used; size_t cap, n; } IMap;
static uint64_t mix64(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33; return x; }
static void im_init(IMap *m, size_t cap) {
    m->cap = 16; while (m->cap < 2 * cap) m->cap <<= 1;
    m->n = 0; m->key = xmalloc(m->cap * sizeof(int64_t)); m->val = xmalloc(m->cap * sizeof(int32_t));
    m->used = calloc(m->cap, 1); if (!m->used) abort();
}
static void im_free(IMap *m) { free(m->key); free(m->val); free(m->used); memset(m, 0, sizeof *m); }
static long im_find(const IMap *m, int64_t k) {
    if (!m->cap) return -1;
    size_t i = mix64((uint64_t)k) & (m->cap - 1);
    while (m->used[i]) { if (m->key[i] == k) return (long)i; i = (i + 1) & (m->cap - 1); }
    return -1;
}
static void im_put(IMap *m, int64_t k, int32_t v);
static void im_grow(IMap *m) {
    IMap o = *m; im_init(m, o.cap);
    for (size_t i = 0; i < o.cap; i++) if (o.used[i]) im_put(m, o.key[i], o.val[i]);
    im_free(&o);
}
static void im_put(IMap *m, int64_t k, int32_t v) {
    if (!m->cap) im_init(m, 16);
    long f = im_find(m, k);
    if (f >= 0) { m->val[f] = v; return; }
    if (2 * (m->n + 1) > m->cap) im_grow(m);
    size_t i = mix64((uint64_t)k) & (m->cap - 1);
    while (m->used[i]) i = (i + 1) & (m->cap - 1);
    m->used[i] = 1; m->key[i] = k; m->val[i] = v; m->n++;
}
static void im_remove(IMap *m, int64_t k) {          /* backward-shift deletion */
    long f = im_find(m, k);
    if (f < 0) return;
    size_t i = (size_t)f;
    m->used[i] = 0; m->n--;
    size_t j = i;
    for (;;) {
        j = (j + 1) & (m->cap - 1);
        if (!m->used[j]) break;
        size_t h = mix64((uint64_t)m->key[j]) & (m->cap - 1);
        int move = (i <= j) ? (h <= i || h > j) : (h <= i && h > j);
        if (move) { m->key[i] = m->key[j]; m->val[i] = m->val[j]; m->used[i] = 1; m->used[j] = 0; i = j; }
    }
}

/* ---------- complex sequence (ComplexSequence) ---------- */
typedef struct { int32_t ev, u; } Pair;
typedef struct { Pair *p; size_t n, cap; int32_t total; int exists; } TP;

typedef struct {
    /* parameters */
    double min_abs;
    int max_dur, check_max, use_eru, out_single, use_cooc, pruning_prefix;
    int fixed;                        /* corrected variant (not in SPMF), see get_maximal_utility_dp */
    /* sequence */
    TP *tp; int32_t largest;          /* time points 1..largest */
    int64_t sequence_utility;
    /* single candidates: per event, mo list and utility list (MoListUtilityList) */
    int32_t max_ev;
    IList *s_mo, *s_ut; uint8_t *s_present;
    /* EEUCS: key (i<<32 | j) */
    IMap cooc_equals, cooc_after;
    uint8_t *row_equals, *row_after;  /* coocMapX.get(i) != null */
    /* output & stats */
    FILE *out;
    long hue_count, candidate_count, combinated, matrix_pruning, ub_pruning, non_max;
    size_t footprint;
} HS;

static inline int64_t key2(int32_t i, int32_t j) { return ((int64_t)(uint32_t)i << 32) | (uint32_t)j; }

static int32_t total_of_duration(const HS *s, int32_t start, int32_t end) {  /* getTotalUtilityOfDuration */
    if (start > s->largest) return 0;
    if (end > s->largest) end = s->largest;
    int32_t tot = 0;
    for (int32_t t = start < 1 ? 1 : start; t <= end; t++)  /* time points < 1 never exist */
        if (s->tp[t].exists) tot = jadd(tot, s->tp[t].total);
    return tot;
}

static int32_t get_irutil(const HS *s, int32_t t, int32_t item) {  /* getIrutil */
    const TP *x = &s->tp[t];
    int32_t iu = 0;
    for (size_t i = x->n - 1; i > 0; i--) {
        if (x->p[i].ev != item) iu = jadd(iu, x->p[i].u);
        else break;
    }
    return iu;
}

/* getPairsForIextension: pairs after the first position whose predecessor is lastItem */
static void pairs_for_iext(const HS *s, int32_t t, int32_t last_item, const Pair **out, size_t *n) {
    const TP *x = &s->tp[t];
    *out = NULL; *n = 0;
    for (size_t i = 1; i < x->n; i++)
        if (x->p[i - 1].ev == last_item) { *out = x->p + i; *n = x->n - i; return; }
}

/* ---------- episodes ---------- */
typedef struct { int32_t *ev; size_t n; } ISet;
typedef struct { ISet *s; size_t n; } Episode;

static Episode ep_copy_prefix(const Episode *e, size_t k) {
    Episode r; r.n = k; r.s = xmalloc((k + 1) * sizeof(ISet));
    for (size_t i = 0; i < k; i++) {
        r.s[i].n = e->s[i].n; r.s[i].ev = xmalloc(e->s[i].n * sizeof(int32_t));
        memcpy(r.s[i].ev, e->s[i].ev, e->s[i].n * sizeof(int32_t));
    }
    return r;
}
static void ep_free(Episode *e) { for (size_t i = 0; i < e->n; i++) free(e->s[i].ev); free(e->s); }

static void save(HS *s, const Episode *e, int32_t util) {
    s->hue_count++;
    size_t len = 0;
    for (size_t i = 0; i < e->n; i++) len += e->s[i].n;
    s->footprint += len;
    if (!s->out) return;
    for (size_t i = 0; i < e->n; i++) {
        for (size_t j = 0; j < e->s[i].n; j++) fprintf(s->out, "%d ", e->s[i].ev[j]);
        fputs("-1 ", s->out);
    }
    fprintf(s->out, "#UTIL: %d\n", util);
}

/* ---------- getEventsUtilityByTID / FiniteStateMachine / getMaximalUtility ---------- */
static int32_t events_utility_by_tid(const HS *s, const ISet *set, int32_t t) {
    const TP *x = &s->tp[t];
    int32_t u = 0; size_t idx = 0;
    for (size_t j = 0; j < x->n; j++) {
        if (idx == set->n) break;
        if (x->p[j].ev == set->ev[idx]) { idx++; u = jadd(u, x->p[j].u); }
    }
    return idx == set->n ? u : 0;
}

typedef struct { size_t pos; int32_t utility; } FSM;

static int fsm_scan(FSM *f, const Episode *e, const TP *x) {    /* FiniteStateMachine.scan */
    int32_t uset = 0; size_t idx = 0, len = e->s[f->pos].n;
    for (size_t j = 0; j < x->n; j++) {
        if (x->p[j].ev == e->s[f->pos].ev[idx]) { idx++; uset = jadd(uset, x->p[j].u); }
        if (idx == len) { f->utility = jadd(f->utility, uset); break; }
    }
    return idx == len;
}

static int32_t get_maximal_utility(const HS *s, const Episode *e, int32_t start, int32_t end) {
    int32_t maxu = 0;
    size_t n = 0, cap = 8;
    FSM *L = xmalloc(cap * sizeof(FSM));
    int32_t first = events_utility_by_tid(s, &e->s[0], start);
    L[n++] = (FSM){1, first};                      /* constructor: utility += first; transit() */
    for (int32_t t = start + 1; t <= end; t++) {
        if (!s->tp[t].exists) continue;
        const TP *x = &s->tp[t];
        for (long j = (long)n - 1; j >= 0; j--) {
            if (fsm_scan(&L[j], e, x)) {
                L[j].pos++;                            /* transit */
                if ((size_t)j == n - 1) {
                    if (n == cap) { cap *= 2; L = xrealloc(L, cap * sizeof(FSM)); }
                    L[n++] = (FSM){1, first};
                }
                if (L[j].pos == e->n) {                /* isEnd */
                    if (L[j].utility > maxu) maxu = L[j].utility;
                    memmove(L + j, L + j + 1, (n - (size_t)j - 1) * sizeof(FSM)); n--;
                }
                /* as in Java, L[j] now denotes the element that followed a removed one */
                if (j >= 1 && L[j].pos == L[j - 1].pos && !fsm_scan(&L[j - 1], e, x)) {
                    if (jsub(L[j].utility, L[j - 1].utility) >= 0) {
                        memmove(L + j - 1, L + j, (n - (size_t)j) * sizeof(FSM)); n--;
                        j--;
                    } else {
                        memmove(L + j, L + j + 1, (n - (size_t)j - 1) * sizeof(FSM)); n--;
                    }
                }
            }
        }
    }
    free(L);
    return maxu;
}

/* Corrected variant (fixed = 1, not part of the reference implementation): maximum of
 * sum_i u(SE_i, t_i) over order-respecting matchings start = t_1 < ... < t_k <= end of the
 * episode e = <SE_1..SE_k>, i.e. the maximal utility of Definition 12 for the prefix of a
 * serial extension. The reference getMaximalUtility over-counts because
 * FiniteStateMachine.scan adds utility as a side effect when it is called again in the
 * duplicate-state test. best[i] = best utility of a matching of SE_1..SE_i so far
 * (INT64_MIN = none); positions are updated in decreasing order so that one time point
 * matches at most one set. */
static int32_t get_maximal_utility_dp(const HS *s, const Episode *e, int32_t start, int32_t end) {
    size_t k = e->n;
    int64_t *best = xmalloc((k + 1) * sizeof(int64_t));
    for (size_t i = 0; i <= k; i++) best[i] = INT64_MIN;
    if (start >= 1 && start <= s->largest && s->tp[start].exists) {
        int32_t u = events_utility_by_tid(s, &e->s[0], start);
        if (u > 0 || e->s[0].n == 0) best[1] = u;
    }
    for (int32_t t = start + 1; t <= end && t <= s->largest; t++) {
        if (!s->tp[t].exists) continue;
        for (size_t i = k; i >= 2; i--) {
            if (best[i - 1] == INT64_MIN) continue;
            int32_t u = events_utility_by_tid(s, &e->s[i - 1], t);
            if (u <= 0) continue;
            if (best[i - 1] + u > best[i]) best[i] = best[i - 1] + u;
        }
    }
    int64_t r = best[k] == INT64_MIN ? 0 : best[k];
    free(best);
    return (int32_t)r;
}

/* ---------- calculateUtilityAndUpperBoundOfMOs / repaiMOSet ---------- */
typedef struct { IList st, en, prev, ut; } BetaInfo;

static void repair_mo_set(HS *s, const BetaInfo *b, int32_t *tot, int32_t *ub) {
    IMap se = {0}, es = {0};
    for (size_t pos = 0; pos < b->st.n; pos++) {
        int32_t sp = b->st.a[pos], ep = b->en.a[pos];
        long f = im_find(&se, sp);
        if (f >= 0) {
            int32_t exist_end = se.val[f];
            if (ep < exist_end) { im_put(&se, sp, ep); im_remove(&es, exist_end); im_put(&es, ep, sp); }
        } else {
            im_put(&se, sp, ep);
            long g = im_find(&es, ep);
            if (g >= 0) {
                int32_t exist_start = es.val[g];
                if (sp > exist_start) { im_put(&es, ep, sp); im_remove(&se, exist_start); im_put(&se, sp, ep); }
            } else im_put(&es, ep, sp);
        }
    }
    int32_t t = 0, u = 0;
    for (size_t pos = 0; pos < b->st.n; pos++) {
        int32_t sp = b->st.a[pos], ep = b->en.a[pos];
        long f = im_find(&se, sp);
        if (f >= 0 && se.val[f] == ep) {
            t = jadd(t, b->ut.a[pos]);
            u = jadd(u, jadd(b->prev.a[pos], total_of_duration(s, ep, sp + s->max_dur - 1)));
        }
    }
    im_free(&se); im_free(&es);
    *tot = t; *ub = u;
}

static void utility_and_ub(HS *s, int32_t beta, const BetaInfo *b, int32_t *tot, int32_t *ub) {
    if (!s->use_eru && !s->use_cooc) { repair_mo_set(s, b, tot, ub); return; }
    int32_t t = 0, u = 0, prev_start = -1;
    for (size_t pos = 0; pos < b->st.n; pos++) {
        int32_t sp = b->st.a[pos];
        if (sp != prev_start) {
            prev_start = sp;
            t = jadd(t, b->ut.a[pos]);
            int32_t ep = b->en.a[pos];
            if (s->use_eru)
                u = jadd(u, jadd(jadd(b->ut.a[pos], get_irutil(s, ep, beta)),
                                 total_of_duration(s, ep + 1, sp + s->max_dur - 1)));
            else
                u = jadd(u, jadd(b->prev.a[pos], total_of_duration(s, ep, sp + s->max_dur - 1)));
        }
    }
    *tot = t; *ub = u;
}

/* ---------- per-call map beta -> BetaInfo (insertion ordered) ---------- */
typedef struct { IMap idx; int32_t *beta; BetaInfo *info; size_t n, cap; } BetaMap;
static BetaInfo *bm_get(BetaMap *m, int32_t beta, int *created) {
    long f = im_find(&m->idx, beta);
    if (f >= 0) { *created = 0; return &m->info[m->idx.val[f]]; }
    if (m->n == m->cap) {
        m->cap = m->cap ? 2 * m->cap : 8;
        m->beta = xrealloc(m->beta, m->cap * sizeof(int32_t));
        m->info = xrealloc(m->info, m->cap * sizeof(BetaInfo));
    }
    im_put(&m->idx, beta, (int32_t)m->n);
    m->beta[m->n] = beta; memset(&m->info[m->n], 0, sizeof(BetaInfo));
    *created = 1;
    return &m->info[m->n++];
}
static void bm_free(BetaMap *m) {
    for (size_t i = 0; i < m->n; i++) { il_free(&m->info[i].st); il_free(&m->info[i].en); il_free(&m->info[i].prev); il_free(&m->info[i].ut); }
    free(m->beta); free(m->info); im_free(&m->idx);
}

/* EEUCS test of one event pair; returns 1 if beta must be pruned (Java: map null -> continue) */
static int cooc_prunes(const HS *s, const IMap *m, const uint8_t *row, int32_t item_i, int32_t beta) {
    if (item_i < 0 || item_i > s->max_ev || !row[item_i]) return 0;
    long f = im_find(m, key2(item_i, beta));
    return f < 0 || (double)m->val[f] < s->min_abs;
}

static void mine_hue(HS *s, const Episode *e, const BetaInfo *a);

static void mine_simultaneous(HS *s, const Episode *e, const BetaInfo *a) {
    BetaMap bm = {0};
    IMap pruning = {0};
    const ISet *last = &e->s[e->n - 1];
    int32_t last_item = last->ev[last->n - 1];
    for (size_t i = 0; i < a->en.n; i++) {
        int32_t sp = a->st.a[i], ep = a->en.a[i], aprev = a->prev.a[i], au = a->ut.a[i];
        const Pair *pp; size_t np;
        pairs_for_iext(s, ep, last_item, &pp, &np);
        for (size_t j = 0; j < np; j++) {
            int32_t beta = pp[j].ev, bu = jadd(au, pp[j].u);
            if (s->use_cooc) {
                if (im_find(&pruning, beta) >= 0) continue;
                int pruned = 0;
                for (size_t k = 0; k < last->n && !pruned; k++)
                    if (cooc_prunes(s, &s->cooc_equals, s->row_equals, last->ev[k], beta)) pruned = 1;
                if (!pruned && s->pruning_prefix)
                    for (size_t z = 0; z + 1 < e->n && !pruned; z++)
                        for (size_t k = 0; k < e->s[z].n && !pruned; k++)
                            if (cooc_prunes(s, &s->cooc_after, s->row_after, e->s[z].ev[k], beta)) pruned = 1;
                if (pruned) { im_put(&pruning, beta, 1); continue; }
            }
            int created;
            BetaInfo *b = bm_get(&bm, beta, &created);
            il_push(&b->st, sp); il_push(&b->en, ep); il_push(&b->prev, aprev); il_push(&b->ut, bu);
        }
    }
    s->matrix_pruning += (long)pruning.n;
    s->combinated += (long)pruning.n;
    for (size_t q = 0; q < bm.n; q++) {
        s->combinated++;
        int32_t beta = bm.beta[q], tot, ub;
        utility_and_ub(s, beta, &bm.info[q], &tot, &ub);
        if ((double)ub >= s->min_abs) {
            Episode be = ep_copy_prefix(e, e->n - 1);
            be.s[be.n].n = last->n + 1;
            be.s[be.n].ev = xmalloc((last->n + 1) * sizeof(int32_t));
            memcpy(be.s[be.n].ev, last->ev, last->n * sizeof(int32_t));
            be.s[be.n].ev[last->n] = beta;
            be.n++;
            if ((double)tot >= s->min_abs) save(s, &be, tot);
            mine_hue(s, &be, &bm.info[q]);
            ep_free(&be);
        } else s->ub_pruning++;
    }
    bm_free(&bm); im_free(&pruning);
}

static void mine_serial(HS *s, const Episode *e, const BetaInfo *a) {
    BetaMap bm = {0};
    IMap pruning = {0};
    const ISet *last = &e->s[e->n - 1];
    int32_t prev_start = -1;
    for (size_t i = 0; i < a->en.n; i++) {
        int32_t sp = a->st.a[i];
        if (sp == prev_start) continue;
        prev_start = sp;
        int32_t ep = a->en.a[i], au = a->ut.a[i];
        size_t j = i + 1;
        while (j < a->en.n && a->st.a[j] == sp) j++;
        int32_t end_next = s->largest;
        if (j < a->en.n) end_next = a->en.a[j];
        int32_t bound = end_next < sp + s->max_dur - 1 ? end_next : sp + s->max_dur - 1;
        if (!s->use_eru && !s->use_cooc) bound = sp + s->max_dur - 1;
        int32_t max_au = au;
        for (int32_t t = ep + 1; t <= bound; t++) {
            if (t > s->largest || !s->tp[t].exists) continue;   /* getEventSetAndItsUtilityByTID: empty */
            const TP *x = &s->tp[t];
            for (size_t k = 0; k < x->n; k++) {
                int32_t beta = x->p[k].ev;
                if (s->check_max && e->n >= 2)
                    max_au = s->fixed ? get_maximal_utility_dp(s, e, sp, t - 1) : get_maximal_utility(s, e, sp, t - 1);
                int32_t bu = jadd(max_au, x->p[k].u);
                if (s->use_cooc) {
                    if (im_find(&pruning, beta) >= 0) continue;
                    int pruned = 0;
                    for (size_t z = 0; z < last->n && !pruned; z++)
                        if (cooc_prunes(s, &s->cooc_after, s->row_after, last->ev[z], beta)) pruned = 1;
                    if (!pruned && s->pruning_prefix)
                        for (size_t z = 0; z + 1 < e->n && !pruned; z++)
                            for (size_t y = 0; y < e->s[z].n && !pruned; y++)
                                if (cooc_prunes(s, &s->cooc_after, s->row_after, e->s[z].ev[y], beta)) pruned = 1;
                    if (pruned) { im_put(&pruning, beta, 1); continue; }
                }
                int created;
                BetaInfo *b = bm_get(&bm, beta, &created);
                if (created && max_au > au) s->non_max++;
                il_push(&b->st, sp); il_push(&b->en, t); il_push(&b->prev, au); il_push(&b->ut, bu);
            }
        }
    }
    s->matrix_pruning += (long)pruning.n;
    s->combinated += (long)pruning.n;
    for (size_t q = 0; q < bm.n; q++) {
        s->combinated++;
        int32_t beta = bm.beta[q], tot, ub;
        utility_and_ub(s, beta, &bm.info[q], &tot, &ub);
        if ((double)ub >= s->min_abs) {
            Episode be = ep_copy_prefix(e, e->n);
            be.s[be.n].n = 1;
            be.s[be.n].ev = xmalloc(sizeof(int32_t));
            be.s[be.n].ev[0] = beta;
            be.n++;
            if ((double)tot >= s->min_abs) save(s, &be, tot);
            mine_hue(s, &be, &bm.info[q]);
            ep_free(&be);
        } else s->ub_pruning++;
    }
    bm_free(&bm); im_free(&pruning);
}

static void mine_hue(HS *s, const Episode *e, const BetaInfo *a) {
    s->candidate_count++;
    mine_simultaneous(s, e, a);
    mine_serial(s, e, a);
}

/* ---------- input (scanDatabaseToFindHighUtilitySingleEpisodes) ---------- */
static void tp_push(TP *x, int32_t ev, int32_t u) {
    if (x->n == x->cap) { x->cap = x->cap ? 2 * x->cap : 4; x->p = xrealloc(x->p, x->cap * sizeof(Pair)); }
    x->p[x->n++] = (Pair){ev, u};
}

static int read_input(HS *s, const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "[HUE-Span] cannot open %s\n", path); return 0; }
    size_t cap = 1024; s->tp = calloc(cap, sizeof(TP));
    char *line = NULL; size_t lcap = 0; ssize_t len;
    int32_t ln = 0;
    s->max_ev = -1;
    /* first pass stores pairs; single-candidate lists are built afterwards in line order */
    while ((len = getline(&line, &lcap, f)) != -1) {
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
        ln++;
        if ((size_t)ln >= cap) { size_t nc = cap * 2; s->tp = xrealloc(s->tp, nc * sizeof(TP)); memset(s->tp + cap, 0, (nc - cap) * sizeof(TP)); cap = nc; }
        if (len == 0 || line[0] == '#' || line[0] == '%' || line[0] == '@') continue;
        char *c1 = strchr(line, ':'); if (!c1) continue;
        char *c2 = strchr(c1 + 1, ':'); if (!c2) continue;
        *c1 = 0; *c2 = 0;
        int32_t total = (int32_t)strtol(c1 + 1, NULL, 10);
        char *pe = line, *pu = c2 + 1;
        TP *x = &s->tp[ln];
        for (;;) {
            char *ee, *ue;
            long ev = strtol(pe, &ee, 10);
            if (ee == pe) break;
            long u = strtol(pu, &ue, 10);
            tp_push(x, (int32_t)ev, (int32_t)u);
            if (ev > s->max_ev) s->max_ev = (int32_t)ev;
            pe = ee; pu = ue;
        }
        x->exists = x->n > 0;
        x->total = total;
        s->sequence_utility += total;
    }
    free(line); fclose(f);
    s->largest = ln;
    if ((size_t)ln + 2 > cap) { s->tp = xrealloc(s->tp, ((size_t)ln + 2) * sizeof(TP)); memset(s->tp + cap, 0, ((size_t)ln + 2 - cap) * sizeof(TP)); }
    return 1;
}

/* stable merge sort of pairs with Java comparator act[o1] - act[o2] (int) or o1 - o2 */
static const int32_t *g_act;
static int g_by_act, g_tie_by_id;
static int pair_cmp(const Pair *x, const Pair *y) {
    int32_t d = g_by_act ? jsub(g_act[x->ev], g_act[y->ev]) : jsub(x->ev, y->ev);
    if (d == 0 && g_tie_by_id) d = jsub(x->ev, y->ev);   /* fixed variant: one total order on events */
    return (d > 0) - (d < 0);
}
static void msort(Pair *a, Pair *tmp, size_t n) {
    if (n < 2) return;
    size_t m = n / 2;
    msort(a, tmp, m); msort(a + m, tmp, n - m);
    size_t i = 0, j = m, k = 0;
    while (i < m && j < n) tmp[k++] = (pair_cmp(&a[j], &a[i]) < 0) ? a[j++] : a[i++];
    while (i < m) tmp[k++] = a[i++];
    while (j < n) tmp[k++] = a[j++];
    memcpy(a, tmp, n * sizeof(Pair));
}

static void prune_single_events(HS *s) {     /* ComplexSequence.pruneSingleEventsByUpperBound */
    int32_t *act = calloc((size_t)s->max_ev + 1, sizeof(int32_t));
    uint8_t *seen = calloc((size_t)s->max_ev + 1, 1);
    for (int32_t t = 1; t <= s->largest; t++) {
        if (!s->tp[t].exists) continue;
        int32_t au = total_of_duration(s, t - s->max_dur + 1, t + s->max_dur - 1);
        for (size_t k = 0; k < s->tp[t].n; k++) { int32_t ev = s->tp[t].p[k].ev; act[ev] = jadd(act[ev], au); seen[ev] = 1; }
    }
    for (int32_t ev = 0; ev <= s->max_ev; ev++)
        if (seen[ev] && (double)act[ev] < s->min_abs) s->s_present[ev] = 0;
    g_act = act; g_by_act = s->use_eru; g_tie_by_id = s->fixed;
    for (int32_t t = 1; t <= s->largest; t++) {
        TP *x = &s->tp[t];
        if (!x->exists) continue;
        int32_t removed = 0; size_t w = 0;
        for (size_t k = 0; k < x->n; k++) {
            if (!s->s_present[x->p[k].ev]) removed = jadd(removed, x->p[k].u);
            else x->p[w++] = x->p[k];
        }
        x->n = w;
        if (x->n > 0) {
            Pair *tmp = xmalloc(x->n * sizeof(Pair));
            msort(x->p, tmp, x->n);
            free(tmp);
            x->total = jsub(x->total, removed);
        } else x->exists = 0;
    }
    free(act); free(seen);
}

static void build_cooc(HS *s) {               /* AlgoHUESpan.buildCoocUtilityMatrix */
    im_init(&s->cooc_equals, 1024); im_init(&s->cooc_after, 1024);
    s->row_equals = calloc((size_t)s->max_ev + 1, 1);
    s->row_after = calloc((size_t)s->max_ev + 1, 1);
    int32_t *processed = xmalloc(((size_t)s->max_ev + 1) * sizeof(int32_t));
    uint8_t *in_after = calloc((size_t)s->max_ev + 1, 1);
    int32_t *after_list = xmalloc(((size_t)s->max_ev + 1) * sizeof(int32_t));
    for (int32_t t = 1; t <= s->largest; t++) {
        if (!s->tp[t].exists) continue;
        const TP *x = &s->tp[t];
        size_t np = 0;
        for (size_t k = 0; k < x->n; k++) {
            int32_t ii = x->p[k].ev;
            int32_t ueq = total_of_duration(s, t - s->max_dur + 1, t + s->max_dur - 1);
            for (size_t q = 0; q < np; q++) {
                int32_t jj = processed[q];
                long f = im_find(&s->cooc_equals, key2(jj, ii));
                im_put(&s->cooc_equals, key2(jj, ii), f < 0 ? ueq : jadd(s->cooc_equals.val[f], ueq));
                s->row_equals[jj] = 1;
            }
            int dup = 0;                                   /* HashSet semantics */
            for (size_t q = 0; q < np; q++) if (processed[q] == ii) { dup = 1; break; }
            if (!dup) processed[np++] = ii;
            size_t na = 0;
            for (int32_t ta = t + 1; ta <= t + s->max_dur - 1; ta++) {
                if (ta > s->largest || !s->tp[ta].exists) continue;
                const TP *y = &s->tp[ta];
                for (size_t m = 0; m < y->n; m++) {
                    int32_t jj = y->p[m].ev;
                    if (in_after[jj]) continue;
                    int32_t ua = total_of_duration(s, ta - s->max_dur + 1, t + s->max_dur - 1);
                    long f = im_find(&s->cooc_after, key2(ii, jj));
                    im_put(&s->cooc_after, key2(ii, jj), f < 0 ? ua : jadd(s->cooc_after.val[f], ua));
                    s->row_after[ii] = 1;
                    in_after[jj] = 1; after_list[na++] = jj;
                }
            }
            for (size_t q = 0; q < na; q++) in_after[after_list[q]] = 0;
        }
    }
    free(processed); free(in_after); free(after_list);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    (void)ds;
    DM_HUE_SPAN_Params *p = (DM_HUE_SPAN_Params *)params;
    if (!p || !p->input_path) { fprintf(stderr, "[HUE-Span] missing parameters\n"); return DM_ERROR_INVALID_PARAM; }
    HS s;
    memset(&s, 0, sizeof s);
    s.max_dur = p->max_duration; s.check_max = p->check_maximum_utility; s.use_eru = p->use_tighter_upper_bound;
    s.out_single = p->output_single_events; s.use_cooc = p->use_cooc_matrix; s.pruning_prefix = p->pruning_prefix;
    s.fixed = p->fixed_utility;
    if (!read_input(&s, p->input_path)) return DM_ERROR_INVALID_PARAM;
    s.min_abs = (double)s.sequence_utility * p->min_utility_ratio;

    /* single candidates: mo list and utility list per event, in time order */
    s.s_mo = calloc((size_t)s.max_ev + 1, sizeof(IList));
    s.s_ut = calloc((size_t)s.max_ev + 1, sizeof(IList));
    s.s_present = calloc((size_t)s.max_ev + 1, 1);
    for (int32_t t = 1; t <= s.largest; t++)
        for (size_t k = 0; k < s.tp[t].n; k++) {
            int32_t ev = s.tp[t].p[k].ev;
            il_push(&s.s_mo[ev], t); il_push(&s.s_ut[ev], s.tp[t].p[k].u); s.s_present[ev] = 1;
        }
    prune_single_events(&s);
    if (s.use_cooc) build_cooc(&s);

    const char *outp = getenv("DM_HUE_OUT");
    if (outp) {
        s.out = fopen(outp, "w");
        if (s.out) fputs("@FILETYPE=\"High-Utility episodes\"\n@SOURCE=\"SPMF SOFTWARE https://philippe-fournier-viger.com/spmf/\"\n", s.out);
    }
    printf("[HUE-Span] time points=%d sequence utility=%lld minUtilAbs=%.6f maxDuration=%d checkMax=%d ERU=%d EEUCP=%d prefix=%d fixed=%d\n",
           s.largest, (long long)s.sequence_utility, s.min_abs, s.max_dur, s.check_max, s.use_eru, s.use_cooc, s.pruning_prefix, s.fixed);

    for (int32_t ev = 0; ev <= s.max_ev; ev++) {
        if (!s.s_present[ev]) continue;
        s.combinated++;
        BetaInfo a;
        a.st = s.s_mo[ev]; a.en = s.s_mo[ev]; a.ut = s.s_ut[ev];
        a.prev.n = a.prev.cap = s.s_mo[ev].n; a.prev.a = calloc(a.prev.n ? a.prev.n : 1, sizeof(int32_t));
        int32_t tot, ub;
        utility_and_ub(&s, ev, &a, &tot, &ub);
        if ((double)ub >= s.min_abs) {
            Episode e; e.n = 1; e.s = xmalloc(2 * sizeof(ISet));
            e.s[0].n = 1; e.s[0].ev = xmalloc(sizeof(int32_t)); e.s[0].ev[0] = ev;
            if (s.out_single && (double)tot >= s.min_abs) save(&s, &e, tot);
            mine_hue(&s, &e, &a);
            ep_free(&e);
        } else s.ub_pruning++;
        free(a.prev.a);
    }
    if (s.out) fclose(s.out);
    printf("[HUE-Span] Episodes counts : %ld\n", s.hue_count);
    printf("[HUE-Span] Candidate counts : %ld\n", s.candidate_count);
    printf("[HUE-Span] Combinated episode counts : %ld\n", s.combinated);
    printf("[HUE-Span] matrix pruning counts : %ld\n", s.matrix_pruning);
    printf("[HUE-Span] upper Bound pruning counts: %ld\n", s.ub_pruning);
    printf("[HUE-Span] non maximal combinated episode counts atleast: %ld (visited_nodes=%ld)\n", s.non_max, s.candidate_count);
    dm_bench_record_results(s.hue_count, s.footprint);

    for (int32_t ev = 0; ev <= s.max_ev; ev++) { il_free(&s.s_mo[ev]); il_free(&s.s_ut[ev]); }
    free(s.s_mo); free(s.s_ut); free(s.s_present);
    for (int32_t t = 0; t <= s.largest; t++) free(s.tp[t].p);
    free(s.tp);
    if (s.use_cooc) { im_free(&s.cooc_equals); im_free(&s.cooc_after); free(s.row_equals); free(s.row_after); }
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "hue_span",
    .name = "HUE-Span",
    .description = "High utility episode mining (Fournier-Viger et al., ADMA 2019), port of the SPMF reference implementation.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL) | (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
