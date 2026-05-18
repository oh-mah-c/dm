#include "algorithms/regular_mine.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WORD_BITS ((int)(sizeof(unsigned long) * 8))

typedef struct {
    uint32_t item;
    unsigned long *tidset;
    int support;
} RM_VItem;

typedef struct {
    RM_VItem *items;
    int count;
    int words;
} RM_VDB;

typedef struct {
    uint32_t *items;
    int len;
    unsigned long *tidset;
    int support;
} RM_FI;

typedef struct {
    RM_FI *buf;
    int count;
    int cap;
} RM_FIList;

typedef struct {
    uint32_t *items;
    int len;
} RM_Set;

typedef struct {
    RM_Set req;
    RM_Set opt;
    RM_Set *plus;
    int plus_count;
} RM_Reg;

typedef struct {
    RM_Reg *buf;
    int count;
    int cap;
} RM_RegList;

typedef struct {
    RM_Set req;
    RM_Set opt;
    RM_Set *minus;
    int minus_count;
} RM_NC;

typedef struct {
    RM_NC *buf;
    int count;
    int cap;
} RM_NCList;

typedef struct {
    RM_FIList *fis;
    const RM_VDB *vdb;
    int minsup;
    int max_items;
    unsigned long candidates;
} RM_Ctx;

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static int cmp_vitem(const void *a, const void *b) {
    const RM_VItem *x = a;
    const RM_VItem *y = b;
    if (x->support != y->support) return x->support - y->support;
    return (x->item > y->item) - (x->item < y->item);
}

static int set_cmp_lex(const RM_Set *a, const RM_Set *b) {
    if (a->len != b->len) return a->len - b->len;
    for (int i = 0; i < a->len; i++) {
        if (a->items[i] != b->items[i]) return (a->items[i] > b->items[i]) - (a->items[i] < b->items[i]);
    }
    return 0;
}

static int fi_cmp_order(const void *a, const void *b) {
    const RM_FI *x = *(RM_FI * const *)a;
    const RM_FI *y = *(RM_FI * const *)b;
    RM_Set xs = { .items = x->items, .len = x->len };
    RM_Set ys = { .items = y->items, .len = y->len };
    return set_cmp_lex(&xs, &ys);
}

static int words_for(int nbits) {
    return (nbits + WORD_BITS - 1) / WORD_BITS;
}

static void bit_set(unsigned long *bits, int pos) {
    bits[pos / WORD_BITS] |= 1UL << (pos % WORD_BITS);
}

static int bit_count(const unsigned long *bits, int words) {
    int total = 0;
    for (int i = 0; i < words; i++) total += __builtin_popcountl(bits[i]);
    return total;
}

static void bit_and(unsigned long *out, const unsigned long *a, const unsigned long *b, int words) {
    for (int i = 0; i < words; i++) out[i] = a[i] & b[i];
}

static int bit_equal(const unsigned long *a, const unsigned long *b, int words) {
    for (int i = 0; i < words; i++) if (a[i] != b[i]) return 0;
    return 1;
}

static unsigned long *bit_clone(const unsigned long *src, int words) {
    unsigned long *out = malloc((size_t)words * sizeof(unsigned long));
    if (out) memcpy(out, src, (size_t)words * sizeof(unsigned long));
    return out;
}

static RM_Set set_copy(const RM_Set *s) {
    RM_Set out = {0};
    out.len = s->len;
    if (out.len > 0) {
        out.items = malloc((size_t)out.len * sizeof(uint32_t));
        memcpy(out.items, s->items, (size_t)out.len * sizeof(uint32_t));
    }
    return out;
}

static RM_Set set_from_items(const uint32_t *items, int len) {
    RM_Set out = {0};
    out.len = len;
    if (len > 0) {
        out.items = malloc((size_t)len * sizeof(uint32_t));
        memcpy(out.items, items, (size_t)len * sizeof(uint32_t));
        qsort(out.items, (size_t)out.len, sizeof(uint32_t), cmp_u32);
        int w = 0;
        for (int i = 0; i < out.len; i++) {
            if (w == 0 || out.items[i] != out.items[w - 1]) out.items[w++] = out.items[i];
        }
        out.len = w;
    }
    return out;
}

static void set_free(RM_Set *s) {
    free(s->items);
    s->items = NULL;
    s->len = 0;
}

static int set_contains(const RM_Set *s, uint32_t item) {
    int lo = 0, hi = s->len - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (s->items[mid] == item) return 1;
        if (s->items[mid] < item) lo = mid + 1;
        else hi = mid - 1;
    }
    return 0;
}

static int set_subset(const RM_Set *a, const RM_Set *b) {
    int i = 0, j = 0;
    while (i < a->len && j < b->len) {
        if (a->items[i] == b->items[j]) { i++; j++; }
        else if (a->items[i] > b->items[j]) j++;
        else return 0;
    }
    return i == a->len;
}

static int set_equal(const RM_Set *a, const RM_Set *b) {
    return a->len == b->len && set_subset(a, b);
}

static RM_Set set_union2(const RM_Set *a, const RM_Set *b) {
    uint32_t *tmp = malloc((size_t)(a->len + b->len) * sizeof(uint32_t));
    int n = 0;
    for (int i = 0; i < a->len; i++) tmp[n++] = a->items[i];
    for (int i = 0; i < b->len; i++) tmp[n++] = b->items[i];
    RM_Set out = set_from_items(tmp, n);
    free(tmp);
    return out;
}

static RM_Set set_diff(const RM_Set *a, const RM_Set *b) {
    uint32_t *tmp = malloc((size_t)a->len * sizeof(uint32_t));
    int n = 0;
    for (int i = 0; i < a->len; i++) if (!set_contains(b, a->items[i])) tmp[n++] = a->items[i];
    RM_Set out = set_from_items(tmp, n);
    free(tmp);
    return out;
}

static void set_add_item(RM_Set *s, uint32_t item) {
    if (set_contains(s, item)) return;
    s->items = realloc(s->items, (size_t)(s->len + 1) * sizeof(uint32_t));
    s->items[s->len++] = item;
    qsort(s->items, (size_t)s->len, sizeof(uint32_t), cmp_u32);
}

static void set_remove_item(RM_Set *s, uint32_t item) {
    int w = 0;
    for (int i = 0; i < s->len; i++) if (s->items[i] != item) s->items[w++] = s->items[i];
    s->len = w;
}

static RM_Reg reg_copy(const RM_Reg *r) {
    RM_Reg out = {0};
    out.req = set_copy(&r->req);
    out.opt = set_copy(&r->opt);
    out.plus_count = r->plus_count;
    if (out.plus_count > 0) {
        out.plus = calloc((size_t)out.plus_count, sizeof(RM_Set));
        for (int i = 0; i < out.plus_count; i++) out.plus[i] = set_copy(&r->plus[i]);
    }
    return out;
}

static void reg_free(RM_Reg *r) {
    set_free(&r->req);
    set_free(&r->opt);
    for (int i = 0; i < r->plus_count; i++) set_free(&r->plus[i]);
    free(r->plus);
    memset(r, 0, sizeof(*r));
}

static int reg_equal(const RM_Reg *a, const RM_Reg *b) {
    if (!set_equal(&a->req, &b->req) || !set_equal(&a->opt, &b->opt) || a->plus_count != b->plus_count) return 0;
    for (int i = 0; i < a->plus_count; i++) if (!set_equal(&a->plus[i], &b->plus[i])) return 0;
    return 1;
}

static void reg_normalize(RM_Reg *r) {
    for (int i = 0; i < r->req.len; i++) set_remove_item(&r->opt, r->req.items[i]);
    for (int i = 0; i < r->plus_count; i++) {
        for (int j = 0; j < r->req.len; j++) set_remove_item(&r->plus[i], r->req.items[j]);
    }
}

static void reglist_add(RM_RegList *list, const RM_Reg *r) {
    for (int i = 0; i < list->count; i++) if (reg_equal(&list->buf[i], r)) return;
    if (list->count >= list->cap) {
        list->cap = list->cap ? list->cap * 2 : 32;
        list->buf = realloc(list->buf, (size_t)list->cap * sizeof(RM_Reg));
    }
    list->buf[list->count++] = reg_copy(r);
}

static void reglist_free(RM_RegList *list) {
    for (int i = 0; i < list->count; i++) reg_free(&list->buf[i]);
    free(list->buf);
    memset(list, 0, sizeof(*list));
}

static void nc_free(RM_NC *r) {
    set_free(&r->req);
    set_free(&r->opt);
    for (int i = 0; i < r->minus_count; i++) set_free(&r->minus[i]);
    free(r->minus);
    memset(r, 0, sizeof(*r));
}

static void nclist_add_take(RM_NCList *list, RM_NC *r) {
    if (list->count >= list->cap) {
        list->cap = list->cap ? list->cap * 2 : 16;
        list->buf = realloc(list->buf, (size_t)list->cap * sizeof(RM_NC));
    }
    list->buf[list->count++] = *r;
    memset(r, 0, sizeof(*r));
}

static void nclist_free(RM_NCList *list) {
    for (int i = 0; i < list->count; i++) nc_free(&list->buf[i]);
    free(list->buf);
    memset(list, 0, sizeof(*list));
}

static void filist_add(RM_FIList *list, const uint32_t *items, int len, const unsigned long *tidset, int words, int support) {
    if (list->count >= list->cap) {
        list->cap = list->cap ? list->cap * 2 : 1024;
        list->buf = realloc(list->buf, (size_t)list->cap * sizeof(RM_FI));
    }
    RM_FI *fi = &list->buf[list->count++];
    fi->items = malloc((size_t)len * sizeof(uint32_t));
    memcpy(fi->items, items, (size_t)len * sizeof(uint32_t));
    qsort(fi->items, (size_t)len, sizeof(uint32_t), cmp_u32);
    fi->len = len;
    fi->tidset = bit_clone(tidset, words);
    fi->support = support;
}

static void filist_free(RM_FIList *list) {
    for (int i = 0; i < list->count; i++) {
        free(list->buf[i].items);
        free(list->buf[i].tidset);
    }
    free(list->buf);
    memset(list, 0, sizeof(*list));
}

static void dfs_fi(RM_Ctx *ctx, uint32_t *prefix, int prefix_len, const unsigned long *prefix_tidset, int start) {
    for (int i = start; i < ctx->vdb->count; i++) {
        unsigned long *tidset = malloc((size_t)ctx->vdb->words * sizeof(unsigned long));
        if (prefix_tidset) bit_and(tidset, prefix_tidset, ctx->vdb->items[i].tidset, ctx->vdb->words);
        else memcpy(tidset, ctx->vdb->items[i].tidset, (size_t)ctx->vdb->words * sizeof(unsigned long));
        int support = bit_count(tidset, ctx->vdb->words);
        ctx->candidates++;
        if (support >= ctx->minsup) {
            prefix[prefix_len] = ctx->vdb->items[i].item;
            filist_add(ctx->fis, prefix, prefix_len + 1, tidset, ctx->vdb->words, support);
            dfs_fi(ctx, prefix, prefix_len + 1, tidset, i + 1);
        }
        free(tidset);
    }
}

static RM_VDB build_vdb(DM_Dataset *ds, int minsup) {
    DM_Trans_Simple *tr = ds->payload;
    RM_VDB vdb = {0};
    vdb.words = words_for((int)ds->count);
    int *counts = calloc((size_t)ds->max_id + 1, sizeof(int));
    for (size_t t = 0; t < ds->count; t++) {
        qsort(tr[t].items, tr[t].count, sizeof(uint32_t), cmp_u32);
        uint32_t prev = UINT32_MAX;
        for (size_t i = 0; i < tr[t].count; i++) {
            if (tr[t].items[i] != prev) counts[tr[t].items[i]]++;
            prev = tr[t].items[i];
        }
    }
    for (uint32_t item = 0; item <= ds->max_id; item++) if (counts[item] >= minsup) vdb.count++;
    vdb.items = calloc((size_t)vdb.count, sizeof(RM_VItem));
    int *pos = malloc(((size_t)ds->max_id + 1) * sizeof(int));
    for (uint32_t item = 0; item <= ds->max_id; item++) pos[item] = -1;
    int p = 0;
    for (uint32_t item = 0; item <= ds->max_id; item++) {
        if (counts[item] >= minsup) {
            vdb.items[p].item = item;
            vdb.items[p].support = counts[item];
            vdb.items[p].tidset = calloc((size_t)vdb.words, sizeof(unsigned long));
            pos[item] = p++;
        }
    }
    for (size_t t = 0; t < ds->count; t++) {
        uint32_t prev = UINT32_MAX;
        for (size_t i = 0; i < tr[t].count; i++) {
            if (tr[t].items[i] == prev) continue;
            prev = tr[t].items[i];
            if (pos[tr[t].items[i]] >= 0) bit_set(vdb.items[pos[tr[t].items[i]]].tidset, (int)t);
        }
    }
    qsort(vdb.items, (size_t)vdb.count, sizeof(RM_VItem), cmp_vitem);
    free(pos);
    free(counts);
    return vdb;
}

static void vdb_free(RM_VDB *vdb) {
    for (int i = 0; i < vdb->count; i++) free(vdb->items[i].tidset);
    free(vdb->items);
}

static RM_Set class_closed(RM_FI **arr, int n) {
    RM_Set out = {0};
    for (int i = 0; i < n; i++) {
        RM_Set s = { .items = arr[i]->items, .len = arr[i]->len };
        RM_Set u = set_union2(&out, &s);
        set_free(&out);
        out = u;
    }
    return out;
}

static RM_FI **class_free_sets(RM_FI **arr, int n, int *out_n) {
    if (n <= 0) {
        *out_n = 0;
        return NULL;
    }
    RM_FI **free_sets = calloc((size_t)n, sizeof(RM_FI *));
    int count = 0;
    for (int i = 0; i < n; i++) {
        RM_Set si = { .items = arr[i]->items, .len = arr[i]->len };
        int minimal = 1;
        for (int j = 0; j < n; j++) {
            if (i == j || arr[j]->len >= arr[i]->len) continue;
            RM_Set sj = { .items = arr[j]->items, .len = arr[j]->len };
            if (set_subset(&sj, &si)) {
                minimal = 0;
                break;
            }
        }
        if (minimal) free_sets[count++] = arr[i];
    }
    qsort(free_sets, (size_t)count, sizeof(RM_FI *), fi_cmp_order);
    *out_n = count;
    return free_sets;
}

static RM_Reg nc_to_reg(const RM_NC *nc) {
    RM_Reg r = {0};
    r.req = set_copy(&nc->req);
    r.opt = set_copy(&nc->opt);
    reg_normalize(&r);
    return r;
}

static RM_RegList covering(const RM_Set *x, RM_Set *prev, int prev_n, const RM_Set *closed) {
    RM_RegList out = {0};
    RM_NC init = {0};
    init.req = set_copy(x);
    init.opt = set_diff(closed, x);
    init.minus_count = prev_n;
    init.minus = calloc((size_t)prev_n, sizeof(RM_Set));
    int ok = 1;
    for (int i = 0; i < prev_n; i++) {
        init.minus[i] = set_diff(&prev[i], x);
        if (init.minus[i].len == 0) ok = 0;
    }
    if (!ok) {
        nc_free(&init);
        return out;
    }

    RM_NCList work = {0};
    nclist_add_take(&work, &init);
    while (work.count > 0) {
        RM_NC cur = work.buf[--work.count];
        while (cur.minus_count > 0) {
            int mi = 0;
            for (int i = 1; i < cur.minus_count; i++) if (cur.minus[i].len < cur.minus[mi].len) mi = i;
            uint32_t a = cur.minus[mi].items[0];

            RM_NC no_a = {0};
            no_a.req = set_copy(&cur.req);
            no_a.opt = set_copy(&cur.opt);
            set_remove_item(&no_a.opt, a);
            no_a.minus = calloc((size_t)cur.minus_count, sizeof(RM_Set));
            for (int i = 0; i < cur.minus_count; i++) {
                if (set_contains(&cur.minus[i], a)) {
                    RM_Set rem = set_copy(&cur.minus[i]);
                    set_remove_item(&rem, a);
                    RM_Set merged = set_union2(&no_a.opt, &rem);
                    set_free(&no_a.opt);
                    no_a.opt = merged;
                    set_free(&rem);
                } else {
                    no_a.minus[no_a.minus_count++] = set_copy(&cur.minus[i]);
                }
            }

            if (cur.minus[mi].len > 1) {
                RM_NC yes_a = {0};
                yes_a.req = set_copy(&cur.req);
                set_add_item(&yes_a.req, a);
                yes_a.opt = set_copy(&cur.opt);
                set_remove_item(&yes_a.opt, a);
                yes_a.minus = calloc((size_t)cur.minus_count, sizeof(RM_Set));
                int valid = 1;
                for (int i = 0; i < cur.minus_count; i++) {
                    RM_Set m = set_copy(&cur.minus[i]);
                    if (set_contains(&m, a)) set_remove_item(&m, a);
                    if (m.len == 0) {
                        set_free(&m);
                        valid = 0;
                        break;
                    }
                    yes_a.minus[yes_a.minus_count++] = m;
                }
                if (valid) nclist_add_take(&work, &yes_a);
                else nc_free(&yes_a);
            }
            nc_free(&cur);
            cur = no_a;
        }
        RM_Reg r = nc_to_reg(&cur);
        reglist_add(&out, &r);
        reg_free(&r);
        nc_free(&cur);
    }
    nclist_free(&work);
    return out;
}

static int same_except_req(const RM_Reg *small, const RM_Reg *big, uint32_t *diff) {
    if (!set_equal(&small->opt, &big->opt) || small->plus_count != big->plus_count) return 0;
    for (int i = 0; i < small->plus_count; i++) if (!set_equal(&small->plus[i], &big->plus[i])) return 0;
    if (big->req.len != small->req.len + 1 || !set_subset(&small->req, &big->req)) return 0;
    for (int i = 0; i < big->req.len; i++) if (!set_contains(&small->req, big->req.items[i])) { *diff = big->req.items[i]; return 1; }
    return 0;
}

static void reg_add_plus(RM_Reg *r, RM_Set *g) {
    if (g->len == 1) {
        set_add_item(&r->req, g->items[0]);
        return;
    }
    r->plus = realloc(r->plus, (size_t)(r->plus_count + 1) * sizeof(RM_Set));
    r->plus[r->plus_count++] = set_copy(g);
}

static int try_merge_pair(const RM_Reg *a, const RM_Reg *b, RM_Reg *out) {
    uint32_t diff = 0;
    if (same_except_req(a, b, &diff)) {
        *out = reg_copy(a);
        set_add_item(&out->opt, diff);
        reg_normalize(out);
        return 1; /* M1 generalized */
    }
    if (same_except_req(b, a, &diff)) {
        *out = reg_copy(b);
        set_add_item(&out->opt, diff);
        reg_normalize(out);
        return 1;
    }

    if (set_equal(&a->req, &b->req) && set_equal(&a->opt, &b->opt)) {
        if (a->plus_count + 1 == b->plus_count) {
            *out = reg_copy(a);
            RM_Set y = set_copy(&b->plus[b->plus_count - 1]);
            RM_Set merged = set_union2(&out->opt, &y);
            set_free(&out->opt);
            out->opt = merged;
            set_free(&y);
            return 1; /* M2-like */
        }
        if (b->plus_count + 1 == a->plus_count) {
            *out = reg_copy(b);
            RM_Set y = set_copy(&a->plus[a->plus_count - 1]);
            RM_Set merged = set_union2(&out->opt, &y);
            set_free(&out->opt);
            out->opt = merged;
            set_free(&y);
            return 1;
        }
    }

    if (a->plus_count == b->plus_count && set_equal(&a->opt, &b->opt)) {
        RM_Set common = {0};
        for (int i = 0; i < a->req.len; i++) if (set_contains(&b->req, a->req.items[i])) set_add_item(&common, a->req.items[i]);
        if (common.len + 1 == a->req.len && common.len + 1 == b->req.len) {
            uint32_t da = 0, db = 0;
            for (int i = 0; i < a->req.len; i++) if (!set_contains(&common, a->req.items[i])) da = a->req.items[i];
            for (int i = 0; i < b->req.len; i++) if (!set_contains(&common, b->req.items[i])) db = b->req.items[i];
            *out = reg_copy(a);
            set_free(&out->req);
            out->req = common;
            set_remove_item(&out->opt, da);
            set_remove_item(&out->opt, db);
            RM_Set g = {0};
            set_add_item(&g, da);
            set_add_item(&g, db);
            reg_add_plus(out, &g);
            set_free(&g);
            reg_normalize(out);
            return 1; /* M3/M4 family */
        }
        set_free(&common);
    }
    return 0;
}

static RM_RegList merging(RM_RegList *in) {
    RM_RegList cur = {0};
    for (int i = 0; i < in->count; i++) reglist_add(&cur, &in->buf[i]);
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < cur.count && !changed; i++) {
            for (int j = i + 1; j < cur.count && !changed; j++) {
                RM_Reg m = {0};
                if (try_merge_pair(&cur.buf[i], &cur.buf[j], &m)) {
                    RM_RegList next = {0};
                    for (int k = 0; k < cur.count; k++) if (k != i && k != j) reglist_add(&next, &cur.buf[k]);
                    reglist_add(&next, &m);
                    reg_free(&m);
                    reglist_free(&cur);
                    cur = next;
                    changed = 1;
                }
            }
        }
    }
    return cur;
}

static void print_set(const RM_Set *s) {
    for (int i = 0; i < s->len; i++) printf("%s%u", i ? " " : "", s->items[i]);
}

static void print_reg(const RM_Reg *r) {
    print_set(&r->req);
    if (r->opt.len > 0) {
        printf("%s{", r->req.len ? " " : "");
        print_set(&r->opt);
        printf("}?");
    }
    for (int i = 0; i < r->plus_count; i++) {
        printf(" {");
        print_set(&r->plus[i]);
        printf("}+");
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_TRANSACTIONAL || !ds->payload) return DM_ERROR_INCOMPATIBLE;
    DM_REGULAR_MINE_Params *p = params;
    double min_support = p ? p->min_support : 0.1;
    int minsup = min_support < 1.0 ? (int)ceil(min_support * (double)ds->count) : (int)ceil(min_support);
    if (minsup < 1) minsup = 1;

    RM_VDB vdb = build_vdb(ds, minsup);
    RM_FIList fis = {0};
    RM_Ctx ctx = { .fis = &fis, .vdb = &vdb, .minsup = minsup, .max_items = vdb.count };
    uint32_t *prefix = malloc((size_t)(vdb.count ? vdb.count : 1) * sizeof(uint32_t));
    dfs_fi(&ctx, prefix, 0, NULL, 0);
    free(prefix);

    int *used = calloc((size_t)fis.count, sizeof(int));
    RM_RegList output = {0};
    int closed_count = 0, free_count = 0;

    for (int i = 0; i < fis.count; i++) {
        if (used[i]) continue;
        RM_FI **cls = calloc((size_t)fis.count, sizeof(RM_FI *));
        int cls_n = 0;
        for (int j = i; j < fis.count; j++) {
            if (!used[j] && bit_equal(fis.buf[i].tidset, fis.buf[j].tidset, vdb.words)) {
                used[j] = 1;
                cls[cls_n++] = &fis.buf[j];
            }
        }
        closed_count++;
        RM_Set closed = class_closed(cls, cls_n);
        int fs_n = 0;
        RM_FI **fs = class_free_sets(cls, cls_n, &fs_n);
        free_count += fs_n;
        RM_Set *prev = calloc((size_t)fs_n, sizeof(RM_Set));
        RM_RegList class_regs = {0};
        for (int f = 0; f < fs_n; f++) {
            RM_Set x = { .items = fs[f]->items, .len = fs[f]->len };
            RM_RegList cov = covering(&x, prev, f, &closed);
            for (int c = 0; c < cov.count; c++) reglist_add(&class_regs, &cov.buf[c]);
            prev[f] = set_copy(&x);
            reglist_free(&cov);
        }
        RM_RegList merged = merging(&class_regs);
        for (int r = 0; r < merged.count; r++) reglist_add(&output, &merged.buf[r]);
        reglist_free(&merged);
        reglist_free(&class_regs);
        for (int f = 0; f < fs_n; f++) set_free(&prev[f]);
        free(prev);
        free(fs);
        set_free(&closed);
        free(cls);
    }

    printf("[RegularMine] minsup=%d transactions=%zu\n", minsup, ds->count);
    printf("[RegularMine] frequent_itemsets=%d closed_classes=%d free_sets=%d regular_itemsets=%d candidates=%lu\n",
           fis.count, closed_count, free_count, output.count, ctx.candidates);
    for (int i = 0; i < output.count && i < 20; i++) {
        printf("  ");
        print_reg(&output.buf[i]);
        printf("\n");
    }

    size_t total_items = 0;
    for (int i = 0; i < output.count; i++) total_items += (size_t)output.buf[i].req.len + (size_t)output.buf[i].opt.len;
    dm_bench_record_results((size_t)output.count, total_items);

    reglist_free(&output);
    free(used);
    filist_free(&fis);
    vdb_free(&vdb);
    return DM_SUCCESS;
}

DM_Algorithm regular_mine_algo = {
    .id = "regular_mine",
    .name = "RegularMine",
    .description = "Frequent regular itemset concise representation mining (Ruggieri KDD 2010).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(regular_mine_algo)
