#include "algorithms/huci_miner.h"
#include "core/dm_benchmark.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t tid;
    double iutil;
    double rutil;
} HUCIElement;

typedef struct {
    uint32_t *items;
    size_t len;
    HUCIElement *elements;
    size_t count;
    size_t cap;
    double sum_iutil;
    double sum_rutil;
} HUCIUtilityList;

typedef struct HUCIGenRef {
    size_t index;
    struct HUCIGenRef *next;
} HUCIGenRef;

typedef struct {
    uint32_t *items;
    size_t len;
    size_t support;
    double utility;
    int closed;
    int key;
    HUCIGenRef *generators;
} HUCIRecord;

typedef struct {
    HUCIRecord *data;
    size_t count;
    size_t cap;
} HUCILevel;

typedef struct {
    HUCILevel *levels;
    size_t count;
    size_t cap;
} HUCIStore;

typedef struct {
    size_t *data;
    size_t count;
    size_t cap;
} HUCIIndexList;

typedef struct {
    DM_Trans_Utility *transactions;
    size_t transaction_count;
    uint32_t max_id;
    double min_utility;
    double min_confidence;
    double *twu;
    double **eucs;
    uint32_t *promising_items;
    uint32_t *rank;
    size_t promising_count;
    HUCIStore store;
    HUCIIndexList hg;
    DM_HUCI_Miner_Stats stats;
} HUCIContext;

static int cmp_items_by_twu(void *arg, const void *a, const void *b) {
    const uint32_t ia = *(const uint32_t *)a;
    const uint32_t ib = *(const uint32_t *)b;
    const double *twu = (const double *)arg;
    if (twu[ia] < twu[ib]) return -1;
    if (twu[ia] > twu[ib]) return 1;
    return ia < ib ? -1 : (ia > ib);
}

static void free_generators(HUCIGenRef *g) {
    while (g) {
        HUCIGenRef *next = g->next;
        free(g);
        g = next;
    }
}

static void store_free(HUCIStore *s) {
    if (!s) return;
    for (size_t l = 0; l < s->count; l++) {
        for (size_t i = 0; i < s->levels[l].count; i++) {
            free(s->levels[l].data[i].items);
            free_generators(s->levels[l].data[i].generators);
        }
        free(s->levels[l].data);
    }
    free(s->levels);
    memset(s, 0, sizeof(*s));
}

static int store_ensure_level(HUCIStore *s, size_t len) {
    if (len == 0) return -1;
    while (s->count < len) {
        if (s->count >= s->cap) {
            size_t next = s->cap ? s->cap * 2 : 8;
            HUCILevel *tmp = (HUCILevel *)realloc(s->levels, next * sizeof(*tmp));
            if (!tmp) return -1;
            memset(tmp + s->cap, 0, (next - s->cap) * sizeof(*tmp));
            s->levels = tmp;
            s->cap = next;
        }
        s->count++;
    }
    return 0;
}

static int store_add(HUCIStore *s, const uint32_t *items, size_t len, size_t support, double utility) {
    if (store_ensure_level(s, len) != 0) return -1;
    HUCILevel *level = &s->levels[len - 1];
    if (level->count >= level->cap) {
        size_t next = level->cap ? level->cap * 2 : 64;
        HUCIRecord *tmp = (HUCIRecord *)realloc(level->data, next * sizeof(*tmp));
        if (!tmp) return -1;
        level->data = tmp;
        level->cap = next;
    }
    HUCIRecord *r = &level->data[level->count++];
    memset(r, 0, sizeof(*r));
    r->items = (uint32_t *)malloc(len * sizeof(uint32_t));
    if (!r->items) return -1;
    memcpy(r->items, items, len * sizeof(uint32_t));
    r->len = len;
    r->support = support;
    r->utility = utility;
    r->closed = 1;
    r->key = 1;
    return 0;
}

static int index_list_add(HUCIIndexList *l, size_t index) {
    if (l->count >= l->cap) {
        size_t next = l->cap ? l->cap * 2 : 64;
        size_t *tmp = (size_t *)realloc(l->data, next * sizeof(*tmp));
        if (!tmp) return -1;
        l->data = tmp;
        l->cap = next;
    }
    l->data[l->count++] = index;
    return 0;
}

static HUCIRecord *record_by_global(HUCIStore *s, size_t index) {
    for (size_t l = 0; l < s->count; l++) {
        if (index < s->levels[l].count) return &s->levels[l].data[index];
        index -= s->levels[l].count;
    }
    return NULL;
}

static size_t global_index(HUCIStore *s, size_t level, size_t pos) {
    size_t idx = pos;
    for (size_t l = 0; l < level; l++) idx += s->levels[l].count;
    return idx;
}

static int ul_push(HUCIUtilityList *ul, HUCIElement e) {
    if (ul->count >= ul->cap) {
        size_t next = ul->cap ? ul->cap * 2 : 16;
        HUCIElement *tmp = (HUCIElement *)realloc(ul->elements, next * sizeof(*tmp));
        if (!tmp) return -1;
        ul->elements = tmp;
        ul->cap = next;
    }
    ul->elements[ul->count++] = e;
    ul->sum_iutil += e.iutil;
    ul->sum_rutil += e.rutil;
    return 0;
}

static void ul_free(HUCIUtilityList *ul) {
    if (!ul) return;
    free(ul->items);
    free(ul->elements);
    free(ul);
}

static HUCIUtilityList *ul_new(const uint32_t *items, size_t len) {
    HUCIUtilityList *ul = (HUCIUtilityList *)calloc(1, sizeof(*ul));
    if (!ul) return NULL;
    ul->items = (uint32_t *)malloc(len * sizeof(uint32_t));
    if (!ul->items) {
        free(ul);
        return NULL;
    }
    memcpy(ul->items, items, len * sizeof(uint32_t));
    ul->len = len;
    return ul;
}

static HUCIElement *find_element(const HUCIUtilityList *ul, uint32_t tid, size_t *cursor) {
    if (!ul) return NULL;
    while (*cursor < ul->count && ul->elements[*cursor].tid < tid) (*cursor)++;
    if (*cursor < ul->count && ul->elements[*cursor].tid == tid) return &ul->elements[*cursor];
    return NULL;
}

static HUCIUtilityList *construct_ul(const HUCIUtilityList *p, const HUCIUtilityList *px, const HUCIUtilityList *py) {
    uint32_t *items = (uint32_t *)malloc((px->len + 1) * sizeof(uint32_t));
    if (!items) return NULL;
    memcpy(items, px->items, px->len * sizeof(uint32_t));
    items[px->len] = py->items[py->len - 1];
    HUCIUtilityList *out = ul_new(items, px->len + 1);
    free(items);
    if (!out) return NULL;

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < px->count && iy < py->count) {
        HUCIElement ex = px->elements[ix];
        HUCIElement ey = py->elements[iy];
        if (ex.tid == ey.tid) {
            double putil = 0.0;
            if (p) {
                HUCIElement *pe = find_element(p, ex.tid, &ip);
                if (pe) putil = pe->iutil;
            }
            HUCIElement e;
            e.tid = ex.tid;
            e.iutil = ex.iutil + ey.iutil - putil;
            e.rutil = ey.rutil;
            if (ul_push(out, e) != 0) {
                ul_free(out);
                return NULL;
            }
            ix++;
            iy++;
        } else if (ex.tid < ey.tid) {
            ix++;
        } else {
            iy++;
        }
    }
    return out;
}

static void free_ul_array(HUCIUtilityList **arr, size_t n) {
    for (size_t i = 0; i < n; i++) ul_free(arr[i]);
    free(arr);
}

static int is_subset_items(const uint32_t *a, size_t alen, const uint32_t *b, size_t blen) {
    size_t i = 0, j = 0;
    while (i < alen && j < blen) {
        if (a[i] == b[j]) {
            i++;
            j++;
        } else if (a[i] > b[j]) {
            j++;
        } else {
            return 0;
        }
    }
    return i == alen;
}

static int record_contains_subset(const HUCIRecord *sup, const HUCIRecord *sub) {
    return is_subset_items(sub->items, sub->len, sup->items, sup->len);
}

static int huci_search(HUCIContext *ctx, HUCIUtilityList *p, HUCIUtilityList **ext, size_t ext_count) {
    for (size_t i = 0; i < ext_count; i++) {
        HUCIUtilityList *px = ext[i];
        if (px->sum_iutil >= ctx->min_utility) {
            if (store_add(&ctx->store, px->items, px->len, px->count, px->sum_iutil) != 0) return -1;
            ctx->stats.high_utility_itemsets++;
            if (px->len > ctx->stats.max_depth) ctx->stats.max_depth = px->len;
        }
        if (px->sum_iutil + px->sum_rutil < ctx->min_utility) {
            ctx->stats.pruned_subtree_utility++;
            continue;
        }
        HUCIUtilityList **next = NULL;
        size_t next_count = 0, next_cap = 0;
        for (size_t j = i + 1; j < ext_count; j++) {
            HUCIUtilityList *py = ext[j];
            uint32_t rx = ctx->rank[px->items[px->len - 1]];
            uint32_t ry = ctx->rank[py->items[py->len - 1]];
            if (ctx->eucs && ctx->eucs[rx] && ctx->eucs[rx][ry] < ctx->min_utility) {
                ctx->stats.pruned_eucs++;
                continue;
            }
            HUCIUtilityList *pxy = construct_ul(p, px, py);
            ctx->stats.joins++;
            ctx->stats.utility_lists_constructed++;
            if (!pxy) {
                free_ul_array(next, next_count);
                return -1;
            }
            if (next_count >= next_cap) {
                size_t ncap = next_cap ? next_cap * 2 : 16;
                HUCIUtilityList **tmp = (HUCIUtilityList **)realloc(next, ncap * sizeof(*tmp));
                if (!tmp) {
                    ul_free(pxy);
                    free_ul_array(next, next_count);
                    return -1;
                }
                next = tmp;
                next_cap = ncap;
            }
            next[next_count++] = pxy;
        }
        if (next_count && huci_search(ctx, px, next, next_count) != 0) {
            free_ul_array(next, next_count);
            return -1;
        }
        free_ul_array(next, next_count);
    }
    return 0;
}

static int build_initial_utility_lists(HUCIContext *ctx, HUCIUtilityList ***out, size_t *out_count) {
    HUCIUtilityList **lists = (HUCIUtilityList **)calloc(ctx->promising_count, sizeof(*lists));
    if (!lists) return -1;
    for (size_t i = 0; i < ctx->promising_count; i++) {
        lists[i] = ul_new(&ctx->promising_items[i], 1);
        if (!lists[i]) {
            free_ul_array(lists, i);
            return -1;
        }
        ctx->stats.utility_lists_constructed++;
    }

    for (size_t t = 0; t < ctx->transaction_count; t++) {
        DM_Trans_Utility *tr = &ctx->transactions[t];
        uint32_t *ids = (uint32_t *)malloc(tr->count * sizeof(uint32_t));
        double *utils = (double *)malloc(tr->count * sizeof(double));
        if (!ids || !utils) {
            free(ids);
            free(utils);
            free_ul_array(lists, ctx->promising_count);
            return -1;
        }
        size_t n = 0;
        for (size_t i = 0; i < tr->count; i++) {
            uint32_t id = tr->items[i].id;
            if (ctx->rank[id] != UINT32_MAX) {
                ids[n] = id;
                utils[n] = tr->items[i].utility;
                n++;
            }
        }
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < n; j++) {
                if (ctx->rank[ids[i]] > ctx->rank[ids[j]]) {
                    uint32_t ti = ids[i]; ids[i] = ids[j]; ids[j] = ti;
                    double tu = utils[i]; utils[i] = utils[j]; utils[j] = tu;
                }
            }
        }
        double remaining = 0.0;
        for (size_t r = n; r > 0; r--) {
            size_t i = r - 1;
            uint32_t rank = ctx->rank[ids[i]];
            HUCIElement e;
            e.tid = (uint32_t)t;
            e.iutil = utils[i];
            e.rutil = remaining;
            if (ul_push(lists[rank], e) != 0) {
                free(ids);
                free(utils);
                free_ul_array(lists, ctx->promising_count);
                return -1;
            }
            remaining += utils[i];
        }
        free(ids);
        free(utils);
    }
    *out = lists;
    *out_count = ctx->promising_count;
    return 0;
}

static int build_twu_eucs(HUCIContext *ctx) {
    ctx->twu = (double *)calloc(ctx->max_id + 1, sizeof(double));
    ctx->rank = (uint32_t *)malloc((ctx->max_id + 1) * sizeof(uint32_t));
    if (!ctx->twu || !ctx->rank) return -1;
    for (uint32_t i = 0; i <= ctx->max_id; i++) ctx->rank[i] = UINT32_MAX;

    for (size_t t = 0; t < ctx->transaction_count; t++) {
        DM_Trans_Utility *tr = &ctx->transactions[t];
        for (size_t i = 0; i < tr->count; i++) ctx->twu[tr->items[i].id] += tr->total_utility;
    }
    ctx->promising_items = (uint32_t *)malloc((ctx->max_id + 1) * sizeof(uint32_t));
    if (!ctx->promising_items) return -1;
    for (uint32_t id = 0; id <= ctx->max_id; id++) {
        if (ctx->twu[id] >= ctx->min_utility) ctx->promising_items[ctx->promising_count++] = id;
    }
    qsort_s(ctx->promising_items, ctx->promising_count, sizeof(uint32_t), cmp_items_by_twu, ctx->twu);
    for (size_t i = 0; i < ctx->promising_count; i++) ctx->rank[ctx->promising_items[i]] = (uint32_t)i;

    ctx->eucs = (double **)calloc(ctx->promising_count, sizeof(double *));
    if (!ctx->eucs) return -1;
    for (size_t i = 0; i < ctx->promising_count; i++) {
        ctx->eucs[i] = (double *)calloc(ctx->promising_count, sizeof(double));
        if (!ctx->eucs[i]) return -1;
    }
    for (size_t t = 0; t < ctx->transaction_count; t++) {
        DM_Trans_Utility *tr = &ctx->transactions[t];
        uint32_t *ranks = (uint32_t *)malloc(tr->count * sizeof(uint32_t));
        if (!ranks) return -1;
        size_t n = 0;
        for (size_t i = 0; i < tr->count; i++) {
            uint32_t r = ctx->rank[tr->items[i].id];
            if (r != UINT32_MAX) ranks[n++] = r;
        }
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i + 1; j < n; j++) {
                ctx->eucs[ranks[i]][ranks[j]] += tr->total_utility;
                ctx->eucs[ranks[j]][ranks[i]] += tr->total_utility;
            }
        }
        free(ranks);
    }
    return 0;
}

static HUCIRecord *find_subset_record(HUCILevel *level, const HUCIRecord *h, size_t remove_pos) {
    for (size_t i = 0; i < level->count; i++) {
        HUCIRecord *cand = &level->data[i];
        if (cand->len + 1 != h->len) continue;
        size_t ci = 0;
        int ok = 1;
        for (size_t hi = 0; hi < h->len; hi++) {
            if (hi == remove_pos) continue;
            if (ci >= cand->len || cand->items[ci++] != h->items[hi]) {
                ok = 0;
                break;
            }
        }
        if (ok) return cand;
    }
    return NULL;
}

static int add_generator_ref(HUCIRecord *ch, size_t global_idx) {
    for (HUCIGenRef *g = ch->generators; g; g = g->next) {
        if (g->index == global_idx) return 0;
    }
    HUCIGenRef *node = (HUCIGenRef *)malloc(sizeof(*node));
    if (!node) return -1;
    node->index = global_idx;
    node->next = ch->generators;
    ch->generators = node;
    return 0;
}

static int remove_hg_at(HUCIIndexList *hg, size_t pos) {
    if (pos >= hg->count) return -1;
    memmove(hg->data + pos, hg->data + pos + 1, (hg->count - pos - 1) * sizeof(*hg->data));
    hg->count--;
    return 0;
}

static int get_generators(HUCIContext *ctx, HUCILevel *closed_level, HUCILevel *current_level) {
    (void)current_level;
    for (size_t i = 0; i < closed_level->count; i++) {
        HUCIRecord *ch = &closed_level->data[i];
        if (!ch->closed) continue;
        for (size_t g = 0; g < ctx->hg.count; ) {
            HUCIRecord *gen = record_by_global(&ctx->store, ctx->hg.data[g]);
            if (gen && gen->len < ch->len && record_contains_subset(ch, gen)) {
                if (add_generator_ref(ch, ctx->hg.data[g]) != 0) return -1;
                remove_hg_at(&ctx->hg, g);
            } else {
                g++;
            }
        }
        if (!ch->generators) {
            size_t idx = 0;
            for (size_t l = 0; l < ctx->store.count; l++) {
                if (&ctx->store.levels[l] == closed_level) {
                    idx = global_index(&ctx->store, l, i);
                    break;
                }
            }
            if (add_generator_ref(ch, idx) != 0) return -1;
        }
    }
    return 0;
}

static int classify_closed_and_generators(HUCIContext *ctx) {
    if (ctx->store.count == 0) return 0;
    for (size_t i = 0; i < ctx->store.levels[0].count; i++) {
        ctx->store.levels[0].data[i].closed = 1;
        ctx->store.levels[0].data[i].key = 1;
    }

    for (size_t k = 2; k <= ctx->store.count; k++) {
        HUCILevel *prev = &ctx->store.levels[k - 2];
        HUCILevel *cur = &ctx->store.levels[k - 1];
        if (cur->count) {
            for (size_t i = 0; i < cur->count; i++) {
                HUCIRecord *h = &cur->data[i];
                h->closed = 1;
                h->key = 1;
                for (size_t r = 0; r < h->len; r++) {
                    HUCIRecord *sub = find_subset_record(prev, h, r);
                    if (sub && sub->support == h->support) {
                        h->key = 0;
                        sub->closed = 0;
                    }
                }
            }
            if (get_generators(ctx, prev, cur) != 0) return -1;
            for (size_t i = 0; i < cur->count; i++) {
                HUCIRecord *h = &cur->data[i];
                if (h->key && !h->closed) {
                    if (index_list_add(&ctx->hg, global_index(&ctx->store, k - 1, i)) != 0) return -1;
                }
            }
        } else {
            if (get_generators(ctx, prev, NULL) != 0) return -1;
        }
    }
    HUCILevel *last = &ctx->store.levels[ctx->store.count - 1];
    if (get_generators(ctx, last, NULL) != 0) return -1;

    for (size_t l = 0; l < ctx->store.count; l++) {
        for (size_t i = 0; i < ctx->store.levels[l].count; i++) {
            HUCIRecord *r = &ctx->store.levels[l].data[i];
            if (r->closed) ctx->stats.high_utility_closed_itemsets++;
            if (r->key) ctx->stats.high_utility_generators++;
        }
    }
    return 0;
}

static HUCIRecord *find_record_exact(HUCIStore *s, const uint32_t *items, size_t len) {
    if (len == 0 || len > s->count) return NULL;
    HUCILevel *level = &s->levels[len - 1];
    for (size_t i = 0; i < level->count; i++) {
        HUCIRecord *r = &level->data[i];
        if (r->len == len && memcmp(r->items, items, len * sizeof(uint32_t)) == 0) return r;
    }
    return NULL;
}

static double item_utility_in_transaction(const DM_Trans_Utility *tr, uint32_t item) {
    for (size_t i = 0; i < tr->count; i++) {
        if (tr->items[i].id == item) return tr->items[i].utility;
    }
    return 0.0;
}

static int transaction_contains(const DM_Trans_Utility *tr, const uint32_t *items, size_t len) {
    for (size_t i = 0; i < len; i++) {
        int found = 0;
        for (size_t j = 0; j < tr->count; j++) {
            if (tr->items[j].id == items[i]) {
                found = 1;
                break;
            }
        }
        if (!found) return 0;
    }
    return 1;
}

static double local_utility(HUCIContext *ctx, const uint32_t *ante, size_t alen, const uint32_t *closed, size_t clen) {
    double sum = 0.0;
    for (size_t t = 0; t < ctx->transaction_count; t++) {
        DM_Trans_Utility *tr = &ctx->transactions[t];
        if (!transaction_contains(tr, closed, clen)) continue;
        for (size_t i = 0; i < alen; i++) sum += item_utility_in_transaction(tr, ante[i]);
    }
    return sum;
}

static int has_minimal_subset(uint32_t **sets, size_t *lens, size_t count, const uint32_t *items, size_t len) {
    for (size_t i = 0; i < count; i++) {
        if (lens[i] < len && is_subset_items(sets[i], lens[i], items, len)) return 1;
    }
    return 0;
}

static int add_minimal(uint32_t ***sets, size_t **lens, size_t *count, size_t *cap, const uint32_t *items, size_t len) {
    for (size_t i = 0; i < *count; ) {
        if (len < (*lens)[i] && is_subset_items(items, len, (*sets)[i], (*lens)[i])) {
            free((*sets)[i]);
            memmove(*sets + i, *sets + i + 1, (*count - i - 1) * sizeof(**sets));
            memmove(*lens + i, *lens + i + 1, (*count - i - 1) * sizeof(**lens));
            (*count)--;
        } else {
            i++;
        }
    }
    if (*count >= *cap) {
        size_t next = *cap ? *cap * 2 : 16;
        uint32_t **nsets = (uint32_t **)realloc(*sets, next * sizeof(**sets));
        size_t *nlens = (size_t *)realloc(*lens, next * sizeof(**lens));
        if (!nsets || !nlens) return -1;
        *sets = nsets;
        *lens = nlens;
        *cap = next;
    }
    (*sets)[*count] = (uint32_t *)malloc(len * sizeof(uint32_t));
    if (!(*sets)[*count]) return -1;
    memcpy((*sets)[*count], items, len * sizeof(uint32_t));
    (*lens)[*count] = len;
    (*count)++;
    return 0;
}

static int enumerate_antecedents(HUCIContext *ctx, HUCIRecord *h, size_t pos,
                                 uint32_t *buf, size_t blen,
                                 uint32_t ***mins, size_t **min_lens,
                                 size_t *min_count, size_t *min_cap) {
    if (pos == h->len) {
        if (blen == 0 || blen == h->len) return 0;
        if (has_minimal_subset(*mins, *min_lens, *min_count, buf, blen)) return 0;
        HUCIRecord *ante = find_record_exact(&ctx->store, buf, blen);
        if (!ante) return 0;
        double luv = local_utility(ctx, buf, blen, h->items, h->len);
        if (ante->utility > 0.0 && luv / ante->utility >= ctx->min_confidence) {
            if (add_minimal(mins, min_lens, min_count, min_cap, buf, blen) != 0) return -1;
        }
        return 0;
    }
    if (enumerate_antecedents(ctx, h, pos + 1, buf, blen, mins, min_lens, min_count, min_cap) != 0) return -1;
    buf[blen] = h->items[pos];
    return enumerate_antecedents(ctx, h, pos + 1, buf, blen + 1, mins, min_lens, min_count, min_cap);
}

static int build_hgb(HUCIContext *ctx) {
    for (size_t l = 0; l < ctx->store.count; l++) {
        HUCILevel *level = &ctx->store.levels[l];
        for (size_t i = 0; i < level->count; i++) {
            HUCIRecord *h = &level->data[i];
            if (!h->closed || h->len < 2) continue;
            uint32_t **mins = NULL;
            size_t *lens = NULL;
            size_t min_count = 0, min_cap = 0;
            uint32_t *buf = (uint32_t *)malloc(h->len * sizeof(uint32_t));
            if (!buf) return -1;
            int rc = enumerate_antecedents(ctx, h, 0, buf, 0, &mins, &lens, &min_count, &min_cap);
            free(buf);
            if (rc != 0) return -1;
            ctx->stats.hgb_rules += min_count;
            for (size_t m = 0; m < min_count; m++) free(mins[m]);
            free(mins);
            free(lens);
        }
    }
    return 0;
}

static void context_free(HUCIContext *ctx) {
    if (!ctx) return;
    if (ctx->eucs) {
        for (size_t i = 0; i < ctx->promising_count; i++) free(ctx->eucs[i]);
        free(ctx->eucs);
    }
    free(ctx->twu);
    free(ctx->promising_items);
    free(ctx->rank);
    free(ctx->hg.data);
    store_free(&ctx->store);
}

int huci_mine_dataset(DM_Dataset *ds, const DM_HUCI_Miner_Params *params, DM_HUCI_Miner_Stats *stats) {
    if (!ds || ds->type != DM_TYPE_UTILITY || !stats) return -1;
    memset(stats, 0, sizeof(*stats));
    HUCIContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.transactions = (DM_Trans_Utility *)ds->payload;
    ctx.transaction_count = ds->count;
    ctx.max_id = ds->max_id;
    ctx.min_utility = params ? params->min_utility : 1000.0;
    ctx.min_confidence = params ? params->min_confidence : 0.8;
    if (ctx.min_confidence > 1.0) ctx.min_confidence /= 100.0;
    if (ctx.min_utility <= 0.0 || ctx.min_confidence <= 0.0) return -1;

    int rc = 0;
    HUCIUtilityList **initial = NULL;
    size_t initial_count = 0;
    if (build_twu_eucs(&ctx) != 0 ||
        build_initial_utility_lists(&ctx, &initial, &initial_count) != 0 ||
        huci_search(&ctx, NULL, initial, initial_count) != 0 ||
        classify_closed_and_generators(&ctx) != 0 ||
        build_hgb(&ctx) != 0) {
        rc = -1;
    }
    free_ul_array(initial, initial_count);
    *stats = ctx.stats;
    context_free(&ctx);
    return rc;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_HUCI_Miner_Params defaults;
    defaults.min_utility = 1000.0;
    defaults.min_confidence = 0.8;
    const DM_HUCI_Miner_Params *p = params ? (const DM_HUCI_Miner_Params *)params : &defaults;
    DM_HUCI_Miner_Stats stats;
    if (huci_mine_dataset(ds, p, &stats) != 0) return DM_ERROR_GENERIC;
    printf("[HUCI-Miner] MinUtil: %.2f MinUConf: %.4f\n", p->min_utility, p->min_confidence);
    printf("[HUCI-Miner] High Utility Itemsets: %zu\n", stats.high_utility_itemsets);
    printf("[HUCI-Miner] High Utility Closed Itemsets: %zu\n", stats.high_utility_closed_itemsets);
    printf("[HUCI-Miner] High Utility Generators: %zu\n", stats.high_utility_generators);
    printf("[HUCI-Miner] HGB Rules: %zu\n", stats.hgb_rules);
    printf("[HUCI-Miner] Max Depth: %zu\n", stats.max_depth);
    printf("[HUCI-Miner] Utility Lists Constructed: %zu\n", stats.utility_lists_constructed);
    printf("[HUCI-Miner] Joins: %zu\n", stats.joins);
    printf("[HUCI-Miner] EUCS Prunes: %zu\n", stats.pruned_eucs);
    printf("[HUCI-Miner] Subtree Utility Prunes: %zu\n", stats.pruned_subtree_utility);
    dm_bench_record_results(stats.high_utility_closed_itemsets, 0);
    return DM_SUCCESS;
}

static DM_Algorithm huci_algo = {
    .id = "huciminer",
    .name = "HUCI-Miner",
    .description = "High Utility Closed Itemset Miner with high utility generators and HGB rules.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(huci_algo)
