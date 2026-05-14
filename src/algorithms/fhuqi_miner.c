#include "algorithms/fhuqi_miner.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t item;
    int low;
    int high;
} FQItem;

typedef struct {
    uint32_t tid;
    double eutil;
    double rutil;
} FTuple;

typedef struct {
    FQItem *qitems;
    size_t count;
    FTuple *tuples;
    size_t tuple_count;
    size_t tuple_capacity;
    double sum_eutil;
    double sum_rutil;
    double twu;
} FUL;

typedef struct {
    FUL **lists;
    size_t count;
    size_t capacity;
} FSet;

typedef struct {
    FQItem qitem;
    double twu;
    double utility;
} FExact;

typedef struct {
    FQItem a;
    FQItem b;
    uint64_t key;
    double twu;
} TQCS_Entry;

typedef struct {
    TQCS_Entry *entries;
    size_t count;
    size_t capacity;
} TQCS;

typedef struct {
    FQItem qitem;
    double eutil;
    int rank;
} FTransItem;

typedef struct {
    FQItem qitem;
    int rank;
} FRankEntry;

static size_t total_huqi_count;
static size_t total_qitems_sum;
static FExact *g_ordered_exact;
static size_t g_ordered_exact_count;
static FRankEntry *g_rank_index;
static double *g_transaction_utilities;
static size_t g_transaction_count;
static TQCS g_tqcs;

static bool load_profit_table(const char *path, double *profits, size_t count) {
    for (size_t i = 0; i < count; i++) profits[i] = 1.0;
    if (!path || path[0] == '\0') return true;

    FILE *file = fopen(path, "r");
    if (!file) return false;

    char line[4096];
    while (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, ", \t\r\n");
        if (!token) continue;
        uint32_t id = (uint32_t)atoi(token);
        token = strtok(NULL, ", \t\r\n");
        if (!token) continue;
        if (id < count) profits[id] = atof(token);
    }
    fclose(file);
    return true;
}

static bool qitem_equal(FQItem a, FQItem b) {
    return a.item == b.item && a.low == b.low && a.high == b.high;
}

static bool qitem_exact(FQItem q) {
    return q.low == q.high;
}

static bool ul_exact(FUL *ul) {
    for (size_t i = 0; i < ul->count; i++) {
        if (!qitem_exact(ul->qitems[i])) return false;
    }
    return true;
}

static bool same_prefix(FUL *x, FUL *y) {
    if (x->count != y->count || x->count == 0) return false;
    for (size_t i = 0; i + 1 < x->count; i++) {
        if (!qitem_equal(x->qitems[i], y->qitems[i])) return false;
    }
    return true;
}

static int cmp_exact_desc(const void *a, const void *b) {
    const FExact *x = (const FExact *)a;
    const FExact *y = (const FExact *)b;
    if (x->utility > y->utility) return -1;
    if (x->utility < y->utility) return 1;
    if (x->qitem.item < y->qitem.item) return -1;
    if (x->qitem.item > y->qitem.item) return 1;
    return x->qitem.low - y->qitem.low;
}

static int cmp_rank_entry(const void *a, const void *b) {
    const FRankEntry *x = (const FRankEntry *)a;
    const FRankEntry *y = (const FRankEntry *)b;
    if (x->qitem.item < y->qitem.item) return -1;
    if (x->qitem.item > y->qitem.item) return 1;
    if (x->qitem.low < y->qitem.low) return -1;
    if (x->qitem.low > y->qitem.low) return 1;
    return 0;
}

static int qitem_order(FQItem q) {
    size_t lo = 0;
    size_t hi = g_ordered_exact_count;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        FQItem m = g_rank_index[mid].qitem;
        if (m.item < q.item || (m.item == q.item && m.low < q.low)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo < g_ordered_exact_count && g_rank_index[lo].qitem.item == q.item && g_rank_index[lo].qitem.low == q.low) {
        return g_rank_index[lo].rank;
    }
    return -1;
}

static int cmp_list_order(const void *a, const void *b) {
    FUL *x = *(FUL **)a;
    FUL *y = *(FUL **)b;
    int ox = qitem_order(x->qitems[x->count - 1]);
    int oy = qitem_order(y->qitems[y->count - 1]);
    if (ox != oy) return ox - oy;
    return 0;
}

static int cmp_trans_order(const void *a, const void *b) {
    const FTransItem *x = (const FTransItem *)a;
    const FTransItem *y = (const FTransItem *)b;
    return x->rank - y->rank;
}

static void set_init(FSet *set) {
    set->lists = NULL;
    set->count = 0;
    set->capacity = 0;
}

static bool set_push(FSet *set, FUL *list) {
    if (set->count == set->capacity) {
        size_t new_capacity = set->capacity ? set->capacity * 2 : 16;
        FUL **new_lists = realloc(set->lists, sizeof(FUL *) * new_capacity);
        if (!new_lists) return false;
        set->lists = new_lists;
        set->capacity = new_capacity;
    }
    set->lists[set->count++] = list;
    return true;
}

static bool set_push_unique(FSet *set, FUL *list) {
    for (size_t i = 0; i < set->count; i++) {
        if (set->lists[i] == list) return true;
    }
    return set_push(set, list);
}

static void set_free_shallow(FSet *set) {
    free(set->lists);
}

static void set_free_lists(FSet *set) {
    for (size_t i = 0; i < set->count; i++) {
        free(set->lists[i]->qitems);
        free(set->lists[i]->tuples);
        free(set->lists[i]);
    }
    free(set->lists);
}

static FUL *create_ul(size_t qitem_count, size_t tuple_capacity) {
    FUL *ul = calloc(1, sizeof(FUL));
    if (!ul) return NULL;
    ul->qitems = malloc(sizeof(FQItem) * qitem_count);
    ul->tuples = malloc(sizeof(FTuple) * (tuple_capacity ? tuple_capacity : 1));
    if (!ul->qitems || !ul->tuples) {
        free(ul->qitems);
        free(ul->tuples);
        free(ul);
        return NULL;
    }
    ul->count = qitem_count;
    ul->tuple_capacity = tuple_capacity ? tuple_capacity : 1;
    return ul;
}

static void free_ul(FUL *ul) {
    if (!ul) return;
    free(ul->qitems);
    free(ul->tuples);
    free(ul);
}

static FUL *clone_ul(FUL *ul) {
    FUL *copy = create_ul(ul->count, ul->tuple_count);
    if (!copy) return NULL;
    memcpy(copy->qitems, ul->qitems, sizeof(FQItem) * ul->count);
    memcpy(copy->tuples, ul->tuples, sizeof(FTuple) * ul->tuple_count);
    copy->tuple_count = ul->tuple_count;
    copy->sum_eutil = ul->sum_eutil;
    copy->sum_rutil = ul->sum_rutil;
    copy->twu = ul->twu;
    return copy;
}

static bool append_tuple(FUL *ul, uint32_t tid, double eutil, double rutil) {
    if (ul->tuple_count == ul->tuple_capacity) {
        size_t new_capacity = ul->tuple_capacity * 2;
        FTuple *new_tuples = realloc(ul->tuples, sizeof(FTuple) * new_capacity);
        if (!new_tuples) return false;
        ul->tuples = new_tuples;
        ul->tuple_capacity = new_capacity;
    }
    ul->tuples[ul->tuple_count].tid = tid;
    ul->tuples[ul->tuple_count].eutil = eutil;
    ul->tuples[ul->tuple_count].rutil = rutil;
    ul->tuple_count++;
    ul->sum_eutil += eutil;
    ul->sum_rutil += rutil;
    return true;
}

static int exact_index(FExact *items, size_t count, FQItem qitem) {
    for (size_t i = 0; i < count; i++) {
        if (qitem_equal(items[i].qitem, qitem)) return (int)i;
    }
    return -1;
}

static FUL *find_single_ul(FUL **uls, size_t count, FQItem qitem) {
    for (size_t i = 0; i < count; i++) {
        if (qitem_equal(uls[i]->qitems[0], qitem)) return uls[i];
    }
    return NULL;
}

static void tqcs_init(TQCS *tqcs) {
    tqcs->entries = NULL;
    tqcs->count = 0;
    tqcs->capacity = 0;
}

static void tqcs_free(TQCS *tqcs) {
    free(tqcs->entries);
    tqcs->entries = NULL;
    tqcs->count = 0;
    tqcs->capacity = 0;
}

static uint64_t tqcs_key_from_order(int oa, int ob) {
    if (oa > ob) {
        int tmp = oa;
        oa = ob;
        ob = tmp;
    }
    return (((uint64_t)(uint32_t)(oa + 1)) << 32) | (uint32_t)(ob + 1);
}

static size_t tqcs_hash(uint64_t key, size_t capacity) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    return (size_t)key & (capacity - 1);
}

static bool tqcs_rehash(TQCS *tqcs, size_t new_capacity) {
    TQCS_Entry *old_entries = tqcs->entries;
    size_t old_capacity = tqcs->capacity;
    tqcs->entries = calloc(new_capacity, sizeof(TQCS_Entry));
    if (!tqcs->entries) {
        tqcs->entries = old_entries;
        return false;
    }
    tqcs->capacity = new_capacity;
    tqcs->count = 0;
    for (size_t i = 0; i < old_capacity; i++) {
        if (old_entries[i].key == 0) continue;
        size_t slot = tqcs_hash(old_entries[i].key, tqcs->capacity);
        while (tqcs->entries[slot].key != 0) slot = (slot + 1) & (tqcs->capacity - 1);
        tqcs->entries[slot] = old_entries[i];
        tqcs->count++;
    }
    free(old_entries);
    return true;
}

static void tqcs_add_ordered(TQCS *tqcs, FQItem a, FQItem b, int oa, int ob, double twu) {
    if (tqcs->capacity == 0 && !tqcs_rehash(tqcs, 2048)) return;
    if ((tqcs->count + 1) * 2 >= tqcs->capacity && !tqcs_rehash(tqcs, tqcs->capacity * 2)) return;

    if (oa > ob) {
        FQItem tmp = a;
        a = b;
        b = tmp;
    }
    uint64_t key = tqcs_key_from_order(oa, ob);
    size_t slot = tqcs_hash(key, tqcs->capacity);
    while (tqcs->entries[slot].key != 0 && tqcs->entries[slot].key != key) {
        slot = (slot + 1) & (tqcs->capacity - 1);
    }
    if (tqcs->entries[slot].key == key) {
        tqcs->entries[slot].twu += twu;
        return;
    }
    tqcs->entries[slot].a = a;
    tqcs->entries[slot].b = b;
    tqcs->entries[slot].key = key;
    tqcs->entries[slot].twu = twu;
    tqcs->count++;
}

static double tqcs_get(TQCS *tqcs, FQItem a, FQItem b) {
    int oa = qitem_order(a);
    int ob = qitem_order(b);
    if (oa < 0 || ob < 0 || tqcs->capacity == 0) return 0.0;
    uint64_t key = tqcs_key_from_order(oa, ob);
    size_t slot = tqcs_hash(key, tqcs->capacity);
    while (tqcs->entries[slot].key != 0) {
        if (tqcs->entries[slot].key == key) return tqcs->entries[slot].twu;
        slot = (slot + 1) & (tqcs->capacity - 1);
    }
    return 0.0;
}

static double prefix_eutil(FUL *p, uint32_t tid, size_t *pos) {
    if (!p) return 0.0;
    while (*pos < p->tuple_count && p->tuples[*pos].tid < tid) (*pos)++;
    if (*pos < p->tuple_count && p->tuples[*pos].tid == tid) return p->tuples[*pos].eutil;
    return 0.0;
}

static FUL *construct_ul(FUL *p, FUL *x, FUL *y) {
    if (!same_prefix(x, y)) return NULL;
    if (x->qitems[x->count - 1].item == y->qitems[y->count - 1].item) return NULL;

    size_t capacity = x->tuple_count < y->tuple_count ? x->tuple_count : y->tuple_count;
    FUL *z = create_ul(x->count + 1, capacity);
    if (!z) return NULL;
    memcpy(z->qitems, x->qitems, sizeof(FQItem) * x->count);
    z->qitems[x->count] = y->qitems[y->count - 1];

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < x->tuple_count && iy < y->tuple_count) {
        if (x->tuples[ix].tid == y->tuples[iy].tid) {
            uint32_t tid = x->tuples[ix].tid;
            double eutil = x->tuples[ix].eutil + y->tuples[iy].eutil - prefix_eutil(p, tid, &ip);
            if (!append_tuple(z, tid, eutil, y->tuples[iy].rutil)) {
                free_ul(z);
                return NULL;
            }
            if (tid < g_transaction_count) z->twu += g_transaction_utilities[tid];
            ix++;
            iy++;
        } else if (x->tuples[ix].tid < y->tuples[iy].tid) {
            ix++;
        } else {
            iy++;
        }
    }
    if (z->tuple_count == 0) {
        free_ul(z);
        return NULL;
    }
    return z;
}

static FUL *merge_ul(FUL *x, FUL *y) {
    if (!same_prefix(x, y)) return NULL;
    FQItem xlast = x->qitems[x->count - 1];
    FQItem ylast = y->qitems[y->count - 1];
    if (xlast.item != ylast.item || xlast.high + 1 != ylast.low) return NULL;

    FUL *z = create_ul(x->count, x->tuple_count + y->tuple_count);
    if (!z) return NULL;
    memcpy(z->qitems, x->qitems, sizeof(FQItem) * x->count);
    z->qitems[z->count - 1].high = ylast.high;

    size_t ix = 0, iy = 0;
    while (ix < x->tuple_count || iy < y->tuple_count) {
        FTuple t;
        if (iy == y->tuple_count || (ix < x->tuple_count && x->tuples[ix].tid < y->tuples[iy].tid)) {
            t = x->tuples[ix++];
        } else if (ix == x->tuple_count || y->tuples[iy].tid < x->tuples[ix].tid) {
            t = y->tuples[iy++];
        } else {
            t.tid = x->tuples[ix].tid;
            t.eutil = x->tuples[ix].eutil + y->tuples[iy].eutil;
            t.rutil = x->tuples[ix].rutil > y->tuples[iy].rutil ? x->tuples[ix].rutil : y->tuples[iy].rutil;
            ix++;
            iy++;
        }
        if (!append_tuple(z, t.tid, t.eutil, t.rutil)) {
            free_ul(z);
            return NULL;
        }
        if (t.tid < g_transaction_count) z->twu += g_transaction_utilities[t.tid];
    }
    return z;
}

static bool tqcs_allows(FQItem x, FQItem y, double min_utility, double qrc) {
    double threshold = min_utility / qrc;
    if (qitem_exact(x)) {
        return tqcs_get(&g_tqcs, x, y) >= threshold;
    }
    double sum = 0.0;
    for (int q = x.low; q <= x.high; q++) {
        FQItem exact = { x.item, q, q };
        sum += tqcs_get(&g_tqcs, exact, y);
    }
    return sum >= threshold;
}

static void collect_status(FUL *z, double min_utility, double qrc, FSet *h, FSet *e, FSet *c) {
    if (z->sum_eutil >= min_utility) {
        total_huqi_count++;
        total_qitems_sum += z->count;
        set_push_unique(h, z);
    } else {
        if (z->sum_eutil + z->sum_rutil >= min_utility) set_push_unique(e, z);
        if (z->sum_eutil >= min_utility / qrc && z->sum_eutil <= min_utility) set_push_unique(c, z);
    }
}

static bool same_interval_family(FUL *a, FUL *b) {
    if (!same_prefix(a, b)) return false;
    FQItem alast = a->qitems[a->count - 1];
    FQItem blast = b->qitems[b->count - 1];
    return alast.item == blast.item;
}

static bool interval_contains(FQItem outer, FQItem inner) {
    return outer.item == inner.item && outer.low <= inner.low && outer.high >= inner.high;
}

static void filter_minimal_intervals(FSet *set) {
    size_t out = 0;
    for (size_t i = 0; i < set->count; i++) {
        bool discard = false;
        FQItem ilast = set->lists[i]->qitems[set->lists[i]->count - 1];
        int isize = ilast.high - ilast.low + 1;
        for (size_t j = 0; j < set->count; j++) {
            if (i == j || !same_interval_family(set->lists[i], set->lists[j])) continue;
            FQItem jlast = set->lists[j]->qitems[set->lists[j]->count - 1];
            int jsize = jlast.high - jlast.low + 1;
            if (jsize < isize && interval_contains(ilast, jlast)) {
                discard = true;
                break;
            }
        }
        if (discard) free_ul(set->lists[i]);
        else set->lists[out++] = set->lists[i];
    }
    set->count = out;
}

static void filter_maximal_intervals(FSet *set) {
    size_t out = 0;
    for (size_t i = 0; i < set->count; i++) {
        bool discard = false;
        FQItem ilast = set->lists[i]->qitems[set->lists[i]->count - 1];
        int isize = ilast.high - ilast.low + 1;
        for (size_t j = 0; j < set->count; j++) {
            if (i == j || !same_interval_family(set->lists[i], set->lists[j])) continue;
            FQItem jlast = set->lists[j]->qitems[set->lists[j]->count - 1];
            int jsize = jlast.high - jlast.low + 1;
            if (jsize > isize && interval_contains(jlast, ilast)) {
                discard = true;
                break;
            }
        }
        if (discard) free_ul(set->lists[i]);
        else set->lists[out++] = set->lists[i];
    }
    set->count = out;
}

static void combine_candidates(FUL **candidates, size_t count, double min_utility, double qrc,
                               DM_FHUQI_Combine_Method method, FSet *hr) {
    set_init(hr);
    for (size_t i = 0; i < count; i++) {
        FUL *z = NULL;
        bool stop_for_min = false;
        FUL *max_huqi = NULL;
        for (size_t j = i + 1; j < count && !stop_for_min; j++) {
            FQItem xlast = candidates[i]->qitems[candidates[i]->count - 1];
            FQItem ylast = candidates[j]->qitems[candidates[j]->count - 1];
            if (!same_prefix(candidates[i], candidates[j]) || xlast.item != ylast.item) continue;

            if (!z) {
                if (xlast.high + 1 != ylast.low || ylast.high - xlast.low + 1 > (int)qrc) continue;
                z = merge_ul(candidates[i], candidates[j]);
            } else {
                FQItem zlast = z->qitems[z->count - 1];
                if (zlast.item != ylast.item || zlast.high + 1 != ylast.low || ylast.high - zlast.low + 1 > (int)qrc) continue;
                FUL *merged = merge_ul(z, candidates[j]);
                free_ul(z);
                z = merged;
            }
            if (!z) continue;

            if (z->sum_eutil >= min_utility) {
                if (method == DM_FHUQI_COMBINE_ALL) {
                    FUL *out = clone_ul(z);
                    if (out) set_push(hr, out);
                } else if (method == DM_FHUQI_COMBINE_MIN) {
                    set_push(hr, z);
                    z = NULL;
                    stop_for_min = true;
                } else {
                    free_ul(max_huqi);
                    max_huqi = clone_ul(z);
                }
            }
        }
        if (method == DM_FHUQI_COMBINE_MAX && max_huqi) set_push(hr, max_huqi);
        free_ul(z);
    }
    if (method == DM_FHUQI_COMBINE_MIN) filter_minimal_intervals(hr);
    else if (method == DM_FHUQI_COMBINE_MAX) filter_maximal_intervals(hr);
}

static void recursive_search(FUL *prefix, FUL **qis, size_t qis_count, FUL **pstar, size_t pstar_count,
                             double min_utility, double qrc, DM_FHUQI_Combine_Method method) {
    for (size_t i = 0; i < qis_count; i++) {
        FUL *px = qis[i];
        FSet h, e, c, pstar_next, hr, next_qis;
        set_init(&h);
        set_init(&e);
        set_init(&c);
        set_init(&pstar_next);
        set_init(&next_qis);

        FQItem x = px->qitems[px->count - 1];
        int ox = qitem_order(x);
        for (size_t j = 0; j < pstar_count; j++) {
            FUL *py = pstar[j];
            FQItem y = py->qitems[py->count - 1];
            int oy = qitem_order(y);
            if (oy <= ox || x.item == y.item) continue;
            if (ul_exact(px)) {
                if (!tqcs_allows(x, y, min_utility, qrc)) continue;
            } else if (!tqcs_allows(x, y, min_utility, qrc)) {
                continue;
            }

            FUL *z = construct_ul(prefix, px, py);
            if (!z) continue;
            if (z->twu < min_utility / qrc) {
                free_ul(z);
                continue;
            }
            set_push(&pstar_next, z);
            collect_status(z, min_utility, qrc, &h, &e, &c);
        }

        combine_candidates(c.lists, c.count, min_utility, qrc, method, &hr);
        for (size_t r = 0; r < hr.count; r++) {
            total_huqi_count++;
            total_qitems_sum += hr.lists[r]->count;
        }

        for (size_t k = 0; k < h.count; k++) set_push_unique(&next_qis, h.lists[k]);
        for (size_t k = 0; k < e.count; k++) set_push_unique(&next_qis, e.lists[k]);
        for (size_t k = 0; k < hr.count; k++) set_push_unique(&next_qis, hr.lists[k]);
        if (next_qis.count > 1) qsort(next_qis.lists, next_qis.count, sizeof(FUL *), cmp_list_order);

        if (next_qis.count > 0 && pstar_next.count > 0) {
            recursive_search(px, next_qis.lists, next_qis.count, pstar_next.lists, pstar_next.count,
                             min_utility, qrc, method);
        }

        set_free_shallow(&h);
        set_free_shallow(&e);
        set_free_shallow(&c);
        set_free_shallow(&next_qis);
        set_free_lists(&pstar_next);
        set_free_lists(&hr);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_QUANTITY) return DM_ERROR_INCOMPATIBLE;

    DM_FHUQI_Miner_Params *fp = (DM_FHUQI_Miner_Params *)params;
    double min_utility = fp ? fp->min_utility : 1000.0;
    double qrc = (fp && fp->qrc > 0.0) ? fp->qrc : 3.0;
    DM_FHUQI_Combine_Method method = fp ? fp->combine_method : DM_FHUQI_COMBINE_ALL;
    DM_Trans_Quantity *data = (DM_Trans_Quantity *)ds->payload;
    double *profits = malloc(sizeof(double) * (ds->max_id + 1));
    if (!profits) return DM_ERROR_MEMORY;
    if (!load_profit_table(fp ? fp->profit_path : NULL, profits, ds->max_id + 1)) {
        free(profits);
        return DM_ERROR_IO;
    }

    total_huqi_count = 0;
    total_qitems_sum = 0;
    tqcs_init(&g_tqcs);
    g_transaction_count = ds->count;
    g_transaction_utilities = malloc(sizeof(double) * ds->count);
    if (!g_transaction_utilities && ds->count > 0) {
        free(profits);
        return DM_ERROR_MEMORY;
    }
    for (size_t i = 0; i < ds->count; i++) {
        double transaction_utility = 0.0;
        for (size_t j = 0; j < data[i].count; j++) {
            transaction_utility += profits[data[i].items[j].id] * data[i].items[j].quantity;
        }
        g_transaction_utilities[i] = transaction_utility;
    }

    FExact *all = NULL;
    size_t all_count = 0, all_capacity = 0;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            int quantity = (int)llround(data[i].items[j].quantity);
            FQItem qitem = { data[i].items[j].id, quantity, quantity };
            int idx = exact_index(all, all_count, qitem);
            if (idx < 0) {
                if (all_count == all_capacity) {
                    size_t new_capacity = all_capacity ? all_capacity * 2 : 128;
                    FExact *new_all = realloc(all, sizeof(FExact) * new_capacity);
                    if (!new_all) {
                        free(all);
                        free(profits);
                        free(g_transaction_utilities);
                        g_transaction_utilities = NULL;
                        tqcs_free(&g_tqcs);
                        return DM_ERROR_MEMORY;
                    }
                    all = new_all;
                    all_capacity = new_capacity;
                }
                all[all_count].qitem = qitem;
                all[all_count].twu = g_transaction_utilities[i];
                all[all_count].utility = profits[data[i].items[j].id] * data[i].items[j].quantity;
                all_count++;
            } else {
                all[idx].twu += g_transaction_utilities[i];
                all[idx].utility += profits[data[i].items[j].id] * data[i].items[j].quantity;
            }
        }
    }

    FExact *promising = malloc(sizeof(FExact) * all_count);
    if (!promising && all_count > 0) {
        free(all);
        free(profits);
        free(g_transaction_utilities);
        g_transaction_utilities = NULL;
        tqcs_free(&g_tqcs);
        return DM_ERROR_MEMORY;
    }
    size_t promising_count = 0;
    for (size_t i = 0; i < all_count; i++) {
        if (all[i].twu >= min_utility / qrc) promising[promising_count++] = all[i];
    }
    free(all);
    if (promising_count > 1) qsort(promising, promising_count, sizeof(FExact), cmp_exact_desc);
    g_ordered_exact = promising;
    g_ordered_exact_count = promising_count;
    g_rank_index = malloc(sizeof(FRankEntry) * promising_count);
    if (!g_rank_index && promising_count > 0) {
        free(promising);
        free(profits);
        free(g_transaction_utilities);
        g_transaction_utilities = NULL;
        tqcs_free(&g_tqcs);
        return DM_ERROR_MEMORY;
    }
    for (size_t i = 0; i < promising_count; i++) {
        g_rank_index[i].qitem = promising[i].qitem;
        g_rank_index[i].rank = (int)i;
    }
    if (promising_count > 1) qsort(g_rank_index, promising_count, sizeof(FRankEntry), cmp_rank_entry);

    FUL **initial = malloc(sizeof(FUL *) * promising_count);
    if (!initial && promising_count > 0) {
        free(promising);
        free(g_rank_index);
        g_rank_index = NULL;
        free(profits);
        free(g_transaction_utilities);
        g_transaction_utilities = NULL;
        tqcs_free(&g_tqcs);
        return DM_ERROR_MEMORY;
    }
    for (size_t i = 0; i < promising_count; i++) {
        initial[i] = create_ul(1, 16);
        if (!initial[i]) {
            for (size_t k = 0; k < i; k++) free_ul(initial[k]);
            free(initial);
            free(promising);
            free(g_rank_index);
            g_rank_index = NULL;
            free(profits);
            free(g_transaction_utilities);
            g_transaction_utilities = NULL;
            tqcs_free(&g_tqcs);
            return DM_ERROR_MEMORY;
        }
        initial[i]->qitems[0] = promising[i].qitem;
        initial[i]->twu = promising[i].twu;
    }

    for (size_t i = 0; i < ds->count; i++) {
        FTransItem *transaction = malloc(sizeof(FTransItem) * data[i].count);
        if (!transaction) continue;
        size_t count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            int quantity = (int)llround(data[i].items[j].quantity);
            FQItem qitem = { data[i].items[j].id, quantity, quantity };
            int rank = qitem_order(qitem);
            if (rank >= 0) {
                transaction[count].qitem = qitem;
                transaction[count].eutil = profits[data[i].items[j].id] * data[i].items[j].quantity;
                transaction[count].rank = rank;
                count++;
            }
        }
        qsort(transaction, count, sizeof(FTransItem), cmp_trans_order);

        for (size_t a = 0; a < count; a++) {
            for (size_t b = a + 1; b < count; b++) {
                tqcs_add_ordered(&g_tqcs, transaction[a].qitem, transaction[b].qitem,
                                 transaction[a].rank, transaction[b].rank, g_transaction_utilities[i]);
            }
        }

        double remaining = 0.0;
        for (size_t r = count; r-- > 0;) {
            FUL *ul = find_single_ul(initial, promising_count, transaction[r].qitem);
            if (ul) append_tuple(ul, (uint32_t)i, transaction[r].eutil, remaining);
            remaining += transaction[r].eutil;
        }
        free(transaction);
    }

    FSet h, e, c, hr, qis;
    set_init(&h);
    set_init(&e);
    set_init(&c);
    set_init(&qis);

    for (size_t i = 0; i < promising_count; i++) {
        collect_status(initial[i], min_utility, qrc, &h, &e, &c);
    }

    combine_candidates(c.lists, c.count, min_utility, qrc, method, &hr);
    for (size_t i = 0; i < hr.count; i++) {
        total_huqi_count++;
        total_qitems_sum += hr.lists[i]->count;
    }

    for (size_t i = 0; i < h.count; i++) set_push_unique(&qis, h.lists[i]);
    for (size_t i = 0; i < e.count; i++) set_push_unique(&qis, e.lists[i]);
    for (size_t i = 0; i < hr.count; i++) set_push_unique(&qis, hr.lists[i]);
    if (qis.count > 1) qsort(qis.lists, qis.count, sizeof(FUL *), cmp_list_order);

    printf("[FHUQI-Miner] Starting search with theta %.3f, qrc %.3f, method %d...\n",
           min_utility, qrc, (int)method);
    recursive_search(NULL, qis.lists, qis.count, initial, promising_count, min_utility, qrc, method);
    printf("[FHUQI-Miner] Found %zu High Utility Quantitative Itemsets.\n", total_huqi_count);

    set_free_shallow(&h);
    set_free_shallow(&e);
    set_free_shallow(&c);
    set_free_shallow(&qis);
    set_free_lists(&hr);
    for (size_t i = 0; i < promising_count; i++) free_ul(initial[i]);
    free(initial);
    free(g_rank_index);
    g_rank_index = NULL;
    free(profits);
    free(g_transaction_utilities);
    g_transaction_utilities = NULL;
    g_transaction_count = 0;
    free(promising);
    tqcs_free(&g_tqcs);

    dm_bench_record_results(total_huqi_count, total_qitems_sum);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "fhuqi_miner",
    .name = "FHUQI-Miner",
    .description = "Fast High Utility Quantitative Itemset Miner with EQCPS and RQCPS pruning.",
    .supported_types = (1 << DM_TYPE_QUANTITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
