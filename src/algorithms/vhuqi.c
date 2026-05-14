#include "algorithms/vhuqi.h"
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
} QItem;

typedef struct {
    uint32_t tid;
    double eutil;
    double rutil;
} VTuple;

typedef struct {
    QItem *qitems;
    size_t count;
    VTuple *tuples;
    size_t tuple_count;
    size_t tuple_capacity;
    double sum_eutil;
    double max_rutil;
} VUtilityList;

typedef struct {
    VUtilityList **lists;
    size_t count;
    size_t capacity;
} VListSet;

typedef struct {
    QItem qitem;
    double sum_eutil;
} InitialQItem;

static size_t total_huqi_count;
static size_t total_qitems_sum;

static bool qitem_equal(QItem a, QItem b) {
    return a.item == b.item && a.low == b.low && a.high == b.high;
}

static bool same_prefix(VUtilityList *x, VUtilityList *y) {
    if (x->count != y->count || x->count == 0) return false;
    for (size_t i = 0; i + 1 < x->count; i++) {
        if (!qitem_equal(x->qitems[i], y->qitems[i])) return false;
    }
    return true;
}

static int cmp_initial_qitem_desc(const void *a, const void *b) {
    const InitialQItem *x = (const InitialQItem *)a;
    const InitialQItem *y = (const InitialQItem *)b;
    if (x->sum_eutil > y->sum_eutil) return -1;
    if (x->sum_eutil < y->sum_eutil) return 1;
    if (x->qitem.item < y->qitem.item) return -1;
    if (x->qitem.item > y->qitem.item) return 1;
    if (x->qitem.low < y->qitem.low) return -1;
    if (x->qitem.low > y->qitem.low) return 1;
    return 0;
}

static int cmp_list_desc(const void *a, const void *b) {
    VUtilityList *x = *(VUtilityList **)a;
    VUtilityList *y = *(VUtilityList **)b;
    if (x->sum_eutil > y->sum_eutil) return -1;
    if (x->sum_eutil < y->sum_eutil) return 1;
    QItem xq = x->qitems[x->count - 1];
    QItem yq = y->qitems[y->count - 1];
    if (xq.item < yq.item) return -1;
    if (xq.item > yq.item) return 1;
    if (xq.low < yq.low) return -1;
    if (xq.low > yq.low) return 1;
    return 0;
}

static void listset_init(VListSet *set) {
    set->lists = NULL;
    set->count = 0;
    set->capacity = 0;
}

static bool listset_push(VListSet *set, VUtilityList *list) {
    if (set->count == set->capacity) {
        size_t new_capacity = set->capacity ? set->capacity * 2 : 16;
        VUtilityList **new_lists = realloc(set->lists, sizeof(VUtilityList *) * new_capacity);
        if (!new_lists) return false;
        set->lists = new_lists;
        set->capacity = new_capacity;
    }
    set->lists[set->count++] = list;
    return true;
}

static VUtilityList *create_list(size_t qitem_count, size_t tuple_capacity) {
    VUtilityList *list = calloc(1, sizeof(VUtilityList));
    if (!list) return NULL;
    list->qitems = malloc(sizeof(QItem) * qitem_count);
    list->tuples = malloc(sizeof(VTuple) * (tuple_capacity ? tuple_capacity : 1));
    if (!list->qitems || !list->tuples) {
        free(list->qitems);
        free(list->tuples);
        free(list);
        return NULL;
    }
    list->count = qitem_count;
    list->tuple_capacity = tuple_capacity ? tuple_capacity : 1;
    return list;
}

static void free_list(VUtilityList *list) {
    if (!list) return;
    free(list->qitems);
    free(list->tuples);
    free(list);
}

static void free_listset_lists(VListSet *set) {
    for (size_t i = 0; i < set->count; i++) free_list(set->lists[i]);
    free(set->lists);
}

static void free_listset_shallow(VListSet *set) {
    free(set->lists);
}

static bool append_tuple(VUtilityList *list, uint32_t tid, double eutil, double rutil) {
    if (list->tuple_count == list->tuple_capacity) {
        size_t new_capacity = list->tuple_capacity * 2;
        VTuple *new_tuples = realloc(list->tuples, sizeof(VTuple) * new_capacity);
        if (!new_tuples) return false;
        list->tuples = new_tuples;
        list->tuple_capacity = new_capacity;
    }
    list->tuples[list->tuple_count].tid = tid;
    list->tuples[list->tuple_count].eutil = eutil;
    list->tuples[list->tuple_count].rutil = rutil;
    list->tuple_count++;
    list->sum_eutil += eutil;
    if (rutil > list->max_rutil) list->max_rutil = rutil;
    return true;
}

static VUtilityList *find_list(VUtilityList **lists, size_t count, QItem qitem) {
    for (size_t i = 0; i < count; i++) {
        if (qitem_equal(lists[i]->qitems[0], qitem)) return lists[i];
    }
    return NULL;
}

static double prefix_eutil(VUtilityList *p, uint32_t tid, size_t *pos) {
    if (!p) return 0.0;
    while (*pos < p->tuple_count && p->tuples[*pos].tid < tid) (*pos)++;
    if (*pos < p->tuple_count && p->tuples[*pos].tid == tid) return p->tuples[*pos].eutil;
    return 0.0;
}

static VUtilityList *construct_list(VUtilityList *p, VUtilityList *x, VUtilityList *y) {
    if (!same_prefix(x, y)) return NULL;
    QItem xlast = x->qitems[x->count - 1];
    QItem ylast = y->qitems[y->count - 1];
    if (xlast.item == ylast.item) return NULL;

    size_t capacity = x->tuple_count < y->tuple_count ? x->tuple_count : y->tuple_count;
    VUtilityList *z = create_list(x->count + 1, capacity);
    if (!z) return NULL;
    memcpy(z->qitems, x->qitems, sizeof(QItem) * x->count);
    z->qitems[x->count] = ylast;

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < x->tuple_count && iy < y->tuple_count) {
        if (x->tuples[ix].tid == y->tuples[iy].tid) {
            uint32_t tid = x->tuples[ix].tid;
            double eutil = x->tuples[ix].eutil + y->tuples[iy].eutil - prefix_eutil(p, tid, &ip);
            if (!append_tuple(z, tid, eutil, y->tuples[iy].rutil)) {
                free_list(z);
                return NULL;
            }
            ix++;
            iy++;
        } else if (x->tuples[ix].tid < y->tuples[iy].tid) {
            ix++;
        } else {
            iy++;
        }
    }

    if (z->tuple_count == 0) {
        free_list(z);
        return NULL;
    }
    return z;
}

static VUtilityList *merge_utility_lists(VUtilityList *x, VUtilityList *y) {
    if (!same_prefix(x, y)) return NULL;
    QItem xlast = x->qitems[x->count - 1];
    QItem ylast = y->qitems[y->count - 1];
    if (xlast.item != ylast.item || xlast.high + 1 != ylast.low) return NULL;

    VUtilityList *z = create_list(x->count, x->tuple_count + y->tuple_count);
    if (!z) return NULL;
    memcpy(z->qitems, x->qitems, sizeof(QItem) * x->count);
    z->qitems[z->count - 1].high = ylast.low;

    size_t ix = 0, iy = 0;
    while (ix < x->tuple_count || iy < y->tuple_count) {
        VTuple t;
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
            free_list(z);
            return NULL;
        }
    }
    return z;
}

static double k_support_bound(VUtilityList *list, double min_abs_util) {
    double bound = list->sum_eutil + list->max_rutil;
    if (bound <= 0.0) return HUGE_VAL;
    return ceil(min_abs_util / bound);
}

static void combine_lists(VUtilityList **cql, size_t cql_count, double min_abs_util, VListSet *hc, VListSet *culs) {
    listset_init(hc);
    listset_init(culs);
    for (size_t i = 0; i < cql_count; i++) {
        VUtilityList *z = NULL;
        for (size_t j = i + 1; j < cql_count; j++) {
            QItem xlast = cql[i]->qitems[cql[i]->count - 1];
            QItem ylast = cql[j]->qitems[cql[j]->count - 1];
            if (!same_prefix(cql[i], cql[j]) || xlast.item != ylast.item) continue;

            if (!z) {
                if (xlast.high + 1 == ylast.low) z = merge_utility_lists(cql[i], cql[j]);
            } else {
                QItem zlast = z->qitems[z->count - 1];
                if (zlast.item == ylast.item && zlast.high + 1 == ylast.low) {
                    VUtilityList *merged = merge_utility_lists(z, cql[j]);
                    free_list(z);
                    z = merged;
                }
            }

            if (z && z->sum_eutil >= min_abs_util) {
                if (!listset_push(hc, z) || !listset_push(culs, z)) {
                    free_list(z);
                }
                z = NULL;
                break;
            }
        }
        free_list(z);
    }
}

static bool add_unique_shallow(VListSet *set, VUtilityList *list) {
    for (size_t i = 0; i < set->count; i++) {
        if (set->lists[i] == list) return true;
    }
    return listset_push(set, list);
}

static void huqi_search(VUtilityList *prefix, VUtilityList **uls, size_t ul_count, double min_abs_util, double qrc) {
    VListSet h, w, c, hc, culs, join;
    listset_init(&h);
    listset_init(&w);
    listset_init(&c);
    listset_init(&join);

    for (size_t i = 0; i < ul_count; i++) {
        VUtilityList *x = uls[i];
        if (x->sum_eutil >= min_abs_util) {
            total_huqi_count++;
            total_qitems_sum += x->count;
            add_unique_shallow(&h, x);
        } else if ((double)x->tuple_count >= k_support_bound(x, min_abs_util)) {
            add_unique_shallow(&w, x);
        }
        if (x->sum_eutil >= min_abs_util / qrc) {
            add_unique_shallow(&c, x);
        }
    }

    combine_lists(c.lists, c.count, min_abs_util, &hc, &culs);
    for (size_t i = 0; i < hc.count; i++) {
        total_huqi_count++;
        total_qitems_sum += hc.lists[i]->count;
    }

    for (size_t i = 0; i < h.count; i++) add_unique_shallow(&join, h.lists[i]);
    for (size_t i = 0; i < w.count; i++) add_unique_shallow(&join, w.lists[i]);
    for (size_t i = 0; i < hc.count; i++) add_unique_shallow(&join, hc.lists[i]);
    if (join.count > 1) qsort(join.lists, join.count, sizeof(VUtilityList *), cmp_list_desc);

    for (size_t i = 0; i < join.count; i++) {
        VUtilityList *x = join.lists[i];
        VListSet x_uls;
        listset_init(&x_uls);
        for (size_t j = i + 1; j < join.count; j++) {
            VUtilityList *y = join.lists[j];
            if (!same_prefix(x, y)) continue;
            if (x->qitems[x->count - 1].item == y->qitems[y->count - 1].item) continue;
            VUtilityList *z = construct_list(prefix, x, y);
            if (z) listset_push(&x_uls, z);
        }
        if (x_uls.count > 0) huqi_search(x, x_uls.lists, x_uls.count, min_abs_util, qrc);
        free_listset_lists(&x_uls);
    }

    free_listset_shallow(&h);
    free_listset_shallow(&w);
    free_listset_shallow(&c);
    free_listset_shallow(&join);
    for (size_t i = 0; i < culs.count; i++) free_list(culs.lists[i]);
    free(hc.lists);
    free(culs.lists);
}

static int qitem_index(InitialQItem *items, size_t count, QItem qitem) {
    for (size_t i = 0; i < count; i++) {
        if (qitem_equal(items[i].qitem, qitem)) return (int)i;
    }
    return -1;
}

static int list_rank(InitialQItem *items, size_t count, QItem qitem) {
    for (size_t i = 0; i < count; i++) {
        if (qitem_equal(items[i].qitem, qitem)) return (int)i;
    }
    return -1;
}

typedef struct {
    QItem qitem;
    double eutil;
    int rank;
} TransQItem;

static int cmp_trans_rank_desc(const void *a, const void *b) {
    const TransQItem *x = (const TransQItem *)a;
    const TransQItem *y = (const TransQItem *)b;
    if (x->rank < y->rank) return -1;
    if (x->rank > y->rank) return 1;
    return 0;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;

    DM_VHUQI_Params *vp = (DM_VHUQI_Params *)params;
    double min_abs_util = vp ? vp->min_abs_utility : 1000.0;
    double qrc = (vp && vp->qrc >= 1.0) ? vp->qrc : 3.0;
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;

    total_huqi_count = 0;
    total_qitems_sum = 0;

    InitialQItem *qitems = NULL;
    size_t qitem_count = 0, qitem_capacity = 0;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            int quantity = (int)llround(data[i].items[j].utility);
            QItem qitem = { data[i].items[j].id, quantity, quantity };
            int idx = qitem_index(qitems, qitem_count, qitem);
            if (idx < 0) {
                if (qitem_count == qitem_capacity) {
                    size_t new_capacity = qitem_capacity ? qitem_capacity * 2 : 128;
                    InitialQItem *new_qitems = realloc(qitems, sizeof(InitialQItem) * new_capacity);
                    if (!new_qitems) {
                        free(qitems);
                        return DM_ERROR_MEMORY;
                    }
                    qitems = new_qitems;
                    qitem_capacity = new_capacity;
                }
                qitems[qitem_count].qitem = qitem;
                qitems[qitem_count].sum_eutil = data[i].items[j].utility;
                qitem_count++;
            } else {
                qitems[idx].sum_eutil += data[i].items[j].utility;
            }
        }
    }

    if (qitem_count > 1) qsort(qitems, qitem_count, sizeof(InitialQItem), cmp_initial_qitem_desc);

    VUtilityList **initial = malloc(sizeof(VUtilityList *) * qitem_count);
    if (!initial && qitem_count > 0) {
        free(qitems);
        return DM_ERROR_MEMORY;
    }
    for (size_t i = 0; i < qitem_count; i++) {
        initial[i] = create_list(1, 16);
        if (!initial[i]) {
            for (size_t k = 0; k < i; k++) free_list(initial[k]);
            free(initial);
            free(qitems);
            return DM_ERROR_MEMORY;
        }
        initial[i]->qitems[0] = qitems[i].qitem;
    }

    for (size_t i = 0; i < ds->count; i++) {
        TransQItem *transaction = malloc(sizeof(TransQItem) * data[i].count);
        if (!transaction) continue;
        size_t count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            int quantity = (int)llround(data[i].items[j].utility);
            QItem qitem = { data[i].items[j].id, quantity, quantity };
            int rank = list_rank(qitems, qitem_count, qitem);
            if (rank >= 0) {
                transaction[count].qitem = qitem;
                transaction[count].eutil = data[i].items[j].utility;
                transaction[count].rank = rank;
                count++;
            }
        }
        qsort(transaction, count, sizeof(TransQItem), cmp_trans_rank_desc);

        double remaining_utility = 0.0;
        for (size_t r = count; r-- > 0;) {
            VUtilityList *list = find_list(initial, qitem_count, transaction[r].qitem);
            if (list) append_tuple(list, (uint32_t)i, transaction[r].eutil, remaining_utility);
            remaining_utility += transaction[r].eutil;
        }
        free(transaction);
    }

    printf("[VHUQI] Starting HUQI-Search with min_abs_util %.3f and qrc %.3f...\n", min_abs_util, qrc);
    huqi_search(NULL, initial, qitem_count, min_abs_util, qrc);
    printf("[VHUQI] Found %zu High Utility Quantitative Itemsets.\n", total_huqi_count);

    for (size_t i = 0; i < qitem_count; i++) free_list(initial[i]);
    free(initial);
    free(qitems);

    dm_bench_record_results(total_huqi_count, total_qitems_sum);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "vhuqi",
    .name = "VHUQI",
    .description = "Vertical mining of High Utility Quantitative Itemsets.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
