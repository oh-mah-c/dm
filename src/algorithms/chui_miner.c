#include "algorithms/chui_miner.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t tid;
    double eu;
    double ru;
} CHUI_Element;

typedef struct {
    uint32_t item_rank;
    uint32_t item_id;
    CHUI_Element *elements;
    size_t count;
    size_t capacity;
    double sum_eu;
    double sum_ru;
} CHUI_Initial_List;

typedef struct {
    uint32_t *items;
    size_t item_count;
    size_t item_capacity;
    CHUI_Element *elements;
    size_t count;
    size_t capacity;
    double sum_eu;
    double sum_ru;
} CHUI_EUList;

typedef struct {
    uint32_t *items;
    size_t count;
    size_t capacity;
} CHUI_ItemVec;

typedef struct {
    uint32_t **sets;
    size_t *lengths;
    size_t count;
    size_t capacity;
} CHUI_Emitted;

typedef struct {
    const DM_Dataset *dataset;
    const DM_Trans_Utility *transactions;
    double min_utility;
    uint32_t *promising_ids;
    uint32_t *rank_of_item;
    size_t promising_count;
    CHUI_Initial_List *initial;
    CHUI_Emitted emitted;
    size_t closed_count;
    size_t total_closed_items;
    size_t constructed_lists;
    size_t closure_extensions;
    size_t pruned_by_ru;
    size_t pruned_by_subsumption;
} CHUI_Context;

static double *g_sort_twu = NULL;

static int compare_item_by_twu_desc(const void *lhs, const void *rhs) {
    uint32_t a = *(const uint32_t *)lhs;
    uint32_t b = *(const uint32_t *)rhs;
    if (g_sort_twu[a] > g_sort_twu[b]) return -1;
    if (g_sort_twu[a] < g_sort_twu[b]) return 1;
    return (a > b) - (a < b);
}

static void *xcalloc(size_t n, size_t size) {
    void *ptr = calloc(n, size);
    if (!ptr) {
        fprintf(stderr, "[CHUI-Miner] Out of memory.\n");
        abort();
    }
    return ptr;
}

static void *xrealloc(void *ptr, size_t size) {
    void *next = realloc(ptr, size);
    if (!next) {
        fprintf(stderr, "[CHUI-Miner] Out of memory.\n");
        abort();
    }
    return next;
}

static void itemvec_push(CHUI_ItemVec *vec, uint32_t item) {
    if (vec->count == vec->capacity) {
        vec->capacity = vec->capacity ? vec->capacity * 2 : 8;
        vec->items = xrealloc(vec->items, vec->capacity * sizeof(uint32_t));
    }
    vec->items[vec->count++] = item;
}

static CHUI_ItemVec itemvec_copy(const uint32_t *items, size_t count) {
    CHUI_ItemVec vec = {0};
    if (count > 0) {
        vec.items = xcalloc(count, sizeof(uint32_t));
        memcpy(vec.items, items, count * sizeof(uint32_t));
        vec.count = count;
        vec.capacity = count;
    }
    return vec;
}

static void itemvec_free(CHUI_ItemVec *vec) {
    free(vec->items);
    vec->items = NULL;
    vec->count = 0;
    vec->capacity = 0;
}

static void initial_append(CHUI_Initial_List *list, uint32_t tid, double eu, double ru) {
    if (list->count == list->capacity) {
        list->capacity = list->capacity ? list->capacity * 2 : 16;
        list->elements = xrealloc(list->elements, list->capacity * sizeof(CHUI_Element));
    }
    list->elements[list->count].tid = tid;
    list->elements[list->count].eu = eu;
    list->elements[list->count].ru = ru;
    list->count++;
    list->sum_eu += eu;
    list->sum_ru += ru;
}

static CHUI_EUList *eulist_new(size_t item_capacity, size_t elem_capacity) {
    CHUI_EUList *list = xcalloc(1, sizeof(CHUI_EUList));
    list->item_capacity = item_capacity ? item_capacity : 1;
    list->items = xcalloc(list->item_capacity, sizeof(uint32_t));
    list->capacity = elem_capacity ? elem_capacity : 1;
    list->elements = xcalloc(list->capacity, sizeof(CHUI_Element));
    return list;
}

static void eulist_append_item(CHUI_EUList *list, uint32_t item_rank) {
    if (list->item_count == list->item_capacity) {
        list->item_capacity *= 2;
        list->items = xrealloc(list->items, list->item_capacity * sizeof(uint32_t));
    }
    list->items[list->item_count++] = item_rank;
}

static void eulist_append_element(CHUI_EUList *list, uint32_t tid, double eu, double ru) {
    if (list->count == list->capacity) {
        list->capacity *= 2;
        list->elements = xrealloc(list->elements, list->capacity * sizeof(CHUI_Element));
    }
    list->elements[list->count].tid = tid;
    list->elements[list->count].eu = eu;
    list->elements[list->count].ru = ru;
    list->count++;
    list->sum_eu += eu;
    list->sum_ru += ru;
}

static CHUI_EUList *eulist_copy(const CHUI_EUList *src) {
    CHUI_EUList *dst = eulist_new(src->item_count, src->count);
    memcpy(dst->items, src->items, src->item_count * sizeof(uint32_t));
    memcpy(dst->elements, src->elements, src->count * sizeof(CHUI_Element));
    dst->item_count = src->item_count;
    dst->count = src->count;
    dst->sum_eu = src->sum_eu;
    dst->sum_ru = src->sum_ru;
    return dst;
}

static CHUI_EUList *eulist_from_initial(const CHUI_Initial_List *src) {
    CHUI_EUList *dst = eulist_new(1, src->count);
    dst->items[0] = src->item_rank;
    dst->item_count = 1;
    memcpy(dst->elements, src->elements, src->count * sizeof(CHUI_Element));
    dst->count = src->count;
    dst->sum_eu = src->sum_eu;
    dst->sum_ru = src->sum_ru;
    return dst;
}

static void eulist_free(CHUI_EUList *list) {
    if (!list) return;
    free(list->items);
    free(list->elements);
    free(list);
}

static bool tidset_subset_initial(const CHUI_EUList *subset, const CHUI_Initial_List *super) {
    if (subset->count > super->count) return false;
    size_t i = 0;
    size_t j = 0;
    while (i < subset->count && j < super->count) {
        uint32_t a = subset->elements[i].tid;
        uint32_t b = super->elements[j].tid;
        if (a == b) {
            i++;
            j++;
        } else if (a > b) {
            j++;
        } else {
            return false;
        }
    }
    return i == subset->count;
}

static bool next_initial_utility(const CHUI_Initial_List *list, uint32_t tid, size_t *cursor, double *utility) {
    while (*cursor < list->count && list->elements[*cursor].tid < tid) {
        (*cursor)++;
    }
    if (*cursor < list->count && list->elements[*cursor].tid == tid) {
        *utility = list->elements[*cursor].eu;
        return true;
    }
    return false;
}

static bool is_subsumed_by_prev(const CHUI_Context *ctx, const CHUI_EUList *list,
                                const uint32_t *prev_set, size_t prev_count) {
    for (size_t i = 0; i < prev_count; i++) {
        if (tidset_subset_initial(list, &ctx->initial[prev_set[i]])) {
            return true;
        }
    }
    return false;
}

static CHUI_EUList *construct_extension(CHUI_Context *ctx, const CHUI_EUList *prefix,
                                        uint32_t item_rank) {
    const CHUI_Initial_List *item_list = &ctx->initial[item_rank];
    ctx->constructed_lists++;

    if (!prefix) {
        return eulist_from_initial(item_list);
    }

    CHUI_EUList *out = eulist_new(prefix->item_count + 1,
                                  prefix->count < item_list->count ? prefix->count : item_list->count);
    memcpy(out->items, prefix->items, prefix->item_count * sizeof(uint32_t));
    out->item_count = prefix->item_count;
    eulist_append_item(out, item_rank);

    size_t i = 0;
    size_t j = 0;
    while (i < prefix->count && j < item_list->count) {
        uint32_t tid_x = prefix->elements[i].tid;
        uint32_t tid_i = item_list->elements[j].tid;
        if (tid_x == tid_i) {
            eulist_append_element(out, tid_x,
                                  prefix->elements[i].eu + item_list->elements[j].eu,
                                  item_list->elements[j].ru);
            i++;
            j++;
        } else if (tid_x < tid_i) {
            i++;
        } else {
            j++;
        }
    }

    return out;
}

static CHUI_EUList *compute_closure(CHUI_Context *ctx, const CHUI_EUList *base,
                                    const uint32_t *post_set, size_t post_count,
                                    uint32_t **out_post, size_t *out_post_count) {
    CHUI_EUList *closed = eulist_copy(base);
    uint32_t *next_post = NULL;
    size_t next_count = 0;
    size_t next_capacity = post_count;

    if (next_capacity > 0) {
        next_post = xcalloc(next_capacity, sizeof(uint32_t));
    }

    for (size_t i = 0; i < post_count; i++) {
        uint32_t item_rank = post_set[i];
        const CHUI_Initial_List *item_list = &ctx->initial[item_rank];
        if (tidset_subset_initial(base, item_list)) {
            eulist_append_item(closed, item_rank);
            size_t cursor = 0;
            for (size_t e = 0; e < closed->count; e++) {
                double utility = 0.0;
                if (next_initial_utility(item_list, closed->elements[e].tid, &cursor, &utility)) {
                    closed->elements[e].eu += utility;
                    closed->elements[e].ru -= utility;
                    if (closed->elements[e].ru < 0.0 && closed->elements[e].ru > -1e-9) {
                        closed->elements[e].ru = 0.0;
                    }
                    closed->sum_eu += utility;
                    closed->sum_ru -= utility;
                }
            }
            ctx->closure_extensions++;
        } else {
            next_post[next_count++] = item_rank;
        }
    }

    *out_post = next_post;
    *out_post_count = next_count;
    return closed;
}

static bool emitted_contains(const CHUI_Emitted *emitted, const uint32_t *items, size_t count) {
    for (size_t i = 0; i < emitted->count; i++) {
        if (emitted->lengths[i] == count &&
            memcmp(emitted->sets[i], items, count * sizeof(uint32_t)) == 0) {
            return true;
        }
    }
    return false;
}

static void emitted_add(CHUI_Emitted *emitted, const uint32_t *items, size_t count) {
    if (emitted->count == emitted->capacity) {
        emitted->capacity = emitted->capacity ? emitted->capacity * 2 : 64;
        emitted->sets = xrealloc(emitted->sets, emitted->capacity * sizeof(uint32_t *));
        emitted->lengths = xrealloc(emitted->lengths, emitted->capacity * sizeof(size_t));
    }
    emitted->sets[emitted->count] = xcalloc(count, sizeof(uint32_t));
    memcpy(emitted->sets[emitted->count], items, count * sizeof(uint32_t));
    emitted->lengths[emitted->count] = count;
    emitted->count++;
}

static void emitted_free(CHUI_Emitted *emitted) {
    for (size_t i = 0; i < emitted->count; i++) {
        free(emitted->sets[i]);
    }
    free(emitted->sets);
    free(emitted->lengths);
}

static void update_utility_unit_array(const CHUI_Context *ctx, const CHUI_EUList *closed,
                                      double *unit_array) {
    for (size_t i = 0; i < closed->item_count; i++) {
        const CHUI_Initial_List *item_list = &ctx->initial[closed->items[i]];
        size_t cursor = 0;
        double sum = 0.0;
        for (size_t e = 0; e < closed->count; e++) {
            double utility = 0.0;
            if (next_initial_utility(item_list, closed->elements[e].tid, &cursor, &utility)) {
                sum += utility;
            }
        }
        unit_array[i] = sum;
    }
}

static void emit_if_chui(CHUI_Context *ctx, const CHUI_EUList *closed) {
    if (closed->sum_eu + 1e-9 < ctx->min_utility) return;
    if (emitted_contains(&ctx->emitted, closed->items, closed->item_count)) return;

    double *unit_array = xcalloc(closed->item_count ? closed->item_count : 1, sizeof(double));
    update_utility_unit_array(ctx, closed, unit_array);
    free(unit_array);

    FILE *f = fopen("chui_results.txt", "a");
    if (f) {
        uint32_t *sorted_items = malloc(sizeof(uint32_t) * (closed->item_count ? closed->item_count : 1));
        if (sorted_items && closed->item_count > 0) {
            for (size_t i = 0; i < closed->item_count; i++) {
                sorted_items[i] = ctx->promising_ids[closed->items[i]];
            }
            for (size_t i = 0; i < closed->item_count; i++) {
                for (size_t j = i + 1; j < closed->item_count; j++) {
                    if (sorted_items[i] > sorted_items[j]) {
                        uint32_t tmp = sorted_items[i];
                        sorted_items[i] = sorted_items[j];
                        sorted_items[j] = tmp;
                    }
                }
            }
            for (size_t i = 0; i < closed->item_count; i++) {
                fprintf(f, "%u ", sorted_items[i]);
            }
            fprintf(f, "#UTIL: %.2f\n", closed->sum_eu);
        }
        free(sorted_items);
        fclose(f);
    }

    emitted_add(&ctx->emitted, closed->items, closed->item_count);
    ctx->closed_count++;
    ctx->total_closed_items += closed->item_count;
}

static void gen_chui(CHUI_Context *ctx, const CHUI_EUList *prefix,
                     const uint32_t *prev_set, size_t prev_count,
                     const uint32_t *post_set, size_t post_count) {
    CHUI_ItemVec local_prev = itemvec_copy(prev_set, prev_count);

    for (size_t i = 0; i < post_count; i++) {
        uint32_t item_rank = post_set[i];
        CHUI_EUList *candidate = construct_extension(ctx, prefix, item_rank);

        if (candidate->sum_eu + candidate->sum_ru + 1e-9 >= ctx->min_utility) {
            if (!is_subsumed_by_prev(ctx, candidate, local_prev.items, local_prev.count)) {
                uint32_t *closed_post = NULL;
                size_t closed_post_count = 0;
                CHUI_EUList *closed = compute_closure(ctx, candidate, post_set + i + 1,
                                                       post_count - i - 1,
                                                       &closed_post, &closed_post_count);
                emit_if_chui(ctx, closed);
                if (closed_post_count > 0) {
                    gen_chui(ctx, closed, local_prev.items, local_prev.count,
                             closed_post, closed_post_count);
                }
                free(closed_post);
                eulist_free(closed);
            } else {
                ctx->pruned_by_subsumption++;
            }
        } else {
            ctx->pruned_by_ru++;
        }

        eulist_free(candidate);
        itemvec_push(&local_prev, item_rank);
    }

    itemvec_free(&local_prev);
}


static void build_initial_lists(CHUI_Context *ctx) {
    ctx->initial = xcalloc(ctx->promising_count, sizeof(CHUI_Initial_List));
    for (size_t i = 0; i < ctx->promising_count; i++) {
        ctx->initial[i].item_rank = (uint32_t)i;
        ctx->initial[i].item_id = ctx->promising_ids[i];
    }

    for (size_t t = 0; t < ctx->dataset->count; t++) {
        const DM_Trans_Utility *tr = &ctx->transactions[t];
        uint32_t (*ranked)[2] = NULL;
        double *utilities = NULL;
        size_t kept = 0;

        if (tr->count > 0) {
            ranked = xcalloc(tr->count, sizeof(uint32_t[2]));
            utilities = xcalloc(tr->count, sizeof(double));
        }

        for (size_t i = 0; i < tr->count; i++) {
            uint32_t item = tr->items[i].id;
            if (item <= ctx->dataset->max_id && ctx->rank_of_item[item] != UINT32_MAX) {
                ranked[kept][0] = ctx->rank_of_item[item];
                ranked[kept][1] = item;
                utilities[kept] = tr->items[i].utility;
                kept++;
            }
        }

        for (size_t i = 1; i < kept; i++) {
            uint32_t rank = ranked[i][0];
            uint32_t item = ranked[i][1];
            double utility = utilities[i];
            size_t j = i;
            while (j > 0 && ranked[j - 1][0] > rank) {
                ranked[j][0] = ranked[j - 1][0];
                ranked[j][1] = ranked[j - 1][1];
                utilities[j] = utilities[j - 1];
                j--;
            }
            ranked[j][0] = rank;
            ranked[j][1] = item;
            utilities[j] = utility;
        }

        double remaining_utility = 0.0;
        for (size_t i = kept; i > 0; i--) {
            size_t pos = i - 1;
            uint32_t rank = ranked[pos][0];
            initial_append(&ctx->initial[rank], (uint32_t)t, utilities[pos], remaining_utility);
            remaining_utility += utilities[pos];
        }

        free(ranked);
        free(utilities);
    }
}

static void free_context(CHUI_Context *ctx) {
    if (ctx->initial) {
        for (size_t i = 0; i < ctx->promising_count; i++) {
            free(ctx->initial[i].elements);
        }
        free(ctx->initial);
    }
    free(ctx->promising_ids);
    free(ctx->rank_of_item);
    emitted_free(&ctx->emitted);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;

    DM_CHUI_Miner_Params *p = (DM_CHUI_Miner_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    DM_Trans_Utility *transactions = (DM_Trans_Utility *)ds->payload;

    FILE *f_init = fopen("chui_results.txt", "w");
    if (f_init) fclose(f_init);

    CHUI_Context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.dataset = ds;
    ctx.transactions = transactions;
    ctx.min_utility = min_util;

    double *twu = xcalloc((size_t)ds->max_id + 1, sizeof(double));
    for (size_t t = 0; t < ds->count; t++) {
        for (size_t i = 0; i < transactions[t].count; i++) {
            uint32_t item = transactions[t].items[i].id;
            if (item <= ds->max_id) {
                twu[item] += transactions[t].total_utility;
            }
        }
    }

    ctx.promising_ids = xcalloc((size_t)ds->max_id + 1, sizeof(uint32_t));
    for (uint32_t item = 0; item <= ds->max_id; item++) {
        if (twu[item] + 1e-9 >= min_util) {
            ctx.promising_ids[ctx.promising_count++] = item;
        }
    }

    g_sort_twu = twu;
    qsort(ctx.promising_ids, ctx.promising_count, sizeof(uint32_t), compare_item_by_twu_desc);
    g_sort_twu = NULL;

    ctx.rank_of_item = xcalloc((size_t)ds->max_id + 1, sizeof(uint32_t));
    for (uint32_t item = 0; item <= ds->max_id; item++) {
        ctx.rank_of_item[item] = UINT32_MAX;
    }
    for (size_t rank = 0; rank < ctx.promising_count; rank++) {
        ctx.rank_of_item[ctx.promising_ids[rank]] = (uint32_t)rank;
    }

    build_initial_lists(&ctx);

    uint32_t *post_set = NULL;
    if (ctx.promising_count > 0) {
        post_set = xcalloc(ctx.promising_count, sizeof(uint32_t));
    }
    for (size_t i = 0; i < ctx.promising_count; i++) {
        post_set[i] = (uint32_t)i;
    }

    gen_chui(&ctx, NULL, NULL, 0, post_set, ctx.promising_count);

    printf("[CHUI-Miner] MinUtil: %.2f\n", min_util);
    printf("[CHUI-Miner] Promising items: %zu\n", ctx.promising_count);
    printf("[CHUI-Miner] Constructed EU-lists: %zu\n", ctx.constructed_lists);
    printf("[CHUI-Miner] Closure extensions: %zu\n", ctx.closure_extensions);
    printf("[CHUI-Miner] Pruned by SumEU+SumRU: %zu\n", ctx.pruned_by_ru);
    printf("[CHUI-Miner] Pruned by subsumption: %zu\n", ctx.pruned_by_subsumption);
    printf("[CHUI-Miner] Found %zu Closed High Utility Itemsets.\n", ctx.closed_count);

    dm_bench_record_results(ctx.closed_count, ctx.total_closed_items);

    free(post_set);
    free(twu);
    free_context(&ctx);
    return DM_SUCCESS;
}

static DM_Algorithm chui_miner_algo = {
    .id = "chui_miner",
    .name = "CHUI-Miner",
    .description = "Mining Closed High Utility Itemsets without Candidate Generation using Extended Utility-Lists.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(chui_miner_algo)
