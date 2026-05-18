#include "algorithms/mheinu.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t tid;
    double u;
    double pub;
} ELNUEntry;

typedef struct {
    uint32_t *items;
    size_t item_count;
    size_t last_pos;
    ELNUEntry *entries;
    size_t entry_count;
    size_t entry_capacity;
    double u;
    double pub;
    double inv;
} ELNU;

typedef struct {
    ELNU **lists;
    size_t count;
    double min_e;
    size_t hei_count;
    size_t total_items;
    size_t join_count;
    size_t pruned_uben;
    size_t pruned_ubeni;
} MHEINUContext;

static double rand_unit(unsigned int *state) {
    *state = (*state * 1103515245u) + 12345u;
    return ((double)((*state / 65536u) % 32768u) + 0.5) / 32768.0;
}

static double rand_normal(unsigned int *state, double mean, double stddev) {
    double u1 = rand_unit(state);
    double u2 = rand_unit(state);
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
    return mean + stddev * z;
}

static void generate_paper_investments(double *inv, size_t count) {
    unsigned int state = 42u;
    for (size_t i = 0; i < count; i++) {
        double value = rand_normal(&state, 10000.0, 10.0);
        if (value <= 0.0) {
            do {
                value = rand_normal(&state, 100.0, 5.0);
            } while (value <= 0.0);
        }
        inv[i] = value;
    }
}

static int load_investments(const char *path, double *inv, size_t count) {
    if (!path || !path[0]) return 0;
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    char line[4096];
    size_t sequential = 0;
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#' || *p == '@' || *p == '%') continue;

        char *end = NULL;
        unsigned long id = strtoul(p, &end, 10);
        while (end && isspace((unsigned char)*end)) end++;
        if (end && (*end == ':' || *end == ',' || *end == '=')) {
            double value = atof(end + 1);
            if (id < count && value > 0.0) inv[id] = value;
        } else {
            double value = atof(p);
            if (sequential < count && value > 0.0) inv[sequential++] = value;
        }
    }

    fclose(f);
    return 1;
}

static void elnu_entry_add(ELNU *list, uint32_t tid, double u, double pub) {
    if (list->entry_count >= list->entry_capacity) {
        list->entry_capacity = list->entry_capacity ? list->entry_capacity * 2 : 8;
        list->entries = realloc(list->entries, list->entry_capacity * sizeof(ELNUEntry));
    }
    list->entries[list->entry_count].tid = tid;
    list->entries[list->entry_count].u = u;
    list->entries[list->entry_count].pub = pub;
    list->entry_count++;
    list->u += u;
    list->pub += pub;
}

static ELNU *elnu_new_single(uint32_t item, double inv) {
    ELNU *list = calloc(1, sizeof(ELNU));
    if (!list) return NULL;
    list->items = malloc(sizeof(uint32_t));
    if (!list->items) {
        free(list);
        return NULL;
    }
    list->items[0] = item;
    list->item_count = 1;
    list->last_pos = 0;
    list->inv = inv;
    return list;
}

static void elnu_free(ELNU *list) {
    if (!list) return;
    free(list->items);
    free(list->entries);
    free(list);
}

static double efficiency(const ELNU *list) {
    return list->inv > 0.0 ? list->u / list->inv : -INFINITY;
}

static double uben(const ELNU *list) {
    return list->inv > 0.0 ? list->pub / list->inv : -INFINITY;
}

static double ubeni(const ELNU *x, const ELNU *y) {
    double denom = x->inv + y->inv;
    return denom > 0.0 ? fmin(x->pub, y->pub) / denom : -INFINITY;
}

static int cmp_elnu_uben(const void *a, const void *b) {
    const ELNU *la = *(ELNU * const *)a;
    const ELNU *lb = *(ELNU * const *)b;
    double ua = uben(la);
    double ub = uben(lb);
    if (ua < ub) return -1;
    if (ua > ub) return 1;
    if (la->items[0] < lb->items[0]) return -1;
    if (la->items[0] > lb->items[0]) return 1;
    return 0;
}

static ELNU *construct_elnu(const ELNU *x, const ELNU *y) {
    ELNU *xy = calloc(1, sizeof(ELNU));
    if (!xy) return NULL;

    xy->item_count = x->item_count + 1;
    xy->items = malloc(xy->item_count * sizeof(uint32_t));
    if (!xy->items) {
        free(xy);
        return NULL;
    }
    memcpy(xy->items, x->items, x->item_count * sizeof(uint32_t));
    xy->items[x->item_count] = y->items[0];
    xy->last_pos = y->last_pos;
    xy->inv = x->inv + y->inv;

    size_t i = 0;
    size_t j = 0;
    while (i < x->entry_count && j < y->entry_count) {
        if (x->entries[i].tid == y->entries[j].tid) {
            elnu_entry_add(xy, x->entries[i].tid,
                           x->entries[i].u + y->entries[j].u,
                           fmin(x->entries[i].pub, y->entries[j].pub));
            i++;
            j++;
        } else if (x->entries[i].tid < y->entries[j].tid) {
            i++;
        } else {
            j++;
        }
    }

    return xy;
}

static void output_if_hei(MHEINUContext *ctx, const ELNU *list) {
    if (efficiency(list) >= ctx->min_e) {
        ctx->hei_count++;
        ctx->total_items += list->item_count;
    }
}

static void search(MHEINUContext *ctx, ELNU *x) {
    for (size_t i = x->last_pos + 1; i < ctx->count; i++) {
        ELNU *y = ctx->lists[i];
        if (ubeni(x, y) < ctx->min_e) {
            ctx->pruned_ubeni++;
            continue;
        }

        ELNU *xy = construct_elnu(x, y);
        ctx->join_count++;
        if (!xy) continue;

        if (xy->entry_count > 0 && uben(xy) >= ctx->min_e) {
            output_if_hei(ctx, xy);
            search(ctx, xy);
        } else {
            ctx->pruned_uben++;
        }
        elnu_free(xy);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_UTILITY || !ds->payload || ds->count == 0) {
        fprintf(stderr, "[MHEINU] Utility dataset required. Use type_id 1.\n");
        return DM_ERROR_INVALID_PARAM;
    }

    DM_MHEINU_Params *p = (DM_MHEINU_Params *)params;
    double min_e = p ? p->min_efficiency : 1.0;
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    size_t item_slots = (size_t)ds->max_id + 1;

    double *investments = calloc(item_slots, sizeof(double));
    ELNU **by_id = calloc(item_slots, sizeof(ELNU *));
    if (!investments || !by_id) {
        free(investments);
        free(by_id);
        return DM_ERROR_MEMORY;
    }

    generate_paper_investments(investments, item_slots);
    int loaded_inv = load_investments(p ? p->investment_path : NULL, investments, item_slots);

    for (size_t tid = 0; tid < ds->count; tid++) {
        double pu = 0.0;
        for (size_t j = 0; j < data[tid].count; j++) {
            if (data[tid].items[j].utility >= 0.0) pu += data[tid].items[j].utility;
        }

        for (size_t j = 0; j < data[tid].count; j++) {
            uint32_t item = data[tid].items[j].id;
            double u = data[tid].items[j].utility;
            double pub = (u >= 0.0) ? pu : fmax(pu + u, 0.0);

            if (!by_id[item]) {
                by_id[item] = elnu_new_single(item, investments[item]);
                if (!by_id[item]) {
                    free(investments);
                    free(by_id);
                    return DM_ERROR_MEMORY;
                }
            }
            elnu_entry_add(by_id[item], (uint32_t)tid, u, pub);
        }
    }

    size_t list_count = 0;
    for (size_t i = 0; i < item_slots; i++) {
        if (by_id[i] && by_id[i]->entry_count > 0) list_count++;
    }

    ELNU **lists = malloc(list_count * sizeof(ELNU *));
    if (!lists) {
        for (size_t i = 0; i < item_slots; i++) elnu_free(by_id[i]);
        free(investments);
        free(by_id);
        return DM_ERROR_MEMORY;
    }

    size_t pos = 0;
    for (size_t i = 0; i < item_slots; i++) {
        if (by_id[i] && by_id[i]->entry_count > 0) lists[pos++] = by_id[i];
    }
    qsort(lists, list_count, sizeof(ELNU *), cmp_elnu_uben);
    for (size_t i = 0; i < list_count; i++) lists[i]->last_pos = i;

    MHEINUContext ctx = {
        .lists = lists,
        .count = list_count,
        .min_e = min_e
    };

    printf("[MHEINU] Starting on %zu transactions, %zu items. minE=%.6g, investments=%s\n",
           ds->count, list_count, min_e, loaded_inv ? "file" : "paper-style generated seed=42");

    for (size_t i = 0; i < list_count; i++) {
        ELNU *x = lists[i];
        if (uben(x) >= min_e) {
            output_if_hei(&ctx, x);
            search(&ctx, x);
        } else {
            ctx.pruned_uben++;
        }
    }

    printf("[MHEINU] Complete. HEIs=%zu, joins=%zu, pruned_uben=%zu, pruned_ubeni=%zu\n",
           ctx.hei_count, ctx.join_count, ctx.pruned_uben, ctx.pruned_ubeni);

    dm_bench_record_results(ctx.hei_count, ctx.total_items);

    for (size_t i = 0; i < list_count; i++) elnu_free(lists[i]);
    free(lists);
    free(by_id);
    free(investments);
    return DM_SUCCESS;
}

DM_Algorithm mheinu_algo = {
    .id = "mheinu",
    .name = "MHEINU",
    .description = "Mining high-efficiency itemsets with negative utilities using ELNU, uben, and ubeni pruning.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(mheinu_algo)
