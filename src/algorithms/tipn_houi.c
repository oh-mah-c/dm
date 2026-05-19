#include "algorithms/tipn_houi.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint32_t id;
    double util;
} ItemUtil;

typedef struct {
    ItemUtil *items;
    size_t count;
    double total;
    size_t interval;
} Trans;

typedef struct {
    Trans *trans;
    size_t count;
    size_t cap;
    uint32_t max_item;
} DB;

typedef struct {
    uint32_t item;
    double twugc;
    int positive;
} OrderItem;

typedef struct {
    uint32_t *items;
    size_t len;
    double relu;
} TopItemset;

typedef struct {
    TopItemset *data;
    size_t count;
    size_t cap;
    size_t k;
    double threshold;
    size_t raises;
} TopK;

typedef struct {
    DB db;
    TIPNHouiParams params;
    TIPNHouiStats *stats;
    double *total_ti;
    double *twugc;
    double *item_exact_utility;
    double *item_rel;
    unsigned char *positive;
    uint32_t *rank_to_item;
    uint32_t *item_to_rank;
    size_t item_count;
    clock_t start_clock;
} Ctx;

static double elapsed_sec(Ctx *ctx) {
    return (double)(clock() - ctx->start_clock) / (double)CLOCKS_PER_SEC;
}

static int should_stop(Ctx *ctx) {
    if (ctx->params.max_seconds > 0.0 && elapsed_sec(ctx) >= ctx->params.max_seconds) {
        ctx->stats->limited = 1;
        return 1;
    }
    if (ctx->params.max_candidates > 0 && ctx->stats->candidates >= ctx->params.max_candidates) {
        ctx->stats->limited = 1;
        return 1;
    }
    return 0;
}

static void db_free(DB *db) {
    for (size_t i = 0; i < db->count; i++) free(db->trans[i].items);
    free(db->trans);
    memset(db, 0, sizeof(*db));
}

static int db_add(DB *db, ItemUtil *items, size_t count, double total, size_t interval) {
    if (count == 0) {
        free(items);
        return 0;
    }
    if (db->count == db->cap) {
        size_t nc = db->cap ? db->cap * 2 : 1024;
        Trans *nt = (Trans *)realloc(db->trans, nc * sizeof(*nt));
        if (!nt) {
            free(items);
            return -1;
        }
        db->trans = nt;
        db->cap = nc;
    }
    db->trans[db->count].items = items;
    db->trans[db->count].count = count;
    db->trans[db->count].total = total;
    db->trans[db->count].interval = interval;
    for (size_t i = 0; i < count; i++) if (items[i].id > db->max_item) db->max_item = items[i].id;
    db->count++;
    return 0;
}

static int parse_line(char *line, size_t tid, const TIPNHouiParams *params, DB *db) {
    char *colon1 = strchr(line, ':');
    char *colon2 = colon1 ? strchr(colon1 + 1, ':') : NULL;
    if (!colon1 || !colon2) return 0;
    *colon1 = '\0';
    *colon2 = '\0';
    char *items_s = line;
    char *total_s = colon1 + 1;
    char *utils_s = colon2 + 1;
    char *interval_s = strchr(utils_s, ':');
    size_t interval = params->intervals ? (tid % params->intervals) : 0;
    if (interval_s) {
        *interval_s = '\0';
        long v = strtol(interval_s + 1, NULL, 10);
        if (v > 0) interval = (size_t)(v - 1);
        else if (v >= 0) interval = (size_t)v;
    }
    uint32_t *ids = NULL;
    double *utils = NULL;
    size_t id_count = 0, util_count = 0, cap = 0;
    char *p = items_s;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (id_count == cap) {
            size_t nc = cap ? cap * 2 : 16;
            uint32_t *ni = (uint32_t *)realloc(ids, nc * sizeof(*ids));
            if (!ni) {
                free(ids); free(utils);
                return -1;
            }
            ids = ni;
            double *nu = (double *)realloc(utils, nc * sizeof(*utils));
            if (!nu) {
                free(ids); free(utils);
                return -1;
            }
            utils = nu; cap = nc;
        }
        ids[id_count++] = (uint32_t)v;
        p = end;
    }
    p = utils_s;
    while (*p && util_count < id_count) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        char *end = NULL;
        double v = strtod(p, &end);
        if (end == p) break;
        utils[util_count++] = v;
        p = end;
    }
    if (id_count == 0 || util_count != id_count) {
        free(ids); free(utils);
        return 0;
    }
    ItemUtil *items = (ItemUtil *)malloc(id_count * sizeof(*items));
    if (!items) {
        free(ids); free(utils);
        return -1;
    }
    for (size_t i = 0; i < id_count; i++) {
        items[i].id = ids[i];
        items[i].util = utils[i];
    }
    double total = atof(total_s);
    free(ids); free(utils);
    return db_add(db, items, id_count, total, interval);
}

static int load_db(const char *path, const TIPNHouiParams *params, DB *db) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char *line = NULL;
    size_t n = 0, tid = 0;
    while (getline(&line, &n, fp) != -1) {
        if (params->max_transactions && tid >= params->max_transactions) break;
        if (line[0] == '@' || line[0] == '#' || line[0] == '%' || line[0] == '\n') continue;
        if (parse_line(line, tid, params, db) != 0) {
            free(line);
            fclose(fp);
            return -1;
        }
        tid++;
    }
    free(line);
    fclose(fp);
    return db->count ? 0 : -1;
}

static int contains_item(const Trans *t, uint32_t item, double *util) {
    for (size_t i = 0; i < t->count; i++) {
        if (t->items[i].id == item) {
            if (util) *util = t->items[i].util;
            return 1;
        }
    }
    return 0;
}

static int trans_contains_itemset(const Trans *t, const uint32_t *items, size_t len, double *util_sum) {
    double sum = 0.0;
    for (size_t i = 0; i < len; i++) {
        double u = 0.0;
        if (!contains_item(t, items[i], &u)) return 0;
        sum += u;
    }
    if (util_sum) *util_sum = sum;
    return 1;
}

static double interval_denominator(Ctx *ctx, const unsigned char *seen) {
    double d = 0.0;
    for (size_t i = 0; i < ctx->params.intervals; i++) {
        if (seen[i]) d += ctx->total_ti[i];
    }
    return d;
}

static double exact_relative(Ctx *ctx, const uint32_t *items, size_t len, double *utility_out, unsigned char *interval_seen) {
    memset(interval_seen, 0, ctx->params.intervals);
    double util = 0.0;
    for (size_t t = 0; t < ctx->db.count; t++) {
        double u = 0.0;
        if (trans_contains_itemset(&ctx->db.trans[t], items, len, &u)) {
            util += u;
            if (ctx->db.trans[t].interval < ctx->params.intervals) interval_seen[ctx->db.trans[t].interval] = 1;
        }
    }
    double denom = interval_denominator(ctx, interval_seen);
    if (utility_out) *utility_out = util;
    if (denom <= 0.0) return -INFINITY;
    return util / denom;
}

static int cmp_order_item(const void *a, const void *b) {
    const OrderItem *x = (const OrderItem *)a;
    const OrderItem *y = (const OrderItem *)b;
    if (x->positive != y->positive) return y->positive - x->positive;
    if (x->twugc < y->twugc) return 1;
    if (x->twugc > y->twugc) return -1;
    return (x->item > y->item) - (x->item < y->item);
}

static int cmp_double_desc(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x < y) - (x > y);
}

static int topk_same(const TopItemset *p, const uint32_t *items, size_t len) {
    if (p->len != len) return 0;
    for (size_t i = 0; i < len; i++) if (p->items[i] != items[i]) return 0;
    return 1;
}

static void topk_free(TopK *tk) {
    for (size_t i = 0; i < tk->count; i++) free(tk->data[i].items);
    free(tk->data);
    memset(tk, 0, sizeof(*tk));
}

static int topk_cmp(const void *a, const void *b) {
    const TopItemset *x = (const TopItemset *)a;
    const TopItemset *y = (const TopItemset *)b;
    if (x->relu < y->relu) return 1;
    if (x->relu > y->relu) return -1;
    return (x->len > y->len) - (x->len < y->len);
}

static int topk_add(TopK *tk, const uint32_t *items, size_t len, double relu) {
    if (tk->k == 0) return 0;
    if (tk->count >= tk->k && relu <= tk->threshold) return 0;
    for (size_t i = 0; i < tk->count; i++) if (topk_same(&tk->data[i], items, len)) return 0;
    if (tk->count == tk->cap) {
        size_t nc = tk->cap ? tk->cap * 2 : 64;
        TopItemset *nd = (TopItemset *)realloc(tk->data, nc * sizeof(*nd));
        if (!nd) return -1;
        tk->data = nd;
        tk->cap = nc;
    }
    tk->data[tk->count].items = (uint32_t *)malloc(len * sizeof(uint32_t));
    if (!tk->data[tk->count].items) return -1;
    memcpy(tk->data[tk->count].items, items, len * sizeof(uint32_t));
    tk->data[tk->count].len = len;
    tk->data[tk->count].relu = relu;
    tk->count++;
    qsort(tk->data, tk->count, sizeof(*tk->data), topk_cmp);
    if (tk->count >= tk->k) {
        double old = tk->threshold;
        tk->threshold = tk->data[tk->k - 1].relu;
        if (tk->threshold > old) tk->raises++;
        while (tk->count > tk->k && tk->data[tk->count - 1].relu < tk->threshold) {
            free(tk->data[tk->count - 1].items);
            tk->count--;
        }
    }
    return 0;
}

static double upper_twugc(Ctx *ctx, const uint32_t *items, size_t len, unsigned char *interval_seen) {
    memset(interval_seen, 0, ctx->params.intervals);
    double ub = 0.0;
    for (size_t t = 0; t < ctx->db.count; t++) {
        if (trans_contains_itemset(&ctx->db.trans[t], items, len, NULL)) {
            ub += ctx->db.trans[t].total;
            if (ctx->db.trans[t].interval < ctx->params.intervals) interval_seen[ctx->db.trans[t].interval] = 1;
        }
    }
    double denom = interval_denominator(ctx, interval_seen);
    return denom > 0.0 ? ub / denom : -INFINITY;
}

static int any_occurs(Ctx *ctx, const uint32_t *items, size_t len) {
    for (size_t t = 0; t < ctx->db.count; t++) if (trans_contains_itemset(&ctx->db.trans[t], items, len, NULL)) return 1;
    return 0;
}

static int mine_rec(Ctx *ctx, TopK *tk, uint32_t *prefix, size_t plen, size_t start_rank, unsigned char *interval_seen) {
    if (should_stop(ctx)) return 0;
    if (ctx->params.max_depth > 0 && plen >= ctx->params.max_depth) return 0;
    for (size_t r = start_rank; r < ctx->item_count; r++) {
        prefix[plen] = ctx->rank_to_item[r];
        size_t len = plen + 1;
        ctx->stats->candidates++;
        if (!any_occurs(ctx, prefix, len)) {
            ctx->stats->pruned_tio++;
            continue;
        }
        double twu_rel = upper_twugc(ctx, prefix, len, interval_seen);
        if (twu_rel < tk->threshold) {
            ctx->stats->pruned_twugc++;
            continue;
        }
        double util = 0.0;
        double rel = exact_relative(ctx, prefix, len, &util, interval_seen);
        if (util < 0.0) {
            ctx->stats->pruned_rlc++;
            continue;
        }
        if (rel >= tk->threshold && topk_add(tk, prefix, len, rel) != 0) return -1;
        ctx->stats->joins++;
        if (mine_rec(ctx, tk, prefix, len, r + 1, interval_seen) != 0) return -1;
    }
    return 0;
}

TIPNHouiParams tipn_houi_default_params(void) {
    TIPNHouiParams p;
    p.k = 50;
    p.intervals = 5;
    p.max_depth = 4;
    p.max_transactions = 0;
    p.max_items = 0;
    p.max_candidates = 500000;
    p.max_seconds = 60.0;
    return p;
}

int tipn_houi_mine_file(const char *path, const TIPNHouiParams *params_in, TIPNHouiStats *stats) {
    if (!path || !stats) return -1;
    Ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.params = params_in ? *params_in : tipn_houi_default_params();
    if (ctx.params.k == 0) ctx.params.k = 50;
    if (ctx.params.intervals == 0) ctx.params.intervals = 5;
    ctx.stats = stats;
    ctx.start_clock = clock();
    memset(stats, 0, sizeof(*stats));
    if (load_db(path, &ctx.params, &ctx.db) != 0) return -1;
    ctx.total_ti = (double *)calloc(ctx.params.intervals, sizeof(double));
    ctx.twugc = (double *)calloc(ctx.db.max_item + 1, sizeof(double));
    ctx.item_exact_utility = (double *)calloc(ctx.db.max_item + 1, sizeof(double));
    ctx.item_rel = (double *)calloc(ctx.db.max_item + 1, sizeof(double));
    ctx.positive = (unsigned char *)calloc(ctx.db.max_item + 1, 1);
    if (!ctx.total_ti || !ctx.twugc || !ctx.item_exact_utility || !ctx.item_rel || !ctx.positive) {
        db_free(&ctx.db); free(ctx.total_ti); free(ctx.twugc); free(ctx.item_exact_utility); free(ctx.item_rel); free(ctx.positive);
        return -1;
    }
    for (size_t t = 0; t < ctx.db.count; t++) {
        if (ctx.db.trans[t].interval >= ctx.params.intervals) ctx.db.trans[t].interval %= ctx.params.intervals;
        ctx.total_ti[ctx.db.trans[t].interval] += ctx.db.trans[t].total;
        for (size_t i = 0; i < ctx.db.trans[t].count; i++) {
            ItemUtil it = ctx.db.trans[t].items[i];
            ctx.item_exact_utility[it.id] += it.util;
        }
    }
    for (uint32_t id = 0; id <= ctx.db.max_item; id++) {
        if (ctx.item_exact_utility[id] >= 0.0) ctx.positive[id] = 1;
    }
    for (size_t t = 0; t < ctx.db.count; t++) {
        for (size_t i = 0; i < ctx.db.trans[t].count; i++) {
            ctx.twugc[ctx.db.trans[t].items[i].id] += ctx.db.trans[t].total;
        }
    }
    OrderItem *order = (OrderItem *)malloc((ctx.db.max_item + 1) * sizeof(*order));
    unsigned char *interval_seen = (unsigned char *)malloc(ctx.params.intervals);
    if (!order || !interval_seen) {
        free(order); free(interval_seen); db_free(&ctx.db); free(ctx.total_ti); free(ctx.twugc); free(ctx.item_exact_utility); free(ctx.item_rel); free(ctx.positive);
        return -1;
    }
    for (uint32_t id = 0; id <= ctx.db.max_item; id++) {
        if (ctx.twugc[id] == 0.0) continue;
        uint32_t item = id;
        double util = 0.0;
        double rel = exact_relative(&ctx, &item, 1, &util, interval_seen);
        ctx.item_rel[id] = rel;
        order[ctx.item_count].item = id;
        order[ctx.item_count].twugc = ctx.twugc[id];
        order[ctx.item_count].positive = ctx.positive[id] ? 1 : 0;
        ctx.item_count++;
    }
    qsort(order, ctx.item_count, sizeof(*order), cmp_order_item);
    if (ctx.params.max_items > 0 && ctx.item_count > ctx.params.max_items) ctx.item_count = ctx.params.max_items;
    ctx.rank_to_item = (uint32_t *)malloc(ctx.item_count * sizeof(uint32_t));
    ctx.item_to_rank = (uint32_t *)malloc((ctx.db.max_item + 1) * sizeof(uint32_t));
    if (!ctx.rank_to_item || !ctx.item_to_rank) {
        free(order); free(interval_seen); db_free(&ctx.db); free(ctx.total_ti); free(ctx.twugc); free(ctx.item_exact_utility); free(ctx.item_rel); free(ctx.positive); free(ctx.rank_to_item); free(ctx.item_to_rank);
        return -1;
    }
    for (uint32_t id = 0; id <= ctx.db.max_item; id++) ctx.item_to_rank[id] = UINT32_MAX;
    for (size_t r = 0; r < ctx.item_count; r++) {
        ctx.rank_to_item[r] = order[r].item;
        ctx.item_to_rank[order[r].item] = (uint32_t)r;
        if (order[r].positive) stats->positive_items++;
        else stats->negative_items++;
    }
    TopK tk = {0};
    tk.k = ctx.params.k;
    double *rpru = (double *)malloc(stats->positive_items * sizeof(double));
    size_t rp = 0;
    for (size_t r = 0; r < ctx.item_count; r++) {
        uint32_t item = ctx.rank_to_item[r];
        if (ctx.positive[item] && ctx.item_rel[item] > -INFINITY) {
            rpru[rp++] = ctx.item_rel[item];
            topk_add(&tk, &item, 1, ctx.item_rel[item]);
        }
    }
    qsort(rpru, rp, sizeof(double), cmp_double_desc);
    if (rp >= ctx.params.k) {
        tk.threshold = rpru[ctx.params.k - 1];
        stats->rpru_size1_threshold = tk.threshold;
    }
    double *rru2 = (double *)malloc(ctx.item_count * ctx.item_count * sizeof(double));
    size_t rru_count = 0;
    uint32_t pair[2];
    for (size_t i = 0; i < ctx.item_count; i++) {
        for (size_t j = i + 1; j < ctx.item_count; j++) {
            pair[0] = ctx.rank_to_item[i];
            pair[1] = ctx.rank_to_item[j];
            double util = 0.0;
            double rel = exact_relative(&ctx, pair, 2, &util, interval_seen);
            if (rel > -INFINITY && util >= 0.0) {
                rru2[rru_count++] = rel;
                topk_add(&tk, pair, 2, rel);
            }
        }
    }
    qsort(rru2, rru_count, sizeof(double), cmp_double_desc);
    if (rru_count >= ctx.params.k && rru2[ctx.params.k - 1] > tk.threshold) tk.threshold = rru2[ctx.params.k - 1];
    stats->rru_size2_threshold = rru_count >= ctx.params.k ? rru2[ctx.params.k - 1] : 0.0;
    uint32_t *prefix = (uint32_t *)malloc((ctx.params.max_depth ? ctx.params.max_depth : ctx.item_count) * sizeof(uint32_t));
    if (!prefix || mine_rec(&ctx, &tk, prefix, 0, 0, interval_seen) != 0) {
        free(prefix); free(rpru); free(rru2); topk_free(&tk); free(order); free(interval_seen); db_free(&ctx.db);
        free(ctx.total_ti); free(ctx.twugc); free(ctx.item_exact_utility); free(ctx.item_rel); free(ctx.positive); free(ctx.rank_to_item); free(ctx.item_to_rank);
        return -1;
    }
    stats->transactions = ctx.db.count;
    stats->intervals = ctx.params.intervals;
    stats->distinct_items = ctx.item_count;
    stats->k = ctx.params.k;
    stats->threshold = tk.threshold;
    stats->threshold_raises = tk.raises;
    stats->output_count = tk.count;
    for (size_t i = 0; i < tk.count; i++) {
        if (i == 0 || tk.data[i].relu > stats->best_relative_utility) stats->best_relative_utility = tk.data[i].relu;
        stats->avg_relative_utility += tk.data[i].relu;
        stats->avg_itemset_length += (double)tk.data[i].len;
        stats->result_ram_bytes += sizeof(TopItemset) + tk.data[i].len * sizeof(uint32_t);
        stats->result_disk_est_bytes += 32 + tk.data[i].len * 12;
    }
    if (tk.count) {
        stats->avg_relative_utility /= (double)tk.count;
        stats->avg_itemset_length /= (double)tk.count;
    }
    free(prefix); free(rpru); free(rru2); topk_free(&tk); free(order); free(interval_seen); db_free(&ctx.db);
    free(ctx.total_ti); free(ctx.twugc); free(ctx.item_exact_utility); free(ctx.item_rel); free(ctx.positive); free(ctx.rank_to_item); free(ctx.item_to_rank);
    return 0;
}
