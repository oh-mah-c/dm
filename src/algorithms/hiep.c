#include "algorithms/hiep.h"
#include "tokenizer/tokenizer.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HIEP_NO_RANK UINT32_MAX
#define HIEP_SIGNAL_COUNT 4
#define HIEP_SIGNAL_LEN 3

typedef struct {
    char *name;
    uint32_t df;
    uint32_t rank;
    double weight;
    double tiub;
    unsigned char survives;
} HIEPToken;

typedef struct {
    uint32_t *items;
    size_t count;
    double itu;
} HIEPTransaction;

typedef struct {
    uint32_t tid;
    uint32_t last;
    uint32_t span;
    double eu;
    double ru;
} HIEPEntry;

typedef struct {
    HIEPEntry *entries;
    size_t count;
    size_t cap;
    uint32_t item;
    double pwt;
} HIEPList;

typedef struct {
    HIEPToken *tokens;
    size_t token_count;
    size_t token_cap;
    HIEPTransaction *tx;
    size_t tx_count;
    size_t tx_cap;
    size_t stream_count;
    size_t raw_nnz;
    size_t input_bytes;
    uint32_t *rank_to_item;
    uint32_t *item_to_rank;
    size_t survivor_count;
} HIEPDB;

typedef struct {
    uint32_t *map;
    size_t cap;
} HIEPNumericMap;

typedef struct {
    HIEPDB *db;
    const HIEPParams *params;
    HIEPStats *stats;
    HIEPList *singletons;
    FILE *out;
    clock_t started;
    uint32_t *tail_marks;
    uint32_t tail_stamp;
    double ul_len_sum;
    double ido_sum;
    size_t junk_only_outputs;
    uint32_t signal_ids[HIEP_SIGNAL_COUNT][HIEP_SIGNAL_LEN];
    unsigned char signal_present[HIEP_SIGNAL_COUNT];
    unsigned int signal_mask;
    int has_signal_vocab;
    int stop;
} HIEPContext;

static uint32_t effective_min_support(const HIEPParams *params, size_t transactions) {
    if (params->min_support_ratio > 0.0 && params->min_support_ratio < 1.0) {
        uint32_t out = (uint32_t)ceil(params->min_support_ratio * (double)transactions);
        return out ? out : 1;
    }
    return params->min_support ? params->min_support : 1;
}

static int db_init(HIEPDB *db) {
    memset(db, 0, sizeof(*db));
    return 0;
}

static void list_free(HIEPList *list) {
    if (!list) return;
    free(list->entries);
    list->entries = NULL;
    list->count = 0;
    list->cap = 0;
}

static void db_free(HIEPDB *db) {
    if (!db) return;
    for (size_t i = 0; i < db->token_count; i++) free(db->tokens[i].name);
    for (size_t i = 0; i < db->tx_count; i++) free(db->tx[i].items);
    free(db->tokens);
    free(db->tx);
    free(db->rank_to_item);
    free(db->item_to_rank);
    memset(db, 0, sizeof(*db));
}

static int add_token_bytes(HIEPDB *db, const char *name, size_t len, uint32_t *id_out) {
    if (db->token_count >= db->token_cap) {
        size_t next = db->token_cap ? db->token_cap * 2 : 1024;
        HIEPToken *tmp = (HIEPToken *)realloc(db->tokens, next * sizeof(HIEPToken));
        if (!tmp) return -1;
        db->tokens = tmp;
        db->token_cap = next;
    }
    uint32_t id = (uint32_t)db->token_count++;
    HIEPToken *tok = &db->tokens[id];
    memset(tok, 0, sizeof(*tok));
    tok->name = (char *)malloc(len + 1);
    tok->rank = HIEP_NO_RANK;
    if (!tok->name) return -1;
    memcpy(tok->name, name, len);
    tok->name[len] = '\0';
    *id_out = id;
    return 0;
}

static int add_token(HIEPDB *db, const char *name, uint32_t *id_out) {
    return add_token_bytes(db, name, strlen(name), id_out);
}

static int uint32_asc_cmp(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static size_t unique_sorted_ids(uint32_t *items, size_t count) {
    if (count == 0) return 0;
    qsort(items, count, sizeof(uint32_t), uint32_asc_cmp);
    size_t out = 1;
    for (size_t i = 1; i < count; i++) {
        if (items[i] != items[out - 1]) items[out++] = items[i];
    }
    return out;
}

static int add_transaction(HIEPDB *db, const uint32_t *items, size_t count) {
    if (count == 0) return 0;
    if (db->tx_count >= db->tx_cap) {
        size_t next = db->tx_cap ? db->tx_cap * 2 : 1024;
        HIEPTransaction *tmp = (HIEPTransaction *)realloc(db->tx, next * sizeof(HIEPTransaction));
        if (!tmp) return -1;
        db->tx = tmp;
        db->tx_cap = next;
    }
    HIEPTransaction *t = &db->tx[db->tx_count++];
    t->items = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!t->items) return -1;
    memcpy(t->items, items, count * sizeof(uint32_t));
    t->count = count;
    t->itu = 0.0;
    db->raw_nnz += count;
    return 0;
}

typedef struct {
    HIEPDB *db;
    Tokenizer *tokenizer;
    const HIEPParams *params;
    uint32_t *external_to_internal;
    size_t map_cap;
    size_t emitted_windows;
    size_t accepted_windows;
    size_t last_window_count;
    int error;
} HIEPTextBuildCtx;

static void text_build_ctx_free(HIEPTextBuildCtx *ctx) {
    free(ctx->external_to_internal);
    ctx->external_to_internal = NULL;
    ctx->map_cap = 0;
}

static int text_map_get(HIEPTextBuildCtx *ctx, uint32_t external, uint32_t *internal) {
    if (external == 0) return -1;
    if ((size_t)external >= ctx->map_cap) {
        size_t next = ctx->map_cap ? ctx->map_cap : 1024;
        while ((size_t)external >= next) next *= 2;
        uint32_t *tmp = (uint32_t *)realloc(ctx->external_to_internal, next * sizeof(uint32_t));
        if (!tmp) return -1;
        for (size_t i = ctx->map_cap; i < next; i++) tmp[i] = HIEP_NO_RANK;
        ctx->external_to_internal = tmp;
        ctx->map_cap = next;
    }

    if (ctx->external_to_internal[external] == HIEP_NO_RANK) {
        uint32_t len = 0;
        const char *name = dm_tokenizer_token_text(ctx->tokenizer, external, &len);
        uint32_t id;
        if (name && len > 0) {
            if (add_token_bytes(ctx->db, name, len, &id) != 0) return -1;
        } else {
            char fallback[32];
            int n = snprintf(fallback, sizeof(fallback), "tok_%u", external);
            if (n <= 0 || add_token_bytes(ctx->db, fallback, (size_t)n, &id) != 0) return -1;
        }
        ctx->external_to_internal[external] = id;
    }

    *internal = ctx->external_to_internal[external];
    return 0;
}

static void hiep_emit_tokenized_window(const uint32_t *tokens, size_t count, void *user_data) {
    HIEPTextBuildCtx *ctx = (HIEPTextBuildCtx *)user_data;
    const HIEPParams *params = ctx->params;
    size_t stride = params->stride ? params->stride : (params->window_length ? params->window_length : 64);
    size_t start = ctx->emitted_windows * stride;
    ctx->emitted_windows++;

    if (ctx->error || count == 0) return;
    if (params->max_transactions && ctx->accepted_windows >= params->max_transactions) return;
    if (params->max_tokens && start >= params->max_tokens) return;
    if (params->max_tokens && start + count > params->max_tokens) count = params->max_tokens - start;
    if (count == 0) return;
    size_t raw_window_count = count;

    uint32_t *mapped = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!mapped) {
        ctx->error = 1;
        return;
    }
    for (size_t i = 0; i < count; i++) {
        if (text_map_get(ctx, tokens[i], &mapped[i]) != 0) {
            free(mapped);
            ctx->error = 1;
            return;
        }
    }
    if (params->mode == HIEP_MODE_ITEMSET) count = unique_sorted_ids(mapped, count);
    if (add_transaction(ctx->db, mapped, count) != 0) {
        free(mapped);
        ctx->error = 1;
        return;
    }
    free(mapped);
    ctx->accepted_windows++;
    ctx->last_window_count = raw_window_count;
}

static int read_limited_file(const char *path, const HIEPParams *params,
                             unsigned char **data_out, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long end = ftell(f);
    if (end < 0) {
        fclose(f);
        return -1;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }

    size_t len = (size_t)end;
    if (params->max_bytes && params->max_bytes < len) len = params->max_bytes;
    unsigned char *data = (unsigned char *)malloc(len ? len : 1);
    if (!data) {
        fclose(f);
        return -1;
    }
    size_t got = len ? fread(data, 1, len, f) : 0;
    fclose(f);
    if (got != len) {
        free(data);
        return -1;
    }
    *data_out = data;
    *len_out = len;
    return 0;
}

static int load_text_file(const char *path, const HIEPParams *params, HIEPDB *db) {
    unsigned char *data = NULL;
    size_t len = 0;
    if (read_limited_file(path, params, &data, &len) != 0) return -1;
    db->input_bytes = len;

    const char *tokenizer_name = params->tokenizer_name ? params->tokenizer_name : "faro";
    Tokenizer *tokenizer = dm_tokenizer_create(tokenizer_name, 65536);
    if (!tokenizer) {
        fprintf(stderr, "Unknown tokenizer '%s'. Supported tokenizers: %s\n",
                tokenizer_name, dm_tokenizer_supported_names());
        free(data);
        return -1;
    }

    HIEPTextBuildCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.db = db;
    ctx.tokenizer = tokenizer;
    ctx.params = params;

    uint32_t window = (uint32_t)(params->window_length ? params->window_length : 64);
    uint32_t stride = (uint32_t)(params->stride ? params->stride : window);
    TransactionMode mode = params->mode == HIEP_MODE_SEQUENCE ? MODE_SLIDING_SEQUENCE : MODE_SLIDING;
    tokenizer->tokenize_buffer(tokenizer, data, len, mode, window, stride, hiep_emit_tokenized_window, &ctx);

    if (ctx.accepted_windows == 0) {
        db->stream_count = 0;
    } else if (params->max_tokens) {
        size_t observed = (ctx.accepted_windows - 1) * stride + ctx.last_window_count;
        db->stream_count = observed < params->max_tokens ? observed : params->max_tokens;
    } else {
        db->stream_count = (ctx.accepted_windows - 1) * stride + ctx.last_window_count;
    }

    int rc = ctx.error ? -1 : 0;
    text_build_ctx_free(&ctx);
    tokenizer->free(tokenizer);
    free(data);
    return rc;
}

static void numeric_map_free(HIEPNumericMap *map) {
    free(map->map);
    map->map = NULL;
    map->cap = 0;
}

static int numeric_map_get(HIEPNumericMap *map, HIEPDB *db, uint32_t external, uint32_t *internal) {
    if ((size_t)external >= map->cap) {
        size_t next = map->cap ? map->cap : 1024;
        while ((size_t)external >= next) next *= 2;
        uint32_t *tmp = (uint32_t *)realloc(map->map, next * sizeof(uint32_t));
        if (!tmp) return -1;
        for (size_t i = map->cap; i < next; i++) tmp[i] = HIEP_NO_RANK;
        map->map = tmp;
        map->cap = next;
    }
    if (map->map[external] == HIEP_NO_RANK) {
        char name[32];
        snprintf(name, sizeof(name), "%u", external);
        uint32_t id;
        if (add_token(db, name, &id) != 0) return -1;
        map->map[external] = id;
    }
    *internal = map->map[external];
    return 0;
}

static char *read_line_dynamic(FILE *f, size_t *len_out) {
    size_t cap = 4096;
    size_t len = 0;
    char *line = (char *)malloc(cap);
    if (!line) return NULL;
    int c;
    while ((c = fgetc(f)) != EOF) {
        if (len + 2 >= cap) {
            cap *= 2;
            char *tmp = (char *)realloc(line, cap);
            if (!tmp) {
                free(line);
                return NULL;
            }
            line = tmp;
        }
        line[len++] = (char)c;
        if (c == '\n') break;
    }
    if (len == 0 && c == EOF) {
        free(line);
        return NULL;
    }
    line[len] = '\0';
    if (len_out) *len_out = len;
    return line;
}

static int load_transaction_file(const char *path, const HIEPParams *params, HIEPDB *db) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    HIEPNumericMap map;
    memset(&map, 0, sizeof(map));
    uint32_t *items = NULL;
    size_t cap = 0;
    size_t line_len = 0;
    char *line;
    int rc = 0;
    while ((line = read_line_dynamic(f, &line_len)) != NULL) {
        db->input_bytes += line_len;
        char *p = line;
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#' || *p == '%' || *p == '@') {
            free(line);
            continue;
        }
        size_t count = 0;
        while (*p) {
            while (*p && isspace((unsigned char)*p)) p++;
            if (!*p) break;
            char *end = p;
            long value = strtol(p, &end, 10);
            if (end == p) {
                while (*p && !isspace((unsigned char)*p)) p++;
                continue;
            }
            p = end;
            if (value <= 0) continue;
            if (count >= cap) {
                size_t next = cap ? cap * 2 : 64;
                uint32_t *tmp = (uint32_t *)realloc(items, next * sizeof(uint32_t));
                if (!tmp) {
                    rc = -1;
                    free(line);
                    goto done;
                }
                items = tmp;
                cap = next;
            }
            uint32_t internal;
            if (numeric_map_get(&map, db, (uint32_t)value, &internal) != 0) {
                rc = -1;
                free(line);
                goto done;
            }
            items[count++] = internal;
        }
        if (count > 0) {
            if (params->mode == HIEP_MODE_ITEMSET) count = unique_sorted_ids(items, count);
            if (add_transaction(db, items, count) != 0) {
                rc = -1;
                free(line);
                goto done;
            }
        }
        free(line);
        if (params->max_transactions && db->tx_count >= params->max_transactions) break;
        if (params->max_tokens && db->raw_nnz >= params->max_tokens) break;
        if (params->max_bytes && db->input_bytes >= params->max_bytes) break;
    }
done:
    free(items);
    numeric_map_free(&map);
    fclose(f);
    return rc;
}

static int list_push(HIEPList *list, HIEPEntry entry) {
    if (list->count >= list->cap) {
        size_t next = list->cap ? list->cap * 2 : 64;
        HIEPEntry *tmp = (HIEPEntry *)realloc(list->entries, next * sizeof(HIEPEntry));
        if (!tmp) return -1;
        list->entries = tmp;
        list->cap = next;
    }
    list->entries[list->count++] = entry;
    return 0;
}

static int compute_df_weights_bounds(HIEPDB *db, const HIEPParams *params, HIEPStats *stats) {
    if (db->token_count == 0 || db->tx_count == 0) return -1;
    uint32_t *marks = (uint32_t *)calloc(db->token_count, sizeof(uint32_t));
    if (!marks) return -1;
    uint32_t stamp = 1;
    for (size_t tid = 0; tid < db->tx_count; tid++, stamp++) {
        if (stamp == 0) {
            memset(marks, 0, db->token_count * sizeof(uint32_t));
            stamp = 1;
        }
        HIEPTransaction *t = &db->tx[tid];
        for (size_t i = 0; i < t->count; i++) {
            uint32_t item = t->items[i];
            if (marks[item] != stamp) {
                marks[item] = stamp;
                db->tokens[item].df++;
            }
        }
    }

    double alpha = params->alpha > 0.0 ? params->alpha : 0.5;
    double denom = (double)db->tx_count + alpha * (double)db->token_count;
    for (size_t i = 0; i < db->token_count; i++) {
        if (params->uniform_weights) {
            db->tokens[i].weight = 1.0;
        } else {
            double p = ((double)db->tokens[i].df + alpha) / denom;
            if (p <= 0.0) p = DBL_MIN;
            db->tokens[i].weight = -log(p) / log(2.0);
        }
    }

    memset(marks, 0, db->token_count * sizeof(uint32_t));
    stamp = 1;
    for (size_t tid = 0; tid < db->tx_count; tid++, stamp++) {
        if (stamp == 0) {
            memset(marks, 0, db->token_count * sizeof(uint32_t));
            stamp = 1;
        }
        HIEPTransaction *t = &db->tx[tid];
        double itu = 0.0;
        for (size_t i = 0; i < t->count; i++) {
            uint32_t item = t->items[i];
            if (marks[item] != stamp) {
                marks[item] = stamp;
                itu += db->tokens[item].weight;
            }
        }
        t->itu = itu;
        for (size_t i = 0; i < t->count; i++) {
            uint32_t item = t->items[i];
            if (marks[item] == stamp) {
                db->tokens[item].tiub += itu;
                marks[item] = stamp + 1;
            }
        }
        for (size_t i = 0; i < t->count; i++) {
            uint32_t item = t->items[i];
            if (marks[item] == stamp + 1) marks[item] = stamp;
        }
    }
    free(marks);

    double best_singleton = 0.0;
    uint32_t sigma = effective_min_support(params, db->tx_count);
    for (size_t i = 0; i < db->token_count; i++) {
        if (db->tokens[i].df >= sigma) {
            double u = (double)db->tokens[i].df * db->tokens[i].weight;
            if (u > best_singleton) best_singleton = u;
        }
    }
    double ratio = params->theta_ratio > 0.0 ? params->theta_ratio : 0.25;
    stats->theta_ratio = ratio;
    stats->theta = params->theta > 0.0 ? params->theta : best_singleton * ratio;
    if (stats->theta <= 0.0) stats->theta = 1.0;
    return 0;
}

static HIEPDB *g_order_db = NULL;

static int token_order_cmp(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    const HIEPToken *tx = &g_order_db->tokens[x];
    const HIEPToken *ty = &g_order_db->tokens[y];
    if (tx->tiub < ty->tiub) return -1;
    if (tx->tiub > ty->tiub) return 1;
    if (tx->weight > ty->weight) return -1;
    if (tx->weight < ty->weight) return 1;
    return (x > y) - (x < y);
}

static uint32_t *g_rank_map = NULL;

static int rank_order_cmp(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    uint32_t rx = g_rank_map[x];
    uint32_t ry = g_rank_map[y];
    return (rx > ry) - (rx < ry);
}

static int prepare_survivors(HIEPDB *db, const HIEPParams *params, HIEPStats *stats) {
    uint32_t sigma = effective_min_support(params, stats->transactions);
    uint32_t *surv = (uint32_t *)malloc(db->token_count * sizeof(uint32_t));
    if (!surv) return -1;
    for (size_t i = 0; i < db->token_count; i++) {
        if (db->tokens[i].df < sigma) {
            if (db->tokens[i].df > 0) stats->pruned_support++;
            continue;
        }
        if (!params->disable_tiub && db->tokens[i].tiub < stats->theta) {
            stats->pruned_tiub++;
            continue;
        }
        db->tokens[i].survives = 1;
        surv[db->survivor_count++] = (uint32_t)i;
    }
    g_order_db = db;
    qsort(surv, db->survivor_count, sizeof(uint32_t), token_order_cmp);
    g_order_db = NULL;

    db->rank_to_item = (uint32_t *)malloc(db->survivor_count * sizeof(uint32_t));
    db->item_to_rank = (uint32_t *)malloc(db->token_count * sizeof(uint32_t));
    if ((!db->rank_to_item && db->survivor_count) || !db->item_to_rank) {
        free(surv);
        return -1;
    }
    for (size_t i = 0; i < db->token_count; i++) db->item_to_rank[i] = HIEP_NO_RANK;
    for (size_t r = 0; r < db->survivor_count; r++) {
        uint32_t item = surv[r];
        db->rank_to_item[r] = item;
        db->item_to_rank[item] = (uint32_t)r;
        db->tokens[item].rank = (uint32_t)r;
    }
    free(surv);
    stats->surviving_items = db->survivor_count;
    return 0;
}

static int project_transactions(HIEPDB *db, HIEPMode mode) {
    g_rank_map = db->item_to_rank;
    for (size_t tid = 0; tid < db->tx_count; tid++) {
        HIEPTransaction *t = &db->tx[tid];
        size_t out = 0;
        for (size_t i = 0; i < t->count; i++) {
            uint32_t item = t->items[i];
            if (item < db->token_count && db->item_to_rank[item] != HIEP_NO_RANK) {
                t->items[out++] = item;
            }
        }
        t->count = out;
        if (mode == HIEP_MODE_ITEMSET && t->count > 1) {
            qsort(t->items, t->count, sizeof(uint32_t), rank_order_cmp);
            size_t unique = 1;
            for (size_t i = 1; i < t->count; i++) {
                if (t->items[i] != t->items[unique - 1]) t->items[unique++] = t->items[i];
            }
            t->count = unique;
        }
    }
    g_rank_map = NULL;
    return 0;
}

static double tail_ru_itemset(const HIEPDB *db, const HIEPTransaction *t, uint32_t pos) {
    double sum = 0.0;
    for (size_t i = (size_t)pos + 1; i < t->count; i++) sum += db->tokens[t->items[i]].weight;
    return sum;
}

static double tail_ru_sequence(HIEPContext *ctx, const HIEPTransaction *t, uint32_t pos, uint32_t min_rank) {
    double sum = 0.0;
    ctx->tail_stamp++;
    if (ctx->tail_stamp == 0) {
        memset(ctx->tail_marks, 0, ctx->db->token_count * sizeof(uint32_t));
        ctx->tail_stamp = 1;
    }
    for (size_t i = (size_t)pos + 1; i < t->count; i++) {
        uint32_t item = t->items[i];
        uint32_t rank = ctx->db->item_to_rank[item];
        if (rank <= min_rank) continue;
        if (ctx->tail_marks[item] == ctx->tail_stamp) continue;
        ctx->tail_marks[item] = ctx->tail_stamp;
        sum += ctx->db->tokens[item].weight;
    }
    return sum;
}

static int build_singletons(HIEPContext *ctx) {
    HIEPDB *db = ctx->db;
    ctx->singletons = (HIEPList *)calloc(db->survivor_count, sizeof(HIEPList));
    ctx->tail_marks = (uint32_t *)calloc(db->token_count ? db->token_count : 1, sizeof(uint32_t));
    if ((!ctx->singletons && db->survivor_count) || !ctx->tail_marks) return -1;
    ctx->tail_stamp = 1;
    uint32_t *seen = (uint32_t *)calloc(db->token_count ? db->token_count : 1, sizeof(uint32_t));
    if (!seen) return -1;
    uint32_t stamp = 1;
    for (size_t r = 0; r < db->survivor_count; r++) {
        uint32_t item = db->rank_to_item[r];
        ctx->singletons[r].item = item;
        ctx->singletons[r].pwt = db->tokens[item].weight;
    }
    for (size_t tid = 0; tid < db->tx_count; tid++, stamp++) {
        if (stamp == 0) {
            memset(seen, 0, db->token_count * sizeof(uint32_t));
            stamp = 1;
        }
        HIEPTransaction *t = &db->tx[tid];
        if (ctx->params->mode == HIEP_MODE_ITEMSET) {
            for (uint32_t p = 0; p < t->count; p++) {
                uint32_t item = t->items[p];
                uint32_t rank = db->item_to_rank[item];
                HIEPEntry e;
                e.tid = (uint32_t)tid;
                e.last = p;
                e.span = 1;
                e.eu = db->tokens[item].weight;
                e.ru = tail_ru_itemset(db, t, p);
                if (list_push(&ctx->singletons[rank], e) != 0) {
                    free(seen);
                    return -1;
                }
                ctx->stats->singleton_occurrences++;
            }
        } else {
            for (uint32_t p = 0; p < t->count; p++) {
                uint32_t item = t->items[p];
                if (seen[item] == stamp) continue;
                seen[item] = stamp;
                uint32_t rank = db->item_to_rank[item];
                HIEPEntry e;
                e.tid = (uint32_t)tid;
                e.last = p;
                e.span = 1;
                e.eu = db->tokens[item].weight;
                e.ru = tail_ru_sequence(ctx, t, p, rank);
                if (list_push(&ctx->singletons[rank], e) != 0) {
                    free(seen);
                    return -1;
                }
                ctx->stats->singleton_occurrences++;
            }
        }
    }
    free(seen);
    return 0;
}

static int find_after_sequence(const HIEPTransaction *t, uint32_t pos, uint32_t item, uint32_t *found) {
    for (uint32_t p = pos + 1; p < t->count; p++) {
        if (t->items[p] == item) {
            *found = p;
            return 1;
        }
    }
    return 0;
}

static uint32_t sequence_min_span(const HIEPTransaction *t, const uint32_t *pattern, size_t len) {
    uint32_t best = UINT32_MAX;
    if (len == 0 || t->count == 0) return best;
    for (uint32_t start = 0; start < t->count; start++) {
        if (t->items[start] != pattern[0]) continue;
        uint32_t pos = start;
        int ok = 1;
        for (size_t k = 1; k < len; k++) {
            uint32_t next = pos + 1;
            while (next < t->count && t->items[next] != pattern[k]) next++;
            if (next >= t->count) {
                ok = 0;
                break;
            }
            pos = next;
        }
        if (ok) {
            uint32_t span = pos - start + 1;
            if (span < best) best = span;
        }
    }
    return best;
}

static int join_hiep(HIEPContext *ctx, const HIEPList *prefix, uint32_t ext_rank,
                     const uint32_t *child_pattern, size_t child_len, HIEPList *out) {
    memset(out, 0, sizeof(*out));
    HIEPDB *db = ctx->db;
    uint32_t ext_item = db->rank_to_item[ext_rank];
    HIEPList *single = &ctx->singletons[ext_rank];
    out->item = ext_item;
    out->pwt = prefix->pwt + db->tokens[ext_item].weight;
    ctx->stats->joins++;
    size_t i = 0, j = 0;
    while (i < prefix->count && j < single->count) {
        uint32_t ti = prefix->entries[i].tid;
        uint32_t tj = single->entries[j].tid;
        if (ti < tj) {
            i++;
            continue;
        }
        if (tj < ti) {
            j++;
            continue;
        }
        HIEPTransaction *t = &db->tx[ti];
        HIEPEntry e;
        memset(&e, 0, sizeof(e));
        e.tid = ti;
        if (ctx->params->mode == HIEP_MODE_ITEMSET) {
            e.last = single->entries[j].last;
            e.span = (uint32_t)child_len;
            e.eu = out->pwt;
            e.ru = tail_ru_itemset(db, t, e.last);
            if (list_push(out, e) != 0) return -1;
        } else {
            uint32_t pos;
            if (find_after_sequence(t, prefix->entries[i].last, ext_item, &pos)) {
                uint32_t span = sequence_min_span(t, child_pattern, child_len);
                if (span != UINT32_MAX) {
                    double kappa = 1.0;
                    if (!ctx->params->disable_compactness && ctx->params->gamma > 0.0) {
                        kappa = exp(-ctx->params->gamma * ((double)span - (double)child_len));
                    }
                    e.last = pos;
                    e.span = span;
                    e.eu = kappa * out->pwt;
                    e.ru = tail_ru_sequence(ctx, t, pos, ext_rank);
                    if (list_push(out, e) != 0) return -1;
                }
            }
        }
        i++;
        j++;
    }
    ctx->stats->joined_entries += out->count;
    return 0;
}

static int time_limited(HIEPContext *ctx) {
    if (ctx->stop) return 1;
    if (ctx->params->max_seconds <= 0.0) return 0;
    double elapsed = (double)(clock() - ctx->started) / (double)CLOCKS_PER_SEC;
    if (elapsed >= ctx->params->max_seconds) {
        ctx->stats->limited = 1;
        ctx->stop = 1;
        return 1;
    }
    return 0;
}

static int token_has_prefix(const char *name, const char *prefix) {
    return strncmp(name, prefix, strlen(prefix)) == 0;
}

static int is_junk_item(const HIEPDB *db, uint32_t item) {
    const char *name = db->tokens[item].name;
    return token_has_prefix(name, "stop_") ||
           token_has_prefix(name, "junk_") ||
           token_has_prefix(name, "filler_");
}

static int pattern_is_junk_only(const HIEPDB *db, const uint32_t *pattern, size_t len) {
    if (len == 0) return 0;
    for (size_t i = 0; i < len; i++) {
        if (!is_junk_item(db, pattern[i])) return 0;
    }
    return 1;
}

static int pattern_matches_signal(HIEPContext *ctx, const uint32_t *pattern, size_t len, size_t sig) {
    if (!ctx->signal_present[sig] || len != HIEP_SIGNAL_LEN) return 0;
    for (size_t i = 0; i < HIEP_SIGNAL_LEN; i++) {
        int found = 0;
        for (size_t j = 0; j < len; j++) {
            if (pattern[j] == ctx->signal_ids[sig][i]) {
                found = 1;
                break;
            }
        }
        if (!found) return 0;
    }
    return 1;
}

static void discover_signal_vocab(HIEPContext *ctx) {
    static const char *signals[HIEP_SIGNAL_COUNT][HIEP_SIGNAL_LEN] = {
        {"rare_api_timeout", "rare_retry_backoff", "rare_circuit_breaker"},
        {"rare_memory_leak", "rare_heap_snapshot", "rare_gc_pause"},
        {"rare_auth_refresh", "rare_token_rotation", "rare_oauth_scope"},
        {"rare_vector_index", "rare_embedding_drift", "rare_recall_drop"}
    };
    for (size_t s = 0; s < HIEP_SIGNAL_COUNT; s++) {
        int ok = 1;
        for (size_t k = 0; k < HIEP_SIGNAL_LEN; k++) {
            uint32_t id = HIEP_NO_RANK;
            for (size_t i = 0; i < ctx->db->token_count; i++) {
                if (strcmp(ctx->db->tokens[i].name, signals[s][k]) == 0) {
                    id = (uint32_t)i;
                    break;
                }
            }
            if (id == HIEP_NO_RANK) {
                ok = 0;
                break;
            }
            ctx->signal_ids[s][k] = id;
        }
        ctx->signal_present[s] = (unsigned char)ok;
        if (ok) ctx->has_signal_vocab = 1;
    }
}

static int emit_pattern(HIEPContext *ctx, const uint32_t *pattern, size_t len,
                        size_t support, double utility, double pwt) {
    HIEPStats *s = ctx->stats;
    s->emitted_patterns++;
    s->total_output_items += len;
    s->avg_support += (double)support;
    s->avg_utility += utility;
    s->avg_pattern_weight += pwt;
    ctx->ido_sum += len ? utility / (double)len : 0.0;
    if (utility > s->best_utility) s->best_utility = utility;
    if (pattern_is_junk_only(ctx->db, pattern, len)) ctx->junk_only_outputs++;
    for (size_t sig = 0; sig < HIEP_SIGNAL_COUNT; sig++) {
        if (pattern_matches_signal(ctx, pattern, len, sig)) {
            ctx->signal_mask |= 1u << sig;
        }
    }
    if (ctx->out) {
        fprintf(ctx->out, "%.10g\t%zu\t%zu\t%.10g\t", utility, support, len, pwt);
        for (size_t i = 0; i < len; i++) {
            if (i) fputc(' ', ctx->out);
            fputs(ctx->db->tokens[pattern[i]].name, ctx->out);
        }
        fputc('\n', ctx->out);
    }
    if (ctx->params->max_patterns && s->emitted_patterns >= ctx->params->max_patterns) {
        s->limited = 1;
        ctx->stop = 1;
    }
    return 0;
}

static void list_score(const HIEPList *list, double *utility, double *iwru) {
    double u = 0.0;
    double b = 0.0;
    for (size_t i = 0; i < list->count; i++) {
        u += list->entries[i].eu;
        b += list->pwt + list->entries[i].ru;
    }
    *utility = u;
    *iwru = b;
}

static int dfs_hiep(HIEPContext *ctx, const uint32_t *pattern, size_t depth,
                    HIEPList *list, size_t suffix_start) {
    if (time_limited(ctx)) return 0;
    if (list->count < ctx->stats->min_support) {
        ctx->stats->pruned_support++;
        return 0;
    }
    HIEPStats *stats = ctx->stats;
    stats->visited_nodes++;
    ctx->ul_len_sum += (double)list->count;
    if ((double)list->count > stats->max_utility_list_length) {
        stats->max_utility_list_length = (double)list->count;
    }
    if (depth > stats->max_depth_seen) stats->max_depth_seen = depth;

    double utility, iwru;
    list_score(list, &utility, &iwru);
    if (utility >= stats->theta) {
        emit_pattern(ctx, pattern, depth, list->count, utility, list->pwt);
    }
    if (time_limited(ctx)) return 0;
    if (!ctx->params->disable_iwru && iwru < stats->theta) {
        stats->pruned_iwru++;
        return 0;
    }
    if (ctx->params->max_depth && depth >= ctx->params->max_depth) return 0;

    for (size_t r = suffix_start; r < ctx->db->survivor_count; r++) {
        if (time_limited(ctx)) break;
        uint32_t *child_pattern = (uint32_t *)malloc((depth + 1) * sizeof(uint32_t));
        if (!child_pattern) return -1;
        memcpy(child_pattern, pattern, depth * sizeof(uint32_t));
        child_pattern[depth] = ctx->db->rank_to_item[r];
        HIEPList child;
        stats->generated_children++;
        if (join_hiep(ctx, list, (uint32_t)r, child_pattern, depth + 1, &child) != 0) {
            free(child_pattern);
            return -1;
        }
        if (child.count >= stats->min_support) {
            if (dfs_hiep(ctx, child_pattern, depth + 1, &child, r + 1) != 0) {
                list_free(&child);
                free(child_pattern);
                return -1;
            }
        } else if (child.count > 0) {
            stats->pruned_support++;
        }
        list_free(&child);
        free(child_pattern);
    }
    return 0;
}

HIEPParams hiep_default_params(void) {
    HIEPParams p;
    memset(&p, 0, sizeof(p));
    p.mode = HIEP_MODE_ITEMSET;
    p.input_type = HIEP_INPUT_TEXT;
    p.window_length = 64;
    p.stride = 64;
    p.theta_ratio = 0.25;
    p.min_support = 2;
    p.min_support_ratio = 0.0;
    p.alpha = 0.5;
    p.gamma = 0.15;
    p.tokenizer_name = "faro";
    return p;
}

int hiep_parse_mode(const char *name, HIEPMode *mode) {
    if (!name || strcmp(name, "itemset") == 0 || strcmp(name, "items") == 0) {
        *mode = HIEP_MODE_ITEMSET;
        return 0;
    }
    if (strcmp(name, "sequence") == 0 || strcmp(name, "seq") == 0) {
        *mode = HIEP_MODE_SEQUENCE;
        return 0;
    }
    return -1;
}

const char *hiep_mode_name(HIEPMode mode) {
    return mode == HIEP_MODE_SEQUENCE ? "sequence" : "itemset";
}

int hiep_parse_input_type(const char *name, HIEPInputType *type) {
    if (!name || strcmp(name, "text") == 0 || strcmp(name, "stream") == 0) {
        *type = HIEP_INPUT_TEXT;
        return 0;
    }
    if (strcmp(name, "transactions") == 0 || strcmp(name, "transaction") == 0 || strcmp(name, "itemsets") == 0) {
        *type = HIEP_INPUT_TRANSACTIONS;
        return 0;
    }
    return -1;
}

const char *hiep_input_type_name(HIEPInputType type) {
    return type == HIEP_INPUT_TRANSACTIONS ? "transactions" : "text";
}

int hiep_mine_file(const char *path, const HIEPParams *params, HIEPStats *stats) {
    if (!path || !params || !stats) return -1;
    memset(stats, 0, sizeof(*stats));
    stats->alpha = params->alpha > 0.0 ? params->alpha : 0.5;
    stats->gamma = params->disable_compactness ? 0.0 : params->gamma;
    stats->min_support = params->min_support ? params->min_support : 1;

    HIEPDB db;
    if (db_init(&db) != 0) return -1;
    int rc = params->input_type == HIEP_INPUT_TRANSACTIONS
        ? load_transaction_file(path, params, &db)
        : load_text_file(path, params, &db);
    if (rc != 0 || db.tx_count == 0) {
        db_free(&db);
        return -1;
    }
    stats->input_bytes = db.input_bytes;
    stats->token_stream_length = params->input_type == HIEP_INPUT_TEXT ? db.stream_count : db.raw_nnz;
    stats->transactions = db.tx_count;
    stats->nnz = db.raw_nnz;
    stats->vocabulary_size = db.token_count;
    stats->min_support = effective_min_support(params, db.tx_count);

    if (compute_df_weights_bounds(&db, params, stats) != 0 ||
        prepare_survivors(&db, params, stats) != 0 ||
        project_transactions(&db, params->mode) != 0) {
        db_free(&db);
        return -1;
    }

    HIEPContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.db = &db;
    ctx.params = params;
    ctx.stats = stats;
    ctx.started = clock();
    discover_signal_vocab(&ctx);
    if (params->output_path) {
        ctx.out = fopen(params->output_path, "w");
        if (!ctx.out) {
            db_free(&db);
            return -1;
        }
        fprintf(ctx.out, "utility\tsupport\tlength\tpattern_weight\tpattern\n");
    }
    if (build_singletons(&ctx) != 0) {
        if (ctx.out) fclose(ctx.out);
        db_free(&db);
        return -1;
    }

    for (size_t r = 0; r < db.survivor_count; r++) {
        if (time_limited(&ctx)) break;
        HIEPList *list = &ctx.singletons[r];
        uint32_t item = db.rank_to_item[r];
        if (list->count >= stats->min_support) {
            if (dfs_hiep(&ctx, &item, 1, list, r + 1) != 0) {
                rc = -1;
                break;
            }
        } else if (list->count > 0) {
            stats->pruned_support++;
        }
    }

    if (stats->emitted_patterns) {
        stats->avg_support /= (double)stats->emitted_patterns;
        stats->avg_utility /= (double)stats->emitted_patterns;
        stats->avg_pattern_weight /= (double)stats->emitted_patterns;
        stats->information_density_optimization = ctx.ido_sum / (double)stats->emitted_patterns;
        stats->noise_filtering_efficiency = 1.0 - ((double)ctx.junk_only_outputs / (double)stats->emitted_patterns);
    }
    if (stats->visited_nodes) {
        stats->avg_utility_list_length = ctx.ul_len_sum / (double)stats->visited_nodes;
    }
    if (ctx.has_signal_vocab) {
        unsigned int recovered = 0;
        for (size_t i = 0; i < HIEP_SIGNAL_COUNT; i++) recovered += (ctx.signal_mask >> i) & 1u;
        stats->signal_recall = (double)recovered / (double)HIEP_SIGNAL_COUNT;
    }
    stats->result_ram_bytes = stats->emitted_patterns * 40 + stats->total_output_items * sizeof(uint32_t);
    stats->result_disk_est_bytes = stats->emitted_patterns * 96 + stats->total_output_items * 18;

    if (ctx.out) fclose(ctx.out);
    for (size_t r = 0; r < db.survivor_count; r++) list_free(&ctx.singletons[r]);
    free(ctx.singletons);
    free(ctx.tail_marks);
    db_free(&db);
    return rc;
}

static DM_Status hiep_algorithm_run(DM_Dataset *ds, void *params) {
    (void)ds;
    HIEPRunConfig *cfg = (HIEPRunConfig *)params;
    if (!cfg || !cfg->input_path || !cfg->stats) return DM_ERROR_INVALID_PARAM;
    return hiep_mine_file(cfg->input_path, &cfg->params, cfg->stats) == 0
        ? DM_SUCCESS
        : DM_ERROR_GENERIC;
}

DM_Algorithm hiep_algo = {
    .id = "hiep",
    .name = "HIEP-Miner",
    .description = "High-information entropy itemset/sequence miner over tokenizer-backed text windows and numeric transactions",
    .supported_types = 0,
    .run = hiep_algorithm_run
};
