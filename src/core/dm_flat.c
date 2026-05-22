#include "core/dm_flat.h"
#include "tokenizer/tokenizer.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t *items;
    size_t item_count;
    size_t item_cap;
    size_t *offsets;
    size_t row_count;
    size_t row_cap;
    uint32_t max_item;
} FlatBuilder;

static int builder_init(FlatBuilder *b) {
    memset(b, 0, sizeof(*b));
    b->row_cap = 1024;
    b->offsets = (size_t *)malloc((b->row_cap + 1) * sizeof(size_t));
    if (!b->offsets) return -1;
    b->offsets[0] = 0;
    return 0;
}

static void builder_free(FlatBuilder *b) {
    free(b->items);
    free(b->offsets);
    memset(b, 0, sizeof(*b));
}

static int builder_reserve_items(FlatBuilder *b, size_t extra) {
    if (extra <= b->item_cap - b->item_count) return 0;
    size_t nc = b->item_cap ? b->item_cap * 2 : 4096;
    while (extra > nc - b->item_count) nc *= 2;
    uint32_t *ni = (uint32_t *)realloc(b->items, nc * sizeof(uint32_t));
    if (!ni) return -1;
    b->items = ni;
    b->item_cap = nc;
    return 0;
}

static int builder_begin_row(FlatBuilder *b) {
    if (b->row_count + 1 >= b->row_cap) {
        size_t nc = b->row_cap * 2;
        size_t *no = (size_t *)realloc(b->offsets, (nc + 1) * sizeof(size_t));
        if (!no) return -1;
        b->offsets = no;
        b->row_cap = nc;
    }
    return 0;
}

static int builder_add_item(FlatBuilder *b, uint32_t item) {
    if (builder_reserve_items(b, 1) != 0) return -1;
    b->items[b->item_count++] = item;
    if (item > b->max_item) b->max_item = item;
    return 0;
}

static int builder_end_row(FlatBuilder *b) {
    if (b->item_count == b->offsets[b->row_count]) return 0;
    b->row_count++;
    b->offsets[b->row_count] = b->item_count;
    return 0;
}

static int finalize_builder(FlatBuilder *b, const DM_MMap *map, const DM_ConnectorOptions *opt, DM_Arena *arena, DM_FlatDataset *out) {
    memset(out, 0, sizeof(*out));
    out->items = DM_ARENA_NEW(arena, uint32_t, b->item_count ? b->item_count : 1);
    out->row_offsets = DM_ARENA_NEW(arena, size_t, b->row_count + 1);
    if (!out->items || !out->row_offsets) return -1;
    memcpy(out->items, b->items, b->item_count * sizeof(uint32_t));
    memcpy(out->row_offsets, b->offsets, (b->row_count + 1) * sizeof(size_t));
    out->item_count = b->item_count;
    out->row_count = b->row_count;
    out->max_item = b->max_item;
    out->source_kind = opt->kind;
    out->arena = arena;
    out->borrowed_base = NULL;
    out->borrowed_bytes = map->size;
    return 0;
}

const char *dm_connector_name(DM_ConnectorKind kind) {
    switch (kind) {
        case DM_CONNECTOR_SPMF: return "spmf";
        case DM_CONNECTOR_TEXT: return "text";
        case DM_CONNECTOR_GRAPH: return "graph";
    }
    return "unknown";
}

int dm_connector_parse_kind(const char *name, DM_ConnectorKind *kind) {
    if (!name || !kind) return -1;
    if (strcmp(name, "spmf") == 0 || strcmp(name, "transactions") == 0) {
        *kind = DM_CONNECTOR_SPMF;
        return 0;
    }
    if (strcmp(name, "text") == 0 || strcmp(name, "language") == 0) {
        *kind = DM_CONNECTOR_TEXT;
        return 0;
    }
    if (strcmp(name, "graph") == 0 || strcmp(name, "network") == 0) {
        *kind = DM_CONNECTOR_GRAPH;
        return 0;
    }
    return -1;
}

DM_ConnectorOptions dm_connector_default_options(DM_ConnectorKind kind) {
    DM_ConnectorOptions opt;
    memset(&opt, 0, sizeof(opt));
    opt.kind = kind;
    opt.text_window = 64;
    opt.text_stride = 32;
    opt.text_sequence = 1;
    opt.graph_undirected = 0;
    return opt;
}

static int load_spmf(const DM_MMap *map, FlatBuilder *b) {
    const unsigned char *p = map->data;
    const unsigned char *end = map->data + map->size;
    if (builder_begin_row(b) != 0) return -1;
    while (p < end) {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\r')) p++;
        if (p >= end) break;
        if (*p == '\n') {
            if (builder_end_row(b) != 0 || builder_begin_row(b) != 0) return -1;
            p++;
            continue;
        }
        if (*p == '#') {
            while (p < end && *p != '\n') p++;
            continue;
        }
        char *next = NULL;
        unsigned long v = strtoul((const char *)p, &next, 10);
        if (next == (const char *)p) {
            while (p < end && *p != '\n') p++;
            continue;
        }
        if (builder_add_item(b, (uint32_t)v) != 0) return -1;
        p = (const unsigned char *)next;
    }
    if (builder_end_row(b) != 0) return -1;
    return 0;
}

typedef struct {
    FlatBuilder *builder;
    int sequence;
    int failed;
} TextCtx;

static int uint32_cmp(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static size_t uniq_u32(uint32_t *v, size_t n) {
    if (n == 0) return 0;
    qsort(v, n, sizeof(uint32_t), uint32_cmp);
    size_t w = 1;
    for (size_t i = 1; i < n; i++) {
        if (v[i] != v[w - 1]) v[w++] = v[i];
    }
    return w;
}

static void text_emit(const uint32_t *tokens, size_t count, void *user_data) {
    TextCtx *ctx = (TextCtx *)user_data;
    if (ctx->failed || count == 0) return;
    uint32_t *tmp = NULL;
    if (!ctx->sequence) {
        tmp = (uint32_t *)malloc(count * sizeof(uint32_t));
        if (!tmp) {
            ctx->failed = 1;
            return;
        }
        memcpy(tmp, tokens, count * sizeof(uint32_t));
        count = uniq_u32(tmp, count);
        tokens = tmp;
    }
    if (builder_begin_row(ctx->builder) != 0) ctx->failed = 1;
    for (size_t i = 0; !ctx->failed && i < count; i++) {
        if (builder_add_item(ctx->builder, tokens[i]) != 0) ctx->failed = 1;
    }
    if (!ctx->failed && builder_end_row(ctx->builder) != 0) ctx->failed = 1;
    free(tmp);
}

static int load_text(const DM_MMap *map, const DM_ConnectorOptions *opt, FlatBuilder *b) {
    Tokenizer *tok = dm_faro_tok_create("rh-arena", 1u << 16);
    if (!tok) return -1;
    tok->base_address = map->data;
    TextCtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.builder = b;
    ctx.sequence = opt->text_sequence;
    TransactionMode mode = opt->text_sequence ? MODE_SLIDING_SEQUENCE : MODE_SLIDING;
    tok->tokenize_buffer(tok, map->data, map->size, mode, (uint32_t)opt->text_window, (uint32_t)opt->text_stride, text_emit, &ctx);
    tok->free(tok);
    return ctx.failed ? -1 : 0;
}

typedef struct {
    uint32_t src;
    uint32_t dst;
} Edge;

static int edge_cmp(const void *a, const void *b) {
    const Edge *x = (const Edge *)a;
    const Edge *y = (const Edge *)b;
    if (x->src != y->src) return (x->src > y->src) - (x->src < y->src);
    return (x->dst > y->dst) - (x->dst < y->dst);
}

static int parse_edge_line(const unsigned char *line, size_t len, uint32_t *a, uint32_t *b) {
    const unsigned char *p = line;
    const unsigned char *end = line + len;
    while (p < end && isspace(*p)) p++;
    if (p >= end || *p == '#') return 0;
    char *next = NULL;
    unsigned long x = strtoul((const char *)p, &next, 10);
    if (next == (const char *)p) return 0;
    p = (const unsigned char *)next;
    unsigned long y = strtoul((const char *)p, &next, 10);
    if (next == (const char *)p) return 0;
    *a = (uint32_t)x;
    *b = (uint32_t)y;
    return 1;
}

static int load_graph(const DM_MMap *map, const DM_ConnectorOptions *opt, FlatBuilder *b) {
    Edge *edges = NULL;
    size_t count = 0, cap = 0;
    const unsigned char *p = map->data;
    const unsigned char *end = map->data + map->size;
    while (p < end) {
        const unsigned char *line = p;
        while (p < end && *p != '\n') p++;
        uint32_t a = 0, c = 0;
        int ok = parse_edge_line(line, (size_t)(p - line), &a, &c);
        if (ok) {
            size_t need = opt->graph_undirected ? 2 : 1;
            if (count + need > cap) {
                size_t nc = cap ? cap * 2 : 1024;
                while (count + need > nc) nc *= 2;
                Edge *ne = (Edge *)realloc(edges, nc * sizeof(*ne));
                if (!ne) {
                    free(edges);
                    return -1;
                }
                edges = ne;
                cap = nc;
            }
            edges[count++] = (Edge){a, c};
            if (opt->graph_undirected) edges[count++] = (Edge){c, a};
        }
        if (p < end) p++;
    }
    qsort(edges, count, sizeof(*edges), edge_cmp);
    size_t i = 0;
    while (i < count) {
        uint32_t src = edges[i].src;
        if (builder_begin_row(b) != 0 || builder_add_item(b, src) != 0) {
            free(edges);
            return -1;
        }
        while (i < count && edges[i].src == src) {
            if (builder_add_item(b, edges[i].dst) != 0) {
                free(edges);
                return -1;
            }
            i++;
        }
        if (builder_end_row(b) != 0) {
            free(edges);
            return -1;
        }
    }
    free(edges);
    return 0;
}

static void fill_stats(const DM_MMap *map, const DM_FlatDataset *flat, DM_ConnectorStats *stats) {
    if (!stats) return;
    memset(stats, 0, sizeof(*stats));
    stats->input_bytes = map->size;
    stats->rows = flat->row_count;
    stats->items = flat->item_count;
    stats->max_item = flat->max_item;
    stats->avg_row_len = flat->row_count ? (double)flat->item_count / (double)flat->row_count : 0.0;
    stats->distinct_estimate = flat->max_item ? (size_t)flat->max_item + 1 : 0;
}

int dm_flat_load_mmap(const char *path, const DM_ConnectorOptions *options, DM_Arena *arena, DM_FlatDataset *out, DM_ConnectorStats *stats) {
    if (!path || !options || !arena || !out) return -1;
    DM_MMap map;
    if (dm_mmap_open(path, &map) != 0) return -1;
    FlatBuilder b;
    if (builder_init(&b) != 0) {
        dm_mmap_close(&map);
        return -1;
    }
    int rc = -1;
    if (options->kind == DM_CONNECTOR_SPMF) rc = load_spmf(&map, &b);
    else if (options->kind == DM_CONNECTOR_TEXT) rc = load_text(&map, options, &b);
    else if (options->kind == DM_CONNECTOR_GRAPH) rc = load_graph(&map, options, &b);
    if (rc == 0) rc = finalize_builder(&b, &map, options, arena, out);
    if (rc == 0) fill_stats(&map, out, stats);
    builder_free(&b);
    dm_mmap_close(&map);
    return rc;
}
