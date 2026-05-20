#include "core/dm_plugin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint32_t item;
    size_t count;
} ItemCount;

static int cmp_item_count_desc(const void *a, const void *b) {
    const ItemCount *x = (const ItemCount *)a;
    const ItemCount *y = (const ItemCount *)b;
    if (x->count < y->count) return 1;
    if (x->count > y->count) return -1;
    return (x->item > y->item) - (x->item < y->item);
}

static int flat_stats_run(const DM_PluginInput *input, DM_PluginResult *result) {
    const DM_FlatDataset *flat = input->flat;
    clock_t start = clock();
    size_t min_len = flat->row_count ? (size_t)-1 : 0;
    size_t max_len = 0;
    for (size_t r = 0; r < flat->row_count; r++) {
        size_t len = flat->row_offsets[r + 1] - flat->row_offsets[r];
        if (len < min_len) min_len = len;
        if (len > max_len) max_len = len;
    }
    printf("plugin=flat_stats\n");
    printf("connector=%s\n", dm_connector_name(flat->source_kind));
    printf("rows=%zu\n", flat->row_count);
    printf("items=%zu\n", flat->item_count);
    printf("max_item=%u\n", flat->max_item);
    printf("min_row_len=%zu\n", min_len);
    printf("max_row_len=%zu\n", max_len);
    printf("avg_row_len=%.6f\n", flat->row_count ? (double)flat->item_count / (double)flat->row_count : 0.0);
    if (result) {
        result->patterns = flat->row_count;
        result->total_items = flat->item_count;
        result->best_score = (double)max_len;
        result->runtime_sec = (double)(clock() - start) / (double)CLOCKS_PER_SEC;
    }
    return 0;
}

static int topk_items_run(const DM_PluginInput *input, DM_PluginResult *result) {
    const DM_FlatDataset *flat = input->flat;
    clock_t start = clock();
    size_t k = (size_t)dm_plugin_arg_long(input, "k", 20);
    size_t cap = (size_t)flat->max_item + 1;
    size_t *counts = DM_ARENA_NEW(input->arena, size_t, cap ? cap : 1);
    ItemCount *pairs = DM_ARENA_NEW(input->arena, ItemCount, cap ? cap : 1);
    if (!counts || !pairs) return -1;
    for (size_t i = 0; i < flat->item_count; i++) counts[flat->items[i]]++;
    size_t n = 0;
    for (uint32_t item = 0; item <= flat->max_item; item++) {
        if (counts[item] == 0) continue;
        pairs[n].item = item;
        pairs[n].count = counts[item];
        n++;
    }
    qsort(pairs, n, sizeof(*pairs), cmp_item_count_desc);
    if (k > n) k = n;
    printf("plugin=topk_items\n");
    printf("connector=%s\n", dm_connector_name(flat->source_kind));
    printf("k=%zu\n", k);
    printf("distinct_items=%zu\n", n);
    for (size_t i = 0; i < k; i++) {
        printf("rank_%zu_item=%u rank_%zu_count=%zu\n", i + 1, pairs[i].item, i + 1, pairs[i].count);
    }
    if (result) {
        result->patterns = k;
        result->total_items = k;
        result->best_score = k ? (double)pairs[0].count : 0.0;
        result->runtime_sec = (double)(clock() - start) / (double)CLOCKS_PER_SEC;
    }
    return 0;
}

static DM_Plugin flat_stats_plugin = {
    .id = "flat_stats",
    .name = "Flat Dataset Statistics",
    .description = "Reports row/item statistics for any connector-backed flat dataset.",
    .accepts_connectors = DM_PLUGIN_ACCEPT_ALL,
    .run = flat_stats_run
};

static DM_Plugin topk_items_plugin = {
    .id = "topk_items",
    .name = "Top-k Item Counter",
    .description = "Mines the top-k most common integer items from any flat dataset.",
    .accepts_connectors = DM_PLUGIN_ACCEPT_ALL,
    .run = topk_items_run
};

DM_REGISTER_PLUGIN(flat_stats_plugin)
DM_REGISTER_PLUGIN(topk_items_plugin)
