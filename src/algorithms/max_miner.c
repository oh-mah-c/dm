#include "algorithms/max_miner.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

typedef struct {
    uint64_t *words;
    size_t num_words;
} BitSet;

typedef struct {
    uint64_t *words;
    size_t num_items;
} ItemBitSet;

typedef struct {
    ItemBitSet *itemsets;
    size_t count;
    size_t capacity;
} F_Registry;

typedef struct {
    BitSet *db;
    uint64_t *depth_tidlists;
    size_t num_words;
    uint32_t min_sup;
    F_Registry f_reg;
    size_t item_num_words;
} MaxMinerContext;

static inline size_t intersect_bitsets_count(const BitSet *a, const BitSet *b) {
    size_t count = 0;
    for (size_t i = 0; i < a->num_words; i++) {
        count += (size_t)__builtin_popcountll(a->words[i] & b->words[i]);
    }
    return count;
}

static inline void intersect_bitsets_store(BitSet *out, const BitSet *a, const BitSet *b) {
    for (size_t i = 0; i < a->num_words; i++) {
        out->words[i] = a->words[i] & b->words[i];
    }
}

static inline void itemset_set_bit(ItemBitSet *bs, uint32_t item) {
    bs->words[item / 64] |= (1ULL << (item % 64));
}

static inline bool itemset_is_subset(const ItemBitSet *sub, const ItemBitSet *sup, size_t num_words) {
    for (size_t i = 0; i < num_words; i++) {
        if ((sub->words[i] & ~sup->words[i]) != 0) return false;
    }
    return true;
}

static void f_add(F_Registry *f, const uint32_t *head, size_t head_len, const uint32_t *tail, size_t tail_len, size_t item_num_words) {
    ItemBitSet new_bs;
    new_bs.words = calloc(item_num_words, sizeof(uint64_t));
    new_bs.num_items = head_len + tail_len;
    for (size_t i = 0; i < head_len; i++) itemset_set_bit(&new_bs, head[i]);
    for (size_t i = 0; i < tail_len; i++) itemset_set_bit(&new_bs, tail[i]);
    
    for (size_t i = 0; i < f->count; i++) {
        if (f->itemsets[i].words != NULL) {
            if (new_bs.num_items <= f->itemsets[i].num_items) {
                if (itemset_is_subset(&new_bs, &f->itemsets[i], item_num_words)) {
                    free(new_bs.words);
                    return; // Has a superset
                }
            }
        }
    }
    
    for (size_t i = 0; i < f->count; i++) {
        if (f->itemsets[i].words != NULL) {
            if (new_bs.num_items >= f->itemsets[i].num_items) {
                if (itemset_is_subset(&f->itemsets[i], &new_bs, item_num_words)) {
                    free(f->itemsets[i].words);
                    f->itemsets[i].words = NULL; // Remove subset
                }
            }
        }
    }
    
    for (size_t i = 0; i < f->count; i++) {
        if (f->itemsets[i].words == NULL) {
            f->itemsets[i] = new_bs;
            return;
        }
    }
    
    if (f->count >= f->capacity) {
        f->capacity = f->capacity == 0 ? 1024 : f->capacity * 2;
        f->itemsets = realloc(f->itemsets, f->capacity * sizeof(ItemBitSet));
    }
    f->itemsets[f->count++] = new_bs;
}

static bool is_subset_of_any_in_F(const ItemBitSet *ht_bs, MaxMinerContext *ctx) {
    for (size_t i = 0; i < ctx->f_reg.count; i++) {
        if (ctx->f_reg.itemsets[i].words != NULL) {
            if (ht_bs->num_items <= ctx->f_reg.itemsets[i].num_items) {
                if (itemset_is_subset(ht_bs, &ctx->f_reg.itemsets[i], ctx->item_num_words)) {
                    return true;
                }
            }
        }
    }
    return false;
}

typedef struct {
    uint32_t item;
    uint32_t sup;
} TailItem;

static int cmp_tail(const void *a, const void *b) {
    TailItem *ta = (TailItem *)a;
    TailItem *tb = (TailItem *)b;
    if (ta->sup < tb->sup) return -1;
    if (ta->sup > tb->sup) return 1;
    if (ta->item < tb->item) return -1;
    if (ta->item > tb->item) return 1;
    return 0;
}

static void sort_tail(uint32_t *tail, uint32_t *tail_sup, size_t tail_len) {
    TailItem *arr = malloc(tail_len * sizeof(TailItem));
    for (size_t i = 0; i < tail_len; i++) {
        arr[i].item = tail[i];
        arr[i].sup = tail_sup[i];
    }
    qsort(arr, tail_len, sizeof(TailItem), cmp_tail);
    for (size_t i = 0; i < tail_len; i++) {
        tail[i] = arr[i].item;
        tail_sup[i] = arr[i].sup;
    }
    free(arr);
}

static void dfs_max_miner(uint32_t *head, size_t head_len, BitSet *head_bs,
                          uint32_t *tail, size_t tail_len, MaxMinerContext *ctx, int depth) {
    if (tail_len == 0) {
        f_add(&ctx->f_reg, head, head_len, NULL, 0, ctx->item_num_words);
        return;
    }
    
    ItemBitSet ht_bs;
    ht_bs.words = calloc(ctx->item_num_words, sizeof(uint64_t));
    ht_bs.num_items = head_len + tail_len;
    for (size_t i = 0; i < head_len; i++) itemset_set_bit(&ht_bs, head[i]);
    for (size_t i = 0; i < tail_len; i++) itemset_set_bit(&ht_bs, tail[i]);
    
    if (is_subset_of_any_in_F(&ht_bs, ctx)) {
        free(ht_bs.words);
        return;
    }
    free(ht_bs.words);
    
    BitSet ht_tidlist;
    ht_tidlist.num_words = ctx->num_words;
    ht_tidlist.words = &ctx->depth_tidlists[depth * ctx->num_words];
    for (size_t w = 0; w < ctx->num_words; w++) {
        ht_tidlist.words[w] = head_bs->words[w];
    }
    
    uint32_t *tail_sup = malloc(tail_len * sizeof(uint32_t));
    for (size_t i = 0; i < tail_len; i++) {
        uint32_t item = tail[i];
        tail_sup[i] = intersect_bitsets_count(head_bs, &ctx->db[item]);
        intersect_bitsets_store(&ht_tidlist, &ht_tidlist, &ctx->db[item]);
    }
    
    uint32_t head_tail_sup = 0;
    for (size_t w = 0; w < ctx->num_words; w++) {
        head_tail_sup += (uint32_t)__builtin_popcountll(ht_tidlist.words[w]);
    }
    
    if (head_tail_sup >= ctx->min_sup) {
        f_add(&ctx->f_reg, head, head_len, tail, tail_len, ctx->item_num_words);
        free(tail_sup);
        return;
    }
    
    size_t new_tail_len = 0;
    for (size_t i = 0; i < tail_len; i++) {
        if (tail_sup[i] >= ctx->min_sup) {
            tail[new_tail_len] = tail[i];
            tail_sup[new_tail_len] = tail_sup[i];
            new_tail_len++;
        }
    }
    tail_len = new_tail_len;
    
    if (tail_len == 0) {
        f_add(&ctx->f_reg, head, head_len, NULL, 0, ctx->item_num_words);
        free(tail_sup);
        return;
    }
    
    sort_tail(tail, tail_sup, tail_len);
    
    for (size_t i = 0; i < tail_len - 1; i++) {
        uint32_t item = tail[i];
        uint32_t *new_head = malloc((head_len + 1) * sizeof(uint32_t));
        if (head_len > 0) memcpy(new_head, head, head_len * sizeof(uint32_t));
        new_head[head_len] = item;
        
        uint32_t *new_tail = NULL;
        if (tail_len - i - 1 > 0) {
            new_tail = malloc((tail_len - i - 1) * sizeof(uint32_t));
            memcpy(new_tail, tail + i + 1, (tail_len - i - 1) * sizeof(uint32_t));
        }
        
        BitSet new_head_bs;
        new_head_bs.num_words = ctx->num_words;
        new_head_bs.words = &ctx->depth_tidlists[(depth + 1) * ctx->num_words];
        intersect_bitsets_store(&new_head_bs, head_bs, &ctx->db[item]);
        
        dfs_max_miner(new_head, head_len + 1, &new_head_bs, new_tail, tail_len - i - 1, ctx, depth + 2);
        
        free(new_head);
        if (new_tail) free(new_tail);
    }
    
    uint32_t greatest_item = tail[tail_len - 1];
    uint32_t *new_head = malloc((head_len + 1) * sizeof(uint32_t));
    if (head_len > 0) memcpy(new_head, head, head_len * sizeof(uint32_t));
    new_head[head_len] = greatest_item;
    f_add(&ctx->f_reg, new_head, head_len + 1, NULL, 0, ctx->item_num_words);
    free(new_head);
    
    free(tail_sup);
}

static uint32_t *g_counts = NULL;
static int cmp_freq_asc(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    if (g_counts[ia] < g_counts[ib]) return -1;
    if (g_counts[ia] > g_counts[ib]) return 1;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_MAX_MINER_Params *mm_params = (DM_MAX_MINER_Params *)params;
    double min_sup_param = mm_params ? mm_params->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[Max-Miner] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }

    uint32_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) freq_count++;
    }

    if (freq_count == 0) {
        printf("[Max-Miner] Complete. Total maximal frequent itemsets found: 0\n");
        free(counts);
        return DM_SUCCESS;
    }

    size_t num_words = (ds->count + 63) / 64;
    BitSet *db = malloc((ds->max_id + 1) * sizeof(BitSet));
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            db[i].words = calloc(num_words, sizeof(uint64_t));
            db[i].num_words = num_words;
        } else {
            db[i].words = NULL;
            db[i].num_words = 0;
        }
    }

    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            if (counts[item] >= min_sup) {
                db[item].words[i / 64] |= (1ULL << (i % 64));
            }
        }
    }

    uint32_t *F1 = malloc(freq_count * sizeof(uint32_t));
    size_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) F1[idx++] = i;
    }
    
    g_counts = counts;
    qsort(F1, freq_count, sizeof(uint32_t), cmp_freq_asc);

    MaxMinerContext ctx = {0};
    ctx.db = db;
    ctx.num_words = num_words;
    ctx.min_sup = min_sup;
    ctx.item_num_words = (ds->max_id + 64) / 64;
    ctx.f_reg.capacity = 1024;
    ctx.f_reg.itemsets = calloc(ctx.f_reg.capacity, sizeof(ItemBitSet));
    ctx.depth_tidlists = malloc(freq_count * 2 * num_words * sizeof(uint64_t));

    for (size_t i = 0; i < freq_count - 1; i++) {
        uint32_t item = F1[i];
        
        uint32_t *head = malloc(sizeof(uint32_t));
        head[0] = item;
        
        uint32_t tail_len = freq_count - i - 1;
        uint32_t *tail = malloc(tail_len * sizeof(uint32_t));
        memcpy(tail, F1 + i + 1, tail_len * sizeof(uint32_t));
        
        BitSet head_bs;
        head_bs.num_words = num_words;
        head_bs.words = &ctx.depth_tidlists[0];
        for (size_t w = 0; w < num_words; w++) head_bs.words[w] = ctx.db[item].words[w];
        
        dfs_max_miner(head, 1, &head_bs, tail, tail_len, &ctx, 1);
        
        free(head);
        free(tail);
    }
    
    if (freq_count > 0) {
        uint32_t greatest = F1[freq_count - 1];
        f_add(&ctx.f_reg, &greatest, 1, NULL, 0, ctx.item_num_words);
    }

    size_t actual_mfi_count = 0;
    size_t total_footprint = 0;
    for (size_t i = 0; i < ctx.f_reg.count; i++) {
        if (ctx.f_reg.itemsets[i].words != NULL) {
            actual_mfi_count++;
            total_footprint += ctx.f_reg.itemsets[i].num_items;
        }
    }

    printf("[Max-Miner] Complete. Total maximal frequent itemsets found: %zu\n", actual_mfi_count);
    dm_bench_record_results(actual_mfi_count, total_footprint);

    for (size_t i = 0; i < ctx.f_reg.count; i++) {
        if (ctx.f_reg.itemsets[i].words != NULL) {
            free(ctx.f_reg.itemsets[i].words);
        }
    }
    free(ctx.f_reg.itemsets);
    free(ctx.depth_tidlists);
    free(F1);
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (db[i].words) free(db[i].words);
    }
    free(db);
    free(counts);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "max_miner",
    .name = "Max-Miner Algorithm",
    .description = "Efficiently Mining Long Patterns (Maximal Frequent Itemsets) from Databases.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
