#include "algorithms/mafia.h"
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
    ItemBitSet **items;
    size_t count;
    size_t capacity;
    size_t total_footprint;
} MFI;

typedef struct {
    uint32_t min_sup;
    size_t item_num_words;
    size_t num_words;
    ItemBitSet *hut_bs;
} MafiaContext;

typedef struct {
    uint32_t *head;
    size_t head_size;
    uint32_t *tail_items;
    size_t tail_size;
    BitSet *bm;
    uint32_t support;
} MafiaNode;

typedef struct {
    uint32_t item;
    uint32_t support;
    BitSet *bm;
} TailElement;

static inline BitSet* alloc_bitset(size_t num_words) {
    BitSet *bs = malloc(sizeof(BitSet));
    bs->words = calloc(num_words, sizeof(uint64_t));
    bs->num_words = num_words;
    return bs;
}

static inline void free_bitset(BitSet *bs) {
    if (bs) {
        free(bs->words);
        free(bs);
    }
}

static inline size_t popcount_bitset(const BitSet *bs) {
    size_t count = 0;
    for (size_t i = 0; i < bs->num_words; i++) {
        count += (size_t)__builtin_popcountll(bs->words[i]);
    }
    return count;
}

static inline void intersect_bitsets(BitSet *out, const BitSet *a, const BitSet *b) {
    for (size_t i = 0; i < a->num_words; i++) {
        out->words[i] = a->words[i] & b->words[i];
    }
}

static inline void itemset_set_bit(ItemBitSet *bs, uint32_t item) {
    bs->words[item / 64] |= (1ULL << (item % 64));
}

static inline void itemset_clear_bit(ItemBitSet *bs, uint32_t item) {
    bs->words[item / 64] &= ~(1ULL << (item % 64));
}

static inline bool itemset_is_subset(const ItemBitSet *sub, const ItemBitSet *sup, size_t num_words) {
    for (size_t i = 0; i < num_words; i++) {
        if ((sub->words[i] & ~sup->words[i]) != 0) return false;
    }
    return true;
}

static bool mfi_has_superset(MFI *mfi, const ItemBitSet *bs, size_t num_words) {
    for (size_t i = 0; i < mfi->count; i++) {
        if (bs->num_items <= mfi->items[i]->num_items &&
            itemset_is_subset(bs, mfi->items[i], num_words)) {
            return true;
        }
    }
    return false;
}

static void mfi_add(MFI *mfi, uint32_t *head, size_t head_size, size_t item_num_words) {
    if (mfi->count >= mfi->capacity) {
        mfi->capacity = mfi->capacity == 0 ? 1024 : mfi->capacity * 2;
        mfi->items = realloc(mfi->items, mfi->capacity * sizeof(ItemBitSet*));
    }
    
    ItemBitSet *bs = malloc(sizeof(ItemBitSet));
    bs->words = calloc(item_num_words, sizeof(uint64_t));
    bs->num_items = head_size;
    for (size_t i = 0; i < head_size; i++) {
        itemset_set_bit(bs, head[i]);
    }
    
    mfi->items[mfi->count++] = bs;
    mfi->total_footprint += head_size;
}

static int cmp_tail(const void *a, const void *b) {
    TailElement *ta = (TailElement *)a;
    TailElement *tb = (TailElement *)b;
    if (ta->support < tb->support) return -1;
    if (ta->support > tb->support) return 1;
    if (ta->item < tb->item) return -1;
    if (ta->item > tb->item) return 1;
    return 0;
}

static bool mafia_dfs(MafiaNode *C, MFI *mfi, bool is_leftmost, MafiaContext *ctx, BitSet **db) {
    if (C->tail_size > 0) {
        for (size_t i = 0; i < C->head_size; i++) itemset_set_bit(ctx->hut_bs, C->head[i]);
        for (size_t i = 0; i < C->tail_size; i++) itemset_set_bit(ctx->hut_bs, C->tail_items[i]);
        ctx->hut_bs->num_items = C->head_size + C->tail_size;
        
        bool has_super = mfi_has_superset(mfi, ctx->hut_bs, ctx->item_num_words);
        
        for (size_t i = 0; i < C->head_size; i++) itemset_clear_bit(ctx->hut_bs, C->head[i]);
        for (size_t i = 0; i < C->tail_size; i++) itemset_clear_bit(ctx->hut_bs, C->tail_items[i]);
        ctx->hut_bs->num_items = 0;
        
        if (has_super) return false;
    }
    
    size_t original_tail_size = C->tail_size;
    size_t frequent_count = 0;
    
    TailElement *trimmed = NULL;
    if (original_tail_size > 0) {
        trimmed = malloc(original_tail_size * sizeof(TailElement));
    }
    size_t trimmed_size = 0;
    
    for (size_t i = 0; i < original_tail_size; i++) {
        uint32_t item = C->tail_items[i];
        BitSet *new_bm = alloc_bitset(C->bm->num_words);
        intersect_bitsets(new_bm, C->bm, db[item]);
        uint32_t sup = popcount_bitset(new_bm);
        
        if (sup >= ctx->min_sup) {
            frequent_count++;
            if (sup == C->support) {
                C->head[C->head_size++] = item;
                free_bitset(new_bm);
            } else {
                trimmed[trimmed_size].item = item;
                trimmed[trimmed_size].support = sup;
                trimmed[trimmed_size].bm = new_bm;
                trimmed_size++;
            }
        } else {
            free_bitset(new_bm);
        }
    }
    
    bool all_frequent = (frequent_count == original_tail_size);
    
    if (trimmed_size == 0) {
        for (size_t i = 0; i < C->head_size; i++) itemset_set_bit(ctx->hut_bs, C->head[i]);
        ctx->hut_bs->num_items = C->head_size;
        
        if (!mfi_has_superset(mfi, ctx->hut_bs, ctx->item_num_words)) {
            mfi_add(mfi, C->head, C->head_size, ctx->item_num_words);
        }
        
        for (size_t i = 0; i < C->head_size; i++) itemset_clear_bit(ctx->hut_bs, C->head[i]);
        ctx->hut_bs->num_items = 0;
        
        if (trimmed) free(trimmed);
        return (is_leftmost && all_frequent);
    }
    
    qsort(trimmed, trimmed_size, sizeof(TailElement), cmp_tail);
    
    bool stop_search = false;
    for (size_t i = 0; i < trimmed_size; i++) {
        bool child_leftmost = is_leftmost && (i == 0);
        
        MafiaNode child;
        child.head = malloc((C->head_size + trimmed_size) * sizeof(uint32_t));
        if (C->head_size > 0) memcpy(child.head, C->head, C->head_size * sizeof(uint32_t));
        child.head[C->head_size] = trimmed[i].item;
        child.head_size = C->head_size + 1;
        
        child.bm = trimmed[i].bm;
        child.support = trimmed[i].support;
        
        if (trimmed_size - i - 1 > 0) {
            child.tail_items = malloc((trimmed_size - i - 1) * sizeof(uint32_t));
            for (size_t j = i + 1; j < trimmed_size; j++) {
                child.tail_items[j - i - 1] = trimmed[j].item;
            }
        } else {
            child.tail_items = NULL;
        }
        child.tail_size = trimmed_size - i - 1;
        
        bool child_stopped = mafia_dfs(&child, mfi, child_leftmost, ctx, db);
        
        if (child.head) free(child.head);
        if (child.tail_items) free(child.tail_items);
        
        if (child_stopped) {
            stop_search = true;
            break;
        }
    }
    
    for (size_t i = 0; i < trimmed_size; i++) {
        free_bitset(trimmed[i].bm);
    }
    if (trimmed) free(trimmed);
    
    return (is_leftmost && all_frequent && stop_search);
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
    DM_MAFIA_Params *mafia_params = (DM_MAFIA_Params *)params;
    double min_sup_param = mafia_params ? mafia_params->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[MAFIA] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

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
        printf("[MAFIA] Complete. Total maximal frequent itemsets found: 0\n");
        free(counts);
        return DM_SUCCESS;
    }

    MafiaContext ctx = {0};
    ctx.min_sup = min_sup;
    ctx.item_num_words = (ds->max_id + 64) / 64;
    ctx.num_words = (ds->count + 63) / 64;
    ctx.hut_bs = malloc(sizeof(ItemBitSet));
    ctx.hut_bs->words = calloc(ctx.item_num_words, sizeof(uint64_t));
    ctx.hut_bs->num_items = 0;

    BitSet **db = malloc((ds->max_id + 1) * sizeof(BitSet*));
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            db[i] = alloc_bitset(ctx.num_words);
        } else {
            db[i] = NULL;
        }
    }

    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            if (counts[item] >= min_sup) {
                db[item]->words[i / 64] |= (1ULL << (i % 64));
            }
        }
    }

    uint32_t *L1 = malloc(freq_count * sizeof(uint32_t));
    size_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) L1[idx++] = i;
    }
    
    g_counts = counts;
    qsort(L1, freq_count, sizeof(uint32_t), cmp_freq_asc);

    MFI mfi = {0};

    MafiaNode root;
    root.head = malloc(freq_count * sizeof(uint32_t));
    root.head_size = 0;
    
    root.tail_items = malloc(freq_count * sizeof(uint32_t));
    root.tail_size = freq_count;
    for (size_t i = 0; i < freq_count; i++) {
        root.tail_items[i] = L1[i];
    }
    
    root.bm = alloc_bitset(ctx.num_words);
    for (size_t i = 0; i < ds->count; i++) {
        root.bm->words[i / 64] |= (1ULL << (i % 64));
    }
    root.support = ds->count;

    mafia_dfs(&root, &mfi, true, &ctx, db);

    printf("[MAFIA] Complete. Total maximal frequent itemsets found: %zu\n", mfi.count);
    dm_bench_record_results(mfi.count, mfi.total_footprint);

    free(root.head);
    free(root.tail_items);
    free_bitset(root.bm);

    for (size_t i = 0; i < mfi.count; i++) {
        free(mfi.items[i]->words);
        free(mfi.items[i]);
    }
    if (mfi.items) free(mfi.items);
    
    free(ctx.hut_bs->words);
    free(ctx.hut_bs);
    
    free(L1);
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (db[i]) free_bitset(db[i]);
    }
    free(db);
    free(counts);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "mafia",
    .name = "MAFIA Algorithm",
    .description = "MAximal Frequent Itemset Algorithm using Vertical Bitmaps, PEP, HUTMFI, and FHUT Pruning.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
