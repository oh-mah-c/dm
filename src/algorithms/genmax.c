#include "algorithms/genmax.h"
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
    ItemBitSet **global_mfi;
    size_t global_mfi_count;
    size_t global_mfi_capacity;
    uint32_t min_sup;
    size_t item_num_words;
    size_t total_footprint;
    size_t num_words; // For tidsets
} GenMaxContext;

static inline BitSet *alloc_bitset(GenMaxContext *ctx) {
    BitSet *bs = malloc(sizeof(BitSet));
    bs->words = calloc(ctx->num_words, sizeof(uint64_t));
    bs->num_words = ctx->num_words;
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

static inline void intersect_bitsets_store(BitSet *out, const BitSet *a, const BitSet *b) {
    for (size_t i = 0; i < a->num_words; i++) {
        out->words[i] = a->words[i] & b->words[i];
    }
}

static inline void itemset_set_bit(ItemBitSet *bs, uint32_t item) {
    bs->words[item / 64] |= (1ULL << (item % 64));
}

static inline bool itemset_has_bit(const ItemBitSet *bs, uint32_t item) {
    return (bs->words[item / 64] & (1ULL << (item % 64))) != 0;
}

static inline bool itemset_is_subset(const ItemBitSet *sub, const ItemBitSet *sup, size_t num_words) {
    for (size_t i = 0; i < num_words; i++) {
        if ((sub->words[i] & ~sup->words[i]) != 0) return false;
    }
    return true;
}

static void add_to_global_mfi(uint32_t *items, size_t count, GenMaxContext *ctx) {
    if (ctx->global_mfi_count >= ctx->global_mfi_capacity) {
        ctx->global_mfi_capacity = ctx->global_mfi_capacity == 0 ? 1024 : ctx->global_mfi_capacity * 2;
        ctx->global_mfi = realloc(ctx->global_mfi, ctx->global_mfi_capacity * sizeof(ItemBitSet*));
    }
    
    ItemBitSet *new_mfi = malloc(sizeof(ItemBitSet));
    new_mfi->words = calloc(ctx->item_num_words, sizeof(uint64_t));
    new_mfi->num_items = count;
    for (size_t i = 0; i < count; i++) {
        itemset_set_bit(new_mfi, items[i]);
    }
    
    ctx->global_mfi[ctx->global_mfi_count++] = new_mfi;
    ctx->total_footprint += count;
}

static void genmax_dfs(uint32_t *prefix, size_t prefix_len,
                       uint32_t *E, BitSet **E_bs, size_t E_len,
                       ItemBitSet **LMFI_in, size_t LMFI_count,
                       GenMaxContext *ctx) {
    
    ItemBitSet **local_LMFI = NULL;
    if (LMFI_count > 0) {
        local_LMFI = malloc(LMFI_count * sizeof(ItemBitSet*));
        memcpy(local_LMFI, LMFI_in, LMFI_count * sizeof(ItemBitSet*));
    }
    
    for (size_t i = 0; i < E_len; i++) {
        uint32_t item_i = E[i];
        BitSet *bs_i = E_bs[i];
        
        uint32_t *E_new = NULL;
        BitSet **E_new_bs = NULL;
        size_t E_new_len = 0;
        
        if (i + 1 < E_len) {
            E_new = malloc((E_len - i - 1) * sizeof(uint32_t));
            E_new_bs = malloc((E_len - i - 1) * sizeof(BitSet*));
            
            for (size_t j = i + 1; j < E_len; j++) {
                uint32_t item_j = E[j];
                BitSet *bs_j = alloc_bitset(ctx);
                intersect_bitsets_store(bs_j, bs_i, E_bs[j]);
                
                if (popcount_bitset(bs_j) >= ctx->min_sup) {
                    E_new[E_new_len] = item_j;
                    E_new_bs[E_new_len] = bs_j;
                    E_new_len++;
                } else {
                    free_bitset(bs_j);
                }
            }
        }
        
        bool subsumed = false;
        ItemBitSet cand_bs;
        cand_bs.words = calloc(ctx->item_num_words, sizeof(uint64_t));
        cand_bs.num_items = prefix_len + 1 + E_new_len;
        for (size_t k = 0; k < prefix_len; k++) itemset_set_bit(&cand_bs, prefix[k]);
        itemset_set_bit(&cand_bs, item_i);
        for (size_t k = 0; k < E_new_len; k++) itemset_set_bit(&cand_bs, E_new[k]);
        
        for (size_t m = 0; m < LMFI_count; m++) {
            if (cand_bs.num_items <= local_LMFI[m]->num_items &&
                itemset_is_subset(&cand_bs, local_LMFI[m], ctx->item_num_words)) {
                subsumed = true;
                break;
            }
        }
        
        free(cand_bs.words);
        
        if (subsumed) {
            for (size_t k = 0; k < E_new_len; k++) free_bitset(E_new_bs[k]);
            if (E_new) free(E_new);
            if (E_new_bs) free(E_new_bs);
            continue;
        }
        
        if (E_new_len == 0) {
            uint32_t *new_mfi = malloc((prefix_len + 1) * sizeof(uint32_t));
            if (prefix_len > 0) memcpy(new_mfi, prefix, prefix_len * sizeof(uint32_t));
            new_mfi[prefix_len] = item_i;
            
            size_t old_count = ctx->global_mfi_count;
            add_to_global_mfi(new_mfi, prefix_len + 1, ctx);
            
            local_LMFI = realloc(local_LMFI, (LMFI_count + 1) * sizeof(ItemBitSet*));
            local_LMFI[LMFI_count++] = ctx->global_mfi[old_count];
            
            free(new_mfi);
        } else {
            uint32_t *new_prefix = malloc((prefix_len + 1) * sizeof(uint32_t));
            if (prefix_len > 0) memcpy(new_prefix, prefix, prefix_len * sizeof(uint32_t));
            new_prefix[prefix_len] = item_i;
            
            ItemBitSet **LMFI_new = NULL;
            size_t LMFI_new_count = 0;
            if (LMFI_count > 0) {
                LMFI_new = malloc(LMFI_count * sizeof(ItemBitSet*));
                for (size_t m = 0; m < LMFI_count; m++) {
                    if (itemset_has_bit(local_LMFI[m], item_i)) {
                        LMFI_new[LMFI_new_count++] = local_LMFI[m];
                    }
                }
            }
            
            size_t old_global_mfi_count = ctx->global_mfi_count;
            
            genmax_dfs(new_prefix, prefix_len + 1, E_new, E_new_bs, E_new_len, LMFI_new, LMFI_new_count, ctx);
            
            size_t new_mfis_added = ctx->global_mfi_count - old_global_mfi_count;
            if (new_mfis_added > 0) {
                local_LMFI = realloc(local_LMFI, (LMFI_count + new_mfis_added) * sizeof(ItemBitSet*));
                for (size_t m = 0; m < new_mfis_added; m++) {
                    local_LMFI[LMFI_count++] = ctx->global_mfi[old_global_mfi_count + m];
                }
            }
            
            free(new_prefix);
            if (LMFI_new) free(LMFI_new);
        }
        
        for (size_t k = 0; k < E_new_len; k++) free_bitset(E_new_bs[k]);
        if (E_new) free(E_new);
        if (E_new_bs) free(E_new_bs);
    }
    
    if (local_LMFI) free(local_LMFI);
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
    DM_GENMAX_Params *gm_params = (DM_GENMAX_Params *)params;
    double min_sup_param = gm_params ? gm_params->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0 && ds->count > 0) min_sup = 1;

    printf("[GenMax] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

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
        printf("[GenMax] Complete. Total maximal frequent itemsets found: 0\n");
        free(counts);
        return DM_SUCCESS;
    }

    GenMaxContext ctx = {0};
    ctx.min_sup = min_sup;
    ctx.item_num_words = (ds->max_id + 64) / 64;
    ctx.num_words = (ds->count + 63) / 64;
    
    BitSet **db = malloc((ds->max_id + 1) * sizeof(BitSet*));
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            db[i] = alloc_bitset(&ctx);
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
    
    BitSet **L1_bs = malloc(freq_count * sizeof(BitSet*));
    for (size_t i = 0; i < freq_count; i++) {
        L1_bs[i] = db[L1[i]];
    }

    genmax_dfs(NULL, 0, L1, L1_bs, freq_count, NULL, 0, &ctx);

    printf("[GenMax] Complete. Total maximal frequent itemsets found: %zu\n", ctx.global_mfi_count);
    dm_bench_record_results(ctx.global_mfi_count, ctx.total_footprint);

    for (size_t i = 0; i < ctx.global_mfi_count; i++) {
        free(ctx.global_mfi[i]->words);
        free(ctx.global_mfi[i]);
    }
    if (ctx.global_mfi) free(ctx.global_mfi);
    
    free(L1);
    free(L1_bs);
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (db[i]) free_bitset(db[i]);
    }
    free(db);
    free(counts);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "genmax",
    .name = "GenMax Algorithm",
    .description = "Efficiently Mining Maximal Frequent Itemsets using Progressive Focusing and Vertical Format.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
