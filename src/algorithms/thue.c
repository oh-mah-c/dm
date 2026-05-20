#include "algorithms/thue.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* --- Data Structures --- */

typedef struct {
    size_t Ts;
    size_t Te;
    double utility;
} THUE_MO;

typedef struct {
    THUE_MO *mos;
    size_t count;
    size_t capacity;
} THUE_MOSet;

typedef struct {
    uint32_t *items;
    size_t count;
} THUE_Itemset;

typedef struct {
    THUE_Itemset *itemsets;
    size_t count;
} THUE_Episode;

typedef struct {
    THUE_Episode *ep;
    double utility;
} THUE_HeapNode;

typedef struct {
    THUE_HeapNode *array;
    size_t capacity;
    size_t size;
} THUE_MinHeap;

typedef struct {
    DM_Trans_Utility *transactions;
    size_t txn_count;
    uint32_t max_item_id;
    
    double *tu; // Total utility of each transaction
    double TU;  // Total utility of whole dataset
    
    double *rtu; // Utility of each transaction for RUS
    double *riu; // Real utility of each item for RUS
    
    double minUtil;
    size_t k;
    size_t MTD;
    
    THUE_MinHeap heap;
    DM_THUE_Stats *stats;
    time_t start_time;
} THUE_Context;

/* --- Utility Functions --- */

static THUE_MOSet create_moset(size_t cap) {
    THUE_MOSet ms;
    ms.count = 0;
    ms.capacity = cap > 0 ? cap : 4;
    ms.mos = (THUE_MO*)malloc(ms.capacity * sizeof(THUE_MO));
    return ms;
}

static void add_mo(THUE_MOSet *ms, size_t Ts, size_t Te, double util) {
    if (ms->count == ms->capacity) {
        ms->capacity *= 2;
        ms->mos = (THUE_MO*)realloc(ms->mos, ms->capacity * sizeof(THUE_MO));
    }
    ms->mos[ms->count].Ts = Ts;
    ms->mos[ms->count].Te = Te;
    ms->mos[ms->count].utility = util;
    ms->count++;
}

static void free_moset(THUE_MOSet *ms) {
    if (ms->mos) free(ms->mos);
    ms->mos = NULL;
    ms->count = 0;
    ms->capacity = 0;
}

static THUE_Episode* clone_episode(const THUE_Episode *ep) {
    THUE_Episode *copy = (THUE_Episode*)malloc(sizeof(THUE_Episode));
    copy->count = ep->count;
    copy->itemsets = (THUE_Itemset*)malloc(copy->count * sizeof(THUE_Itemset));
    for (size_t i = 0; i < ep->count; i++) {
        copy->itemsets[i].count = ep->itemsets[i].count;
        copy->itemsets[i].items = (uint32_t*)malloc(copy->itemsets[i].count * sizeof(uint32_t));
        memcpy(copy->itemsets[i].items, ep->itemsets[i].items, copy->itemsets[i].count * sizeof(uint32_t));
    }
    return copy;
}

static void free_episode(THUE_Episode *ep) {
    if (!ep) return;
    for (size_t i = 0; i < ep->count; i++) {
        free(ep->itemsets[i].items);
    }
    free(ep->itemsets);
    free(ep);
}

static void free_episode_contents(THUE_Episode *ep) {
    if (!ep) return;
    for (size_t i = 0; i < ep->count; i++) {
        free(ep->itemsets[i].items);
    }
    free(ep->itemsets);
    ep->itemsets = NULL;
    ep->count = 0;
}

/* --- Min Heap --- */

static void heap_init(THUE_MinHeap *h, size_t cap) {
    h->size = 0;
    h->capacity = cap > 0 ? cap : 1;
    h->array = (THUE_HeapNode*)malloc(h->capacity * sizeof(THUE_HeapNode));
}

static void heap_free(THUE_MinHeap *h) {
    for (size_t i = 0; i < h->size; i++) {
        free_episode(h->array[i].ep);
    }
    free(h->array);
}

static int episode_equals(const THUE_Episode *a, const THUE_Episode *b) {
    if (!a || !b || a->count != b->count) return 0;
    for (size_t i = 0; i < a->count; i++) {
        if (a->itemsets[i].count != b->itemsets[i].count) return 0;
        for (size_t j = 0; j < a->itemsets[i].count; j++) {
            if (a->itemsets[i].items[j] != b->itemsets[i].items[j]) return 0;
        }
    }
    return 1;
}

static int cmp_heap_node_desc(const void *a, const void *b) {
    const THUE_HeapNode *x = (const THUE_HeapNode *)a;
    const THUE_HeapNode *y = (const THUE_HeapNode *)b;
    if (x->utility < y->utility) return 1;
    if (x->utility > y->utility) return -1;
    return 0;
}

static int topk_buffer_add(THUE_MinHeap *h, size_t k, const THUE_Episode *ep, double utility, double *threshold) {
    if (k == 0) return 0;
    if (h->size >= k && utility < *threshold) return 0;
    for (size_t i = 0; i < h->size; i++) {
        if (episode_equals(h->array[i].ep, ep)) {
            if (utility > h->array[i].utility) h->array[i].utility = utility;
            qsort(h->array, h->size, sizeof(THUE_HeapNode), cmp_heap_node_desc);
            if (h->size >= k && h->array[k - 1].utility > *threshold) *threshold = h->array[k - 1].utility;
            return 0;
        }
    }
    if (h->size == h->capacity) {
        size_t nc = h->capacity * 2;
        THUE_HeapNode *na = (THUE_HeapNode *)realloc(h->array, nc * sizeof(THUE_HeapNode));
        if (!na) return -1;
        h->array = na;
        h->capacity = nc;
    }
    h->array[h->size].ep = clone_episode(ep);
    h->array[h->size].utility = utility;
    h->size++;
    qsort(h->array, h->size, sizeof(THUE_HeapNode), cmp_heap_node_desc);
    if (h->size >= k) {
        double kth = h->array[k - 1].utility;
        if (kth > *threshold) *threshold = kth;
        size_t keep = h->size;
        while (keep > 0 && h->array[keep - 1].utility < *threshold) {
            free_episode(h->array[keep - 1].ep);
            keep--;
        }
        h->size = keep;
    }
    return 0;
}

static void RUC(THUE_Context *ctx, const THUE_Episode *ep, double utility) {
    if (utility >= ctx->minUtil) {
        if (topk_buffer_add(&ctx->heap, ctx->k, ep, utility, &ctx->minUtil) == 0 &&
            ctx->heap.size >= ctx->k) {
            ctx->minUtil = ctx->heap.array[ctx->k - 1].utility;
        }
    }
}

/* --- RUS Strategies --- */

static int cmp_double_desc(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da > db) return -1;
    if (da < db) return 1;
    return 0;
}

static void RUS(THUE_Context *ctx, double *uList, size_t len) {
    if (len == 0) return;
    
    double *tmp = (double*)malloc(len * sizeof(double));
    memcpy(tmp, uList, len * sizeof(double));
    qsort(tmp, len, sizeof(double), cmp_double_desc);
    
    double new_min = 0.0;
    if (len >= ctx->k) {
        new_min = tmp[ctx->k - 1];
    } else {
        new_min = tmp[len - 1];
    }
    
    if (new_min > ctx->minUtil) {
        ctx->minUtil = new_min;
    }
    free(tmp);
}

/* --- EWU and Helpers --- */

static double get_item_utility(THUE_Context *ctx, size_t tid, uint32_t item) {
    DM_Trans_Utility *t = &ctx->transactions[tid];
    double utility = 0.0;
    for (size_t i = 0; i < t->count; i++) {
        if (t->items[i].id == item) utility += t->items[i].utility;
    }
    return utility;
}

static int contains_item(DM_Trans_Utility *t, uint32_t item) {
    for (size_t i = 0; i < t->count; i++) {
        if (t->items[i].id == item) return 1;
    }
    return 0;
}

static double remaining_utility(THUE_Context *ctx, size_t tid, const THUE_Episode *ep) {
    DM_Trans_Utility *t = &ctx->transactions[tid];
    const THUE_Itemset *last_se = &ep->itemsets[ep->count - 1];
    uint32_t last_item = last_se->items[last_se->count - 1];
    
    double ru = 0.0;
    for (size_t i = 0; i < t->count; i++) {
        if (t->items[i].id > last_item) {
            // Check if it's not in alpha
            int in_alpha = 0;
            for (size_t j = 0; j < ep->count; j++) {
                for (size_t k = 0; k < ep->itemsets[j].count; k++) {
                    if (ep->itemsets[j].items[k] == t->items[i].id) {
                        in_alpha = 1;
                        break;
                    }
                }
                if (in_alpha) break;
            }
            if (!in_alpha) {
                ru += t->items[i].utility;
            }
        }
    }
    return ru;
}

static void compute_EWU_opt(THUE_Context *ctx, const THUE_Episode *ep, const THUE_MOSet *moSet, double *out_utility, double *out_ewu) {
    double total_util = 0.0;
    double ewu_opt = 0.0;
    
    for (size_t i = 0; i < moSet->count; i++) {
        total_util += moSet->mos[i].utility;
        
        double ru = remaining_utility(ctx, moSet->mos[i].Te, ep);
        double suffix_tu = 0.0;
        size_t limit = moSet->mos[i].Ts + ctx->MTD;
        if (limit >= ctx->txn_count) limit = ctx->txn_count - 1;
        
        for (size_t t = moSet->mos[i].Te + 1; t <= limit; t++) {
            suffix_tu += ctx->tu[t];
        }
        
        ewu_opt += moSet->mos[i].utility + ru + suffix_tu;
    }
    
    *out_utility = total_util;
    *out_ewu = ewu_opt;
}

static void filter_minimal(THUE_MOSet *moSet) {
    if (moSet->count <= 1) return;
    
    THUE_MOSet filtered = create_moset(moSet->count);
    for (size_t i = 0; i < moSet->count; i++) {
        int is_minimal = 1;
        for (size_t j = 0; j < moSet->count; j++) {
            if (i == j) continue;
            // Paper Definition 3.6 uses a strict sub-time interval:
            // [Ts',Te'] is inside [Ts,Te] iff Ts < Ts' and Te' < Te.
            if (moSet->mos[i].Ts < moSet->mos[j].Ts && moSet->mos[j].Te < moSet->mos[i].Te) {
                is_minimal = 0;
                break;
            }
        }
        if (is_minimal) {
            int duplicate = 0;
            for (size_t j = 0; j < filtered.count; j++) {
                if (filtered.mos[j].Ts == moSet->mos[i].Ts && filtered.mos[j].Te == moSet->mos[i].Te) {
                    duplicate = 1;
                    if (moSet->mos[i].utility > filtered.mos[j].utility) {
                        filtered.mos[j].utility = moSet->mos[i].utility;
                    }
                    break;
                }
            }
            if (!duplicate) add_mo(&filtered, moSet->mos[i].Ts, moSet->mos[i].Te, moSet->mos[i].utility);
        }
    }
    
    free(moSet->mos);
    moSet->mos = filtered.mos;
    moSet->count = filtered.count;
    moSet->capacity = filtered.capacity;
}

/* --- Core Spanning --- */

static void Span_SerialHUE(THUE_Context *ctx, const THUE_Episode *ep, const THUE_MOSet *moSet);

static void Span_SimultHUE(THUE_Context *ctx, const THUE_Episode *ep, const THUE_MOSet *moSet) {
    if (time(NULL) - ctx->start_time > 10000) {
        ctx->stats->timeout_or_memory_limit = 1;
        return;
    }
    
    ctx->stats->candidate_episodes_generated++;
    
    const THUE_Itemset *last_se = &ep->itemsets[ep->count - 1];
    uint32_t last_item = last_se->items[last_se->count - 1];
    
    // Collect candidate items
    uint32_t *cands = (uint32_t*)malloc((ctx->max_item_id + 1) * sizeof(uint32_t));
    size_t cand_count = 0;
    
    for (size_t i = 0; i < moSet->count; i++) {
        size_t Te = moSet->mos[i].Te;
        DM_Trans_Utility *t = &ctx->transactions[Te];
        for (size_t j = 0; j < t->count; j++) {
            if (t->items[j].id > last_item) {
                // Add to cands if not exists
                int exists = 0;
                for (size_t c = 0; c < cand_count; c++) {
                    if (cands[c] == t->items[j].id) {
                        exists = 1; break;
                    }
                }
                if (!exists) {
                    cands[cand_count++] = t->items[j].id;
                }
            }
        }
    }
    
    for (size_t c = 0; c < cand_count; c++) {
        uint32_t e = cands[c];
        
        THUE_MOSet beta_mos = create_moset(moSet->count);
        for (size_t i = 0; i < moSet->count; i++) {
            size_t Te = moSet->mos[i].Te;
            if (contains_item(&ctx->transactions[Te], e)) {
                double u_e = get_item_utility(ctx, Te, e);
                add_mo(&beta_mos, moSet->mos[i].Ts, Te, moSet->mos[i].utility + u_e);
            }
        }
        
        filter_minimal(&beta_mos);
        
        if (beta_mos.count > 0) {
            // Construct beta
            THUE_Episode beta;
            beta.count = ep->count;
            beta.itemsets = (THUE_Itemset*)malloc(beta.count * sizeof(THUE_Itemset));
            for (size_t i = 0; i < beta.count - 1; i++) {
                beta.itemsets[i].count = ep->itemsets[i].count;
                beta.itemsets[i].items = (uint32_t*)malloc(beta.itemsets[i].count * sizeof(uint32_t));
                memcpy(beta.itemsets[i].items, ep->itemsets[i].items, beta.itemsets[i].count * sizeof(uint32_t));
            }
            
            beta.itemsets[beta.count - 1].count = last_se->count + 1;
            beta.itemsets[beta.count - 1].items = (uint32_t*)malloc(beta.itemsets[beta.count - 1].count * sizeof(uint32_t));
            memcpy(beta.itemsets[beta.count - 1].items, last_se->items, last_se->count * sizeof(uint32_t));
            beta.itemsets[beta.count - 1].items[last_se->count] = e;
            
            // Calculate EWU
            double util, ewu;
            compute_EWU_opt(ctx, &beta, &beta_mos, &util, &ewu);
            
            if (ewu >= ctx->minUtil) {
                RUC(ctx, &beta, util);
                Span_SimultHUE(ctx, &beta, &beta_mos);
                Span_SerialHUE(ctx, &beta, &beta_mos);
            }
            
            free_episode_contents(&beta); // RUC clones retained winners.
        }
        free_moset(&beta_mos);
    }
    free(cands);
}

static void Span_SerialHUE(THUE_Context *ctx, const THUE_Episode *ep, const THUE_MOSet *moSet) {
    if (time(NULL) - ctx->start_time > 10000) {
        ctx->stats->timeout_or_memory_limit = 1;
        return;
    }
    
    ctx->stats->candidate_episodes_generated++;
    
    // Collect candidate items
    uint32_t *cands = (uint32_t*)malloc((ctx->max_item_id + 1) * sizeof(uint32_t));
    size_t cand_count = 0;
    
    for (size_t i = 0; i < moSet->count; i++) {
        size_t Te = moSet->mos[i].Te;
        size_t Ts = moSet->mos[i].Ts;
        size_t limit = Ts + ctx->MTD;
        if (limit >= ctx->txn_count) limit = ctx->txn_count - 1;
        
        for (size_t t = Te + 1; t <= limit; t++) {
            DM_Trans_Utility *trans = &ctx->transactions[t];
            for (size_t j = 0; j < trans->count; j++) {
                int exists = 0;
                for (size_t c = 0; c < cand_count; c++) {
                    if (cands[c] == trans->items[j].id) {
                        exists = 1; break;
                    }
                }
                if (!exists) {
                    cands[cand_count++] = trans->items[j].id;
                }
            }
        }
    }
    
    for (size_t c = 0; c < cand_count; c++) {
        uint32_t e = cands[c];
        
        THUE_MOSet beta_mos = create_moset(moSet->count);
        for (size_t i = 0; i < moSet->count; i++) {
            size_t Te = moSet->mos[i].Te;
            size_t Ts = moSet->mos[i].Ts;
            size_t limit = Ts + ctx->MTD;
            if (limit >= ctx->txn_count) limit = ctx->txn_count - 1;
            
            for (size_t t = Te + 1; t <= limit; t++) {
                if (contains_item(&ctx->transactions[t], e)) {
                    double u_e = get_item_utility(ctx, t, e);
                    add_mo(&beta_mos, Ts, t, moSet->mos[i].utility + u_e);
                }
            }
        }
        
        filter_minimal(&beta_mos);
        
        if (beta_mos.count > 0) {
            // Construct beta
            THUE_Episode beta;
            beta.count = ep->count + 1;
            beta.itemsets = (THUE_Itemset*)malloc(beta.count * sizeof(THUE_Itemset));
            for (size_t i = 0; i < ep->count; i++) {
                beta.itemsets[i].count = ep->itemsets[i].count;
                beta.itemsets[i].items = (uint32_t*)malloc(beta.itemsets[i].count * sizeof(uint32_t));
                memcpy(beta.itemsets[i].items, ep->itemsets[i].items, beta.itemsets[i].count * sizeof(uint32_t));
            }
            
            beta.itemsets[beta.count - 1].count = 1;
            beta.itemsets[beta.count - 1].items = (uint32_t*)malloc(sizeof(uint32_t));
            beta.itemsets[beta.count - 1].items[0] = e;
            
            // Calculate EWU
            double util, ewu;
            compute_EWU_opt(ctx, &beta, &beta_mos, &util, &ewu);
            
            if (ewu >= ctx->minUtil) {
                RUC(ctx, &beta, util);
                // I-Concatenation should be called first to save memory according to the paper
                Span_SimultHUE(ctx, &beta, &beta_mos);
                Span_SerialHUE(ctx, &beta, &beta_mos);
            }
            
            free_episode_contents(&beta);
        }
        free_moset(&beta_mos);
    }
    free(cands);
}

/* --- Entry Point --- */

static DM_Status thue_run(DM_Dataset *dataset, void *parameters) {
    if (dataset->type != DM_TYPE_UTILITY && dataset->type != DM_TYPE_SEQUENCE_UTILITY) {
        printf("THUE only supports utility datasets\n");
        return DM_ERROR_INCOMPATIBLE;
    }
    
    DM_THUE_Params *params = (DM_THUE_Params*)parameters;
    THUE_Context ctx;
    memset(&ctx, 0, sizeof(ctx));
    
    ctx.start_time = time(NULL);
    ctx.k = params->k;
    ctx.MTD = params->MTD;
    ctx.minUtil = 0.0;
    
    ctx.transactions = (DM_Trans_Utility*)dataset->payload;
    ctx.txn_count = dataset->count;
    ctx.max_item_id = dataset->max_id;
    
    heap_init(&ctx.heap, ctx.k);
    
    ctx.stats = (DM_THUE_Stats*)calloc(1, sizeof(DM_THUE_Stats));
    ctx.stats->transactions = ctx.txn_count;
    
    // Phase 1: Compute TU, RTU, RIU
    ctx.tu = (double*)calloc(ctx.txn_count, sizeof(double));
    ctx.rtu = (double*)calloc(ctx.txn_count, sizeof(double));
    ctx.riu = (double*)calloc(ctx.max_item_id + 1, sizeof(double));
    
    for (size_t i = 0; i < ctx.txn_count; i++) {
        double trans_util = 0.0;
        for (size_t j = 0; j < ctx.transactions[i].count; j++) {
            double u = ctx.transactions[i].items[j].utility;
            trans_util += u;
            ctx.riu[ctx.transactions[i].items[j].id] += u;
            ctx.stats->items_processed++;
        }
        ctx.tu[i] = trans_util;
        ctx.rtu[i] = trans_util;
        ctx.TU += trans_util;
    }
    
    ctx.stats->total_utility = ctx.TU;
    
    RUS(&ctx, ctx.rtu, ctx.txn_count);
    RUS(&ctx, ctx.riu, ctx.max_item_id + 1);
    
    ctx.stats->initial_min_util = ctx.minUtil;
    
    // Phase 2: Compute 1-episodes
    
    // Find all valid 1-items
    uint32_t *valid_items = (uint32_t*)malloc((ctx.max_item_id + 1) * sizeof(uint32_t));
    size_t valid_count = 0;
    for (uint32_t e = 0; e <= ctx.max_item_id; e++) {
        if (ctx.riu[e] > 0) { // e is present
            valid_items[valid_count++] = e;
        }
    }
    
    // We could sort valid_items by riu decreasing here, which helps pruning
    for (size_t i = 0; i < valid_count; i++) {
        for (size_t j = i + 1; j < valid_count; j++) {
            if (ctx.riu[valid_items[i]] < ctx.riu[valid_items[j]]) {
                uint32_t tmp = valid_items[i];
                valid_items[i] = valid_items[j];
                valid_items[j] = tmp;
            }
        }
    }
    
    for (size_t c = 0; c < valid_count; c++) {
        uint32_t e = valid_items[c];
        
        THUE_MOSet moSet = create_moset(10);
        for (size_t i = 0; i < ctx.txn_count; i++) {
            if (contains_item(&ctx.transactions[i], e)) {
                double u_e = get_item_utility(&ctx, i, e);
                add_mo(&moSet, i, i, u_e);
            }
        }
        
        filter_minimal(&moSet);
        
        if (moSet.count > 0) {
            THUE_Episode ep;
            ep.count = 1;
            ep.itemsets = (THUE_Itemset*)malloc(sizeof(THUE_Itemset));
            ep.itemsets[0].count = 1;
            ep.itemsets[0].items = (uint32_t*)malloc(sizeof(uint32_t));
            ep.itemsets[0].items[0] = e;
            
            double util, ewu;
            compute_EWU_opt(&ctx, &ep, &moSet, &util, &ewu);
            
            if (ewu >= ctx.minUtil) {
                RUC(&ctx, &ep, util);
                Span_SimultHUE(&ctx, &ep, &moSet);
                Span_SerialHUE(&ctx, &ep, &moSet);
            }
            
            free(ep.itemsets[0].items);
            free(ep.itemsets);
        }
        free_moset(&moSet);
        
        if (ctx.stats->timeout_or_memory_limit) break;
    }
    
    ctx.stats->final_hues = ctx.heap.size;
    ctx.stats->final_min_util = ctx.minUtil;
    size_t total_episode_items = 0;
    for (size_t i = 0; i < ctx.heap.size; i++) {
        for (size_t j = 0; j < ctx.heap.array[i].ep->count; j++) {
            total_episode_items += ctx.heap.array[i].ep->itemsets[j].count;
        }
    }
    dm_bench_record_results(ctx.stats->final_hues, total_episode_items);
    
    printf("\nTHUE Mining Complete:\n");
    printf("Timeout/OOM: %s\n", ctx.stats->timeout_or_memory_limit ? "Yes" : "No");
    printf("Initial minUtil: %.2f\n", ctx.stats->initial_min_util);
    printf("Final minUtil: %.2f\n", ctx.stats->final_min_util);
    printf("Candidate Episodes Evaluated: %zu\n", ctx.stats->candidate_episodes_generated);
    printf("Top-K Found: %zu\n", ctx.stats->final_hues);
    
    // Sort and print the top K
    double *res_utils = (double*)malloc(ctx.heap.size * sizeof(double));
    for (size_t i = 0; i < ctx.heap.size; i++) {
        res_utils[i] = ctx.heap.array[i].utility;
    }
    qsort(res_utils, ctx.heap.size, sizeof(double), cmp_double_desc);
    
    printf("Top Utilities: ");
    for (size_t i = 0; i < ctx.heap.size && i < 10; i++) {
        printf("%.2f ", res_utils[i]);
    }
    printf("\n");
    
    free(res_utils);
    free(valid_items);
    free(ctx.tu);
    free(ctx.rtu);
    free(ctx.riu);
    heap_free(&ctx.heap);
    free(ctx.stats);
    
    return DM_SUCCESS;
}

DM_Algorithm thue_algo = {
    .id = "thue",
    .name = "THUE",
    .description = "Top-k High Utility Episode mining with RIU/RTU/RUC threshold raising and optimized EWU pruning (Wan et al. 2021).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = thue_run
};

DM_REGISTER_ALGORITHM(thue_algo)
