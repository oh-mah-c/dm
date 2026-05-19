#include "algorithms/tmku.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include "core/dm_algorithm.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t tid;
    double iutil;
    double rutil;
} UtilityTuple;

typedef struct {
    uint32_t *items;
    size_t count;
    UtilityTuple *tuples;
    size_t tuple_count;
    double sum_iutil;
    double sum_rutil;
} UtilityList;

typedef struct {
    uint32_t id;
    double twu;
} ItemTWU;

typedef struct TPNode {
    uint32_t item;
    double twu;
    double sumIu;
    double sumRu;
    bool isEnd;
    struct TPNode *parent;
    struct TPNode **children;
    size_t child_count;
    size_t child_cap;
    struct TPNode *link;
} TPNode;

typedef struct {
    uint32_t *items;
    size_t len;
    double utility;
} Pattern;

typedef struct {
    Pattern *data;
    size_t count;
    size_t cap;
    size_t k;
    double threshold;
    size_t threshold_raises;
} TopK;

/* --- GLOBAL/STATIC UTILS --- */

static int cmp_item_twu(const void *a, const void *b) {
    double twu1 = ((ItemTWU*)a)->twu;
    double twu2 = ((ItemTWU*)b)->twu;
    if (twu1 < twu2) return -1;
    if (twu1 > twu2) return 1;
    return (int)(((ItemTWU*)a)->id - ((ItemTWU*)b)->id);
}

static uint32_t *g_rank = NULL;

static int cmp_target_rank(const void *a, const void *b) {
    uint32_t item_a = *(const uint32_t *)a;
    uint32_t item_b = *(const uint32_t *)b;
    uint32_t r_a = g_rank[item_a];
    uint32_t r_b = g_rank[item_b];
    return (r_a < r_b) ? -1 : ((r_a > r_b) ? 1 : 0);
}

static int cmp_pattern_desc(const void *a, const void *b) {
    const Pattern *x = (const Pattern *)a;
    const Pattern *y = (const Pattern *)b;
    if (x->utility < y->utility) return 1;
    if (x->utility > y->utility) return -1;
    if (x->len < y->len) return -1;
    if (x->len > y->len) return 1;
    return 0;
}

/* --- UTILITY LIST CONSTRUCTION --- */

static UtilityList* construct(UtilityList *p, UtilityList *px, UtilityList *py) {
    UtilityList *pxy = malloc(sizeof(UtilityList));
    pxy->count = px->count + 1;
    pxy->items = malloc(sizeof(uint32_t) * pxy->count);
    memcpy(pxy->items, px->items, sizeof(uint32_t) * px->count);
    pxy->items[px->count] = py->items[py->count - 1];
    
    size_t capacity = px->tuple_count < py->tuple_count ? px->tuple_count : py->tuple_count;
    pxy->tuples = malloc(sizeof(UtilityTuple) * capacity);
    pxy->tuple_count = 0;
    pxy->sum_iutil = 0;
    pxy->sum_rutil = 0;

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < px->tuple_count && iy < py->tuple_count) {
        if (px->tuples[ix].tid == py->tuples[iy].tid) {
            uint32_t tid = px->tuples[ix].tid;
            double iutil = px->tuples[ix].iutil + py->tuples[iy].iutil;
            
            if (p != NULL) {
                while (ip < p->tuple_count && p->tuples[ip].tid < tid) ip++;
                if (ip < p->tuple_count && p->tuples[ip].tid == tid) {
                    iutil -= p->tuples[ip].iutil;
                }
            }
            
            pxy->tuples[pxy->tuple_count].tid = tid;
            pxy->tuples[pxy->tuple_count].iutil = iutil;
            pxy->tuples[pxy->tuple_count].rutil = py->tuples[iy].rutil;
            pxy->sum_iutil += iutil;
            pxy->sum_rutil += py->tuples[iy].rutil;
            pxy->tuple_count++;
            ix++; iy++;
        } else if (px->tuples[ix].tid < py->tuples[iy].tid) ix++;
        else iy++;
    }
    return pxy;
}

static void free_utility_list(UtilityList *ul) {
    if (!ul) return;
    free(ul->items);
    free(ul->tuples);
    free(ul);
}

/* --- TP-TREE FUNCTIONS --- */

static TPNode* create_node(uint32_t item, double twu, TPNode *parent) {
    TPNode *node = malloc(sizeof(TPNode));
    node->item = item;
    node->twu = twu;
    node->sumIu = 0;
    node->sumRu = 0;
    node->isEnd = false;
    node->parent = parent;
    node->children = NULL;
    node->child_count = 0;
    node->child_cap = 0;
    node->link = NULL;
    return node;
}

static void add_child(TPNode *parent, TPNode *child) {
    if (parent->child_count == parent->child_cap) {
        parent->child_cap = parent->child_cap ? parent->child_cap * 2 : 4;
        parent->children = realloc(parent->children, sizeof(TPNode*) * parent->child_cap);
    }
    parent->children[parent->child_count++] = child;
}

static TPNode* find_child(TPNode *parent, uint32_t item) {
    for (size_t i = 0; i < parent->child_count; i++) {
        if (parent->children[i]->item == item) {
            return parent->children[i];
        }
    }
    return NULL;
}

static void free_tree(TPNode *node) {
    if (!node) return;
    for (size_t i = 0; i < node->child_count; i++) {
        free_tree(node->children[i]);
    }
    free(node->children);
    free(node);
}

static void insert_tp_tree(TPNode *root, uint32_t *items, size_t count, double sum_iutil, double sum_rutil, double *twu_counts, TPNode **header_table, TPNode **tail_table, double min_util) {
    TPNode *curr = root;
    for (size_t i = 0; i < count; i++) {
        uint32_t item = items[i];
        TPNode *child = find_child(curr, item);
        if (!child) {
            child = create_node(item, twu_counts[item], curr);
            add_child(curr, child);
            if (header_table[item] == NULL) {
                header_table[item] = child;
                tail_table[item] = child;
            } else {
                tail_table[item]->link = child;
                tail_table[item] = child;
            }
        }
        curr = child;
    }
    curr->sumIu = sum_iutil;
    curr->sumRu = sum_rutil;
    if (sum_iutil >= min_util) {
        curr->isEnd = true;
    }
}

/* --- RECURSIVE MINING --- */

static void tmku_mine_recursive(UtilityList *p, UtilityList **extensions, size_t ext_count, double min_util, TPNode *root, double *twu_counts, TPNode **header_table, TPNode **tail_table, DM_TMKU_Stats *stats) {
    for (size_t i = 0; i < ext_count; i++) {
        UtilityList *px = extensions[i];
        stats->visited_nodes++;

        insert_tp_tree(root, px->items, px->count, px->sum_iutil, px->sum_rutil, twu_counts, header_table, tail_table, min_util);

        if (px->sum_iutil + px->sum_rutil >= min_util) {
            UtilityList **ex_px = malloc(sizeof(UtilityList*) * (ext_count - i - 1));
            size_t ex_px_count = 0;

            for (size_t j = i + 1; j < ext_count; j++) {
                UtilityList *py = extensions[j];
                UtilityList *pxy = construct(p, px, py);
                stats->joins++;
                stats->joined_entries += pxy->tuple_count;
                if (pxy->tuple_count > 0) {
                    ex_px[ex_px_count++] = pxy;
                } else {
                    free_utility_list(pxy);
                }
            }

            if (ex_px_count > 0) {
                tmku_mine_recursive(px, ex_px, ex_px_count, min_util, root, twu_counts, header_table, tail_table, stats);
            }

            for (size_t j = 0; j < ex_px_count; j++) free_utility_list(ex_px[j]);
            free(ex_px);
        } else {
            stats->pruned_subtree_utility++;
        }
    }
}

/* --- TOP-K BUFFER FUNCTIONS --- */

static void topk_init(TopK *t, size_t k) {
    t->k = k;
    t->count = 0;
    t->cap = 16;
    t->data = malloc(sizeof(Pattern) * t->cap);
    t->threshold = 0.0;
    t->threshold_raises = 0;
}

static void topk_free(TopK *t) {
    for (size_t i = 0; i < t->count; i++) {
        free(t->data[i].items);
    }
    free(t->data);
}

static bool same_items(const Pattern *p, const uint32_t *items, size_t len) {
    if (p->len != len) return false;
    for (size_t i = 0; i < len; i++) {
        if (p->items[i] != items[i]) return false;
    }
    return true;
}

static void topk_add(TopK *t, const uint32_t *items, size_t len, double utility) {
    if (t->k == 0) return;
    if (t->count >= t->k && utility < t->threshold) return;
    
    for (size_t i = 0; i < t->count; i++) {
        if (same_items(&t->data[i], items, len)) return;
    }
    
    if (t->count == t->cap) {
        t->cap *= 2;
        t->data = realloc(t->data, sizeof(Pattern) * t->cap);
    }
    t->data[t->count].items = malloc(sizeof(uint32_t) * len);
    memcpy(t->data[t->count].items, items, sizeof(uint32_t) * len);
    t->data[t->count].len = len;
    t->data[t->count].utility = utility;
    t->count++;
    
    qsort(t->data, t->count, sizeof(Pattern), cmp_pattern_desc);
    
    if (t->count > t->k) {
        free(t->data[t->count - 1].items);
        t->count--;
    }
    
    if (t->count == t->k) {
        double old = t->threshold;
        t->threshold = t->data[t->k - 1].utility;
        if (t->threshold > old) {
            t->threshold_raises++;
        }
    }
}

/* --- DISCOVERY PROCEDURE --- */

static void explore_descendants(TPNode *node, TopK *topk, DM_TMKU_Stats *stats) {
    if (node->sumIu + node->sumRu < topk->threshold) {
        stats->pruned_subtree_utility++;
        return;
    }
    if (node->isEnd) {
        size_t len = 0;
        TPNode *curr = node;
        while (curr->parent != NULL) {
            len++;
            curr = curr->parent;
        }
        uint32_t *items = malloc(sizeof(uint32_t) * len);
        curr = node;
        size_t idx = len;
        while (curr->parent != NULL) {
            items[--idx] = curr->item;
            curr = curr->parent;
        }
        topk_add(topk, items, len, node->sumIu);
        free(items);
    }
    for (size_t i = 0; i < node->child_count; i++) {
        explore_descendants(node->children[i], topk, stats);
    }
}

static void tmku_discover(TPNode **header_table, double *twu_counts, uint32_t *target_pattern, size_t target_len, TopK *topk, DM_TMKU_Stats *stats) {
    if (target_len == 0) return;
    
    uint32_t tr = target_pattern[target_len - 1];
    
    TPNode *curr = header_table[tr];
    while (curr != NULL) {
        stats->candidates++;
        int posToMatch = (int)target_len - 2;
        TPNode *parent_node = curr->parent;
        
        while (parent_node != NULL && posToMatch >= 0) {
            uint32_t y = target_pattern[posToMatch];
            if (parent_node->twu < twu_counts[y]) {
                stats->pruned_twu++;
                break;
            }
            if (parent_node->twu == twu_counts[y] && parent_node->item == y) {
                posToMatch--;
            }
            parent_node = parent_node->parent;
        }
        
        if (posToMatch == -1) {
            explore_descendants(curr, topk, stats);
        }
        
        curr = curr->link;
    }
}

/* --- MAIN API --- */

int tmku_mine_dataset(DM_Dataset *ds, const DM_TMKU_Params *params, DM_TMKU_Stats *stats) {
    if (!ds || ds->type != DM_TYPE_UTILITY || !params || !stats) return -1;
    
    memset(stats, 0, sizeof(*stats));
    stats->transactions = ds->count;
    stats->k = params->k;

    double min_util = params->min_utility;

    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;

    double *twu_counts = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            twu_counts[data[i].items[j].id] += data[i].total_utility;
        }
    }

    // Verify all items in the target pattern exist and satisfy TWU >= min_util
    for (size_t i = 0; i < params->target_len; i++) {
        uint32_t target_item = params->target_pattern[i];
        if (target_item > ds->max_id || twu_counts[target_item] < min_util) {
            free(twu_counts);
            return 0; // Return empty results safely
        }
    }

    ItemTWU *items = malloc(sizeof(ItemTWU) * (ds->max_id + 1));
    size_t item_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu_counts[i] >= min_util) {
            items[item_count].id = i;
            items[item_count].twu = twu_counts[i];
            item_count++;
        }
    }
    qsort(items, item_count, sizeof(ItemTWU), cmp_item_twu);
    stats->distinct_items = item_count;

    g_rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(g_rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < item_count; i++) {
        g_rank[items[i].id] = (uint32_t)i;
    }

    // Sort target pattern by rank
    uint32_t *sorted_target = malloc(sizeof(uint32_t) * params->target_len);
    memcpy(sorted_target, params->target_pattern, sizeof(uint32_t) * params->target_len);
    qsort(sorted_target, params->target_len, sizeof(uint32_t), cmp_target_rank);

    UtilityList **initial_ext = malloc(sizeof(UtilityList*) * item_count);
    for (size_t i = 0; i < item_count; i++) {
        initial_ext[i] = malloc(sizeof(UtilityList));
        initial_ext[i]->items = malloc(sizeof(uint32_t));
        initial_ext[i]->items[0] = items[i].id;
        initial_ext[i]->count = 1;
        initial_ext[i]->tuples = malloc(sizeof(UtilityTuple) * 8);
        initial_ext[i]->tuple_count = 0;
        initial_ext[i]->sum_iutil = 0;
        initial_ext[i]->sum_rutil = 0;
    }

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *t_items = malloc(sizeof(uint32_t) * data[i].count);
        double *t_utils = malloc(sizeof(double) * data[i].count);
        size_t t_count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (g_rank[data[i].items[j].id] != 0xFFFFFFFF) {
                t_items[t_count] = data[i].items[j].id;
                t_utils[t_count] = data[i].items[j].utility;
                t_count++;
            }
        }
        
        for (size_t j = 0; j < t_count; j++) {
            for (size_t k = j + 1; k < t_count; k++) {
                if (g_rank[t_items[j]] > g_rank[t_items[k]]) {
                    uint32_t tmp_i = t_items[j]; t_items[j] = t_items[k]; t_items[k] = tmp_i;
                    double tmp_u = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = tmp_u;
                }
            }
        }

        double remaining_utility = 0;
        for (int j = (int)t_count - 1; j >= 0; j--) {
            uint32_t r = g_rank[t_items[j]];
            UtilityList *ul = initial_ext[r];
            
            if (ul->tuple_count > 0 && ul->tuple_count % 8 == 0) {
                ul->tuples = realloc(ul->tuples, sizeof(UtilityTuple) * (ul->tuple_count + 8));
            }
            
            ul->tuples[ul->tuple_count].tid = (uint32_t)i;
            ul->tuples[ul->tuple_count].iutil = t_utils[j];
            ul->tuples[ul->tuple_count].rutil = remaining_utility;
            ul->sum_iutil += t_utils[j];
            ul->sum_rutil += remaining_utility;
            ul->tuple_count++;
            
            remaining_utility += t_utils[j];
        }
        free(t_items); free(t_utils);
    }

    TPNode *root = create_node(0, 0.0, NULL);
    TPNode **header_table = calloc(ds->max_id + 1, sizeof(TPNode*));
    TPNode **tail_table = calloc(ds->max_id + 1, sizeof(TPNode*));

    tmku_mine_recursive(NULL, initial_ext, item_count, min_util, root, twu_counts, header_table, tail_table, stats);

    TopK topk;
    topk_init(&topk, params->k);

    tmku_discover(header_table, twu_counts, sorted_target, params->target_len, &topk, stats);

    stats->final_threshold = topk.threshold;
    stats->output_count = topk.count;
    stats->threshold_raises = topk.threshold_raises;

    printf("\nTargeted Top-%zu High Utility Itemsets containing: ", params->k);
    for (size_t i = 0; i < params->target_len; i++) {
        printf("%u ", params->target_pattern[i]);
    }
    printf("\n");
    printf("===================================================\n");
    for (size_t i = 0; i < topk.count; i++) {
        stats->total_output_items += topk.data[i].len;
        stats->avg_utility += topk.data[i].utility;
        if (i == 0 || topk.data[i].utility > stats->best_utility) {
            stats->best_utility = topk.data[i].utility;
        }
        
        printf("{ ");
        for (size_t j = 0; j < topk.data[i].len; j++) {
            printf("%u ", topk.data[i].items[j]);
        }
        printf("} #UTILITY: %.6f\n", topk.data[i].utility);
    }
    printf("===================================================\n\n");

    if (topk.count > 0) {
        stats->avg_utility /= (double)topk.count;
        stats->avg_length = (double)stats->total_output_items / (double)topk.count;
    }

    stats->result_ram_bytes = topk.count * sizeof(Pattern) + stats->total_output_items * sizeof(uint32_t);
    stats->result_disk_est_bytes = stats->total_output_items * 12 + topk.count * 32;

    // Cleanup
    topk_free(&topk);
    free_tree(root);
    free(header_table);
    free(tail_table);
    for (size_t i = 0; i < item_count; i++) free_utility_list(initial_ext[i]);
    free(initial_ext);
    free(items);
    free(g_rank);
    free(twu_counts);
    free(sorted_target);

    return 0;
}

static DM_Status run_algo(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_TMKU_Params *p = (DM_TMKU_Params *)params;
    DM_TMKU_Stats stats;
    int rc = tmku_mine_dataset(ds, p, &stats);
    if (rc != 0) return DM_ERROR_GENERIC;
    dm_bench_record_results(stats.output_count, stats.total_output_items);
    return DM_SUCCESS;
}

DM_Algorithm tmku_algo = {
    .id = "tmku",
    .name = "TMKU",
    .description = "Targeted Mining of Top-k High Utility Itemsets containing a target pattern.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run_algo
};

DM_REGISTER_ALGORITHM(tmku_algo)
