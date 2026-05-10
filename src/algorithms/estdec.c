#include "algorithms/estdec.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * estDec Algorithm for finding recent frequent itemsets over data streams.
 * Reference: J. H. Chang and W. S. Lee, "Finding Recent Frequent Itemsets 
 * Adaptively over Online Data Streams", KDD 2003.
 */

typedef struct MLNode {
    uint32_t item;
    double cnt;
    double err;
    uint32_t mrtid;
    struct MLNode *parent;
    struct MLNode *first_child;
    struct MLNode *next_sibling;
} MLNode;

typedef struct {
    MLNode *root;
    double d;         // decay rate
    double Dk;        // total decayed transaction count
    uint32_t k;       // current transaction index
    double s_min;     // Smin
    double s_ins;     // Sins
    double s_prn;     // Sprn
} EstDecContext;

static MLNode* create_node(uint32_t item, MLNode *parent) {
    MLNode *n = calloc(1, sizeof(MLNode));
    n->item = item;
    n->parent = parent;
    return n;
}

static void free_ml(MLNode *n) {
    if (!n) return;
    free_ml(n->first_child);
    free_ml(n->next_sibling);
    free(n);
}

static int compare_uint32(const void *a, const void *b) {
    uint32_t v1 = *(uint32_t *)a;
    uint32_t v2 = *(uint32_t *)b;
    if (v1 < v2) return -1;
    if (v1 > v2) return 1;
    return 0;
}

static double get_node_count(MLNode *n, uint32_t current_k, double d) {
    if (n->mrtid == 0) return n->cnt;
    return n->cnt * pow(d, (double)current_k - n->mrtid);
}

static double get_node_error(MLNode *n, uint32_t current_k, double d) {
    if (n->mrtid == 0) return n->err;
    return n->err * pow(d, (double)current_k - n->mrtid);
}

// Phase II: Count Updating
static void update_counts(MLNode *curr, uint32_t *t_items, uint32_t t_len, uint32_t start_idx, uint32_t k, double d, double Dk, double s_prn) {
    for (uint32_t i = start_idx; i < t_len; i++) {
        uint32_t item = t_items[i];
        MLNode *child = curr->first_child;
        MLNode *prev = NULL;
        while (child && child->item < item) { prev = child; child = child->next_sibling; }
        
        if (child && child->item == item) {
            // Update node
            child->cnt = get_node_count(child, k, d) + 1.0;
            child->err = get_node_error(child, k, d);
            child->mrtid = k;
            
            // Recursively update children
            update_counts(child, t_items, t_len, i + 1, k, d, Dk, s_prn);
            
            // Pruning (Line 7-8)
            if (child->cnt / Dk < s_prn && child->parent != NULL && child->parent->item != 0xFFFFFFFF) {
                // Eliminate e and its children
                if (prev) prev->next_sibling = child->next_sibling;
                else curr->first_child = child->next_sibling;
                MLNode *next = child->next_sibling;
                child->next_sibling = NULL;
                free_ml(child);
                child = next;
                continue; 
            }
        }
    }
}

// Estimation helper
static double find_count_in_ml(MLNode *root, uint32_t *items, uint32_t len, uint32_t k, double d) {
    MLNode *curr = root;
    for (uint32_t i = 0; i < len; i++) {
        uint32_t item = items[i];
        MLNode *child = curr->first_child;
        while (child && child->item < item) child = child->next_sibling;
        if (!child || child->item != item) return 0.0;
        curr = child;
    }
    return get_node_count(curr, k, d);
}

// Delayed Insertion
static void delayed_insertion(MLNode *curr, uint32_t *t_items, uint32_t t_len, uint32_t start_idx, EstDecContext *ctx, uint32_t *prefix, uint32_t prefix_len) {
    for (uint32_t i = start_idx; i < t_len; i++) {
        uint32_t item = t_items[i];
        MLNode *child = curr->first_child;
        MLNode *prev = NULL;
        while (child && child->item < item) { prev = child; child = child->next_sibling; }
        
        if (child && child->item == item) {
            // Already in ML, go deeper
            prefix[prefix_len] = item;
            delayed_insertion(child, t_items, t_len, i + 1, ctx, prefix, prefix_len + 1);
        } else {
            // Potential for insertion (Line 14-20)
            if (prefix_len > 0) {
                uint32_t current_itemset[128];
                memcpy(current_itemset, prefix, prefix_len * sizeof(uint32_t));
                current_itemset[prefix_len] = item;
                uint32_t n_len = prefix_len + 1;
                
                double c_max = 1e18;
                bool all_subsets_exist = true;
                for (uint32_t j = 0; j < n_len; j++) {
                    uint32_t subset[128];
                    uint32_t slen = 0;
                    for (uint32_t m = 0; m < n_len; m++) if (m != j) subset[slen++] = current_itemset[m];
                    double c = find_count_in_ml(ctx->root, subset, slen, ctx->k, ctx->d);
                    if (c == 0.0) { all_subsets_exist = false; break; }
                    if (c < c_max) c_max = c;
                }
                
                if (all_subsets_exist) {
                    double cnt_for_subsets = (1.0 - pow(ctx->d, (double)n_len - 1.0)) / (1.0 - ctx->d);
                    double max_cnt_before_subsets = ctx->s_ins * (ctx->Dk - (double)n_len + 1.0) * pow(ctx->d, (double)n_len - 1.0);
                    double c_upper = max_cnt_before_subsets + cnt_for_subsets;
                    if (c_max > c_upper) c_max = c_upper;
                    
                    if (c_max / ctx->Dk >= ctx->s_ins) {
                        MLNode *node = create_node(item, curr);
                        node->cnt = c_max;
                        node->err = c_max; 
                        node->mrtid = ctx->k;
                        if (prev) { node->next_sibling = prev->next_sibling; prev->next_sibling = node; }
                        else { node->next_sibling = curr->first_child; curr->first_child = node; }
                        
                        prefix[prefix_len] = item;
                        delayed_insertion(node, t_items, t_len, i + 1, ctx, prefix, prefix_len + 1);
                    }
                }
            }
        }
    }
}

static void select_frequent(MLNode *curr, uint32_t *prefix, uint32_t prefix_len, EstDecContext *ctx, size_t *total_fi, size_t *total_footprint) {
    MLNode *child = curr->first_child;
    while (child) {
        prefix[prefix_len] = child->item;
        double support = get_node_count(child, ctx->k, ctx->d) / ctx->Dk;
        if (support >= ctx->s_min - 1e-7) {
            (*total_fi)++;
            (*total_footprint) += (prefix_len + 1);
        }
        select_frequent(child, prefix, prefix_len + 1, ctx, total_fi, total_footprint);
        child = child->next_sibling;
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_ESTDEC_Params *p = (DM_ESTDEC_Params *)params;
    double s_min = p ? p->min_support : 0.1;
    double s_ins = p ? p->ins_threshold : 0.05;
    double s_prn = p ? p->prn_threshold : 0.01;
    double b = p ? p->decay_base : 2.0;
    double h = p ? p->decay_life : 10000.0;
    double d = pow(b, -1.0/h);

    EstDecContext ctx = {
        .root = create_node(0xFFFFFFFF, NULL),
        .d = d,
        .Dk = 0.0,
        .k = 0,
        .s_min = s_min,
        .s_ins = s_ins,
        .s_prn = s_prn
    };

    printf("[estDec] Starting. Smin: %.4f, Sins: %.4f, Sprn: %.4f, b: %.1f, h: %.1f\n", s_min, s_ins, s_prn, b, h);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    uint32_t *filt_items = malloc(ds->max_id * sizeof(uint32_t));

    for (size_t i = 0; i < ds->count; i++) {
        ctx.k++;
        uint32_t *t_items = data[i].items;
        uint32_t t_len = (uint32_t)data[i].count;
        qsort(t_items, t_len, sizeof(uint32_t), compare_uint32);
        
        // Phase I
        ctx.Dk = ctx.Dk * d + 1.0;
        
        // Phase II
        update_counts(ctx.root, t_items, t_len, 0, ctx.k, d, ctx.Dk, s_prn);
        
        // Phase III - Item Filtering (Line 10)
        uint32_t filt_len = 0;
        for (uint32_t j = 0; j < t_len; j++) {
            double sup = find_count_in_ml(ctx.root, &t_items[j], 1, ctx.k, d) / ctx.Dk;
            if (sup >= s_ins || find_count_in_ml(ctx.root, &t_items[j], 1, ctx.k, d) == 0.0) {
                filt_items[filt_len++] = t_items[j];
            }
        }
        
        // Ensure all 1-itemsets in Tk' are in ML (Line 12-13)
        for (uint32_t j = 0; j < filt_len; j++) {
            uint32_t item = filt_items[j];
            MLNode *child = ctx.root->first_child;
            MLNode *prev = NULL;
            while (child && child->item < item) { prev = child; child = child->next_sibling; }
            if (!child || child->item != item) {
                MLNode *n = create_node(item, ctx.root);
                n->cnt = 1.0; n->err = 0.0; n->mrtid = ctx.k;
                if (prev) { n->next_sibling = prev->next_sibling; prev->next_sibling = n; }
                else { n->next_sibling = ctx.root->first_child; ctx.root->first_child = n; }
            }
        }

        uint32_t prefix[128];
        delayed_insertion(ctx.root, filt_items, filt_len, 0, &ctx, prefix, 0);
    }
    free(filt_items);

    // Phase IV
    size_t total_fi = 0;
    size_t total_footprint = 0;
    uint32_t prefix[128];
    select_frequent(ctx.root, prefix, 0, &ctx, &total_fi, &total_footprint);

    printf("[estDec] Complete. Total Frequent Itemsets found: %zu\n", total_fi);
    dm_bench_record_results(total_fi, total_footprint);

    free_ml(ctx.root);
    return DM_SUCCESS;
}

static DM_Algorithm algo_estdec = {
    .id = "estdec",
    .name = "estDec Algorithm",
    .description = "Finding recent frequent itemsets adaptively over online data streams.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo_estdec)
