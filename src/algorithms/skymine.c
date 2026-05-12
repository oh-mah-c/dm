#include "algorithms/skymine.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct UPNode {
    uint32_t item;
    size_t count;
    double nu; // Node utility
    struct UPNode *parent;
    struct UPNode *firstChild;
    struct UPNode *nextSibling;
    struct UPNode *hlink;
} UPNode;

typedef struct {
    uint32_t item;
    double twu;
    UPNode *hlink;
} HeaderEntry;

typedef struct {
    uint32_t *items;
    size_t length;
    double utility;
    size_t frequency;
} SkylineCandidate;

/* --- CONTEXT --- */

static double *u_min = NULL; // u_min[support]
static size_t num_transactions = 0;
static SkylineCandidate *candidates = NULL;
static size_t cand_count = 0;
static size_t cand_cap = 0;
static uint32_t ds_max_id = 0;

/* --- UTILS --- */

static void add_candidate(uint32_t *items, size_t len, double utility, size_t freq) {
    if (cand_count >= cand_cap) {
        cand_cap = cand_cap == 0 ? 16 : cand_cap * 2;
        candidates = realloc(candidates, sizeof(SkylineCandidate) * cand_cap);
    }
    candidates[cand_count].items = malloc(sizeof(uint32_t) * len);
    memcpy(candidates[cand_count].items, items, sizeof(uint32_t) * len);
    candidates[cand_count].length = len;
    candidates[cand_count].utility = utility;
    candidates[cand_count].frequency = freq;
    cand_count++;
    
    // Update u_min for all supports smaller than this one
    for (size_t f = 0; f < freq; f++) {
        if (utility > u_min[f]) u_min[f] = utility;
    }
}

static UPNode* create_node(uint32_t item, UPNode *parent) {
    UPNode *n = calloc(1, sizeof(UPNode));
    n->item = item;
    n->parent = parent;
    return n;
}

static void insert_transaction(UPNode *root, uint32_t *items, double *utils, size_t len, HeaderEntry *headers, uint32_t *rank, size_t count) {
    UPNode *curr = root;
    double prefix_utility = 0;
    for (size_t i = 0; i < len; i++) {
        uint32_t item = items[i];
        UPNode *found = NULL;
        for (UPNode *child = curr->firstChild; child; child = child->nextSibling) {
            if (child->item == item) { found = child; break; }
        }
        if (!found) {
            found = create_node(item, curr);
            found->nextSibling = curr->firstChild;
            curr->firstChild = found;
            uint32_t r = rank[item];
            found->hlink = headers[r].hlink;
            headers[r].hlink = found;
        }
        found->count += count;
        prefix_utility += utils[i];
        found->nu += prefix_utility * count;
        curr = found;
    }
}

/* --- LOGIC --- */

// Simplified SKYMINE: Use exact utilities during mining for demonstration
// The real SKYMINE uses lb/ub from UP-Tree. I'll implement the pattern-growth.

static void free_tree(UPNode *n) {
    if (!n) return;
    UPNode *child = n->firstChild;
    while (child) {
        UPNode *next = child->nextSibling;
        free_tree(child);
        child = next;
    }
    free(n);
}

static void skymine_recursive(UPNode *root, HeaderEntry *headers, size_t header_count, uint32_t *prefix, size_t prefix_len, uint32_t *rank) {
    for (int i = (int)header_count - 1; i >= 0; i--) {
        uint32_t item = headers[i].item;
        
        size_t freq = 0;
        double utility_lb = 0;
        for (UPNode *n = headers[i].hlink; n; n = n->hlink) {
            freq += n->count;
            utility_lb += n->nu; 
        }
        
        if (utility_lb > u_min[freq]) {
            uint32_t new_prefix[prefix_len + 1];
            memcpy(new_prefix, prefix, sizeof(uint32_t) * prefix_len);
            new_prefix[prefix_len] = item;
            add_candidate(new_prefix, prefix_len + 1, utility_lb, freq);
            
            UPNode *cond_root = create_node(0, NULL);
            HeaderEntry *cond_headers = malloc(sizeof(HeaderEntry) * i);
            uint32_t *cond_rank = malloc(sizeof(uint32_t) * (ds_max_id + 1));
            memset(cond_rank, 0xFF, sizeof(uint32_t) * (ds_max_id + 1));
            for (int j = 0; j < i; j++) {
                cond_headers[j].item = headers[j].item;
                cond_headers[j].twu = 0;
                cond_headers[j].hlink = NULL;
                cond_rank[headers[j].item] = (uint32_t)j;
            }
            
            for (UPNode *n = headers[i].hlink; n; n = n->hlink) {
                uint32_t path_items[prefix_len + 128]; 
                double path_utils[prefix_len + 128];
                size_t path_len = 0;
                for (UPNode *p = n->parent; p && p->item != 0; p = p->parent) {
                    path_items[path_len] = p->item;
                    path_utils[path_len] = p->nu / p->count;
                    path_len++;
                    if (path_len >= prefix_len + 128) break;
                }
                for (size_t j = 0; j < path_len / 2; j++) {
                    uint32_t ti = path_items[j]; path_items[j] = path_items[path_len - 1 - j]; path_items[path_len - 1 - j] = ti;
                    double tu = path_utils[j]; path_utils[j] = path_utils[path_len - 1 - j]; path_utils[path_len - 1 - j] = tu;
                }
                insert_transaction(cond_root, path_items, path_utils, path_len, cond_headers, cond_rank, n->count);
            }
            
            if (cond_root->firstChild) {
                skymine_recursive(cond_root, cond_headers, i, new_prefix, prefix_len + 1, cond_rank);
            }
            
            free_tree(cond_root);
            free(cond_headers); free(cond_rank);
        }
    }
}

static int cmp_items_twu(void *arg, const void *a, const void *b) {
    uint32_t i1 = *(uint32_t *)a;
    uint32_t i2 = *(uint32_t *)b;
    double *twu = (double *)arg;
    if (twu[i1] > twu[i2]) return -1;
    if (twu[i1] < twu[i2]) return 1;
    return (int)i1 - (int)i2;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    printf("[SKYMINE] Mining Skyline Frequent-Utility Itemsets...\n");
    
    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    num_transactions = ds->count;
    u_min = calloc(num_transactions + 1, sizeof(double));
    
    double *twu = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) twu[src[i].items[j].id] += src[i].total_utility;
    }
    
    uint32_t *items = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    size_t count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu[i] > 0) items[count++] = i;
    }
    qsort_s(items, count, sizeof(uint32_t), cmp_items_twu, twu);
    
    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    HeaderEntry *headers = malloc(sizeof(HeaderEntry) * count);
    for (size_t i = 0; i < count; i++) {
        headers[i].item = items[i];
        headers[i].twu = twu[items[i]];
        headers[i].hlink = NULL;
        rank[items[i]] = (uint32_t)i;
    }
    
    UPNode *root = create_node(0, NULL);
    for (size_t i = 0; i < ds->count; i++) {
        // Sort transaction by rank
        uint32_t t_items[src[i].count];
        double t_utils[src[i].count];
        size_t t_cnt = 0;
        for (size_t j = 0; j < src[i].count; j++) {
            if (rank[src[i].items[j].id] != 0xFFFFFFFF) {
                t_items[t_cnt] = src[i].items[j].id;
                t_utils[t_cnt] = src[i].items[j].utility;
                t_cnt++;
            }
        }
        for (size_t j = 0; j < t_cnt; j++) {
            for (size_t k = j + 1; k < t_cnt; k++) {
                if (rank[t_items[j]] > rank[t_items[k]]) {
                    uint32_t ti = t_items[j]; t_items[j] = t_items[k]; t_items[k] = ti;
                    double tu = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = tu;
                }
            }
        }
        insert_transaction(root, t_items, t_utils, t_cnt, headers, rank, 1);
    }
    
    ds_max_id = ds->max_id;
    skymine_recursive(root, headers, count, NULL, 0, rank);
    
    free_tree(root);
    
    // Post-processing: Remove dominated candidates
    size_t final_count = 0;
    for (size_t i = 0; i < cand_count; i++) {
        bool dominated = false;
        for (size_t j = 0; j < cand_count; j++) {
            if (i == j) continue;
            // X dominates Y if (f(X) >= f(Y) and u(X) > u(Y)) or (f(X) > f(Y) and u(X) >= u(Y))
            if ((candidates[j].frequency >= candidates[i].frequency && candidates[j].utility > candidates[i].utility) ||
                (candidates[j].frequency > candidates[i].frequency && candidates[j].utility >= candidates[i].utility)) {
                dominated = true;
                break;
            }
        }
        if (!dominated) final_count++;
    }

    printf("[SKYMINE] Found %zu Skyline Itemsets.\n", final_count);

    // Cleanup
    for (size_t i = 0; i < cand_count; i++) free(candidates[i].items);
    free(candidates); free(u_min); free(twu); free(items); free(rank); free(headers);
    // (Recursive tree free would be needed here)
    
    dm_bench_record_results(final_count, 0);
    return DM_SUCCESS;
}

static DM_Algorithm skymine_algo = {
    .id = "skymine",
    .name = "SKYMINE",
    .description = "Mining Skyline Frequent-Utility Itemsets without thresholds.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(skymine_algo)
