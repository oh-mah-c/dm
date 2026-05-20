#include "algorithms/up_growth.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct UPTreeNode {
    uint32_t id;
    uint32_t count;
    double nu;
    struct UPTreeNode *parent;
    struct UPTreeNode *hlink;
    struct UPTreeNode **children;
    size_t children_count;
} UPTreeNode;

typedef struct {
    UPTreeNode *root;
} UPTree;

typedef struct {
    uint32_t item;
    double utility;
    UPTreeNode *head;
} HeaderEntry;

typedef struct {
    HeaderEntry *entries;
    size_t count;
} HeaderTable;

typedef struct {
    uint32_t *items;
    size_t count;
} PHUI;

static UPTreeNode* create_node(uint32_t id, UPTreeNode *parent) {
    UPTreeNode *node = calloc(1, sizeof(UPTreeNode));
    node->id = id;
    node->parent = parent;
    return node;
}

static void free_tree(UPTreeNode *node) {
    if (!node) return;
    for (size_t i = 0; i < node->children_count; i++) {
        free_tree(node->children[i]);
    }
    free(node->children);
    free(node);
}

static PHUI *phui_list = NULL;
static size_t phui_count = 0;
static size_t phui_capacity = 0;

static void add_phui(const uint32_t *items, size_t count) {
    if (phui_count >= phui_capacity) {
        phui_capacity = phui_capacity == 0 ? 1024 : phui_capacity * 2;
        phui_list = realloc(phui_list, sizeof(PHUI) * phui_capacity);
    }
    phui_list[phui_count].items = malloc(sizeof(uint32_t) * count);
    memcpy(phui_list[phui_count].items, items, sizeof(uint32_t) * count);
    phui_list[phui_count].count = count;
    phui_count++;
}

static uint32_t *global_rank = NULL;
static double *global_miu_table = NULL;

typedef struct {
    uint32_t id;
    double utility;
} SortedItem;

static int cmp_sorted_items(const void *a, const void *b) {
    uint32_t id_a = ((const SortedItem *)a)->id;
    uint32_t id_b = ((const SortedItem *)b)->id;
    uint32_t rank_a = global_rank[id_a];
    uint32_t rank_b = global_rank[id_b];
    if (rank_a < rank_b) return -1;
    if (rank_a > rank_b) return 1;
    return 0;
}

typedef struct {
    uint32_t *items;
    size_t count;
    uint32_t support;
    double path_utility;
} CPB_Path;

static void up_growth_recurse(UPTree *tree, HeaderTable *header_table, uint32_t *prefix, size_t prefix_len, double min_util) {
    for (int i = (int)header_table->count - 1; i >= 0; i--) {
        HeaderEntry *entry = &header_table->entries[i];
        uint32_t item = entry->item;
        
        // Generate prefix extension
        uint32_t *new_prefix = malloc(sizeof(uint32_t) * (prefix_len + 1));
        memcpy(new_prefix, prefix, sizeof(uint32_t) * prefix_len);
        new_prefix[prefix_len] = item;
        
        add_phui(new_prefix, prefix_len + 1);
        
        // Construct Y-CPB
        size_t path_count = 0;
        UPTreeNode *curr = entry->head;
        while (curr) {
            path_count++;
            curr = curr->hlink;
        }
        
        if (path_count == 0) {
            free(new_prefix);
            continue;
        }
        
        CPB_Path *cpb = malloc(sizeof(CPB_Path) * path_count);
        curr = entry->head;
        size_t path_idx = 0;
        uint32_t max_id_in_cpb = 0;
        
        while (curr) {
            size_t ancestor_count = 0;
            UPTreeNode *p = curr->parent;
            while (p && p->id != 0xFFFFFFFF) {
                ancestor_count++;
                p = p->parent;
            }
            
            cpb[path_idx].items = malloc(sizeof(uint32_t) * ancestor_count);
            cpb[path_idx].count = ancestor_count;
            cpb[path_idx].support = curr->count;
            cpb[path_idx].path_utility = curr->nu;
            
            p = curr->parent;
            size_t idx = 0;
            while (p && p->id != 0xFFFFFFFF) {
                cpb[path_idx].items[idx++] = p->id;
                if (p->id > max_id_in_cpb) {
                    max_id_in_cpb = p->id;
                }
                p = p->parent;
            }
            
            path_idx++;
            curr = curr->hlink;
        }
        
        // Calculate path utility for local items in Y-CPB
        double *local_path_utilities = calloc((size_t)max_id_in_cpb + 1, sizeof(double));
        for (size_t p = 0; p < path_count; p++) {
            for (size_t j = 0; j < cpb[p].count; j++) {
                uint32_t it = cpb[p].items[j];
                local_path_utilities[it] += cpb[p].path_utility;
            }
        }
        
        // Identify local promising items
        size_t local_prom_count = 0;
        uint32_t *local_promising = malloc(sizeof(uint32_t) * ((size_t)max_id_in_cpb + 1));
        for (uint32_t it = 0; it <= max_id_in_cpb; it++) {
            if (local_path_utilities[it] >= min_util) {
                local_promising[local_prom_count++] = it;
            }
        }
        
        if (local_prom_count == 0) {
            // Free cpb paths
            for (size_t p = 0; p < path_count; p++) {
                free(cpb[p].items);
            }
            free(cpb);
            free(local_path_utilities);
            free(local_promising);
            free(new_prefix);
            continue;
        }
        
        // Sort local promising items by local path utility descending
        for (size_t a = 0; a < local_prom_count; a++) {
            for (size_t b = a + 1; b < local_prom_count; b++) {
                if (local_path_utilities[local_promising[a]] < local_path_utilities[local_promising[b]] ||
                    (local_path_utilities[local_promising[a]] == local_path_utilities[local_promising[b]] &&
                     local_promising[a] < local_promising[b])) {
                    uint32_t tmp = local_promising[a];
                    local_promising[a] = local_promising[b];
                    local_promising[b] = tmp;
                }
            }
        }
        
        uint32_t *local_rank = malloc(sizeof(uint32_t) * ((size_t)max_id_in_cpb + 1));
        memset(local_rank, 0xFF, sizeof(uint32_t) * ((size_t)max_id_in_cpb + 1));
        for (size_t r = 0; r < local_prom_count; r++) {
            local_rank[local_promising[r]] = (uint32_t)r;
        }
        
        // Build local header table
        HeaderTable local_header;
        local_header.count = local_prom_count;
        local_header.entries = malloc(sizeof(HeaderEntry) * local_prom_count);
        for (size_t r = 0; r < local_prom_count; r++) {
            local_header.entries[r].item = local_promising[r];
            local_header.entries[r].utility = local_path_utilities[local_promising[r]];
            local_header.entries[r].head = NULL;
        }
        
        // Construct local conditional UP-Tree root
        UPTreeNode *local_root = create_node(0xFFFFFFFF, NULL);
        
        // Insert filtered and reorganized paths
        for (size_t p = 0; p < path_count; p++) {
            uint32_t *filtered_path = malloc(sizeof(uint32_t) * cpb[p].count);
            size_t filtered_len = 0;
            for (size_t j = 0; j < cpb[p].count; j++) {
                uint32_t it = cpb[p].items[j];
                if (local_rank[it] != 0xFFFFFFFF) {
                    filtered_path[filtered_len++] = it;
                }
            }
            
            if (filtered_len == 0) {
                free(filtered_path);
                continue;
            }
            
            // Sort filtered_path according to local_rank
            for (size_t a = 0; a < filtered_len; a++) {
                for (size_t b = a + 1; b < filtered_len; b++) {
                    if (local_rank[filtered_path[a]] > local_rank[filtered_path[b]]) {
                        uint32_t tmp = filtered_path[a];
                        filtered_path[a] = filtered_path[b];
                        filtered_path[b] = tmp;
                    }
                }
            }
            
            // Strategy 3: DLU
            double reduced_utility = cpb[p].path_utility;
            for (size_t j = 0; j < cpb[p].count; j++) {
                uint32_t it = cpb[p].items[j];
                if (local_rank[it] == 0xFFFFFFFF) {
                    reduced_utility -= global_miu_table[it] * cpb[p].support;
                }
            }
            if (reduced_utility < 0) reduced_utility = 0;
            
            // Insert reorganized path (DLN Strategy 4)
            UPTreeNode *curr_node = local_root;
            for (size_t x = 0; x < filtered_len; x++) {
                uint32_t item_id = filtered_path[x];
                
                UPTreeNode *child = NULL;
                for (size_t c = 0; c < curr_node->children_count; c++) {
                    if (curr_node->children[c]->id == item_id) {
                        child = curr_node->children[c];
                        break;
                    }
                }
                if (!child) {
                    child = create_node(item_id, curr_node);
                    
                    curr_node->children_count++;
                    curr_node->children = realloc(curr_node->children, sizeof(UPTreeNode*) * curr_node->children_count);
                    curr_node->children[curr_node->children_count - 1] = child;
                    
                    for (size_t h = 0; h < local_prom_count; h++) {
                        if (local_header.entries[h].item == item_id) {
                            child->hlink = local_header.entries[h].head;
                            local_header.entries[h].head = child;
                            break;
                        }
                    }
                }
                
                child->count += cpb[p].support;
                
                double sum_miu = 0;
                for (size_t desc = x + 1; desc < filtered_len; desc++) {
                    sum_miu += global_miu_table[filtered_path[desc]];
                }
                child->nu += reduced_utility - cpb[p].support * sum_miu;
                
                curr_node = child;
            }
            
            free(filtered_path);
        }
        
        // Recursive mining
        if (local_root->children_count > 0) {
            UPTree local_tree;
            local_tree.root = local_root;
            up_growth_recurse(&local_tree, &local_header, new_prefix, prefix_len + 1, min_util);
        }
        
        // Cleanup local structures
        free_tree(local_root);
        free(local_header.entries);
        free(local_rank);
        free(local_promising);
        free(local_path_utilities);
        
        for (size_t p = 0; p < path_count; p++) {
            free(cpb[p].items);
        }
        free(cpb);
        free(new_prefix);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_UP_Growth_Params *p = (DM_UP_Growth_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    phui_count = 0; phui_capacity = 0; phui_list = NULL;

    // Phase 1: TWU and Order
    double *twu = calloc(ds->max_id + 1, sizeof(double));
    global_miu_table = malloc(sizeof(double) * (ds->max_id + 1));
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        global_miu_table[i] = 1e18;
    }

    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item_id = data[i].items[j].id;
            double item_util = data[i].items[j].utility;
            twu[item_id] += data[i].total_utility;
            if (item_util < global_miu_table[item_id]) {
                global_miu_table[item_id] = item_util;
            }
        }
    }

    uint32_t *promising = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    size_t prom_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu[i] >= min_util) {
            promising[prom_count++] = i;
        }
    }

    // Sort TWU Descending
    for (size_t i = 0; i < prom_count; i++) {
        for (size_t j = i + 1; j < prom_count; j++) {
            if (twu[promising[i]] < twu[promising[j]] ||
                (twu[promising[i]] == twu[promising[j]] && promising[i] < promising[j])) {
                uint32_t tmp = promising[i];
                promising[i] = promising[j];
                promising[j] = tmp;
            }
        }
    }

    global_rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(global_rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < prom_count; i++) {
        global_rank[promising[i]] = (uint32_t)i;
    }

    // Construct global header table
    HeaderTable global_header;
    global_header.count = prom_count;
    global_header.entries = malloc(sizeof(HeaderEntry) * prom_count);
    for (size_t i = 0; i < prom_count; i++) {
        global_header.entries[i].item = promising[i];
        global_header.entries[i].utility = twu[promising[i]];
        global_header.entries[i].head = NULL;
    }

    // Create global root node
    UPTreeNode *global_root = create_node(0xFFFFFFFF, NULL);

    // Scan 2: Insert reorganised transactions (DGU/DGN Strategies 1 & 2)
    for (size_t i = 0; i < ds->count; i++) {
        SortedItem *sorted_items = malloc(sizeof(SortedItem) * data[i].count);
        size_t sorted_count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item_id = data[i].items[j].id;
            if (global_rank[item_id] != 0xFFFFFFFF) {
                sorted_items[sorted_count].id = item_id;
                sorted_items[sorted_count].utility = data[i].items[j].utility;
                sorted_count++;
            }
        }

        if (sorted_count == 0) {
            free(sorted_items);
            continue;
        }

        // Sort items by rank ascending
        qsort(sorted_items, sorted_count, sizeof(SortedItem), cmp_sorted_items);

        double rtu = 0;
        for (size_t j = 0; j < sorted_count; j++) {
            rtu += sorted_items[j].utility;
        }

        UPTreeNode *curr_node = global_root;
        for (size_t x = 0; x < sorted_count; x++) {
            uint32_t item_id = sorted_items[x].id;

            UPTreeNode *child = NULL;
            for (size_t c = 0; c < curr_node->children_count; c++) {
                if (curr_node->children[c]->id == item_id) {
                    child = curr_node->children[c];
                    break;
                }
            }
            if (!child) {
                child = create_node(item_id, curr_node);

                curr_node->children_count++;
                curr_node->children = realloc(curr_node->children, sizeof(UPTreeNode*) * curr_node->children_count);
                curr_node->children[curr_node->children_count - 1] = child;

                for (size_t h = 0; h < prom_count; h++) {
                    if (global_header.entries[h].item == item_id) {
                        child->hlink = global_header.entries[h].head;
                        global_header.entries[h].head = child;
                        break;
                    }
                }
            }

            child->count++;

            double suffix_utility = 0;
            for (size_t p = x + 1; p < sorted_count; p++) {
                suffix_utility += sorted_items[p].utility;
            }
            child->nu += rtu - suffix_utility;

            curr_node = child;
        }

        free(sorted_items);
    }

    // Call recursive UP-Growth search to populate phui_list
    if (global_root->children_count > 0) {
        UPTree global_tree;
        global_tree.root = global_root;
        up_growth_recurse(&global_tree, &global_header, NULL, 0, min_util);
    }

    // Phase 3: Verification
    size_t hui_count = 0;
    size_t total_items = 0;
    for (size_t i = 0; i < phui_count; i++) {
        double total_u = 0;
        for (size_t j = 0; j < ds->count; j++) {
            double u_in_t = 0;
            size_t match_count = 0;
            for (size_t l = 0; l < phui_list[i].count; l++) {
                bool found = false;
                for (size_t m = 0; m < data[j].count; m++) {
                    if (data[j].items[m].id == phui_list[i].items[l]) {
                        u_in_t += data[j].items[m].utility;
                        found = true;
                        break;
                    }
                }
                if (found) {
                    match_count++;
                } else {
                    break;
                }
            }
            if (match_count == phui_list[i].count) {
                total_u += u_in_t;
            }
        }
        if (total_u >= min_util) {
            hui_count++;
            total_items += phui_list[i].count;
        }
    }

    printf("[UP-Growth] Found %zu High Utility Itemsets.\n", hui_count);

    // Cleanup
    for (size_t i = 0; i < phui_count; i++) {
        free(phui_list[i].items);
    }
    free(phui_list);
    free_tree(global_root);
    free(global_header.entries);
    free(global_rank);
    free(twu);
    free(global_miu_table);
    free(promising);

    dm_bench_record_results(hui_count, total_items);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "upgrowth",
    .name = "UP-Growth",
    .description = "Utility Pattern Growth algorithm using UP-Tree and candidate pruning strategies (DGU, DGN).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
