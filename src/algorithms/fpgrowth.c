#include "algorithms/fpgrowth.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#define ROOT_ITEM 0xFFFFFFFF

/* --- INTERNAL DATA STRUCTURES --- */

typedef struct FP_Node {
    uint32_t item;
    uint32_t count;
    struct FP_Node *parent;
    struct FP_Node *first_child;
    struct FP_Node *next_sibling;
    struct FP_Node *node_link;
} FP_Node;

typedef struct {
    uint32_t item;
    uint32_t support;
    FP_Node *head;
    FP_Node *tail;
} Header_Entry;

typedef struct {
    Header_Entry *entries;
    size_t count;
} Header_Table;

typedef struct {
    FP_Node *root;
    Header_Table header_table;
} FP_Tree;

typedef struct {
    uint32_t *items;
    size_t count;
    uint32_t weight;
} Weighted_Trans;

typedef struct {
    Weighted_Trans *array;
    size_t count;
    size_t capacity;
    uint32_t max_id;
} Weighted_Trans_List;

/* --- MEMORY POOL (ARENA ALLOCATOR) --- */
typedef struct NodePool {
    FP_Node *nodes;
    size_t capacity;
    size_t used;
    struct NodePool *next;
} NodePool;

typedef struct {
    NodePool *head;
} FP_Allocator;

static void alloc_init(FP_Allocator *alloc) {
    alloc->head = NULL;
}

static FP_Node* alloc_node(FP_Allocator *alloc, uint32_t item, FP_Node *parent) {
    if (!alloc->head || alloc->head->used >= alloc->head->capacity) {
        NodePool *np = (NodePool *)malloc(sizeof(NodePool));
        np->capacity = 4096;
        np->nodes = (FP_Node *)malloc(sizeof(FP_Node) * np->capacity);
        np->used = 0;
        np->next = alloc->head;
        alloc->head = np;
    }
    FP_Node *n = &alloc->head->nodes[alloc->head->used++];
    n->item = item;
    n->count = 0;
    n->parent = parent;
    n->first_child = NULL;
    n->next_sibling = NULL;
    n->node_link = NULL;
    return n;
}

static void alloc_free(FP_Allocator *alloc) {
    NodePool *curr = alloc->head;
    while (curr) {
        NodePool *next = curr->next;
        free(curr->nodes);
        free(curr);
        curr = next;
    }
    alloc->head = NULL;
}

/* --- TREE CONSTRUCTION --- */

static int cmp_support_desc(const void *a, const void *b) {
    const Header_Entry *ha = (const Header_Entry *)a;
    const Header_Entry *hb = (const Header_Entry *)b;
    if (ha->support > hb->support) return -1;
    if (ha->support < hb->support) return 1;
    if (ha->item < hb->item) return -1; // deterministic tie-break
    if (ha->item > hb->item) return 1;
    return 0;
}

static void sort_transaction(uint32_t *items, size_t count, uint32_t *rank_map) {
    for (size_t i = 1; i < count; i++) {
        uint32_t key = items[i];
        size_t j = i;
        while (j > 0 && rank_map[items[j - 1]] > rank_map[key]) {
            items[j] = items[j - 1];
            j--;
        }
        items[j] = key;
    }
}

static FP_Tree* build_tree(Weighted_Trans_List *db, uint32_t min_sup, FP_Allocator *alloc) {
    uint32_t *counts = (uint32_t *)calloc(db->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < db->count; i++) {
        for (size_t j = 0; j < db->array[i].count; j++) {
            counts[db->array[i].items[j]] += db->array[i].weight;
        }
    }

    size_t freq_count = 0;
    for (uint32_t i = 0; i <= db->max_id; i++) {
        if (counts[i] >= min_sup) freq_count++;
    }

    if (freq_count == 0) {
        free(counts);
        return NULL;
    }

    FP_Tree *tree = (FP_Tree *)malloc(sizeof(FP_Tree));
    tree->header_table.count = freq_count;
    tree->header_table.entries = (Header_Entry *)malloc(sizeof(Header_Entry) * freq_count);
    
    size_t idx = 0;
    for (uint32_t i = 0; i <= db->max_id; i++) {
        if (counts[i] >= min_sup) {
            tree->header_table.entries[idx].item = i;
            tree->header_table.entries[idx].support = counts[i];
            tree->header_table.entries[idx].head = NULL;
            tree->header_table.entries[idx].tail = NULL;
            idx++;
        }
    }

    qsort(tree->header_table.entries, freq_count, sizeof(Header_Entry), cmp_support_desc);

    uint32_t *rank_map = (uint32_t *)calloc(db->max_id + 1, sizeof(uint32_t));
    Header_Entry **entry_map = (Header_Entry **)calloc(db->max_id + 1, sizeof(Header_Entry*));
    
    for (uint32_t i = 0; i <= db->max_id; i++) {
        rank_map[i] = 0xFFFFFFFF; // Inf
    }
    for (size_t i = 0; i < freq_count; i++) {
        rank_map[tree->header_table.entries[i].item] = (uint32_t)i;
        entry_map[tree->header_table.entries[i].item] = &tree->header_table.entries[i];
    }

    tree->root = alloc_node(alloc, ROOT_ITEM, NULL);

    uint32_t *filtered = (uint32_t *)malloc(sizeof(uint32_t) * (db->max_id + 1));
    for (size_t i = 0; i < db->count; i++) {
        size_t f_count = 0;
        for (size_t j = 0; j < db->array[i].count; j++) {
            uint32_t item = db->array[i].items[j];
            if (counts[item] >= min_sup) {
                filtered[f_count++] = item;
            }
        }
        if (f_count > 0) {
            sort_transaction(filtered, f_count, rank_map);
            
            FP_Node *curr = tree->root;
            for (size_t j = 0; j < f_count; j++) {
                uint32_t item = filtered[j];
                FP_Node *child = curr->first_child;
                FP_Node *prev = NULL;
                while (child) {
                    if (child->item == item) break;
                    prev = child;
                    child = child->next_sibling;
                }
                if (child) {
                    child->count += db->array[i].weight;
                    curr = child;
                } else {
                    FP_Node *n = alloc_node(alloc, item, curr);
                    n->count = db->array[i].weight;
                    if (prev) prev->next_sibling = n;
                    else curr->first_child = n;
                    
                    Header_Entry *he = entry_map[item];
                    if (he->tail) he->tail->node_link = n;
                    else he->head = n;
                    he->tail = n;
                    
                    curr = n;
                }
            }
        }
    }

    free(filtered);
    free(rank_map);
    free(entry_map);
    free(counts);

    return tree;
}

/* --- DFS MINING --- */

static void mine_tree(FP_Tree *tree, uint32_t min_sup, size_t k, size_t *total_freq, size_t *total_footprint) {
    // Process from least frequent to most
    for (int i = tree->header_table.count - 1; i >= 0; i--) {
        Header_Entry *he = &tree->header_table.entries[i];
        
        (*total_freq)++;
        (*total_footprint) += k;

        Weighted_Trans_List cond_db;
        cond_db.capacity = 16;
        cond_db.count = 0;
        cond_db.max_id = 0;
        cond_db.array = (Weighted_Trans *)malloc(sizeof(Weighted_Trans) * cond_db.capacity);

        FP_Node *node = he->head;
        uint32_t *path = (uint32_t *)malloc(sizeof(uint32_t) * 1024); // Assume max path depth 1024

        while (node) {
            if (node->parent && node->parent->item != ROOT_ITEM) {
                size_t p_len = 0;
                FP_Node *p = node->parent;
                while (p && p->item != ROOT_ITEM) {
                    path[p_len++] = p->item;
                    if (p->item > cond_db.max_id) cond_db.max_id = p->item;
                    p = p->parent;
                }
                if (p_len > 0) {
                    if (cond_db.count >= cond_db.capacity) {
                        cond_db.capacity *= 2;
                        cond_db.array = (Weighted_Trans *)realloc(cond_db.array, sizeof(Weighted_Trans) * cond_db.capacity);
                    }
                    cond_db.array[cond_db.count].items = (uint32_t *)malloc(sizeof(uint32_t) * p_len);
                    memcpy(cond_db.array[cond_db.count].items, path, sizeof(uint32_t) * p_len);
                    cond_db.array[cond_db.count].count = p_len;
                    cond_db.array[cond_db.count].weight = node->count;
                    cond_db.count++;
                }
            }
            node = node->node_link;
        }
        free(path);

        if (cond_db.count > 0) {
            FP_Allocator local_alloc;
            alloc_init(&local_alloc);
            
            FP_Tree *cond_tree = build_tree(&cond_db, min_sup, &local_alloc);
            if (cond_tree) {
                mine_tree(cond_tree, min_sup, k + 1, total_freq, total_footprint);
                free(cond_tree->header_table.entries);
                free(cond_tree);
            }
            
            alloc_free(&local_alloc);
            
            for (size_t t = 0; t < cond_db.count; t++) {
                free(cond_db.array[t].items);
            }
        }
        free(cond_db.array);
    }
}

/* --- MAIN ENTRY --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_FPGROWTH_Params *p = (DM_FPGROWTH_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0) min_sup = 1;

    printf("[FP-Growth] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;

    // Convert raw dataset into Weighted_Trans_List for initial tree build
    Weighted_Trans_List db;
    db.count = ds->count;
    db.capacity = ds->count;
    db.max_id = ds->max_id;
    db.array = (Weighted_Trans *)malloc(sizeof(Weighted_Trans) * ds->count);
    
    for (size_t i = 0; i < ds->count; i++) {
        db.array[i].items = data[i].items; // Borrow pointers, read-only
        db.array[i].count = data[i].count;
        db.array[i].weight = 1;
    }

    FP_Allocator global_alloc;
    alloc_init(&global_alloc);

    FP_Tree *tree = build_tree(&db, min_sup, &global_alloc);
    free(db.array);

    size_t total_freq = 0;
    size_t total_footprint = 0;

    if (tree) {
        mine_tree(tree, min_sup, 1, &total_freq, &total_footprint);
        free(tree->header_table.entries);
        free(tree);
    }

    alloc_free(&global_alloc);

    printf("[FP-Growth] Complete. Total frequent itemsets found: %zu\n", total_freq);
    dm_bench_record_results(total_freq, total_footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "fpgrowth",
    .name = "FP-Growth Algorithm",
    .description = "Mining Frequent Patterns Without Candidate Generation (Han SIGMOD 2000).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
