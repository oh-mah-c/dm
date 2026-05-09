#include "algorithms/fcfia.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

#define ROOT_ITEM 0xFFFFFFFF
#define MAX_DEPTH 1024

/* --- DATA STRUCTURES --- */

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
    FP_Node *root;
    Header_Entry *header;
    size_t header_count;
    uint32_t *rank_map;
} FP_Tree;

typedef struct CFI_Node {
    uint32_t item;
    uint32_t count;
    uint32_t level;
    struct CFI_Node *parent;
    struct CFI_Node *first_child;
    struct CFI_Node *next_sibling;
    struct CFI_Node *node_link;
} CFI_Node;

typedef struct {
    CFI_Node *root;
    CFI_Node **head_table;
    size_t header_count;
} CFI_Tree;

typedef struct {
    CFI_Node **nodes;
    size_t count;
    size_t capacity;
} CFIP_Set;

typedef struct {
    uint32_t min_sup;
    uint32_t max_item_id;
    size_t total_cfi;
    size_t total_footprint;
    
    CFI_Tree *cfi_tree;
    uint32_t *root_rank_map;
    
    CFIP_Set cfip_sets[MAX_DEPTH];
    uint32_t *current_head;
    int head_len;
} FCFIA_Context;

/* --- MEMORY --- */

typedef struct NodePool {
    FP_Node *nodes;
    size_t capacity;
    size_t used;
    struct NodePool *next;
} NodePool;

typedef struct {
    NodePool *head;
} FP_Allocator;

static FP_Allocator g_fp_alloc;
static void fp_alloc_init() { g_fp_alloc.head = NULL; }
static FP_Node* alloc_fp_node(uint32_t item, FP_Node *parent) {
    if (!g_fp_alloc.head || g_fp_alloc.head->used >= g_fp_alloc.head->capacity) {
        NodePool *np = (NodePool *)malloc(sizeof(NodePool));
        np->capacity = 8192;
        np->nodes = (FP_Node *)malloc(sizeof(FP_Node) * np->capacity);
        np->used = 0;
        np->next = g_fp_alloc.head;
        g_fp_alloc.head = np;
    }
    FP_Node *n = &g_fp_alloc.head->nodes[g_fp_alloc.head->used++];
    n->item = item; n->count = 0; n->parent = parent;
    n->first_child = n->next_sibling = n->node_link = NULL;
    return n;
}
static void fp_alloc_free() {
    NodePool *curr = g_fp_alloc.head;
    while (curr) {
        NodePool *next = curr->next;
        free(curr->nodes); free(curr); curr = next;
    }
    g_fp_alloc.head = NULL;
}

typedef struct CFINodePool {
    CFI_Node *nodes;
    size_t capacity;
    size_t used;
    struct CFINodePool *next;
} CFINodePool;
typedef struct {
    CFINodePool *head;
} CFI_Allocator;
static CFI_Allocator g_cfi_alloc;
static void cfi_alloc_init() { g_cfi_alloc.head = NULL; }
static CFI_Node* alloc_cfi_node(uint32_t item, uint32_t count, uint32_t level, CFI_Node *parent) {
    if (!g_cfi_alloc.head || g_cfi_alloc.head->used >= g_cfi_alloc.head->capacity) {
        CFINodePool *np = (CFINodePool *)malloc(sizeof(CFINodePool));
        np->capacity = 8192;
        np->nodes = (CFI_Node *)malloc(sizeof(CFI_Node) * np->capacity);
        np->used = 0;
        np->next = g_cfi_alloc.head;
        g_cfi_alloc.head = np;
    }
    CFI_Node *n = &g_cfi_alloc.head->nodes[g_cfi_alloc.head->used++];
    n->item = item; n->count = count; n->level = level; n->parent = parent;
    n->first_child = n->next_sibling = n->node_link = NULL;
    return n;
}
static void cfi_alloc_free() {
    CFINodePool *curr = g_cfi_alloc.head;
    while (curr) {
        CFINodePool *next = curr->next;
        free(curr->nodes); free(curr); curr = next;
    }
    g_cfi_alloc.head = NULL;
}

/* --- UTILS --- */

static int cmp_support_desc(const void *a, const void *b) {
    const Header_Entry *ha = (const Header_Entry *)a;
    const Header_Entry *hb = (const Header_Entry *)b;
    if (ha->support > hb->support) return -1;
    if (ha->support < hb->support) return 1;
    return (ha->item < hb->item) ? -1 : 1;
}

static void sort_by_rank(uint32_t *items, int len, uint32_t *rank_map) {
    for (int i = 1; i < len; i++) {
        uint32_t key = items[i];
        int j = i - 1;
        while (j >= 0 && rank_map[items[j]] > rank_map[key]) {
            items[j+1] = items[j]; j--;
        }
        items[j+1] = key;
    }
}

/* --- LOGIC --- */

static void cfip_set_add(CFIP_Set *set, CFI_Node *node) {
    for (size_t i = 0; i < set->count; i++) if (set->nodes[i] == node) return;
    if (set->count >= set->capacity) {
        set->capacity = set->capacity ? set->capacity * 2 : 16;
        set->nodes = (CFI_Node **)realloc(set->nodes, sizeof(CFI_Node *) * set->capacity);
    }
    set->nodes[set->count++] = node;
}

static void update_cfips(FCFIA_Context *ctx, CFI_Node *leaf, uint32_t *itemset, int len) {
    CFI_Node *q = leaf;
    int h = len;
    while (q && q->item != ROOT_ITEM && h >= 1) {
        if (q->item == itemset[h-1]) {
            bool found = false;
            for (size_t i = 0; i < ctx->cfip_sets[h].count; i++) {
                if (ctx->cfip_sets[h].nodes[i] == q) { found = true; break; }
            }
            if (!found) {
                cfip_set_add(&ctx->cfip_sets[h], q);
                h--; q = q->parent;
            } else break;
        } else q = q->parent;
    }
}

static void cfips_projection(FCFIA_Context *ctx, uint32_t x, int h) {
    CFIP_Set *src = &ctx->cfip_sets[h];
    CFIP_Set *dst = &ctx->cfip_sets[h+1];
    dst->count = 0;
    for (size_t i = 0; i < src->count; i++) {
        CFI_Node *p = src->nodes[i];
        CFI_Node *curr = p->parent;
        while (curr && curr->item != ROOT_ITEM) {
            if (curr->item == x) { cfip_set_add(dst, curr); break; }
            curr = curr->parent;
        }
    }
}

static bool is_closed(FCFIA_Context *ctx, uint32_t support) {
    int h = ctx->head_len;
    CFIP_Set *set = &ctx->cfip_sets[h];
    for (size_t i = 0; i < set->count; i++) if (set->nodes[i]->count == support) return false;
    return true;
}

static void insert_into_cfi_tree(FCFIA_Context *ctx, uint32_t *itemset, int len, uint32_t support) {
    uint32_t *sorted = (uint32_t *)malloc(len * sizeof(uint32_t));
    memcpy(sorted, itemset, len * sizeof(uint32_t));
    sort_by_rank(sorted, len, ctx->root_rank_map);
    
    CFI_Node *curr = ctx->cfi_tree->root;
    for (int i = 0; i < len; i++) {
        uint32_t item = sorted[i];
        CFI_Node *child = curr->first_child;
        CFI_Node *prev = NULL;
        while (child) {
            if (child->item == item) break;
            prev = child; child = child->next_sibling;
        }
        if (child) { if (support > child->count) child->count = support; curr = child; }
        else {
            CFI_Node *new_node = alloc_cfi_node(item, support, i, curr);
            if (prev) prev->next_sibling = new_node; else curr->first_child = new_node;
            uint32_t r = ctx->root_rank_map[item];
            new_node->node_link = ctx->cfi_tree->head_table[r];
            ctx->cfi_tree->head_table[r] = new_node;
            curr = new_node;
        }
    }
    ctx->total_cfi++;
    ctx->total_footprint += len;
    update_cfips(ctx, curr, sorted, len);
    free(sorted);
}

static void fcfia_mine(FP_Tree *T, FCFIA_Context *ctx, int depth) {
    if (depth >= MAX_DEPTH - 1) return;
    if (!T->root->first_child) return;
    
    // Single Path Optimization
    bool single_path = true;
    FP_Node *sn = T->root->first_child;
    while (sn) {
        if (sn->next_sibling) { single_path = false; break; }
        sn = sn->first_child;
    }
    
    if (single_path) {
        sn = T->root->first_child;
        uint32_t path_items[MAX_DEPTH];
        uint32_t path_counts[MAX_DEPTH];
        int plen = 0;
        while (sn) {
            path_items[plen] = sn->item;
            path_counts[plen] = sn->count;
            plen++; sn = sn->first_child;
        }
        for (int i = 1; i <= plen; i++) {
            uint32_t support = path_counts[i-1];
            if (i == plen || path_counts[i-1] > path_counts[i]) {
                int total_len = ctx->head_len + i;
                uint32_t *combined = (uint32_t *)malloc(total_len * sizeof(uint32_t));
                memcpy(combined, ctx->current_head, ctx->head_len * sizeof(uint32_t));
                memcpy(combined + ctx->head_len, path_items, i * sizeof(uint32_t));
                
                bool subsumed = false;
                uint32_t last_item = combined[total_len - 1];
                uint32_t r = ctx->root_rank_map[last_item];
                uint32_t *sorted_combined = malloc(total_len * sizeof(uint32_t));
                memcpy(sorted_combined, combined, total_len * sizeof(uint32_t));
                sort_by_rank(sorted_combined, total_len, ctx->root_rank_map);
                for (CFI_Node *n = ctx->cfi_tree->head_table[r]; n; n = n->node_link) {
                    if (n->count >= support) {
                        int m_count = 0; CFI_Node *c = n; int c_idx = total_len - 1;
                        while (c && c->item != ROOT_ITEM && c_idx >= 0) {
                            if (c->item == sorted_combined[c_idx]) { m_count++; c_idx--; }
                            c = c->parent;
                        }
                        if (m_count == total_len) { subsumed = true; break; }
                    }
                }
                free(sorted_combined);
                if (!subsumed) insert_into_cfi_tree(ctx, combined, total_len, support);
                free(combined);
            }
        }
        return;
    }

    // General Case
    for (int i = (int)T->header_count - 1; i >= 0; i--) {
        uint32_t item = T->header[i].item;
        uint32_t support = T->header[i].support;
        
        // Before recursion, check if this branch is even needed
        // (Closed checking using projection)
        cfips_projection(ctx, item, ctx->head_len);
        
        bool closed = true;
        CFIP_Set *proj_set = &ctx->cfip_sets[ctx->head_len + 1];
        for (size_t pidx = 0; pidx < proj_set->count; pidx++) {
            if (proj_set->nodes[pidx]->count == support) { closed = false; break; }
        }
        
        if (closed) {
            // Temporarily update head for insertion
            ctx->current_head[ctx->head_len++] = item;
            insert_into_cfi_tree(ctx, ctx->current_head, ctx->head_len, support);
            
            // Build conditional tree
            uint32_t *cond_counts = (uint32_t *)calloc(T->header_count, sizeof(uint32_t));
            for (FP_Node *n = T->header[i].head; n; n = n->node_link) {
                FP_Node *p = n->parent;
                while (p && p->item != ROOT_ITEM) {
                    uint32_t r = T->rank_map[p->item];
                    if (r != ROOT_ITEM) cond_counts[r] += n->count;
                    p = p->parent;
                }
            }
            
            int next_header_count = 0;
            for (int j = 0; j < i; j++) if (cond_counts[j] >= ctx->min_sup) next_header_count++;
            
            if (next_header_count > 0) {
                // PEP Pruning Strategy
                // If support(head U item) == support(head U item U j), move j to head
                int pep_count = 0;
                uint32_t pep_items[MAX_DEPTH];
                for (int j = 0; j < i; j++) {
                    if (cond_counts[j] == support) {
                        pep_items[pep_count++] = T->header[j].item;
                        cond_counts[j] = 0; // Mark as processed
                    }
                }
                
                // Update next_header_count after PEP
                next_header_count = 0;
                for (int j = 0; j < i; j++) if (cond_counts[j] >= ctx->min_sup) next_header_count++;
                
                // Add PEP items to head and projection
                int old_head_len = ctx->head_len;
                for (int p = 0; p < pep_count; p++) {
                    uint32_t pitem = pep_items[p];
                    cfips_projection(ctx, pitem, ctx->head_len);
                    ctx->current_head[ctx->head_len++] = pitem;
                }
                
                if (next_header_count > 0) {
                    Header_Entry *next_header = (Header_Entry *)malloc(next_header_count * sizeof(Header_Entry));
                    uint32_t *next_rank_map = (uint32_t *)malloc((ctx->max_item_id + 1) * sizeof(uint32_t));
                    for (uint32_t j = 0; j <= ctx->max_item_id; j++) next_rank_map[j] = ROOT_ITEM;
                    int idx = 0;
                    for (int j = 0; j < i; j++) {
                        if (cond_counts[j] >= ctx->min_sup) {
                            next_header[idx].item = T->header[j].item;
                            next_header[idx].support = cond_counts[j];
                            next_header[idx].head = next_header[idx].tail = NULL;
                            idx++;
                        }
                    }
                    qsort(next_header, next_header_count, sizeof(Header_Entry), cmp_support_desc);
                    for (int j = 0; j < next_header_count; j++) next_rank_map[next_header[j].item] = j;
                    FP_Tree next_T;
                    next_T.header = next_header; next_T.header_count = next_header_count;
                    next_T.rank_map = next_rank_map; next_T.root = alloc_fp_node(ROOT_ITEM, NULL);
                    for (FP_Node *n = T->header[i].head; n; n = n->node_link) {
                        uint32_t path[MAX_DEPTH]; int plen = 0; FP_Node *p = n->parent;
                        while (p && p->item != ROOT_ITEM) {
                            if (next_rank_map[p->item] != ROOT_ITEM) path[plen++] = p->item;
                            p = p->parent;
                        }
                        if (plen > 0) {
                            sort_by_rank(path, plen, next_rank_map);
                            FP_Node *curr = next_T.root;
                            for (int j = 0; j < plen; j++) {
                                uint32_t pitem = path[j];
                                FP_Node *child = curr->first_child; FP_Node *prev = NULL;
                                while (child) { if (child->item == pitem) break; prev = child; child = child->next_sibling; }
                                if (child) { child->count += n->count; curr = child; }
                                else {
                                    FP_Node *nn = alloc_fp_node(pitem, curr); nn->count = n->count;
                                    if (prev) prev->next_sibling = nn; else curr->first_child = nn;
                                    uint32_t r = next_rank_map[pitem];
                                    if (next_header[r].tail) next_header[r].tail->node_link = nn; else next_header[r].head = nn;
                                    next_header[r].tail = nn; curr = nn;
                                }
                            }
                        }
                    }
                    fcfia_mine(&next_T, ctx, depth + 1);
                    free(next_header); free(next_rank_map);
                }
                ctx->head_len = old_head_len; // Backtrack PEP
            }
            ctx->head_len--; // Backtrack current item
            free(cond_counts);
        }
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_FCFIA_Params *p = (DM_FCFIA_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0) min_sup = 1;
    printf("[FCFIA] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    uint32_t *counts = (uint32_t *)calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    size_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= min_sup) freq_count++;
    if (freq_count == 0) { free(counts); return DM_SUCCESS; }
    Header_Entry *header = (Header_Entry *)malloc(sizeof(Header_Entry) * freq_count);
    uint32_t *rank_map = (uint32_t *)malloc((ds->max_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= ds->max_id; i++) rank_map[i] = ROOT_ITEM;
    size_t h_idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= min_sup) {
        header[h_idx].item = i; header[h_idx].support = counts[i];
        header[h_idx].head = header[h_idx].tail = NULL; h_idx++;
    }
    free(counts); qsort(header, freq_count, sizeof(Header_Entry), cmp_support_desc);
    for (size_t i = 0; i < freq_count; i++) rank_map[header[i].item] = (uint32_t)i;
    fp_alloc_init(); cfi_alloc_init();
    FP_Tree T_orig; T_orig.header = header; T_orig.header_count = freq_count; T_orig.rank_map = rank_map;
    T_orig.root = alloc_fp_node(ROOT_ITEM, NULL);
    uint32_t *filtered = (uint32_t *)malloc(sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < ds->count; i++) {
        int f_count = 0;
        for (size_t j = 0; j < data[i].count; j++) if (rank_map[data[i].items[j]] != ROOT_ITEM) filtered[f_count++] = data[i].items[j];
        if (f_count > 0) {
            sort_by_rank(filtered, f_count, rank_map);
            FP_Node *curr = T_orig.root;
            for (int j = 0; j < f_count; j++) {
                uint32_t item = filtered[j]; FP_Node *child = curr->first_child; FP_Node *prev = NULL;
                while (child) { if (child->item == item) break; prev = child; child = child->next_sibling; }
                if (child) { child->count++; curr = child; }
                else {
                    FP_Node *n = alloc_fp_node(item, curr); n->count = 1;
                    if (prev) prev->next_sibling = n; else curr->first_child = n;
                    uint32_t r = rank_map[item];
                    if (header[r].tail) header[r].tail->node_link = n; else header[r].head = n;
                    header[r].tail = n; curr = n;
                }
            }
        }
    }
    free(filtered);
    FCFIA_Context ctx; ctx.min_sup = min_sup; ctx.max_item_id = ds->max_id;
    ctx.total_cfi = 0; ctx.total_footprint = 0; ctx.head_len = 0;
    ctx.current_head = (uint32_t *)malloc(MAX_DEPTH * sizeof(uint32_t));
    ctx.root_rank_map = rank_map;
    ctx.cfi_tree = (CFI_Tree *)malloc(sizeof(CFI_Tree));
    ctx.cfi_tree->root = alloc_cfi_node(ROOT_ITEM, 0, 0, NULL);
    ctx.cfi_tree->header_count = freq_count;
    ctx.cfi_tree->head_table = (CFI_Node **)calloc(freq_count, sizeof(CFI_Node *));
    for (int i = 0; i < MAX_DEPTH; i++) { ctx.cfip_sets[i].nodes = NULL; ctx.cfip_sets[i].count = 0; ctx.cfip_sets[i].capacity = 0; }
    cfip_set_add(&ctx.cfip_sets[0], ctx.cfi_tree->root);
    fcfia_mine(&T_orig, &ctx, 0);
    printf("[FCFIA] Complete. Total frequent closed itemsets: %zu\n", ctx.total_cfi);
    dm_bench_record_results(ctx.total_cfi, ctx.total_footprint);
    free(ctx.current_head);
    for (int i = 0; i < MAX_DEPTH; i++) if (ctx.cfip_sets[i].nodes) free(ctx.cfip_sets[i].nodes);
    free(ctx.cfi_tree->head_table); free(ctx.cfi_tree); free(header); free(rank_map);
    fp_alloc_free(); cfi_alloc_free();
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "fcfia", .name = "FCFIA Algorithm",
    .description = "Efficient Algorithm for Frequent Closed Itemsets Mining (Lisheng Ma & Yi Qi, 2008).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL), .run = run
};
DM_REGISTER_ALGORITHM(algo)
