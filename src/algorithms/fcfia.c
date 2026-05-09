#include "algorithms/fcfia.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

#define ROOT_ITEM 0xFFFFFFFF
#define MAX_DEPTH 4096

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
} FP_Header;

typedef struct {
    FP_Node *root;
    FP_Header *header;
    size_t header_count;
    uint32_t *rank_map;
} FP_Tree;

typedef struct CFINode {
    uint32_t item;
    uint32_t count;
    struct CFINode *parent;
    struct CFINode *first_child;
    struct CFINode *next_sibling;
    struct CFINode *node_link;
} CFINode;

typedef struct {
    CFINode *head;
    CFINode *tail;
} CFI_Header;

typedef struct {
    CFINode *root;
    CFI_Header *header;
    size_t header_size;
} CFITree;

typedef struct {
    CFINode **nodes;
    size_t count;
    size_t capacity;
} CFIPSet;

typedef struct {
    uint32_t min_sup;
    uint32_t max_item_id;
    CFITree *cfi_tree;
    size_t total_cfi;
    size_t total_footprint;
    uint32_t *global_rank_map;
} FCFIA_Context;

/* --- FP TREE MEMORY --- */

static FP_Node* alloc_fp_node(uint32_t item, FP_Node *parent) {
    FP_Node *n = calloc(1, sizeof(FP_Node));
    n->item = item;
    n->parent = parent;
    return n;
}

static void free_fp_tree_nodes(FP_Node *n) {
    if (!n) return;
    free_fp_tree_nodes(n->first_child);
    free_fp_tree_nodes(n->next_sibling);
    free(n);
}

/* --- LOGIC --- */

static int cmp_sup_desc(const void *a, const void *b) {
    const FP_Header *ha = a, *hb = b;
    if (ha->support > hb->support) return -1;
    if (ha->support < hb->support) return 1;
    return (ha->item < hb->item) ? -1 : 1;
}

static void sort_by_rank(uint32_t *items, int len, uint32_t *rank_map) {
    for (int i = 1; i < len; i++) {
        uint32_t key = items[i]; int j = i - 1;
        while (j >= 0 && rank_map[items[j]] > rank_map[key]) { items[j+1] = items[j]; j--; }
        items[j+1] = key;
    }
}

static bool is_single_path(FP_Tree *T) {
    FP_Node *curr = T->root;
    while (curr) {
        if (curr->first_child && curr->first_child->next_sibling) return false;
        curr = curr->first_child;
    }
    return true;
}

static bool node_has_ancestor(CFINode *node, uint32_t item) {
    CFINode *curr = node->parent;
    while (curr && curr->item != ROOT_ITEM) {
        if (curr->item == item) return true;
        curr = curr->parent;
    }
    return false;
}

static void CFIPs_Projection(uint32_t x, int head_idx, CFIPSet *CFIPs, CFITree *tree) {
    if (head_idx == 0) {
        CFIPs[0].count = 0;
        for (CFINode *n = tree->header[x].head; n; n = n->node_link) {
            if (CFIPs[0].count >= CFIPs[0].capacity) {
                CFIPs[0].capacity = CFIPs[0].capacity ? CFIPs[0].capacity * 2 : 16;
                CFIPs[0].nodes = realloc(CFIPs[0].nodes, CFIPs[0].capacity * sizeof(CFINode*));
            }
            CFIPs[0].nodes[CFIPs[0].count++] = n;
        }
    } else {
        int prev_idx = head_idx - 1;
        CFIPs[head_idx].count = 0;
        for (size_t i = 0; i < CFIPs[prev_idx].count; i++) {
            CFINode *P = CFIPs[prev_idx].nodes[i];
            if (node_has_ancestor(P, x)) {
                if (CFIPs[head_idx].count >= CFIPs[head_idx].capacity) {
                    CFIPs[head_idx].capacity = CFIPs[head_idx].capacity ? CFIPs[head_idx].capacity * 2 : 16;
                    CFIPs[head_idx].nodes = realloc(CFIPs[head_idx].nodes, CFIPs[head_idx].capacity * sizeof(CFINode*));
                }
                CFIPs[head_idx].nodes[CFIPs[head_idx].count++] = P;
            }
        }
    }
}

static bool closed_checking(int head_len, uint32_t support, CFIPSet *CFIPs, uint32_t *head, int total_head_len) {
    if (head_len == 0) return false;
    int last_idx = head_len - 1;
    for (size_t i = 0; i < CFIPs[last_idx].count; i++) {
        CFINode *P = CFIPs[last_idx].nodes[i];
        if (P->count == support) {
            bool all_found = true;
            for (int k = 0; k < total_head_len; k++) {
                if (head[k] == P->item) continue;
                if (!node_has_ancestor(P, head[k])) {
                    all_found = false;
                    break;
                }
            }
            if (all_found) return true;
        }
    }
    return false;
}

static bool closed_checking_root(uint32_t *head, int len, uint32_t support, CFITree *tree, uint32_t *rank_map) {
    if (len == 0) return false;
    uint32_t least_freq = head[0];
    for (int i = 1; i < len; i++) {
        if (rank_map[head[i]] > rank_map[least_freq]) least_freq = head[i];
    }
    CFINode *n = tree->header[least_freq].head;
    while (n) {
        if (n->count == support) {
            bool all_found = true;
            for (int k = 0; k < len; k++) {
                if (head[k] == n->item) continue;
                if (!node_has_ancestor(n, head[k])) { all_found = false; break; }
            }
            if (all_found) return true;
        }
        n = n->node_link;
    }
    return false;
}

static CFINode* Insert_CFI_Tree(CFITree *tree, uint32_t *head, int len, uint32_t support, uint32_t *rank_map) {
    uint32_t sorted_head[MAX_DEPTH];
    memcpy(sorted_head, head, len * sizeof(uint32_t));
    for (int i = 0; i < len - 1; i++) {
        for (int j = i + 1; j < len; j++) {
            if (rank_map[sorted_head[i]] > rank_map[sorted_head[j]]) {
                uint32_t tmp = sorted_head[i]; sorted_head[i] = sorted_head[j]; sorted_head[j] = tmp;
            }
        }
    }
    
    CFINode *curr = tree->root;
    for (int i = 0; i < len; i++) {
        uint32_t item = sorted_head[i];
        CFINode *child = curr->first_child;
        CFINode *prev = NULL;
        while (child) {
            if (child->item == item) break;
            prev = child;
            child = child->next_sibling;
        }
        if (child) {
            curr = child;
        } else {
            CFINode *n = calloc(1, sizeof(CFINode));
            n->item = item;
            n->count = 0;
            n->parent = curr;
            if (prev) prev->next_sibling = n;
            else curr->first_child = n;
            
            if (tree->header[item].tail) tree->header[item].tail->node_link = n;
            else tree->header[item].head = n;
            tree->header[item].tail = n;
            
            curr = n;
        }
    }
    curr->count = support;
    return curr;
}

static void Update_CFIPs(CFINode *leaf, int head_len, CFIPSet *CFIPs) {
    for (int h = 0; h < head_len; h++) {
        if (CFIPs[h].count >= CFIPs[h].capacity) {
            CFIPs[h].capacity = CFIPs[h].capacity ? CFIPs[h].capacity * 2 : 16;
            CFIPs[h].nodes = realloc(CFIPs[h].nodes, CFIPs[h].capacity * sizeof(CFINode*));
        }
        CFIPs[h].nodes[CFIPs[h].count++] = leaf;
    }
}

static FP_Tree* construct_conditional_tree(FP_Tree *T, int header_idx, uint32_t min_sup, uint32_t *pruned, int *p_count, FCFIA_Context *ctx) {
    uint32_t support = T->header[header_idx].support;
    
    uint32_t *cond_counts = calloc(header_idx > 0 ? header_idx : 1, sizeof(uint32_t));
    for (FP_Node *n = T->header[header_idx].head; n; n = n->node_link) {
        FP_Node *p = n->parent;
        while (p && p->item != ROOT_ITEM) {
            uint32_t r = T->rank_map[p->item];
            if (r != ROOT_ITEM && r < (uint32_t)header_idx) cond_counts[r] += n->count;
            p = p->parent;
        }
    }
    
    *p_count = 0;
    int next_h_count = 0;
    for (int j = 0; j < header_idx; j++) {
        if (cond_counts[j] == support) {
            pruned[(*p_count)++] = T->header[j].item;
            cond_counts[j] = 0;
        } else if (cond_counts[j] >= min_sup) {
            next_h_count++;
        }
    }
    
    if (next_h_count == 0) {
        free(cond_counts);
        return NULL;
    }
    
    FP_Tree *next_T = calloc(1, sizeof(FP_Tree));
    next_T->header_count = next_h_count;
    next_T->header = calloc(next_h_count, sizeof(FP_Header));
    
    typedef struct { uint32_t item, support; } TempH;
    TempH *th = calloc(header_idx, sizeof(TempH));
    int th_idx = 0;
    for (int j = 0; j < header_idx; j++) {
        if (cond_counts[j] >= min_sup) {
            th[th_idx].item = T->header[j].item;
            th[th_idx].support = cond_counts[j];
            th_idx++;
        }
    }
    
    for (int i = 0; i < next_h_count - 1; i++) {
        for (int j = i + 1; j < next_h_count; j++) {
            if (th[i].support < th[j].support || (th[i].support == th[j].support && th[i].item > th[j].item)) {
                TempH t = th[i]; th[i] = th[j]; th[j] = t;
            }
        }
    }
    
    next_T->rank_map = malloc((ctx->max_item_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= ctx->max_item_id; i++) next_T->rank_map[i] = ROOT_ITEM;
    
    for (int i = 0; i < next_h_count; i++) {
        next_T->header[i].item = th[i].item;
        next_T->header[i].support = th[i].support;
        next_T->rank_map[th[i].item] = i;
    }
    
    next_T->root = calloc(1, sizeof(FP_Node));
    next_T->root->item = ROOT_ITEM;
    
    for (FP_Node *n = T->header[header_idx].head; n; n = n->node_link) {
        uint32_t path[MAX_DEPTH];
        int plen = 0;
        FP_Node *p = n->parent;
        while (p && p->item != ROOT_ITEM) {
            if (next_T->rank_map[p->item] != ROOT_ITEM) {
                path[plen++] = p->item;
            }
            p = p->parent;
        }
        if (plen > 0) {
            sort_by_rank(path, plen, next_T->rank_map);
            FP_Node *curr = next_T->root;
            for (int i = 0; i < plen; i++) {
                uint32_t pitem = path[i];
                FP_Node *child = curr->first_child, *prev = NULL;
                while (child) {
                    if (child->item == pitem) break;
                    prev = child; child = child->next_sibling;
                }
                if (child) {
                    child->count += n->count;
                    curr = child;
                } else {
                    FP_Node *nn = alloc_fp_node(pitem, curr);
                    nn->count = n->count;
                    if (prev) prev->next_sibling = nn; else curr->first_child = nn;
                    uint32_t r = next_T->rank_map[pitem];
                    if (next_T->header[r].tail) next_T->header[r].tail->node_link = nn;
                    else next_T->header[r].head = nn;
                    next_T->header[r].tail = nn;
                    curr = nn;
                }
            }
        }
    }
    free(th);
    free(cond_counts);
    return next_T;
}

static void FCFIA_Mine(FP_Tree *T, uint32_t *head, int head_len, int total_head_len, CFIPSet *CFIPs, FCFIA_Context *ctx) {
    if (is_single_path(T)) {
        FP_Node *curr = T->root->first_child;
        uint32_t path_items[MAX_DEPTH];
        int path_len = 0;
        while (curr) {
            path_items[path_len++] = curr->item;
            bool is_end = (!curr->first_child || curr->first_child->count != curr->count);
            if (is_end) {
                uint32_t support = curr->count;
                uint32_t new_head[MAX_DEPTH];
                memcpy(new_head, head, total_head_len * sizeof(uint32_t));
                for (int j = path_len - 1; j >= 0; j--) {
                    new_head[total_head_len + (path_len - 1 - j)] = path_items[j];
                }
                
                bool is_closed = true;
                if (head_len > 0) {
                    is_closed = !closed_checking(head_len, support, CFIPs, new_head, total_head_len + path_len);
                } else {
                    is_closed = !closed_checking_root(new_head, total_head_len + path_len, support, ctx->cfi_tree, ctx->global_rank_map);
                }
                
                if (is_closed) {
                    CFINode *leaf = Insert_CFI_Tree(ctx->cfi_tree, new_head, total_head_len + path_len, support, ctx->global_rank_map);
                    Update_CFIPs(leaf, head_len, CFIPs);
                    ctx->total_cfi++;
                    ctx->total_footprint += (total_head_len + path_len);
                }
            }
            curr = curr->first_child;
        }
    } else {
        for (int i = (int)T->header_count - 1; i >= 0; i--) {
            uint32_t item = T->header[i].item;
            uint32_t support = T->header[i].support;
            
            head[total_head_len] = item;
            CFIPs_Projection(item, head_len, CFIPs, ctx->cfi_tree);
            
            bool is_closed = !closed_checking(head_len + 1, support, CFIPs, head, total_head_len + 1);
            
            if (is_closed) {
                CFINode *leaf = Insert_CFI_Tree(ctx->cfi_tree, head, total_head_len + 1, support, ctx->global_rank_map);
                Update_CFIPs(leaf, head_len + 1, CFIPs);
                ctx->total_cfi++;
                ctx->total_footprint += (total_head_len + 1);
            } else {
                continue;
            }
            
            uint32_t pruned[MAX_DEPTH];
            int p_count = 0;
            FP_Tree *Thead = construct_conditional_tree(T, i, ctx->min_sup, pruned, &p_count, ctx);
            
            for (int k = 0; k < p_count; k++) {
                head[total_head_len + 1 + k] = pruned[k];
            }
            
            if (Thead) {
                FCFIA_Mine(Thead, head, head_len + 1, total_head_len + 1 + p_count, CFIPs, ctx);
                free_fp_tree_nodes(Thead->root);
                free(Thead->header);
                free(Thead->rank_map);
                free(Thead);
            }
        }
    }
}

static void free_cfi_tree_nodes(CFINode *n) {
    if (!n) return;
    free_cfi_tree_nodes(n->first_child);
    free_cfi_tree_nodes(n->next_sibling);
    free(n);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_FCFIA_Params *p = params; double ms_p = p ? p->min_support : 0.01;
    uint32_t min_sup = (ms_p < 1.0) ? (uint32_t)ceil(ms_p * ds->count) : (uint32_t)ms_p;
    if (min_sup == 0) min_sup = 1;
    printf("[FCFIA] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);
    DM_Trans_Simple *data = ds->payload;
    
    uint32_t *cnts = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) cnts[data[i].items[j]]++;
    }
    
    size_t f_cnt = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (cnts[i] >= min_sup) f_cnt++;
    }
    
    if (f_cnt == 0) { free(cnts); return DM_SUCCESS; }
    
    FP_Header *h = calloc(f_cnt, sizeof(FP_Header));
    uint32_t *rm = malloc((ds->max_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= ds->max_id; i++) rm[i] = ROOT_ITEM;
    
    size_t hi = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (cnts[i] >= min_sup) {
            h[hi].item = i; h[hi].support = cnts[i]; hi++;
        }
    }
    free(cnts);
    
    qsort(h, f_cnt, sizeof(FP_Header), cmp_sup_desc);
    for (size_t i = 0; i < f_cnt; i++) rm[h[i].item] = (uint32_t)i;
    
    FP_Tree T;
    T.header = h;
    T.header_count = f_cnt;
    T.rank_map = rm;
    T.root = alloc_fp_node(ROOT_ITEM, NULL);
    
    uint32_t *filt = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < ds->count; i++) {
        int fc = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (rm[data[i].items[j]] != ROOT_ITEM) filt[fc++] = data[i].items[j];
        }
        if (fc > 0) {
            sort_by_rank(filt, fc, rm);
            FP_Node *curr = T.root;
            for (int j = 0; j < fc; j++) {
                uint32_t it = filt[j];
                FP_Node *ch = curr->first_child, *pr = NULL;
                while (ch) { if (ch->item == it) break; pr = ch; ch = ch->next_sibling; }
                if (ch) { ch->count++; curr = ch; }
                else {
                    FP_Node *n = alloc_fp_node(it, curr);
                    n->count = 1;
                    if (pr) pr->next_sibling = n; else curr->first_child = n;
                    uint32_t r = rm[it];
                    if (h[r].tail) h[r].tail->node_link = n; else h[r].head = n;
                    h[r].tail = n; curr = n;
                }
            }
        }
    }
    free(filt);
    
    FCFIA_Context ctx;
    ctx.min_sup = min_sup;
    ctx.max_item_id = ds->max_id;
    ctx.total_cfi = 0;
    ctx.total_footprint = 0;
    ctx.global_rank_map = rm;
    
    ctx.cfi_tree = calloc(1, sizeof(CFITree));
    ctx.cfi_tree->root = calloc(1, sizeof(CFINode));
    ctx.cfi_tree->root->item = ROOT_ITEM;
    ctx.cfi_tree->header_size = ds->max_id + 1;
    ctx.cfi_tree->header = calloc(ctx.cfi_tree->header_size, sizeof(CFI_Header));
    
    uint32_t head[MAX_DEPTH];
    CFIPSet CFIPs[MAX_DEPTH] = {0};
    
    FCFIA_Mine(&T, head, 0, 0, CFIPs, &ctx);
    
    printf("[FCFIA] Complete. Total frequent closed itemsets found: %zu\n", ctx.total_cfi);
    dm_bench_record_results(ctx.total_cfi, ctx.total_footprint);
    
    for (int i = 0; i < MAX_DEPTH; i++) free(CFIPs[i].nodes);
    free_cfi_tree_nodes(ctx.cfi_tree->root);
    free(ctx.cfi_tree->header);
    free(ctx.cfi_tree);
    
    free_fp_tree_nodes(T.root);
    free(h);
    free(rm);
    
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "fcfia",
    .name = "FCFIA Algorithm",
    .description = "FCFIA with CFIPs Projection",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};
DM_REGISTER_ALGORITHM(algo)
