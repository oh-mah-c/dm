#include "algorithms/fpclose.h"
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

typedef struct {
    uint32_t *counts;
    int size;
} FP_Array;

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
    FP_Array *array;
} FP_Tree;

typedef struct FCINode {
    uint32_t item;
    uint32_t count;
    uint32_t level;
    struct FCINode *parent;
    struct FCINode *first_child;
    struct FCINode *next_sibling;
    struct FCINode *node_link;
} FCINode;

typedef struct {
    FCINode *head;
    FCINode *tail;
} FCI_HeaderEntry;

typedef struct {
    FCINode *root;
    FCI_HeaderEntry *header; 
    uint32_t *rank_map; 
} CFITree;

typedef struct {
    uint32_t min_sup;
    uint32_t max_item_id;
    size_t total_cfi;
    size_t total_footprint;
    
    uint32_t prefix[MAX_DEPTH];
    uint32_t *rank_maps[MAX_DEPTH];
    CFITree *C_trees[MAX_DEPTH];
    int depth;
} FPclose_Context;

/* --- LOGIC --- */

static inline int get_fp_array_idx(int i, int j, int n) {
    if (i > j) { int t = i; i = j; j = t; }
    return i * n - (i * (i + 1)) / 2 + (j - i - 1);
}

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

static FCINode* insert_cfi_tree(CFITree *C, uint32_t *items, int len, uint32_t count) {
    FCINode *curr = C->root;
    for (int i = 0; i < len; i++) {
        uint32_t item = items[i];
        FCINode *child = curr->first_child, *prev = NULL;
        while (child) {
            if (child->item == item) break;
            prev = child;
            child = child->next_sibling;
        }
        if (child) {
            curr = child;
        } else {
            FCINode *n = calloc(1, sizeof(FCINode));
            n->item = item;
            n->level = curr == C->root ? 1 : curr->level + 1;
            n->parent = curr;
            if (prev) prev->next_sibling = n;
            else curr->first_child = n;
            
            uint32_t r = C->rank_map[item];
            if (C->header[r].tail) C->header[r].tail->node_link = n;
            else C->header[r].head = n;
            C->header[r].tail = n;
            
            curr = n;
        }
    }
    if (count > curr->count) curr->count = count;
    return curr;
}

static bool check_single_item_closed(uint32_t item, uint32_t count, CFITree *C) {
    uint32_t r = C->rank_map[item];
    for (FCINode *n = C->header[r].head; n; n = n->node_link) {
        if (n->count >= count) return true;
    }
    return false;
}

static bool check_path_closed(uint32_t *path, int path_len, uint32_t count, CFITree *C) {
    if (path_len == 0) return false;
    uint32_t least_freq_item = path[path_len - 1];
    uint32_t r = C->rank_map[least_freq_item];
    
    for (FCINode *n = C->header[r].head; n; n = n->node_link) {
        if (n->count >= count) {
            if (n->level < (uint32_t)path_len) continue;
            bool all_found = true;
            for (int i = 0; i < path_len - 1; i++) {
                uint32_t item = path[i];
                FCINode *curr = n->parent;
                bool found = false;
                while (curr && curr->item != ROOT_ITEM) {
                    if (curr->item == item) { found = true; break; }
                    curr = curr->parent;
                }
                if (!found) { all_found = false; break; }
            }
            if (all_found) return true;
        }
    }
    return false;
}

static void push_up_cfi(uint32_t *z_part, int z_len, uint32_t support, FPclose_Context *ctx) {
    uint32_t current_z[MAX_DEPTH];
    memcpy(current_z, z_part, z_len * sizeof(uint32_t));
    int curr_len = z_len;
    
    for (int d = ctx->depth - 1; d >= 0; d--) {
        current_z[curr_len++] = ctx->prefix[d];
        sort_by_rank(current_z, curr_len, ctx->rank_maps[d]);
        insert_cfi_tree(ctx->C_trees[d], current_z, curr_len, support);
    }
}

static CFITree* create_local_cfi(CFITree *C_parent, uint32_t item_i, FP_Tree *T_Y) {
    CFITree *C_Y = calloc(1, sizeof(CFITree));
    C_Y->root = calloc(1, sizeof(FCINode));
    C_Y->root->item = ROOT_ITEM;
    C_Y->header = calloc(T_Y->header_count, sizeof(FCI_HeaderEntry));
    C_Y->rank_map = T_Y->rank_map;
    
    uint32_t r_i = C_parent->rank_map[item_i];
    for (FCINode *n = C_parent->header[r_i].head; n; n = n->node_link) {
        uint32_t Z[MAX_DEPTH];
        int z_len = 0;
        FCINode *curr = n->parent;
        while (curr && curr->item != ROOT_ITEM) {
            if (T_Y->rank_map[curr->item] != ROOT_ITEM) {
                Z[z_len++] = curr->item;
            }
            curr = curr->parent;
        }
        if (z_len > 0) {
            sort_by_rank(Z, z_len, T_Y->rank_map);
            insert_cfi_tree(C_Y, Z, z_len, n->count);
        }
    }
    return C_Y;
}

static FP_Tree* construct_conditional_tree(FP_Tree *T, int i, uint32_t *cond_counts, int next_h_count, FPclose_Context *ctx) {
    FP_Tree *Ty = calloc(1, sizeof(FP_Tree));
    Ty->header_count = next_h_count;
    Ty->header = calloc(next_h_count, sizeof(FP_Header));
    
    typedef struct { uint32_t item, support; } TempH;
    TempH *th = calloc(i > 0 ? i : 1, sizeof(TempH));
    int th_idx = 0;
    for (int j = 0; j < i; j++) {
        if (cond_counts[j] >= ctx->min_sup) {
            th[th_idx].item = T->header[j].item;
            th[th_idx].support = cond_counts[j];
            th_idx++;
        }
    }
    
    for (int x = 0; x < next_h_count - 1; x++) {
        for (int y = x + 1; y < next_h_count; y++) {
            if (th[x].support < th[y].support || (th[x].support == th[y].support && th[x].item > th[y].item)) {
                TempH t = th[x]; th[x] = th[y]; th[y] = t;
            }
        }
    }
    
    Ty->rank_map = malloc((ctx->max_item_id + 1) * sizeof(uint32_t));
    for (uint32_t k = 0; k <= ctx->max_item_id; k++) Ty->rank_map[k] = ROOT_ITEM;
    
    for (int x = 0; x < next_h_count; x++) {
        Ty->header[x].item = th[x].item;
        Ty->header[x].support = th[x].support;
        Ty->rank_map[th[x].item] = x;
    }
    free(th);
    
    Ty->root = calloc(1, sizeof(FP_Node));
    Ty->root->item = ROOT_ITEM;
    
    bool use_array = (next_h_count < 2000);
    if (use_array && next_h_count > 1) {
        Ty->array = calloc(1, sizeof(FP_Array));
        Ty->array->size = next_h_count;
        Ty->array->counts = calloc(next_h_count * (next_h_count - 1) / 2, sizeof(uint32_t));
    }
    
    for (FP_Node *n = T->header[i].head; n; n = n->node_link) {
        uint32_t path[MAX_DEPTH];
        int plen = 0;
        FP_Node *p = n->parent;
        while (p && p->item != ROOT_ITEM) {
            if (Ty->rank_map[p->item] != ROOT_ITEM) {
                path[plen++] = p->item;
            }
            p = p->parent;
        }
        if (plen > 0) {
            sort_by_rank(path, plen, Ty->rank_map);
            
            if (Ty->array) {
                for (int x = 0; x < plen - 1; x++) {
                    for (int y = x + 1; y < plen; y++) {
                        int r_x = Ty->rank_map[path[x]];
                        int r_y = Ty->rank_map[path[y]];
                        Ty->array->counts[get_fp_array_idx(r_x, r_y, next_h_count)] += n->count;
                    }
                }
            }
            
            FP_Node *curr = Ty->root;
            for (int x = 0; x < plen; x++) {
                uint32_t pitem = path[x];
                FP_Node *child = curr->first_child, *prev = NULL;
                while (child) {
                    if (child->item == pitem) break;
                    prev = child; child = child->next_sibling;
                }
                if (child) {
                    child->count += n->count;
                    curr = child;
                } else {
                    FP_Node *nn = calloc(1, sizeof(FP_Node));
                    nn->item = pitem;
                    nn->count = n->count;
                    nn->parent = curr;
                    if (prev) prev->next_sibling = nn; else curr->first_child = nn;
                    uint32_t r = Ty->rank_map[pitem];
                    if (Ty->header[r].tail) Ty->header[r].tail->node_link = nn;
                    else Ty->header[r].head = nn;
                    Ty->header[r].tail = nn;
                    curr = nn;
                }
            }
        }
    }
    return Ty;
}

static void free_cfi_tree_nodes(FCINode *n) {
    if (!n) return;
    free_cfi_tree_nodes(n->first_child);
    free_cfi_tree_nodes(n->next_sibling);
    free(n);
}

static void free_fp_tree_nodes(FP_Node *n) {
    if (!n) return;
    free_fp_tree_nodes(n->first_child);
    free_fp_tree_nodes(n->next_sibling);
    free(n);
}

static void fpclose_mine(FP_Tree *T, CFITree *C, FPclose_Context *ctx) {
    if (is_single_path(T)) {
        FP_Node *curr = T->root->first_child;
        uint32_t path[MAX_DEPTH];
        int path_len = 0;
        while (curr) {
            path[path_len++] = curr->item;
            bool is_end = (!curr->first_child || curr->first_child->count != curr->count);
            if (is_end) {
                uint32_t support = curr->count;
                bool backward_subsumed = check_path_closed(path, path_len, support, C);
                if (!backward_subsumed) {
                    insert_cfi_tree(C, path, path_len, support);
                    ctx->total_cfi++;
                    ctx->total_footprint += (ctx->depth + path_len);
                    push_up_cfi(path, path_len, support, ctx);
                }
            }
            curr = curr->first_child;
        }
    } else {
        for (int i = (int)T->header_count - 1; i >= 0; i--) {
            uint32_t item = T->header[i].item;
            uint32_t support = T->header[i].support;
            
            bool backward_subsumed = check_single_item_closed(item, support, C);
            if (backward_subsumed) continue;
            
            bool forward_subsumed = false;
            uint32_t cond_counts[MAX_DEPTH] = {0};
            int next_h_count = 0;
            
            if (T->array) {
                for (int j = 0; j < i; j++) {
                    cond_counts[j] = T->array->counts[get_fp_array_idx(j, i, T->header_count)];
                    if (cond_counts[j] == support) forward_subsumed = true;
                    if (cond_counts[j] >= ctx->min_sup) next_h_count++;
                }
            } else {
                for (FP_Node *n = T->header[i].head; n; n = n->node_link) {
                    FP_Node *p = n->parent;
                    while (p && p->item != ROOT_ITEM) {
                        uint32_t r = T->rank_map[p->item];
                        if (r != ROOT_ITEM && r < (uint32_t)i) cond_counts[r] += n->count;
                        p = p->parent;
                    }
                }
                for (int j = 0; j < i; j++) {
                    if (cond_counts[j] == support) forward_subsumed = true;
                    if (cond_counts[j] >= ctx->min_sup) next_h_count++;
                }
            }
            
            if (!forward_subsumed) {
                insert_cfi_tree(C, &item, 1, support);
                ctx->total_cfi++;
                ctx->total_footprint += (ctx->depth + 1);
                push_up_cfi(&item, 1, support, ctx);
            }
            
            if (next_h_count > 0) {
                FP_Tree *Ty = construct_conditional_tree(T, i, cond_counts, next_h_count, ctx);
                CFITree *Cy = create_local_cfi(C, item, Ty);
                
                ctx->prefix[ctx->depth] = item;
                ctx->rank_maps[ctx->depth] = C->rank_map;
                ctx->C_trees[ctx->depth] = C;
                ctx->depth++;
                
                fpclose_mine(Ty, Cy, ctx);
                
                ctx->depth--;
                
                free_cfi_tree_nodes(Cy->root);
                free(Cy->header);
                free(Cy);
                
                free_fp_tree_nodes(Ty->root);
                free(Ty->header);
                free(Ty->rank_map);
                if (Ty->array) { free(Ty->array->counts); free(Ty->array); }
                free(Ty);
            }
        }
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_FPCLOSE_Params *p = params; double ms_p = p ? p->min_support : 0.01;
    uint32_t min_sup = (ms_p < 1.0) ? (uint32_t)ceil(ms_p * ds->count) : (uint32_t)ms_p;
    if (min_sup == 0) min_sup = 1;
    printf("[FPclose] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);
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
    memset(&T, 0, sizeof(FP_Tree));
    T.header = h; T.header_count = f_cnt; T.rank_map = rm; 
    T.root = calloc(1, sizeof(FP_Node));
    T.root->item = ROOT_ITEM;
    
    bool use_array = (f_cnt < 2000);
    if (use_array && f_cnt > 1) {
        T.array = calloc(1, sizeof(FP_Array));
        T.array->size = f_cnt;
        T.array->counts = calloc(f_cnt * (f_cnt - 1) / 2, sizeof(uint32_t));
    }
    
    uint32_t *filt = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < ds->count; i++) {
        int fc = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (rm[data[i].items[j]] != ROOT_ITEM) filt[fc++] = data[i].items[j];
        }
        if (fc > 0) {
            sort_by_rank(filt, fc, rm);
            FP_Node *curr = T.root;
            
            if (T.array) {
                for (int x = 0; x < fc - 1; x++) {
                    for (int y = x + 1; y < fc; y++) {
                        int r_x = rm[filt[x]];
                        int r_y = rm[filt[y]];
                        T.array->counts[get_fp_array_idx(r_x, r_y, f_cnt)]++;
                    }
                }
            }
            
            for (int j = 0; j < fc; j++) {
                uint32_t it = filt[j];
                FP_Node *ch = curr->first_child, *pr = NULL;
                while (ch) { if (ch->item == it) break; pr = ch; ch = ch->next_sibling; }
                if (ch) { ch->count++; curr = ch; }
                else {
                    FP_Node *n = calloc(1, sizeof(FP_Node));
                    n->item = it; n->count = 1; n->parent = curr;
                    if (pr) pr->next_sibling = n; else curr->first_child = n;
                    uint32_t r = rm[it];
                    if (h[r].tail) h[r].tail->node_link = n; else h[r].head = n;
                    h[r].tail = n; curr = n;
                }
            }
        }
    }
    free(filt);
    
    CFITree *C = calloc(1, sizeof(CFITree));
    C->root = calloc(1, sizeof(FCINode));
    C->root->item = ROOT_ITEM;
    C->header = calloc(f_cnt, sizeof(FCI_HeaderEntry));
    C->rank_map = rm;
    
    FPclose_Context ctx;
    memset(&ctx, 0, sizeof(FPclose_Context));
    ctx.min_sup = min_sup;
    ctx.max_item_id = ds->max_id;
    
    fpclose_mine(&T, C, &ctx);
    
    printf("[FPclose] Complete. Total frequent closed itemsets found: %zu\n", ctx.total_cfi);
    dm_bench_record_results(ctx.total_cfi, ctx.total_footprint);
    
    free_cfi_tree_nodes(C->root);
    free(C->header);
    free(C);
    
    free_fp_tree_nodes(T.root);
    free(T.header);
    free(T.rank_map);
    if (T.array) { free(T.array->counts); free(T.array); }
    
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "fpclose",
    .name = "FPclose Algorithm",
    .description = "FPclose with FP-Array and local CFI-trees",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
