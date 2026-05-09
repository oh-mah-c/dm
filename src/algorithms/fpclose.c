#include "algorithms/fpclose.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

#define ROOT_ITEM 0xFFFFFFFF
#define MAX_DEPTH 1024

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
    FP_Node *root;
    Header_Entry *header;
    size_t header_count;
    uint32_t *rank_map;
} FP_Tree;

typedef struct CFI_Node {
    uint32_t item;
    uint32_t count;
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
    uint32_t *rank_maps[MAX_DEPTH];
    CFI_Tree *active_trees[MAX_DEPTH];
    uint32_t *prefixes[MAX_DEPTH];
    int prefixes_len[MAX_DEPTH];
    size_t total_fci;
    size_t total_footprint;
    uint32_t min_sup;
    uint32_t max_item_id;
} FPClose_Context;

/* --- MEMORY POOLS --- */

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

static void fp_alloc_init(void) {
    g_fp_alloc.head = NULL;
}

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
    n->item = item;
    n->count = 0;
    n->parent = parent;
    n->first_child = NULL;
    n->next_sibling = NULL;
    n->node_link = NULL;
    return n;
}

static void fp_alloc_free(void) {
    NodePool *curr = g_fp_alloc.head;
    while (curr) {
        NodePool *next = curr->next;
        free(curr->nodes);
        free(curr);
        curr = next;
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

static void cfi_alloc_init(void) {
    g_cfi_alloc.head = NULL;
}

static CFI_Node* alloc_cfi_node(uint32_t item, uint32_t count, CFI_Node *parent) {
    if (!g_cfi_alloc.head || g_cfi_alloc.head->used >= g_cfi_alloc.head->capacity) {
        CFINodePool *np = (CFINodePool *)malloc(sizeof(CFINodePool));
        np->capacity = 8192;
        np->nodes = (CFI_Node *)malloc(sizeof(CFI_Node) * np->capacity);
        np->used = 0;
        np->next = g_cfi_alloc.head;
        g_cfi_alloc.head = np;
    }
    CFI_Node *n = &g_cfi_alloc.head->nodes[g_cfi_alloc.head->used++];
    n->item = item;
    n->count = count;
    n->parent = parent;
    n->first_child = NULL;
    n->next_sibling = NULL;
    n->node_link = NULL;
    return n;
}

static void cfi_alloc_free(void) {
    CFINodePool *curr = g_cfi_alloc.head;
    while (curr) {
        CFINodePool *next = curr->next;
        free(curr->nodes);
        free(curr);
        curr = next;
    }
    g_cfi_alloc.head = NULL;
}

/* --- UTILS --- */

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

static int cmp_support_desc(const void *a, const void *b) {
    const Header_Entry *ha = (const Header_Entry *)a;
    const Header_Entry *hb = (const Header_Entry *)b;
    if (ha->support > hb->support) return -1;
    if (ha->support < hb->support) return 1;
    if (ha->item < hb->item) return -1;
    if (ha->item > hb->item) return 1;
    return 0;
}

static void sort_path(uint32_t *items, size_t count, uint32_t *rank_map) {
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

static bool in_array(uint32_t item, const uint32_t *arr, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (arr[i] == item) return true;
        if (arr[i] > item) break;
    }
    return false;
}

static bool is_single_path(FP_Tree *T) {
    if (!T->root->first_child) return true;
    FP_Node *n = T->root->first_child;
    while (n) {
        if (n->next_sibling) return false;
        n = n->first_child;
    }
    return true;
}

static CFI_Tree* alloc_cfi_tree(size_t header_count) {
    CFI_Tree *C = (CFI_Tree *)malloc(sizeof(CFI_Tree));
    C->header_count = header_count;
    if (header_count > 0) {
        C->head_table = (CFI_Node **)calloc(header_count, sizeof(CFI_Node*));
    } else {
        C->head_table = NULL;
    }
    C->root = alloc_cfi_node(ROOT_ITEM, 0, NULL);
    return C;
}

static void free_cfi_tree(CFI_Tree *C) {
    if (C->head_table) free(C->head_table);
    free(C);
}

/* --- FPCLOSE LOGIC --- */

static void insert_cfi_to_all(FPClose_Context *ctx, uint32_t *full_cfi, int len, uint32_t c, int max_depth) {
    ctx->total_fci++;
    ctx->total_footprint += len;
    
    for (int d = 0; d <= max_depth; d++) {
        uint32_t *Z_d = (uint32_t *)malloc(len * sizeof(uint32_t));
        int z_len = 0;
        for (int j = 0; j < len; j++) {
            if (!in_array(full_cfi[j], ctx->prefixes[d], ctx->prefixes_len[d])) {
                Z_d[z_len++] = full_cfi[j];
            }
        }
        
        if (z_len > 0) {
            sort_path(Z_d, z_len, ctx->rank_maps[d]);
            
            CFI_Node *curr = ctx->active_trees[d]->root;
            for (int x = 0; x < z_len; x++) {
                uint32_t item = Z_d[x];
                CFI_Node *child = curr->first_child;
                CFI_Node *prev = NULL;
                while (child) {
                    if (child->item == item) break;
                    prev = child;
                    child = child->next_sibling;
                }
                if (child) {
                    if (c > child->count) child->count = c;
                    curr = child;
                } else {
                    CFI_Node *new_node = alloc_cfi_node(item, c, curr);
                    if (prev) prev->next_sibling = new_node;
                    else curr->first_child = new_node;
                    
                    uint32_t idx = ctx->rank_maps[d][item];
                    new_node->node_link = ctx->active_trees[d]->head_table[idx];
                    ctx->active_trees[d]->head_table[idx] = new_node;
                    
                    curr = new_node;
                }
            }
        }
        free(Z_d);
    }
}

static void mine_tree(FP_Tree *T_X, uint32_t *A_X, FPClose_Context *ctx, int depth) {
    int k = (int)T_X->header_count;
    for (int i = k - 1; i >= 0; i--) {
        uint32_t item_i = T_X->header[i].item;
        uint32_t supp_i = T_X->header[i].support;
        
        CFI_Tree *C_X = ctx->active_trees[depth];
        bool subsumed = false;
        if (C_X && C_X->head_table) {
            for (CFI_Node *n = C_X->head_table[i]; n; n = n->node_link) {
                if (n->count >= supp_i) {
                    subsumed = true;
                    break;
                }
            }
        }
        if (subsumed) continue;
        
        uint32_t *new_prefix = (uint32_t *)malloc((ctx->prefixes_len[depth] + 1) * sizeof(uint32_t));
        if (ctx->prefixes_len[depth] > 0) {
            memcpy(new_prefix, ctx->prefixes[depth], ctx->prefixes_len[depth] * sizeof(uint32_t));
        }
        new_prefix[ctx->prefixes_len[depth]] = item_i;
        qsort(new_prefix, ctx->prefixes_len[depth] + 1, sizeof(uint32_t), cmp_uint32);
        
        ctx->prefixes[depth + 1] = new_prefix;
        ctx->prefixes_len[depth + 1] = ctx->prefixes_len[depth] + 1;
        
        uint32_t *cond_counts = (uint32_t *)calloc(i > 0 ? i : 1, sizeof(uint32_t));
        int new_k = 0;
        for (int v = 0; v < i; v++) {
            uint32_t pair_count = A_X[i * (i - 1) / 2 + v];
            if (pair_count >= ctx->min_sup) {
                cond_counts[v] = pair_count;
                new_k++;
            }
        }
        
        if (new_k == 0) {
            insert_cfi_to_all(ctx, new_prefix, ctx->prefixes_len[depth + 1], supp_i, depth);
        } else {
            Header_Entry *new_header = (Header_Entry *)malloc(new_k * sizeof(Header_Entry));
            int idx = 0;
            for (int v = 0; v < i; v++) {
                if (cond_counts[v] >= ctx->min_sup) {
                    new_header[idx].item = T_X->header[v].item;
                    new_header[idx].support = cond_counts[v];
                    new_header[idx].head = NULL;
                    new_header[idx].tail = NULL;
                    idx++;
                }
            }
            qsort(new_header, new_k, sizeof(Header_Entry), cmp_support_desc);
            
            uint32_t *new_rank_map = (uint32_t *)malloc((ctx->max_item_id + 1) * sizeof(uint32_t));
            for (uint32_t y = 0; y <= ctx->max_item_id; y++) new_rank_map[y] = ROOT_ITEM;
            for (int y = 0; y < new_k; y++) new_rank_map[new_header[y].item] = y;
            
            ctx->rank_maps[depth + 1] = new_rank_map;
            
            FP_Tree *T_Y = (FP_Tree *)malloc(sizeof(FP_Tree));
            T_Y->header = new_header;
            T_Y->header_count = new_k;
            T_Y->rank_map = new_rank_map;
            T_Y->root = alloc_fp_node(ROOT_ITEM, NULL);
            
            size_t A_Y_size = (size_t)new_k * (new_k - 1) / 2;
            uint32_t *A_Y = A_Y_size > 0 ? (uint32_t *)calloc(A_Y_size, sizeof(uint32_t)) : NULL;
            
            uint32_t *path = (uint32_t *)malloc(i * sizeof(uint32_t));
            for (FP_Node *n = T_X->header[i].head; n; n = n->node_link) {
                int p_len = 0;
                FP_Node *p = n->parent;
                while (p && p->item != ROOT_ITEM) {
                    if (new_rank_map[p->item] != ROOT_ITEM) {
                        path[p_len++] = p->item;
                    }
                    p = p->parent;
                }
                if (p_len > 0) {
                    sort_path(path, p_len, new_rank_map);
                    
                    FP_Node *curr = T_Y->root;
                    for (int x = 0; x < p_len; x++) {
                        uint32_t item = path[x];
                        FP_Node *child = curr->first_child;
                        FP_Node *prev = NULL;
                        while (child) {
                            if (child->item == item) break;
                            prev = child;
                            child = child->next_sibling;
                        }
                        if (child) {
                            child->count += n->count;
                            curr = child;
                        } else {
                            FP_Node *new_node = alloc_fp_node(item, curr);
                            new_node->count = n->count;
                            if (prev) prev->next_sibling = new_node;
                            else curr->first_child = new_node;
                            
                            uint32_t r = new_rank_map[item];
                            if (new_header[r].tail) new_header[r].tail->node_link = new_node;
                            else new_header[r].head = new_node;
                            new_header[r].tail = new_node;
                            
                            curr = new_node;
                        }
                    }
                    
                    if (A_Y) {
                        for (int x = 0; x < p_len; x++) {
                            uint32_t idx_x = new_rank_map[path[x]];
                            for (int y = 0; y < x; y++) {
                                uint32_t idx_y = new_rank_map[path[y]];
                                A_Y[idx_x * (idx_x - 1) / 2 + idx_y] += n->count;
                            }
                        }
                    }
                }
            }
            free(path);
            
            if (is_single_path(T_Y)) {
                FP_Node *n = T_Y->root->first_child;
                uint32_t *p_items = (uint32_t *)malloc(new_k * sizeof(uint32_t));
                uint32_t *p_counts = (uint32_t *)malloc(new_k * sizeof(uint32_t));
                int len = 0;
                while (n) {
                    p_items[len] = n->item;
                    p_counts[len] = n->count;
                    len++;
                    n = n->first_child;
                }
                
                for (int m = 0; m <= len; m++) {
                    uint32_t cfi_count = (m == 0) ? supp_i : p_counts[m-1];
                    bool is_cfi = false;
                    if (m == len) is_cfi = true;
                    else if (m == 0 && cfi_count > p_counts[0]) is_cfi = true;
                    else if (m > 0 && cfi_count > p_counts[m]) is_cfi = true;
                    
                    if (is_cfi) {
                        uint32_t *z_local = (uint32_t *)malloc((m + 1) * sizeof(uint32_t));
                        for (int j = 0; j < m; j++) z_local[j] = p_items[j];
                        z_local[m] = item_i;
                        sort_path(z_local, m + 1, T_X->rank_map);
                        
                        bool sub = false;
                        uint32_t ik = z_local[m];
                        uint32_t idx_ik = T_X->rank_map[ik];
                        if (C_X && C_X->head_table) {
                            for (CFI_Node *cn = C_X->head_table[idx_ik]; cn; cn = cn->node_link) {
                                if (cn->count >= cfi_count) {
                                    CFI_Node *cp = cn->parent;
                                    int match_idx = m - 1;
                                    while (cp && cp->item != ROOT_ITEM && match_idx >= 0) {
                                        if (cp->item == z_local[match_idx]) match_idx--;
                                        cp = cp->parent;
                                    }
                                    if (match_idx < 0) {
                                        sub = true;
                                        break;
                                    }
                                }
                            }
                        }
                        
                        if (!sub) {
                            uint32_t *full_cfi = (uint32_t *)malloc((ctx->prefixes_len[depth] + 1 + m) * sizeof(uint32_t));
                            if (ctx->prefixes_len[depth] > 0) {
                                memcpy(full_cfi, ctx->prefixes[depth], ctx->prefixes_len[depth] * sizeof(uint32_t));
                            }
                            full_cfi[ctx->prefixes_len[depth]] = item_i;
                            for (int j = 0; j < m; j++) full_cfi[ctx->prefixes_len[depth] + 1 + j] = p_items[j];
                            qsort(full_cfi, ctx->prefixes_len[depth] + 1 + m, sizeof(uint32_t), cmp_uint32);
                            
                            insert_cfi_to_all(ctx, full_cfi, ctx->prefixes_len[depth] + 1 + m, cfi_count, depth);
                            free(full_cfi);
                        }
                        free(z_local);
                    }
                }
                free(p_items);
                free(p_counts);
            } else {
                if (new_header[0].support < supp_i) {
                    insert_cfi_to_all(ctx, new_prefix, ctx->prefixes_len[depth + 1], supp_i, depth);
                }
                CFI_Tree *C_Y = alloc_cfi_tree(new_k);
                ctx->active_trees[depth + 1] = C_Y;
                mine_tree(T_Y, A_Y, ctx, depth + 1);
                free_cfi_tree(C_Y);
            }
            
            if (A_Y) free(A_Y);
            free(new_header);
            free(new_rank_map);
            free(T_Y);
        }
        
        free(cond_counts);
        free(new_prefix);
    }
}

/* --- MAIN ENTRY --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_FPCLOSE_Params *p = (DM_FPCLOSE_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0) min_sup = 1;

    printf("[FPclose] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    
    uint32_t *counts = (uint32_t *)calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }

    size_t freq_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) freq_count++;
    }

    if (freq_count == 0) {
        free(counts);
        printf("[FPclose] Complete. Total frequent closed itemsets found: 0\n");
        return DM_SUCCESS;
    }

    Header_Entry *header = (Header_Entry *)malloc(sizeof(Header_Entry) * freq_count);
    size_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            header[idx].item = i;
            header[idx].support = counts[i];
            header[idx].head = NULL;
            header[idx].tail = NULL;
            idx++;
        }
    }
    free(counts);

    qsort(header, freq_count, sizeof(Header_Entry), cmp_support_desc);

    uint32_t *rank_map = (uint32_t *)malloc((ds->max_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= ds->max_id; i++) rank_map[i] = ROOT_ITEM;
    for (size_t i = 0; i < freq_count; i++) rank_map[header[i].item] = (uint32_t)i;

    fp_alloc_init();
    cfi_alloc_init();

    FP_Tree T_orig;
    T_orig.header = header;
    T_orig.header_count = freq_count;
    T_orig.rank_map = rank_map;
    T_orig.root = alloc_fp_node(ROOT_ITEM, NULL);

    size_t A_orig_size = (size_t)freq_count * (freq_count - 1) / 2;
    uint32_t *A_orig = A_orig_size > 0 ? (uint32_t *)calloc(A_orig_size, sizeof(uint32_t)) : NULL;

    uint32_t *filtered = (uint32_t *)malloc(sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < ds->count; i++) {
        size_t f_count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            if (rank_map[item] != ROOT_ITEM) {
                filtered[f_count++] = item;
            }
        }
        if (f_count > 0) {
            sort_path(filtered, f_count, rank_map);
            
            FP_Node *curr = T_orig.root;
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
                    child->count++;
                    curr = child;
                } else {
                    FP_Node *n = alloc_fp_node(item, curr);
                    n->count = 1;
                    if (prev) prev->next_sibling = n;
                    else curr->first_child = n;
                    
                    uint32_t r = rank_map[item];
                    if (header[r].tail) header[r].tail->node_link = n;
                    else header[r].head = n;
                    header[r].tail = n;
                    
                    curr = n;
                }
            }
            
            if (A_orig) {
                for (size_t x = 0; x < f_count; x++) {
                    uint32_t idx_x = rank_map[filtered[x]];
                    for (size_t y = 0; y < x; y++) {
                        uint32_t idx_y = rank_map[filtered[y]];
                        A_orig[idx_x * (idx_x - 1) / 2 + idx_y]++;
                    }
                }
            }
        }
    }
    free(filtered);

    FPClose_Context ctx;
    ctx.min_sup = min_sup;
    ctx.max_item_id = ds->max_id;
    ctx.total_fci = 0;
    ctx.total_footprint = 0;
    ctx.rank_maps[0] = rank_map;
    ctx.prefixes_len[0] = 0;
    ctx.prefixes[0] = NULL;

    if (is_single_path(&T_orig)) {
        FP_Node *n = T_orig.root->first_child;
        uint32_t *p_items = (uint32_t *)malloc(freq_count * sizeof(uint32_t));
        uint32_t *p_counts = (uint32_t *)malloc(freq_count * sizeof(uint32_t));
        int len = 0;
        while (n) {
            p_items[len] = n->item;
            p_counts[len] = n->count;
            len++;
            n = n->first_child;
        }
        for (int m = 1; m <= len; m++) {
            if (m == len || p_counts[m-1] > p_counts[m]) {
                ctx.total_fci++;
                ctx.total_footprint += m;
            }
        }
        free(p_items);
        free(p_counts);
    } else {
        CFI_Tree *C_root = alloc_cfi_tree(freq_count);
        ctx.active_trees[0] = C_root;
        
        mine_tree(&T_orig, A_orig, &ctx, 0);
        
        free_cfi_tree(C_root);
    }

    if (A_orig) free(A_orig);
    free(header);
    free(rank_map);
    fp_alloc_free();
    cfi_alloc_free();

    printf("[FPclose] Complete. Total frequent closed itemsets found: %zu\n", ctx.total_fci);
    dm_bench_record_results(ctx.total_fci, ctx.total_footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "fpclose",
    .name = "FPclose Algorithm",
    .description = "Mining Closed Frequent Itemsets with FP-tree Array Technique (Grahne & Zhu 2003).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
