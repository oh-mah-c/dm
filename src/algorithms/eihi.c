#include "algorithms/eihi.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t tid;
    double iutil;
    double rutil;
} EIHI_Tuple;

typedef struct {
    uint32_t item;
    EIHI_Tuple *tuples;
    size_t tuple_count;
    size_t n_start_idx; // Index where tuples for N start
    double sum_iutil_d;
    double sum_iutil_n;
    double sum_rutil_d;
    double sum_rutil_n;
} EIHI_UtilityList;

typedef struct EIHI_TrieNode {
    uint32_t item;
    double utility;
    struct EIHI_TrieNode **children;
    size_t child_count;
    size_t child_cap;
} EIHI_TrieNode;

typedef struct {
    uint32_t u, v;
    double twu;
} EIHI_EUCS_Entry;

typedef struct {
    EIHI_EUCS_Entry *entries;
    size_t count;
    size_t capacity;
} EIHI_EUCS;

/* --- TRIE MANAGEMENT --- */

static EIHI_TrieNode* create_trie_node(uint32_t item) {
    EIHI_TrieNode *node = calloc(1, sizeof(EIHI_TrieNode));
    node->item = item;
    node->child_cap = 4;
    node->children = malloc(sizeof(EIHI_TrieNode*) * node->child_cap);
    return node;
}

static void trie_insert(EIHI_TrieNode *root, uint32_t *items, size_t count, double utility) {
    EIHI_TrieNode *curr = root;
    for (size_t i = 0; i < count; i++) {
        uint32_t item = items[i];
        EIHI_TrieNode *next = NULL;
        for (size_t j = 0; j < curr->child_count; j++) {
            if (curr->children[j]->item == item) {
                next = curr->children[j];
                break;
            }
        }
        if (!next) {
            next = create_trie_node(item);
            if (curr->child_count >= curr->child_cap) {
                curr->child_cap *= 2;
                curr->children = realloc(curr->children, sizeof(EIHI_TrieNode*) * curr->child_cap);
            }
            curr->children[curr->child_count++] = next;
        }
        curr = next;
    }
    curr->utility = utility;
}

static void free_trie(EIHI_TrieNode *node) {
    if (!node) return;
    for (size_t i = 0; i < node->child_count; i++) {
        free_trie(node->children[i]);
    }
    free(node->children);
    free(node);
}

/* --- UTILITY LIST JOIN --- */

static EIHI_UtilityList* construct(EIHI_UtilityList *p, EIHI_UtilityList *px, EIHI_UtilityList *py, size_t d_count) {
    EIHI_UtilityList *pxy = calloc(1, sizeof(EIHI_UtilityList));
    pxy->item = py->item;
    pxy->tuples = malloc(sizeof(EIHI_Tuple) * (px->tuple_count < py->tuple_count ? px->tuple_count : py->tuple_count));
    pxy->n_start_idx = 0;

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
            
            if (tid < d_count) {
                pxy->sum_iutil_d += iutil;
                pxy->sum_rutil_d += py->tuples[iy].rutil;
                pxy->n_start_idx++;
            } else {
                pxy->sum_iutil_n += iutil;
                pxy->sum_rutil_n += py->tuples[iy].rutil;
            }
            
            pxy->tuple_count++;
            ix++; iy++;
        } else if (px->tuples[ix].tid < py->tuples[iy].tid) ix++;
        else iy++;
    }
    return pxy;
}

static void free_utility_list(EIHI_UtilityList *ul) {
    if (!ul) return;
    free(ul->tuples);
    free(ul);
}

/* --- SEARCH --- */

static size_t trie_count_huis(EIHI_TrieNode *node) {
    if (!node) return 0;
    size_t count = (node->utility > 0) ? 1 : 0;
    for (size_t i = 0; i < node->child_count; i++) {
        count += trie_count_huis(node->children[i]);
    }
    return count;
}

static size_t total_hui_count = 0;
static EIHI_TrieNode *hui_trie = NULL;

static void search_baseline(EIHI_UtilityList *p, EIHI_UtilityList **extensions, size_t ext_count, double min_util, size_t d_count, EIHI_EUCS *eucs, uint32_t *prefix, size_t prefix_len) {
    for (size_t i = 0; i < ext_count; i++) {
        EIHI_UtilityList *px = extensions[i];
        if (px->sum_iutil_d >= min_util) {
            prefix[prefix_len] = px->item;
            trie_insert(hui_trie, prefix, prefix_len + 1, px->sum_iutil_d);
            total_hui_count++;
        }

        if (px->sum_iutil_d + px->sum_rutil_d >= min_util) {
            EIHI_UtilityList **ext_px = malloc(sizeof(EIHI_UtilityList*) * (ext_count - i - 1));
            size_t ext_px_count = 0;
            prefix[prefix_len] = px->item;
            for (size_t j = i + 1; j < ext_count; j++) {
                EIHI_UtilityList *py = extensions[j];
                EIHI_UtilityList *pxy = construct(p, px, py, d_count);
                if (pxy->sum_iutil_d + pxy->sum_rutil_d >= min_util) {
                    ext_px[ext_px_count++] = pxy;
                } else {
                    free_utility_list(pxy);
                }
            }
            if (ext_px_count > 0) {
                search_baseline(px, ext_px, ext_px_count, min_util, d_count, eucs, prefix, prefix_len + 1);
            }
            for (size_t k = 0; k < ext_px_count; k++) free_utility_list(ext_px[k]);
            free(ext_px);
        }
    }
}

static void search(EIHI_UtilityList *p, EIHI_UtilityList **extensions, size_t ext_count, double min_util, size_t d_count, EIHI_EUCS *eucs, uint32_t *prefix, size_t prefix_len) {
    for (size_t i = 0; i < ext_count; i++) {
        EIHI_UtilityList *px = extensions[i];
        double total_iutil = px->sum_iutil_d + px->sum_iutil_n;
        if (total_iutil >= min_util) {
            prefix[prefix_len] = px->item;
            trie_insert(hui_trie, prefix, prefix_len + 1, total_iutil);
            total_hui_count++;
        }

        if (total_iutil + px->sum_rutil_d + px->sum_rutil_n >= min_util) {
            if (px->tuple_count > px->n_start_idx) {
                EIHI_UtilityList **ext_px = malloc(sizeof(EIHI_UtilityList*) * (ext_count - i - 1));
                size_t ext_px_count = 0;
                prefix[prefix_len] = px->item;
                for (size_t j = i + 1; j < ext_count; j++) {
                    EIHI_UtilityList *py = extensions[j];
                    EIHI_UtilityList *pxy = construct(p, px, py, d_count);
                    if (pxy->tuple_count > pxy->n_start_idx) {
                        ext_px[ext_px_count++] = pxy;
                    } else {
                        free_utility_list(pxy);
                    }
                }
                if (ext_px_count > 0) {
                    search(px, ext_px, ext_px_count, min_util, d_count, eucs, prefix, prefix_len + 1);
                }
                for (size_t k = 0; k < ext_px_count; k++) free_utility_list(ext_px[k]);
                free(ext_px);
            }
        }
    }
}

static void run_incremental(EIHI_UtilityList **initial_ext, size_t i_star_count, double min_util, size_t d_count, EIHI_EUCS *eucs, bool *appears_in_n) {
    printf("[EIHI] Phase 1: Mining Baseline D...\n");
    uint32_t *prefix = malloc(sizeof(uint32_t) * 128);
    search_baseline(NULL, initial_ext, i_star_count, min_util, d_count, eucs, prefix, 0);

    printf("[EIHI] Phase 2: Mining Incremental N...\n");
    EIHI_UtilityList **seeds = malloc(sizeof(EIHI_UtilityList*) * i_star_count);
    size_t seed_count = 0;
    for (size_t i = 0; i < i_star_count; i++) {
        if (appears_in_n[initial_ext[i]->item]) {
            seeds[seed_count++] = initial_ext[i];
        }
    }
    search(NULL, seeds, seed_count, min_util, d_count, eucs, prefix, 0);
    free(prefix);
    free(seeds);
}

static int cmp_item_twu(void *twu_arr, const void *a, const void *b) {
    uint32_t i1 = *(const uint32_t *)a;
    uint32_t i2 = *(const uint32_t *)b;
    double *twu = (double *)twu_arr;
    if (twu[i1] < twu[i2]) return -1;
    if (twu[i1] > twu[i2]) return 1;
    return (int)i1 - (int)i2;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_EIHI_Params *p = (DM_EIHI_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    
    // Determine split point (Incremental simulation)
    // If not specified, we take last 20% as 'N'
    size_t total_count = ds->count;
    size_t d_count = (size_t)(total_count * 0.8);
    if (d_count == 0 && total_count > 0) d_count = 1;
    
    printf("[EIHI] Running Incremental Miner. Total: %zu, D: %zu, N: %zu\n", total_count, d_count, total_count - d_count);

    total_hui_count = 0;
    hui_trie = create_trie_node(0);

    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;

    // 1. Calculate TWU for all items (D + N)
    double *twu = calloc(ds->max_id + 1, sizeof(double));
    bool *appears_in_n = calloc(ds->max_id + 1, sizeof(bool));
    for (size_t i = 0; i < total_count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            twu[data[i].items[j].id] += data[i].total_utility;
            if (i >= d_count) appears_in_n[data[i].items[j].id] = true;
        }
    }

    // 2. Identify I* (TWU >= minutil)
    uint32_t *i_star = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    size_t i_star_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu[i] >= min_util) {
            i_star[i_star_count++] = i;
        }
    }

    // 3. Establish Order (Ascending TWU)
    // Note: To be 100% correct for multiple updates, we should keep the order from D.
    // For this simulation, we just sort.
    qsort_s(i_star, i_star_count, sizeof(uint32_t), cmp_item_twu, twu);

    uint32_t *rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < i_star_count; i++) rank[i_star[i]] = (uint32_t)i;

    // 4. Build Initial Utility Lists and EUCS
    EIHI_UtilityList **initial_ext = malloc(sizeof(EIHI_UtilityList*) * i_star_count);
    for (size_t i = 0; i < i_star_count; i++) {
        initial_ext[i] = calloc(1, sizeof(EIHI_UtilityList));
        initial_ext[i]->item = i_star[i];
        initial_ext[i]->tuples = malloc(sizeof(EIHI_Tuple) * 16);
        initial_ext[i]->tuple_count = 0;
    }

    EIHI_EUCS eucs = { malloc(sizeof(EIHI_EUCS_Entry) * 1024), 0, 1024 };

    for (size_t i = 0; i < total_count; i++) {
        // Filter and sort items in transaction
        uint32_t t_items[data[i].count];
        double t_utils[data[i].count];
        size_t t_count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (rank[data[i].items[j].id] != 0xFFFFFFFF) {
                t_items[t_count] = data[i].items[j].id;
                t_utils[t_count] = data[i].items[j].utility;
                t_count++;
            }
        }
        // Sort by rank
        for (size_t j = 0; j < t_count; j++) {
            for (size_t k = j + 1; k < t_count; k++) {
                if (rank[t_items[j]] > rank[t_items[k]]) {
                    uint32_t tmp_i = t_items[j]; t_items[j] = t_items[k]; t_items[k] = tmp_i;
                    double tmp_u = t_utils[j]; t_utils[j] = t_utils[k]; t_utils[k] = tmp_u;
                }
            }
        }

        double remaining_utility = 0;
        for (size_t j = t_count; j-- > 0; ) {
            uint32_t item_id = t_items[j];
            uint32_t r = rank[item_id];
            EIHI_UtilityList *ul = initial_ext[r];
            
            if (ul->tuple_count % 16 == 0 && ul->tuple_count > 0) {
                ul->tuples = realloc(ul->tuples, sizeof(EIHI_Tuple) * (ul->tuple_count + 16));
            }
            
            ul->tuples[ul->tuple_count].tid = (uint32_t)i;
            ul->tuples[ul->tuple_count].iutil = t_utils[j];
            ul->tuples[ul->tuple_count].rutil = remaining_utility;
            
            if (i < d_count) {
                ul->sum_iutil_d += t_utils[j];
                ul->sum_rutil_d += remaining_utility;
                ul->n_start_idx++;
            } else {
                ul->sum_iutil_n += t_utils[j];
                ul->sum_rutil_n += remaining_utility;
            }
            ul->tuple_count++;

            for (size_t k = 0; k < j; k++) {
                uint32_t u = item_id, v = t_items[k];
                if (u > v) { uint32_t t = u; u = v; v = t; }
                bool found = false;
                for (size_t m = 0; m < eucs.count; m++) {
                    if (eucs.entries[m].u == u && eucs.entries[m].v == v) {
                        eucs.entries[m].twu += data[i].total_utility;
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    if (eucs.count >= eucs.capacity) {
                        eucs.capacity *= 2;
                        eucs.entries = realloc(eucs.entries, sizeof(EIHI_EUCS_Entry) * eucs.capacity);
                    }
                    eucs.entries[eucs.count].u = u;
                    eucs.entries[eucs.count].v = v;
                    eucs.entries[eucs.count].twu = data[i].total_utility;
                    eucs.count++;
                }
            }
            remaining_utility += t_utils[j];
        }
    }

    // 5. Execute Incremental Mining Flow
    run_incremental(initial_ext, i_star_count, min_util, d_count, &eucs, appears_in_n);

    total_hui_count = trie_count_huis(hui_trie);
    printf("[EIHI] Found %zu HUIs in Trie.\n", total_hui_count);

    // Cleanup
    for (size_t i = 0; i < i_star_count; i++) free_utility_list(initial_ext[i]);
    free(initial_ext);
    free(i_star);
    free(rank);
    free(twu);
    free(appears_in_n);
    free(eucs.entries);
    free_trie(hui_trie);

    dm_bench_record_results(total_hui_count, 0);
    return DM_SUCCESS;
}

static DM_Algorithm eihi_algo = {
    .id = "eihi",
    .name = "EIHI Algorithm",
    .description = "Efficient Incremental High Utility Itemset Mining.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(eihi_algo)
