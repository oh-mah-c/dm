#include "algorithms/huim_aco.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include "core/dm_bitset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t id;
    double twu;
    DM_BitSet *bitset;
} ACO_Item;

typedef struct {
    int current_item_idx;
    uint32_t *path;
    int path_len;
    DM_BitSet *ts; // Transactions containing the path
} Ant;

/* --- CONTEXT --- */

static ACO_Item *htwuis = NULL;
static size_t htwui_count = 0;
static double total_min_util = 0;
static double *pheromone_matrix = NULL;

/* --- UTILS --- */

static double get_utility(uint32_t *path, int len, DM_Trans_Utility *src, DM_BitSet *ts) {
    double total_u = 0;
    for (size_t i = 0; i < ts->size; i++) {
        if (dm_bitset_get(ts, i)) {
            double trans_u = 0;
            for (int j = 0; j < len; j++) {
                uint32_t item_id = htwuis[path[j]].id;
                for (size_t k = 0; k < src[i].count; k++) {
                    if (src[i].items[k].id == item_id) {
                        trans_u += src[i].items[k].utility;
                        break;
                    }
                }
            }
            total_u += trans_u;
        }
    }
    return total_u;
}

static double get_twu_of_pair(int i, int j, DM_Trans_Utility *src, size_t ds_count) {
    DM_BitSet *bs = dm_bitset_copy(htwuis[i].bitset);
    dm_bitset_and(bs, htwuis[j].bitset);
    
    double twu = 0;
    for (size_t t = 0; t < ds_count; t++) {
        if (dm_bitset_get(bs, t)) twu += src[t].total_utility;
    }
    dm_bitset_free(bs);
    return twu;
}

static double get_u_of_pair(int i, int j, DM_Trans_Utility *src, size_t ds_count) {
    DM_BitSet *bs = dm_bitset_copy(htwuis[i].bitset);
    dm_bitset_and(bs, htwuis[j].bitset);
    
    double u = 0;
    uint32_t id_i = htwuis[i].id;
    uint32_t id_j = htwuis[j].id;
    
    for (size_t t = 0; t < ds_count; t++) {
        if (dm_bitset_get(bs, t)) {
            for (size_t k = 0; k < src[t].count; k++) {
                if (src[t].items[k].id == id_i || src[t].items[k].id == id_j) {
                    u += src[t].items[k].utility;
                }
            }
        }
    }
    dm_bitset_free(bs);
    return u;
}

/* --- LOGIC --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_HUIM_ACO_Params *p = (DM_HUIM_ACO_Params *)params;
    double min_util_ratio = p ? p->min_utility : 0.3;
    int sn = p ? p->pop_size : 20;
    int max_iter = p ? p->max_iter : 100;
    double alpha = p ? p->alpha : 0.1;
    double beta = p ? p->beta : 3.0;
    double gamma = p ? p->gamma : 5.0;
    double lambda = p ? p->lambda : 1000.0;
    double tau = p ? p->tau : 0.8;
    
    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    double total_u = 0;
    for (size_t i = 0; i < ds->count; i++) total_u += src[i].total_utility;
    total_min_util = total_u * min_util_ratio;
    
    // 1-HTWUIs
    double *item_twu = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) {
            item_twu[src[i].items[j].id] += src[i].total_utility;
        }
    }
    
    htwui_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_twu[i] >= total_min_util) htwui_count++;
    }
    printf("[HUIM-ACO] Threshold: %.2f, 1-HTWUIs: %zu\n", total_min_util, htwui_count);
    fflush(stdout);
    
    if (htwui_count == 0) { free(item_twu); return DM_SUCCESS; }
    
    htwuis = malloc(sizeof(ACO_Item) * htwui_count);
    size_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_twu[i] >= total_min_util) {
            htwuis[idx].id = i;
            htwuis[idx].twu = item_twu[i];
            htwuis[idx].bitset = dm_bitset_create(ds->count);
            for (size_t t = 0; t < ds->count; t++) {
                for (size_t k = 0; k < src[t].count; k++) {
                    if (src[t].items[k].id == i) {
                        dm_bitset_set(htwuis[idx].bitset, t);
                        break;
                    }
                }
            }
            idx++;
        }
    }
    free(item_twu);
    
    // Pheromone Matrix
    pheromone_matrix = calloc(htwui_count * htwui_count, sizeof(double));
    for (size_t i = 0; i < htwui_count; i++) {
        for (size_t j = i + 1; j < htwui_count; j++) {
            double twu_ij = get_twu_of_pair(i, j, src, ds->count);
            if (twu_ij >= total_min_util) {
                double val = get_u_of_pair(i, j, src, ds->count) / 2.0;
                pheromone_matrix[i * htwui_count + j] = val;
                pheromone_matrix[j * htwui_count + i] = val;
            }
        }
    }
    
    srand((unsigned int)time(NULL));
    Ant *ants = malloc(sizeof(Ant) * sn);
    for (int i = 0; i < sn; i++) {
        ants[i].path = malloc(sizeof(uint32_t) * htwui_count);
        ants[i].ts = dm_bitset_create(ds->count);
    }
    
    // ACO Outer Loop
    size_t total_found = 0;
    size_t last_total = 0;
    do {
        last_total = total_found;
        for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < sn; i++) {
            // Step 1: Select first item (Eq. 4)
            double sum_p = 0;
            for (size_t j = 0; j < htwui_count; j++) sum_p += pow(htwuis[j].twu, gamma) / lambda;
            double r = ((double)rand() / RAND_MAX) * sum_p;
            double cur_sum = 0;
            int first_item = 0;
            for (size_t j = 0; j < htwui_count; j++) {
                cur_sum += pow(htwuis[j].twu, gamma) / lambda;
                if (cur_sum >= r) { first_item = (int)j; break; }
            }
            
            ants[i].current_item_idx = first_item;
            ants[i].path[0] = first_item;
            ants[i].path_len = 1;
            dm_bitset_set_all(ants[i].ts);
            dm_bitset_and(ants[i].ts, htwuis[first_item].bitset);
            
            // Build path
            bool finished = false;
            while (!finished) {
                // Determine Successors (Succ_t: items not in path and > current index to avoid duplicates?)
                // The paper says "argmax {e_cj}", usually ACO uses a graph. 
                // In itemset mining, we can restrict to j > current to build sets.
                
                int next_item = -1;
                double rand_val = (double)rand() / RAND_MAX;
                if (rand_val >= tau) {
                    // Exploitation: argmax
                    double max_e = -1;
                    for (size_t j = ants[i].current_item_idx + 1; j < htwui_count; j++) {
                        double e = pheromone_matrix[ants[i].current_item_idx * htwui_count + j];
                        if (e > max_e) { max_e = e; next_item = (int)j; }
                    }
                } else {
                    // Exploration: Probabilistic
                    double sum_e = 0;
                    for (size_t j = ants[i].current_item_idx + 1; j < htwui_count; j++) {
                        sum_e += pheromone_matrix[ants[i].current_item_idx * htwui_count + j];
                    }
                    if (sum_e > 0) {
                        double r2 = ((double)rand() / RAND_MAX) * sum_e;
                        double cur_e = 0;
                        for (size_t j = ants[i].current_item_idx + 1; j < htwui_count; j++) {
                            cur_e += pheromone_matrix[ants[i].current_item_idx * htwui_count + j];
                            if (cur_e >= r2) { next_item = (int)j; break; }
                        }
                    }
                }
                
                if (next_item == -1) { finished = true; break; }
                
                // Check TWU of Pa + next_item
                DM_BitSet *next_ts = dm_bitset_copy(ants[i].ts);
                dm_bitset_and(next_ts, htwuis[next_item].bitset);
                
                double twu_next = 0;
                for (size_t t = 0; t < ds->count; t++) {
                    if (dm_bitset_get(next_ts, t)) twu_next += src[t].total_utility;
                }
                
                if (twu_next >= total_min_util) {
                    // Local Update (Eq. 2)
                    double u_pair = get_u_of_pair(ants[i].current_item_idx, next_item, src, ds->count);
                    double delta_e = pow(u_pair, alpha) / lambda;
                    pheromone_matrix[ants[i].current_item_idx * htwui_count + next_item] += delta_e;
                    pheromone_matrix[next_item * htwui_count + ants[i].current_item_idx] += delta_e;
                    
                    // Update Path
                    ants[i].path[ants[i].path_len++] = next_item;
                    ants[i].current_item_idx = next_item;
                    dm_bitset_free(ants[i].ts);
                    ants[i].ts = next_ts;
                    
                    // Check if HUI
                    double util = get_utility(ants[i].path, ants[i].path_len, src, ants[i].ts);
                    if (util >= total_min_util) {
                        total_found++;
                        // Global Update (Eq. 3)
                        for (int l = 0; l < ants[i].path_len - 1; l++) {
                            int u = ants[i].path[l];
                            int v = ants[i].path[l+1];
                            double twu_uv = get_twu_of_pair(u, v, src, ds->count);
                            double delta_global = pow(twu_uv, beta) / lambda;
                            pheromone_matrix[u * htwui_count + v] += delta_global;
                            pheromone_matrix[v * htwui_count + u] += delta_global;
                        }
                    }
                } else {
                    dm_bitset_free(next_ts);
                    finished = true;
                }
            }
        }
    }
    
    } while (total_found > last_total);
    
    // Cleanup
    for (size_t i = 0; i < htwui_count; i++) dm_bitset_free(htwuis[i].bitset);
    free(htwuis);
    free(pheromone_matrix);
    for (int i = 0; i < sn; i++) {
        free(ants[i].path);
        dm_bitset_free(ants[i].ts);
    }
    free(ants);
    
    dm_bench_record_results(total_found, 0);
    return DM_SUCCESS;
}

static DM_Algorithm huim_aco_algo = {
    .id = "huim_aco",
    .name = "HUIM-ACO",
    .description = "Mining High-Utility Itemsets using Ant Colony Optimization.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(huim_aco_algo)
