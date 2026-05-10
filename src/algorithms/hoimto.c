#include "algorithms/hoimto.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    uint32_t item;
    uint32_t *tids;
    size_t num_tids;
    double ioub;
} HOIMTOItemset;

static void mine_hoimto_dfs(HOIMTOItemset *classes, size_t class_size, size_t k, uint32_t *prefix, uint32_t min_sup, double min_io, double *to_over_len, double *base_io, size_t *total_ho_count, size_t *total_ho_footprint, uint32_t *buffer) {
    if (class_size < 2) return;
    
    for (size_t i = 0; i < class_size; i++) {
        HOIMTOItemset *P1 = &classes[i];
        
        HOIMTOItemset *children = NULL;
        if (class_size - i - 1 > 0) {
            children = malloc((class_size - i - 1) * sizeof(HOIMTOItemset));
        }
        size_t child_count = 0;
        
        for (size_t j = i + 1; j < class_size; j++) {
            HOIMTOItemset *P2 = &classes[j];
            
            size_t p1_idx = 0, p2_idx = 0, new_idx = 0;
            while (p1_idx < P1->num_tids && p2_idx < P2->num_tids) {
                if (P1->tids[p1_idx] < P2->tids[p2_idx]) {
                    p1_idx++;
                } else if (P1->tids[p1_idx] > P2->tids[p2_idx]) {
                    p2_idx++;
                } else {
                    buffer[new_idx++] = P1->tids[p1_idx];
                    p1_idx++;
                    p2_idx++;
                }
            }
            
            if (new_idx >= min_sup) {
                double ioub = P1->ioub + base_io[P2->item];
                
                HOIMTOItemset *P = &children[child_count++];
                P->item = P2->item;
                P->num_tids = new_idx;
                P->tids = malloc(new_idx * sizeof(uint32_t));
                memcpy(P->tids, buffer, new_idx * sizeof(uint32_t));
                P->ioub = ioub;
                
                if (ioub >= min_io) {
                    double io = 0.0;
                    for (size_t t = 0; t < new_idx; t++) {
                        io += to_over_len[buffer[t]];
                    }
                    io *= (double)k;
                    
                    if (io >= min_io) {
                        (*total_ho_count)++;
                        (*total_ho_footprint) += k;
                    }
                }
            }
        }
        
        if (child_count > 0) {
            prefix[k - 2] = P1->item;
            mine_hoimto_dfs(children, child_count, k + 1, prefix, min_sup, min_io, to_over_len, base_io, total_ho_count, total_ho_footprint, buffer);
            
            for (size_t c = 0; c < child_count; c++) {
                free(children[c].tids);
            }
        }
        if (children) free(children);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_HOIMTO_Params *hoimto_params = (DM_HOIMTO_Params *)params;
    double min_sup_param = hoimto_params ? hoimto_params->min_sup : 0.5;
    double min_io_param  = hoimto_params ? hoimto_params->min_io : 0.025;
    
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    double min_io = min_io_param;
    if (min_sup == 0) min_sup = 1;

    printf("[HOIMTO] Starting on %zu transactions. Min Support: %u, Min IO: %.4f\n", ds->count, min_sup, min_io);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    uint32_t *g_tsize = malloc(ds->count * sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        g_tsize[i] = data[i].count;
    }

    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            counts[data[i].items[j]]++;
        }
    }

    uint32_t **item_tids = malloc((ds->max_id + 1) * sizeof(uint32_t*));
    uint32_t *item_idx = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] > 0) item_tids[i] = malloc(counts[i] * sizeof(uint32_t));
        else item_tids[i] = NULL;
    }

    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            item_tids[item][item_idx[item]++] = i;
        }
    }
    free(item_idx);

    double *to_table = malloc(ds->count * sizeof(double));
    uint32_t *buffer1 = malloc(ds->count * sizeof(uint32_t));
    uint32_t *buffer2 = malloc(ds->count * sizeof(uint32_t));
    
    for (size_t i = 0; i < ds->count; i++) {
        if (data[i].count == 0) {
            to_table[i] = 1.0;
            continue;
        }
        size_t min_item_idx = 0;
        uint32_t min_sup_val = counts[data[i].items[0]];
        for (size_t j = 1; j < data[i].count; j++) {
            if (counts[data[i].items[j]] < min_sup_val) {
                min_sup_val = counts[data[i].items[j]];
                min_item_idx = j;
            }
        }
        
        uint32_t start_item = data[i].items[min_item_idx];
        size_t current_len = counts[start_item];
        memcpy(buffer1, item_tids[start_item], current_len * sizeof(uint32_t));
        
        for (size_t j = 0; j < data[i].count; j++) {
            if (j == min_item_idx) continue;
            uint32_t item = data[i].items[j];
            
            size_t p1 = 0, p2 = 0, new_len = 0;
            while (p1 < current_len && p2 < counts[item]) {
                if (buffer1[p1] < item_tids[item][p2]) p1++;
                else if (buffer1[p1] > item_tids[item][p2]) p2++;
                else {
                    buffer2[new_len++] = buffer1[p1];
                    p1++; p2++;
                }
            }
            current_len = new_len;
            memcpy(buffer1, buffer2, current_len * sizeof(uint32_t));
            if (current_len == 0) break;
        }
        to_table[i] = (double)current_len / (double)ds->count;
    }
    free(buffer1);
    free(buffer2);
    
    double *to_over_len = malloc(ds->count * sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        if (data[i].count > 0) {
            to_over_len[i] = to_table[i] / (double)data[i].count;
        } else {
            to_over_len[i] = 0.0;
        }
    }
    free(to_table);

    double *base_io = calloc(ds->max_id + 1, sizeof(double));
    HOIMTOItemset *C1 = malloc((ds->max_id + 1) * sizeof(HOIMTOItemset));
    size_t c1_count = 0;
    
    size_t total_ho_count = 0;
    size_t total_ho_footprint = 0;

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            double io = 0.0;
            for (size_t t = 0; t < counts[i]; t++) {
                io += to_over_len[item_tids[i][t]];
            }
            base_io[i] = io;
            
            if (io >= min_io) {
                total_ho_count++;
                total_ho_footprint += 1;
            }
            
            HOIMTOItemset *it = &C1[c1_count++];
            it->item = i;
            it->num_tids = counts[i];
            it->tids = malloc(counts[i] * sizeof(uint32_t));
            memcpy(it->tids, item_tids[i], counts[i] * sizeof(uint32_t));
            it->ioub = io;
        }
    }
    
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_tids[i]) free(item_tids[i]);
    }
    free(item_tids);
    free(counts);

    uint32_t *buffer = malloc(ds->count * sizeof(uint32_t));
    uint32_t *prefix = malloc((ds->max_id + 1) * sizeof(uint32_t));
    
    mine_hoimto_dfs(C1, c1_count, 2, prefix, min_sup, min_io, to_over_len, base_io, &total_ho_count, &total_ho_footprint, buffer);
    
    free(buffer);
    free(prefix);
    
    for (size_t i = 0; i < c1_count; i++) {
        free(C1[i].tids);
    }
    free(C1);
    
    free(to_over_len);
    free(base_io);
    free(g_tsize);

    printf("[HOIMTO] Complete. Total high occupancy itemsets found: %zu\n", total_ho_count);
    dm_bench_record_results(total_ho_count, total_ho_footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "hoimto",
    .name = "HOIMTO Algorithm",
    .description = "High Occupancy Itemset Mining with Transaction Occupancy (Datta et al. 2021).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
