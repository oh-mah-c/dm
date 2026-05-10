#include "algorithms/r_miner.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t tid;
    double util;
} TIDPair;

typedef struct {
    uint32_t *items;
    size_t item_count;
    TIDPair *tids;
    size_t tid_count;
    double total_util;
    double residue;
} ResidueMap;

/* --- UTILS --- */

static ResidueMap* compute_join(ResidueMap *rm1, ResidueMap *rm2, double min_util) {
    ResidueMap *res = malloc(sizeof(ResidueMap));
    res->item_count = rm1->item_count + 1;
    res->items = malloc(sizeof(uint32_t) * res->item_count);
    memcpy(res->items, rm1->items, sizeof(uint32_t) * rm1->item_count);
    res->items[rm1->item_count] = rm2->items[0]; // Assuming rm2 is cardinality 1

    res->tids = malloc(sizeof(TIDPair) * (rm1->tid_count < rm2->tid_count ? rm1->tid_count : rm2->tid_count));
    res->tid_count = 0;
    res->total_util = 0;

    size_t i = 0, j = 0;
    while (i < rm1->tid_count && j < rm2->tid_count) {
        if (rm1->tids[i].tid == rm2->tids[j].tid) {
            res->tids[res->tid_count].tid = rm1->tids[i].tid;
            res->tids[res->tid_count].util = rm1->tids[i].util + rm2->tids[j].util;
            res->total_util += res->tids[res->tid_count].util;
            res->tid_count++;
            i++; j++;
        } else if (rm1->tids[i].tid < rm2->tids[j].tid) {
            i++;
        } else {
            j++;
        }
    }

    if (res->tid_count == 0) {
        free(res->tids); free(res->items); free(res);
        return NULL;
    }

    res->residue = min_util - res->total_util;
    return res;
}

/* --- RECURSIVE MINING --- */

static size_t hui_count = 0;

static void r_miner_recursive(ResidueMap *rm_i, ResidueMap **master, int master_idx, double min_util) {
    // Traverse master map in reverse order (Condition 2)
    for (int j = master_idx - 1; j >= 0; j--) {
        ResidueMap *rm_j = master[j];
        
        // Condition 1: Join only if total utility of rm_j >= residue of rm_i
        if (rm_j->total_util < rm_i->residue) {
            // Since master is sorted ascending, all previous elements will also fail Pruning Condition 1
            break; 
        }

        ResidueMap *join = compute_join(rm_i, rm_j, min_util);
        if (join) {
            if (join->total_util >= min_util) {
                hui_count++;
            }
            r_miner_recursive(join, master, j, min_util);
            
            // Cleanup
            free(join->tids); free(join->items); free(join);
        }
    }
}

/* --- MAIN RUN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_R_Miner_Params *p = (DM_R_Miner_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    uint32_t max_id = ds->max_id;

    printf("[R-Miner] Phase 1: Scanning database and building Residue Maps...\n");
    ResidueMap **rms = calloc(max_id + 1, sizeof(ResidueMap*));
    double *twu = calloc(max_id + 1, sizeof(double));

    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t id = data[i].items[j].id;
            twu[id] += data[i].total_utility;
            
            if (!rms[id]) {
                rms[id] = malloc(sizeof(ResidueMap));
                rms[id]->items = malloc(sizeof(uint32_t));
                rms[id]->items[0] = id;
                rms[id]->item_count = 1;
                rms[id]->tids = malloc(sizeof(TIDPair) * ds->count);
                rms[id]->tid_count = 0;
                rms[id]->total_util = 0;
            }
            rms[id]->tids[rms[id]->tid_count].tid = (uint32_t)i;
            rms[id]->tids[rms[id]->tid_count].util = data[i].items[j].utility;
            rms[id]->total_util += data[i].items[j].utility;
            rms[id]->tid_count++;
        }
    }

    // Build Master Map
    ResidueMap **master = malloc(sizeof(ResidueMap*) * (max_id + 1));
    size_t master_size = 0;
    for (uint32_t i = 0; i <= max_id; i++) {
        if (rms[i] && twu[i] >= min_util) {
            rms[i]->residue = min_util - rms[i]->total_util;
            master[master_size++] = rms[i];
        } else if (rms[i]) {
            free(rms[i]->tids); free(rms[i]->items); free(rms[i]);
        }
    }

    // Sort Master Map by TotalUtility Ascending
    for (size_t i = 0; i < master_size; i++) {
        for (size_t j = i + 1; j < master_size; j++) {
            if (master[i]->total_util > master[j]->total_util) {
                ResidueMap *tmp = master[i]; master[i] = master[j]; master[j] = tmp;
            }
        }
    }

    printf("[R-Miner] Phase 2: Recursive Mining...\n");
    hui_count = 0;
    for (size_t i = 0; i < master_size; i++) {
        if (master[i]->total_util >= min_util) {
            hui_count++;
        }
        r_miner_recursive(master[i], master, (int)i, min_util);
    }

    printf("[R-Miner] Found %zu High Utility Itemsets.\n", hui_count);

    // Cleanup
    for (size_t i = 0; i < master_size; i++) {
        free(master[i]->tids); free(master[i]->items); free(master[i]);
    }
    free(master); free(rms); free(twu);

    dm_bench_record_results(hui_count, 0);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "rminer",
    .name = "R-Miner",
    .description = "A residual utility-based algorithm for high-utility itemset mining using Residue Maps and Master Map pruning.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
