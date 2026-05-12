#include "algorithms/sum.h"
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

/* --- GLOBAL STATE --- */
static size_t total_hui_count = 0;

/* --- UTILS --- */

static ResidueMap* compute_join(ResidueMap *rm1, ResidueMap *rm2, double min_util) {
    ResidueMap *res = malloc(sizeof(ResidueMap));
    res->item_count = rm1->item_count + 1;
    res->items = malloc(sizeof(uint32_t) * res->item_count);
    memcpy(res->items, rm1->items, sizeof(uint32_t) * rm1->item_count);
    res->items[rm1->item_count] = rm2->items[0]; 

    // Worst case size: min of tid_counts
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

static void sum_recursive(ResidueMap *rm_i, ResidueMap **master, int master_idx, double min_util) {
    for (int j = master_idx - 1; j >= 0; j--) {
        ResidueMap *rm_j = master[j];
        
        // Pruning Condition 1: rm_j->total_util >= rm_i->residue
        if (rm_j->total_util < rm_i->residue) {
            break; 
        }

        ResidueMap *join = compute_join(rm_i, rm_j, min_util);
        if (join) {
            if (join->total_util >= min_util) {
                total_hui_count++;
            }
            sum_recursive(join, master, j, min_util);
            
            free(join->tids); free(join->items); free(join);
        }
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_SUM_Params *p = (DM_SUM_Params *)params;
    double mu = p ? p->min_utility : 1000.0;
    int window_size = p ? p->window_size : 150;
    int increment_size = (p && p->increment_size > 0) ? p->increment_size : (int)ds->count;
    bool dynamic_threshold = p ? p->dynamic_threshold : true;
    
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    uint32_t max_id = ds->max_id;

    int *rc = calloc(max_id + 1, sizeof(int));
    double *twu = calloc(max_id + 1, sizeof(double));
    ResidueMap **rms = calloc(max_id + 1, sizeof(ResidueMap*));

    size_t processed_transactions = 0;
    total_hui_count = 0;

    printf("[SUM] Starting Scented Utility Miner...\n");
    printf("[SUM] Initial MU: %.2f, Window Size: %d, Increment Size: %d\n", mu, window_size, increment_size);

    while (processed_transactions < ds->count) {
        size_t chunk_end = processed_transactions + increment_size;
        if (chunk_end > ds->count) chunk_end = ds->count;

        printf("[SUM] Processing batch: transactions %zu to %zu\n", processed_transactions, chunk_end);

        // Update RC and Residue Maps for items in this batch
        for (size_t i = processed_transactions; i < chunk_end; i++) {
            // Decrement RC for all items that exist
            for (uint32_t id = 0; id <= max_id; id++) {
                if (rc[id] > 0) rc[id]--;
            }

            // Set RC to window_size for items in current transaction
            for (size_t j = 0; j < data[i].count; j++) {
                uint32_t id = data[i].items[j].id;
                rc[id] = window_size;
                twu[id] += data[i].total_utility;

                if (!rms[id]) {
                    rms[id] = malloc(sizeof(ResidueMap));
                    rms[id]->items = malloc(sizeof(uint32_t));
                    rms[id]->items[0] = id;
                    rms[id]->item_count = 1;
                    rms[id]->tids = malloc(sizeof(TIDPair) * ds->count); // Max possible
                    rms[id]->tid_count = 0;
                    rms[id]->total_util = 0;
                }
                rms[id]->tids[rms[id]->tid_count].tid = (uint32_t)i;
                rms[id]->tids[rms[id]->tid_count].util = data[i].items[j].utility;
                rms[id]->total_util += data[i].items[j].utility;
                rms[id]->tid_count++;
            }
        }

        // Build Master Map from relevant items
        ResidueMap **master = malloc(sizeof(ResidueMap*) * (max_id + 1));
        size_t master_size = 0;
        double tu_min = -1;

        for (uint32_t i = 0; i <= max_id; i++) {
            if (rms[i] && rc[i] > 0 && twu[i] >= mu) {
                rms[i]->residue = mu - rms[i]->total_util;
                master[master_size++] = rms[i];
                if (tu_min < 0 || rms[i]->total_util < tu_min) {
                    tu_min = rms[i]->total_util;
                }
            }
        }

        if (master_size > 0) {
            // Sort Master Map by TotalUtility Ascending
            for (size_t i = 0; i < master_size; i++) {
                for (size_t j = i + 1; j < master_size; j++) {
                    if (master[i]->total_util > master[j]->total_util) {
                        ResidueMap *tmp = master[i]; master[i] = master[j]; master[j] = tmp;
                    }
                }
            }

            // Mine
            size_t batch_hui_count = 0;
            size_t start_hui = total_hui_count;
            for (size_t i = 0; i < master_size; i++) {
                if (master[i]->total_util >= mu) {
                    total_hui_count++;
                }
                sum_recursive(master[i], master, (int)i, mu);
            }
            batch_hui_count = total_hui_count - start_hui;
            printf("[SUM] Batch complete. Found %zu HUIs (Total: %zu)\n", batch_hui_count, total_hui_count);

            // Update Dynamic Threshold
            if (dynamic_threshold) {
                mu += tu_min;
                printf("[SUM] Raised MU to %.2f (added tu_min=%.2f)\n", mu, tu_min);
            }
        } else {
            printf("[SUM] Batch complete. No high utility items found.\n");
        }

        free(master);
        processed_transactions = chunk_end;
    }

    printf("[SUM] Total High Utility Itemsets found: %zu\n", total_hui_count);

    // Cleanup
    for (uint32_t i = 0; i <= max_id; i++) {
        if (rms[i]) {
            free(rms[i]->tids); free(rms[i]->items); free(rms[i]);
        }
    }
    free(rms); free(rc); free(twu);

    dm_bench_record_results(total_hui_count, 0);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "sum",
    .name = "Scented Utility Miner",
    .description = "An incremental HUIM algorithm using reinduction strategy and dynamic threshold setting.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
