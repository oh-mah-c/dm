#ifndef DM_TKU_MINER_H
#define DM_TKU_MINER_H

#include "core/dm_dataset.h"

#include <stddef.h>

typedef struct {
    size_t k;
    size_t max_depth;
    double max_seconds;
} DM_TKU_Params;

typedef struct {
    size_t transactions;
    size_t distinct_items;
    size_t k;
    double final_threshold;
    size_t output_count;
    size_t total_output_items;
    double best_utility;
    double avg_utility;
    double avg_length;
    size_t candidates;
    size_t visited_nodes;
    size_t joins;
    size_t joined_entries;
    size_t pruned_twu;
    size_t pruned_subtree_utility;
    size_t threshold_raises;
    double pe_threshold;
    double singleton_threshold;
    size_t phase2_checked;
    size_t result_ram_bytes;
    size_t result_disk_est_bytes;
    int limited;
} DM_TKU_Stats;

int tku_mine_dataset(DM_Dataset *ds, const DM_TKU_Params *params, DM_TKU_Stats *stats);

#endif
