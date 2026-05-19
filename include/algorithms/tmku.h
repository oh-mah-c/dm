#ifndef DM_TMKU_H
#define DM_TMKU_H

#include "core/dm_dataset.h"
#include <stddef.h>

typedef struct {
    size_t k;
    uint32_t *target_pattern;
    size_t target_len;
    double min_utility; // target utility threshold \xi
    double max_seconds;
} DM_TMKU_Params;

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
    size_t result_ram_bytes;
    size_t result_disk_est_bytes;
    int limited;
} DM_TMKU_Stats;

int tmku_mine_dataset(DM_Dataset *ds, const DM_TMKU_Params *params, DM_TMKU_Stats *stats);

#endif
