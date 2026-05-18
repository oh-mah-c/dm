#ifndef DM_CHUO_MINER_H
#define DM_CHUO_MINER_H

#include "core/dm_dataset.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    double min_utility;
    uint32_t min_support;
    double min_occupancy;
    size_t max_patterns;
    size_t max_depth;
    double max_seconds;
    int verify_reconstruction;
} DM_CHUO_Params;

typedef struct {
    size_t transactions;
    uint32_t max_item_id;
    uint32_t min_support;
    double min_utility;
    double min_occupancy;
    size_t surviving_items;
    size_t singleton_occurrences;
    size_t visited_nodes;
    size_t generated_children;
    size_t joined_entries;
    size_t pruned_twu;
    size_t pruned_support;
    size_t pruned_ruu;
    size_t pruned_oub;
    size_t pruned_backward;
    size_t closure_jumps;
    size_t closure_hash_hits;
    size_t emitted_chuois;
    size_t total_output_items;
    size_t signatures_built;
    size_t signature_value_records;
    size_t reconstruction_checks;
    size_t reconstruction_failures;
    size_t max_depth_seen;
    double avg_support;
    double avg_utility;
    double avg_occupancy;
    double best_utility;
    double best_occupancy;
    size_t result_ram_bytes;
    size_t result_disk_est_bytes;
    int limited;
} DM_CHUO_Stats;

int chuo_mine_dataset(DM_Dataset *ds, const DM_CHUO_Params *params, DM_CHUO_Stats *stats);

#endif
