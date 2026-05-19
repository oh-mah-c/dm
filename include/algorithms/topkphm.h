#ifndef DM_TOPKPHM_H
#define DM_TOPKPHM_H

#include <stddef.h>

typedef struct {
    size_t k;
    size_t max_period;
    double max_avg_period;
    size_t max_depth;
    size_t max_candidates;
    double max_seconds;
} DM_TOPKPHM_Params;

typedef struct {
    size_t transactions;
    size_t distinct_items;
    size_t kept_items;
    size_t k;
    size_t support_threshold;
    double max_avg_period;
    size_t max_period;
    double final_threshold;
    size_t threshold_raises;
    size_t output_count;
    size_t total_output_items;
    double best_utility;
    double avg_utility;
    double avg_output_length;
    size_t candidates;
    size_t visited_nodes;
    size_t joins;
    size_t joined_entries;
    size_t euscs_pairs;
    size_t pruned_support;
    size_t pruned_periodicity;
    size_t pruned_twu;
    size_t pruned_subtree_utility;
    size_t result_ram_bytes;
    size_t result_disk_est_bytes;
    int limited;
} DM_TOPKPHM_Stats;

int topkphm_mine_file(const char *path, const DM_TOPKPHM_Params *params, DM_TOPKPHM_Stats *stats);

#endif
