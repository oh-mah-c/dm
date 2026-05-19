#ifndef DM_TIPN_HOUI_H
#define DM_TIPN_HOUI_H

#include <stddef.h>

typedef struct {
    size_t k;
    size_t intervals;
    size_t max_depth;
    size_t max_transactions;
    size_t max_items;
    size_t max_candidates;
    double max_seconds;
} TIPNHouiParams;

typedef struct {
    size_t transactions;
    size_t intervals;
    size_t distinct_items;
    size_t positive_items;
    size_t negative_items;
    size_t k;
    double threshold;
    size_t output_count;
    double best_relative_utility;
    double avg_relative_utility;
    double avg_itemset_length;
    size_t candidates;
    size_t joins;
    size_t pruned_twugc;
    size_t pruned_rlc;
    size_t pruned_tio;
    size_t threshold_raises;
    double rpru_size1_threshold;
    double rru_size2_threshold;
    size_t result_ram_bytes;
    size_t result_disk_est_bytes;
    int limited;
} TIPNHouiStats;

TIPNHouiParams tipn_houi_default_params(void);
int tipn_houi_mine_file(const char *path, const TIPNHouiParams *params, TIPNHouiStats *stats);

#endif
