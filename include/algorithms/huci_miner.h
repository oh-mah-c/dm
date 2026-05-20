#ifndef DM_HUCI_MINER_H
#define DM_HUCI_MINER_H

#include "core/dm_algorithm.h"

typedef struct {
    double min_utility;
    double min_confidence;
} DM_HUCI_Miner_Params;

typedef struct {
    size_t high_utility_itemsets;
    size_t high_utility_closed_itemsets;
    size_t high_utility_generators;
    size_t hgb_rules;
    size_t max_depth;
    size_t utility_lists_constructed;
    size_t joins;
    size_t pruned_eucs;
    size_t pruned_subtree_utility;
} DM_HUCI_Miner_Stats;

int huci_mine_dataset(DM_Dataset *ds,
                      const DM_HUCI_Miner_Params *params,
                      DM_HUCI_Miner_Stats *stats);

#endif // DM_HUCI_MINER_H
