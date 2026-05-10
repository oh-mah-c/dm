#ifndef DM_DFI_GROWTH_H
#define DM_DFI_GROWTH_H

#include "core/dm_algorithm.h"

/**
 * @brief Parameters for the DFI-Growth Algorithm
 * 
 * DFI-Growth: Deriving Frequent Itemsets based on Pattern Growth.
 * Paper: Huang et al. (2019) "An Efficient Algorithm for Deriving Frequent Itemsets from Lossless Condensed Representation"
 */
typedef struct {
    double min_support;
    const char *fci_miner; // Optional: specify which miner to use to get FCIs first. Default "charm" or "close".
} DM_DFI_GROWTH_Params;

#endif // DM_DFI_GROWTH_H
