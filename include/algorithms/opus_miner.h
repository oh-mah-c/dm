#ifndef DM_OPUS_MINER_H
#define DM_OPUS_MINER_H

#include "core/dm_algorithm.h"

/**
 * @brief Interest measures for OPUS Miner
 */
typedef enum {
    DM_OPUS_MEASURE_LEVERAGE,
    DM_OPUS_MEASURE_LIFT
} DM_OPUS_Measure;

/**
 * @brief Parameters for the OPUS Miner Algorithm
 */
typedef struct {
    int k;                      /**< Number of top-k itemsets to find */
    DM_OPUS_Measure measure;     /**< Interest measure to use */
    double alpha;               /**< Statistical significance level (default 0.05) */
    bool check_indep;           /**< Whether to perform independent productivity check (default true) */
} DM_OPUS_MINER_Params;

#endif // DM_OPUS_MINER_H
