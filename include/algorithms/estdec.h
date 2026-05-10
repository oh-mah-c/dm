#ifndef DM_ALGORITHM_ESTDEC_H
#define DM_ALGORITHM_ESTDEC_H

#include "core/dm_algorithm.h"

/**
 * estDec Algorithm for finding recent frequent itemsets over data streams.
 * Reference: J. H. Chang and W. S. Lee, "Finding Recent Frequent Itemsets 
 * Adaptively over Online Data Streams", KDD 2003.
 */

typedef struct {
    double min_support;  // Smin
    double ins_threshold; // Sins
    double prn_threshold; // Sprn
    double decay_base;    // b
    double decay_life;    // h
} DM_ESTDEC_Params;

#endif // DM_ALGORITHM_ESTDEC_H
