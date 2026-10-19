#ifndef DM_ALGORITHM_WALKY_G_H
#define DM_ALGORITHM_WALKY_G_H

#include "core/dm_algorithm.h"

/**
 * Walky-G: depth-first mining of minimal rare itemsets over frequent generators.
 * Reference: L. Szathmary, P. Valtchev, A. Napoli, R. Godin, "Efficient Vertical Mining of
 * Minimal Rare Itemsets", CLA 2012, pp. 269-280.
 *
 * min_support < 1 is relative (lambda = ceil(min_support * n)); otherwise absolute.
 * An itemset is rare if its support is below lambda.
 */
typedef struct {
    double min_support;
} DM_WALKY_G_Params;

#endif // DM_ALGORITHM_WALKY_G_H
