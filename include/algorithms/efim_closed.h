#ifndef DM_EFIM_CLOSED_H
#define DM_EFIM_CLOSED_H

#include "core/dm_algorithm.h"

/**
 * @file efim_closed.h
 * @brief EFIM-Closed: Fast and Memory Efficient Discovery of Closed High-Utility Itemsets
 *
 * Reference:
 *   P. Fournier-Viger, S. Zida, J. C.-W. Lin, C.-W. Wu, and V. S. Tseng,
 *   "EFIM-Closed: Fast and Memory Efficient Discovery of Closed High-Utility Itemsets,"
 *   in Proc. 12th Int. Conf. on Machine Learning and Data Mining in Pattern Recognition (MLDM 2016),
 *   Springer LNAI 9729, 2016, pp. 199-213.
 *   DOI: 10.1007/978-3-319-41920-6_16
 */

typedef struct {
    double min_utility; /**< Minimum utility threshold (absolute) */
} DM_EFIM_Closed_Params;

#endif // DM_EFIM_CLOSED_H

