#ifndef DM_ALGORITHM_FLCHUSPM_H
#define DM_ALGORITHM_FLCHUSPM_H

#include "core/dm_algorithm.h"

/**
 * FLCHUSPM: frequent, low average cost and high average utility sequential pattern mining.
 * Reference: T. Truong, H. Duong, B. Le, P. Fournier-Viger, U. Yun, "Mining interesting
 * sequences with low average cost and high average utility", Applied Intelligence 52 (2022)
 * 7136-7157, doi:10.1007/s10489-021-02505-0.
 *
 * Implemented from the paper (no reference code is available): Definitions 1-6, Theorems 1-3,
 * the CUL structure of Section 5.1 and the pseudo-code of Figs. 3-4.
 *
 * Input file: one q-sequence per line, q-items "item:cost:utility", "-1" closes an itemset,
 * "-2" closes the sequence; lines starting with '#', '%' or '@' are ignored.
 */
typedef struct {
    const char *input_path;
    double mins;          /* minimum support (absolute count, or percent of |D| if percent) */
    double minu;          /* minimum average utility (absolute, or percent of maxUSeq) */
    double maxc;          /* maximum average cost (absolute, or percent of maxCSeq) */
    int percent;          /* thresholds given in the percentage form of Section 3 */
    int use_dps;          /* depth pruning (FUB, Flb), Fig. 4 line 1 */
    int use_wps;          /* width pruning (AMUB, supp) incl. RedS, TS and CMAPS */
} DM_FLCHUSPM_Params;

#endif // DM_ALGORITHM_FLCHUSPM_H
