#ifndef DM_ALGORITHM_FLCHUS_X_H
#define DM_ALGORITHM_FLCHUS_X_H

#include "core/dm_algorithm.h"

/**
 * FLCHUS-X (proposed): sound and complete miner for FLCHUSM (frequent, low average cost and
 * high average utility sequences, Truong et al., Applied Intelligence 2022).
 * Same input format and thresholds as flchuspm (see flchuspm.h). Proofs of the bounds are in
 * docs/core/FLCHUS/notes.md.
 */
typedef struct {
    const char *input_path;
    double mins, minu, maxc;
    int percent;
    int cost_width;       /* MLB in RedS, CMAPs and the candidate sets I/S */
    int reds_fixpoint;    /* repeat RedS until no item is removed */
    int fast_select;      /* O(n) selection of the m extreme values and early abandon of joins */
    int desc_bound;       /* descendant cost bound MDB for depth pruning (most expensive occurrence per end) */
} DM_FLCHUS_X_Params;

#endif // DM_ALGORITHM_FLCHUS_X_H
