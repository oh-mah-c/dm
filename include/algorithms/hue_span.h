#ifndef DM_ALGORITHM_HUE_SPAN_H
#define DM_ALGORITHM_HUE_SPAN_H

#include "core/dm_algorithm.h"

/**
 * HUE-Span: high utility episode mining in a complex event sequence.
 * Reference: P. Fournier-Viger, P. Yang, J. C.-W. Lin, U. Yun, "HUE-Span: Fast High Utility
 * Episode Mining", Proc. ADMA 2019, Springer LNAI, pp. 169-184,
 * doi:10.1007/978-3-030-35231-8_12.
 *
 * This is a port of the authors' reference implementation (SPMF, package
 * ca.pfv.spmf.algorithms.episodes.huespan, classes AlgoHUESpan, ComplexSequence and
 * FiniteStateMachine), which resolves the points the paper leaves open.
 * The input is read directly from the file (SPMF format "events:total:utilities", one time
 * point per physical line), as the reference implementation does.
 */
typedef struct {
    const char *input_path;
    double min_utility_ratio;     /* minimum utility as a fraction of the sequence utility */
    int max_duration;             /* a minimal occurrence [ts, te] must satisfy te - ts + 1 <= maxDuration */
    int check_maximum_utility;    /* SPMF default: true  (false = "traditional utility") */
    int use_tighter_upper_bound;  /* SPMF default: true  (ERU; false = EWU) */
    int output_single_events;     /* SPMF parameter "Output single events" */
    int use_cooc_matrix;          /* SPMF default: true  (EEUCP) */
    int pruning_prefix;           /* SPMF default: true */
    int fixed_utility;            /* 0 = reference behaviour; 1 = corrected maximal utility and event order (not in SPMF) */
} DM_HUE_SPAN_Params;

#endif // DM_ALGORITHM_HUE_SPAN_H
