/*
 * CMA-ES: Covariance Matrix Adaptation Evolution Strategy.
 *
 * Implements the (mu/mu_W, lambda)-CMA-ES as described in:
 *   N. Hansen, "The CMA Evolution Strategy: A Tutorial,"
 *   arXiv:1604.00772v2, 2023.
 *
 * Algorithm summary: Appendix A (Figure 6) + Table 1 (default parameters).
 * Sampling: Eq. (38)-(40).  Mean update: Eq. (42).
 * Step-size control (CSA): Eq. (43)-(44).  Covariance update: Eq. (45)-(47).
 */

#ifndef DM_CMAES_H
#define DM_CMAES_H

#include "core/dm_common.h"
#include "core/dm_algorithm.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Public types
 * ------------------------------------------------------------------------- */

/* Objective function: minimise f(x) where x has dimension n. */
typedef double (*CMAES_ObjectiveFn)(const double *x, int n, void *userdata);

/*
 * CMAES_Params -- all fields optional: zero-init then set what you need.
 * Defaults are applied by cmaes_init() following Table 1 of the tutorial.
 */
typedef struct {
    int    n;           /* Search space dimension (required, > 0). */
    int    lambda;      /* Offspring per generation; 0 -> default = 4+floor(3*ln(n)). */
    double sigma0;      /* Initial step-size; 0.0 -> 0.3*(x_hi - x_lo) or 0.3. */
    double *x0;         /* Initial mean, length n; NULL -> zeros. */
    int    max_evals;   /* Budget (# f calls); 0 -> 1e4 * n^2. */
    double ftol;        /* Stop if best f-value < ftol; default -HUGE_VAL. */
    double xtol;        /* Stop if sigma * max(diag(D)) < xtol * sigma0; default 1e-12. */

    /* Objective function and optional user data pointer. */
    CMAES_ObjectiveFn objective;
    void             *userdata;
} CMAES_Params;

/* State returned after optimisation. */
typedef struct {
    double  *xbest;     /* Best point found, length n (caller must free). */
    double   fbest;     /* f(xbest). */
    int      evals;     /* Total function evaluations used. */
    int      gens;      /* Total generations completed. */
    int      stop_flag; /* Reason for termination (CMAES_STOP_* constants below). */
} CMAES_Result;

#define CMAES_STOP_BUDGET     1  /* max_evals reached */
#define CMAES_STOP_FTOL       2  /* fbest < ftol */
#define CMAES_STOP_XTOL       3  /* sigma * max_D < xtol * sigma0 */
#define CMAES_STOP_CONDCOV    4  /* cond(C) > 1e14 */
#define CMAES_STOP_STAGNATION 5  /* no progress over history window */
#define CMAES_STOP_NOEFFECT   6  /* step too small to change mean */

/* -------------------------------------------------------------------------
 * Public API
 * ------------------------------------------------------------------------- */

/*
 * Run CMA-ES.  Returns DM_SUCCESS on normal termination (stop criterion met)
 * or an error code if parameters are invalid / allocation fails.
 * The caller must free result->xbest with free().
 */
DM_Status cmaes_run(const CMAES_Params *params, CMAES_Result *result);

/* Convenience: run and print a short summary to stdout. */
void cmaes_print_result(const CMAES_Result *result, int n);

/* DM framework plugin descriptor (registered automatically at startup). */
extern DM_Algorithm cmaes_algorithm;

#ifdef __cplusplus
}
#endif

#endif /* DM_CMAES_H */
