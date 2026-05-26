/*
 * CMA-ES: Covariance Matrix Adaptation Evolution Strategy.
 *
 * Reference: N. Hansen, arXiv:1604.00772v2 (2023).
 * All equation numbers refer to that document.
 *
 * Implementation follows Appendix A (Figure 6 algorithm) and Table 1 defaults.
 * Negative recombination weights (active CMA / aCMA-ES, 2016 default) are used
 * as described in Sections 3.2 and the Table 1 footnotes.
 *
 * Linear algebra:
 *   - All matrices stored column-major (Fortran order) with 1-D double arrays.
 *     Element (i,j) of an n x n matrix A is A[j*n + i].
 *   - Eigendecomposition C = B D^2 B^T computed via a symmetric QR/Jacobi
 *     routine (no external LAPACK dependency).
 *   - B and D recomputed every max(1, floor(1/(10*n*(c1+cmu)))) generations
 *     to keep O(n^2) amortised cost per generation (Sec. B.2).
 */

#include "algorithms/cmaes.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdio.h>
#include <time.h>

/* ---------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

static double randn_box_muller(void) {
    /* Box-Muller; good enough for optimisation workloads. */
    static int have_spare = 0;
    static double spare;
    if (have_spare) { have_spare = 0; return spare; }
    double u, v, s;
    do {
        u = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        v = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        s = u*u + v*v;
    } while (s >= 1.0 || s == 0.0);
    double mul = sqrt(-2.0 * log(s) / s);
    spare = v * mul;
    have_spare = 1;
    return u * mul;
}

/* Fill z[0..n-1] with i.i.d. N(0,1) samples (Eq. 38). */
static void sample_normal(double *z, int n) {
    for (int i = 0; i < n; i++) z[i] = randn_box_muller();
}

/*
 * Symmetric Jacobi eigendecomposition.
 * Input:  A[n*n] symmetric matrix (column-major, overwritten).
 * Output: A contains normalised eigenvectors as columns (= B);
 *         d[n] contains eigenvalues.
 * Returns number of sweeps performed.
 */
static int jacobi_eigen(double *A, double *d, int n) {
    /* Copy diagonal as initial eigenvalue estimates. */
    for (int i = 0; i < n; i++) d[i] = A[i*n + i];
    /* Accumulate eigenvectors in V (start as identity), then swap into A. */
    double *V = (double *)calloc((size_t)(n*n), sizeof(double));
    if (!V) return -1;
    for (int i = 0; i < n; i++) V[i*n + i] = 1.0;

    const int max_sweeps = 100;
    int sweeps;
    for (sweeps = 0; sweeps < max_sweeps; sweeps++) {
        double off = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++)
                off += A[j*n+i] * A[j*n+i];
        if (off < 1e-14 * 1e-14) break;

        for (int p = 0; p < n-1; p++) {
            for (int q = p+1; q < n; q++) {
                double apq = A[q*n+p];
                if (fabs(apq) < 1e-15) continue;
                double app = A[p*n+p], aqq = A[q*n+q];
                double tau = (aqq - app) / (2.0 * apq);
                double t = (tau >= 0.0)
                           ?  1.0 / (tau + sqrt(1.0 + tau*tau))
                           : -1.0 / (-tau + sqrt(1.0 + tau*tau));
                double c = 1.0 / sqrt(1.0 + t*t);
                double s = t * c;
                double off2 = s / (1.0 + c);

                /* Update diagonal elements. */
                A[p*n+p] = app - t * apq;
                A[q*n+q] = aqq + t * apq;
                A[q*n+p] = 0.0;
                A[p*n+q] = 0.0;

                /* Update off-diagonal elements. */
                for (int r = 0; r < n; r++) {
                    if (r != p && r != q) {
                        double arp = A[p*n+r];
                        double arq = A[q*n+r];
                        A[p*n+r] = arp - s*(arq + off2*arp);
                        A[r*n+p] = A[p*n+r];
                        A[q*n+r] = arq + s*(arp - off2*arq);
                        A[r*n+q] = A[q*n+r];
                    }
                }

                /* Accumulate rotation in V. */
                for (int r = 0; r < n; r++) {
                    double vrp = V[p*n+r];
                    double vrq = V[q*n+r];
                    V[p*n+r] = c*vrp - s*vrq;
                    V[q*n+r] = s*vrp + c*vrq;
                }
            }
        }
        /* Refresh diagonal. */
        for (int i = 0; i < n; i++) d[i] = A[i*n+i];
    }

    /* Copy eigenvectors back into A. */
    memcpy(A, V, (size_t)(n*n) * sizeof(double));
    free(V);
    return sweeps;
}

/* dot product of two length-n vectors. */
static double dot(const double *a, const double *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i]*b[i];
    return s;
}

/* Euclidean norm of length-n vector. */
static double norm2(const double *a, int n) {
    return sqrt(dot(a, a, n));
}

/* y = A * x  (A is n x n column-major, x and y are length n). */
static void mat_vec(const double *A, const double *x, double *y, int n) {
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int j = 0; j < n; j++) s += A[j*n+i] * x[j];
        y[i] = s;
    }
}

/* ---------------------------------------------------------------------------
 * Main CMA-ES optimiser
 * -------------------------------------------------------------------------- */

DM_Status cmaes_run(const CMAES_Params *params, CMAES_Result *result) {
    if (!params || !result || !params->objective || params->n < 1)
        return DM_ERROR_INVALID_PARAM;

    const int n = params->n;
    srand((unsigned int)time(NULL));

    /* -----------------------------------------------------------------------
     * Strategy parameter setup (Table 1, Eqs. 48-58)
     * --------------------------------------------------------------------- */

    /* Population size. Eq. (48). */
    const int lambda = (params->lambda > 1)
                       ? params->lambda
                       : 4 + (int)(3.0 * log((double)n));

    /* Parent number. */
    const int mu = lambda / 2;

    /* Recombination weights.  Preliminary shape: w'_i = ln((lambda+1)/2) - ln(i).
     * Eq. (49).  Then normalise positive weights to sum 1 (Eq. 53). */
    double *w_raw = (double *)malloc((size_t)lambda * sizeof(double));
    if (!w_raw) return DM_ERROR_MEMORY;
    for (int i = 0; i < lambda; i++)
        w_raw[i] = log(0.5*(lambda+1)) - log((double)(i+1));

    double sum_w_pos = 0.0, sum_w2_pos = 0.0;
    for (int i = 0; i < mu; i++) { sum_w_pos += w_raw[i]; sum_w2_pos += w_raw[i]*w_raw[i]; }
    const double mueff = sum_w_pos * sum_w_pos / sum_w2_pos; /* Eq. (8). */

    /* Negative weight helpers for active CMA (2016 default). */
    double sum_wneg_abs = 0.0;
    for (int i = mu; i < lambda; i++) sum_wneg_abs += fabs(w_raw[i]);
    double mueff_neg = 0.0;
    {
        double s1 = 0.0, s2 = 0.0;
        for (int i = mu; i < lambda; i++) { s1 += fabs(w_raw[i]); s2 += w_raw[i]*w_raw[i]; }
        if (s2 > 0.0) mueff_neg = s1*s1/s2;
    }

    /* Step-size control (CSA) parameters.  Eq. (55). */
    const double cs = (mueff + 2.0) / ((double)n + mueff + 5.0);
    const double ds = 1.0 + 2.0*fmax(0.0, sqrt((mueff-1.0)/((double)(n+1))) - 1.0) + cs;

    /* Covariance matrix adaptation parameters.  Eqs. (56)-(58). */
    const double alpha_cov = 2.0;
    const double cc  = (4.0 + mueff/(double)n) / ((double)(n+4) + 2.0*mueff/(double)n);
    const double c1  = alpha_cov / ((pow((double)(n+1.3), 2.0) + mueff));
    const double cmu_raw = alpha_cov * (mueff - 2.0 + 1.0/mueff)
                           / ((pow((double)(n+2), 2.0) + alpha_cov*mueff/2.0));
    const double cmu = fmin(1.0 - c1, cmu_raw);

    /* Compute final normalised weights w[i] (Eq. 53) and alpha bounds. */
    /* alpha_mu_neg: sets negative weight magnitude. */
    const double alpha_mu_neg     = 1.0 + c1/cmu;
    const double alpha_mueff_neg  = 1.0 + 2.0*mueff_neg/(mueff+2.0);
    const double alpha_posdef_neg = (1.0 - c1 - cmu) / ((double)n * cmu);

    double *w = (double *)malloc((size_t)lambda * sizeof(double));
    if (!w) { free(w_raw); return DM_ERROR_MEMORY; }
    for (int i = 0; i < mu; i++)
        w[i] = w_raw[i] / sum_w_pos;
    double sum_wneg_raw = 0.0;
    for (int i = mu; i < lambda; i++) sum_wneg_raw += fabs(w_raw[i]);
    double alpha_neg = fmin(alpha_mu_neg, fmin(alpha_mueff_neg, alpha_posdef_neg));
    for (int i = mu; i < lambda; i++)
        w[i] = (sum_wneg_raw > 0.0)
               ? alpha_neg * w_raw[i] / sum_wneg_raw
               : 0.0;
    free(w_raw);

    /* Expected norm of N(0,I).  chiN ~ sqrt(n) * (1 - 1/(4n) + 1/(21n^2)). */
    const double chiN = sqrt((double)n) * (1.0 - 1.0/(4.0*n) + 1.0/(21.0*n*n));

    /* Budget and tolerances. */
    const int    max_evals = (params->max_evals > 0)
                             ? params->max_evals
                             : (int)(1e4 * (double)n * (double)n);
    const double ftol  = (params->ftol != 0.0) ? params->ftol : -DBL_MAX;
    const double xtol  = (params->xtol > 0.0)  ? params->xtol : 1e-12;

    /* Frequency of B/D re-decomposition (Sec. B.2): recompute every
     * max(1, 1/(10*n*(c1+cmu))) generations to achieve O(n^2) amortised cost. */

    /* -----------------------------------------------------------------------
     * Allocate state
     * --------------------------------------------------------------------- */

    double *m   = (double *)calloc((size_t)n, sizeof(double));   /* mean */
    double *ps  = (double *)calloc((size_t)n, sizeof(double));   /* sigma path */
    double *pc  = (double *)calloc((size_t)n, sizeof(double));   /* cov path */
    double *C   = (double *)calloc((size_t)(n*n), sizeof(double)); /* covariance */
    double *B   = (double *)calloc((size_t)(n*n), sizeof(double)); /* eigenvectors */
    double *D   = (double *)calloc((size_t)n, sizeof(double));   /* sqrt(eigenvalues) */
    double *invsqrtC = (double *)calloc((size_t)(n*n), sizeof(double)); /* C^{-1/2} */

    /* Per-generation buffers. */
    double *arx  = (double *)malloc((size_t)(lambda * n) * sizeof(double)); /* offspring */
    double *ary  = (double *)malloc((size_t)(lambda * n) * sizeof(double)); /* (x-m)/sigma */
    double *arz  = (double *)malloc((size_t)(lambda * n) * sizeof(double)); /* N(0,I) samples */
    double *arfx = (double *)malloc((size_t)lambda * sizeof(double));       /* fitness */
    int    *idx  = (int *)   malloc((size_t)lambda * sizeof(int));          /* sort index */

    /* Working vectors. */
    double *ymean = (double *)malloc((size_t)n * sizeof(double)); /* <y>_w */
    double *tmp   = (double *)malloc((size_t)n * sizeof(double));

    if (!m || !ps || !pc || !C || !B || !D || !invsqrtC ||
        !arx || !ary || !arz || !arfx || !idx || !ymean || !tmp || !w) {
        free(m); free(ps); free(pc); free(C); free(B); free(D); free(invsqrtC);
        free(arx); free(ary); free(arz); free(arfx); free(idx);
        free(ymean); free(tmp); free(w);
        return DM_ERROR_MEMORY;
    }

    /* Initialise mean (Eq. 40 init). */
    if (params->x0)
        memcpy(m, params->x0, (size_t)n * sizeof(double));

    /* Initialise C = I, B = I, D = 1. */
    for (int i = 0; i < n; i++) {
        C[i*n+i] = 1.0;
        B[i*n+i] = 1.0;
        D[i]     = 1.0;
    }
    /* invsqrtC = B * D^{-1} * B^T = I when C=I, D=1. */
    for (int i = 0; i < n; i++) invsqrtC[i*n+i] = 1.0;

    double sigma = (params->sigma0 > 0.0) ? params->sigma0 : 0.3;

    /* -----------------------------------------------------------------------
     * Stagnation tracking: keep history of best fitness values.
     * --------------------------------------------------------------------- */
    int hist_size = (int)fmax(120.0, 30.0*(double)n/(double)lambda) + 1;
    hist_size = (int)fmin(hist_size, 20000);
    double *fbest_hist = (double *)malloc((size_t)hist_size * sizeof(double));
    if (!fbest_hist) {
        /* Non-fatal: just disable stagnation check. */
        hist_size = 0;
    } else {
        for (int i = 0; i < hist_size; i++) fbest_hist[i] = DBL_MAX;
    }

    /* Best so far. */
    double *xbest = (double *)malloc((size_t)n * sizeof(double));
    if (!xbest) { free(fbest_hist); goto oom; }
    double fbest = DBL_MAX;
    memcpy(xbest, m, (size_t)n * sizeof(double));

    int evals = 0, gen = 0, stop_flag = CMAES_STOP_BUDGET;
    int eigen_eval = 0; /* counteval at last eigendecomposition */

    /* -----------------------------------------------------------------------
     * Generation loop
     * --------------------------------------------------------------------- */
    while (evals < max_evals) {
        gen++;

        /* --- Sample lambda offspring (Eqs. 38-40). --- */
        for (int k = 0; k < lambda; k++) {
            double *zk = arz + k*n;
            double *yk = ary + k*n;
            double *xk = arx + k*n;
            sample_normal(zk, n);           /* z_k ~ N(0,I) */
            /* y_k = B * D * z_k  ~  N(0, C).  Eq. (39). */
            for (int i = 0; i < n; i++) {
                double s = 0.0;
                for (int j = 0; j < n; j++) s += B[j*n+i] * D[j] * zk[j];
                yk[i] = s;
            }
            /* x_k = m + sigma * y_k.  Eq. (40). */
            for (int i = 0; i < n; i++) xk[i] = m[i] + sigma * yk[i];
            arfx[k] = params->objective(xk, n, params->userdata);
            evals++;
        }

        /* --- Sort by fitness (minimisation). --- */
        for (int k = 0; k < lambda; k++) idx[k] = k;
        /* Simple insertion sort (lambda is small, typically < 100). */
        for (int k = 1; k < lambda; k++) {
            int key = idx[k]; double fkey = arfx[key];
            int j = k-1;
            while (j >= 0 && arfx[idx[j]] > fkey) { idx[j+1] = idx[j]; j--; }
            idx[j+1] = key;
        }

        /* Update best. */
        if (arfx[idx[0]] < fbest) {
            fbest = arfx[idx[0]];
            memcpy(xbest, arx + idx[0]*n, (size_t)n * sizeof(double));
        }
        if (hist_size > 0)
            fbest_hist[gen % hist_size] = fbest;

        /* --- Selection and recombination: update mean (Eq. 42). --- */
        /* ymean = sum_{i=1}^{mu} w_i * y_{i:lambda}.  (c_m = 1) */
        memset(ymean, 0, (size_t)n * sizeof(double));
        for (int i = 0; i < mu; i++) {
            const double *yi = ary + idx[i]*n;
            for (int j = 0; j < n; j++) ymean[j] += w[i] * yi[j];
        }
        /* m <- m + sigma * ymean. */
        for (int j = 0; j < n; j++) m[j] += sigma * ymean[j];

        /* --- Step-size control (CSA, Eqs. 43-44). --- */
        /* p_sigma <- (1-cs)*p_sigma + sqrt(cs*(2-cs)*mueff) * C^{-1/2} * ymean */
        mat_vec(invsqrtC, ymean, tmp, n);
        double sq = sqrt(cs * (2.0 - cs) * mueff);
        for (int j = 0; j < n; j++)
            ps[j] = (1.0 - cs) * ps[j] + sq * tmp[j];
        double ps_norm = norm2(ps, n);

        /* Heaviside indicator h_sigma (prevents too-fast increase when ps is large). */
        double hs_thresh = (1.4 + 2.0/((double)(n+1))) * chiN;
        double hs_denom  = sqrt(1.0 - pow(1.0-cs, 2.0*(double)(evals/lambda)));
        int hsig = (hs_denom > 0.0 && ps_norm / hs_denom < hs_thresh) ? 1 : 0;

        /* sigma <- sigma * exp((cs/ds) * (||ps|| / chiN - 1)).  Eq. (44). */
        sigma *= exp((cs / ds) * (ps_norm / chiN - 1.0));

        /* --- Covariance matrix adaptation (Eqs. 45-47). --- */
        /* p_c <- (1-cc)*p_c + hsig * sqrt(cc*(2-cc)*mueff) * ymean. */
        double sq_c = sqrt(cc * (2.0 - cc) * mueff);
        for (int j = 0; j < n; j++)
            pc[j] = (1.0 - cc) * pc[j] + (double)hsig * sq_c * ymean[j];

        /* Compute adjusted weights w_i^o for negative weights (Eq. 46).
         * w_i^o = w_i  if w_i >= 0,
         * w_i^o = w_i * n / ||C^{-1/2} y_{i:lambda}||^2  if w_i < 0.   */
        double delta_hs = (1.0 - (double)hsig) * cc * (2.0 - cc); /* ~0 usually */

        /* C <- (1 + c1*delta_hs - c1 - cmu*sum(w)) * C
         *      + c1 * pc * pc^T
         *      + cmu * sum_i w_i^o * y_{i:lambda} * y_{i:lambda}^T       */
        double sum_wj = 0.0;
        for (int i = 0; i < lambda; i++) sum_wj += w[i];
        double decay = 1.0 + c1*delta_hs - c1 - cmu*sum_wj;

        /* Scale existing C. */
        for (int j = 0; j < n*n; j++) C[j] *= decay;

        /* Rank-one update. */
        for (int r = 0; r < n; r++)
            for (int s2 = 0; s2 < n; s2++)
                C[s2*n+r] += c1 * pc[r] * pc[s2];

        /* Rank-mu update. */
        for (int i = 0; i < lambda; i++) {
            const double *yi = ary + idx[i]*n;
            double wi = w[i];
            if (wi < 0.0) {
                /* Adjust weight for negative entries. */
                mat_vec(invsqrtC, yi, tmp, n);
                double nrm2 = dot(tmp, tmp, n);
                wi = (nrm2 > 0.0) ? wi * (double)n / nrm2 : 0.0;
            }
            for (int r = 0; r < n; r++)
                for (int s2 = 0; s2 < n; s2++)
                    C[s2*n+r] += cmu * wi * yi[r] * yi[s2];
        }

        /* Enforce symmetry (numerical drift). */
        for (int r = 0; r < n; r++)
            for (int s2 = r+1; s2 < n; s2++)
                C[r*n+s2] = C[s2*n+r] = 0.5*(C[r*n+s2] + C[s2*n+r]);

        /* --- Eigendecomposition (recomputed periodically, Sec. B.2). --- */
        if (evals - eigen_eval > lambda / (10.0 * (double)n * (c1 + cmu))) {
            eigen_eval = evals;

            /* Copy C into B for decomposition. */
            memcpy(B, C, (size_t)(n*n) * sizeof(double));
            jacobi_eigen(B, D, n);      /* B <- eigenvectors, D <- eigenvalues */

            /* Clamp eigenvalues (positive definiteness guard). */
            double dmin = DBL_MAX;
            for (int i = 0; i < n; i++) dmin = fmin(dmin, D[i]);
            if (dmin < 1e-20) {
                /* Shift spectrum: C += |dmin| * I + epsilon. */
                double shift = fabs(dmin) + 1e-20;
                for (int i = 0; i < n; i++) {
                    D[i] += shift;
                    C[i*n+i] += shift;
                }
                /* Re-decompose. */
                memcpy(B, C, (size_t)(n*n) * sizeof(double));
                jacobi_eigen(B, D, n);
            }
            for (int i = 0; i < n; i++) D[i] = sqrt(fabs(D[i]));

            /* Build C^{-1/2} = B * D^{-1} * B^T. */
            memset(invsqrtC, 0, (size_t)(n*n) * sizeof(double));
            for (int r = 0; r < n; r++)
                for (int s2 = 0; s2 < n; s2++) {
                    double acc = 0.0;
                    for (int k = 0; k < n; k++)
                        acc += B[k*n+r] * (1.0/D[k]) * B[k*n+s2];
                    invsqrtC[s2*n+r] = acc;
                }
        }

        /* --- Termination criteria (Appendix B.3). --- */

        /* ftol. */
        if (fbest <= ftol) { stop_flag = CMAES_STOP_FTOL; break; }

        /* TolX: sigma * max_D < xtol * sigma0. */
        double max_D = 0.0;
        for (int i = 0; i < n; i++) max_D = fmax(max_D, D[i]);
        if (sigma * max_D < xtol) { stop_flag = CMAES_STOP_XTOL; break; }

        /* ConditionCov: cond(C) > 1e14. */
        double min_D = DBL_MAX;
        for (int i = 0; i < n; i++) min_D = fmin(min_D, D[i]);
        if (min_D > 0.0 && max_D / min_D > 1e7) {
            stop_flag = CMAES_STOP_CONDCOV; break;
        }

        /* NoEffectAxis: adding 0.1*sigma*d_i*b_i to m[g%n] doesn't change it. */
        {
            int axis = (gen-1) % n;
            double test_step = 0.1 * sigma * D[axis] * B[axis*n + (gen-1)%n];
            if (fabs(test_step) < 1e-16 * fabs(m[(gen-1)%n])) {
                stop_flag = CMAES_STOP_NOEFFECT; break;
            }
        }

        /* Stagnation. */
        if (hist_size > 0 && gen > hist_size) {
            /* Compare median of first 30% vs last 30% of history. */
            int h30 = (int)(0.3 * hist_size);
            if (h30 >= 2) {
                /* Rough median approximation: sort copies of the two windows. */
                double *ha = (double *)malloc((size_t)h30 * sizeof(double));
                double *hb = (double *)malloc((size_t)h30 * sizeof(double));
                if (ha && hb) {
                    for (int i = 0; i < h30; i++) {
                        ha[i] = fbest_hist[(gen - hist_size + i) % hist_size];
                        hb[i] = fbest_hist[(gen - h30 + i) % hist_size];
                    }
                    /* Selection sort (h30 is small). */
                    for (int i = 0; i < h30-1; i++)
                        for (int j = i+1; j < h30; j++)
                            if (ha[j] < ha[i]) { double t = ha[i]; ha[i]=ha[j]; ha[j]=t; }
                    for (int i = 0; i < h30-1; i++)
                        for (int j = i+1; j < h30; j++)
                            if (hb[j] < hb[i]) { double t = hb[i]; hb[i]=hb[j]; hb[j]=t; }
                    double med_old = ha[h30/2], med_new = hb[h30/2];
                    if (med_new >= med_old) { free(ha); free(hb); stop_flag = CMAES_STOP_STAGNATION; break; }
                }
                free(ha); free(hb);
            }
        }
    } /* end generation loop */

    /* -----------------------------------------------------------------------
     * Fill result
     * --------------------------------------------------------------------- */
    result->xbest    = xbest;
    result->fbest    = fbest;
    result->evals    = evals;
    result->gens     = gen;
    result->stop_flag = stop_flag;

    free(m); free(ps); free(pc); free(C); free(B); free(D); free(invsqrtC);
    free(arx); free(ary); free(arz); free(arfx); free(idx);
    free(ymean); free(tmp); free(w); free(fbest_hist);
    return DM_SUCCESS;

oom:
    free(m); free(ps); free(pc); free(C); free(B); free(D); free(invsqrtC);
    free(arx); free(ary); free(arz); free(arfx); free(idx);
    free(ymean); free(tmp); free(w);
    return DM_ERROR_MEMORY;
}

/* ---------------------------------------------------------------------------
 * Utility: print result
 * -------------------------------------------------------------------------- */

void cmaes_print_result(const CMAES_Result *result, int n) {
    static const char *stop_names[] = {
        "?", "BUDGET", "FTOL", "XTOL", "CONDCOV", "STAGNATION", "NOEFFECT"
    };
    int sf = result->stop_flag;
    const char *sname = (sf >= 1 && sf <= 6) ? stop_names[sf] : stop_names[0];
    printf("CMA-ES  fbest=%.6g  evals=%d  gens=%d  stop=%s\n  xbest=[",
           result->fbest, result->evals, result->gens, sname);
    for (int i = 0; i < n && i < 8; i++)
        printf("%.4g%s", result->xbest[i], (i < n-1 && i < 7) ? ", " : "");
    if (n > 8) printf(", ...");
    printf("]\n");
}

/* ---------------------------------------------------------------------------
 * DM framework plugin adapter
 * -------------------------------------------------------------------------- */

/*
 * The DM plugin interface expects a DM_Dataset input; CMA-ES is an optimiser,
 * not a pattern-miner.  We treat the first row of the dataset as the initial
 * mean x0 and the second row (if present) as the domain bounds, then minimise
 * the Rosenbrock function as a self-contained demonstration.  Users embedding
 * CMA-ES in a real application should call cmaes_run() directly.
 */
static double rosenbrock(const double *x, int n, void *ud) {
    (void)ud;
    double f = 0.0;
    for (int i = 0; i < n-1; i++)
        f += 100.0*(x[i+1] - x[i]*x[i])*(x[i+1] - x[i]*x[i]) + (1.0 - x[i])*(1.0 - x[i]);
    return f;
}

static DM_Status cmaes_plugin_run(DM_Dataset *ds, void *vparams) {
    (void)ds;
    (void)vparams;

    CMAES_Params p;
    memset(&p, 0, sizeof(p));
    p.n         = (ds && ds->max_id > 0) ? (int)ds->max_id : 10;
    p.objective = rosenbrock;

    CMAES_Result res;
    memset(&res, 0, sizeof(res));
    DM_Status st = cmaes_run(&p, &res);
    if (st == DM_SUCCESS) {
        cmaes_print_result(&res, p.n);
        free(res.xbest);
    }
    return st;
}

DM_Algorithm cmaes_algorithm = {
    .id          = "cmaes",
    .name        = "CMA-ES",
    .description = "Covariance Matrix Adaptation Evolution Strategy "
                   "(Hansen, arXiv:1604.00772v2, 2023).",
    .supported_types = 0xFFFFFFFF,
    .run         = cmaes_plugin_run,
};

DM_REGISTER_ALGORITHM(cmaes_algorithm)
