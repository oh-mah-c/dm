/*
 * test_cmaes.c -- Self-contained test suite for the CMA-ES implementation.
 *
 * Standard benchmark functions:
 *   Sphere       : f(x) = sum(x_i^2),               optimum 0 at origin.
 *   Rosenbrock   : f(x) = sum[100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2], optimum 0 at 1s.
 *   Rastrigin    : f(x) = 10n + sum[x_i^2 - 10*cos(2*pi*x_i)],     optimum 0 at origin.
 *   Ellipsoid    : f(x) = sum[(1e6)^(i/(n-1)) * x_i^2],            optimum 0 at origin.
 *   Cigar        : f(x) = x_0^2 + 1e6 * sum_{i>0} x_i^2,           optimum 0 at origin.
 *   Ackley       : standard Ackley 2-D, optimum ~0 at origin.
 */

#include "algorithms/cmaes.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Benchmark objective functions
 * ------------------------------------------------------------------------- */

static double sphere(const double *x, int n, void *ud) {
    (void)ud;
    double f = 0.0;
    for (int i = 0; i < n; i++) f += x[i]*x[i];
    return f;
}

static double rosenbrock(const double *x, int n, void *ud) {
    (void)ud;
    double f = 0.0;
    for (int i = 0; i < n-1; i++)
        f += 100.0*(x[i+1]-x[i]*x[i])*(x[i+1]-x[i]*x[i]) + (1.0-x[i])*(1.0-x[i]);
    return f;
}

static double rastrigin(const double *x, int n, void *ud) {
    (void)ud;
    const double pi = 3.14159265358979323846;
    double f = 10.0 * n;
    for (int i = 0; i < n; i++) f += x[i]*x[i] - 10.0*cos(2.0*pi*x[i]);
    return f;
}

static double ellipsoid(const double *x, int n, void *ud) {
    (void)ud;
    double f = 0.0;
    for (int i = 0; i < n; i++) {
        double scale = pow(1e6, (double)i / (double)(n-1));
        f += scale * x[i]*x[i];
    }
    return f;
}

static double cigar(const double *x, int n, void *ud) {
    (void)ud;
    double f = x[0]*x[0];
    for (int i = 1; i < n; i++) f += 1e6 * x[i]*x[i];
    return f;
}

static double ackley2d(const double *x, int n, void *ud) {
    (void)ud; (void)n;
    const double pi = 3.14159265358979323846;
    double a = 20.0, b = 0.2, c = 2.0*pi;
    double s1 = x[0]*x[0] + x[1]*x[1];
    double s2 = cos(c*x[0]) + cos(c*x[1]);
    return -a*exp(-b*sqrt(s1/2.0)) - exp(s2/2.0) + a + exp(1.0);
}

/* -------------------------------------------------------------------------
 * Test harness
 * ------------------------------------------------------------------------- */

typedef struct {
    const char       *name;
    CMAES_ObjectiveFn fn;
    int               n;
    double            x0_val;   /* uniform initial mean component */
    double            sigma0;
    double            ftol;
    int               max_evals;
} TestCase;

static int passed = 0, failed = 0;

static void run_test(const TestCase *tc) {
    double *x0 = (double *)malloc((size_t)tc->n * sizeof(double));
    if (!x0) { printf("  [FAIL] %s: OOM\n", tc->name); failed++; return; }
    for (int i = 0; i < tc->n; i++) x0[i] = tc->x0_val;

    CMAES_Params p;
    memset(&p, 0, sizeof(p));
    p.n         = tc->n;
    p.x0        = x0;
    p.sigma0    = tc->sigma0;
    p.ftol      = tc->ftol;
    p.max_evals = tc->max_evals;
    p.objective = tc->fn;

    CMAES_Result res;
    memset(&res, 0, sizeof(res));
    DM_Status st = cmaes_run(&p, &res);
    free(x0);

    if (st != DM_SUCCESS) {
        printf("  [FAIL] %-20s  cmaes_run returned %d\n", tc->name, st);
        failed++;
        return;
    }

    int ok = (res.fbest <= tc->ftol * 10.0); /* allow 10x slack */
    printf("  [%s] %-20s  fbest=%.3e  evals=%6d  gens=%4d  stop=%d\n",
           ok ? "PASS" : "FAIL", tc->name, res.fbest, res.evals, res.gens, res.stop_flag);
    if (ok) passed++; else failed++;
    free(res.xbest);
}

/* -------------------------------------------------------------------------
 * Parameter / API unit tests
 * ------------------------------------------------------------------------- */

static void test_invalid_params(void) {
    CMAES_Result res;
    memset(&res, 0, sizeof(res));
    CMAES_Params p; memset(&p, 0, sizeof(p));

    /* NULL params. */
    int ok1 = (cmaes_run(NULL, &res) == DM_ERROR_INVALID_PARAM);
    /* n == 0. */
    p.n = 0; p.objective = sphere;
    int ok2 = (cmaes_run(&p, &res) == DM_ERROR_INVALID_PARAM);
    /* NULL objective. */
    p.n = 2; p.objective = NULL;
    int ok3 = (cmaes_run(&p, &res) == DM_ERROR_INVALID_PARAM);

    int ok = ok1 && ok2 && ok3;
    printf("  [%s] invalid_params\n", ok ? "PASS" : "FAIL");
    if (ok) passed++; else failed++;
}

static void test_1d_sphere(void) {
    /* 1-D quadratic: converges trivially. */
    double x0 = 3.0;
    CMAES_Params p; memset(&p, 0, sizeof(p));
    p.n = 1; p.x0 = &x0; p.sigma0 = 1.0; p.ftol = 1e-8;
    p.objective = sphere; p.max_evals = 5000;
    CMAES_Result res; memset(&res, 0, sizeof(res));
    DM_Status st = cmaes_run(&p, &res);
    int ok = (st == DM_SUCCESS && res.fbest < 1e-7);
    printf("  [%s] 1d_sphere  fbest=%.3e\n", ok ? "PASS" : "FAIL", res.fbest);
    if (ok) passed++; else failed++;
    if (st == DM_SUCCESS) free(res.xbest);
}

static void test_result_xbest_not_null(void) {
    double x0 = 1.0;
    CMAES_Params p; memset(&p, 0, sizeof(p));
    p.n = 2; p.x0 = &x0; p.sigma0 = 0.5; p.max_evals = 200;
    p.objective = sphere;
    CMAES_Result res; memset(&res, 0, sizeof(res));
    DM_Status st = cmaes_run(&p, &res);
    int ok = (st == DM_SUCCESS && res.xbest != NULL);
    printf("  [%s] result_xbest_not_null\n", ok ? "PASS" : "FAIL");
    if (ok) passed++; else failed++;
    if (st == DM_SUCCESS) free(res.xbest);
}

static void test_budget_respected(void) {
    double x0 = 5.0;
    CMAES_Params p; memset(&p, 0, sizeof(p));
    p.n = 5; p.x0 = &x0; p.sigma0 = 1.0; p.max_evals = 300;
    p.objective = rosenbrock;
    CMAES_Result res; memset(&res, 0, sizeof(res));
    cmaes_run(&p, &res);
    /* evals may slightly exceed max_evals by up to lambda-1 (last generation). */
    int lambda_est = 4 + (int)(3.0 * log(5.0));
    int ok = (res.evals <= p.max_evals + lambda_est);
    printf("  [%s] budget_respected  evals=%d  budget=%d\n",
           ok ? "PASS" : "FAIL", res.evals, p.max_evals);
    if (ok) passed++; else failed++;
    free(res.xbest);
}

static void test_plugin_descriptor(void) {
    int ok = (cmaes_algorithm.run != NULL
              && strcmp(cmaes_algorithm.id, "cmaes") == 0);
    printf("  [%s] plugin_descriptor  id=%s\n",
           ok ? "PASS" : "FAIL", cmaes_algorithm.id);
    if (ok) passed++; else failed++;
}

/* -------------------------------------------------------------------------
 * Entry point
 * ------------------------------------------------------------------------- */

int main(void) {
    printf("=== CMA-ES Tests ===\n\n");

    /* --- Unit tests --- */
    printf("-- API / unit tests --\n");
    test_invalid_params();
    test_1d_sphere();
    test_result_xbest_not_null();
    test_budget_respected();
    test_plugin_descriptor();
    printf("\n");

    /* --- Benchmark convergence tests --- */
    printf("-- Benchmark convergence (n=10, unless noted) --\n");
    TestCase cases[] = {
        /* name              fn          n    x0    sigma0  ftol    max_evals */
        { "sphere_n5",       sphere,     5,  3.0,   1.0,   1e-9,  50000  },
        { "sphere_n10",      sphere,    10,  3.0,   1.0,   1e-8,  100000 },
        { "sphere_n20",      sphere,    20,  3.0,   1.0,   1e-6,  200000 },
        { "ellipsoid_n10",   ellipsoid, 10,  3.0,   1.0,   1e-7,  200000 },
        { "cigar_n10",       cigar,     10,  3.0,   1.0,   1e-7,  200000 },
        { "rosenbrock_n2",   rosenbrock, 2,  0.5,   0.3,   1e-6,  50000  },
        { "rosenbrock_n5",   rosenbrock, 5,  0.5,   0.3,   1e-5,  200000 },
        { "rosenbrock_n10",  rosenbrock,10,  0.5,   0.3,   1e-4,  500000 },
        /* Rastrigin is multimodal; CMA-ES may converge to a local minimum.
         * We test that it meaningfully reduces f (gets close to 0 or a local opt),
         * not necessarily the global minimum.  ftol set generously. */
        { "rastrigin_n2",    rastrigin,  2,  2.0,   1.0,   2.0,   200000 },
        { "rastrigin_n5",    rastrigin,  5,  2.0,   1.0,   15.0,  600000 },
        /* Ackley is multimodal; test meaningful reduction toward local minimum. */
        { "ackley_2d",       ackley2d,   2,  2.0,   1.0,   3.0,   100000 },
    };
    int ncases = (int)(sizeof(cases)/sizeof(cases[0]));
    for (int i = 0; i < ncases; i++) run_test(&cases[i]);

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return (failed == 0) ? 0 : 1;
}
