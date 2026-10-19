/*
 * Exact comparison of summed occupancy with a threshold.
 *
 * occ(X) = |X| * sum_{q in T(X)} 1/|t_q| is computed in double precision by the
 * miners. A double comparison with the threshold xi can be wrong when occ(X) and
 * xi differ by less than the rounding error. The functions below decide
 * occ(X) >= xi exactly:
 *   - xi = alpha * n (or an absolute xi) is represented as a fraction P/Q, where
 *     alpha is the shortest decimal that rounds to the given double (the value
 *     the user typed, e.g. 0.0005 = 5/10^4);
 *   - when the double value is farther than DM_OCC_TOL from xi, the double
 *     comparison is correct and is used;
 *   - otherwise the sum is evaluated exactly with arbitrary-precision integers:
 *     k * Q * sum_L c_L * (M / L) >= P * M, M = lcm of the distinct lengths L,
 *     c_L = number of supporting transactions of length L.
 * Pruning tests are not affected: they compare an upper bound with
 * xi - DM_OCC_TOL, which can only keep more candidates.
 */
#ifndef DM_EXACT_OCC_H
#define DM_EXACT_OCC_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned __int128 dm_u128;

/* relative width of the band in which the exact test is used; the rounding
 * error of a sum of up to 2^32 terms 1/L is far below it */
#define DM_OCC_TOL(xi) (1e-7 * (1.0 + fabs(xi)))

typedef struct {
    dm_u128 P, Q;      /* xi = P / Q */
    double value;      /* xi in double precision */
} DmThreshold;

/* ── decimal parsing ─────────────────────────────────────────────────────── */
/* Writes the shortest decimal that rounds to x as mantissa m and exponent e
 * (x = m * 10^e). Returns 0 on failure (x <= 0 or not representable). */
static inline int dm_shortest_decimal(double x, uint64_t *m, int *e) {
    if (!(x > 0.0) || !isfinite(x)) return 0;
    char buf[64];
    for (int p = 1; p <= 17; p++) {
        snprintf(buf, sizeof buf, "%.*e", p - 1, x);
        if (strtod(buf, NULL) != x) continue;
        uint64_t mant = 0;
        int digits = 0, exp10 = 0;
        const char *c = buf;
        for (; *c && *c != 'e'; c++) {
            if (*c >= '0' && *c <= '9') { mant = mant * 10 + (uint64_t)(*c - '0'); digits++; }
        }
        if (*c == 'e') exp10 = atoi(c + 1);
        *m = mant;
        *e = exp10 - (digits - 1);
        return 1;
    }
    return 0;
}

static inline dm_u128 dm_pow10(int k) {
    dm_u128 r = 1;
    while (k-- > 0) r *= 10;
    return r;
}

/* xi = arg * n if relative, else xi = arg; arg is read as the shortest decimal */
static inline DmThreshold dm_threshold_make(double arg, size_t n, int relative) {
    DmThreshold t;
    uint64_t m = 0;
    int e = 0;
    if (!dm_shortest_decimal(arg, &m, &e)) { m = 0; e = 0; }
    dm_u128 num = (dm_u128)m * (relative ? (dm_u128)n : 1);
    if (e >= 0) { t.P = num * dm_pow10(e); t.Q = 1; }
    else { t.P = num; t.Q = dm_pow10(-e); }
    t.value = relative ? arg * (double)n : arg;
    return t;
}

/* sigma = ceil(xi), the smallest support an HOI can have */
static inline size_t dm_threshold_ceil(DmThreshold t) {
    if (t.Q == 0) return 1;
    dm_u128 c = t.P / t.Q + (t.P % t.Q != 0);
    return c < 1 ? 1 : (size_t)c;
}

/* ── minimal arbitrary-precision unsigned integers (base 2^32) ───────────── */
typedef struct { uint32_t *d; size_t n, cap; } DmBig;

static inline void dm_big_init(DmBig *b) { b->cap = 8; b->n = 0; b->d = (uint32_t *)calloc(b->cap, sizeof(uint32_t)); }
static inline void dm_big_free(DmBig *b) { free(b->d); b->d = NULL; b->n = b->cap = 0; }
static inline void dm_big_reserve(DmBig *b, size_t need) {
    if (need <= b->cap) return;
    size_t c = b->cap;
    while (c < need) c *= 2;
    b->d = (uint32_t *)realloc(b->d, c * sizeof(uint32_t));
    memset(b->d + b->cap, 0, (c - b->cap) * sizeof(uint32_t));
    b->cap = c;
}
static inline void dm_big_trim(DmBig *b) { while (b->n > 0 && b->d[b->n - 1] == 0) b->n--; }
static inline void dm_big_set_u128(DmBig *b, dm_u128 v) {
    b->n = 0;
    dm_big_reserve(b, 4);
    while (v) { b->d[b->n++] = (uint32_t)v; v >>= 32; }
}
static inline void dm_big_copy(DmBig *dst, const DmBig *src) {
    dm_big_reserve(dst, src->n + 1);
    memcpy(dst->d, src->d, src->n * sizeof(uint32_t));
    dst->n = src->n;
}
static inline void dm_big_mul_u32(DmBig *b, uint32_t f) {
    uint64_t carry = 0;
    for (size_t i = 0; i < b->n; i++) {
        uint64_t x = (uint64_t)b->d[i] * f + carry;
        b->d[i] = (uint32_t)x;
        carry = x >> 32;
    }
    if (carry) { dm_big_reserve(b, b->n + 1); b->d[b->n++] = (uint32_t)carry; }
    dm_big_trim(b);
}
static inline void dm_big_mul_u128(DmBig *b, dm_u128 f) {
    /* b *= f, with f split into 32-bit words */
    DmBig acc, part;
    dm_big_init(&acc);
    dm_big_init(&part);
    int shift = 0;
    while (f) {
        uint32_t w = (uint32_t)f;
        f >>= 32;
        if (w) {
            dm_big_copy(&part, b);
            dm_big_mul_u32(&part, w);
            /* acc += part << (32 * shift) */
            dm_big_reserve(&acc, part.n + (size_t)shift + 1);
            uint64_t carry = 0;
            size_t i = 0;
            for (; i < part.n || carry; i++) {
                size_t k = i + (size_t)shift;
                if (k >= acc.n) { dm_big_reserve(&acc, k + 2); acc.n = k + 1; }
                uint64_t x = (uint64_t)acc.d[k] + (i < part.n ? part.d[i] : 0) + carry;
                acc.d[k] = (uint32_t)x;
                carry = x >> 32;
            }
        }
        shift++;
    }
    dm_big_trim(&acc);
    dm_big_copy(b, &acc);
    dm_big_free(&acc);
    dm_big_free(&part);
}
static inline void dm_big_add(DmBig *a, const DmBig *b) {
    size_t n = (a->n > b->n ? a->n : b->n) + 1;
    dm_big_reserve(a, n);
    uint64_t carry = 0;
    for (size_t i = 0; i < n; i++) {
        uint64_t x = (uint64_t)(i < a->n ? a->d[i] : 0) + (i < b->n ? b->d[i] : 0) + carry;
        a->d[i] = (uint32_t)x;
        carry = x >> 32;
    }
    a->n = n;
    dm_big_trim(a);
}
static inline uint32_t dm_big_mod_u32(const DmBig *b, uint32_t m) {
    uint64_t r = 0;
    for (size_t i = b->n; i-- > 0;) r = ((r << 32) | b->d[i]) % m;
    return (uint32_t)r;
}
static inline void dm_big_div_u32(DmBig *b, uint32_t m) {   /* exact or floor division */
    uint64_t r = 0;
    for (size_t i = b->n; i-- > 0;) {
        uint64_t x = (r << 32) | b->d[i];
        b->d[i] = (uint32_t)(x / m);
        r = x % m;
    }
    dm_big_trim(b);
}
static inline int dm_big_cmp(const DmBig *a, const DmBig *b) {
    if (a->n != b->n) return a->n < b->n ? -1 : 1;
    for (size_t i = a->n; i-- > 0;)
        if (a->d[i] != b->d[i]) return a->d[i] < b->d[i] ? -1 : 1;
    return 0;
}
static inline uint32_t dm_gcd_u32(uint32_t a, uint32_t b) { while (b) { uint32_t t = a % b; a = b; b = t; } return a; }

/* ── exact test ──────────────────────────────────────────────────────────── */
/* Decides k * sum_i w_i / len_i >= xi exactly; (len_i, w_i) need not be grouped. */
static inline int dm_occ_ge_exact(uint64_t k, const uint32_t *len, const uint64_t *w, size_t m, DmThreshold xi) {
    /* group by length: sort a copy of the pairs by length */
    uint64_t *pairs = (uint64_t *)malloc((m ? m : 1) * 2 * sizeof(uint64_t));
    size_t g = 0;
    for (size_t i = 0; i < m; i++) {
        size_t j = 0;
        for (j = 0; j < g; j++) if (pairs[2 * j] == len[i]) break;
        if (j == g) { pairs[2 * g] = len[i]; pairs[2 * g + 1] = 0; g++; }
        pairs[2 * j + 1] += w ? w[i] : 1;
    }
    DmBig M, N, term, lhs, rhs;
    dm_big_init(&M); dm_big_init(&N); dm_big_init(&term); dm_big_init(&lhs); dm_big_init(&rhs);
    dm_big_set_u128(&M, 1);
    for (size_t j = 0; j < g; j++) {
        uint32_t L = (uint32_t)pairs[2 * j];
        uint32_t gc = dm_gcd_u32(dm_big_mod_u32(&M, L), L);
        dm_big_mul_u32(&M, L / gc);
    }
    dm_big_set_u128(&N, 0);
    for (size_t j = 0; j < g; j++) {
        uint32_t L = (uint32_t)pairs[2 * j];
        dm_big_copy(&term, &M);
        dm_big_div_u32(&term, L);
        dm_big_mul_u128(&term, (dm_u128)pairs[2 * j + 1]);
        dm_big_add(&N, &term);
    }
    dm_big_copy(&lhs, &N);
    dm_big_mul_u128(&lhs, (dm_u128)k);
    dm_big_mul_u128(&lhs, xi.Q);
    dm_big_copy(&rhs, &M);
    dm_big_mul_u128(&rhs, xi.P);
    int ge = dm_big_cmp(&lhs, &rhs) >= 0;
    dm_big_free(&M); dm_big_free(&N); dm_big_free(&term); dm_big_free(&lhs); dm_big_free(&rhs);
    free(pairs);
    return ge;
}

/* Exact test for equal lengths L: k * cnt / L >= P / Q. */
static inline int dm_occ_ge_uniform(uint64_t k, uint64_t cnt, uint32_t L, DmThreshold xi) {
    DmBig lhs, rhs;
    dm_big_init(&lhs); dm_big_init(&rhs);
    dm_big_set_u128(&lhs, (dm_u128)k * cnt);
    dm_big_mul_u128(&lhs, xi.Q);
    dm_big_set_u128(&rhs, xi.P);
    dm_big_mul_u32(&rhs, L);
    int ge = dm_big_cmp(&lhs, &rhs) >= 0;
    dm_big_free(&lhs); dm_big_free(&rhs);
    return ge;
}

/* 1 if the double value decides the test, with *res set; 0 if the exact test is needed */
static inline int dm_occ_decided(double occ, DmThreshold xi, int *res) {
    double tol = DM_OCC_TOL(xi.value);
    if (occ >= xi.value + tol) { *res = 1; return 1; }
    if (occ < xi.value - tol) { *res = 0; return 1; }
    return 0;
}

/* pruning test: keep (return 1) unless the bound is clearly below xi */
static inline int dm_bound_ge(double bound, DmThreshold xi) {
    return bound >= xi.value - DM_OCC_TOL(xi.value);
}

#endif
