/*
 * VOLT: Vocabulary Learning via Optimal Transport for NMT
 * ACL 2021  -  Xu, Zhou, Gan, Zheng, Li (ByteDance AI Lab)
 * GPU acceleration via Vulkan compute (dm_gpu_sinkhorn).
 *
 * Implements Algorithm 1 exactly as in the paper:
 *   For t in S:
 *     T = L[:t]                          top-t BPE candidates by frequency
 *     K[i][j] = 1/len(T[i])             if char_j in T[i],  else 0
 *     u = P(T)/(K@v),  v = P(C)/(K^T@u) Sinkhorn iterations
 *     P_opt = diag(u)*K*diag(v)
 *     vocab = {T[i]: row_sum(P_opt[i]) >= threshold * P(T[i])}
 *     H_v = -1/l_v * sum_{i in vocab} P(i)*log P(i)
 *   Return vocab with max MUV = -(H(t)-H(t-1)) / (S[t]-S[t-1])
 */

#include "tokenizer/volt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DM_GPU
#include "core/gpu/dm_gpu.h"
#endif

/* -------------------------------------------------------------------------
 * String utilities
 * ---------------------------------------------------------------------- */

static char *xstrdup(const char *s) {
    size_t n = strlen(s); char *o = (char *)malloc(n + 1);
    if (o) memcpy(o, s, n + 1); return o;
}
static char *xstrndup(const char *s, size_t n) {
    char *o = (char *)malloc(n + 1); if (o) { memcpy(o, s, n); o[n] = '\0'; } return o;
}

/* -------------------------------------------------------------------------
 * Simple open-addressing string→double hash map
 * ---------------------------------------------------------------------- */

typedef struct { char *key; double val; } KVD;
typedef struct { KVD *data; size_t n, cap; } KVDMap;

static size_t _kvd_h(const char *s) {
    size_t h = 5381;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++)
        h = (h << 5) + h + *p;
    return h;
}
static KVDMap *kvdmap_new(void) { return (KVDMap *)calloc(1, sizeof(KVDMap)); }
static void kvdmap_grow(KVDMap *m) {
    size_t nc = m->cap ? m->cap * 2 : 64;
    KVD *nd = (KVD *)calloc(nc, sizeof(KVD));
    for (size_t i = 0; i < m->cap; i++) {
        if (!m->data[i].key) continue;
        size_t h = _kvd_h(m->data[i].key) % nc;
        while (nd[h].key) h = (h + 1) % nc;
        nd[h] = m->data[i];
    }
    free(m->data); m->data = nd; m->cap = nc;
}
static void kvdmap_add(KVDMap *m, const char *key, double d) {
    if (m->n * 2 >= m->cap) kvdmap_grow(m);
    size_t h = _kvd_h(key) % m->cap;
    while (m->data[h].key && strcmp(m->data[h].key, key) != 0)
        h = (h + 1) % m->cap;
    if (!m->data[h].key) { m->data[h].key = xstrdup(key); m->n++; }
    m->data[h].val += d;
}
static double kvdmap_get(const KVDMap *m, const char *key) {
    if (!m->cap) return 0.0;
    size_t h = _kvd_h(key) % m->cap;
    while (m->data[h].key && strcmp(m->data[h].key, key) != 0)
        h = (h + 1) % m->cap;
    return m->data[h].key ? m->data[h].val : 0.0;
}
static void kvdmap_free(KVDMap *m) {
    if (!m) return;
    for (size_t i = 0; i < m->cap; i++) free(m->data[i].key);
    free(m->data); free(m);
}

/* -------------------------------------------------------------------------
 * String→size_t map (char index lookup)
 * ---------------------------------------------------------------------- */

typedef struct { char *key; size_t val; } KVZ;
typedef struct { KVZ *data; size_t n, cap; } KVZMap;

static size_t _kvz_h(const char *s) { return _kvd_h(s); }
static KVZMap *kvzmap_new(void) { return (KVZMap *)calloc(1, sizeof(KVZMap)); }
static void kvzmap_grow(KVZMap *m) {
    size_t nc = m->cap ? m->cap * 2 : 64;
    KVZ *nd = (KVZ *)calloc(nc, sizeof(KVZ));
    for (size_t i = 0; i < m->cap; i++) {
        if (!m->data[i].key) continue;
        size_t h = _kvz_h(m->data[i].key) % nc;
        while (nd[h].key) h = (h + 1) % nc;
        nd[h] = m->data[i];
    }
    free(m->data); m->data = nd; m->cap = nc;
}
static void kvzmap_set(KVZMap *m, const char *key, size_t val) {
    if (m->n * 2 >= m->cap) kvzmap_grow(m);
    size_t h = _kvz_h(key) % m->cap;
    while (m->data[h].key && strcmp(m->data[h].key, key) != 0)
        h = (h + 1) % m->cap;
    if (!m->data[h].key) { m->data[h].key = xstrdup(key); m->n++; }
    m->data[h].val = val;
}
static int kvzmap_get(const KVZMap *m, const char *key, size_t *out) {
    if (!m->cap) return 0;
    size_t h = _kvz_h(key) % m->cap;
    while (m->data[h].key && strcmp(m->data[h].key, key) != 0)
        h = (h + 1) % m->cap;
    if (m->data[h].key) { *out = m->data[h].val; return 1; }
    return 0;
}
static void kvzmap_free(KVZMap *m) {
    if (!m) return;
    for (size_t i = 0; i < m->cap; i++) free(m->data[i].key);
    free(m->data); free(m);
}

/* -------------------------------------------------------------------------
 * UTF-8 utilities
 * ---------------------------------------------------------------------- */

static size_t utf8_clen(unsigned char c) {
    if ((c & 0x80) == 0)    return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}
static size_t utf8_ncodepoints(const char *s) {
    size_t n = 0;
    const unsigned char *p = (const unsigned char *)s;
    while (*p) { n++; p += utf8_clen(*p); }
    return n;
}

/* -------------------------------------------------------------------------
 * BPE merge rules loading and encoding
 * ---------------------------------------------------------------------- */

typedef struct { char *left, *right, *merged; } BPEMerge;
typedef struct { BPEMerge *data; size_t n, cap; } BPEMerges;

static void bpe_push(BPEMerges *m, const char *l, const char *r) {
    if (m->n == m->cap) {
        size_t nc = m->cap ? m->cap * 2 : 256;
        m->data = (BPEMerge *)realloc(m->data, nc * sizeof(BPEMerge));
        m->cap = nc;
    }
    BPEMerge *e = &m->data[m->n++];
    e->left = xstrdup(l); e->right = xstrdup(r);
    size_t nl = strlen(l), nr = strlen(r);
    e->merged = (char *)malloc(nl + nr + 1);
    memcpy(e->merged, l, nl); memcpy(e->merged + nl, r, nr + 1);
}
static void bpe_free(BPEMerges *m) {
    for (size_t i = 0; i < m->n; i++) {
        free(m->data[i].left); free(m->data[i].right); free(m->data[i].merged);
    }
    free(m->data); memset(m, 0, sizeof(*m));
}
static int bpe_load(const char *path, BPEMerges *out) {
    FILE *f = fopen(path, "r"); if (!f) return -1;
    char buf[4096];
    while (fgets(buf, sizeof(buf), f)) {
        char *p = buf;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\r') continue;
        char *l = p; while (*p && *p != ' ' && *p != '\t') p++;
        if (!*p) continue; *p++ = '\0';
        while (*p == ' ' || *p == '\t') p++;
        char *r = p; while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
        *p = '\0';
        if (*r) bpe_push(out, l, r);
    }
    fclose(f); return 0;
}

/* BPE-encode one word (whitespace pre-tokenized) */
static char **bpe_encode_word(const char *word, const BPEMerges *m, size_t *out_n) {
    /* Initial: each UTF-8 codepoint + </w> */
    size_t ncps = utf8_ncodepoints(word);
    char **syms = (char **)malloc((ncps + 2) * sizeof(char *));
    size_t n = 0;
    const char *p = word;
    while (*p) {
        size_t cl = utf8_clen((unsigned char)*p);
        syms[n++] = xstrndup(p, cl); p += cl;
    }
    syms[n++] = xstrdup("</w>");

    for (size_t mi = 0; mi < m->n; mi++) {
        const char *left = m->data[mi].left, *right = m->data[mi].right;
        const char *merged = m->data[mi].merged;
        size_t new_n = 0;
        char **ns = (char **)malloc(n * sizeof(char *));
        size_t i = 0;
        while (i < n) {
            if (i + 1 < n && strcmp(syms[i], left) == 0
                           && strcmp(syms[i + 1], right) == 0) {
                ns[new_n++] = xstrdup(merged);
                free(syms[i]); free(syms[i + 1]); i += 2;
            } else { ns[new_n++] = syms[i++]; }
        }
        free(syms); syms = ns; n = new_n;
    }
    *out_n = n; return syms;
}

/* -------------------------------------------------------------------------
 * Corpus reading + frequency counting
 * ---------------------------------------------------------------------- */

static KVDMap *read_word_freq(const char *path, size_t max_lines) {
    FILE *f = fopen(path, "r"); if (!f) return NULL;
    KVDMap *wf = kvdmap_new();
    char buf[65536]; size_t lines = 0;
    while (fgets(buf, sizeof(buf), f)) {
        if (max_lines && lines >= max_lines) break; lines++;
        char *p = buf;
        while (*p) {
            while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
            if (!*p) break;
            char *s = p;
            while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
            char *w = xstrndup(s, (size_t)(p - s));
            kvdmap_add(wf, w, 1.0); free(w);
        }
    }
    fclose(f); return wf;
}

static KVDMap *token_freqs_from_corpus(const KVDMap *wf, const BPEMerges *m) {
    KVDMap *tf = kvdmap_new();
    for (size_t i = 0; i < wf->cap; i++) {
        if (!wf->data[i].key) continue;
        double freq = wf->data[i].val;
        size_t n; char **toks = bpe_encode_word(wf->data[i].key, m, &n);
        for (size_t j = 0; j < n; j++) { kvdmap_add(tf, toks[j], freq); free(toks[j]); }
        free(toks);
    }
    return tf;
}

static KVDMap *char_freqs_from_corpus(const KVDMap *wf) {
    KVDMap *cf = kvdmap_new(); char cbuf[8];
    for (size_t i = 0; i < wf->cap; i++) {
        if (!wf->data[i].key) continue;
        double freq = wf->data[i].val;
        const char *p = wf->data[i].key;
        while (*p) {
            size_t cl = utf8_clen((unsigned char)*p);
            memcpy(cbuf, p, cl); cbuf[cl] = '\0';
            kvdmap_add(cf, cbuf, freq); p += cl;
        }
    }
    return cf;
}

/* -------------------------------------------------------------------------
 * Sparse kernel K  (CSR-like per-row list of (col, val))
 * ---------------------------------------------------------------------- */

typedef struct { size_t *col; double *val; size_t n, cap; } SRow;

static void srow_push(SRow *r, size_t col, double val) {
    if (r->n == r->cap) {
        size_t nc = r->cap ? r->cap * 2 : 8;
        r->col = (size_t *)realloc(r->col, nc * sizeof(size_t));
        r->val = (double *)realloc(r->val, nc * sizeof(double));
        r->cap = nc;
    }
    r->col[r->n] = col; r->val[r->n] = val; r->n++;
}
static void srow_free(SRow *r) { free(r->col); free(r->val); memset(r, 0, sizeof(*r)); }

/*
 * Build sparse K: K[i][j] = 1/len(T[i])  if char_j appears in T[i] (excl. </w>)
 *                           = 0           otherwise
 * len = number of Unicode codepoints in T[i] (excluding the </w> marker)
 */
static SRow *build_kernel(char **T, size_t n_tok, KVZMap *char_idx) {
    SRow *K = (SRow *)calloc(n_tok, sizeof(SRow));
    char cbuf[8];
    for (size_t i = 0; i < n_tok; i++) {
        const char *tok = T[i];
        size_t tl = strlen(tok), eow = strlen("</w>");
        char *clean = (tl >= eow && strcmp(tok + tl - eow, "</w>") == 0)
                      ? xstrndup(tok, tl - eow) : xstrdup(tok);
        size_t tok_cp = utf8_ncodepoints(clean);
        if (tok_cp == 0) { free(clean); continue; }
        double kval = 1.0 / (double)tok_cp;
        /* Collect unique chars (skip duplicates within the same token) */
        size_t seen_buf[128], seen_n = 0;
        const char *p = clean;
        while (*p) {
            size_t cl = utf8_clen((unsigned char)*p);
            memcpy(cbuf, p, cl); cbuf[cl] = '\0'; p += cl;
            size_t cidx;
            if (!kvzmap_get(char_idx, cbuf, &cidx)) continue;
            int dup = 0;
            for (size_t s = 0; s < seen_n; s++) if (seen_buf[s] == cidx) { dup = 1; break; }
            if (dup) continue;
            if (seen_n < 128) seen_buf[seen_n++] = cidx;
            srow_push(&K[i], cidx, kval);
        }
        free(clean);
    }
    return K;
}
static void kernel_free(SRow *K, size_t n) {
    for (size_t i = 0; i < n; i++) srow_free(&K[i]);
    free(K);
}

/* -------------------------------------------------------------------------
 * Sinkhorn (Algorithm 1, VOLT paper)
 *   u = P(T) / (K @ v)
 *   v = P(C) / (K^T @ u)
 *   P_opt row sums = u[i] * (K[i] @ v)
 * ---------------------------------------------------------------------- */

static void sinkhorn(const SRow *K, size_t n_tok, size_t n_char,
                     const double *p_tok, const double *p_char,
                     int max_iter, double tol,
                     double *row_sums_out) {
    const double EPS = 1e-300;
    double *u = (double *)malloc(n_tok  * sizeof(double));
    double *v = (double *)malloc(n_char * sizeof(double));
    double *u0 = (double *)malloc(n_tok  * sizeof(double));
    double *v0 = (double *)malloc(n_char * sizeof(double));
    double *Kv  = (double *)malloc(n_tok  * sizeof(double));
    double *KTu = (double *)malloc(n_char * sizeof(double));

    for (size_t i = 0; i < n_tok; i++)  u[i] = 1.0;
    for (size_t j = 0; j < n_char; j++) v[j] = 1.0;

    for (int it = 0; it < max_iter; it++) {
        memcpy(u0, u, n_tok  * sizeof(double));
        memcpy(v0, v, n_char * sizeof(double));

        /* Kv = K @ v */
        for (size_t i = 0; i < n_tok; i++) {
            double s = 0.0;
            for (size_t k = 0; k < K[i].n; k++) s += K[i].val[k] * v[K[i].col[k]];
            Kv[i] = s;
        }
        /* u = P(T) / Kv */
        for (size_t i = 0; i < n_tok; i++)
            u[i] = p_tok[i] / (Kv[i] > EPS ? Kv[i] : EPS);

        /* KTu = K^T @ u */
        for (size_t j = 0; j < n_char; j++) KTu[j] = 0.0;
        for (size_t i = 0; i < n_tok; i++)
            for (size_t k = 0; k < K[i].n; k++)
                KTu[K[i].col[k]] += K[i].val[k] * u[i];

        /* v = P(C) / KTu */
        for (size_t j = 0; j < n_char; j++)
            v[j] = p_char[j] / (KTu[j] > EPS ? KTu[j] : EPS);

        /* Convergence check */
        double du = 0.0, dv = 0.0;
        for (size_t i = 0; i < n_tok; i++) { double d = fabs(u[i]-u0[i]); if (d>du) du=d; }
        for (size_t j = 0; j < n_char; j++) { double d = fabs(v[j]-v0[j]); if (d>dv) dv=d; }
        if (du < tol && dv < tol) break;
    }

    /* row_sums[i] = u[i] * (K[i] @ v) */
    for (size_t i = 0; i < n_tok; i++) {
        double s = 0.0;
        for (size_t k = 0; k < K[i].n; k++) s += K[i].val[k] * v[K[i].col[k]];
        row_sums_out[i] = u[i] * s;
    }

    free(u); free(v); free(u0); free(v0); free(Kv); free(KTu);
}

/* -------------------------------------------------------------------------
 * Entropy Eq. 2:  H_v = -1/l_v * sum_{i in v} P(i)*log P(i)
 * P(i) = token_freq[i] / sum_{j in v} token_freq[j]
 * l_v  = mean Unicode length of tokens in v (excluding </w>)
 * ---------------------------------------------------------------------- */

static double compute_entropy(char **vocab, size_t nv, const KVDMap *tf) {
    if (!nv) return 0.0;
    double total_f = 0.0, total_l = 0.0;
    for (size_t i = 0; i < nv; i++) {
        total_f += kvdmap_get(tf, vocab[i]);
        size_t tl = strlen(vocab[i]), eow = strlen("</w>");
        char *c = (tl >= eow && strcmp(vocab[i] + tl - eow, "</w>") == 0)
                  ? xstrndup(vocab[i], tl - eow) : xstrdup(vocab[i]);
        total_l += (double)utf8_ncodepoints(c); free(c);
    }
    if (total_f == 0.0 || nv == 0) return 0.0;
    double l_v = total_l / (double)nv;
    if (l_v <= 0.0) return 0.0;
    double H = 0.0;
    for (size_t i = 0; i < nv; i++) {
        double f = kvdmap_get(tf, vocab[i]);
        if (f > 0.0) { double p = f / total_f; H -= p * log(p); }
    }
    return H / l_v;
}

/* -------------------------------------------------------------------------
 * Sort comparator: descending by value
 * ---------------------------------------------------------------------- */

static int cmp_kv_desc(const void *a, const void *b) {
    double va = ((const KVD *)a)->val, vb = ((const KVD *)b)->val;
    return (vb > va) ? 1 : (vb < va) ? -1 : 0;
}

/* -------------------------------------------------------------------------
 * JSON output
 * ---------------------------------------------------------------------- */

static void json_s(FILE *o, const char *s) {
    fputc('"', o);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p == '"' || *p == '\\') { fputc('\\', o); fputc(*p, o); }
        else if (*p < 0x20) fprintf(o, "\\u%04x", *p);
        else fputc(*p, o);
    }
    fputc('"', o);
}

typedef struct { int size; double entropy; char **vocab; size_t nv; } VStep;

static void write_result(const char *path, const VStep *best, double muv,
                          const VStep *steps, int ns) {
    FILE *o = fopen(path, "w"); if (!o) { perror(path); return; }
    fprintf(o, "{\n  \"version\": \"dm-volt-acl2021\",\n"
               "  \"algorithm\": \"vocabulary-learning-optimal-transport\",\n"
               "  \"optimal_S\": %d,\n  \"vocab_size\": %zu,\n"
               "  \"muv\": %.10g,\n  \"entropy\": %.10g,\n"
               "  \"vocab\": [\n",
               best->size, best->nv, muv, best->entropy);
    for (size_t i = 0; i < best->nv; i++) {
        const char *tok = best->vocab[i];
        size_t tl = strlen(tok), eow = strlen("</w>");
        char *clean = (tl >= eow && strcmp(tok + tl - eow, "</w>") == 0)
                      ? xstrndup(tok, tl - eow) : xstrdup(tok);
        fprintf(o, "    "); json_s(o, clean); free(clean);
        if (i + 1 < best->nv) fputc(',', o);
        fputc('\n', o);
    }
    fprintf(o, "  ],\n  \"all_sizes\": [\n");
    for (int t = 0; t < ns; t++) {
        fprintf(o, "    {\"size\": %d, \"vocab_size\": %zu, \"entropy\": %.8g}",
                steps[t].size, steps[t].nv, steps[t].entropy);
        if (t + 1 < ns) fputc(',', o);
        fputc('\n', o);
    }
    fprintf(o, "  ]\n}\n");
    fclose(o);
}

/* -------------------------------------------------------------------------
 * Main VOLT driver
 * ---------------------------------------------------------------------- */

#ifdef DM_GPU
/* Convert SRow *K (double/size_t) → float CSR and call dm_gpu_sinkhorn.
 * row_sums_out is double; conversion is done internally.
 * Returns 0 on success, -1 to fall back to the CPU sinkhorn. */
static int sinkhorn_gpu(DmGpuCtx *gpu, const SRow *K,
                        size_t n_tok, size_t n_char,
                        const double *p_tok, const double *p_char,
                        int max_iter, float tol, double *row_sums_out) {
    /* Count NNZ */
    uint32_t nnz = 0;
    for (size_t i = 0; i < n_tok; i++) nnz += (uint32_t)K[i].n;

    uint32_t *coo_row = (uint32_t *)malloc(nnz * sizeof(uint32_t));
    uint32_t *coo_col = (uint32_t *)malloc(nnz * sizeof(uint32_t));
    float    *coo_val = (float    *)malloc(nnz * sizeof(float));
    float    *fp_tok  = (float    *)malloc(n_tok  * sizeof(float));
    float    *fp_char = (float    *)malloc(n_char * sizeof(float));
    float    *frow    = (float    *)malloc(n_tok  * sizeof(float));
    if (!coo_row || !coo_col || !coo_val || !fp_tok || !fp_char || !frow) goto fail;

    { uint32_t k = 0;
      for (size_t i = 0; i < n_tok; i++)
          for (size_t j = 0; j < K[i].n; j++) {
              coo_row[k] = (uint32_t)i;
              coo_col[k] = (uint32_t)K[i].col[j];
              coo_val[k] = (float)K[i].val[j];
              k++;
          }
    }
    for (size_t i = 0; i < n_tok;  i++) fp_tok[i]  = (float)p_tok[i];
    for (size_t j = 0; j < n_char; j++) fp_char[j] = (float)p_char[j];

    uint32_t *K_rp = NULL, *K_ci = NULL; float *K_v = NULL;
    dm_gpu_build_csr(coo_row, coo_col, coo_val, nnz,
                     (uint32_t)n_tok, (uint32_t)n_char,
                     &K_rp, &K_ci, &K_v);
    if (!K_rp) goto fail;

    DmGpuCSR Kcsr = { K_rp, K_ci, K_v, (uint32_t)n_tok, (uint32_t)n_char, nnz };

    uint32_t *Kt_rp = NULL, *Kt_ci = NULL; float *Kt_v = NULL;
    dm_gpu_csr_transpose(&Kcsr, &Kt_rp, &Kt_ci, &Kt_v);
    if (!Kt_rp) { free(K_rp); free(K_ci); free(K_v); goto fail; }

    DmGpuCSR Ktcsr = { Kt_rp, Kt_ci, Kt_v, (uint32_t)n_char, (uint32_t)n_tok, nnz };

    int rc = dm_gpu_sinkhorn(gpu, &Kcsr, &Ktcsr, fp_tok, fp_char, max_iter, tol, frow);

    if (rc == DM_GPU_OK)
        for (size_t i = 0; i < n_tok; i++) row_sums_out[i] = (double)frow[i];

    free(K_rp); free(K_ci); free(K_v);
    free(Kt_rp); free(Kt_ci); free(Kt_v);
    free(coo_row); free(coo_col); free(coo_val);
    free(fp_tok); free(fp_char); free(frow);
    return (rc == DM_GPU_OK) ? 0 : -1;

fail:
    free(coo_row); free(coo_col); free(coo_val);
    free(fp_tok); free(fp_char); free(frow);
    return -1;
}
#endif /* DM_GPU */

static int volt_run(const char *corpus, const char *bpe_path,
                    const char *output,
                    int *S, int S_len,
                    double threshold, int sink_iters, double sink_tol,
                    size_t max_lines, int stats, void *gpu_ctx) {

    fprintf(stderr, "Loading corpus: %s\n", corpus);
    KVDMap *wf = read_word_freq(corpus, max_lines);
    if (!wf) { fprintf(stderr, "volt: cannot open corpus\n"); return 1; }
    size_t nw = 0; double tw = 0;
    for (size_t i = 0; i < wf->cap; i++) {
        if (!wf->data[i].key) continue; nw++; tw += wf->data[i].val;
    }
    fprintf(stderr, "  %zu word types, %.0f tokens\n", nw, tw);

    BPEMerges merges = {0};
    if (bpe_load(bpe_path, &merges) != 0) {
        fprintf(stderr, "volt: cannot load BPE model %s\n", bpe_path);
        kvdmap_free(wf); return 1;
    }
    fprintf(stderr, "Loaded %zu BPE merges from %s\n", merges.n, bpe_path);

    fprintf(stderr, "Computing frequencies...\n");
    KVDMap *tf = token_freqs_from_corpus(wf, &merges);
    KVDMap *cf = char_freqs_from_corpus(wf);
    fprintf(stderr, "  %zu token types,  %zu char types\n", tf->n, cf->n);

    /* Sort tokens by frequency descending → candidate list L */
    size_t nt = tf->n;
    KVD *L = (KVD *)malloc(nt * sizeof(KVD));
    size_t li = 0;
    for (size_t i = 0; i < tf->cap; i++)
        if (tf->data[i].key) L[li++] = tf->data[i];
    qsort(L, nt, sizeof(KVD), cmp_kv_desc);

    /* Build char array + index */
    size_t nc = cf->n;
    char **chars_arr = (char **)malloc(nc * sizeof(char *));
    double *p_char = (double *)malloc(nc * sizeof(double));
    KVZMap *cidx = kvzmap_new();
    double tc = 0; size_t ci = 0;
    for (size_t i = 0; i < cf->cap; i++)
        if (cf->data[i].key) tc += cf->data[i].val;
    for (size_t i = 0; i < cf->cap; i++) {
        if (!cf->data[i].key) continue;
        chars_arr[ci] = cf->data[i].key;       /* borrowed */
        p_char[ci] = cf->data[i].val / (tc > 0 ? tc : 1.0);
        kvzmap_set(cidx, chars_arr[ci], ci);
        ci++;
    }

    /* Clamp S to available tokens */
    while (S_len > 0 && S[S_len - 1] > (int)nt) S_len--;
    if (S_len == 0) {
        fprintf(stderr, "volt: S sequence exceeds available tokens (%zu)\n", nt);
        free(L); free(chars_arr); free(p_char); kvzmap_free(cidx);
        kvdmap_free(tf); kvdmap_free(cf); kvdmap_free(wf); bpe_free(&merges);
        return 1;
    }

    fprintf(stderr, "Running VOLT: S=%d..%d (%d steps)\n",
            S[0], S[S_len - 1], S_len);

    VStep *steps = (VStep *)calloc((size_t)S_len, sizeof(VStep));
    int nv_steps = 0;

    for (int si = 0; si < S_len; si++) {
        int sz = S[si];
        if (sz <= 0 || (size_t)sz > nt) continue;
        size_t n_tok = (size_t)sz;

        /* Token array and distribution for this step */
        char **T = (char **)malloc(n_tok * sizeof(char *));
        double *p_tok = (double *)malloc(n_tok * sizeof(double));
        double st = 0;
        for (size_t i = 0; i < n_tok; i++) { T[i] = L[i].key; st += L[i].val; }
        for (size_t i = 0; i < n_tok; i++) p_tok[i] = L[i].val / (st > 0 ? st : 1.0);

        SRow *K = build_kernel(T, n_tok, cidx);
        double *row_sums = (double *)calloc(n_tok, sizeof(double));
        int gpu_used = 0;
#ifdef DM_GPU
        if (gpu_ctx && dm_gpu_ready((DmGpuCtx *)gpu_ctx))
            gpu_used = sinkhorn_gpu((DmGpuCtx *)gpu_ctx, K, n_tok, nc,
                                    p_tok, p_char, sink_iters, (float)sink_tol,
                                    row_sums) == 0;
#endif
        if (!gpu_used)
            sinkhorn(K, n_tok, nc, p_tok, p_char, sink_iters, sink_tol, row_sums);

        /* Extract vocabulary */
        size_t nv = 0;
        char **vocab = (char **)malloc(n_tok * sizeof(char *));
        for (size_t i = 0; i < n_tok; i++)
            if (row_sums[i] >= threshold * p_tok[i]) vocab[nv++] = T[i];

        double H = compute_entropy(vocab, nv, tf);
        steps[nv_steps].size = sz;
        steps[nv_steps].entropy = H;
        steps[nv_steps].nv = nv;
        steps[nv_steps].vocab = (char **)malloc(nv * sizeof(char *));
        memcpy(steps[nv_steps].vocab, vocab, nv * sizeof(char *)); /* borrowed ptrs */
        nv_steps++;

        fprintf(stderr, "  t=%6d: |vocab|=%5zu  H=%.6f\n", sz, nv, H);

        free(vocab); free(row_sums); kernel_free(K, n_tok); free(p_tok); free(T);
    }

    if (nv_steps == 0) {
        fprintf(stderr, "volt: no valid results\n");
        free(L); free(chars_arr); free(p_char); kvzmap_free(cidx);
        for (int t = 0; t < nv_steps; t++) free(steps[t].vocab);
        free(steps);
        kvdmap_free(tf); kvdmap_free(cf); kvdmap_free(wf); bpe_free(&merges);
        return 1;
    }

    /* MUV selection (Eq. 3) */
    double best_muv = -1e300; int best_t = nv_steps - 1;
    for (int t = 1; t < nv_steps; t++) {
        int step = steps[t].size - steps[t - 1].size;
        if (step <= 0) continue;
        double muv = -(steps[t].entropy - steps[t - 1].entropy) / (double)step;
        if (muv > best_muv) { best_muv = muv; best_t = t; }
    }

    fprintf(stderr, "\nOptimal: S=%d  |vocab|=%zu  MUV=%.8g  H=%.6f\n",
            steps[best_t].size, steps[best_t].nv, best_muv, steps[best_t].entropy);

    write_result(output, &steps[best_t], best_muv, steps, nv_steps);
    fprintf(stderr, "Written to %s\n", output);

    if (stats) {
        fprintf(stderr,
            "{\"vocab_size\":%zu,\"optimal_S\":%d,\"muv\":%.8g,"
            "\"entropy\":%.8g,\"num_chars\":%zu,\"num_token_candidates\":%zu}\n",
            steps[best_t].nv, steps[best_t].size, best_muv,
            steps[best_t].entropy, nc, nt);
    }

    for (int t = 0; t < nv_steps; t++) free(steps[t].vocab);
    free(steps); free(L); free(chars_arr); free(p_char);
    kvzmap_free(cidx); kvdmap_free(tf); kvdmap_free(cf); kvdmap_free(wf);
    bpe_free(&merges);
    return 0;
}

/* -------------------------------------------------------------------------
 * CLI entry point
 * ---------------------------------------------------------------------- */

int dm_volt_cli(int argc, char **argv) {
    int start = 1;
    if (argc >= 2 && (strcmp(argv[1], "volt") == 0 ||
                      strcmp(argv[1], "dm_volt") == 0)) start = 2;
    if (start >= argc) goto usage;

    const char *corpus = NULL, *bpe_model = NULL, *output = NULL;
    int S_min = 0, S_max = 0, S_step = 0;
    double threshold = 1e-3;
    int sink_iters = 1000;
    double sink_tol = 1e-9;
    size_t max_lines = 0;
    int stats_flag = 0;
    int use_gpu = 0, gpu_device = 0;

    for (int i = start; i < argc; i++) {
        if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--input"))
            corpus = argv[++i];
        else if (!strcmp(argv[i], "--bpe-model"))
            bpe_model = argv[++i];
        else if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--output"))
            output = argv[++i];
        else if (!strcmp(argv[i], "--S-min"))
            S_min = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--S-max"))
            S_max = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--S-step"))
            S_step = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threshold"))
            threshold = atof(argv[++i]);
        else if (!strcmp(argv[i], "--sinkhorn-iters"))
            sink_iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sinkhorn-tol"))
            sink_tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "--max-lines"))
            max_lines = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stats"))
            stats_flag = 1;
        else if (!strcmp(argv[i], "--gpu"))
            use_gpu = 1;
        else if (!strcmp(argv[i], "--gpu-device") && i + 1 < argc)
            { gpu_device = atoi(argv[++i]); use_gpu = 1; }
        else { fprintf(stderr, "volt: unknown option: %s\n", argv[i]); goto usage; }
    }

    if (!corpus || !bpe_model || !output) goto usage;

    if (S_step  <= 0) S_step  = 500;
    if (S_max   <= 0) S_max   = 10000;
    if (S_min   <= 0) S_min   = S_step;
    int S_len = (S_max - S_min) / S_step + 1;
    int *S_seq = (int *)malloc((size_t)S_len * sizeof(int));
    for (int i = 0; i < S_len; i++) S_seq[i] = S_min + i * S_step;

    void *gpu_ctx = NULL;
#ifdef DM_GPU
    if (use_gpu) {
        gpu_ctx = dm_gpu_create(gpu_device, NULL);
        if (!gpu_ctx || !dm_gpu_ready((DmGpuCtx *)gpu_ctx)) {
            fprintf(stderr, "[volt] GPU init failed, falling back to CPU\n");
            dm_gpu_destroy((DmGpuCtx *)gpu_ctx); gpu_ctx = NULL;
        } else { char _dname[256]={0}; dm_gpu_device_name((DmGpuCtx*)gpu_ctx,_dname,sizeof(_dname)); fprintf(stderr,"[volt] GPU: %s\n",_dname); }
    }
#else
    if (use_gpu) fprintf(stderr, "[volt] built without GPU support, using CPU\n");
#endif

    int rc = volt_run(corpus, bpe_model, output, S_seq, S_len,
                      threshold, sink_iters, sink_tol, max_lines, stats_flag, gpu_ctx);
    free(S_seq);
#ifdef DM_GPU
    dm_gpu_destroy((DmGpuCtx *)gpu_ctx);
#endif
    return rc;

usage:
    fprintf(stderr,
        "Usage: volt -i corpus.txt --bpe-model bpe.txt -o vocab.json\n"
        "       [--S-min N] [--S-max N] [--S-step N]\n"
        "       [--threshold F] [--sinkhorn-iters N] [--sinkhorn-tol F]\n"
        "       [--max-lines N] [--stats] [--gpu] [--gpu-device N]\n\n"
        "VOLT: Vocabulary Learning via Optimal Transport (ACL 2021)\n"
        "Requires a BPE model file (from: dm --train --algo bpe).\n"
        "Example:\n"
        "  dm --train --algo bpe -i corpus.txt -m 10000 -o bpe.txt\n"
        "  volt -i corpus.txt --bpe-model bpe.txt --S-max 5000 -o vocab.json\n");
    return 2;
}
