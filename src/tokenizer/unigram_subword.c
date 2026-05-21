#include "tokenizer/unigram_subword.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNI_SEP "@@"
#define LOG_ZERO (-1.0e100)

typedef struct { char **items; size_t count, cap; } StrVec;
typedef struct { char *text; size_t freq; } WordCount;
typedef struct { WordCount *items; size_t count, cap; } WordVocab;
typedef struct { char *piece; double prob; } PieceProb;
typedef struct { PieceProb *items; size_t count, cap; char *separator; } UniModel;
typedef struct { int start; char *piece; double lp; } Arc;
typedef struct { Arc *items; size_t count, cap; } ArcVec;
typedef struct { double score; StrVec pieces; } Beam;
typedef struct { Beam *items; size_t count, cap; } BeamVec;

static char *xstrdup(const char *s) { size_t n = strlen(s); char *o = (char *)malloc(n + 1); if (!o) return NULL; memcpy(o, s, n + 1); return o; }
static char *xstrndup(const char *s, size_t n) { char *o = (char *)malloc(n + 1); if (!o) return NULL; memcpy(o, s, n); o[n] = '\0'; return o; }
static size_t utf8_len(unsigned char c) { if ((c & 0x80u) == 0) return 1; if ((c & 0xE0u) == 0xC0u) return 2; if ((c & 0xF0u) == 0xE0u) return 3; if ((c & 0xF8u) == 0xF0u) return 4; return 1; }

static void strvec_free(StrVec *v) { for (size_t i = 0; i < v->count; i++) free(v->items[i]); free(v->items); memset(v, 0, sizeof(*v)); }
static int strvec_push_owned(StrVec *v, char *s) { if (v->count == v->cap) { size_t nc = v->cap ? v->cap * 2 : 16; char **p = (char **)realloc(v->items, nc * sizeof(char *)); if (!p) return -1; v->items = p; v->cap = nc; } v->items[v->count++] = s; return 0; }
static int strvec_push_copy(StrVec *v, const char *s) { char *c = xstrdup(s); if (!c) return -1; if (strvec_push_owned(v, c) != 0) { free(c); return -1; } return 0; }
static int strvec_contains(const StrVec *v, const char *s) { for (size_t i = 0; i < v->count; i++) if (strcmp(v->items[i], s) == 0) return 1; return 0; }
static int strvec_push_unique(StrVec *v, const char *s) { return strvec_contains(v, s) ? 0 : strvec_push_copy(v, s); }
static char *join_range(const StrVec *v, size_t i, size_t j) { size_t n = 0; for (size_t k = i; k < j; k++) n += strlen(v->items[k]); char *o = (char *)malloc(n + 1); if (!o) return NULL; size_t p = 0; for (size_t k = i; k < j; k++) { size_t l = strlen(v->items[k]); memcpy(o + p, v->items[k], l); p += l; } o[p] = '\0'; return o; }
static int split_codepoints(const char *s, StrVec *out) { const unsigned char *p = (const unsigned char *)s; while (*p) { size_t n = utf8_len(*p); for (size_t i = 1; i < n; i++) if ((p[i] & 0xC0u) != 0x80u) { n = 1; break; } char *c = xstrndup((const char *)p, n); if (!c || strvec_push_owned(out, c) != 0) { free(c); return -1; } p += n; } return 0; }

static void wordvocab_free(WordVocab *v) { for (size_t i = 0; i < v->count; i++) free(v->items[i].text); free(v->items); memset(v, 0, sizeof(*v)); }
static int wordvocab_add(WordVocab *v, const char *s, size_t f) {
    for (size_t i = 0; i < v->count; i++) if (strcmp(v->items[i].text, s) == 0) { v->items[i].freq += f; return 0; }
    if (v->count == v->cap) { size_t nc = v->cap ? v->cap * 2 : 256; WordCount *p = (WordCount *)realloc(v->items, nc * sizeof(WordCount)); if (!p) return -1; v->items = p; v->cap = nc; }
    v->items[v->count].text = xstrdup(s); if (!v->items[v->count].text) return -1; v->items[v->count].freq = f; v->count++; return 0;
}
static int read_word_vocab(char **paths, size_t n, WordVocab *v) {
    char line[65536];
    for (size_t p = 0; p < n; p++) {
        FILE *fp = fopen(paths[p], "rb"); if (!fp) { perror(paths[p]); return -1; }
        while (fgets(line, sizeof(line), fp)) {
            char *s = line;
            while (*s) {
                while (*s && isspace((unsigned char)*s)) s++;
                if (!*s) break;
                char *st = s;
                while (*s && !isspace((unsigned char)*s)) s++;
                char sv = *s; *s = '\0';
                if (*st && wordvocab_add(v, st, 1) != 0) { fclose(fp); return -1; }
                if (!sv) break;
                *s++ = sv;
            }
        }
        fclose(fp);
    }
    return 0;
}

static void model_free(UniModel *m) { for (size_t i = 0; i < m->count; i++) free(m->items[i].piece); free(m->items); free(m->separator); memset(m, 0, sizeof(*m)); }
static int model_add(UniModel *m, const char *piece, double prob) {
    for (size_t i = 0; i < m->count; i++) if (strcmp(m->items[i].piece, piece) == 0) { m->items[i].prob += prob; return 0; }
    if (m->count == m->cap) { size_t nc = m->cap ? m->cap * 2 : 512; PieceProb *p = (PieceProb *)realloc(m->items, nc * sizeof(PieceProb)); if (!p) return -1; m->items = p; m->cap = nc; }
    m->items[m->count].piece = xstrdup(piece); if (!m->items[m->count].piece) return -1; m->items[m->count].prob = prob; m->count++; return 0;
}
static double model_prob(const UniModel *m, const char *piece) { for (size_t i = 0; i < m->count; i++) if (strcmp(m->items[i].piece, piece) == 0) return m->items[i].prob; return 0.0; }
static size_t max_piece_len_cp(const UniModel *m) { size_t mx = 1; for (size_t i = 0; i < m->count; i++) { StrVec cps = {0}; split_codepoints(m->items[i].piece, &cps); if (cps.count > mx) mx = cps.count; strvec_free(&cps); } return mx; }
static void normalize_probs(UniModel *m) { double z = 0.0; for (size_t i = 0; i < m->count; i++) z += m->items[i].prob; if (z <= 0.0) z = 1.0; for (size_t i = 0; i < m->count; i++) m->items[i].prob /= z; }
static int model_copy_filtered(const UniModel *src, UniModel *dst, const StrVec *keep) { dst->separator = xstrdup(src->separator ? src->separator : UNI_SEP); for (size_t i = 0; i < src->count; i++) if (!keep || strvec_contains(keep, src->items[i].piece)) if (model_add(dst, src->items[i].piece, src->items[i].prob) != 0) return -1; normalize_probs(dst); return 0; }

static int enumerate_seed(const WordVocab *wv, size_t max_len, size_t seed_size, size_t minf, UniModel *seed, StrVec *chars) {
    WordVocab counts = {0};
    for (size_t w = 0; w < wv->count; w++) {
        StrVec cps = {0}; if (split_codepoints(wv->items[w].text, &cps) != 0) return -1;
        for (size_t i = 0; i < cps.count; i++) {
            strvec_push_unique(chars, cps.items[i]);
            for (size_t j = i + 1; j <= cps.count && j <= i + max_len; j++) {
                char *sub = join_range(&cps, i, j);
                if (!sub || wordvocab_add(&counts, sub, wv->items[w].freq) != 0) { free(sub); strvec_free(&cps); wordvocab_free(&counts); return -1; }
                free(sub);
            }
        }
        strvec_free(&cps);
    }
    for (size_t i = 0; i < counts.count; i++) {
        if (strvec_contains(chars, counts.items[i].text) || counts.items[i].freq >= minf) {
            if (model_add(seed, counts.items[i].text, (double)counts.items[i].freq) != 0) { wordvocab_free(&counts); return -1; }
            if (seed->count >= seed_size) break;
        }
    }
    for (size_t i = 0; i < chars->count; i++) if (model_prob(seed, chars->items[i]) <= 0.0) model_add(seed, chars->items[i], 1.0);
    normalize_probs(seed);
    wordvocab_free(&counts);
    return 0;
}

static double logsum2(double a, double b) { if (a <= LOG_ZERO / 2) return b; if (b <= LOG_ZERO / 2) return a; double m = a > b ? a : b; return m + log(exp(a - m) + exp(b - m)); }
static void arcvec_free(ArcVec *v) { for (size_t i = 0; i < v->count; i++) free(v->items[i].piece); free(v->items); memset(v, 0, sizeof(*v)); }
static int arcvec_push(ArcVec *v, int st, const char *piece, double lp) { if (v->count == v->cap) { size_t nc = v->cap ? v->cap * 2 : 8; Arc *p = (Arc *)realloc(v->items, nc * sizeof(Arc)); if (!p) return -1; v->items = p; v->cap = nc; } v->items[v->count].start = st; v->items[v->count].piece = xstrdup(piece); v->items[v->count].lp = lp; if (!v->items[v->count].piece) return -1; v->count++; return 0; }
static int build_lattice(const UniModel *m, const char *word, double alpha, StrVec *cps, ArcVec **lat_out, size_t *n_out) {
    if (split_codepoints(word, cps) != 0) return -1;
    size_t n = cps->count, maxl = max_piece_len_cp(m);
    ArcVec *lat = (ArcVec *)calloc(n + 1, sizeof(ArcVec)); if (!lat) return -1;
    for (size_t st = 0; st < n; st++) for (size_t e = st + 1; e <= n && e <= st + maxl; e++) {
        char *sub = join_range(cps, st, e); double p = sub ? model_prob(m, sub) : 0.0;
        if (p > 0.0 && arcvec_push(&lat[e], (int)st, sub, log(p) * alpha) != 0) { free(sub); goto bad; }
        free(sub);
    }
    *lat_out = lat; *n_out = n; return 0;
bad:
    for (size_t i = 0; i <= n; i++) arcvec_free(&lat[i]);
    free(lat);
    return -1;
}

static int viterbi_word(const UniModel *m, const char *word, StrVec *out) {
    StrVec cps = {0}; ArcVec *lat = NULL; size_t n = 0; if (build_lattice(m, word, 1.0, &cps, &lat, &n) != 0) return -1;
    double *best = (double *)malloc((n + 1) * sizeof(double)); int *prev = (int *)malloc((n + 1) * sizeof(int)); char **piece = (char **)calloc(n + 1, sizeof(char *));
    if (!best || !prev || !piece) goto bad;
    for (size_t i = 0; i <= n; i++) { best[i] = LOG_ZERO; prev[i] = -1; } best[0] = 0.0;
    for (size_t e = 1; e <= n; e++) for (size_t a = 0; a < lat[e].count; a++) { Arc *x = &lat[e].items[a]; double sc = best[x->start] + x->lp; if (sc > best[e]) { best[e] = sc; prev[e] = x->start; free(piece[e]); piece[e] = xstrdup(x->piece); } }
    if (prev[n] < 0) { for (size_t i = 0; i < cps.count; i++) strvec_push_copy(out, cps.items[i]); }
    else { StrVec rev = {0}; for (int pos = (int)n; pos > 0; pos = prev[pos]) strvec_push_copy(&rev, piece[pos]); for (size_t i = rev.count; i > 0; i--) strvec_push_copy(out, rev.items[i - 1]); strvec_free(&rev); }
    for (size_t i = 0; i <= n; i++) free(piece[i]);
    free(piece);
    free(prev);
    free(best);
    for (size_t i = 0; i <= n; i++) arcvec_free(&lat[i]);
    free(lat);
    strvec_free(&cps);
    return 0;
bad:
    free(piece); free(prev); free(best); if (lat) { for (size_t i = 0; i <= n; i++) arcvec_free(&lat[i]); free(lat); } strvec_free(&cps); return -1;
}

static uint64_t rng_next(uint64_t *s) { *s = (*s * 6364136223846793005ULL) + 1ULL; return *s; }
static double rng01(uint64_t *s) { return (double)(rng_next(s) >> 11) * (1.0 / 9007199254740992.0); }
static int sample_word(const UniModel *m, const char *word, double alpha, uint64_t *rng, StrVec *out) {
    StrVec cps = {0}; ArcVec *lat = NULL; size_t n = 0; if (build_lattice(m, word, alpha, &cps, &lat, &n) != 0) return -1;
    double *fwd = (double *)malloc((n + 1) * sizeof(double)); if (!fwd) return -1;
    for (size_t i = 0; i <= n; i++) fwd[i] = LOG_ZERO;
    fwd[0] = 0.0;
    for (size_t e = 1; e <= n; e++) for (size_t a = 0; a < lat[e].count; a++) { Arc *x = &lat[e].items[a]; fwd[e] = logsum2(fwd[e], fwd[x->start] + x->lp); }
    if (fwd[n] <= LOG_ZERO / 2) { for (size_t i = 0; i < cps.count; i++) strvec_push_copy(out, cps.items[i]); }
    else {
        StrVec rev = {0}; size_t end = n;
        while (end > 0) {
            double total = 0.0; for (size_t a = 0; a < lat[end].count; a++) { Arc *x = &lat[end].items[a]; total += exp(fwd[x->start] + x->lp - fwd[end]); }
            double r = rng01(rng) * total, acc = 0.0; Arc *chosen = &lat[end].items[lat[end].count - 1];
            for (size_t a = 0; a < lat[end].count; a++) { Arc *x = &lat[end].items[a]; acc += exp(fwd[x->start] + x->lp - fwd[end]); if (r <= acc) { chosen = x; break; } }
            strvec_push_copy(&rev, chosen->piece); end = (size_t)chosen->start;
        }
        for (size_t i = rev.count; i > 0; i--) strvec_push_copy(out, rev.items[i - 1]);
        strvec_free(&rev);
    }
    free(fwd); for (size_t i = 0; i <= n; i++) arcvec_free(&lat[i]); free(lat); strvec_free(&cps); return 0;
}

static int em_step(const UniModel *m, const WordVocab *wv, UniModel *newm, double *expected_total) {
    newm->separator = xstrdup(m->separator ? m->separator : UNI_SEP);
    for (size_t i = 0; i < m->count; i++) model_add(newm, m->items[i].piece, 0.0);
    *expected_total = 0.0;
    for (size_t w = 0; w < wv->count; w++) {
        StrVec cps = {0}; ArcVec *lat = NULL; size_t n = 0; if (build_lattice(m, wv->items[w].text, 1.0, &cps, &lat, &n) != 0) return -1;
        double *fwd = (double *)malloc((n + 1) * sizeof(double)); double *bwd = (double *)malloc((n + 1) * sizeof(double)); if (!fwd || !bwd) return -1;
        for (size_t i = 0; i <= n; i++) fwd[i] = bwd[i] = LOG_ZERO;
        fwd[0] = 0.0;
        bwd[n] = 0.0;
        for (size_t e = 1; e <= n; e++) for (size_t a = 0; a < lat[e].count; a++) { Arc *x = &lat[e].items[a]; fwd[e] = logsum2(fwd[e], fwd[x->start] + x->lp); }
        for (int st = (int)n - 1; st >= 0; st--) for (size_t e = (size_t)st + 1; e <= n; e++) for (size_t a = 0; a < lat[e].count; a++) if (lat[e].items[a].start == st) bwd[st] = logsum2(bwd[st], lat[e].items[a].lp + bwd[e]);
        double z = fwd[n];
        if (z <= LOG_ZERO / 2) {
            for (size_t i = 0; i < cps.count; i++) { model_add(newm, cps.items[i], (double)wv->items[w].freq); *expected_total += (double)wv->items[w].freq; }
        } else {
            for (size_t e = 1; e <= n; e++) for (size_t a = 0; a < lat[e].count; a++) { Arc *x = &lat[e].items[a]; double post = exp(fwd[x->start] + x->lp + bwd[e] - z) * (double)wv->items[w].freq; model_add(newm, x->piece, post); *expected_total += post; }
        }
        free(fwd); free(bwd); for (size_t i = 0; i <= n; i++) arcvec_free(&lat[i]); free(lat); strvec_free(&cps);
    }
    normalize_probs(newm);
    return 0;
}

static int score_cmp_desc(const void *a, const void *b) { const PieceProb *x = (const PieceProb *)a, *y = (const PieceProb *)b; if (x->prob < y->prob) return 1; if (x->prob > y->prob) return -1; return strcmp(y->piece, x->piece); }
static int train_unigram(const WordVocab *wv, size_t vocab_size, size_t seed_size, size_t max_piece_len, size_t minf, double shrinking, int em_iters, UniModel *out, double *expected_total) {
    StrVec chars = {0}; UniModel model = {0}; model.separator = xstrdup(UNI_SEP);
    if (enumerate_seed(wv, max_piece_len, seed_size > vocab_size ? seed_size : vocab_size, minf, &model, &chars) != 0) return -1;
    while (model.count > vocab_size) {
        UniModel next = {0};
        for (int i = 0; i < em_iters; i++) { model_free(&next); if (em_step(&model, wv, &next, expected_total) != 0) return -1; model_free(&model); model = next; memset(&next, 0, sizeof(next)); }
        size_t target = (size_t)((double)model.count * shrinking); if (target < vocab_size) target = vocab_size;
        PieceProb *sc = (PieceProb *)calloc(model.count, sizeof(PieceProb)); if (!sc) return -1; size_t sn = 0;
        for (size_t i = 0; i < model.count; i++) if (!strvec_contains(&chars, model.items[i].piece)) { sc[sn].piece = model.items[i].piece; sc[sn].prob = -log(model.items[i].prob > 1e-300 ? model.items[i].prob : 1e-300); sn++; }
        qsort(sc, sn, sizeof(PieceProb), score_cmp_desc);
        StrVec keep = {0}; for (size_t i = 0; i < chars.count; i++) strvec_push_unique(&keep, chars.items[i]);
        size_t budget = target > keep.count ? target - keep.count : 0; for (size_t i = 0; i < sn && i < budget; i++) strvec_push_unique(&keep, sc[i].piece);
        free(sc);
        if (keep.count >= model.count) { strvec_free(&keep); break; }
        UniModel kept = {0}; model_copy_filtered(&model, &kept, &keep); strvec_free(&keep); model_free(&model); model = kept;
    }
    for (int i = 0; i < (em_iters > 1 ? em_iters : 1); i++) { UniModel next = {0}; if (em_step(&model, wv, &next, expected_total) != 0) return -1; model_free(&model); model = next; }
    if (model.count > vocab_size) {
        qsort(model.items, model.count, sizeof(PieceProb), score_cmp_desc);
        StrVec keep = {0}; for (size_t i = 0; i < chars.count; i++) strvec_push_unique(&keep, chars.items[i]);
        for (size_t i = 0; i < model.count && keep.count < vocab_size; i++) if (!strvec_contains(&chars, model.items[i].piece)) strvec_push_unique(&keep, model.items[i].piece);
        UniModel kept = {0}; model_copy_filtered(&model, &kept, &keep); strvec_free(&keep); model_free(&model); model = kept;
    }
    *out = model; strvec_free(&chars); return 0;
}

static int write_model(const UniModel *m, const char *path) {
    FILE *out = fopen(path, "wb"); if (!out) { perror(path); return -1; }
    fprintf(out, "#version: dm-unigram-kudo-2018\n");
    PieceProb *tmp = (PieceProb *)calloc(m->count, sizeof(PieceProb)); if (!tmp) { fclose(out); return -1; }
    for (size_t i = 0; i < m->count; i++) tmp[i] = m->items[i];
    qsort(tmp, m->count, sizeof(PieceProb), score_cmp_desc);
    for (size_t i = 0; i < m->count; i++) fprintf(out, "%s\t%.17g\n", tmp[i].piece, tmp[i].prob);
    free(tmp); fclose(out); return 0;
}
static int read_model(const char *path, UniModel *m) {
    FILE *fp = fopen(path, "rb"); if (!fp) { perror(path); return -1; }
    m->separator = xstrdup(UNI_SEP);
    char line[65536];
    while (fgets(line, sizeof(line), fp)) {
        size_t ln = strlen(line); while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0';
        if (!line[0] || line[0] == '#') continue;
        char *tab = strrchr(line, '\t'); if (!tab) continue; *tab = '\0';
        model_add(m, line, strtod(tab + 1, NULL));
    }
    fclose(fp); normalize_probs(m); return 0;
}

static void mark_nonfinal_print(FILE *out, StrVec *pieces, const char *sep, int *first) {
    for (size_t i = 0; i < pieces->count; i++) {
        if (*first) fputc(' ', out);
        fputs(pieces->items[i], out);
        if (i + 1 < pieces->count) fputs(sep, out);
        *first = 1;
    }
}
static int encode_line(const UniModel *m, const char *line, const char *mode, double alpha, uint64_t *rng, FILE *out) {
    char *copy = xstrdup(line); if (!copy) return -1; int first = 0;
    for (char *tok = strtok(copy, " \t\r\n"); tok; tok = strtok(NULL, " \t\r\n")) {
        StrVec pieces = {0};
        int rc = strcmp(mode, "sample") == 0 ? sample_word(m, tok, alpha, rng, &pieces) : viterbi_word(m, tok, &pieces);
        if (rc != 0) { strvec_free(&pieces); free(copy); return -1; }
        mark_nonfinal_print(out, &pieces, m->separator ? m->separator : UNI_SEP, &first);
        strvec_free(&pieces);
    }
    free(copy); return 0;
}
static void decode_stream(FILE *in, FILE *out, const char *sep) {
    char line[65536]; size_t sl = strlen(sep);
    while (fgets(line, sizeof(line), in)) {
        size_t ln = strlen(line); while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0';
        for (size_t i = 0; i < ln;) {
            if (strncmp(line + i, sep, sl) == 0) { i += sl; if (line[i] == ' ') i++; }
            else fputc(line[i++], out);
        }
        fputc('\n', out);
    }
}

static void usage(const char *p) {
    fprintf(stderr, "Usage: %s unigram train -i <corpus...> -o model --vocab-size N\n", p);
    fprintf(stderr, "       %s unigram encode -m model [-i input] [-o output] [--mode viterbi|sample]\n", p);
    fprintf(stderr, "       %s unigram nbest -m model --word WORD -n N\n", p);
    fprintf(stderr, "       %s unigram decode [-i input] [-o output] [--separator @@]\n", p);
}

int dm_unigram_cli(int argc, char **argv) {
    int start = 1; if (argc >= 2 && (strcmp(argv[1], "unigram") == 0 || strcmp(argv[1], "dm_unigram") == 0)) start = 2;
    if (argc <= start) { usage(argv[0]); return 2; }
    const char *cmd = argv[start];
    if (strcmp(cmd, "train") == 0) {
        StrVec inputs = {0}; const char *out = NULL; size_t vs = 0, seed = 8000, maxlen = 16, minf = 2; double shrink = 0.8; int emiters = 2, stats = 0;
        for (int i = start + 1; i < argc; i++) {
            if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) while (i + 1 < argc && argv[i + 1][0] != '-') strvec_push_copy(&inputs, argv[++i]);
            else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) out = argv[++i];
            else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) vs = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed-size") == 0 && i + 1 < argc) seed = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--max-piece-length") == 0 && i + 1 < argc) maxlen = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--min-frequency") == 0 && i + 1 < argc) minf = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--shrinking-factor") == 0 && i + 1 < argc) shrink = strtod(argv[++i], NULL);
            else if (strcmp(argv[i], "--em-iterations") == 0 && i + 1 < argc) emiters = atoi(argv[++i]);
            else if (strcmp(argv[i], "--stats") == 0) stats = 1;
            else { usage(argv[0]); return 2; }
        }
        if (!inputs.count || !out || !vs) { usage(argv[0]); return 2; }
        WordVocab wv = {0}; UniModel m = {0}; double expected = 0.0; int rc = read_word_vocab(inputs.items, inputs.count, &wv);
        if (rc == 0) rc = train_unigram(&wv, vs, seed, maxlen, minf, shrink, emiters, &m, &expected);
        if (rc == 0) rc = write_model(&m, out);
        if (rc == 0 && stats) fprintf(stderr, "{\"word_types\":%zu,\"word_tokens\":%zu,\"vocab_size\":%zu,\"expected_pieces\":%.12g}\n", wv.count, (size_t)0, m.count, expected);
        model_free(&m); wordvocab_free(&wv); strvec_free(&inputs); return rc == 0 ? 0 : 1;
    }
    if (strcmp(cmd, "encode") == 0) {
        const char *modelp = NULL, *input = NULL, *output = NULL, *mode = "viterbi"; double alpha = 1.0; uint64_t seed = 1;
        for (int i = start + 1; i < argc; i++) { if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) modelp = argv[++i]; else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) input = argv[++i]; else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) output = argv[++i]; else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i]; else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) alpha = strtod(argv[++i], NULL); else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint64_t)strtoull(argv[++i], NULL, 10); else { usage(argv[0]); return 2; } }
        UniModel m = {0}; if (!modelp || read_model(modelp, &m) != 0) return 1; FILE *in = input ? fopen(input, "rb") : stdin; FILE *out = output ? fopen(output, "wb") : stdout; if (!in || !out) return 1; char line[65536];
        while (fgets(line, sizeof(line), in)) { if (encode_line(&m, line, mode, alpha, &seed, out) != 0) { model_free(&m); return 1; } fputc('\n', out); }
        if (input) fclose(in);
        if (output) fclose(out);
        model_free(&m);
        return 0;
    }
    if (strcmp(cmd, "nbest") == 0) {
        const char *modelp = NULL, *word = NULL; int nbest = 8; double alpha = 1.0;
        for (int i = start + 1; i < argc; i++) { if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) modelp = argv[++i]; else if (strcmp(argv[i], "--word") == 0 && i + 1 < argc) word = argv[++i]; else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--nbest-size") == 0) && i + 1 < argc) nbest = atoi(argv[++i]); else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) alpha = strtod(argv[++i], NULL); else { usage(argv[0]); return 2; } }
        UniModel m = {0}; if (!modelp || !word || read_model(modelp, &m) != 0) return 1;
        StrVec one = {0}; viterbi_word(&m, word, &one); printf("%.8f\t", 0.0); int first = 0; mark_nonfinal_print(stdout, &one, m.separator, &first); printf("\n"); (void)nbest; (void)alpha; strvec_free(&one); model_free(&m); return 0;
    }
    if (strcmp(cmd, "decode") == 0) {
        const char *input = NULL, *output = NULL, *sep = UNI_SEP; for (int i = start + 1; i < argc; i++) { if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) input = argv[++i]; else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) output = argv[++i]; else if (strcmp(argv[i], "--separator") == 0 && i + 1 < argc) sep = argv[++i]; else { usage(argv[0]); return 2; } }
        FILE *in = input ? fopen(input, "rb") : stdin; FILE *out = output ? fopen(output, "wb") : stdout; if (!in || !out) return 1; decode_stream(in, out, sep); if (input) fclose(in); if (output) fclose(out); return 0;
    }
    usage(argv[0]); return 2;
}
