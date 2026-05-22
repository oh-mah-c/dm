#include "tokenizer/sentencepiece_lite.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DM_GPU
#include "gpu/dm_gpu.h"
#endif
#ifndef DM_NO_ICU
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#endif

#define SP_SPACE "\xE2\x96\x81"
#define SP_UNK "<unk>"
#define SP_BOS "<s>"
#define SP_EOS "</s>"
#define SP_PAD "<pad>"
#define LOG_ZERO (-1.0e100)

typedef struct { char **items; size_t count, cap; } StrVec;
typedef struct { char *text; size_t freq; } TextCount;
typedef struct { TextCount *items; size_t count, cap; } TextVocab;
typedef struct { char *a, *b; size_t freq; } PairStat;
typedef struct { PairStat *items; size_t count, cap; } PairStats;
typedef struct { char **syms; size_t len, cap; size_t freq; } SymSeq;
typedef struct { SymSeq *items; size_t count, cap; } SymCorpus;
typedef struct { char *src, *dst; } Rule;
typedef struct { Rule *items; size_t count, cap; } Rules;
typedef struct { char *a, *b; } Merge;
typedef struct { Merge *items; size_t count, cap; } Merges;
typedef struct { char *piece; double prob; } UniPiece;
typedef struct { UniPiece *items; size_t count, cap; } UniTable;
typedef struct {
    char *model_type;
    StrVec vocab;
    char *normalization;
    Rules rules;
    int add_dummy_prefix;
    char *unk_token, *bos_token, *eos_token, *pad_token;
    Merges merges;
    UniTable uni;
} SPModel;

static char *xstrdup(const char *s) { size_t n = strlen(s); char *o = (char *)malloc(n + 1); if (!o) return NULL; memcpy(o, s, n + 1); return o; }
static char *xstrndup(const char *s, size_t n) { char *o = (char *)malloc(n + 1); if (!o) return NULL; memcpy(o, s, n); o[n] = '\0'; return o; }
static char *concat2(const char *a, const char *b) { size_t na = strlen(a), nb = strlen(b); char *o = (char *)malloc(na + nb + 1); if (!o) return NULL; memcpy(o, a, na); memcpy(o + na, b, nb + 1); return o; }
static size_t utf8_len(unsigned char c) { if ((c & 0x80u) == 0) return 1; if ((c & 0xE0u) == 0xC0u) return 2; if ((c & 0xF0u) == 0xE0u) return 3; if ((c & 0xF8u) == 0xF0u) return 4; return 1; }
static void json_string(FILE *out, const char *s) { fputc('"', out); for (; *s; s++) { unsigned char c = (unsigned char)*s; if (c == '"' || c == '\\') { fputc('\\', out); fputc(c, out); } else if (c == '\n') fputs("\\n", out); else if (c == '\r') fputs("\\r", out); else if (c == '\t') fputs("\\t", out); else if (c < 32) fprintf(out, "\\u%04x", c); else fputc(c, out); } fputc('"', out); }

static void strvec_free(StrVec *v) { for (size_t i = 0; i < v->count; i++) free(v->items[i]); free(v->items); memset(v, 0, sizeof(*v)); }
static int strvec_push_owned(StrVec *v, char *s) { if (v->count == v->cap) { size_t nc = v->cap ? v->cap * 2 : 16; char **p = (char **)realloc(v->items, nc * sizeof(char *)); if (!p) return -1; v->items = p; v->cap = nc; } v->items[v->count++] = s; return 0; }
static int strvec_push_copy(StrVec *v, const char *s) { char *c = xstrdup(s); if (!c) return -1; if (strvec_push_owned(v, c) != 0) { free(c); return -1; } return 0; }
static int strvec_contains(const StrVec *v, const char *s) { for (size_t i = 0; i < v->count; i++) if (strcmp(v->items[i], s) == 0) return 1; return 0; }
static int strvec_push_unique(StrVec *v, const char *s) { return strvec_contains(v, s) ? 0 : strvec_push_copy(v, s); }

static void textvocab_free(TextVocab *v) { for (size_t i = 0; i < v->count; i++) free(v->items[i].text); free(v->items); memset(v, 0, sizeof(*v)); }
static int textvocab_add(TextVocab *v, const char *s, size_t f) {
    for (size_t i = 0; i < v->count; i++) if (strcmp(v->items[i].text, s) == 0) { v->items[i].freq += f; return 0; }
    if (v->count == v->cap) { size_t nc = v->cap ? v->cap * 2 : 128; TextCount *p = (TextCount *)realloc(v->items, nc * sizeof(TextCount)); if (!p) return -1; v->items = p; v->cap = nc; }
    v->items[v->count].text = xstrdup(s); if (!v->items[v->count].text) return -1; v->items[v->count].freq = f; v->count++; return 0;
}

static int split_codepoints(const char *s, StrVec *out) {
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        size_t n = utf8_len(*p);
        for (size_t i = 1; i < n; i++) if ((p[i] & 0xC0u) != 0x80u) { n = 1; break; }
        char *c = xstrndup((const char *)p, n);
        if (!c || strvec_push_owned(out, c) != 0) { free(c); return -1; }
        p += n;
    }
    return 0;
}
static char *join_range(StrVec *v, size_t i, size_t j) { size_t n = 0; for (size_t k = i; k < j; k++) n += strlen(v->items[k]); char *o = (char *)malloc(n + 1); if (!o) return NULL; size_t p = 0; for (size_t k = i; k < j; k++) { size_t l = strlen(v->items[k]); memcpy(o + p, v->items[k], l); p += l; } o[p] = '\0'; return o; }

static char *lower_ascii(const char *s) { char *o = xstrdup(s); if (!o) return NULL; for (char *p = o; *p; p++) *p = (char)tolower((unsigned char)*p); return o; }
static char *icu_normalize_utf8(const char *s, const char *name) {
#ifndef DM_NO_ICU
    UErrorCode status = U_ZERO_ERROR;
    const UNormalizer2 *norm = strcmp(name, "nfc") == 0 ? unorm2_getNFCInstance(&status) : unorm2_getNFKCInstance(&status);
    if (U_FAILURE(status)) return xstrdup(s);
    int32_t ulen = 0;
    u_strFromUTF8(NULL, 0, &ulen, s, -1, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) return xstrdup(s);
    status = U_ZERO_ERROR;
    UChar *ubuf = (UChar *)malloc((size_t)(ulen + 1) * sizeof(UChar));
    if (!ubuf) return NULL;
    u_strFromUTF8(ubuf, ulen + 1, NULL, s, -1, &status);
    if (U_FAILURE(status)) { free(ubuf); return xstrdup(s); }
    int32_t nlen = unorm2_normalize(norm, ubuf, ulen, NULL, 0, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) { free(ubuf); return xstrdup(s); }
    status = U_ZERO_ERROR;
    UChar *nbuf = (UChar *)malloc((size_t)(nlen + 1) * sizeof(UChar));
    if (!nbuf) { free(ubuf); return NULL; }
    unorm2_normalize(norm, ubuf, ulen, nbuf, nlen + 1, &status);
    free(ubuf);
    if (U_FAILURE(status)) { free(nbuf); return xstrdup(s); }
    int32_t outlen = 0;
    u_strToUTF8(NULL, 0, &outlen, nbuf, nlen, &status);
    if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) { free(nbuf); return xstrdup(s); }
    status = U_ZERO_ERROR;
    char *out = (char *)malloc((size_t)outlen + 1);
    if (!out) { free(nbuf); return NULL; }
    u_strToUTF8(out, outlen + 1, NULL, nbuf, nlen, &status);
    free(nbuf);
    if (U_FAILURE(status)) { free(out); return xstrdup(s); }
    return out;
#else
    (void)name;
    return xstrdup(s);
#endif
}
static char *apply_rules(const char *s, const Rules *rules) {
    StrVec out = {0}; size_t i = 0, n = strlen(s);
    while (i < n) {
        const Rule *match = NULL;
        for (size_t r = 0; r < rules->count; r++) if (strncmp(s + i, rules->items[r].src, strlen(rules->items[r].src)) == 0) { match = &rules->items[r]; break; }
        if (match) { if (strvec_push_copy(&out, match->dst) != 0) { strvec_free(&out); return NULL; } i += strlen(match->src); }
        else { size_t l = utf8_len((unsigned char)s[i]); if (strvec_push_owned(&out, xstrndup(s + i, l)) != 0) { strvec_free(&out); return NULL; } i += l; }
    }
    char *joined = join_range(&out, 0, out.count); strvec_free(&out); return joined;
}
static char *normalize_text(const SPModel *m, const char *s) {
    char *base;
    if (strcmp(m->normalization, "lower") == 0) base = lower_ascii(s);
    else if (strcmp(m->normalization, "nfc") == 0 || strcmp(m->normalization, "nfkc") == 0) base = icu_normalize_utf8(s, m->normalization);
    else base = xstrdup(s);
    if (!base) return NULL;
    if (!m->rules.count) return base;
    char *r = apply_rules(base, &m->rules); free(base); return r;
}
static char *escape_ws(const char *s, int add_dummy) {
    size_t n = strlen(s), space_len = strlen(SP_SPACE), outn = add_dummy ? space_len : 0;
    for (size_t i = 0; i < n; i++) outn += (s[i] == ' ') ? space_len : 1;
    char *o = (char *)malloc(outn + 1); if (!o) return NULL; size_t p = 0;
    if (add_dummy) { memcpy(o + p, SP_SPACE, space_len); p += space_len; }
    for (size_t i = 0; i < n; i++) if (s[i] == ' ') { memcpy(o + p, SP_SPACE, space_len); p += space_len; } else o[p++] = s[i];
    o[p] = '\0'; return o;
}
static char *unescape_ws(const char *s, int add_dummy) {
    size_t n = strlen(s), sl = strlen(SP_SPACE); char *o = (char *)malloc(n + 1); if (!o) return NULL; size_t p = 0;
    for (size_t i = 0; i < n;) {
        if (strncmp(s + i, SP_SPACE, sl) == 0) { o[p++] = ' '; i += sl; }
        else o[p++] = s[i++];
    }
    o[p] = '\0';
    if (add_dummy && o[0] == ' ') memmove(o, o + 1, strlen(o));
    return o;
}
static char *normalize_escape(const SPModel *m, const char *s) { char *n = normalize_text(m, s); if (!n) return NULL; char *e = escape_ws(n, m->add_dummy_prefix); free(n); return e; }

static void symseq_free(SymSeq *s) { for (size_t i = 0; i < s->len; i++) free(s->syms[i]); free(s->syms); memset(s, 0, sizeof(*s)); }
static int symseq_push_owned(SymSeq *s, char *x) { if (s->len == s->cap) { size_t nc = s->cap ? s->cap * 2 : 16; char **p = (char **)realloc(s->syms, nc * sizeof(char *)); if (!p) return -1; s->syms = p; s->cap = nc; } s->syms[s->len++] = x; return 0; }
static int symseq_push_copy(SymSeq *s, const char *x) { char *c = xstrdup(x); if (!c) return -1; if (symseq_push_owned(s, c) != 0) { free(c); return -1; } return 0; }
static void symcorpus_free(SymCorpus *c) { for (size_t i = 0; i < c->count; i++) symseq_free(&c->items[i]); free(c->items); memset(c, 0, sizeof(*c)); }
static int symcorpus_push_owned(SymCorpus *c, SymSeq s) { if (c->count == c->cap) { size_t nc = c->cap ? c->cap * 2 : 64; SymSeq *p = (SymSeq *)realloc(c->items, nc * sizeof(SymSeq)); if (!p) return -1; c->items = p; c->cap = nc; } c->items[c->count++] = s; return 0; }
static int symseq_from_text(const char *s, size_t freq, SymSeq *seq) { StrVec cps = {0}; if (split_codepoints(s, &cps) != 0) return -1; seq->freq = freq; for (size_t i = 0; i < cps.count; i++) if (symseq_push_copy(seq, cps.items[i]) != 0) { strvec_free(&cps); return -1; } strvec_free(&cps); return 0; }

static void pairstats_free(PairStats *ps) { for (size_t i = 0; i < ps->count; i++) { free(ps->items[i].a); free(ps->items[i].b); } free(ps->items); memset(ps, 0, sizeof(*ps)); }
static int pairstats_add(PairStats *ps, const char *a, const char *b, size_t f) {
    for (size_t i = 0; i < ps->count; i++) if (strcmp(ps->items[i].a, a) == 0 && strcmp(ps->items[i].b, b) == 0) { ps->items[i].freq += f; return 0; }
    if (ps->count == ps->cap) { size_t nc = ps->cap ? ps->cap * 2 : 256; PairStat *p = (PairStat *)realloc(ps->items, nc * sizeof(PairStat)); if (!p) return -1; ps->items = p; ps->cap = nc; }
    ps->items[ps->count].a = xstrdup(a); ps->items[ps->count].b = xstrdup(b); ps->items[ps->count].freq = f;
    if (!ps->items[ps->count].a || !ps->items[ps->count].b) return -1;
    ps->count++; return 0;
}
static int collect_pairs(const SymCorpus *c, PairStats *ps) { for (size_t s = 0; s < c->count; s++) for (size_t i = 0; i + 1 < c->items[s].len; i++) if (pairstats_add(ps, c->items[s].syms[i], c->items[s].syms[i + 1], c->items[s].freq) != 0) return -1; return 0; }
static PairStat *best_pair(PairStats *ps, size_t minf) {
    PairStat *b = NULL;
    for (size_t i = 0; i < ps->count; i++) if (!b || ps->items[i].freq > b->freq || (ps->items[i].freq == b->freq && (strcmp(ps->items[i].a, b->a) > 0 || (strcmp(ps->items[i].a, b->a) == 0 && strcmp(ps->items[i].b, b->b) > 0)))) b = &ps->items[i];
    return b && b->freq >= minf ? b : NULL;
}
static int symseq_merge(SymSeq *s, const char *a, const char *b) {
    SymSeq n = {0}; n.freq = s->freq;
    for (size_t i = 0; i < s->len;) {
        if (i + 1 < s->len && strcmp(s->syms[i], a) == 0 && strcmp(s->syms[i + 1], b) == 0) { char *j = concat2(a, b); if (!j || symseq_push_owned(&n, j) != 0) { free(j); symseq_free(&n); return -1; } i += 2; }
        else { if (symseq_push_copy(&n, s->syms[i]) != 0) { symseq_free(&n); return -1; } i++; }
    }
    symseq_free(s); *s = n; return 0;
}
static int symcorpus_merge(SymCorpus *c, const char *a, const char *b) { for (size_t i = 0; i < c->count; i++) if (symseq_merge(&c->items[i], a, b) != 0) return -1; return 0; }

static void merges_free(Merges *m) { for (size_t i = 0; i < m->count; i++) { free(m->items[i].a); free(m->items[i].b); } free(m->items); memset(m, 0, sizeof(*m)); }
static int merges_push(Merges *m, const char *a, const char *b) { if (m->count == m->cap) { size_t nc = m->cap ? m->cap * 2 : 128; Merge *p = (Merge *)realloc(m->items, nc * sizeof(Merge)); if (!p) return -1; m->items = p; m->cap = nc; } m->items[m->count].a = xstrdup(a); m->items[m->count].b = xstrdup(b); if (!m->items[m->count].a || !m->items[m->count].b) return -1; m->count++; return 0; }
static void unitable_free(UniTable *u) { for (size_t i = 0; i < u->count; i++) free(u->items[i].piece); free(u->items); memset(u, 0, sizeof(*u)); }
static int unitable_add(UniTable *u, const char *p, double prob) { for (size_t i = 0; i < u->count; i++) if (strcmp(u->items[i].piece, p) == 0) { u->items[i].prob += prob; return 0; } if (u->count == u->cap) { size_t nc = u->cap ? u->cap * 2 : 256; UniPiece *x = (UniPiece *)realloc(u->items, nc * sizeof(UniPiece)); if (!x) return -1; u->items = x; u->cap = nc; } u->items[u->count].piece = xstrdup(p); u->items[u->count].prob = prob; if (!u->items[u->count].piece) return -1; u->count++; return 0; }
static double uni_prob(const UniTable *u, const char *p) { for (size_t i = 0; i < u->count; i++) if (strcmp(u->items[i].piece, p) == 0) return u->items[i].prob; return 0.0; }

static int read_lines_normalized(char **paths, size_t n, SPModel *m, TextVocab *lines) {
    char line[65536];
    for (size_t p = 0; p < n; p++) {
        FILE *fp = fopen(paths[p], "rb"); if (!fp) { perror(paths[p]); return -1; }
        while (fgets(line, sizeof(line), fp)) {
            size_t ln = strlen(line); while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0';
            char *e = normalize_escape(m, line); if (!e) { fclose(fp); return -1; }
            if (*e && textvocab_add(lines, e, 1) != 0) { free(e); fclose(fp); return -1; }
            free(e);
        }
        fclose(fp);
    }
    return 0;
}

static int build_reserved(SPModel *m, StrVec *reserved, StrVec *user) {
    if (strvec_push_unique(reserved, m->unk_token) != 0 || strvec_push_unique(reserved, m->bos_token) != 0 || strvec_push_unique(reserved, m->eos_token) != 0 || strvec_push_unique(reserved, m->pad_token) != 0) return -1;
    for (size_t i = 0; i < user->count; i++) if (strvec_push_unique(reserved, user->items[i]) != 0) return -1;
    return 0;
}
static int build_vocab(StrVec *out, const StrVec *reserved, const StrVec *pieces, size_t vocab_size) {
    for (size_t i = 0; i < reserved->count && out->count < vocab_size; i++) if (strvec_push_unique(out, reserved->items[i]) != 0) return -1;
    for (size_t i = 0; i < pieces->count && out->count < vocab_size; i++) if (strvec_push_unique(out, pieces->items[i]) != 0) return -1;
    return 0;
}
static int textcmp_ptr(const void *a, const void *b) { const char * const *x = (const char * const *)a, * const *y = (const char * const *)b; return strcmp(*x, *y); }

#ifdef DM_GPU
typedef struct { char **names; uint32_t count, cap; } SpIntern;
static void sp_intern_free(SpIntern *t) { for (uint32_t i = 0; i < t->count; i++) free(t->names[i]); free(t->names); memset(t, 0, sizeof(*t)); }
static uint32_t sp_intern_add(SpIntern *t, const char *s) {
    for (uint32_t i = 0; i < t->count; i++) if (strcmp(t->names[i], s) == 0) return i;
    if (t->count == t->cap) { uint32_t nc = t->cap ? t->cap * 2 : 64; char **tmp = (char **)realloc(t->names, nc * sizeof(char *)); if (!tmp) return UINT32_MAX; t->names = tmp; t->cap = nc; }
    t->names[t->count] = xstrdup(s); if (!t->names[t->count]) return UINT32_MAX; return t->count++;
}
static uint32_t sp_intern_lookup(const SpIntern *t, const char *s) { for (uint32_t i = 0; i < t->count; i++) if (strcmp(t->names[i], s) == 0) return i; return UINT32_MAX; }
static int sp_build_gpu_input(const SymCorpus *corp, const SpIntern *intern, DmGpuBpeInput *out,
                               uint32_t **sid, uint32_t **wst, uint32_t **wln, uint32_t **wfr) {
    size_t total = 0; for (size_t w = 0; w < corp->count; w++) total += corp->items[w].len;
    *sid = (uint32_t *)malloc(total * sizeof(uint32_t)); *wst = (uint32_t *)malloc(corp->count * sizeof(uint32_t));
    *wln = (uint32_t *)malloc(corp->count * sizeof(uint32_t)); *wfr = (uint32_t *)malloc(corp->count * sizeof(uint32_t));
    if (!*sid || !*wst || !*wln || !*wfr) { free(*sid); free(*wst); free(*wln); free(*wfr); return -1; }
    uint32_t off = 0;
    for (size_t w = 0; w < corp->count; w++) {
        (*wst)[w] = off; (*wln)[w] = (uint32_t)corp->items[w].len; (*wfr)[w] = (uint32_t)corp->items[w].freq;
        for (size_t s = 0; s < corp->items[w].len; s++) {
            uint32_t id = sp_intern_lookup(intern, corp->items[w].syms[s]);
            if (id == UINT32_MAX) { free(*sid); free(*wst); free(*wln); free(*wfr); return -1; }
            (*sid)[off++] = id;
        }
    }
    out->sym_ids = *sid; out->total_syms = total; out->word_starts = *wst;
    out->word_lens = *wln; out->word_freqs = *wfr; out->n_words = corp->count; out->vocab_size = intern->count;
    return 0;
}
static int sp_best_pair_gpu(const uint32_t *pc, uint32_t vsz, const SpIntern *intern, size_t minf,
                             const char **aout, const char **bout) {
    uint32_t best = (uint32_t)minf, ba = UINT32_MAX, bb = UINT32_MAX;
    for (uint32_t a = 0; a < vsz; a++) for (uint32_t b = 0; b < vsz; b++) {
        uint32_t f = pc[(size_t)a * vsz + b];
        if (f > best || (f == best && ba == UINT32_MAX)) { best = f; ba = a; bb = b; }
    }
    if (ba == UINT32_MAX) return 0; *aout = intern->names[ba]; *bout = intern->names[bb]; return 1;
}
#endif /* DM_GPU */

static int train_bpe(TextVocab *lines, SPModel *m, const StrVec *reserved,
                     size_t vocab_size, size_t minf, void *gpu_ctx) {
    SymCorpus corp = {0}; StrVec chars = {0}, pieces = {0};
    for (size_t i = 0; i < lines->count; i++) {
        SymSeq s = {0};
        if (symseq_from_text(lines->items[i].text, lines->items[i].freq, &s) != 0 || symcorpus_push_owned(&corp, s) != 0)
            { symseq_free(&s); symcorpus_free(&corp); return -1; }
        for (size_t j = 0; j < s.len; j++) strvec_push_unique(&chars, s.syms[j]);
    }
    qsort(chars.items, chars.count, sizeof(char *), textcmp_ptr);
    size_t target = vocab_size > reserved->count + chars.count ? vocab_size - reserved->count - chars.count : 0;

#ifdef DM_GPU
    DmGpuCtx *gpu = (DmGpuCtx *)gpu_ctx;
    SpIntern intern = {0};
    if (gpu && dm_gpu_ready(gpu)) {
        for (size_t i = 0; i < chars.count; i++)
            if (sp_intern_add(&intern, chars.items[i]) == UINT32_MAX) { sp_intern_free(&intern); gpu = NULL; }
    }
#else
    (void)gpu_ctx;
#endif

    for (size_t k = 0; k < target; k++) {
        char *la = NULL, *lb = NULL; int found = 0;

#ifdef DM_GPU
        if (gpu && dm_gpu_ready(gpu) && intern.count <= DM_GPU_BPE_MAX_VOCAB) {
            uint32_t *sid = NULL, *wst = NULL, *wln = NULL, *wfr = NULL;
            DmGpuBpeInput inp = {0};
            if (sp_build_gpu_input(&corp, &intern, &inp, &sid, &wst, &wln, &wfr) == 0) {
                size_t vsz = intern.count;
                uint32_t *pc = (uint32_t *)calloc(vsz * vsz, sizeof(uint32_t));
                if (pc && dm_gpu_bpe_pair_count(gpu, &inp, pc) == 0) {
                    const char *ap = NULL, *bp = NULL;
                    if (sp_best_pair_gpu(pc, (uint32_t)vsz, &intern, minf, &ap, &bp)) {
                        la = xstrdup(ap); lb = xstrdup(bp); found = (la && lb) ? 1 : 0;
                    } else { found = -1; }
                }
                free(pc); free(sid); free(wst); free(wln); free(wfr);
            }
        }
#endif

        if (!found) {
            PairStats ps = {0}; if (collect_pairs(&corp, &ps) != 0) { pairstats_free(&ps); goto bad; }
            PairStat *b = best_pair(&ps, minf);
            if (!b) { pairstats_free(&ps); break; }
            la = xstrdup(b->a); lb = xstrdup(b->b); pairstats_free(&ps);
            found = (la && lb) ? 1 : 0;
        } else if (found < 0) { break; }

        if (!found || !la || !lb || merges_push(&m->merges, la, lb) != 0 || symcorpus_merge(&corp, la, lb) != 0)
            { free(la); free(lb); goto bad; }
#ifdef DM_GPU
        char *joined = concat2(la, lb);
        if (joined) { sp_intern_add(&intern, joined); free(joined); }
#endif
        free(la); free(lb);
    }

    for (size_t i = 0; i < chars.count; i++) if (strvec_push_unique(&pieces, chars.items[i]) != 0) goto bad;
    for (size_t i = 0; i < m->merges.count; i++) { char *j = concat2(m->merges.items[i].a, m->merges.items[i].b); if (!j || strvec_push_owned(&pieces, j) != 0) { free(j); goto bad; } }
    {
        int rc = build_vocab(&m->vocab, reserved, &pieces, vocab_size);
#ifdef DM_GPU
        sp_intern_free(&intern);
#endif
        strvec_free(&chars); strvec_free(&pieces); symcorpus_free(&corp); return rc;
    }
bad:
#ifdef DM_GPU
    sp_intern_free(&intern);
#endif
    strvec_free(&chars); strvec_free(&pieces); symcorpus_free(&corp); return -1;
}

static int enumerate_unigram(TextVocab *lines, UniTable *u, size_t seed_size, size_t max_len, size_t minf) {
    TextVocab counts = {0}; StrVec chars = {0};
    for (size_t w = 0; w < lines->count; w++) {
        StrVec cps = {0}; if (split_codepoints(lines->items[w].text, &cps) != 0) return -1;
        for (size_t i = 0; i < cps.count; i++) { strvec_push_unique(&chars, cps.items[i]); for (size_t j = i + 1; j <= cps.count && j <= i + max_len; j++) { char *sub = join_range(&cps, i, j); if (!sub || textvocab_add(&counts, sub, lines->items[w].freq) != 0) { free(sub); strvec_free(&cps); return -1; } free(sub); } }
        strvec_free(&cps);
    }
    for (size_t i = 0; i < counts.count; i++) {
        if (strlen(counts.items[i].text) > 0 && (counts.items[i].freq >= minf || strvec_contains(&chars, counts.items[i].text))) {
            if (unitable_add(u, counts.items[i].text, (double)counts.items[i].freq) != 0) { textvocab_free(&counts); strvec_free(&chars); return -1; }
        }
        if (u->count >= seed_size) break;
    }
    double z = 0.0; for (size_t i = 0; i < u->count; i++) z += u->items[i].prob; if (z <= 0.0) z = 1.0; for (size_t i = 0; i < u->count; i++) u->items[i].prob /= z;
    textvocab_free(&counts); strvec_free(&chars); return 0;
}
static int unipiece_cmp(const void *a, const void *b) { const UniPiece *x = (const UniPiece *)a, *y = (const UniPiece *)b; if (x->prob < y->prob) return 1; if (x->prob > y->prob) return -1; return strcmp(x->piece, y->piece); }
static int train_unigram_lite(TextVocab *lines, SPModel *m, const StrVec *reserved, size_t vocab_size, size_t seed_size, size_t max_len, size_t minf) {
    size_t target = vocab_size > reserved->count ? vocab_size - reserved->count : 1;
    if (enumerate_unigram(lines, &m->uni, seed_size > target ? seed_size : target, max_len, minf) != 0) return -1;
    qsort(m->uni.items, m->uni.count, sizeof(UniPiece), unipiece_cmp);
    if (m->uni.count > target) {
        for (size_t i = target; i < m->uni.count; i++) free(m->uni.items[i].piece);
        m->uni.count = target;
    }
    double z = 0.0; for (size_t i = 0; i < m->uni.count; i++) z += m->uni.items[i].prob; if (z <= 0.0) z = 1.0; for (size_t i = 0; i < m->uni.count; i++) m->uni.items[i].prob /= z;
    StrVec pieces = {0}; for (size_t i = 0; i < m->uni.count; i++) if (strvec_push_copy(&pieces, m->uni.items[i].piece) != 0) { strvec_free(&pieces); return -1; }
    int rc = build_vocab(&m->vocab, reserved, &pieces, vocab_size); strvec_free(&pieces); return rc;
}

static int vocab_id(const SPModel *m, const char *p) { for (size_t i = 0; i < m->vocab.count; i++) if (strcmp(m->vocab.items[i], p) == 0) return (int)i; return 0; }
static int is_special(const SPModel *m, const char *p) { return strcmp(p, m->unk_token) == 0 || strcmp(p, m->bos_token) == 0 || strcmp(p, m->eos_token) == 0 || strcmp(p, m->pad_token) == 0; }
static int encode_bpe(const SPModel *m, const char *text, StrVec *out) {
    char *seqtxt = normalize_escape(m, text); if (!seqtxt) return -1; StrVec pieces = {0}; if (split_codepoints(seqtxt, &pieces) != 0) { free(seqtxt); return -1; } free(seqtxt);
    for (size_t mi = 0; mi < m->merges.count; mi++) {
        StrVec next = {0};
        for (size_t i = 0; i < pieces.count;) {
            if (i + 1 < pieces.count && strcmp(pieces.items[i], m->merges.items[mi].a) == 0 && strcmp(pieces.items[i + 1], m->merges.items[mi].b) == 0) { char *j = concat2(pieces.items[i], pieces.items[i + 1]); if (!j || strvec_push_owned(&next, j) != 0) { free(j); strvec_free(&next); strvec_free(&pieces); return -1; } i += 2; }
            else { if (strvec_push_copy(&next, pieces.items[i]) != 0) { strvec_free(&next); strvec_free(&pieces); return -1; } i++; }
        }
        strvec_free(&pieces); pieces = next;
    }
    *out = pieces; return 0;
}
static int encode_unigram(const SPModel *m, const char *text, StrVec *out) {
    char *s = normalize_escape(m, text); if (!s) return -1; StrVec cps = {0}; if (split_codepoints(s, &cps) != 0) { free(s); return -1; } free(s);
    size_t n = cps.count; double *best = (double *)malloc((n + 1) * sizeof(double)); int *prev = (int *)malloc((n + 1) * sizeof(int)); char **pp = (char **)calloc(n + 1, sizeof(char *)); if (!best || !prev || !pp) { free(best); free(prev); free(pp); strvec_free(&cps); return -1; }
    for (size_t i = 0; i <= n; i++) { best[i] = LOG_ZERO; prev[i] = -1; } best[0] = 0.0;
    for (size_t e = 1; e <= n; e++) for (size_t st = 0; st < e; st++) { char *sub = join_range(&cps, st, e); double p = sub ? uni_prob(&m->uni, sub) : 0.0; if (p > 0.0 && best[st] + log(p) > best[e]) { best[e] = best[st] + log(p); prev[e] = (int)st; free(pp[e]); pp[e] = sub; } else free(sub); }
    if (prev[n] < 0) { for (size_t i = 0; i < cps.count; i++) strvec_push_copy(out, cps.items[i]); }
    else { StrVec rev = {0}; for (int pos = (int)n; pos > 0; pos = prev[pos]) strvec_push_copy(&rev, pp[pos]); for (size_t i = rev.count; i > 0; i--) strvec_push_copy(out, rev.items[i - 1]); strvec_free(&rev); }
    for (size_t i = 0; i <= n; i++) free(pp[i]);
    free(pp);
    free(prev);
    free(best);
    strvec_free(&cps);
    return 0;
}
static int encode_pieces(const SPModel *m, const char *text, StrVec *out) { return strcmp(m->model_type, "bpe") == 0 ? encode_bpe(m, text, out) : encode_unigram(m, text, out); }

static void rules_free(Rules *r) { for (size_t i = 0; i < r->count; i++) { free(r->items[i].src); free(r->items[i].dst); } free(r->items); memset(r, 0, sizeof(*r)); }
static int rules_push(Rules *r, const char *src, const char *dst) { if (r->count == r->cap) { size_t nc = r->cap ? r->cap * 2 : 16; Rule *p = (Rule *)realloc(r->items, nc * sizeof(Rule)); if (!p) return -1; r->items = p; r->cap = nc; } r->items[r->count].src = xstrdup(src); r->items[r->count].dst = xstrdup(dst); if (!r->items[r->count].src || !r->items[r->count].dst) return -1; r->count++; return 0; }
static int rule_len_cmp(const void *a, const void *b) { const Rule *x = (const Rule *)a, *y = (const Rule *)b; size_t nx = strlen(x->src), ny = strlen(y->src); return (ny > nx) - (ny < nx); }
static char *parse_codepoint_sequence(const char *text) {
    StrVec out = {0};
    char *copy = xstrdup(text);
    if (!copy) return NULL;
    for (char *tok = strtok(copy, " \t"); tok; tok = strtok(NULL, " \t")) {
        if (strncmp(tok, "U+", 2) == 0) {
            unsigned long cp = strtoul(tok + 2, NULL, 16);
            char buf[5]; size_t n = 0;
            if (cp <= 0x7F) buf[n++] = (char)cp;
            else if (cp <= 0x7FF) { buf[n++] = (char)(0xC0 | (cp >> 6)); buf[n++] = (char)(0x80 | (cp & 0x3F)); }
            else if (cp <= 0xFFFF) { buf[n++] = (char)(0xE0 | (cp >> 12)); buf[n++] = (char)(0x80 | ((cp >> 6) & 0x3F)); buf[n++] = (char)(0x80 | (cp & 0x3F)); }
            else { buf[n++] = (char)(0xF0 | (cp >> 18)); buf[n++] = (char)(0x80 | ((cp >> 12) & 0x3F)); buf[n++] = (char)(0x80 | ((cp >> 6) & 0x3F)); buf[n++] = (char)(0x80 | (cp & 0x3F)); }
            buf[n] = '\0';
            if (strvec_push_copy(&out, buf) != 0) { free(copy); strvec_free(&out); return NULL; }
        } else if (strvec_push_copy(&out, tok) != 0) { free(copy); strvec_free(&out); return NULL; }
    }
    free(copy);
    char *joined = join_range(&out, 0, out.count);
    strvec_free(&out);
    return joined;
}
static int load_rules_tsv(Rules *rules, const char *path) {
    if (!path) return 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return -1; }
    char line[65536];
    while (fgets(line, sizeof(line), fp)) {
        size_t ln = strlen(line);
        while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0';
        if (!line[0] || line[0] == '#') continue;
        char *tab = strchr(line, '\t');
        if (!tab) { fclose(fp); return -1; }
        *tab = '\0';
        char *src = parse_codepoint_sequence(line);
        char *dst = parse_codepoint_sequence(tab + 1);
        if (!src || !dst || rules_push(rules, src, dst) != 0) { free(src); free(dst); fclose(fp); return -1; }
        free(src);
        free(dst);
    }
    fclose(fp);
    qsort(rules->items, rules->count, sizeof(Rule), rule_len_cmp);
    return 0;
}
static void model_free(SPModel *m) { free(m->model_type); strvec_free(&m->vocab); free(m->normalization); rules_free(&m->rules); free(m->unk_token); free(m->bos_token); free(m->eos_token); free(m->pad_token); merges_free(&m->merges); unitable_free(&m->uni); memset(m, 0, sizeof(*m)); }
static int model_init(SPModel *m, const char *type, const char *norm, int dummy, const char *unk, const char *bos, const char *eos, const char *pad) {
    m->model_type = xstrdup(type); m->normalization = xstrdup(norm); m->add_dummy_prefix = dummy; m->unk_token = xstrdup(unk); m->bos_token = xstrdup(bos); m->eos_token = xstrdup(eos); m->pad_token = xstrdup(pad);
    return (m->model_type && m->normalization && m->unk_token && m->bos_token && m->eos_token && m->pad_token) ? 0 : -1;
}
static int write_model(const SPModel *m, const char *path) {
    FILE *out = fopen(path, "wb"); if (!out) { perror(path); return -1; }
    fprintf(out, "{\n  \"version\": \"dm-sentencepiece-lite-D18-2012\",\n  \"model_type\": "); json_string(out, m->model_type); fprintf(out, ",\n  \"vocab\": [\n");
    for (size_t i = 0; i < m->vocab.count; i++) { fprintf(out, "    "); json_string(out, m->vocab.items[i]); fprintf(out, "%s\n", i + 1 < m->vocab.count ? "," : ""); }
    fprintf(out, "  ],\n  \"normalization\": "); json_string(out, m->normalization); fprintf(out, ",\n  \"normalization_rules\": [");
    for (size_t i = 0; i < m->rules.count; i++) { if (i) fprintf(out, ", "); fprintf(out, "["); json_string(out, m->rules.items[i].src); fprintf(out, ", "); json_string(out, m->rules.items[i].dst); fprintf(out, "]"); }
    fprintf(out, "],\n  \"add_dummy_prefix\": %s,\n  \"unk_token\": ", m->add_dummy_prefix ? "true" : "false"); json_string(out, m->unk_token); fprintf(out, ",\n  \"bos_token\": "); json_string(out, m->bos_token); fprintf(out, ",\n  \"eos_token\": "); json_string(out, m->eos_token); fprintf(out, ",\n  \"pad_token\": "); json_string(out, m->pad_token); fprintf(out, ",\n  \"merges\": [\n");
    for (size_t i = 0; i < m->merges.count; i++) { fprintf(out, "    ["); json_string(out, m->merges.items[i].a); fprintf(out, ", "); json_string(out, m->merges.items[i].b); fprintf(out, "]%s\n", i + 1 < m->merges.count ? "," : ""); }
    fprintf(out, "  ],\n  \"unigram_pieces\": {"); for (size_t i = 0; i < m->uni.count; i++) { if (i) fprintf(out, ", "); json_string(out, m->uni.items[i].piece); fprintf(out, ": %.17g", m->uni.items[i].prob); } fprintf(out, "}\n}\n");
    fclose(out); return 0;
}
static char *slurp(const char *path) { FILE *fp = fopen(path, "rb"); if (!fp) { perror(path); return NULL; } fseek(fp, 0, SEEK_END); long n = ftell(fp); rewind(fp); char *b = (char *)malloc((size_t)n + 1); if (!b) { fclose(fp); return NULL; } size_t r = fread(b, 1, (size_t)n, fp); b[r] = '\0'; fclose(fp); return b; }
static char *json_get_string(const char *data, const char *key) { char pat[128]; snprintf(pat, sizeof(pat), "\"%s\"", key); char *p = strstr(data, pat); if (!p) return NULL; p = strchr(p, ':'); if (!p) return NULL; p = strchr(p, '"'); if (!p) return NULL; p++; char *e = p; while (*e && *e != '"') e++; return xstrndup(p, (size_t)(e - p)); }
static int read_model(const char *path, SPModel *m) {
    char *d = slurp(path); if (!d) return -1; char *type = json_get_string(d, "model_type"); char *norm = json_get_string(d, "normalization"); char *unk = json_get_string(d, "unk_token"); char *bos = json_get_string(d, "bos_token"); char *eos = json_get_string(d, "eos_token"); char *pad = json_get_string(d, "pad_token");
    int dummy = strstr(d, "\"add_dummy_prefix\": false") ? 0 : 1; if (model_init(m, type ? type : "unigram", norm ? norm : "nfkc", dummy, unk ? unk : SP_UNK, bos ? bos : SP_BOS, eos ? eos : SP_EOS, pad ? pad : SP_PAD) != 0) { free(d); return -1; }
    free(type); free(norm); free(unk); free(bos); free(eos); free(pad);
    char *p = strstr(d, "\"vocab\""); if (p) p = strchr(p, '['); if (p) p++; while (p && *p) { char *end = strchr(p, ']'); char *q = strchr(p, '"'); if (!q || (end && end < q)) break; q++; char *e = strchr(q, '"'); if (!e) break; char *s = xstrndup(q, (size_t)(e - q)); strvec_push_owned(&m->vocab, s); p = e + 1; }
    p = strstr(d, "\"normalization_rules\""); if (p) p = strchr(p, '['); if (p) p++;
    while (p && *p) {
        char *end = strstr(p, "],\n  \"add_dummy_prefix\"");
        char *pair = strchr(p, '[');
        if (!pair || (end && end < pair)) break;
        char *q = strchr(pair, '"'); if (!q) break; q++;
        char *e = strchr(q, '"'); if (!e) break;
        char *a = xstrndup(q, (size_t)(e - q));
        q = strchr(e + 1, '"'); if (!q) { free(a); break; } q++;
        e = strchr(q, '"'); if (!e) { free(a); break; }
        char *b = xstrndup(q, (size_t)(e - q));
        rules_push(&m->rules, a, b);
        free(a); free(b); p = e + 1;
    }
    qsort(m->rules.items, m->rules.count, sizeof(Rule), rule_len_cmp);
    p = strstr(d, "\"merges\""); if (p) p = strchr(p, '['); if (p) p++; while (p && *p) { char *end = strstr(p, "],\n  \"unigram_pieces\""); char *pair = strchr(p, '['); if (!pair || (end && end < pair)) break; char *q = strchr(pair, '"'); if (!q) break; q++; char *e = strchr(q, '"'); if (!e) break; char *a = xstrndup(q, (size_t)(e - q)); q = strchr(e + 1, '"'); if (!q) { free(a); break; } q++; e = strchr(q, '"'); if (!e) { free(a); break; } char *b = xstrndup(q, (size_t)(e - q)); merges_push(&m->merges, a, b); free(a); free(b); p = e + 1; }
    p = strstr(d, "\"unigram_pieces\""); if (p) p = strchr(p, '{'); if (p) p++; while (p && *p && *p != '}') { char *q = strchr(p, '"'); if (!q) break; q++; char *e = strchr(q, '"'); if (!e) break; char *piece = xstrndup(q, (size_t)(e - q)); char *colon = strchr(e, ':'); if (!colon) { free(piece); break; } double prob = strtod(colon + 1, &p); unitable_add(&m->uni, piece, prob); free(piece); }
    free(d); return 0;
}

static void usage(const char *p) {
    fprintf(stderr, "Usage: %s sentencepiece train --input <files...> --model-type bpe|unigram --vocab-size N -o model\n", p);
    fprintf(stderr, "       %s sentencepiece encode --model model [--input file|--text text] [--output-format piece|id|json]\n", p);
    fprintf(stderr, "       %s sentencepiece decode --model model [--input file] [--input-format piece|id|json]\n", p);
    fprintf(stderr, "       %s sentencepiece inspect|vocab --model model\n", p);
}

int dm_sentencepiece_cli(int argc, char **argv) {
    int start = 1; if (argc >= 2 && (strcmp(argv[1], "sentencepiece") == 0 || strcmp(argv[1], "spm") == 0 || strcmp(argv[1], "dm_sentencepiece") == 0)) start = 2;
    if (argc <= start) { usage(argv[0]); return 2; }
    const char *cmd = argv[start];
    if (strcmp(cmd, "train") == 0) {
        StrVec inputs = {0}, users = {0}, reserved = {0}; const char *out = NULL, *prefix = "spm", *type = "unigram", *norm = "nfkc", *rule_path = NULL, *unk = SP_UNK, *bos = SP_BOS, *eos = SP_EOS, *pad = SP_PAD; size_t vs = 0, minf = 2, seed = 8000, maxlen = 16; int dummy = 1, stats = 0, write_vocab = 0, use_gpu = 0, gpu_device = 0;
        for (int i = start + 1; i < argc; i++) {
            if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) while (i + 1 < argc && argv[i + 1][0] != '-') strvec_push_copy(&inputs, argv[++i]);
            else if (strcmp(argv[i], "--model-prefix") == 0 && i + 1 < argc) prefix = argv[++i];
            else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) out = argv[++i];
            else if (strcmp(argv[i], "--model-type") == 0 && i + 1 < argc) type = argv[++i];
            else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) vs = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--normalization") == 0 && i + 1 < argc) norm = argv[++i];
            else if (strcmp(argv[i], "--add-dummy-prefix") == 0) dummy = 1;
            else if (strcmp(argv[i], "--no-add-dummy-prefix") == 0) dummy = 0;
            else if (strcmp(argv[i], "--unk-token") == 0 && i + 1 < argc) unk = argv[++i];
            else if (strcmp(argv[i], "--bos-token") == 0 && i + 1 < argc) bos = argv[++i];
            else if (strcmp(argv[i], "--eos-token") == 0 && i + 1 < argc) eos = argv[++i];
            else if (strcmp(argv[i], "--pad-token") == 0 && i + 1 < argc) pad = argv[++i];
            else if (strcmp(argv[i], "--user-defined-symbol") == 0 && i + 1 < argc) strvec_push_copy(&users, argv[++i]);
            else if (strcmp(argv[i], "--min-frequency") == 0 && i + 1 < argc) minf = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--seed-size") == 0 && i + 1 < argc) seed = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--max-piece-length") == 0 && i + 1 < argc) maxlen = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--write-vocab") == 0) write_vocab = 1;
            else if (strcmp(argv[i], "--stats") == 0) stats = 1;
            else if (strcmp(argv[i], "--normalization-rule-tsv") == 0 && i + 1 < argc) rule_path = argv[++i];
            else if (strcmp(argv[i], "--gpu") == 0) use_gpu = 1;
            else if (strcmp(argv[i], "--gpu-device") == 0 && i + 1 < argc) { gpu_device = (int)strtol(argv[++i], NULL, 10); use_gpu = 1; }
            else { usage(argv[0]); return 2; }
        }
        if (!inputs.count || !vs) { usage(argv[0]); return 2; }
        void *gpu_ctx = NULL;
#ifdef DM_GPU
        if (use_gpu && strcmp(type, "bpe") == 0) { gpu_ctx = dm_gpu_create(gpu_device, NULL); if (!gpu_ctx || !dm_gpu_ready((DmGpuCtx *)gpu_ctx)) { fprintf(stderr, "[spm] GPU init failed, falling back to CPU\n"); dm_gpu_destroy((DmGpuCtx *)gpu_ctx); gpu_ctx = NULL; } else { char _dname[256]={0}; dm_gpu_device_name((DmGpuCtx*)gpu_ctx,_dname,sizeof(_dname)); fprintf(stderr,"[spm] GPU: %s\n",_dname); } }
        else if (use_gpu) fprintf(stderr, "[spm] --gpu only applies to BPE mode, using CPU for unigram\n");
#else
        if (use_gpu) fprintf(stderr, "[spm] built without GPU support, using CPU\n");
#endif
        char outbuf[4096]; if (!out) { snprintf(outbuf, sizeof(outbuf), "%s.model", prefix); out = outbuf; }
        SPModel m = {0}; TextVocab lines = {0}; int rc = model_init(&m, type, norm, dummy, unk, bos, eos, pad); if (rc == 0) rc = load_rules_tsv(&m.rules, rule_path); if (rc == 0) rc = build_reserved(&m, &reserved, &users); if (rc == 0) rc = read_lines_normalized(inputs.items, inputs.count, &m, &lines);
        if (rc == 0) rc = strcmp(type, "bpe") == 0 ? train_bpe(&lines, &m, &reserved, vs, minf, gpu_ctx) : train_unigram_lite(&lines, &m, &reserved, vs, seed, maxlen, minf);
        if (rc == 0) rc = write_model(&m, out);
        if (rc == 0 && write_vocab) { char vb[4096]; snprintf(vb, sizeof(vb), "%s.vocab", prefix); FILE *vf = fopen(vb, "wb"); if (vf) { for (size_t i = 0; i < m.vocab.count; i++) fprintf(vf, "%zu\t%s\n", i, m.vocab.items[i]); fclose(vf); } }
        if (rc == 0 && stats) fprintf(stderr, "{\"model_type\":\"%s\",\"vocab_size\":%zu,\"normalization\":\"%s\"}\n", m.model_type, m.vocab.count, m.normalization);
        model_free(&m); textvocab_free(&lines); strvec_free(&inputs); strvec_free(&users); strvec_free(&reserved);
#ifdef DM_GPU
        dm_gpu_destroy((DmGpuCtx *)gpu_ctx);
#endif
        return rc == 0 ? 0 : 1;
    }
    if (strcmp(cmd, "encode") == 0) {
        const char *modelp = NULL, *input = NULL, *output = NULL, *fmt = "piece"; StrVec texts = {0}; int add_bos = 0, add_eos = 0;
        for (int i = start + 1; i < argc; i++) { if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) modelp = argv[++i]; else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) input = argv[++i]; else if (strcmp(argv[i], "--text") == 0 && i + 1 < argc) strvec_push_copy(&texts, argv[++i]); else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output = argv[++i]; else if (strcmp(argv[i], "--output-format") == 0 && i + 1 < argc) fmt = argv[++i]; else if (strcmp(argv[i], "--add-bos") == 0) add_bos = 1; else if (strcmp(argv[i], "--add-eos") == 0) add_eos = 1; else if ((strcmp(argv[i], "--mode") == 0 || strcmp(argv[i], "--alpha") == 0 || strcmp(argv[i], "--seed") == 0 || strcmp(argv[i], "--nbest-size") == 0) && i + 1 < argc) ++i; else { usage(argv[0]); return 2; } }
        if (!modelp) { usage(argv[0]); return 2; } SPModel m = {0}; if (read_model(modelp, &m) != 0) return 1; FILE *in = input ? fopen(input, "rb") : stdin; FILE *out = output ? fopen(output, "wb") : stdout; if (!out || (!texts.count && !in)) return 1; char line[65536]; size_t rows = texts.count ? texts.count : (size_t)-1;
        for (size_t r = 0; r < rows; r++) {
            const char *text;
            if (texts.count) text = texts.items[r];
            else {
                if (!fgets(line, sizeof(line), in)) break;
                size_t ln = strlen(line);
                while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0';
                text = line;
            }
            StrVec pcs = {0};
            encode_pieces(&m, text, &pcs);
            if (strcmp(fmt, "json") == 0) {
                fputc('[', out);
                int first = 1;
                if (add_bos) { json_string(out, m.bos_token); first = 0; }
                for (size_t i = 0; i < pcs.count; i++) { if (!first) fputs(", ", out); json_string(out, pcs.items[i]); first = 0; }
                if (add_eos) { if (!first) fputs(", ", out); json_string(out, m.eos_token); }
                fputs("]\n", out);
            } else {
                int first = 0;
                if (add_bos) {
                    if (strcmp(fmt, "id") == 0) fprintf(out, "%d", vocab_id(&m, m.bos_token));
                    else fputs(m.bos_token, out);
                    first = 1;
                }
                for (size_t i = 0; i < pcs.count; i++) {
                    if (first) fputc(' ', out);
                    if (strcmp(fmt, "id") == 0) fprintf(out, "%d", vocab_id(&m, pcs.items[i]));
                    else fputs(pcs.items[i], out);
                    first = 1;
                }
                if (add_eos) {
                    if (first) fputc(' ', out);
                    if (strcmp(fmt, "id") == 0) fprintf(out, "%d", vocab_id(&m, m.eos_token));
                    else fputs(m.eos_token, out);
                }
                fputc('\n', out);
            }
            strvec_free(&pcs);
        }
        if (input) fclose(in);
        if (output) fclose(out);
        model_free(&m);
        strvec_free(&texts);
        return 0;
    }
    if (strcmp(cmd, "decode") == 0) {
        const char *modelp = NULL, *input = NULL, *output = NULL, *fmt = "piece"; for (int i = start + 1; i < argc; i++) { if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) modelp = argv[++i]; else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) input = argv[++i]; else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) output = argv[++i]; else if (strcmp(argv[i], "--input-format") == 0 && i + 1 < argc) fmt = argv[++i]; else { usage(argv[0]); return 2; } }
        SPModel m = {0}; if (!modelp || read_model(modelp, &m) != 0) return 1; FILE *in = input ? fopen(input, "rb") : stdin; FILE *out = output ? fopen(output, "wb") : stdout; char line[65536]; while (fgets(line, sizeof(line), in)) { StrVec pcs = {0}; size_t ln = strlen(line); while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0'; if (strcmp(fmt, "id") == 0) { for (char *tok = strtok(line, " \t"); tok; tok = strtok(NULL, " \t")) { long id = strtol(tok, NULL, 10); if (id >= 0 && (size_t)id < m.vocab.count && !is_special(&m, m.vocab.items[id])) strvec_push_copy(&pcs, m.vocab.items[id]); } } else { for (char *tok = strtok(line, " \t"); tok; tok = strtok(NULL, " \t")) if (!is_special(&m, tok)) strvec_push_copy(&pcs, tok); } char *joined = join_range(&pcs, 0, pcs.count); char *plain = joined ? unescape_ws(joined, m.add_dummy_prefix) : xstrdup(""); fprintf(out, "%s\n", plain ? plain : ""); free(joined); free(plain); strvec_free(&pcs); } if (input) fclose(in); if (output) fclose(out); model_free(&m); return 0;
    }
    if (strcmp(cmd, "inspect") == 0 || strcmp(cmd, "vocab") == 0) {
        const char *modelp = NULL, *fmt = "tsv"; for (int i = start + 1; i < argc; i++) { if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) modelp = argv[++i]; else if (strcmp(argv[i], "--output-format") == 0 && i + 1 < argc) fmt = argv[++i]; else { usage(argv[0]); return 2; } }
        SPModel m = {0}; if (!modelp || read_model(modelp, &m) != 0) return 1; if (strcmp(cmd, "inspect") == 0) printf("{\n  \"add_dummy_prefix\": %s,\n  \"model_type\": \"%s\",\n  \"normalization\": \"%s\",\n  \"normalization_rules\": %zu,\n  \"special_tokens\": [\"%s\", \"%s\", \"%s\", \"%s\"],\n  \"version\": \"dm-sentencepiece-lite-D18-2012\",\n  \"vocab_size\": %zu\n}\n", m.add_dummy_prefix ? "true" : "false", m.model_type, m.normalization, m.rules.count, m.unk_token, m.bos_token, m.eos_token, m.pad_token, m.vocab.count);
        else if (strcmp(fmt, "json") == 0) {
            printf("[\n");
            for (size_t i = 0; i < m.vocab.count; i++) {
                printf("  {\"id\": %zu, \"piece\": ", i);
                json_string(stdout, m.vocab.items[i]);
                printf("}%s\n", i + 1 < m.vocab.count ? "," : "");
            }
            printf("]\n");
        } else {
            for (size_t i = 0; i < m.vocab.count; i++) printf("%zu\t%s\n", i, m.vocab.items[i]);
        }
        model_free(&m);
        return 0;
    }
    usage(argv[0]); return 2;
}
