#include "tokenizer/parity_bpe.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned char *data; size_t len; } Tok;
typedef struct { Tok *items; size_t len, cap; } Seq;
typedef struct { Seq *items; size_t count, cap; } SeqVec;
typedef struct { char *lang; SeqVec docs; } LangDocs;
typedef struct { LangDocs *items; size_t count, cap; } Corpus;
typedef struct { Tok left, right; size_t freq; } PairStat;
typedef struct { PairStat *items; size_t count, cap; } PairStats;
typedef struct { Tok left, right, joined; } Merge;
typedef struct { Merge *items; size_t count, cap; } MergeTable;
typedef struct { char *pretokenizer; char *strategy; MergeTable merges; } PBPEModel;
typedef struct { char *focus; Tok left, right; size_t pair_count; double min_cr, gini; } TraceRow;
typedef struct { TraceRow *items; size_t count, cap; } TraceVec;
typedef struct { char **items; size_t count, cap; } StrVec;
typedef struct { Tok tok; size_t count; } TokCount;
typedef struct { TokCount *items; size_t count, cap; } TokCounts;

static char *xstrdup(const char *s) { size_t n = strlen(s); char *o = (char *)malloc(n + 1); if (!o) return NULL; memcpy(o, s, n + 1); return o; }
static char *xstrndup(const char *s, size_t n) { char *o = (char *)malloc(n + 1); if (!o) return NULL; memcpy(o, s, n); o[n] = '\0'; return o; }
static Tok tok_copy(const unsigned char *p, size_t n) { Tok t; t.data = (unsigned char *)malloc(n ? n : 1); t.len = n; if (t.data && n) memcpy(t.data, p, n); return t; }
static void tok_free(Tok *t) { free(t->data); t->data = NULL; t->len = 0; }
static int tok_eq(Tok a, Tok b) { return a.len == b.len && (a.len == 0 || memcmp(a.data, b.data, a.len) == 0); }
static int tok_cmp(Tok a, Tok b) { size_t n = a.len < b.len ? a.len : b.len; int c = n ? memcmp(a.data, b.data, n) : 0; if (c) return c; return (a.len > b.len) - (a.len < b.len); }
static Tok tok_concat(Tok a, Tok b) { Tok t; t.len = a.len + b.len; t.data = (unsigned char *)malloc(t.len ? t.len : 1); if (!t.data) { t.len = 0; return t; } if (a.len) memcpy(t.data, a.data, a.len); if (b.len) memcpy(t.data + a.len, b.data, b.len); return t; }

static void json_string(FILE *out, const char *s) { fputc('"', out); for (; *s; s++) { unsigned char c = (unsigned char)*s; if (c == '"' || c == '\\') { fputc('\\', out); fputc(c, out); } else if (c == '\n') fputs("\\n", out); else if (c == '\r') fputs("\\r", out); else if (c == '\t') fputs("\\t", out); else if (c < 32) fprintf(out, "\\u%04x", c); else fputc(c, out); } fputc('"', out); }

static const char b64tab[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static char *b64_encode(Tok t) {
    size_t outn = ((t.len + 2) / 3) * 4;
    char *out = (char *)malloc(outn + 1);
    if (!out) return NULL;
    size_t o = 0;
    for (size_t i = 0; i < t.len; i += 3) {
        uint32_t v = (uint32_t)t.data[i] << 16;
        if (i + 1 < t.len) v |= (uint32_t)t.data[i + 1] << 8;
        if (i + 2 < t.len) v |= (uint32_t)t.data[i + 2];
        out[o++] = b64tab[(v >> 18) & 63];
        out[o++] = b64tab[(v >> 12) & 63];
        out[o++] = (i + 1 < t.len) ? b64tab[(v >> 6) & 63] : '=';
        out[o++] = (i + 2 < t.len) ? b64tab[v & 63] : '=';
    }
    out[o] = '\0';
    return out;
}
static int b64_val(char c) { const char *p = strchr(b64tab, c); return p ? (int)(p - b64tab) : -1; }
static Tok b64_decode(const char *s) {
    size_t n = strlen(s), cap = (n / 4) * 3 + 3, o = 0;
    Tok t; t.data = (unsigned char *)malloc(cap); t.len = 0;
    if (!t.data) return t;
    for (size_t i = 0; i < n;) {
        int v[4]; int pad = 0;
        for (int k = 0; k < 4 && i < n; k++, i++) {
            if (s[i] == '=') { v[k] = 0; pad++; }
            else { v[k] = b64_val(s[i]); if (v[k] < 0) { tok_free(&t); return t; } }
        }
        uint32_t x = ((uint32_t)v[0] << 18) | ((uint32_t)v[1] << 12) | ((uint32_t)v[2] << 6) | (uint32_t)v[3];
        if (o < cap) t.data[o++] = (unsigned char)((x >> 16) & 255);
        if (pad < 2 && o < cap) t.data[o++] = (unsigned char)((x >> 8) & 255);
        if (pad < 1 && o < cap) t.data[o++] = (unsigned char)(x & 255);
    }
    t.len = o;
    return t;
}

static void strvec_free(StrVec *v) { for (size_t i = 0; i < v->count; i++) free(v->items[i]); free(v->items); memset(v, 0, sizeof(*v)); }
static int strvec_push_owned(StrVec *v, char *s) { if (v->count == v->cap) { size_t nc = v->cap ? v->cap * 2 : 8; char **t = (char **)realloc(v->items, nc * sizeof(char *)); if (!t) return -1; v->items = t; v->cap = nc; } v->items[v->count++] = s; return 0; }
static int strvec_push_copy(StrVec *v, const char *s) { char *c = xstrdup(s); if (!c) return -1; if (strvec_push_owned(v, c) != 0) { free(c); return -1; } return 0; }

static void seq_free(Seq *s) { for (size_t i = 0; i < s->len; i++) tok_free(&s->items[i]); free(s->items); memset(s, 0, sizeof(*s)); }
static int seq_push_owned(Seq *s, Tok t) { if (s->len == s->cap) { size_t nc = s->cap ? s->cap * 2 : 16; Tok *p = (Tok *)realloc(s->items, nc * sizeof(Tok)); if (!p) return -1; s->items = p; s->cap = nc; } s->items[s->len++] = t; return 0; }
static int seq_push_copy(Seq *s, Tok t) { Tok c = tok_copy(t.data, t.len); if (!c.data && t.len) return -1; if (seq_push_owned(s, c) != 0) { tok_free(&c); return -1; } return 0; }
static void seqvec_free(SeqVec *v) { for (size_t i = 0; i < v->count; i++) seq_free(&v->items[i]); free(v->items); memset(v, 0, sizeof(*v)); }
static int seqvec_push_owned(SeqVec *v, Seq s) { if (v->count == v->cap) { size_t nc = v->cap ? v->cap * 2 : 16; Seq *p = (Seq *)realloc(v->items, nc * sizeof(Seq)); if (!p) return -1; v->items = p; v->cap = nc; } v->items[v->count++] = s; return 0; }

static void corpus_free(Corpus *c) { for (size_t i = 0; i < c->count; i++) { free(c->items[i].lang); seqvec_free(&c->items[i].docs); } free(c->items); memset(c, 0, sizeof(*c)); }
static LangDocs *corpus_lang(Corpus *c, const char *lang) { for (size_t i = 0; i < c->count; i++) if (strcmp(c->items[i].lang, lang) == 0) return &c->items[i]; return NULL; }
static LangDocs *corpus_lang_add(Corpus *c, const char *lang) {
    LangDocs *e = corpus_lang(c, lang);
    if (e) return e;
    if (c->count == c->cap) { size_t nc = c->cap ? c->cap * 2 : 4; LangDocs *p = (LangDocs *)realloc(c->items, nc * sizeof(LangDocs)); if (!p) return NULL; c->items = p; c->cap = nc; }
    e = &c->items[c->count++]; memset(e, 0, sizeof(*e)); e->lang = xstrdup(lang); if (!e->lang) return NULL; return e;
}
static int corpus_copy(const Corpus *src, Corpus *dst) {
    for (size_t i = 0; i < src->count; i++) {
        LangDocs *ld = corpus_lang_add(dst, src->items[i].lang); if (!ld) return -1;
        for (size_t j = 0; j < src->items[i].docs.count; j++) {
            Seq s = {0};
            for (size_t k = 0; k < src->items[i].docs.items[j].len; k++) if (seq_push_copy(&s, src->items[i].docs.items[j].items[k]) != 0) { seq_free(&s); return -1; }
            if (seqvec_push_owned(&ld->docs, s) != 0) { seq_free(&s); return -1; }
        }
    }
    return 0;
}

static int is_ascii_alpha(unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
static int is_ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }
static int pretokenize(const char *text, const char *mode, StrVec *out) {
    if (strcmp(mode, "none") == 0) return *text ? strvec_push_copy(out, text) : 0;
    if (strcmp(mode, "whitespace") == 0) {
        char *copy = xstrdup(text); if (!copy) return -1; char *s = copy;
        while (*s) { while (*s && isspace((unsigned char)*s)) s++; if (!*s) break; char *st = s; while (*s && !isspace((unsigned char)*s)) s++; char sv = *s; *s = '\0'; if (strvec_push_copy(out, st) != 0) { free(copy); return -1; } if (!sv) break; *s++ = sv; }
        free(copy); return 0;
    }
    int gpt4 = strcmp(mode, "gpt4") == 0; if (strcmp(mode, "gpt2") != 0 && !gpt4) return -1;
    size_t i = 0, n = strlen(text);
    while (i < n) {
        if (text[i] == '\'' && i + 1 < n) { const char *tails[] = {"s","t","re","ve","m","ll","d"}; for (size_t k = 0; k < 7; k++) { size_t l = strlen(tails[k]); if (i + 1 + l <= n && strncmp(text + i + 1, tails[k], l) == 0) { if (strvec_push_owned(out, xstrndup(text + i, 1 + l)) != 0) return -1; i += 1 + l; goto cont; } } }
        size_t start = i; int had_space = 0;
        if (text[i] == ' ' && i + 1 < n && !isspace((unsigned char)text[i + 1])) { had_space = 1; i++; }
        if (i < n && is_ascii_alpha((unsigned char)text[i])) while (i < n && is_ascii_alpha((unsigned char)text[i])) i++;
        else if (i < n && is_ascii_digit((unsigned char)text[i])) { size_t max = gpt4 ? 3 : (size_t)-1, c = 0; while (i < n && is_ascii_digit((unsigned char)text[i]) && c < max) { i++; c++; } }
        else if (i < n && !isspace((unsigned char)text[i])) while (i < n && !isspace((unsigned char)text[i]) && !is_ascii_alpha((unsigned char)text[i]) && !is_ascii_digit((unsigned char)text[i])) i++;
        else while (i < n && isspace((unsigned char)text[i])) i++;
        if (i > start && strvec_push_owned(out, xstrndup(text + start, i - start)) != 0) return -1;
        (void)had_space;
cont: ;
    }
    return 0;
}

static int seq_from_text(const char *text, const char *pretok, Seq *seq) {
    StrVec chunks = {0};
    if (pretokenize(text, pretok, &chunks) != 0) return -1;
    for (size_t c = 0; c < chunks.count; c++) {
        const unsigned char *p = (const unsigned char *)chunks.items[c];
        for (size_t i = 0; p[i]; i++) {
            Tok t = tok_copy(&p[i], 1);
            if (!t.data || seq_push_owned(seq, t) != 0) { tok_free(&t); strvec_free(&chunks); return -1; }
        }
    }
    strvec_free(&chunks);
    return 0;
}

static int parse_lang_path(const char *entry, char **lang, char **path) {
    const char *eq = strchr(entry, '='); if (!eq || eq == entry) return -1;
    *lang = xstrndup(entry, (size_t)(eq - entry)); *path = xstrdup(eq + 1);
    return (*lang && *path) ? 0 : -1;
}
static int load_lang_corpora(StrVec *entries, const char *pretok, size_t max_lines, Corpus *corpus) {
    char line[65536];
    for (size_t e = 0; e < entries->count; e++) {
        char *lang = NULL, *path = NULL; if (parse_lang_path(entries->items[e], &lang, &path) != 0) return -1;
        FILE *fp = fopen(path, "rb"); if (!fp) { perror(path); free(lang); free(path); return -1; }
        LangDocs *ld = corpus_lang_add(corpus, lang); if (!ld) { fclose(fp); free(lang); free(path); return -1; }
        size_t n = 0;
        while (fgets(line, sizeof(line), fp)) {
            if (max_lines && n >= max_lines) break;
            n++;
            size_t ln = strlen(line); while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0';
            Seq seq = {0}; if (seq_from_text(line, pretok, &seq) != 0) { fclose(fp); free(lang); free(path); return -1; }
            if (seq.len && seqvec_push_owned(&ld->docs, seq) != 0) { seq_free(&seq); fclose(fp); free(lang); free(path); return -1; }
            if (!seq.len) seq_free(&seq);
        }
        fclose(fp); free(lang); free(path);
    }
    return 0;
}
static int load_labeled_tsv(const char *path, const char *pretok, size_t max_lines, Corpus *corpus) {
    FILE *fp = fopen(path, "rb"); if (!fp) { perror(path); return -1; }
    char line[65536]; size_t n = 0;
    while (fgets(line, sizeof(line), fp)) {
        if (max_lines && n >= max_lines) break;
        n++;
        size_t ln = strlen(line); while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0';
        if (!line[0]) continue;
        char *tab = strchr(line, '\t'); if (!tab) { fclose(fp); fprintf(stderr, "labeled corpus line %zu must be LANG<TAB>TEXT\n", n); return -1; }
        *tab = '\0'; LangDocs *ld = corpus_lang_add(corpus, line); if (!ld) { fclose(fp); return -1; }
        Seq seq = {0}; if (seq_from_text(tab + 1, pretok, &seq) != 0) { fclose(fp); return -1; }
        if (seq.len && seqvec_push_owned(&ld->docs, seq) != 0) { seq_free(&seq); fclose(fp); return -1; }
        if (!seq.len) seq_free(&seq);
    }
    fclose(fp); return 0;
}

static void pairstats_free(PairStats *ps) { for (size_t i = 0; i < ps->count; i++) { tok_free(&ps->items[i].left); tok_free(&ps->items[i].right); } free(ps->items); memset(ps, 0, sizeof(*ps)); }
static int pairstats_add(PairStats *ps, Tok l, Tok r, size_t f) {
    for (size_t i = 0; i < ps->count; i++) if (tok_eq(ps->items[i].left, l) && tok_eq(ps->items[i].right, r)) { ps->items[i].freq += f; return 0; }
    if (ps->count == ps->cap) { size_t nc = ps->cap ? ps->cap * 2 : 256; PairStat *p = (PairStat *)realloc(ps->items, nc * sizeof(PairStat)); if (!p) return -1; ps->items = p; ps->cap = nc; }
    ps->items[ps->count].left = tok_copy(l.data, l.len); ps->items[ps->count].right = tok_copy(r.data, r.len); ps->items[ps->count].freq = f;
    if (!ps->items[ps->count].left.data || !ps->items[ps->count].right.data) return -1;
    ps->count++;
    return 0;
}
static int pair_counts_docs(const SeqVec *docs, PairStats *ps) { for (size_t d = 0; d < docs->count; d++) for (size_t i = 0; i + 1 < docs->items[d].len; i++) if (pairstats_add(ps, docs->items[d].items[i], docs->items[d].items[i + 1], 1) != 0) return -1; return 0; }
static int count_all_pairs(const Corpus *c, PairStats *ps) { for (size_t i = 0; i < c->count; i++) if (pair_counts_docs(&c->items[i].docs, ps) != 0) return -1; return 0; }
static PairStat *best_pair(PairStats *ps, size_t minf) {
    PairStat *b = NULL;
    for (size_t i = 0; i < ps->count; i++) {
        PairStat *p = &ps->items[i];
        if (!b || p->freq > b->freq || (p->freq == b->freq && (tok_cmp(p->left, b->left) > 0 || (tok_cmp(p->left, b->left) == 0 && tok_cmp(p->right, b->right) > 0)))) b = p;
    }
    return (b && b->freq >= minf) ? b : NULL;
}

static int merge_seq(Seq *s, Tok l, Tok r) {
    Seq n = {0}; Tok joined = tok_concat(l, r); if (!joined.data && joined.len) return -1;
    for (size_t i = 0; i < s->len;) {
        if (i + 1 < s->len && tok_eq(s->items[i], l) && tok_eq(s->items[i + 1], r)) {
            if (seq_push_copy(&n, joined) != 0) { tok_free(&joined); seq_free(&n); return -1; } i += 2;
        } else { if (seq_push_copy(&n, s->items[i]) != 0) { tok_free(&joined); seq_free(&n); return -1; } i++; }
    }
    tok_free(&joined); seq_free(s); *s = n; return 0;
}
static int apply_merge(Corpus *c, Tok l, Tok r) { for (size_t a = 0; a < c->count; a++) for (size_t d = 0; d < c->items[a].docs.count; d++) if (merge_seq(&c->items[a].docs.items[d], l, r) != 0) return -1; return 0; }

static double seq_original_len(const Seq *s, const char *unit) {
    size_t bytes = 0; for (size_t i = 0; i < s->len; i++) bytes += s->items[i].len;
    if (strcmp(unit, "byte") == 0) return (double)bytes;
    if (strcmp(unit, "line") == 0) return 1.0;
    if (strcmp(unit, "char") == 0) {
        size_t chars = 0; for (size_t i = 0; i < s->len; i++) for (size_t j = 0; j < s->items[i].len; j++) if ((s->items[i].data[j] & 0xC0u) != 0x80u) chars++;
        return (double)chars;
    }
    return (double)bytes;
}
static double compression_rate_docs(const SeqVec *docs, const char *unit) {
    double sum = 0.0; size_t n = 0;
    for (size_t d = 0; d < docs->count; d++) if (docs->items[d].len) { sum += seq_original_len(&docs->items[d], unit) / (double)docs->items[d].len; n++; }
    return n ? sum / (double)n : 0.0;
}
static double gini_from_rates(const double *rates, size_t n) {
    double *costs = (double *)malloc(n * sizeof(double)); if (!costs) return 0.0; size_t m = 0; for (size_t i = 0; i < n; i++) if (rates[i] > 0.0) costs[m++] = 1.0 / rates[i];
    for (size_t i = 0; i < m; i++) for (size_t j = i + 1; j < m; j++) if (costs[i] > costs[j]) { double x = costs[i]; costs[i] = costs[j]; costs[j] = x; }
    double total = 0.0, weighted = 0.0; for (size_t i = 0; i < m; i++) { total += costs[i]; weighted += (double)(m - i) * costs[i]; }
    double g = (m && total) ? (1.0 / (double)m) * ((double)m + 1.0 - 2.0 * weighted / total) : 0.0; free(costs); return g;
}
static char *select_language(const Corpus *dev, const double *rates, StrVec *recent, size_t window, size_t limit) {
    size_t best = 0;
    for (size_t i = 1; i < dev->count; i++) if (rates[i] < rates[best] || (rates[i] == rates[best] && strcmp(dev->items[i].lang, dev->items[best].lang) < 0)) best = i;
    if (!window || !limit) return dev->items[best].lang;
    for (size_t pass = 0; pass < dev->count; pass++) {
        size_t bi = (size_t)-1;
        for (size_t i = 0; i < dev->count; i++) {
            size_t cnt = 0, start = recent->count > window ? recent->count - window : 0;
            for (size_t r = start; r < recent->count; r++) if (strcmp(recent->items[r], dev->items[i].lang) == 0) cnt++;
            if (cnt <= limit && (bi == (size_t)-1 || rates[i] < rates[bi] || (rates[i] == rates[bi] && strcmp(dev->items[i].lang, dev->items[bi].lang) < 0))) bi = i;
        }
        if (bi != (size_t)-1) return dev->items[bi].lang;
    }
    return dev->items[best].lang;
}

static void merges_free(MergeTable *m) { for (size_t i = 0; i < m->count; i++) { tok_free(&m->items[i].left); tok_free(&m->items[i].right); tok_free(&m->items[i].joined); } free(m->items); memset(m, 0, sizeof(*m)); }
static int merges_push(MergeTable *m, Tok l, Tok r) {
    if (m->count == m->cap) { size_t nc = m->cap ? m->cap * 2 : 128; Merge *p = (Merge *)realloc(m->items, nc * sizeof(Merge)); if (!p) return -1; m->items = p; m->cap = nc; }
    m->items[m->count].left = tok_copy(l.data, l.len); m->items[m->count].right = tok_copy(r.data, r.len); m->items[m->count].joined = tok_concat(l, r);
    if (!m->items[m->count].left.data || !m->items[m->count].right.data || !m->items[m->count].joined.data) return -1;
    m->count++;
    return 0;
}
static void trace_free(TraceVec *tv) { for (size_t i = 0; i < tv->count; i++) { free(tv->items[i].focus); tok_free(&tv->items[i].left); tok_free(&tv->items[i].right); } free(tv->items); memset(tv, 0, sizeof(*tv)); }
static int trace_push(TraceVec *tv, const char *focus, Tok l, Tok r, size_t freq, double min_cr, double gini) {
    if (tv->count == tv->cap) { size_t nc = tv->cap ? tv->cap * 2 : 64; TraceRow *p = (TraceRow *)realloc(tv->items, nc * sizeof(TraceRow)); if (!p) return -1; tv->items = p; tv->cap = nc; }
    TraceRow *x = &tv->items[tv->count]; memset(x, 0, sizeof(*x)); x->focus = xstrdup(focus); x->left = tok_copy(l.data, l.len); x->right = tok_copy(r.data, r.len); x->pair_count = freq; x->min_cr = min_cr; x->gini = gini;
    if (!x->focus || !x->left.data || !x->right.data) return -1;
    tv->count++;
    return 0;
}
static void model_free(PBPEModel *m) { free(m->pretokenizer); free(m->strategy); merges_free(&m->merges); memset(m, 0, sizeof(*m)); }

static int learn_pbpe(const Corpus *train, const Corpus *dev, size_t num_merges, size_t minf, const char *unit, const char *pretok, const char *strategy, size_t hybrid_global, size_t window, size_t window_limit, PBPEModel *model, TraceVec *trace) {
    Corpus tw = {0}, dw = {0}; if (!train->count || !dev->count || corpus_copy(train, &tw) != 0 || corpus_copy(dev, &dw) != 0) { corpus_free(&tw); corpus_free(&dw); return -1; }
    for (size_t i = 0; i < dw.count; i++) if (!corpus_lang(&tw, dw.items[i].lang)) { fprintf(stderr, "development language missing in training data: %s\n", dw.items[i].lang); corpus_free(&tw); corpus_free(&dw); return -1; }
    model->pretokenizer = xstrdup(pretok); model->strategy = xstrdup(strategy); if (!model->pretokenizer || !model->strategy) { corpus_free(&tw); corpus_free(&dw); return -1; }
    StrVec recent = {0};
    for (size_t k = 0; k < num_merges; k++) {
        double *rates = (double *)calloc(dw.count ? dw.count : 1, sizeof(double)); if (!rates) { corpus_free(&tw); corpus_free(&dw); strvec_free(&recent); return -1; }
        double min_cr = 0.0; for (size_t i = 0; i < dw.count; i++) { rates[i] = compression_rate_docs(&dw.items[i].docs, unit); if (i == 0 || rates[i] < min_cr) min_cr = rates[i]; }
        double gini = gini_from_rates(rates, dw.count);
        int global = strcmp(strategy, "classic") == 0 || (strcmp(strategy, "hybrid") == 0 && k < hybrid_global);
        char *focus = global ? "__global__" : select_language(&dw, rates, &recent, window, window_limit);
        PairStats ps = {0};
        if (global) { if (count_all_pairs(&tw, &ps) != 0) { free(rates); return -1; } }
        else {
            LangDocs *ld = corpus_lang(&tw, focus);
            if (!ld || pair_counts_docs(&ld->docs, &ps) != 0 || ps.count == 0) { pairstats_free(&ps); if (count_all_pairs(&tw, &ps) != 0) { free(rates); return -1; } focus = "__global__"; }
        }
        PairStat *best = best_pair(&ps, minf);
        if (!best) { pairstats_free(&ps); free(rates); break; }
        if (merges_push(&model->merges, best->left, best->right) != 0 || apply_merge(&tw, best->left, best->right) != 0 || apply_merge(&dw, best->left, best->right) != 0 || trace_push(trace, focus, best->left, best->right, best->freq, min_cr, gini) != 0) { pairstats_free(&ps); free(rates); corpus_free(&tw); corpus_free(&dw); strvec_free(&recent); return -1; }
        if (strcmp(focus, "__global__") != 0 && strvec_push_copy(&recent, focus) != 0) { pairstats_free(&ps); free(rates); return -1; }
        pairstats_free(&ps); free(rates);
    }
    corpus_free(&tw); corpus_free(&dw); strvec_free(&recent); return 0;
}

static int encode_bytes_model(const PBPEModel *m, const unsigned char *data, size_t len, Seq *out) {
    for (size_t i = 0; i < len; i++) { Tok t = tok_copy(data + i, 1); if (!t.data || seq_push_owned(out, t) != 0) { tok_free(&t); return -1; } }
    for (size_t mi = 0; mi < m->merges.count && out->len > 1; mi++) if (merge_seq(out, m->merges.items[mi].left, m->merges.items[mi].right) != 0) return -1;
    return 0;
}
static int encode_text_model(const PBPEModel *m, const char *text, Seq *out) {
    StrVec chunks = {0}; if (pretokenize(text, m->pretokenizer, &chunks) != 0) return -1;
    for (size_t i = 0; i < chunks.count; i++) { Seq part = {0}; if (encode_bytes_model(m, (const unsigned char *)chunks.items[i], strlen(chunks.items[i]), &part) != 0) { seq_free(&part); strvec_free(&chunks); return -1; } for (size_t j = 0; j < part.len; j++) { if (seq_push_owned(out, part.items[j]) != 0) { part.items[j].data = NULL; seq_free(&part); strvec_free(&chunks); return -1; } part.items[j].data = NULL; } seq_free(&part); }
    strvec_free(&chunks); return 0;
}

static int tokcounts_add(TokCounts *tc, Tok t) { for (size_t i = 0; i < tc->count; i++) if (tok_eq(tc->items[i].tok, t)) { tc->items[i].count++; return 0; } if (tc->count == tc->cap) { size_t nc = tc->cap ? tc->cap * 2 : 512; TokCount *p = (TokCount *)realloc(tc->items, nc * sizeof(TokCount)); if (!p) return -1; tc->items = p; tc->cap = nc; } tc->items[tc->count].tok = tok_copy(t.data, t.len); tc->items[tc->count].count = 1; if (!tc->items[tc->count].tok.data) return -1; tc->count++; return 0; }
static void tokcounts_free(TokCounts *tc) { for (size_t i = 0; i < tc->count; i++) tok_free(&tc->items[i].tok); free(tc->items); memset(tc, 0, sizeof(*tc)); }
static double renyi(const TokCounts *tc, double alpha) {
    size_t total = 0, maxc = 0; for (size_t i = 0; i < tc->count; i++) { total += tc->items[i].count; if (tc->items[i].count > maxc) maxc = tc->items[i].count; }
    if (!total) return 0.0;
    if (fabs(alpha - 1.0) < 1e-12) { double h = 0.0; for (size_t i = 0; i < tc->count; i++) { double p = (double)tc->items[i].count / (double)total; h -= p * log(p) / log(2.0); } return h; }
    if (isinf(alpha)) return -log((double)maxc / (double)total) / log(2.0);
    double s = 0.0; for (size_t i = 0; i < tc->count; i++) s += pow((double)tc->items[i].count / (double)total, alpha);
    return log(s) / log(2.0) / (1.0 - alpha);
}

static int model_vocab_size(const PBPEModel *m) {
    TokCounts tc = {0};
    for (int i = 0; i < 256; i++) { unsigned char b = (unsigned char)i; Tok t = {&b, 1}; if (tokcounts_add(&tc, t) != 0) { tokcounts_free(&tc); return 256 + (int)m->merges.count; } }
    for (size_t i = 0; i < m->merges.count; i++) { tokcounts_add(&tc, m->merges.items[i].left); tokcounts_add(&tc, m->merges.items[i].right); tokcounts_add(&tc, m->merges.items[i].joined); }
    int n = (int)tc.count; tokcounts_free(&tc); return n;
}

static int write_model(const PBPEModel *m, const char *path, const TraceVec *trace) {
    FILE *out = fopen(path, "wb"); if (!out) { perror(path); return -1; }
    fprintf(out, "{\n  \"version\": \"dm-parity-aware-bpe-2508.04796v2\",\n  \"algorithm\": \"parity-aware-byte-pair-encoding\",\n  \"pretokenizer\": "); json_string(out, m->pretokenizer);
    fprintf(out, ",\n  \"strategy\": "); json_string(out, m->strategy); fprintf(out, ",\n  \"merges\": [\n");
    for (size_t i = 0; i < m->merges.count; i++) { char *a = b64_encode(m->merges.items[i].left), *b = b64_encode(m->merges.items[i].right); fprintf(out, "    ["); json_string(out, a ? a : ""); fprintf(out, ", "); json_string(out, b ? b : ""); fprintf(out, "]%s\n", i + 1 < m->merges.count ? "," : ""); free(a); free(b); }
    fprintf(out, "  ],\n  \"trace\": [\n");
    for (size_t i = 0; i < trace->count; i++) { char *a = b64_encode(trace->items[i].left), *b = b64_encode(trace->items[i].right); fprintf(out, "    {\"step\": %zu, \"focus_language\": ", i + 1); json_string(out, trace->items[i].focus); fprintf(out, ", \"pair\": ["); json_string(out, a ? a : ""); fprintf(out, ", "); json_string(out, b ? b : ""); fprintf(out, "], \"pair_count\": %zu, \"min_cr_before\": %.17g, \"gini_before\": %.17g}%s\n", trace->items[i].pair_count, trace->items[i].min_cr, trace->items[i].gini, i + 1 < trace->count ? "," : ""); free(a); free(b); }
    fprintf(out, "  ]\n}\n"); fclose(out); return 0;
}
static char *slurp(const char *path) { FILE *fp = fopen(path, "rb"); if (!fp) { perror(path); return NULL; } fseek(fp, 0, SEEK_END); long n = ftell(fp); rewind(fp); char *b = (char *)malloc((size_t)n + 1); if (!b) { fclose(fp); return NULL; } size_t r = fread(b, 1, (size_t)n, fp); b[r] = '\0'; fclose(fp); return b; }
static char *json_get_string(const char *data, const char *key) { char pat[128]; snprintf(pat, sizeof(pat), "\"%s\"", key); char *p = strstr(data, pat); if (!p) return NULL; p = strchr(p, ':'); if (!p) return NULL; p = strchr(p, '"'); if (!p) return NULL; p++; char *e = p; while (*e && *e != '"') e++; return xstrndup(p, (size_t)(e - p)); }
static int read_model(const char *path, PBPEModel *m) {
    char *d = slurp(path); if (!d) return -1;
    m->pretokenizer = json_get_string(d, "pretokenizer"); m->strategy = json_get_string(d, "strategy"); if (!m->pretokenizer) m->pretokenizer = xstrdup("none"); if (!m->strategy) m->strategy = xstrdup("parity");
    char *p = strstr(d, "\"merges\""); if (p) p = strchr(p, '['); if (p) p++;
    while (p && *p) { char *pair = strchr(p, '['); char *endarr = strchr(p, ']'); if (!pair || (endarr && endarr < pair)) break; char *q = strchr(pair, '"'); if (!q) break; q++; char *e = strchr(q, '"'); if (!e) break; char *sa = xstrndup(q, (size_t)(e - q)); q = strchr(e + 1, '"'); if (!q) { free(sa); break; } q++; e = strchr(q, '"'); if (!e) { free(sa); break; } char *sb = xstrndup(q, (size_t)(e - q)); Tok a = b64_decode(sa), b = b64_decode(sb); if (!a.data || !b.data || merges_push(&m->merges, a, b) != 0) { free(sa); free(sb); tok_free(&a); tok_free(&b); free(d); return -1; } free(sa); free(sb); tok_free(&a); tok_free(&b); p = e + 1; }
    free(d); return 0;
}

static int encode_corpus_with_model(const PBPEModel *m, StrVec *entries, const char *labeled, Corpus *out) {
    char line[65536];
    if (labeled) {
        FILE *fp = fopen(labeled, "rb"); if (!fp) { perror(labeled); return -1; } size_t n = 0;
        while (fgets(line, sizeof(line), fp)) { n++; size_t ln = strlen(line); while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0'; if (!line[0]) continue; char *tab = strchr(line, '\t'); if (!tab) { fclose(fp); fprintf(stderr, "labeled corpus line %zu must be LANG<TAB>TEXT\n", n); return -1; } *tab = '\0'; LangDocs *ld = corpus_lang_add(out, line); if (!ld) { fclose(fp); return -1; } Seq seq = {0}; if (encode_text_model(m, tab + 1, &seq) != 0) { fclose(fp); return -1; } if (seq.len && seqvec_push_owned(&ld->docs, seq) != 0) { seq_free(&seq); fclose(fp); return -1; } }
        fclose(fp); return 0;
    }
    for (size_t e = 0; e < entries->count; e++) {
        char *lang = NULL, *path = NULL; if (parse_lang_path(entries->items[e], &lang, &path) != 0) return -1;
        FILE *fp = fopen(path, "rb"); if (!fp) { perror(path); free(lang); free(path); return -1; } LangDocs *ld = corpus_lang_add(out, lang); if (!ld) { fclose(fp); free(lang); free(path); return -1; }
        while (fgets(line, sizeof(line), fp)) { size_t ln = strlen(line); while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0'; Seq seq = {0}; if (encode_text_model(m, line, &seq) != 0) { fclose(fp); free(lang); free(path); return -1; } if (seq.len && seqvec_push_owned(&ld->docs, seq) != 0) { seq_free(&seq); fclose(fp); free(lang); free(path); return -1; } }
        fclose(fp); free(lang); free(path);
    }
    return 0;
}

static int print_evaluate(const PBPEModel *m, const Corpus *c, const char *unit) {
    double *rates = (double *)calloc(c->count ? c->count : 1, sizeof(double)); TokCounts freq = {0}; size_t tokens = 0, nbytes = 0;
    if (!rates) return -1;
    for (size_t i = 0; i < c->count; i++) {
        rates[i] = compression_rate_docs(&c->items[i].docs, unit);
        for (size_t d = 0; d < c->items[i].docs.count; d++) for (size_t j = 0; j < c->items[i].docs.items[d].len; j++) { Tok t = c->items[i].docs.items[d].items[j]; tokens++; nbytes += t.len; if (tokcounts_add(&freq, t) != 0) { free(rates); tokcounts_free(&freq); return -1; } }
    }
    double mean = 0.0, mincr = c->count ? rates[0] : 0.0; for (size_t i = 0; i < c->count; i++) { mean += rates[i]; if (rates[i] < mincr) mincr = rates[i]; } if (c->count) mean /= (double)c->count;
    int vocab = model_vocab_size(m);
    printf("{\n  \"languages\": {"); for (size_t i = 0; i < c->count; i++) { if (i) printf(", "); json_string(stdout, c->items[i].lang); printf(": %.12g", rates[i]); }
    printf("},\n  \"min_compression_rate\": %.12g,\n  \"mean_compression_rate\": %.12g,\n  \"tokenizer_fairness_gini\": %.12g,\n  \"tokens\": %zu,\n  \"utf8_bytes\": %zu,\n  \"bytes_per_token\": %.12g,\n  \"vocab_size\": %d,\n  \"vocab_observed\": %zu,\n  \"vocab_utilization\": %.12g,\n  \"type_token_ratio\": %.12g,\n  \"renyi_h1\": %.12g,\n  \"renyi_h2\": %.12g,\n  \"renyi_hinf\": %.12g\n}\n", mincr, mean, gini_from_rates(rates, c->count), tokens, nbytes, tokens ? (double)nbytes / (double)tokens : 0.0, vocab, freq.count, vocab ? (double)freq.count / (double)vocab : 0.0, tokens ? (double)freq.count / (double)tokens : 0.0, renyi(&freq, 1.0), renyi(&freq, 2.0), renyi(&freq, INFINITY));
    free(rates); tokcounts_free(&freq); return 0;
}

static void token_text(FILE *out, Tok t) { for (size_t i = 0; i < t.len; i++) fputc(t.data[i] ? t.data[i] : '?', out); }
static void token_hex(FILE *out, Tok t) { static const char hx[] = "0123456789abcdef"; for (size_t i = 0; i < t.len; i++) { fputc(hx[t.data[i] >> 4], out); fputc(hx[t.data[i] & 15], out); } }
static int hexval(char c) { if (c >= '0' && c <= '9') return c - '0'; if (c >= 'a' && c <= 'f') return c - 'a' + 10; if (c >= 'A' && c <= 'F') return c - 'A' + 10; return -1; }
static Tok hex_decode(const char *s) { size_t n = strlen(s), o = 0; Tok t; t.data = (unsigned char *)malloc(n / 2 + 1); t.len = 0; if (!t.data) return t; for (size_t i = 0; i + 1 < n; i += 2) { int a = hexval(s[i]), b = hexval(s[i + 1]); if (a < 0 || b < 0) { tok_free(&t); return t; } t.data[o++] = (unsigned char)((a << 4) | b); } t.len = o; return t; }

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s parity_bpe train --lang-corpus LANG=PATH ... --merges K -o model\n", prog);
    fprintf(stderr, "       %s parity_bpe train --input-labeled TSV --merges K -o model\n", prog);
    fprintf(stderr, "       %s parity_bpe encode -m model [-i input] [-o output] [--hex-tokens|--base64-tokens]\n", prog);
    fprintf(stderr, "       %s parity_bpe decode -m model [-i input] [-o output] [--base64-tokens]\n", prog);
    fprintf(stderr, "       %s parity_bpe evaluate -m model --lang-corpus LANG=PATH ...\n", prog);
    fprintf(stderr, "       %s parity_bpe trace -m model\n", prog);
}

int dm_parity_bpe_cli(int argc, char **argv) {
    int start = 1; if (argc >= 2 && (strcmp(argv[1], "parity_bpe") == 0 || strcmp(argv[1], "pbpe") == 0 || strcmp(argv[1], "dm_parity_bpe") == 0)) start = 2;
    if (argc <= start) { usage(argv[0]); return 2; }
    const char *cmd = argv[start];
    if (strcmp(cmd, "train") == 0) {
        StrVec train_entries = {0}, dev_entries = {0}; const char *input_labeled = NULL, *dev_labeled = NULL, *out = NULL, *pretok = "none", *unit = "byte", *strategy = "parity"; size_t merges = 0, minf = 2, max_lines = 0, max_dev_lines = 0, hybrid = 0, window = 0, window_limit = 0; int keep_trace = 0, stats = 0;
        for (int i = start + 1; i < argc; i++) {
            if (strcmp(argv[i], "--lang-corpus") == 0 && i + 1 < argc) strvec_push_copy(&train_entries, argv[++i]);
            else if (strcmp(argv[i], "--dev-corpus") == 0 && i + 1 < argc) strvec_push_copy(&dev_entries, argv[++i]);
            else if (strcmp(argv[i], "--input-labeled") == 0 && i + 1 < argc) input_labeled = argv[++i];
            else if (strcmp(argv[i], "--dev-labeled") == 0 && i + 1 < argc) dev_labeled = argv[++i];
            else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) out = argv[++i];
            else if (strcmp(argv[i], "--merges") == 0 && i + 1 < argc) merges = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--min-frequency") == 0 && i + 1 < argc) minf = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--pretokenizer") == 0 && i + 1 < argc) pretok = argv[++i];
            else if (strcmp(argv[i], "--cr-unit") == 0 && i + 1 < argc) unit = argv[++i];
            else if (strcmp(argv[i], "--strategy") == 0 && i + 1 < argc) strategy = argv[++i];
            else if (strcmp(argv[i], "--hybrid-global-merges") == 0 && i + 1 < argc) hybrid = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--window") == 0 && i + 1 < argc) window = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--window-limit") == 0 && i + 1 < argc) window_limit = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--max-lines") == 0 && i + 1 < argc) max_lines = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--max-dev-lines") == 0 && i + 1 < argc) max_dev_lines = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--keep-trace") == 0) keep_trace = 1;
            else if (strcmp(argv[i], "--stats") == 0) stats = 1;
            else if (strcmp(argv[i], "--gpu") == 0) { /* GPU accepted but not used: training uses language steering between merges which serializes the algorithm */ }
            else if (strcmp(argv[i], "--gpu-device") == 0 && i + 1 < argc) { i++; /* skip device arg */ }
            else { usage(argv[0]); strvec_free(&train_entries); strvec_free(&dev_entries); return 2; }
        }
        if (!out || !merges || (!input_labeled && !train_entries.count)) { usage(argv[0]); strvec_free(&train_entries); strvec_free(&dev_entries); return 2; }
        Corpus train = {0}, dev = {0}; int rc = input_labeled ? load_labeled_tsv(input_labeled, pretok, max_lines, &train) : load_lang_corpora(&train_entries, pretok, max_lines, &train);
        if (rc == 0) rc = dev_labeled ? load_labeled_tsv(dev_labeled, pretok, max_dev_lines, &dev) : (dev_entries.count ? load_lang_corpora(&dev_entries, pretok, max_dev_lines, &dev) : corpus_copy(&train, &dev));
        PBPEModel model = {0}; TraceVec trace = {0}; if (rc == 0) rc = learn_pbpe(&train, &dev, merges, minf, unit, pretok, strategy, hybrid, window, window_limit, &model, &trace);
        TraceVec empty = {0}; if (rc == 0) rc = write_model(&model, out, keep_trace ? &trace : &empty);
        if (rc == 0 && stats) { Corpus eval = {0}; encode_corpus_with_model(&model, dev_entries.count ? &dev_entries : &train_entries, dev_labeled ? dev_labeled : input_labeled, &eval); print_evaluate(&model, &eval, unit); corpus_free(&eval); }
        model_free(&model); trace_free(&trace); corpus_free(&train); corpus_free(&dev); strvec_free(&train_entries); strvec_free(&dev_entries); return rc == 0 ? 0 : 1;
    }
    if (strcmp(cmd, "encode") == 0) {
        const char *modelp = NULL, *input = NULL, *output = NULL; int hex = 0, b64 = 0;
        for (int i = start + 1; i < argc; i++) { if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) modelp = argv[++i]; else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) input = argv[++i]; else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) output = argv[++i]; else if (strcmp(argv[i], "--hex-tokens") == 0) hex = 1; else if (strcmp(argv[i], "--base64-tokens") == 0) b64 = 1; else { usage(argv[0]); return 2; } }
        if (!modelp) { usage(argv[0]); return 2; } PBPEModel m = {0}; if (read_model(modelp, &m) != 0) return 1; FILE *in = input ? fopen(input, "rb") : stdin; FILE *outf = output ? fopen(output, "wb") : stdout; if (!in || !outf) { model_free(&m); return 1; }
        char line[65536]; while (fgets(line, sizeof(line), in)) { size_t ln = strlen(line); while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0'; Seq s = {0}; if (encode_text_model(&m, line, &s) != 0) { seq_free(&s); model_free(&m); return 1; } if (hex || b64) { fputc('[', outf); for (size_t i = 0; i < s.len; i++) { if (i) fputs(", ", outf); if (hex) { fputc('"', outf); token_hex(outf, s.items[i]); fputc('"', outf); } else { char *v = b64_encode(s.items[i]); json_string(outf, v ? v : ""); free(v); } } fputs("]\n", outf); } else { for (size_t i = 0; i < s.len; i++) { if (i) fputc(' ', outf); token_text(outf, s.items[i]); } fputc('\n', outf); } seq_free(&s); }
        if (input) fclose(in);
        if (output) fclose(outf);
        model_free(&m);
        return 0;
    }
    if (strcmp(cmd, "decode") == 0) {
        const char *modelp = NULL, *input = NULL, *output = NULL; int b64 = 0; for (int i = start + 1; i < argc; i++) { if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) modelp = argv[++i]; else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) input = argv[++i]; else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) output = argv[++i]; else if (strcmp(argv[i], "--base64-tokens") == 0) b64 = 1; else { usage(argv[0]); return 2; } }
        (void)modelp; FILE *in = input ? fopen(input, "rb") : stdin; FILE *outf = output ? fopen(output, "wb") : stdout; if (!in || !outf) return 1; char line[65536]; while (fgets(line, sizeof(line), in)) { char *p = line; while ((p = strchr(p, '"'))) { char *q = ++p; while (*q && *q != '"') q++; if (!*q) break; char *s = xstrndup(p, (size_t)(q - p)); Tok t = b64 ? b64_decode(s) : hex_decode(s); fwrite(t.data, 1, t.len, outf); tok_free(&t); free(s); p = q + 1; } fputc('\n', outf); } if (input) fclose(in); if (output) fclose(outf); return 0;
    }
    if (strcmp(cmd, "evaluate") == 0) {
        const char *modelp = NULL, *labeled = NULL, *unit = "byte"; StrVec entries = {0}; for (int i = start + 1; i < argc; i++) { if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) modelp = argv[++i]; else if (strcmp(argv[i], "--lang-corpus") == 0 && i + 1 < argc) strvec_push_copy(&entries, argv[++i]); else if (strcmp(argv[i], "--input-labeled") == 0 && i + 1 < argc) labeled = argv[++i]; else if (strcmp(argv[i], "--cr-unit") == 0 && i + 1 < argc) unit = argv[++i]; else { usage(argv[0]); strvec_free(&entries); return 2; } }
        if (!modelp || (!entries.count && !labeled)) { usage(argv[0]); strvec_free(&entries); return 2; } PBPEModel m = {0}; Corpus c = {0}; int rc = read_model(modelp, &m); if (rc == 0) rc = encode_corpus_with_model(&m, &entries, labeled, &c); if (rc == 0) rc = print_evaluate(&m, &c, unit); model_free(&m); corpus_free(&c); strvec_free(&entries); return rc == 0 ? 0 : 1;
    }
    if (strcmp(cmd, "trace") == 0) {
        const char *modelp = NULL; for (int i = start + 1; i < argc; i++) if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) modelp = argv[++i]; else { usage(argv[0]); return 2; }
        if (!modelp) { usage(argv[0]); return 2; } char *d = slurp(modelp); if (!d) return 1; printf("step,focus_language,pair_count,min_cr_before,gini_before\n"); char *p = strstr(d, "\"trace\""); while (p && (p = strstr(p, "\"step\""))) { long step = strtol(strchr(p, ':') + 1, NULL, 10); char *f = json_get_string(p, "focus_language"); char *pc = strstr(p, "\"pair_count\""); char *mc = strstr(p, "\"min_cr_before\""); char *gi = strstr(p, "\"gini_before\""); printf("%ld,%s,%ld,%.17g,%.17g\n", step, f ? f : "", pc ? strtol(strchr(pc, ':') + 1, NULL, 10) : 0, mc ? strtod(strchr(mc, ':') + 1, NULL) : 0.0, gi ? strtod(strchr(gi, ':') + 1, NULL) : 0.0); free(f); p++; } free(d); return 0;
    }
    usage(argv[0]); return 2;
}
