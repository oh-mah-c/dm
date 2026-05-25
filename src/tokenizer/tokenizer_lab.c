#include "tokenizer/tokenizer_lab.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DM_GPU
#include "core/gpu/dm_gpu.h"
#endif

#define LAB_EOW "</w>"

typedef struct { char **items; size_t count, cap; } StrVec;
typedef struct { char *text; size_t freq; } TextCount;
typedef struct { TextCount *items; size_t count, cap; } TextVocab;
typedef struct { char **syms; size_t len, cap, freq; } SymWord;
typedef struct { SymWord *items; size_t count, cap; } SymVocab;
typedef struct { char *left, *right; size_t freq; } PairStat;
typedef struct { PairStat *items; size_t count, cap; } PairStats;
typedef struct { char *left, *right, *joined; } Merge;
typedef struct { Merge *items; size_t count, cap; } MergeTable;
typedef struct { char *pretokenizer; MergeTable merges; } LabModel;

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

static int is_ascii_alpha(unsigned char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
static int is_ascii_digit(unsigned char c) { return c >= '0' && c <= '9'; }
static int pretokenize(const char *text, const char *mode, StrVec *out) {
    if (strcmp(mode, "identity") == 0) return *text ? strvec_push_copy(out, text) : 0;
    int gpt4 = strcmp(mode, "gpt4") == 0;
    int gpt2 = strcmp(mode, "gpt2") == 0;
    int punct = strcmp(mode, "punct") == 0;
    if (!gpt2 && !gpt4 && !punct) return -1;
    size_t i = 0, n = strlen(text);
    while (i < n) {
        if (!punct && text[i] == '\'' && i + 1 < n) {
            const char *tails[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            for (size_t k = 0; k < 7; k++) {
                size_t l = strlen(tails[k]);
                if (i + 1 + l <= n && strncmp(text + i + 1, tails[k], l) == 0) {
                    if (strvec_push_owned(out, xstrndup(text + i, 1 + l)) != 0) return -1;
                    i += 1 + l;
                    goto cont;
                }
            }
        }
        size_t start = i;
        if (!punct && text[i] == ' ' && i + 1 < n && !isspace((unsigned char)text[i + 1])) i++;
        if (i < n && is_ascii_alpha((unsigned char)text[i])) {
            while (i < n && is_ascii_alpha((unsigned char)text[i])) i++;
        } else if (i < n && is_ascii_digit((unsigned char)text[i])) {
            size_t max = (gpt4 || punct) ? 3 : (size_t)-1, c = 0;
            while (i < n && is_ascii_digit((unsigned char)text[i]) && c < max) { i++; c++; }
        } else if (punct && (text[i] == '\r' || text[i] == '\n')) {
            if (text[i] == '\r' && i + 1 < n && text[i + 1] == '\n') i += 2; else i++;
        } else if (punct && i < n && isspace((unsigned char)text[i])) {
            while (i < n && isspace((unsigned char)text[i]) && text[i] != '\r' && text[i] != '\n') i++;
        } else if (punct) {
            size_t l = utf8_len((unsigned char)text[i]); i += l;
        } else if (i < n && !isspace((unsigned char)text[i])) {
            while (i < n && !isspace((unsigned char)text[i]) && !is_ascii_alpha((unsigned char)text[i]) && !is_ascii_digit((unsigned char)text[i])) i++;
        } else {
            while (i < n && isspace((unsigned char)text[i])) i++;
        }
        if (i > start && strvec_push_owned(out, xstrndup(text + start, i - start)) != 0) return -1;
cont: ;
    }
    return 0;
}

static void textvocab_free(TextVocab *v) { for (size_t i = 0; i < v->count; i++) free(v->items[i].text); free(v->items); memset(v, 0, sizeof(*v)); }
static int textvocab_add(TextVocab *v, const char *s, size_t f) {
    for (size_t i = 0; i < v->count; i++) if (strcmp(v->items[i].text, s) == 0) { v->items[i].freq += f; return 0; }
    if (v->count == v->cap) { size_t nc = v->cap ? v->cap * 2 : 256; TextCount *p = (TextCount *)realloc(v->items, nc * sizeof(TextCount)); if (!p) return -1; v->items = p; v->cap = nc; }
    v->items[v->count].text = xstrdup(s); if (!v->items[v->count].text) return -1; v->items[v->count].freq = f; v->count++; return 0;
}

static int split_symbols(const char *text, StrVec *out) {
    const unsigned char *p = (const unsigned char *)text;
    while (*p) {
        size_t n = utf8_len(*p);
        for (size_t i = 1; i < n; i++) if ((p[i] & 0xC0u) != 0x80u) { n = 1; break; }
        char *s = xstrndup((const char *)p, n);
        if (!s || strvec_push_owned(out, s) != 0) { free(s); return -1; }
        p += n;
    }
    return strvec_push_copy(out, LAB_EOW);
}
static void symword_free(SymWord *w) { for (size_t i = 0; i < w->len; i++) free(w->syms[i]); free(w->syms); memset(w, 0, sizeof(*w)); }
static void symvocab_free(SymVocab *v) { for (size_t i = 0; i < v->count; i++) symword_free(&v->items[i]); free(v->items); memset(v, 0, sizeof(*v)); }
static int symvocab_add_from_text(SymVocab *v, const char *text, size_t freq) {
    if (v->count == v->cap) { size_t nc = v->cap ? v->cap * 2 : 256; SymWord *p = (SymWord *)realloc(v->items, nc * sizeof(SymWord)); if (!p) return -1; v->items = p; v->cap = nc; }
    SymWord *w = &v->items[v->count]; memset(w, 0, sizeof(*w)); StrVec s = {0}; if (split_symbols(text, &s) != 0) return -1;
    w->syms = s.items; w->len = s.count; w->cap = s.cap; w->freq = freq; v->count++; return 0;
}

static void pairstats_free(PairStats *ps) { for (size_t i = 0; i < ps->count; i++) { free(ps->items[i].left); free(ps->items[i].right); } free(ps->items); memset(ps, 0, sizeof(*ps)); }
static int pairstats_add(PairStats *ps, const char *l, const char *r, size_t f) {
    for (size_t i = 0; i < ps->count; i++) if (strcmp(ps->items[i].left, l) == 0 && strcmp(ps->items[i].right, r) == 0) { ps->items[i].freq += f; return 0; }
    if (ps->count == ps->cap) { size_t nc = ps->cap ? ps->cap * 2 : 256; PairStat *p = (PairStat *)realloc(ps->items, nc * sizeof(PairStat)); if (!p) return -1; ps->items = p; ps->cap = nc; }
    ps->items[ps->count].left = xstrdup(l); ps->items[ps->count].right = xstrdup(r); ps->items[ps->count].freq = f;
    if (!ps->items[ps->count].left || !ps->items[ps->count].right) return -1;
    ps->count++;
    return 0;
}
static int collect_pairs(const SymVocab *v, PairStats *ps) { for (size_t w = 0; w < v->count; w++) for (size_t i = 0; i + 1 < v->items[w].len; i++) if (pairstats_add(ps, v->items[w].syms[i], v->items[w].syms[i + 1], v->items[w].freq) != 0) return -1; return 0; }
static PairStat *best_pair(PairStats *ps, size_t minf) {
    PairStat *b = NULL;
    for (size_t i = 0; i < ps->count; i++) {
        PairStat *p = &ps->items[i];
        if (!b || p->freq > b->freq || (p->freq == b->freq && (strcmp(p->left, b->left) > 0 || (strcmp(p->left, b->left) == 0 && strcmp(p->right, b->right) > 0)))) b = p;
    }
    return b && b->freq >= minf ? b : NULL;
}

#ifdef DM_GPU
typedef struct { char **names; uint32_t count, cap; } LabIntern;
static void lab_intern_free(LabIntern *t) { for (uint32_t i = 0; i < t->count; i++) free(t->names[i]); free(t->names); memset(t, 0, sizeof(*t)); }
static uint32_t lab_intern_add(LabIntern *t, const char *s) {
    for (uint32_t i = 0; i < t->count; i++) if (strcmp(t->names[i], s) == 0) return i;
    if (t->count == t->cap) { uint32_t nc = t->cap ? t->cap * 2 : 64; char **tmp = (char **)realloc(t->names, nc * sizeof(char *)); if (!tmp) return UINT32_MAX; t->names = tmp; t->cap = nc; }
    t->names[t->count] = xstrdup(s); if (!t->names[t->count]) return UINT32_MAX; return t->count++;
}
static uint32_t lab_intern_lookup(const LabIntern *t, const char *s) { for (uint32_t i = 0; i < t->count; i++) if (strcmp(t->names[i], s) == 0) return i; return UINT32_MAX; }
static int lab_build_gpu_input(const SymVocab *sv, const LabIntern *intern, DmGpuBpeInput *out,
                                uint32_t **sid, uint32_t **wst, uint32_t **wln, uint32_t **wfr) {
    size_t total = 0; for (size_t w = 0; w < sv->count; w++) total += sv->items[w].len;
    *sid = (uint32_t *)malloc(total * sizeof(uint32_t)); *wst = (uint32_t *)malloc(sv->count * sizeof(uint32_t));
    *wln = (uint32_t *)malloc(sv->count * sizeof(uint32_t)); *wfr = (uint32_t *)malloc(sv->count * sizeof(uint32_t));
    if (!*sid || !*wst || !*wln || !*wfr) { free(*sid); free(*wst); free(*wln); free(*wfr); return -1; }
    uint32_t off = 0;
    for (size_t w = 0; w < sv->count; w++) {
        (*wst)[w] = off; (*wln)[w] = (uint32_t)sv->items[w].len; (*wfr)[w] = (uint32_t)sv->items[w].freq;
        for (size_t s = 0; s < sv->items[w].len; s++) {
            uint32_t id = lab_intern_lookup(intern, sv->items[w].syms[s]);
            if (id == UINT32_MAX) { free(*sid); free(*wst); free(*wln); free(*wfr); return -1; }
            (*sid)[off++] = id;
        }
    }
    out->sym_ids = *sid; out->total_syms = total; out->word_starts = *wst;
    out->word_lens = *wln; out->word_freqs = *wfr; out->n_words = sv->count; out->vocab_size = intern->count;
    return 0;
}
static int lab_best_pair_gpu(const uint32_t *pc, uint32_t vsz, const LabIntern *intern, size_t minf,
                              const char **lout, const char **rout) {
    uint32_t best = (uint32_t)minf, ba = UINT32_MAX, bb = UINT32_MAX;
    for (uint32_t a = 0; a < vsz; a++) for (uint32_t b = 0; b < vsz; b++) {
        uint32_t f = pc[(size_t)a * vsz + b];
        if (f > best || (f == best && ba == UINT32_MAX)) { best = f; ba = a; bb = b; }
    }
    if (ba == UINT32_MAX) return 0; *lout = intern->names[ba]; *rout = intern->names[bb]; return 1;
}
#endif /* DM_GPU */

static int symword_merge(SymWord *w, const char *l, const char *r) {
    char **next = (char **)calloc(w->len ? w->len : 1, sizeof(char *)); if (!next) return -1; size_t o = 0;
    for (size_t i = 0; i < w->len;) {
        if (i + 1 < w->len && strcmp(w->syms[i], l) == 0 && strcmp(w->syms[i + 1], r) == 0) { next[o] = concat2(l, r); if (!next[o]) goto bad; i += 2; o++; }
        else { next[o] = xstrdup(w->syms[i]); if (!next[o]) goto bad; i++; o++; }
    }
    for (size_t i = 0; i < w->len; i++) free(w->syms[i]);
    free(w->syms);
    w->syms = next;
    w->len = w->cap = o;
    return 0;
bad:
    for (size_t i = 0; i < o; i++) free(next[i]);
    free(next);
    return -1;
}
static int symvocab_merge(SymVocab *v, const char *l, const char *r) { for (size_t i = 0; i < v->count; i++) if (symword_merge(&v->items[i], l, r) != 0) return -1; return 0; }

static void merges_free(MergeTable *m) { for (size_t i = 0; i < m->count; i++) { free(m->items[i].left); free(m->items[i].right); free(m->items[i].joined); } free(m->items); memset(m, 0, sizeof(*m)); }
static int merges_push(MergeTable *m, const char *l, const char *r) {
    if (m->count == m->cap) { size_t nc = m->cap ? m->cap * 2 : 128; Merge *p = (Merge *)realloc(m->items, nc * sizeof(Merge)); if (!p) return -1; m->items = p; m->cap = nc; }
    m->items[m->count].left = xstrdup(l); m->items[m->count].right = xstrdup(r); m->items[m->count].joined = concat2(l, r);
    if (!m->items[m->count].left || !m->items[m->count].right || !m->items[m->count].joined) return -1;
    m->count++;
    return 0;
}
static void model_free(LabModel *m) { free(m->pretokenizer); merges_free(&m->merges); memset(m, 0, sizeof(*m)); }

static int build_chunk_vocab(char **paths, size_t n, const char *pretok, size_t max_chars, TextVocab *chunks) {
    char line[65536]; size_t remaining = max_chars;
    for (size_t p = 0; p < n; p++) {
        FILE *fp = fopen(paths[p], "rb"); if (!fp) { perror(paths[p]); return -1; }
        while (fgets(line, sizeof(line), fp)) {
            char *text = line;
            if (max_chars) {
                if (!remaining) break;
                if (strlen(line) > remaining) line[remaining] = '\0';
                remaining -= strlen(line);
            }
            StrVec toks = {0}; if (pretokenize(text, pretok, &toks) != 0) { strvec_free(&toks); fclose(fp); return -1; }
            for (size_t i = 0; i < toks.count; i++) if (textvocab_add(chunks, toks.items[i], 1) != 0) { strvec_free(&toks); fclose(fp); return -1; }
            strvec_free(&toks);
        }
        fclose(fp);
        if (max_chars && !remaining) break;
    }
    return 0;
}

static int learn_bpe(const TextVocab *chunks, size_t vocab_size, size_t minf, LabModel *m, void *gpu_ctx) {
    SymVocab sv = {0}; StrVec chars = {0};
    for (size_t i = 0; i < chunks->count; i++) {
        if (symvocab_add_from_text(&sv, chunks->items[i].text, chunks->items[i].freq) != 0) goto bad;
        for (size_t j = 0; j < sv.items[sv.count - 1].len; j++) if (strvec_push_unique(&chars, sv.items[sv.count - 1].syms[j]) != 0) goto bad;
    }
    size_t target = vocab_size > chars.count ? vocab_size - chars.count : 0;

#ifdef DM_GPU
    DmGpuCtx *gpu = (DmGpuCtx *)gpu_ctx;
    LabIntern intern = {0};
    if (gpu && dm_gpu_ready(gpu)) {
        for (size_t i = 0; i < chars.count; i++)
            if (lab_intern_add(&intern, chars.items[i]) == UINT32_MAX) { lab_intern_free(&intern); gpu = NULL; }
    }
#else
    (void)gpu_ctx;
#endif

    for (size_t step = 0; step < target; step++) {
        char *l = NULL, *r = NULL; int found = 0;

#ifdef DM_GPU
        if (gpu && dm_gpu_ready(gpu) && intern.count <= DM_GPU_BPE_MAX_VOCAB) {
            uint32_t *sid = NULL, *wst = NULL, *wln = NULL, *wfr = NULL;
            DmGpuBpeInput inp = {0};
            if (lab_build_gpu_input(&sv, &intern, &inp, &sid, &wst, &wln, &wfr) == 0) {
                size_t vsz = intern.count;
                uint32_t *pc = (uint32_t *)calloc(vsz * vsz, sizeof(uint32_t));
                if (pc && dm_gpu_bpe_pair_count(gpu, &inp, pc) == 0) {
                    const char *lp = NULL, *rp = NULL;
                    if (lab_best_pair_gpu(pc, (uint32_t)vsz, &intern, minf, &lp, &rp)) {
                        l = xstrdup(lp); r = xstrdup(rp); found = (l && r) ? 1 : 0;
                    } else { found = -1; }
                }
                free(pc); free(sid); free(wst); free(wln); free(wfr);
            }
        }
#endif

        if (!found) {
            PairStats ps = {0}; if (collect_pairs(&sv, &ps) != 0) { pairstats_free(&ps); goto bad; }
            PairStat *b = best_pair(&ps, minf);
            if (!b) { pairstats_free(&ps); break; }
            l = xstrdup(b->left); r = xstrdup(b->right); pairstats_free(&ps);
            found = (l && r) ? 1 : 0;
        } else if (found < 0) { break; }

        if (!found || !l || !r || merges_push(&m->merges, l, r) != 0 || symvocab_merge(&sv, l, r) != 0) {
            free(l); free(r); goto bad;
        }
#ifdef DM_GPU
        char *joined = concat2(l, r);
        if (joined) { lab_intern_add(&intern, joined); free(joined); }
#endif
        free(l); free(r);
    }

#ifdef DM_GPU
    lab_intern_free(&intern);
#endif
    strvec_free(&chars); symvocab_free(&sv); return 0;
bad:
#ifdef DM_GPU
    lab_intern_free(&intern);
#endif
    strvec_free(&chars); symvocab_free(&sv); return -1;
}

static void strip_eow(StrVec *v) {
    if (!v->count) return;
    size_t e = strlen(LAB_EOW);
    char *last = v->items[v->count - 1];
    size_t n = strlen(last);
    if (strcmp(last, LAB_EOW) == 0) { free(last); v->count--; }
    else if (n >= e && strcmp(last + n - e, LAB_EOW) == 0) { last[n - e] = '\0'; if (!*last) { free(last); v->count--; } }
}
static int encode_chunk(const LabModel *m, const char *chunk, StrVec *out) {
    if (split_symbols(chunk, out) != 0) return -1;
    for (size_t mi = 0; mi < m->merges.count && out->count > 1; mi++) {
        StrVec next = {0};
        for (size_t i = 0; i < out->count;) {
            Merge *mg = &m->merges.items[mi];
            if (i + 1 < out->count && strcmp(out->items[i], mg->left) == 0 && strcmp(out->items[i + 1], mg->right) == 0) {
                char *j = concat2(out->items[i], out->items[i + 1]);
                if (!j || strvec_push_owned(&next, j) != 0) { free(j); strvec_free(&next); return -1; }
                i += 2;
            } else { if (strvec_push_copy(&next, out->items[i]) != 0) { strvec_free(&next); return -1; } i++; }
        }
        strvec_free(out); *out = next;
    }
    strip_eow(out);
    return 0;
}
static int encode_text(const LabModel *m, const char *text, StrVec *out) {
    StrVec chunks = {0}; if (pretokenize(text, m->pretokenizer, &chunks) != 0) return -1;
    for (size_t i = 0; i < chunks.count; i++) {
        StrVec pcs = {0}; if (encode_chunk(m, chunks.items[i], &pcs) != 0) { strvec_free(&chunks); strvec_free(&pcs); return -1; }
        for (size_t j = 0; j < pcs.count; j++) { char *owned = pcs.items[j]; if (strvec_push_owned(out, owned) != 0) { pcs.items[j] = NULL; strvec_free(&pcs); strvec_free(&chunks); return -1; } pcs.items[j] = NULL; }
        strvec_free(&pcs);
    }
    strvec_free(&chunks);
    return 0;
}

static int tokenfreq_add(TextVocab *v, const char *s) { return textvocab_add(v, s, 1); }
static double renyi_entropy(const TextVocab *freq, double alpha) {
    double total = 0.0; for (size_t i = 0; i < freq->count; i++) total += (double)freq->items[i].freq;
    if (total <= 0.0) return 0.0;
    if (fabs(alpha - 1.0) < 1e-12) { double h = 0.0; for (size_t i = 0; i < freq->count; i++) { double p = (double)freq->items[i].freq / total; h -= p * log(p); } return h; }
    double s = 0.0; for (size_t i = 0; i < freq->count; i++) { double p = (double)freq->items[i].freq / total; s += pow(p, alpha); }
    return log(s) / (1.0 - alpha);
}
static int count_tokens(const LabModel *m, char **paths, size_t n, double *tokens, double *bytes, TextVocab *freq) {
    char line[65536]; *tokens = 0.0; *bytes = 0.0;
    for (size_t p = 0; p < n; p++) {
        FILE *fp = fopen(paths[p], "rb"); if (!fp) { perror(paths[p]); return -1; }
        while (fgets(line, sizeof(line), fp)) {
            *bytes += (double)strlen(line);
            StrVec toks = {0}; if (encode_text(m, line, &toks) != 0) { strvec_free(&toks); fclose(fp); return -1; }
            *tokens += (double)toks.count;
            for (size_t i = 0; i < toks.count; i++) if (tokenfreq_add(freq, toks.items[i]) != 0) { strvec_free(&toks); fclose(fp); return -1; }
            strvec_free(&toks);
        }
        fclose(fp);
    }
    return 0;
}

static int write_model(const LabModel *m, const char *path) {
    FILE *out = fopen(path, "wb"); if (!out) { perror(path); return -1; }
    fprintf(out, "{\n  \"version\": \"dm-tokenizer-lab-dagan-2024\",\n  \"algorithm\": \"bpe\",\n  \"pretokenizer\": ");
    json_string(out, m->pretokenizer);
    fprintf(out, ",\n  \"merges\": [\n");
    for (size_t i = 0; i < m->merges.count; i++) { fprintf(out, "    ["); json_string(out, m->merges.items[i].left); fprintf(out, ", "); json_string(out, m->merges.items[i].right); fprintf(out, "]%s\n", i + 1 < m->merges.count ? "," : ""); }
    fprintf(out, "  ]\n}\n");
    fclose(out);
    return 0;
}
static char *slurp(const char *path) { FILE *fp = fopen(path, "rb"); if (!fp) { perror(path); return NULL; } fseek(fp, 0, SEEK_END); long n = ftell(fp); rewind(fp); char *b = (char *)malloc((size_t)n + 1); if (!b) { fclose(fp); return NULL; } size_t r = fread(b, 1, (size_t)n, fp); b[r] = '\0'; fclose(fp); return b; }
static char *json_get_string(const char *data, const char *key) { char pat[128]; snprintf(pat, sizeof(pat), "\"%s\"", key); char *p = strstr(data, pat); if (!p) return NULL; p = strchr(p, ':'); if (!p) return NULL; p = strchr(p, '"'); if (!p) return NULL; p++; char *e = p; while (*e && *e != '"') e++; return xstrndup(p, (size_t)(e - p)); }
static int read_model(const char *path, LabModel *m) {
    char *d = slurp(path); if (!d) return -1;
    m->pretokenizer = json_get_string(d, "pretokenizer"); if (!m->pretokenizer) { free(d); return -1; }
    char *p = strstr(d, "\"merges\""); if (p) p = strchr(p, '['); if (p) p++;
    while (p && *p) {
        char *pair = strchr(p, '['), *end = strrchr(p, ']');
        if (!pair || (end && end < pair)) break;
        char *q = strchr(pair, '"'); if (!q) break; q++; char *e = strchr(q, '"'); if (!e) break; char *l = xstrndup(q, (size_t)(e - q));
        q = strchr(e + 1, '"'); if (!q) { free(l); break; } q++; e = strchr(q, '"'); if (!e) { free(l); break; } char *r = xstrndup(q, (size_t)(e - q));
        if (!l || !r || merges_push(&m->merges, l, r) != 0) { free(l); free(r); free(d); return -1; }
        free(l); free(r); p = e + 1;
    }
    free(d);
    return 0;
}

static void print_eval(const LabModel *model, char **inputs, size_t n, LabModel *baseline) {
    double toks = 0.0, bytes = 0.0, btoks = 0.0, bbytes = 0.0; TextVocab freq = {0}, bfreq = {0};
    count_tokens(model, inputs, n, &toks, &bytes, &freq);
    if (baseline) count_tokens(baseline, inputs, n, &btoks, &bbytes, &bfreq);
    printf("{\n  \"bytes_per_token\": %.12g,\n  \"nsl\": %.12g,\n  \"renyi_alpha_2p5\": %.12g,\n  \"tokens\": %.12g,\n  \"utf8_bytes\": %.12g,\n  \"vocab_observed\": %.12g\n}\n", toks ? bytes / toks : 0.0, btoks ? toks / btoks : 1.0, renyi_entropy(&freq, 2.5), toks, bytes, (double)freq.count);
    textvocab_free(&freq); textvocab_free(&bfreq);
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s tokenizer_lab train-bpe -i <corpus...> -o model --vocab-size N [--pretokenizer gpt2|gpt4|punct|identity]\n", prog);
    fprintf(stderr, "       %s tokenizer_lab encode -m model [-i input] [-o output] [--json-tokens]\n", prog);
    fprintf(stderr, "       %s tokenizer_lab evaluate -m model -i <eval...> [--baseline model]\n", prog);
    fprintf(stderr, "       %s tokenizer_lab compare -m <models...> -i <eval...> [--baseline model]\n", prog);
    fprintf(stderr, "       %s tokenizer_lab vocab-tradeoff --nsl-csv file --dim D --layers L --heads H --kv-heads K\n", prog);
}

int dm_tokenizer_lab_cli(int argc, char **argv) {
    int start = 1; if (argc >= 2 && (strcmp(argv[1], "tokenizer_lab") == 0 || strcmp(argv[1], "toklab") == 0 || strcmp(argv[1], "dm_tokenizer_lab") == 0)) start = 2;
    if (argc <= start) { usage(argv[0]); return 2; }
    const char *cmd = argv[start];
    if (strcmp(cmd, "train-bpe") == 0) {
        StrVec inputs = {0}; const char *out = NULL, *pretok = "gpt4"; size_t vs = 0, minf = 2, max_chars = 0; int stats = 0, use_gpu = 0, gpu_device = 0;
        for (int i = start + 1; i < argc; i++) {
            if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) while (i + 1 < argc && argv[i + 1][0] != '-') strvec_push_copy(&inputs, argv[++i]);
            else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) out = argv[++i];
            else if (strcmp(argv[i], "--vocab-size") == 0 && i + 1 < argc) vs = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--pretokenizer") == 0 && i + 1 < argc) pretok = argv[++i];
            else if (strcmp(argv[i], "--min-frequency") == 0 && i + 1 < argc) minf = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--max-chars") == 0 && i + 1 < argc) max_chars = (size_t)strtoull(argv[++i], NULL, 10);
            else if (strcmp(argv[i], "--stats") == 0) stats = 1;
            else if (strcmp(argv[i], "--gpu") == 0) use_gpu = 1;
            else if (strcmp(argv[i], "--gpu-device") == 0 && i + 1 < argc) { gpu_device = (int)strtol(argv[++i], NULL, 10); use_gpu = 1; }
            else { usage(argv[0]); strvec_free(&inputs); return 2; }
        }
        if (!inputs.count || !out || !vs) { usage(argv[0]); strvec_free(&inputs); return 2; }
        void *gpu_ctx = NULL;
#ifdef DM_GPU
        if (use_gpu) { gpu_ctx = dm_gpu_create(gpu_device, NULL); if (!gpu_ctx || !dm_gpu_ready((DmGpuCtx *)gpu_ctx)) { fprintf(stderr, "[tokenizer_lab] GPU init failed, falling back to CPU\n"); dm_gpu_destroy((DmGpuCtx *)gpu_ctx); gpu_ctx = NULL; } else { char _dname[256]={0}; dm_gpu_device_name((DmGpuCtx*)gpu_ctx,_dname,sizeof(_dname)); fprintf(stderr,"[tokenizer_lab] GPU: %s\n",_dname); } }
#else
        if (use_gpu) fprintf(stderr, "[tokenizer_lab] built without GPU support, using CPU\n");
#endif
        TextVocab chunks = {0}; LabModel m = {0}; m.pretokenizer = xstrdup(pretok);
        int rc = m.pretokenizer ? build_chunk_vocab(inputs.items, inputs.count, pretok, max_chars, &chunks) : -1;
        if (rc == 0) rc = learn_bpe(&chunks, vs, minf, &m, gpu_ctx);
        if (rc == 0) rc = write_model(&m, out);
        if (rc == 0 && stats) fprintf(stderr, "{\"chunk_types\":%zu,\"chunk_tokens\":%zu,\"pretokenizer\":\"%s\",\"requested_vocab_size\":%zu,\"merges\":%zu}\n", chunks.count, (size_t)0, pretok, vs, m.merges.count);
        model_free(&m); textvocab_free(&chunks); strvec_free(&inputs);
#ifdef DM_GPU
        dm_gpu_destroy((DmGpuCtx *)gpu_ctx);
#endif
        return rc == 0 ? 0 : 1;
    }
    if (strcmp(cmd, "encode") == 0) {
        const char *modelp = NULL, *input = NULL, *output = NULL; int json = 0;
        for (int i = start + 1; i < argc; i++) { if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) modelp = argv[++i]; else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) input = argv[++i]; else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) output = argv[++i]; else if (strcmp(argv[i], "--json-tokens") == 0) json = 1; else { usage(argv[0]); return 2; } }
        LabModel m = {0}; if (!modelp || read_model(modelp, &m) != 0) return 1; FILE *in = input ? fopen(input, "rb") : stdin; FILE *out = output ? fopen(output, "wb") : stdout; if (!in || !out) { model_free(&m); return 1; }
        char line[65536]; while (fgets(line, sizeof(line), in)) { StrVec toks = {0}; if (encode_text(&m, line, &toks) != 0) { strvec_free(&toks); model_free(&m); return 1; } if (json) { fputc('[', out); for (size_t i = 0; i < toks.count; i++) { if (i) fputs(", ", out); json_string(out, toks.items[i]); } fputs("]\n", out); } else { for (size_t i = 0; i < toks.count; i++) { if (i) fputc(' ', out); fputs(toks.items[i], out); } fputc('\n', out); } strvec_free(&toks); }
        if (input) fclose(in);
        if (output) fclose(out);
        model_free(&m);
        return 0;
    }
    if (strcmp(cmd, "evaluate") == 0) {
        const char *modelp = NULL, *basep = NULL; StrVec inputs = {0};
        for (int i = start + 1; i < argc; i++) { if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) && i + 1 < argc) modelp = argv[++i]; else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) while (i + 1 < argc && argv[i + 1][0] != '-') strvec_push_copy(&inputs, argv[++i]); else if (strcmp(argv[i], "--baseline") == 0 && i + 1 < argc) basep = argv[++i]; else { usage(argv[0]); strvec_free(&inputs); return 2; } }
        LabModel m = {0}, b = {0}; if (!modelp || !inputs.count || read_model(modelp, &m) != 0) { strvec_free(&inputs); return 1; } LabModel *bp = NULL; if (basep && read_model(basep, &b) == 0) bp = &b; print_eval(&m, inputs.items, inputs.count, bp); model_free(&m); model_free(&b); strvec_free(&inputs); return 0;
    }
    if (strcmp(cmd, "compare") == 0) {
        StrVec models = {0}, inputs = {0}; const char *basep = NULL;
        for (int i = start + 1; i < argc; i++) { if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--models") == 0) && i + 1 < argc) while (i + 1 < argc && argv[i + 1][0] != '-') strvec_push_copy(&models, argv[++i]); else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) while (i + 1 < argc && argv[i + 1][0] != '-') strvec_push_copy(&inputs, argv[++i]); else if (strcmp(argv[i], "--baseline") == 0 && i + 1 < argc) basep = argv[++i]; else { usage(argv[0]); return 2; } }
        LabModel b = {0}; LabModel *bp = NULL; if (basep && read_model(basep, &b) == 0) bp = &b;
        printf("model,pretokenizer,tokens,utf8_bytes,bytes_per_token,nsl,vocab_observed,renyi_alpha_2p5\n");
        for (size_t i = 0; i < models.count; i++) { LabModel m = {0}; if (read_model(models.items[i], &m) != 0) continue; double toks = 0, bytes = 0, btoks = 0, bbytes = 0; TextVocab freq = {0}, bfreq = {0}; count_tokens(&m, inputs.items, inputs.count, &toks, &bytes, &freq); if (bp) count_tokens(bp, inputs.items, inputs.count, &btoks, &bbytes, &bfreq); printf("%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n", models.items[i], m.pretokenizer, toks, bytes, toks ? bytes / toks : 0.0, btoks ? toks / btoks : 1.0, (double)freq.count, renyi_entropy(&freq, 2.5)); textvocab_free(&freq); textvocab_free(&bfreq); model_free(&m); }
        model_free(&b); strvec_free(&models); strvec_free(&inputs); return 0;
    }
    if (strcmp(cmd, "vocab-tradeoff") == 0) {
        const char *csv = NULL; int dim = 0, layers = 0, heads = 0, kv_heads = 0, batch = 1; double seq32 = 4096.0, slope = 0.04;
        for (int i = start + 1; i < argc; i++) { if (strcmp(argv[i], "--nsl-csv") == 0 && i + 1 < argc) csv = argv[++i]; else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) dim = atoi(argv[++i]); else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) layers = atoi(argv[++i]); else if (strcmp(argv[i], "--heads") == 0 && i + 1 < argc) heads = atoi(argv[++i]); else if (strcmp(argv[i], "--kv-heads") == 0 && i + 1 < argc) kv_heads = atoi(argv[++i]); else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) batch = atoi(argv[++i]); else if (strcmp(argv[i], "--sequence-len-32k") == 0 && i + 1 < argc) seq32 = strtod(argv[++i], NULL); else if (strcmp(argv[i], "--softmax-slope") == 0 && i + 1 < argc) slope = strtod(argv[++i], NULL); else { usage(argv[0]); return 2; } }
        FILE *fp = csv ? fopen(csv, "rb") : NULL; if (!fp) return 1; char line[4096]; if (!fgets(line, sizeof(line), fp)) { fclose(fp); return 1; } double best_mem = 0, best_inf = 0; int best_mem_v = 0, best_inf_v = 0; double best_mem_nsl = 0, best_inf_nsl = 0, best_mem_vocab = 0, best_mem_cache = 0; int first = 1;
        while (fgets(line, sizeof(line), fp)) { int v = 0; double nsl = 0; if (sscanf(line, "%d,%lf", &v, &nsl) != 2) continue; double seq = seq32 * nsl; double vocab_mem = 2.0 * dim * v; double cache = 2.0 * layers * batch * dim * ((double)kv_heads / (double)heads) * seq; double total = vocab_mem + cache; double inf = nsl * (1.0 + slope * ((double)v - 32000.0) / 32000.0); if (first || total < best_mem) { best_mem = total; best_mem_v = v; best_mem_nsl = nsl; best_mem_vocab = vocab_mem; best_mem_cache = cache; } if (first || inf < best_inf) { best_inf = inf; best_inf_v = v; best_inf_nsl = nsl; } first = 0; }
        fclose(fp);
        printf("{\n  \"inference_optimal\": {\n    \"cost_proxy\": %.12g,\n    \"nsl32k\": %.12g,\n    \"vocab_size\": %d\n  },\n  \"inference_optimal_vocab_size\": %d,\n  \"memory_optimal\": {\n    \"cache_params\": %.12g,\n    \"nsl32k\": %.12g,\n    \"total_params_proxy\": %.12g,\n    \"vocab_params\": %.12g,\n    \"vocab_size\": %d\n  },\n  \"memory_optimal_vocab_size\": %d\n}\n", best_inf, best_inf_nsl, best_inf_v, best_inf_v, best_mem_cache, best_mem_nsl, best_mem, best_mem_vocab, best_mem_v, best_mem_v);
        return 0;
    }
    usage(argv[0]); return 2;
}
