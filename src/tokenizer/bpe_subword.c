#include "tokenizer/bpe_subword.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef DM_GPU
#include "gpu/dm_gpu.h"
#endif

#define DM_BPE_EOW "</w>"
#define DM_BPE_SEPARATOR "@@"

typedef struct {
    char **items;
    size_t count;
    size_t cap;
} StrVec;

typedef struct {
    char *word;
    size_t freq;
} WordCount;

typedef struct {
    WordCount *items;
    size_t count;
    size_t cap;
} WordVocab;

typedef struct {
    char **symbols;
    size_t len;
    size_t cap;
    size_t freq;
} SymbolWord;

typedef struct {
    SymbolWord *items;
    size_t count;
    size_t cap;
} SymbolVocab;

typedef struct {
    char *left;
    char *right;
    size_t freq;
} PairStat;

typedef struct {
    PairStat *items;
    size_t count;
    size_t cap;
} PairStats;

typedef struct {
    char *left;
    char *right;
    char *joined;
} Merge;

typedef struct {
    Merge *items;
    size_t count;
    size_t cap;
} MergeTable;

typedef struct {
    char *token;
    size_t count;
} TokenCount;

typedef struct {
    TokenCount *items;
    size_t count;
    size_t cap;
} TokenVocab;

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n + 1);
    return out;
}

static char *xstrndup(const char *s, size_t n) {
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static char *concat2(const char *a, const char *b) {
    size_t na = strlen(a), nb = strlen(b);
    char *out = (char *)malloc(na + nb + 1);
    if (!out) return NULL;
    memcpy(out, a, na);
    memcpy(out + na, b, nb + 1);
    return out;
}

static char *trim(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) *--end = '\0';
    return s;
}

static size_t utf8_len(unsigned char c) {
    if ((c & 0x80u) == 0) return 1;
    if ((c & 0xE0u) == 0xC0u) return 2;
    if ((c & 0xF0u) == 0xE0u) return 3;
    if ((c & 0xF8u) == 0xF0u) return 4;
    return 1;
}

static void strvec_free(StrVec *v) {
    if (!v) return;
    for (size_t i = 0; i < v->count; i++) free(v->items[i]);
    free(v->items);
    v->items = NULL;
    v->count = v->cap = 0;
}

static int strvec_push_owned(StrVec *v, char *s) {
    if (v->count == v->cap) {
        size_t next = v->cap ? v->cap * 2 : 16;
        char **tmp = (char **)realloc(v->items, next * sizeof(char *));
        if (!tmp) return -1;
        v->items = tmp;
        v->cap = next;
    }
    v->items[v->count++] = s;
    return 0;
}

static int strvec_push_copy(StrVec *v, const char *s) {
    char *copy = xstrdup(s);
    if (!copy) return -1;
    if (strvec_push_owned(v, copy) != 0) {
        free(copy);
        return -1;
    }
    return 0;
}

static int word_to_symbols(const char *word, StrVec *symbols) {
    const unsigned char *p = (const unsigned char *)word;
    while (*p) {
        size_t n = utf8_len(*p);
        for (size_t i = 1; i < n; i++) {
            if ((p[i] & 0xC0u) != 0x80u) {
                n = 1;
                break;
            }
        }
        char *sym = xstrndup((const char *)p, n);
        if (!sym || strvec_push_owned(symbols, sym) != 0) {
            free(sym);
            return -1;
        }
        p += n;
    }
    return strvec_push_copy(symbols, DM_BPE_EOW);
}

static void word_vocab_free(WordVocab *v) {
    for (size_t i = 0; i < v->count; i++) free(v->items[i].word);
    free(v->items);
    v->items = NULL;
    v->count = v->cap = 0;
}

static int word_vocab_add(WordVocab *v, const char *word, size_t freq) {
    for (size_t i = 0; i < v->count; i++) {
        if (strcmp(v->items[i].word, word) == 0) {
            v->items[i].freq += freq;
            return 0;
        }
    }
    if (v->count == v->cap) {
        size_t next = v->cap ? v->cap * 2 : 256;
        WordCount *tmp = (WordCount *)realloc(v->items, next * sizeof(WordCount));
        if (!tmp) return -1;
        v->items = tmp;
        v->cap = next;
    }
    v->items[v->count].word = xstrdup(word);
    if (!v->items[v->count].word) return -1;
    v->items[v->count].freq = freq;
    v->count++;
    return 0;
}

static int read_words_from_file(const char *path, WordVocab *vocab) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return -1;
    }
    char line[32768];
    while (fgets(line, sizeof(line), fp)) {
        char *s = trim(line);
        while (*s) {
            while (*s && isspace((unsigned char)*s)) s++;
            if (!*s) break;
            char *start = s;
            while (*s && !isspace((unsigned char)*s)) s++;
            char saved = *s;
            *s = '\0';
            if (*start && word_vocab_add(vocab, start, 1) != 0) {
                fclose(fp);
                return -1;
            }
            if (!saved) break;
            *s++ = saved;
        }
    }
    fclose(fp);
    return 0;
}

static int read_words_from_stdin(WordVocab *vocab) {
    char line[32768];
    while (fgets(line, sizeof(line), stdin)) {
        char *s = trim(line);
        while (*s) {
            while (*s && isspace((unsigned char)*s)) s++;
            if (!*s) break;
            char *start = s;
            while (*s && !isspace((unsigned char)*s)) s++;
            char saved = *s;
            *s = '\0';
            if (*start && word_vocab_add(vocab, start, 1) != 0) return -1;
            if (!saved) break;
            *s++ = saved;
        }
    }
    return 0;
}

static void symbol_word_free(SymbolWord *w) {
    for (size_t i = 0; i < w->len; i++) free(w->symbols[i]);
    free(w->symbols);
    w->symbols = NULL;
    w->len = w->cap = w->freq = 0;
}

static void symbol_vocab_free(SymbolVocab *v) {
    for (size_t i = 0; i < v->count; i++) symbol_word_free(&v->items[i]);
    free(v->items);
    v->items = NULL;
    v->count = v->cap = 0;
}

static int symbol_word_from_word(SymbolWord *out, const char *word, size_t freq) {
    StrVec syms = {0};
    if (word_to_symbols(word, &syms) != 0) return -1;
    out->symbols = syms.items;
    out->len = syms.count;
    out->cap = syms.cap;
    out->freq = freq;
    return 0;
}

static int symbol_vocab_from_words(const WordVocab *words, SymbolVocab *symbols) {
    symbols->items = (SymbolWord *)calloc(words->count ? words->count : 1, sizeof(SymbolWord));
    if (!symbols->items) return -1;
    symbols->cap = words->count;
    for (size_t i = 0; i < words->count; i++) {
        if (symbol_word_from_word(&symbols->items[symbols->count], words->items[i].word, words->items[i].freq) != 0) return -1;
        symbols->count++;
    }
    return 0;
}

static void pair_stats_free(PairStats *stats) {
    for (size_t i = 0; i < stats->count; i++) {
        free(stats->items[i].left);
        free(stats->items[i].right);
    }
    free(stats->items);
    stats->items = NULL;
    stats->count = stats->cap = 0;
}

static int pair_stats_add(PairStats *stats, const char *left, const char *right, size_t freq) {
    for (size_t i = 0; i < stats->count; i++) {
        if (strcmp(stats->items[i].left, left) == 0 && strcmp(stats->items[i].right, right) == 0) {
            stats->items[i].freq += freq;
            return 0;
        }
    }
    if (stats->count == stats->cap) {
        size_t next = stats->cap ? stats->cap * 2 : 256;
        PairStat *tmp = (PairStat *)realloc(stats->items, next * sizeof(PairStat));
        if (!tmp) return -1;
        stats->items = tmp;
        stats->cap = next;
    }
    stats->items[stats->count].left = xstrdup(left);
    stats->items[stats->count].right = xstrdup(right);
    if (!stats->items[stats->count].left || !stats->items[stats->count].right) return -1;
    stats->items[stats->count].freq = freq;
    stats->count++;
    return 0;
}

static int collect_pair_stats(const SymbolVocab *vocab, PairStats *stats) {
    for (size_t w = 0; w < vocab->count; w++) {
        const SymbolWord *word = &vocab->items[w];
        for (size_t i = 0; i + 1 < word->len; i++) {
            if (pair_stats_add(stats, word->symbols[i], word->symbols[i + 1], word->freq) != 0) return -1;
        }
    }
    return 0;
}

static int best_pair(const PairStats *stats, size_t min_freq, const PairStat **best_out) {
    const PairStat *best = NULL;
    for (size_t i = 0; i < stats->count; i++) {
        const PairStat *p = &stats->items[i];
        if (!best || p->freq > best->freq ||
            (p->freq == best->freq && (strcmp(p->left, best->left) > 0 ||
                                       (strcmp(p->left, best->left) == 0 && strcmp(p->right, best->right) > 0)))) {
            best = p;
        }
    }
    if (!best || best->freq < min_freq) return 0;
    *best_out = best;
    return 1;
}

static int symbol_word_merge(SymbolWord *word, const char *left, const char *right) {
    if (word->len < 2) return 0;
    char **next = (char **)calloc(word->len, sizeof(char *));
    if (!next) return -1;
    size_t out = 0;
    for (size_t i = 0; i < word->len;) {
        if (i + 1 < word->len && strcmp(word->symbols[i], left) == 0 && strcmp(word->symbols[i + 1], right) == 0) {
            next[out] = concat2(left, right);
            if (!next[out]) {
                for (size_t k = 0; k < out; k++) free(next[k]);
                free(next);
                return -1;
            }
            i += 2;
            out++;
        } else {
            next[out] = xstrdup(word->symbols[i]);
            if (!next[out]) {
                for (size_t k = 0; k < out; k++) free(next[k]);
                free(next);
                return -1;
            }
            i++;
            out++;
        }
    }
    for (size_t i = 0; i < word->len; i++) free(word->symbols[i]);
    free(word->symbols);
    word->symbols = next;
    word->len = out;
    word->cap = out;
    return 0;
}

static int symbol_vocab_merge(SymbolVocab *vocab, const char *left, const char *right) {
    for (size_t i = 0; i < vocab->count; i++) {
        if (symbol_word_merge(&vocab->items[i], left, right) != 0) return -1;
    }
    return 0;
}

static void merge_table_free(MergeTable *table) {
    for (size_t i = 0; i < table->count; i++) {
        free(table->items[i].left);
        free(table->items[i].right);
        free(table->items[i].joined);
    }
    free(table->items);
    table->items = NULL;
    table->count = table->cap = 0;
}

static int merge_table_push(MergeTable *table, const char *left, const char *right) {
    if (table->count == table->cap) {
        size_t next = table->cap ? table->cap * 2 : 128;
        Merge *tmp = (Merge *)realloc(table->items, next * sizeof(Merge));
        if (!tmp) return -1;
        table->items = tmp;
        table->cap = next;
    }
    Merge *m = &table->items[table->count];
    m->left = xstrdup(left);
    m->right = xstrdup(right);
    m->joined = (m->left && m->right) ? concat2(left, right) : NULL;
    if (!m->left || !m->right || !m->joined) return -1;
    table->count++;
    return 0;
}

#ifdef DM_GPU
/* Map a symbol string to its uint32 ID using symbol_counts as intern table.
 * Returns UINT32_MAX if not found. */
static uint32_t sym_intern_id(const TokenVocab *sym_counts, const char *sym) {
    for (size_t i = 0; i < sym_counts->count; i++) {
        if (strcmp(sym_counts->items[i].token, sym) == 0) return (uint32_t)i;
    }
    return UINT32_MAX;
}

/* Build DmGpuBpeInput flat arrays from SymbolVocab + intern table.
 * Caller must free sym_ids, word_starts, word_lens, word_freqs. */
static int build_gpu_bpe_input(const SymbolVocab *vocab, const TokenVocab *sym_counts,
                                DmGpuBpeInput *out,
                                uint32_t **sym_ids_buf, uint32_t **starts_buf,
                                uint32_t **lens_buf,   uint32_t **freqs_buf) {
    size_t total = 0;
    for (size_t w = 0; w < vocab->count; w++) total += vocab->items[w].len;

    *sym_ids_buf = (uint32_t *)malloc(total * sizeof(uint32_t));
    *starts_buf  = (uint32_t *)malloc(vocab->count * sizeof(uint32_t));
    *lens_buf    = (uint32_t *)malloc(vocab->count * sizeof(uint32_t));
    *freqs_buf   = (uint32_t *)malloc(vocab->count * sizeof(uint32_t));
    if (!*sym_ids_buf || !*starts_buf || !*lens_buf || !*freqs_buf) {
        free(*sym_ids_buf); free(*starts_buf); free(*lens_buf); free(*freqs_buf);
        return -1;
    }

    uint32_t offset = 0;
    for (size_t w = 0; w < vocab->count; w++) {
        const SymbolWord *word = &vocab->items[w];
        (*starts_buf)[w] = offset;
        (*lens_buf)[w]   = (uint32_t)word->len;
        (*freqs_buf)[w]  = (uint32_t)word->freq;
        for (size_t s = 0; s < word->len; s++) {
            uint32_t id = sym_intern_id(sym_counts, word->symbols[s]);
            if (id == UINT32_MAX) { free(*sym_ids_buf); free(*starts_buf); free(*lens_buf); free(*freqs_buf); return -1; }
            (*sym_ids_buf)[offset++] = id;
        }
    }

    out->sym_ids     = *sym_ids_buf;
    out->total_syms  = total;
    out->word_starts = *starts_buf;
    out->word_lens   = *lens_buf;
    out->word_freqs  = *freqs_buf;
    out->n_words     = vocab->count;
    out->vocab_size  = (uint32_t)sym_counts->count;
    return 0;
}

/* Find best pair from dense pair_counts[V*V], returning left/right strings.
 * Returns 1 if found, 0 if nothing exceeds min_freq. */
static int best_pair_gpu(const uint32_t *pair_counts, uint32_t vocab_size,
                          const TokenVocab *sym_counts, size_t min_freq,
                          const char **left_out, const char **right_out) {
    uint32_t best_freq = (uint32_t)min_freq;
    uint32_t best_a = UINT32_MAX, best_b = UINT32_MAX;
    for (uint32_t a = 0; a < vocab_size; a++) {
        for (uint32_t b = 0; b < vocab_size; b++) {
            uint32_t f = pair_counts[(size_t)a * vocab_size + b];
            if (f > best_freq || (f == best_freq && best_a == UINT32_MAX)) {
                best_freq = f;
                best_a = a; best_b = b;
            }
        }
    }
    if (best_a == UINT32_MAX) return 0;
    *left_out  = sym_counts->items[best_a].token;
    *right_out = sym_counts->items[best_b].token;
    return 1;
}
#endif /* DM_GPU */

static int learn_bpe(const WordVocab *words, size_t num_merges, size_t min_freq, MergeTable *merges, TokenVocab *symbol_counts, void *gpu_ctx);

static void token_vocab_free(TokenVocab *v) {
    for (size_t i = 0; i < v->count; i++) free(v->items[i].token);
    free(v->items);
    v->items = NULL;
    v->count = v->cap = 0;
}

static int token_vocab_add(TokenVocab *v, const char *tok, size_t count) {
    for (size_t i = 0; i < v->count; i++) {
        if (strcmp(v->items[i].token, tok) == 0) {
            v->items[i].count += count;
            return 0;
        }
    }
    if (v->count == v->cap) {
        size_t next = v->cap ? v->cap * 2 : 256;
        TokenCount *tmp = (TokenCount *)realloc(v->items, next * sizeof(TokenCount));
        if (!tmp) return -1;
        v->items = tmp;
        v->cap = next;
    }
    v->items[v->count].token = xstrdup(tok);
    if (!v->items[v->count].token) return -1;
    v->items[v->count].count = count;
    v->count++;
    return 0;
}

static int token_vocab_find(const TokenVocab *v, const char *tok, size_t *count) {
    for (size_t i = 0; i < v->count; i++) {
        if (strcmp(v->items[i].token, tok) == 0) {
            if (count) *count = v->items[i].count;
            return 1;
        }
    }
    return 0;
}

static int collect_symbol_counts(const SymbolVocab *vocab, TokenVocab *counts) {
    for (size_t w = 0; w < vocab->count; w++) {
        const SymbolWord *word = &vocab->items[w];
        for (size_t i = 0; i < word->len; i++) {
            if (token_vocab_add(counts, word->symbols[i], word->freq) != 0) return -1;
        }
    }
    return 0;
}

static int learn_bpe(const WordVocab *words, size_t num_merges, size_t min_freq,
                      MergeTable *merges, TokenVocab *symbol_counts, void *gpu_ctx) {
    SymbolVocab vocab = {0};
    if (symbol_vocab_from_words(words, &vocab) != 0) return -1;
    if (collect_symbol_counts(&vocab, symbol_counts) != 0) {
        symbol_vocab_free(&vocab);
        return -1;
    }

#ifdef DM_GPU
    DmGpuCtx *gpu = (DmGpuCtx *)gpu_ctx;
#else
    (void)gpu_ctx;
#endif

    for (size_t i = 0; i < num_merges; i++) {
        char *left = NULL, *right = NULL;
        size_t freq = 0;
        int found = 0;

#ifdef DM_GPU
        /* GPU path: dense pair-count matrix, viable when vocab fits. */
        if (gpu && dm_gpu_ready(gpu) && symbol_counts->count <= DM_GPU_BPE_MAX_VOCAB) {
            uint32_t *sid = NULL, *wst = NULL, *wln = NULL, *wfr = NULL;
            DmGpuBpeInput inp = {0};
            if (build_gpu_bpe_input(&vocab, symbol_counts, &inp, &sid, &wst, &wln, &wfr) == 0) {
                size_t vsz = (size_t)inp.vocab_size;
                uint32_t *pair_counts = (uint32_t *)calloc(vsz * vsz, sizeof(uint32_t));
                if (pair_counts && dm_gpu_bpe_pair_count(gpu, &inp, pair_counts) == 0) {
                    const char *lstr = NULL, *rstr = NULL;
                    if (best_pair_gpu(pair_counts, (uint32_t)vsz, symbol_counts, min_freq, &lstr, &rstr)) {
                        uint32_t best_id_a = sym_intern_id(symbol_counts, lstr);
                        uint32_t best_id_b = sym_intern_id(symbol_counts, rstr);
                        freq  = (best_id_a < vsz && best_id_b < vsz)
                                ? pair_counts[(size_t)best_id_a * vsz + best_id_b] : 0;
                        left  = xstrdup(lstr);
                        right = xstrdup(rstr);
                        found = (left && right) ? 1 : 0;
                    } else {
                        found = -1; /* no pair above min_freq */
                    }
                }
                free(pair_counts);
                free(sid); free(wst); free(wln); free(wfr);
            }
            /* found == 0 here means GPU allocation failed — fall through to CPU */
        }
#endif /* DM_GPU */

        /* CPU fallback (also used when GPU not available or vocab too large). */
        if (!found) {
            PairStats stats = {0};
            if (collect_pair_stats(&vocab, &stats) != 0) {
                pair_stats_free(&stats);
                symbol_vocab_free(&vocab);
                return -1;
            }
            const PairStat *best = NULL;
            int has = best_pair(&stats, min_freq, &best);
            if (has <= 0) { pair_stats_free(&stats); break; }
            left  = xstrdup(best->left);
            right = xstrdup(best->right);
            freq  = best->freq;
            pair_stats_free(&stats);
            found = (left && right) ? 1 : 0;
        } else if (found < 0) {
            /* GPU found no pair above min_freq */
            break;
        }

        if (!found || !left || !right || merge_table_push(merges, left, right) != 0) {
            free(left); free(right);
            symbol_vocab_free(&vocab);
            return -1;
        }
        char *joined = concat2(left, right);
        if (!joined || token_vocab_add(symbol_counts, joined, freq) != 0 ||
            symbol_vocab_merge(&vocab, left, right) != 0) {
            free(joined); free(left); free(right);
            symbol_vocab_free(&vocab);
            return -1;
        }
        free(joined); free(left); free(right);
    }
    symbol_vocab_free(&vocab);
    return 0;
}

static int load_merges(const char *path, MergeTable *table) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return -1;
    }
    char line[8192];
    size_t lineno = 0;
    while (fgets(line, sizeof(line), fp)) {
        lineno++;
        char *s = trim(line);
        if (!*s || *s == '#') continue;
        char *left = strtok(s, " \t\r\n");
        char *right = strtok(NULL, " \t\r\n");
        char *extra = strtok(NULL, " \t\r\n");
        if (!left || !right || extra || merge_table_push(table, left, right) != 0) {
            fprintf(stderr, "invalid BPE merge line at %s:%zu\n", path, lineno);
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}

static void strip_eow_inplace(StrVec *pieces) {
    if (!pieces->count) return;
    char *last = pieces->items[pieces->count - 1];
    size_t n = strlen(last), e = strlen(DM_BPE_EOW);
    if (strcmp(last, DM_BPE_EOW) == 0) {
        free(last);
        pieces->count--;
    } else if (n >= e && strcmp(last + n - e, DM_BPE_EOW) == 0) {
        last[n - e] = '\0';
        if (!*last) {
            free(last);
            pieces->count--;
        }
    }
}

static int encode_word_raw(const MergeTable *merges, const char *word, StrVec *pieces) {
    if (word_to_symbols(word, pieces) != 0) return -1;
    for (size_t m = 0; m < merges->count; m++) {
        StrVec next = {0};
        for (size_t i = 0; i < pieces->count;) {
            if (i + 1 < pieces->count && strcmp(pieces->items[i], merges->items[m].left) == 0 && strcmp(pieces->items[i + 1], merges->items[m].right) == 0) {
                char *joined = concat2(pieces->items[i], pieces->items[i + 1]);
                if (!joined || strvec_push_owned(&next, joined) != 0) {
                    free(joined);
                    strvec_free(&next);
                    return -1;
                }
                i += 2;
            } else {
                if (strvec_push_copy(&next, pieces->items[i]) != 0) {
                    strvec_free(&next);
                    return -1;
                }
                i++;
            }
        }
        strvec_free(pieces);
        *pieces = next;
        if (pieces->count == 1) break;
    }
    strip_eow_inplace(pieces);
    return 0;
}

static const Merge *find_reverse_merge(const MergeTable *merges, const char *piece) {
    for (size_t i = 0; i < merges->count; i++) {
        if (strcmp(merges->items[i].joined, piece) == 0) return &merges->items[i];
    }
    return NULL;
}

static int recursive_split_piece(const char *piece, const MergeTable *merges, const TokenVocab *vocab, size_t threshold, StrVec *out) {
    size_t count = 0;
    if (token_vocab_find(vocab, piece, &count) && count >= threshold) return strvec_push_copy(out, piece);
    const Merge *m = find_reverse_merge(merges, piece);
    if (!m) return strvec_push_copy(out, piece);
    if (recursive_split_piece(m->left, merges, vocab, threshold, out) != 0) return -1;
    return recursive_split_piece(m->right, merges, vocab, threshold, out);
}

static int repair_unknowns(StrVec *pieces, const MergeTable *merges, const TokenVocab *vocab, size_t threshold) {
    StrVec out = {0};
    for (size_t i = 0; i < pieces->count; i++) {
        if (recursive_split_piece(pieces->items[i], merges, vocab, threshold, &out) != 0) {
            strvec_free(&out);
            return -1;
        }
    }
    strvec_free(pieces);
    *pieces = out;
    return 0;
}

static int read_token_vocab(const char *path, TokenVocab *vocab) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return -1;
    }
    char line[8192];
    while (fgets(line, sizeof(line), fp)) {
        char *s = trim(line);
        if (!*s) continue;
        char *tab = strrchr(s, '\t');
        size_t count = 1;
        if (tab) {
            *tab = '\0';
            count = (size_t)strtoull(tab + 1, NULL, 10);
            if (!count) count = 1;
        }
        if (token_vocab_add(vocab, s, count) != 0) {
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}

static int write_merges_file(const MergeTable *merges, const char *path) {
    FILE *out = path ? fopen(path, "wb") : stdout;
    if (!out) {
        perror(path);
        return -1;
    }
    fprintf(out, "#version: dm-bpe-sennrich-2016\n");
    for (size_t i = 0; i < merges->count; i++) fprintf(out, "%s %s\n", merges->items[i].left, merges->items[i].right);
    if (path) fclose(out);
    return 0;
}

static int cmp_token_count(const void *a, const void *b) {
    const TokenCount *x = (const TokenCount *)a;
    const TokenCount *y = (const TokenCount *)b;
    if (x->count < y->count) return 1;
    if (x->count > y->count) return -1;
    return strcmp(x->token, y->token);
}

static int write_token_vocab(TokenVocab *vocab, const char *path) {
    FILE *out = path ? fopen(path, "wb") : stdout;
    if (!out) {
        perror(path);
        return -1;
    }
    qsort(vocab->items, vocab->count, sizeof(TokenCount), cmp_token_count);
    for (size_t i = 0; i < vocab->count; i++) fprintf(out, "%s\t%zu\n", vocab->items[i].token, vocab->items[i].count);
    if (path) fclose(out);
    return 0;
}

static int write_word_vocab(const WordVocab *words, const char *path) {
    TokenVocab tmp = {0};
    for (size_t i = 0; i < words->count; i++) {
        if (token_vocab_add(&tmp, words->items[i].word, words->items[i].freq) != 0) {
            token_vocab_free(&tmp);
            return -1;
        }
    }
    int rc = write_token_vocab(&tmp, path);
    token_vocab_free(&tmp);
    return rc;
}

static int encode_line(const MergeTable *merges, const char *line, const TokenVocab *vocab, size_t threshold, FILE *out) {
    char *copy = xstrdup(line);
    if (!copy) return -1;
    char *s = trim(copy);
    int first = 1;
    while (*s) {
        while (*s && isspace((unsigned char)*s)) s++;
        if (!*s) break;
        char *start = s;
        while (*s && !isspace((unsigned char)*s)) s++;
        char saved = *s;
        *s = '\0';
        StrVec pieces = {0};
        if (encode_word_raw(merges, start, &pieces) != 0 || (vocab && repair_unknowns(&pieces, merges, vocab, threshold) != 0)) {
            strvec_free(&pieces);
            free(copy);
            return -1;
        }
        for (size_t i = 0; i < pieces.count; i++) {
            if (!first) fputc(' ', out);
            fputs(pieces.items[i], out);
            if (i + 1 < pieces.count) fputs(DM_BPE_SEPARATOR, out);
            first = 0;
        }
        strvec_free(&pieces);
        if (!saved) break;
        *s++ = saved;
    }
    fputc('\n', out);
    free(copy);
    return 0;
}

static void decode_text(const char *line, const char *sep, FILE *out) {
    size_t sep_len = strlen(sep);
    size_t n = strlen(line);
    while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) n--;
    for (size_t i = 0; i < n;) {
        if (sep_len && i + sep_len <= n && memcmp(line + i, sep, sep_len) == 0) {
            i += sep_len;
            if (i < n && line[i] == ' ') i++;
        } else {
            fputc(line[i++], out);
        }
    }
    fputc('\n', out);
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s bpe learn-bpe -i <corpus...> -m <merges> [-o codes] [--min-frequency N] [--vocab-out path] [--stats] [--gpu] [--gpu-device N]\n", prog);
    fprintf(stderr, "       %s bpe apply-bpe -c codes [-i input] [-o output] [--vocabulary vocab] [--vocabulary-threshold N]\n", prog);
    fprintf(stderr, "       %s bpe decode [-i input] [-o output] [--separator @@]\n", prog);
    fprintf(stderr, "       %s bpe vocab -i <corpus...> [-o vocab]\n", prog);
}

int dm_bpe_cli(int argc, char **argv) {
    int start = 1;
    if (argc >= 2 && (strcmp(argv[1], "bpe") == 0 || strcmp(argv[1], "dm_bpe") == 0)) start = 2;
    if (argc <= start) {
        usage(argv[0]);
        return 2;
    }
    const char *cmd = argv[start];
    if (strcmp(cmd, "learn-bpe") == 0) {
        StrVec inputs = {0};
        const char *output = NULL;
        const char *vocab_out = NULL;
        size_t merges_n = 0, min_frequency = 2;
        int stats = 0;
        int use_gpu = 0;
        int gpu_device = 0;
        for (int i = start + 1; i < argc; i++) {
            if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) {
                while (i + 1 < argc && argv[i + 1][0] != '-') {
                    if (strvec_push_copy(&inputs, argv[++i]) != 0) return 1;
                }
            } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
                output = argv[++i];
            } else if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--merges") == 0) && i + 1 < argc) {
                merges_n = (size_t)strtoull(argv[++i], NULL, 10);
            } else if (strcmp(argv[i], "--min-frequency") == 0 && i + 1 < argc) {
                min_frequency = (size_t)strtoull(argv[++i], NULL, 10);
            } else if (strcmp(argv[i], "--vocab-out") == 0 && i + 1 < argc) {
                vocab_out = argv[++i];
            } else if (strcmp(argv[i], "--stats") == 0) {
                stats = 1;
            } else if (strcmp(argv[i], "--gpu") == 0) {
                use_gpu = 1;
            } else if (strcmp(argv[i], "--gpu-device") == 0 && i + 1 < argc) {
                gpu_device = (int)strtol(argv[++i], NULL, 10);
                use_gpu = 1;
            } else {
                usage(argv[0]);
                strvec_free(&inputs);
                return 2;
            }
        }
        if (!merges_n) {
            usage(argv[0]);
            strvec_free(&inputs);
            return 2;
        }
        WordVocab words = {0};
        int rc = 0;
        if (inputs.count == 0) rc = read_words_from_stdin(&words);
        for (size_t i = 0; rc == 0 && i < inputs.count; i++) rc = read_words_from_file(inputs.items[i], &words);

        void *gpu_ctx = NULL;
#ifdef DM_GPU
        if (use_gpu) {
            gpu_ctx = dm_gpu_create(gpu_device, NULL);
            if (!gpu_ctx || !dm_gpu_ready((DmGpuCtx *)gpu_ctx)) {
                fprintf(stderr, "[bpe] GPU init failed, falling back to CPU\n");
                dm_gpu_destroy((DmGpuCtx *)gpu_ctx);
                gpu_ctx = NULL;
            } else {
                char _dname[256] = {0};
                dm_gpu_device_name((DmGpuCtx *)gpu_ctx, _dname, sizeof(_dname));
                fprintf(stderr, "[bpe] GPU: %s\n", _dname);
            }
        }
#else
        if (use_gpu) fprintf(stderr, "[bpe] built without GPU support, using CPU\n");
#endif

        MergeTable merges = {0};
        TokenVocab symbols = {0};
        if (rc == 0 && learn_bpe(&words, merges_n, min_frequency, &merges, &symbols, gpu_ctx) != 0) rc = -1;
        if (rc == 0 && write_merges_file(&merges, output) != 0) rc = -1;
        if (rc == 0 && vocab_out && write_token_vocab(&symbols, vocab_out) != 0) rc = -1;
        if (rc == 0 && stats) {
            size_t tokens = 0;
            for (size_t i = 0; i < words.count; i++) tokens += words.items[i].freq;
            fprintf(stderr, "{\"merges\": %zu, \"symbol_types\": %zu, \"word_tokens\": %zu, \"word_types\": %zu}\n", merges.count, symbols.count, tokens, words.count);
        }
        merge_table_free(&merges);
        token_vocab_free(&symbols);
        word_vocab_free(&words);
        strvec_free(&inputs);
#ifdef DM_GPU
        dm_gpu_destroy((DmGpuCtx *)gpu_ctx);
#endif
        return rc == 0 ? 0 : 1;
    }
    if (strcmp(cmd, "apply-bpe") == 0) {
        const char *codes = NULL, *input = NULL, *output = NULL, *vocab_path = NULL;
        size_t threshold = 1;
        for (int i = start + 1; i < argc; i++) {
            if ((strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--codes") == 0) && i + 1 < argc) codes = argv[++i];
            else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) input = argv[++i];
            else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) output = argv[++i];
            else if (strcmp(argv[i], "--vocabulary") == 0 && i + 1 < argc) vocab_path = argv[++i];
            else if (strcmp(argv[i], "--vocabulary-threshold") == 0 && i + 1 < argc) threshold = (size_t)strtoull(argv[++i], NULL, 10);
            else {
                usage(argv[0]);
                return 2;
            }
        }
        if (!codes) {
            usage(argv[0]);
            return 2;
        }
        MergeTable merges = {0};
        TokenVocab vocab = {0};
        int has_vocab = 0;
        if (load_merges(codes, &merges) != 0) return 1;
        if (vocab_path) {
            if (read_token_vocab(vocab_path, &vocab) != 0) {
                merge_table_free(&merges);
                return 1;
            }
            has_vocab = 1;
        }
        FILE *in = input ? fopen(input, "rb") : stdin;
        FILE *out = output ? fopen(output, "wb") : stdout;
        if (!in || !out) {
            if (input && !in) perror(input);
            if (output && !out) perror(output);
            if (in && input) fclose(in);
            if (out && output) fclose(out);
            merge_table_free(&merges);
            token_vocab_free(&vocab);
            return 1;
        }
        char line[32768];
        int rc = 0;
        while (fgets(line, sizeof(line), in)) {
            if (encode_line(&merges, line, has_vocab ? &vocab : NULL, threshold, out) != 0) {
                rc = 1;
                break;
            }
        }
        if (input) fclose(in);
        if (output) fclose(out);
        merge_table_free(&merges);
        token_vocab_free(&vocab);
        return rc;
    }
    if (strcmp(cmd, "decode") == 0) {
        const char *input = NULL, *output = NULL, *sep = DM_BPE_SEPARATOR;
        for (int i = start + 1; i < argc; i++) {
            if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) input = argv[++i];
            else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) output = argv[++i];
            else if (strcmp(argv[i], "--separator") == 0 && i + 1 < argc) sep = argv[++i];
            else {
                usage(argv[0]);
                return 2;
            }
        }
        FILE *in = input ? fopen(input, "rb") : stdin;
        FILE *out = output ? fopen(output, "wb") : stdout;
        if (!in || !out) return 1;
        char line[32768];
        while (fgets(line, sizeof(line), in)) decode_text(line, sep, out);
        if (input) fclose(in);
        if (output) fclose(out);
        return 0;
    }
    if (strcmp(cmd, "vocab") == 0) {
        StrVec inputs = {0};
        const char *output = NULL;
        for (int i = start + 1; i < argc; i++) {
            if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) {
                while (i + 1 < argc && argv[i + 1][0] != '-') {
                    if (strvec_push_copy(&inputs, argv[++i]) != 0) return 1;
                }
            } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
                output = argv[++i];
            } else {
                usage(argv[0]);
                strvec_free(&inputs);
                return 2;
            }
        }
        WordVocab words = {0};
        int rc = 0;
        if (inputs.count == 0) rc = read_words_from_stdin(&words);
        for (size_t i = 0; rc == 0 && i < inputs.count; i++) rc = read_words_from_file(inputs.items[i], &words);
        if (rc == 0 && write_word_vocab(&words, output) != 0) rc = -1;
        word_vocab_free(&words);
        strvec_free(&inputs);
        return rc == 0 ? 0 : 1;
    }
    usage(argv[0]);
    return 2;
}
