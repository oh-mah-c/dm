#include "algorithms/laga.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#define strtok_r(str, delim, saveptr) strtok_s((str), (delim), (saveptr))
#define mkdir(path, mode) _mkdir(path)
#endif

typedef enum {
    LAGA_TRANSACTION,
    LAGA_SEQUENCE,
    LAGA_TEXT,
    LAGA_TABULAR,
    LAGA_UTILITY
} LAGA_Schema;

typedef struct {
    uint32_t *items;
    double *utils;
    size_t len;
    double total_utility;
    char *raw;
} LAGA_Record;

typedef struct {
    char **tokens;
    size_t count;
    size_t cap;
} LAGA_Dict;

typedef struct {
    uint32_t a;
    uint32_t b;
    size_t count;
    double weight;
} LAGA_Pair;

typedef struct {
    LAGA_Pair *pairs;
    size_t count;
    size_t cap;
} LAGA_PairTable;

typedef struct {
    LAGA_Record *records;
    size_t count;
    size_t cap;
    LAGA_Dict dict;
    LAGA_Schema schema;
    size_t max_len;
    size_t *length_hist;
    size_t length_cap;
    size_t *support;
    double *utility_sum;
    double *feedback;
    size_t *start_count;
    LAGA_PairTable cooc;
    LAGA_PairTable trans;
    LAGA_PairTable real_patterns;
    double avg_entropy;
    double avg_utility;
} LAGA_Layout;

typedef struct {
    double min_support;
    double alpha;
    double tau_copy;
    double tau_quality;
    size_t target_size;
    size_t iterations;
    unsigned int seed;
    double support_tolerance;
    double noise;
    int disable_feedback;
    int disable_anchors;
} LAGA_Params;

typedef struct {
    size_t generated;
    size_t accepted;
    size_t rejected_copy;
    size_t rejected_quality;
    size_t repairs;
    size_t vocab;
    size_t real_records;
    size_t patterns_real;
    size_t patterns_syn;
    double pattern_recall;
    double pattern_precision;
    double support_loss;
    double entropy_loss;
    double copy_rate;
    double avg_nearest_similarity;
    double runtime_sec;
} LAGA_Stats;

static uint32_t rng_next(uint32_t *s) {
    uint32_t x = *s ? *s : 1u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

static double rng_double(uint32_t *s) {
    return (double)rng_next(s) / (double)UINT32_MAX;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (!p) abort();
    memcpy(p, s, n);
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) abort();
    return p;
}

static void *xrealloc(void *p, size_t sz) {
    void *q = realloc(p, sz);
    if (!q) abort();
    return q;
}

static void trim(char *s) {
    char *a = s;
    while (*a && isspace((unsigned char)*a)) a++;
    if (a != s) memmove(s, a, strlen(a) + 1);
    size_t n = strlen(s);
    while (n && isspace((unsigned char)s[n - 1])) s[--n] = '\0';
}

static int mkdir_p(const char *path) {
    char tmp[4096];
    size_t n = strlen(path);
    if (n >= sizeof(tmp)) return -1;
    memcpy(tmp, path, n + 1);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0775) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0775) != 0 && errno != EEXIST) return -1;
    return 0;
}

static void ensure_parent_dir(const char *path) {
    char tmp[4096];
    size_t n = strlen(path);
    if (n >= sizeof(tmp)) return;
    memcpy(tmp, path, n + 1);
    char *slash = strrchr(tmp, '/');
    if (slash) {
        *slash = '\0';
        if (tmp[0]) mkdir_p(tmp);
    }
}

static int dict_get_or_add(LAGA_Dict *d, const char *tok) {
    for (size_t i = 0; i < d->count; i++) {
        if (strcmp(d->tokens[i], tok) == 0) return (int)i;
    }
    if (d->count == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 128;
        d->tokens = (char **)xrealloc(d->tokens, d->cap * sizeof(char *));
    }
    d->tokens[d->count] = xstrdup(tok);
    return (int)d->count++;
}

static void pair_add(LAGA_PairTable *t, uint32_t a, uint32_t b, double delta) {
    if (a > b) {
        uint32_t tmp = a;
        a = b;
        b = tmp;
    }
    for (size_t i = 0; i < t->count; i++) {
        if (t->pairs[i].a == a && t->pairs[i].b == b) {
            t->pairs[i].count++;
            t->pairs[i].weight += delta;
            return;
        }
    }
    if (t->count == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 256;
        t->pairs = (LAGA_Pair *)xrealloc(t->pairs, t->cap * sizeof(LAGA_Pair));
    }
    t->pairs[t->count].a = a;
    t->pairs[t->count].b = b;
    t->pairs[t->count].count = 1;
    t->pairs[t->count].weight = delta;
    t->count++;
}

static void trans_add(LAGA_PairTable *t, uint32_t a, uint32_t b) {
    for (size_t i = 0; i < t->count; i++) {
        if (t->pairs[i].a == a && t->pairs[i].b == b) {
            t->pairs[i].count++;
            t->pairs[i].weight += 1.0;
            return;
        }
    }
    if (t->count == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 256;
        t->pairs = (LAGA_Pair *)xrealloc(t->pairs, t->cap * sizeof(LAGA_Pair));
    }
    t->pairs[t->count].a = a;
    t->pairs[t->count].b = b;
    t->pairs[t->count].count = 1;
    t->pairs[t->count].weight = 1.0;
    t->count++;
}

static size_t pair_count(const LAGA_PairTable *t, uint32_t a, uint32_t b) {
    if (a > b) {
        uint32_t tmp = a;
        a = b;
        b = tmp;
    }
    for (size_t i = 0; i < t->count; i++) {
        if (t->pairs[i].a == a && t->pairs[i].b == b) return t->pairs[i].count;
    }
    return 0;
}

static size_t trans_count(const LAGA_PairTable *t, uint32_t a, uint32_t b) {
    for (size_t i = 0; i < t->count; i++) {
        if (t->pairs[i].a == a && t->pairs[i].b == b) return t->pairs[i].count;
    }
    return 0;
}

static void ensure_vocab_arrays(LAGA_Layout *m) {
    size_t v = m->dict.count ? m->dict.count : 1;
    m->support = (size_t *)xrealloc(m->support, v * sizeof(size_t));
    m->utility_sum = (double *)xrealloc(m->utility_sum, v * sizeof(double));
    m->feedback = (double *)xrealloc(m->feedback, v * sizeof(double));
    m->start_count = (size_t *)xrealloc(m->start_count, v * sizeof(size_t));
    for (size_t i = 0; i < v; i++) {
        m->support[i] = 0;
        m->utility_sum[i] = 0.0;
        m->feedback[i] = 1.0;
        m->start_count[i] = 0;
    }
}

static void record_push(LAGA_Layout *m, LAGA_Record rec) {
    if (m->count == m->cap) {
        m->cap = m->cap ? m->cap * 2 : 256;
        m->records = (LAGA_Record *)xrealloc(m->records, m->cap * sizeof(LAGA_Record));
    }
    m->records[m->count++] = rec;
}

static int looks_utility(const char *line) {
    return strchr(line, ':') != NULL;
}

static LAGA_Schema parse_schema(const char *s, const char *input) {
    if (!s || strcmp(s, "auto") == 0) {
        FILE *fp = fopen(input, "r");
        if (!fp) return LAGA_TRANSACTION;
        char line[4096];
        LAGA_Schema out = LAGA_TRANSACTION;
        if (fgets(line, sizeof(line), fp)) {
            if (strchr(line, ',')) out = LAGA_TABULAR;
            else if (looks_utility(line)) out = LAGA_UTILITY;
            else {
                int text = 0;
                for (char *p = line; *p; p++) {
                    if (isalpha((unsigned char)*p)) {
                        text = 1;
                        break;
                    }
                }
                out = text ? LAGA_TEXT : LAGA_TRANSACTION;
            }
        }
        fclose(fp);
        return out;
    }
    if (strcmp(s, "text") == 0) return LAGA_TEXT;
    if (strcmp(s, "sequence") == 0) return LAGA_SEQUENCE;
    if (strcmp(s, "tabular") == 0) return LAGA_TABULAR;
    if (strcmp(s, "utility") == 0) return LAGA_UTILITY;
    return LAGA_TRANSACTION;
}

static void parse_delimited_tokens(char *line, const char *delim, LAGA_Layout *m, LAGA_Record *rec, int with_column) {
    size_t cap = 16;
    rec->items = (uint32_t *)xcalloc(cap, sizeof(uint32_t));
    char *save = NULL;
    size_t col = 0;
    for (char *tok = strtok_r(line, delim, &save); tok; tok = strtok_r(NULL, delim, &save), col++) {
        trim(tok);
        if (!tok[0]) continue;
        char key[1024];
        if (with_column) snprintf(key, sizeof(key), "c%zu=%s", col, tok);
        else snprintf(key, sizeof(key), "%s", tok);
        int id = dict_get_or_add(&m->dict, key);
        if (rec->len == cap) {
            cap *= 2;
            rec->items = (uint32_t *)xrealloc(rec->items, cap * sizeof(uint32_t));
        }
        rec->items[rec->len++] = (uint32_t)id;
    }
}

static int parse_utility_line(char *line, LAGA_Layout *m, LAGA_Record *rec) {
    char *first = strtok(line, ":");
    char *tu = strtok(NULL, ":");
    char *utils = strtok(NULL, ":");
    if (!first) return -1;
    rec->total_utility = tu ? atof(tu) : 0.0;
    parse_delimited_tokens(first, " \t\r\n", m, rec, 0);
    rec->utils = (double *)xcalloc(rec->len ? rec->len : 1, sizeof(double));
    if (utils) {
        char *save = NULL;
        size_t i = 0;
        for (char *tok = strtok_r(utils, " \t\r\n", &save); tok && i < rec->len; tok = strtok_r(NULL, " \t\r\n", &save)) {
            rec->utils[i++] = atof(tok);
        }
    }
    if (rec->total_utility <= 0.0) {
        for (size_t i = 0; i < rec->len; i++) rec->total_utility += rec->utils[i] > 0.0 ? rec->utils[i] : 1.0;
    }
    return 0;
}

static int load_dataset(const char *path, LAGA_Schema schema, LAGA_Layout *m) {
    memset(m, 0, sizeof(*m));
    m->schema = schema;
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char line[16384];
    int first_tabular = 1;
    while (fgets(line, sizeof(line), fp)) {
        trim(line);
        if (!line[0]) continue;
        if (schema == LAGA_TABULAR && first_tabular && strchr(line, ',')) {
            first_tabular = 0;
            int has_alpha = 0;
            for (char *p = line; *p; p++) if (isalpha((unsigned char)*p)) has_alpha = 1;
            if (has_alpha) continue;
        }
        first_tabular = 0;
        LAGA_Record rec;
        memset(&rec, 0, sizeof(rec));
        rec.raw = xstrdup(line);
        char *work = xstrdup(line);
        if (schema == LAGA_UTILITY) parse_utility_line(work, m, &rec);
        else if (schema == LAGA_TABULAR) parse_delimited_tokens(work, ",", m, &rec, 1);
        else if (schema == LAGA_TEXT || schema == LAGA_SEQUENCE) parse_delimited_tokens(work, " \t\r\n", m, &rec, 0);
        else parse_delimited_tokens(work, " \t\r\n", m, &rec, 0);
        free(work);
        if (rec.len) record_push(m, rec);
        else {
            free(rec.items);
            free(rec.utils);
            free(rec.raw);
        }
    }
    fclose(fp);
    return m->count ? 0 : -1;
}

static int contains_item(const uint32_t *items, size_t n, uint32_t x) {
    for (size_t i = 0; i < n; i++) if (items[i] == x) return 1;
    return 0;
}

static double record_entropy(const LAGA_Record *r) {
    if (!r->len) return 0.0;
    double h = 0.0;
    for (size_t i = 0; i < r->len; i++) {
        size_t c = 0;
        for (size_t j = 0; j < r->len; j++) if (r->items[j] == r->items[i]) c++;
        double p = (double)c / (double)r->len;
        h -= log(p);
    }
    return h / (double)r->len;
}

static void build_layout(LAGA_Layout *m, double min_support) {
    ensure_vocab_arrays(m);
    m->length_cap = 1;
    for (size_t i = 0; i < m->count; i++) if (m->records[i].len + 1 > m->length_cap) m->length_cap = m->records[i].len + 1;
    m->length_hist = (size_t *)xcalloc(m->length_cap, sizeof(size_t));
    double entropy_sum = 0.0;
    double util_sum = 0.0;
    for (size_t r = 0; r < m->count; r++) {
        LAGA_Record *rec = &m->records[r];
        if (rec->len < m->length_cap) m->length_hist[rec->len]++;
        if (rec->len) m->start_count[rec->items[0]]++;
        for (size_t i = 0; i < rec->len; i++) {
            if (!contains_item(rec->items, i, rec->items[i])) m->support[rec->items[i]]++;
            if (rec->utils) m->utility_sum[rec->items[i]] += rec->utils[i];
        }
        for (size_t i = 0; i < rec->len; i++) {
            for (size_t j = i + 1; j < rec->len; j++) {
                if (rec->items[i] != rec->items[j]) pair_add(&m->cooc, rec->items[i], rec->items[j], 1.0);
            }
        }
        for (size_t i = 1; i < rec->len; i++) trans_add(&m->trans, rec->items[i - 1], rec->items[i]);
        entropy_sum += record_entropy(rec);
        util_sum += rec->total_utility;
    }
    m->avg_entropy = entropy_sum / (double)m->count;
    m->avg_utility = util_sum / (double)m->count;
    size_t minsup = min_support < 1.0 ? (size_t)ceil(min_support * (double)m->count) : (size_t)min_support;
    if (minsup < 1) minsup = 1;
    for (uint32_t i = 0; i < m->dict.count; i++) {
        if (m->support[i] >= minsup) pair_add(&m->real_patterns, i, i, (double)m->support[i]);
    }
    for (size_t p = 0; p < m->cooc.count; p++) {
        if (m->cooc.pairs[p].count >= minsup) {
            pair_add(&m->real_patterns, m->cooc.pairs[p].a, m->cooc.pairs[p].b, (double)m->cooc.pairs[p].count);
        }
    }
}

static size_t sample_length(const LAGA_Layout *m, uint32_t *rng) {
    size_t total = 0;
    for (size_t i = 1; i < m->length_cap; i++) total += m->length_hist[i];
    if (!total) return 1;
    size_t pick = (size_t)(rng_double(rng) * (double)total);
    size_t acc = 0;
    for (size_t i = 1; i < m->length_cap; i++) {
        acc += m->length_hist[i];
        if (pick < acc) return i;
    }
    return m->length_cap - 1;
}

static uint32_t weighted_item(const LAGA_Layout *m, uint32_t *rng, uint32_t prev, int use_prev, double alpha) {
    double total = 0.0;
    for (uint32_t i = 0; i < m->dict.count; i++) {
        double w = alpha + (double)m->support[i] * m->feedback[i];
        if (use_prev) w += (double)trans_count(&m->trans, prev, i) * 3.0 + (double)pair_count(&m->cooc, prev, i);
        total += w;
    }
    double pick = rng_double(rng) * total;
    double acc = 0.0;
    for (uint32_t i = 0; i < m->dict.count; i++) {
        double w = alpha + (double)m->support[i] * m->feedback[i];
        if (use_prev) w += (double)trans_count(&m->trans, prev, i) * 3.0 + (double)pair_count(&m->cooc, prev, i);
        acc += w;
        if (pick <= acc) return i;
    }
    return (uint32_t)(m->dict.count - 1);
}

static double record_similarity(const LAGA_Record *a, const LAGA_Record *b) {
    size_t inter = 0, uni = 0;
    for (size_t i = 0; i < a->len; i++) if (!contains_item(a->items, i, a->items[i])) uni++;
    for (size_t i = 0; i < b->len; i++) if (!contains_item(b->items, i, b->items[i]) && !contains_item(a->items, a->len, b->items[i])) uni++;
    for (size_t i = 0; i < a->len; i++) {
        if (!contains_item(a->items, i, a->items[i]) && contains_item(b->items, b->len, a->items[i])) inter++;
    }
    return uni ? (double)inter / (double)uni : 0.0;
}

static double nearest_similarity(const LAGA_Layout *real, const LAGA_Record *cand) {
    double best = 0.0;
    for (size_t i = 0; i < real->count; i++) {
        double s = record_similarity(cand, &real->records[i]);
        if (s > best) best = s;
    }
    return best;
}

static double quality_score(const LAGA_Layout *m, const LAGA_Record *r) {
    if (!r->len) return 0.0;
    double score = 0.0;
    for (size_t i = 0; i < r->len; i++) score += (double)m->support[r->items[i]] / (double)m->count;
    for (size_t i = 1; i < r->len; i++) score += (double)pair_count(&m->trans, r->items[i - 1], r->items[i]) / (double)m->count;
    return score / (double)(2 * r->len);
}

static LAGA_Record generate_record(const LAGA_Layout *m, const LAGA_Params *p, uint32_t *rng) {
    LAGA_Record r;
    memset(&r, 0, sizeof(r));
    r.len = sample_length(m, rng);
    r.items = (uint32_t *)xcalloc(r.len ? r.len : 1, sizeof(uint32_t));
    for (size_t i = 0; i < r.len; i++) r.items[i] = UINT32_MAX;
    int sequential = (m->schema == LAGA_SEQUENCE || m->schema == LAGA_TEXT);
    if (sequential) {
        r.items[0] = weighted_item(m, rng, 0, 0, p->alpha);
        for (size_t i = 1; i < r.len; i++) r.items[i] = weighted_item(m, rng, r.items[i - 1], 1, p->alpha);
    } else {
        size_t pattern_count = m->real_patterns.count;
        if (!p->disable_anchors && pattern_count && rng_double(rng) < 0.65) {
            const LAGA_Pair *pat = &m->real_patterns.pairs[(size_t)(rng_double(rng) * (double)pattern_count) % pattern_count];
            r.items[0] = pat->a;
            if (r.len > 1) r.items[1] = pat->b;
        }
        for (size_t i = 0; i < r.len; i++) {
            if (r.items[i] == UINT32_MAX) {
                uint32_t x;
                size_t guard = 0;
                do {
                    uint32_t prev = i ? r.items[i - 1] : 0;
                    x = weighted_item(m, rng, prev, i > 0, p->alpha);
                } while (contains_item(r.items, i, x) && ++guard < 64);
                r.items[i] = x;
            }
        }
    }
    if (m->schema == LAGA_UTILITY) {
        r.utils = (double *)xcalloc(r.len ? r.len : 1, sizeof(double));
        for (size_t i = 0; i < r.len; i++) {
            double avg = m->support[r.items[i]] ? m->utility_sum[r.items[i]] / (double)m->support[r.items[i]] : 1.0;
            double jitter = 1.0 + p->noise * (2.0 * rng_double(rng) - 1.0);
            if (jitter < 0.1) jitter = 0.1;
            r.utils[i] = avg * jitter;
            r.total_utility += r.utils[i];
        }
    }
    return r;
}

static void free_record(LAGA_Record *r) {
    free(r->items);
    free(r->utils);
    free(r->raw);
    memset(r, 0, sizeof(*r));
}

static void free_layout(LAGA_Layout *m) {
    for (size_t i = 0; i < m->count; i++) free_record(&m->records[i]);
    for (size_t i = 0; i < m->dict.count; i++) free(m->dict.tokens[i]);
    free(m->dict.tokens);
    free(m->records);
    free(m->length_hist);
    free(m->support);
    free(m->utility_sum);
    free(m->feedback);
    free(m->start_count);
    free(m->cooc.pairs);
    free(m->trans.pairs);
    free(m->real_patterns.pairs);
}

static void output_record(FILE *fp, const LAGA_Layout *m, const LAGA_Record *r) {
    if (m->schema == LAGA_UTILITY) {
        for (size_t i = 0; i < r->len; i++) fprintf(fp, "%s%s", i ? " " : "", m->dict.tokens[r->items[i]]);
        fprintf(fp, ":%.6f:", r->total_utility);
        for (size_t i = 0; i < r->len; i++) fprintf(fp, "%s%.6f", i ? " " : "", r->utils ? r->utils[i] : 1.0);
        fprintf(fp, "\n");
    } else if (m->schema == LAGA_TABULAR) {
        for (size_t i = 0; i < r->len; i++) {
            const char *tok = m->dict.tokens[r->items[i]];
            const char *eq = strchr(tok, '=');
            fprintf(fp, "%s%s", i ? "," : "", eq ? eq + 1 : tok);
        }
        fprintf(fp, "\n");
    } else {
        for (size_t i = 0; i < r->len; i++) fprintf(fp, "%s%s", i ? " " : "", m->dict.tokens[r->items[i]]);
        fprintf(fp, "\n");
    }
}

static void evaluate_synthetic(LAGA_Layout *syn, const LAGA_Layout *real, LAGA_Stats *st) {
    size_t inter = 0;
    double support_loss = 0.0;
    for (size_t j = 0; j < syn->real_patterns.count; j++) {
        for (size_t i = 0; i < real->real_patterns.count; i++) {
            if (syn->real_patterns.pairs[j].a == real->real_patterns.pairs[i].a &&
                syn->real_patterns.pairs[j].b == real->real_patterns.pairs[i].b) {
                inter++;
                break;
            }
        }
    }
    for (size_t i = 0; i < real->real_patterns.count; i++) {
        uint32_t a = real->real_patterns.pairs[i].a;
        uint32_t b = real->real_patterns.pairs[i].b;
        size_t sc = (a == b) ? (a < syn->dict.count ? syn->support[a] : 0) : pair_count(&syn->cooc, a, b);
        double rs = (double)real->real_patterns.pairs[i].count / (double)real->count;
        double ss = syn->count ? (double)sc / (double)syn->count : 0.0;
        support_loss += fabs(rs - ss);
    }
    st->patterns_real = real->real_patterns.count;
    st->patterns_syn = syn->real_patterns.count;
    st->pattern_recall = real->real_patterns.count ? (double)inter / (double)real->real_patterns.count : 1.0;
    st->pattern_precision = syn->real_patterns.count ? (double)inter / (double)syn->real_patterns.count : 1.0;
    st->support_loss = real->real_patterns.count ? support_loss / (double)real->real_patterns.count : 0.0;
    st->entropy_loss = fabs(real->avg_entropy - syn->avg_entropy);
}

static int generate_dataset(LAGA_Layout *real, const LAGA_Params *p, const char *output, LAGA_Stats *stats, int update_feedback) {
    ensure_parent_dir(output);
    FILE *fp = fopen(output, "w");
    if (!fp) return -1;
    uint32_t rng = p->seed;
    size_t attempts = 0, max_attempts = p->target_size * 200 + 1000;
    LAGA_Record *accepted = (LAGA_Record *)xcalloc(p->target_size ? p->target_size : 1, sizeof(LAGA_Record));
    while (stats->accepted < p->target_size && attempts++ < max_attempts) {
        stats->generated++;
        LAGA_Record r = generate_record(real, p, &rng);
        double sim = nearest_similarity(real, &r);
        if (sim > p->tau_copy) {
            stats->rejected_copy++;
            free_record(&r);
            continue;
        }
        if (quality_score(real, &r) < p->tau_quality) {
            stats->rejected_quality++;
            if (r.len) r.items[0] = weighted_item(real, &rng, 0, 0, p->alpha);
            stats->repairs++;
            if (quality_score(real, &r) < p->tau_quality) {
                free_record(&r);
                continue;
            }
        }
        output_record(fp, real, &r);
        stats->avg_nearest_similarity += sim;
        accepted[stats->accepted++] = r;
    }
    fclose(fp);
    stats->copy_rate = 0.0;
    stats->avg_nearest_similarity = stats->accepted ? stats->avg_nearest_similarity / (double)stats->accepted : 0.0;
    LAGA_Layout syn;
    memset(&syn, 0, sizeof(syn));
    syn.schema = real->schema;
    syn.dict.count = real->dict.count;
    syn.dict.cap = real->dict.count;
    syn.dict.tokens = (char **)xcalloc(real->dict.count ? real->dict.count : 1, sizeof(char *));
    for (size_t i = 0; i < real->dict.count; i++) syn.dict.tokens[i] = xstrdup(real->dict.tokens[i]);
    for (size_t i = 0; i < stats->accepted; i++) {
        LAGA_Record cp;
        memset(&cp, 0, sizeof(cp));
        cp.len = accepted[i].len;
        cp.items = (uint32_t *)xcalloc(cp.len ? cp.len : 1, sizeof(uint32_t));
        memcpy(cp.items, accepted[i].items, cp.len * sizeof(uint32_t));
        cp.total_utility = accepted[i].total_utility;
        if (accepted[i].utils) {
            cp.utils = (double *)xcalloc(cp.len ? cp.len : 1, sizeof(double));
            memcpy(cp.utils, accepted[i].utils, cp.len * sizeof(double));
        }
        record_push(&syn, cp);
        free_record(&accepted[i]);
    }
    free(accepted);
    build_layout(&syn, p->min_support);
    evaluate_synthetic(&syn, real, stats);
    if (update_feedback && !p->disable_feedback) {
        for (uint32_t item = 0; item < real->dict.count; item++) {
            double real_s = real->count ? (double)real->support[item] / (double)real->count : 0.0;
            double syn_s = syn.count ? (double)syn.support[item] / (double)syn.count : 0.0;
            double diff = real_s - syn_s;
            if (fabs(diff) > p->support_tolerance * 0.25) {
                double factor = 1.0 + diff;
                if (factor < 0.25) factor = 0.25;
                if (factor > 4.0) factor = 4.0;
                real->feedback[item] *= factor;
                if (real->feedback[item] < 0.05) real->feedback[item] = 0.05;
                if (real->feedback[item] > 20.0) real->feedback[item] = 20.0;
            }
        }
    }
    free_layout(&syn);
    return stats->accepted == p->target_size ? 0 : -1;
}

static const char *arg_value(int argc, char **argv, const char *key, const char *fallback) {
    for (int i = 2; i + 1 < argc; i++) if (strcmp(argv[i], key) == 0) return argv[i + 1];
    return fallback;
}

static int has_flag(int argc, char **argv, const char *key) {
    for (int i = 2; i < argc; i++) if (strcmp(argv[i], key) == 0) return 1;
    return 0;
}

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s laga --input <dataset> --output <synthetic_file> [--schema auto|transaction|sequence|text|tabular|utility]\n", prog);
    printf("     [--target-size N] [--minsup ratio|count] [--alpha V] [--tau-copy V]\n");
    printf("     [--tau-quality V] [--iterations N] [--support-tolerance V] [--noise V] [--seed N]\n");
    printf("     [--disable-feedback] [--disable-anchors]\n");
}

int dm_laga_cli(int argc, char **argv) {
    if (has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return 0;
    }
    const char *input = arg_value(argc, argv, "--input", NULL);
    const char *output = arg_value(argc, argv, "--output", NULL);
    if (!input || !output) {
        usage(argv[0]);
        return 1;
    }
    LAGA_Params p;
    p.min_support = atof(arg_value(argc, argv, "--minsup", "0.05"));
    p.alpha = atof(arg_value(argc, argv, "--alpha", "0.5"));
    p.tau_copy = atof(arg_value(argc, argv, "--tau-copy", "0.85"));
    p.tau_quality = atof(arg_value(argc, argv, "--tau-quality", "0.001"));
    p.target_size = (size_t)strtoull(arg_value(argc, argv, "--target-size", "0"), NULL, 10);
    p.iterations = (size_t)strtoull(arg_value(argc, argv, "--iterations", "3"), NULL, 10);
    p.seed = (unsigned int)strtoul(arg_value(argc, argv, "--seed", "42"), NULL, 10);
    p.support_tolerance = atof(arg_value(argc, argv, "--support-tolerance", "0.05"));
    p.noise = atof(arg_value(argc, argv, "--noise", "0.15"));
    p.disable_feedback = has_flag(argc, argv, "--disable-feedback");
    p.disable_anchors = has_flag(argc, argv, "--disable-anchors");
    clock_t start = clock();
    LAGA_Layout real;
    LAGA_Schema schema = parse_schema(arg_value(argc, argv, "--schema", "auto"), input);
    if (load_dataset(input, schema, &real) != 0) {
        fprintf(stderr, "LAGA error: could not load dataset %s\n", input);
        return 1;
    }
    if (p.target_size == 0) p.target_size = real.count;
    build_layout(&real, p.min_support);
    LAGA_Stats best;
    memset(&best, 0, sizeof(best));
    for (size_t it = 0; it < (p.iterations ? p.iterations : 1); it++) {
        char tmp[4096];
        snprintf(tmp, sizeof(tmp), "%s.tmp.%zu", output, it);
        LAGA_Stats st;
        memset(&st, 0, sizeof(st));
        generate_dataset(&real, &p, tmp, &st, 1);
        if (it == 0 || st.support_loss < best.support_loss) {
            if (it != 0) remove(output);
            rename(tmp, output);
            best = st;
        } else {
            remove(tmp);
        }
        if (st.support_loss <= p.support_tolerance) break;
    }
    best.runtime_sec = (double)(clock() - start) / (double)CLOCKS_PER_SEC;
    char stats_path[4096];
    snprintf(stats_path, sizeof(stats_path), "%s.stats", output);
    FILE *sf = fopen(stats_path, "w");
    if (sf) {
        fprintf(sf, "algorithm=LAGA\n");
        fprintf(sf, "input=%s\noutput=%s\n", input, output);
        fprintf(sf, "real_records=%zu\nsynthetic_records=%zu\nvocab=%zu\n", real.count, best.accepted, real.dict.count);
        fprintf(sf, "patterns_real=%zu\npatterns_syn=%zu\n", best.patterns_real, best.patterns_syn);
        fprintf(sf, "pattern_recall=%.10g\npattern_precision=%.10g\nsupport_loss=%.10g\nentropy_loss=%.10g\n", best.pattern_recall, best.pattern_precision, best.support_loss, best.entropy_loss);
        fprintf(sf, "copy_rate=%.10g\navg_nearest_similarity=%.10g\nrejected_copy=%zu\nrejected_quality=%zu\nrepairs=%zu\nruntime_sec=%.10g\n",
                best.copy_rate, best.avg_nearest_similarity, best.rejected_copy, best.rejected_quality, best.repairs, best.runtime_sec);
        fprintf(sf, "disable_feedback=%d\ndisable_anchors=%d\n", p.disable_feedback, p.disable_anchors);
        fclose(sf);
    }
    printf("LAGA Self-Contained Generator\n");
    printf("input=%s\n", input);
    printf("output=%s\n", output);
    printf("stats=%s\n", stats_path);
    printf("real_records=%zu\n", real.count);
    printf("synthetic_records=%zu\n", best.accepted);
    printf("vocab=%zu\n", real.dict.count);
    printf("patterns_real=%zu\n", best.patterns_real);
    printf("patterns_syn=%zu\n", best.patterns_syn);
    printf("pattern_recall=%.6f\n", best.pattern_recall);
    printf("pattern_precision=%.6f\n", best.pattern_precision);
    printf("support_loss=%.6f\n", best.support_loss);
    printf("entropy_loss=%.6f\n", best.entropy_loss);
    printf("copy_rate=%.6f\n", best.copy_rate);
    printf("avg_nearest_similarity=%.6f\n", best.avg_nearest_similarity);
    printf("rejected_copy=%zu\n", best.rejected_copy);
    printf("rejected_quality=%zu\n", best.rejected_quality);
    printf("repairs=%zu\n", best.repairs);
    printf("disable_feedback=%d\n", p.disable_feedback);
    printf("disable_anchors=%d\n", p.disable_anchors);
    printf("runtime_sec=%.6f\n", best.runtime_sec);
    free_layout(&real);
    return best.accepted == p.target_size ? 0 : 2;
}

// Generate data directly to Arena
BenchmarkDataset* dm_laga_generate_ram(DM_Arena* arena, size_t target_bytes, double alpha, double noise) {
    BenchmarkDataset* ds = (BenchmarkDataset*)dm_arena_alloc(arena, sizeof(BenchmarkDataset), 8);
    if (!ds) return NULL;
    
    // We will simulate a Zipfian distribution for items and length
    size_t avg_len = 10;
    size_t num_txns = target_bytes / (avg_len * sizeof(int));
    if (num_txns < 10) num_txns = 10;
    
    ds->txn_count = num_txns;
    ds->max_item_id = 1000;
    ds->total_bytes = target_bytes;
    
    ds->transactions = (int**)dm_arena_alloc(arena, num_txns * sizeof(int*), 8);
    ds->txn_lengths = (size_t*)dm_arena_alloc(arena, num_txns * sizeof(size_t), 8);
    
    uint32_t seed = 42;
    for (size_t i = 0; i < num_txns; i++) {
        // Simple length distribution
        size_t len = 5 + (size_t)(rng_double(&seed) * 10.0);
        ds->txn_lengths[i] = len;
        
        int* items = (int*)dm_arena_alloc(arena, len * sizeof(int), 4);
        ds->transactions[i] = items;
        
        for (size_t j = 0; j < len; j++) {
            // Zipfian-like distribution (simplified for benchmark speed)
            double r = rng_double(&seed);
            int item = (int)(pow(r, alpha) * ds->max_item_id);
            if (item >= ds->max_item_id) item = ds->max_item_id - 1;
            if (item < 0) item = 0;
            items[j] = item;
        }
    }
    
    return ds;
}
