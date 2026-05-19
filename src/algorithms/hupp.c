#define _GNU_SOURCE
#include "algorithms/hupp.h"
#include "core/dm_portability.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HUPP_MAX_CANDIDATES 8
#define HUPP_HASH_SIZE 262144u

typedef struct {
    uint32_t tid;
    uint16_t pos;
    uint64_t feas_mask;
    float best_gain;
    float exact_acc;
    float acc_ub;
    float rem_ic;
} HUPPEntry;

typedef struct {
    HUPPEntry *entries;
    size_t count;
    size_t cap;
    double omega;
    uint32_t item;
} HUPPList;

typedef struct {
    char *name;
    uint32_t df;
    uint32_t rank;
    double omega;
    uint8_t structural;
    uint8_t critical;
} HUPPConcept;

typedef struct {
    uint32_t *items;
    uint64_t *keep_masks;
    float *suffix_ic;
    uint16_t item_count;
    uint16_t token_count;
    uint8_t k;
    float gain[HUPP_MAX_CANDIDATES];
    float acc[HUPP_MAX_CANDIDATES];
    float saving[HUPP_MAX_CANDIDATES];
    float latency[HUPP_MAX_CANDIDATES];
    float cache[HUPP_MAX_CANDIDATES];
} HUPPTransaction;

typedef struct HUPPDictNode {
    char *key;
    uint32_t id;
    struct HUPPDictNode *next;
} HUPPDictNode;

typedef struct {
    HUPPConcept *concepts;
    size_t concept_count;
    size_t concept_cap;
    HUPPDictNode **buckets;
    HUPPTransaction *tx;
    size_t tx_count;
    size_t tx_cap;
    size_t total_tokens;
} HUPPDB;

typedef struct {
    HUPPDB *db;
    HUPPList *singletons;
    uint32_t *frequent_items;
    size_t frequent_count;
    uint32_t minsup;
    double theta;
    double alpha_min;
    size_t max_patterns;
    size_t max_depth;
    double max_seconds;
    clock_t started;
    HUPPStats *stats;
} HUPPContext;

typedef struct {
    uint32_t item;
    uint32_t support;
    double omega;
} HUPPOrderItem;

static char *hupp_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *out = (char *)malloc(n);
    if (out) memcpy(out, s, n);
    return out;
}

static double clamp01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

static unsigned hash_string(const char *s) {
    unsigned h = 2166136261u;
    while (*s) {
        h ^= (unsigned char)*s++;
        h *= 16777619u;
    }
    return h & (HUPP_HASH_SIZE - 1u);
}

static int db_init(HUPPDB *db) {
    memset(db, 0, sizeof(*db));
    db->buckets = (HUPPDictNode **)calloc(HUPP_HASH_SIZE, sizeof(HUPPDictNode *));
    return db->buckets ? 0 : -1;
}

static void tx_free(HUPPTransaction *t) {
    free(t->items);
    free(t->keep_masks);
    free(t->suffix_ic);
}

static void db_free(HUPPDB *db) {
    if (!db) return;
    for (size_t i = 0; i < db->tx_count; i++) tx_free(&db->tx[i]);
    free(db->tx);
    for (size_t i = 0; i < db->concept_count; i++) free(db->concepts[i].name);
    free(db->concepts);
    if (db->buckets) {
        for (size_t b = 0; b < HUPP_HASH_SIZE; b++) {
            HUPPDictNode *n = db->buckets[b];
            while (n) {
                HUPPDictNode *next = n->next;
                free(n);
                n = next;
            }
        }
    }
    free(db->buckets);
}

static uint8_t is_structural_name(const char *name) {
    return strncmp(name, "kw:", 3) != 0;
}

static uint8_t is_critical_name(const char *name) {
    return strncmp(name, "cat:", 4) == 0 || strncmp(name, "task:", 5) == 0 ||
           strncmp(name, "fmt:", 4) == 0 || strncmp(name, "lang:", 5) == 0 ||
           strcmp(name, "has_context") == 0 || strcmp(name, "code_task") == 0 ||
           strncmp(name, "constraint:", 11) == 0;
}

static int dict_get_or_add(HUPPDB *db, const char *key, uint32_t *id) {
    unsigned h = hash_string(key);
    for (HUPPDictNode *n = db->buckets[h]; n; n = n->next) {
        if (strcmp(n->key, key) == 0) {
            *id = n->id;
            return 0;
        }
    }
    if (db->concept_count >= db->concept_cap) {
        size_t next = db->concept_cap ? db->concept_cap * 2 : 4096;
        HUPPConcept *tmp = (HUPPConcept *)realloc(db->concepts, next * sizeof(HUPPConcept));
        if (!tmp) return -1;
        db->concepts = tmp;
        db->concept_cap = next;
    }
    char *copy = hupp_strdup(key);
    HUPPDictNode *node = (HUPPDictNode *)malloc(sizeof(HUPPDictNode));
    if (!copy || !node) {
        free(copy);
        free(node);
        return -1;
    }
    uint32_t new_id = (uint32_t)db->concept_count++;
    db->concepts[new_id].name = copy;
    db->concepts[new_id].df = 0;
    db->concepts[new_id].rank = UINT32_MAX;
    db->concepts[new_id].omega = 0.0;
    db->concepts[new_id].structural = is_structural_name(copy);
    db->concepts[new_id].critical = is_critical_name(copy);
    node->key = copy;
    node->id = new_id;
    node->next = db->buckets[h];
    db->buckets[h] = node;
    *id = new_id;
    return 0;
}

static int tx_add_id(uint32_t *ids, size_t *count, size_t cap, uint32_t id) {
    for (size_t i = 0; i < *count; i++) {
        if (ids[i] == id) return 0;
    }
    if (*count >= cap) return 0;
    ids[(*count)++] = id;
    return 0;
}

static int add_concept(HUPPDB *db, uint32_t *ids, size_t *count, size_t cap, const char *name) {
    uint32_t id;
    if (dict_get_or_add(db, name, &id) != 0) return -1;
    return tx_add_id(ids, count, cap, id);
}

static int is_stopword(const char *w) {
    static const char *stops[] = {
        "the","and","for","that","with","this","from","into","have","has","had","are","was","were","will","would",
        "could","should","what","when","where","which","there","their","about","using","without","your","you","can",
        "please","write","make","give","show","tell","does","did","not","but","all","any","each","than","then","them",
        "also","more","most","some","such","how","why","who","its","his","her","our","out","one","two","get","set",
        NULL
    };
    for (int i = 0; stops[i]; i++) {
        if (strcmp(w, stops[i]) == 0) return 1;
    }
    return 0;
}

static void lower_ascii(char *s) {
    for (; *s; s++) *s = (char)tolower((unsigned char)*s);
}

static char *json_extract_string(const char *line, const char *key) {
    char needle[96];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(line, needle);
    if (!p) return NULL;
    p = strchr(p + strlen(needle), ':');
    if (!p) return NULL;
    p++;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != '"') return NULL;
    p++;
    size_t cap = 256, len = 0;
    char *out = (char *)malloc(cap);
    if (!out) return NULL;
    while (*p) {
        char c = *p++;
        if (c == '"') break;
        if (c == '\\' && *p) {
            char e = *p++;
            if (e == 'n') c = '\n';
            else if (e == 'r') c = '\r';
            else if (e == 't') c = '\t';
            else if (e == '"' || e == '\\' || e == '/') c = e;
            else c = ' ';
        }
        if (len + 2 >= cap) {
            cap *= 2;
            char *tmp = (char *)realloc(out, cap);
            if (!tmp) {
                free(out);
                return NULL;
            }
            out = tmp;
        }
        out[len++] = c;
    }
    out[len] = '\0';
    return out;
}

static char *append_text(char *base, const char *extra) {
    if (!extra || !*extra) return base;
    size_t a = base ? strlen(base) : 0;
    size_t b = strlen(extra);
    char *out = (char *)realloc(base, a + b + 2);
    if (!out) {
        free(base);
        return NULL;
    }
    if (a) out[a++] = ' ';
    memcpy(out + a, extra, b + 1);
    return out;
}

static char *extract_code_feedback_human_text(const char *line) {
    char *combined = NULL;
    const char *p = line;
    while ((p = strstr(p, "\"role\"")) != NULL) {
        const char *human = strstr(p, "\"human\"");
        const char *value = strstr(p, "\"value\"");
        if (!human || !value || human > value) {
            p += 6;
            continue;
        }
        const char *next_role = strstr(value + 7, "\"role\"");
        const char *segment_end = next_role ? next_role : line + strlen(line);
        if (human > segment_end) {
            p = value + 7;
            continue;
        }
        char *v = json_extract_string(value, "value");
        if (v) {
            combined = append_text(combined, v);
            free(v);
            if (!combined) return NULL;
        }
        p = segment_end;
    }
    return combined;
}

static void add_phrase_features(HUPPDB *db, uint32_t *ids, size_t *count, size_t cap, const char *text) {
    char tmp[128];
    const char *checks[][2] = {
        {"json", "fmt:json"}, {"csv", "fmt:csv"}, {"table", "fmt:table"}, {"bullet", "fmt:list"},
        {"list", "fmt:list"}, {"markdown", "fmt:markdown"}, {"cite", "fmt:citation"}, {"citation", "fmt:citation"},
        {"python", "lang:python"}, {"ruby", "lang:ruby"}, {"java", "lang:java"}, {"javascript", "lang:javascript"},
        {"typescript", "lang:typescript"}, {"sql", "lang:sql"}, {"html", "lang:html"}, {"css", "lang:css"},
        {"code", "code_task"}, {"function", "code_task"}, {"algorithm", "code_task"},
        {"without", "constraint:negative"}, {"must", "constraint:hard"}, {"only", "constraint:hard"},
        {"compare", "task:compare"}, {"classify", "task:classify"}, {"summarize", "task:summarize"},
        {"explain", "task:explain"}, {"generate", "task:generate"}, {"convert", "task:convert"},
        {"fix", "task:debug"}, {"debug", "task:debug"}, {"optimize", "task:optimize"}, {NULL, NULL}
    };
    size_t n = strlen(text);
    char *lower = (char *)malloc(n + 1);
    if (!lower) return;
    memcpy(lower, text, n + 1);
    lower_ascii(lower);
    for (int i = 0; checks[i][0]; i++) {
        if (strstr(lower, checks[i][0])) add_concept(db, ids, count, cap, checks[i][1]);
    }
    if (strchr(lower, '?')) add_concept(db, ids, count, cap, "task:question_answering");
    if (strstr(lower, "step by step") || strstr(lower, "reason")) add_concept(db, ids, count, cap, "task:reasoning");
    if (strstr(lower, "do not") || strstr(lower, "don't")) add_concept(db, ids, count, cap, "constraint:negative");
    free(lower);
    (void)tmp;
}

static size_t tokenize_concepts(HUPPDB *db, uint32_t *ids, size_t *count, size_t cap, const char *text) {
    size_t tokens = 0;
    char word[96];
    size_t len = 0;
    for (const char *p = text; ; p++) {
        unsigned char c = (unsigned char)*p;
        if (isalnum(c) || c == '_' || c == '-') {
            if (len + 1 < sizeof(word)) word[len++] = (char)tolower(c);
        } else {
            if (len > 0) {
                word[len] = '\0';
                tokens++;
                if (len >= 3 && !is_stopword(word) && !isdigit((unsigned char)word[0])) {
                    char concept[128];
                    snprintf(concept, sizeof(concept), "kw:%s", word);
                    add_concept(db, ids, count, cap, concept);
                }
                len = 0;
            }
            if (!*p) break;
        }
    }
    return tokens;
}

static int add_transaction(HUPPDB *db, uint32_t *ids, size_t count, size_t tokens) {
    if (count == 0) return 0;
    if (db->tx_count >= db->tx_cap) {
        size_t next = db->tx_cap ? db->tx_cap * 2 : 4096;
        HUPPTransaction *tmp = (HUPPTransaction *)realloc(db->tx, next * sizeof(HUPPTransaction));
        if (!tmp) return -1;
        db->tx = tmp;
        db->tx_cap = next;
    }
    HUPPTransaction *t = &db->tx[db->tx_count++];
    memset(t, 0, sizeof(*t));
    t->items = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!t->items) return -1;
    memcpy(t->items, ids, count * sizeof(uint32_t));
    t->item_count = (uint16_t)count;
    t->token_count = (uint16_t)(tokens > 65535 ? 65535 : tokens);
    for (size_t i = 0; i < count; i++) db->concepts[ids[i]].df++;
    db->total_tokens += tokens;
    return 0;
}

static int parse_dolly_line(HUPPDB *db, const char *line, size_t cap) {
    char *instruction = json_extract_string(line, "instruction");
    char *context = json_extract_string(line, "context");
    char *category = json_extract_string(line, "category");
    char *text = NULL;
    text = append_text(text, instruction);
    text = append_text(text, context);
    if (!text) text = hupp_strdup("");
    uint32_t *ids = (uint32_t *)calloc(cap, sizeof(uint32_t));
    if (!ids || !text) {
        free(ids); free(instruction); free(context); free(category); free(text);
        return -1;
    }
    size_t count = 0;
    if (category && *category) {
        char concept[160];
        lower_ascii(category);
        snprintf(concept, sizeof(concept), "cat:%s", category);
        add_concept(db, ids, &count, cap, concept);
    }
    if (context && *context) add_concept(db, ids, &count, cap, "has_context");
    add_phrase_features(db, ids, &count, cap, text);
    size_t tokens = tokenize_concepts(db, ids, &count, cap, text);
    int rc = add_transaction(db, ids, count, tokens);
    free(ids); free(instruction); free(context); free(category); free(text);
    return rc;
}

static int parse_code_feedback_line(HUPPDB *db, const char *line, size_t cap) {
    char *text = extract_code_feedback_human_text(line);
    if (!text) text = hupp_strdup("");
    uint32_t *ids = (uint32_t *)calloc(cap, sizeof(uint32_t));
    if (!ids || !text) {
        free(ids); free(text);
        return -1;
    }
    size_t count = 0;
    add_concept(db, ids, &count, cap, "cat:code_feedback");
    add_concept(db, ids, &count, cap, "code_task");
    add_phrase_features(db, ids, &count, cap, text);
    size_t tokens = tokenize_concepts(db, ids, &count, cap, text);
    int rc = add_transaction(db, ids, count, tokens);
    free(ids); free(text);
    return rc;
}

static HUPPDatasetType auto_type(const char *path, const char *line) {
    if (path && strstr(path, "dolly")) return HUPP_DATASET_DOLLY;
    if (path && (strstr(path, "Code") || strstr(path, "Feedback"))) return HUPP_DATASET_CODE_FEEDBACK;
    if (line && strstr(line, "\"messages\"")) return HUPP_DATASET_CODE_FEEDBACK;
    return HUPP_DATASET_DOLLY;
}

static int load_prompt_db(const char *path, const HUPPParams *params, HUPPDB *db) {
    FILE *fp = fopen(path, "r");
    if (!fp) return -1;
    char *line = NULL;
    size_t cap_line = 0;
    ssize_t len;
    HUPPDatasetType type = params->dataset_type;
    int first = 1;
    size_t concept_cap = params->max_concepts_per_prompt ? params->max_concepts_per_prompt : 64;
    if (concept_cap > 512) concept_cap = 512;
    while ((len = dm_getline(&line, &cap_line, fp)) != -1) {
        if (len <= 1) continue;
        if (first && type == HUPP_DATASET_AUTO) type = auto_type(path, line);
        first = 0;
        int rc = type == HUPP_DATASET_CODE_FEEDBACK ? parse_code_feedback_line(db, line, concept_cap)
                                                     : parse_dolly_line(db, line, concept_cap);
        if (rc != 0) {
            free(line);
            fclose(fp);
            return -1;
        }
        if (params->max_transactions && db->tx_count >= params->max_transactions) break;
    }
    free(line);
    fclose(fp);
    return 0;
}

static int order_cmp(const void *a, const void *b) {
    const HUPPOrderItem *ia = (const HUPPOrderItem *)a;
    const HUPPOrderItem *ib = (const HUPPOrderItem *)b;
    if (ia->support < ib->support) return -1;
    if (ia->support > ib->support) return 1;
    if (ia->omega > ib->omega) return -1;
    if (ia->omega < ib->omega) return 1;
    return ia->item < ib->item ? -1 : (ia->item > ib->item);
}

static HUPPDB *g_sort_db;
static int item_rank_cmp(const void *a, const void *b) {
    uint32_t ia = *(const uint32_t *)a;
    uint32_t ib = *(const uint32_t *)b;
    uint32_t ra = g_sort_db->concepts[ia].rank;
    uint32_t rb = g_sort_db->concepts[ib].rank;
    if (ra < rb) return -1;
    if (ra > rb) return 1;
    return ia < ib ? -1 : (ia > ib);
}

static int prepare_order(HUPPDB *db) {
    HUPPOrderItem *order = (HUPPOrderItem *)malloc(db->concept_count * sizeof(HUPPOrderItem));
    if (!order) return -1;
    double n = (double)db->tx_count;
    for (size_t i = 0; i < db->concept_count; i++) {
        db->concepts[i].omega = log((n + 1.0) / ((double)db->concepts[i].df + 1.0));
        order[i].item = (uint32_t)i;
        order[i].support = db->concepts[i].df;
        order[i].omega = db->concepts[i].omega;
    }
    qsort(order, db->concept_count, sizeof(HUPPOrderItem), order_cmp);
    for (size_t r = 0; r < db->concept_count; r++) db->concepts[order[r].item].rank = (uint32_t)r;
    free(order);
    g_sort_db = db;
    for (size_t t = 0; t < db->tx_count; t++) {
        qsort(db->tx[t].items, db->tx[t].item_count, sizeof(uint32_t), item_rank_cmp);
    }
    return 0;
}

static int cand_dominated(const float *s, const float *l, const float *a, const float *h, int x, int y) {
    return s[y] >= s[x] && l[y] >= l[x] && a[y] >= a[x] && h[y] >= h[x] &&
           (s[y] > s[x] || l[y] > l[x] || a[y] > a[x] || h[y] > h[x]);
}

static int candidate_preserves(int cand, const HUPPConcept *c, double omega, double avg_omega, size_t pos, size_t count) {
    if (cand == 0) return 1;
    if (cand == 1) return c->structural || omega >= avg_omega * 0.55 || (pos % 5) != 0;
    if (cand == 2) return c->critical || omega >= avg_omega || pos < (count * 3) / 5;
    if (cand == 3) return c->critical || omega >= avg_omega * 1.25 || pos < (count + 2) / 3;
    if (cand == 4) {
        const char *n = c->name;
        return c->critical || strncmp(n, "cat:", 4) == 0 || strncmp(n, "task:", 5) == 0 ||
               strncmp(n, "lang:", 5) == 0 || strncmp(n, "fmt:", 4) == 0;
    }
    return c->critical || omega >= avg_omega * 1.5;
}

static int build_candidates(HUPPDB *db, const HUPPParams *params, HUPPStats *stats) {
    double ws = params->w_token, wl = params->w_latency, wa = params->w_alignment, wh = params->w_cache;
    double sumw = ws + wl + wa + wh;
    if (sumw <= 0.0) { ws = 0.35; wl = 0.20; wa = 0.35; wh = 0.10; sumw = 1.0; }
    ws /= sumw; wl /= sumw; wa /= sumw; wh /= sumw;

    for (size_t ti = 0; ti < db->tx_count; ti++) {
        HUPPTransaction *t = &db->tx[ti];
        int base_k = 6;
        float s[HUPP_MAX_CANDIDATES], l[HUPP_MAX_CANDIDATES], a[HUPP_MAX_CANDIDATES], h[HUPP_MAX_CANDIDATES];
        uint64_t keep_by_pos[512];
        double total_ic = 0.0;
        double avg_ic = 0.0;
        size_t critical = 0;
        for (size_t p = 0; p < t->item_count; p++) {
            HUPPConcept *c = &db->concepts[t->items[p]];
            total_ic += c->omega;
            if (c->critical) critical++;
        }
        avg_ic = t->item_count ? total_ic / (double)t->item_count : 0.0;
        for (int c = 0; c < base_k; c++) {
            double kept_ic = 0.0;
            size_t kept = 0, kept_critical = 0;
            for (size_t p = 0; p < t->item_count; p++) {
                HUPPConcept *concept = &db->concepts[t->items[p]];
                if (candidate_preserves(c, concept, concept->omega, avg_ic, p, t->item_count)) {
                    kept++;
                    kept_ic += concept->omega;
                    if (concept->critical) kept_critical++;
                }
            }
            double frac_items = t->item_count ? (double)kept / (double)t->item_count : 1.0;
            double frac_ic = total_ic > 0.0 ? kept_ic / total_ic : frac_items;
            double crit_frac = critical ? (double)kept_critical / (double)critical : 1.0;
            double target_saving[] = {0.00, 0.18, 0.34, 0.52, 0.30, 0.62};
            double cache_bias[] = {0.20, 0.42, 0.50, 0.46, 0.86, 0.55};
            s[c] = (float)clamp01(target_saving[c] * (1.05 - 0.25 * frac_items));
            l[c] = (float)clamp01(0.10 + 0.82 * s[c]);
            a[c] = (float)clamp01(0.58 + 0.30 * frac_ic + 0.12 * crit_frac - (c == 5 ? 0.05 : 0.0));
            h[c] = (float)cache_bias[c];
        }
        uint8_t live[HUPP_MAX_CANDIDATES];
        for (int i = 0; i < base_k; i++) live[i] = 1;
        for (int i = 0; i < base_k; i++) {
            for (int j = 0; j < base_k; j++) {
                if (i != j && cand_dominated(s, l, a, h, i, j)) {
                    live[i] = 0;
                    stats->pareto_removed++;
                    break;
                }
            }
        }
        int map[HUPP_MAX_CANDIDATES];
        t->k = 0;
        for (int i = 0; i < base_k; i++) {
            if (!live[i]) continue;
            map[t->k] = i;
            t->saving[t->k] = s[i];
            t->latency[t->k] = l[i];
            t->acc[t->k] = a[i];
            t->cache[t->k] = h[i];
            t->gain[t->k] = (float)(ws * s[i] + wl * l[i] + wa * a[i] + wh * h[i]);
            t->k++;
        }
        if (t->k == 0) return -1;
        t->keep_masks = (uint64_t *)calloc(t->item_count, sizeof(uint64_t));
        t->suffix_ic = (float *)calloc((size_t)t->item_count + 1, sizeof(float));
        if (!t->keep_masks || !t->suffix_ic) return -1;
        for (size_t p = 0; p < t->item_count; p++) {
            uint64_t mask = 0;
            for (uint8_t k = 0; k < t->k; k++) {
                int cand = map[k];
                HUPPConcept *c = &db->concepts[t->items[p]];
                if (candidate_preserves(cand, c, c->omega, avg_ic, p, t->item_count)) mask |= (1ULL << k);
            }
            keep_by_pos[p] = mask;
            t->keep_masks[p] = mask;
        }
        t->suffix_ic[t->item_count] = 0.0f;
        for (int p = (int)t->item_count - 1; p >= 0; p--) {
            t->suffix_ic[p] = (float)(t->suffix_ic[p + 1] + db->concepts[t->items[p]].omega);
        }
        stats->candidate_optimizers += t->k;
        (void)keep_by_pos;
    }
    return 0;
}

static int list_push(HUPPList *l, HUPPEntry e) {
    if (l->count >= l->cap) {
        size_t next = l->cap ? l->cap * 2 : 64;
        HUPPEntry *tmp = (HUPPEntry *)realloc(l->entries, next * sizeof(HUPPEntry));
        if (!tmp) return -1;
        l->entries = tmp;
        l->cap = next;
    }
    l->entries[l->count++] = e;
    return 0;
}

static void list_free(HUPPList *l) {
    free(l->entries);
    memset(l, 0, sizeof(*l));
}

static void mask_best(const HUPPTransaction *t, uint64_t mask, float *best_gain, float *best_acc, float *acc_ub) {
    float bg = -1.0f, ba = 0.0f, au = 0.0f;
    for (uint8_t k = 0; k < t->k; k++) {
        if ((mask & (1ULL << k)) == 0) continue;
        if (t->gain[k] > bg) {
            bg = t->gain[k];
            ba = t->acc[k];
        }
        if (t->acc[k] > au) au = t->acc[k];
    }
    *best_gain = bg < 0.0f ? 0.0f : bg;
    *best_acc = ba;
    *acc_ub = au;
}

static int build_singletons(HUPPDB *db, HUPPContext *ctx) {
    ctx->singletons = (HUPPList *)calloc(db->concept_count, sizeof(HUPPList));
    if (!ctx->singletons) return -1;
    for (size_t ti = 0; ti < db->tx_count; ti++) {
        HUPPTransaction *t = &db->tx[ti];
        for (uint16_t p = 0; p < t->item_count; p++) {
            uint32_t item = t->items[p];
            uint64_t mask = t->keep_masks[p];
            if (!mask) continue;
            HUPPEntry e;
            e.tid = (uint32_t)ti;
            e.pos = p;
            e.feas_mask = mask;
            mask_best(t, mask, &e.best_gain, &e.exact_acc, &e.acc_ub);
            e.rem_ic = t->suffix_ic[p + 1];
            if (list_push(&ctx->singletons[item], e) != 0) return -1;
        }
    }
    ctx->frequent_items = (uint32_t *)malloc(db->concept_count * sizeof(uint32_t));
    if (!ctx->frequent_items) return -1;
    for (size_t i = 0; i < db->concept_count; i++) {
        ctx->singletons[i].item = (uint32_t)i;
        ctx->singletons[i].omega = db->concepts[i].omega;
        if (ctx->singletons[i].count >= ctx->minsup) {
            ctx->frequent_items[ctx->frequent_count++] = (uint32_t)i;
            ctx->stats->singleton_occurrences += ctx->singletons[i].count;
        }
    }
    ctx->stats->frequent_singletons = ctx->frequent_count;
    return 0;
}

static int time_limited(HUPPContext *ctx) {
    if (ctx->stats->limited) return 1;
    if (ctx->max_patterns && ctx->stats->emitted_patterns >= ctx->max_patterns) {
        ctx->stats->limited = 1;
        return 1;
    }
    if (ctx->max_seconds > 0.0) {
        double elapsed = (double)(clock() - ctx->started) / (double)CLOCKS_PER_SEC;
        if (elapsed >= ctx->max_seconds) {
            ctx->stats->limited = 1;
            return 1;
        }
    }
    return 0;
}

static int find_item_after(const HUPPTransaction *t, uint16_t pos, uint32_t item, uint16_t *found) {
    for (uint16_t p = (uint16_t)(pos + 1); p < t->item_count; p++) {
        if (t->items[p] == item) {
            *found = p;
            return 1;
        }
    }
    return 0;
}

static int join_list(HUPPContext *ctx, const HUPPList *prefix, uint32_t ext_item, HUPPList *out) {
    memset(out, 0, sizeof(*out));
    out->item = ext_item;
    out->omega = prefix->omega + ctx->db->concepts[ext_item].omega;
    ctx->stats->joins++;
    for (size_t i = 0; i < prefix->count; i++) {
        const HUPPEntry *e = &prefix->entries[i];
        HUPPTransaction *t = &ctx->db->tx[e->tid];
        uint16_t p;
        if (!find_item_after(t, e->pos, ext_item, &p)) continue;
        uint64_t mask = e->feas_mask & t->keep_masks[p];
        if (!mask) continue;
        HUPPEntry ne;
        ne.tid = e->tid;
        ne.pos = p;
        ne.feas_mask = mask;
        mask_best(t, mask, &ne.best_gain, &ne.exact_acc, &ne.acc_ub);
        ne.rem_ic = t->suffix_ic[p + 1];
        if (list_push(out, ne) != 0) return -1;
    }
    ctx->stats->joined_entries += out->count;
    return 0;
}

static int float_desc_cmp(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return x > y ? -1 : (x < y);
}

static double aaub_top_sigma(const HUPPList *list, uint32_t sigma) {
    if (list->count < sigma || sigma == 0) return 0.0;
    float *vals = (float *)malloc(list->count * sizeof(float));
    if (!vals) return 0.0;
    for (size_t i = 0; i < list->count; i++) vals[i] = list->entries[i].acc_ub;
    qsort(vals, list->count, sizeof(float), float_desc_cmp);
    double sum = 0.0;
    for (uint32_t i = 0; i < sigma; i++) sum += vals[i];
    free(vals);
    return sum / (double)sigma;
}

static int dfs(HUPPContext *ctx, HUPPList *list, size_t suffix_start, size_t depth) {
    if (time_limited(ctx)) return 0;
    if (list->count < ctx->minsup) {
        ctx->stats->pruned_support++;
        return 0;
    }
    ctx->stats->visited_nodes++;
    if (depth > ctx->stats->max_depth_seen) ctx->stats->max_depth_seen = depth;

    double gain_sum = 0.0, acc_sum = 0.0, ptwo = 0.0;
    for (size_t i = 0; i < list->count; i++) {
        gain_sum += list->entries[i].best_gain;
        acc_sum += list->entries[i].exact_acc;
        ptwo += list->entries[i].best_gain * (list->omega + list->entries[i].rem_ic);
    }
    double utility = list->omega * gain_sum;
    double avg_acc = acc_sum / (double)list->count;
    if (utility >= ctx->theta && avg_acc >= ctx->alpha_min) {
        HUPPStats *s = ctx->stats;
        s->emitted_patterns++;
        s->total_pattern_items += depth;
        s->avg_utility += utility;
        s->avg_alignment += avg_acc;
        s->avg_support += (double)list->count;
        if (utility > s->best_utility) s->best_utility = utility;
        if (avg_acc > s->best_alignment) s->best_alignment = avg_acc;
    }
    if (ptwo < ctx->theta) {
        ctx->stats->pruned_ptwo++;
        return 0;
    }
    if (ctx->alpha_min > 0.0 && aaub_top_sigma(list, ctx->minsup) < ctx->alpha_min) {
        ctx->stats->pruned_aaub++;
        return 0;
    }
    if (ctx->max_depth && depth >= ctx->max_depth) return 0;
    for (size_t si = suffix_start; si < ctx->frequent_count; si++) {
        if (time_limited(ctx)) break;
        uint32_t ext = ctx->frequent_items[si];
        HUPPList child;
        if (join_list(ctx, list, ext, &child) != 0) return -1;
        if (child.count >= ctx->minsup) {
            if (dfs(ctx, &child, si + 1, depth + 1) != 0) {
                list_free(&child);
                return -1;
            }
        } else if (child.count > 0) {
            ctx->stats->pruned_support++;
        }
        list_free(&child);
    }
    return 0;
}

int hupp_parse_dataset_type(const char *name, HUPPDatasetType *type) {
    if (!name || strcmp(name, "auto") == 0) { *type = HUPP_DATASET_AUTO; return 0; }
    if (strcmp(name, "dolly") == 0 || strcmp(name, "databricks-dolly") == 0) { *type = HUPP_DATASET_DOLLY; return 0; }
    if (strcmp(name, "code_feedback") == 0 || strcmp(name, "modified-code-feedback") == 0 || strcmp(name, "code") == 0) {
        *type = HUPP_DATASET_CODE_FEEDBACK;
        return 0;
    }
    return -1;
}

const char *hupp_dataset_type_name(HUPPDatasetType type) {
    switch (type) {
        case HUPP_DATASET_DOLLY: return "dolly";
        case HUPP_DATASET_CODE_FEEDBACK: return "code_feedback";
        default: return "auto";
    }
}

HUPPParams hupp_default_params(void) {
    HUPPParams p;
    memset(&p, 0, sizeof(p));
    p.dataset_type = HUPP_DATASET_AUTO;
    p.min_support = 0.02;
    p.min_utility = 0.0;
    p.min_utility_ratio = 0.002;
    p.min_alignment = 0.80;
    p.w_token = 0.35;
    p.w_latency = 0.20;
    p.w_alignment = 0.35;
    p.w_cache = 0.10;
    p.max_concepts_per_prompt = 64;
    p.max_depth = 6;
    return p;
}

int hupp_mine_file(const char *path, const HUPPParams *params, HUPPStats *stats) {
    if (!path || !params || !stats) return -1;
    memset(stats, 0, sizeof(*stats));
    HUPPDB db;
    if (db_init(&db) != 0) return -1;
    if (load_prompt_db(path, params, &db) != 0 || db.tx_count == 0) {
        db_free(&db);
        return -1;
    }
    stats->transactions = db.tx_count;
    stats->semantic_concepts = db.concept_count;
    stats->total_prompt_tokens = db.total_tokens;
    if (prepare_order(&db) != 0 || build_candidates(&db, params, stats) != 0) {
        db_free(&db);
        return -1;
    }

    HUPPContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.db = &db;
    ctx.stats = stats;
    ctx.alpha_min = params->min_alignment;
    ctx.max_patterns = params->max_patterns;
    ctx.max_depth = params->max_depth;
    ctx.max_seconds = params->max_seconds;
    ctx.started = clock();
    double ms = params->min_support;
    ctx.minsup = ms < 1.0 ? (uint32_t)ceil(ms * (double)db.tx_count) : (uint32_t)ms;
    if (ctx.minsup == 0) ctx.minsup = 1;
    ctx.theta = params->min_utility;

    if (build_singletons(&db, &ctx) != 0) {
        db_free(&db);
        return -1;
    }
    if (ctx.theta <= 0.0) {
        double opportunity = 0.0;
        for (size_t i = 0; i < ctx.frequent_count; i++) {
            HUPPList *l = &ctx.singletons[ctx.frequent_items[i]];
            double g = 0.0;
            for (size_t e = 0; e < l->count; e++) g += l->entries[e].best_gain;
            double u = l->omega * g;
            if (u > opportunity) opportunity = u;
        }
        ctx.theta = opportunity * (params->min_utility_ratio > 0.0 ? params->min_utility_ratio : 0.002);
    }
    stats->minsup_count = ctx.minsup;
    stats->theta = ctx.theta;
    stats->alpha_min = ctx.alpha_min;

    int rc = 0;
    for (size_t i = 0; i < ctx.frequent_count; i++) {
        uint32_t item = ctx.frequent_items[i];
        HUPPList *l = &ctx.singletons[item];
        if (dfs(&ctx, l, i + 1, 1) != 0) {
            rc = -1;
            break;
        }
        if (time_limited(&ctx)) break;
    }
    if (stats->emitted_patterns) {
        stats->avg_utility /= (double)stats->emitted_patterns;
        stats->avg_alignment /= (double)stats->emitted_patterns;
        stats->avg_support /= (double)stats->emitted_patterns;
    }
    stats->result_ram_bytes = stats->emitted_patterns * 32 + stats->total_pattern_items * 4;
    stats->result_disk_est_bytes = stats->emitted_patterns * 96 + stats->total_pattern_items * 16;

    for (size_t i = 0; i < db.concept_count; i++) list_free(&ctx.singletons[i]);
    free(ctx.singletons);
    free(ctx.frequent_items);
    db_free(&db);
    return rc;
}
