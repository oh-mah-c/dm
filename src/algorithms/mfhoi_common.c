#include "algorithms/mfhoi.h"
#include "core/dm_portability.h"
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
#include <intrin.h>
#include <windows.h>
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
static int clock_gettime(int unused, struct timespec *ts) {
    static LARGE_INTEGER freq;
    static int initialized = 0;
    LARGE_INTEGER counter;
    (void)unused;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    QueryPerformanceCounter(&counter);
    ts->tv_sec = (time_t)(counter.QuadPart / freq.QuadPart);
    ts->tv_nsec = (long)(((counter.QuadPart % freq.QuadPart) * 1000000000LL) / freq.QuadPart);
    return 0;
}
#endif
#endif

#define WORD_BITS ((int)(sizeof(unsigned long) * 8))

typedef struct {
    VerticalItem *items;
    int count;
    int word_count;
} VerticalDB;

typedef struct {
    const TransactionDB *db;
    const VerticalDB *vdb;
    int minsup;
    double minocc;
    int max_patterns;
    double max_seconds;
    struct timespec start;
    PatternList *frequent;
    PatternList *fhoi;
    MFHOIStats *stats;
} MineContext;

static double elapsed_seconds(const struct timespec *start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - start->tv_sec) + (double)(now.tv_nsec - start->tv_nsec) / 1000000000.0;
}

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a;
    int y = *(const int *)b;
    return (x > y) - (x < y);
}

static int cmp_vertical_support_asc(const void *a, const void *b) {
    const VerticalItem *x = (const VerticalItem *)a;
    const VerticalItem *y = (const VerticalItem *)b;
    if (x->support != y->support) return x->support - y->support;
    return x->item - y->item;
}

static int cmp_pattern_len_desc(const void *a, const void *b) {
    const Pattern *x = (const Pattern *)a;
    const Pattern *y = (const Pattern *)b;
    if (x->length != y->length) return y->length - x->length;
    if (x->avg_occ < y->avg_occ) return 1;
    if (x->avg_occ > y->avg_occ) return -1;
    return y->support - x->support;
}

static int cmp_int_desc(const void *a, const void *b) {
    int x = *(const int *)a;
    int y = *(const int *)b;
    return (y > x) - (y < x);
}

static int cmp_double_desc(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (y > x) - (y < x);
}

static int bit_words(int nbits) {
    return (nbits + WORD_BITS - 1) / WORD_BITS;
}

static void bitset_set(unsigned long *bits, int pos) {
    bits[pos / WORD_BITS] |= (1UL << (pos % WORD_BITS));
}

static int bitset_get(const unsigned long *bits, int pos) {
    return (bits[pos / WORD_BITS] >> (pos % WORD_BITS)) & 1UL;
}

static int bitset_popcount(const unsigned long *bits, int words) {
    int total = 0;
    for (int i = 0; i < words; i++) {
#ifdef _MSC_VER
        total += (int)__popcnt64(bits[i]);
#else
        total += __builtin_popcountl(bits[i]);
#endif
    }
    return total;
}

static void bitset_and(unsigned long *out, const unsigned long *a, const unsigned long *b, int words) {
    for (int i = 0; i < words; i++) out[i] = a[i] & b[i];
}

static unsigned long *bitset_clone(const unsigned long *src, int words) {
    unsigned long *copy = malloc((size_t)words * sizeof(unsigned long));
    if (copy) memcpy(copy, src, (size_t)words * sizeof(unsigned long));
    return copy;
}

void mfhoi_pattern_list_init(PatternList *list) {
    list->count = 0;
    list->capacity = 256;
    list->patterns = malloc((size_t)list->capacity * sizeof(Pattern));
}

static void pattern_free(Pattern *p) {
    free(p->items);
    free(p->tidset);
}

void mfhoi_pattern_list_free(PatternList *list) {
    if (!list || !list->patterns) return;
    for (int i = 0; i < list->count; i++) pattern_free(&list->patterns[i]);
    free(list->patterns);
    list->patterns = NULL;
    list->count = 0;
    list->capacity = 0;
}

static int pattern_list_add(PatternList *list, const int *items, int length, const unsigned long *tidset, int word_count, int support, double avg_occ) {
    if (list->count >= list->capacity) {
        int new_cap = list->capacity * 2;
        Pattern *new_patterns = realloc(list->patterns, (size_t)new_cap * sizeof(Pattern));
        if (!new_patterns) return -1;
        list->patterns = new_patterns;
        list->capacity = new_cap;
    }

    Pattern *p = &list->patterns[list->count++];
    p->items = malloc((size_t)length * sizeof(int));
    p->tidset = bitset_clone(tidset, word_count);
    if (!p->items || !p->tidset) return -1;
    memcpy(p->items, items, (size_t)length * sizeof(int));
    qsort(p->items, (size_t)length, sizeof(int), cmp_int);
    p->length = length;
    p->support = support;
    p->avg_occ = avg_occ;
    return 0;
}

static int proper_subset(const Pattern *x, const Pattern *y) {
    if (x->length >= y->length) return 0;
    int i = 0;
    int j = 0;
    while (i < x->length && j < y->length) {
        if (x->items[i] == y->items[j]) {
            i++;
            j++;
        } else if (x->items[i] > y->items[j]) {
            j++;
        } else {
            return 0;
        }
    }
    return i == x->length;
}

static int same_pattern(const Pattern *x, const Pattern *y) {
    if (x->length != y->length) return 0;
    for (int i = 0; i < x->length; i++) {
        if (x->items[i] != y->items[i]) return 0;
    }
    return 1;
}

static double avg_occ_for_tidset(const TransactionDB *db, const unsigned long *tidset, int support, int length) {
    if (support <= 0) return 0.0;
    double sum = 0.0;
    for (int tid = 0; tid < db->transaction_count; tid++) {
        if (bitset_get(tidset, tid)) {
            sum += (double)length / (double)db->transaction_lengths[tid];
        }
    }
    return sum / (double)support;
}

static int transaction_contains_item(const Transaction *tr, int item) {
    int lo = 0;
    int hi = tr->length - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (tr->items[mid] == item) return 1;
        if (tr->items[mid] < item) lo = mid + 1;
        else hi = mid - 1;
    }
    return 0;
}

static int node_occupancy_bounds(const MineContext *ctx, const unsigned long *tidset, int prefix_len, int suffix_start, double *ub1, double *ub2) {
    *ub1 = 0.0;
    *ub2 = 0.0;
    if (suffix_start >= ctx->vdb->count || ctx->minsup <= 0) return 0;

    int support = bitset_popcount(tidset, ctx->vdb->word_count);
    if (support < ctx->minsup) return -1;

    int *caps = malloc((size_t)support * sizeof(int));
    double *inv = malloc((size_t)support * sizeof(double));
    double *env = malloc((size_t)support * sizeof(double));
    if (!caps || !inv || !env) {
        free(caps);
        free(inv);
        free(env);
        return -1;
    }

    int pos = 0;
    for (int tid = 0; tid < ctx->db->transaction_count; tid++) {
        if (!bitset_get(tidset, tid)) continue;
        const Transaction *tr = &ctx->db->transactions[tid];
        int cap = 0;
        for (int idx = suffix_start; idx < ctx->vdb->count; idx++) {
            if (transaction_contains_item(tr, ctx->vdb->items[idx].item)) cap++;
        }
        caps[pos] = cap;
        inv[pos] = 1.0 / (double)ctx->db->transaction_lengths[tid];
        env[pos] = (double)(prefix_len + cap) / (double)ctx->db->transaction_lengths[tid];
        pos++;
    }

    qsort(caps, (size_t)support, sizeof(int), cmp_int_desc);
    qsort(inv, (size_t)support, sizeof(double), cmp_double_desc);
    qsort(env, (size_t)support, sizeof(double), cmp_double_desc);

    int top = ctx->minsup;
    double inv_top_sum = 0.0;
    double env_top_sum = 0.0;
    for (int i = 0; i < top; i++) {
        inv_top_sum += inv[i];
        env_top_sum += env[i];
    }

    *ub1 = (double)(prefix_len + caps[top - 1]) * (inv_top_sum / (double)top);
    *ub2 = env_top_sum / (double)top;

    free(caps);
    free(inv);
    free(env);
    return 0;
}

static void strip_comment(char *line) {
    char *hash = strchr(line, '#');
    if (hash) *hash = '\0';
}

TransactionDB *mfhoi_load_transaction_db(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) return NULL;

    TransactionDB *db = calloc(1, sizeof(TransactionDB));
    int cap = 1024;
    db->transactions = calloc((size_t)cap, sizeof(Transaction));
    db->transaction_lengths = calloc((size_t)cap, sizeof(int));
    db->max_item_id = -1;

    char line[65536];
    while (fgets(line, sizeof(line), fp)) {
        strip_comment(line);
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0') continue;

        if (db->transaction_count >= cap) {
            cap *= 2;
            db->transactions = realloc(db->transactions, (size_t)cap * sizeof(Transaction));
            db->transaction_lengths = realloc(db->transaction_lengths, (size_t)cap * sizeof(int));
        }

        int item_cap = 16;
        int len = 0;
        int *items = malloc((size_t)item_cap * sizeof(int));
        char *tok = strtok(p, " \t\r\n");
        while (tok) {
            if (len >= item_cap) {
                item_cap *= 2;
                items = realloc(items, (size_t)item_cap * sizeof(int));
            }
            int item = atoi(tok);
            items[len++] = item;
            if (item > db->max_item_id) db->max_item_id = item;
            tok = strtok(NULL, " \t\r\n");
        }
        if (len > 0) {
            qsort(items, (size_t)len, sizeof(int), cmp_int);
            int unique_len = 0;
            for (int i = 0; i < len; i++) {
                if (unique_len == 0 || items[i] != items[unique_len - 1]) {
                    items[unique_len++] = items[i];
                }
            }
            db->transactions[db->transaction_count].items = items;
            db->transactions[db->transaction_count].length = unique_len;
            db->transaction_lengths[db->transaction_count] = unique_len;
            db->transaction_count++;
        } else {
            free(items);
        }
    }

    fclose(fp);
    db->item_count = db->max_item_id + 1;
    return db;
}

void mfhoi_free_transaction_db(TransactionDB *db) {
    if (!db) return;
    for (int i = 0; i < db->transaction_count; i++) free(db->transactions[i].items);
    free(db->transactions);
    free(db->transaction_lengths);
    free(db);
}

static void vertical_db_free(VerticalDB *vdb) {
    if (!vdb || !vdb->items) return;
    for (int i = 0; i < vdb->count; i++) free(vdb->items[i].bitset);
    free(vdb->items);
    vdb->items = NULL;
    vdb->count = 0;
}

static int build_vertical_db(const TransactionDB *db, int minsup, VerticalDB *out) {
    out->word_count = bit_words(db->transaction_count);
    int *counts = calloc((size_t)db->item_count, sizeof(int));
    if (!counts) return -1;

    for (int tid = 0; tid < db->transaction_count; tid++) {
        for (int i = 0; i < db->transactions[tid].length; i++) counts[db->transactions[tid].items[i]]++;
    }

    int n = 0;
    for (int item = 0; item < db->item_count; item++) {
        if (counts[item] >= minsup) n++;
    }

    out->items = calloc((size_t)n, sizeof(VerticalItem));
    out->count = n;
    int *item_to_pos = malloc((size_t)db->item_count * sizeof(int));
    for (int i = 0; i < db->item_count; i++) item_to_pos[i] = -1;

    int pos = 0;
    for (int item = 0; item < db->item_count; item++) {
        if (counts[item] >= minsup) {
            out->items[pos].item = item;
            out->items[pos].support = counts[item];
            out->items[pos].bitset = calloc((size_t)out->word_count, sizeof(unsigned long));
            item_to_pos[item] = pos;
            pos++;
        }
    }

    for (int tid = 0; tid < db->transaction_count; tid++) {
        for (int i = 0; i < db->transactions[tid].length; i++) {
            int p = item_to_pos[db->transactions[tid].items[i]];
            if (p >= 0) bitset_set(out->items[p].bitset, tid);
        }
    }

    qsort(out->items, (size_t)out->count, sizeof(VerticalItem), cmp_vertical_support_asc);
    free(item_to_pos);
    free(counts);
    return 0;
}

static void dfs_mine(MineContext *ctx, int *prefix, int prefix_len, const unsigned long *prefix_tidset, int suffix_start) {
    if (ctx->stats->status_limited) return;
    if (ctx->max_seconds > 0.0 && elapsed_seconds(&ctx->start) > ctx->max_seconds) {
        ctx->stats->status_limited = 1;
        return;
    }

    for (int idx = suffix_start; idx < ctx->vdb->count; idx++) {
        unsigned long *tidset = malloc((size_t)ctx->vdb->word_count * sizeof(unsigned long));
        if (!tidset) {
            ctx->stats->status_limited = 1;
            return;
        }

        if (prefix_tidset) bitset_and(tidset, prefix_tidset, ctx->vdb->items[idx].bitset, ctx->vdb->word_count);
        else memcpy(tidset, ctx->vdb->items[idx].bitset, (size_t)ctx->vdb->word_count * sizeof(unsigned long));

        int support = bitset_popcount(tidset, ctx->vdb->word_count);
        ctx->stats->num_generated_candidates++;
        if (support < ctx->minsup) {
            free(tidset);
            continue;
        }

        prefix[prefix_len] = ctx->vdb->items[idx].item;
        int new_len = prefix_len + 1;
        double avg_occ = avg_occ_for_tidset(ctx->db, tidset, support, new_len);

        pattern_list_add(ctx->frequent, prefix, new_len, tidset, ctx->vdb->word_count, support, avg_occ);
        ctx->stats->num_frequent_itemsets = ctx->frequent->count;
        if (avg_occ + 1e-12 >= ctx->minocc) {
            pattern_list_add(ctx->fhoi, prefix, new_len, tidset, ctx->vdb->word_count, support, avg_occ);
            ctx->stats->num_fhoi = ctx->fhoi->count;
        }

        if (ctx->max_patterns > 0 && ctx->frequent->count >= ctx->max_patterns) {
            ctx->stats->status_limited = 1;
            free(tidset);
            return;
        }

        double ub1 = 1.0;
        double ub2 = 1.0;
        if (idx + 1 < ctx->vdb->count && node_occupancy_bounds(ctx, tidset, new_len, idx + 1, &ub1, &ub2) == 0) {
            if (ub1 + 1e-12 < ctx->minocc) {
                ctx->stats->pruned_ub1_count++;
                free(tidset);
                continue;
            }
            if (ub2 + 1e-12 < ctx->minocc) {
                ctx->stats->pruned_ub2_count++;
                free(tidset);
                continue;
            }
        }

        dfs_mine(ctx, prefix, new_len, tidset, idx + 1);
        free(tidset);
        if (ctx->stats->status_limited) return;
    }
}

static void copy_pattern_to_list(PatternList *out, const Pattern *p, int word_count) {
    pattern_list_add(out, p->items, p->length, p->tidset, word_count, p->support, p->avg_occ);
}

static void filter_mfi(const PatternList *frequent, PatternList *out, int word_count) {
    mfhoi_pattern_list_init(out);
    for (int i = 0; i < frequent->count; i++) {
        int dominated = 0;
        for (int j = 0; j < frequent->count; j++) {
            if (proper_subset(&frequent->patterns[i], &frequent->patterns[j])) {
                dominated = 1;
                break;
            }
        }
        if (!dominated) copy_pattern_to_list(out, &frequent->patterns[i], word_count);
    }
}

static void filter_mfhoi(const PatternList *fhoi, PatternList *out, int word_count, int strong, int *removed_count) {
    mfhoi_pattern_list_init(out);
    if (fhoi->count == 0) {
        *removed_count = 0;
        return;
    }

    Pattern *sorted = malloc((size_t)fhoi->count * sizeof(Pattern));
    memcpy(sorted, fhoi->patterns, (size_t)fhoi->count * sizeof(Pattern));
    qsort(sorted, (size_t)fhoi->count, sizeof(Pattern), cmp_pattern_len_desc);

    *removed_count = 0;
    for (int i = 0; i < fhoi->count; i++) {
        int dominated = 0;
        for (int j = 0; j < i; j++) {
            Pattern *y = &sorted[j];
            int occ_dominates = strong ? (y->avg_occ + 1e-12 >= sorted[i].avg_occ) : (y->avg_occ > sorted[i].avg_occ + 1e-12);
            if (occ_dominates && proper_subset(&sorted[i], y)) {
                dominated = 1;
                break;
            }
        }
        if (dominated) (*removed_count)++;
        else copy_pattern_to_list(out, &sorted[i], word_count);
    }
    free(sorted);
}

void mfhoi_result_free(MFHOIResult *result) {
    mfhoi_pattern_list_free(&result->frequent);
    mfhoi_pattern_list_free(&result->mfi);
    mfhoi_pattern_list_free(&result->fhoi);
    mfhoi_pattern_list_free(&result->weak_mfhoi);
    mfhoi_pattern_list_free(&result->strong_mfhoi);
}

int mfhoi_mine_all(const TransactionDB *db, int minsup_count, double minocc, const MFHOILimits *limits, MFHOIResult *out) {
    memset(out, 0, sizeof(*out));
    if (!db || db->transaction_count <= 0 || minsup_count <= 0) return -1;

    VerticalDB vdb = {0};
    if (build_vertical_db(db, minsup_count, &vdb) != 0) return -1;
    mfhoi_pattern_list_init(&out->frequent);
    mfhoi_pattern_list_init(&out->fhoi);

    int *prefix = malloc((size_t)(vdb.count > 0 ? vdb.count : 1) * sizeof(int));
    MineContext ctx = {
        .db = db,
        .vdb = &vdb,
        .minsup = minsup_count,
        .minocc = minocc,
        .max_patterns = limits ? limits->max_patterns : 2000000,
        .max_seconds = limits ? limits->max_seconds : 0.0,
        .frequent = &out->frequent,
        .fhoi = &out->fhoi,
        .stats = &out->stats
    };
    clock_gettime(CLOCK_MONOTONIC, &ctx.start);
    dfs_mine(&ctx, prefix, 0, NULL, 0);

    filter_mfi(&out->frequent, &out->mfi, vdb.word_count);
    filter_mfhoi(&out->fhoi, &out->weak_mfhoi, vdb.word_count, 0, &out->stats.weak_removed_count);
    filter_mfhoi(&out->fhoi, &out->strong_mfhoi, vdb.word_count, 1, &out->stats.dominance_removed_count);
    out->stats.num_frequent_itemsets = out->frequent.count;
    out->stats.num_fhoi = out->fhoi.count;
    out->stats.num_weak_mfhoi = out->weak_mfhoi.count;
    out->stats.num_strong_mfhoi = out->strong_mfhoi.count;

    free(prefix);
    vertical_db_free(&vdb);
    return out->stats.status_limited ? 1 : 0;
}

const PatternList *mfhoi_select_patterns(const MFHOIResult *result, MFHOIAlgorithm algo) {
    switch (algo) {
        case MFHOI_ALGO_APRIORI: return &result->frequent;
        case MFHOI_ALGO_FHOI: return &result->fhoi;
        case MFHOI_ALGO_WEAK: return &result->weak_mfhoi;
        case MFHOI_ALGO_STRONG: return &result->strong_mfhoi;
        case MFHOI_ALGO_MFI: return &result->mfi;
    }
    return &result->strong_mfhoi;
}

const char *mfhoi_algorithm_name(MFHOIAlgorithm algo) {
    switch (algo) {
        case MFHOI_ALGO_APRIORI: return "apriori";
        case MFHOI_ALGO_FHOI: return "fhoi";
        case MFHOI_ALGO_WEAK: return "weak_mfhoi";
        case MFHOI_ALGO_STRONG: return "strong_mfhoi";
        case MFHOI_ALGO_MFI: return "mfi_baseline";
    }
    return "unknown";
}

int mfhoi_parse_algorithm(const char *name, MFHOIAlgorithm *algo) {
    if (strcmp(name, "apriori") == 0 || strcmp(name, "fi") == 0) *algo = MFHOI_ALGO_APRIORI;
    else if (strcmp(name, "fhoi") == 0) *algo = MFHOI_ALGO_FHOI;
    else if (strcmp(name, "weak_mfhoi") == 0 || strcmp(name, "weak") == 0) *algo = MFHOI_ALGO_WEAK;
    else if (strcmp(name, "strong_mfhoi") == 0 || strcmp(name, "strong") == 0) *algo = MFHOI_ALGO_STRONG;
    else if (strcmp(name, "mfi_baseline") == 0 || strcmp(name, "mfi") == 0) *algo = MFHOI_ALGO_MFI;
    else return -1;
    return 0;
}

int mfhoi_write_patterns(const char *path, const PatternList *patterns, int transaction_count) {
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;
    for (int i = 0; i < patterns->count; i++) {
        const Pattern *p = &patterns->patterns[i];
        for (int j = 0; j < p->length; j++) fprintf(fp, "%s%d", j ? " " : "", p->items[j]);
        fprintf(fp, " | %d | %.8f | %.8f | %d\n", p->support, (double)p->support / (double)transaction_count, p->avg_occ, p->length);
    }
    fclose(fp);
    return 0;
}

int mfhoi_write_dominance_examples(const char *path, const TransactionDB *db, double minsup_ratio, double minocc, const PatternList *fhoi, int append) {
    FILE *fp = fopen(path, append ? "a" : "w");
    if (!fp) return -1;
    if (!append) {
        fprintf(fp, "dataset,minsup_ratio,minocc,removed_itemset_X,support_X,avg_occ_X,dominating_father_Y,support_Y,avg_occ_Y,size_X,size_Y\n");
    }
    int written = 0;
    for (int i = 0; i < fhoi->count && written < 50; i++) {
        const Pattern *x = &fhoi->patterns[i];
        for (int j = 0; j < fhoi->count; j++) {
            const Pattern *y = &fhoi->patterns[j];
            if (proper_subset(x, y) && y->avg_occ + 1e-12 >= x->avg_occ) {
                fprintf(fp, "%s,%.6f,%.6f,\"", db->name ? db->name : "dataset", minsup_ratio, minocc);
                for (int k = 0; k < x->length; k++) fprintf(fp, "%s%d", k ? " " : "", x->items[k]);
                fprintf(fp, "\",%d,%.8f,\"", x->support, x->avg_occ);
                for (int k = 0; k < y->length; k++) fprintf(fp, "%s%d", k ? " " : "", y->items[k]);
                fprintf(fp, "\",%d,%.8f,%d,%d\n", y->support, y->avg_occ, x->length, y->length);
                written++;
                break;
            }
        }
    }
    fclose(fp);
    return 0;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

void mfhoi_compute_quality(const PatternList *patterns, int transaction_count, MFHOIPatternQuality *q) {
    memset(q, 0, sizeof(*q));
    if (!patterns || patterns->count == 0) return;
    double *occs = malloc((size_t)patterns->count * sizeof(double));
    for (int i = 0; i < patterns->count; i++) {
        const Pattern *p = &patterns->patterns[i];
        q->avg_itemset_length += p->length;
        if (p->length > q->max_itemset_length) q->max_itemset_length = p->length;
        q->avg_support += p->support;
        q->avg_relative_support += (double)p->support / (double)transaction_count;
        q->avg_occupancy += p->avg_occ;
        if (p->avg_occ > q->max_occupancy) q->max_occupancy = p->avg_occ;
        occs[i] = p->avg_occ;
    }
    q->avg_itemset_length /= patterns->count;
    q->avg_support /= patterns->count;
    q->avg_relative_support /= patterns->count;
    q->avg_occupancy /= patterns->count;
    qsort(occs, (size_t)patterns->count, sizeof(double), cmp_double);
    if (patterns->count % 2) q->median_occupancy = occs[patterns->count / 2];
    else q->median_occupancy = (occs[patterns->count / 2 - 1] + occs[patterns->count / 2]) / 2.0;
    free(occs);
}

double mfhoi_exact_overlap_ratio(const PatternList *a, const PatternList *b, int *overlap_count) {
    int overlap = 0;
    if (!a || !b || a->count == 0) {
        if (overlap_count) *overlap_count = 0;
        return 0.0;
    }
    for (int i = 0; i < a->count; i++) {
        for (int j = 0; j < b->count; j++) {
            if (same_pattern(&a->patterns[i], &b->patterns[j])) {
                overlap++;
                break;
            }
        }
    }
    if (overlap_count) *overlap_count = overlap;
    return (double)overlap / (double)a->count;
}

static double jaccard_items(const Pattern *a, const Pattern *b) {
    int i = 0, j = 0, inter = 0, uni = 0;
    while (i < a->length || j < b->length) {
        if (i < a->length && (j >= b->length || a->items[i] < b->items[j])) {
            uni++; i++;
        } else if (j < b->length && (i >= a->length || b->items[j] < a->items[i])) {
            uni++; j++;
        } else {
            inter++; uni++; i++; j++;
        }
    }
    return uni ? (double)inter / (double)uni : 0.0;
}

double mfhoi_avg_jaccard_to_nearest(const PatternList *source, const PatternList *target) {
    if (!source || !target || source->count == 0 || target->count == 0) return 0.0;
    double total = 0.0;
    for (int i = 0; i < source->count; i++) {
        double best = 0.0;
        for (int j = 0; j < target->count; j++) {
            double jac = jaccard_items(&source->patterns[i], &target->patterns[j]);
            if (jac > best) best = jac;
        }
        total += best;
    }
    return total / (double)source->count;
}

static void ensure_dir(const char *path) {
    dm_mkdir(path, 0775);
}

static void write_random_transactions(FILE *fp, int transactions, int items, int len, unsigned int *seed) {
    int *buf = malloc((size_t)len * sizeof(int));
    for (int t = 0; t < transactions; t++) {
        for (int i = 0; i < len; i++) buf[i] = (int)(dm_rand_r(seed) % (unsigned int)items) + 1;
        qsort(buf, (size_t)len, sizeof(int), cmp_int);
        int last = -1;
        int printed = 0;
        for (int i = 0; i < len; i++) {
            if (buf[i] != last) {
                fprintf(fp, "%s%d", printed ? " " : "", buf[i]);
                printed = 1;
                last = buf[i];
            }
        }
        fprintf(fp, "\n");
    }
    free(buf);
}

int mfhoi_write_synthetic_datasets(const char *dataset_root) {
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s/synthetic", dataset_root);
    ensure_dir(dataset_root);
    ensure_dir(dir);
    unsigned int seed = 42;

    struct { const char *name; int transactions; int items; int len; } specs[] = {
        {"syn_sparse.txt", 10000, 1000, 10},
        {"syn_dense.txt", 10000, 200, 60},
        {"syn_long.txt", 10000, 500, 100}
    };
    for (size_t s = 0; s < sizeof(specs) / sizeof(specs[0]); s++) {
        char path[1200];
        snprintf(path, sizeof(path), "%s/%s", dir, specs[s].name);
        FILE *fp = fopen(path, "w");
        if (!fp) return -1;
        write_random_transactions(fp, specs[s].transactions, specs[s].items, specs[s].len, &seed);
        fclose(fp);
    }

    char path[1200];
    snprintf(path, sizeof(path), "%s/syn_correlated.txt", dir);
    FILE *fp = fopen(path, "w");
    if (!fp) return -1;
    for (int t = 0; t < 10000; t++) {
        int printed = 0;
        if (dm_rand_r(&seed) % 100 < 20) for (int i = 1; i <= 10; i++) { fprintf(fp, "%s%d", printed ? " " : "", i); printed = 1; }
        if (dm_rand_r(&seed) % 100 < 10) for (int i = 101; i <= 115; i++) { fprintf(fp, "%s%d", printed ? " " : "", i); printed = 1; }
        if (dm_rand_r(&seed) % 100 < 35) for (int i = 201; i <= 205; i++) { fprintf(fp, "%s%d", printed ? " " : "", i); printed = 1; }
        int extra = 5 + (int)(dm_rand_r(&seed) % 12);
        for (int i = 0; i < extra; i++) {
            int item = (int)(dm_rand_r(&seed) % 500) + 1;
            fprintf(fp, "%s%d", printed ? " " : "", item);
            printed = 1;
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}
