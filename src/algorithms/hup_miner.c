#include "algorithms/hup_miner.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t *items;
    size_t count;
} HUP_Itemset;

typedef struct {
    HUP_Itemset *itemsets;
    size_t itemset_count;
    size_t length;
    size_t support;
    double utility_sum;
    double average_utility;
    double upper_bound;
} HUP_Pattern;

typedef struct {
    HUP_Pattern *patterns;
    size_t count;
    size_t capacity;
} HUP_PatternList;

typedef struct {
    size_t *values;
    size_t count;
    size_t capacity;
} HUP_Positions;

typedef struct {
    DM_Sequence_Utility *seqs;
    size_t seq_count;
    double *item_utility;
    double max_item_utility;
    double threshold;
    size_t generated_candidates;
    size_t hubp_count;
    bool candidate_cap_reached;
} HUP_Context;

static void pos_add(HUP_Positions *pos, size_t value) {
    if (pos->count >= pos->capacity) {
        pos->capacity = pos->capacity ? pos->capacity * 2 : 8;
        pos->values = realloc(pos->values, sizeof(size_t) * pos->capacity);
    }
    pos->values[pos->count++] = value;
}

static void pos_free(HUP_Positions *pos) {
    free(pos->values);
    pos->values = NULL;
    pos->count = 0;
    pos->capacity = 0;
}

static void pattern_free(HUP_Pattern *p) {
    if (!p) return;
    for (size_t i = 0; i < p->itemset_count; i++) free(p->itemsets[i].items);
    free(p->itemsets);
    memset(p, 0, sizeof(*p));
}

static HUP_Pattern pattern_copy(const HUP_Pattern *src) {
    HUP_Pattern dst = {0};
    dst.itemset_count = src->itemset_count;
    dst.length = src->length;
    dst.support = src->support;
    dst.utility_sum = src->utility_sum;
    dst.average_utility = src->average_utility;
    dst.upper_bound = src->upper_bound;
    dst.itemsets = calloc(dst.itemset_count, sizeof(HUP_Itemset));
    for (size_t i = 0; i < dst.itemset_count; i++) {
        dst.itemsets[i].count = src->itemsets[i].count;
        dst.itemsets[i].items = malloc(sizeof(uint32_t) * dst.itemsets[i].count);
        memcpy(dst.itemsets[i].items, src->itemsets[i].items,
               sizeof(uint32_t) * dst.itemsets[i].count);
    }
    return dst;
}

static void list_add_copy(HUP_PatternList *list, const HUP_Pattern *pattern) {
    if (list->count >= list->capacity) {
        list->capacity = list->capacity ? list->capacity * 2 : 64;
        list->patterns = realloc(list->patterns, sizeof(HUP_Pattern) * list->capacity);
    }
    list->patterns[list->count++] = pattern_copy(pattern);
}

static void list_free(HUP_PatternList *list) {
    for (size_t i = 0; i < list->count; i++) pattern_free(&list->patterns[i]);
    free(list->patterns);
    list->patterns = NULL;
    list->count = 0;
    list->capacity = 0;
}

static bool itemset_contains_all(const DM_Trans_Sequence_Utility *itemset,
                                 const HUP_Itemset *pattern_itemset) {
    for (size_t i = 0; i < pattern_itemset->count; i++) {
        bool found = false;
        for (size_t j = 0; j < itemset->count; j++) {
            if (itemset->items[j].id == pattern_itemset->items[i]) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

static HUP_Positions itemset_positions(const DM_Sequence_Utility *seq, const HUP_Itemset *itemset) {
    HUP_Positions pos = {0};
    for (size_t i = 0; i < seq->count; i++) {
        if (itemset_contains_all(&seq->itemsets[i], itemset)) pos_add(&pos, i);
    }
    return pos;
}

static HUP_Positions pattern_positions(const DM_Sequence_Utility *seq, const HUP_Pattern *pattern) {
    if (pattern->itemset_count == 1) {
        return itemset_positions(seq, &pattern->itemsets[0]);
    }

    HUP_Pattern prefix = *pattern;
    prefix.itemset_count = pattern->itemset_count - 1;

    HUP_Positions prefix_pos = pattern_positions(seq, &prefix);
    HUP_Positions suffix_pos = itemset_positions(seq, &pattern->itemsets[pattern->itemset_count - 1]);
    HUP_Positions out = {0};
    size_t j = 0;

    for (size_t i = 0; i < prefix_pos.count; i++) {
        while (j < suffix_pos.count && suffix_pos.values[j] <= prefix_pos.values[i]) j++;
        if (j < suffix_pos.count) {
            pos_add(&out, suffix_pos.values[j]);
            j++;
        }
    }

    pos_free(&prefix_pos);
    pos_free(&suffix_pos);
    return out;
}

static bool pattern_equals(const HUP_Pattern *a, const HUP_Pattern *b) {
    if (a->itemset_count != b->itemset_count || a->length != b->length) return false;
    for (size_t i = 0; i < a->itemset_count; i++) {
        if (a->itemsets[i].count != b->itemsets[i].count) return false;
        for (size_t j = 0; j < a->itemsets[i].count; j++) {
            if (a->itemsets[i].items[j] != b->itemsets[i].items[j]) return false;
        }
    }
    return true;
}

static bool list_contains(const HUP_PatternList *list, const HUP_Pattern *pattern) {
    for (size_t i = 0; i < list->count; i++) {
        if (pattern_equals(&list->patterns[i], pattern)) return true;
    }
    return false;
}

static HUP_Pattern make_singleton(uint32_t item) {
    HUP_Pattern p = {0};
    p.itemset_count = 1;
    p.length = 1;
    p.itemsets = calloc(1, sizeof(HUP_Itemset));
    p.itemsets[0].count = 1;
    p.itemsets[0].items = malloc(sizeof(uint32_t));
    p.itemsets[0].items[0] = item;
    return p;
}

static HUP_Pattern make_length2(uint32_t first, uint32_t second, bool same_itemset) {
    HUP_Pattern p = {0};
    p.length = 2;
    p.itemset_count = same_itemset ? 1 : 2;
    p.itemsets = calloc(p.itemset_count, sizeof(HUP_Itemset));
    if (same_itemset) {
        p.itemsets[0].count = 2;
        p.itemsets[0].items = malloc(sizeof(uint32_t) * 2);
        p.itemsets[0].items[0] = first;
        p.itemsets[0].items[1] = second;
    } else {
        p.itemsets[0].count = 1;
        p.itemsets[0].items = malloc(sizeof(uint32_t));
        p.itemsets[0].items[0] = first;
        p.itemsets[1].count = 1;
        p.itemsets[1].items = malloc(sizeof(uint32_t));
        p.itemsets[1].items[0] = second;
    }
    return p;
}

static void evaluate_pattern(HUP_Context *ctx, HUP_Pattern *pattern) {
    pattern->support = 0;
    pattern->utility_sum = 0.0;
    for (size_t i = 0; i < pattern->itemset_count; i++) {
        for (size_t j = 0; j < pattern->itemsets[i].count; j++) {
            pattern->utility_sum += ctx->item_utility[pattern->itemsets[i].items[j]];
        }
    }
    for (size_t sid = 0; sid < ctx->seq_count; sid++) {
        HUP_Positions pos = pattern_positions(&ctx->seqs[sid], pattern);
        pattern->support += pos.count;
        pos_free(&pos);
    }
    pattern->average_utility = pattern->length
        ? pattern->utility_sum * (double)pattern->support / (double)pattern->length
        : 0.0;
    pattern->upper_bound = ctx->max_item_utility * (double)pattern->support;
}

static HUP_Pattern prefix_pattern(const HUP_Pattern *p) {
    HUP_Pattern out = pattern_copy(p);
    HUP_Itemset *last = &out.itemsets[out.itemset_count - 1];
    last->count--;
    out.length--;
    if (last->count == 0) {
        free(last->items);
        out.itemset_count--;
    }
    return out;
}

static HUP_Pattern suffix_pattern(const HUP_Pattern *p) {
    HUP_Pattern out = pattern_copy(p);
    HUP_Itemset *first = &out.itemsets[0];
    memmove(first->items, first->items + 1, sizeof(uint32_t) * (first->count - 1));
    first->count--;
    out.length--;
    if (first->count == 0) {
        free(first->items);
        memmove(out.itemsets, out.itemsets + 1, sizeof(HUP_Itemset) * (out.itemset_count - 1));
        out.itemset_count--;
    }
    return out;
}

static HUP_Pattern join_patterns(const HUP_Pattern *p, const HUP_Pattern *q) {
    HUP_Pattern out = pattern_copy(p);
    const HUP_Itemset *q_last = &q->itemsets[q->itemset_count - 1];
    uint32_t item = q_last->items[q_last->count - 1];

    if (q_last->count > 1) {
        HUP_Itemset *out_last = &out.itemsets[out.itemset_count - 1];
        out_last->items = realloc(out_last->items, sizeof(uint32_t) * (out_last->count + 1));
        out_last->items[out_last->count++] = item;
    } else {
        out.itemsets = realloc(out.itemsets, sizeof(HUP_Itemset) * (out.itemset_count + 1));
        out.itemsets[out.itemset_count].count = 1;
        out.itemsets[out.itemset_count].items = malloc(sizeof(uint32_t));
        out.itemsets[out.itemset_count].items[0] = item;
        out.itemset_count++;
    }
    out.length++;
    out.support = 0;
    out.utility_sum = 0.0;
    out.average_utility = 0.0;
    out.upper_bound = 0.0;
    return out;
}

static void format_pattern(const HUP_Pattern *p, char *buf, size_t buf_size) {
    size_t used = 0;
    for (size_t i = 0; i < p->itemset_count; i++) {
        used += (size_t)snprintf(buf + used, used < buf_size ? buf_size - used : 0, "[");
        for (size_t j = 0; j < p->itemsets[i].count; j++) {
            used += (size_t)snprintf(buf + used, used < buf_size ? buf_size - used : 0,
                                     "%s%u", j ? " " : "", p->itemsets[i].items[j]);
        }
        used += (size_t)snprintf(buf + used, used < buf_size ? buf_size - used : 0, "]");
    }
}

static void mine_level(HUP_Context *ctx, HUP_PatternList *level, HUP_PatternList *hups,
                       size_t max_length, size_t max_candidates) {
    size_t length = 3;
    while (level->count > 0 && (max_length == 0 || length <= max_length)) {
        HUP_PatternList next = {0};
        for (size_t i = 0; i < level->count; i++) {
            HUP_Pattern suffix = suffix_pattern(&level->patterns[i]);
            for (size_t j = 0; j < level->count; j++) {
                HUP_Pattern prefix = prefix_pattern(&level->patterns[j]);
                bool can_join = pattern_equals(&suffix, &prefix);
                pattern_free(&prefix);
                if (!can_join) continue;

                if (max_candidates && ctx->generated_candidates >= max_candidates) {
                    ctx->candidate_cap_reached = true;
                    break;
                }

                HUP_Pattern candidate = join_patterns(&level->patterns[i], &level->patterns[j]);
                if (!list_contains(&next, &candidate)) {
                    ctx->generated_candidates++;
                    evaluate_pattern(ctx, &candidate);
                    if (candidate.average_utility >= ctx->threshold) list_add_copy(hups, &candidate);
                    if (candidate.upper_bound >= ctx->threshold) {
                        list_add_copy(&next, &candidate);
                        ctx->hubp_count++;
                    }
                }
                pattern_free(&candidate);
            }
            pattern_free(&suffix);
            if (ctx->candidate_cap_reached) break;
        }
        list_free(level);
        *level = next;
        if (ctx->candidate_cap_reached) break;
        length++;
    }
}

static int cmp_uint32(const void *a, const void *b) {
    uint32_t av = *(const uint32_t *)a;
    uint32_t bv = *(const uint32_t *)b;
    return (av > bv) - (av < bv);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_SEQUENCE_UTILITY) return DM_ERROR_INCOMPATIBLE;

    DM_HUP_Miner_Params *p = (DM_HUP_Miner_Params *)params;
    double threshold = p ? p->min_average_utility : 100.0;
    size_t max_length = p ? p->max_pattern_length : 0;
    size_t max_candidates = p ? p->max_candidates : 0;
    DM_Sequence_Utility *seqs = (DM_Sequence_Utility *)ds->payload;

    double *utility_sum = calloc((size_t)ds->max_id + 1, sizeof(double));
    size_t *utility_count = calloc((size_t)ds->max_id + 1, sizeof(size_t));
    uint32_t *items = NULL;
    size_t item_count = 0, item_capacity = 0;
    bool *seen = calloc((size_t)ds->max_id + 1, sizeof(bool));

    for (size_t sid = 0; sid < ds->count; sid++) {
        for (size_t is = 0; is < seqs[sid].count; is++) {
            for (size_t ii = 0; ii < seqs[sid].itemsets[is].count; ii++) {
                DM_Item it = seqs[sid].itemsets[is].items[ii];
                utility_sum[it.id] += it.utility;
                utility_count[it.id]++;
                if (!seen[it.id]) {
                    seen[it.id] = true;
                    if (item_count >= item_capacity) {
                        item_capacity = item_capacity ? item_capacity * 2 : 128;
                        items = realloc(items, sizeof(uint32_t) * item_capacity);
                    }
                    items[item_count++] = it.id;
                }
            }
        }
    }
    qsort(items, item_count, sizeof(uint32_t), cmp_uint32);

    double *item_utility = calloc((size_t)ds->max_id + 1, sizeof(double));
    double max_item_utility = 0.0;
    for (size_t i = 0; i < item_count; i++) {
        uint32_t item = items[i];
        item_utility[item] = utility_count[item] ? utility_sum[item] / (double)utility_count[item] : 0.0;
        if (item_utility[item] > max_item_utility) max_item_utility = item_utility[item];
    }

    HUP_Context ctx = {
        .seqs = seqs,
        .seq_count = ds->count,
        .item_utility = item_utility,
        .max_item_utility = max_item_utility,
        .threshold = threshold
    };
    HUP_PatternList hups = {0};
    HUP_PatternList level1 = {0};
    HUP_PatternList level2 = {0};

    printf("[HUP-Miner] Mining HUPs (rho: %.2f, max length: %s, max candidates: %s)...\n",
           threshold, max_length ? "bounded" : "unbounded", max_candidates ? "bounded" : "unbounded");

    for (size_t i = 0; i < item_count; i++) {
        HUP_Pattern pattern = make_singleton(items[i]);
        ctx.generated_candidates++;
        evaluate_pattern(&ctx, &pattern);
        if (pattern.average_utility >= threshold) list_add_copy(&hups, &pattern);
        if (pattern.upper_bound >= threshold) {
            list_add_copy(&level1, &pattern);
            ctx.hubp_count++;
        }
        pattern_free(&pattern);
    }

    if (max_length == 0 || max_length >= 2) {
        for (size_t i = 0; i < level1.count; i++) {
            uint32_t first = level1.patterns[i].itemsets[0].items[0];
            for (size_t j = 0; j < level1.count; j++) {
                uint32_t second = level1.patterns[j].itemsets[0].items[0];
                if (!max_candidates || ctx.generated_candidates < max_candidates) {
                    HUP_Pattern s_pattern = make_length2(first, second, false);
                    ctx.generated_candidates++;
                    evaluate_pattern(&ctx, &s_pattern);
                    if (s_pattern.average_utility >= threshold) list_add_copy(&hups, &s_pattern);
                    if (s_pattern.upper_bound >= threshold) {
                        list_add_copy(&level2, &s_pattern);
                        ctx.hubp_count++;
                    }
                    pattern_free(&s_pattern);
                } else {
                    ctx.candidate_cap_reached = true;
                    break;
                }

                if (first < second) {
                    if (max_candidates && ctx.generated_candidates >= max_candidates) {
                        ctx.candidate_cap_reached = true;
                        break;
                    }
                    HUP_Pattern i_pattern = make_length2(first, second, true);
                    ctx.generated_candidates++;
                    evaluate_pattern(&ctx, &i_pattern);
                    if (i_pattern.average_utility >= threshold) list_add_copy(&hups, &i_pattern);
                    if (i_pattern.upper_bound >= threshold) {
                        list_add_copy(&level2, &i_pattern);
                        ctx.hubp_count++;
                    }
                    pattern_free(&i_pattern);
                }
            }
            if (ctx.candidate_cap_reached) break;
        }
    }

    list_free(&level1);
    if (!ctx.candidate_cap_reached && (max_length == 0 || max_length >= 3)) {
        mine_level(&ctx, &level2, &hups, max_length, max_candidates);
    } else {
        list_free(&level2);
    }

    size_t total_items = 0;
    for (size_t i = 0; i < hups.count; i++) total_items += hups.patterns[i].length;

    printf("[HUP-Miner] Found %zu High Average Utility Nonoverlapping Patterns.\n", hups.count);
    printf("[HUP-Miner] Generated candidates: %zu, HUBPs kept: %zu\n",
           ctx.generated_candidates, ctx.hubp_count);
    if (ctx.candidate_cap_reached) {
        printf("[HUP-Miner] Candidate cap reached; increase max_candidates for a fuller run.\n");
    }
    for (size_t i = 0; i < hups.count && i < 10; i++) {
        char buf[512];
        format_pattern(&hups.patterns[i], buf, sizeof(buf));
        printf("  %s | support=%zu | avg_utility=%.4f | length=%zu\n",
               buf, hups.patterns[i].support, hups.patterns[i].average_utility,
               hups.patterns[i].length);
    }

    dm_bench_record_results(hups.count, total_items);

    list_free(&hups);
    free(utility_sum);
    free(utility_count);
    free(item_utility);
    free(items);
    free(seen);
    return DM_SUCCESS;
}

DM_Algorithm hup_miner_algo = {
    .id = "hup_miner",
    .name = "HUP-Miner",
    .description = "High Average Utility Nonoverlapping Pattern mining for sequence utility databases.",
    .supported_types = (1 << DM_TYPE_SEQUENCE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(hup_miner_algo)
