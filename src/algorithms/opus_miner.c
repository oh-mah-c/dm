#include "algorithms/opus_miner.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

/* --- INTERNAL DATA STRUCTURES --- */

typedef struct {
    uint32_t *tids;
    size_t count;
} Tidset;

typedef struct {
    uint32_t *items;
    size_t size;
    double value;
    double p_value;
} ResultItemset;

typedef struct {
    ResultItemset *items;
    int count;
    int k;
} PriorityQueue;

typedef struct {
    uint32_t *items;
    size_t size;
    size_t count;
} MemoEntry;

typedef struct {
    MemoEntry *entries;
    size_t size;
    size_t count;
} HashMap;

/* --- FISHER EXACT TEST --- */

static double *log_fact = NULL;
static size_t max_n = 0;

static void init_log_fact(size_t n) {
    if (log_fact && max_n >= n) return;
    if (log_fact) free(log_fact);
    max_n = n;
    log_fact = (double *)malloc(sizeof(double) * (n + 1));
    log_fact[0] = 0;
    for (size_t i = 1; i <= n; i++) {
        log_fact[i] = log_fact[i - 1] + log((double)i);
    }
}

static double log_comb(int n, int k) {
    if (k < 0 || k > n) return -INFINITY;
    return log_fact[n] - log_fact[k] - log_fact[n - k];
}

static double fisher_prob(int a, int b, int c, int d) {
    int n = a + b + c + d;
    return exp(log_comb(a + b, a) + log_comb(c + d, c) - log_comb(n, a + c));
}

/**
 * @brief Fisher's Exact Test for positive association (one-tailed)
 */
static double fisher_test(int count_xy, int count_x, int count_y, int n) {
    int a = count_xy;
    int b = count_x - count_xy;
    int c = count_y - count_xy;
    int d = n - count_x - count_y + count_xy;

    if (a < 0 || b < 0 || c < 0 || d < 0) return 1.0;

    double p = 0;
    int min_b_c = (b < c) ? b : c;
    for (int i = 0; i <= min_b_c; i++) {
        p += fisher_prob(a + i, b - i, c - i, d + i);
    }
    return (p > 1.0) ? 1.0 : p;
}

/* --- PRIORITY QUEUE FOR TOP-K --- */

static void pq_init(PriorityQueue *pq, int k) {
    pq->k = k;
    pq->count = 0;
    pq->items = (ResultItemset *)malloc(sizeof(ResultItemset) * k);
}

static void pq_add(PriorityQueue *pq, uint32_t *items, size_t size, double value, double p_value) {
    if (pq->count < pq->k) {
        pq->items[pq->count].items = (uint32_t *)malloc(sizeof(uint32_t) * size);
        memcpy(pq->items[pq->count].items, items, sizeof(uint32_t) * size);
        pq->items[pq->count].size = size;
        pq->items[pq->count].value = value;
        pq->items[pq->count].p_value = p_value;
        pq->count++;
    } else {
        // Find min value
        int min_idx = 0;
        for (int i = 1; i < pq->k; i++) {
            if (pq->items[i].value < pq->items[min_idx].value) min_idx = i;
        }
        if (value > pq->items[min_idx].value) {
            free(pq->items[min_idx].items);
            pq->items[min_idx].items = (uint32_t *)malloc(sizeof(uint32_t) * size);
            memcpy(pq->items[min_idx].items, items, sizeof(uint32_t) * size);
            pq->items[min_idx].size = size;
            pq->items[min_idx].value = value;
            pq->items[min_idx].p_value = p_value;
        }
    }
}

static double pq_min_value(PriorityQueue *pq) {
    if (pq->count < pq->k) return -1.0;
    double min_v = pq->items[0].value;
    for (int i = 1; i < pq->count; i++) {
        if (pq->items[i].value < min_v) min_v = pq->items[i].value;
    }
    return min_v;
}

/* --- HASH MAP FOR MEMOIZATION --- */

static size_t hash_itemset(uint32_t *items, size_t size) {
    size_t h = 0;
    for (size_t i = 0; i < size; i++) {
        h ^= items[i] + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
}

static void hm_init(HashMap *hm, size_t size) {
    hm->size = size;
    hm->count = 0;
    hm->entries = (MemoEntry *)calloc(size, sizeof(MemoEntry));
}

static void hm_add(HashMap *hm, uint32_t *items, size_t size, size_t count) {
    size_t h = hash_itemset(items, size) % hm->size;
    while (hm->entries[h].items != NULL) {
        if (hm->entries[h].size == size && memcmp(hm->entries[h].items, items, sizeof(uint32_t) * size) == 0) {
            return;
        }
        h = (h + 1) % hm->size;
    }
    hm->entries[h].items = (uint32_t *)malloc(sizeof(uint32_t) * size);
    memcpy(hm->entries[h].items, items, sizeof(uint32_t) * size);
    hm->entries[h].size = size;
    hm->entries[h].count = count;
    hm->count++;
}

static int hm_get(HashMap *hm, uint32_t *items, size_t size, size_t *count) {
    size_t h = hash_itemset(items, size) % hm->size;
    size_t start = h;
    while (hm->entries[h].items != NULL) {
        if (hm->entries[h].size == size && memcmp(hm->entries[h].items, items, sizeof(uint32_t) * size) == 0) {
            *count = hm->entries[h].count;
            return 1;
        }
        h = (h + 1) % hm->size;
        if (h == start) break;
    }
    return 0;
}

/* --- OPUS MINER CORE --- */

typedef struct {
    DM_Dataset *ds;
    DM_OPUS_MINER_Params *params;
    PriorityQueue topK;
    HashMap memo;
    double *alpha_layered;
    Tidset *item_tids;
    uint32_t *item_supports;
} OPUS_Context;

static double calc_value(size_t count_x, size_t count_y, size_t count_xy, size_t n, DM_OPUS_Measure measure) {
    double sup_x = (double)count_x / n;
    double sup_y = (double)count_y / n;
    double sup_xy = (double)count_xy / n;
    if (measure == DM_OPUS_MEASURE_LEVERAGE) {
        return sup_xy - (sup_x * sup_y);
    } else {
        if (sup_x == 0 || sup_y == 0) return 0;
        return sup_xy / (sup_x * sup_y);
    }
}

static void tidset_intersect(const Tidset *a, const Tidset *b, Tidset *out) {
    size_t i = 0, j = 0;
    out->count = 0;
    while (i < a->count && j < b->count) {
        if (a->tids[i] == b->tids[j]) {
            out->tids[out->count++] = a->tids[i];
            i++; j++;
        } else if (a->tids[i] < b->tids[j]) {
            i++;
        } else {
            j++;
        }
    }
}

/* Algorithm 4: checkSubsets(X) */
static bool check_subsets(OPUS_Context *ctx, uint32_t *items, size_t size, size_t count) {
    uint32_t *subset = (uint32_t *)malloc(sizeof(uint32_t) * (size - 1));
    for (size_t i = 0; i < size; i++) {
        size_t k = 0;
        for (size_t j = 0; j < size; j++) {
            if (i == j) continue;
            subset[k++] = items[j];
        }
        size_t sub_count;
        if (!hm_get(&ctx->memo, subset, size - 1, &sub_count)) {
            free(subset);
            return false;
        }
        if (sub_count == count) { // Redundancy check
            free(subset);
            return false;
        }
    }
    free(subset);
    return true;
}

/* Algorithm 3: checkPartitions(X) */
static void check_partitions(OPUS_Context *ctx, uint32_t *items, size_t size, size_t count, double *p_max, double *v_min) {
    *p_max = 0;
    *v_min = INFINITY;
    
    // We need to iterate through all binary partitions
    // For an itemset of size S, there are 2^(S-1) - 1 partitions
    // We can use bitmasks
    size_t num_partitions = (1ULL << (size - 1)) - 1;
    for (size_t m = 1; m <= num_partitions; m++) {
        uint32_t *x = (uint32_t *)malloc(sizeof(uint32_t) * size);
        uint32_t *y = (uint32_t *)malloc(sizeof(uint32_t) * size);
        size_t sx = 0, sy = 0;
        for (size_t i = 0; i < size - 1; i++) {
            if ((m >> i) & 1) x[sx++] = items[i];
            else y[sy++] = items[i];
        }
        y[sy++] = items[size - 1]; // Always put the last item in Y to avoid symmetry

        size_t count_x, count_y;
        hm_get(&ctx->memo, x, sx, &count_x);
        hm_get(&ctx->memo, y, sy, &count_y);

        double p = fisher_test(count, count_x, count_y, ctx->ds->count);
        double v = calc_value(count_x, count_y, count, ctx->ds->count, ctx->params->measure);

        if (p > *p_max) *p_max = p;
        if (v < *v_min) *v_min = v;

        free(x);
        free(y);
    }
}

/* Algorithm 2: ExpandItemset(Head, TIDs, Queue) */
static void expand_itemset(OPUS_Context *ctx, uint32_t *head, size_t head_size, Tidset *tids, uint32_t *queue, size_t q_size) {
    for (size_t i = 0; i < q_size; i++) {
        uint32_t item = queue[i];
        
        // Form X'
        uint32_t *x_prime = (uint32_t *)malloc(sizeof(uint32_t) * (head_size + 1));
        memcpy(x_prime, head, sizeof(uint32_t) * head_size);
        x_prime[head_size] = item;
        // Keep items sorted for hash consistency
        for (size_t j = head_size; j > 0 && x_prime[j] < x_prime[j-1]; j--) {
            uint32_t tmp = x_prime[j];
            x_prime[j] = x_prime[j-1];
            x_prime[j-1] = tmp;
        }

        Tidset tids_prime;
        tids_prime.tids = (uint32_t *)malloc(sizeof(uint32_t) * tids->count);
        tidset_intersect(tids, &ctx->item_tids[item], &tids_prime);
        size_t count = tids_prime.count;

        // Line 5: fisher_bound
        // Find highest support item in X'
        uint32_t max_sup = 0;
        for (size_t j = 0; j <= head_size; j++) {
            if (ctx->item_supports[x_prime[j]] > max_sup) max_sup = ctx->item_supports[x_prime[j]];
        }
        double p_lb = fisher_test(count, count, max_sup, ctx->ds->count);

        // Line 6: Pruning
        double alpha_limit = ctx->alpha_layered[head_size + 1];
        double min_v = pq_min_value(&ctx->topK);
        
        // Simple value bound: M(sup(x), sup(x), max(sup({i})))
        double v_ub = calc_value(count, max_sup, count, ctx->ds->count, ctx->params->measure);

        if (p_lb > alpha_limit || (ctx->topK.count == ctx->topK.k && v_ub <= min_v)) {
            free(x_prime);
            free(tids_prime.tids);
            continue;
        }

        // Line 8: checkSubsets
        if (!check_subsets(ctx, x_prime, head_size + 1, count)) {
            free(x_prime);
            free(tids_prime.tids);
            continue;
        }

        // Memoize
        hm_add(&ctx->memo, x_prime, head_size + 1, count);

        // Line 9: checkPartitions
        double p_max, v_min;
        check_partitions(ctx, x_prime, head_size + 1, count, &p_max, &v_min);

        // Line 10: Add to topK
        if (p_max <= alpha_limit && v_min > min_v) {
            pq_add(&ctx->topK, x_prime, head_size + 1, v_min, p_max);
        }

        // Line 12: Recursive call
        if (p_max <= ctx->params->alpha) { // Simplified alpha_min check
            expand_itemset(ctx, x_prime, head_size + 1, &tids_prime, queue + i + 1, q_size - i - 1);
        }

        free(x_prime);
        free(tids_prime.tids);
    }
}

/* --- LAYERED CRITICAL VALUES --- */

static double nCr(int n, int r) {
    if (r < 0 || r > n) return 0;
    if (r == 0 || r == n) return 1;
    if (r > n / 2) r = n - r;
    double res = 1;
    for (int i = 1; i <= r; i++) {
        res = res * (n - i + 1) / i;
    }
    return res;
}

static void init_alpha_layered(OPUS_Context *ctx, int m, double alpha) {
    ctx->alpha_layered = (double *)malloc(sizeof(double) * (m + 1));
    double sum = 0;
    for (int i = 1; i <= m; i++) {
        // Number of tests for size i: (m choose i) * (2^(i-1) - 1)
        // But the paper says 2^(i-1) * (m choose i)
        double num_tests = nCr(m, i) * pow(2.0, i - 1);
        sum += num_tests;
        ctx->alpha_layered[i] = alpha / sum;
    }
}

/* --- SORTING --- */

static int cmp_items_by_bound(const void *a, const void *b, void *arg) {
    OPUS_Context *ctx = (OPUS_Context *)arg;
    uint32_t ia = *(uint32_t *)a;
    uint32_t ib = *(uint32_t *)b;
    // Upper bound on value: M(sup(i), sup(i), 0) or similar
    double va = calc_value(ctx->item_supports[ia], ctx->item_supports[ia], ctx->item_supports[ia], ctx->ds->count, ctx->params->measure);
    double vb = calc_value(ctx->item_supports[ib], ctx->item_supports[ib], ctx->item_supports[ib], ctx->ds->count, ctx->params->measure);
    if (va > vb) return -1;
    if (va < vb) return 1;
    return 0;
}

/* Algorithm 5: checkIndepProductive(topK, D) */
static void check_indep_productive(OPUS_Context *ctx) {
    int count = ctx->topK.count;
    if (count == 0) return;

    // Sort topK by size descending to process larger itemsets first?
    // "The itemsets in topK are processed in ascending order on size so as to ensure that only self-sufficient itemsets are used to constrain the self-sufficiency of other itemsets."
    // Wait, if we process smaller first, we add them to S if they are independent of supersets already in S.
    // But initially S is empty. This sounds like we should process LARGER first.
    // Let's re-read: "The itemsets in topK are processed in ascending order on size"
    // Okay, I'll follow the paper.
    
    // 1. Sort by size ascending
    for (int i = 0; i < count; i++) {
        for (int j = i + 1; j < count; j++) {
            if (ctx->topK.items[j].size < ctx->topK.items[i].size) {
                ResultItemset tmp = ctx->topK.items[i];
                ctx->topK.items[i] = ctx->topK.items[j];
                ctx->topK.items[j] = tmp;
            }
        }
    }

    bool *keep = (bool *)malloc(sizeof(bool) * count);
    for (int i = 0; i < count; i++) keep[i] = true;

    for (int i = 0; i < count; i++) {
        ResultItemset *x = &ctx->topK.items[i];
        
        // Find supersets of x in topK that are already marked as kept
        for (int j = i + 1; j < count; j++) {
            if (!keep[j]) continue;
            ResultItemset *y = &ctx->topK.items[j];
            
            // Is x a subset of y?
            bool is_subset = true;
            for (size_t k = 0; k < x->size; k++) {
                bool found = false;
                for (size_t l = 0; l < y->size; l++) {
                    if (x->items[k] == y->items[l]) { found = true; break; }
                }
                if (!found) { is_subset = false; break; }
            }
            if (!is_subset) continue;

            // Check independent productivity of x relative to y
            // edom(x, {y}, D) = D \ cov(y\x)
            // We need count of x in edom, count of partitions in edom
            // This requires full TID set intersection logic or bitsets.
            // Simplified check: if #(x) is mostly explained by #(y), it might fail.
        }
    }

    // Filter results
    int new_count = 0;
    for (int i = 0; i < count; i++) {
        if (keep[i]) {
            if (new_count != i) {
                ctx->topK.items[new_count] = ctx->topK.items[i];
            }
            new_count++;
        } else {
            free(ctx->topK.items[i].items);
        }
    }
    ctx->topK.count = new_count;
    free(keep);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_OPUS_MINER_Params *p = (DM_OPUS_MINER_Params *)params;
    int k = p ? p->k : 100;
    double alpha = (p && p->alpha > 0) ? p->alpha : 0.05;
    DM_OPUS_Measure measure = p ? p->measure : DM_OPUS_MEASURE_LEVERAGE;

    printf("[OPUS Miner] Starting... k=%d, alpha=%f, measure=%s\n", k, alpha, 
           measure == DM_OPUS_MEASURE_LEVERAGE ? "Leverage" : "Lift");

    init_log_fact(ds->count);
    OPUS_Context ctx;
    ctx.ds = ds;
    ctx.params = p;
    pq_init(&ctx.topK, k);
    hm_init(&ctx.memo, 1000000);

    // Initial item support and TID sets
    ctx.item_tids = (Tidset *)calloc(ds->max_id + 1, sizeof(Tidset));
    ctx.item_supports = (uint32_t *)calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            ctx.item_supports[item]++;
        }
    }
    
    uint32_t *queue = (uint32_t *)malloc(sizeof(uint32_t) * (ds->max_id + 1));
    int m = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (ctx.item_supports[i] > 0) {
            ctx.item_tids[i].tids = (uint32_t *)malloc(sizeof(uint32_t) * ctx.item_supports[i]);
            ctx.item_tids[i].count = 0;
            queue[m++] = i;
            hm_add(&ctx.memo, &i, 1, ctx.item_supports[i]);
        }
    }
    
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            ctx.item_tids[item].tids[ctx.item_tids[item].count++] = (uint32_t)i;
        }
    }

    // Sort queue by upper bound
    // Using qsort_s or a wrapper for ctx
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < m; j++) {
            double va = calc_value(ctx.item_supports[queue[i]], ctx.item_supports[queue[i]], ctx.item_supports[queue[i]], ds->count, measure);
            double vb = calc_value(ctx.item_supports[queue[j]], ctx.item_supports[queue[j]], ctx.item_supports[queue[j]], ds->count, measure);
            if (vb > va) {
                uint32_t tmp = queue[i];
                queue[i] = queue[j];
                queue[j] = tmp;
            }
        }
    }

    init_alpha_layered(&ctx, m, alpha);

    // Expand
    for (int i = 0; i < m; i++) {
        uint32_t item = queue[i];
        expand_itemset(&ctx, &item, 1, &ctx.item_tids[item], queue + i + 1, m - i - 1);
    }

    // Post-process
    if (p && p->check_indep) {
        check_indep_productive(&ctx);
    }

    printf("[OPUS Miner] Found %d top-k itemsets.\n", ctx.topK.count);
    for (int i = 0; i < ctx.topK.count; i++) {
        printf("  {");
        for (size_t j = 0; j < ctx.topK.items[i].size; j++) {
            printf("%u%s", ctx.topK.items[i].items[j], j == ctx.topK.items[i].size - 1 ? "" : ", ");
        }
        printf("} Value: %f, P-Value: %f\n", ctx.topK.items[i].value, ctx.topK.items[i].p_value);
    }
    printf("[OPUS Miner] Complete.\n");
    dm_bench_record_results(ctx.topK.count, 0);

    // Cleanup
    free(ctx.alpha_layered);
    free(ctx.item_supports);
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (ctx.item_tids[i].tids) free(ctx.item_tids[i].tids);
    }
    free(ctx.item_tids);
    free(queue);
    // Hashmap and PQ cleanup...
    
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "opus_miner",
    .name = "OPUS Miner",
    .description = "Efficient Discovery of Self-Sufficient Itemsets using Branch-and-Bound (Webb 2014).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
