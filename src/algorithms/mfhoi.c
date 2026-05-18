#include "algorithms/mfhoi.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int cmp_int_local(const void *a, const void *b) {
    int x = *(const int *)a;
    int y = *(const int *)b;
    return (x > y) - (x < y);
}

static TransactionDB *db_from_dm_dataset(DM_Dataset *ds) {
    DM_Trans_Simple *src = (DM_Trans_Simple *)ds->payload;
    TransactionDB *db = calloc(1, sizeof(TransactionDB));
    db->transaction_count = (int)ds->count;
    db->item_count = (int)ds->max_id + 1;
    db->max_item_id = (int)ds->max_id;
    db->transactions = calloc(ds->count, sizeof(Transaction));
    db->transaction_lengths = calloc(ds->count, sizeof(int));

    for (size_t tid = 0; tid < ds->count; tid++) {
        db->transactions[tid].items = malloc(src[tid].count * sizeof(int));
        for (size_t j = 0; j < src[tid].count; j++) {
            db->transactions[tid].items[j] = (int)src[tid].items[j];
        }
        qsort(db->transactions[tid].items, src[tid].count, sizeof(int), cmp_int_local);
        int unique_len = 0;
        for (size_t j = 0; j < src[tid].count; j++) {
            if (unique_len == 0 || db->transactions[tid].items[j] != db->transactions[tid].items[unique_len - 1]) {
                db->transactions[tid].items[unique_len++] = db->transactions[tid].items[j];
            }
        }
        db->transactions[tid].length = unique_len;
        db->transaction_lengths[tid] = unique_len;
    }
    return db;
}

DM_Status mfhoi_adapter_run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_TRANSACTIONAL || !ds->payload || ds->count == 0) {
        return DM_ERROR_INVALID_PARAM;
    }

    DM_MFHOI_Params *p = (DM_MFHOI_Params *)params;
    double alpha = p ? p->min_support : 0.01;
    double minocc = p ? p->min_occupancy : 0.3;
    int minsup = alpha < 1.0 ? (int)ceil(alpha * (double)ds->count) : (int)ceil(alpha);
    if (minsup < 1) minsup = 1;

    printf("[MFHOI] Starting on %zu transactions. min_support=%d, min_occupancy=%.6g, variant=%s\n",
           ds->count, minsup, minocc, (p && p->strong) ? "strong" : "weak");

    TransactionDB *db = db_from_dm_dataset(ds);
    MFHOILimits limits = { .max_patterns = 2000000, .max_seconds = 0.0 };
    MFHOIResult result;
    int status = mfhoi_mine_all(db, minsup, minocc, &limits, &result);
    MFHOIAlgorithm output_algo = (p && p->strong) ? MFHOI_ALGO_STRONG : MFHOI_ALGO_WEAK;
    if (p && p->output_algorithm >= MFHOI_ALGO_APRIORI && p->output_algorithm <= MFHOI_ALGO_MFI) {
        output_algo = (MFHOIAlgorithm)p->output_algorithm;
    }
    const PatternList *out = mfhoi_select_patterns(&result, output_algo);

    printf("[MFHOI] Complete. Frequent=%d FHOI=%d Weak=%d Strong=%d Output=%d%s\n",
           result.stats.num_frequent_itemsets, result.stats.num_fhoi,
           result.stats.num_weak_mfhoi, result.stats.num_strong_mfhoi,
           out->count, status ? " (limited)" : "");
    printf("[MFHOI] Candidates=%d UB1-pruned=%d UB2-pruned=%d Dominance-removed=%d Weak-removed=%d\n",
           result.stats.num_generated_candidates, result.stats.pruned_ub1_count,
           result.stats.pruned_ub2_count, result.stats.dominance_removed_count,
           result.stats.weak_removed_count);

    int footprint = 0;
    for (int i = 0; i < out->count; i++) footprint += out->patterns[i].length;
    dm_bench_record_results((size_t)out->count, (size_t)footprint);

    mfhoi_result_free(&result);
    mfhoi_free_transaction_db(db);
    return DM_SUCCESS;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    return mfhoi_adapter_run(ds, params);
}

static DM_Algorithm algo = {
    .id = "mfhoi",
    .name = "MFHOI Algorithm",
    .description = "Weak/Strong Maximal Frequent High-Occupancy Itemset Mining.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
