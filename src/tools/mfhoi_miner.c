#include "algorithms/mfhoi.h"
#include "core/experiment.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s --input <dataset> --algorithm <apriori|fhoi|weak_mfhoi|strong_mfhoi|mfi_baseline> --minsup <ratio|count> --minocc <value> --output <path> [--max-patterns N] [--max-seconds S]\n", prog);
}

static const char *arg_value(int argc, char **argv, const char *key) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return NULL;
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input");
    const char *algo_name = arg_value(argc, argv, "--algorithm");
    const char *minsup_s = arg_value(argc, argv, "--minsup");
    const char *minocc_s = arg_value(argc, argv, "--minocc");
    const char *output = arg_value(argc, argv, "--output");
    const char *max_patterns_s = arg_value(argc, argv, "--max-patterns");
    const char *max_seconds_s = arg_value(argc, argv, "--max-seconds");

    if (!input || !algo_name || !minsup_s || !minocc_s || !output) {
        usage(argv[0]);
        return 1;
    }

    MFHOIAlgorithm algo;
    if (mfhoi_parse_algorithm(algo_name, &algo) != 0) {
        fprintf(stderr, "Unknown MFHOI algorithm: %s\n", algo_name);
        return 1;
    }

    TransactionDB *db = mfhoi_load_transaction_db(input);
    if (!db) {
        fprintf(stderr, "Cannot load dataset: %s\n", input);
        return 1;
    }
    db->name = dm_path_basename(input);

    double minsup_arg = atof(minsup_s);
    int minsup_count = minsup_arg < 1.0 ? (int)ceil(minsup_arg * (double)db->transaction_count) : (int)ceil(minsup_arg);
    if (minsup_count < 1) minsup_count = 1;

    MFHOILimits limits = {
        .max_patterns = max_patterns_s ? atoi(max_patterns_s) : 2000000,
        .max_seconds = max_seconds_s ? atof(max_seconds_s) : 0.0
    };

    MFHOIResult result;
    double start = dm_timer_now_seconds();
    int status = mfhoi_mine_all(db, minsup_count, atof(minocc_s), &limits, &result);
    double runtime = dm_timer_now_seconds() - start;
    const PatternList *patterns = mfhoi_select_patterns(&result, algo);
    mfhoi_write_patterns(output, patterns, db->transaction_count);

    printf("MFHOI Miner\n");
    printf("dataset=%s\n", input);
    printf("algorithm=%s\n", mfhoi_algorithm_name(algo));
    printf("minsup_count=%d\n", minsup_count);
    printf("minocc=%.8f\n", atof(minocc_s));
    printf("status=%s\n", status ? "LIMITED" : "OK");
    printf("runtime_seconds=%.8f\n", runtime);
    printf("peak_ram_mb=%.4f\n", get_peak_ram_mb());
    printf("output_disk_mb=%.6f\n", dm_file_size_mb(output));
    printf("num_generated_candidates=%d\n", result.stats.num_generated_candidates);
    printf("num_frequent_itemsets=%d\n", result.stats.num_frequent_itemsets);
    printf("num_fhoi=%d\n", result.stats.num_fhoi);
    printf("num_weak_mfhoi=%d\n", result.stats.num_weak_mfhoi);
    printf("num_strong_mfhoi=%d\n", result.stats.num_strong_mfhoi);
    printf("num_output_itemsets=%d\n", patterns->count);
    printf("pruned_ub1=%d\n", result.stats.pruned_ub1_count);
    printf("pruned_ub2=%d\n", result.stats.pruned_ub2_count);
    printf("dominance_removed_count=%d\n", result.stats.dominance_removed_count);

    mfhoi_result_free(&result);
    mfhoi_free_transaction_db(db);
    return status ? 2 : 0;
}
