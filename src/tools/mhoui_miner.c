#include "algorithms/mhoui.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset.h"
#include "core/dm_dataset_types.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s --input <utility_dataset> --algorithm <houi|weak_mhoui|strong_mhoui|direct_weak_mhoui|direct_strong_mhoui> --minsup <ratio|count> --minocc <value> --minutil <abs> --output <path> [--max-patterns N] [--max-seconds S]\n", prog);
}

static const char *arg_value(int argc, char **argv, const char *key, const char *fallback) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return fallback;
}

static int has_flag(int argc, char **argv, const char *key) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], key) == 0) return 1;
    }
    return 0;
}

static double total_utility(DM_Dataset *ds) {
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    double total = 0.0;
    for (size_t i = 0; i < ds->count; i++) total += data[i].total_utility;
    return total;
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    const char *algo_name = arg_value(argc, argv, "--algorithm", "direct_strong_mhoui");
    const char *minsup_s = arg_value(argc, argv, "--minsup", "0.05");
    const char *minocc_s = arg_value(argc, argv, "--minocc", "0.4");
    const char *minutil_s = arg_value(argc, argv, "--minutil", NULL);
    const char *minutil_ratio_s = arg_value(argc, argv, "--minutil-ratio", NULL);
    const char *output = arg_value(argc, argv, "--output", "results/patterns/mhoui_patterns.csv");
    const char *max_patterns_s = arg_value(argc, argv, "--max-patterns", "200000");
    const char *max_seconds_s = arg_value(argc, argv, "--max-seconds", "0");

    if (!input || (!minutil_s && !minutil_ratio_s) || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return input ? 0 : 1;
    }

    MHOUIAlgorithm algo;
    if (mhoui_parse_algorithm(algo_name, &algo) != 0) {
        fprintf(stderr, "Unknown MHOUI algorithm: %s\n", algo_name);
        return 1;
    }

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    dm_bench_start(DM_PHASE_LOAD);
    DM_Dataset *ds = dm_dataset_load(input, DM_TYPE_UTILITY);
    dm_bench_stop(DM_PHASE_LOAD);
    if (!ds) {
        fprintf(stderr, "Could not load utility dataset: %s\n", input);
        return 1;
    }

    double minsup_value = atof(minsup_s);
    int minsup_count = minsup_value < 1.0 ? (int)ceil(minsup_value * (double)ds->count) : (int)minsup_value;
    double dbu = total_utility(ds);
    double minutil = minutil_s ? atof(minutil_s) : ceil(atof(minutil_ratio_s) * dbu);
    double minutil_ratio = dbu > 0.0 ? minutil / dbu : 0.0;
    MHOUILimits limits = {
        .max_patterns = atoi(max_patterns_s),
        .max_seconds = atof(max_seconds_s),
        .direct = (algo == MHOUI_ALGO_DIRECT_WEAK || algo == MHOUI_ALGO_DIRECT_STRONG),
        .strong_direct = (algo == MHOUI_ALGO_DIRECT_STRONG)
    };

    MHOUIResult result;
    dm_bench_start(DM_PHASE_ALGO);
    int status = mhoui_mine_dataset(ds, minsup_count, atof(minocc_s), minutil, &limits, &result);
    dm_bench_stop(DM_PHASE_ALGO);

    const MHOUPatternList *patterns = mhoui_select_patterns(&result, algo);
    mkdir("results", 0775);
    mkdir("results/patterns", 0775);
    dm_bench_start(DM_PHASE_WRITE);
    mhoui_write_patterns(output, patterns, ds->count);
    dm_bench_stop(DM_PHASE_WRITE);
    dm_bench_stop(DM_PHASE_TOTAL);

    size_t total_items = 0;
    for (int i = 0; i < patterns->count; i++) total_items += (size_t)patterns->patterns[i].length;
    dm_bench_record_results((size_t)patterns->count, total_items);
    DM_BenchmarkReport report = dm_bench_get_report();

    double avg_len, avg_sup, avg_util, avg_occ;
    int max_len;
    mhoui_pattern_quality(patterns, &avg_len, &max_len, &avg_sup, &avg_util, &avg_occ);

    printf("MHOUI Miner\n");
    printf("input=%s\n", input);
    printf("algorithm=%s\n", mhoui_algorithm_name(algo));
    printf("transactions=%zu\n", ds->count);
    printf("minsup_count=%d\n", minsup_count);
    printf("minocc=%.10g\n", atof(minocc_s));
    printf("minutil=%.10g\n", minutil);
    printf("minutil_ratio=%.10g\n", minutil_ratio);
    printf("status=%s\n", status == 0 ? "OK" : "LIMITED");
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_ALGO] / 1000.0);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    printf("visited_nodes=%d\n", result.stats.visited_nodes);
    printf("candidates=%d\n", result.stats.generated_candidates);
    printf("pruned_support=%d\n", result.stats.pruned_support);
    printf("pruned_twu=%d\n", result.stats.pruned_twu);
    printf("pruned_uub=%d\n", result.stats.pruned_uub);
    printf("pruned_oub1=%d\n", result.stats.pruned_oub1);
    printf("pruned_oub2=%d\n", result.stats.pruned_oub2);
    printf("pruned_dom=%d\n", result.stats.pruned_dominance);
    printf("num_houi=%d\n", result.stats.houi_count);
    printf("num_weak_mhoui=%d\n", result.stats.weak_count);
    printf("num_strong_mhoui=%d\n", result.stats.strong_count);
    printf("output_count=%d\n", patterns->count);
    printf("avg_len=%.8f\n", avg_len);
    printf("max_len=%d\n", max_len);
    printf("avg_support=%.8f\n", avg_sup);
    printf("avg_utility=%.8f\n", avg_util);
    printf("avg_occupancy=%.8f\n", avg_occ);
    printf("output_file=%s\n", output);

    mhoui_result_free(&result);
    dm_dataset_free(ds);
    return status == 0 ? 0 : 2;
}
