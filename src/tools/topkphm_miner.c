#include "algorithms/topkphm.h"
#include "core/dm_benchmark.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s --input <utility_dataset> --k <N> --maxper <N> --maxavg <N> [--max-depth N] [--max-candidates N] [--max-seconds S]\n", prog);
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    if (!input || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return input ? 0 : 1;
    }

    DM_TOPKPHM_Params params;
    memset(&params, 0, sizeof(params));
    params.k = (size_t)strtoull(arg_value(argc, argv, "--k", "100"), NULL, 10);
    params.max_period = (size_t)strtoull(arg_value(argc, argv, "--maxper", "1000"), NULL, 10);
    params.max_avg_period = atof(arg_value(argc, argv, "--maxavg", "500"));
    params.max_depth = (size_t)strtoull(arg_value(argc, argv, "--max-depth", "0"), NULL, 10);
    params.max_candidates = (size_t)strtoull(arg_value(argc, argv, "--max-candidates", "0"), NULL, 10);
    params.max_seconds = atof(arg_value(argc, argv, "--max-seconds", "0"));

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    DM_TOPKPHM_Stats stats;
    int rc = topkphm_mine_file(input, &params, &stats);
    dm_bench_stop(DM_PHASE_TOTAL);
    DM_BenchmarkReport report = dm_bench_get_report();
    if (rc != 0) {
        fprintf(stderr, "TOPKPHM mining failed for %s\n", input);
        return 1;
    }
    dm_bench_record_results(stats.output_count, stats.total_output_items);

    printf("TOPKPHM Miner\n");
    printf("input=%s\n", input);
    printf("transactions=%zu\n", stats.transactions);
    printf("distinct_items=%zu\n", stats.distinct_items);
    printf("kept_items=%zu\n", stats.kept_items);
    printf("k=%zu\n", stats.k);
    printf("max_period=%zu\n", stats.max_period);
    printf("max_avg_period=%.6f\n", stats.max_avg_period);
    printf("support_threshold=%zu\n", stats.support_threshold);
    printf("status=%s\n", stats.limited ? "LIMITED" : "OK");
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_TOTAL] / 1000.0);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    printf("final_threshold=%.6f\n", stats.final_threshold);
    printf("threshold_raises=%zu\n", stats.threshold_raises);
    printf("output_count=%zu\n", stats.output_count);
    printf("total_output_items=%zu\n", stats.total_output_items);
    printf("avg_output_length=%.6f\n", stats.avg_output_length);
    printf("best_utility=%.6f\n", stats.best_utility);
    printf("avg_utility=%.6f\n", stats.avg_utility);
    printf("visited_nodes=%zu\n", stats.visited_nodes);
    printf("candidates=%zu\n", stats.candidates);
    printf("joins=%zu\n", stats.joins);
    printf("joined_entries=%zu\n", stats.joined_entries);
    printf("euscs_pairs=%zu\n", stats.euscs_pairs);
    printf("pruned_support=%zu\n", stats.pruned_support);
    printf("pruned_periodicity=%zu\n", stats.pruned_periodicity);
    printf("pruned_twu=%zu\n", stats.pruned_twu);
    printf("pruned_subtree_utility=%zu\n", stats.pruned_subtree_utility);
    printf("result_ram_bytes=%zu\n", stats.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", stats.result_disk_est_bytes);
    return stats.limited ? 2 : 0;
}
