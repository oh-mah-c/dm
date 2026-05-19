#include "algorithms/tipn_houi.h"
#include "core/dm_benchmark.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *arg_value(int argc, char **argv, const char *key, const char *fallback) {
    for (int i = 1; i + 1 < argc; i++) if (strcmp(argv[i], key) == 0) return argv[i + 1];
    return fallback;
}

static int has_flag(int argc, char **argv, const char *key) {
    for (int i = 1; i < argc; i++) if (strcmp(argv[i], key) == 0) return 1;
    return 0;
}

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s --input <negative_utility_dataset> --k <N> [--intervals N]\n", prog);
    printf("     [--max-depth N] [--max-transactions N] [--max-candidates N] [--max-seconds S]\n");
    printf("     [--max-items N]\n");
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    if (!input || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return input ? 0 : 1;
    }
    TIPNHouiParams params = tipn_houi_default_params();
    params.k = (size_t)strtoull(arg_value(argc, argv, "--k", "50"), NULL, 10);
    params.intervals = (size_t)strtoull(arg_value(argc, argv, "--intervals", "5"), NULL, 10);
    params.max_depth = (size_t)strtoull(arg_value(argc, argv, "--max-depth", "4"), NULL, 10);
    params.max_transactions = (size_t)strtoull(arg_value(argc, argv, "--max-transactions", "0"), NULL, 10);
    params.max_items = (size_t)strtoull(arg_value(argc, argv, "--max-items", "0"), NULL, 10);
    params.max_candidates = (size_t)strtoull(arg_value(argc, argv, "--max-candidates", "500000"), NULL, 10);
    params.max_seconds = atof(arg_value(argc, argv, "--max-seconds", "60"));

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    dm_bench_start(DM_PHASE_ALGO);
    TIPNHouiStats stats;
    int rc = tipn_houi_mine_file(input, &params, &stats);
    dm_bench_stop(DM_PHASE_ALGO);
    dm_bench_stop(DM_PHASE_TOTAL);
    DM_BenchmarkReport report = dm_bench_get_report();
    if (rc != 0) {
        fprintf(stderr, "TIPN-HOUI mining failed for %s\n", input);
        return 1;
    }
    printf("TIPN-HOUI Miner\n");
    printf("input=%s\n", input);
    printf("transactions=%zu\n", stats.transactions);
    printf("intervals=%zu\n", stats.intervals);
    printf("distinct_items=%zu\n", stats.distinct_items);
    printf("max_items=%zu\n", params.max_items);
    printf("positive_items=%zu\n", stats.positive_items);
    printf("negative_items=%zu\n", stats.negative_items);
    printf("k=%zu\n", stats.k);
    printf("status=%s\n", stats.limited ? "LIMITED" : "OK");
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_ALGO] / 1000.0);
    printf("total_sec=%.6f\n", report.phase_times_ms[DM_PHASE_TOTAL] / 1000.0);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    printf("rpru_size1_threshold=%.10g\n", stats.rpru_size1_threshold);
    printf("rru_size2_threshold=%.10g\n", stats.rru_size2_threshold);
    printf("final_threshold=%.10g\n", stats.threshold);
    printf("threshold_raises=%zu\n", stats.threshold_raises);
    printf("output_count=%zu\n", stats.output_count);
    printf("best_relative_utility=%.10g\n", stats.best_relative_utility);
    printf("avg_relative_utility=%.10g\n", stats.avg_relative_utility);
    printf("avg_itemset_length=%.6f\n", stats.avg_itemset_length);
    printf("candidates=%zu\n", stats.candidates);
    printf("joins=%zu\n", stats.joins);
    printf("pruned_twugc=%zu\n", stats.pruned_twugc);
    printf("pruned_rlc=%zu\n", stats.pruned_rlc);
    printf("pruned_tio=%zu\n", stats.pruned_tio);
    printf("result_ram_bytes=%zu\n", stats.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", stats.result_disk_est_bytes);
    return stats.limited ? 2 : 0;
}
