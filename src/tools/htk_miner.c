#include "algorithms/htk_miner.h"
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
    printf("  %s --input <transaction_dataset> --k <N> [--mode bsn] [--max-depth N] [--max-candidates N] [--max-seconds S]\n", prog);
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    if (!input || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return input ? 0 : 1;
    }

    DM_HTK_Params params;
    memset(&params, 0, sizeof(params));
    params.k = (size_t)strtoull(arg_value(argc, argv, "--k", "100"), NULL, 10);
    params.max_depth = (size_t)strtoull(arg_value(argc, argv, "--max-depth", "0"), NULL, 10);
    params.max_candidates = (size_t)strtoull(arg_value(argc, argv, "--max-candidates", "0"), NULL, 10);
    params.max_seconds = atof(arg_value(argc, argv, "--max-seconds", "0"));
    if (htk_parse_mode(arg_value(argc, argv, "--mode", "bsn"), &params.mode) != 0) {
        fprintf(stderr, "Unknown HTK mode. This implementation supports: bsn\n");
        return 1;
    }

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    DM_HTK_Stats stats;
    int rc = htk_mine_file(input, &params, &stats);
    dm_bench_stop(DM_PHASE_TOTAL);
    DM_BenchmarkReport report = dm_bench_get_report();
    if (rc != 0) {
        fprintf(stderr, "HTK mining failed for %s\n", input);
        return 1;
    }
    dm_bench_record_results(stats.output_count, stats.total_output_items);

    printf("HTK-Miner\n");
    printf("input=%s\n", input);
    printf("mode=%s\n", htk_mode_name(params.mode));
    printf("transactions=%zu\n", stats.transactions);
    printf("nonempty_transactions=%zu\n", stats.nonempty_transactions);
    printf("distinct_items=%zu\n", stats.distinct_items);
    printf("k=%zu\n", stats.k);
    printf("status=%s\n", stats.limited ? "LIMITED" : "OK");
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_TOTAL] / 1000.0);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    printf("levels=%zu\n", stats.levels);
    printf("bitset_words=%zu\n", stats.bitset_words);
    printf("singleton_kept=%zu\n", stats.singleton_kept);
    printf("pruned_singletons=%zu\n", stats.pruned_singletons);
    printf("final_threshold=%zu\n", stats.final_threshold);
    printf("threshold_raises=%zu\n", stats.threshold_raises);
    printf("output_count=%zu\n", stats.output_count);
    printf("total_output_items=%zu\n", stats.total_output_items);
    printf("avg_output_length=%.6f\n", stats.avg_output_length);
    printf("best_support=%.6f\n", stats.best_support);
    printf("avg_support=%.6f\n", stats.avg_support);
    printf("candidates=%zu\n", stats.candidates);
    printf("joins=%zu\n", stats.joins);
    printf("intersections=%zu\n", stats.intersections);
    printf("pruned_support=%zu\n", stats.pruned_support);
    printf("result_ram_bytes=%zu\n", stats.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", stats.result_disk_est_bytes);
    return stats.limited ? 2 : 0;
}
