#include "algorithms/tku_miner.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset.h"

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
    printf("  %s --input <utility_dataset> --k <N> [--max-depth N] [--max-seconds S]\n", prog);
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    if (!input || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return input ? 0 : 1;
    }
    DM_TKU_Params params;
    params.k = (size_t)strtoull(arg_value(argc, argv, "--k", "10"), NULL, 10);
    params.max_depth = (size_t)strtoull(arg_value(argc, argv, "--max-depth", "0"), NULL, 10);
    params.max_seconds = atof(arg_value(argc, argv, "--max-seconds", "0"));

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    dm_bench_start(DM_PHASE_LOAD);
    DM_Dataset *ds = dm_dataset_load(input, DM_TYPE_UTILITY);
    dm_bench_stop(DM_PHASE_LOAD);
    if (!ds) {
        fprintf(stderr, "Could not load utility dataset: %s\n", input);
        return 1;
    }

    DM_TKU_Stats stats;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = tku_mine_dataset(ds, &params, &stats);
    dm_bench_stop(DM_PHASE_ALGO);
    dm_bench_stop(DM_PHASE_TOTAL);
    DM_BenchmarkReport report = dm_bench_get_report();
    dm_dataset_free(ds);
    if (rc != 0) {
        fprintf(stderr, "TKU mining failed for %s\n", input);
        return 1;
    }

    printf("TKU Miner\n");
    printf("input=%s\n", input);
    printf("transactions=%zu\n", stats.transactions);
    printf("distinct_items=%zu\n", stats.distinct_items);
    printf("k=%zu\n", stats.k);
    printf("status=%s\n", stats.limited ? "LIMITED" : "OK");
    printf("load_sec=%.6f\n", report.phase_times_ms[DM_PHASE_LOAD] / 1000.0);
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_ALGO] / 1000.0);
    printf("total_sec=%.6f\n", report.phase_times_ms[DM_PHASE_TOTAL] / 1000.0);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    printf("pe_threshold=%.6f\n", stats.pe_threshold);
    printf("singleton_threshold=%.6f\n", stats.singleton_threshold);
    printf("final_threshold=%.6f\n", stats.final_threshold);
    printf("threshold_raises=%zu\n", stats.threshold_raises);
    printf("output_count=%zu\n", stats.output_count);
    printf("total_output_items=%zu\n", stats.total_output_items);
    printf("avg_output_length=%.6f\n", stats.avg_length);
    printf("best_utility=%.6f\n", stats.best_utility);
    printf("avg_utility=%.6f\n", stats.avg_utility);
    printf("visited_nodes=%zu\n", stats.visited_nodes);
    printf("candidates=%zu\n", stats.candidates);
    printf("joins=%zu\n", stats.joins);
    printf("joined_entries=%zu\n", stats.joined_entries);
    printf("pruned_twu=%zu\n", stats.pruned_twu);
    printf("pruned_subtree_utility=%zu\n", stats.pruned_subtree_utility);
    printf("phase2_checked=%zu\n", stats.phase2_checked);
    printf("result_ram_bytes=%zu\n", stats.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", stats.result_disk_est_bytes);
    return stats.limited ? 2 : 0;
}
