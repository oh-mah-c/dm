#include "algorithms/kclotree_miner.h"
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
    printf("  %s --input <spmf_sequence_file_or_folder> [--k N] [--type generic|group|redundancy_aware]\n", prog);
    printf("     [--max-depth N] [--max-candidates N] [--max-seconds S]\n");
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    if (!input || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return input ? 0 : 1;
    }
    KCloParams params = kclotree_default_params();
    params.k = (size_t)strtoull(arg_value(argc, argv, "--k", "10"), NULL, 10);
    params.max_depth = (size_t)strtoull(arg_value(argc, argv, "--max-depth", "6"), NULL, 10);
    params.max_candidates = (size_t)strtoull(arg_value(argc, argv, "--max-candidates", "250000"), NULL, 10);
    params.max_seconds = atof(arg_value(argc, argv, "--max-seconds", "60"));
    if (kclotree_parse_type(arg_value(argc, argv, "--type", "generic"), &params.type) != 0) {
        fprintf(stderr, "Unknown mining type\n");
        return 1;
    }

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    dm_bench_start(DM_PHASE_ALGO);
    KCloStats stats;
    int rc = kclotree_mine_path(input, &params, &stats);
    dm_bench_stop(DM_PHASE_ALGO);
    dm_bench_stop(DM_PHASE_TOTAL);
    DM_BenchmarkReport report = dm_bench_get_report();
    if (rc != 0) {
        fprintf(stderr, "KCloTree mining failed for %s\n", input);
        return 1;
    }

    printf("KCloTreeMiner\n");
    printf("input=%s\n", input);
    printf("type=%s\n", kclotree_type_name(params.type));
    printf("sequences=%zu\n", stats.sequences);
    printf("distinct_items=%zu\n", stats.distinct_items);
    printf("max_sequence_length=%zu\n", stats.max_sequence_length);
    printf("k=%zu\n", stats.k);
    printf("max_depth=%zu\n", params.max_depth);
    printf("status=%s\n", stats.limited ? "LIMITED" : "OK");
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_ALGO] / 1000.0);
    printf("total_sec=%.6f\n", report.phase_times_ms[DM_PHASE_TOTAL] / 1000.0);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    printf("output_count=%zu\n", stats.output_count);
    printf("unique_supports_reported=%zu\n", stats.unique_supports_reported);
    printf("min_reported_support=%zu\n", stats.min_reported_support);
    printf("max_reported_support=%zu\n", stats.max_reported_support);
    printf("avg_reported_support=%.6f\n", stats.avg_reported_support);
    printf("avg_pattern_length=%.6f\n", stats.avg_pattern_length);
    printf("candidates_created=%zu\n", stats.candidates_created);
    printf("candidates_processed=%zu\n", stats.candidates_processed);
    printf("projected_extensions=%zu\n", stats.projected_extensions);
    printf("same_support_extensions=%zu\n", stats.same_support_extensions);
    printf("closed_candidates=%zu\n", stats.closed_candidates);
    printf("absorbed_patterns=%zu\n", stats.absorbed_patterns);
    printf("pruned_by_bound=%zu\n", stats.pruned_by_bound);
    printf("max_heap_size=%zu\n", stats.max_heap_size);
    printf("result_ram_bytes=%zu\n", stats.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", stats.result_disk_est_bytes);
    return stats.limited ? 2 : 0;
}
