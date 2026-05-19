#include "algorithms/tmku.h"
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
    printf("  %s --input <utility_dataset> --k <N> --target <items> --threshold <val> [--max-seconds S]\n", prog);
    printf("Example:\n");
    printf("  %s --input datasets/utilities/foodmart.txt --k 3 --target 1340 --threshold 20\n", prog);
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    if (!input || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return input ? 0 : 1;
    }

    DM_TMKU_Params params;
    memset(&params, 0, sizeof(params));
    params.k = (size_t)strtoull(arg_value(argc, argv, "--k", "10"), NULL, 10);
    params.min_utility = atof(arg_value(argc, argv, "--threshold", "0.0"));
    params.max_seconds = atof(arg_value(argc, argv, "--max-seconds", "0"));

    const char *target_str = arg_value(argc, argv, "--target", "");
    size_t target_len = 0;
    uint32_t *target_pattern = NULL;
    if (strlen(target_str) > 0) {
        size_t commas = 0;
        for (size_t i = 0; target_str[i] != '\0'; i++) {
            if (target_str[i] == ',') commas++;
        }
        target_pattern = malloc(sizeof(uint32_t) * (commas + 1));
        char *str_copy = strdup(target_str);
        char *tok = strtok(str_copy, ",");
        while (tok != NULL) {
            target_pattern[target_len++] = (uint32_t)strtoul(tok, NULL, 10);
            tok = strtok(NULL, ",");
        }
        free(str_copy);
    }
    params.target_pattern = target_pattern;
    params.target_len = target_len;

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    dm_bench_start(DM_PHASE_LOAD);
    DM_Dataset *ds = dm_dataset_load(input, DM_TYPE_UTILITY);
    dm_bench_stop(DM_PHASE_LOAD);
    if (!ds) {
        fprintf(stderr, "Could not load utility dataset: %s\n", input);
        free(target_pattern);
        return 1;
    }

    DM_TMKU_Stats stats;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = tmku_mine_dataset(ds, &params, &stats);
    dm_bench_stop(DM_PHASE_ALGO);
    dm_bench_stop(DM_PHASE_TOTAL);
    DM_BenchmarkReport report = dm_bench_get_report();
    dm_dataset_free(ds);
    free(target_pattern);

    if (rc != 0) {
        fprintf(stderr, "TMKU mining failed for %s\n", input);
        return 1;
    }

    printf("TMKU Miner\n");
    printf("input=%s\n", input);
    printf("transactions=%zu\n", stats.transactions);
    printf("distinct_items=%zu\n", stats.distinct_items);
    printf("k=%zu\n", stats.k);
    printf("status=%s\n", stats.limited ? "LIMITED" : "OK");
    printf("load_sec=%.6f\n", report.phase_times_ms[DM_PHASE_LOAD] / 1000.0);
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_ALGO] / 1000.0);
    printf("total_sec=%.6f\n", report.phase_times_ms[DM_PHASE_TOTAL] / 1000.0);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
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
    printf("result_ram_bytes=%zu\n", stats.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", stats.result_disk_est_bytes);
    return stats.limited ? 2 : 0;
}
