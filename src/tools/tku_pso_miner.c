#include "algorithms/tku_pso.h"
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
    printf("  %s --input <utility_dataset> --k <N> [--population N] [--iterations N] [--seed N] [--max-seconds S]\n", prog);
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    if (!input || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return input ? 0 : 1;
    }

    DM_TKU_PSO_Params params;
    memset(&params, 0, sizeof(params));
    params.k = (size_t)strtoull(arg_value(argc, argv, "--k", "100"), NULL, 10);
    params.population_size = (size_t)strtoull(arg_value(argc, argv, "--population", "20"), NULL, 10);
    params.iterations = (size_t)strtoull(arg_value(argc, argv, "--iterations", "10000"), NULL, 10);
    params.seed = (unsigned int)strtoul(arg_value(argc, argv, "--seed", "42"), NULL, 10);
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

    DM_TKU_PSO_Stats stats;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = tku_pso_mine_dataset(ds, &params, &stats);
    dm_bench_stop(DM_PHASE_ALGO);
    dm_bench_stop(DM_PHASE_TOTAL);
    DM_BenchmarkReport report = dm_bench_get_report();
    dm_dataset_free(ds);
    if (rc != 0) {
        fprintf(stderr, "TKU-PSO mining failed for %s\n", input);
        return 1;
    }
    dm_bench_record_results(stats.output_count, stats.total_output_items);

    printf("TKU-PSO Miner\n");
    printf("input=%s\n", input);
    printf("transactions=%zu\n", stats.transactions);
    printf("distinct_items=%zu\n", stats.distinct_items);
    printf("kept_items=%zu\n", stats.kept_items);
    printf("k=%zu\n", stats.k);
    printf("population_size=%zu\n", stats.population_size);
    printf("iterations=%zu\n", stats.iterations);
    printf("status=%s\n", stats.limited ? "LIMITED" : "OK");
    printf("load_sec=%.6f\n", report.phase_times_ms[DM_PHASE_LOAD] / 1000.0);
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_ALGO] / 1000.0);
    printf("total_sec=%.6f\n", report.phase_times_ms[DM_PHASE_TOTAL] / 1000.0);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    printf("cuv_threshold=%.6f\n", stats.cuv_threshold);
    printf("final_threshold=%.6f\n", stats.final_threshold);
    printf("threshold_raises=%zu\n", stats.threshold_raises);
    printf("output_count=%zu\n", stats.output_count);
    printf("total_output_items=%zu\n", stats.total_output_items);
    printf("avg_output_length=%.6f\n", stats.avg_length);
    printf("best_utility=%.6f\n", stats.best_utility);
    printf("avg_utility=%.6f\n", stats.avg_utility);
    printf("deviation=%.6f\n", stats.deviation);
    printf("initialized_singletons=%zu\n", stats.initialized_singletons);
    printf("roulette_initialized=%zu\n", stats.roulette_initialized);
    printf("pev_repairs=%zu\n", stats.pev_repairs);
    printf("explored_particles=%zu\n", stats.explored_particles);
    printf("redundant_particles=%zu\n", stats.redundant_particles);
    printf("evaluated_particles=%zu\n", stats.evaluated_particles);
    printf("skipped_estimation=%zu\n", stats.skipped_estimation);
    printf("overestimates=%zu\n", stats.overestimates);
    printf("underestimates=%zu\n", stats.underestimates);
    printf("pruned_twu_items=%zu\n", stats.pruned_twu_items);
    printf("result_ram_bytes=%zu\n", stats.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", stats.result_disk_est_bytes);
    return stats.limited ? 2 : 0;
}
