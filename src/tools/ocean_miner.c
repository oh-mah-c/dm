/**
 * @file ocean_miner.c
 * @brief Standalone CLI runner for OCEAN algorithm (Shen et al. DASFAA 2016)
 */

#include "algorithms/ocean.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset.h"

#include <math.h>
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
    printf("  %s --input <utility_dataset> --minocc <value> [--minsup <ratio|count>]\n", prog);
    printf("Example:\n");
    printf("  %s --input datasets/utilities/foodmart.txt --minocc 0.50 --minsup 0.01\n", prog);
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    const char *minocc_s = arg_value(argc, argv, "--minocc", NULL);
    const char *minsup_s = arg_value(argc, argv, "--minsup", "0.01");

    if (!input || !minocc_s || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return (input && minocc_s) ? 0 : 1;
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

    DM_OCEAN_Params params;
    params.min_occupancy = atof(minocc_s);
    params.min_support = atof(minsup_s);

    extern DM_Algorithm ocean_algo;
    dm_bench_start(DM_PHASE_ALGO);
    DM_Status status = ocean_algo.run(ds, &params);
    dm_bench_stop(DM_PHASE_ALGO);
    dm_bench_stop(DM_PHASE_TOTAL);

    DM_BenchmarkReport report = dm_bench_get_report();

    if (status != DM_SUCCESS) {
        fprintf(stderr, "OCEAN execution failed with code %d\n", status);
        dm_dataset_free(ds);
        return 1;
    }

    printf("OCEAN Miner\n");
    printf("input=%s\n", input);
    printf("transactions=%zu\n", ds->count);
    printf("min_occupancy=%.6f\n", params.min_occupancy);
    printf("min_support=%.6f\n", params.min_support);
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_ALGO] / 1000.0);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    printf("output_count=%zu\n", report.num_itemsets);

    dm_dataset_free(ds);
    return 0;
}
