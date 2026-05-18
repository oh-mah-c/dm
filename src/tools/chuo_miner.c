#include "algorithms/chuo_miner.h"
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
    printf("  %s --input <utility_dataset> --minutil <value> --minsup <ratio|count> --minocc <value>\n", prog);
    printf("     [--max-patterns N] [--max-depth N] [--max-seconds S]\n");
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    const char *minutil_s = arg_value(argc, argv, "--minutil", NULL);
    const char *minsup_s = arg_value(argc, argv, "--minsup", "0.01");
    const char *minocc_s = arg_value(argc, argv, "--minocc", "0.30");
    if (!input || !minutil_s || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return (input && minutil_s) ? 0 : 1;
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

    DM_CHUO_Params params;
    memset(&params, 0, sizeof(params));
    params.min_utility = atof(minutil_s);
    params.min_occupancy = atof(minocc_s);
    double minsup_v = atof(minsup_s);
    params.min_support = minsup_v < 1.0 ? (uint32_t)ceil(minsup_v * (double)ds->count) : (uint32_t)minsup_v;
    if (params.min_support == 0) params.min_support = 1;
    params.max_patterns = (size_t)strtoull(arg_value(argc, argv, "--max-patterns", "0"), NULL, 10);
    params.max_depth = (size_t)strtoull(arg_value(argc, argv, "--max-depth", "0"), NULL, 10);
    params.max_seconds = atof(arg_value(argc, argv, "--max-seconds", "0"));
    params.verify_reconstruction = 1;

    DM_CHUO_Stats stats;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = chuo_mine_dataset(ds, &params, &stats);
    dm_bench_stop(DM_PHASE_ALGO);
    dm_bench_stop(DM_PHASE_TOTAL);
    DM_BenchmarkReport report = dm_bench_get_report();

    if (rc != 0) {
        fprintf(stderr, "CHUO-Miner failed for %s\n", input);
        dm_dataset_free(ds);
        return 1;
    }

    printf("CHUO-Miner\n");
    printf("input=%s\n", input);
    printf("transactions=%zu\n", stats.transactions);
    printf("max_item_id=%u\n", stats.max_item_id);
    printf("minsup_ratio=%.10g\n", minsup_v < 1.0 ? minsup_v : (stats.transactions ? (double)stats.min_support / (double)stats.transactions : 0.0));
    printf("minsup_count=%u\n", stats.min_support);
    printf("min_utility=%.10g\n", stats.min_utility);
    printf("min_occupancy=%.10g\n", stats.min_occupancy);
    printf("status=%s\n", stats.limited ? "LIMITED" : "OK");
    printf("load_sec=%.6f\n", report.phase_times_ms[DM_PHASE_LOAD] / 1000.0);
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_ALGO] / 1000.0);
    printf("total_sec=%.6f\n", report.phase_times_ms[DM_PHASE_TOTAL] / 1000.0);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    printf("surviving_items=%zu\n", stats.surviving_items);
    printf("visited_nodes=%zu\n", stats.visited_nodes);
    printf("generated_children=%zu\n", stats.generated_children);
    printf("joined_entries=%zu\n", stats.joined_entries);
    printf("pruned_twu=%zu\n", stats.pruned_twu);
    printf("pruned_support=%zu\n", stats.pruned_support);
    printf("pruned_ruu=%zu\n", stats.pruned_ruu);
    printf("pruned_oub=%zu\n", stats.pruned_oub);
    printf("pruned_backward=%zu\n", stats.pruned_backward);
    printf("closure_jumps=%zu\n", stats.closure_jumps);
    printf("closure_hash_hits=%zu\n", stats.closure_hash_hits);
    printf("output_count=%zu\n", stats.emitted_chuois);
    printf("total_output_items=%zu\n", stats.total_output_items);
    printf("avg_output_length=%.6f\n", stats.emitted_chuois ? (double)stats.total_output_items / (double)stats.emitted_chuois : 0.0);
    printf("avg_support=%.6f\n", stats.avg_support);
    printf("avg_utility=%.6f\n", stats.avg_utility);
    printf("avg_occupancy=%.6f\n", stats.avg_occupancy);
    printf("best_utility=%.6f\n", stats.best_utility);
    printf("best_occupancy=%.6f\n", stats.best_occupancy);
    printf("max_depth=%zu\n", stats.max_depth_seen);
    printf("signatures_built=%zu\n", stats.signatures_built);
    printf("signature_value_records=%zu\n", stats.signature_value_records);
    printf("reconstruction_checks=%zu\n", stats.reconstruction_checks);
    printf("reconstruction_failures=%zu\n", stats.reconstruction_failures);
    printf("result_ram_bytes=%zu\n", stats.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", stats.result_disk_est_bytes);

    dm_dataset_free(ds);
    return stats.limited ? 2 : 0;
}
