#include "algorithms/vifp.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s --input <transaction_dataset> --minsup <ratio|count> --mode <plaintext|smpc|fhe> [--max-itemsets N] [--max-seconds S]\n", prog);
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

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    const char *minsup_s = arg_value(argc, argv, "--minsup", "0.05");
    const char *mode_s = arg_value(argc, argv, "--mode", "plaintext");
    const char *max_itemsets_s = arg_value(argc, argv, "--max-itemsets", "0");
    const char *max_seconds_s = arg_value(argc, argv, "--max-seconds", "0");

    if (!input || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return input ? 0 : 1;
    }

    VIFPMode mode;
    if (vifp_parse_mode(mode_s, &mode) != 0) {
        fprintf(stderr, "Unknown VIFP mode: %s\n", mode_s);
        return 1;
    }

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    dm_bench_start(DM_PHASE_LOAD);
    DM_Dataset *ds = dm_dataset_load(input, DM_TYPE_TRANSACTIONAL);
    dm_bench_stop(DM_PHASE_LOAD);
    if (!ds) {
        fprintf(stderr, "Could not load transactional dataset: %s\n", input);
        return 1;
    }

    double minsup_value = atof(minsup_s);
    uint32_t minsup_count = minsup_value < 1.0 ? (uint32_t)ceil(minsup_value * (double)ds->count) : (uint32_t)minsup_value;
    if (minsup_count == 0) minsup_count = 1;

    VIFPStats stats;
    dm_bench_start(DM_PHASE_ALGO);
    int status = vifp_mine_dataset(ds, minsup_count, mode, (size_t)strtoull(max_itemsets_s, NULL, 10), atof(max_seconds_s), &stats);
    dm_bench_stop(DM_PHASE_ALGO);
    dm_bench_stop(DM_PHASE_TOTAL);
    DM_BenchmarkReport report = dm_bench_get_report();

    printf("VIFP Miner\n");
    printf("input=%s\n", input);
    printf("mode=%s\n", vifp_mode_name(mode));
    printf("transactions=%zu\n", ds->count);
    printf("max_item_id=%u\n", ds->max_id);
    printf("minsup_ratio=%.10g\n", minsup_value < 1.0 ? minsup_value : ((double)minsup_count / (double)ds->count));
    printf("minsup_count=%u\n", minsup_count);
    printf("status=%s\n", status == 0 ? "OK" : (status == 2 ? "LIMITED" : "FAILED"));
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_ALGO] / 1000.0);
    printf("total_sec=%.6f\n", report.phase_times_ms[DM_PHASE_TOTAL] / 1000.0);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    printf("frequent_singletons=%zu\n", stats.frequent_singletons);
    printf("occurrence_records=%zu\n", stats.occurrence_records);
    printf("hpa_records=%zu\n", stats.hpa_records);
    printf("output_count=%zu\n", stats.emitted_itemsets);
    printf("total_output_items=%zu\n", stats.total_output_items);
    printf("projected_states=%zu\n", stats.projected_states);
    printf("pid_descriptors=%zu\n", stats.pid_descriptors);
    printf("cpb_records=%zu\n", stats.cpb_records);
    printf("max_cpb_records=%zu\n", stats.max_cpb_records);
    printf("predecessor_fetches=%zu\n", stats.predecessor_fetches);
    printf("histogram_updates=%zu\n", stats.histogram_updates);
    printf("secure_comparisons=%zu\n", stats.secure_comparisons);
    printf("stable_partitions=%zu\n", stats.stable_partitions);
    printf("oblivious_sorts=%zu\n", stats.oblivious_sorts);
    printf("public_capacity_slots=%zu\n", stats.public_capacity_slots);
    printf("active_capacity_slots=%zu\n", stats.active_capacity_slots);
    printf("max_depth=%zu\n", stats.max_depth);
    printf("estimated_comm_bytes=%zu\n", stats.estimated_comm_bytes);
    printf("estimated_ciphertext_bytes=%zu\n", stats.estimated_ciphertext_bytes);
    printf("estimated_bootstraps=%zu\n", stats.estimated_bootstraps);
    printf("result_ram_bytes=%zu\n", report.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", report.result_disk_est_bytes);

    dm_dataset_free(ds);
    return status == 0 ? 0 : (status == 2 ? 2 : 1);
}
