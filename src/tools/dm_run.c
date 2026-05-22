#include "core/dm_benchmark.h"
#include "core/dm_algorithm.h"
#include "core/dm_dataset.h"
#include "core/dm_flat.h"
#include "core/dm_flat_adapter.h"
#include "core/dm_plugin.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int output_json = 0;

extern DM_Algorithm bio_huif_ga_algo;
extern DM_Algorithm bio_huif_pso_algo;
extern DM_Algorithm bio_huif_ba_algo;
extern DM_Algorithm huim_afsa_algo;
extern DM_Algorithm huim_abc_algo;
extern DM_Algorithm skyline_miner_algo;
extern DM_Algorithm thui_algo;
extern DM_Algorithm tku_ce_algo;
extern DM_Algorithm tku_ce_plus_algo;
extern DM_Algorithm fhmds_algo;
extern DM_Algorithm haui_miner_algo;
extern DM_Algorithm ehaupm_algo;
extern DM_Algorithm hauim_gmu_algo;
extern DM_Algorithm nam_hep_algo;
extern DM_Algorithm memu_algo;
extern DM_Algorithm mheinu_algo;
extern DM_Algorithm dphim_algo;
extern DM_Algorithm closed_fhuim_kinana_algo;
extern DM_Algorithm hup_miner_algo;
extern DM_Algorithm regular_mine_algo;
extern DM_Algorithm mhoui_algo;
extern DM_Algorithm vifp_algo;
extern DM_Algorithm tmku_algo;
extern DM_Algorithm hiep_algo;

static void register_exported_legacy_algorithms(void) {
    dm_register_algorithm(&bio_huif_ga_algo);
    dm_register_algorithm(&bio_huif_pso_algo);
    dm_register_algorithm(&bio_huif_ba_algo);
    dm_register_algorithm(&huim_afsa_algo);
    dm_register_algorithm(&huim_abc_algo);
    dm_register_algorithm(&skyline_miner_algo);
    dm_register_algorithm(&thui_algo);
    dm_register_algorithm(&tku_ce_algo);
    dm_register_algorithm(&tku_ce_plus_algo);
    dm_register_algorithm(&fhmds_algo);
    dm_register_algorithm(&haui_miner_algo);
    dm_register_algorithm(&ehaupm_algo);
    dm_register_algorithm(&hauim_gmu_algo);
    dm_register_algorithm(&nam_hep_algo);
    dm_register_algorithm(&memu_algo);
    dm_register_algorithm(&mheinu_algo);
    dm_register_algorithm(&dphim_algo);
    dm_register_algorithm(&closed_fhuim_kinana_algo);
    dm_register_algorithm(&hup_miner_algo);
    dm_register_algorithm(&regular_mine_algo);
    dm_register_algorithm(&mhoui_algo);
    dm_register_algorithm(&vifp_algo);
    dm_register_algorithm(&tmku_algo);
    dm_register_algorithm(&hiep_algo);
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

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s --algorithm <id> --input <path> --connector spmf|text|graph [--k N] [--legacy-type 0..4]\n", prog);
    printf("  %s --list\n", prog);
}

static DM_PluginArg collect_arg(const char *key, const char *value) {
    DM_PluginArg arg;
    arg.key = key;
    arg.value = value;
    return arg;
}

int main(int argc, char **argv) {
    register_exported_legacy_algorithms();

    if (has_flag(argc, argv, "--list")) {
        printf("Plugins/adapters:\n");
        dm_plugin_print_all();
        printf("\nLegacy algorithms reachable through DM_FlatDataset adapters:\n");
        dm_list_algorithms();
        return 0;
    }
    const char *algorithm = arg_value(argc, argv, "--algorithm", NULL);
    const char *input = arg_value(argc, argv, "--input", NULL);
    const char *connector = arg_value(argc, argv, "--connector", "spmf");
    if (!algorithm || !input || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return (algorithm && input) ? 0 : 1;
    }

    DM_Plugin *plugin = dm_plugin_get(algorithm);
    DM_Algorithm *legacy = plugin ? NULL : dm_get_algorithm(algorithm);
    if (!plugin && !legacy) {
        fprintf(stderr, "Unknown algorithm: %s\n", algorithm);
        printf("Plugins/adapters:\n");
        dm_plugin_print_all();
        printf("\nLegacy algorithms reachable through DM_FlatDataset adapters:\n");
        dm_list_algorithms();
        return 1;
    }

    int raw_only_plugin = plugin && plugin->accepts_connectors == 0;
    DM_ConnectorKind kind = DM_CONNECTOR_SPMF;
    if (!raw_only_plugin && dm_connector_parse_kind(connector, &kind) != 0) {
        fprintf(stderr, "Unknown connector: %s\n", connector);
        return 1;
    }
    if (plugin && !raw_only_plugin && !(plugin->accepts_connectors & (1u << kind))) {
        fprintf(stderr, "Plugin %s does not accept connector %s\n", algorithm, connector);
        return 1;
    }

    DM_ConnectorOptions opt = dm_connector_default_options(kind);
    opt.text_window = (size_t)strtoull(arg_value(argc, argv, "--window", "64"), NULL, 10);
    opt.text_stride = (size_t)strtoull(arg_value(argc, argv, "--stride", "32"), NULL, 10);
    opt.text_sequence = !has_flag(argc, argv, "--itemset");
    opt.graph_undirected = has_flag(argc, argv, "--undirected");
    size_t arena_mb = (size_t)strtoull(arg_value(argc, argv, "--arena-mb", "256"), NULL, 10);

    DM_Arena arena;
    if (dm_arena_init(&arena, arena_mb * 1024 * 1024) != 0) {
        fprintf(stderr, "Could not allocate arena\n");
        return 1;
    }
    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    DM_FlatDataset flat;
    memset(&flat, 0, sizeof(flat));
    DM_ConnectorStats cstats;
    if (!raw_only_plugin && dm_flat_load_mmap(input, &opt, &arena, &flat, &cstats) != 0) {
        dm_arena_free(&arena);
        fprintf(stderr, "Could not load flat dataset\n");
        return 1;
    }

    int rc = 0;
    double algorithm_runtime_sec = 0.0;
    if (plugin) {
        DM_PluginArg args[8];
        size_t arg_count = 0;
        args[arg_count++] = collect_arg("k", arg_value(argc, argv, "--k", "20"));
        args[arg_count++] = collect_arg("minsup", arg_value(argc, argv, "--minsup", "1"));
        args[arg_count++] = collect_arg("theta", arg_value(argc, argv, "--theta", "0"));
        args[arg_count++] = collect_arg("minutil", arg_value(argc, argv, "--minutil", arg_value(argc, argv, "--theta", "0")));
        args[arg_count++] = collect_arg("minconf", arg_value(argc, argv, "--minconf", "0.8"));
        args[arg_count++] = collect_arg("minocc", arg_value(argc, argv, "--minocc", "0.1"));
        DM_PluginInput pin;
        memset(&pin, 0, sizeof(pin));
        pin.flat = &flat;
        pin.arena = &arena;
        pin.source_path = input;
        pin.connector = kind;
        pin.args = args;
        pin.arg_count = arg_count;
        DM_PluginResult result;
        memset(&result, 0, sizeof(result));
        rc = plugin->run(&pin, &result);
        algorithm_runtime_sec = result.runtime_sec;
        printf("algorithm_interface=%s\n", raw_only_plugin ? "raw_plugin" : "flat_plugin");
    } else {
        int requested_type = (int)strtol(arg_value(argc, argv, "--legacy-type", "-1"), NULL, 10);
        DM_DatasetType legacy_type = dm_flat_choose_legacy_type(legacy->supported_types, kind, requested_type);
        DM_Dataset *ds = dm_flat_to_dataset(&flat, legacy_type);
        if (!ds) {
            dm_arena_free(&arena);
            fprintf(stderr, "Could not adapt flat dataset for legacy algorithm %s\n", algorithm);
            return 1;
        }
        dm_bench_start(DM_PHASE_ALGO);
        DM_Status status = legacy->run(ds, NULL);
        dm_bench_stop(DM_PHASE_ALGO);
        algorithm_runtime_sec = dm_bench_get_report().phase_times_ms[DM_PHASE_ALGO] / 1000.0;
        rc = status == DM_SUCCESS ? 0 : 1;
        printf("algorithm_interface=legacy_adapter\n");
        printf("legacy_dataset_type=%d\n", (int)legacy_type);
        dm_dataset_free(ds);
    }
    dm_bench_stop(DM_PHASE_TOTAL);
    DM_BenchmarkReport report = dm_bench_get_report();
    printf("algorithm_runtime_sec=%.6f\n", algorithm_runtime_sec);
    printf("total_runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_TOTAL] / 1000.0);
    if (!raw_only_plugin) {
        printf("flat_rows=%zu\n", flat.row_count);
        printf("flat_items=%zu\n", flat.item_count);
        printf("flat_max_item=%u\n", flat.max_item);
    }
    printf("arena_used_bytes=%zu\n", arena.offset);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    dm_arena_free(&arena);
    return rc == 0 ? 0 : 1;
}
