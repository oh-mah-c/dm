#include "core/dm_flat_adapter.h"
#include "core/dm_benchmark.h"
#include "core/dm_plugin.h"

#include "algorithms/chuo_miner.h"
#include "algorithms/htk_miner.h"
#include "algorithms/hupp.h"
#include "algorithms/huci_miner.h"
#include "algorithms/kclotree_miner.h"
#include "algorithms/pso_classifier.h"
#include "algorithms/tipn_houi.h"
#include "algorithms/tku_miner.h"
#include "algorithms/tku_pso.h"
#include "algorithms/topkphm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t arg_size(const DM_PluginInput *in, const char *key, size_t fallback) {
    const char *v = dm_plugin_arg(in, key, NULL);
    return v ? (size_t)strtoull(v, NULL, 10) : fallback;
}

static double arg_double(const DM_PluginInput *in, const char *key, double fallback) {
    const char *v = dm_plugin_arg(in, key, NULL);
    return v ? strtod(v, NULL) : fallback;
}

static int run_chuo_adapter(const DM_PluginInput *in, DM_PluginResult *out) {
    DM_Dataset *ds = dm_flat_to_dataset(in->flat, DM_TYPE_UTILITY);
    if (!ds) return -1;
    DM_CHUO_Params p;
    memset(&p, 0, sizeof(p));
    p.min_support = (uint32_t)arg_size(in, "minsup", 1);
    p.min_utility = arg_double(in, "minutil", arg_double(in, "theta", 1.0));
    p.min_occupancy = arg_double(in, "minocc", 0.1);
    p.max_patterns = arg_size(in, "max_patterns", 100000);
    p.max_depth = arg_size(in, "max_depth", 0);
    p.max_seconds = arg_double(in, "max_seconds", 0.0);
    DM_CHUO_Stats s;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = chuo_mine_dataset(ds, &p, &s);
    dm_bench_stop(DM_PHASE_ALGO);
    DM_BenchmarkReport r = dm_bench_get_report();
    if (out) {
        out->runtime_sec = r.phase_times_ms[DM_PHASE_ALGO] / 1000.0;
        out->patterns = s.emitted_chuois;
    }
    printf("algorithm=chuo\n");
    printf("output_count=%zu\n", s.emitted_chuois);
    printf("visited_nodes=%zu\n", s.visited_nodes);
    printf("best_utility=%.6f\n", s.best_utility);
    printf("best_occupancy=%.6f\n", s.best_occupancy);
    printf("result_ram_bytes=%zu\n", s.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", s.result_disk_est_bytes);
    dm_dataset_free(ds);
    return rc;
}

static int run_huci_adapter(const DM_PluginInput *in, DM_PluginResult *out) {
    DM_Dataset *ds = dm_dataset_load(in->source_path, DM_TYPE_UTILITY);
    if (!ds) return -1;
    DM_HUCI_Miner_Params p;
    p.min_utility = arg_double(in, "minutil", arg_double(in, "theta", 1000.0));
    p.min_confidence = arg_double(in, "minconf", 0.8);
    DM_HUCI_Miner_Stats s;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = huci_mine_dataset(ds, &p, &s);
    dm_bench_stop(DM_PHASE_ALGO);
    DM_BenchmarkReport r = dm_bench_get_report();
    if (out) {
        out->runtime_sec = r.phase_times_ms[DM_PHASE_ALGO] / 1000.0;
        out->patterns = s.high_utility_closed_itemsets;
    }
    printf("algorithm=huciminer\n");
    printf("high_utility_itemsets=%zu\n", s.high_utility_itemsets);
    printf("high_utility_closed_itemsets=%zu\n", s.high_utility_closed_itemsets);
    printf("high_utility_generators=%zu\n", s.high_utility_generators);
    printf("hgb_rules=%zu\n", s.hgb_rules);
    printf("max_depth=%zu\n", s.max_depth);
    printf("utility_lists_constructed=%zu\n", s.utility_lists_constructed);
    printf("joins=%zu\n", s.joins);
    printf("pruned_eucs=%zu\n", s.pruned_eucs);
    printf("pruned_subtree_utility=%zu\n", s.pruned_subtree_utility);
    dm_dataset_free(ds);
    return rc;
}

static int run_tku_adapter(const DM_PluginInput *in, DM_PluginResult *out) {
    DM_Dataset *ds = dm_flat_to_dataset(in->flat, DM_TYPE_UTILITY);
    if (!ds) return -1;
    DM_TKU_Params p;
    memset(&p, 0, sizeof(p));
    p.k = arg_size(in, "k", 20);
    p.max_depth = arg_size(in, "max_depth", 0);
    p.max_seconds = arg_double(in, "max_seconds", 0.0);
    DM_TKU_Stats s;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = tku_mine_dataset(ds, &p, &s);
    dm_bench_stop(DM_PHASE_ALGO);
    DM_BenchmarkReport r = dm_bench_get_report();
    if (out) {
        out->runtime_sec = r.phase_times_ms[DM_PHASE_ALGO] / 1000.0;
        out->patterns = s.output_count;
    }
    printf("algorithm=tku_miner\n");
    printf("output_count=%zu\n", s.output_count);
    printf("final_threshold=%.6f\n", s.final_threshold);
    printf("best_utility=%.6f\n", s.best_utility);
    printf("avg_utility=%.6f\n", s.avg_utility);
    printf("result_ram_bytes=%zu\n", s.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", s.result_disk_est_bytes);
    dm_dataset_free(ds);
    return rc;
}

static int run_tku_pso_adapter(const DM_PluginInput *in, DM_PluginResult *out) {
    DM_Dataset *ds = dm_flat_to_dataset(in->flat, DM_TYPE_UTILITY);
    if (!ds) return -1;
    DM_TKU_PSO_Params p;
    memset(&p, 0, sizeof(p));
    p.k = arg_size(in, "k", 20);
    p.population_size = arg_size(in, "population", 50);
    p.iterations = arg_size(in, "iterations", 50);
    p.seed = (unsigned int)arg_size(in, "seed", 7);
    p.max_seconds = arg_double(in, "max_seconds", 0.0);
    DM_TKU_PSO_Stats s;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = tku_pso_mine_dataset(ds, &p, &s);
    dm_bench_stop(DM_PHASE_ALGO);
    DM_BenchmarkReport r = dm_bench_get_report();
    if (out) {
        out->runtime_sec = r.phase_times_ms[DM_PHASE_ALGO] / 1000.0;
        out->patterns = s.output_count;
    }
    printf("algorithm=tku_pso\n");
    printf("output_count=%zu\n", s.output_count);
    printf("final_threshold=%.6f\n", s.final_threshold);
    printf("best_utility=%.6f\n", s.best_utility);
    printf("evaluated_particles=%zu\n", s.evaluated_particles);
    printf("result_ram_bytes=%zu\n", s.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", s.result_disk_est_bytes);
    dm_dataset_free(ds);
    return rc;
}

static int run_htk_adapter(const DM_PluginInput *in, DM_PluginResult *out) {
    DM_HTK_Params p;
    memset(&p, 0, sizeof(p));
    p.k = arg_size(in, "k", 20);
    p.max_depth = arg_size(in, "max_depth", 0);
    p.max_candidates = arg_size(in, "max_candidates", 0);
    p.max_seconds = arg_double(in, "max_seconds", 0.0);
    p.mode = DM_HTK_MODE_BSN;
    DM_HTK_Stats s;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = htk_mine_file(in->source_path, &p, &s);
    dm_bench_stop(DM_PHASE_ALGO);
    DM_BenchmarkReport r = dm_bench_get_report();
    if (out) {
        out->runtime_sec = r.phase_times_ms[DM_PHASE_ALGO] / 1000.0;
        out->patterns = s.output_count;
    }
    printf("algorithm=htk_miner\n");
    printf("output_count=%zu\n", s.output_count);
    printf("final_threshold=%zu\n", s.final_threshold);
    printf("candidates=%zu\n", s.candidates);
    printf("result_ram_bytes=%zu\n", s.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", s.result_disk_est_bytes);
    return rc;
}

static int run_hupp_adapter(const DM_PluginInput *in, DM_PluginResult *out) {
    HUPPParams p = hupp_default_params();
    p.min_support = arg_double(in, "minsup", p.min_support);
    p.min_utility = arg_double(in, "theta", p.min_utility);
    p.max_patterns = arg_size(in, "max_patterns", p.max_patterns);
    p.max_depth = arg_size(in, "max_depth", p.max_depth);
    p.max_seconds = arg_double(in, "max_seconds", p.max_seconds);
    HUPPStats s;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = hupp_mine_file(in->source_path, &p, &s);
    dm_bench_stop(DM_PHASE_ALGO);
    DM_BenchmarkReport r = dm_bench_get_report();
    if (out) {
        out->runtime_sec = r.phase_times_ms[DM_PHASE_ALGO] / 1000.0;
        out->patterns = s.emitted_patterns;
    }
    printf("algorithm=hupp\n");
    printf("output_count=%zu\n", s.emitted_patterns);
    printf("semantic_concepts=%zu\n", s.semantic_concepts);
    printf("best_utility=%.6f\n", s.best_utility);
    printf("best_alignment=%.6f\n", s.best_alignment);
    printf("result_ram_bytes=%zu\n", s.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", s.result_disk_est_bytes);
    return rc;
}

static int run_kclotree_adapter(const DM_PluginInput *in, DM_PluginResult *out) {
    KCloParams p = kclotree_default_params();
    p.k = arg_size(in, "k", p.k);
    p.max_depth = arg_size(in, "max_depth", p.max_depth);
    p.max_candidates = arg_size(in, "max_candidates", p.max_candidates);
    p.max_seconds = arg_double(in, "max_seconds", p.max_seconds);
    KCloStats s;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = kclotree_mine_path(in->source_path, &p, &s);
    dm_bench_stop(DM_PHASE_ALGO);
    DM_BenchmarkReport r = dm_bench_get_report();
    if (out) {
        out->runtime_sec = r.phase_times_ms[DM_PHASE_ALGO] / 1000.0;
        out->patterns = s.output_count;
    }
    printf("algorithm=kclotree_miner\n");
    printf("output_count=%zu\n", s.output_count);
    printf("min_reported_support=%zu\n", s.min_reported_support);
    printf("max_reported_support=%zu\n", s.max_reported_support);
    printf("result_ram_bytes=%zu\n", s.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", s.result_disk_est_bytes);
    return rc;
}

static int run_tipn_adapter(const DM_PluginInput *in, DM_PluginResult *out) {
    TIPNHouiParams p = tipn_houi_default_params();
    p.k = arg_size(in, "k", p.k);
    p.intervals = arg_size(in, "intervals", p.intervals);
    p.max_depth = arg_size(in, "max_depth", p.max_depth);
    p.max_seconds = arg_double(in, "max_seconds", p.max_seconds);
    TIPNHouiStats s;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = tipn_houi_mine_file(in->source_path, &p, &s);
    dm_bench_stop(DM_PHASE_ALGO);
    DM_BenchmarkReport r = dm_bench_get_report();
    if (out) {
        out->runtime_sec = r.phase_times_ms[DM_PHASE_ALGO] / 1000.0;
        out->patterns = s.output_count;
    }
    printf("algorithm=tipn_houi\n");
    printf("output_count=%zu\n", s.output_count);
    printf("threshold=%.6f\n", s.threshold);
    printf("best_relative_utility=%.6f\n", s.best_relative_utility);
    printf("result_ram_bytes=%zu\n", s.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", s.result_disk_est_bytes);
    return rc;
}

static int run_topkphm_adapter(const DM_PluginInput *in, DM_PluginResult *out) {
    DM_TOPKPHM_Params p;
    memset(&p, 0, sizeof(p));
    p.k = arg_size(in, "k", 20);
    p.max_period = arg_size(in, "max_period", 1000000);
    p.max_avg_period = arg_double(in, "max_avg_period", 1000000.0);
    p.max_depth = arg_size(in, "max_depth", 0);
    p.max_candidates = arg_size(in, "max_candidates", 0);
    p.max_seconds = arg_double(in, "max_seconds", 0.0);
    DM_TOPKPHM_Stats s;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = topkphm_mine_file(in->source_path, &p, &s);
    dm_bench_stop(DM_PHASE_ALGO);
    DM_BenchmarkReport r = dm_bench_get_report();
    if (out) {
        out->runtime_sec = r.phase_times_ms[DM_PHASE_ALGO] / 1000.0;
        out->patterns = s.output_count;
    }
    printf("algorithm=topkphm\n");
    printf("output_count=%zu\n", s.output_count);
    printf("final_threshold=%.6f\n", s.final_threshold);
    printf("best_utility=%.6f\n", s.best_utility);
    printf("result_ram_bytes=%zu\n", s.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", s.result_disk_est_bytes);
    return rc;
}

static int run_pso_classifier_adapter(const DM_PluginInput *in, DM_PluginResult *out) {
    PSOClassifierParams p = pso_classifier_default_params();
    p.particles = arg_size(in, "particles", p.particles);
    p.max_iterations = arg_size(in, "iterations", p.max_iterations);
    p.seed = (unsigned int)arg_size(in, "seed", p.seed);
    p.indifference_threshold = arg_double(in, "threshold", p.indifference_threshold);
    p.convergence_radius = arg_double(in, "radius", p.convergence_radius);
    PSOClassifierStats s;
    dm_bench_start(DM_PHASE_ALGO);
    int rc = pso_classifier_run_spmf_folder(in->source_path, &p, &s);
    dm_bench_stop(DM_PHASE_ALGO);
    DM_BenchmarkReport r = dm_bench_get_report();
    if (out) {
        out->runtime_sec = r.phase_times_ms[DM_PHASE_ALGO] / 1000.0;
        out->patterns = s.total_rules;
    }
    printf("algorithm=pso_classifier\n");
    printf("records=%zu\n", s.records);
    printf("classes=%zu\n", s.classes);
    printf("total_rules=%zu\n", s.total_rules);
    printf("accuracy_mean=%.6f\n", s.accuracy_mean);
    printf("accuracy_stddev=%.6f\n", s.accuracy_stddev);
    printf("result_ram_bytes=%zu\n", s.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", s.result_disk_est_bytes);
    return rc;
}

#define ALL_CONNECTORS ((1u << DM_CONNECTOR_SPMF) | (1u << DM_CONNECTOR_TEXT) | (1u << DM_CONNECTOR_GRAPH))

static DM_Plugin chuo_plugin = {"chuo", "CHUO-Miner", "Closed high utility occupancy miner through the flat utility adapter.", ALL_CONNECTORS, run_chuo_adapter};
static DM_Plugin huci_plugin = {"huciminer", "HUCI-Miner", "High utility closed itemsets, generators, and HGB rules using the exact utility parser.", 0, run_huci_adapter};
static DM_Plugin tku_plugin = {"tku_miner", "TKU-Miner", "Top-k high utility miner through the flat utility adapter.", ALL_CONNECTORS, run_tku_adapter};
static DM_Plugin tku_pso_plugin = {"tku_pso", "TKU-PSO", "Particle-swarm top-k high utility miner through the flat utility adapter.", ALL_CONNECTORS, run_tku_pso_adapter};
static DM_Plugin htk_plugin = {"htk_miner", "HTK-Miner", "Top-k frequent itemset miner using its exact raw-format parser.", ALL_CONNECTORS, run_htk_adapter};
static DM_Plugin hupp_plugin = {"hupp", "HUPP", "Prompt-pattern miner using its exact raw prompt parser.", ALL_CONNECTORS, run_hupp_adapter};
static DM_Plugin kclotree_plugin = {"kclotree_miner", "KCloTree", "Top-k closed tree/sequence miner using its exact raw parser.", ALL_CONNECTORS, run_kclotree_adapter};
static DM_Plugin tipn_plugin = {"tipn_houi", "TIPN-HOUI", "Periodic negative high utility occupancy miner using its exact raw parser.", ALL_CONNECTORS, run_tipn_adapter};
static DM_Plugin topkphm_plugin = {"topkphm", "TOPKPHM", "Top-k periodic high utility miner using its exact raw parser.", ALL_CONNECTORS, run_topkphm_adapter};
static DM_Plugin pso_classifier_plugin = {"pso_classifier", "PSO Classifier", "Folder/classification-rule miner exposed as a raw-input dm_run plugin.", 0, run_pso_classifier_adapter};

DM_REGISTER_PLUGIN(chuo_plugin)
DM_REGISTER_PLUGIN(huci_plugin)
DM_REGISTER_PLUGIN(tku_plugin)
DM_REGISTER_PLUGIN(tku_pso_plugin)
DM_REGISTER_PLUGIN(htk_plugin)
DM_REGISTER_PLUGIN(hupp_plugin)
DM_REGISTER_PLUGIN(kclotree_plugin)
DM_REGISTER_PLUGIN(tipn_plugin)
DM_REGISTER_PLUGIN(topkphm_plugin)
DM_REGISTER_PLUGIN(pso_classifier_plugin)
