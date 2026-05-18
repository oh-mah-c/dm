#include "algorithms/hupp.h"
#include "core/dm_benchmark.h"

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
    printf("  %s --input <prompt_jsonl> [--dataset-type auto|dolly|code_feedback]\n", prog);
    printf("     [--minsup ratio|count] [--minutil value] [--minutil-ratio value] [--minalign value]\n");
    printf("     [--weights token,latency,alignment,cache] [--max-transactions N] [--max-patterns N]\n");
    printf("     [--max-depth N] [--max-seconds S]\n");
}

static int parse_weights(const char *s, HUPPParams *p) {
    if (!s) return 0;
    double a, b, c, d;
    if (sscanf(s, "%lf,%lf,%lf,%lf", &a, &b, &c, &d) != 4) return -1;
    p->w_token = a;
    p->w_latency = b;
    p->w_alignment = c;
    p->w_cache = d;
    return 0;
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    if (!input || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return input ? 0 : 1;
    }

    HUPPParams params = hupp_default_params();
    const char *type_s = arg_value(argc, argv, "--dataset-type", "auto");
    if (hupp_parse_dataset_type(type_s, &params.dataset_type) != 0) {
        fprintf(stderr, "Unknown HUPP dataset type: %s\n", type_s);
        return 1;
    }
    params.min_support = atof(arg_value(argc, argv, "--minsup", "0.02"));
    params.min_utility = atof(arg_value(argc, argv, "--minutil", "0"));
    params.min_utility_ratio = atof(arg_value(argc, argv, "--minutil-ratio", "0.002"));
    params.min_alignment = atof(arg_value(argc, argv, "--minalign", "0.80"));
    params.max_transactions = (size_t)strtoull(arg_value(argc, argv, "--max-transactions", "0"), NULL, 10);
    params.max_concepts_per_prompt = (size_t)strtoull(arg_value(argc, argv, "--max-concepts", "64"), NULL, 10);
    params.max_patterns = (size_t)strtoull(arg_value(argc, argv, "--max-patterns", "0"), NULL, 10);
    params.max_depth = (size_t)strtoull(arg_value(argc, argv, "--max-depth", "6"), NULL, 10);
    params.max_seconds = atof(arg_value(argc, argv, "--max-seconds", "0"));
    if (parse_weights(arg_value(argc, argv, "--weights", NULL), &params) != 0) {
        fprintf(stderr, "Weights must be token,latency,alignment,cache\n");
        return 1;
    }

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    dm_bench_start(DM_PHASE_ALGO);
    HUPPStats stats;
    int rc = hupp_mine_file(input, &params, &stats);
    dm_bench_stop(DM_PHASE_ALGO);
    dm_bench_stop(DM_PHASE_TOTAL);
    dm_bench_record_results(stats.emitted_patterns, stats.total_pattern_items);
    DM_BenchmarkReport report = dm_bench_get_report();

    if (rc != 0) {
        fprintf(stderr, "HUPP mining failed for %s\n", input);
        return 1;
    }

    printf("HUPP Miner\n");
    printf("input=%s\n", input);
    printf("dataset_type=%s\n", hupp_dataset_type_name(params.dataset_type));
    printf("transactions=%zu\n", stats.transactions);
    printf("semantic_concepts=%zu\n", stats.semantic_concepts);
    printf("total_prompt_tokens=%zu\n", stats.total_prompt_tokens);
    printf("avg_prompt_tokens=%.6f\n", stats.transactions ? (double)stats.total_prompt_tokens / (double)stats.transactions : 0.0);
    printf("minsup_ratio=%.10g\n", params.min_support < 1.0 ? params.min_support : (stats.transactions ? (double)stats.minsup_count / (double)stats.transactions : 0.0));
    printf("minsup_count=%u\n", stats.minsup_count);
    printf("theta=%.10g\n", stats.theta);
    printf("alpha_min=%.10g\n", stats.alpha_min);
    printf("weights=%.6f,%.6f,%.6f,%.6f\n", params.w_token, params.w_latency, params.w_alignment, params.w_cache);
    printf("status=%s\n", stats.limited ? "LIMITED" : "OK");
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_ALGO] / 1000.0);
    printf("total_sec=%.6f\n", report.phase_times_ms[DM_PHASE_TOTAL] / 1000.0);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    printf("candidate_optimizers=%zu\n", stats.candidate_optimizers);
    printf("pareto_removed=%zu\n", stats.pareto_removed);
    printf("frequent_singletons=%zu\n", stats.frequent_singletons);
    printf("singleton_occurrences=%zu\n", stats.singleton_occurrences);
    printf("visited_nodes=%zu\n", stats.visited_nodes);
    printf("joins=%zu\n", stats.joins);
    printf("joined_entries=%zu\n", stats.joined_entries);
    printf("pruned_support=%zu\n", stats.pruned_support);
    printf("pruned_ptwo=%zu\n", stats.pruned_ptwo);
    printf("pruned_aaub=%zu\n", stats.pruned_aaub);
    printf("output_count=%zu\n", stats.emitted_patterns);
    printf("total_output_items=%zu\n", stats.total_pattern_items);
    printf("avg_output_length=%.6f\n", stats.emitted_patterns ? (double)stats.total_pattern_items / (double)stats.emitted_patterns : 0.0);
    printf("avg_support=%.6f\n", stats.avg_support);
    printf("avg_utility=%.6f\n", stats.avg_utility);
    printf("avg_alignment=%.6f\n", stats.avg_alignment);
    printf("best_utility=%.6f\n", stats.best_utility);
    printf("best_alignment=%.6f\n", stats.best_alignment);
    printf("max_depth=%zu\n", stats.max_depth_seen);
    printf("result_ram_bytes=%zu\n", stats.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", stats.result_disk_est_bytes);
    return stats.limited ? 2 : 0;
}
