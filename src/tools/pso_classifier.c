#include "algorithms/pso_classifier.h"
#include "core/dm_benchmark.h"

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
    printf("  %s --input <spmf_class_folder> [--particles N] [--max-iterations N]\n", prog);
    printf("     [--threshold T] [--radius R] [--uncovered R] [--seed N] [--max-records N]\n");
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    if (!input || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return input ? 0 : 1;
    }

    PSOClassifierParams params = pso_classifier_default_params();
    params.particles = (size_t)strtoull(arg_value(argc, argv, "--particles", "25"), NULL, 10);
    params.max_iterations = (size_t)strtoull(arg_value(argc, argv, "--max-iterations", "1000"), NULL, 10);
    params.indifference_threshold = atof(arg_value(argc, argv, "--threshold", "0.90"));
    params.convergence_radius = atof(arg_value(argc, argv, "--radius", "0.01"));
    params.uncovered_ratio = atof(arg_value(argc, argv, "--uncovered", "0.10"));
    params.seed = (unsigned int)strtoul(arg_value(argc, argv, "--seed", "7"), NULL, 10);
    params.max_records = (size_t)strtoull(arg_value(argc, argv, "--max-records", "0"), NULL, 10);

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    dm_bench_start(DM_PHASE_ALGO);
    PSOClassifierStats stats;
    int rc = pso_classifier_run_spmf_folder(input, &params, &stats);
    dm_bench_stop(DM_PHASE_ALGO);
    dm_bench_stop(DM_PHASE_TOTAL);
    DM_BenchmarkReport report = dm_bench_get_report();

    if (rc != 0) {
        fprintf(stderr, "PSO classifier failed. Expected a folder of labeled SPMF sequence files named <class>translated.txt.\n");
        return 1;
    }

    printf("Particle Swarm Data Mining Classifier\n");
    printf("input=%s\n", input);
    printf("records=%zu\n", stats.records);
    printf("attributes=%zu\n", stats.attributes);
    printf("classes=%zu\n", stats.classes);
    printf("folds=%zu\n", stats.folds);
    printf("particles=%zu\n", stats.particles);
    printf("max_iterations=%zu\n", params.max_iterations);
    printf("indifference_threshold=%.10g\n", params.indifference_threshold);
    printf("convergence_radius=%.10g\n", params.convergence_radius);
    printf("uncovered_ratio=%.10g\n", params.uncovered_ratio);
    printf("constriction=%.10g\n", params.constriction);
    printf("acceleration_limit=%.10g\n", params.acceleration_limit);
    printf("seed=%u\n", params.seed);
    printf("accuracy_mean=%.6f\n", stats.accuracy_mean);
    printf("accuracy_stddev=%.6f\n", stats.accuracy_stddev);
    printf("accuracy_percent=%.3f\n", stats.accuracy_mean * 100.0);
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_ALGO] / 1000.0);
    printf("total_sec=%.6f\n", report.phase_times_ms[DM_PHASE_TOTAL] / 1000.0);
    printf("training_sec=%.6f\n", stats.training_sec);
    printf("validation_sec=%.6f\n", stats.validation_sec);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    printf("total_rules=%zu\n", stats.total_rules);
    printf("rules_per_set=%.6f\n", stats.rules_per_set);
    printf("total_attribute_tests=%zu\n", stats.total_attribute_tests);
    printf("tests_per_rule=%.6f\n", stats.tests_per_rule);
    printf("total_iterations=%zu\n", stats.total_iterations);
    printf("iterations_per_rule=%.6f\n", stats.iterations_per_rule);
    printf("removed_training_instances=%zu\n", stats.total_removed_instances);
    printf("cleaned_rules=%zu\n", stats.total_cleaned_rules);
    printf("result_ram_bytes=%zu\n", stats.result_ram_bytes);
    printf("result_disk_est_bytes=%zu\n", stats.result_disk_est_bytes);
    return 0;
}
