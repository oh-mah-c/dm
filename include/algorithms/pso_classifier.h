#ifndef DM_PSO_CLASSIFIER_H
#define DM_PSO_CLASSIFIER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t particles;
    size_t max_iterations;
    double indifference_threshold;
    double convergence_radius;
    double uncovered_ratio;
    double constriction;
    double acceleration_limit;
    unsigned int seed;
    size_t max_records;
} PSOClassifierParams;

typedef struct {
    size_t records;
    size_t attributes;
    size_t classes;
    size_t folds;
    size_t particles;
    size_t total_rules;
    size_t total_attribute_tests;
    size_t total_iterations;
    size_t total_removed_instances;
    size_t total_cleaned_rules;
    double accuracy_mean;
    double accuracy_stddev;
    double rules_per_set;
    double tests_per_rule;
    double iterations_per_rule;
    double training_sec;
    double validation_sec;
    double peak_ram_mb;
    size_t result_ram_bytes;
    size_t result_disk_est_bytes;
} PSOClassifierStats;

PSOClassifierParams pso_classifier_default_params(void);
int pso_classifier_run_spmf_folder(const char *folder,
                                   const PSOClassifierParams *params,
                                   PSOClassifierStats *stats);

#endif
