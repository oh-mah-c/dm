#ifndef DM_TKU_PSO_H
#define DM_TKU_PSO_H

#include "core/dm_dataset.h"

#include <stddef.h>

typedef struct {
    size_t k;
    size_t population_size;
    size_t iterations;
    unsigned int seed;
    double max_seconds;
} DM_TKU_PSO_Params;

typedef struct {
    size_t transactions;
    size_t distinct_items;
    size_t kept_items;
    size_t k;
    size_t population_size;
    size_t iterations;
    double cuv_threshold;
    double final_threshold;
    size_t threshold_raises;
    size_t output_count;
    size_t total_output_items;
    double best_utility;
    double avg_utility;
    double avg_length;
    double deviation;
    size_t initialized_singletons;
    size_t roulette_initialized;
    size_t pev_repairs;
    size_t explored_particles;
    size_t redundant_particles;
    size_t evaluated_particles;
    size_t skipped_estimation;
    size_t overestimates;
    size_t underestimates;
    size_t pruned_twu_items;
    size_t result_ram_bytes;
    size_t result_disk_est_bytes;
    int limited;
} DM_TKU_PSO_Stats;

int tku_pso_mine_dataset(DM_Dataset *ds, const DM_TKU_PSO_Params *params, DM_TKU_PSO_Stats *stats);

#endif
