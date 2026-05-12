#include "algorithms/hupe_garm.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include "core/dm_bitset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint8_t *bits;
    double fitness;
    int rank;
} Chromosome;

typedef struct {
    uint32_t id;
    double twu;
    double profit;      // External utility
    DM_BitSet *bitset;  // Transactions containing this item
} GARM_Item;

/* --- CONTEXT --- */

static GARM_Item *promising_items = NULL;
static size_t item_count = 0;
static double total_min_util = 0;

/* --- UTILS --- */

static double calculate_fitness(uint8_t *bits, DM_Trans_Utility *src, size_t ds_count) {
    double total_u = 0;
    bool has_positive = false;
    
    DM_BitSet *intersection = dm_bitset_create(ds_count);
    dm_bitset_set_all(intersection);
    
    bool empty = true;
    for (size_t j = 0; j < item_count; j++) {
        if (bits[j]) {
            dm_bitset_and(intersection, promising_items[j].bitset);
            empty = false;
            if (promising_items[j].profit > 0) has_positive = true;
        }
    }
    
    if (empty || !has_positive) {
        dm_bitset_free(intersection);
        return -1e18; // Very low fitness
    }
    
    for (size_t i = 0; i < ds_count; i++) {
        if (dm_bitset_get(intersection, i)) {
            double trans_u = 0;
            for (size_t j = 0; j < item_count; j++) {
                if (bits[j]) {
                    for (size_t k = 0; k < src[i].count; k++) {
                        if (src[i].items[k].id == promising_items[j].id) {
                            trans_u += src[i].items[k].utility;
                            break;
                        }
                    }
                }
            }
            total_u += trans_u;
        }
    }
    
    dm_bitset_free(intersection);
    return total_u;
}

static int compare_chromosomes(const void *a, const void *b) {
    Chromosome *c1 = (Chromosome *)a;
    Chromosome *c2 = (Chromosome *)b;
    if (c1->fitness > c2->fitness) return -1;
    if (c1->fitness < c2->fitness) return 1;
    return 0;
}

/* --- LOGIC --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_HUPE_GARM_Params *p = (DM_HUPE_GARM_Params *)params;
    double min_util_ratio = p ? p->min_utility : 0.02;
    int pop_size = p ? p->pop_size : 50;
    int max_iter = p ? p->max_iter : 100;
    double p_max = p ? p->p_max : 0.1;
    double p_min = p ? p->p_min : 0.01;
    int k_top = p ? p->k : 0;
    
    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    double total_u = 0;
    for (size_t i = 0; i < ds->count; i++) total_u += src[i].total_utility;
    total_min_util = (k_top > 0) ? -1e18 : (total_u * min_util_ratio);
    
    // Database reorganization (only for UMU)
    double *item_twu = calloc(ds->max_id + 1, sizeof(double));
    if (k_top == 0) {
        for (size_t i = 0; i < ds->count; i++) {
            double tu_pos = 0;
            for (size_t j = 0; j < src[i].count; j++) {
                if (src[i].items[j].utility > 0) tu_pos += src[i].items[j].utility;
            }
            for (size_t j = 0; j < src[i].count; j++) item_twu[src[i].items[j].id] += tu_pos;
        }
    } else {
        // For WUMU, keep all items that appear in DB
        for (size_t i = 0; i < ds->count; i++) {
            for (size_t j = 0; j < src[i].count; j++) item_twu[src[i].items[j].id] = 1.0;
        }
    }
    
    item_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_twu[i] >= total_min_util || k_top > 0) {
            if (item_twu[i] > 0) item_count++;
        }
    }
    
    if (item_count == 0) {
        free(item_twu);
        return DM_SUCCESS;
    }
    
    double *item_profit = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) {
            item_profit[src[i].items[j].id] = (src[i].items[j].utility >= 0) ? 1.0 : -1.0;
        }
    }

    promising_items = malloc(sizeof(GARM_Item) * item_count);
    size_t idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_twu[i] >= total_min_util || (k_top > 0 && item_twu[i] > 0)) {
            promising_items[idx].id = i;
            promising_items[idx].twu = item_twu[i];
            promising_items[idx].profit = item_profit[i];
            promising_items[idx].bitset = dm_bitset_create(ds->count);
            for (size_t t = 0; t < ds->count; t++) {
                for (size_t k = 0; k < src[t].count; k++) {
                    if (src[t].items[k].id == i) {
                        dm_bitset_set(promising_items[idx].bitset, t);
                        break;
                    }
                }
            }
            idx++;
            if (idx >= item_count) break;
        }
    }
    free(item_twu); free(item_profit);
    
    srand((unsigned int)time(NULL));
    
    // Initial Population
    Chromosome *pop = malloc(sizeof(Chromosome) * pop_size);
    for (int i = 0; i < pop_size; i++) {
        pop[i].bits = calloc(item_count, 1);
        // Randomly select k items (k between 1 and item_count)
        int k = (rand() % item_count) + 1;
        for (int j = 0; j < k; j++) {
            int pos = rand() % item_count;
            pop[i].bits[pos] = 1;
        }
        pop[i].fitness = calculate_fitness(pop[i].bits, src, ds->count);
    }
    
    size_t found_hui_count = 0;
    
    // GA Generations
    for (int gen = 0; gen < max_iter; gen++) {
        // Selection (Roulette Wheel)
        double total_fitness = 0;
        double min_f = 0;
        for (int i = 0; i < pop_size; i++) {
            if (pop[i].fitness > min_f) total_fitness += pop[i].fitness;
        }
        
        Chromosome *next_pop = malloc(sizeof(Chromosome) * pop_size);
        for (int i = 0; i < pop_size; i++) {
            double r = ((double)rand() / RAND_MAX) * total_fitness;
            double sum = 0;
            int selected = 0;
            for (int j = 0; j < pop_size; j++) {
                if (pop[j].fitness > 0) sum += pop[j].fitness;
                if (sum >= r) { selected = j; break; }
            }
            next_pop[i].bits = malloc(item_count);
            memcpy(next_pop[i].bits, pop[selected].bits, item_count);
        }
        
        // Crossover (Single Point)
        for (int i = 0; i < pop_size; i += 2) {
            if (i + 1 < pop_size) {
                int point = rand() % item_count;
                for (int j = point; j < (int)item_count; j++) {
                    uint8_t tmp = next_pop[i].bits[j];
                    next_pop[i].bits[j] = next_pop[i+1].bits[j];
                    next_pop[i+1].bits[j] = tmp;
                }
            }
        }
        
        // Ranked Mutation
        // First, rank the next_pop to apply ranked mutation
        for (int i = 0; i < pop_size; i++) {
            next_pop[i].fitness = calculate_fitness(next_pop[i].bits, src, ds->count);
        }
        qsort(next_pop, pop_size, sizeof(Chromosome), compare_chromosomes);
        for (int i = 0; i < pop_size; i++) next_pop[i].rank = i + 1;
        
        for (int i = 0; i < pop_size; i++) {
            double pm = p_max - ((p_max - p_min) / max_iter) * gen * ((double)next_pop[i].rank / pop_size);
            for (size_t j = 0; j < item_count; j++) {
                if (((double)rand() / RAND_MAX) < pm) {
                    next_pop[i].bits[j] = !next_pop[i].bits[j];
                }
            }
            next_pop[i].fitness = calculate_fitness(next_pop[i].bits, src, ds->count);
            if (next_pop[i].fitness >= total_min_util) found_hui_count++;
        }
        
        // Evaluation (Elitist): Combine pop and next_pop, take top pop_size
        Chromosome *combined = malloc(sizeof(Chromosome) * pop_size * 2);
        memcpy(combined, pop, sizeof(Chromosome) * pop_size);
        memcpy(combined + pop_size, next_pop, sizeof(Chromosome) * pop_size);
        
        qsort(combined, pop_size * 2, sizeof(Chromosome), compare_chromosomes);
        
        for (int i = 0; i < pop_size; i++) {
            memcpy(pop[i].bits, combined[i].bits, item_count);
            pop[i].fitness = combined[i].fitness;
        }
        
        for (int i = pop_size; i < pop_size * 2; i++) free(combined[i].bits);
        free(combined);
        free(next_pop);
    }
    
    // Cleanup
    for (size_t i = 0; i < item_count; i++) dm_bitset_free(promising_items[i].bitset);
    free(promising_items);
    for (int i = 0; i < pop_size; i++) free(pop[i].bits);
    free(pop);
    
    dm_bench_record_results(found_hui_count, 0);
    
    return DM_SUCCESS;
}

static DM_Algorithm hupe_garm_algo = {
    .id = "hupe_garm",
    .name = "HUPE-GARM",
    .description = "High Utility Pattern Extraction using Genetic Algorithm with Ranked Mutation.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(hupe_garm_algo)
