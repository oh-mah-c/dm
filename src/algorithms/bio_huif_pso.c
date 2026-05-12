#include "algorithms/bio_huif.h"
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
    uint8_t *pbest;
    double pbest_fitness;
} Particle;

typedef struct {
    uint32_t id;
    double twu;
    DM_BitSet *bitset;
    double *utilities; 
} HTWUI;

/* --- CONTEXT --- */

static HTWUI *htwuis = NULL;
static size_t htwui_count = 0;
static double total_min_util = 0;
static BioHUIF_Context ctx;

/* --- UTILS --- */

static void add_to_shui(uint8_t *bits, double utility) {
    size_t len = 0;
    for (size_t i = 0; i < htwui_count; i++) if (bits[i]) len++;
    if (len == 0) return;

    // Check uniqueness
    for (size_t i = 0; i < ctx.shui_count; i++) {
        if (ctx.shui[i].len == len) {
            bool match = true;
            size_t k = 0;
            for (size_t j = 0; j < htwui_count; j++) {
                if (bits[j]) {
                    if (ctx.shui[i].items[k++] != htwuis[j].id) { match = false; break; }
                }
            }
            if (match) return;
        }
    }

    if (ctx.shui_count >= ctx.shui_capacity) {
        ctx.shui_capacity = ctx.shui_capacity == 0 ? 1000 : ctx.shui_capacity * 2;
        ctx.shui = realloc(ctx.shui, sizeof(SHUI_Entry) * ctx.shui_capacity);
    }
    ctx.shui[ctx.shui_count].items = malloc(sizeof(uint32_t) * len);
    size_t k = 0;
    for (size_t j = 0; j < htwui_count; j++) if (bits[j]) ctx.shui[ctx.shui_count].items[k++] = htwuis[j].id;
    ctx.shui[ctx.shui_count].len = len;
    ctx.shui[ctx.shui_count].utility = utility;
    ctx.total_shui_utility += utility;
    ctx.shui_count++;
}

static double calculate_utility(const uint8_t *bits) {
    DM_BitSet *ts = dm_bitset_create(htwuis[0].bitset->size);
    dm_bitset_set_all(ts);
    bool first = true;
    bool has_bits = false;
    for (size_t j = 0; j < htwui_count; j++) {
        if (bits[j]) {
            has_bits = true;
            if (first) { dm_bitset_copy_to(ts, htwuis[j].bitset); first = false; }
            else { dm_bitset_and(ts, htwuis[j].bitset); }
        }
    }
    if (!has_bits) { dm_bitset_free(ts); return 0; }

    double total_u = 0;
    for (size_t i = 0; i < ts->size; i++) {
        if (dm_bitset_get(ts, i)) {
            double trans_u = 0;
            for (size_t j = 0; j < htwui_count; j++) {
                if (bits[j]) trans_u += htwuis[j].utilities[i];
            }
            total_u += trans_u;
        }
    }
    dm_bitset_free(ts);
    return total_u;
}

// Algorithm 1: PEV_Check
static void pev_check(uint8_t *bits) {
    DM_BitSet *rv = dm_bitset_create(htwuis[0].bitset->size);
    DM_BitSet *rv_prime = dm_bitset_create(htwuis[0].bitset->size);
    
    bool initialized = false;
    for (size_t j = 0; j < htwui_count; j++) {
        if (bits[j]) {
            if (!initialized) {
                dm_bitset_copy_to(rv, htwuis[j].bitset);
                initialized = true;
            } else {
                dm_bitset_copy_to(rv_prime, rv);
                dm_bitset_and(rv_prime, htwuis[j].bitset);
                
                // Check if UPEV (empty)
                bool is_upev = true;
                for (size_t w = 0; w < rv_prime->count; w++) {
                    if (rv_prime->bits[w] != 0) { is_upev = false; break; }
                }
                
                if (is_upev) {
                    bits[j] = 0; // Remove item
                } else {
                    dm_bitset_copy_to(rv, rv_prime);
                }
            }
        }
    }
    dm_bitset_free(rv);
    dm_bitset_free(rv_prime);
}

static size_t get_bit_diff(uint8_t *v1, uint8_t *v2, int *diff_indices) {
    size_t count = 0;
    for (size_t i = 0; i < htwui_count; i++) {
        if (v1[i] != v2[i]) diff_indices[count++] = i;
    }
    return count;
}

static void roulette_wheel_init_gbest(uint8_t *gbest) {
    if (ctx.shui_count == 0) { memset(gbest, 0, htwui_count); return; }
    double r = ((double)rand() / RAND_MAX) * ctx.total_shui_utility;
    double sum = 0;
    for (size_t i = 0; i < ctx.shui_count; i++) {
        sum += ctx.shui[i].utility;
        if (sum >= r) {
            memset(gbest, 0, htwui_count);
            for (size_t j = 0; j < ctx.shui[i].len; j++) {
                for (size_t k = 0; k < htwui_count; k++) {
                    if (htwuis[k].id == ctx.shui[i].items[j]) { gbest[k] = 1; break; }
                }
            }
            return;
        }
    }
}

/* --- LOGIC --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_BioHUIF_Params *p = (DM_BioHUIF_Params *)params;
    double min_util_ratio = p ? p->min_utility : 0.01;
    int pop_size = p ? p->pop_size : 30;
    int max_iter = p ? p->max_iter : 1000;

    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    double total_u = 0;
    for (size_t i = 0; i < ds->count; i++) total_u += src[i].total_utility;
    total_min_util = total_u * min_util_ratio;

    double *item_twu = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) item_twu[src[i].items[j].id] += src[i].total_utility;
    }

    htwui_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_twu[i] >= total_min_util) htwui_count++;
    }

    if (htwui_count == 0) { free(item_twu); return DM_SUCCESS; }

    htwuis = malloc(sizeof(HTWUI) * htwui_count);
    double twu_sum = 0;
    size_t h_idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_twu[i] >= total_min_util) {
            htwuis[h_idx].id = i;
            htwuis[h_idx].twu = item_twu[i];
            twu_sum += item_twu[i];
            htwuis[h_idx].bitset = dm_bitset_create(ds->count);
            htwuis[h_idx].utilities = calloc(ds->count, sizeof(double));
            for (size_t t = 0; t < ds->count; t++) {
                for (size_t k = 0; k < src[t].count; k++) {
                    if (src[t].items[k].id == i) {
                        dm_bitset_set(htwuis[h_idx].bitset, t);
                        htwuis[h_idx].utilities[t] = src[t].items[k].utility;
                        break;
                    }
                }
            }
            h_idx++;
        }
    }
    free(item_twu);

    // --- DIAGNOSTIC: verify utility computation ---
    printf("[Bio-HUIF-PSO] total_min_util=%.2f, htwui_count=%zu\n", total_min_util, htwui_count);
    double max_single_util = 0;
    for (size_t j = 0; j < htwui_count; j++) {
        double u = 0;
        for (size_t t = 0; t < ds->count; t++) {
            if (dm_bitset_get(htwuis[j].bitset, t)) u += htwuis[j].utilities[t];
        }
        if (u > max_single_util) max_single_util = u;
    }
    printf("[Bio-HUIF-PSO] Max single-item utility: %.2f (min_util=%.2f)\n", max_single_util, total_min_util);
    // -----------------------------------------------

    // Initialization
    srand((unsigned int)time(NULL));
    ctx.shui_count = 0;
    ctx.shui_capacity = 1000;
    ctx.shui = malloc(sizeof(SHUI_Entry) * ctx.shui_capacity);
    ctx.total_shui_utility = 0;

    Particle *pop = malloc(sizeof(Particle) * pop_size);
    for (int i = 0; i < pop_size; i++) {
        pop[i].bits = calloc(htwui_count, 1);
        pop[i].pbest = calloc(htwui_count, 1);
        int num = (rand() % htwui_count) + 1;
        for (int n = 0; n < num; n++) {
            double r = ((double)rand() / RAND_MAX) * twu_sum;
            double s = 0;
            for (size_t j = 0; j < htwui_count; j++) {
                s += htwuis[j].twu;
                if (s >= r) { pop[i].bits[j] = 1; break; }
            }
        }
        if (num > 1) pev_check(pop[i].bits);
        pop[i].fitness = calculate_utility(pop[i].bits);
        if (pop[i].fitness >= total_min_util) add_to_shui(pop[i].bits, pop[i].fitness);
        memcpy(pop[i].pbest, pop[i].bits, htwui_count);
        pop[i].pbest_fitness = pop[i].fitness;
    }

    uint8_t *gbest = calloc(htwui_count, 1);
    double gbest_fitness = -1;
    for (int i = 0; i < pop_size; i++) {
        if (pop[i].fitness > gbest_fitness) {
            gbest_fitness = pop[i].fitness;
            memcpy(gbest, pop[i].bits, htwui_count);
        }
    }

    int *diff_indices = malloc(sizeof(int) * htwui_count);

    // PSO Loop
    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < pop_size; i++) {
            // Update Position according to Algorithm 6
            // 1. Randomly change one bit
            int r_idx = rand() % htwui_count;
            pop[i].bits[r_idx] = 1 - pop[i].bits[r_idx];

            // 2. Diff with pbest
            size_t diff_len = get_bit_diff(pop[i].bits, pop[i].pbest, diff_indices);
            if (diff_len > 0) {
                int v_i2 = rand() % diff_len;
                for (int k = 0; k < v_i2; k++) {
                    int idx = diff_indices[rand() % diff_len];
                    pop[i].bits[idx] = 1 - pop[i].bits[idx];
                }
            }

            // 3. Diff with gbest
            diff_len = get_bit_diff(pop[i].bits, gbest, diff_indices);
            if (diff_len > 0) {
                int v_i3 = rand() % diff_len;
                for (int k = 0; k < v_i3; k++) {
                    int idx = diff_indices[rand() % diff_len];
                    pop[i].bits[idx] = 1 - pop[i].bits[idx];
                }
            }

            pev_check(pop[i].bits);
            pop[i].fitness = calculate_utility(pop[i].bits);
            if (pop[i].fitness >= total_min_util) add_to_shui(pop[i].bits, pop[i].fitness);

            if (pop[i].fitness > pop[i].pbest_fitness) {
                pop[i].pbest_fitness = pop[i].fitness;
                memcpy(pop[i].pbest, pop[i].bits, htwui_count);
            }
        }
        // Roulette Wheel Selection for gbest of next population (Song & Huang Strategy)
        roulette_wheel_init_gbest(gbest);
    }

    dm_bench_record_results(ctx.shui_count, 0);

    // Cleanup
    free(diff_indices);
    free(gbest);
    for (int i = 0; i < pop_size; i++) { free(pop[i].bits); free(pop[i].pbest); }
    free(pop);
    for (size_t i = 0; i < htwui_count; i++) { dm_bitset_free(htwuis[i].bitset); free(htwuis[i].utilities); }
    free(htwuis);
    for (size_t i = 0; i < ctx.shui_count; i++) free(ctx.shui[i].items);
    free(ctx.shui);

    return DM_SUCCESS;
}

DM_Algorithm bio_huif_pso_algo = {
    .id = "bio_huif_pso",
    .name = "Bio-HUIF-PSO",
    .description = "Bio-inspired HUIF based on PSO (Song & Huang 2018) with Diverse Optimal Value Framework.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(bio_huif_pso_algo)
