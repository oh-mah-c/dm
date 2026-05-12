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
} Chromosome;

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

static void pev_check(uint8_t *bits) {
    DM_BitSet *rv = dm_bitset_create(htwuis[0].bitset->size);
    DM_BitSet *rv_prime = dm_bitset_create(htwuis[0].bitset->size);
    bool initialized = false;
    for (size_t j = 0; j < htwui_count; j++) {
        if (bits[j]) {
            if (!initialized) { dm_bitset_copy_to(rv, htwuis[j].bitset); initialized = true; }
            else {
                dm_bitset_copy_to(rv_prime, rv);
                dm_bitset_and(rv_prime, htwuis[j].bitset);
                bool is_upev = true;
                for (size_t w = 0; w < rv_prime->count; w++) { if (rv_prime->bits[w] != 0) { is_upev = false; break; } }
                if (is_upev) bits[j] = 0;
                else dm_bitset_copy_to(rv, rv_prime);
            }
        }
    }
    dm_bitset_free(rv);
    dm_bitset_free(rv_prime);
}

static size_t get_bit_diff(uint8_t *v1, uint8_t *v2, int *diff_indices) {
    size_t count = 0;
    for (size_t i = 0; i < htwui_count; i++) { if (v1[i] != v2[i]) diff_indices[count++] = i; }
    return count;
}

static int select_chromosome(Chromosome *pop, int pop_size) {
    double total_fitness = 0;
    for (int i = 0; i < pop_size; i++) total_fitness += pop[i].fitness;
    if (total_fitness == 0) return rand() % pop_size;
    double r = ((double)rand() / RAND_MAX) * total_fitness;
    double sum = 0;
    for (int i = 0; i < pop_size; i++) {
        sum += pop[i].fitness;
        if (sum >= r) return i;
    }
    return pop_size - 1;
}

static void roulette_wheel_init_hui(uint8_t *bits) {
    if (ctx.shui_count == 0) { memset(bits, 0, htwui_count); return; }
    double r = ((double)rand() / RAND_MAX) * ctx.total_shui_utility;
    double sum = 0;
    for (size_t i = 0; i < ctx.shui_count; i++) {
        sum += ctx.shui[i].utility;
        if (sum >= r) {
            memset(bits, 0, htwui_count);
            for (size_t j = 0; j < ctx.shui[i].len; j++) {
                for (size_t k = 0; k < htwui_count; k++) {
                    if (htwuis[k].id == ctx.shui[i].items[j]) { bits[k] = 1; break; }
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
    for (uint32_t i = 0; i <= ds->max_id; i++) { if (item_twu[i] >= total_min_util) htwui_count++; }
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

    srand((unsigned int)time(NULL));
    ctx.shui_count = 0;
    ctx.shui_capacity = 1000;
    ctx.shui = malloc(sizeof(SHUI_Entry) * ctx.shui_capacity);
    ctx.total_shui_utility = 0;

    Chromosome *pop = malloc(sizeof(Chromosome) * pop_size);
    for (int i = 0; i < pop_size; i++) {
        pop[i].bits = calloc(htwui_count, 1);
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
    }

    Chromosome *next_pop = malloc(sizeof(Chromosome) * pop_size);
    for (int i = 0; i < pop_size; i++) next_pop[i].bits = calloc(htwui_count, 1);
    int *diff_indices = malloc(sizeof(int) * htwui_count);

    // GA Loop
    for (int iter = 0; iter < max_iter; iter++) {
        int next_idx = 0;
        while (next_idx < pop_size) {
            int i = select_chromosome(pop, pop_size);
            int j = select_chromosome(pop, pop_size);
            size_t diff_len = get_bit_diff(pop[i].bits, pop[j].bits, diff_indices);
            
            // Crossover for two offspring
            for (int k = 0; k < 2 && next_idx < pop_size; k++) {
                uint8_t *parent = (k == 0) ? pop[i].bits : pop[j].bits;
                memcpy(next_pop[next_idx].bits, parent, htwui_count);
                
                if (diff_len > 0) {
                    int cnum = (int)((double)diff_len * ((double)rand() / RAND_MAX));
                    for (int m = 0; m < cnum; m++) {
                        int idx = diff_indices[rand() % diff_len];
                        next_pop[next_idx].bits[idx] = 1 - next_pop[next_idx].bits[idx];
                    }
                }
                
                // Mutation
                int m_idx = rand() % htwui_count;
                next_pop[next_idx].bits[m_idx] = 1 - next_pop[next_idx].bits[m_idx];
                
                pev_check(next_pop[next_idx].bits);
                next_pop[next_idx].fitness = calculate_utility(next_pop[next_idx].bits);
                if (next_pop[next_idx].fitness >= total_min_util) add_to_shui(next_pop[next_idx].bits, next_pop[next_idx].fitness);
                next_idx++;
            }
        }
        // Replace two random chromosomes with diverse HUIs
        if (ctx.shui_count >= 2) {
            int r1 = rand() % pop_size;
            int r2 = rand() % pop_size;
            roulette_wheel_init_hui(next_pop[r1].bits);
            next_pop[r1].fitness = calculate_utility(next_pop[r1].bits);
            roulette_wheel_init_hui(next_pop[r2].bits);
            next_pop[r2].fitness = calculate_utility(next_pop[r2].bits);
        }

        // Swap populations
        for (int i = 0; i < pop_size; i++) {
            uint8_t *tmp = pop[i].bits;
            pop[i].bits = next_pop[i].bits;
            next_pop[i].bits = tmp;
            pop[i].fitness = next_pop[i].fitness;
        }
    }

    dm_bench_record_results(ctx.shui_count, 0);

    // Cleanup
    free(diff_indices);
    for (int i = 0; i < pop_size; i++) { free(pop[i].bits); free(next_pop[i].bits); }
    free(pop); free(next_pop);
    for (size_t i = 0; i < htwui_count; i++) { dm_bitset_free(htwuis[i].bitset); free(htwuis[i].utilities); }
    free(htwuis);
    for (size_t i = 0; i < ctx.shui_count; i++) free(ctx.shui[i].items);
    free(ctx.shui);
    return DM_SUCCESS;
}

DM_Algorithm bio_huif_ga_algo = {
    .id = "bio_huif_ga",
    .name = "Bio-HUIF-GA",
    .description = "Bio-inspired HUIF based on GA (Song & Huang 2018) with Diverse Optimal Value Framework.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(bio_huif_ga_algo)
