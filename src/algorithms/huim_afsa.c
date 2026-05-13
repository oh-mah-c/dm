#include "algorithms/huim_afsa.h"
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
} Fish;

typedef struct {
    uint32_t id;
    double twu;
    DM_BitSet *bitset;
    double *utilities; 
} HTWUI;

typedef struct {
    uint32_t *items;
    size_t len;
    double utility;
} SHUI_Entry;

typedef struct {
    SHUI_Entry *shui;
    size_t shui_count;
    size_t shui_capacity;
    double total_shui_utility;
} AFSA_Context;

/* --- CONTEXT --- */

static HTWUI *htwuis = NULL;
static size_t htwui_count = 0;
static double total_min_util = 0;
static AFSA_Context ctx;

/* --- UTILS --- */

static void add_to_shui(uint8_t *bits, double utility) {
    size_t len = 0;
    for (size_t i = 0; i < htwui_count; i++) if (bits[i]) len++;
    if (len == 0) return;

    // Uniqueness check
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
                bool is_empty = true;
                for (size_t w = 0; w < rv_prime->count; w++) {
                    if (rv_prime->bits[w] != 0) { is_empty = false; break; }
                }
                if (is_empty) bits[j] = 0;
                else dm_bitset_copy_to(rv, rv_prime);
            }
        }
    }
    dm_bitset_free(rv);
    dm_bitset_free(rv_prime);
}

static int get_hamming_dist(const uint8_t *v1, const uint8_t *v2) {
    int dist = 0;
    for (size_t i = 0; i < htwui_count; i++) if (v1[i] != v2[i]) dist++;
    return dist;
}

static void move_towards(uint8_t *current, const uint8_t *target, int step) {
    int *diffs = malloc(sizeof(int) * htwui_count);
    int diff_count = 0;
    for (size_t i = 0; i < htwui_count; i++) if (current[i] != target[i]) diffs[diff_count++] = i;
    
    if (diff_count > 0) {
        int move_bits = (step < diff_count) ? step : diff_count;
        for (int i = 0; i < move_bits; i++) {
            int r = rand() % (diff_count - i);
            int idx = diffs[r];
            current[idx] = target[idx];
            diffs[r] = diffs[diff_count - 1 - i];
        }
    }
    free(diffs);
}

/* --- AFSA BEHAVIORS --- */

static void behavior_prey(uint8_t *bits, int visual, int step, int try_number) {
    uint8_t *trial = malloc(htwui_count);
    double current_fitness = calculate_utility(bits);
    
    for (int i = 0; i < try_number; i++) {
        memcpy(trial, bits, htwui_count);
        // Random move in visual range
        int v_step = (rand() % visual) + 1;
        for (int j = 0; j < v_step; j++) {
            int idx = rand() % htwui_count;
            trial[idx] = 1 - trial[idx];
        }
        pev_check(trial);
        double f = calculate_utility(trial);
        if (f > current_fitness) {
            move_towards(bits, trial, step);
            pev_check(bits);
            free(trial);
            return;
        }
    }
    
    // If not found, take a random small move
    int r_idx = rand() % htwui_count;
    bits[r_idx] = 1 - bits[r_idx];
    pev_check(bits);
    free(trial);
}

static void behavior_swarm(uint8_t *bits, Fish *pop, int pop_size, int visual, int step, double delta) {
    int *neighbors = malloc(sizeof(int) * pop_size);
    int n_count = 0;
    for (int i = 0; i < pop_size; i++) {
        if (get_hamming_dist(bits, pop[i].bits) <= visual) neighbors[n_count++] = i;
    }

    if (n_count > 0) {
        uint8_t *center = calloc(htwui_count, 1);
        for (size_t j = 0; j < htwui_count; j++) {
            int count = 0;
            for (int k = 0; k < n_count; k++) if (pop[neighbors[k]].bits[j]) count++;
            if (count * 2 >= n_count) center[j] = 1; // Majority rule for center
        }
        pev_check(center);
        double f_center = calculate_utility(center);
        double current_f = calculate_utility(bits);
        
        // Crowding check: n_count / pop_size < delta
        if (f_center > current_f && (double)n_count / pop_size < delta) {
            move_towards(bits, center, step);
            pev_check(bits);
        } else {
            behavior_prey(bits, visual, step, 5);
        }
        free(center);
    } else {
        behavior_prey(bits, visual, step, 5);
    }
    free(neighbors);
}

static void behavior_follow(uint8_t *bits, Fish *pop, int pop_size, int visual, int step, double delta) {
    int best_neighbor = -1;
    double best_f = -1;
    int n_count = 0;

    for (int i = 0; i < pop_size; i++) {
        int dist = get_hamming_dist(bits, pop[i].bits);
        if (dist <= visual) {
            n_count++;
            if (pop[i].fitness > best_f) {
                best_f = pop[i].fitness;
                best_neighbor = i;
            }
        }
    }

    if (best_neighbor != -1) {
        double current_f = calculate_utility(bits);
        if (best_f > current_f && (double)n_count / pop_size < delta) {
            move_towards(bits, pop[best_neighbor].bits, step);
            pev_check(bits);
        } else {
            behavior_prey(bits, visual, step, 5);
        }
    } else {
        behavior_prey(bits, visual, step, 5);
    }
}

/* --- MAIN LOGIC --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_HUIM_AFSA_Params *p = (DM_HUIM_AFSA_Params *)params;
    
    double min_util_ratio = p ? p->min_utility : 0.01;
    int pop_size = p ? p->pop_size : 30;
    int max_iter = p ? p->max_iter : 1000;
    int visual = p ? p->visual : 10;
    int step = p ? p->step : 2;
    int try_number = p ? p->try_number : 5;
    double delta = p ? p->delta : 0.618;

    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    double total_u = 0;
    for (size_t i = 0; i < ds->count; i++) total_u += src[i].total_utility;
    total_min_util = total_u * min_util_ratio;

    double *item_twu = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) item_twu[src[i].items[j].id] += src[i].total_utility;
    }

    htwui_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (item_twu[i] >= total_min_util) htwui_count++;
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

    Fish *pop = malloc(sizeof(Fish) * pop_size);
    for (int i = 0; i < pop_size; i++) {
        pop[i].bits = calloc(htwui_count, 1);
        int num = (rand() % 5) + 1; // Small initial sets
        for (int n = 0; n < num; n++) {
            double r = ((double)rand() / RAND_MAX) * twu_sum;
            double s = 0;
            for (size_t j = 0; j < htwui_count; j++) {
                s += htwuis[j].twu;
                if (s >= r) { pop[i].bits[j] = 1; break; }
            }
        }
        pev_check(pop[i].bits);
        pop[i].fitness = calculate_utility(pop[i].bits);
        if (pop[i].fitness >= total_min_util) add_to_shui(pop[i].bits, pop[i].fitness);
    }

    // AFSA Loop
    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < pop_size; i++) {
            // Decide behavior: Follow and Swarm are evaluated, then Prey as default
            // In discrete version, we can try both and pick best move or choose one
            if (rand() % 2 == 0) {
                behavior_swarm(pop[i].bits, pop, pop_size, visual, step, delta);
            } else {
                behavior_follow(pop[i].bits, pop, pop_size, visual, step, delta);
            }
            
            pop[i].fitness = calculate_utility(pop[i].bits);
            if (pop[i].fitness >= total_min_util) add_to_shui(pop[i].bits, pop[i].fitness);
        }
    }

    dm_bench_record_results(ctx.shui_count, 0);

    // Cleanup
    for (int i = 0; i < pop_size; i++) free(pop[i].bits);
    free(pop);
    for (size_t i = 0; i < htwui_count; i++) { dm_bitset_free(htwuis[i].bitset); free(htwuis[i].utilities); }
    free(htwuis);
    for (size_t i = 0; i < ctx.shui_count; i++) free(ctx.shui[i].items);
    free(ctx.shui);
    
    return DM_SUCCESS;
}

DM_Algorithm huim_afsa_algo = {
    .id = "huim_afsa",
    .name = "HUIM-AFSA",
    .description = "Artificial Fish Swarm Algorithm for Mining High Utility Itemsets (Song et al. 2021).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(huim_afsa_algo)
