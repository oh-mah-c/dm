#include "algorithms/huim_abc.h"
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
    int trial;
} FoodSource;

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
} ABC_Context;

/* --- CONTEXT --- */

static HTWUI *htwuis = NULL;
static size_t htwui_count = 0;
static double total_min_util = 0;
static ABC_Context ctx;

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
            for (size_t j = 0; j < htwui_count; j++) if (bits[j]) trans_u += htwuis[j].utilities[i];
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
                for (size_t w = 0; w < rv_prime->count; w++) { if (rv_prime->bits[w] != 0) { is_empty = false; break; } }
                if (is_empty) bits[j] = 0;
                else dm_bitset_copy_to(rv, rv_prime);
            }
        }
    }
    dm_bitset_free(rv);
    dm_bitset_free(rv_prime);
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

static void initialize_food_source(FoodSource *fs, double twu_sum) {
    memset(fs->bits, 0, htwui_count);
    int num = (rand() % 5) + 1;
    for (int n = 0; n < num; n++) {
        double r = ((double)rand() / RAND_MAX) * twu_sum;
        double s = 0;
        for (size_t j = 0; j < htwui_count; j++) {
            s += htwuis[j].twu;
            if (s >= r) { fs->bits[j] = 1; break; }
        }
    }
    pev_check(fs->bits);
    fs->fitness = calculate_utility(fs->bits);
    fs->trial = 0;
}

/* --- LOGIC --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_HUIM_ABC_Params *p = (DM_HUIM_ABC_Params *)params;
    
    double min_util_ratio = p ? p->min_utility : 0.01;
    int pop_size = p ? p->pop_size : 30; // SN
    int max_iter = p ? p->max_iter : 1000;
    int limit = p ? p->limit : 100;
    int step = p ? p->step : 2;

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

    FoodSource *pop = malloc(sizeof(FoodSource) * pop_size);
    for (int i = 0; i < pop_size; i++) {
        pop[i].bits = calloc(htwui_count, 1);
        initialize_food_source(&pop[i], twu_sum);
        if (pop[i].fitness >= total_min_util) add_to_shui(pop[i].bits, pop[i].fitness);
    }

    uint8_t *trial_v = malloc(htwui_count);
    double *probs = malloc(sizeof(double) * pop_size);

    // ABC Loop
    for (int iter = 0; iter < max_iter; iter++) {
        // 1. Employed Bees
        for (int i = 0; i < pop_size; i++) {
            int k;
            do { k = rand() % pop_size; } while (k == i);
            memcpy(trial_v, pop[i].bits, htwui_count);
            move_towards(trial_v, pop[k].bits, step);
            pev_check(trial_v);
            double f = calculate_utility(trial_v);
            if (f >= total_min_util) add_to_shui(trial_v, f);
            
            if (f > pop[i].fitness) {
                memcpy(pop[i].bits, trial_v, htwui_count);
                pop[i].fitness = f;
                pop[i].trial = 0;
            } else {
                pop[i].trial++;
            }
        }

        // 2. Onlooker Bees
        double total_fit = 0;
        for (int i = 0; i < pop_size; i++) total_fit += pop[i].fitness;
        for (int i = 0; i < pop_size; i++) probs[i] = (total_fit > 0) ? (pop[i].fitness / total_fit) : (1.0 / pop_size);

        for (int m = 0; m < pop_size; m++) {
            double r = (double)rand() / RAND_MAX;
            double s = 0;
            int i = 0;
            for (; i < pop_size - 1; i++) {
                s += probs[i];
                if (s >= r) break;
            }
            
            int k;
            do { k = rand() % pop_size; } while (k == i);
            memcpy(trial_v, pop[i].bits, htwui_count);
            move_towards(trial_v, pop[k].bits, step);
            pev_check(trial_v);
            double f = calculate_utility(trial_v);
            if (f >= total_min_util) add_to_shui(trial_v, f);
            
            if (f > pop[i].fitness) {
                memcpy(pop[i].bits, trial_v, htwui_count);
                pop[i].fitness = f;
                pop[i].trial = 0;
            } else {
                pop[i].trial++;
            }
        }

        // 3. Scout Bees
        for (int i = 0; i < pop_size; i++) {
            if (pop[i].trial >= limit) {
                initialize_food_source(&pop[i], twu_sum);
                if (pop[i].fitness >= total_min_util) add_to_shui(pop[i].bits, pop[i].fitness);
            }
        }
    }

    dm_bench_record_results(ctx.shui_count, 0);

    // Cleanup
    free(trial_v);
    free(probs);
    for (int i = 0; i < pop_size; i++) free(pop[i].bits);
    free(pop);
    for (size_t i = 0; i < htwui_count; i++) { dm_bitset_free(htwuis[i].bitset); free(htwuis[i].utilities); }
    free(htwuis);
    for (size_t i = 0; i < ctx.shui_count; i++) free(ctx.shui[i].items);
    free(ctx.shui);
    
    return DM_SUCCESS;
}

DM_Algorithm huim_abc_algo = {
    .id = "huim_abc",
    .name = "HUIM-ABC",
    .description = "Artificial Bee Colony algorithm for High Utility Itemset Mining (Song & Huang 2018).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(huim_abc_algo)
