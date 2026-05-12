#include "algorithms/huim_bpso_tree.h"
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
    double *velocity;
    double fitness;
    uint8_t *pbest;
    double pbest_fitness;
} Particle;

typedef struct {
    uint32_t id;
    double twu;
    DM_BitSet *bitset;
    double *utilities; // Utility of this item in each transaction
} HTWUI;

typedef struct {
    uint32_t *items;
    size_t len;
    double utility;
} HUI_Entry;

/* --- CONTEXT --- */

static HTWUI *htwuis = NULL;
static size_t htwui_count = 0;
static double total_min_util = 0;

static HUI_Entry *shui = NULL;
static size_t shui_count = 0;
static size_t shui_capacity = 0;

/* --- UTILS --- */

static double sigmoid(double v) {
    if (v > 20) return 1.0;
    if (v < -20) return 0.0;
    return 1.0 / (1.0 + exp(-v));
}

static void add_to_shui(uint32_t *items, size_t len, double utility) {
    // Basic uniqueness check (can be slow if shui_count is huge)
    // For heuristic miners, we typically find fewer unique HUIs than exact ones.
    for (size_t i = 0; i < shui_count; i++) {
        if (shui[i].len == len) {
            bool match = true;
            for (size_t j = 0; j < len; j++) {
                if (shui[i].items[j] != items[j]) { match = false; break; }
            }
            if (match) return;
        }
    }

    if (shui_count >= shui_capacity) {
        shui_capacity = shui_capacity == 0 ? 1000 : shui_capacity * 2;
        shui = realloc(shui, sizeof(HUI_Entry) * shui_capacity);
    }
    shui[shui_count].items = malloc(sizeof(uint32_t) * len);
    memcpy(shui[shui_count].items, items, sizeof(uint32_t) * len);
    shui[shui_count].len = len;
    shui[shui_count].utility = utility;
    shui_count++;
}

static double calculate_utility(const uint8_t *bits, size_t ds_count, const DM_BitSet *ts) {
    double total_u = 0;
    for (size_t i = 0; i < ds_count; i++) {
        if (dm_bitset_get((DM_BitSet*)ts, i)) {
            double trans_u = 0;
            for (size_t j = 0; j < htwui_count; j++) {
                if (bits[j]) {
                    trans_u += htwuis[j].utilities[i];
                }
            }
            total_u += trans_u;
        }
    }
    return total_u;
}

static bool is_or(size_t j, const DM_BitSet *current_ts) {
    for (size_t i = 0; i < current_ts->count; i++) {
        if (current_ts->bits[i] & htwuis[j].bitset->bits[i]) return true;
    }
    return false;
}

/* --- LOGIC --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_HUIM_BPSO_Tree_Params *p = (DM_HUIM_BPSO_Tree_Params *)params;
    double min_util_ratio = p ? p->min_utility : 0.01;
    int pop_size = p ? p->pop_size : 20;
    int max_iter = p ? p->max_iter : 1000;
    double w_param = p ? p->w : 0.9;
    double c1 = p ? p->c1 : 2.0;
    double c2 = p ? p->c2 : 2.0;

    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    double total_u = 0;
    for (size_t i = 0; i < ds->count; i++) total_u += src[i].total_utility;
    total_min_util = total_u * min_util_ratio;

    double *item_twu = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) {
            item_twu[src[i].items[j].id] += src[i].total_utility;
        }
    }

    htwui_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_twu[i] >= total_min_util) htwui_count++;
    }

    if (htwui_count == 0) { free(item_twu); return DM_SUCCESS; }

    htwuis = malloc(sizeof(HTWUI) * htwui_count);
    size_t h_idx = 0;
    double max_twu = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_twu[i] >= total_min_util) {
            htwuis[h_idx].id = i;
            htwuis[h_idx].twu = item_twu[i];
            if (htwuis[h_idx].twu > max_twu) max_twu = htwuis[h_idx].twu;
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
    Particle *pop = malloc(sizeof(Particle) * pop_size);
    uint8_t *gbest = calloc(htwui_count, 1);
    double gbest_fitness = -1;
    DM_BitSet *ts_buffer = dm_bitset_create(ds->count);

    // Initialization using paper's TWU probability strategy
    for (int i = 0; i < pop_size; i++) {
        pop[i].bits = calloc(htwui_count, 1);
        pop[i].velocity = malloc(sizeof(double) * htwui_count);
        pop[i].pbest = malloc(htwui_count);

        dm_bitset_set_all(ts_buffer);
        // Randomize item order for initialization to avoid early-item bias
        int *order = malloc(sizeof(int) * htwui_count);
        for(size_t j=0; j<htwui_count; j++) order[j] = j;
        for(size_t j=0; j<htwui_count; j++) {
            int r = j + rand() % (htwui_count - j);
            int temp = order[j]; order[j] = order[r]; order[r] = temp;
        }

        for (size_t k = 0; k < htwui_count; k++) {
            size_t j = order[k];
            pop[i].velocity[j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            if (is_or(j, ts_buffer)) {
                // Use relative TWU as probability, with a base explore factor
                double prob = (htwuis[j].twu / max_twu) * 0.2; 
                if ((double)rand() / RAND_MAX < prob) {
                    pop[i].bits[j] = 1;
                    dm_bitset_and(ts_buffer, htwuis[j].bitset);
                }
            }
        }
        free(order);

        pop[i].fitness = calculate_utility(pop[i].bits, ds->count, ts_buffer);
        memcpy(pop[i].pbest, pop[i].bits, htwui_count);
        pop[i].pbest_fitness = pop[i].fitness;
        if (pop[i].fitness > gbest_fitness) {
            gbest_fitness = pop[i].fitness;
            memcpy(gbest, pop[i].bits, htwui_count);
        }
    }

    shui_count = 0;
    shui_capacity = 1000;
    shui = malloc(sizeof(HUI_Entry) * shui_capacity);

    // Main PSO Loop
    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < pop_size; i++) {
            dm_bitset_set_all(ts_buffer);
            // In the update step, we stick to the paper's sorted order to match "OR/NOR-tree" traversal logic
            for (size_t j = 0; j < htwui_count; j++) {
                double r1 = (double)rand() / RAND_MAX;
                double r2 = (double)rand() / RAND_MAX;
                pop[i].velocity[j] = w_param * pop[i].velocity[j] +
                                     c1 * r1 * (pop[i].pbest[j] - pop[i].bits[j]) +
                                     c2 * r2 * (gbest[j] - pop[i].bits[j]);

                // OR/NOR check: Step 24 in Algorithm 1
                if (is_or(j, ts_buffer) && (double)rand() / RAND_MAX < sigmoid(pop[i].velocity[j])) {
                    pop[i].bits[j] = 1;
                    dm_bitset_and(ts_buffer, htwuis[j].bitset);
                } else {
                    pop[i].bits[j] = 0;
                }
            }

            pop[i].fitness = calculate_utility(pop[i].bits, ds->count, ts_buffer);
            if (pop[i].fitness >= total_min_util) {
                uint32_t *items = malloc(sizeof(uint32_t) * htwui_count);
                size_t len = 0;
                for (size_t j = 0; j < htwui_count; j++) if (pop[i].bits[j]) items[len++] = htwuis[j].id;
                if (len > 0) add_to_shui(items, len, pop[i].fitness);
                free(items);
            }

            if (pop[i].fitness > pop[i].pbest_fitness) {
                pop[i].pbest_fitness = pop[i].fitness;
                memcpy(pop[i].pbest, pop[i].bits, htwui_count);
            }
            if (pop[i].fitness > gbest_fitness) {
                gbest_fitness = pop[i].fitness;
                memcpy(gbest, pop[i].bits, htwui_count);
            }
        }
    }

    dm_bench_record_results(shui_count, 0);

    // Cleanup
    dm_bitset_free(ts_buffer);
    for (size_t i = 0; i < htwui_count; i++) {
        dm_bitset_free(htwuis[i].bitset);
        free(htwuis[i].utilities);
    }
    free(htwuis);
    for (int i = 0; i < pop_size; i++) {
        free(pop[i].bits);
        free(pop[i].velocity);
        free(pop[i].pbest);
    }
    free(pop);
    free(gbest);
    for (size_t i = 0; i < shui_count; i++) free(shui[i].items);
    free(shui);
    return DM_SUCCESS;
}

static DM_Algorithm huim_bpso_tree_algo = {
    .id = "huim_bpso_tree",
    .name = "HUIM-BPSO-tree",
    .description = "Mining High-Utility Itemsets with Binary PSO and OR/NOR-tree (Lin et al. 2016).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(huim_bpso_tree_algo)
