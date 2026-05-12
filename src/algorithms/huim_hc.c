#include "algorithms/huim_hc.h"
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
    uint8_t *bits;      // Encoding vector
    double fitness;
    DM_BitSet *ts;      // Bitmap cover (TS)
} Chromosome;

typedef struct {
    uint32_t id;
    double twu;
    DM_BitSet *bitset;  // Column vector in bitmap
    double *utilities;  // Utility of this item in each transaction
} HC_Item;

typedef struct {
    uint32_t *items;
    size_t len;
    double utility;
} HUI_Entry;

/* --- CONTEXT --- */

static HC_Item *htwuis = NULL;
static size_t htwui_count = 0;
static double total_min_util = 0;

static HUI_Entry *shui = NULL;
static size_t shui_count = 0;
static size_t shui_capacity = 0;

/* --- UTILS --- */

static void add_to_shui(uint32_t *items, size_t len, double utility) {
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
        shui_capacity = (shui_capacity == 0) ? 1000 : shui_capacity * 2;
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
    // Check if TS is empty to avoid loop
    bool is_empty = true;
    for (size_t w = 0; w < ts->count; w++) if (ts->bits[w] != 0) { is_empty = false; break; }
    if (is_empty) return 0;

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

// Algorithm 1: Checking PEV (PEVC)
static bool pevc(uint8_t *bits, DM_BitSet *ts, DM_Trans_Utility *src) {
    dm_bitset_set_all(ts);
    bool first = true;
    bool has_bits = false;
    for (size_t i = 0; i < htwui_count; i++) {
        if (bits[i]) {
            has_bits = true;
            if (first) {
                dm_bitset_and(ts, htwuis[i].bitset);
                first = false;
            } else {
                DM_BitSet *temp = dm_bitset_copy(ts);
                dm_bitset_and(temp, htwuis[i].bitset);
                
                bool is_empty = true;
                for (size_t w = 0; w < temp->count; w++) {
                    if (temp->bits[w] != 0) { is_empty = false; break; }
                }

                if (is_empty) {
                    bits[i] = 0;
                    dm_bitset_free(temp);
                } else {
                    dm_bitset_and(ts, htwuis[i].bitset);
                    dm_bitset_free(temp);
                }
            }
        }
    }
    
    if (!has_bits) return false;
    
    // Paper Algorithm 1 Step 4: Check if sum of TWU >= min_util
    double twu_sum = 0;
    for (size_t i = 0; i < ts->size; i++) {
        if (dm_bitset_get(ts, i)) {
            twu_sum += src[i].total_utility;
        }
    }
    
    return (twu_sum >= total_min_util);
}

/* --- LOGIC --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_HUIM_HC_Params *p = (DM_HUIM_HC_Params *)params;
    double min_util_ratio = p ? p->min_utility : 0.01;
    int pop_size = p ? p->pop_size : 30;
    int max_gen = p ? p->max_gen : 10000;

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

    htwuis = malloc(sizeof(HC_Item) * htwui_count);
    size_t h_idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_twu[i] >= total_min_util) {
            htwuis[h_idx].id = i;
            htwuis[h_idx].twu = item_twu[i];
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
    
    // Initial Population (Algorithm 2)
    Chromosome *pop = malloc(sizeof(Chromosome) * pop_size);
    double twu_sum = 0;
    for (size_t i = 0; i < htwui_count; i++) twu_sum += htwuis[i].twu;

    for (int i = 0; i < pop_size; i++) {
        pop[i].bits = calloc(htwui_count, 1);
        pop[i].ts = dm_bitset_create(ds->count);
        int num_ones = (rand() % 5) + 1; // Start small
        for (int j = 0; j < num_ones; j++) {
            double r = ((double)rand() / RAND_MAX) * twu_sum;
            double cur = 0;
            for (size_t k = 0; k < htwui_count; k++) {
                cur += htwuis[k].twu;
                if (cur >= r) { pop[i].bits[k] = 1; break; }
            }
        }
        pevc(pop[i].bits, pop[i].ts, src);
        pop[i].fitness = calculate_utility(pop[i].bits, ds->count, pop[i].ts);
    }

    shui_count = 0;
    shui_capacity = 1000;
    shui = malloc(sizeof(HUI_Entry) * shui_capacity);

    // Mining Loop (Algorithm 3)
    for (int gen = 0; gen < max_gen; gen++) {
        for (int i = 0; i < pop_size; i++) {
            if (pop[i].fitness >= total_min_util) {
                uint32_t *items = malloc(sizeof(uint32_t) * htwui_count);
                size_t len = 0;
                for (size_t j = 0; j < htwui_count; j++) if (pop[i].bits[j]) items[len++] = htwuis[j].id;
                if (len > 0) add_to_shui(items, len, pop[i].fitness);
                free(items);
            }
        }

        // GN(): Get Neighbor (Algorithm 4 logic: move if neighbor is a PEV)
        for (int i = 0; i < pop_size; i++) {
            uint8_t *neighbor_bits = malloc(htwui_count);
            memcpy(neighbor_bits, pop[i].bits, htwui_count);
            DM_BitSet *neighbor_ts = dm_bitset_create(ds->count);
            
            int pos = rand() % htwui_count;
            neighbor_bits[pos] = !neighbor_bits[pos];
            
            if (pevc(neighbor_bits, neighbor_ts, src)) {
                double neighbor_fitness = calculate_utility(neighbor_bits, ds->count, neighbor_ts);
                // Hill Climbing: Move if better or equal (plateau exploration)
                if (neighbor_fitness >= pop[i].fitness) {
                    free(pop[i].bits);
                    dm_bitset_free(pop[i].ts);
                    pop[i].bits = neighbor_bits;
                    pop[i].ts = neighbor_ts;
                    pop[i].fitness = neighbor_fitness;
                } else {
                    free(neighbor_bits);
                    dm_bitset_free(neighbor_ts);
                }
            } else {
                free(neighbor_bits);
                dm_bitset_free(neighbor_ts);
            }
        }

        // Diversity: Replace with discovered HUIs
        if (shui_count > 0 && (rand() % 10 == 0)) {
            int target = rand() % pop_size;
            int h_idx_rand = rand() % shui_count;
            memset(pop[target].bits, 0, htwui_count);
            for (size_t j = 0; j < shui[h_idx_rand].len; j++) {
                for (size_t k = 0; k < htwui_count; k++) {
                    if (htwuis[k].id == shui[h_idx_rand].items[j]) { pop[target].bits[k] = 1; break; }
                }
            }
            pevc(pop[target].bits, pop[target].ts, src);
            pop[target].fitness = calculate_utility(pop[target].bits, ds->count, pop[target].ts);
        }
    }

    dm_bench_record_results(shui_count, 0);

    // Cleanup
    for (size_t i = 0; i < htwui_count; i++) {
        dm_bitset_free(htwuis[i].bitset);
        free(htwuis[i].utilities);
    }
    free(htwuis);
    for (int i = 0; i < pop_size; i++) {
        free(pop[i].bits);
        dm_bitset_free(pop[i].ts);
    }
    free(pop);
    for (size_t i = 0; i < shui_count; i++) free(shui[i].items);
    free(shui);

    return DM_SUCCESS;
}

static DM_Algorithm huim_hc_algo = {
    .id = "huim_hc",
    .name = "HUIM-HC",
    .description = "Mining High-Utility Itemsets with Hill Climbing (Nawaz et al. 2021).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(huim_hc_algo)
