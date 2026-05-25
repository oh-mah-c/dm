#include "algorithms/hupspm.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t seq_idx;
    uint32_t itemset_idx;
    uint32_t item_idx;
    double utility;
} H_Occurrence;

typedef struct {
    H_Occurrence *occs;
    size_t count;
    size_t capacity;
} H_ProjectedDatabase;

/* --- CONTEXT --- */

static double min_util = 0;
static double min_prob_count = 0;
static size_t found_count = 0;

/* --- UTILS --- */

static H_ProjectedDatabase* create_projected() {
    H_ProjectedDatabase *pdb = malloc(sizeof(H_ProjectedDatabase));
    pdb->count = 0;
    pdb->capacity = 16;
    pdb->occs = malloc(sizeof(H_Occurrence) * pdb->capacity);
    return pdb;
}

static void add_occurrence(H_ProjectedDatabase *pdb, uint32_t sid, uint32_t tid, uint32_t iid, double util) {
    if (pdb->count >= pdb->capacity) {
        pdb->capacity *= 2;
        pdb->occs = realloc(pdb->occs, sizeof(H_Occurrence) * pdb->capacity);
    }
    pdb->occs[pdb->count].seq_idx = sid;
    pdb->occs[pdb->count].itemset_idx = tid;
    pdb->occs[pdb->count].item_idx = iid;
    pdb->occs[pdb->count].utility = util;
    pdb->count++;
}

static void free_projected(H_ProjectedDatabase *pdb) {
    if (!pdb) return;
    free(pdb->occs);
    free(pdb);
}

/* --- LOGIC --- */

static void hupspm_recursive(H_ProjectedDatabase *pdb, DM_Sequence_Utility *ds_payload, size_t ds_count, uint32_t *prefix, size_t prefix_len, uint32_t max_id) {
    
    // I-Concatenations
    for (uint32_t item = 0; item <= max_id; item++) {
        H_ProjectedDatabase *next_pdb = create_projected();
        double swu = 0;
        double prob_sum = 0;
        uint32_t last_sid_swu = 0xFFFFFFFF;
        uint32_t last_sid_prob = 0xFFFFFFFF;
        
        for (size_t i = 0; i < pdb->count; i++) {
            H_Occurrence occ = pdb->occs[i];
            DM_Sequence_Utility seq = ds_payload[occ.seq_idx];
            DM_Trans_Sequence_Utility ts = seq.itemsets[occ.itemset_idx];
            
            for (size_t j = occ.item_idx + 1; j < ts.count; j++) {
                if (ts.items[j].id == item) {
                    double new_util = occ.utility + ts.items[j].utility;
                    add_occurrence(next_pdb, occ.seq_idx, occ.itemset_idx, (uint32_t)j, new_util);
                    if (occ.seq_idx != last_sid_swu) {
                        swu += seq.total_utility;
                        last_sid_swu = occ.seq_idx;
                    }
                    if (occ.seq_idx != last_sid_prob) {
                        prob_sum += seq.probability;
                        last_sid_prob = occ.seq_idx;
                    }
                    break;
                }
            }
        }
        
        if (swu >= min_util && prob_sum >= min_prob_count) {
            // Check actual utility for output
            double actual_u = 0;
            uint32_t current_sid = 0xFFFFFFFF;
            double sid_max = 0;
            for (size_t k = 0; k < next_pdb->count; k++) {
                if (next_pdb->occs[k].seq_idx != current_sid) {
                    actual_u += sid_max;
                    current_sid = next_pdb->occs[k].seq_idx;
                    sid_max = next_pdb->occs[k].utility;
                } else if (next_pdb->occs[k].utility > sid_max) sid_max = next_pdb->occs[k].utility;
            }
            actual_u += sid_max;
            
            if (actual_u >= min_util) found_count++;
            
            uint32_t *next_prefix = malloc(sizeof(uint32_t) * (prefix_len + 1));
            if (next_prefix) {
                memcpy(next_prefix, prefix, sizeof(uint32_t) * prefix_len);
                next_prefix[prefix_len] = item;
                hupspm_recursive(next_pdb, ds_payload, ds_count, next_prefix, prefix_len + 1, max_id);
                free(next_prefix);
            }
        }
        free_projected(next_pdb);
    }
    
    // S-Concatenations
    for (uint32_t item = 0; item <= max_id; item++) {
        H_ProjectedDatabase *next_pdb = create_projected();
        double swu = 0;
        double prob_sum = 0;
        uint32_t last_sid_swu = 0xFFFFFFFF;
        uint32_t last_sid_prob = 0xFFFFFFFF;
        
        for (size_t i = 0; i < pdb->count; i++) {
            H_Occurrence occ = pdb->occs[i];
            DM_Sequence_Utility seq = ds_payload[occ.seq_idx];
            
            bool found_in_sid = false;
            for (size_t t = occ.itemset_idx + 1; t < seq.count; t++) {
                DM_Trans_Sequence_Utility ts = seq.itemsets[t];
                for (size_t j = 0; j < ts.count; j++) {
                    if (ts.items[j].id == item) {
                        double new_util = occ.utility + ts.items[j].utility;
                        add_occurrence(next_pdb, occ.seq_idx, (uint32_t)t, (uint32_t)j, new_util);
                        found_in_sid = true;
                        break; 
                    }
                }
            }
            if (found_in_sid) {
                if (occ.seq_idx != last_sid_swu) {
                    swu += seq.total_utility;
                    last_sid_swu = occ.seq_idx;
                }
                if (occ.seq_idx != last_sid_prob) {
                    prob_sum += seq.probability;
                    last_sid_prob = occ.seq_idx;
                }
            }
        }
        
        if (swu >= min_util && prob_sum >= min_prob_count) {
            double actual_u = 0;
            uint32_t current_sid = 0xFFFFFFFF;
            double sid_max = 0;
            for (size_t k = 0; k < next_pdb->count; k++) {
                if (next_pdb->occs[k].seq_idx != current_sid) {
                    actual_u += sid_max;
                    current_sid = next_pdb->occs[k].seq_idx;
                    sid_max = next_pdb->occs[k].utility;
                } else if (next_pdb->occs[k].utility > sid_max) sid_max = next_pdb->occs[k].utility;
            }
            actual_u += sid_max;
            
            if (actual_u >= min_util) found_count++;
            
            uint32_t *next_prefix = malloc(sizeof(uint32_t) * (prefix_len + 1));
            if (next_prefix) {
                memcpy(next_prefix, prefix, sizeof(uint32_t) * prefix_len);
                next_prefix[prefix_len] = item;
                hupspm_recursive(next_pdb, ds_payload, ds_count, next_prefix, prefix_len + 1, max_id);
                free(next_prefix);
            }
        }
        free_projected(next_pdb);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_SEQUENCE_UTILITY && ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_HUPSPM_Params *p = (DM_HUPSPM_Params *)params;
    min_util = p ? p->min_utility : 100.0;
    double min_prob = p ? p->min_probability : 0.5;
    min_prob_count = ds->count * min_prob;
    found_count = 0;
    
    printf("[HUPSPM] Mining High Utility-Probability Sequential Patterns (MinUtil: %.2f, MinProb: %.2f)...\n", min_util, min_prob);
    
    DM_Sequence_Utility *seq_ds;
    size_t seq_count = ds->count;
    
    if (ds->type == DM_TYPE_UTILITY) {
        DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
        seq_ds = malloc(sizeof(DM_Sequence_Utility) * ds->count);
        for (size_t i = 0; i < ds->count; i++) {
            seq_ds[i].count = 1;
            seq_ds[i].total_utility = src[i].total_utility;
            seq_ds[i].probability = 1.0; // Default for non-uncertain data
            seq_ds[i].itemsets = malloc(sizeof(DM_Trans_Sequence_Utility));
            seq_ds[i].itemsets[0].count = src[i].count;
            seq_ds[i].itemsets[0].items = src[i].items;
        }
    } else {
        seq_ds = (DM_Sequence_Utility *)ds->payload;
    }
    
    // Initial 1-sequences
    for (uint32_t item = 0; item <= ds->max_id; item++) {
        H_ProjectedDatabase *pdb = create_projected();
        double swu = 0;
        double prob_sum = 0;
        uint32_t last_sid_swu = 0xFFFFFFFF;
        uint32_t last_sid_prob = 0xFFFFFFFF;
        
        for (size_t i = 0; i < seq_count; i++) {
            bool found_in_seq = false;
            for (size_t t = 0; t < seq_ds[i].count; t++) {
                for (size_t j = 0; j < seq_ds[i].itemsets[t].count; j++) {
                    if (seq_ds[i].itemsets[t].items[j].id == item) {
                        add_occurrence(pdb, (uint32_t)i, (uint32_t)t, (uint32_t)j, seq_ds[i].itemsets[t].items[j].utility);
                        found_in_seq = true;
                    }
                }
            }
            if (found_in_seq) {
                swu += seq_ds[i].total_utility;
                prob_sum += seq_ds[i].probability;
            }
        }
        
        if (swu >= min_util && prob_sum >= min_prob_count) {
            double actual_u = 0;
            uint32_t current_sid = 0xFFFFFFFF;
            double sid_max = 0;
            for (size_t k = 0; k < pdb->count; k++) {
                if (pdb->occs[k].seq_idx != current_sid) {
                    actual_u += sid_max;
                    current_sid = pdb->occs[k].seq_idx;
                    sid_max = pdb->occs[k].utility;
                } else if (pdb->occs[k].utility > sid_max) sid_max = pdb->occs[k].utility;
            }
            actual_u += sid_max;
            
            if (actual_u >= min_util) found_count++;
            
            uint32_t prefix[1] = {item};
            hupspm_recursive(pdb, seq_ds, seq_count, prefix, 1, ds->max_id);
        }
        free_projected(pdb);
    }
    
    printf("[HUPSPM] Found %zu High Utility-Probability Sequential Patterns.\n", found_count);
    
    if (ds->type == DM_TYPE_UTILITY) {
        for (size_t i = 0; i < ds->count; i++) free(seq_ds[i].itemsets);
        free(seq_ds);
    }
    
    dm_bench_record_results(found_count, 0);
    return DM_SUCCESS;
}

static DM_Algorithm hupspm_algo = {
    .id = "hupspm",
    .name = "HUPSPM",
    .description = "High Utility-Probability Sequential Pattern Mining from Uncertain Databases.",
    .supported_types = (1 << DM_TYPE_UTILITY) | (1 << DM_TYPE_SEQUENCE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(hupspm_algo)
