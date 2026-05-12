#include "algorithms/uspan.h"
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
} Occurrence;

typedef struct {
    Occurrence *occs;
    size_t count;
    size_t capacity;
} ProjectedDatabase;

/* --- CONTEXT --- */

static double min_util = 0;
static size_t found_count = 0;

/* --- UTILS --- */

static ProjectedDatabase* create_projected() {
    ProjectedDatabase *pdb = malloc(sizeof(ProjectedDatabase));
    pdb->count = 0;
    pdb->capacity = 16;
    pdb->occs = malloc(sizeof(Occurrence) * pdb->capacity);
    return pdb;
}

static void add_occurrence(ProjectedDatabase *pdb, uint32_t sid, uint32_t tid, uint32_t iid, double util) {
    if (pdb->count >= pdb->capacity) {
        pdb->capacity *= 2;
        pdb->occs = realloc(pdb->occs, sizeof(Occurrence) * pdb->capacity);
    }
    pdb->occs[pdb->count].seq_idx = sid;
    pdb->occs[pdb->count].itemset_idx = tid;
    pdb->occs[pdb->count].item_idx = iid;
    pdb->occs[pdb->count].utility = util;
    pdb->count++;
}

static void free_projected(ProjectedDatabase *pdb) {
    if (!pdb) return;
    free(pdb->occs);
    free(pdb);
}

/* --- USPAN LOGIC --- */

static void uspan_recursive(ProjectedDatabase *pdb, DM_Sequence_Utility *ds_payload, size_t ds_count, uint32_t *prefix, size_t prefix_len, bool last_is_iconcat, uint32_t max_id) {
    
    // Width Pruning (SWU) is typically done during candidate generation.
    // Here we'll generate all possible I-Concatenations and S-Concatenations.
    
    // 1. I-Concatenations (Add to the same itemset)
    for (uint32_t item = 0; item <= max_id; item++) {
        ProjectedDatabase *next_pdb = create_projected();
        double swu = 0;
        uint32_t last_sid = 0xFFFFFFFF;
        
        for (size_t i = 0; i < pdb->count; i++) {
            Occurrence occ = pdb->occs[i];
            DM_Sequence_Utility seq = ds_payload[occ.seq_idx];
            DM_Trans_Sequence_Utility ts = seq.itemsets[occ.itemset_idx];
            
            // Only items in the same itemset AFTER occ.item_idx
            for (size_t j = occ.item_idx + 1; j < ts.count; j++) {
                if (ts.items[j].id == item) {
                    double new_util = occ.utility + ts.items[j].utility;
                    add_occurrence(next_pdb, occ.seq_idx, occ.itemset_idx, (uint32_t)j, new_util);
                    if (occ.seq_idx != last_sid) {
                        swu += seq.total_utility; // SWU bound
                        last_sid = occ.seq_idx;
                    }
                    break;
                }
            }
        }
        
        if (swu >= min_util) {
            uint32_t next_prefix[prefix_len + 1];
            memcpy(next_prefix, prefix, sizeof(uint32_t) * prefix_len);
            next_prefix[prefix_len] = item;
            
            // Depth Pruning: Theorem 2
            double max_potential = 0;
            uint32_t cur_sid = 0xFFFFFFFF;
            double sid_pot = 0;
            for (size_t k = 0; k < next_pdb->count; k++) {
                Occurrence occ = next_pdb->occs[k];
                double pot = occ.utility;
                for (size_t t = occ.itemset_idx + 1; t < ds_payload[occ.seq_idx].count; t++) {
                    for (size_t j = 0; j < ds_payload[occ.seq_idx].itemsets[t].count; j++) {
                        pot += ds_payload[occ.seq_idx].itemsets[t].items[j].utility;
                    }
                }
                for (size_t j = occ.item_idx + 1; j < ds_payload[occ.seq_idx].itemsets[occ.itemset_idx].count; j++) {
                    pot += ds_payload[occ.seq_idx].itemsets[occ.itemset_idx].items[j].utility;
                }
                if (occ.seq_idx != cur_sid) {
                    max_potential += sid_pot;
                    cur_sid = occ.seq_idx;
                    sid_pot = pot;
                } else if (pot > sid_pot) sid_pot = pot;
            }
            max_potential += sid_pot;

            if (max_potential >= min_util) {
                // Calculate max utility of this sequence across the database
                double max_u = 0;
                uint32_t current_sid = 0xFFFFFFFF;
                double sid_max = 0;
                for (size_t k = 0; k < next_pdb->count; k++) {
                    if (next_pdb->occs[k].seq_idx != current_sid) {
                        max_u += sid_max;
                        current_sid = next_pdb->occs[k].seq_idx;
                        sid_max = next_pdb->occs[k].utility;
                    } else if (next_pdb->occs[k].utility > sid_max) sid_max = next_pdb->occs[k].utility;
                }
                max_u += sid_max;
                
                if (max_u >= min_util) found_count++;
                
                uspan_recursive(next_pdb, ds_payload, ds_count, next_prefix, prefix_len + 1, true, max_id);
            }
        }
        free_projected(next_pdb);
    }
    
    // 2. S-Concatenations (Add to a new itemset)
    for (uint32_t item = 0; item <= max_id; item++) {
        ProjectedDatabase *next_pdb = create_projected();
        double swu = 0;
        uint32_t last_sid = 0xFFFFFFFF;
        
        for (size_t i = 0; i < pdb->count; i++) {
            Occurrence occ = pdb->occs[i];
            DM_Sequence_Utility seq = ds_payload[occ.seq_idx];
            
            // Items in ANY itemset AFTER occ.itemset_idx
            bool found_in_sid = false;
            for (size_t t = occ.itemset_idx + 1; t < seq.count; t++) {
                DM_Trans_Sequence_Utility ts = seq.itemsets[t];
                for (size_t j = 0; j < ts.count; j++) {
                    if (ts.items[j].id == item) {
                        double new_util = occ.utility + ts.items[j].utility;
                        add_occurrence(next_pdb, occ.seq_idx, (uint32_t)t, (uint32_t)j, new_util);
                        found_in_sid = true;
                        // For a SID, we can have multiple occurrences, but SWU is added once per SID
                        break; 
                    }
                }
            }
            if (found_in_sid && occ.seq_idx != last_sid) {
                swu += seq.total_utility;
                last_sid = occ.seq_idx;
            }
        }
        
        if (swu >= min_util) {
            uint32_t next_prefix[prefix_len + 1];
            memcpy(next_prefix, prefix, sizeof(uint32_t) * prefix_len);
            next_prefix[prefix_len] = item;
            
            // Depth Pruning: Theorem 2
            double max_potential = 0;
            uint32_t cur_sid = 0xFFFFFFFF;
            double sid_pot = 0;
            for (size_t k = 0; k < next_pdb->count; k++) {
                Occurrence occ = next_pdb->occs[k];
                double pot = occ.utility;
                for (size_t t = occ.itemset_idx + 1; t < ds_payload[occ.seq_idx].count; t++) {
                    for (size_t j = 0; j < ds_payload[occ.seq_idx].itemsets[t].count; j++) {
                        pot += ds_payload[occ.seq_idx].itemsets[t].items[j].utility;
                    }
                }
                for (size_t j = occ.item_idx + 1; j < ds_payload[occ.seq_idx].itemsets[occ.itemset_idx].count; j++) {
                    pot += ds_payload[occ.seq_idx].itemsets[occ.itemset_idx].items[j].utility;
                }
                if (occ.seq_idx != cur_sid) {
                    max_potential += sid_pot;
                    cur_sid = occ.seq_idx;
                    sid_pot = pot;
                } else if (pot > sid_pot) sid_pot = pot;
            }
            max_potential += sid_pot;

            if (max_potential >= min_util) {
                double max_u = 0;
                uint32_t current_sid = 0xFFFFFFFF;
                double sid_max = 0;
                for (size_t k = 0; k < next_pdb->count; k++) {
                    if (next_pdb->occs[k].seq_idx != current_sid) {
                        max_u += sid_max;
                        current_sid = next_pdb->occs[k].seq_idx;
                        sid_max = next_pdb->occs[k].utility;
                    } else if (next_pdb->occs[k].utility > sid_max) sid_max = next_pdb->occs[k].utility;
                }
                max_u += sid_max;
                
                if (max_u >= min_util) found_count++;
                
                uspan_recursive(next_pdb, ds_payload, ds_count, next_prefix, prefix_len + 1, false, max_id);
            }
        }
        free_projected(next_pdb);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_SEQUENCE_UTILITY) {
        // For testing, let's allow DM_TYPE_UTILITY and treat each transaction as a single-itemset sequence
        if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    }
    
    DM_USPAN_Params *p = (DM_USPAN_Params *)params;
    min_util = p ? p->min_utility : 100.0;
    found_count = 0;
    
    printf("[USpan] Mining High Utility Sequential Patterns (MinUtil: %.2f)...\n", min_util);
    
    // Adapt DM_TYPE_UTILITY to DM_TYPE_SEQUENCE_UTILITY internally if needed
    DM_Sequence_Utility *seq_ds;
    size_t seq_count = ds->count;
    
    if (ds->type == DM_TYPE_UTILITY) {
        DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
        seq_ds = malloc(sizeof(DM_Sequence_Utility) * ds->count);
        for (size_t i = 0; i < ds->count; i++) {
            seq_ds[i].count = 1;
            seq_ds[i].total_utility = src[i].total_utility;
            seq_ds[i].itemsets = malloc(sizeof(DM_Trans_Sequence_Utility));
            seq_ds[i].itemsets[0].count = src[i].count;
            seq_ds[i].itemsets[0].items = src[i].items;
        }
    } else {
        seq_ds = (DM_Sequence_Utility *)ds->payload;
    }
    
    // Initial occurrences (1-sequences)
    for (uint32_t item = 0; item <= ds->max_id; item++) {
        ProjectedDatabase *pdb = create_projected();
        double swu = 0;
        uint32_t last_sid = 0xFFFFFFFF;
        
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
            if (found_in_seq) swu += seq_ds[i].total_utility;
        }
        
        if (swu >= min_util) {
            // Depth Pruning: Theorem 2 (Remaining Utility)
            double max_potential = 0;
            uint32_t cur_sid = 0xFFFFFFFF;
            double sid_pot = 0;
            for (size_t k = 0; k < pdb->count; k++) {
                Occurrence occ = pdb->occs[k];
                // Potential = current utility + sum of all utilities in remaining itemsets
                double pot = occ.utility;
                for (size_t t = occ.itemset_idx + 1; t < seq_ds[occ.seq_idx].count; t++) {
                    for (size_t j = 0; j < seq_ds[occ.seq_idx].itemsets[t].count; j++) {
                        pot += seq_ds[occ.seq_idx].itemsets[t].items[j].utility;
                    }
                }
                // Add remaining items in SAME itemset
                for (size_t j = occ.item_idx + 1; j < seq_ds[occ.seq_idx].itemsets[occ.itemset_idx].count; j++) {
                    pot += seq_ds[occ.seq_idx].itemsets[occ.itemset_idx].items[j].utility;
                }

                if (occ.seq_idx != cur_sid) {
                    max_potential += sid_pot;
                    cur_sid = occ.seq_idx;
                    sid_pot = pot;
                } else if (pot > sid_pot) sid_pot = pot;
            }
            max_potential += sid_pot;

            if (max_potential >= min_util) {
                double max_u = 0;
                uint32_t current_sid = 0xFFFFFFFF;
                double sid_max = 0;
                for (size_t k = 0; k < pdb->count; k++) {
                    if (pdb->occs[k].seq_idx != current_sid) {
                        max_u += sid_max;
                        current_sid = pdb->occs[k].seq_idx;
                        sid_max = pdb->occs[k].utility;
                    } else if (pdb->occs[k].utility > sid_max) sid_max = pdb->occs[k].utility;
                }
                max_u += sid_max;
                
                if (max_u >= min_util) found_count++;
                
                uint32_t prefix[1] = {item};
                uspan_recursive(pdb, seq_ds, seq_count, prefix, 1, false, ds->max_id);
            }
        }
        free_projected(pdb);
    }
    
    printf("[USpan] Found %zu High Utility Sequential Patterns.\n", found_count);
    
    if (ds->type == DM_TYPE_UTILITY) {
        for (size_t i = 0; i < ds->count; i++) free(seq_ds[i].itemsets);
        free(seq_ds);
    }
    
    dm_bench_record_results(found_count, 0);
    return DM_SUCCESS;
}

static DM_Algorithm uspan_algo = {
    .id = "uspan",
    .name = "USpan",
    .description = "Mining High Utility Sequential Patterns using LQS-Tree.",
    .supported_types = (1 << DM_TYPE_UTILITY) | (1 << DM_TYPE_SEQUENCE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(uspan_algo)
