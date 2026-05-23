#include "algorithms/prefixspan.h"
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// Represents an occurrence of the prefix in the sequence
typedef struct {
    uint32_t seq_idx;
    uint32_t itemset_idx;
    uint32_t item_idx;
} Occurrence;

typedef struct {
    Occurrence *occs;
    size_t count;
    size_t capacity;
} ProjectedDatabase;

static size_t min_supp_count = 0;
static size_t found_count = 0;

static ProjectedDatabase* create_projected() {
    ProjectedDatabase *pdb = malloc(sizeof(ProjectedDatabase));
    pdb->count = 0;
    pdb->capacity = 16;
    pdb->occs = malloc(sizeof(Occurrence) * pdb->capacity);
    return pdb;
}

static void add_occurrence(ProjectedDatabase *pdb, uint32_t sid, uint32_t tid, uint32_t iid) {
    if (pdb->count >= pdb->capacity) {
        pdb->capacity *= 2;
        pdb->occs = realloc(pdb->occs, sizeof(Occurrence) * pdb->capacity);
    }
    pdb->occs[pdb->count].seq_idx = sid;
    pdb->occs[pdb->count].itemset_idx = tid;
    pdb->occs[pdb->count].item_idx = iid;
    pdb->count++;
}

static void free_projected(ProjectedDatabase *pdb) {
    if (!pdb) return;
    free(pdb->occs);
    free(pdb);
}

// prefix_span_recursive
static void prefix_span_recursive(ProjectedDatabase *pdb, DM_Sequence_Utility *seq_ds, size_t seq_count, uint32_t *prefix, size_t prefix_len, bool last_is_iconcat, uint32_t max_id) {
    
    // Arrays to count support of S-extensions and I-extensions
    uint32_t *s_sup = calloc(max_id + 1, sizeof(uint32_t));
    uint32_t *i_sup = calloc(max_id + 1, sizeof(uint32_t));
    
    uint32_t *s_last_sid = calloc(max_id + 1, sizeof(uint32_t));
    uint32_t *i_last_sid = calloc(max_id + 1, sizeof(uint32_t));
    
    // Initialize last_sid to 0xFFFFFFFF
    for (uint32_t i = 0; i <= max_id; i++) {
        s_last_sid[i] = 0xFFFFFFFF;
        i_last_sid[i] = 0xFFFFFFFF;
    }

    uint32_t current_sid = 0xFFFFFFFF;

    // Scan the projected database
    for (size_t i = 0; i < pdb->count; i++) {
        Occurrence occ = pdb->occs[i];
        DM_Sequence_Utility seq = seq_ds[occ.seq_idx];
        
        // I-extensions: items in the SAME itemset AFTER occ.item_idx
        DM_Trans_Sequence_Utility ts = seq.itemsets[occ.itemset_idx];
        for (size_t j = occ.item_idx + 1; j < ts.count; j++) {
            uint32_t item = ts.items[j].id;
            if (i_last_sid[item] != occ.seq_idx) {
                i_sup[item]++;
                i_last_sid[item] = occ.seq_idx;
            }
        }
        
        // S-extensions: items in ANY itemset AFTER occ.itemset_idx
        // ONLY FOR THE FIRST OCCURRENCE IN EACH SEQUENCE
        if (occ.seq_idx != current_sid) {
            current_sid = occ.seq_idx;
            for (size_t t = occ.itemset_idx + 1; t < seq.count; t++) {
                DM_Trans_Sequence_Utility ts_next = seq.itemsets[t];
                for (size_t j = 0; j < ts_next.count; j++) {
                    uint32_t item = ts_next.items[j].id;
                    if (s_last_sid[item] != occ.seq_idx) {
                        s_sup[item]++;
                        s_last_sid[item] = occ.seq_idx;
                    }
                }
            }
        }
    }
    
    // Next, build projected databases for frequent I-extensions
    for (uint32_t item = 0; item <= max_id; item++) {
        if (i_sup[item] >= min_supp_count) {
            found_count++;
            
            uint32_t next_prefix[prefix_len + 1];
            memcpy(next_prefix, prefix, sizeof(uint32_t) * prefix_len);
            next_prefix[prefix_len] = item;
            
            ProjectedDatabase *next_pdb = create_projected();
            
            for (size_t i = 0; i < pdb->count; i++) {
                Occurrence occ = pdb->occs[i];
                DM_Sequence_Utility seq = seq_ds[occ.seq_idx];
                DM_Trans_Sequence_Utility ts = seq.itemsets[occ.itemset_idx];
                for (size_t j = occ.item_idx + 1; j < ts.count; j++) {
                    if (ts.items[j].id == item) {
                        add_occurrence(next_pdb, occ.seq_idx, occ.itemset_idx, (uint32_t)j);
                        break;
                    }
                }
            }
            prefix_span_recursive(next_pdb, seq_ds, seq_count, next_prefix, prefix_len + 1, true, max_id);
            free_projected(next_pdb);
        }
    }
    
    // Build projected databases for frequent S-extensions
    for (uint32_t item = 0; item <= max_id; item++) {
        if (s_sup[item] >= min_supp_count) {
            found_count++;
            
            uint32_t next_prefix[prefix_len + 1];
            memcpy(next_prefix, prefix, sizeof(uint32_t) * prefix_len);
            next_prefix[prefix_len] = item;
            
            ProjectedDatabase *next_pdb = create_projected();
            
            uint32_t cur_sid = 0xFFFFFFFF;
            for (size_t i = 0; i < pdb->count; i++) {
                Occurrence occ = pdb->occs[i];
                if (occ.seq_idx == cur_sid) continue; // Only first occurrence
                cur_sid = occ.seq_idx;
                
                DM_Sequence_Utility seq = seq_ds[occ.seq_idx];
                for (size_t t = occ.itemset_idx + 1; t < seq.count; t++) {
                    DM_Trans_Sequence_Utility ts_next = seq.itemsets[t];
                    for (size_t j = 0; j < ts_next.count; j++) {
                        if (ts_next.items[j].id == item) {
                            add_occurrence(next_pdb, occ.seq_idx, (uint32_t)t, (uint32_t)j);
                            break;
                        }
                    }
                }
            }
            prefix_span_recursive(next_pdb, seq_ds, seq_count, next_prefix, prefix_len + 1, false, max_id);
            free_projected(next_pdb);
        }
    }
    
    free(s_sup); free(i_sup);
    free(s_last_sid); free(i_last_sid);
}

// run
static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_SEQUENCE_UTILITY && ds->type != DM_TYPE_UTILITY) {
        return DM_ERROR_INCOMPATIBLE;
    }
    
    DM_PrefixSpan_Params *p = (DM_PrefixSpan_Params *)params;
    double minsup_param = p ? p->min_support : 0.01;
    
    if (minsup_param > 0.0 && minsup_param < 1.0) {
        min_supp_count = (size_t)(minsup_param * ds->count);
    } else {
        min_supp_count = (size_t)minsup_param;
    }
    if (min_supp_count == 0) min_supp_count = 1;
    
    found_count = 0;
    
    printf("[PrefixSpan] Mining Sequential Patterns (MinSup: %zu)...\n", min_supp_count);
    
    // Adapt DM_TYPE_UTILITY to DM_TYPE_SEQUENCE_UTILITY
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
    
    uint32_t *sup = calloc(ds->max_id + 1, sizeof(uint32_t));
    uint32_t *last_sid = calloc(ds->max_id + 1, sizeof(uint32_t));
    for (uint32_t i = 0; i <= ds->max_id; i++) last_sid[i] = 0xFFFFFFFF;
    
    for (size_t i = 0; i < seq_count; i++) {
        for (size_t t = 0; t < seq_ds[i].count; t++) {
            for (size_t j = 0; j < seq_ds[i].itemsets[t].count; j++) {
                uint32_t item = seq_ds[i].itemsets[t].items[j].id;
                if (last_sid[item] != i) {
                    sup[item]++;
                    last_sid[item] = i;
                }
            }
        }
    }
    
    for (uint32_t item = 0; item <= ds->max_id; item++) {
        if (sup[item] >= min_supp_count) {
            found_count++;
            
            ProjectedDatabase *pdb = create_projected();
            for (size_t i = 0; i < seq_count; i++) {
                for (size_t t = 0; t < seq_ds[i].count; t++) {
                    for (size_t j = 0; j < seq_ds[i].itemsets[t].count; j++) {
                        if (seq_ds[i].itemsets[t].items[j].id == item) {
                            add_occurrence(pdb, (uint32_t)i, (uint32_t)t, (uint32_t)j);
                            break; // only one instance per itemset
                        }
                    }
                }
            }
            
            uint32_t prefix[1] = {item};
            prefix_span_recursive(pdb, seq_ds, seq_count, prefix, 1, false, ds->max_id);
            free_projected(pdb);
        }
    }
    
    free(sup);
    free(last_sid);
    
    printf("[PrefixSpan] Found %zu Sequential Patterns.\n", found_count);
    
    if (ds->type == DM_TYPE_UTILITY) {
        for (size_t i = 0; i < ds->count; i++) free(seq_ds[i].itemsets);
        free(seq_ds);
    }
    
    dm_bench_record_results(found_count, 0);
    return DM_SUCCESS;
}

DM_Algorithm prefixspan_algo = {
    .id = "prefixspan",
    .name = "PrefixSpan",
    .description = "Prefix-projected Sequential pattern mining",
    .supported_types = (1 << DM_TYPE_UTILITY) | (1 << DM_TYPE_SEQUENCE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(prefixspan_algo)
