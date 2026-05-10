#include "algorithms/krimp.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

/* --- KRIMP DATA STRUCTURES --- */

typedef struct {
    uint32_t *items;
    size_t count;
    uint32_t support;
    uint32_t usage;
    double code_length;
} Krimp_Itemset;

typedef struct {
    Krimp_Itemset *itemsets;
    size_t count;
    size_t capacity;
} Krimp_List;

/* --- UTILS --- */

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

static int cmp_lexicographical(const uint32_t *a, size_t a_len, const uint32_t *b, size_t b_len) {
    size_t min_len = a_len < b_len ? a_len : b_len;
    for (size_t i = 0; i < min_len; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    if (a_len < b_len) return -1;
    if (a_len > b_len) return 1;
    return 0;
}

// Standard Cover Order: |X|↓ supp(X)↓ lexicographically↑
static int cmp_cover_order(const void *a, const void *b) {
    const Krimp_Itemset *ia = (const Krimp_Itemset *)a;
    const Krimp_Itemset *ib = (const Krimp_Itemset *)b;
    
    if (ia->count > ib->count) return -1;
    if (ia->count < ib->count) return 1;
    
    if (ia->support > ib->support) return -1;
    if (ia->support < ib->support) return 1;
    
    return cmp_lexicographical(ia->items, ia->count, ib->items, ib->count);
}

// Standard Candidate Order: supp(X)↓ |X|↓ lexicographically↑
static int cmp_candidate_order(const void *a, const void *b) {
    const Krimp_Itemset *ia = (const Krimp_Itemset *)a;
    const Krimp_Itemset *ib = (const Krimp_Itemset *)b;
    
    if (ia->support > ib->support) return -1;
    if (ia->support < ib->support) return 1;
    
    if (ia->count > ib->count) return -1;
    if (ia->count < ib->count) return 1;
    
    return cmp_lexicographical(ia->items, ia->count, ib->items, ib->count);
}

static bool is_subset(const uint32_t *sub, size_t sub_len, const uint32_t *set, size_t set_len) {
    if (sub_len > set_len) return false;
    size_t i = 0, j = 0;
    while (i < sub_len && j < set_len) {
        if (sub[i] == set[j]) { i++; j++; }
        else if (sub[i] > set[j]) j++;
        else return false;
    }
    return i == sub_len;
}

static void remove_subset(uint32_t *set, size_t *set_len, const uint32_t *sub, size_t sub_len) {
    uint32_t *new_set = (uint32_t*)malloc(sizeof(uint32_t) * (*set_len));
    size_t new_len = 0;
    size_t si = 0;
    for (size_t i = 0; i < *set_len; i++) {
        if (si < sub_len && set[i] == sub[si]) {
            si++;
        } else {
            new_set[new_len++] = set[i];
        }
    }
    memcpy(set, new_set, sizeof(uint32_t) * new_len);
    *set_len = new_len;
    free(new_set);
}

/* --- KRIMP LOGIC --- */

static void calculate_usage(Krimp_List *ct, DM_Dataset *ds) {
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ct->count; i++) ct->itemsets[i].usage = 0;
    
    uint32_t *temp_items = (uint32_t*)malloc(sizeof(uint32_t) * ds->max_id + 100);

    for (size_t i = 0; i < ds->count; i++) {
        size_t current_len = data[i].count;
        memcpy(temp_items, data[i].items, sizeof(uint32_t) * current_len);
        
        for (size_t j = 0; j < ct->count; j++) {
            if (current_len == 0) break;
            if (is_subset(ct->itemsets[j].items, ct->itemsets[j].count, temp_items, current_len)) {
                ct->itemsets[j].usage++;
                remove_subset(temp_items, &current_len, ct->itemsets[j].items, ct->itemsets[j].count);
                // Standard Cover continues to find next itemsets in the SAME transaction.
                // Re-scan from current j is not needed because itemsets are disjoint.
                // But we must stay at current j if it could cover more? 
                // No, "disjoint set of elements... An itemset X is included... X is removed from t and the process continues".
                // Actually, Algorithm 2 is recursive. It finds the FIRST match, then recurses on the remainder.
                // So we should restart the loop from j=0? 
                // Paper says: "For a given transaction t, the code table is traversed in a fixed order."
                // "An itemset X is included... iff X subset t. Then X is removed from t and the process continues to cover the uncovered remainder."
                // This means we should re-scan the CT from the beginning for the remainder?
                // Usually KRIMP implementations just continue the loop because the CT is sorted.
                // If a later itemset matches, an earlier one would have matched if it were possible.
                // Actually, let's follow Algorithm 2 exactly: "StandardCover(t\S, CT)".
                // This means we start from the beginning of CT again.
                j = -1; // Reset loop
            }
        }
    }
    free(temp_items);
}

static double calculate_mdl_score(Krimp_List *ct, DM_Dataset *ds, Krimp_List *st) {
    uint64_t total_usage = 0;
    for (size_t i = 0; i < ct->count; i++) total_usage += ct->itemsets[i].usage;
    
    if (total_usage == 0) return 1e18;

    double l_d_ct = 0;
    for (size_t i = 0; i < ct->count; i++) {
        if (ct->itemsets[i].usage > 0) {
            ct->itemsets[i].code_length = -log2((double)ct->itemsets[i].usage / total_usage);
            l_d_ct += ct->itemsets[i].usage * ct->itemsets[i].code_length;
        } else {
            ct->itemsets[i].code_length = 0;
        }
    }

    double l_ct_d = 0;
    for (size_t i = 0; i < ct->count; i++) {
        if (ct->itemsets[i].usage == 0) continue;
        
        // L(code_CT(X))
        l_ct_d += ct->itemsets[i].code_length;
        
        // L(code_ST(X))
        double l_st_x = 0;
        for (size_t j = 0; j < ct->itemsets[i].count; j++) {
            uint32_t item = ct->itemsets[i].items[j];
            // Find item in ST
            for (size_t k = 0; k < st->count; k++) {
                if (st->itemsets[k].count == 1 && st->itemsets[k].items[0] == item) {
                    l_st_x += st->itemsets[k].code_length;
                    break;
                }
            }
        }
        l_ct_d += l_st_x;
    }

    return l_d_ct + l_ct_d;
}

/* --- CANDIDATE GENERATION (APRIORI) --- */

static void generate_candidates(DM_Dataset *ds, double min_sup_val, Krimp_List *out) {
    // Simple Apriori to get candidates
    uint32_t min_sup = (uint32_t)ceil(min_sup_val * ds->count);
    if (min_sup == 0) min_sup = 1;

    uint32_t *counts = (uint32_t*)calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }

    out->count = 0;
    out->capacity = 1024;
    out->itemsets = (Krimp_Itemset*)malloc(sizeof(Krimp_Itemset) * out->capacity);

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            Krimp_Itemset *it = &out->itemsets[out->count++];
            it->count = 1;
            it->items = (uint32_t*)malloc(sizeof(uint32_t));
            it->items[0] = i;
            it->support = counts[i];
            it->usage = 0;
        }
    }
    free(counts);

    // For KRIMP, we usually want candidates of length > 1 as well.
    // I'll implement a simple 2-itemset generation for demonstration.
    // In a real scenario, we'd use a full miner.
    size_t l1_count = out->count;
    for (size_t i = 0; i < l1_count; i++) {
        for (size_t j = i + 1; j < l1_count; j++) {
            uint32_t pair[2] = {out->itemsets[i].items[0], out->itemsets[j].items[0]};
            if (pair[0] > pair[1]) { uint32_t t = pair[0]; pair[0] = pair[1]; pair[1] = t; }
            
            uint32_t sup = 0;
            for (size_t k = 0; k < ds->count; k++) {
                if (is_subset(pair, 2, data[k].items, data[k].count)) sup++;
            }
            
            if (sup >= min_sup) {
                if (out->count >= out->capacity) {
                    out->capacity *= 2;
                    out->itemsets = (Krimp_Itemset*)realloc(out->itemsets, sizeof(Krimp_Itemset) * out->capacity);
                }
                Krimp_Itemset *it = &out->itemsets[out->count++];
                it->count = 2;
                it->items = (uint32_t*)malloc(sizeof(uint32_t) * 2);
                it->items[0] = pair[0];
                it->items[1] = pair[1];
                it->support = sup;
                it->usage = 0;
            }
        }
    }
}

/* --- MAIN RUN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_KRIMP_Params *p = (DM_KRIMP_Params *)params;
    double min_sup_val = p ? p->min_support : 0.1;
    bool prune = p ? p->prune : true;

    printf("[KRIMP] Starting. Min Support for Candidates: %.2f, Pruning: %s\n", min_sup_val, prune ? "ON" : "OFF");

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) qsort(data[i].items, data[i].count, sizeof(uint32_t), cmp_uint32);

    // 1. Standard Code Table (ST)
    Krimp_List st = {NULL, 0, 0};
    st.capacity = ds->max_id + 1;
    st.itemsets = (Krimp_Itemset*)malloc(sizeof(Krimp_Itemset) * st.capacity);
    uint32_t *counts = (uint32_t*)calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] > 0) {
            st.itemsets[st.count].items = (uint32_t*)malloc(sizeof(uint32_t));
            st.itemsets[st.count].items[0] = i;
            st.itemsets[st.count].count = 1;
            st.itemsets[st.count].support = counts[i];
            st.itemsets[st.count].usage = counts[i];
            st.count++;
        }
    }
    free(counts);
    
    uint64_t total_st_usage = 0;
    for (size_t i = 0; i < st.count; i++) total_st_usage += st.itemsets[i].usage;
    for (size_t i = 0; i < st.count; i++) {
        st.itemsets[i].code_length = -log2((double)st.itemsets[i].usage / total_st_usage);
    }

    // 2. Candidate Generation
    Krimp_List candidates = {NULL, 0, 0};
    generate_candidates(ds, min_sup_val, &candidates);
    qsort(candidates.itemsets, candidates.count, sizeof(Krimp_Itemset), cmp_candidate_order);
    printf("[KRIMP] Generated %zu candidates.\n", candidates.count);

    // 3. Main Loop
    Krimp_List ct = {NULL, 0, 0};
    ct.capacity = st.count + 10;
    ct.itemsets = (Krimp_Itemset*)malloc(sizeof(Krimp_Itemset) * ct.capacity);
    for (size_t i = 0; i < st.count; i++) {
        ct.itemsets[i].items = (uint32_t*)malloc(sizeof(uint32_t));
        ct.itemsets[i].items[0] = st.itemsets[i].items[0];
        ct.itemsets[i].count = 1;
        ct.itemsets[i].support = st.itemsets[i].support;
        ct.itemsets[i].usage = st.itemsets[i].usage;
        ct.count++;
    }
    qsort(ct.itemsets, ct.count, sizeof(Krimp_Itemset), cmp_cover_order);

    double current_score = calculate_mdl_score(&ct, ds, &st);
    printf("[KRIMP] Initial Score: %.2f bits\n", current_score);

    for (size_t i = 0; i < candidates.count; i++) {
        if (candidates.itemsets[i].count <= 1) continue; // Skip singletons already in CT
        
        // Add candidate to CT
        if (ct.count >= ct.capacity) {
            ct.capacity *= 2;
            ct.itemsets = (Krimp_Itemset*)realloc(ct.itemsets, sizeof(Krimp_Itemset) * ct.capacity);
        }
        Krimp_Itemset *new_it = &ct.itemsets[ct.count++];
        new_it->items = (uint32_t*)malloc(sizeof(uint32_t) * candidates.itemsets[i].count);
        memcpy(new_it->items, candidates.itemsets[i].items, sizeof(uint32_t) * candidates.itemsets[i].count);
        new_it->count = candidates.itemsets[i].count;
        new_it->support = candidates.itemsets[i].support;
        
        qsort(ct.itemsets, ct.count, sizeof(Krimp_Itemset), cmp_cover_order);
        
        calculate_usage(&ct, ds);
        double new_score = calculate_mdl_score(&ct, ds, &st);
        
        if (new_score < current_score) {
            current_score = new_score;
            // Pruning
            if (prune) {
                // Algorithm 4: Post-Acceptance Pruning
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (size_t j = 0; j < ct.count; j++) {
                        if (ct.itemsets[j].count <= 1) continue;
                        
                        Krimp_Itemset backup = ct.itemsets[j];
                        // Try removing j
                        for (size_t k = j; k < ct.count - 1; k++) ct.itemsets[k] = ct.itemsets[k+1];
                        ct.count--;
                        
                        calculate_usage(&ct, ds);
                        double p_score = calculate_mdl_score(&ct, ds, &st);
                        if (p_score <= new_score) {
                            new_score = p_score;
                            current_score = p_score;
                            free(backup.items);
                            changed = true;
                            break;
                        } else {
                            // Restore
                            ct.count++;
                            for (size_t k = ct.count - 1; k > j; k--) ct.itemsets[k] = ct.itemsets[k-1];
                            ct.itemsets[j] = backup;
                            calculate_usage(&ct, ds); // Restore usage
                        }
                    }
                }
            }
        } else {
            // Reject
            free(ct.itemsets[ct.count - 1].items);
            ct.count--;
            qsort(ct.itemsets, ct.count, sizeof(Krimp_Itemset), cmp_cover_order);
            calculate_usage(&ct, ds); // Restore usage for next candidate
        }
    }

    printf("[KRIMP] Final Score: %.2f bits. Selected %zu itemsets.\n", current_score, ct.count);

    // Cleanup
    size_t total_items = 0;
    for (size_t i = 0; i < ct.count; i++) {
        total_items += ct.itemsets[i].count;
        free(ct.itemsets[i].items);
    }
    free(ct.itemsets);
    for (size_t i = 0; i < st.count; i++) free(st.itemsets[i].items);
    free(st.itemsets);
    for (size_t i = 0; i < candidates.count; i++) free(candidates.itemsets[i].items);
    free(candidates.itemsets);

    dm_bench_record_results(ct.count, total_items);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "krimp",
    .name = "KRIMP Algorithm",
    .description = "MDL-based pattern selection algorithm that finds the set of itemsets that compress the database best.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
