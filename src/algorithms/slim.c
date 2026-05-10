#include "algorithms/slim.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

/* --- SLIM DATA STRUCTURES --- */

typedef struct {
    uint32_t *items;
    size_t count;
    uint32_t support;
    uint32_t usage;
    uint32_t *tids;      // List of transaction IDs where this itemset is used in cover
    size_t tid_count;
    double code_length;
} Slim_Itemset;

typedef struct {
    Slim_Itemset *itemsets;
    size_t count;
    size_t capacity;
} Slim_List;

typedef struct {
    size_t i, j;
    double estimated_gain;
} Slim_Candidate;

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
    const Slim_Itemset *ia = (const Slim_Itemset *)a;
    const Slim_Itemset *ib = (const Slim_Itemset *)b;
    if (ia->count > ib->count) return -1;
    if (ia->count < ib->count) return 1;
    if (ia->support > ib->support) return -1;
    if (ia->support < ib->support) return 1;
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
    size_t wr = 0, si = 0;
    for (size_t i = 0; i < *set_len; i++) {
        if (si < sub_len && set[i] == sub[si]) { si++; }
        else { set[wr++] = set[i]; }
    }
    *set_len = wr;
}

/* --- SLIM LOGIC --- */

static void calculate_usage_and_tids(Slim_List *ct, DM_Dataset *ds) {
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ct->count; i++) {
        ct->itemsets[i].usage = 0;
        ct->itemsets[i].tid_count = 0;
        if (!ct->itemsets[i].tids) ct->itemsets[i].tids = (uint32_t*)malloc(sizeof(uint32_t) * ds->count);
    }
    
    uint32_t *temp_items = (uint32_t*)malloc(sizeof(uint32_t) * (ds->max_id + 1));
    for (size_t i = 0; i < ds->count; i++) {
        size_t current_len = data[i].count;
        memcpy(temp_items, data[i].items, sizeof(uint32_t) * current_len);
        for (size_t j = 0; j < ct->count; j++) {
            if (current_len == 0) break;
            if (is_subset(ct->itemsets[j].items, ct->itemsets[j].count, temp_items, current_len)) {
                ct->itemsets[j].usage++;
                ct->itemsets[j].tids[ct->itemsets[j].tid_count++] = i;
                remove_subset(temp_items, &current_len, ct->itemsets[j].items, ct->itemsets[j].count);
                j = -1; // Reset loop for StandardCover
            }
        }
    }
    free(temp_items);
}

static double calculate_mdl_score(Slim_List *ct, DM_Dataset *ds, Slim_List *st) {
    uint64_t total_usage = 0;
    for (size_t i = 0; i < ct->count; i++) total_usage += ct->itemsets[i].usage;
    if (total_usage == 0) return 1e18;

    double l_d_ct = 0;
    for (size_t i = 0; i < ct->count; i++) {
        if (ct->itemsets[i].usage > 0) {
            ct->itemsets[i].code_length = -log2((double)ct->itemsets[i].usage / total_usage);
            l_d_ct += ct->itemsets[i].usage * ct->itemsets[i].code_length;
        } else ct->itemsets[i].code_length = 0;
    }

    double l_ct_d = 0;
    for (size_t i = 0; i < ct->count; i++) {
        if (ct->itemsets[i].usage == 0) continue;
        l_ct_d += ct->itemsets[i].code_length;
        double l_st_x = 0;
        for (size_t j = 0; j < ct->itemsets[i].count; j++) {
            uint32_t item = ct->itemsets[i].items[j];
            for (size_t k = 0; k < st->count; k++) {
                if (st->itemsets[k].items[0] == item) { l_st_x += st->itemsets[k].code_length; break; }
            }
        }
        l_ct_d += l_st_x;
    }
    return l_d_ct + l_ct_d;
}

static size_t intersect_size(uint32_t *a, size_t n_a, uint32_t *b, size_t n_b) {
    size_t count = 0, i = 0, j = 0;
    while (i < n_a && j < n_b) {
        if (a[i] == b[j]) { count++; i++; j++; }
        else if (a[i] < b[j]) i++;
        else j++;
    }
    return count;
}

static double f_log_f(uint32_t f) {
    return f > 0 ? (double)f * log2((double)f) : 0;
}

static double estimate_gain(Slim_List *ct, size_t idx_x, size_t idx_y, DM_Dataset *ds, Slim_List *st) {
    Slim_Itemset *X = &ct->itemsets[idx_x];
    Slim_Itemset *Y = &ct->itemsets[idx_y];
    
    uint32_t xy_usage = intersect_size(X->tids, X->tid_count, Y->tids, Y->tid_count);
    if (xy_usage == 0) return -1e18;

    uint32_t s = 0;
    for (size_t i = 0; i < ct->count; i++) s += ct->itemsets[i].usage;
    
    uint32_t x = X->usage, y = Y->usage;
    uint32_t s_prime = s - xy_usage;
    uint32_t x_prime = x - xy_usage;
    uint32_t y_prime = y - xy_usage;

    // ΔL(D | CT) = L(D|CT) - L(D|CT')
    // L(D|CT) = s log s - Σ x log x
    double delta_l_d = (f_log_f(s) - f_log_f(s_prime)) - (f_log_f(x) + f_log_f(y)) + 
                       (f_log_f(x_prime) + f_log_f(y_prime) + f_log_f(xy_usage));

    // ΔL(CT) approx: code lengths change for all, but we focus on X, Y, XY
    // Model change: L(CT') - L(CT)
    // New itemset XY adds: L(code_ST(XY)) + L(code_CT'(XY))
    // X and Y codes change slightly.
    // Simplified:
    double l_st_xy = 0;
    for (size_t i = 0; i < X->count; i++) {
        for (size_t k = 0; k < st->count; k++) if (st->itemsets[k].items[0] == X->items[i]) { l_st_xy += st->itemsets[k].code_length; break; }
    }
    for (size_t i = 0; i < Y->count; i++) {
        // Only count items in Y NOT in X
        bool in_x = false;
        for (size_t j = 0; j < X->count; j++) if (Y->items[i] == X->items[j]) { in_x = true; break; }
        if (!in_x) {
            for (size_t k = 0; k < st->count; k++) if (st->itemsets[k].items[0] == Y->items[i]) { l_st_xy += st->itemsets[k].code_length; break; }
        }
    }
    
    double l_code_xy = -log2((double)xy_usage / s_prime);
    double delta_l_ct = l_st_xy + l_code_xy;

    return delta_l_d - delta_l_ct;
}

/* --- MAIN RUN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_SLIM_Params *p = (DM_SLIM_Params *)params;
    bool prune = p ? p->prune : true;

    printf("[SLIM] Starting. Pruning: %s\n", prune ? "ON" : "OFF");

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) qsort(data[i].items, data[i].count, sizeof(uint32_t), cmp_uint32);

    // 1. ST
    Slim_List st = {NULL, 0, 0};
    st.capacity = ds->max_id + 1;
    st.itemsets = (Slim_Itemset*)calloc(st.capacity, sizeof(Slim_Itemset));
    uint32_t *counts = (uint32_t*)calloc(ds->max_id + 1, sizeof(uint32_t));
    for (size_t i = 0; i < ds->count; i++) for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
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
    for (size_t i = 0; i < st.count; i++) st.itemsets[i].code_length = -log2((double)st.itemsets[i].usage / total_st_usage);

    // 2. Initial CT (Singletons)
    Slim_List ct = {NULL, 0, 0};
    ct.capacity = st.count + 256;
    ct.itemsets = (Slim_Itemset*)calloc(ct.capacity, sizeof(Slim_Itemset));
    for (size_t i = 0; i < st.count; i++) {
        ct.itemsets[i].items = (uint32_t*)malloc(sizeof(uint32_t));
        ct.itemsets[i].items[0] = st.itemsets[i].items[0];
        ct.itemsets[i].count = 1;
        ct.itemsets[i].support = st.itemsets[i].support;
        ct.count++;
    }
    qsort(ct.itemsets, ct.count, sizeof(Slim_Itemset), cmp_cover_order);
    calculate_usage_and_tids(&ct, ds);
    double current_score = calculate_mdl_score(&ct, ds, &st);
    printf("[SLIM] Initial Score: %.2f bits\n", current_score);

    // 3. Main Loop
    bool found_improvement = true;
    while (found_improvement) {
        found_improvement = false;
        
        // Find best candidate pair
        size_t best_i = 0, best_j = 0;
        double best_est_gain = -1.0;
        
        for (size_t i = 0; i < ct.count; i++) {
            for (size_t j = i + 1; j < ct.count; j++) {
                double est = estimate_gain(&ct, i, j, ds, &st);
                if (est > best_est_gain) {
                    best_est_gain = est;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        
        if (best_est_gain <= 0) break;

        // Try adding the union of best_i and best_j
        Slim_Itemset *X = &ct.itemsets[best_i];
        Slim_Itemset *Y = &ct.itemsets[best_j];
        
        // Create union
        size_t u_count = 0;
        uint32_t *u_items = (uint32_t*)malloc(sizeof(uint32_t) * (X->count + Y->count));
        // Simple merge (both are sorted)
        size_t px = 0, py = 0;
        while (px < X->count && py < Y->count) {
            if (X->items[px] == Y->items[py]) { u_items[u_count++] = X->items[px]; px++; py++; }
            else if (X->items[px] < Y->items[py]) { u_items[u_count++] = X->items[px++]; }
            else { u_items[u_count++] = Y->items[py++]; }
        }
        while (px < X->count) u_items[u_count++] = X->items[px++];
        while (py < Y->count) u_items[u_count++] = Y->items[py++];
        
        // Check if already in CT
        bool exists = false;
        for (size_t k = 0; k < ct.count; k++) {
            if (ct.itemsets[k].count == u_count && memcmp(ct.itemsets[k].items, u_items, sizeof(uint32_t)*u_count) == 0) {
                exists = true; break;
            }
        }
        
        if (exists) {
            // Mark this pair as invalid for next iteration by setting a flag or similar
            // For now, I'll just skip and we need a better way to track "tried pairs".
            // Paper says we sort by gain, so we'd try the next best.
            free(u_items);
            continue; 
        }

        // Calculate exact support for the union
        uint32_t u_sup = 0;
        for (size_t k = 0; k < ds->count; k++) if (is_subset(u_items, u_count, data[k].items, data[k].count)) u_sup++;

        // Add to CT
        if (ct.count >= ct.capacity) {
            ct.capacity *= 2;
            ct.itemsets = (Slim_Itemset*)realloc(ct.itemsets, sizeof(Slim_Itemset) * ct.capacity);
            // Re-zero new memory to avoid garbage tids
            memset(ct.itemsets + ct.count, 0, sizeof(Slim_Itemset) * (ct.capacity - ct.count));
        }
        
        Slim_Itemset *new_it = &ct.itemsets[ct.count++];
        new_it->items = u_items;
        new_it->count = u_count;
        new_it->support = u_sup;
        new_it->tids = NULL;
        
        qsort(ct.itemsets, ct.count, sizeof(Slim_Itemset), cmp_cover_order);
        calculate_usage_and_tids(&ct, ds);
        double new_score = calculate_mdl_score(&ct, ds, &st);
        
        if (new_score < current_score) {
            printf("[SLIM] Accepted itemset (size %zu). Gain: %.2f bits. Score: %.2f\n", u_count, current_score - new_score, new_score);
            current_score = new_score;
            found_improvement = true;
            
            if (prune) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (size_t k = 0; k < ct.count; k++) {
                        if (ct.itemsets[k].count <= 1) continue;
                        Slim_Itemset backup = ct.itemsets[k];
                        for (size_t m = k; m < ct.count - 1; m++) ct.itemsets[m] = ct.itemsets[m+1];
                        ct.count--;
                        calculate_usage_and_tids(&ct, ds);
                        double p_score = calculate_mdl_score(&ct, ds, &st);
                        if (p_score <= current_score) {
                            current_score = p_score;
                            free(backup.items); free(backup.tids);
                            changed = true; break;
                        } else {
                            ct.count++;
                            for (size_t m = ct.count - 1; m > k; m--) ct.itemsets[m] = ct.itemsets[m-1];
                            ct.itemsets[k] = backup;
                            calculate_usage_and_tids(&ct, ds);
                        }
                    }
                }
            }
        } else {
            // Reject
            free(new_it->items);
            ct.count--;
            qsort(ct.itemsets, ct.count, sizeof(Slim_Itemset), cmp_cover_order);
            calculate_usage_and_tids(&ct, ds);
        }
    }

    printf("[SLIM] Final Score: %.2f bits. Selected %zu itemsets.\n", current_score, ct.count);

    // Cleanup
    size_t total_fi_items = 0;
    for (size_t i = 0; i < ct.count; i++) {
        total_fi_items += ct.itemsets[i].count;
        free(ct.itemsets[i].items);
        if (ct.itemsets[i].tids) free(ct.itemsets[i].tids);
    }
    free(ct.itemsets);
    for (size_t i = 0; i < st.count; i++) free(st.itemsets[i].items);
    free(st.itemsets);
    dm_bench_record_results(ct.count, total_fi_items);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "slim",
    .name = "SLIM Algorithm",
    .description = "Directly mines descriptive patterns using MDL by iteratively merging itemsets in the code table.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
