#include "algorithms/close.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

/* --- INTERNAL DATA STRUCTURES --- */

typedef struct {
    uint32_t *generator;
    size_t gen_count;
    uint32_t *closure;
    size_t closure_count;
    uint32_t support;
} Close_Itemset;

typedef struct {
    Close_Itemset *array;
    size_t count;
    size_t capacity;
    size_t k;
} Close_List;

typedef struct {
    uint32_t *items;
    size_t count;
    uint32_t support;
} FrequentItemset;

typedef struct {
    FrequentItemset *array;
    size_t count;
    size_t capacity;
    size_t k;
} FI_List;

/* --- UTILITY FUNCTIONS --- */

static int cmp_uint32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

static int cmp_itemset(const uint32_t *a, const uint32_t *b, size_t k) {
    for (size_t i = 0; i < k; i++) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

static void list_init(Close_List *list, size_t k) {
    list->capacity = 1024;
    list->count = 0;
    list->k = k;
    list->array = (Close_Itemset *)malloc(sizeof(Close_Itemset) * list->capacity);
}

static void list_free(Close_List *list) {
    for (size_t i = 0; i < list->count; i++) {
        free(list->array[i].generator);
        if (list->array[i].closure) free(list->array[i].closure);
    }
    free(list->array);
}

static void list_append(Close_List *list, uint32_t *gen, uint32_t support) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->array = (Close_Itemset *)realloc(list->array, sizeof(Close_Itemset) * list->capacity);
    }
    list->array[list->count].generator = gen;
    list->array[list->count].gen_count = list->k;
    list->array[list->count].closure = NULL;
    list->array[list->count].closure_count = 0;
    list->array[list->count].support = support;
    list->count++;
}

static long list_find(Close_List *list, const uint32_t *items) {
    size_t low = 0, high = list->count;
    size_t k = list->k;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int cmp = cmp_itemset(list->array[mid].generator, items, k);
        if (cmp == 0) return (long)mid;
        if (cmp < 0) low = mid + 1;
        else high = mid;
    }
    return -1;
}

static bool is_subset(const uint32_t *sub, size_t sub_len, const uint32_t *set, size_t set_len) {
    size_t i = 0, j = 0;
    while (i < sub_len && j < set_len) {
        if (sub[i] == set[j]) { i++; j++; }
        else if (sub[i] > set[j]) { j++; }
        else return false;
    }
    return i == sub_len;
}

static size_t intersect(uint32_t *a, size_t a_len, const uint32_t *b, size_t b_len) {
    size_t i = 0, j = 0, k = 0;
    while (i < a_len && j < b_len) {
        if (a[i] == b[j]) {
            a[k++] = a[i];
            i++; j++;
        } else if (a[i] < b[j]) {
            i++;
        } else {
            j++;
        }
    }
    return k;
}

/* --- ALGORITHM 4: GEN-CLOER --- */
static void gen_closure(Close_List *FCC, DM_Dataset *ds) {
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t t = 0; t < ds->count; t++) {
        DM_Trans_Simple tr = data[t];
        for (size_t i = 0; i < FCC->count; i++) {
            if (is_subset(FCC->array[i].generator, FCC->k, tr.items, tr.count)) {
                if (FCC->array[i].closure == NULL) {
                    FCC->array[i].closure = (uint32_t *)malloc(sizeof(uint32_t) * tr.count);
                    memcpy(FCC->array[i].closure, tr.items, sizeof(uint32_t) * tr.count);
                    FCC->array[i].closure_count = tr.count;
                } else {
                    FCC->array[i].closure_count = intersect(FCC->array[i].closure, FCC->array[i].closure_count, tr.items, tr.count);
                }
                FCC->array[i].support++;
            }
        }
    }
}

/* --- ALGORITHM 5: GEN-GENERATOR --- */
static void gen_generator(Close_List *FC_prev, Close_List *FCC_next) {
    size_t k = FC_prev->k + 1;
    list_init(FCC_next, k);
    uint32_t *subset = (uint32_t *)malloc(sizeof(uint32_t) * (k - 1));

    for (size_t i = 0; i < FC_prev->count; i++) {
        for (size_t j = i + 1; j < FC_prev->count; j++) {
            bool match = true;
            for (size_t m = 0; m < k - 2; m++) {
                if (FC_prev->array[i].generator[m] != FC_prev->array[j].generator[m]) {
                    match = false;
                    break;
                }
            }
            if (!match) break;

            uint32_t *c_gen = (uint32_t *)malloc(sizeof(uint32_t) * k);
            memcpy(c_gen, FC_prev->array[i].generator, sizeof(uint32_t) * (k - 1));
            c_gen[k - 1] = FC_prev->array[j].generator[k - 2];

            // Strategy 1: All subsets must be generators in FC_prev
            bool all_subsets_present = true;
            for (size_t m = 0; m < k; m++) {
                if (m == k - 1 || m == k - 2) continue;
                size_t s_idx = 0;
                for (size_t n = 0; n < k; n++) {
                    if (n != m) subset[s_idx++] = c_gen[n];
                }
                if (list_find(FC_prev, subset) < 0) {
                    all_subsets_present = false;
                    break;
                }
            }

            if (all_subsets_present) {
                // Strategy 2: Generator not included in any of its subsets' closures
                bool redundant = false;
                for (size_t m = 0; m < k; m++) {
                    size_t s_idx = 0;
                    for (size_t n = 0; n < k; n++) {
                        if (n != m) subset[s_idx++] = c_gen[n];
                    }
                    long sub_idx = list_find(FC_prev, subset);
                    if (sub_idx >= 0) {
                        if (is_subset(c_gen, k, FC_prev->array[sub_idx].closure, FC_prev->array[sub_idx].closure_count)) {
                            redundant = true;
                            break;
                        }
                    }
                }

                if (!redundant) {
                    list_append(FCC_next, c_gen, 0);
                } else {
                    free(c_gen);
                }
            } else {
                free(c_gen);
            }
        }
    }
    free(subset);
}

/* --- ALGORITHM 6: DERIVING FREQUENT ITEMSETS --- */
static void fi_list_init(FI_List *list, size_t k) {
    list->capacity = 1024;
    list->count = 0;
    list->k = k;
    list->array = (FrequentItemset *)malloc(sizeof(FrequentItemset) * list->capacity);
}

static void fi_list_free(FI_List *list) {
    for (size_t i = 0; i < list->count; i++) {
        free(list->array[i].items);
    }
    free(list->array);
}

static void fi_list_append(FI_List *list, const uint32_t *items, size_t count, uint32_t support) {
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->array = (FrequentItemset *)realloc(list->array, sizeof(FrequentItemset) * list->capacity);
    }
    list->array[list->count].items = (uint32_t *)malloc(sizeof(uint32_t) * count);
    memcpy(list->array[list->count].items, items, sizeof(uint32_t) * count);
    list->array[list->count].count = count;
    list->array[list->count].support = support;
    list->count++;
}

static int cmp_fi(const void *a, const void *b) {
    const FrequentItemset *fa = (const FrequentItemset *)a;
    const FrequentItemset *fb = (const FrequentItemset *)b;
    if (fa->count != fb->count) return (fa->count < fb->count) ? -1 : 1;
    return cmp_itemset(fa->items, fb->items, fa->count);
}

static bool fi_list_contains(FI_List *list, const uint32_t *items, size_t k) {
    size_t low = 0, high = list->count;
    while (low < high) {
        size_t mid = low + (high - low) / 2;
        int cmp = cmp_itemset(list->array[mid].items, items, k);
        if (cmp == 0) return true;
        if (cmp < 0) low = mid + 1;
        else high = mid;
    }
    return false;
}

static void fi_apriori_gen(FI_List *L_prev, FI_List *C_curr) {
    size_t k = L_prev->k + 1;
    fi_list_init(C_curr, k);
    uint32_t *subset = (uint32_t *)malloc(sizeof(uint32_t) * (k - 1));

    for (size_t i = 0; i < L_prev->count; i++) {
        for (size_t j = i + 1; j < L_prev->count; j++) {
            bool match = true;
            for (size_t m = 0; m < k - 2; m++) {
                if (L_prev->array[i].items[m] != L_prev->array[j].items[m]) {
                    match = false;
                    break;
                }
            }
            if (!match) break;

            uint32_t *c_items = (uint32_t *)malloc(sizeof(uint32_t) * k);
            memcpy(c_items, L_prev->array[i].items, sizeof(uint32_t) * (k - 1));
            c_items[k - 1] = L_prev->array[j].items[k - 2];

            bool all_subsets_present = true;
            for (size_t drop = 0; drop < k; drop++) {
                if (drop == k - 1 || drop == k - 2) continue;
                size_t s_idx = 0;
                for (size_t m = 0; m < k; m++) {
                    if (m != drop) subset[s_idx++] = c_items[m];
                }
                if (!fi_list_contains(L_prev, subset, k - 1)) {
                    all_subsets_present = false;
                    break;
                }
            }

            if (all_subsets_present) {
                fi_list_append(C_curr, c_items, k, 0);
            }
            free(c_items);
        }
    }
    free(subset);
}

void gen_rules_recursive(FrequentItemset l_k, FI_List *H_m, FI_List *all_fi_global, double min_conf, size_t *rule_count_ptr) {
    if (l_k.count > H_m->k + 1) {
        FI_List H_next;
        fi_apriori_gen(H_m, &H_next);
        
        for (size_t i = 0; i < H_next.count; ) {
            uint32_t *h = H_next.array[i].items;
            size_t ant_len = l_k.count - H_next.k;
            uint32_t *ant = (uint32_t *)malloc(sizeof(uint32_t) * ant_len);
            
            // ant = l_k \ h
            size_t a_ptr = 0;
            for (size_t m = 0; m < l_k.count; m++) {
                bool in_h = false;
                for (size_t n = 0; n < H_next.k; n++) {
                    if (l_k.items[m] == h[n]) { in_h = true; break; }
                }
                if (!in_h) ant[a_ptr++] = l_k.items[m];
            }
            
            uint32_t ant_supp = 0;
            size_t low = 0, high = all_fi_global[ant_len].count;
            while (low < high) {
                size_t mid = low + (high - low) / 2;
                int cmp = cmp_itemset(all_fi_global[ant_len].array[mid].items, ant, ant_len);
                if (cmp == 0) { ant_supp = all_fi_global[ant_len].array[mid].support; break; }
                if (cmp < 0) low = mid + 1; else high = mid;
            }
            free(ant);

            if (ant_supp > 0 && (double)l_k.support / ant_supp >= min_conf) {
                (*rule_count_ptr)++;
                i++;
            } else {
                // Delete h from H_next
                free(H_next.array[i].items);
                for (size_t m = i; m < H_next.count - 1; m++) H_next.array[m] = H_next.array[m+1];
                H_next.count--;
            }
        }
        
        if (H_next.count > 0) {
            gen_rules_recursive(l_k, &H_next, all_fi_global, min_conf, rule_count_ptr);
        }
        fi_list_free(&H_next);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_CLOSE_Params *p = (DM_CLOSE_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0) min_sup = 1;

    printf("[Close] Starting. Min Support: %u\n", min_sup);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        qsort(data[i].items, data[i].count, sizeof(uint32_t), cmp_uint32);
    }

    // Step 1: Initialize generators in FCC_1
    Close_List *all_fcc = NULL;
    size_t fcc_levels = 0;
    size_t fcc_cap = 16;
    all_fcc = (Close_List *)malloc(sizeof(Close_List) * fcc_cap);

    Close_List FCC1;
    list_init(&FCC1, 1);
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        uint32_t *gen = (uint32_t *)malloc(sizeof(uint32_t));
        gen[0] = i;
        list_append(&FCC1, gen, 0);
    }
    
    gen_closure(&FCC1, ds);

    Close_List FC1;
    list_init(&FC1, 1);
    for (size_t i = 0; i < FCC1.count; i++) {
        if (FCC1.array[i].support >= min_sup) {
            uint32_t *gen = (uint32_t *)malloc(sizeof(uint32_t));
            gen[0] = FCC1.array[i].generator[0];
            list_append(&FC1, gen, FCC1.array[i].support);
            FC1.array[FC1.count - 1].closure = (uint32_t *)malloc(sizeof(uint32_t) * FCC1.array[i].closure_count);
            memcpy(FC1.array[FC1.count - 1].closure, FCC1.array[i].closure, sizeof(uint32_t) * FCC1.array[i].closure_count);
            FC1.array[FC1.count - 1].closure_count = FCC1.array[i].closure_count;
        }
    }
    list_free(&FCC1);

    all_fcc[fcc_levels++] = FC1;
    printf("[Close] Found %zu frequent 1-generators\n", FC1.count);

    size_t k = 2;
    while (all_fcc[fcc_levels - 1].count > 0) {
        Close_List FCC_next_cand;
        gen_generator(&all_fcc[fcc_levels - 1], &FCC_next_cand);
        if (FCC_next_cand.count == 0) {
            list_free(&FCC_next_cand);
            break;
        }

        gen_closure(&FCC_next_cand, ds);

        Close_List FC_next;
        list_init(&FC_next, k);
        for (size_t i = 0; i < FCC_next_cand.count; i++) {
            if (FCC_next_cand.array[i].support >= min_sup) {
                uint32_t *gen = (uint32_t *)malloc(sizeof(uint32_t) * k);
                memcpy(gen, FCC_next_cand.array[i].generator, sizeof(uint32_t) * k);
                list_append(&FC_next, gen, FCC_next_cand.array[i].support);
                FC_next.array[FC_next.count - 1].closure = (uint32_t *)malloc(sizeof(uint32_t) * FCC_next_cand.array[i].closure_count);
                memcpy(FC_next.array[FC_next.count - 1].closure, FCC_next_cand.array[i].closure, sizeof(uint32_t) * FCC_next_cand.array[i].closure_count);
                FC_next.array[FC_next.count - 1].closure_count = FCC_next_cand.array[i].closure_count;
            }
        }
        list_free(&FCC_next_cand);
        
        if (fcc_levels == fcc_cap) {
            fcc_cap *= 2;
            all_fcc = (Close_List *)realloc(all_fcc, sizeof(Close_List) * fcc_cap);
        }
        all_fcc[fcc_levels++] = FC_next;
        printf("[Close] Found %zu frequent %zu-generators\n", FC_next.count, k);
        k++;
    }

    // Phase 2: Deriving Frequent Itemsets (Algorithm 6)
    size_t max_k = 0;
    for (size_t i = 0; i < fcc_levels; i++) {
        for (size_t j = 0; j < all_fcc[i].count; j++) {
            if (all_fcc[i].array[j].closure_count > max_k) {
                max_k = all_fcc[i].array[j].closure_count;
            }
        }
    }

    FI_List *all_fi = (FI_List *)malloc(sizeof(FI_List) * (max_k + 1));
    for (size_t i = 0; i <= max_k; i++) fi_list_init(&all_fi[i], i);

    for (size_t i = 0; i < fcc_levels; i++) {
        for (size_t j = 0; j < all_fcc[i].count; j++) {
            fi_list_append(&all_fi[all_fcc[i].array[j].closure_count], 
                           all_fcc[i].array[j].closure, all_fcc[i].array[j].closure_count, 
                           all_fcc[i].array[j].support);
        }
    }

    for (size_t i = 0; i <= max_k; i++) {
        qsort(all_fi[i].array, all_fi[i].count, sizeof(FrequentItemset), cmp_fi);
    }

    uint32_t *subset_buf = (uint32_t *)malloc(sizeof(uint32_t) * max_k);
    for (size_t curr_k = max_k; curr_k > 1; curr_k--) {
        for (size_t i = 0; i < all_fi[curr_k].count; i++) {
            FrequentItemset c = all_fi[curr_k].array[i];
            for (size_t drop = 0; drop < curr_k; drop++) {
                size_t s_idx = 0;
                for (size_t m = 0; m < curr_k; m++) {
                    if (m != drop) subset_buf[s_idx++] = c.items[m];
                }
                if (!fi_list_contains(&all_fi[curr_k - 1], subset_buf, curr_k - 1)) {
                    fi_list_append(&all_fi[curr_k - 1], subset_buf, curr_k - 1, c.support);
                    // Re-sort and keep unique - simplified for this implementation
                    qsort(all_fi[curr_k - 1].array, all_fi[curr_k - 1].count, sizeof(FrequentItemset), cmp_fi);
                }
            }
        }
    }
    free(subset_buf);

    size_t total_fi = 0;
    size_t footprint = 0;
    for (size_t i = 1; i <= max_k; i++) {
        total_fi += all_fi[i].count;
        footprint += all_fi[i].count * i;
    }

    // Phase 3: Association Rule Generation (Algorithm 7)
    double min_conf = p ? p->min_confidence : 0.8;
    size_t rule_count = 0;
    
    printf("[Close] Phase 2 complete. Found %zu frequent itemsets.\n", total_fi);
    printf("[Close] Phase 3: Generating rules (confidence >= %.2f)...\n", min_conf);

    for (size_t k_idx = 2; k_idx <= max_k; k_idx++) {
        for (size_t i = 0; i < all_fi[k_idx].count; i++) {
            FrequentItemset l_k = all_fi[k_idx].array[i];
            
            // H1 = {itemsets of size 1 that are subsets of l_k}
            FI_List H1;
            fi_list_init(&H1, 1);
            for (size_t m = 0; m < k_idx; m++) {
                uint32_t item = l_k.items[m];
                fi_list_append(&H1, &item, 1, 0); 
            }
            
            // Check confidence for 1-item consequents
            for (size_t h_idx = 0; h_idx < H1.count; ) {
                uint32_t *h1 = H1.array[h_idx].items;
                // find l_k \ h1
                size_t ant_len = k_idx - 1;
                uint32_t *ant = (uint32_t *)malloc(sizeof(uint32_t) * ant_len);
                size_t a_ptr = 0;
                for (size_t m = 0; m < k_idx; m++) {
                    if (l_k.items[m] != h1[0]) ant[a_ptr++] = l_k.items[m];
                }
                
                // Lookup support of antecedent
                uint32_t ant_supp = 0;
                size_t low = 0, high = all_fi[ant_len].count;
                while (low < high) {
                    size_t mid = low + (high - low) / 2;
                    int cmp = cmp_itemset(all_fi[ant_len].array[mid].items, ant, ant_len);
                    if (cmp == 0) { ant_supp = all_fi[ant_len].array[mid].support; break; }
                    if (cmp < 0) low = mid + 1; else high = mid;
                }
                free(ant);

                if (ant_supp > 0 && (double)l_k.support / ant_supp >= min_conf) {
                    rule_count++;
                    h_idx++;
                } else {
                    // Remove h1 from H1
                    free(H1.array[h_idx].items);
                    for (size_t m = h_idx; m < H1.count - 1; m++) H1.array[m] = H1.array[m+1];
                    H1.count--;
                }
            }
            
            if (H1.count > 0) {
                // Procedure Gen-Rules(l_k, H_m)
                // We use an iterative approach or recursive helper
                void gen_rules_recursive(FrequentItemset l_k, FI_List *H_m, FI_List *all_fi_global, double min_conf, size_t *rule_count_ptr);
                gen_rules_recursive(l_k, &H1, all_fi, min_conf, &rule_count);
            }
            fi_list_free(&H1);
        }
    }

    printf("[Close] Phase 3 complete. Found %zu association rules.\n", rule_count);
    printf("[Close] Complete. Total frequent itemsets found: %zu\n", total_fi);
    dm_bench_record_results(total_fi, footprint);

    // Cleanup
    for (size_t i = 0; i < fcc_levels; i++) list_free(&all_fcc[i]);
    free(all_fcc);
    for (size_t i = 0; i <= max_k; i++) fi_list_free(&all_fi[i]);
    free(all_fi);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "close",
    .name = "Close Algorithm",
    .description = "Efficient mining of association rules using closed itemset lattices (Pasquier et al. 1999).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
