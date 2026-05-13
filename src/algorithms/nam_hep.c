#include "algorithms/nam_hep.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t *tids;
    size_t count;
} TIDSet;

typedef struct {
    uint32_t *items;
    size_t len;
    TIDSet tids;
    double occupancy;
} Node;

typedef struct {
    Node *nodes;
    size_t count;
} Level;

/* --- UTILS --- */

static int cmp_uint32(const void *a, const void *b) {
    return (int)(*(uint32_t*)a - *(uint32_t*)b);
}

static int cmp_double(const void *a, const void *b) {
    double v1 = *(double*)a;
    double v2 = *(double*)b;
    if (v1 < v2) return -1;
    if (v1 > v2) return 1;
    return 0;
}

static double calculate_median(double *values, size_t count) {
    if (count == 0) return 0;
    qsort(values, count, sizeof(double), cmp_double);
    if (count % 2 == 1) return values[count / 2];
    return (values[count / 2 - 1] + values[count / 2]) / 2.0;
}

static TIDSet intersect(TIDSet s1, TIDSet s2) {
    TIDSet res;
    res.tids = malloc(sizeof(uint32_t) * (s1.count < s2.count ? s1.count : s2.count));
    res.count = 0;
    size_t i = 0, j = 0;
    while (i < s1.count && j < s2.count) {
        if (s1.tids[i] == s2.tids[j]) {
            res.tids[res.count++] = s1.tids[i];
            i++; j++;
        } else if (s1.tids[i] < s2.tids[j]) i++;
        else j++;
    }
    return res;
}

static void free_level(Level *lvl) {
    for (size_t i = 0; i < lvl->count; i++) {
        free(lvl->nodes[i].items);
        free(lvl->nodes[i].tids.tids);
    }
    free(lvl->nodes);
    free(lvl);
}

/* --- MAIN RUN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_TRANSACTIONAL) return DM_ERROR_INCOMPATIBLE;

    DM_NAM_HEP_Params *p = (DM_NAM_HEP_Params *)params;
    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;
    size_t total_trans = ds->count;

    // Level 1 Initialization
    TIDSet *item_tidsets = calloc(ds->max_id + 1, sizeof(TIDSet));
    size_t *item_sups = calloc(ds->max_id + 1, sizeof(size_t));
    for (size_t i = 0; i < total_trans; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            item_sups[data[i].items[j]]++;
        }
    }
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_sups[i] > 0) {
            item_tidsets[i].tids = malloc(sizeof(uint32_t) * item_sups[i]);
            item_tidsets[i].count = 0;
        }
    }
    for (size_t i = 0; i < total_trans; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t id = data[i].items[j];
            item_tidsets[id].tids[item_tidsets[id].count++] = (uint32_t)i;
        }
    }

    double *supports_1 = malloc(sizeof(double) * (ds->max_id + 1));
    double *occupancies_1 = malloc(sizeof(double) * (ds->max_id + 1));
    size_t active_count_1 = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_sups[i] > 0) {
            supports_1[active_count_1] = (double)item_sups[i];
            double occ = 0;
            for (size_t j = 0; j < item_tidsets[i].count; j++) {
                occ += 1.0 / data[item_tidsets[i].tids[j]].count;
            }
            occupancies_1[active_count_1] = occ;
            active_count_1++;
        }
    }

    double delta = (p && p->support_threshold >= 0) ? p->support_threshold : calculate_median(supports_1, active_count_1);
    double xi_prev = (p && p->occupancy_threshold >= 0) ? p->occupancy_threshold : calculate_median(occupancies_1, active_count_1);
    
    printf("[NAM-HEP] Adaptive Support threshold (delta): %.2f\n", delta);
    printf("[NAM-HEP] Initial Occupancy threshold (xi_1): %.2f\n", xi_prev);

    Level *h_prev = malloc(sizeof(Level));
    h_prev->nodes = malloc(sizeof(Node) * active_count_1);
    h_prev->count = 0;

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (item_sups[i] >= delta) {
            double occ = 0;
            for (size_t j = 0; j < item_tidsets[i].count; j++) {
                occ += 1.0 / data[item_tidsets[i].tids[j]].count;
            }
            if (occ >= xi_prev) {
                h_prev->nodes[h_prev->count].items = malloc(sizeof(uint32_t));
                h_prev->nodes[h_prev->count].items[0] = i;
                h_prev->nodes[h_prev->count].len = 1;
                h_prev->nodes[h_prev->count].tids.count = item_tidsets[i].count;
                h_prev->nodes[h_prev->count].tids.tids = malloc(sizeof(uint32_t) * item_tidsets[i].count);
                memcpy(h_prev->nodes[h_prev->count].tids.tids, item_tidsets[i].tids, sizeof(uint32_t) * item_tidsets[i].count);
                h_prev->nodes[h_prev->count].occupancy = occ;
                h_prev->count++;
            }
        }
    }
    printf("[NAM-HEP] Level 1: Found %zu HO itemsets\n", h_prev->count);

    size_t total_ho_count = h_prev->count;
    size_t total_items_sum = h_prev->count;

    // Recursive Level k
    for (size_t k = 2; h_prev->count > 0; k++) {
        // I_{k-1} = Scan(H_{k-1})
        uint32_t *ik_minus_1 = malloc(sizeof(uint32_t) * (ds->max_id + 1));
        size_t ik_count = 0;
        bool *present = calloc(ds->max_id + 1, sizeof(bool));
        for (size_t i = 0; i < h_prev->count; i++) {
            for (size_t j = 0; j < h_prev->nodes[i].len; j++) {
                uint32_t id = h_prev->nodes[i].items[j];
                if (!present[id]) {
                    ik_minus_1[ik_count++] = id;
                    present[id] = true;
                }
            }
        }
        qsort(ik_minus_1, ik_count, sizeof(uint32_t), cmp_uint32);
        free(present);

        // Generate P_k' (filtered by delta)
        Node *pk_prime = malloc(sizeof(Node) * h_prev->count * ik_count);
        size_t pk_prime_count = 0;
        double *pk_occupancies = malloc(sizeof(double) * h_prev->count * ik_count);

        for (size_t i = 0; i < h_prev->count; i++) {
            uint32_t max_item = h_prev->nodes[i].items[h_prev->nodes[i].len - 1];
            for (size_t j = 0; j < ik_count; j++) {
                if (ik_minus_1[j] > max_item) {
                    TIDSet new_tids = intersect(h_prev->nodes[i].tids, item_tidsets[ik_minus_1[j]]);
                    if (new_tids.count >= delta) {
                        pk_prime[pk_prime_count].len = k;
                        pk_prime[pk_prime_count].items = malloc(sizeof(uint32_t) * k);
                        memcpy(pk_prime[pk_prime_count].items, h_prev->nodes[i].items, sizeof(uint32_t) * (k - 1));
                        pk_prime[pk_prime_count].items[k - 1] = ik_minus_1[j];
                        pk_prime[pk_prime_count].tids = new_tids;
                        
                        double occ = 0;
                        for (size_t l = 0; l < new_tids.count; l++) {
                            occ += (double)k / data[new_tids.tids[l]].count;
                        }
                        pk_prime[pk_prime_count].occupancy = occ;
                        pk_occupancies[pk_prime_count] = occ;
                        pk_prime_count++;
                    } else {
                        free(new_tids.tids);
                    }
                }
            }
        }
        free(ik_minus_1);

        if (pk_prime_count == 0) {
            free(pk_prime); free(pk_occupancies);
            break;
        }

        double xi_k = calculate_median(pk_occupancies, pk_prime_count);
        free(pk_occupancies);

        if (xi_k < xi_prev) {
            printf("[NAM-HEP] Level %zu: xi_k (%.2f) < xi_{k-1} (%.2f). Stopping.\n", k, xi_k, xi_prev);
            for (size_t i = 0; i < pk_prime_count; i++) {
                free(pk_prime[i].items); free(pk_prime[i].tids.tids);
            }
            free(pk_prime);
            break;
        }

        Level *h_k = malloc(sizeof(Level));
        h_k->nodes = malloc(sizeof(Node) * pk_prime_count);
        h_k->count = 0;

        for (size_t i = 0; i < pk_prime_count; i++) {
            if (pk_prime[i].occupancy >= xi_k) {
                h_k->nodes[h_k->count++] = pk_prime[i];
            } else {
                free(pk_prime[i].items); free(pk_prime[i].tids.tids);
            }
        }
        free(pk_prime);

        printf("[NAM-HEP] Level %zu: Found %zu HO itemsets (xi_k=%.2f)\n", k, h_k->count, xi_k);
        total_ho_count += h_k->count;
        total_items_sum += h_k->count * k;
        
        free_level(h_prev);
        h_prev = h_k;
        xi_prev = xi_k;
    }

    printf("[NAM-HEP] Total HO itemsets found: %zu\n", total_ho_count);
    dm_bench_record_results(total_ho_count, total_items_sum);

    // Cleanup
    free_level(h_prev);
    for (uint32_t i = 0; i <= ds->max_id; i++) if (item_sups[i] > 0) free(item_tidsets[i].tids);
    free(item_tidsets); free(item_sups); free(supports_1); free(occupancies_1);

    return DM_SUCCESS;
}

DM_Algorithm nam_hep_algo = {
    .id = "nam_hep",
    .name = "NAM-HEP",
    .description = "Adaptive High Occupancy Itemset Mining with SE-tree (Tran et al. 2025).",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(nam_hep_algo)
