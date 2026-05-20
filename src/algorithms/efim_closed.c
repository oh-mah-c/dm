#include "algorithms/efim_closed.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t id;
    double utility;
} EFC_Item;

typedef struct {
    EFC_Item *items;
    size_t count;
    double total_utility;
    size_t weight; // For merged transactions
} EFC_Transaction;

typedef struct {
    EFC_Transaction *transactions;
    size_t count;
    uint32_t max_item_id;
} EFC_Database;

typedef struct {
    EFC_Transaction t;
    double u_alpha;
} EFC_Entry;

/* --- UTILS --- */

static uint32_t *rank = NULL;

static int cmp_efc_items(const void *a, const void *b) {
    uint32_t r1 = rank[((EFC_Item*)a)->id];
    uint32_t r2 = rank[((EFC_Item*)b)->id];
    return (r1 < r2) ? -1 : ((r1 > r2) ? 1 : 0);
}

static int cmp_entries(const void *a, const void *b) {
    const EFC_Entry *e1 = (const EFC_Entry *)a;
    const EFC_Entry *e2 = (const EFC_Entry *)b;
    size_t min_len = e1->t.count < e2->t.count ? e1->t.count : e2->t.count;
    for (size_t i = 0; i < min_len; i++) {
        uint32_t id1 = e1->t.items[e1->t.count - 1 - i].id;
        uint32_t id2 = e2->t.items[e2->t.count - 1 - i].id;
        if (id1 < id2) return -1;
        if (id1 > id2) return 1;
    }
    if (e1->t.count < e2->t.count) return -1;
    if (e1->t.count > e2->t.count) return 1;
    return 0;
}

static void merge_database(EFC_Database *db, double *u_alpha_per_trans) {
    if (db->count <= 1) return;
    EFC_Entry *entries = malloc(sizeof(EFC_Entry) * db->count);
    for (size_t i = 0; i < db->count; i++) {
        entries[i].t = db->transactions[i];
        entries[i].u_alpha = u_alpha_per_trans[i];
    }
    qsort(entries, db->count, sizeof(EFC_Entry), cmp_entries);
    size_t write_idx = 0;
    for (size_t i = 1; i < db->count; i++) {
        bool identical = (entries[i].t.count == entries[write_idx].t.count);
        if (identical) {
            for (size_t j = 0; j < entries[i].t.count; j++) {
                if (entries[i].t.items[j].id != entries[write_idx].t.items[j].id) {
                    identical = false; break;
                }
            }
        }
        if (identical) {
            for (size_t j = 0; j < entries[i].t.count; j++) {
                entries[write_idx].t.items[j].utility += entries[i].t.items[j].utility;
            }
            entries[write_idx].t.total_utility += entries[i].t.total_utility;
            entries[write_idx].t.weight += entries[i].t.weight;
            entries[write_idx].u_alpha += entries[i].u_alpha;
            free(entries[i].t.items);
        } else {
            write_idx++;
            entries[write_idx] = entries[i];
        }
    }
    db->count = write_idx + 1;
    for (size_t i = 0; i < db->count; i++) {
        db->transactions[i] = entries[i].t;
        u_alpha_per_trans[i] = entries[i].u_alpha;
    }
    free(entries);
}

typedef struct { uint32_t id; double twu; } ItemTWU;

static int cmp_item_twu_static(const void *a, const void *b) {
    double t1 = ((ItemTWU*)a)->twu;
    double t2 = ((ItemTWU*)b)->twu;
    if (t1 < t2) return -1;
    if (t1 > t2) return 1;
    return (int)(((ItemTWU*)a)->id - ((ItemTWU*)b)->id);
}

/* --- CORE --- */

static size_t total_chui_count = 0;
static size_t total_chui_items = 0;

static void efc_search(uint32_t *alpha, size_t alpha_len, EFC_Database *db, double *u_alpha_per_trans, 
                       uint32_t *primary, size_t prim_count, uint32_t *secondary, size_t sec_count, 
                       double min_util) {
    for (size_t i = 0; i < prim_count; i++) {
        uint32_t item = primary[i];
        uint32_t item_rank = rank[item];
        
        uint32_t *beta = malloc(sizeof(uint32_t) * (alpha_len + 1));
        if (alpha_len > 0) memcpy(beta, alpha, sizeof(uint32_t) * alpha_len);
        beta[alpha_len] = item;

        // Calculate support, utility of beta, and back_counts for BCC
        double total_u_beta = 0;
        size_t sup_beta = 0;
        uint32_t *back_counts = calloc(db->max_item_id + 1, sizeof(uint32_t));
        
        for (size_t j = 0; j < db->count; j++) {
            EFC_Transaction *t = &db->transactions[j];
            bool found = false;
            size_t item_idx = 0;
            for (size_t k = 0; k < t->count; k++) {
                if (t->items[k].id == item) {
                    found = true;
                    item_idx = k;
                    break;
                }
            }
            if (found) {
                sup_beta += t->weight;
                total_u_beta += (u_alpha_per_trans[j] + t->items[item_idx].utility);
                for (size_t k = 0; k < item_idx; k++) {
                    back_counts[t->items[k].id] += t->weight;
                }
            }
        }
        
        // BCC check: does there exist an item z < i (rank[z] < item_rank) in Secondary(alpha)
        // such that back_counts[z] == sup_beta?
        bool has_backward = false;
        for (uint32_t z = 0; z <= db->max_item_id; z++) {
            if (back_counts[z] == sup_beta) {
                has_backward = true;
                break;
            }
        }
        free(back_counts);
        
        if (has_backward) {
            free(beta);
            continue; // Pruned by BCC!
        }
        
        // If no backward extension, construct projected database beta_db
        EFC_Database beta_db;
        beta_db.max_item_id = db->max_item_id;
        beta_db.transactions = malloc(sizeof(EFC_Transaction) * db->count);
        beta_db.count = 0;
        
        double *u_beta_per_trans = malloc(sizeof(double) * db->count);
        
        for (size_t j = 0; j < db->count; j++) {
            EFC_Transaction *t = &db->transactions[j];
            bool found = false;
            size_t item_idx = 0;
            for (size_t k = 0; k < t->count; k++) {
                if (t->items[k].id == item) {
                    found = true;
                    item_idx = k;
                    break;
                }
            }
            if (found) {
                u_beta_per_trans[beta_db.count] = u_alpha_per_trans[j] + t->items[item_idx].utility;
                
                // Copy both backward items and forward items
                size_t p_count = t->count - 1; // excluding the item itself
                if (p_count > 0) {
                    beta_db.transactions[beta_db.count].items = malloc(sizeof(EFC_Item) * p_count);
                    beta_db.transactions[beta_db.count].count = 0;
                    beta_db.transactions[beta_db.count].weight = t->weight;
                    double t_u = 0;
                    for (size_t k = 0; k < t->count; k++) {
                        if (k == item_idx) continue;
                        beta_db.transactions[beta_db.count].items[beta_db.transactions[beta_db.count].count++] = t->items[k];
                        if (k > item_idx) {
                            t_u += t->items[k].utility;
                        }
                    }
                    beta_db.transactions[beta_db.count].total_utility = t_u;
                    beta_db.count++;
                }
            }
        }
        
        merge_database(&beta_db, u_beta_per_trans);
        
        // Calculate sup_bins, su_bins, lu_bins for forward items
        double *su_bins = calloc(db->max_item_id + 1, sizeof(double));
        double *lu_bins = calloc(db->max_item_id + 1, sizeof(double));
        uint32_t *sup_bins = calloc(db->max_item_id + 1, sizeof(uint32_t));
        
        for (size_t j = 0; j < beta_db.count; j++) {
            EFC_Transaction *t = &beta_db.transactions[j];
            double remaining = 0;
            for (size_t k = t->count; k-- > 0; ) {
                uint32_t id = t->items[k].id;
                if (rank[id] < item_rank) {
                    sup_bins[id] += (uint32_t)t->weight;
                } else {
                    lu_bins[id] += u_beta_per_trans[j] + t->total_utility;
                    su_bins[id] += u_beta_per_trans[j] + t->items[k].utility + remaining;
                    sup_bins[id] += (uint32_t)t->weight;
                    remaining += t->items[k].utility;
                }
            }
        }
        
        // Determine new primary and secondary items
        uint32_t *new_primary = malloc(sizeof(uint32_t) * sec_count);
        size_t new_prim_count = 0;
        uint32_t *new_secondary = malloc(sizeof(uint32_t) * sec_count);
        size_t new_sec_count = 0;
        
        bool has_forward = false;
        bool all_items_same_sup = true;
        size_t forward_items_checked = 0;
        
        // Find index of 'item' in 'secondary'
        size_t item_idx_in_sec = 0;
        for (size_t j = 0; j < sec_count; j++) {
            if (secondary[j] == item) {
                item_idx_in_sec = j;
                break;
            }
        }
        
        for (size_t j = item_idx_in_sec + 1; j < sec_count; j++) {
            uint32_t z = secondary[j];
            forward_items_checked++;
            if (sup_bins[z] == sup_beta) {
                has_forward = true;
            } else {
                all_items_same_sup = false;
            }
            
            if (lu_bins[z] >= min_util) {
                new_secondary[new_sec_count++] = z;
                if (su_bins[z] >= min_util) {
                    new_primary[new_prim_count++] = z;
                }
            }
        }
        
        if (forward_items_checked == 0) {
            all_items_same_sup = true; // vacuously true
        }
        
        // CJU: Closure Jumping
        if (has_forward && all_items_same_sup) {
            // Calculate closure utility
            double jump_util = total_u_beta;
            for (size_t j = 0; j < beta_db.count; j++) {
                jump_util += beta_db.transactions[j].total_utility;
            }
            if (jump_util >= min_util) {
                total_chui_count++;
                total_chui_items += (alpha_len + 1 + (sec_count - 1 - item_idx_in_sec));

                FILE *f = fopen("efim_results.txt", "a");
                if (f) {
                    size_t len = alpha_len + 1 + (sec_count - 1 - item_idx_in_sec);
                    uint32_t *sorted_items = malloc(sizeof(uint32_t) * len);
                    if (sorted_items) {
                        memcpy(sorted_items, beta, sizeof(uint32_t) * (alpha_len + 1));
                        size_t idx = alpha_len + 1;
                        for (size_t j = item_idx_in_sec + 1; j < sec_count; j++) {
                            sorted_items[idx++] = secondary[j];
                        }
                        for (size_t x = 0; x < len; x++) {
                            for (size_t y = x + 1; y < len; y++) {
                                if (sorted_items[x] > sorted_items[y]) {
                                    uint32_t tmp = sorted_items[x];
                                    sorted_items[x] = sorted_items[y];
                                    sorted_items[y] = tmp;
                                }
                            }
                        }
                        for (size_t x = 0; x < len; x++) {
                            fprintf(f, "%u ", sorted_items[x]);
                        }
                        fprintf(f, "#UTIL: %.2f\n", jump_util);
                    }
                    free(sorted_items);
                    fclose(f);
                }
            }
        } else {
            if (new_prim_count > 0) {
                // Filter beta_db to keep only valid forward and backward items
                bool *is_sec = calloc(beta_db.max_item_id + 1, sizeof(bool));
                if (is_sec) {
                    for (size_t j = 0; j < new_sec_count; j++) {
                        is_sec[new_secondary[j]] = true;
                    }
                    for (size_t j = 0; j < beta_db.count; j++) {
                        EFC_Transaction *t = &beta_db.transactions[j];
                        size_t write_idx = 0;
                        double t_u = 0;
                        for (size_t k = 0; k < t->count; k++) {
                            uint32_t id = t->items[k].id;
                            if (rank[id] < item_rank) {
                                // Backward item: keep it
                                t->items[write_idx++] = t->items[k];
                            } else if (is_sec[id]) {
                                // Forward item in Secondary(beta): keep it and add utility to t_u
                                t->items[write_idx++] = t->items[k];
                                t_u += t->items[k].utility;
                            }
                        }
                        t->count = write_idx;
                        t->total_utility = t_u;
                    }
                    free(is_sec);
                }
                
                efc_search(beta, alpha_len + 1, &beta_db, u_beta_per_trans, new_primary, new_prim_count, new_secondary, new_sec_count, min_util);
            }
            if (!has_forward && total_u_beta >= min_util) {
                total_chui_count++;
                total_chui_items += (alpha_len + 1);

                FILE *f = fopen("efim_results.txt", "a");
                if (f) {
                    size_t len = alpha_len + 1;
                    uint32_t *sorted_items = malloc(sizeof(uint32_t) * len);
                    if (sorted_items) {
                        memcpy(sorted_items, beta, sizeof(uint32_t) * len);
                        for (size_t x = 0; x < len; x++) {
                            for (size_t y = x + 1; y < len; y++) {
                                if (sorted_items[x] > sorted_items[y]) {
                                    uint32_t tmp = sorted_items[x];
                                    sorted_items[x] = sorted_items[y];
                                    sorted_items[y] = tmp;
                                }
                            }
                        }
                        for (size_t x = 0; x < len; x++) {
                            fprintf(f, "%u ", sorted_items[x]);
                        }
                        fprintf(f, "#UTIL: %.2f\n", total_u_beta);
                    }
                    free(sorted_items);
                    fclose(f);
                }
            }
        }
        
        // Free resources
        free(su_bins); free(lu_bins); free(sup_bins);
        free(new_primary); free(new_secondary);
        
        for (size_t j = 0; j < beta_db.count; j++) {
            if (beta_db.transactions[j].items) {
                free(beta_db.transactions[j].items);
            }
        }
        free(beta_db.transactions);
        free(u_beta_per_trans);
        free(beta);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_EFIM_Closed_Params *p = (DM_EFIM_Closed_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;

    FILE *f_init = fopen("efim_results.txt", "w");
    if (f_init) fclose(f_init);

    DM_Trans_Utility *src_data = (DM_Trans_Utility *)ds->payload;
    total_chui_count = 0;
    total_chui_items = 0;

    double *twu_counts = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src_data[i].count; j++) {
            twu_counts[src_data[i].items[j].id] += src_data[i].total_utility;
        }
    }

    ItemTWU *items = malloc(sizeof(ItemTWU) * (ds->max_id + 1));
    size_t item_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu_counts[i] >= min_util) {
            items[item_count].id = i;
            items[item_count].twu = twu_counts[i];
            item_count++;
        }
    }
    qsort(items, item_count, sizeof(ItemTWU), cmp_item_twu_static);

    rank = malloc(sizeof(uint32_t) * (ds->max_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (ds->max_id + 1));
    uint32_t *secondary = malloc(sizeof(uint32_t) * item_count);
    for (size_t i = 0; i < item_count; i++) {
        rank[items[i].id] = i;
        secondary[i] = items[i].id;
    }

    EFC_Database db;
    db.max_item_id = ds->max_id;
    db.transactions = malloc(sizeof(EFC_Transaction) * ds->count);
    db.count = 0;
    double *u_alpha_per_trans = calloc(ds->count, sizeof(double));

    for (size_t i = 0; i < ds->count; i++) {
        size_t valid_count = 0;
        for (size_t j = 0; j < src_data[i].count; j++) {
            if (rank[src_data[i].items[j].id] != 0xFFFFFFFF) valid_count++;
        }
        if (valid_count > 0) {
            db.transactions[db.count].items = malloc(sizeof(EFC_Item) * valid_count);
            db.transactions[db.count].count = 0;
            db.transactions[db.count].weight = 1;
            double t_u = 0;
            for (size_t j = 0; j < src_data[i].count; j++) {
                if (rank[src_data[i].items[j].id] != 0xFFFFFFFF) {
                    db.transactions[db.count].items[db.transactions[db.count].count].id = src_data[i].items[j].id;
                    db.transactions[db.count].items[db.transactions[db.count].count].utility = src_data[i].items[j].utility;
                    t_u += src_data[i].items[j].utility;
                    db.transactions[db.count].count++;
                }
            }
            db.transactions[db.count].total_utility = t_u;
            qsort(db.transactions[db.count].items, db.transactions[db.count].count, sizeof(EFC_Item), cmp_efc_items);
            db.count++;
        }
    }

    merge_database(&db, u_alpha_per_trans);

    double *su_bins = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < db.count; i++) {
        EFC_Transaction *t = &db.transactions[i];
        double remaining = 0;
        for (size_t j = t->count; j-- > 0; ) {
            su_bins[t->items[j].id] += t->items[j].utility + remaining;
            remaining += t->items[j].utility;
        }
    }

    uint32_t *primary = malloc(sizeof(uint32_t) * item_count);
    size_t prim_count = 0;
    for (size_t i = 0; i < item_count; i++) {
        if (su_bins[secondary[i]] >= min_util) primary[prim_count++] = secondary[i];
    }

    efc_search(NULL, 0, &db, u_alpha_per_trans, primary, prim_count, secondary, item_count, min_util);

    printf("[EFIM-Closed] Found %zu Closed High Utility Itemsets.\n", total_chui_count);

    for (size_t i = 0; i < db.count; i++) {
        if (db.transactions[i].items) free(db.transactions[i].items);
    }
    free(db.transactions);
    free(u_alpha_per_trans);
    free(primary); free(secondary);
    free(rank); free(twu_counts); free(items); free(su_bins);

    dm_bench_record_results(total_chui_count, total_chui_items);
    return DM_SUCCESS;
}

static DM_Algorithm efc_algo = {
    .id = "efim_closed",
    .name = "EFIM-Closed Algorithm",
    .description = "Fast and Memory Efficient Discovery of Closed High-Utility Itemsets.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(efc_algo)
