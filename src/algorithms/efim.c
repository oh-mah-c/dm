#include "algorithms/efim.h"
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
} EFIM_Item;

typedef struct {
    EFIM_Item *items;
    size_t count;
    double total_utility;
} EFIM_Transaction;

typedef struct {
    EFIM_Transaction *transactions;
    size_t count;
    uint32_t max_item_id;
} EFIM_Database;

typedef struct {
    EFIM_Transaction t;
    double u_alpha;
} EFIM_Entry;

/* --- UTILS --- */

static uint32_t *rank = NULL;

static int cmp_efim_items(const void *a, const void *b) {
    uint32_t r1 = rank[((EFIM_Item*)a)->id];
    uint32_t r2 = rank[((EFIM_Item*)b)->id];
    return (r1 < r2) ? -1 : ((r1 > r2) ? 1 : 0);
}

static int cmp_entries(const void *a, const void *b) {
    const EFIM_Entry *e1 = (const EFIM_Entry *)a;
    const EFIM_Entry *e2 = (const EFIM_Entry *)b;
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

static void merge_database(EFIM_Database *db, double *u_alpha_per_trans) {
    if (db->count <= 1) return;
    
    EFIM_Entry *entries = malloc(sizeof(EFIM_Entry) * db->count);
    for (size_t i = 0; i < db->count; i++) {
        entries[i].t = db->transactions[i];
        entries[i].u_alpha = u_alpha_per_trans[i];
    }
    
    qsort(entries, db->count, sizeof(EFIM_Entry), cmp_entries);
    
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

/* --- EFIM CORE --- */

static size_t total_hui_count = 0;
static size_t total_items_sum = 0;

static void efim_search(uint32_t *alpha, size_t alpha_len, EFIM_Database *db, double *u_alpha_per_trans, uint32_t *primary, size_t prim_count, uint32_t *secondary, size_t sec_count, double min_util) {
    for (size_t i = 0; i < prim_count; i++) {
        uint32_t item = primary[i];
        uint32_t item_rank = rank[item];
        
        uint32_t *beta = malloc(sizeof(uint32_t) * (alpha_len + 1));
        if (alpha_len > 0) memcpy(beta, alpha, sizeof(uint32_t) * alpha_len);
        beta[alpha_len] = item;

        double *u_beta_per_trans = malloc(sizeof(double) * db->count);
        double total_u_beta = 0;
        
        EFIM_Database beta_db;
        beta_db.max_item_id = db->max_item_id;
        beta_db.transactions = malloc(sizeof(EFIM_Transaction) * db->count);
        beta_db.count = 0;

        for (size_t j = 0; j < db->count; j++) {
            EFIM_Transaction *t = &db->transactions[j];
            double u_item = 0;
            bool found = false;
            size_t item_idx = 0;
            for (size_t k = 0; k < t->count; k++) {
                if (t->items[k].id == item) {
                    u_item = t->items[k].utility;
                    found = true;
                    item_idx = k;
                    break;
                }
            }

            if (found) {
                u_beta_per_trans[beta_db.count] = u_alpha_per_trans[j] + u_item;
                total_u_beta += u_beta_per_trans[beta_db.count];
                
                size_t p_count = t->count - 1 - item_idx;
                if (p_count > 0) {
                    beta_db.transactions[beta_db.count].items = malloc(sizeof(EFIM_Item) * p_count);
                    beta_db.transactions[beta_db.count].count = 0;
                    double t_u = 0;
                    for (size_t k = item_idx + 1; k < t->count; k++) {
                        beta_db.transactions[beta_db.count].items[beta_db.transactions[beta_db.count].count++] = t->items[k];
                        t_u += t->items[k].utility;
                    }
                    beta_db.transactions[beta_db.count].total_utility = t_u;
                    beta_db.count++;
                }
            }
        }

        if (total_u_beta >= min_util) {
            total_hui_count++;
            total_items_sum += (alpha_len + 1);
        }

        if (beta_db.count > 0) {
            merge_database(&beta_db, u_beta_per_trans);

            double *su_bins = calloc(db->max_item_id + 1, sizeof(double));
            double *lu_bins = calloc(db->max_item_id + 1, sizeof(double));

            for (size_t j = 0; j < beta_db.count; j++) {
                EFIM_Transaction *t = &beta_db.transactions[j];
                double remaining = 0;
                for (size_t k = t->count; k-- > 0; ) {
                    uint32_t id = t->items[k].id;
                    lu_bins[id] += u_beta_per_trans[j] + t->total_utility;
                    su_bins[id] += u_beta_per_trans[j] + t->items[k].utility + remaining;
                    remaining += t->items[k].utility;
                }
            }

            uint32_t *new_primary = malloc(sizeof(uint32_t) * sec_count);
            size_t new_prim_count = 0;
            uint32_t *new_secondary = malloc(sizeof(uint32_t) * sec_count);
            size_t new_sec_count = 0;

            for (size_t j = 0; j < sec_count; j++) {
                uint32_t z = secondary[j];
                if (rank[z] <= item_rank) continue;
                if (lu_bins[z] >= min_util) {
                    new_secondary[new_sec_count++] = z;
                    if (su_bins[z] >= min_util) {
                        new_primary[new_prim_count++] = z;
                    }
                }
            }

            if (new_prim_count > 0) {
                efim_search(beta, alpha_len + 1, &beta_db, u_beta_per_trans, new_primary, new_prim_count, new_secondary, new_sec_count, min_util);
            }

            free(su_bins); free(lu_bins);
            free(new_primary); free(new_secondary);
        }

        for (size_t j = 0; j < beta_db.count; j++) free(beta_db.transactions[j].items);
        free(beta_db.transactions);
        free(u_beta_per_trans);
        free(beta);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    
    DM_EFIM_Params *p = (DM_EFIM_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    
    DM_Trans_Utility *src_data = (DM_Trans_Utility *)ds->payload;
    total_hui_count = 0;
    total_items_sum = 0;

    double *twu_counts = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src_data[i].count; j++) twu_counts[src_data[i].items[j].id] += src_data[i].total_utility;
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

    EFIM_Database db;
    db.max_item_id = ds->max_id;
    db.transactions = malloc(sizeof(EFIM_Transaction) * ds->count);
    db.count = 0;
    double *u_alpha_per_trans = calloc(ds->count, sizeof(double));

    for (size_t i = 0; i < ds->count; i++) {
        size_t valid_count = 0;
        for (size_t j = 0; j < src_data[i].count; j++) {
            if (rank[src_data[i].items[j].id] != 0xFFFFFFFF) valid_count++;
        }
        if (valid_count > 0) {
            db.transactions[db.count].items = malloc(sizeof(EFIM_Item) * valid_count);
            db.transactions[db.count].count = 0;
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
            qsort(db.transactions[db.count].items, db.transactions[db.count].count, sizeof(EFIM_Item), cmp_efim_items);
            db.count++;
        }
    }

    merge_database(&db, u_alpha_per_trans);

    double *su_bins = calloc(ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < db.count; i++) {
        EFIM_Transaction *t = &db.transactions[i];
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

    efim_search(NULL, 0, &db, u_alpha_per_trans, primary, prim_count, secondary, item_count, min_util);

    printf("[EFIM] Found %zu High Utility Itemsets.\n", total_hui_count);

    for (size_t i = 0; i < db.count; i++) free(db.transactions[i].items);
    free(db.transactions);
    free(u_alpha_per_trans);
    free(primary); free(secondary);
    free(rank); free(twu_counts); free(items); free(su_bins);

    dm_bench_record_results(total_hui_count, total_items_sum);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "efim",
    .name = "EFIM Algorithm",
    .description = "A Highly Efficient Algorithm for High-Utility Itemset Mining using Database Projection and Transaction Merging.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
