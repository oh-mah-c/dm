#include "algorithms/feacp.h"
#include "core/dm_dataset_types.h"
#include "core/dm_algorithm.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t id;
    double utility;
} FEACP_Item;

typedef struct {
    FEACP_Item *items;
    size_t count;
    double total_utility;
} FEACP_Transaction;

typedef struct {
    FEACP_Transaction *transactions;
    size_t count;
    uint32_t max_item_id;
} FEACP_Database;

typedef struct {
    FEACP_Transaction t;
    double u_alpha;
} FEACP_Entry;

typedef struct {
    uint32_t id;
    uint32_t parent;
    uint32_t *children;
    size_t child_count;
    int level;
    double gwu;
    bool is_promising;
} TaxonomyNode;

/* --- GLOBAL STATE --- */

static TaxonomyNode *nodes = NULL;
static size_t max_node_id = 0;
static uint32_t *rank = NULL;
static size_t feacp_count = 0;
static size_t total_items_sum = 0;

/* --- TAXONOMY HELPERS --- */

static bool is_descendant(uint32_t desc, uint32_t anc) {
    if (anc == 0 || anc == 0xFFFFFFFF || desc > max_node_id || anc > max_node_id) return false;
    uint32_t curr = desc;
    while (curr != 0xFFFFFFFF && curr != 0) {
        if (nodes[curr].parent == anc) return true;
        curr = nodes[curr].parent;
    }
    return false;
}

static bool has_descendant_in_set(uint32_t item, uint32_t *set, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (is_descendant(item, set[i]) || is_descendant(set[i], item)) return true;
    }
    return false;
}

static void calculate_levels(uint32_t node_id, int level) {
    nodes[node_id].level = level;
    for (size_t i = 0; i < nodes[node_id].child_count; i++) {
        calculate_levels(nodes[node_id].children[i], level + 1);
    }
}

/* --- EFIM-STYLE UTILS --- */

static int cmp_feacp_items(const void *a, const void *b) {
    uint32_t r1 = rank[((FEACP_Item*)a)->id];
    uint32_t r2 = rank[((FEACP_Item*)b)->id];
    return (r1 < r2) ? -1 : ((r1 > r2) ? 1 : 0);
}

static int cmp_entries(const void *a, const void *b) {
    const FEACP_Entry *e1 = (const FEACP_Entry *)a;
    const FEACP_Entry *e2 = (const FEACP_Entry *)b;
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

static void merge_database(FEACP_Database *db, double *u_alpha_per_trans) {
    if (db->count <= 1) return;
    
    FEACP_Entry *entries = malloc(sizeof(FEACP_Entry) * db->count);
    for (size_t i = 0; i < db->count; i++) {
        entries[i].t = db->transactions[i];
        entries[i].u_alpha = u_alpha_per_trans[i];
    }
    
    qsort(entries, db->count, sizeof(FEACP_Entry), cmp_entries);
    
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

/* --- FEACP CORE --- */

static void feacp_search(uint32_t *alpha, size_t alpha_len, FEACP_Database *db, double *u_alpha_per_trans, uint32_t *primary, size_t prim_count, uint32_t *secondary, size_t sec_count, double min_util) {
    for (size_t i = 0; i < prim_count; i++) {
        uint32_t item = primary[i];
        uint32_t item_rank = rank[item];
        
        uint32_t *beta = malloc(sizeof(uint32_t) * (alpha_len + 1));
        if (alpha_len > 0) memcpy(beta, alpha, sizeof(uint32_t) * alpha_len);
        beta[alpha_len] = item;

        double *u_beta_per_trans = malloc(sizeof(double) * db->count);
        double total_u_beta = 0;
        
        FEACP_Database beta_db;
        beta_db.max_item_id = db->max_item_id;
        beta_db.transactions = malloc(sizeof(FEACP_Transaction) * db->count);
        beta_db.count = 0;

        for (size_t j = 0; j < db->count; j++) {
            FEACP_Transaction *t = &db->transactions[j];
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
                
                // Projection: only include items after 'item' and not descendants of anything in beta
                size_t p_cap = t->count - 1 - item_idx;
                if (p_cap > 0) {
                    beta_db.transactions[beta_db.count].items = malloc(sizeof(FEACP_Item) * p_cap);
                    beta_db.transactions[beta_db.count].count = 0;
                    double t_u = 0;
                    for (size_t k = item_idx + 1; k < t->count; k++) {
                        uint32_t next_id = t->items[k].id;
                        // In FEACP, we should also check if next_id is descendant of the NEWLY added 'item'
                        // Since beta = alpha + {item}, and we already checked descendants for alpha.
                        if (!is_descendant(next_id, item) && !is_descendant(item, next_id)) {
                            beta_db.transactions[beta_db.count].items[beta_db.transactions[beta_db.count].count++] = t->items[k];
                            t_u += t->items[k].utility;
                        }
                    }
                    beta_db.transactions[beta_db.count].total_utility = t_u;
                    if (beta_db.transactions[beta_db.count].count > 0) {
                        beta_db.count++;
                    } else {
                        free(beta_db.transactions[beta_db.count].items);
                    }
                }
            }
        }

        if (total_u_beta >= min_util) {
            feacp_count++;
            total_items_sum += (alpha_len + 1);
        }

        if (beta_db.count > 0) {
            merge_database(&beta_db, u_beta_per_trans);

            double *su_bins = calloc(max_node_id + 1, sizeof(double));
            double *lu_bins = calloc(max_node_id + 1, sizeof(double));

            for (size_t j = 0; j < beta_db.count; j++) {
                FEACP_Transaction *t = &beta_db.transactions[j];
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
                if (rank[z] == 0xFFFFFFFF || rank[z] <= item_rank) continue;
                // Descendant check: z must not be descendant of any item in beta
                if (has_descendant_in_set(z, beta, alpha_len + 1)) continue;

                if (lu_bins[z] >= min_util) {
                    new_secondary[new_sec_count++] = z;
                    if (su_bins[z] >= min_util) {
                        new_primary[new_prim_count++] = z;
                    }
                }
            }

            if (new_prim_count > 0) {
                feacp_search(beta, alpha_len + 1, &beta_db, u_beta_per_trans, new_primary, new_prim_count, new_secondary, new_sec_count, min_util);
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
    DM_FEACP_Params *p = (DM_FEACP_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    const char *tax_path = p ? p->taxonomy_path : NULL;

    max_node_id = ds->max_id;
    nodes = calloc(max_node_id + 10001, sizeof(TaxonomyNode)); 
    for(size_t i=0; i <= max_node_id+10000; i++) {
        nodes[i].id = (uint32_t)i;
        nodes[i].parent = 0xFFFFFFFF;
    }

    if (tax_path) {
        FILE *f = fopen(tax_path, "r");
        if (f) {
            uint32_t child, parent;
            while (fscanf(f, "%u,%u", &child, &parent) == 2) {
                if (child > max_node_id + 10000 || parent > max_node_id + 10000) continue; 
                nodes[child].parent = parent;
                nodes[parent].child_count++;
                nodes[parent].children = realloc(nodes[parent].children, sizeof(uint32_t) * nodes[parent].child_count);
                nodes[parent].children[nodes[parent].child_count-1] = child;
                if (parent > max_node_id) max_node_id = (size_t)parent;
            }
            fclose(f);
        }
    }

    for (uint32_t i = 0; i <= max_node_id; i++) {
        if (nodes[i].parent == 0xFFFFFFFF || nodes[i].parent == 0) {
            calculate_levels(i, 1);
        }
    }

    DM_Trans_Utility *src_data = (DM_Trans_Utility *)ds->payload;
    feacp_count = 0;
    total_items_sum = 0;

    // Phase 1: GTWU and initial pruning
    bool *present = malloc(sizeof(bool) * (max_node_id + 1));
    for (size_t i = 0; i < ds->count; i++) {
        memset(present, 0, sizeof(bool) * (max_node_id + 1));
        for (size_t j = 0; j < src_data[i].count; j++) {
            uint32_t curr = src_data[i].items[j].id;
            while (curr != 0xFFFFFFFF && curr != 0) {
                present[curr] = true;
                curr = nodes[curr].parent;
            }
        }
        for (uint32_t j = 1; j <= max_node_id; j++) {
            if (present[j]) nodes[j].gwu += src_data[i].total_utility;
        }
    }
    free(present);

    uint32_t *promising_ids = malloc(sizeof(uint32_t) * (max_node_id + 1));
    size_t prom_count = 0;
    for (uint32_t i = 1; i <= max_node_id; i++) {
        if (nodes[i].gwu >= min_util) {
            // Inherit pruning from parents
            bool all_anc = true;
            uint32_t curr = nodes[i].parent;
            while (curr != 0xFFFFFFFF && curr != 0) {
                if (nodes[curr].gwu < min_util) { all_anc = false; break; }
                curr = nodes[curr].parent;
            }
            if (all_anc) {
                nodes[i].is_promising = true;
                promising_ids[prom_count++] = i;
            }
        }
    }

    // Sorting by Level then GTWU
    for (size_t i = 0; i < prom_count; i++) {
        for (size_t j = i + 1; j < prom_count; j++) {
            uint32_t id1 = promising_ids[i];
            uint32_t id2 = promising_ids[j];
            bool swap = false;
            if (nodes[id1].level > nodes[id2].level) swap = true;
            else if (nodes[id1].level == nodes[id2].level && nodes[id1].gwu > nodes[id2].gwu) swap = true;
            
            if (swap) {
                uint32_t tmp = promising_ids[i];
                promising_ids[i] = promising_ids[j];
                promising_ids[j] = tmp;
            }
        }
    }

    rank = malloc(sizeof(uint32_t) * (max_node_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (max_node_id + 1));
    for (size_t i = 0; i < prom_count; i++) rank[promising_ids[i]] = (uint32_t)i;

    // Phase 2: Build Database (Expanded with generalized items)
    FEACP_Database db;
    db.max_item_id = (uint32_t)max_node_id;
    db.transactions = malloc(sizeof(FEACP_Transaction) * ds->count);
    db.count = 0;
    double *u_alpha_per_trans = calloc(ds->count, sizeof(double));
    double *node_utils = malloc(sizeof(double) * (max_node_id + 1));

    for (size_t i = 0; i < ds->count; i++) {
        memset(node_utils, 0, sizeof(double) * (max_node_id + 1));
        size_t valid_count = 0;
        for (size_t j = 0; j < src_data[i].count; j++) {
            uint32_t curr = src_data[i].items[j].id;
            double u = src_data[i].items[j].utility;
            while (curr != 0xFFFFFFFF && curr != 0) {
                if (rank[curr] != 0xFFFFFFFF) {
                    if (node_utils[curr] == 0) valid_count++;
                    node_utils[curr] += u;
                }
                curr = nodes[curr].parent;
            }
        }

        if (valid_count > 0) {
            db.transactions[db.count].items = malloc(sizeof(FEACP_Item) * valid_count);
            db.transactions[db.count].count = 0;
            double t_u = 0;
            for (uint32_t j = 1; j <= max_node_id; j++) {
                if (node_utils[j] > 0) {
                    db.transactions[db.count].items[db.transactions[db.count].count].id = j;
                    db.transactions[db.count].items[db.transactions[db.count].count].utility = node_utils[j];
                    t_u += node_utils[j]; // Actually total_utility in EFIM is sum of items in transaction
                    db.transactions[db.count].count++;
                }
            }
            // In FEACP, the total_utility of a transaction is NOT simply sum of expanded items
            // because an itemset cannot have both parent and child.
            // However, EFIM uses total_utility for LU calculation. 
            // In cross-level, LU(P, v) = u(P) + re(P, T) where re(P, T) is items after v that don't conflict.
            // So we'll set total_utility = t_u for now and re-calculate precisely in search if needed.
            db.transactions[db.count].total_utility = t_u; 
            qsort(db.transactions[db.count].items, db.transactions[db.count].count, sizeof(FEACP_Item), cmp_feacp_items);
            db.count++;
        }
    }
    free(node_utils);

    merge_database(&db, u_alpha_per_trans);

    // Initial LU/SU calculation
    double *su_bins = calloc(max_node_id + 1, sizeof(double));
    for (size_t i = 0; i < db.count; i++) {
        FEACP_Transaction *t = &db.transactions[i];
        double remaining = 0;
        for (size_t j = t->count; j-- > 0; ) {
            su_bins[t->items[j].id] += t->items[j].utility + remaining;
            remaining += t->items[j].utility;
        }
    }

    uint32_t *primary = malloc(sizeof(uint32_t) * prom_count);
    size_t prim_cnt = 0;
    for (size_t i = 0; i < prom_count; i++) {
        if (su_bins[promising_ids[i]] >= min_util) {
            primary[prim_cnt++] = promising_ids[i];
        }
    }

    feacp_search(NULL, 0, &db, u_alpha_per_trans, primary, prim_cnt, promising_ids, prom_count, min_util);

    printf("[FEACP] Found %zu Cross-Level High Utility Itemsets.\n", feacp_count);

    // Cleanup
    for (size_t i = 0; i < db.count; i++) free(db.transactions[i].items);
    free(db.transactions);
    free(u_alpha_per_trans);
    free(primary); free(promising_ids); free(rank);
    for (size_t i = 0; i <= max_node_id; i++) free(nodes[i].children);
    free(nodes);
    free(su_bins);

    dm_bench_record_results(feacp_count, total_items_sum);
    return DM_SUCCESS;
}

static DM_Algorithm feacp_algo = {
    .id = "feacp",
    .name = "FEACP Algorithm",
    .description = "Fast Efficient Algorithm for Cross-level high-utility Pattern mining using EFIM-style optimizations.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(feacp_algo)
