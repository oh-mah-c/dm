#include "algorithms/lcmver2.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * LCM ver. 2: Linear time Closed itemset Miner.
 * Reference: Takeaki Uno, Masashi Kiyomi, Hiroki Arimura, "LCM ver. 2: Efficient Mining Algorithms for 
 * Frequent/Closed/Maximal Itemsets", in Proc. FIMI'04, 2004.
 */

/* --- Data Structures --- */

typedef struct {
    uint32_t *items;
    uint32_t len;
    uint32_t weight;
} Transaction;

typedef struct {
    Transaction *trans;
    uint32_t count;
} Database;

typedef struct {
    uint32_t min_sup;
    int mode; // 0: All, 1: Closed, 2: Maximal
    size_t total_found;
    size_t total_footprint;
    uint32_t *rank_map;
    uint32_t *inv_map;
    uint32_t num_items;
} Context;

/* --- Utilities --- */

static void free_database(Database *db) {
    if (!db) return;
    for (uint32_t i = 0; i < db->count; i++) free(db->trans[i].items);
    free(db->trans);
}

static int cmp_uint32(const void *a, const void *b) {
    uint32_t ua = *(uint32_t*)a, ub = *(uint32_t*)b;
    return (ua < ub) ? -1 : (ua > ub ? 1 : 0);
}

static int cmp_trans(const void *a, const void *b) {
    const Transaction *ta = (const Transaction *)a;
    const Transaction *tb = (const Transaction *)b;
    uint32_t min_len = ta->len < tb->len ? ta->len : tb->len;
    for (uint32_t i = 0; i < min_len; i++) {
        if (ta->items[i] < tb->items[i]) return -1;
        if (ta->items[i] > tb->items[i]) return 1;
    }
    if (ta->len < tb->len) return -1;
    if (ta->len > tb->len) return 1;
    return 0;
}

/* --- Database Reduction --- */

static void reduce_db(Database *db, int32_t tail, uint32_t min_sup, uint32_t *out_in_all_len) {
    if (db->count == 0) { *out_in_all_len = 0; return; }

    uint32_t max_item = 0;
    for (uint32_t i = 0; i < db->count; i++)
        for (uint32_t j = 0; j < db->trans[i].len; j++)
            if (db->trans[i].items[j] > max_item) max_item = db->trans[i].items[j];

    uint32_t *counts = calloc(max_item + 1, sizeof(uint32_t));
    uint32_t total_weight = 0;
    for (uint32_t i = 0; i < db->count; i++) {
        total_weight += db->trans[i].weight;
        for (uint32_t j = 0; j < db->trans[i].len; j++)
            counts[db->trans[i].items[j]] += db->trans[i].weight;
    }

    uint32_t in_all_cnt = 0;
    for (uint32_t i = (uint32_t)(tail + 1); i <= max_item; i++)
        if (counts[i] == total_weight) in_all_cnt++;
    *out_in_all_len = in_all_cnt;

    for (uint32_t i = 0; i < db->count; i++) {
        uint32_t new_len = 0;
        for (uint32_t j = 0; j < db->trans[i].len; j++) {
            uint32_t item = db->trans[i].items[j];
            if ((int32_t)item > tail && counts[item] >= min_sup && counts[item] < total_weight) {
                db->trans[i].items[new_len++] = item;
            }
        }
        db->trans[i].len = new_len;
    }
    free(counts);

    if (db->count > 1) {
        qsort(db->trans, db->count, sizeof(Transaction), cmp_trans);
        uint32_t new_count = 0;
        for (uint32_t i = 0; i < db->count; i++) {
            if (new_count > 0 && cmp_trans(&db->trans[i], &db->trans[new_count - 1]) == 0) {
                db->trans[new_count - 1].weight += db->trans[i].weight;
                free(db->trans[i].items);
            } else {
                db->trans[new_count++] = db->trans[i];
            }
        }
        db->count = new_count;
    }
}

/* --- Algorithm 3.1: Enumerating Frequent Itemsets (LCMfreq) --- */

static size_t lcm_freq_recursive(int32_t tail, Database *db, uint32_t min_sup) {
    if (db->count == 0) return 0;

    uint32_t max_item = 0;
    for (uint32_t i = 0; i < db->count; i++)
        for (uint32_t j = 0; j < db->trans[i].len; j++)
            if (db->trans[i].items[j] > max_item) max_item = db->trans[i].items[j];

    if ((int32_t)max_item <= tail) return 0;

    uint32_t *counts = calloc(max_item + 1, sizeof(uint32_t));
    for (uint32_t i = 0; i < db->count; i++)
        for (uint32_t j = 0; j < db->trans[i].len; j++)
            counts[db->trans[i].items[j]] += db->trans[i].weight;

    size_t found_in_this_level = 0;
    for (uint32_t e = (uint32_t)(tail + 1); e <= max_item; e++) {
        if (counts[e] >= min_sup) {
            Database next_db;
            next_db.count = 0;
            next_db.trans = malloc(db->count * sizeof(Transaction));
            for (uint32_t i = 0; i < db->count; i++) {
                bool has_e = false;
                for (uint32_t j = 0; j < db->trans[i].len; j++) if (db->trans[i].items[j] == e) { has_e = true; break; }
                if (has_e) {
                    Transaction *t = &next_db.trans[next_db.count++];
                    t->weight = db->trans[i].weight;
                    t->len = 0;
                    t->items = malloc(db->trans[i].len * sizeof(uint32_t));
                    for (uint32_t j = 0; j < db->trans[i].len; j++) {
                        if ((int32_t)db->trans[i].items[j] > (int32_t)e)
                            t->items[t->len++] = db->trans[i].items[j];
                    }
                }
            }

            uint32_t in_all_len = 0;
            reduce_db(&next_db, (int32_t)e, min_sup, &in_all_len);

            size_t sub_found = lcm_freq_recursive((int32_t)e, &next_db, min_sup);
            
            // Total itemsets starting with e: (1 + sub_found) * 2^in_all_len
            size_t multiplier = (size_t)1 << in_all_len;
            found_in_this_level += (1 + sub_found) * multiplier;

            free_database(&next_db);
        }
    }
    free(counts);
    return found_in_this_level;
}

/* --- Entry Point --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_LCMVER2_Params *p = (DM_LCMVER2_Params*)params;
    double ms_val = p ? p->min_support : 0.01;
    uint32_t min_sup = (ms_val < 1.0) ? (uint32_t)ceil(ms_val * ds->count) : (uint32_t)ms_val;
    if (min_sup == 0) min_sup = 1;

    printf("[LCMv2] Starting. Min Support: %u, Mode: ALL\n", min_sup);

    uint32_t *raw_counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple*)ds->payload;
    for (size_t i = 0; i < ds->count; i++)
        for (size_t j = 0; j < data[i].count; j++) raw_counts[data[i].items[j]]++;

    uint32_t freq_cnt = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (raw_counts[i] >= min_sup) freq_cnt++;

    uint32_t *freq_items = malloc(freq_cnt * sizeof(uint32_t));
    uint32_t f_idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (raw_counts[i] >= min_sup) freq_items[f_idx++] = i;

    static uint32_t *g_s_counts; g_s_counts = raw_counts;
    int cmp_freq(const void *a, const void *b) {
        uint32_t ia = *(uint32_t*)a, ib = *(uint32_t*)b;
        if (g_s_counts[ia] < g_s_counts[ib]) return -1;
        if (g_s_counts[ia] > g_s_counts[ib]) return 1;
        return (ia < ib) ? -1 : 1;
    }
    qsort(freq_items, freq_cnt, sizeof(uint32_t), cmp_freq);

    uint32_t *rank_map = malloc((ds->max_id + 1) * sizeof(uint32_t));
    memset(rank_map, 0xFF, (ds->max_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i < freq_cnt; i++) rank_map[freq_items[i]] = i;

    Database db;
    db.count = (uint32_t)ds->count;
    db.trans = malloc(db.count * sizeof(Transaction));
    for (uint32_t i = 0; i < db.count; i++) {
        db.trans[i].len = 0;
        db.trans[i].items = malloc(data[i].count * sizeof(uint32_t));
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t item = data[i].items[j];
            if (rank_map[item] != 0xFFFFFFFF) db.trans[i].items[db.trans[i].len++] = rank_map[item];
        }
        db.trans[i].weight = 1;
        qsort(db.trans[i].items, db.trans[i].len, sizeof(uint32_t), cmp_uint32);
    }
    free(raw_counts);

    uint32_t in_all_len = 0;
    reduce_db(&db, -1, min_sup, &in_all_len);

    size_t core_found = lcm_freq_recursive(-1, &db, min_sup);
    size_t total_found = (1 + core_found) * ((size_t)1 << in_all_len) - 1;

    printf("[LCMv2] Complete. FIs found: %zu\n", total_found);
    dm_bench_record_results(total_found, 0);

    free_database(&db); free(freq_items); free(rank_map);
    return DM_SUCCESS;
}

static DM_Algorithm algo_lcmv2 = {
    .id = "lcmver2", .name = "LCM ver. 2 Algorithm",
    .description = "Efficient mining of Frequent/Closed/Maximal itemsets using database reduction and occurrence deliver.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL), .run = run
};
DM_REGISTER_ALGORITHM(algo_lcmv2)
