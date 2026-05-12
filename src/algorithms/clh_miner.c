#include "algorithms/clh_miner.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

/* --- DATA STRUCTURES --- */

typedef struct {
    uint32_t tid;
    double iutil;
    double rutil;
} TaxUtilityTuple;

typedef struct {
    uint32_t *items;
    size_t item_count;
    TaxUtilityTuple *tuples;
    size_t tuple_count;
    double sum_iutil;
    double sum_rutil;
    uint32_t last; 
} TaxUtilityList;

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
static size_t clhui_count = 0;
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

static void calculate_levels(uint32_t node_id, int level) {
    nodes[node_id].level = level;
    for (size_t i = 0; i < nodes[node_id].child_count; i++) {
        calculate_levels(nodes[node_id].children[i], level + 1);
    }
}

/* --- UTILITY LIST HELPERS --- */

static TaxUtilityList* construct(TaxUtilityList *p, TaxUtilityList *px, TaxUtilityList *py) {
    TaxUtilityList *pxy = malloc(sizeof(TaxUtilityList));
    pxy->item_count = px->item_count + 1;
    pxy->items = malloc(sizeof(uint32_t) * pxy->item_count);
    memcpy(pxy->items, px->items, sizeof(uint32_t) * px->item_count);
    pxy->items[px->item_count] = py->last;
    pxy->last = py->last;
    
    size_t capacity = px->tuple_count < py->tuple_count ? px->tuple_count : py->tuple_count;
    pxy->tuples = malloc(sizeof(TaxUtilityTuple) * capacity);
    pxy->tuple_count = 0;
    pxy->sum_iutil = 0;
    pxy->sum_rutil = 0;

    size_t ix = 0, iy = 0, ip = 0;
    while (ix < px->tuple_count && iy < py->tuple_count) {
        if (px->tuples[ix].tid == py->tuples[iy].tid) {
            uint32_t tid = px->tuples[ix].tid;
            double iutil = px->tuples[ix].iutil + py->tuples[iy].iutil;
            
            if (p != NULL) {
                while (ip < p->tuple_count && p->tuples[ip].tid < tid) ip++;
                if (ip < p->tuple_count && p->tuples[ip].tid == tid) {
                    iutil -= p->tuples[ip].iutil;
                }
            }
            
            pxy->tuples[pxy->tuple_count].tid = tid;
            pxy->tuples[pxy->tuple_count].iutil = iutil;
            pxy->tuples[pxy->tuple_count].rutil = py->tuples[iy].rutil;
            pxy->sum_iutil += iutil;
            pxy->sum_rutil += py->tuples[iy].rutil;
            pxy->tuple_count++;
            ix++; iy++;
        } else if (px->tuples[ix].tid < py->tuples[iy].tid) ix++;
        else iy++;
    }
    return pxy;
}

static void free_tu_list(TaxUtilityList *ul) {
    if (!ul) return;
    free(ul->items);
    free(ul->tuples);
    free(ul);
}

/* --- RECURSIVE SEARCH --- */

static void search(TaxUtilityList **tuls, size_t count, double min_util, TaxUtilityList *prefix_tul) {
    for (size_t i = 0; i < count; i++) {
        TaxUtilityList *x = tuls[i];
        
        if (x->sum_iutil >= min_util) {
            clhui_count++;
            total_items_sum += x->item_count;
        }

        TaxUtilityList **extensions = malloc(sizeof(TaxUtilityList*) * (count - i - 1));
        size_t ext_count = 0;

        for (size_t j = i + 1; j < count; j++) {
            TaxUtilityList *y = tuls[j];
            
            bool can_add = true;
            for (size_t k = 0; k < x->item_count; k++) {
                if (is_descendant(y->last, x->items[k]) || is_descendant(x->items[k], y->last)) {
                    can_add = false;
                    break;
                }
            }

            if (can_add) {
                TaxUtilityList *pxy = construct(prefix_tul, x, y);
                extensions[ext_count++] = pxy;
            }
        }

        if (ext_count > 0) {
            if (x->sum_iutil + x->sum_rutil >= min_util) {
                search(extensions, ext_count, min_util, x);
            }
            for (size_t j = 0; j < ext_count; j++) free_tu_list(extensions[j]);
        }
        free(extensions);
    }
}

/* --- MAIN RUN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_CLH_Miner_Params *p = (DM_CLH_Miner_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    const char *tax_path = p ? p->taxonomy_path : NULL;

    max_node_id = ds->max_id;
    // Initial nodes
    nodes = calloc(max_node_id + 5001, sizeof(TaxonomyNode)); 
    for(size_t i=0; i <= max_node_id+5000; i++) {
        nodes[i].id = (uint32_t)i;
        nodes[i].parent = 0xFFFFFFFF;
    }

    if (tax_path) {
        FILE *f = fopen(tax_path, "r");
        if (f) {
            uint32_t child, parent;
            while (fscanf(f, "%u,%u", &child, &parent) == 2) {
                if (child > max_node_id + 5000 || parent > max_node_id + 5000) continue; 
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

    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    bool *present = malloc(sizeof(bool) * (max_node_id + 1));
    for (size_t i = 0; i < ds->count; i++) {
        memset(present, 0, sizeof(bool) * (max_node_id + 1));
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t curr = data[i].items[j].id;
            while (curr != 0xFFFFFFFF && curr != 0) {
                present[curr] = true;
                curr = nodes[curr].parent;
            }
        }
        for (uint32_t j = 1; j <= max_node_id; j++) {
            if (present[j]) nodes[j].gwu += data[i].total_utility;
        }
    }
    free(present);

    uint32_t *promising_ids = malloc(sizeof(uint32_t) * (max_node_id + 1));
    size_t prom_count = 0;
    for (uint32_t i = 1; i <= max_node_id; i++) {
        if (nodes[i].gwu >= min_util) {
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

    TaxUtilityList **initial_tuls = malloc(sizeof(TaxUtilityList*) * prom_count);
    for (size_t i = 0; i < prom_count; i++) {
        initial_tuls[i] = malloc(sizeof(TaxUtilityList));
        initial_tuls[i]->item_count = 1;
        initial_tuls[i]->items = malloc(sizeof(uint32_t));
        initial_tuls[i]->items[0] = promising_ids[i];
        initial_tuls[i]->last = promising_ids[i];
        initial_tuls[i]->tuples = malloc(sizeof(TaxUtilityTuple) * 8);
        initial_tuls[i]->tuple_count = 0;
        initial_tuls[i]->sum_iutil = 0;
        initial_tuls[i]->sum_rutil = 0;
    }

    double *node_utils = malloc(sizeof(double) * (max_node_id + 1));
    for (size_t i = 0; i < ds->count; i++) {
        memset(node_utils, 0, sizeof(double) * (max_node_id + 1));
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t curr = data[i].items[j].id;
            double u = data[i].items[j].utility;
            while (curr != 0xFFFFFFFF && curr != 0) {
                node_utils[curr] += u;
                curr = nodes[curr].parent;
            }
        }

        for (size_t j = 0; j < prom_count; j++) {
            uint32_t id = promising_ids[j];
            if (node_utils[id] > 0) {
                double rutil = 0;
                for (size_t k = j + 1; k < prom_count; k++) {
                    uint32_t next_id = promising_ids[k];
                    if (node_utils[next_id] > 0 && !is_descendant(next_id, id)) {
                        rutil += node_utils[next_id];
                    }
                }
                
                TaxUtilityList *ul = initial_tuls[j];
                if (ul->tuple_count > 0 && ul->tuple_count % 8 == 0) {
                    ul->tuples = realloc(ul->tuples, sizeof(TaxUtilityTuple) * (ul->tuple_count + 8));
                }
                ul->tuples[ul->tuple_count].tid = (uint32_t)i;
                ul->tuples[ul->tuple_count].iutil = node_utils[id];
                ul->tuples[ul->tuple_count].rutil = rutil;
                ul->sum_iutil += node_utils[id];
                ul->sum_rutil += rutil;
                ul->tuple_count++;
            }
        }
    }
    free(node_utils);

    clhui_count = 0;
    total_items_sum = 0;
    search(initial_tuls, prom_count, min_util, NULL);

    printf("[CLH-Miner] Found %zu Cross-Level High Utility Itemsets.\n", clhui_count);

    for (size_t i = 0; i < prom_count; i++) free_tu_list(initial_tuls[i]);
    free(initial_tuls); free(promising_ids); free(rank);
    for (size_t i = 0; i <= max_node_id; i++) free(nodes[i].children);
    free(nodes);

    dm_bench_record_results(clhui_count, total_items_sum);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "clhminer",
    .name = "CLH-Miner",
    .description = "Mines Cross-Level High Utility Itemsets using a taxonomy and tax-utility-lists.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
