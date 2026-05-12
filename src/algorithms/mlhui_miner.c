#include "algorithms/mlhui_miner.h"
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
} UtilityTuple;

typedef struct {
    uint32_t *items;
    size_t count;
    UtilityTuple *tuples;
    size_t tuple_count;
    double sum_iutil;
    double sum_rutil;
} UtilityList;

typedef struct {
    uint32_t id;
    uint32_t parent;
    uint32_t *children;
    size_t child_count;
    int level;
    double gwu;
    bool is_promising;
} TaxonomyNode;

typedef struct {
    uint32_t u, v;
    double twu;
} EUCS_Entry;

typedef struct {
    double *matrix;
    size_t size;
} EUCS;

/* --- GLOBAL STATE --- */

static TaxonomyNode *nodes = NULL;
static size_t max_node_id = 0;
static uint32_t *rank = NULL;
static size_t mlhui_count = 0;
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

/* --- EUCS --- */

static void eucs_add(EUCS *eucs, uint32_t u, uint32_t v, double twu) {
    if (u >= eucs->size || v >= eucs->size) return;
    eucs->matrix[u * eucs->size + v] += twu;
    eucs->matrix[v * eucs->size + u] += twu;
}

static double eucs_get(EUCS *eucs, uint32_t u, uint32_t v) {
    if (u >= eucs->size || v >= eucs->size) return 0;
    return eucs->matrix[u * eucs->size + v];
}

/* --- UTILITY LIST JOIN --- */

static UtilityList* construct(UtilityList *p, UtilityList *px, UtilityList *py) {
    UtilityList *pxy = malloc(sizeof(UtilityList));
    pxy->count = px->count + 1;
    pxy->items = malloc(sizeof(uint32_t) * pxy->count);
    memcpy(pxy->items, px->items, sizeof(uint32_t) * px->count);
    pxy->items[px->count] = py->items[py->count - 1];
    
    size_t capacity = px->tuple_count < py->tuple_count ? px->tuple_count : py->tuple_count;
    pxy->tuples = malloc(sizeof(UtilityTuple) * capacity);
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

static void free_utility_list(UtilityList *ul) {
    if (!ul) return;
    free(ul->items);
    free(ul->tuples);
    free(ul);
}

/* --- RECURSIVE SEARCH --- */

static void search(UtilityList *p, UtilityList **extensions, size_t ext_count, double min_util, EUCS *eucs) {
    for (size_t i = 0; i < ext_count; i++) {
        UtilityList *px = extensions[i];
        if (px->sum_iutil >= min_util) {
            mlhui_count++;
            total_items_sum += px->count;
        }

        if (px->sum_iutil + px->sum_rutil >= min_util) {
            UtilityList **ext_px = malloc(sizeof(UtilityList*) * (ext_count - i - 1));
            size_t ext_px_count = 0;

            uint32_t last_x = px->items[px->count - 1];
            for (size_t j = i + 1; j < ext_count; j++) {
                UtilityList *py = extensions[j];
                uint32_t last_y = py->items[py->count - 1];
                
                // Conflict Check: No ancestor/descendant relationship in itemset
                // In MLHUI-Miner, we must check if last_y conflicts with ANY item in px
                bool conflict = false;
                for (size_t k = 0; k < px->count; k++) {
                    if (is_descendant(last_y, px->items[k]) || is_descendant(px->items[k], last_y)) {
                        conflict = true;
                        break;
                    }
                }

                if (!conflict && eucs_get(eucs, last_x, last_y) >= min_util) {
                    ext_px[ext_px_count++] = construct(p, px, py);
                }
            }

            if (ext_px_count > 0) {
                search(px, ext_px, ext_px_count, min_util, eucs);
                for (size_t j = 0; j < ext_px_count; j++) free_utility_list(ext_px[j]);
            }
            free(ext_px);
        }
    }
}

/* --- MAIN RUN --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_UTILITY) return DM_ERROR_INCOMPATIBLE;
    DM_MLHUI_Miner_Params *p = (DM_MLHUI_Miner_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    const char *tax_path = p ? p->taxonomy_path : NULL;

    max_node_id = ds->max_id;
    // Initial nodes (reserve extra space for potential categories not in dataset)
    nodes = calloc(max_node_id + 10001, sizeof(TaxonomyNode)); 
    for(size_t i=0; i <= max_node_id+10000; i++) {
        nodes[i].id = (uint32_t)i;
        nodes[i].parent = 0xFFFFFFFF;
    }

    if (tax_path) {
        FILE *f = fopen(tax_path, "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                uint32_t child, parent;
                if (sscanf(line, "%u,%u", &child, &parent) == 2 || sscanf(line, "%u %u", &child, &parent) == 2) {
                    if (child > max_node_id + 10000 || parent > max_node_id + 10000) continue; 
                    nodes[child].parent = parent;
                    nodes[parent].child_count++;
                    nodes[parent].children = realloc(nodes[parent].children, sizeof(uint32_t) * nodes[parent].child_count);
                    nodes[parent].children[nodes[parent].child_count-1] = child;
                    if (parent > max_node_id) max_node_id = (size_t)parent;
                }
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
            // Pruning: if an ancestor is not promising, then its children shouldn't be either?
            // Actually GWU property: GWU(parent) >= GWU(child). 
            // So if GWU(parent) < min_util, then GWU(child) < min_util.
            nodes[i].is_promising = true;
            promising_ids[prom_count++] = i;
        }
    }

    // Sort by GWU (ascending as in FHM/HUI-Miner)
    for (size_t i = 0; i < prom_count; i++) {
        for (size_t j = i + 1; j < prom_count; j++) {
            if (nodes[promising_ids[i]].gwu > nodes[promising_ids[j]].gwu) {
                uint32_t tmp = promising_ids[i];
                promising_ids[i] = promising_ids[j];
                promising_ids[j] = tmp;
            }
        }
    }
    
    rank = malloc(sizeof(uint32_t) * (max_node_id + 1));
    memset(rank, 0xFF, sizeof(uint32_t) * (max_node_id + 1));
    for (size_t i = 0; i < prom_count; i++) rank[promising_ids[i]] = (uint32_t)i;

    UtilityList **initial_ext = malloc(sizeof(UtilityList*) * prom_count);
    for (size_t i = 0; i < prom_count; i++) {
        initial_ext[i] = malloc(sizeof(UtilityList));
        initial_ext[i]->items = malloc(sizeof(uint32_t));
        initial_ext[i]->items[0] = promising_ids[i];
        initial_ext[i]->count = 1;
        initial_ext[i]->tuples = malloc(sizeof(UtilityTuple) * 8);
        initial_ext[i]->tuple_count = 0;
        initial_ext[i]->sum_iutil = 0;
        initial_ext[i]->sum_rutil = 0;
    }

    EUCS eucs;
    eucs.size = max_node_id + 1;
    eucs.matrix = calloc(eucs.size * eucs.size, sizeof(double));
    
    double *node_utils = malloc(sizeof(double) * (max_node_id + 1));
    uint32_t *t_prom = malloc(sizeof(uint32_t) * (max_node_id + 1));

    for (size_t i = 0; i < ds->count; i++) {
        memset(node_utils, 0, sizeof(double) * (max_node_id + 1));
        size_t t_prom_count = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            uint32_t curr = data[i].items[j].id;
            double u = data[i].items[j].utility;
            while (curr != 0xFFFFFFFF && curr != 0) {
                if (rank[curr] != 0xFFFFFFFF) {
                    if (node_utils[curr] == 0) t_prom[t_prom_count++] = curr;
                    node_utils[curr] += u;
                }
                curr = nodes[curr].parent;
            }
        }

        // Sort t_prom by rank
        for (size_t j = 0; j < t_prom_count; j++) {
            for (size_t k = j + 1; k < t_prom_count; k++) {
                if (rank[t_prom[j]] > rank[t_prom[k]]) {
                    uint32_t tmp = t_prom[j]; t_prom[j] = t_prom[k]; t_prom[k] = tmp;
                }
            }
        }

        double remaining = 0;
        for (size_t j = t_prom_count; j-- > 0; ) {
            uint32_t id = t_prom[j];
            UtilityList *ul = initial_ext[rank[id]];
            if (ul->tuple_count > 0 && ul->tuple_count % 8 == 0) {
                ul->tuples = realloc(ul->tuples, sizeof(UtilityTuple) * (ul->tuple_count + 8));
            }
            ul->tuples[ul->tuple_count].tid = (uint32_t)i;
            ul->tuples[ul->tuple_count].iutil = node_utils[id];
            ul->tuples[ul->tuple_count].rutil = remaining;
            ul->sum_iutil += node_utils[id];
            ul->sum_rutil += remaining;
            ul->tuple_count++;

            for (size_t k = 0; k < j; k++) {
                // EUCS only for non-conflicting pairs
                if (!is_descendant(t_prom[j], t_prom[k]) && !is_descendant(t_prom[k], t_prom[j])) {
                    eucs_add(&eucs, t_prom[j], t_prom[k], data[i].total_utility);
                }
            }
            remaining += node_utils[id];
        }
    }
    free(node_utils); free(t_prom);

    mlhui_count = 0;
    total_items_sum = 0;
    search(NULL, initial_ext, prom_count, min_util, &eucs);

    printf("[MLHUI-Miner] Found %zu Multi-Level High Utility Itemsets.\n", mlhui_count);

    for (size_t i = 0; i < prom_count; i++) free_utility_list(initial_ext[i]);
    free(initial_ext); free(promising_ids); free(rank);
    for (size_t i = 0; i <= max_node_id; i++) free(nodes[i].children);
    free(nodes); free(eucs.matrix);

    dm_bench_record_results(mlhui_count, total_items_sum);
    return DM_SUCCESS;
}

static DM_Algorithm mlhui_algo = {
    .id = "mlhui_miner",
    .name = "MLHUI-Miner",
    .description = "Mines Multi-Level High Utility Itemsets using a taxonomy and utility-lists (Cagliero et al. 2017).",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(mlhui_algo)
