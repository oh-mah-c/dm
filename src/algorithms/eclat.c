#include "algorithms/eclat.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

/* --- INTERNAL DATA STRUCTURES (Vertical Format) --- */

typedef struct {
    uint32_t *tids;
    size_t count;
    size_t capacity;
} Tidset;

typedef struct {
    uint32_t item;
    Tidset tids;
} Eclat_Node;

typedef struct {
    Eclat_Node *nodes;
    size_t count;
    size_t capacity;
} Eclat_Class;

static void tidset_init(Tidset *ts, size_t capacity) {
    ts->capacity = capacity > 0 ? capacity : 8;
    ts->count = 0;
    ts->tids = (uint32_t *)malloc(sizeof(uint32_t) * ts->capacity);
}

static void tidset_add(Tidset *ts, uint32_t tid) {
    if (ts->count >= ts->capacity) {
        ts->capacity *= 2;
        ts->tids = (uint32_t *)realloc(ts->tids, sizeof(uint32_t) * ts->capacity);
    }
    ts->tids[ts->count++] = tid;
}

static void tidset_free(Tidset *ts) {
    if (ts->tids) free(ts->tids);
}

static void tidset_intersect(const Tidset *a, const Tidset *b, Tidset *out) {
    size_t i = 0, j = 0;
    while (i < a->count && j < b->count) {
        if (a->tids[i] == b->tids[j]) {
            out->tids[out->count++] = a->tids[i]; // Pre-allocated exactly
            i++; j++;
        } else if (a->tids[i] < b->tids[j]) {
            i++;
        } else {
            j++;
        }
    }
}

// Zaki's Optimization: Sort equivalence classes by ascending support to minimize intersection sizes
static int cmp_eclat_node_support_asc(const void *a, const void *b) {
    const Eclat_Node *na = (const Eclat_Node *)a;
    const Eclat_Node *nb = (const Eclat_Node *)b;
    if (na->tids.count < nb->tids.count) return -1;
    if (na->tids.count > nb->tids.count) return 1;
    if (na->item < nb->item) return -1;
    if (na->item > nb->item) return 1;
    return 0;
}

/* --- DFS CORE ALGORITHM --- */

static void eclat_mine(Eclat_Class *eq_class, size_t k, uint32_t min_sup, size_t *total_freq, size_t *total_footprint) {
    for (size_t i = 0; i < eq_class->count; i++) {
        // eq_class->nodes[i] is a frequent k-itemset
        (*total_freq)++;
        (*total_footprint) += k;

        Eclat_Class next_class;
        next_class.count = 0;
        next_class.capacity = eq_class->count - i - 1;
        
        if (next_class.capacity > 0) {
            next_class.nodes = (Eclat_Node *)malloc(sizeof(Eclat_Node) * next_class.capacity);
        } else {
            next_class.nodes = NULL;
        }

        for (size_t j = i + 1; j < eq_class->count; j++) {
            // Optimization: The intersection can never be larger than the smallest tidset
            size_t min_count = eq_class->nodes[i].tids.count < eq_class->nodes[j].tids.count ? 
                               eq_class->nodes[i].tids.count : eq_class->nodes[j].tids.count;
            
            if (min_count >= min_sup) {
                Tidset out;
                tidset_init(&out, min_count); // Pre-allocate accurately to avoid realloc overhead
                tidset_intersect(&eq_class->nodes[i].tids, &eq_class->nodes[j].tids, &out);
                
                if (out.count >= min_sup) {
                    next_class.nodes[next_class.count].item = eq_class->nodes[j].item;
                    next_class.nodes[next_class.count].tids = out;
                    next_class.count++;
                } else {
                    tidset_free(&out);
                }
            }
        }

        if (next_class.count > 0) {
            eclat_mine(&next_class, k + 1, min_sup, total_freq, total_footprint);
            
            // Clean up children's tidsets after backtracking
            for (size_t m = 0; m < next_class.count; m++) {
                tidset_free(&next_class.nodes[m].tids);
            }
        }
        
        if (next_class.nodes) {
            free(next_class.nodes);
        }
    }
}

/* --- MAIN ENTRY --- */

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_ECLAT_Params *p = (DM_ECLAT_Params *)params;
    double min_sup_param = p ? p->min_support : 0.01;
    uint32_t min_sup = (min_sup_param < 1.0) ? (uint32_t)ceil(min_sup_param * ds->count) : (uint32_t)min_sup_param;
    if (min_sup == 0) min_sup = 1;

    printf("[Eclat] Starting on %zu transactions. Min Support: %u\n", ds->count, min_sup);

    DM_Trans_Simple *data = (DM_Trans_Simple *)ds->payload;

    // 1. Convert Horizontal Database to Vertical Format
    Tidset *vertical_db = (Tidset *)calloc(ds->max_id + 1, sizeof(Tidset));
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        vertical_db[i].capacity = 0; 
    }

    for (size_t i = 0; i < ds->count; i++) {
        DM_Trans_Simple tr = data[i];
        for (size_t j = 0; j < tr.count; j++) {
            uint32_t item = tr.items[j];
            if (vertical_db[item].capacity == 0) {
                tidset_init(&vertical_db[item], 16);
            }
            // Ensure no duplicate TIDs are added if the dataset contains duplicate items in a transaction
            if (vertical_db[item].count == 0 || vertical_db[item].tids[vertical_db[item].count - 1] != (uint32_t)i) {
                tidset_add(&vertical_db[item], (uint32_t)i);
            }
        }
    }

    // 2. Generate L1 Equivalence Class
    Eclat_Class root_class;
    root_class.capacity = ds->max_id + 1;
    root_class.count = 0;
    root_class.nodes = (Eclat_Node *)malloc(sizeof(Eclat_Node) * root_class.capacity);

    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (vertical_db[i].capacity > 0 && vertical_db[i].count >= min_sup) {
            root_class.nodes[root_class.count].item = i;
            root_class.nodes[root_class.count].tids = vertical_db[i];
            root_class.count++;
        } else {
            if (vertical_db[i].capacity > 0) {
                tidset_free(&vertical_db[i]);
            }
        }
    }
    free(vertical_db);

    // 3. Optimization: Sort root class by ascending support (Zaki's heuristic)
    qsort(root_class.nodes, root_class.count, sizeof(Eclat_Node), cmp_eclat_node_support_asc);

    // 4. Start DFS Mining
    size_t total_freq = 0;
    size_t total_footprint = 0;
    
    eclat_mine(&root_class, 1, min_sup, &total_freq, &total_footprint);

    // 5. Clean up Root Class
    for (size_t i = 0; i < root_class.count; i++) {
        tidset_free(&root_class.nodes[i].tids);
    }
    free(root_class.nodes);

    printf("[Eclat] Complete. Total frequent itemsets found: %zu\n", total_freq);
    dm_bench_record_results(total_freq, total_footprint);

    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "eclat",
    .name = "Eclat Algorithm",
    .description = "Equivalence CLASS Transformation (Zaki KDD 1997) using Vertical Data Format.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
