#include "algorithms/ltm.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * Frequent Itemset Mining Algorithm Based on Linear Table (LTM).
 * Reference: Jun Lu, Wenhe Xu, Kailong Zhou, Zhicong Guo,
 * "Frequent Itemset Mining Algorithm Based on Linear Table", 
 * Journal of Database Management, Volume 34, Issue 1, 2023.
 */

typedef struct {
    uint32_t name;
    uint32_t child;
    uint32_t sibling;
    uint32_t parent;
    uint32_t prev_occurrence;
    uint32_t frequency;
    uint32_t *binary_bits; // Pointer to bitmask in pool
} LTM_Node;

typedef struct {
    uint32_t last_index;
} LTM_Header;

typedef struct {
    LTM_Node *nodes;
    size_t count;
    size_t capacity;
    LTM_Header *headers;
    uint32_t num_frequent;
    uint32_t words;
    uint32_t *item_to_bit;
    uint32_t *bit_to_item;
} LTM_Context;

static uint32_t get_node(LTM_Context *ctx, uint32_t parent_idx, uint32_t item_name) {
    uint32_t curr = ctx->nodes[parent_idx].child;
    while (curr != 0) {
        if (ctx->nodes[curr].name == item_name) return curr;
        curr = ctx->nodes[curr].sibling;
    }
    return 0;
}

static int cmp_items_desc(const void *a, const void *b, void *arg) {
    uint32_t *counts = (uint32_t*)arg;
    uint32_t i1 = *(uint32_t*)a;
    uint32_t i2 = *(uint32_t*)b;
    if (counts[i1] != counts[i2]) return (int)counts[i2] - (int)counts[i1];
    return (int)i2 - (int)i1;
}

// Helper for bitwise operations on bit arrays
static bool is_superset(uint32_t *a, uint32_t *b, uint32_t words) {
    for (uint32_t i = 0; i < words; i++) {
        if ((a[i] & b[i]) != b[i]) return false;
    }
    return true;
}

static void increment_bit_array(uint32_t *a, uint32_t words) {
    for (uint32_t i = 0; i < words; i++) {
        a[i]++;
        if (a[i] != 0) return;
    }
}

static void prune_bit_array(uint32_t *a, uint32_t words) {
    // Formula: A = A + (A & -A)
    // Find LSB
    bool carry = false;
    bool found_lsb = false;
    for (uint32_t i = 0; i < words; i++) {
        if (!found_lsb) {
            if (a[i] != 0) {
                uint32_t lsb = a[i] & -a[i];
                uint32_t next = a[i] + lsb;
                if (next < a[i]) carry = true;
                else carry = false;
                a[i] = next;
                found_lsb = true;
            }
        } else if (carry) {
            a[i]++;
            if (a[i] != 0) carry = false;
        }
    }
}

static int get_lsb_bit(uint32_t *a, uint32_t words) {
    for (uint32_t i = 0; i < words; i++) {
        if (a[i] != 0) {
#ifdef _MSC_VER
            unsigned long index;
            _BitScanForward(&index, a[i]);
            return i * 32 + (int)index;
#else
            return i * 32 + __builtin_ctz(a[i]);
#endif
        }
    }
    return -1;
}

static int count_set_bits(uint32_t *a, uint32_t words) {
    int count = 0;
    for (uint32_t i = 0; i < words; i++) {
#ifdef _MSC_VER
        count += __popcnt(a[i]);
#else
        count += __builtin_popcount(a[i]);
#endif
    }
    return count;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_LTM_Params *p = (DM_LTM_Params*)params;
    double ms_val = p ? p->min_support : 0.05;
    uint32_t min_sup = (ms_val < 1.0) ? (uint32_t)ceil(ms_val * ds->count) : (uint32_t)ms_val;
    if (min_sup == 0) min_sup = 1;

    printf("[LTM] Starting. Min Support: %u\n", min_sup);
    fflush(stdout);

    // 1. Preprocessing
    uint32_t *counts = calloc(ds->max_id + 1, sizeof(uint32_t));
    DM_Trans_Simple *data = (DM_Trans_Simple*)ds->payload;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) counts[data[i].items[j]]++;
    }

    uint32_t num_frequent = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) if (counts[i] >= min_sup) num_frequent++;

    if (num_frequent == 0) {
        printf("[LTM] No frequent items found.\n");
        free(counts);
        return DM_SUCCESS;
    }

    // Sort frequent items by support DESC
    typedef struct { uint32_t id; uint32_t count; } ItemCount;
    ItemCount *f_items = malloc(num_frequent * sizeof(ItemCount));
    uint32_t f_idx = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (counts[i] >= min_sup) {
            f_items[f_idx].id = i;
            f_items[f_idx].count = counts[i];
            f_idx++;
        }
    }
    int cmp_item_desc_local(const void *a, const void *b) {
        return (int)((ItemCount*)b)->count - (int)((ItemCount*)a)->count;
    }
    qsort(f_items, num_frequent, sizeof(ItemCount), cmp_item_desc_local);

    LTM_Context ctx;
    ctx.num_frequent = num_frequent;
    ctx.words = (num_frequent + 31) / 32;
    ctx.item_to_bit = malloc((ds->max_id + 1) * sizeof(uint32_t));
    memset(ctx.item_to_bit, 0xFF, (ds->max_id + 1) * sizeof(uint32_t));
    ctx.bit_to_item = malloc(num_frequent * sizeof(uint32_t));

    for (uint32_t i = 0; i < num_frequent; i++) {
        // Bit 0 is lowest support, Bit num_frequent-1 is highest support
        uint32_t bit_pos = (num_frequent - 1) - i;
        ctx.item_to_bit[f_items[i].id] = bit_pos;
        ctx.bit_to_item[bit_pos] = f_items[i].id;
    }

    // 2. Build Linear Table
    ctx.capacity = 1024 * 1024; // Initial capacity
    ctx.nodes = calloc(ctx.capacity, sizeof(LTM_Node));
    ctx.count = 1; // Index 0 is root
    ctx.headers = calloc(num_frequent, sizeof(LTM_Header));

    uint32_t *bit_pool = calloc(ctx.capacity * ctx.words, sizeof(uint32_t));
    for (size_t i = 0; i < ctx.capacity; i++) ctx.nodes[i].binary_bits = &bit_pool[i * ctx.words];

    for (size_t i = 0; i < ds->count; i++) {
        uint32_t *t_items = malloc(data[i].count * sizeof(uint32_t));
        uint32_t t_len = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (ctx.item_to_bit[data[i].items[j]] != 0xFFFFFFFF) {
                t_items[t_len++] = data[i].items[j];
            }
        }
        // Sort items in transaction by support DESC (highest bit first)
        int cmp_t(const void *a, const void *b) {
            uint32_t b1 = ctx.item_to_bit[*(uint32_t*)a];
            uint32_t b2 = ctx.item_to_bit[*(uint32_t*)b];
            return (int)b2 - (int)b1;
        }
        qsort(t_items, t_len, sizeof(uint32_t), cmp_t);

        uint32_t curr = 0;
        for (uint32_t j = 0; j < t_len; j++) {
            uint32_t item = t_items[j];
            uint32_t bit = ctx.item_to_bit[item];
            uint32_t next = get_node(&ctx, curr, item);
            if (next != 0) {
                ctx.nodes[next].frequency++;
                curr = next;
            } else {
                if (ctx.count >= ctx.capacity) {
                    size_t old_cap = ctx.capacity;
                    ctx.capacity *= 2;
                    ctx.nodes = realloc(ctx.nodes, ctx.capacity * sizeof(LTM_Node));
                    memset(ctx.nodes + old_cap, 0, old_cap * sizeof(LTM_Node));
                    uint32_t *new_pool = realloc(bit_pool, ctx.capacity * ctx.words * sizeof(uint32_t));
                    bit_pool = new_pool;
                    for (size_t k = 0; k < ctx.capacity; k++) ctx.nodes[k].binary_bits = &bit_pool[k * ctx.words];
                }
                uint32_t n = ctx.count++;
                ctx.nodes[n].name = item;
                ctx.nodes[n].parent = curr;
                ctx.nodes[n].frequency = 1;
                ctx.nodes[n].sibling = ctx.nodes[curr].child;
                ctx.nodes[curr].child = n;

                memcpy(ctx.nodes[n].binary_bits, ctx.nodes[curr].binary_bits, ctx.words * sizeof(uint32_t));
                ctx.nodes[n].binary_bits[bit / 32] |= (1 << (bit % 32));

                ctx.nodes[n].prev_occurrence = ctx.headers[bit].last_index;
                ctx.headers[bit].last_index = n;

                curr = n;
            }
        }
        free(t_items);
    }

    // 3. Mining
    uint32_t *A = calloc(ctx.words, sizeof(uint32_t));
    A[0] = 1; // Start with first item (bit 0)

    size_t total_frequent = 0;
    size_t total_footprint = 0;

    // Boundary for A
    uint32_t *max_A = calloc(ctx.words, sizeof(uint32_t));
    for (uint32_t i = 0; i < num_frequent; i++) max_A[i / 32] |= (1 << (i % 32));

    while (true) {
        // Check if A > max_A
        bool overflow = false;
        for (int i = (int)ctx.words - 1; i >= 0; i--) {
            if (A[i] > max_A[i]) { overflow = true; break; }
            if (A[i] < max_A[i]) break;
            if (i == 0 && A[i] == max_A[i]) { /* exactly max, proceed */ }
        }
        if (overflow) break;

        int last_bit = get_lsb_bit(A, ctx.words);
        if (last_bit == -1 || last_bit >= (int)num_frequent) break;

        uint32_t support = 0;
        uint32_t node_idx = ctx.headers[last_bit].last_index;
        while (node_idx != 0) {
            if (is_superset(ctx.nodes[node_idx].binary_bits, A, ctx.words)) {
                support += ctx.nodes[node_idx].frequency;
            }
            node_idx = ctx.nodes[node_idx].prev_occurrence;
        }

        if (support >= min_sup) {
            total_frequent++;
            total_footprint += count_set_bits(A, ctx.words);
            increment_bit_array(A, ctx.words);
        } else {
            prune_bit_array(A, ctx.words);
        }
        
        // Safety break for very large search spaces if needed, but the paper implies this should run.
        // In practice, M must be small for this to finish.
    }

    printf("[LTM] Complete. FIs found: %zu\n", total_frequent);
    dm_bench_record_results(total_frequent, total_footprint);

    free(A); free(max_A);
    free(ctx.item_to_bit); free(ctx.bit_to_item);
    free(ctx.headers); free(ctx.nodes); free(bit_pool);
    free(counts); free(f_items);

    return DM_SUCCESS;
}

static DM_Algorithm algo_ltm = {
    .id = "ltm", .name = "Linear Table Miner",
    .description = "A frequent itemset mining algorithm based on a linear table and bitwise pruning.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL), .run = run
};
DM_REGISTER_ALGORITHM(algo_ltm)
