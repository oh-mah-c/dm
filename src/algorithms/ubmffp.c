#include "algorithms/ubmffp.h"
#include "core/dm_dataset_types.h"
#include "core/dm_benchmark.h"
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

/**
 * UBMFFP-Tree Algorithm for Mining Multiple Fuzzy Frequent Itemsets.
 * Reference: Lin et al., "An UBMFFP Tree for Mining Multiple Fuzzy Frequent Itemsets", 
 * International Journal of Uncertainty, Fuzziness and Knowledge-Based Systems 23 (2015) 861–879.
 */

typedef enum { FUZZY_LOW = 0, FUZZY_MID = 1, FUZZY_HIGH = 2 } FuzzyTerm;

static double get_membership(double val, FuzzyTerm term) {
    if (term == FUZZY_LOW) {
        if (val <= 1.0) return 1.0;
        if (val >= 6.0) return 0.0;
        return (6.0 - val) / (6.0 - 1.0);
    } else if (term == FUZZY_MID) {
        if (val <= 1.0 || val >= 11.0) return 0.0;
        if (val <= 6.0) return (val - 1.0) / (6.0 - 1.0);
        return (11.0 - val) / (11.0 - 6.0);
    } else if (term == FUZZY_HIGH) {
        if (val <= 6.0) return 0.0;
        if (val >= 11.0) return 1.0;
        return (val - 6.0) / (11.0 - 6.0);
    }
    return 0.0;
}

#define REGION_ID(item_id, term) ((item_id) * 3 + (term))
#define ITEM_ID_FROM_REGION(rid) ((rid) / 3)

typedef struct UBMFFP_Node {
    uint32_t rid;
    double fuzzy_val;
    struct UBMFFP_Node *parent;
    struct UBMFFP_Node *children;
    struct UBMFFP_Node *sibling;
    struct UBMFFP_Node *next_homonym;
} UBMFFP_Node;

typedef struct {
    uint32_t rid;
    double support;
    uint32_t occurrence;
    UBMFFP_Node *first_node;
} HeaderEntry;

typedef struct {
    uint32_t *rids;
    size_t k;
    double upper_bound;
    double actual_support;
} Candidate;

static UBMFFP_Node* create_node(uint32_t rid, double val, UBMFFP_Node *parent) {
    UBMFFP_Node *node = malloc(sizeof(UBMFFP_Node));
    node->rid = rid;
    node->fuzzy_val = val;
    node->parent = parent;
    node->children = NULL;
    node->sibling = NULL;
    node->next_homonym = NULL;
    return node;
}

static int cmp_header(const void *a, const void *b) {
    const HeaderEntry *x = (const HeaderEntry *)a;
    const HeaderEntry *y = (const HeaderEntry *)b;
    if (x->occurrence != y->occurrence) return (y->occurrence - x->occurrence);
    return (int)y->rid - (int)x->rid;
}

static void add_candidate(uint32_t *items, size_t k, double upper_bound, Candidate **cands, size_t *count, size_t *cap) {
    if (*count >= *cap) {
        *cap *= 2;
        *cands = realloc(*cands, sizeof(Candidate) * (*cap));
    }
    (*cands)[*count].rids = malloc(sizeof(uint32_t) * k);
    memcpy((*cands)[*count].rids, items, sizeof(uint32_t) * k);
    (*cands)[*count].k = k;
    (*cands)[*count].upper_bound = upper_bound;
    (*cands)[*count].actual_support = 0;
    (*count)++;
}

static double threshold_global;
static int *rid_to_order_global;

static void generate_combinations(uint32_t *prefix, size_t prefix_len, int last_idx, uint32_t suffix_rid, double *ub_counts, HeaderEntry *L1, Candidate **cands, size_t *count, size_t *cap) {
    for (int i = last_idx + 1; i < (int)rid_to_order_global[suffix_rid]; i++) {
        if (ub_counts[i] >= threshold_global - 1e-9) {
            bool item_ok = true;
            uint32_t new_iid = ITEM_ID_FROM_REGION(L1[i].rid);
            if (new_iid == ITEM_ID_FROM_REGION(suffix_rid)) item_ok = false;
            else {
                for (size_t p = 0; p < prefix_len; p++) {
                    if (ITEM_ID_FROM_REGION(prefix[p]) == new_iid) { item_ok = false; break; }
                }
            }

            if (item_ok) {
                uint32_t *new_prefix = malloc(sizeof(uint32_t) * (prefix_len + 2));
                if (prefix_len > 0) memcpy(new_prefix, prefix, sizeof(uint32_t) * prefix_len);
                new_prefix[prefix_len] = L1[i].rid;
                new_prefix[prefix_len + 1] = suffix_rid;
                add_candidate(new_prefix, prefix_len + 2, ub_counts[i], cands, count, cap);
                
                // Recurse
                new_prefix[prefix_len + 1] = L1[i].rid; 
                generate_combinations(new_prefix, prefix_len + 1, i, suffix_rid, ub_counts, L1, cands, count, cap);
                free(new_prefix);
            }
        }
    }
}

static void free_tree(UBMFFP_Node *node) {
    if (!node) return;
    UBMFFP_Node *child = node->children;
    while (child) {
        UBMFFP_Node *next = child->sibling;
        free_tree(child);
        child = next;
    }
    free(node);
}

static DM_Status run(DM_Dataset *ds, void *params) {
    DM_UBMFFP_Params *p = (DM_UBMFFP_Params *)params;
    double delta = p ? p->min_support : 0.1;
    double threshold = delta * ds->count;
    threshold_global = threshold;

    printf("[UBMFFP-Tree] Starting. Delta: %.2f (Threshold: %.2f)\n", delta, threshold);

    HeaderEntry *full_header = calloc((ds->max_id + 1) * 3, sizeof(HeaderEntry));
    for (uint32_t i = 0; i < (ds->max_id + 1) * 3; i++) full_header[i].rid = i;

    for (size_t i = 0; i < ds->count; i++) {
        if (ds->type == DM_TYPE_TRANSACTIONAL) {
            DM_Trans_Simple *tr = &((DM_Trans_Simple *)ds->payload)[i];
            for (size_t j = 0; j < tr->count; j++) {
                for (int t = 0; t < 3; t++) {
                    double m = get_membership(1.0, (FuzzyTerm)t);
                    if (m > 0) {
                        full_header[REGION_ID(tr->items[j], t)].support += m;
                        full_header[REGION_ID(tr->items[j], t)].occurrence++;
                    }
                }
            }
        } else {
            DM_Trans_Utility *tr = &((DM_Trans_Utility *)ds->payload)[i];
            for (size_t j = 0; j < tr->count; j++) {
                for (int t = 0; t < 3; t++) {
                    double m = get_membership(tr->items[j].utility, (FuzzyTerm)t);
                    if (m > 0) {
                        full_header[REGION_ID(tr->items[j].id, t)].support += m;
                        full_header[REGION_ID(tr->items[j].id, t)].occurrence++;
                    }
                }
            }
        }
    }

    HeaderEntry *L1 = malloc(sizeof(HeaderEntry) * (ds->max_id + 1) * 3);
    size_t L1_count = 0;
    for (uint32_t i = 0; i < (ds->max_id + 1) * 3; i++) {
        if (full_header[i].support >= threshold - 1e-9) {
            L1[L1_count++] = full_header[i];
            L1[L1_count-1].first_node = NULL;
        }
    }
    qsort(L1, L1_count, sizeof(HeaderEntry), cmp_header);

    int *rid_to_order = malloc(sizeof(int) * (ds->max_id + 1) * 3);
    for (int i = 0; i < (ds->max_id + 1) * 3; i++) rid_to_order[i] = -1;
    for (size_t i = 0; i < L1_count; i++) rid_to_order[L1[i].rid] = (int)i;
    rid_to_order_global = rid_to_order;

    UBMFFP_Node *root = create_node(0xFFFFFFFF, 0, NULL);
    for (size_t i = 0; i < ds->count; i++) {
        typedef struct { int order; double val; uint32_t rid; } TransRegion;
        TransRegion tr_regions[256];
        size_t tr_count = 0;

        if (ds->type == DM_TYPE_TRANSACTIONAL) {
            DM_Trans_Simple *tr = &((DM_Trans_Simple *)ds->payload)[i];
            for (size_t j = 0; j < tr->count; j++) {
                for (int t = 0; t < 3; t++) {
                    uint32_t rid = REGION_ID(tr->items[j], t);
                    if (rid_to_order[rid] != -1) {
                        tr_regions[tr_count].order = rid_to_order[rid];
                        tr_regions[tr_count].rid = rid;
                        tr_regions[tr_count].val = get_membership(1.0, (FuzzyTerm)t);
                        tr_count++;
                    }
                }
            }
        } else {
            DM_Trans_Utility *tr = &((DM_Trans_Utility *)ds->payload)[i];
            for (size_t j = 0; j < tr->count; j++) {
                for (int t = 0; t < 3; t++) {
                    uint32_t rid = REGION_ID(tr->items[j].id, t);
                    if (rid_to_order[rid] != -1) {
                        tr_regions[tr_count].order = rid_to_order[rid];
                        tr_regions[tr_count].rid = rid;
                        tr_regions[tr_count].val = get_membership(tr->items[j].utility, (FuzzyTerm)t);
                        tr_count++;
                    }
                }
            }
        }

        for (size_t a = 0; a < tr_count; a++) {
            for (size_t b = a + 1; b < tr_count; b++) {
                if (tr_regions[a].order > tr_regions[b].order) {
                    TransRegion tmp = tr_regions[a]; tr_regions[a] = tr_regions[b]; tr_regions[b] = tmp;
                }
            }
        }

        UBMFFP_Node *curr = root;
        for (size_t j = 0; j < tr_count; j++) {
            UBMFFP_Node *child = curr->children;
            UBMFFP_Node *target = NULL;
            while (child) {
                if (child->rid == tr_regions[j].rid) { target = child; break; }
                child = child->sibling;
            }
            if (target) {
                target->fuzzy_val += tr_regions[j].val;
            } else {
                target = create_node(tr_regions[j].rid, tr_regions[j].val, curr);
                target->sibling = curr->children;
                curr->children = target;
                target->next_homonym = L1[tr_regions[j].order].first_node;
                L1[tr_regions[j].order].first_node = target;
            }
            curr = target;
        }
    }

    // 4. Phase 1: Candidate Generation
    size_t cand_cap = 1024, cand_count = 0;
    Candidate *candidates = malloc(sizeof(Candidate) * cand_cap);

    for (int i = (int)L1_count - 1; i >= 0; i--) {
        uint32_t suffix_rid = L1[i].rid;
        UBMFFP_Node *node = L1[i].first_node;
        double *ub_counts = calloc(L1_count, sizeof(double));
        while (node) {
            UBMFFP_Node *p_node = node->parent;
            while (p_node && p_node != root) {
                int order = rid_to_order[p_node->rid];
                ub_counts[order] += node->fuzzy_val;
                p_node = p_node->parent;
            }
            node = node->next_homonym;
        }
        generate_combinations(NULL, 0, -1, suffix_rid, ub_counts, L1, &candidates, &cand_count, &cand_cap);
        free(ub_counts);
    }

    // 5. Phase 2: Verification
    printf("[UBMFFP-Tree] Phase 1 found %zu candidates. Verifying...\n", cand_count);
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t c = 0; c < cand_count; c++) {
            Candidate *can = &candidates[c];
            double min_m = 2.0;
            size_t found_count = 0;

            if (ds->type == DM_TYPE_TRANSACTIONAL) {
                DM_Trans_Simple *tr = &((DM_Trans_Simple *)ds->payload)[i];
                for (size_t k = 0; k < can->k; k++) {
                    uint32_t rid = can->rids[k];
                    uint32_t iid = ITEM_ID_FROM_REGION(rid);
                    int term = rid % 3;
                    for (size_t j = 0; j < tr->count; j++) {
                        if (tr->items[j] == iid) {
                            double m = get_membership(1.0, (FuzzyTerm)term);
                            if (m < min_m) min_m = m;
                            found_count++; break;
                        }
                    }
                }
            } else {
                DM_Trans_Utility *tr = &((DM_Trans_Utility *)ds->payload)[i];
                for (size_t k = 0; k < can->k; k++) {
                    uint32_t rid = can->rids[k];
                    uint32_t iid = ITEM_ID_FROM_REGION(rid);
                    int term = rid % 3;
                    for (size_t j = 0; j < tr->count; j++) {
                        if (tr->items[j].id == iid) {
                            double m = get_membership(tr->items[j].utility, (FuzzyTerm)term);
                            if (m < min_m) min_m = m;
                            found_count++; break;
                        }
                    }
                }
            }
            if (found_count == can->k) can->actual_support += min_m;
        }
    }

    size_t final_count = L1_count;
    size_t total_footprint = L1_count;
    for (size_t i = 0; i < cand_count; i++) {
        if (candidates[i].actual_support >= threshold - 1e-9) {
            final_count++;
            total_footprint += candidates[i].k;
        }
        free(candidates[i].rids);
    }

    free_tree(root);
    free(candidates); free(rid_to_order); free(L1); free(full_header);

    printf("[UBMFFP-Tree] Complete. Total FFIs found: %zu\n", final_count);
    dm_bench_record_results(final_count, total_footprint);
    return DM_SUCCESS;
}

static DM_Algorithm algo = {
    .id = "ubmffp",
    .name = "UBMFFP-Tree Algorithm",
    .description = "Upper-bound Multiple Fuzzy Frequent Pattern Tree mining.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL) | (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(algo)
