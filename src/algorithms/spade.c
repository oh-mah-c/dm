#include "algorithms/spade.h"
#include "core/dm_dataset.h"
#include "core/dm_algorithm.h"
#include "core/dm_benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    SPADE_EVENT,
    SPADE_SEQUENCE
} SpadeAtomType;

typedef struct {
    uint32_t sid;
    uint32_t eid;
} SpadePair;

typedef struct {
    SpadePair *pairs;
    size_t count;
    size_t capacity;
    uint32_t support;
} SpadeIdList;

typedef struct {
    uint32_t item;
    SpadeAtomType type;
    SpadeIdList idlist;
} SpadeAtom;

typedef struct {
    SpadeAtom *atoms;
    size_t count;
    size_t capacity;
    uint32_t *prefix;
    size_t prefix_len;
} SpadeClass;

static size_t min_supp_count = 0;
static size_t found_count = 0;

static void idlist_init(SpadeIdList *list) {
    list->count = 0;
    list->capacity = 16;
    list->support = 0;
    list->pairs = malloc(sizeof(SpadePair) * list->capacity);
}

static void idlist_add(SpadeIdList *list, uint32_t sid, uint32_t eid) {
    // If it's the same as the last one, do we add it?
    // A sequence can occur multiple times in the same sid. We need all occurrences
    // to allow subsequent joins (e.g. A -> B -> B).
    // BUT we only count support once per sid!
    if (list->count == 0 || list->pairs[list->count - 1].sid != sid) {
        list->support++;
    }
    if (list->count >= list->capacity) {
        list->capacity *= 2;
        list->pairs = realloc(list->pairs, sizeof(SpadePair) * list->capacity);
    }
    list->pairs[list->count].sid = sid;
    list->pairs[list->count].eid = eid;
    list->count++;
}

static void idlist_free(SpadeIdList *list) {
    free(list->pairs);
}

static void class_init(SpadeClass *cls, uint32_t *prefix, size_t prefix_len) {
    cls->count = 0;
    cls->capacity = 16;
    cls->atoms = malloc(sizeof(SpadeAtom) * cls->capacity);
    cls->prefix_len = prefix_len;
    if (prefix_len > 0) {
        cls->prefix = malloc(sizeof(uint32_t) * prefix_len);
        memcpy(cls->prefix, prefix, sizeof(uint32_t) * prefix_len);
    } else {
        cls->prefix = NULL;
    }
}

static SpadeAtom* class_add_atom(SpadeClass *cls, uint32_t item, SpadeAtomType type) {
    if (cls->count >= cls->capacity) {
        cls->capacity *= 2;
        cls->atoms = realloc(cls->atoms, sizeof(SpadeAtom) * cls->capacity);
    }
    SpadeAtom *atom = &cls->atoms[cls->count++];
    atom->item = item;
    atom->type = type;
    idlist_init(&atom->idlist);
    return atom;
}

static void class_free(SpadeClass *cls) {
    for (size_t i = 0; i < cls->count; i++) {
        idlist_free(&cls->atoms[i].idlist);
    }
    free(cls->atoms);
    free(cls->prefix);
}

static void join_equality(SpadeIdList *l1, SpadeIdList *l2, SpadeIdList *out) {
    size_t i = 0, j = 0;
    while (i < l1->count && j < l2->count) {
        if (l1->pairs[i].sid < l2->pairs[j].sid) {
            i++;
        } else if (l1->pairs[i].sid > l2->pairs[j].sid) {
            j++;
        } else {
            uint32_t current_sid = l1->pairs[i].sid;
            size_t start_i = i, start_j = j;
            while (i < l1->count && l1->pairs[i].sid == current_sid) i++;
            while (j < l2->count && l2->pairs[j].sid == current_sid) j++;
            
            for (size_t x = start_i; x < i; x++) {
                for (size_t y = start_j; y < j; y++) {
                    if (l1->pairs[x].eid == l2->pairs[y].eid) {
                        idlist_add(out, current_sid, l1->pairs[x].eid);
                    }
                }
            }
        }
    }
}

static void join_temporal(SpadeIdList *l1, SpadeIdList *l2, SpadeIdList *out) {
    size_t i = 0, j = 0;
    while (i < l1->count && j < l2->count) {
        if (l1->pairs[i].sid < l2->pairs[j].sid) {
            i++;
        } else if (l1->pairs[i].sid > l2->pairs[j].sid) {
            j++;
        } else {
            uint32_t current_sid = l1->pairs[i].sid;
            size_t start_i = i, start_j = j;
            while (i < l1->count && l1->pairs[i].sid == current_sid) i++;
            while (j < l2->count && l2->pairs[j].sid == current_sid) j++;
            
            uint32_t min_t1 = l1->pairs[start_i].eid;
            for (size_t y = start_j; y < j; y++) {
                if (min_t1 < l2->pairs[y].eid) {
                    idlist_add(out, current_sid, l2->pairs[y].eid);
                }
            }
        }
    }
}

static void spade_recursive(SpadeClass *cls) {
    for (size_t i = 0; i < cls->count; i++) {
        SpadeAtom *a1 = &cls->atoms[i];
        SpadeClass next_cls; 
        class_init(&next_cls, NULL, 0); // Prefix array isn't needed for just counting!
        
        for (size_t j = 0; j < cls->count; j++) {
            SpadeAtom *a2 = &cls->atoms[j];
            
            if (a1->type == SPADE_EVENT && a2->type == SPADE_EVENT) {
                if (a1->item < a2->item) {
                    SpadeIdList out; idlist_init(&out);
                    join_equality(&a1->idlist, &a2->idlist, &out);
                    if (out.support >= min_supp_count) {
                        found_count++;
                        SpadeAtom *new_a = class_add_atom(&next_cls, a2->item, SPADE_EVENT);
                        idlist_free(&new_a->idlist);
                        new_a->idlist = out;
                    } else { idlist_free(&out); }
                }
            } else if (a1->type == SPADE_EVENT && a2->type == SPADE_SEQUENCE) {
                SpadeIdList out; idlist_init(&out);
                join_temporal(&a1->idlist, &a2->idlist, &out);
                if (out.support >= min_supp_count) {
                    found_count++;
                    SpadeAtom *new_a = class_add_atom(&next_cls, a2->item, SPADE_SEQUENCE);
                    idlist_free(&new_a->idlist);
                    new_a->idlist = out;
                } else { idlist_free(&out); }
            } else if (a1->type == SPADE_SEQUENCE && a2->type == SPADE_SEQUENCE) {
                if (a1->item < a2->item) {
                    SpadeIdList out_eq; idlist_init(&out_eq);
                    join_equality(&a1->idlist, &a2->idlist, &out_eq);
                    if (out_eq.support >= min_supp_count) {
                        found_count++;
                        SpadeAtom *new_a = class_add_atom(&next_cls, a2->item, SPADE_EVENT);
                        idlist_free(&new_a->idlist);
                        new_a->idlist = out_eq;
                    } else { idlist_free(&out_eq); }
                }
                
                SpadeIdList out_seq; idlist_init(&out_seq);
                join_temporal(&a1->idlist, &a2->idlist, &out_seq);
                if (out_seq.support >= min_supp_count) {
                    found_count++;
                    SpadeAtom *new_a = class_add_atom(&next_cls, a2->item, SPADE_SEQUENCE);
                    idlist_free(&new_a->idlist);
                    new_a->idlist = out_seq;
                } else { idlist_free(&out_seq); }
            }
        }
        
        if (next_cls.count > 0) {
            spade_recursive(&next_cls);
        }
        class_free(&next_cls);
    }
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (ds->type != DM_TYPE_SEQUENCE_UTILITY && ds->type != DM_TYPE_UTILITY) {
        return DM_ERROR_INCOMPATIBLE;
    }
    
    DM_SPADE_Params *p = (DM_SPADE_Params *)params;
    double minsup_param = p ? p->min_support : 0.01;
    
    if (minsup_param > 0.0 && minsup_param < 1.0) {
        min_supp_count = (size_t)(minsup_param * ds->count);
    } else {
        min_supp_count = (size_t)minsup_param;
    }
    if (min_supp_count == 0) min_supp_count = 1;
    
    found_count = 0;
    
    printf("[SPADE] Mining Sequential Patterns (MinSup: %zu)...\n", min_supp_count);
    
    DM_Sequence_Utility *seq_ds;
    if (ds->type == DM_TYPE_UTILITY) {
        DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
        seq_ds = malloc(sizeof(DM_Sequence_Utility) * ds->count);
        for (size_t i = 0; i < ds->count; i++) {
            seq_ds[i].count = 1;
            seq_ds[i].total_utility = src[i].total_utility;
            seq_ds[i].itemsets = malloc(sizeof(DM_Trans_Sequence_Utility));
            seq_ds[i].itemsets[0].count = src[i].count;
            seq_ds[i].itemsets[0].items = src[i].items;
        }
    } else {
        seq_ds = (DM_Sequence_Utility *)ds->payload;
    }
    
    // Initial class [root]
    SpadeClass root_cls;
    class_init(&root_cls, NULL, 0);
    
    SpadeIdList *initial_lists = calloc(ds->max_id + 1, sizeof(SpadeIdList));
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        idlist_init(&initial_lists[i]);
    }
    
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t t = 0; t < seq_ds[i].count; t++) {
            for (size_t j = 0; j < seq_ds[i].itemsets[t].count; j++) {
                uint32_t item = seq_ds[i].itemsets[t].items[j].id;
                // Add (sid, eid) to idlist. eid is just t (the event index).
                // Wait! Within the same event t, there could be duplicates if the dataset is weird,
                // but idlist_add handles unique sid counts.
                // Wait, if an item occurs multiple times in the SAME EVENT (t), 
                // idlist_add will add multiple (sid, eid).
                // We should only add once per (sid, eid).
                SpadeIdList *list = &initial_lists[item];
                if (list->count == 0 || list->pairs[list->count - 1].sid != (uint32_t)i || list->pairs[list->count - 1].eid != (uint32_t)t) {
                    idlist_add(list, (uint32_t)i, (uint32_t)t);
                }
            }
        }
    }
    
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (initial_lists[i].support >= min_supp_count) {
            found_count++;
            SpadeAtom *new_a = class_add_atom(&root_cls, i, SPADE_SEQUENCE);
            // We transfer the memory.
            idlist_free(&new_a->idlist);
            new_a->idlist = initial_lists[i];
        } else {
            idlist_free(&initial_lists[i]);
        }
    }
    free(initial_lists);
    
    if (root_cls.count > 0) {
        spade_recursive(&root_cls);
    }
    class_free(&root_cls);
    
    printf("[SPADE] Found %zu Sequential Patterns.\n", found_count);
    
    if (ds->type == DM_TYPE_UTILITY) {
        for (size_t i = 0; i < ds->count; i++) free(seq_ds[i].itemsets);
        free(seq_ds);
    }
    
    dm_bench_record_results(found_count, 0);
    return DM_SUCCESS;
}

DM_Algorithm spade_algo = {
    .id = "spade",
    .name = "SPADE",
    .description = "Sequential PAttern Discovery using Equivalence classes",
    .supported_types = (1 << DM_TYPE_UTILITY) | (1 << DM_TYPE_SEQUENCE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(spade_algo)
