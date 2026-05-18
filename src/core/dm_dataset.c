#include "core/dm_dataset.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/* Internal free functions for different types */

static void free_trans_simple(void *payload, size_t count) {
    DM_Trans_Simple *data = (DM_Trans_Simple *)payload;
    for (size_t i = 0; i < count; i++) free(data[i].items);
    free(data);
}

static void free_trans_utility(void *payload, size_t count) {
    DM_Trans_Utility *data = (DM_Trans_Utility *)payload;
    for (size_t i = 0; i < count; i++) free(data[i].items);
    free(data);
}

static void free_trans_quantity(void *payload, size_t count) {
    DM_Trans_Quantity *data = (DM_Trans_Quantity *)payload;
    for (size_t i = 0; i < count; i++) free(data[i].items);
    free(data);
}

static void free_sequence_utility(void *payload, size_t count) {
    DM_Sequence_Utility *data = (DM_Sequence_Utility *)payload;
    for (size_t i = 0; i < count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            free(data[i].itemsets[j].items);
        }
        free(data[i].itemsets);
    }
    free(data);
}

static void sequence_add_itemset(DM_Sequence_Utility *seq, DM_Item *items, size_t count) {
    if (count == 0) return;
    seq->itemsets = realloc(seq->itemsets, sizeof(DM_Trans_Sequence_Utility) * (seq->count + 1));
    seq->itemsets[seq->count].items = malloc(sizeof(DM_Item) * count);
    memcpy(seq->itemsets[seq->count].items, items, sizeof(DM_Item) * count);
    seq->itemsets[seq->count].count = count;
    seq->count++;
}

DM_Dataset* dm_dataset_load(const char *path, DM_DatasetType type) {
    FILE *file = fopen(path, "r");
    if (!file) return NULL;

    DM_Dataset *ds = (DM_Dataset*)malloc(sizeof(DM_Dataset));
    if (!ds) { fclose(file); return NULL; }

    ds->type = type;
    ds->count = 0;
    ds->max_id = 0;
    
    char line[65536]; // Increased buffer size for large transactions
    size_t capacity = 100;
    
    if (type == DM_TYPE_TRANSACTIONAL) {
        ds->payload = malloc(sizeof(DM_Trans_Simple) * capacity);
        ds->free_payload = free_trans_simple;
    } else if (type == DM_TYPE_UTILITY) {
        ds->payload = malloc(sizeof(DM_Trans_Utility) * capacity);
        ds->free_payload = free_trans_utility;
    } else if (type == DM_TYPE_QUANTITY) {
        ds->payload = malloc(sizeof(DM_Trans_Quantity) * capacity);
        ds->free_payload = free_trans_quantity;
    } else if (type == DM_TYPE_SEQUENCE_UTILITY) {
        ds->payload = malloc(sizeof(DM_Sequence_Utility) * capacity);
        ds->free_payload = free_sequence_utility;
    } else {
        free(ds); fclose(file); return NULL;
    }

    while (fgets(line, sizeof(line), file)) {
        char *line_start = line;
        while (*line_start == ' ' || *line_start == '\t' || *line_start == '\r' || *line_start == '\n') line_start++;
        if (*line_start == '\0' || *line_start == '#' || *line_start == '%' || *line_start == '@') {
            continue;
        }
        if (line_start != line) memmove(line, line_start, strlen(line_start) + 1);

        if (ds->count >= capacity) {
            capacity *= 2;
            size_t sz = sizeof(DM_Trans_Utility);
            if (type == DM_TYPE_TRANSACTIONAL) sz = sizeof(DM_Trans_Simple);
            else if (type == DM_TYPE_QUANTITY) sz = sizeof(DM_Trans_Quantity);
            else if (type == DM_TYPE_SEQUENCE_UTILITY) sz = sizeof(DM_Sequence_Utility);
            ds->payload = realloc(ds->payload, sz * capacity);
        }

        if (type == DM_TYPE_TRANSACTIONAL) {
            DM_Trans_Simple *tr = &((DM_Trans_Simple*)ds->payload)[ds->count];
            size_t tr_cap = 8;
            tr->items = malloc(sizeof(uint32_t) * tr_cap);
            tr->count = 0;
            
            char *token = strtok(line, " \t\n\r");
            while (token) {
                if (tr->count >= tr_cap) {
                    tr_cap *= 2;
                    tr->items = realloc(tr->items, sizeof(uint32_t) * tr_cap);
                }
                uint32_t id = (uint32_t)atoi(token);
                tr->items[tr->count++] = id;
                if (id > ds->max_id) ds->max_id = id;
                token = strtok(NULL, " \t\n\r");
            }
            if (tr->count > 0) ds->count++;
            else free(tr->items);
        } else if (type == DM_TYPE_QUANTITY) {
            DM_Trans_Quantity *tr = &((DM_Trans_Quantity*)ds->payload)[ds->count];
            size_t tr_cap = 8;
            tr->items = malloc(sizeof(DM_Quantity_Item) * tr_cap);
            tr->count = 0;

            char *token = strtok(line, " \t\n\r");
            while (token) {
                char *comma = strchr(token, ',');
                if (comma) {
                    *comma = '\0';
                    if (tr->count >= tr_cap) {
                        tr_cap *= 2;
                        tr->items = realloc(tr->items, sizeof(DM_Quantity_Item) * tr_cap);
                    }
                    uint32_t id = (uint32_t)atoi(token);
                    tr->items[tr->count].id = id;
                    tr->items[tr->count].quantity = atof(comma + 1);
                    if (id > ds->max_id) ds->max_id = id;
                    tr->count++;
                }
                token = strtok(NULL, " \t\n\r");
            }
            if (tr->count > 0) ds->count++;
            else free(tr->items);
        } else if (type == DM_TYPE_UTILITY) {
            char *p1 = strchr(line, ':');
            if (!p1) {
                // Fallback to transactional style with utility 1.0
                DM_Trans_Utility *tr = &((DM_Trans_Utility*)ds->payload)[ds->count];
                tr->total_utility = 0;
                size_t tr_cap = 8;
                tr->items = malloc(sizeof(DM_Item) * tr_cap);
                tr->count = 0;
                char *token = strtok(line, " \t\n\r");
                while (token) {
                    if (tr->count >= tr_cap) {
                        tr_cap *= 2;
                        tr->items = realloc(tr->items, sizeof(DM_Item) * tr_cap);
                    }
                    tr->items[tr->count].id = (uint32_t)atoi(token);
                    tr->items[tr->count].utility = 1.0;
                    tr->total_utility += 1.0;
                    if (tr->items[tr->count].id > ds->max_id) ds->max_id = tr->items[tr->count].id;
                    tr->count++;
                    token = strtok(NULL, " \t\n\r");
                }
                if (tr->count > 0) ds->count++; else free(tr->items);
                continue;
            }
            *p1 = '\0';
            char *p2 = strchr(p1 + 1, ':');
            if (!p2) continue;
            *p2 = '\0';

            DM_Trans_Utility *tr = &((DM_Trans_Utility*)ds->payload)[ds->count];
            tr->total_utility = atof(p1 + 1);
            
            size_t tr_cap = 8;
            tr->items = malloc(sizeof(DM_Item) * tr_cap);
            tr->count = 0;
            
            char *items_part = _strdup(line); 
            char *it_token = strtok(items_part, " ");
            while (it_token) {
                if (tr->count >= tr_cap) {
                    tr_cap *= 2;
                    tr->items = realloc(tr->items, sizeof(DM_Item) * tr_cap);
                }
                tr->items[tr->count].id = (uint32_t)atoi(it_token);
                if (tr->items[tr->count].id > ds->max_id) ds->max_id = tr->items[tr->count].id;
                tr->count++;
                it_token = strtok(NULL, " ");
            }
            free(items_part);

            char *utils_part = _strdup(p2 + 1);
            char *ut_token = strtok(utils_part, " \n\r");
            size_t ut_idx = 0;
            while (ut_token && ut_idx < tr->count) {
                tr->items[ut_idx].utility = atof(ut_token);
                ut_idx++;
                ut_token = strtok(NULL, " \n\r");
            }
            free(utils_part);

            if (tr->count > 0) ds->count++;
            else free(tr->items);
        } else if (type == DM_TYPE_SEQUENCE_UTILITY) {
            DM_Sequence_Utility *seq = &((DM_Sequence_Utility*)ds->payload)[ds->count];
            seq->itemsets = NULL;
            seq->count = 0;
            seq->total_utility = 0.0;
            seq->probability = 1.0;

            size_t cur_cap = 4;
            DM_Item *cur_items = malloc(sizeof(DM_Item) * cur_cap);
            size_t cur_count = 0;
            double computed_total = 0.0;

            char *token = strtok(line, " \t\n\r");
            while (token) {
                if (strcmp(token, "-1") == 0) {
                    sequence_add_itemset(seq, cur_items, cur_count);
                    cur_count = 0;
                } else if (strcmp(token, "-2") == 0) {
                    if (cur_count > 0) {
                        sequence_add_itemset(seq, cur_items, cur_count);
                        cur_count = 0;
                    }
                } else if (strncmp(token, "SUtility:", 9) == 0) {
                    seq->total_utility = atof(token + 9);
                } else if (strncmp(token, "SU:", 3) == 0) {
                    seq->total_utility = atof(token + 3);
                } else if (strncmp(token, "Utility:", 8) == 0) {
                    seq->total_utility = atof(token + 8);
                } else if (strncmp(token, "prob:", 5) == 0 || strncmp(token, "Prob:", 5) == 0) {
                    seq->probability = atof(token + 5);
                } else {
                    char *lb = strchr(token, '[');
                    char *rb = lb ? strchr(lb + 1, ']') : NULL;
                    char *colon = strchr(token, ':');
                    uint32_t id = 0;
                    double util = 1.0;

                    if (lb && rb) {
                        *lb = '\0';
                        *rb = '\0';
                        id = (uint32_t)atoi(token);
                        util = atof(lb + 1);
                    } else if (colon) {
                        *colon = '\0';
                        id = (uint32_t)atoi(token);
                        util = atof(colon + 1);
                    } else {
                        id = (uint32_t)atoi(token);
                    }

                    if (id > 0) {
                        if (cur_count >= cur_cap) {
                            cur_cap *= 2;
                            cur_items = realloc(cur_items, sizeof(DM_Item) * cur_cap);
                        }
                        cur_items[cur_count].id = id;
                        cur_items[cur_count].utility = util;
                        cur_count++;
                        computed_total += util;
                        if (id > ds->max_id) ds->max_id = id;
                    }
                }
                token = strtok(NULL, " \t\n\r");
            }

            if (cur_count > 0) sequence_add_itemset(seq, cur_items, cur_count);
            free(cur_items);

            if (seq->total_utility <= 0.0) seq->total_utility = computed_total;
            if (seq->count > 0) {
                ds->count++;
            } else {
                free(seq->itemsets);
            }
        }
    }

    fclose(file);
    return ds;
}

void dm_dataset_free(DM_Dataset *ds) {
    if (!ds) return;
    if (ds->free_payload && ds->payload) {
        ds->free_payload(ds->payload, ds->count);
    }
    free(ds);
}
