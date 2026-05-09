#include "core/dm_dataset.h"
#include <string.h>

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

DM_Dataset* dm_dataset_load(const char *path, DM_DatasetType type) {
    FILE *file = fopen(path, "r");
    if (!file) return NULL;

    DM_Dataset *ds = (DM_Dataset*)malloc(sizeof(DM_Dataset));
    if (!ds) { fclose(file); return NULL; }

    ds->type = type;
    ds->count = 0;
    ds->max_id = 0;
    
    char line[16384];
    size_t capacity = 100;
    
    if (type == DM_TYPE_TRANSACTIONAL) {
        ds->payload = malloc(sizeof(DM_Trans_Simple) * capacity);
        ds->free_payload = free_trans_simple;
    } else if (type == DM_TYPE_UTILITY) {
        ds->payload = malloc(sizeof(DM_Trans_Utility) * capacity);
        ds->free_payload = free_trans_utility;
    } else {
        // Handle other types or error
        free(ds); fclose(file); return NULL;
    }

    while (fgets(line, sizeof(line), file)) {
        if (ds->count >= capacity) {
            capacity *= 2;
            size_t sz = (type == DM_TYPE_TRANSACTIONAL) ? sizeof(DM_Trans_Simple) : sizeof(DM_Trans_Utility);
            ds->payload = realloc(ds->payload, sz * capacity);
        }

        char *token = strtok(line, " \t\n\r");
        if (!token) continue;

        if (type == DM_TYPE_TRANSACTIONAL) {
            DM_Trans_Simple *tr = &((DM_Trans_Simple*)ds->payload)[ds->count];
            size_t tr_cap = 8;
            tr->items = malloc(sizeof(uint32_t) * tr_cap);
            tr->count = 0;
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
        } else if (type == DM_TYPE_UTILITY) {
            DM_Trans_Utility *tr = &((DM_Trans_Utility*)ds->payload)[ds->count];
            size_t tr_cap = 8;
            tr->items = malloc(sizeof(DM_Item) * tr_cap);
            tr->count = 0;
            tr->total_utility = 0;
            while (token) {
                if (tr->count >= tr_cap) {
                    tr_cap *= 2;
                    tr->items = realloc(tr->items, sizeof(DM_Item) * tr_cap);
                }
                char *colon = strchr(token, ':');
                if (colon) {
                    *colon = '\0';
                    tr->items[tr->count].id = (uint32_t)atoi(token);
                    tr->items[tr->count].utility = atof(colon + 1);
                    tr->total_utility += tr->items[tr->count].utility;
                } else {
                    tr->items[tr->count].id = (uint32_t)atoi(token);
                    tr->items[tr->count].utility = 1.0;
                }
                if (tr->items[tr->count].id > ds->max_id) ds->max_id = tr->items[tr->count].id;
                tr->count++;
                token = strtok(NULL, " \t\n\r");
            }
        }
        ds->count++;
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
