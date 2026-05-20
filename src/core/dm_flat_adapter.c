#include "core/dm_flat_adapter.h"

#include <stdlib.h>
#include <string.h>

static void free_trans_simple_adapted(void *payload, size_t count) {
    DM_Trans_Simple *data = (DM_Trans_Simple *)payload;
    for (size_t i = 0; i < count; i++) free(data[i].items);
    free(data);
}

static void free_trans_utility_adapted(void *payload, size_t count) {
    DM_Trans_Utility *data = (DM_Trans_Utility *)payload;
    for (size_t i = 0; i < count; i++) free(data[i].items);
    free(data);
}

static void free_trans_quantity_adapted(void *payload, size_t count) {
    DM_Trans_Quantity *data = (DM_Trans_Quantity *)payload;
    for (size_t i = 0; i < count; i++) free(data[i].items);
    free(data);
}

static void free_sequence_utility_adapted(void *payload, size_t count) {
    DM_Sequence_Utility *data = (DM_Sequence_Utility *)payload;
    for (size_t i = 0; i < count; i++) {
        for (size_t j = 0; j < data[i].count; j++) free(data[i].itemsets[j].items);
        free(data[i].itemsets);
    }
    free(data);
}

static void free_matrix_adapted(void *payload, size_t count) {
    DM_Matrix_Row *data = (DM_Matrix_Row *)payload;
    for (size_t i = 0; i < count; i++) free(data[i].values);
    free(data);
}

DM_DatasetType dm_flat_choose_legacy_type(uint32_t supported_types, DM_ConnectorKind connector, int requested_type) {
    if (requested_type >= 0 && requested_type <= (int)DM_TYPE_QUANTITY && (supported_types & (1u << requested_type))) {
        return (DM_DatasetType)requested_type;
    }
    if (connector == DM_CONNECTOR_TEXT || connector == DM_CONNECTOR_GRAPH || connector == DM_CONNECTOR_SPMF) {
        if (supported_types & (1u << DM_TYPE_TRANSACTIONAL)) return DM_TYPE_TRANSACTIONAL;
    }
    if (supported_types & (1u << DM_TYPE_UTILITY)) return DM_TYPE_UTILITY;
    if (supported_types & (1u << DM_TYPE_SEQUENCE_UTILITY)) return DM_TYPE_SEQUENCE_UTILITY;
    if (supported_types & (1u << DM_TYPE_QUANTITY)) return DM_TYPE_QUANTITY;
    if (supported_types & (1u << DM_TYPE_MATRIX)) return DM_TYPE_MATRIX;
    if (supported_types & (1u << DM_TYPE_TRANSACTIONAL)) return DM_TYPE_TRANSACTIONAL;
    return DM_TYPE_TRANSACTIONAL;
}

DM_Dataset *dm_flat_to_dataset(const DM_FlatDataset *flat, DM_DatasetType type) {
    if (!flat) return NULL;
    DM_Dataset *ds = (DM_Dataset *)calloc(1, sizeof(*ds));
    if (!ds) return NULL;
    ds->type = type;
    ds->count = flat->row_count;
    ds->max_id = flat->max_item;

    if (type == DM_TYPE_TRANSACTIONAL) {
        DM_Trans_Simple *rows = (DM_Trans_Simple *)calloc(flat->row_count, sizeof(*rows));
        if (!rows) { free(ds); return NULL; }
        for (size_t r = 0; r < flat->row_count; r++) {
            size_t a = flat->row_offsets[r], b = flat->row_offsets[r + 1], n = b - a;
            rows[r].items = (uint32_t *)malloc((n ? n : 1) * sizeof(uint32_t));
            if (!rows[r].items) { free_trans_simple_adapted(rows, r); free(ds); return NULL; }
            memcpy(rows[r].items, flat->items + a, n * sizeof(uint32_t));
            rows[r].count = n;
        }
        ds->payload = rows;
        ds->free_payload = free_trans_simple_adapted;
    } else if (type == DM_TYPE_UTILITY) {
        DM_Trans_Utility *rows = (DM_Trans_Utility *)calloc(flat->row_count, sizeof(*rows));
        if (!rows) { free(ds); return NULL; }
        for (size_t r = 0; r < flat->row_count; r++) {
            size_t a = flat->row_offsets[r], b = flat->row_offsets[r + 1], n = b - a;
            rows[r].items = (DM_Item *)malloc((n ? n : 1) * sizeof(DM_Item));
            if (!rows[r].items) { free_trans_utility_adapted(rows, r); free(ds); return NULL; }
            rows[r].count = n;
            rows[r].total_utility = (double)n;
            for (size_t i = 0; i < n; i++) {
                rows[r].items[i].id = flat->items[a + i];
                rows[r].items[i].utility = 1.0;
            }
        }
        ds->payload = rows;
        ds->free_payload = free_trans_utility_adapted;
    } else if (type == DM_TYPE_QUANTITY) {
        DM_Trans_Quantity *rows = (DM_Trans_Quantity *)calloc(flat->row_count, sizeof(*rows));
        if (!rows) { free(ds); return NULL; }
        for (size_t r = 0; r < flat->row_count; r++) {
            size_t a = flat->row_offsets[r], b = flat->row_offsets[r + 1], n = b - a;
            rows[r].items = (DM_Quantity_Item *)malloc((n ? n : 1) * sizeof(DM_Quantity_Item));
            if (!rows[r].items) { free_trans_quantity_adapted(rows, r); free(ds); return NULL; }
            rows[r].count = n;
            for (size_t i = 0; i < n; i++) {
                rows[r].items[i].id = flat->items[a + i];
                rows[r].items[i].quantity = 1.0;
            }
        }
        ds->payload = rows;
        ds->free_payload = free_trans_quantity_adapted;
    } else if (type == DM_TYPE_SEQUENCE_UTILITY) {
        DM_Sequence_Utility *rows = (DM_Sequence_Utility *)calloc(flat->row_count, sizeof(*rows));
        if (!rows) { free(ds); return NULL; }
        for (size_t r = 0; r < flat->row_count; r++) {
            size_t a = flat->row_offsets[r], b = flat->row_offsets[r + 1], n = b - a;
            rows[r].itemsets = (DM_Trans_Sequence_Utility *)calloc(1, sizeof(DM_Trans_Sequence_Utility));
            if (!rows[r].itemsets) { free_sequence_utility_adapted(rows, r); free(ds); return NULL; }
            rows[r].itemsets[0].items = (DM_Item *)malloc((n ? n : 1) * sizeof(DM_Item));
            if (!rows[r].itemsets[0].items) { free_sequence_utility_adapted(rows, r + 1); free(ds); return NULL; }
            rows[r].count = 1;
            rows[r].itemsets[0].count = n;
            rows[r].total_utility = (double)n;
            rows[r].probability = 1.0;
            for (size_t i = 0; i < n; i++) {
                rows[r].itemsets[0].items[i].id = flat->items[a + i];
                rows[r].itemsets[0].items[i].utility = 1.0;
            }
        }
        ds->payload = rows;
        ds->free_payload = free_sequence_utility_adapted;
    } else if (type == DM_TYPE_MATRIX) {
        DM_Matrix_Row *rows = (DM_Matrix_Row *)calloc(flat->row_count, sizeof(*rows));
        if (!rows) { free(ds); return NULL; }
        for (size_t r = 0; r < flat->row_count; r++) {
            size_t a = flat->row_offsets[r], b = flat->row_offsets[r + 1], n = b - a;
            rows[r].values = (double *)malloc((n ? n : 1) * sizeof(double));
            if (!rows[r].values) { free_matrix_adapted(rows, r); free(ds); return NULL; }
            rows[r].count = n;
            for (size_t i = 0; i < n; i++) rows[r].values[i] = (double)flat->items[a + i];
        }
        ds->payload = rows;
        ds->free_payload = free_matrix_adapted;
    } else {
        free(ds);
        return NULL;
    }
    return ds;
}
