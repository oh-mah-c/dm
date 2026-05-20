#ifndef DM_FLAT_H
#define DM_FLAT_H

#include "core/dm_arena.h"
#include "core/dm_mmap.h"

#include <stddef.h>
#include <stdint.h>

typedef enum {
    DM_CONNECTOR_SPMF = 0,
    DM_CONNECTOR_TEXT = 1,
    DM_CONNECTOR_GRAPH = 2
} DM_ConnectorKind;

typedef struct {
    uint32_t *items;
    size_t item_count;
    size_t *row_offsets;
    size_t row_count;
    uint32_t max_item;
    DM_ConnectorKind source_kind;
    DM_Arena *arena;
    const unsigned char *borrowed_base;
    size_t borrowed_bytes;
} DM_FlatDataset;

typedef struct {
    DM_ConnectorKind kind;
    size_t text_window;
    size_t text_stride;
    int text_sequence;
    int graph_undirected;
} DM_ConnectorOptions;

typedef struct {
    size_t input_bytes;
    size_t rows;
    size_t items;
    size_t distinct_estimate;
    uint32_t max_item;
    double avg_row_len;
} DM_ConnectorStats;

const char *dm_connector_name(DM_ConnectorKind kind);
int dm_connector_parse_kind(const char *name, DM_ConnectorKind *kind);
DM_ConnectorOptions dm_connector_default_options(DM_ConnectorKind kind);
int dm_flat_load_mmap(const char *path, const DM_ConnectorOptions *options, DM_Arena *arena, DM_FlatDataset *out, DM_ConnectorStats *stats);

#endif
