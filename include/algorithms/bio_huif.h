#ifndef DM_BIO_HUIF_H
#define DM_BIO_HUIF_H

#include "core/dm_algorithm.h"
#include "core/dm_bitset.h"

typedef struct {
    double min_utility;
    int pop_size;
    int max_iter;
} DM_BioHUIF_Params;

// Common structure for HUI entries in the SHUI set
typedef struct {
    uint32_t *items;
    size_t len;
    double utility;
} SHUI_Entry;

// Framework context for Bio-HUIF algorithms
typedef struct {
    SHUI_Entry *shui;
    size_t shui_count;
    size_t shui_capacity;
    double total_shui_utility;
} BioHUIF_Context;

#endif
