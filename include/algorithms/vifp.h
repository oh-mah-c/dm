#ifndef DM_VIFP_H
#define DM_VIFP_H

#include "core/dm_algorithm.h"
#include "core/dm_dataset.h"
#include <stddef.h>
#include <stdint.h>

typedef enum {
    VIFP_MODE_PLAINTEXT = 0,
    VIFP_MODE_SMPC = 1,
    VIFP_MODE_FHE = 2
} VIFPMode;

typedef struct {
    double min_support;
    VIFPMode mode;
    size_t max_itemsets;
    double max_seconds;
} DM_VIFP_Params;

typedef struct {
    size_t transactions;
    uint32_t max_item_id;
    uint32_t minsup_count;
    size_t frequent_singletons;
    size_t occurrence_records;
    size_t hpa_records;
    size_t emitted_itemsets;
    size_t total_output_items;
    size_t projected_states;
    size_t cpb_records;
    size_t max_cpb_records;
    size_t predecessor_fetches;
    size_t histogram_updates;
    size_t secure_comparisons;
    size_t stable_partitions;
    size_t oblivious_sorts;
    size_t pid_descriptors;
    size_t max_depth;
    size_t public_capacity_slots;
    size_t active_capacity_slots;
    size_t estimated_comm_bytes;
    size_t estimated_ciphertext_bytes;
    size_t estimated_bootstraps;
    int limited;
} VIFPStats;

int vifp_parse_mode(const char *name, VIFPMode *mode);
const char *vifp_mode_name(VIFPMode mode);
int vifp_mine_dataset(DM_Dataset *ds, uint32_t minsup_count, VIFPMode mode, size_t max_itemsets, double max_seconds, VIFPStats *stats);

#endif
