#ifndef DM_KCLOTREE_MINER_H
#define DM_KCLOTREE_MINER_H

#include <stddef.h>

typedef enum {
    KCLO_TYPE_GENERIC = 0,
    KCLO_TYPE_GROUP = 1,
    KCLO_TYPE_REDUNDANCY_AWARE = 2
} KCloMiningType;

typedef struct {
    size_t k;
    size_t max_depth;
    size_t max_candidates;
    double max_seconds;
    KCloMiningType type;
} KCloParams;

typedef struct {
    size_t sequences;
    size_t distinct_items;
    size_t max_sequence_length;
    size_t k;
    const char *type_name;
    size_t output_count;
    size_t unique_supports_reported;
    size_t min_reported_support;
    size_t max_reported_support;
    double avg_reported_support;
    double avg_pattern_length;
    size_t candidates_created;
    size_t candidates_processed;
    size_t projected_extensions;
    size_t same_support_extensions;
    size_t closed_candidates;
    size_t absorbed_patterns;
    size_t pruned_by_bound;
    size_t max_heap_size;
    size_t result_ram_bytes;
    size_t result_disk_est_bytes;
    int limited;
} KCloStats;

KCloParams kclotree_default_params(void);
int kclotree_parse_type(const char *name, KCloMiningType *type);
const char *kclotree_type_name(KCloMiningType type);
int kclotree_mine_path(const char *path, const KCloParams *params, KCloStats *stats);

#endif
