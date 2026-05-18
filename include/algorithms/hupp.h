#ifndef DM_HUPP_H
#define DM_HUPP_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    HUPP_DATASET_AUTO = 0,
    HUPP_DATASET_DOLLY = 1,
    HUPP_DATASET_CODE_FEEDBACK = 2
} HUPPDatasetType;

typedef struct {
    HUPPDatasetType dataset_type;
    double min_support;
    double min_utility;
    double min_utility_ratio;
    double min_alignment;
    double w_token;
    double w_latency;
    double w_alignment;
    double w_cache;
    size_t max_transactions;
    size_t max_concepts_per_prompt;
    size_t max_patterns;
    size_t max_depth;
    double max_seconds;
} HUPPParams;

typedef struct {
    size_t transactions;
    size_t semantic_concepts;
    size_t total_prompt_tokens;
    size_t candidate_optimizers;
    size_t pareto_removed;
    uint32_t minsup_count;
    double theta;
    double alpha_min;
    size_t frequent_singletons;
    size_t singleton_occurrences;
    size_t visited_nodes;
    size_t joins;
    size_t joined_entries;
    size_t pruned_support;
    size_t pruned_ptwo;
    size_t pruned_aaub;
    size_t emitted_patterns;
    size_t total_pattern_items;
    size_t max_depth_seen;
    double best_utility;
    double best_alignment;
    double avg_utility;
    double avg_alignment;
    double avg_support;
    size_t result_ram_bytes;
    size_t result_disk_est_bytes;
    int limited;
} HUPPStats;

int hupp_parse_dataset_type(const char *name, HUPPDatasetType *type);
const char *hupp_dataset_type_name(HUPPDatasetType type);
HUPPParams hupp_default_params(void);
int hupp_mine_file(const char *path, const HUPPParams *params, HUPPStats *stats);

#endif
