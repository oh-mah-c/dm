#ifndef DM_HIEP_H
#define DM_HIEP_H

#include <stddef.h>
#include <stdint.h>

#include "core/dm_algorithm.h"

typedef enum {
    HIEP_MODE_ITEMSET = 0,
    HIEP_MODE_SEQUENCE = 1
} HIEPMode;

typedef enum {
    HIEP_INPUT_TEXT = 0,
    HIEP_INPUT_TRANSACTIONS = 1
} HIEPInputType;

typedef struct {
    HIEPMode mode;
    HIEPInputType input_type;
    size_t window_length;
    size_t stride;
    double theta;
    double theta_ratio;
    uint32_t min_support;
    double min_support_ratio;
    double alpha;
    double gamma;
    size_t max_patterns;
    size_t max_depth;
    size_t max_transactions;
    size_t max_tokens;
    size_t max_bytes;
    double max_seconds;
    int disable_tiub;
    int disable_iwru;
    int uniform_weights;
    int disable_compactness;
    const char *tokenizer_name;
    const char *output_path;
} HIEPParams;

typedef struct {
    size_t input_bytes;
    size_t token_stream_length;
    size_t transactions;
    size_t nnz;
    size_t vocabulary_size;
    uint32_t min_support;
    double theta;
    double theta_ratio;
    double alpha;
    double gamma;
    size_t surviving_items;
    size_t singleton_occurrences;
    size_t visited_nodes;
    size_t generated_children;
    size_t joins;
    size_t joined_entries;
    size_t pruned_support;
    size_t pruned_tiub;
    size_t pruned_iwru;
    size_t emitted_patterns;
    size_t total_output_items;
    size_t max_depth_seen;
    double avg_support;
    double avg_utility;
    double avg_pattern_weight;
    double best_utility;
    double information_density_optimization;
    double noise_filtering_efficiency;
    double signal_recall;
    double avg_utility_list_length;
    double max_utility_list_length;
    size_t result_ram_bytes;
    size_t result_disk_est_bytes;
    int limited;
} HIEPStats;

typedef struct {
    const char *input_path;
    HIEPParams params;
    HIEPStats *stats;
} HIEPRunConfig;

HIEPParams hiep_default_params(void);
int hiep_parse_mode(const char *name, HIEPMode *mode);
const char *hiep_mode_name(HIEPMode mode);
int hiep_parse_input_type(const char *name, HIEPInputType *type);
const char *hiep_input_type_name(HIEPInputType type);
int hiep_mine_file(const char *path, const HIEPParams *params, HIEPStats *stats);

extern DM_Algorithm hiep_algo;

#endif
