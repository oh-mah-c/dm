#ifndef DM_HTK_MINER_H
#define DM_HTK_MINER_H

#include <stddef.h>

typedef enum {
    DM_HTK_MODE_BSN = 0
} DM_HTK_Mode;

typedef struct {
    size_t k;
    size_t max_depth;
    size_t max_candidates;
    double max_seconds;
    DM_HTK_Mode mode;
} DM_HTK_Params;

typedef struct {
    size_t transactions;
    size_t nonempty_transactions;
    size_t distinct_items;
    size_t k;
    size_t levels;
    size_t final_threshold;
    size_t output_count;
    size_t total_output_items;
    double avg_output_length;
    double best_support;
    double avg_support;
    size_t candidates;
    size_t joins;
    size_t intersections;
    size_t singleton_kept;
    size_t pruned_singletons;
    size_t pruned_support;
    size_t threshold_raises;
    size_t bitset_words;
    size_t result_ram_bytes;
    size_t result_disk_est_bytes;
    int limited;
} DM_HTK_Stats;

int htk_mine_file(const char *path, const DM_HTK_Params *params, DM_HTK_Stats *stats);
const char *htk_mode_name(DM_HTK_Mode mode);
int htk_parse_mode(const char *name, DM_HTK_Mode *mode);

#endif
