#ifndef DM_ALGORITHM_MHOUI_H
#define DM_ALGORITHM_MHOUI_H

#include "core/dm_algorithm.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    double min_support;
    double min_occupancy;
    double min_utility;
    int strong;
    int direct;
    int output_algorithm;
    int max_patterns;
    double max_seconds;
} DM_MHOUI_Params;

typedef enum {
    MHOUI_ALGO_HOUI = 0,
    MHOUI_ALGO_WEAK,
    MHOUI_ALGO_STRONG,
    MHOUI_ALGO_DIRECT_WEAK,
    MHOUI_ALGO_DIRECT_STRONG
} MHOUIAlgorithm;

typedef struct {
    uint32_t *items;
    int length;
    uint64_t *tidset;
    int support;
    double utility;
    double avg_occupancy;
} MHOUPattern;

typedef struct {
    MHOUPattern *patterns;
    int count;
    int capacity;
} MHOUPatternList;

typedef struct {
    int visited_nodes;
    int generated_candidates;
    int pruned_support;
    int pruned_twu;
    int pruned_uub;
    int pruned_oub1;
    int pruned_oub2;
    int pruned_dominance;
    int houi_count;
    int weak_count;
    int strong_count;
    int weak_removed;
    int strong_removed;
    int status_limited;
} MHOUIStats;

typedef struct {
    MHOUPatternList houi;
    MHOUPatternList weak_mhoui;
    MHOUPatternList strong_mhoui;
    MHOUIStats stats;
} MHOUIResult;

typedef struct {
    int max_patterns;
    double max_seconds;
    int direct;
    int strong_direct;
} MHOUILimits;

int mhoui_parse_algorithm(const char *name, MHOUIAlgorithm *algo);
const char *mhoui_algorithm_name(MHOUIAlgorithm algo);

void mhoui_pattern_list_init(MHOUPatternList *list);
void mhoui_pattern_list_free(MHOUPatternList *list);
void mhoui_result_free(MHOUIResult *result);

int mhoui_mine_dataset(DM_Dataset *ds, int minsup_count, double minocc, double minutil, const MHOUILimits *limits, MHOUIResult *out);
const MHOUPatternList *mhoui_select_patterns(const MHOUIResult *result, MHOUIAlgorithm algo);
int mhoui_write_patterns(const char *path, const MHOUPatternList *patterns, size_t transaction_count);
void mhoui_pattern_quality(const MHOUPatternList *patterns, double *avg_len, int *max_len, double *avg_support, double *avg_utility, double *avg_occ);

extern DM_Algorithm mhoui_algo;

#endif
