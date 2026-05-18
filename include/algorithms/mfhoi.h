#ifndef DM_ALGORITHM_MFHOI_H
#define DM_ALGORITHM_MFHOI_H

#include "core/dm_algorithm.h"
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    double min_support;
    double min_occupancy;
    bool strong;
    int output_algorithm;
} DM_MFHOI_Params;

typedef struct {
    int *items;
    int length;
} Transaction;

typedef struct {
    Transaction *transactions;
    int transaction_count;
    int item_count;
    int max_item_id;
    int *transaction_lengths;
    const char *name;
} TransactionDB;

typedef struct {
    int item;
    unsigned long *bitset;
    int support;
} VerticalItem;

typedef struct {
    int *items;
    int length;
    unsigned long *tidset;
    int support;
    double avg_occ;
} Pattern;

typedef struct {
    Pattern *patterns;
    int count;
    int capacity;
} PatternList;

typedef struct {
    int num_generated_candidates;
    int num_frequent_itemsets;
    int num_fhoi;
    int num_weak_mfhoi;
    int num_strong_mfhoi;
    int dominance_removed_count;
    int weak_removed_count;
    int pruned_ub1_count;
    int pruned_ub2_count;
    int status_limited;
} MFHOIStats;

typedef struct {
    PatternList frequent;
    PatternList mfi;
    PatternList fhoi;
    PatternList weak_mfhoi;
    PatternList strong_mfhoi;
    MFHOIStats stats;
} MFHOIResult;

typedef enum {
    MFHOI_ALGO_APRIORI = 0,
    MFHOI_ALGO_FHOI,
    MFHOI_ALGO_WEAK,
    MFHOI_ALGO_STRONG,
    MFHOI_ALGO_MFI
} MFHOIAlgorithm;

typedef struct {
    int max_patterns;
    double max_seconds;
} MFHOILimits;

typedef struct {
    double avg_itemset_length;
    int max_itemset_length;
    double avg_support;
    double avg_relative_support;
    double avg_occupancy;
    double median_occupancy;
    double max_occupancy;
} MFHOIPatternQuality;

TransactionDB *mfhoi_load_transaction_db(const char *path);
void mfhoi_free_transaction_db(TransactionDB *db);
int mfhoi_write_synthetic_datasets(const char *dataset_root);

void mfhoi_pattern_list_init(PatternList *list);
void mfhoi_pattern_list_free(PatternList *list);
void mfhoi_result_free(MFHOIResult *result);

int mfhoi_mine_all(const TransactionDB *db, int minsup_count, double minocc, const MFHOILimits *limits, MFHOIResult *out);
const PatternList *mfhoi_select_patterns(const MFHOIResult *result, MFHOIAlgorithm algo);
const char *mfhoi_algorithm_name(MFHOIAlgorithm algo);
int mfhoi_parse_algorithm(const char *name, MFHOIAlgorithm *algo);

int mfhoi_write_patterns(const char *path, const PatternList *patterns, int transaction_count);
int mfhoi_write_dominance_examples(const char *path, const TransactionDB *db, double minsup_ratio, double minocc, const PatternList *fhoi, int append);
void mfhoi_compute_quality(const PatternList *patterns, int transaction_count, MFHOIPatternQuality *quality);
double mfhoi_exact_overlap_ratio(const PatternList *a, const PatternList *b, int *overlap_count);
double mfhoi_avg_jaccard_to_nearest(const PatternList *source, const PatternList *target);

#endif // DM_ALGORITHM_MFHOI_H
