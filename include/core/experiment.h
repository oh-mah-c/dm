#ifndef DM_EXPERIMENT_H
#define DM_EXPERIMENT_H

#include "algorithms/mfhoi.h"
#include <stddef.h>

typedef struct {
    char dataset[256];
    char algorithm[64];
    double minsup_ratio;
    int minsup_count;
    double minocc;
    int run_id;
    char status[32];
    double runtime_seconds;
    double peak_ram_mb;
    double temp_disk_mb;
    double output_disk_mb;
    int num_generated_candidates;
    int num_output_itemsets;
    int num_fhoi;
    int num_weak_mfhoi;
    int num_strong_mfhoi;
    int dominance_removed_count;
    MFHOIPatternQuality quality;
    double compression_vs_fi;
    double compression_vs_fhoi;
    double reduction_vs_fhoi_percent;
    double strong_extra_reduction_percent;
} ExperimentRow;

double dm_timer_now_seconds(void);
double get_peak_ram_mb(void);
double dm_file_size_mb(const char *path);
double dm_directory_size_mb(const char *path);
int dm_ensure_dir(const char *path);
const char *dm_path_basename(const char *path);
void dm_dataset_stats_write(const char *csv_path, const char **paths, int count);
void dm_csv_write_raw_header(const char *path);
void dm_csv_append_raw_row(const char *path, const ExperimentRow *row);
void dm_pattern_output_write(const char *path, const PatternList *patterns, int transaction_count);
void dm_experiment_generate_report(const char *results_root);

#endif
