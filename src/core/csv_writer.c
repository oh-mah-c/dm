#include "core/experiment.h"
#include <stdio.h>

void dm_csv_write_raw_header(const char *path) {
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp, "dataset,algorithm,minsup_ratio,minsup_count,minocc,run_id,status,runtime_seconds,peak_ram_mb,temp_disk_mb,output_disk_mb,num_generated_candidates,num_output_itemsets,num_fhoi,num_weak_mfhoi,num_strong_mfhoi,dominance_removed_count,avg_itemset_length,max_itemset_length,avg_support,avg_relative_support,avg_occupancy,median_occupancy,max_occupancy,compression_vs_FI,compression_vs_FHOI,reduction_vs_FHOI_percent,strong_extra_reduction_percent\n");
    fclose(fp);
}

void dm_csv_append_raw_row(const char *path, const ExperimentRow *r) {
    FILE *fp = fopen(path, "a");
    if (!fp) return;
    fprintf(fp, "%s,%s,%.6f,%d,%.6f,%d,%s,%.8f,%.4f,%.6f,%.6f,%d,%d,%d,%d,%d,%d,%.8f,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
            r->dataset, r->algorithm, r->minsup_ratio, r->minsup_count, r->minocc, r->run_id, r->status,
            r->runtime_seconds, r->peak_ram_mb, r->temp_disk_mb, r->output_disk_mb,
            r->num_generated_candidates, r->num_output_itemsets, r->num_fhoi, r->num_weak_mfhoi,
            r->num_strong_mfhoi, r->dominance_removed_count,
            r->quality.avg_itemset_length, r->quality.max_itemset_length, r->quality.avg_support,
            r->quality.avg_relative_support, r->quality.avg_occupancy, r->quality.median_occupancy,
            r->quality.max_occupancy, r->compression_vs_fi, r->compression_vs_fhoi,
            r->reduction_vs_fhoi_percent, r->strong_extra_reduction_percent);
    fclose(fp);
}
