#include "algorithms/mfhoi.h"
#include "core/experiment.h"
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char **paths;
    int count;
    int capacity;
} PathList;

static void paths_add(PathList *list, const char *path) {
    if (list->count >= list->capacity) {
        list->capacity = list->capacity ? list->capacity * 2 : 16;
        list->paths = realloc(list->paths, (size_t)list->capacity * sizeof(char *));
    }
    list->paths[list->count++] = strdup(path);
}

static void paths_free(PathList *list) {
    for (int i = 0; i < list->count; i++) free(list->paths[i]);
    free(list->paths);
}

static int ends_with_txt(const char *s) {
    size_t n = strlen(s);
    return n > 4 && strcmp(s + n - 4, ".txt") == 0;
}

static void collect_txt(PathList *list, const char *root) {
    DIR *dir = opendir(root);
    if (!dir) return;
    struct dirent *ent;
    while ((ent = readdir(dir))) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        char child[1024];
        snprintf(child, sizeof(child), "%s/%s", root, ent->d_name);
        if (ent->d_type == DT_DIR) {
            collect_txt(list, child);
        } else if (ends_with_txt(ent->d_name)) {
            paths_add(list, child);
        }
    }
    closedir(dir);
}

static const char *arg_value(int argc, char **argv, const char *key) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return NULL;
}

static void sanitize_name(const char *in, char *out, size_t out_sz) {
    size_t w = 0;
    for (size_t i = 0; in[i] && w + 1 < out_sz; i++) {
        char c = in[i];
        out[w++] = (c == '/' || c == ' ' || c == '.') ? '_' : c;
    }
    out[w] = '\0';
}

static double safe_ratio(double num, double den) {
    return den == 0.0 ? 0.0 : num / den;
}

static void append_summary_headers(const char *out_root) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/csv/results_avg.csv", out_root);
    dm_csv_write_raw_header(path);

    snprintf(path, sizeof(path), "%s/csv/compression_summary.csv", out_root);
    FILE *fp = fopen(path, "w");
    if (fp) {
        fprintf(fp, "dataset,minsup_ratio,minocc,num_fhoi,num_weak_mfhoi,num_strong_mfhoi,ok_chain,reduction_vs_FHOI_percent,strong_extra_reduction_percent\n");
        fclose(fp);
    }

    snprintf(path, sizeof(path), "%s/csv/overlap_summary.csv", out_root);
    fp = fopen(path, "w");
    if (fp) {
        fprintf(fp, "dataset,minsup_ratio,minocc,exact_overlap_count,exact_overlap_ratio,avg_jaccard_to_nearest_MFI\n");
        fclose(fp);
    }

    snprintf(path, sizeof(path), "%s/csv/pattern_quality_summary.csv", out_root);
    fp = fopen(path, "w");
    if (fp) {
        fprintf(fp, "dataset,algorithm,minsup_ratio,minocc,avg_itemset_length,max_itemset_length,avg_support,avg_relative_support,avg_occupancy,median_occupancy,max_occupancy\n");
        fclose(fp);
    }
}

static void append_compression(const char *out_root, const char *dataset, double minsup_ratio, double minocc, const MFHOIResult *result) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/csv/compression_summary.csv", out_root);
    FILE *fp = fopen(path, "a");
    if (!fp) return;
    int ok = result->strong_mfhoi.count <= result->weak_mfhoi.count && result->weak_mfhoi.count <= result->fhoi.count;
    double reduction = result->fhoi.count ? (1.0 - (double)result->strong_mfhoi.count / (double)result->fhoi.count) * 100.0 : 0.0;
    double extra = result->weak_mfhoi.count ? (1.0 - (double)result->strong_mfhoi.count / (double)result->weak_mfhoi.count) * 100.0 : 0.0;
    fprintf(fp, "%s,%.6f,%.6f,%d,%d,%d,%s,%.8f,%.8f\n",
            dataset, minsup_ratio, minocc, result->fhoi.count, result->weak_mfhoi.count,
            result->strong_mfhoi.count, ok ? "OK" : "FAIL", reduction, extra);
    fclose(fp);
}

static void append_overlap(const char *out_root, const char *dataset, double minsup_ratio, double minocc, const MFHOIResult *result) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/csv/overlap_summary.csv", out_root);
    FILE *fp = fopen(path, "a");
    if (!fp) return;
    int exact = 0;
    double ratio = mfhoi_exact_overlap_ratio(&result->strong_mfhoi, &result->mfi, &exact);
    double jac = mfhoi_avg_jaccard_to_nearest(&result->strong_mfhoi, &result->mfi);
    fprintf(fp, "%s,%.6f,%.6f,%d,%.8f,%.8f\n", dataset, minsup_ratio, minocc, exact, ratio, jac);
    fclose(fp);
}

static void append_quality(const char *out_root, const char *dataset, const char *algo, double minsup_ratio, double minocc, const MFHOIPatternQuality *q) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/csv/pattern_quality_summary.csv", out_root);
    FILE *fp = fopen(path, "a");
    if (!fp) return;
    fprintf(fp, "%s,%s,%.6f,%.6f,%.8f,%d,%.8f,%.8f,%.8f,%.8f,%.8f\n",
            dataset, algo, minsup_ratio, minocc, q->avg_itemset_length, q->max_itemset_length,
            q->avg_support, q->avg_relative_support, q->avg_occupancy,
            q->median_occupancy, q->max_occupancy);
    fclose(fp);
}

int main(int argc, char **argv) {
    const char *dataset_root = arg_value(argc, argv, "--datasets");
    const char *out_root = arg_value(argc, argv, "--out");
    const char *runs_s = arg_value(argc, argv, "--runs");
    const char *max_seconds_s = arg_value(argc, argv, "--max-seconds");
    const char *max_patterns_s = arg_value(argc, argv, "--max-patterns");
    int runs = runs_s ? atoi(runs_s) : 3;
    if (runs < 1) runs = 1;
    dataset_root = dataset_root ? dataset_root : "datasets/itemsets";
    out_root = out_root ? out_root : "results";
    double max_seconds = max_seconds_s ? atof(max_seconds_s) : 30.0;
    int max_patterns = max_patterns_s ? atoi(max_patterns_s) : 500000;

    char itemsets_root[1024];
    snprintf(itemsets_root, sizeof(itemsets_root), "%s/itemsets", dataset_root);
    DIR *probe = opendir(itemsets_root);
    if (probe) {
        closedir(probe);
        dataset_root = itemsets_root;
    }

    char path[1024];
    snprintf(path, sizeof(path), "%s/csv", out_root); dm_ensure_dir(path);
    snprintf(path, sizeof(path), "%s/patterns", out_root); dm_ensure_dir(path);
    snprintf(path, sizeof(path), "%s/reports", out_root); dm_ensure_dir(path);
    snprintf(path, sizeof(path), "%s/tmp", out_root); dm_ensure_dir(path);

    PathList datasets = {0};
    collect_txt(&datasets, dataset_root);
    if (datasets.count == 0) {
        fprintf(stderr, "No .txt datasets found under %s\n", dataset_root);
        return 1;
    }

    const char **stat_paths = malloc((size_t)datasets.count * sizeof(char *));
    for (int i = 0; i < datasets.count; i++) stat_paths[i] = datasets.paths[i];
    snprintf(path, sizeof(path), "%s/csv/dataset_stats.csv", out_root);
    dm_dataset_stats_write(path, stat_paths, datasets.count);
    free(stat_paths);

    snprintf(path, sizeof(path), "%s/csv/results_raw.csv", out_root);
    dm_csv_write_raw_header(path);
    append_summary_headers(out_root);
    char dominance_path[1024];
    snprintf(dominance_path, sizeof(dominance_path), "%s/csv/dominance_examples.csv", out_root);
    int dominance_append = 0;

    double minsups[] = {0.01, 0.02, 0.05, 0.10};
    double minoccs[] = {0.2, 0.4, 0.6, 0.8};
    MFHOIAlgorithm algos[] = {MFHOI_ALGO_APRIORI, MFHOI_ALGO_MFI, MFHOI_ALGO_FHOI, MFHOI_ALGO_WEAK, MFHOI_ALGO_STRONG};

    for (int d = 0; d < datasets.count; d++) {
        TransactionDB *db = mfhoi_load_transaction_db(datasets.paths[d]);
        if (!db) continue;
        db->name = dm_path_basename(datasets.paths[d]);
        char dataset_name[256];
        sanitize_name(dm_path_basename(datasets.paths[d]), dataset_name, sizeof(dataset_name));

        for (size_t ms = 0; ms < sizeof(minsups) / sizeof(minsups[0]); ms++) {
            int minsup_count = (int)ceil(minsups[ms] * (double)db->transaction_count);
            if (minsup_count < 1) minsup_count = 1;
            for (size_t mo = 0; mo < sizeof(minoccs) / sizeof(minoccs[0]); mo++) {
                for (int run = 1; run <= runs; run++) {
                    MFHOILimits limits = { .max_patterns = max_patterns, .max_seconds = max_seconds };
                    MFHOIResult result;
                    double start = dm_timer_now_seconds();
                    int status = mfhoi_mine_all(db, minsup_count, minoccs[mo], &limits, &result);
                    double runtime = dm_timer_now_seconds() - start;
                    double peak_ram = get_peak_ram_mb();

                    append_compression(out_root, dataset_name, minsups[ms], minoccs[mo], &result);
                    append_overlap(out_root, dataset_name, minsups[ms], minoccs[mo], &result);
                    mfhoi_write_dominance_examples(dominance_path, db, minsups[ms], minoccs[mo], &result.fhoi, dominance_append);
                    dominance_append = 1;

                    for (size_t a = 0; a < sizeof(algos) / sizeof(algos[0]); a++) {
                        const PatternList *patterns = mfhoi_select_patterns(&result, algos[a]);
                        char pattern_path[1024];
                        snprintf(pattern_path, sizeof(pattern_path), "%s/patterns/%s_%s_%.2f_%.1f_run%d.txt",
                                 out_root, dataset_name, mfhoi_algorithm_name(algos[a]), minsups[ms], minoccs[mo], run);
                        mfhoi_write_patterns(pattern_path, patterns, db->transaction_count);

                        ExperimentRow row = {0};
                        snprintf(row.dataset, sizeof(row.dataset), "%s", dataset_name);
                        snprintf(row.algorithm, sizeof(row.algorithm), "%s", mfhoi_algorithm_name(algos[a]));
                        row.minsup_ratio = minsups[ms];
                        row.minsup_count = minsup_count;
                        row.minocc = minoccs[mo];
                        row.run_id = run;
                        snprintf(row.status, sizeof(row.status), "%s", status ? "LIMITED" : "OK");
                        row.runtime_seconds = runtime;
                        row.peak_ram_mb = peak_ram;
                        snprintf(path, sizeof(path), "%s/tmp", out_root);
                        row.temp_disk_mb = dm_directory_size_mb(path);
                        row.output_disk_mb = dm_file_size_mb(pattern_path);
                        row.num_generated_candidates = result.stats.num_generated_candidates;
                        row.num_output_itemsets = patterns->count;
                        row.num_fhoi = result.fhoi.count;
                        row.num_weak_mfhoi = result.weak_mfhoi.count;
                        row.num_strong_mfhoi = result.strong_mfhoi.count;
                        row.dominance_removed_count = result.stats.dominance_removed_count;
                        mfhoi_compute_quality(patterns, db->transaction_count, &row.quality);
                        row.compression_vs_fi = safe_ratio((double)result.frequent.count, (double)result.strong_mfhoi.count);
                        row.compression_vs_fhoi = safe_ratio((double)result.fhoi.count, (double)result.strong_mfhoi.count);
                        row.reduction_vs_fhoi_percent = result.fhoi.count ? (1.0 - (double)result.strong_mfhoi.count / (double)result.fhoi.count) * 100.0 : 0.0;
                        row.strong_extra_reduction_percent = result.weak_mfhoi.count ? (1.0 - (double)result.strong_mfhoi.count / (double)result.weak_mfhoi.count) * 100.0 : 0.0;
                        snprintf(path, sizeof(path), "%s/csv/results_raw.csv", out_root);
                        dm_csv_append_raw_row(path, &row);
                        snprintf(path, sizeof(path), "%s/csv/results_avg.csv", out_root);
                        dm_csv_append_raw_row(path, &row);
                        append_quality(out_root, dataset_name, row.algorithm, minsups[ms], minoccs[mo], &row.quality);
                    }
                    mfhoi_result_free(&result);
                }
            }
        }
        mfhoi_free_transaction_db(db);
    }

    dm_experiment_generate_report(out_root);
    paths_free(&datasets);
    printf("MFHOI experiments complete. See %s/csv and %s/reports/experiment_report.txt\n", out_root, out_root);
    return 0;
}
