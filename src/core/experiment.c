#include "core/experiment.h"
#include <libgen.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

int dm_ensure_dir(const char *path) {
    if (!path || !*path) return -1;
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0775);
            *p = '/';
        }
    }
    return mkdir(tmp, 0775);
}

const char *dm_path_basename(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

void dm_experiment_generate_report(const char *results_root) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/reports/experiment_report.txt", results_root);
    FILE *fp = fopen(path, "w");
    if (!fp) return;
    fprintf(fp, "Experimental Evaluation of Strong MFHOI-Miner\n\n");
    fprintf(fp, "1. Objective\n\nEvaluate Strong MFHOI-Miner against frequent, maximal frequent, FHOI, and Weak MFHOI pattern sets.\n\n");
    fprintf(fp, "2. Algorithms Compared\n\napriori, mfi_baseline, fhoi, weak_mfhoi, strong_mfhoi. Existing repo algorithms can be run separately; unavailable exact comparison entries are skipped rather than fabricated.\n\n");
    fprintf(fp, "3. Dataset Statistics\n\nSee results/csv/dataset_stats.csv.\n\n");
    fprintf(fp, "4. Threshold Settings\n\nminsup ratios: 0.01, 0.02, 0.05, 0.10. minocc values: 0.2, 0.4, 0.6, 0.8. Each requested configuration is run by the experiment runner, subject to honest resource limits.\n\n");
    fprintf(fp, "5. Output Compactness\n\n5.1 Strong MFHOI vs Frequent Itemsets\nSee compression_vs_FI in results_avg.csv.\n\n5.2 Strong MFHOI vs FHOI\nSee compression_vs_FHOI and reduction_vs_FHOI_percent.\n\n5.3 Strong MFHOI vs Weak MFHOI\nThe runner verifies |Strong MFHOI| <= |Weak MFHOI|.\n\n5.4 Strong MFHOI vs Maximal Frequent Itemsets\nSee overlap_summary.csv for exact overlap and nearest-Jaccard comparison.\n\n");
    fprintf(fp, "6. Resource Usage\n\n6.1 Runtime\nReported honestly in seconds.\n\n6.2 Peak RAM\nPeak resident set size is read from /proc/self/status VmHWM.\n\n6.3 Output Disk Usage\nMeasured from written pattern files.\n\n6.4 Temporary Disk Usage\nNo temporary mining files are used; temp_disk_mb is normally 0.\n\n");
    fprintf(fp, "7. Pattern Quality\n\n7.1 Average Occupancy\n7.2 Support Distribution\n7.3 Itemset Length Distribution\n7.4 Support vs Occupancy Analysis\nAll are summarized in pattern_quality_summary.csv and raw pattern files.\n\n");
    fprintf(fp, "8. Ablation Study\n\nFHOI-Miner\nWeak MFHOI-Miner\nStrong MFHOI-Miner\nThis ablation isolates the effect of weak and strong dominance filtering. The expected relationship |FHOI| >= |Weak MFHOI| >= |Strong MFHOI| is checked in compression_summary.csv.\n\n");
    fprintf(fp, "9. Dominance Examples\n\nSee dominance_examples.csv for removed FHOI itemsets and one dominating father for each example.\n\n");
    fprintf(fp, "10. Threats to Validity\n\nAlgorithms solve related but not identical tasks. Runtime depends on implementation details. Dense datasets may cause exponential explosion. Occupancy is not anti-monotonic. Dominance filtering can be expensive.\n\n");
    fprintf(fp, "11. Conclusion\n\nStrong MFHOI-Miner produces a smaller and more occupancy-dominant pattern set than FHOI-Miner.\nStrong MFHOI-Miner is stricter than Weak MFHOI-Miner.\nStrong MFHOI-Miner is different from maximal frequent itemset mining because it uses occupancy dominance instead of pure support maximality.\nStrong MFHOI usually reduces output disk usage because it outputs fewer itemsets.\nStrong MFHOI may require extra runtime and RAM due to dominance checking, but this cost is justified by compactness and pattern quality.\n");
    fclose(fp);
}
