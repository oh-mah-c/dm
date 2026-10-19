/**
 * @file mhoui_bench.c
 * @brief Unified Standardized Benchmark Harness for MHOUI vs Baseline Algorithms
 *
 * Architecture:
 *   - Direct Mining (Ours):
 *       MHOUI-Miner operates directly in single-phase search-tree mining with
 *       joint dual upper bounds and branch Pareto dominance pruning.
 *       No candidate materialization, Filter Time = 0.0 ms.
 *
 *   - Two-Stage Baselines:
 *       1. OCEAN + Post-Filter (Shen et al. DASFAA 2016 + Modular Filter)
 *       2. HUOPM + Post-Filter (Gan et al. IEEE TCYB 2020 + Modular Filter)
 *       3. IL-HUOPM + Post-Filter (Kim et al. IEEE Access 2023 + Modular Filter)
 *       4. HOUI-Miner + Post-Filter (Unpruned HOUI enumeration + Modular Filter)
 *
 * The post-filter module (mhoui_post_filter.c) is strictly pluggable and decoupled
 * from baseline mining kernels.
 */

#include "algorithms/mhoui.h"
#include "algorithms/ocean.h"
#include "algorithms/huopm.h"
#include "algorithms/il_huopm.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset.h"
#include "core/dm_dataset_types.h"
#include "core/mhoui_post_filter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

int output_json = 0;

typedef struct {
    const char *algorithm;
    double time_mine_sec;
    double time_filter_sec;
    double time_total_sec;
    size_t visited_nodes;
    size_t candidate_count;
    size_t maximal_count;
    double peak_ram_mb;
    int status;
} BenchmarkResult;

static const char *arg_value(int argc, char **argv, const char *key, const char *fallback) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], key) == 0) return argv[i + 1];
    }
    return fallback;
}

static int has_flag(int argc, char **argv, const char *key) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], key) == 0) return 1;
    }
    return 0;
}

static double get_peak_memory_mb(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    return (double)r_usage.ru_maxrss / 1024.0;
#endif
}

static double total_utility(DM_Dataset *ds) {
    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    double total = 0.0;
    for (size_t i = 0; i < ds->count; i++) total += data[i].total_utility;
    return total;
}

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static uint32_t *dense_remap_dataset(DM_Dataset *ds) {
    if (!ds || ds->type != DM_TYPE_UTILITY || ds->count == 0) return NULL;
    if (ds->max_id <= 100000) return NULL; /* Already dense */

    DM_Trans_Utility *src = (DM_Trans_Utility *)ds->payload;
    size_t total_items = 0;
    for (size_t i = 0; i < ds->count; i++) total_items += src[i].count;
    if (total_items == 0) return NULL;

    uint32_t *all_ids = (uint32_t *)malloc(total_items * sizeof(uint32_t));
    if (!all_ids) return NULL;

    size_t pos = 0;
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) {
            all_ids[pos++] = src[i].items[j].id;
        }
    }

    qsort(all_ids, total_items, sizeof(uint32_t), cmp_u32);

    /* Deduplicate */
    size_t ucount = 0;
    for (size_t i = 0; i < total_items; i++) {
        if (i == 0 || all_ids[i] != all_ids[i - 1]) {
            all_ids[ucount++] = all_ids[i];
        }
    }

    uint32_t *orig_ids = (uint32_t *)malloc(ucount * sizeof(uint32_t));
    if (!orig_ids) {
        free(all_ids);
        return NULL;
    }
    memcpy(orig_ids, all_ids, ucount * sizeof(uint32_t));

    /* Remap in-place */
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < src[i].count; j++) {
            uint32_t old_id = src[i].items[j].id;
            size_t lo = 0, hi = ucount - 1;
            uint32_t new_id = 0;
            while (lo <= hi) {
                size_t mid = lo + (hi - lo) / 2;
                if (orig_ids[mid] == old_id) {
                    new_id = (uint32_t)mid;
                    break;
                }
                if (orig_ids[mid] < old_id) lo = mid + 1;
                else hi = mid - 1;
            }
            src[i].items[j].id = new_id;
        }
    }

    ds->max_id = (uint32_t)(ucount - 1);
    free(all_ids);
    printf("[Dataset Remap] Densely remapped %zu unique sparse items (max_id: %u -> %u)\n",
           ucount, orig_ids[ucount - 1], ds->max_id);
    return orig_ids;
}

static void pattern_collector_cb(const uint32_t *items, size_t length, size_t support,
                                 double utility, double occupancy, void *user_data) {
    MHOUI_PatternSet *set = (MHOUI_PatternSet *)user_data;
    mhoui_pattern_set_add(set, items, length, support, utility, occupancy);
}

/* ── Benchmark Runner for Single Algorithm ───────────────────────────────── */
static BenchmarkResult run_single_benchmark(DM_Dataset *ds, const char *algo_name,
                                            int minsup_count, double minocc, double minutil,
                                            int strong, double dbu, const char *dump_path,
                                            const uint32_t *orig_ids) {
    BenchmarkResult res = {0};
    res.algorithm = algo_name;

    struct timespec t_start, t_end;

    if (strncmp(algo_name, "direct_", 7) == 0 ||
        strcmp(algo_name, "mhoui") == 0 ||
        strncmp(algo_name, "ablation_", 9) == 0) {

        /* Direct Mining & Ablation Configurations */
        MHOUILimits limits = {
            .max_patterns = 1000000,
            .max_seconds = 0.0,
            .direct = 1,
            .strong_direct = (strcmp(algo_name, "direct_weak_mhoui") != 0),
            .disable_uub = (strcmp(algo_name, "ablation_no_uub") == 0 || strcmp(algo_name, "ablation_no_bounds") == 0),
            .disable_oub = (strcmp(algo_name, "ablation_no_oub") == 0 || strcmp(algo_name, "ablation_no_bounds") == 0),
            .disable_dominance = (strcmp(algo_name, "ablation_no_dom") == 0 || strcmp(algo_name, "ablation_no_bounds") == 0)
        };

        MHOUIResult mres;
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        int status = mhoui_mine_dataset(ds, minsup_count, minocc, minutil, &limits, &mres);
        clock_gettime(CLOCK_MONOTONIC, &t_end);

        res.time_mine_sec = (double)(t_end.tv_sec - t_start.tv_sec) +
                            (double)(t_end.tv_nsec - t_start.tv_nsec) / 1e9;
        res.time_filter_sec = 0.0; /* Direct mining: ZERO post-filtering */
        res.time_total_sec = res.time_mine_sec;
        res.candidate_count = (size_t)mres.stats.houi_count;
        res.visited_nodes = (size_t)mres.stats.visited_nodes;

        const MHOUPatternList *pats = limits.strong_direct ? &mres.strong_mhoui : &mres.weak_mhoui;
        res.maximal_count = (size_t)pats->count;
        res.peak_ram_mb = get_peak_memory_mb();
        res.status = status;

        if (dump_path) {
            FILE *fp = fopen(dump_path, "w");
            if (fp) {
                for (int i = 0; i < pats->count; i++) {
                    for (int j = 0; j < pats->patterns[i].length; j++) {
                        uint32_t it = pats->patterns[i].items[j];
                        fprintf(fp, "%u%c", orig_ids ? orig_ids[it] : it, (j + 1 == pats->patterns[i].length) ? '\n' : ' ');
                    }
                }
                fclose(fp);
            }
        }

        mhoui_result_free(&mres);

    } else if (strcmp(algo_name, "houi_filter") == 0) {

        /* Two-Stage: HOUI Candidate Enumeration + Pluggable Post-Filter */
        MHOUILimits limits = {
            .max_patterns = 1000000,
            .max_seconds = 0.0,
            .direct = 0,
            .strong_direct = strong
        };

        MHOUIResult mres;
        clock_gettime(CLOCK_MONOTONIC, &t_start);
        mhoui_mine_dataset(ds, minsup_count, minocc, minutil, &limits, &mres);
        clock_gettime(CLOCK_MONOTONIC, &t_end);

        res.time_mine_sec = (double)(t_end.tv_sec - t_start.tv_sec) +
                            (double)(t_end.tv_nsec - t_start.tv_nsec) / 1e9;

        /* Convert candidate HOUI list to MHOUI_PatternSet */
        MHOUI_PatternSet candidates;
        mhoui_pattern_set_init(&candidates);
        for (int i = 0; i < mres.houi.count; i++) {
            const MHOUPattern *p = &mres.houi.patterns[i];
            mhoui_pattern_set_add(&candidates, p->items, (size_t)p->length,
                                  (size_t)p->support, p->utility, p->avg_occupancy);
        }

        MHOUI_PatternSet maximal_set;
        size_t removed = 0;
        double filter_ms = 0.0;
        mhoui_post_filter(&candidates, strong, minutil, &maximal_set, &removed, &filter_ms);

        res.time_filter_sec = filter_ms / 1000.0;
        res.time_total_sec = res.time_mine_sec + res.time_filter_sec;
        res.candidate_count = candidates.count;
        res.maximal_count = maximal_set.count;
        res.visited_nodes = (size_t)mres.stats.visited_nodes;
        res.peak_ram_mb = get_peak_memory_mb();
        res.status = 0;

        if (dump_path) {
            FILE *fp = fopen(dump_path, "w");
            if (fp) {
                for (size_t i = 0; i < maximal_set.count; i++) {
                    for (size_t j = 0; j < maximal_set.patterns[i].length; j++) {
                        uint32_t it = maximal_set.patterns[i].items[j];
                        fprintf(fp, "%u%c", orig_ids ? orig_ids[it] : it, (j + 1 == maximal_set.patterns[i].length) ? '\n' : ' ');
                    }
                }
                fclose(fp);
            }
        }

        mhoui_pattern_set_free(&candidates);
        mhoui_pattern_set_free(&maximal_set);
        mhoui_result_free(&mres);

    } else if (strcmp(algo_name, "ocean_filter") == 0) {

        /* Two-Stage: OCEAN Candidate Mining + Pluggable Post-Filter */
        extern DM_Algorithm ocean_algo;
        extern size_t ocean_get_visited_nodes(void);
        MHOUI_PatternSet candidates;
        mhoui_pattern_set_init(&candidates);

        DM_OCEAN_Params o_params;
        o_params.min_occupancy = minocc;
        o_params.min_support = (double)minsup_count;
        o_params.on_pattern = pattern_collector_cb;
        o_params.user_data = &candidates;

        clock_gettime(CLOCK_MONOTONIC, &t_start);
        int status = (ocean_algo.run(ds, &o_params) == DM_SUCCESS) ? 0 : 1;
        clock_gettime(CLOCK_MONOTONIC, &t_end);

        res.time_mine_sec = (double)(t_end.tv_sec - t_start.tv_sec) +
                            (double)(t_end.tv_nsec - t_start.tv_nsec) / 1e9;

        MHOUI_PatternSet maximal_set;
        size_t removed = 0;
        double filter_ms = 0.0;
        mhoui_post_filter(&candidates, strong, minutil, &maximal_set, &removed, &filter_ms);

        res.time_filter_sec = filter_ms / 1000.0;
        res.time_total_sec = res.time_mine_sec + res.time_filter_sec;
        res.candidate_count = candidates.count;
        res.maximal_count = maximal_set.count;
        res.visited_nodes = ocean_get_visited_nodes();
        res.peak_ram_mb = get_peak_memory_mb();
        res.status = status;

        if (dump_path) {
            FILE *fp = fopen(dump_path, "w");
            if (fp) {
                for (size_t i = 0; i < maximal_set.count; i++) {
                    for (size_t j = 0; j < maximal_set.patterns[i].length; j++) {
                        uint32_t it = maximal_set.patterns[i].items[j];
                        fprintf(fp, "%u%c", orig_ids ? orig_ids[it] : it, (j + 1 == maximal_set.patterns[i].length) ? '\n' : ' ');
                    }
                }
                fclose(fp);
            }
        }

        mhoui_pattern_set_free(&candidates);
        mhoui_pattern_set_free(&maximal_set);

    } else if (strcmp(algo_name, "huopm_filter") == 0) {

        /* Two-Stage: HUOPM Candidate Mining + Pluggable Post-Filter */
        extern DM_Algorithm huopm_algo;
        extern size_t huopm_get_visited_nodes(void);
        MHOUI_PatternSet candidates;
        mhoui_pattern_set_init(&candidates);

        DM_HUOPM_Params h_params;
        h_params.min_occupancy = minocc;
        h_params.min_support = (double)minsup_count;
        h_params.on_pattern = pattern_collector_cb;
        h_params.user_data = &candidates;

        clock_gettime(CLOCK_MONOTONIC, &t_start);
        int status = (huopm_algo.run(ds, &h_params) == DM_SUCCESS) ? 0 : 1;
        clock_gettime(CLOCK_MONOTONIC, &t_end);

        res.time_mine_sec = (double)(t_end.tv_sec - t_start.tv_sec) +
                            (double)(t_end.tv_nsec - t_start.tv_nsec) / 1e9;

        MHOUI_PatternSet maximal_set;
        size_t removed = 0;
        double filter_ms = 0.0;
        mhoui_post_filter(&candidates, strong, minutil, &maximal_set, &removed, &filter_ms);

        res.time_filter_sec = filter_ms / 1000.0;
        res.time_total_sec = res.time_mine_sec + res.time_filter_sec;
        res.candidate_count = candidates.count;
        res.maximal_count = maximal_set.count;
        res.visited_nodes = huopm_get_visited_nodes();
        res.peak_ram_mb = get_peak_memory_mb();
        res.status = status;

        if (dump_path) {
            FILE *fp = fopen(dump_path, "w");
            if (fp) {
                for (size_t i = 0; i < maximal_set.count; i++) {
                    for (size_t j = 0; j < maximal_set.patterns[i].length; j++) {
                        uint32_t it = maximal_set.patterns[i].items[j];
                        fprintf(fp, "%u%c", orig_ids ? orig_ids[it] : it, (j + 1 == maximal_set.patterns[i].length) ? '\n' : ' ');
                    }
                }
                fclose(fp);
            }
        }

        mhoui_pattern_set_free(&candidates);
        mhoui_pattern_set_free(&maximal_set);

    } else if (strcmp(algo_name, "il_huopm_filter") == 0) {

        /* Two-Stage: IL-HUOPM Candidate Mining + Pluggable Post-Filter */
        extern DM_Algorithm il_huopm_algo;
        extern size_t il_huopm_get_visited_nodes(void);
        MHOUI_PatternSet candidates;
        mhoui_pattern_set_init(&candidates);

        DM_IL_HUOPM_Params il_params;
        il_params.min_occupancy = minocc;
        il_params.min_support = (double)minsup_count;
        il_params.on_pattern = pattern_collector_cb;
        il_params.user_data = &candidates;

        clock_gettime(CLOCK_MONOTONIC, &t_start);
        int status = (il_huopm_algo.run(ds, &il_params) == DM_SUCCESS) ? 0 : 1;
        clock_gettime(CLOCK_MONOTONIC, &t_end);

        res.time_mine_sec = (double)(t_end.tv_sec - t_start.tv_sec) +
                            (double)(t_end.tv_nsec - t_start.tv_nsec) / 1e9;

        MHOUI_PatternSet maximal_set;
        size_t removed = 0;
        double filter_ms = 0.0;
        mhoui_post_filter(&candidates, strong, minutil, &maximal_set, &removed, &filter_ms);

        res.time_filter_sec = filter_ms / 1000.0;
        res.time_total_sec = res.time_mine_sec + res.time_filter_sec;
        res.candidate_count = candidates.count;
        res.maximal_count = maximal_set.count;
        res.visited_nodes = il_huopm_get_visited_nodes();
        res.peak_ram_mb = get_peak_memory_mb();
        res.status = status;

        if (dump_path) {
            FILE *fp = fopen(dump_path, "w");
            if (fp) {
                for (size_t i = 0; i < maximal_set.count; i++) {
                    for (size_t j = 0; j < maximal_set.patterns[i].length; j++) {
                        uint32_t it = maximal_set.patterns[i].items[j];
                        fprintf(fp, "%u%c", orig_ids ? orig_ids[it] : it, (j + 1 == maximal_set.patterns[i].length) ? '\n' : ' ');
                    }
                }
                fclose(fp);
            }
        }

        mhoui_pattern_set_free(&candidates);
        mhoui_pattern_set_free(&maximal_set);

    } else {
        fprintf(stderr, "Unknown algorithm: %s\n", algo_name);
        res.status = -1;
    }

    (void)dbu;
    return res;
}

static void print_benchmark_table(const BenchmarkResult *results, size_t count) {
    printf("\n%-22s | %-10s | %-10s | %-10s | %-13s | %-11s | %-9s | %-11s\n",
           "Algorithm", "Mining (s)", "Filter (s)", "Total (s)", "Visited Nodes", "Candidates", "Maximal", "Peak RAM(MB)");
    printf("-----------------------+------------+------------+------------+---------------+-------------+-----------+-------------\n");
    for (size_t i = 0; i < count; i++) {
        const BenchmarkResult *r = &results[i];
        printf("%-22s | %10.4f | %10.4f | %10.4f | %13zu | %11zu | %9zu | %11.2f\n",
               r->algorithm, r->time_mine_sec, r->time_filter_sec, r->time_total_sec,
               r->visited_nodes, r->candidate_count, r->maximal_count, r->peak_ram_mb);
    }
    printf("--------------------------------------------------------------------------------------------------------------------\n\n");
}

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s --input <utility_dataset> [options]\n\n", prog);
    printf("Options:\n");
    printf("  --algorithm <name>    direct_strong_mhoui | direct_weak_mhoui | houi_filter |\n");
    printf("                        ocean_filter | huopm_filter | il_huopm_filter | all\n");
    printf("                        (default: all)\n");
    printf("  --minsup <val>        Minimum support ratio (e.g. 0.05) or absolute count (default: 0.05)\n");
    printf("  --minocc <val>        Minimum occupancy threshold beta in (0, 1] (default: 0.40)\n");
    printf("  --minutil <val>       Absolute minimum utility threshold mu\n");
    printf("  --minutil-ratio <val> Minimum utility as ratio of total database utility (e.g. 0.005)\n");
    printf("  --strong <0|1>        1 for Strong Dominance, 0 for Weak Dominance (default: 1)\n");
    printf("  --dump-patterns <path>Dump maximal patterns to file (one per line)\n");
    printf("  --csv <path>          Save benchmark results table to CSV file\n");
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    const char *algo_req = arg_value(argc, argv, "--algorithm", "all");
    const char *minsup_s = arg_value(argc, argv, "--minsup", "0.05");
    const char *minocc_s = arg_value(argc, argv, "--minocc", "0.40");
    const char *minutil_s = arg_value(argc, argv, "--minutil", NULL);
    const char *minutil_ratio_s = arg_value(argc, argv, "--minutil-ratio", NULL);
    const char *strong_s = arg_value(argc, argv, "--strong", "1");
    const char *dump_path = arg_value(argc, argv, "--dump-patterns", NULL);
    const char *csv_out = arg_value(argc, argv, "--csv", NULL);

    if (!input || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return input ? 0 : 1;
    }

    DM_Dataset *ds = dm_dataset_load(input, DM_TYPE_UTILITY);
    if (!ds) {
        fprintf(stderr, "Could not load utility dataset: %s\n", input);
        return 1;
    }

    uint32_t *orig_ids = dense_remap_dataset(ds);

    double minsup_val = atof(minsup_s);
    int minsup_count = minsup_val < 1.0 ? (int)ceil(minsup_val * (double)ds->count) : (int)minsup_val;
    if (minsup_count < 1) minsup_count = 1;

    double minocc = atof(minocc_s);
    double dbu = total_utility(ds);
    double minutil = 0.0;
    if (minutil_s) {
        minutil = atof(minutil_s);
    } else if (minutil_ratio_s) {
        minutil = ceil(atof(minutil_ratio_s) * dbu);
    } else {
        minutil = ceil(0.005 * dbu);
    }
    int strong = atoi(strong_s);

    printf("======================================================================================================\n");
    printf("MHOUI STANDARDIZED BENCHMARK HARNESS (Direct Single-Phase vs Pluggable Two-Stage Baselines)\n");
    printf("======================================================================================================\n");
    printf("Dataset:            %s\n", input);
    printf("Transactions:       %zu\n", ds->count);
    printf("Total DB Utility:   %.2f\n", dbu);
    printf("MinSup:             %.6f (count = %d)\n", minsup_val, minsup_count);
    printf("MinOccupancy:       %.4f\n", minocc);
    printf("MinUtility:         %.2f (ratio = %.6f)\n", minutil, dbu > 0.0 ? minutil / dbu : 0.0);
    printf("Pareto Mode:        %s\n", strong ? "Strong Pareto Dominance" : "Weak Pareto Dominance");
    printf("======================================================================================================\n");

    const char *all_algos[] = {
        "direct_strong_mhoui",
        "houi_filter",
        "ocean_filter",
        "huopm_filter",
        "il_huopm_filter"
    };
    size_t num_to_run = 0;
    const char *run_list[10];

    if (strcmp(algo_req, "all") == 0) {
        num_to_run = sizeof(all_algos) / sizeof(all_algos[0]);
        for (size_t i = 0; i < num_to_run; i++) run_list[i] = all_algos[i];
    } else {
        run_list[0] = algo_req;
        num_to_run = 1;
    }

    BenchmarkResult results[10];
    for (size_t i = 0; i < num_to_run; i++) {
        printf("Running %-20s ... ", run_list[i]);
        fflush(stdout);
        results[i] = run_single_benchmark(ds, run_list[i], minsup_count, minocc, minutil, strong, dbu, dump_path, orig_ids);
        printf("done in %.4f s (Maximal: %zu, Visited: %zu)\n", results[i].time_total_sec, results[i].maximal_count, results[i].visited_nodes);
    }

    print_benchmark_table(results, num_to_run);

    if (csv_out) {
        FILE *f = fopen(csv_out, "w");
        if (f) {
            fprintf(f, "algorithm,dataset,minsup_count,minocc,minutil,time_mine_sec,time_filter_sec,time_total_sec,visited_nodes,candidates,maximal,peak_ram_mb\n");
            for (size_t i = 0; i < num_to_run; i++) {
                const BenchmarkResult *r = &results[i];
                fprintf(f, "%s,%s,%d,%.6f,%.2f,%.6f,%.6f,%.6f,%zu,%zu,%zu,%.2f\n",
                        r->algorithm, input, minsup_count, minocc, minutil,
                        r->time_mine_sec, r->time_filter_sec, r->time_total_sec,
                        r->visited_nodes, r->candidate_count, r->maximal_count, r->peak_ram_mb);
            }
            fclose(f);
            printf("Saved benchmark CSV to %s\n", csv_out);
        }
    }

    free(orig_ids);
    dm_dataset_free(ds);
    return 0;
}
