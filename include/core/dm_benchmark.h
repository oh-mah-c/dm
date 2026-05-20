#ifndef DM_BENCHMARK_H
#define DM_BENCHMARK_H

#include "core/dm_common.h"
#include <stddef.h>

typedef enum {
    DM_PHASE_LOAD = 0,
    DM_PHASE_ALGO = 1,
    DM_PHASE_WRITE = 2,
    DM_PHASE_TOTAL = 3
} DM_BenchPhase;

typedef struct {
    double phase_times_ms[4];   // Load, Algo, Write, Total
    
    size_t peak_memory_kb;      // High-water mark in KB
    
    double user_cpu_ms;         // User CPU time
    double sys_cpu_ms;          // System CPU time
    
    size_t result_ram_bytes;    // Exact RAM footprint
    size_t result_disk_est_bytes; // Estimated Disk size (CSV/TXT)
    size_t num_itemsets;        // Total FIs found
    size_t total_items;         // Sum of lengths of all FIs
    
    double throughput_mb_s;     // Cache-Locality Metric (Data Size / Time)
} DM_BenchmarkReport;

typedef struct {
    int** transactions;
    size_t txn_count;
    size_t* txn_lengths;
    int max_item_id;
    size_t total_bytes;         // Total payload size in bytes
} BenchmarkDataset;

typedef void (*MiningAlgorithmHook)(const BenchmarkDataset* dataset, float min_support, float min_utility);


/**
 * @brief Initialize the benchmarking module
 */
void dm_bench_reset(void);

/**
 * @brief Start measuring a specific phase
 */
void dm_bench_start(DM_BenchPhase phase);

/**
 * @brief Stop measuring a specific phase
 */
void dm_bench_stop(DM_BenchPhase phase);

/**
 * @brief Record the size of the final itemsets for footprint calculation
 * @param num_itemsets Number of frequent itemsets found
 * @param total_items Sum of the lengths of all frequent itemsets
 */
void dm_bench_record_results(size_t num_itemsets, size_t total_items);

/**
 * @brief Generate and get the report
 */
DM_BenchmarkReport dm_bench_get_report(void);

/**
 * @brief Print a beautifully formatted benchmark table
 */
void dm_bench_print_report(const char *algo_name, const char *dataset_name);

/**
 * @brief Run a full benchmark matrix against an algorithm hook
 * @param algo_name Name of the algorithm
 * @param hook Function pointer to the algorithm hook
 */
void dm_run_benchmark(const char *algo_name, MiningAlgorithmHook hook);

#endif // DM_BENCHMARK_H
