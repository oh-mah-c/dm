#include "core/dm_benchmark.h"
#include "core/dm_flat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int output_json = 0;

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

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s --input <path> --connector spmf|text|graph [--arena-mb N]\n", prog);
    printf("     text options:  [--window N] [--stride N] [--itemset]\n");
    printf("     graph options: [--undirected]\n");
}

static void print_preview(const DM_FlatDataset *flat, size_t rows) {
    size_t limit = flat->row_count < rows ? flat->row_count : rows;
    for (size_t r = 0; r < limit; r++) {
        size_t a = flat->row_offsets[r];
        size_t b = flat->row_offsets[r + 1];
        printf("row_%zu_len=%zu row_%zu=", r, b - a, r);
        size_t n = b - a < 24 ? b - a : 24;
        for (size_t i = 0; i < n; i++) {
            if (i) printf(" ");
            printf("%u", flat->items[a + i]);
        }
        if (n < b - a) printf(" ...");
        printf("\n");
    }
}

int main(int argc, char **argv) {
    const char *input = arg_value(argc, argv, "--input", NULL);
    const char *connector = arg_value(argc, argv, "--connector", "spmf");
    if (!input || has_flag(argc, argv, "--help")) {
        usage(argv[0]);
        return input ? 0 : 1;
    }
    DM_ConnectorKind kind;
    if (dm_connector_parse_kind(connector, &kind) != 0) {
        fprintf(stderr, "Unknown connector: %s\n", connector);
        return 1;
    }
    DM_ConnectorOptions opt = dm_connector_default_options(kind);
    opt.text_window = (size_t)strtoull(arg_value(argc, argv, "--window", "64"), NULL, 10);
    opt.text_stride = (size_t)strtoull(arg_value(argc, argv, "--stride", "32"), NULL, 10);
    opt.text_sequence = !has_flag(argc, argv, "--itemset");
    opt.graph_undirected = has_flag(argc, argv, "--undirected");
    size_t arena_mb = (size_t)strtoull(arg_value(argc, argv, "--arena-mb", "256"), NULL, 10);

    DM_Arena arena;
    if (dm_arena_init(&arena, arena_mb * 1024 * 1024) != 0) {
        fprintf(stderr, "Could not allocate arena\n");
        return 1;
    }

    dm_bench_reset();
    dm_bench_start(DM_PHASE_TOTAL);
    DM_FlatDataset flat;
    DM_ConnectorStats stats;
    int rc = dm_flat_load_mmap(input, &opt, &arena, &flat, &stats);
    dm_bench_stop(DM_PHASE_TOTAL);
    DM_BenchmarkReport report = dm_bench_get_report();
    if (rc != 0) {
        dm_arena_free(&arena);
        fprintf(stderr, "Connector failed for %s\n", input);
        return 1;
    }

    printf("dm Universal Connector\n");
    printf("input=%s\n", input);
    printf("connector=%s\n", dm_connector_name(kind));
    printf("input_bytes=%zu\n", stats.input_bytes);
    printf("rows=%zu\n", stats.rows);
    printf("items=%zu\n", stats.items);
    printf("max_item=%u\n", stats.max_item);
    printf("distinct_estimate=%zu\n", stats.distinct_estimate);
    printf("avg_row_len=%.6f\n", stats.avg_row_len);
    printf("arena_capacity_mb=%zu\n", arena_mb);
    printf("arena_used_bytes=%zu\n", arena.offset);
    printf("runtime_sec=%.6f\n", report.phase_times_ms[DM_PHASE_TOTAL] / 1000.0);
    printf("peak_ram_mb=%.6f\n", report.peak_memory_kb / 1024.0);
    print_preview(&flat, 3);
    dm_arena_free(&arena);
    return 0;
}
