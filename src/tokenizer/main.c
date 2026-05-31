#define _GNU_SOURCE
#include "../../include/tokenizer/tokenizer.h"
#include "../../include/tokenizer/bpe_subword.h"
#include "../../include/tokenizer/unigram_subword.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/mman.h>
#include <sys/resource.h>
#endif
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifdef _WIN32
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
static int clock_gettime(int unused, struct timespec *ts) {
    static LARGE_INTEGER freq;
    static int initialized = 0;
    LARGE_INTEGER counter;
    (void)unused;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    QueryPerformanceCounter(&counter);
    ts->tv_sec = (time_t)(counter.QuadPart / freq.QuadPart);
    ts->tv_nsec = (long)(((counter.QuadPart % freq.QuadPart) * 1000000000LL) / freq.QuadPart);
    return 0;
}
#endif
#endif

typedef struct {
    FILE *out_file;
    size_t transaction_count;
} EmitContext;

static void spmf_emit_callback(const uint32_t *tokens, size_t count, void *user_data) {
    EmitContext *ctx = (EmitContext *)user_data;
    if (count == 0) return;
    
    for (size_t i = 0; i < count; i++) {
        fprintf(ctx->out_file, "%u%c", tokens[i], (i == count - 1) ? '\n' : ' ');
    }
    ctx->transaction_count++;
}

static long get_peak_rss_kb(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc))) {
        return (long)(pmc.PeakWorkingSetSize / 1024);
    }
    return 0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss;
    }
    return 0;
#endif
}

static unsigned char *read_input_file(const char *path, size_t *size_out) {
    FILE *f = fopen(path, "rb");
    long size;
    unsigned char *data;
    if (!f) {
        perror("Failed to open input file");
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        perror("Failed to seek input file");
        fclose(f);
        return NULL;
    }
    size = ftell(f);
    if (size <= 0) {
        fprintf(stderr, "Empty input file\n");
        fclose(f);
        return NULL;
    }
    rewind(f);
    data = (unsigned char *)malloc((size_t)size);
    if (!data) {
        fclose(f);
        return NULL;
    }
    if (fread(data, 1, (size_t)size, f) != (size_t)size) {
        perror("Failed to read input file");
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size_out = (size_t)size;
    return data;
}

int output_json = 0;

int main(int argc, char **argv) {
    if (argc >= 2 && (strcmp(argv[1], "bpe") == 0 || strcmp(argv[1], "dm_bpe") == 0)) {
        return dm_bpe_cli(argc, argv);
    }
    if (argc >= 2 && (strcmp(argv[1], "unigram") == 0 || strcmp(argv[1], "dm_unigram") == 0)) {
        return dm_unigram_cli(argc, argv);
    }

    char *input_path = NULL;
    char *output_path = NULL;
    char *mode_str = "doc";
    char *algo_str = "faro";
    int run_benchmark = 0;
    uint32_t window_size = 10;
    uint32_t stride = 1;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            mode_str = argv[++i];
        } else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) {
            window_size = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            stride = (uint32_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--algo") == 0 && i + 1 < argc) {
            algo_str = argv[++i];
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            run_benchmark = 1;
        } else if (strcmp(argv[i], "--json") == 0) {
            output_json = 1;
        }
    }

    
    if (!input_path) {
        fprintf(stderr, "Usage: %s -i <input_file> [-o <output_file>] [-m <doc|sentence|sliding>] [-w <window_size>] [-s <stride>] [--algo <faro|...>] [--benchmark]\n", argv[0]);
        return 1;
    }
    
    TransactionMode mode = MODE_DOCUMENT;
    if (strcmp(mode_str, "sentence") == 0) {
        mode = MODE_SENTENCE;
    } else if (strcmp(mode_str, "sliding") == 0) {
        mode = MODE_SLIDING;
    }
    
    /* Create selected tokenizer */
    Tokenizer *tok = NULL;
    if (strcmp(algo_str, "lp-raw") == 0) {
        tok = lp_raw_tokenizer_create(65536);
    } else if (strcmp(algo_str, "lp-fp") == 0) {
        tok = lp_fp_tokenizer_create(65536);
    } else if (strcmp(algo_str, "rh-fp") == 0) {
        tok = rh_fp_tokenizer_create(65536);
    } else if (strcmp(algo_str, "rh-arena") == 0 || strcmp(algo_str, "faro") == 0) {
        tok = rh_arena_tokenizer_create(65536);
    } else if (strcmp(algo_str, "rh-borrow") == 0) {
        tok = rh_borrow_tokenizer_create(65536);
    } else {
        fprintf(stderr, "Error: Unknown algorithm '%s'. Supported: lp-raw, lp-fp, rh-fp, rh-arena, rh-borrow\n", algo_str);
        return 1;
    }

    
    if (!tok) {
        fprintf(stderr, "Error: Failed to initialize tokenizer '%s'\n", algo_str);
        return 1;
    }
    
    size_t file_size = 0;
    unsigned char *mapped_data = read_input_file(input_path, &file_size);
    if (!mapped_data) {
        tok->free(tok);
        return 1;
    }
    
    FILE *out = stdout;
    if (output_path) {
        out = fopen(output_path, "w");
        if (!out) {
            perror("Failed to open output file");
            free(mapped_data);
            tok->free(tok);
            return 1;
        }
    }
    
    EmitContext ctx = { .out_file = out, .transaction_count = 0 };
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    /* Run polymorphic Scan Loop */
    tok->tokenize_buffer(tok, mapped_data, file_size, mode, window_size, stride, spmf_emit_callback, &ctx);
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    double elapsed_sec = (end_time.tv_sec - start_time.tv_sec) + 
                         (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0;
    
    if (output_path) {
        fclose(out);
    }
    
    if (run_benchmark) {
        long peak_rss = get_peak_rss_kb();
        tok->print_stats(tok, input_path, file_size, elapsed_sec, ctx.transaction_count, peak_rss);
    }
    
    tok->free(tok);
    free(mapped_data);
    
    return 0;
}
