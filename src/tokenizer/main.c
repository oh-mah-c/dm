#define _GNU_SOURCE
#include "../../include/tokenizer/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/resource.h>

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
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss;
    }
    return 0;
}

int output_json = 0;

int main(int argc, char **argv) {
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
    
    int fd = open(input_path, O_RDONLY);
    if (fd < 0) {
        perror("Failed to open input file");
        tok->free(tok);
        return 1;
    }
    
    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("Failed to get file stats");
        close(fd);
        tok->free(tok);
        return 1;
    }
    
    size_t file_size = st.st_size;
    unsigned char *mapped_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_data == MAP_FAILED) {
        perror("mmap failed");
        close(fd);
        tok->free(tok);
        return 1;
    }
    
    madvise(mapped_data, file_size, MADV_SEQUENTIAL);
    
    FILE *out = stdout;
    if (output_path) {
        out = fopen(output_path, "w");
        if (!out) {
            perror("Failed to open output file");
            munmap(mapped_data, file_size);
            close(fd);
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
    munmap(mapped_data, file_size);
    close(fd);
    
    return 0;
}
