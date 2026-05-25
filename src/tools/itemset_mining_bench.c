#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define mkdir(path, mode) _mkdir(path)
#define pclose _pclose
#define popen _popen
#define sleep(seconds) Sleep((seconds) * 1000)
#define WEXITSTATUS(status) (status)
#define WIFEXITED(status) (1)
#else
#include <sys/wait.h>
#include <unistd.h>
#endif
#include <time.h>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

typedef struct {
    const char *algo;
    int type_id;
    const char *thresholds_raw;
    const char *dataset_path;
    const char *out_path;
    const char *extra_args;
    int timeout_seconds;
    int pause_seconds;
    int replace_output;
    int mem_mb;
} BenchConfig;

typedef struct {
    char threshold[64];
    char status[32];
    char itemsets[64];
    char ram_mb[64];
    char disk_mb[64];
    char runtime_s[64];
} BenchRow;

static void usage(const char *prog) {
    printf("Usage:\n");
    printf("  %s <algorithm> <type_id> \"<minsup list>\" <dataset_path> [options]\n\n", prog);
    printf("Examples:\n");
    printf("  %s fpmax 0 \"0.1 0.2 0.3 0.4 0.5\" datasets/itemsets/mushrooms.txt\n", prog);
    printf("  %s fpmax 0 \"<0.1 0.2 0.3 0.4 0.5>\" datasets/itemsets/mushrooms.txt\n", prog);
    printf("  %s mfhoi 0 \"0.1 0.2 0.3\" datasets/itemsets/mushrooms.txt --extra \"0.3\"\n\n", prog);
    printf("Options:\n");
    printf("  --out <path>        Text report path. Default: results/itemset_mining_bench.txt\n");
    printf("  --timeout <sec>     Per-threshold timeout. Default: 120\n");
    printf("  --pause <sec>       Pause between thresholds. Default: 1\n");
    printf("  --extra \"args\"      Extra args passed after threshold, matching main.c params.\n");
    printf("  --mem-mb <mb>       Virtual memory cap for each child process. Default: 1536\n");
    printf("  --replace           Replace output file instead of appending a new section.\n");
    printf("\nType ids match main.c: 0=Transactional, 1=Utility, 2=Matrix, 4=Quantity.\n");
    printf("By default this tool appends to the report so old benchmarks are preserved.\n");
}

static int ensure_parent_dir(const char *path) {
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s", path);
    char *slash = strrchr(buf, '/');
    if (!slash) return 0;
    *slash = '\0';
    if (buf[0] == '\0') return 0;

    char partial[1024] = {0};
    char *p = buf;
    if (*p == '/') {
        strcpy(partial, "/");
        p++;
    }

    char *tok = strtok(p, "/");
    while (tok) {
        if (strlen(partial) > 1) strcat(partial, "/");
        strcat(partial, tok);
        mkdir(partial, 0775);
        tok = strtok(NULL, "/");
    }
    return 0;
}

static char *shell_quote(const char *s) {
    size_t len = 2;
    for (const char *p = s; *p; p++) len += (*p == '\'') ? 4 : 1;
    char *out = malloc(len + 1);
    char *w = out;
    *w++ = '\'';
    for (const char *p = s; *p; p++) {
        if (*p == '\'') {
            memcpy(w, "'\\''", 4);
            w += 4;
        } else {
            *w++ = *p;
        }
    }
    *w++ = '\'';
    *w = '\0';
    return out;
}

static void strip_angle_brackets(char *s) {
    while (isspace((unsigned char)*s)) memmove(s, s + 1, strlen(s));
    size_t n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) s[--n] = '\0';
    if (n > 0 && s[0] == '<') {
        memmove(s, s + 1, n);
        n--;
    }
    if (n > 0 && s[n - 1] == '>') s[n - 1] = '\0';
    for (char *p = s; *p; p++) {
        if (*p == ',') *p = ' ';
    }
}

static void parse_metric(const char *line, const char *label, char *out, size_t out_sz) {
    const char *p = strstr(line, label);
    if (!p) return;
    p = strchr(p, ':');
    if (!p) return;
    p++;
    while (isspace((unsigned char)*p)) p++;

    size_t i = 0;
    while (p[i] && !isspace((unsigned char)p[i]) && i + 1 < out_sz) {
        out[i] = p[i];
        i++;
    }
    out[i] = '\0';
}

static void ms_to_seconds(const char *ms, char *out, size_t out_sz) {
    if (strcmp(ms, "NA") == 0 || ms[0] == '\0') {
        snprintf(out, out_sz, "NA");
        return;
    }
    snprintf(out, out_sz, "%.6f", atof(ms) / 1000.0);
}

static int run_one(const BenchConfig *cfg, const char *threshold, BenchRow *row) {
    snprintf(row->threshold, sizeof(row->threshold), "%s", threshold);
    snprintf(row->status, sizeof(row->status), "OK");
    snprintf(row->itemsets, sizeof(row->itemsets), "NA");
    snprintf(row->ram_mb, sizeof(row->ram_mb), "NA");
    snprintf(row->disk_mb, sizeof(row->disk_mb), "NA");
    snprintf(row->runtime_s, sizeof(row->runtime_s), "NA");

    char *algo_q = shell_quote(cfg->algo);
    char *dataset_q = shell_quote(cfg->dataset_path);
    char *threshold_q = shell_quote(threshold);

    char command[4096];
    if (cfg->extra_args && cfg->extra_args[0]) {
        snprintf(command, sizeof(command),
                 "sh -c 'ulimit -v %d; exec timeout %d ./bin/dm.exe %s %s %d %s %s' 2>&1",
                 cfg->mem_mb * 1024, cfg->timeout_seconds, algo_q, dataset_q, cfg->type_id, threshold_q, cfg->extra_args);
    } else {
        snprintf(command, sizeof(command),
                 "sh -c 'ulimit -v %d; exec timeout %d ./bin/dm.exe %s %s %d %s' 2>&1",
                 cfg->mem_mb * 1024, cfg->timeout_seconds, algo_q, dataset_q, cfg->type_id, threshold_q);
    }

    free(algo_q);
    free(dataset_q);
    free(threshold_q);

    FILE *pipe = popen(command, "r");
    if (!pipe) {
        snprintf(row->status, sizeof(row->status), "FAILED");
        return -1;
    }

    char line[1024];
    char runtime_ms[64] = "NA";
    while (fgets(line, sizeof(line), pipe)) {
        parse_metric(line, "- Frequent Itemsets", row->itemsets, sizeof(row->itemsets));
        parse_metric(line, "- Peak RAM (VmHWM)", row->ram_mb, sizeof(row->ram_mb));
        parse_metric(line, "- Est. Disk (.txt)", row->disk_mb, sizeof(row->disk_mb));
        parse_metric(line, "- TOTAL WALL TIME", runtime_ms, sizeof(runtime_ms));
    }

    int status = pclose(pipe);
    if (WIFEXITED(status)) {
        int code = WEXITSTATUS(status);
        if (code == 124) snprintf(row->status, sizeof(row->status), "TIMEOUT");
        else if (code != 0) snprintf(row->status, sizeof(row->status), "FAILED(%d)", code);
    } else if (status != 0) {
        snprintf(row->status, sizeof(row->status), "FAILED");
    }

    ms_to_seconds(runtime_ms, row->runtime_s, sizeof(row->runtime_s));

#if defined(__GLIBC__)
    malloc_trim(0);
#endif

    return 0;
}

static int parse_args(int argc, char **argv, BenchConfig *cfg) {
    cfg->algo = NULL;
    cfg->type_id = 0;
    cfg->thresholds_raw = NULL;
    cfg->dataset_path = NULL;
    cfg->out_path = "results/itemset_mining_bench.txt";
    cfg->extra_args = "";
    cfg->timeout_seconds = 120;
    cfg->pause_seconds = 1;
    cfg->replace_output = 0;
    cfg->mem_mb = 1536;

    if (argc < 5) return -1;
    cfg->algo = argv[1];
    cfg->type_id = atoi(argv[2]);
    cfg->thresholds_raw = argv[3];
    cfg->dataset_path = argv[4];

    for (int i = 5; i < argc; i++) {
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            cfg->out_path = argv[++i];
        } else if (strcmp(argv[i], "--timeout") == 0 && i + 1 < argc) {
            cfg->timeout_seconds = atoi(argv[++i]);
            if (cfg->timeout_seconds <= 0) cfg->timeout_seconds = 120;
        } else if (strcmp(argv[i], "--pause") == 0 && i + 1 < argc) {
            cfg->pause_seconds = atoi(argv[++i]);
            if (cfg->pause_seconds < 0) cfg->pause_seconds = 0;
        } else if (strcmp(argv[i], "--extra") == 0 && i + 1 < argc) {
            cfg->extra_args = argv[++i];
        } else if (strcmp(argv[i], "--mem-mb") == 0 && i + 1 < argc) {
            cfg->mem_mb = atoi(argv[++i]);
            if (cfg->mem_mb <= 0) cfg->mem_mb = 1536;
        } else if (strcmp(argv[i], "--replace") == 0) {
            cfg->replace_output = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            return -1;
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            return -1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    BenchConfig cfg;
    if (parse_args(argc, argv, &cfg) != 0) {
        usage(argv[0]);
        return 1;
    }

    char thresholds_buf[2048];
    snprintf(thresholds_buf, sizeof(thresholds_buf), "%s", cfg.thresholds_raw);
    strip_angle_brackets(thresholds_buf);

    ensure_parent_dir(cfg.out_path);
    FILE *out = fopen(cfg.out_path, cfg.replace_output ? "w" : "a");
    if (!out) {
        fprintf(stderr, "Cannot write %s: %s\n", cfg.out_path, strerror(errno));
        return 1;
    }

    time_t now = time(NULL);
    if (!cfg.replace_output) fprintf(out, "\n\n");
    fprintf(out, "Itemset Mining Benchmark\n");
    fprintf(out, "Started: %s", ctime(&now));
    fprintf(out, "Algorithm: %s\n", cfg.algo);
    fprintf(out, "Dataset: %s\n", cfg.dataset_path);
    fprintf(out, "Type ID: %d\n", cfg.type_id);
    fprintf(out, "Thresholds: %s\n", thresholds_buf);
    fprintf(out, "Extra args: %s\n", cfg.extra_args[0] ? cfg.extra_args : "(none)");
    fprintf(out, "Timeout per run: %d seconds\n\n", cfg.timeout_seconds);
    fprintf(out, "Memory cap per run: %d MB\n", cfg.mem_mb);
    fprintf(out, "Pause between runs: %d seconds\n", cfg.pause_seconds);
    fprintf(out, "Memory note: each threshold is executed in a separate dm.exe child process; after the child exits, OS-owned memory is released. The parent also trims its heap between runs when glibc supports it.\n\n");
    fprintf(out, "%-12s %-14s %-12s %-12s %-12s %-12s\n",
            "threshold", "#itemsets", "ram_mb", "disk_mb", "runtime_s", "status");
    fprintf(out, "----------------------------------------------------------------------------\n");

    printf("Writing benchmark txt: %s\n", cfg.out_path);

    char *saveptr = NULL;
#ifdef _WIN32
    char *tok = strtok_s(thresholds_buf, " \t\r\n", &saveptr);
#else
    char *tok = strtok_r(thresholds_buf, " \t\r\n", &saveptr);
#endif
    int count = 0;
    while (tok) {
        BenchRow row;
        printf("[bench] %s threshold=%s dataset=%s\n", cfg.algo, tok, cfg.dataset_path);
        run_one(&cfg, tok, &row);
        fprintf(out, "%-12s %-14s %-12s %-12s %-12s %-12s\n",
                row.threshold, row.itemsets, row.ram_mb, row.disk_mb, row.runtime_s, row.status);
        fflush(out);
        count++;
        if (cfg.pause_seconds > 0) sleep((unsigned int)cfg.pause_seconds);
#ifdef _WIN32
        tok = strtok_s(NULL, " \t\r\n", &saveptr);
#else
        tok = strtok_r(NULL, " \t\r\n", &saveptr);
#endif
    }

    fclose(out);
    if (count == 0) {
        fprintf(stderr, "No thresholds found.\n");
        return 1;
    }

    printf("Done: %s\n", cfg.out_path);
    return 0;
}
