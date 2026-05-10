#include "core/dm_benchmark.h"
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#endif

static DM_BenchmarkReport report;

// High-Res Time state
#ifdef _WIN32
static LARGE_INTEGER freq;
static LARGE_INTEGER start_times[4];
#else
static struct timespec start_times[4];
#endif

// CPU Time state
#ifdef _WIN32
static FILETIME sys_start_ft_cpu, sys_start_ft_sys, sys_start_ft_user;
#else
static struct rusage start_rusage;
#endif

static double get_time_ms_diff(
#ifdef _WIN32
    LARGE_INTEGER start, LARGE_INTEGER end
#else
    struct timespec start, struct timespec end
#endif
) {
#ifdef _WIN32
    return (double)(end.QuadPart - start.QuadPart) * 1000.0 / (double)freq.QuadPart;
#else
    double s = (end.tv_sec - start.tv_sec) * 1000.0;
    double ns = (end.tv_nsec - start.tv_nsec) / 1000000.0;
    return s + ns;
#endif
}

void dm_bench_reset(void) {
    memset(&report, 0, sizeof(report));
#ifdef _WIN32
    QueryPerformanceFrequency(&freq);
#endif
}

void dm_bench_start(DM_BenchPhase phase) {
    if (phase > DM_PHASE_TOTAL) return;

#ifdef _WIN32
    QueryPerformanceCounter(&start_times[phase]);
    if (phase == DM_PHASE_ALGO) {
        FILETIME creation, exit;
        GetProcessTimes(GetCurrentProcess(), &creation, &exit, &sys_start_ft_sys, &sys_start_ft_user);
    }
#else
    clock_gettime(CLOCK_MONOTONIC, &start_times[phase]);
    if (phase == DM_PHASE_ALGO) {
        getrusage(RUSAGE_SELF, &start_rusage);
    }
#endif
}

#ifdef _WIN32
static double filetime_to_ms(FILETIME ft) {
    ULARGE_INTEGER uli;
    uli.LowPart = ft.dwLowDateTime;
    uli.HighPart = ft.dwHighDateTime;
    return (double)uli.QuadPart / 10000.0;
}
#endif

void dm_bench_stop(DM_BenchPhase phase) {
    if (phase > DM_PHASE_TOTAL) return;

#ifdef _WIN32
    LARGE_INTEGER end_time;
    QueryPerformanceCounter(&end_time);
    report.phase_times_ms[phase] = get_time_ms_diff(start_times[phase], end_time);

    if (phase == DM_PHASE_ALGO) {
        FILETIME creation, exit, kernel_time, user_time;
        GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel_time, &user_time);
        report.user_cpu_ms = filetime_to_ms(user_time) - filetime_to_ms(sys_start_ft_user);
        report.sys_cpu_ms = filetime_to_ms(kernel_time) - filetime_to_ms(sys_start_ft_sys);
    }
#else
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    report.phase_times_ms[phase] = get_time_ms_diff(start_times[phase], end_time);

    if (phase == DM_PHASE_ALGO) {
        struct rusage end_rusage;
        getrusage(RUSAGE_SELF, &end_rusage);
        double usr_start = start_rusage.ru_utime.tv_sec * 1000.0 + start_rusage.ru_utime.tv_usec / 1000.0;
        double usr_end = end_rusage.ru_utime.tv_sec * 1000.0 + end_rusage.ru_utime.tv_usec / 1000.0;
        double sys_start = start_rusage.ru_stime.tv_sec * 1000.0 + start_rusage.ru_stime.tv_usec / 1000.0;
        double sys_end = end_rusage.ru_stime.tv_sec * 1000.0 + end_rusage.ru_stime.tv_usec / 1000.0;
        report.user_cpu_ms = usr_end - usr_start;
        report.sys_cpu_ms = sys_end - sys_start;
    }
#endif
}

void dm_bench_record_results(size_t num_itemsets, size_t total_items) {
    report.num_itemsets = num_itemsets;
    report.total_items = total_items;
    report.result_ram_bytes = num_itemsets * 24 + total_items * 4;
    report.result_disk_est_bytes = total_items * 5 + num_itemsets * 8;
}

static void read_peak_memory(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        report.peak_memory_kb = pmc.PeakWorkingSetSize / 1024;
    }
#else
    FILE* file = fopen("/proc/self/status", "r");
    if (file) {
        char line[128];
        while (fgets(line, sizeof(line), file)) {
            if (strncmp(line, "VmHWM:", 6) == 0) {
                long long v;
                if (sscanf(line + 6, "%lld", &v) == 1) {
                    report.peak_memory_kb = (size_t)v;
                }
                break;
            }
        }
        fclose(file);
    } else {
        struct rusage r_usage;
        getrusage(RUSAGE_SELF, &r_usage);
#ifdef __APPLE__
        report.peak_memory_kb = r_usage.ru_maxrss / 1024;
#else
        report.peak_memory_kb = r_usage.ru_maxrss;
#endif
    }
#endif
}

DM_BenchmarkReport dm_bench_get_report(void) {
    read_peak_memory();
    return report;
}

void dm_bench_print_report(const char *algo_name, const char *dataset_name) {
    read_peak_memory();
    
    printf("\n");
    printf("============================================================\n");
    printf("                  DATA MINING BENCHMARK REPORT              \n");
    printf("============================================================\n");
    printf(" Algorithm   : %s\n", algo_name);
    printf(" Dataset     : %s\n", dataset_name);
    printf("------------------------------------------------------------\n");
    printf(" [1] TIMING (High-Res)\n");
    printf("     - I/O Load Data    : %10.3f ms\n", report.phase_times_ms[DM_PHASE_LOAD]);
    printf("     - Algorithm Core   : %10.3f ms\n", report.phase_times_ms[DM_PHASE_ALGO]);
    printf("     - Write Results    : %10.3f ms\n", report.phase_times_ms[DM_PHASE_WRITE]);
    printf("     - TOTAL WALL TIME  : %10.3f ms\n", report.phase_times_ms[DM_PHASE_TOTAL]);
    printf("------------------------------------------------------------\n");
    printf(" [2] CPU UTILIZATION\n");
    printf("     - User Mode (CPU)  : %10.3f ms\n", report.user_cpu_ms);
    printf("     - Kernel Mode (SYS): %10.3f ms\n", report.sys_cpu_ms);
    printf("     - Threading Score  : %10.1f %%\n", 
            (report.phase_times_ms[DM_PHASE_ALGO] > 0) ? 
            ((report.user_cpu_ms + report.sys_cpu_ms) / report.phase_times_ms[DM_PHASE_ALGO] * 100.0) : 0.0);
    printf("------------------------------------------------------------\n");
    printf(" [3] MEMORY PROFILING\n");
    printf("     - Peak RAM (VmHWM) : %10.2f MB  (%zu KB)\n", report.peak_memory_kb / 1024.0, report.peak_memory_kb);
    printf("------------------------------------------------------------\n");
    printf(" [4] STORAGE FOOTPRINT (Result Set)\n");
    printf("     - RAM Occupied     : %10.2f MB  (%zu Bytes)\n", report.result_ram_bytes / 1048576.0, report.result_ram_bytes);
    printf("     - Est. Disk (.txt) : %10.2f MB  (%zu Bytes)\n", report.result_disk_est_bytes / 1048576.0, report.result_disk_est_bytes);
    printf("------------------------------------------------------------\n");
    printf(" [5] MINING RESULTS\n");
    printf("     - Frequent Itemsets: %zu\n", report.num_itemsets);
    printf("     - Total Items      : %zu\n", report.total_items);
    printf("============================================================\n\n");
}
