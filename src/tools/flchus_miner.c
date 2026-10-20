/*
 * flchus_miner: standalone driver for the FLCHUSM algorithms (flchuspm, flchus_x), used for
 * benchmarking without the libraries linked into dm.exe. Same command line and environment
 * variables as "dm.exe flchuspm|flchus_x":
 *
 *   ./build/bin/flchus_miner <flchuspm|flchus_x> <file> <mins> <minu> <maxc> [pct]
 *   env: DM_FLCHUS_OUT, FLCHUS_DPS / FLCHUS_WPS / FLCHUS_FLB (flchuspm),
 *        FLCHUSX_COSTW / FLCHUSX_FIX / FLCHUSX_SEL / FLCHUSX_DESC (flchus_x)
 * Prints "time_s=<wall seconds of the algorithm>" (reading the file included).
 */
#include "algorithms/flchuspm.h"
#include "algorithms/flchus_x.h"
#include "core/dm_algorithm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* the algorithms report their result counts through this hook of the dm benchmark module */
void dm_bench_record_results(size_t num_itemsets, size_t total_items) { (void)num_itemsets; (void)total_items; }

static int flag(const char *name, int def) { const char *v = getenv(name); return v ? atoi(v) : def; }

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <flchuspm|flchus_x> <file> <mins> <minu> <maxc> [pct]\n", argv[0]);
        return 2;
    }
    DM_Algorithm *algo = dm_get_algorithm(argv[1]);
    if (!algo) { fprintf(stderr, "unknown algorithm %s\n", argv[1]); return 2; }
    int pct = argc >= 7 && strcmp(argv[6], "pct") == 0;
    DM_FLCHUSPM_Params fp = { argv[2], atof(argv[3]), atof(argv[4]), atof(argv[5]), pct,
                              flag("FLCHUS_DPS", 1), flag("FLCHUS_WPS", 1) };
    DM_FLCHUS_X_Params xp = { argv[2], atof(argv[3]), atof(argv[4]), atof(argv[5]), pct,
                              flag("FLCHUSX_COSTW", 1), flag("FLCHUSX_FIX", 1), flag("FLCHUSX_SEL", 1),
                              flag("FLCHUSX_DESC", 1) };
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    DM_Status st = algo->run(NULL, strcmp(argv[1], "flchus_x") == 0 ? (void *)&xp : (void *)&fp);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("time_s=%.4f\n", (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9);
    return st == DM_SUCCESS ? 0 : 1;
}
