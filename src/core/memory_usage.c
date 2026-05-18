#include "core/experiment.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

double get_peak_ram_mb(void) {
    FILE *fp = fopen("/proc/self/status", "r");
    if (!fp) return 0.0;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            char *p = line + 6;
            while (*p == ' ' || *p == '\t') p++;
            double value = strtod(p, NULL);
            fclose(fp);
            return value / 1024.0;
        }
    }
    fclose(fp);
    return 0.0;
}
