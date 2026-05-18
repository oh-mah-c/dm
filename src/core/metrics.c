#include "core/experiment.h"
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

double dm_file_size_mb(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0.0;
    return (double)st.st_size / (1024.0 * 1024.0);
}

double dm_directory_size_mb(const char *path) {
    DIR *dir = opendir(path);
    if (!dir) return 0.0;
    double total = 0.0;
    struct dirent *ent;
    while ((ent = readdir(dir))) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        char child[1024];
        snprintf(child, sizeof(child), "%s/%s", path, ent->d_name);
        struct stat st;
        if (stat(child, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) total += dm_directory_size_mb(child);
        else total += (double)st.st_size / (1024.0 * 1024.0);
    }
    closedir(dir);
    return total;
}
