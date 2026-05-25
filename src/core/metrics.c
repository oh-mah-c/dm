#include "core/experiment.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

double dm_file_size_mb(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0.0;
    return (double)st.st_size / (1024.0 * 1024.0);
}

double dm_directory_size_mb(const char *path) {
#ifdef _WIN32
    char pattern[1024];
    WIN32_FIND_DATAA data;
    HANDLE h;
    double total = 0.0;
    snprintf(pattern, sizeof(pattern), "%s\\*", path);
    h = FindFirstFileA(pattern, &data);
    if (h == INVALID_HANDLE_VALUE) return 0.0;
    do {
        char child[1024];
        if (strcmp(data.cFileName, ".") == 0 || strcmp(data.cFileName, "..") == 0) continue;
        snprintf(child, sizeof(child), "%s\\%s", path, data.cFileName);
        if (data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            total += dm_directory_size_mb(child);
        } else {
            LARGE_INTEGER size;
            size.HighPart = data.nFileSizeHigh;
            size.LowPart = data.nFileSizeLow;
            total += (double)size.QuadPart / (1024.0 * 1024.0);
        }
    } while (FindNextFileA(h, &data));
    FindClose(h);
    return total;
#else
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
#endif
}
