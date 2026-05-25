#ifndef DM_PORTABILITY_H
#define DM_PORTABILITY_H

/* Ensure POSIX/GNU extensions are available (sysconf, _SC_NPROCESSORS_ONLN,
   getline, rand_r, etc.) on glibc-based systems. */
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>

#ifndef _SSIZE_T_DEFINED
typedef intptr_t ssize_t;
#define _SSIZE_T_DEFINED
#endif

static inline int dm_mkdir(const char *path, int mode) {
    (void)mode;
    return _mkdir(path);
}

static inline unsigned int dm_rand_r(unsigned int *seed) {
    *seed = (*seed * 1103515245u) + 12345u;
    return (*seed / 65536u) % 32768u;
}

static inline long dm_cpu_count(void) {
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwNumberOfProcessors ? (long)info.dwNumberOfProcessors : 1;
}

static inline ssize_t dm_getline(char **lineptr, size_t *n, FILE *stream) {
    if (!lineptr || !n || !stream) return -1;
    if (!*lineptr || *n == 0) {
        *n = 128;
        *lineptr = (char *)malloc(*n);
        if (!*lineptr) return -1;
    }

    size_t pos = 0;
    int c;
    while ((c = fgetc(stream)) != EOF) {
        if (pos + 1 >= *n) {
            size_t next = (*n) * 2;
            char *tmp = (char *)realloc(*lineptr, next);
            if (!tmp) return -1;
            *lineptr = tmp;
            *n = next;
        }
        (*lineptr)[pos++] = (char)c;
        if (c == '\n') break;
    }

    if (pos == 0 && c == EOF) return -1;
    (*lineptr)[pos] = '\0';
    return (ssize_t)pos;
}

#else
#include <sys/stat.h>
#include <unistd.h>

static inline int dm_mkdir(const char *path, int mode) {
    return mkdir(path, (mode_t)mode);
}

static inline long dm_cpu_count(void) {
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 ? count : 1;
}

#define dm_rand_r rand_r
#define dm_getline getline

#endif

#endif
