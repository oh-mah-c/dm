/* Minimal unistd stub for MSVC/Windows */
#ifndef DM_UNISTD_STUB_H
#define DM_UNISTD_STUB_H

#ifdef _WIN32
#include <io.h>
#include <process.h>
#include <direct.h>
#include <sys/types.h>
#include <basetsd.h>
typedef SSIZE_T ssize_t;
#define STDIN_FILENO  0
#define STDOUT_FILENO 1
void __stdcall Sleep(unsigned long dwMilliseconds);
static inline int usleep(unsigned int us) { Sleep((us + 999) / 1000); return 0; }
#include <stdio.h>
#include <stdlib.h>
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
            size_t new_n = *n * 2;
            char *new_ptr = (char *)realloc(*lineptr, new_n);
            if (!new_ptr) return -1;
            *lineptr = new_ptr;
            *n = new_n;
        }
        (*lineptr)[pos++] = (char)c;
        if (c == '\n') break;
    }
    if (pos == 0 && c == EOF) return -1;
    (*lineptr)[pos] = '\0';
    return (ssize_t)pos;
}
#define getline dm_getline
#else
#if defined(__GNUC__) || defined(__clang__)
#include_next <unistd.h>
#else
#include </usr/include/unistd.h>
#endif
#endif /* _WIN32 */

#endif /* DM_UNISTD_STUB_H */
