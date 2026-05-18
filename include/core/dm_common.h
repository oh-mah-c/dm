#ifndef DM_COMMON_H
#define DM_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#if !defined(_WIN32)
#define _strdup strdup

typedef int (*DM_QsortS_Comparator)(void *, const void *, const void *);

static DM_QsortS_Comparator dm_qsort_s_comparator = NULL;
static void *dm_qsort_s_context = NULL;

static int dm_qsort_s_bridge(const void *a, const void *b) {
    return dm_qsort_s_comparator(dm_qsort_s_context, a, b);
}

static inline int dm_qsort_s(void *base, size_t nmemb, size_t size, DM_QsortS_Comparator comparator, void *context) {
    dm_qsort_s_comparator = comparator;
    dm_qsort_s_context = context;
    qsort(base, nmemb, size, dm_qsort_s_bridge);
    dm_qsort_s_comparator = NULL;
    dm_qsort_s_context = NULL;
    return 0;
}

#define qsort_s dm_qsort_s
#endif

typedef enum {
    DM_SUCCESS = 0,
    DM_ERROR_GENERIC = -1,
    DM_ERROR_IO = -2,
    DM_ERROR_MEMORY = -3,
    DM_ERROR_INVALID_PARAM = -4,
    DM_ERROR_NOT_FOUND = -5,
    DM_ERROR_INCOMPATIBLE = -6
} DM_Status;

#endif // DM_COMMON_H
