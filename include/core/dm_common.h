#ifndef DM_COMMON_H
#define DM_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

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
