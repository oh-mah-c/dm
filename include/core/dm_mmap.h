#ifndef DM_MMAP_H
#define DM_MMAP_H

#include <stddef.h>

typedef struct {
    const unsigned char *data;
    size_t size;
    int fd;
    int mapped;
} DM_MMap;

int dm_mmap_open(const char *path, DM_MMap *map);
void dm_mmap_close(DM_MMap *map);

#endif
