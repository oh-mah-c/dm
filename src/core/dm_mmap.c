#include "core/dm_mmap.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int dm_mmap_open(const char *path, DM_MMap *map) {
    if (!path || !map) return -1;
    memset(map, 0, sizeof(*map));
    map->fd = -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        close(fd);
        return -1;
    }
    if (st.st_size == 0) {
        map->fd = fd;
        return 0;
    }
    void *ptr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        return -1;
    }
    map->data = (const unsigned char *)ptr;
    map->size = (size_t)st.st_size;
    map->fd = fd;
    map->mapped = 1;
    return 0;
}

void dm_mmap_close(DM_MMap *map) {
    if (!map) return;
    if (map->mapped && map->data && map->size) munmap((void *)map->data, map->size);
    if (map->fd >= 0) close(map->fd);
    memset(map, 0, sizeof(*map));
    map->fd = -1;
}
