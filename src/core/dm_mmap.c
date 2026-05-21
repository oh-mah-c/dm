#include "core/dm_mmap.h"

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>

typedef struct { HANDLE file; HANDLE mapping; } WinMapHandles;

int dm_mmap_open(const char *path, DM_MMap *map) {
    if (!path || !map) return -1;
    memset(map, 0, sizeof(*map));
    map->fd = -1;
    HANDLE hf = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hf == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(hf, &sz) || sz.QuadPart < 0) { CloseHandle(hf); return -1; }
    if (sz.QuadPart == 0) { CloseHandle(hf); return 0; }
    HANDLE hm = CreateFileMappingA(hf, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hm) { CloseHandle(hf); return -1; }
    void *ptr = MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0);
    if (!ptr) { CloseHandle(hm); CloseHandle(hf); return -1; }
    WinMapHandles *h = (WinMapHandles *)malloc(sizeof(WinMapHandles));
    if (!h) { UnmapViewOfFile(ptr); CloseHandle(hm); CloseHandle(hf); return -1; }
    h->file = hf; h->mapping = hm;
    map->data = (const unsigned char *)ptr;
    map->size = (size_t)sz.QuadPart;
    map->fd = (int)(intptr_t)h;
    map->mapped = 1;
    return 0;
}

void dm_mmap_close(DM_MMap *map) {
    if (!map) return;
    if (map->mapped && map->data) UnmapViewOfFile((void *)map->data);
    if (map->fd != -1) {
        WinMapHandles *h = (WinMapHandles *)(intptr_t)map->fd;
        CloseHandle(h->mapping);
        CloseHandle(h->file);
        free(h);
    }
    memset(map, 0, sizeof(*map));
    map->fd = -1;
}

#else
#include <fcntl.h>
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
#endif
