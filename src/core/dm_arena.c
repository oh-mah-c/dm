#include "core/dm_arena.h"

#include <stdlib.h>
#include <string.h>

int dm_arena_init(DM_Arena *arena, size_t capacity) {
    if (!arena || capacity == 0) return -1;
    arena->base = (unsigned char *)malloc(capacity);
    if (!arena->base) return -1;
    arena->capacity = capacity;
    arena->offset = 0;
    arena->owns_memory = 1;
    return 0;
}

void dm_arena_wrap(DM_Arena *arena, void *memory, size_t capacity) {
    if (!arena) return;
    arena->base = (unsigned char *)memory;
    arena->capacity = capacity;
    arena->offset = 0;
    arena->owns_memory = 0;
}

void dm_arena_reset(DM_Arena *arena) {
    if (arena) arena->offset = 0;
}

void dm_arena_free(DM_Arena *arena) {
    if (!arena) return;
    if (arena->owns_memory) free(arena->base);
    arena->base = NULL;
    arena->capacity = 0;
    arena->offset = 0;
    arena->owns_memory = 0;
}

void *dm_arena_alloc(DM_Arena *arena, size_t size, size_t alignment) {
    if (!arena || !arena->base || size == 0) return NULL;
    if (alignment == 0) alignment = sizeof(void *);
    size_t mask = alignment - 1;
    size_t aligned = (arena->offset + mask) & ~mask;
    if (aligned > arena->capacity || size > arena->capacity - aligned) return NULL;
    void *ptr = arena->base + aligned;
    arena->offset = aligned + size;
    memset(ptr, 0, size);
    return ptr;
}

char *dm_arena_strndup(DM_Arena *arena, const char *src, size_t len) {
    char *out = (char *)dm_arena_alloc(arena, len + 1, 1);
    if (!out) return NULL;
    memcpy(out, src, len);
    out[len] = '\0';
    return out;
}
