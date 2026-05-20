#ifndef DM_ARENA_H
#define DM_ARENA_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    unsigned char *base;
    size_t capacity;
    size_t offset;
    int owns_memory;
} DM_Arena;

int dm_arena_init(DM_Arena *arena, size_t capacity);
void dm_arena_wrap(DM_Arena *arena, void *memory, size_t capacity);
void dm_arena_reset(DM_Arena *arena);
void dm_arena_free(DM_Arena *arena);
void *dm_arena_alloc(DM_Arena *arena, size_t size, size_t alignment);
char *dm_arena_strndup(DM_Arena *arena, const char *src, size_t len);

#define DM_ARENA_NEW(arena, type, count) ((type *)dm_arena_alloc((arena), sizeof(type) * (count), sizeof(void *)))

#endif
