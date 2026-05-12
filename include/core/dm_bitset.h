#ifndef DM_BITSET_H
#define DM_BITSET_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

typedef struct {
    uint64_t *bits;
    size_t size;     // number of bits
    size_t count;    // number of 64-bit words
} DM_BitSet;

DM_BitSet* dm_bitset_create(size_t size);
DM_BitSet* dm_bitset_copy(const DM_BitSet *src);
void dm_bitset_copy_to(DM_BitSet *dest, const DM_BitSet *src);
void dm_bitset_free(DM_BitSet *bs);
void dm_bitset_set(DM_BitSet *bs, size_t pos);
void dm_bitset_clear(DM_BitSet *bs, size_t pos);
bool dm_bitset_get(DM_BitSet *bs, size_t pos);
void dm_bitset_and(DM_BitSet *dest, const DM_BitSet *src);
void dm_bitset_or(DM_BitSet *dest, const DM_BitSet *src);
void dm_bitset_not(DM_BitSet *bs);
void dm_bitset_set_all(DM_BitSet *bs);
size_t dm_bitset_count(DM_BitSet *bs);

#endif
