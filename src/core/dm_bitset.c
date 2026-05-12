#include "core/dm_bitset.h"
#include <stdlib.h>
#include <string.h>

DM_BitSet* dm_bitset_create(size_t size) {
    DM_BitSet *bs = malloc(sizeof(DM_BitSet));
    bs->size = size;
    bs->count = (size + 63) / 64;
    bs->bits = calloc(bs->count, sizeof(uint64_t));
    return bs;
}

DM_BitSet* dm_bitset_copy(const DM_BitSet *src) {
    if (!src) return NULL;
    DM_BitSet *dest = malloc(sizeof(DM_BitSet));
    dest->size = src->size;
    dest->count = src->count;
    dest->bits = malloc(dest->count * sizeof(uint64_t));
    memcpy(dest->bits, src->bits, dest->count * sizeof(uint64_t));
    return dest;
}

void dm_bitset_copy_to(DM_BitSet *dest, const DM_BitSet *src) {
    if (!dest || !src) return;
    size_t n = dest->count < src->count ? dest->count : src->count;
    memcpy(dest->bits, src->bits, n * sizeof(uint64_t));
}

void dm_bitset_free(DM_BitSet *bs) {
    if (bs) {
        free(bs->bits);
        free(bs);
    }
}

void dm_bitset_set(DM_BitSet *bs, size_t pos) {
    if (pos < bs->size) {
        bs->bits[pos / 64] |= (1ULL << (pos % 64));
    }
}

void dm_bitset_clear(DM_BitSet *bs, size_t pos) {
    if (pos < bs->size) {
        bs->bits[pos / 64] &= ~(1ULL << (pos % 64));
    }
}

bool dm_bitset_get(DM_BitSet *bs, size_t pos) {
    if (pos < bs->size) {
        return (bs->bits[pos / 64] & (1ULL << (pos % 64))) != 0;
    }
    return false;
}

void dm_bitset_and(DM_BitSet *dest, const DM_BitSet *src) {
    size_t n = dest->count < src->count ? dest->count : src->count;
    for (size_t i = 0; i < n; i++) {
        dest->bits[i] &= src->bits[i];
    }
}

void dm_bitset_or(DM_BitSet *dest, const DM_BitSet *src) {
    size_t n = dest->count < src->count ? dest->count : src->count;
    for (size_t i = 0; i < n; i++) {
        dest->bits[i] |= src->bits[i];
    }
}

void dm_bitset_not(DM_BitSet *bs) {
    for (size_t i = 0; i < bs->count; i++) {
        bs->bits[i] = ~bs->bits[i];
    }
}

void dm_bitset_set_all(DM_BitSet *bs) {
    memset(bs->bits, 0xFF, bs->count * sizeof(uint64_t));
}

size_t dm_bitset_count(DM_BitSet *bs) {
    size_t count = 0;
    for (size_t i = 0; i < bs->count; i++) {
        uint64_t v = bs->bits[i];
        // Popcount
        v = v - ((v >> 1) & 0x5555555555555555ULL);
        v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);
        count += (((v + (v >> 4)) & 0x0F0F0F0F0F0F0F0FULL) * 0x0101010101010101ULL) >> 56;
    }
    return count;
}
