#ifndef DM_TOKENIZER_BLOCK_H
#define DM_TOKENIZER_BLOCK_H

#include "tokenizer.h"
#include "core/dm_block.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Fully tokenizes an input buffer into a DM_Block.
 * 
 * Creates a block with:
 *  - kind = DM_KIND_TOKEN_STREAM
 *  - role = DM_ROLE_TOKEN_IDS
 *  - dtype = DM_DTYPE_I32
 *  - layout = DM_LAYOUT_PACKED_TOKENS
 *  - data = Flat contiguous array of `uint32_t` token IDs
 *  - aux = Used to store structural boundaries (lengths array for ragged sequences)
 */
int dm_tokenize_to_block(
    Tokenizer *tok,
    const unsigned char *text,
    size_t length,
    TransactionMode mode,
    uint32_t window,
    uint32_t stride,
    DM_Block *out_block
);

#ifdef __cplusplus
}
#endif

#endif // DM_TOKENIZER_BLOCK_H
