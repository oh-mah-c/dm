#include "tokenizer/dm_tokenizer_block.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t *tokens;
    size_t token_count;
    size_t token_capacity;
    
    uint32_t *lengths;
    size_t lengths_count;
    size_t lengths_capacity;
} Accumulator;

static void accumulator_callback(const uint32_t *tokens, size_t count, void *user_data) {
    Accumulator *acc = (Accumulator*)user_data;
    if (count == 0) return;
    
    // Append tokens
    if (acc->token_count + count > acc->token_capacity) {
        while (acc->token_count + count > acc->token_capacity) {
            acc->token_capacity = acc->token_capacity == 0 ? 1024 : acc->token_capacity * 2;
        }
        acc->tokens = realloc(acc->tokens, acc->token_capacity * sizeof(uint32_t));
    }
    memcpy(acc->tokens + acc->token_count, tokens, count * sizeof(uint32_t));
    acc->token_count += count;
    
    // Append length (for Ragged sequences)
    if (acc->lengths_count + 1 > acc->lengths_capacity) {
        acc->lengths_capacity = acc->lengths_capacity == 0 ? 128 : acc->lengths_capacity * 2;
        acc->lengths = realloc(acc->lengths, acc->lengths_capacity * sizeof(uint32_t));
    }
    acc->lengths[acc->lengths_count++] = (uint32_t)count;
}

int dm_tokenize_to_block(
    Tokenizer *tok,
    const unsigned char *text,
    size_t length,
    TransactionMode mode,
    uint32_t window,
    uint32_t stride,
    DM_Block *out_block)
{
    if (!tok || !text || !out_block) return -1;
    
    Accumulator acc = {0};
    tok->tokenize_buffer(tok, text, length, mode, window, stride, accumulator_callback, &acc);
    
    // Determine number of elements
    int64_t shape[1] = {(int64_t)acc.token_count};
    
    // Note: If no tokens were parsed, token_count will be 0, creating an empty block.
    int rc = dm_block_create(out_block, DM_KIND_TOKEN_STREAM, DM_DTYPE_I32, DM_LAYOUT_PACKED_TOKENS, DM_BACKEND_CPU, 1, shape);
    if (rc != 0) {
        if (acc.tokens) free(acc.tokens);
        if (acc.lengths) free(acc.lengths);
        return rc;
    }
    
    out_block->role = DM_ROLE_TOKEN_IDS;
    
    if (acc.token_count > 0 && acc.tokens) {
        memcpy(out_block->data, acc.tokens, acc.token_count * sizeof(uint32_t));
        free(acc.tokens);
    }
    
    if (mode == MODE_DOCUMENT) {
        // Flat document mode: no ragged structure needed.
        if (acc.lengths) free(acc.lengths);
        out_block->aux = NULL;
    } else {
        // Save the lengths array pointer directly to aux to describe the jagged structure.
        // It must be freed manually by the user or within a custom deallocator later.
        out_block->aux = acc.lengths;
        
        // Also save the number of distinct chunks (sequences) as metadata
        char chunk_str[32];
        snprintf(chunk_str, sizeof(chunk_str), "%zu", acc.lengths_count);
        dm_block_set_meta(out_block, "ragged_chunk_count", chunk_str);
    }
    
    return 0;
}
