#include "core/dm_block.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/dm_tokenizer_block.h"
#include <stdio.h>
#include <string.h>

int main() {
    printf("Creating Tokenizer...\n");
    Tokenizer *tok = rh_arena_tokenizer_create(1024);
    if (!tok) {
        printf("Failed to create tokenizer\n");
        return 1;
    }

    const char *text = "hello world. this is a test. hello again.";
    size_t len = strlen(text);

    // 1. Test Flat Document Mode
    printf("\n=== Document Mode (Flat array) ===\n");
    DM_Block block_doc;
    int rc = dm_tokenize_to_block(tok, (const unsigned char*)text, len, MODE_DOCUMENT, 0, 0, &block_doc);
    if (rc == 0) {
        printf("Block created! count=%zu, dtype=%d, layout=%d\n", block_doc.count, block_doc.dtype, block_doc.layout);
        uint32_t *tokens = (uint32_t*)block_doc.data;
        printf("Tokens: ");
        for (size_t i = 0; i < block_doc.count; i++) {
            printf("%u ", tokens[i]);
        }
        printf("\nAux pointer: %p (Expected NULL for flat document)\n", block_doc.aux);
        dm_block_free(&block_doc);
    } else {
        printf("Failed to tokenize to block.\n");
    }

    // 2. Test Sentence Mode (Ragged array)
    printf("\n=== Sentence Mode (Ragged layout) ===\n");
    DM_Block block_sent;
    rc = dm_tokenize_to_block(tok, (const unsigned char*)text, len, MODE_SENTENCE, 0, 0, &block_sent);
    if (rc == 0) {
        printf("Block created! count=%zu\n", block_sent.count);
        
        char chunk_str[64] = {0};
        for (int i = 0; i < block_sent.meta_count; i++) {
            if (strcmp(block_sent.meta[i].key, "ragged_chunk_count") == 0) {
                strcpy(chunk_str, block_sent.meta[i].value);
            }
        }
        printf("Ragged Chunk Count: %s\n", chunk_str);
        
        uint32_t *tokens = (uint32_t*)block_sent.data;
        uint32_t *lengths = (uint32_t*)block_sent.aux;
        
        size_t offset = 0;
        int num_chunks = atoi(chunk_str);
        for (int c = 0; c < num_chunks; c++) {
            printf("Sentence %d (len %u): ", c, lengths[c]);
            for (uint32_t i = 0; i < lengths[c]; i++) {
                printf("%u ", tokens[offset++]);
            }
            printf("\n");
        }
        
        // Custom cleanup since aux is manually managed
        if (block_sent.aux) free(block_sent.aux);
        dm_block_free(&block_sent);
    }

    tok->free(tok);
    return 0;
}
