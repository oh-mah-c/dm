#ifndef TOKENIZER_FRAMEWORK_H
#define TOKENIZER_FRAMEWORK_H

#include <stdint.h>
#include <stddef.h>

typedef enum {
    MODE_DOCUMENT,   /* Line-by-line transactions */
    MODE_SENTENCE,   /* Split transactions at sentence boundaries (. ? !) */
    MODE_SLIDING     /* Sliding window transactions */
} TransactionMode;

/* Generic Polymorphic Tokenizer Interface matching Section 3 */
typedef struct Tokenizer Tokenizer;

struct Tokenizer {
    const char *name;
    void *impl;
    
    /* Base address for memory mapping, used in borrowed-view mode */
    const unsigned char *base_address;
    
    /* Virtual functions mapping the getOrCreate and scan loop semantics */
    uint32_t (*get_or_create)(Tokenizer *self, const char *token_bytes, uint32_t len, uint64_t hash_val);
    void (*tokenize_buffer)(Tokenizer *self, 
                            const unsigned char *S, size_t B, 
                            TransactionMode mode, 
                            uint32_t window_size, uint32_t stride,
                            void (*emit_callback)(const uint32_t *tokens, size_t count, void *user_data), 
                            void *user_data);
    void (*free)(Tokenizer *self);
    
    /* Print detailed stats for standard benchmark comparisons */
    void (*print_stats)(Tokenizer *self, const char *input_path, size_t file_size, double elapsed_sec, size_t tx_count, long peak_rss);
};

/* --- FARO-Tokenizer Specific Definitions matching Section 4 --- */

#define FNV_OFFSET_BASIS 14695981039346656037ULL
#define FNV_PRIME        1099511628211ULL

typedef struct {
    uint32_t metadata;     /* fingerprint (bits 0-7), DIB (bits 8-23), flags (bits 24-31) */
    uint32_t arena_offset;  /* Offset into the lexeme arena */
    uint32_t token_len;     /* Length of the canonical token */
    uint32_t token_id;      /* Assigned ID */
} FaroSlot;

typedef struct {
    FaroSlot *slots;
    uint32_t capacity;
    char *arena;
    uint32_t arena_size;
    uint32_t arena_capacity;
    uint32_t unique_tokens;
} FaroTokenizerImpl;

/* Factory functions for Ablation Study matching Section 11.2 */
Tokenizer *lp_raw_tokenizer_create(uint32_t initial_capacity);
Tokenizer *lp_fp_tokenizer_create(uint32_t initial_capacity);
Tokenizer *rh_fp_tokenizer_create(uint32_t initial_capacity);
Tokenizer *rh_arena_tokenizer_create(uint32_t initial_capacity);  /* Default FARO */
Tokenizer *rh_borrow_tokenizer_create(uint32_t initial_capacity);

/* Common utilities */
uint64_t faro_hash(const char *bytes, uint32_t len);
size_t faro_sort_uniq(uint32_t *tokens, size_t count);

#endif /* TOKENIZER_FRAMEWORK_H */
