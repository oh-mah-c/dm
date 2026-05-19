#include "../../include/tokenizer/tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* MurmurHash3 fmix64 matching Section 5.1 */
static inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

/* UTF-8 parser matching Section 9.3 */
static int parse_utf8_char(const unsigned char *s, size_t remaining, size_t *char_len) {
    if (remaining == 0) return 0;
    unsigned char c = s[0];
    if (c < 0x80) {
        *char_len = 1;
        return 1;
    }
    if ((c & 0xE0) == 0xC0) {
        if (remaining < 2) return 0;
        if (c < 0xC2) return 0;
        if ((s[1] & 0xC0) != 0x80) return 0;
        *char_len = 2;
        return 1;
    }
    if ((c & 0xF0) == 0xE0) {
        if (remaining < 3) return 0;
        unsigned char c1 = s[1];
        unsigned char c2 = s[2];
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80) return 0;
        if (c == 0xE0 && c1 < 0xA0) return 0;
        if (c == 0xED && c1 > 0x9F) return 0;
        *char_len = 3;
        return 1;
    }
    if ((c & 0xF8) == 0xF0) {
        if (remaining < 4) return 0;
        unsigned char c1 = s[1];
        unsigned char c2 = s[2];
        unsigned char c3 = s[3];
        if ((c1 & 0xC0) != 0x80 || (c2 & 0xC0) != 0x80 || (c3 & 0xC0) != 0x80) return 0;
        if (c == 0xF0 && c1 < 0x90) return 0;
        if (c == 0xF4 && c1 > 0x8F) return 0;
        if (c > 0xF4) return 0;
        *char_len = 4;
        return 1;
    }
    return 0;
}

uint64_t faro_hash(const char *bytes, uint32_t len) {
    uint64_t h = FNV_OFFSET_BASIS;
    for (uint32_t i = 0; i < len; i++) {
        h ^= (unsigned char)bytes[i];
        h *= FNV_PRIME;
    }
    h ^= len;
    return fmix64(h);
}

static int int_compare(const void *a, const void *b) {
    uint32_t val_a = *(const uint32_t *)a;
    uint32_t val_b = *(const uint32_t *)b;
    return (val_a > val_b) - (val_a < val_b);
}

size_t faro_sort_uniq(uint32_t *tokens, size_t count) {
    if (count <= 1) return count;
    qsort(tokens, count, sizeof(uint32_t), int_compare);
    size_t write_idx = 1;
    for (size_t i = 1; i < count; i++) {
        if (tokens[i] != tokens[write_idx - 1]) {
            tokens[write_idx] = tokens[i];
            write_idx++;
        }
    }
    return write_idx;
}

static void faro_rehash(FaroTokenizerImpl *tok) {
    uint32_t old_capacity = tok->capacity;
    FaroSlot *old_slots = tok->slots;
    
    uint32_t new_capacity = old_capacity * 2;
    FaroSlot *new_slots = calloc(new_capacity, sizeof(FaroSlot));
    if (!new_slots) return;
    
    tok->capacity = new_capacity;
    tok->slots = new_slots;
    
    for (uint32_t i = 0; i < old_capacity; i++) {
        FaroSlot *old_slot = &old_slots[i];
        uint8_t old_fp = old_slot->metadata & 0xFF;
        if (old_fp == 0) continue;
        
        const char *bytes = tok->arena + old_slot->arena_offset;
        uint64_t hash_val = faro_hash(bytes, old_slot->token_len);
        
        uint32_t mask = new_capacity - 1;
        uint32_t s = hash_val & mask;
        uint32_t d = 0;
        
        uint8_t fp = (hash_val >> 56) == 0 ? 1 : (hash_val >> 56);
        
        uint8_t cand_fp = fp;
        uint32_t cand_dib = d;
        uint32_t cand_offset = old_slot->arena_offset;
        uint32_t cand_len = old_slot->token_len;
        uint32_t cand_id = old_slot->token_id;
        
        while (1) {
            FaroSlot *slot = &new_slots[s];
            uint8_t slot_fp = slot->metadata & 0xFF;
            uint32_t slot_dib = (slot->metadata >> 8) & 0xFFFF;
            
            if (slot_fp == 0) {
                slot->metadata = cand_fp | (cand_dib << 8);
                slot->arena_offset = cand_offset;
                slot->token_len = cand_len;
                slot->token_id = cand_id;
                break;
            }
            
            if (slot_dib < cand_dib) {
                uint8_t temp_fp = slot_fp;
                uint32_t temp_dib = slot_dib;
                uint32_t temp_offset = slot->arena_offset;
                uint32_t temp_len = slot->token_len;
                uint32_t temp_id = slot->token_id;
                
                slot->metadata = cand_fp | (cand_dib << 8);
                slot->arena_offset = cand_offset;
                slot->token_len = cand_len;
                slot->token_id = cand_id;
                
                cand_fp = temp_fp;
                cand_dib = temp_dib;
                cand_offset = temp_offset;
                cand_len = temp_len;
                cand_id = temp_id;
            }
            
            s = (s + 1) & mask;
            cand_dib++;
        }
    }
    
    free(old_slots);
}

static uint32_t faro_get_or_create_impl(FaroTokenizerImpl *tok, const char *token_bytes, uint32_t len, uint64_t hash_val) {
    uint32_t capacity = tok->capacity;
    uint32_t mask = capacity - 1;
    uint32_t s = hash_val & mask;
    uint32_t d = 0;
    
    uint8_t fingerprint = (hash_val >> 56) == 0 ? 1 : (hash_val >> 56);
    
    uint8_t cand_fp = fingerprint;
    uint32_t cand_dib = d;
    uint32_t cand_len = len;
    uint32_t cand_id = 0;
    uint32_t cand_offset = 0;
    int is_materialized = 0;
    
    while (1) {
        FaroSlot *slot = &tok->slots[s];
        uint8_t slot_fp = slot->metadata & 0xFF;
        uint32_t slot_dib = (slot->metadata >> 8) & 0xFFFF;
        
        if (slot_fp != 0 && slot_dib < cand_dib) {
            goto do_insert_swap;
        }
        
        if (slot_fp == 0) {
            if (!is_materialized) {
                cand_offset = tok->arena_size;
                if (tok->arena_size + cand_len > tok->arena_capacity) {
                    tok->arena_capacity = tok->arena_capacity * 2 + cand_len;
                    tok->arena = realloc(tok->arena, tok->arena_capacity);
                }
                memcpy(tok->arena + cand_offset, token_bytes, cand_len);
                tok->arena_size += cand_len;
                
                tok->unique_tokens++;
                cand_id = tok->unique_tokens;
                is_materialized = 1;
            }
            
            slot->metadata = cand_fp | (cand_dib << 8);
            slot->arena_offset = cand_offset;
            slot->token_len = cand_len;
            slot->token_id = cand_id;
            
            if ((double)tok->unique_tokens / capacity > 0.75) {
                faro_rehash(tok);
            }
            
            return cand_id;
        }
        
        if (slot_fp == cand_fp && slot->token_len == cand_len && 
            memcmp(tok->arena + slot->arena_offset, token_bytes, cand_len) == 0) {
            return slot->token_id;
        }
        
        goto next_probe;
        
    do_insert_swap:
        if (!is_materialized) {
            cand_offset = tok->arena_size;
            if (tok->arena_size + cand_len > tok->arena_capacity) {
                tok->arena_capacity = tok->arena_capacity * 2 + cand_len;
                tok->arena = realloc(tok->arena, tok->arena_capacity);
            }
            memcpy(tok->arena + cand_offset, token_bytes, cand_len);
            tok->arena_size += cand_len;
            
            tok->unique_tokens++;
            cand_id = tok->unique_tokens;
            is_materialized = 1;
        }
        
        uint8_t temp_fp = slot_fp;
        uint32_t temp_dib = slot_dib;
        uint32_t temp_offset = slot->arena_offset;
        uint32_t temp_len = slot->token_len;
        uint32_t temp_id = slot->token_id;
        
        slot->metadata = cand_fp | (cand_dib << 8);
        slot->arena_offset = cand_offset;
        slot->token_len = cand_len;
        slot->token_id = cand_id;
        
        cand_fp = temp_fp;
        cand_dib = temp_dib;
        cand_offset = temp_offset;
        cand_len = temp_len;
        cand_id = temp_id;
        
        token_bytes = tok->arena + cand_offset;
        
    next_probe:
        s = (s + 1) & mask;
        cand_dib++;
    }
}

static void faro_tokenize_buffer_impl(FaroTokenizerImpl *tok, 
                                      const unsigned char *S, size_t B, 
                                      TransactionMode mode, 
                                      uint32_t window_size, uint32_t stride,
                                      void (*emit_callback)(const uint32_t *tokens, size_t count, void *user_data), 
                                      void *user_data) {
    char token_buf[4096];
    size_t token_len = 0;
    
    uint32_t *txn_tokens = NULL;
    size_t txn_count = 0;
    size_t txn_capacity = 0;
    
    uint32_t *global_tokens = NULL;
    size_t global_count = 0;
    size_t global_capacity = 0;
    
    #define FINALIZE_TOKEN() do { \
        if (token_len > 0) { \
            uint64_t hash_val = faro_hash(token_buf, token_len); \
            uint32_t tid = faro_get_or_create_impl(tok, token_buf, token_len, hash_val); \
            if (mode == MODE_SLIDING) { \
                if (global_count >= global_capacity) { \
                    global_capacity = global_capacity * 2 + 128; \
                    global_tokens = realloc(global_tokens, global_capacity * sizeof(uint32_t)); \
                } \
                global_tokens[global_count++] = tid; \
            } else { \
                if (txn_count >= txn_capacity) { \
                    txn_capacity = txn_capacity * 2 + 32; \
                    txn_tokens = realloc(txn_tokens, txn_capacity * sizeof(uint32_t)); \
                } \
                txn_tokens[txn_count++] = tid; \
            } \
            token_len = 0; \
        } \
    } while(0)

    #define FINALIZE_TRANSACTION() do { \
        if (txn_count > 0) { \
            size_t final_count = faro_sort_uniq(txn_tokens, txn_count); \
            emit_callback(txn_tokens, final_count, user_data); \
            txn_count = 0; \
        } \
    } while(0)

    size_t i = 0;
    while (i < B) {
        unsigned char c = S[i];
        
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
            unsigned char folded = c;
            if (c >= 'A' && c <= 'Z') {
                folded = c + 32;
            }
            if (token_len < sizeof(token_buf)) {
                token_buf[token_len++] = folded;
            }
            i++;
        } else if (c >= 0x80) {
            size_t char_len = 0;
            if (parse_utf8_char(S + i, B - i, &char_len)) {
                if (token_len + char_len <= sizeof(token_buf)) {
                    for (size_t k = 0; k < char_len; k++) {
                        token_buf[token_len++] = S[i + k];
                    }
                }
                i += char_len;
            } else {
                FINALIZE_TOKEN();
                i++;
            }
        } else if (c == '-' || c == '_' || c == '\'') {
            int left_token = (token_len > 0);
            int right_token = 0;
            if (i + 1 < B) {
                unsigned char next_c = S[i + 1];
                if ((next_c >= 'a' && next_c <= 'z') || (next_c >= 'A' && next_c <= 'Z') || (next_c >= '0' && next_c <= '9') || next_c >= 0x80) {
                    if (next_c >= 0x80) {
                        size_t dummy;
                        if (parse_utf8_char(S + i + 1, B - (i + 1), &dummy)) {
                            right_token = 1;
                        }
                    } else {
                        right_token = 1;
                    }
                }
            }
            
            if (left_token && right_token) {
                if (token_len < sizeof(token_buf)) {
                    token_buf[token_len++] = c;
                }
                i++;
            } else {
                FINALIZE_TOKEN();
                i++;
            }
        } else {
            FINALIZE_TOKEN();
            if (mode == MODE_DOCUMENT && c == '\n') {
                FINALIZE_TRANSACTION();
            } else if (mode == MODE_SENTENCE && (c == '.' || c == '?' || c == '!')) {
                FINALIZE_TRANSACTION();
            }
            i++;
        }
    }
    
    FINALIZE_TOKEN();
    if (mode == MODE_DOCUMENT || mode == MODE_SENTENCE) {
        FINALIZE_TRANSACTION();
    }
    
    if (mode == MODE_SLIDING && global_count > 0) {
        if (window_size == 0) window_size = 10;
        if (stride == 0) stride = 1;
        
        uint32_t *win_buf = malloc(window_size * sizeof(uint32_t));
        
        for (size_t start = 0; start < global_count; start += stride) {
            size_t end = start + window_size;
            if (end > global_count) end = global_count;
            
            size_t count = end - start;
            memcpy(win_buf, global_tokens + start, count * sizeof(uint32_t));
            
            size_t final_count = faro_sort_uniq(win_buf, count);
            emit_callback(win_buf, final_count, user_data);
        }
        
        free(win_buf);
    }
    
    free(txn_tokens);
    free(global_tokens);
    
    #undef FINALIZE_TOKEN
    #undef FINALIZE_TRANSACTION
}

/* Virtual function implementations */

static uint32_t faro_get_or_create_wrapper(Tokenizer *self, const char *token_bytes, uint32_t len, uint64_t hash_val) {
    return faro_get_or_create_impl((FaroTokenizerImpl *)self->impl, token_bytes, len, hash_val);
}

static void faro_tokenize_buffer_wrapper(Tokenizer *self, 
                                         const unsigned char *S, size_t B, 
                                         TransactionMode mode, 
                                         uint32_t window_size, uint32_t stride,
                                         void (*emit_callback)(const uint32_t *tokens, size_t count, void *user_data), 
                                         void *user_data) {
    faro_tokenize_buffer_impl((FaroTokenizerImpl *)self->impl, S, B, mode, window_size, stride, emit_callback, user_data);
}

static void faro_free_wrapper(Tokenizer *self) {
    if (self) {
        FaroTokenizerImpl *tok = (FaroTokenizerImpl *)self->impl;
        if (tok) {
            free(tok->slots);
            free(tok->arena);
            free(tok);
        }
        free(self);
    }
}

static void faro_print_stats_wrapper(Tokenizer *self, const char *input_path, size_t file_size, double elapsed_sec, size_t tx_count, long peak_rss) {
    FaroTokenizerImpl *tok = (FaroTokenizerImpl *)self->impl;
    double mib_processed = (double)file_size / (1024.0 * 1024.0);
    double throughput = mib_processed / elapsed_sec;
    
    uint64_t total_dib = 0;
    uint32_t max_dib = 0;
    uint64_t collision_count = 0;
    uint32_t occupied_count = 0;
    
    for (uint32_t i = 0; i < tok->capacity; i++) {
        FaroSlot *slot = &tok->slots[i];
        uint8_t fp = slot->metadata & 0xFF;
        if (fp != 0) {
            occupied_count++;
            uint32_t dib = (slot->metadata >> 8) & 0xFFFF;
            total_dib += dib;
            if (dib > max_dib) max_dib = dib;
            if (dib > 0) collision_count++;
        }
    }
    
    double avg_probe = occupied_count ? ((double)total_dib / occupied_count) + 1.0 : 1.0;
    double collision_rate = occupied_count ? (double)collision_count / occupied_count : 0.0;
    double load_factor = (double)tok->unique_tokens / tok->capacity;
    
    double slot_mb = (double)(tok->capacity * sizeof(FaroSlot)) / (1024.0 * 1024.0);
    double arena_mb = (double)tok->arena_size / (1024.0 * 1024.0);
    
    printf("\n==================================================\n");
    printf("         FARO-TOKENIZER EXPERIMENT BENCHMARK      \n");
    printf("==================================================\n");
    printf("Input File          : %s\n", input_path);
    printf("Input Size          : %.2f MiB (%zu bytes)\n", mib_processed, file_size);
    printf("Elapsed Time        : %.6f seconds\n", elapsed_sec);
    printf("Throughput          : %.2f MiB/s\n", throughput);
    printf("Transactions Emitted: %zu\n", tx_count);
    printf("--------------------------------------------------\n");
    printf("Dictionary Capacity : %u\n", tok->capacity);
    printf("Unique Tokens (U)   : %u\n", tok->unique_tokens);
    printf("Load Factor (alpha) : %.4f\n", load_factor);
    printf("Slot Memory         : %.2f MiB\n", slot_mb);
    printf("Arena Memory        : %.2f MiB\n", arena_mb);
    printf("--------------------------------------------------\n");
    printf("Average Probe Length: %.4f\n", avg_probe);
    printf("Maximum Probe Length: %u\n", max_dib + 1);
    printf("Collision Rate      : %.4f (DIB > 0)\n", collision_rate);
    printf("Peak RSS            : %ld KB\n", peak_rss);
    printf("==================================================\n\n");
}

Tokenizer *faro_tokenizer_create(uint32_t initial_capacity) {
    uint32_t cap = 16;
    while (cap < initial_capacity) {
        cap *= 2;
    }
    
    FaroTokenizerImpl *tok = malloc(sizeof(FaroTokenizerImpl));
    if (!tok) return NULL;
    
    tok->slots = calloc(cap, sizeof(FaroSlot));
    if (!tok->slots) {
        free(tok);
        return NULL;
    }
    
    tok->capacity = cap;
    tok->unique_tokens = 0;
    tok->arena_capacity = 4096;
    tok->arena = malloc(tok->arena_capacity);
    tok->arena_size = 0;
    
    Tokenizer *self = malloc(sizeof(Tokenizer));
    if (!self) {
        free(tok->slots);
        free(tok->arena);
        free(tok);
        return NULL;
    }
    
    self->name = "FARO-Tokenizer";
    self->impl = tok;
    self->get_or_create = faro_get_or_create_wrapper;
    self->tokenize_buffer = faro_tokenize_buffer_wrapper;
    self->free = faro_free_wrapper;
    self->print_stats = faro_print_stats_wrapper;
    
    return self;
}
