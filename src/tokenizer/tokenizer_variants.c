#include "../../include/tokenizer/tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Reuse the polymorphic scan loop from the main framework */
extern void faro_tokenize_buffer_polymorphic(Tokenizer *self, 
                                             const unsigned char *S, size_t B, 
                                             TransactionMode mode, 
                                             uint32_t window_size, uint32_t stride,
                                             void (*emit_callback)(const uint32_t *tokens, size_t count, void *user_data), 
                                             void *user_data);

extern int output_json;


/* ========================================================================= */
/* 1. LP-Raw: Linear Probing with raw string keys (no fingerprint filtering) */
/* ========================================================================= */

typedef struct {
    FaroSlot *slots;
    uint32_t capacity;
    char *arena;
    uint32_t arena_size;
    uint32_t arena_capacity;
    uint32_t unique_tokens;
} LpRawImpl;

static const char *arena_like_token_text(FaroSlot *slots, uint32_t capacity, const char *arena, uint32_t token_id, uint32_t *len) {
    if (!slots || !arena || token_id == 0) return NULL;
    for (uint32_t i = 0; i < capacity; i++) {
        if (slots[i].token_id == token_id) {
            if (len) *len = slots[i].token_len;
            return arena + slots[i].arena_offset;
        }
    }
    return NULL;
}

static const char *lp_raw_token_text(Tokenizer *self, uint32_t token_id, uint32_t *len) {
    LpRawImpl *tok = (LpRawImpl *)self->impl;
    return tok ? arena_like_token_text(tok->slots, tok->capacity, tok->arena, token_id, len) : NULL;
}

static uint32_t lp_raw_vocab_size(Tokenizer *self) {
    LpRawImpl *tok = (LpRawImpl *)self->impl;
    return tok ? tok->unique_tokens : 0;
}

static void lp_raw_rehash(LpRawImpl *tok) {
    uint32_t old_capacity = tok->capacity;
    FaroSlot *old_slots = tok->slots;
    
    uint32_t new_capacity = old_capacity * 2;
    FaroSlot *new_slots = calloc(new_capacity, sizeof(FaroSlot));
    if (!new_slots) return;
    
    tok->capacity = new_capacity;
    tok->slots = new_slots;
    
    for (uint32_t i = 0; i < old_capacity; i++) {
        FaroSlot *old_slot = &old_slots[i];
        if (old_slot->token_id == 0) continue;
        
        const char *bytes = tok->arena + old_slot->arena_offset;
        uint64_t hash_val = faro_hash(bytes, old_slot->token_len);
        
        uint32_t mask = new_capacity - 1;
        uint32_t s = hash_val & mask;
        
        while (1) {
            FaroSlot *slot = &new_slots[s];
            if (slot->token_id == 0) {
                *slot = *old_slot;
                break;
            }
            s = (s + 1) & mask;
        }
    }
    
    free(old_slots);
}

static uint32_t lp_raw_get_or_create(Tokenizer *self, const char *token_bytes, uint32_t len, uint64_t hash_val) {
    LpRawImpl *tok = (LpRawImpl *)self->impl;
    uint32_t capacity = tok->capacity;
    uint32_t mask = capacity - 1;
    uint32_t s = hash_val & mask;
    
    while (1) {
        FaroSlot *slot = &tok->slots[s];
        
        if (slot->token_id == 0) {
            /* Materialize new token */
            uint32_t offset = tok->arena_size;
            if (tok->arena_size + len > tok->arena_capacity) {
                tok->arena_capacity = tok->arena_capacity * 2 + len;
                tok->arena = realloc(tok->arena, tok->arena_capacity);
            }
            memcpy(tok->arena + offset, token_bytes, len);
            tok->arena_size += len;
            
            tok->unique_tokens++;
            slot->arena_offset = offset;
            slot->token_len = len;
            slot->token_id = tok->unique_tokens;
            
            if ((double)tok->unique_tokens / capacity > 0.75) {
                lp_raw_rehash(tok);
            }
            
            return slot->token_id;
        }
        
        /* Direct full string compare without fingerprint filtering */
        if (slot->token_len == len && memcmp(tok->arena + slot->arena_offset, token_bytes, len) == 0) {
            return slot->token_id;
        }
        
        s = (s + 1) & mask;
    }
}

static void lp_raw_free(Tokenizer *self) {
    if (self) {
        LpRawImpl *tok = (LpRawImpl *)self->impl;
        if (tok) {
            free(tok->slots);
            free(tok->arena);
            free(tok);
        }
        free(self);
    }
}

static void lp_raw_print_stats(Tokenizer *self, const char *input_path, size_t file_size, double elapsed_sec, size_t tx_count, long peak_rss) {
    (void)input_path;
    (void)tx_count;
    LpRawImpl *tok = (LpRawImpl *)self->impl;
    double mib = (double)file_size / (1024.0 * 1024.0);
    double throughput = mib / elapsed_sec;
    
    uint64_t total_probes = 0;
    uint32_t max_probes = 0;
    uint32_t collisions = 0;
    
    for (uint32_t i = 0; i < tok->capacity; i++) {
        FaroSlot *slot = &tok->slots[i];
        if (slot->token_id != 0) {
            const char *bytes = tok->arena + slot->arena_offset;
            uint64_t hash_val = faro_hash(bytes, slot->token_len);
            uint32_t init_b = hash_val & (tok->capacity - 1);
            uint32_t probes = ((i - init_b + tok->capacity) & (tok->capacity - 1)) + 1;
            total_probes += probes;
            if (probes > max_probes) max_probes = probes;
            if (probes > 1) collisions++;
        }
    }
    
    double avg_probe = tok->unique_tokens ? (double)total_probes / tok->unique_tokens : 1.0;
    double collision_rate = tok->unique_tokens ? (double)collisions / tok->unique_tokens : 0.0;
    double load_factor = (double)tok->unique_tokens / tok->capacity;
    
    if (output_json) {
        printf("{\"variant\":\"LP-Raw\",\"throughput_mib\":%.4f,\"unique_tokens\":%u,\"load_factor\":%.6f,\"avg_probe\":%.6f,\"max_probe\":%u,\"collision_rate\":%.6f,\"peak_rss_kb\":%ld}\n",
               throughput, tok->unique_tokens, load_factor, avg_probe, max_probes, collision_rate, peak_rss);
    } else {
        printf("LP-Raw\t%.2f MiB/s\t%u unique\tPeak RSS: %ld KB\n", throughput, tok->unique_tokens, peak_rss);
    }
}


Tokenizer *lp_raw_tokenizer_create(uint32_t initial_capacity) {
    uint32_t cap = 16;
    while (cap < initial_capacity) cap *= 2;
    
    LpRawImpl *tok = malloc(sizeof(LpRawImpl));
    tok->slots = calloc(cap, sizeof(FaroSlot));
    tok->capacity = cap;
    tok->unique_tokens = 0;
    tok->arena_capacity = 4096;
    tok->arena = malloc(tok->arena_capacity);
    tok->arena_size = 0;
    
    Tokenizer *self = malloc(sizeof(Tokenizer));
    self->name = "LP-Raw";
    self->impl = tok;
    self->base_address = NULL;
    self->get_or_create = lp_raw_get_or_create;
    self->tokenize_buffer = faro_tokenize_buffer_polymorphic;
    self->free = lp_raw_free;
    self->print_stats = lp_raw_print_stats;
    self->token_text = lp_raw_token_text;
    self->vocab_size = lp_raw_vocab_size;
    return self;
}

/* ========================================================================= */
/* 2. LP-FP: Linear Probing with fingerprint filtering                       */
/* ========================================================================= */

typedef struct {
    FaroSlot *slots;
    uint32_t capacity;
    char *arena;
    uint32_t arena_size;
    uint32_t arena_capacity;
    uint32_t unique_tokens;
} LpFpImpl;

static const char *lp_fp_token_text(Tokenizer *self, uint32_t token_id, uint32_t *len) {
    LpFpImpl *tok = (LpFpImpl *)self->impl;
    return tok ? arena_like_token_text(tok->slots, tok->capacity, tok->arena, token_id, len) : NULL;
}

static uint32_t lp_fp_vocab_size(Tokenizer *self) {
    LpFpImpl *tok = (LpFpImpl *)self->impl;
    return tok ? tok->unique_tokens : 0;
}

static void lp_fp_rehash(LpFpImpl *tok) {
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
        
        while (1) {
            FaroSlot *slot = &new_slots[s];
            uint8_t slot_fp = slot->metadata & 0xFF;
            if (slot_fp == 0) {
                *slot = *old_slot;
                break;
            }
            s = (s + 1) & mask;
        }
    }
    
    free(old_slots);
}

static uint32_t lp_fp_get_or_create(Tokenizer *self, const char *token_bytes, uint32_t len, uint64_t hash_val) {
    LpFpImpl *tok = (LpFpImpl *)self->impl;
    uint32_t capacity = tok->capacity;
    uint32_t mask = capacity - 1;
    uint32_t s = hash_val & mask;
    
    uint8_t fp = (hash_val >> 56) == 0 ? 1 : (hash_val >> 56);
    
    while (1) {
        FaroSlot *slot = &tok->slots[s];
        uint8_t slot_fp = slot->metadata & 0xFF;
        
        if (slot_fp == 0) {
            uint32_t offset = tok->arena_size;
            if (tok->arena_size + len > tok->arena_capacity) {
                tok->arena_capacity = tok->arena_capacity * 2 + len;
                tok->arena = realloc(tok->arena, tok->arena_capacity);
            }
            memcpy(tok->arena + offset, token_bytes, len);
            tok->arena_size += len;
            
            tok->unique_tokens++;
            slot->metadata = fp;
            slot->arena_offset = offset;
            slot->token_len = len;
            slot->token_id = tok->unique_tokens;
            
            if ((double)tok->unique_tokens / capacity > 0.75) {
                lp_fp_rehash(tok);
            }
            
            return slot->token_id;
        }
        
        /* Fast-path fingerprint filter check */
        if (slot_fp == fp && slot->token_len == len && memcmp(tok->arena + slot->arena_offset, token_bytes, len) == 0) {
            return slot->token_id;
        }
        
        s = (s + 1) & mask;
    }
}

static void lp_fp_free(Tokenizer *self) {
    if (self) {
        LpFpImpl *tok = (LpFpImpl *)self->impl;
        if (tok) {
            free(tok->slots);
            free(tok->arena);
            free(tok);
        }
        free(self);
    }
}

static void lp_fp_print_stats(Tokenizer *self, const char *input_path, size_t file_size, double elapsed_sec, size_t tx_count, long peak_rss) {
    (void)input_path;
    (void)tx_count;
    LpFpImpl *tok = (LpFpImpl *)self->impl;
    double mib = (double)file_size / (1024.0 * 1024.0);
    double throughput = mib / elapsed_sec;
    
    uint64_t total_probes = 0;
    uint32_t max_probes = 0;
    uint32_t collisions = 0;
    
    for (uint32_t i = 0; i < tok->capacity; i++) {
        FaroSlot *slot = &tok->slots[i];
        uint8_t fp = slot->metadata & 0xFF;
        if (fp != 0) {
            const char *bytes = tok->arena + slot->arena_offset;
            uint64_t hash_val = faro_hash(bytes, slot->token_len);
            uint32_t init_b = hash_val & (tok->capacity - 1);
            uint32_t probes = ((i - init_b + tok->capacity) & (tok->capacity - 1)) + 1;
            total_probes += probes;
            if (probes > max_probes) max_probes = probes;
            if (probes > 1) collisions++;
        }
    }
    
    double avg_probe = tok->unique_tokens ? (double)total_probes / tok->unique_tokens : 1.0;
    double collision_rate = tok->unique_tokens ? (double)collisions / tok->unique_tokens : 0.0;
    double load_factor = (double)tok->unique_tokens / tok->capacity;
    
    if (output_json) {
        printf("{\"variant\":\"LP-FP\",\"throughput_mib\":%.4f,\"unique_tokens\":%u,\"load_factor\":%.6f,\"avg_probe\":%.6f,\"max_probe\":%u,\"collision_rate\":%.6f,\"peak_rss_kb\":%ld}\n",
               throughput, tok->unique_tokens, load_factor, avg_probe, max_probes, collision_rate, peak_rss);
    } else {
        printf("LP-FP\t%.2f MiB/s\t%u unique\tPeak RSS: %ld KB\n", throughput, tok->unique_tokens, peak_rss);
    }
}


Tokenizer *lp_fp_tokenizer_create(uint32_t initial_capacity) {
    uint32_t cap = 16;
    while (cap < initial_capacity) cap *= 2;
    
    LpFpImpl *tok = malloc(sizeof(LpFpImpl));
    tok->slots = calloc(cap, sizeof(FaroSlot));
    tok->capacity = cap;
    tok->unique_tokens = 0;
    tok->arena_capacity = 4096;
    tok->arena = malloc(tok->arena_capacity);
    tok->arena_size = 0;
    
    Tokenizer *self = malloc(sizeof(Tokenizer));
    self->name = "LP-FP";
    self->impl = tok;
    self->base_address = NULL;
    self->get_or_create = lp_fp_get_or_create;
    self->tokenize_buffer = faro_tokenize_buffer_polymorphic;
    self->free = lp_fp_free;
    self->print_stats = lp_fp_print_stats;
    self->token_text = lp_fp_token_text;
    self->vocab_size = lp_fp_vocab_size;
    return self;
}

/* ========================================================================= */
/* 3. RH-FP: Robin Hood with fingerprint (pointer-based lexemes, no arena)   */
/* ========================================================================= */

typedef struct {
    FaroSlot *slots;
    uint32_t capacity;
    char **unique_ptrs;
    uint32_t unique_ptrs_size;
    uint32_t unique_ptrs_capacity;
    uint32_t unique_tokens;
} RhFpImpl;

static void rh_fp_rehash(RhFpImpl *tok) {
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
        
        const char *bytes = tok->unique_ptrs[old_slot->arena_offset];
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

static uint32_t rh_fp_get_or_create(Tokenizer *self, const char *token_bytes, uint32_t len, uint64_t hash_val) {
    RhFpImpl *tok = (RhFpImpl *)self->impl;
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
                /* Allocate separate memory for string lexeme */
                char *copy = malloc(len + 1);
                memcpy(copy, token_bytes, len);
                copy[len] = '\0';
                
                if (tok->unique_ptrs_size >= tok->unique_ptrs_capacity) {
                    tok->unique_ptrs_capacity = tok->unique_ptrs_capacity * 2 + 128;
                    tok->unique_ptrs = realloc(tok->unique_ptrs, tok->unique_ptrs_capacity * sizeof(char *));
                }
                
                cand_offset = tok->unique_ptrs_size;
                tok->unique_ptrs[tok->unique_ptrs_size++] = copy;
                tok->unique_tokens++;
                cand_id = tok->unique_tokens;
                is_materialized = 1;
            }
            
            slot->metadata = cand_fp | (cand_dib << 8);
            slot->arena_offset = cand_offset;
            slot->token_len = cand_len;
            slot->token_id = cand_id;
            
            if ((double)tok->unique_tokens / capacity > 0.75) {
                rh_fp_rehash(tok);
            }
            
            return cand_id;
        }
        
        if (slot_fp == cand_fp && slot->token_len == cand_len && 
            memcmp(tok->unique_ptrs[slot->arena_offset], token_bytes, cand_len) == 0) {
            return slot->token_id;
        }
        
        goto next_probe;
        
    do_insert_swap:
        if (!is_materialized) {
            char *copy = malloc(len + 1);
            memcpy(copy, token_bytes, len);
            copy[len] = '\0';
            
            if (tok->unique_ptrs_size >= tok->unique_ptrs_capacity) {
                tok->unique_ptrs_capacity = tok->unique_ptrs_capacity * 2 + 128;
                tok->unique_ptrs = realloc(tok->unique_ptrs, tok->unique_ptrs_capacity * sizeof(char *));
            }
            
            cand_offset = tok->unique_ptrs_size;
            tok->unique_ptrs[tok->unique_ptrs_size++] = copy;
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
        
        token_bytes = tok->unique_ptrs[cand_offset];
        
    next_probe:
        s = (s + 1) & mask;
        cand_dib++;
    }
}

static void rh_fp_free(Tokenizer *self) {
    if (self) {
        RhFpImpl *tok = (RhFpImpl *)self->impl;
        if (tok) {
            for (uint32_t i = 0; i < tok->unique_ptrs_size; i++) {
                free(tok->unique_ptrs[i]);
            }
            free(tok->unique_ptrs);
            free(tok->slots);
            free(tok);
        }
        free(self);
    }
}

static void rh_fp_print_stats(Tokenizer *self, const char *input_path, size_t file_size, double elapsed_sec, size_t tx_count, long peak_rss) {
    (void)input_path;
    (void)tx_count;
    RhFpImpl *tok = (RhFpImpl *)self->impl;
    double mib = (double)file_size / (1024.0 * 1024.0);
    double throughput = mib / elapsed_sec;
    
    uint64_t total_probes = 0;
    uint32_t max_probes = 0;
    uint32_t collisions = 0;
    
    for (uint32_t i = 0; i < tok->capacity; i++) {
        FaroSlot *slot = &tok->slots[i];
        uint8_t fp = slot->metadata & 0xFF;
        if (fp != 0) {
            uint32_t dib = (slot->metadata >> 8) & 0xFFFF;
            uint32_t probes = dib + 1;
            total_probes += probes;
            if (probes > max_probes) max_probes = probes;
            if (probes > 1) collisions++;
        }
    }
    
    double avg_probe = tok->unique_tokens ? (double)total_probes / tok->unique_tokens : 1.0;
    double collision_rate = tok->unique_tokens ? (double)collisions / tok->unique_tokens : 0.0;
    double load_factor = (double)tok->unique_tokens / tok->capacity;
    
    if (output_json) {
        printf("{\"variant\":\"RH-FP\",\"throughput_mib\":%.4f,\"unique_tokens\":%u,\"load_factor\":%.6f,\"avg_probe\":%.6f,\"max_probe\":%u,\"collision_rate\":%.6f,\"peak_rss_kb\":%ld}\n",
               throughput, tok->unique_tokens, load_factor, avg_probe, max_probes, collision_rate, peak_rss);
    } else {
        printf("RH-FP\t%.2f MiB/s\t%u unique\tPeak RSS: %ld KB\n", throughput, tok->unique_tokens, peak_rss);
    }
}

static const char *rh_fp_token_text(Tokenizer *self, uint32_t token_id, uint32_t *len) {
    RhFpImpl *tok = (RhFpImpl *)self->impl;
    if (!tok || token_id == 0) return NULL;
    for (uint32_t i = 0; i < tok->capacity; i++) {
        FaroSlot *slot = &tok->slots[i];
        if (slot->token_id == token_id) {
            if (len) *len = slot->token_len;
            return tok->unique_ptrs[slot->arena_offset];
        }
    }
    return NULL;
}

static uint32_t rh_fp_vocab_size(Tokenizer *self) {
    RhFpImpl *tok = (RhFpImpl *)self->impl;
    return tok ? tok->unique_tokens : 0;
}


Tokenizer *rh_fp_tokenizer_create(uint32_t initial_capacity) {
    uint32_t cap = 16;
    while (cap < initial_capacity) cap *= 2;
    
    RhFpImpl *tok = malloc(sizeof(RhFpImpl));
    tok->slots = calloc(cap, sizeof(FaroSlot));
    tok->capacity = cap;
    tok->unique_ptrs_capacity = 1024;
    tok->unique_ptrs = malloc(tok->unique_ptrs_capacity * sizeof(char *));
    tok->unique_ptrs_size = 0;
    tok->unique_tokens = 0;
    
    Tokenizer *self = malloc(sizeof(Tokenizer));
    self->name = "RH-FP";
    self->impl = tok;
    self->base_address = NULL;
    self->get_or_create = rh_fp_get_or_create;
    self->tokenize_buffer = faro_tokenize_buffer_polymorphic;
    self->free = rh_fp_free;
    self->print_stats = rh_fp_print_stats;
    self->token_text = rh_fp_token_text;
    self->vocab_size = rh_fp_vocab_size;
    return self;
}

/* ========================================================================= */
/* 4. RH-Borrow: Robin Hood in Zero-Copy borrowed-view mode                  */
/* ========================================================================= */

typedef struct {
    FaroSlot *slots;
    uint32_t capacity;
    uint32_t unique_tokens;
} RhBorrowImpl;

static void rh_borrow_rehash(Tokenizer *self, RhBorrowImpl *tok) {
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
        
        const char *bytes = (const char *)self->base_address + old_slot->arena_offset;
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

static uint32_t rh_borrow_get_or_create(Tokenizer *self, const char *token_bytes, uint32_t len, uint64_t hash_val) {
    RhBorrowImpl *tok = (RhBorrowImpl *)self->impl;
    uint32_t capacity = tok->capacity;
    uint32_t mask = capacity - 1;
    uint32_t s = hash_val & mask;
    uint32_t d = 0;
    
    uint8_t fingerprint = (hash_val >> 56) == 0 ? 1 : (hash_val >> 56);
    
    uint8_t cand_fp = fingerprint;
    uint32_t cand_dib = d;
    uint32_t cand_len = len;
    uint32_t cand_id = 0;
    /* Borrowed offset directly references base_address memory mapping */
    uint32_t cand_offset = (uint32_t)(token_bytes - (const char *)self->base_address);
    
    while (1) {
        FaroSlot *slot = &tok->slots[s];
        uint8_t slot_fp = slot->metadata & 0xFF;
        uint32_t slot_dib = (slot->metadata >> 8) & 0xFFFF;
        
        if (slot_fp != 0 && slot_dib < cand_dib) {
            goto do_insert_swap;
        }
        
        if (slot_fp == 0) {
            if (cand_id == 0) {
                tok->unique_tokens++;
                cand_id = tok->unique_tokens;
            }
            
            slot->metadata = cand_fp | (cand_dib << 8);
            slot->arena_offset = cand_offset;
            slot->token_len = cand_len;
            slot->token_id = cand_id;
            
            if ((double)tok->unique_tokens / capacity > 0.75) {
                rh_borrow_rehash(self, tok);
            }
            
            return cand_id;
        }
        
        if (slot_fp == cand_fp && slot->token_len == cand_len && 
            memcmp((const char *)self->base_address + slot->arena_offset, token_bytes, cand_len) == 0) {
            return slot->token_id;
        }
        
        goto next_probe;
        
    do_insert_swap:
        if (cand_id == 0) {
            tok->unique_tokens++;
            cand_id = tok->unique_tokens;
        }

        /* Swap candidate state with occupant */
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
        
        token_bytes = (const char *)self->base_address + cand_offset;
        
    next_probe:
        s = (s + 1) & mask;
        cand_dib++;
    }
}

static void rh_borrow_free(Tokenizer *self) {
    if (self) {
        RhBorrowImpl *tok = (RhBorrowImpl *)self->impl;
        if (tok) {
            free(tok->slots);
            free(tok);
        }
        free(self);
    }
}

static void rh_borrow_print_stats(Tokenizer *self, const char *input_path, size_t file_size, double elapsed_sec, size_t tx_count, long peak_rss) {
    (void)input_path;
    (void)tx_count;
    RhBorrowImpl *tok = (RhBorrowImpl *)self->impl;
    double mib = (double)file_size / (1024.0 * 1024.0);
    double throughput = mib / elapsed_sec;
    
    uint64_t total_probes = 0;
    uint32_t max_probes = 0;
    uint32_t collisions = 0;
    
    for (uint32_t i = 0; i < tok->capacity; i++) {
        FaroSlot *slot = &tok->slots[i];
        uint8_t fp = slot->metadata & 0xFF;
        if (fp != 0) {
            uint32_t dib = (slot->metadata >> 8) & 0xFFFF;
            uint32_t probes = dib + 1;
            total_probes += probes;
            if (probes > max_probes) max_probes = probes;
            if (probes > 1) collisions++;
        }
    }
    
    double avg_probe = tok->unique_tokens ? (double)total_probes / tok->unique_tokens : 1.0;
    double collision_rate = tok->unique_tokens ? (double)collisions / tok->unique_tokens : 0.0;
    double load_factor = (double)tok->unique_tokens / tok->capacity;
    
    if (output_json) {
        printf("{\"variant\":\"RH-Borrow\",\"throughput_mib\":%.4f,\"unique_tokens\":%u,\"load_factor\":%.6f,\"avg_probe\":%.6f,\"max_probe\":%u,\"collision_rate\":%.6f,\"peak_rss_kb\":%ld}\n",
               throughput, tok->unique_tokens, load_factor, avg_probe, max_probes, collision_rate, peak_rss);
    } else {
        printf("RH-Borrow\t%.2f MiB/s\t%u unique\tPeak RSS: %ld KB\n", throughput, tok->unique_tokens, peak_rss);
    }
}


/* Highly optimized zero-copy scanner for borrowed-view mode matching Section 6 */
static void rh_borrow_tokenize_buffer_impl(Tokenizer *self, 
                                           const unsigned char *S, size_t B, 
                                           TransactionMode mode, 
                                           uint32_t window_size, uint32_t stride,
                                           void (*emit_callback)(const uint32_t *tokens, size_t count, void *user_data), 
                                           void *user_data) {
    self->base_address = S; /* store mapping address */
    
    uint32_t *txn_tokens = NULL;
    size_t txn_count = 0;
    size_t txn_capacity = 0;
    
    uint32_t *global_tokens = NULL;
    size_t global_count = 0;
    size_t global_capacity = 0;
    
    #define EMIT_TID(tid) do { \
        if (mode == MODE_SLIDING || mode == MODE_SLIDING_SEQUENCE) { \
            if (global_count >= global_capacity) { \
                global_capacity = global_capacity * 2 + 128; \
                global_tokens = realloc(global_tokens, global_capacity * sizeof(uint32_t)); \
            } \
            global_tokens[global_count++] = (tid); \
        } else { \
            if (txn_count >= txn_capacity) { \
                txn_capacity = txn_capacity * 2 + 32; \
                txn_tokens = realloc(txn_tokens, txn_capacity * sizeof(uint32_t)); \
            } \
            txn_tokens[txn_count++] = (tid); \
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
    size_t token_start = 0;
    int inside_token = 0;
    
    while (i < B) {
        unsigned char c = S[i];
        
        /* Word bytes check (ASCII identity-canonical only for borrowed mode) */
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c >= 0x80) {
            if (!inside_token) {
                token_start = i;
                inside_token = 1;
            }
            i++;
        } else if (c == '-' || c == '_' || c == '\'') {
            /* Joiner check */
            if (inside_token && i + 1 < B) {
                unsigned char next_c = S[i + 1];
                if ((next_c >= 'a' && next_c <= 'z') || (next_c >= 'A' && next_c <= 'Z') || (next_c >= '0' && next_c <= '9') || next_c >= 0x80) {
                    i += 2;
                    continue;
                }
            }
            
            /* Treat as delimiter */
            if (inside_token) {
                size_t len = i - token_start;
                uint64_t hash_val = faro_hash((const char *)S + token_start, len);
                uint32_t tid = self->get_or_create(self, (const char *)S + token_start, len, hash_val);
                EMIT_TID(tid);
                inside_token = 0;
            }
            i++;
        } else {
            /* Delimiter */
            if (inside_token) {
                size_t len = i - token_start;
                uint64_t hash_val = faro_hash((const char *)S + token_start, len);
                uint32_t tid = self->get_or_create(self, (const char *)S + token_start, len, hash_val);
                EMIT_TID(tid);
                inside_token = 0;
            }
            if (mode == MODE_DOCUMENT && c == '\n') {
                FINALIZE_TRANSACTION();
            } else if (mode == MODE_SENTENCE && (c == '.' || c == '?' || c == '!')) {
                FINALIZE_TRANSACTION();
            }
            i++;
        }
    }
    
    if (inside_token) {
        size_t len = B - token_start;
        uint64_t hash_val = faro_hash((const char *)S + token_start, len);
        uint32_t tid = self->get_or_create(self, (const char *)S + token_start, len, hash_val);
        EMIT_TID(tid);
    }
    
    if (mode == MODE_DOCUMENT || mode == MODE_SENTENCE) {
        FINALIZE_TRANSACTION();
    }
    
    if ((mode == MODE_SLIDING || mode == MODE_SLIDING_SEQUENCE) && global_count > 0) {
        if (window_size == 0) window_size = 10;
        if (stride == 0) stride = 1;
        uint32_t *win_buf = malloc(window_size * sizeof(uint32_t));
        
        for (size_t start = 0; start < global_count; start += stride) {
            size_t end = start + window_size;
            if (end > global_count) end = global_count;
            size_t count = end - start;
            memcpy(win_buf, global_tokens + start, count * sizeof(uint32_t));
            size_t final_count = (mode == MODE_SLIDING_SEQUENCE) ? count : faro_sort_uniq(win_buf, count);
            emit_callback(win_buf, final_count, user_data);
        }
        free(win_buf);
    }
    
    free(txn_tokens);
    free(global_tokens);
    
    #undef EMIT_TID
    #undef FINALIZE_TRANSACTION
}

static const char *rh_borrow_token_text(Tokenizer *self, uint32_t token_id, uint32_t *len) {
    RhBorrowImpl *tok = (RhBorrowImpl *)self->impl;
    if (!tok || !self->base_address || token_id == 0) return NULL;
    for (uint32_t i = 0; i < tok->capacity; i++) {
        FaroSlot *slot = &tok->slots[i];
        if (slot->token_id == token_id) {
            if (len) *len = slot->token_len;
            return (const char *)self->base_address + slot->arena_offset;
        }
    }
    return NULL;
}

static uint32_t rh_borrow_vocab_size(Tokenizer *self) {
    RhBorrowImpl *tok = (RhBorrowImpl *)self->impl;
    return tok ? tok->unique_tokens : 0;
}

Tokenizer *rh_borrow_tokenizer_create(uint32_t initial_capacity) {
    uint32_t cap = 16;
    while (cap < initial_capacity) cap *= 2;
    
    RhBorrowImpl *tok = malloc(sizeof(RhBorrowImpl));
    tok->slots = calloc(cap, sizeof(FaroSlot));
    tok->capacity = cap;
    tok->unique_tokens = 0;
    
    Tokenizer *self = malloc(sizeof(Tokenizer));
    self->name = "RH-Borrow";
    self->impl = tok;
    self->base_address = NULL;
    self->get_or_create = rh_borrow_get_or_create;
    self->tokenize_buffer = rh_borrow_tokenize_buffer_impl;
    self->free = rh_borrow_free;
    self->print_stats = rh_borrow_print_stats;
    self->token_text = rh_borrow_token_text;
    self->vocab_size = rh_borrow_vocab_size;
    return self;
}

Tokenizer *dm_tokenizer_create(const char *name, uint32_t initial_capacity) {
    if (!name || strcmp(name, "faro") == 0 || strcmp(name, "rh-arena") == 0) {
        return rh_arena_tokenizer_create(initial_capacity);
    }
    if (strcmp(name, "lp-raw") == 0) return lp_raw_tokenizer_create(initial_capacity);
    if (strcmp(name, "lp-fp") == 0) return lp_fp_tokenizer_create(initial_capacity);
    if (strcmp(name, "rh-fp") == 0) return rh_fp_tokenizer_create(initial_capacity);
    if (strcmp(name, "rh-borrow") == 0) return rh_borrow_tokenizer_create(initial_capacity);
    return NULL;
}

const char *dm_tokenizer_supported_names(void) {
    return "faro|rh-arena|lp-raw|lp-fp|rh-fp|rh-borrow";
}

const char *dm_tokenizer_token_text(Tokenizer *self, uint32_t token_id, uint32_t *len) {
    if (!self || !self->token_text) return NULL;
    return self->token_text(self, token_id, len);
}

uint32_t dm_tokenizer_vocab_size(Tokenizer *self) {
    if (!self || !self->vocab_size) return 0;
    return self->vocab_size(self);
}
