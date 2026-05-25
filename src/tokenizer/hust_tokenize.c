#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#include <io.h>
#else
#include <sys/mman.h>
#include <sys/resource.h>
#endif
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctype.h>

#include "../../include/tokenizer/tokenizer.h"

#define HUST_NO_RANK UINT32_MAX

#ifdef _WIN32
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
static int clock_gettime(int unused, struct timespec *ts) {
    static LARGE_INTEGER freq;
    static int initialized = 0;
    LARGE_INTEGER counter;
    (void)unused;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }
    QueryPerformanceCounter(&counter);
    ts->tv_sec = (time_t)(counter.QuadPart / freq.QuadPart);
    ts->tv_nsec = (long)(((counter.QuadPart % freq.QuadPart) * 1000000000LL) / freq.QuadPart);
    return 0;
}
#endif
#endif

static unsigned char *read_input_file(const char *path, size_t *size_out) {
    FILE *f = fopen(path, "rb");
    long size;
    unsigned char *data;
    if (!f) {
        perror("Failed to open input file");
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        perror("Failed to seek input file");
        fclose(f);
        return NULL;
    }
    size = ftell(f);
    if (size <= 0) {
        fprintf(stderr, "Empty input file\n");
        fclose(f);
        return NULL;
    }
    rewind(f);
    data = (unsigned char *)malloc((size_t)size);
    if (!data) {
        fclose(f);
        return NULL;
    }
    if (fread(data, 1, (size_t)size, f) != (size_t)size) {
        perror("Failed to read input file");
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size_out = (size_t)size;
    return data;
}

typedef struct {
    char *name;
    uint32_t df;
    uint32_t rank;
    double weight;
    double tiub;
    unsigned char survives;
} HUSTToken;

typedef struct {
    uint32_t *items;
    size_t count;
    double itu;
} HUSTTransaction;

typedef struct {
    uint32_t tid;
    uint32_t last;
    uint32_t span;
    double eu;
    double ru;
} HUSTEntry;

typedef struct {
    HUSTEntry *entries;
    size_t count;
    size_t cap;
    uint32_t item;
    double pwt;
} HUSTList;

typedef struct {
    HUSTToken *tokens;
    size_t token_count;
    size_t token_cap;
    HUSTTransaction *tx;
    size_t tx_count;
    size_t tx_cap;
    size_t stream_count;
    size_t raw_nnz;
    size_t input_bytes;
    uint32_t *rank_to_item;
    uint32_t *item_to_rank;
    size_t survivor_count;
} HUSTDB;

typedef struct {
    uint32_t *tokens;
    size_t len;
    double utility;
    size_t support;
} HUSTPattern;

typedef struct {
    HUSTDB *db;
    HUSTList *singletons;
    FILE *pattern_out;
    clock_t started;
    
    // Parameters
    double theta;
    double theta_ratio;
    size_t sigma_0;
    double alpha;
    int is_sequence_mode;
    int filter_noise;
    size_t max_depth;
    size_t max_patterns;
    
    // Discovered patterns
    HUSTPattern *patterns;
    size_t pattern_count;
    size_t pattern_cap;
    
    // Stats
    size_t visited_nodes;
    size_t joins;
    size_t joined_entries;
    size_t pruned_support;
    size_t pruned_tiub;
    size_t pruned_iwru;
    
    // Workspace marks
    uint32_t *tail_marks;
    uint32_t tail_stamp;
} HUSTContext;

// Global ordering context for qsort
static HUSTDB *g_order_db = NULL;
static int token_order_cmp(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    const HUSTToken *tx = &g_order_db->tokens[x];
    const HUSTToken *ty = &g_order_db->tokens[y];
    if (tx->tiub < ty->tiub) return -1;
    if (tx->tiub > ty->tiub) return 1;
    if (tx->weight > ty->weight) return -1;
    if (tx->weight < ty->weight) return 1;
    return (x > y) - (x < y);
}

static uint32_t *g_rank_map = NULL;
static int rank_order_cmp(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    uint32_t rx = g_rank_map[x];
    uint32_t ry = g_rank_map[y];
    return (rx > ry) - (rx < ry);
}

static int uint32_asc_cmp(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a;
    uint32_t y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

static size_t unique_sorted_ids(uint32_t *items, size_t count) {
    if (count == 0) return 0;
    qsort(items, count, sizeof(uint32_t), uint32_asc_cmp);
    size_t out = 1;
    for (size_t i = 1; i < count; i++) {
        if (items[i] != items[out - 1]) items[out++] = items[i];
    }
    return out;
}

static int db_init(HUSTDB *db) {
    memset(db, 0, sizeof(*db));
    return 0;
}

static void db_free(HUSTDB *db) {
    if (!db) return;
    for (size_t i = 0; i < db->token_count; i++) free(db->tokens[i].name);
    for (size_t i = 0; i < db->tx_count; i++) free(db->tx[i].items);
    free(db->tokens);
    free(db->tx);
    free(db->rank_to_item);
    free(db->item_to_rank);
    memset(db, 0, sizeof(*db));
}

static void list_free(HUSTList *list) {
    if (!list) return;
    free(list->entries);
    list->entries = NULL;
    list->count = 0;
    list->cap = 0;
}

static int list_push(HUSTList *list, HUSTEntry entry) {
    if (list->count >= list->cap) {
        size_t next = list->cap ? list->cap * 2 : 64;
        HUSTEntry *tmp = realloc(list->entries, next * sizeof(HUSTEntry));
        if (!tmp) return -1;
        list->entries = tmp;
        list->cap = next;
    }
    list->entries[list->count++] = entry;
    return 0;
}

static int add_token_bytes(HUSTDB *db, const char *name, size_t len, uint32_t *id_out) {
    if (db->token_count >= db->token_cap) {
        size_t next = db->token_cap ? db->token_cap * 2 : 1024;
        HUSTToken *tmp = realloc(db->tokens, next * sizeof(HUSTToken));
        if (!tmp) return -1;
        db->tokens = tmp;
        db->token_cap = next;
    }
    uint32_t id = (uint32_t)db->token_count++;
    HUSTToken *tok = &db->tokens[id];
    memset(tok, 0, sizeof(*tok));
    tok->name = malloc(len + 1);
    if (!tok->name) return -1;
    memcpy(tok->name, name, len);
    tok->name[len] = '\0';
    tok->rank = HUST_NO_RANK;
    *id_out = id;
    return 0;
}

static int add_transaction(HUSTDB *db, const uint32_t *items, size_t count) {
    if (count == 0) return 0;
    if (db->tx_count >= db->tx_cap) {
        size_t next = db->tx_cap ? db->tx_cap * 2 : 1024;
        HUSTTransaction *tmp = realloc(db->tx, next * sizeof(HUSTTransaction));
        if (!tmp) return -1;
        db->tx = tmp;
        db->tx_cap = next;
    }
    HUSTTransaction *t = &db->tx[db->tx_count++];
    t->items = malloc(count * sizeof(uint32_t));
    if (!t->items) return -1;
    memcpy(t->items, items, count * sizeof(uint32_t));
    t->count = count;
    t->itu = 0.0;
    db->raw_nnz += count;
    return 0;
}

// UTF-8 char parser matching Faro
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
// Custom direct UTF-8 / Whitespace initial tokenizer to avoid state sharing bugs
static int tokenize_text_direct(HUSTDB *db, const unsigned char *S, size_t B, uint32_t **stream_out, size_t *stream_len_out) {
    size_t cap = 65536;
    uint32_t *stream = malloc(cap * sizeof(uint32_t));
    size_t count = 0;
    
    // Hash table to deduplicate tokens
    typedef struct HashNode {
        char *word;
        uint32_t id;
        struct HashNode *next;
    } HashNode;
    size_t h_cap = 65536;
    HashNode **h_table = calloc(h_cap, sizeof(HashNode *));
    
    size_t i = 0;
    char token_buf[4096];
    size_t token_len = 0;
    
    while (i < B) {
        unsigned char c = S[i];
        if (isspace(c) || c == ',' || c == '.' || c == '?' || c == '!' || c == ';' || c == ':') {
            if (token_len > 0) {
                token_buf[token_len] = '\0';
                // Hash lookup
                uint64_t hash = 5381;
                for (size_t k = 0; k < token_len; k++) hash = ((hash << 5) + hash) + token_buf[k];
                size_t bucket = hash % h_cap;
                HashNode *node = h_table[bucket];
                uint32_t int_id = HUST_NO_RANK;
                while (node) {
                    if (strcmp(node->word, token_buf) == 0) {
                        int_id = node->id;
                        break;
                    }
                    node = node->next;
                }
                if (int_id == HUST_NO_RANK) {
                    if (add_token_bytes(db, token_buf, token_len, &int_id) != 0) {
                        free(stream);
                        return -1;
                    }
                    HashNode *new_node = malloc(sizeof(HashNode));
                    new_node->word = strdup(token_buf);
                    new_node->id = int_id;
                    new_node->next = h_table[bucket];
                    h_table[bucket] = new_node;
                }
                if (count >= cap) {
                    cap *= 2;
                    stream = realloc(stream, cap * sizeof(uint32_t));
                }
                stream[count++] = int_id;
                token_len = 0;
            }
            i++;
        } else if (c >= 0x80) {
            size_t char_len = 0;
            if (parse_utf8_char(S + i, B - i, &char_len)) {
                if (token_len + char_len < sizeof(token_buf) - 1) {
                    for (size_t k = 0; k < char_len; k++) {
                        token_buf[token_len++] = S[i + k];
                    }
                }
                i += char_len;
            } else {
                i++; // Skip invalid UTF-8
            }
        } else {
            // Fold to lowercase
            char folded = (c >= 'A' && c <= 'Z') ? (c + 32) : c;
            if (token_len < sizeof(token_buf) - 1) {
                token_buf[token_len++] = folded;
            }
            i++;
        }
    }
    
    if (token_len > 0) {
        token_buf[token_len] = '\0';
        uint64_t hash = 5381;
        for (size_t k = 0; k < token_len; k++) hash = ((hash << 5) + hash) + token_buf[k];
        size_t bucket = hash % h_cap;
        HashNode *node = h_table[bucket];
        uint32_t int_id = HUST_NO_RANK;
        while (node) {
            if (strcmp(node->word, token_buf) == 0) {
                int_id = node->id;
                break;
            }
            node = node->next;
        }
        if (int_id == HUST_NO_RANK) {
            if (add_token_bytes(db, token_buf, token_len, &int_id) != 0) {
                free(stream);
                return -1;
            }
        }
        if (count >= cap) {
            cap *= 2;
            stream = realloc(stream, cap * sizeof(uint32_t));
        }
        stream[count++] = int_id;
    }
    
    // Free hash table
    for (size_t k = 0; k < h_cap; k++) {
        HashNode *node = h_table[k];
        while (node) {
            HashNode *next = node->next;
            free(node->word);
            free(node);
            node = next;
        }
    }
    free(h_table);
    
    *stream_out = stream;
    *stream_len_out = count;
    return 0;
}

// Character level tokenizer
static int tokenize_char_direct(HUSTDB *db, const unsigned char *S, size_t B, uint32_t **stream_out, size_t *stream_len_out) {
    size_t cap = 65536;
    uint32_t *stream = malloc(cap * sizeof(uint32_t));
    size_t count = 0;
    
    uint32_t char_ids[256];
    for (int k = 0; k < 256; k++) char_ids[k] = HUST_NO_RANK;
    
    size_t i = 0;
    while (i < B) {
        size_t char_len = 1;
        unsigned char c = S[i];
        if (c >= 0x80) {
            if (!parse_utf8_char(S + i, B - i, &char_len)) {
                char_len = 1;
            }
        }
        
        // Form token name
        char name[16];
        memcpy(name, S + i, char_len);
        name[char_len] = '\0';
        
        uint32_t int_id = HUST_NO_RANK;
        if (char_len == 1 && char_ids[c] != HUST_NO_RANK) {
            int_id = char_ids[c];
        } else {
            // Find or create in DB
            for (size_t k = 0; k < db->token_count; k++) {
                if (strcmp(db->tokens[k].name, name) == 0) {
                    int_id = (uint32_t)k;
                    break;
                }
            }
            if (int_id == HUST_NO_RANK) {
                if (add_token_bytes(db, name, char_len, &int_id) != 0) {
                    free(stream);
                    return -1;
                }
                if (char_len == 1) char_ids[c] = int_id;
            }
        }
        
        if (count >= cap) {
            cap *= 2;
            stream = realloc(stream, cap * sizeof(uint32_t));
        }
        stream[count++] = int_id;
        i += char_len;
    }
    
    *stream_out = stream;
    *stream_len_out = count;
    return 0;
}

// Compute statistics and bounds matching section 3.2 & 3.4
static int compute_statistics(HUSTContext *ctx, uint32_t *stream, size_t stream_len, size_t L, size_t stride) {
    HUSTDB *db = ctx->db;
    
    // Form initial transactions
    size_t n = 0;
    if (stream_len >= L) {
        n = (stream_len - L) / stride + 1;
    }
    if (n == 0) return -1;
    
    db->tx_cap = n;
    db->tx = calloc(n, sizeof(HUSTTransaction));
    
    uint32_t *txn_buf = malloc(L * sizeof(uint32_t));
    for (size_t q = 0; q < n; q++) {
        size_t start = q * stride;
        size_t count = L;
        memcpy(txn_buf, stream + start, count * sizeof(uint32_t));
        if (!ctx->is_sequence_mode) {
            count = unique_sorted_ids(txn_buf, count);
        }
        if (add_transaction(db, txn_buf, count) != 0) {
            free(txn_buf);
            return -1;
        }
    }
    free(txn_buf);
    
    // Compute df(x)
    uint32_t *marks = calloc(db->token_count, sizeof(uint32_t));
    uint32_t stamp = 1;
    for (size_t tid = 0; tid < db->tx_count; tid++, stamp++) {
        HUSTTransaction *t = &db->tx[tid];
        for (size_t i = 0; i < t->count; i++) {
            uint32_t item = t->items[i];
            if (marks[item] != stamp) {
                marks[item] = stamp;
                db->tokens[item].df++;
            }
        }
    }
    
    // Compute I(x)
    double alpha = ctx->alpha;
    double denom = (double)db->tx_count + alpha * (double)db->token_count;
    double best_singleton_u = 0.0;
    for (size_t i = 0; i < db->token_count; i++) {
        double p = ((double)db->tokens[i].df + alpha) / denom;
        if (p <= 0.0) p = DBL_MIN;
        db->tokens[i].weight = -log(p) / log(2.0); // Self-information in bits
        
        // Track best singleton utility
        if (db->tokens[i].df >= ctx->sigma_0) {
            double u = (double)db->tokens[i].df * db->tokens[i].weight;
            if (u > best_singleton_u) best_singleton_u = u;
        }
    }
    
    // Set utility threshold theta
    if (ctx->theta <= 0.0) {
        ctx->theta = best_singleton_u * ctx->theta_ratio;
    }
    if (ctx->theta <= 0.0) ctx->theta = 1.0;
    
    // Compute ITU(T) and TIUB(x)
    memset(marks, 0, db->token_count * sizeof(uint32_t));
    stamp = 1;
    for (size_t tid = 0; tid < db->tx_count; tid++, stamp++) {
        HUSTTransaction *t = &db->tx[tid];
        double itu = 0.0;
        for (size_t i = 0; i < t->count; i++) {
            uint32_t item = t->items[i];
            if (marks[item] != stamp) {
                marks[item] = stamp;
                itu += db->tokens[item].weight;
            }
        }
        t->itu = itu;
        
        for (size_t i = 0; i < t->count; i++) {
            uint32_t item = t->items[i];
            if (marks[item] == stamp) {
                db->tokens[item].tiub += itu;
                marks[item] = stamp + 1;
            }
        }
    }
    free(marks);
    
    return 0;
}

// Prepare survivors
static int prepare_survivors(HUSTContext *ctx) {
    HUSTDB *db = ctx->db;
    uint32_t *surv = malloc(db->token_count * sizeof(uint32_t));
    if (!surv) return -1;
    
    for (size_t i = 0; i < db->token_count; i++) {
        if (db->tokens[i].df < ctx->sigma_0) {
            if (db->tokens[i].df > 0) ctx->pruned_support++;
            continue;
        }
        if (db->tokens[i].tiub < ctx->theta) {
            ctx->pruned_tiub++;
            continue;
        }
        db->tokens[i].survives = 1;
        surv[db->survivor_count++] = (uint32_t)i;
    }
    
    // Sort survivors by ascending TIUB, then descending weight
    g_order_db = db;
    qsort(surv, db->survivor_count, sizeof(uint32_t), token_order_cmp);
    g_order_db = NULL;
    
    db->rank_to_item = malloc(db->survivor_count * sizeof(uint32_t));
    db->item_to_rank = malloc(db->token_count * sizeof(uint32_t));
    for (size_t i = 0; i < db->token_count; i++) db->item_to_rank[i] = HUST_NO_RANK;
    
    for (size_t r = 0; r < db->survivor_count; r++) {
        uint32_t item = surv[r];
        db->rank_to_item[r] = item;
        db->item_to_rank[item] = (uint32_t)r;
        db->tokens[item].rank = (uint32_t)r;
    }
    free(surv);
    return 0;
}

// Project transactions
static int project_transactions(HUSTContext *ctx) {
    HUSTDB *db = ctx->db;
    g_rank_map = db->item_to_rank;
    for (size_t tid = 0; tid < db->tx_count; tid++) {
        HUSTTransaction *t = &db->tx[tid];
        size_t out = 0;
        for (size_t i = 0; i < t->count; i++) {
            uint32_t item = t->items[i];
            if (item < db->token_count && db->item_to_rank[item] != HUST_NO_RANK) {
                t->items[out++] = item;
            }
        }
        t->count = out;
        if (!ctx->is_sequence_mode && t->count > 1) {
            qsort(t->items, t->count, sizeof(uint32_t), rank_order_cmp);
            size_t unique = 1;
            for (size_t i = 1; i < t->count; i++) {
                if (t->items[i] != t->items[unique - 1]) t->items[unique++] = t->items[i];
            }
            t->count = unique;
        }
    }
    g_rank_map = NULL;
    return 0;
}

static double tail_ru_itemset(const HUSTDB *db, const HUSTTransaction *t, uint32_t pos) {
    double sum = 0.0;
    for (size_t i = (size_t)pos + 1; i < t->count; i++) {
        sum += db->tokens[t->items[i]].weight;
    }
    return sum;
}

static double tail_ru_sequence(HUSTContext *ctx, const HUSTTransaction *t, uint32_t pos, uint32_t min_rank) {
    double sum = 0.0;
    ctx->tail_stamp++;
    if (ctx->tail_stamp == 0) {
        memset(ctx->tail_marks, 0, ctx->db->token_count * sizeof(uint32_t));
        ctx->tail_stamp = 1;
    }
    for (size_t i = (size_t)pos + 1; i < t->count; i++) {
        uint32_t item = t->items[i];
        uint32_t rank = ctx->db->item_to_rank[item];
        if (rank <= min_rank) continue;
        if (ctx->tail_marks[item] == ctx->tail_stamp) continue;
        ctx->tail_marks[item] = ctx->tail_stamp;
        sum += ctx->db->tokens[item].weight;
    }
    return sum;
}

// Build singletons
static int build_singletons(HUSTContext *ctx) {
    HUSTDB *db = ctx->db;
    ctx->singletons = calloc(db->survivor_count, sizeof(HUSTList));
    ctx->tail_marks = calloc(db->token_count ? db->token_count : 1, sizeof(uint32_t));
    ctx->tail_stamp = 1;
    
    uint32_t *seen = calloc(db->token_count ? db->token_count : 1, sizeof(uint32_t));
    uint32_t stamp = 1;
    
    for (size_t r = 0; r < db->survivor_count; r++) {
        uint32_t item = db->rank_to_item[r];
        ctx->singletons[r].item = item;
        ctx->singletons[r].pwt = db->tokens[item].weight;
    }
    
    for (size_t tid = 0; tid < db->tx_count; tid++, stamp++) {
        HUSTTransaction *t = &db->tx[tid];
        if (!ctx->is_sequence_mode) {
            for (uint32_t p = 0; p < t->count; p++) {
                uint32_t item = t->items[p];
                uint32_t rank = db->item_to_rank[item];
                HUSTEntry e;
                e.tid = (uint32_t)tid;
                e.last = p;
                e.span = 1;
                e.eu = db->tokens[item].weight;
                e.ru = tail_ru_itemset(db, t, p);
                if (list_push(&ctx->singletons[rank], e) != 0) {
                    free(seen);
                    return -1;
                }
            }
        } else {
            for (uint32_t p = 0; p < t->count; p++) {
                uint32_t item = t->items[p];
                if (seen[item] == stamp) continue;
                seen[item] = stamp;
                uint32_t rank = db->item_to_rank[item];
                HUSTEntry e;
                e.tid = (uint32_t)tid;
                e.last = p;
                e.span = 1;
                e.eu = db->tokens[item].weight;
                e.ru = tail_ru_sequence(ctx, t, p, rank);
                if (list_push(&ctx->singletons[rank], e) != 0) {
                    free(seen);
                    return -1;
                }
            }
        }
    }
    free(seen);
    return 0;
}

static int find_after_sequence(const HUSTTransaction *t, uint32_t pos, uint32_t item, uint32_t *found) {
    for (uint32_t p = pos + 1; p < t->count; p++) {
        if (t->items[p] == item) {
            *found = p;
            return 1;
        }
    }
    return 0;
}

static int join_hust(HUSTContext *ctx, const HUSTList *prefix, uint32_t ext_rank, HUSTList *out) {
    memset(out, 0, sizeof(*out));
    HUSTDB *db = ctx->db;
    uint32_t ext_item = db->rank_to_item[ext_rank];
    HUSTList *single = &ctx->singletons[ext_rank];
    out->item = ext_item;
    out->pwt = prefix->pwt + db->tokens[ext_item].weight;
    ctx->joins++;
    
    size_t i = 0, j = 0;
    while (i < prefix->count && j < single->count) {
        uint32_t ti = prefix->entries[i].tid;
        uint32_t tj = single->entries[j].tid;
        if (ti < tj) {
            i++;
            continue;
        }
        if (tj < ti) {
            j++;
            continue;
        }
        HUSTTransaction *t = &db->tx[ti];
        HUSTEntry e;
        memset(&e, 0, sizeof(e));
        e.tid = ti;
        
        if (!ctx->is_sequence_mode) {
            e.last = single->entries[j].last;
            e.span = prefix->entries[i].span + 1;
            e.eu = out->pwt;
            e.ru = tail_ru_itemset(db, t, e.last);
            if (list_push(out, e) != 0) return -1;
        } else {
            uint32_t pos;
            if (find_after_sequence(t, prefix->entries[i].last, ext_item, &pos)) {
                e.last = pos;
                e.span = pos - prefix->entries[i].last + prefix->entries[i].span;
                e.eu = out->pwt;
                e.ru = tail_ru_sequence(ctx, t, pos, ext_rank);
                if (list_push(out, e) != 0) return -1;
            }
        }
        i++;
        j++;
    }
    ctx->joined_entries += out->count;
    return 0;
}

static int add_discovered_pattern(HUSTContext *ctx, const uint32_t *pattern, size_t len, double utility, size_t support) {
    if (ctx->pattern_count >= ctx->pattern_cap) {
        ctx->pattern_cap = ctx->pattern_cap ? ctx->pattern_cap * 2 : 1024;
        HUSTPattern *tmp = realloc(ctx->patterns, ctx->pattern_cap * sizeof(HUSTPattern));
        if (!tmp) return -1;
        ctx->patterns = tmp;
    }
    HUSTPattern *p = &ctx->patterns[ctx->pattern_count++];
    p->tokens = malloc(len * sizeof(uint32_t));
    memcpy(p->tokens, pattern, len * sizeof(uint32_t));
    p->len = len;
    p->utility = utility;
    p->support = support;
    return 0;
}

static int emit_pattern(HUSTContext *ctx, const uint32_t *pattern, size_t len, size_t support, double utility, double pwt) {
    if (add_discovered_pattern(ctx, pattern, len, utility, support) != 0) return -1;
    if (ctx->pattern_out) {
        fprintf(ctx->pattern_out, "%.6f\t%zu\t%zu\t%.6f\t", utility, support, len, pwt);
        for (size_t i = 0; i < len; i++) {
            if (i) fputc(' ', ctx->pattern_out);
            fputs(ctx->db->tokens[pattern[i]].name, ctx->pattern_out);
        }
        fputc('\n', ctx->pattern_out);
    }
    return 0;
}

// DFS-Merge Search Algorithm (Algorithm 2 & 3)
static int dfs_hust(HUSTContext *ctx, const uint32_t *pattern, size_t depth, HUSTList *list, size_t suffix_start) {
    if (ctx->max_depth && depth > ctx->max_depth) return 0;
    if (ctx->max_patterns && ctx->pattern_count >= ctx->max_patterns) return 0;
    ctx->visited_nodes++;
    
    // Check minsup
    if (list->count < ctx->sigma_0) {
        ctx->pruned_support++;
        return 0;
    }
    
    // Score list
    double utility = 0.0;
    double iwru = 0.0;
    for (size_t i = 0; i < list->count; i++) {
        utility += list->entries[i].eu;
        iwru += list->pwt + list->entries[i].ru;
    }
    
    if (utility >= ctx->theta) {
        if (emit_pattern(ctx, pattern, depth, list->count, utility, list->pwt) != 0) return -1;
    }
    
    if (iwru < ctx->theta) {
        ctx->pruned_iwru++;
        return 0;
    }
    
    for (size_t r = suffix_start; r < ctx->db->survivor_count; r++) {
        uint32_t *child_pattern = malloc((depth + 1) * sizeof(uint32_t));
        if (!child_pattern) return -1;
        memcpy(child_pattern, pattern, depth * sizeof(uint32_t));
        child_pattern[depth] = ctx->db->rank_to_item[r];
        
        HUSTList child;
        if (join_hust(ctx, list, (uint32_t)r, &child) != 0) {
            free(child_pattern);
            return -1;
        }
        
        if (child.count >= ctx->sigma_0) {
            if (dfs_hust(ctx, child_pattern, depth + 1, &child, r + 1) != 0) {
                list_free(&child);
                free(child_pattern);
                return -1;
            }
        } else if (child.count > 0) {
            ctx->pruned_support++;
        }
        list_free(&child);
        free(child_pattern);
    }
    return 0;
}

// Sorting patterns by descending length, then descending utility
static int pattern_sort_cmp(const void *a, const void *b) {
    const HUSTPattern *pa = (const HUSTPattern *)a;
    const HUSTPattern *pb = (const HUSTPattern *)b;
    if (pa->len > pb->len) return -1;
    if (pa->len < pb->len) return 1;
    if (pa->utility > pb->utility) return -1;
    if (pa->utility < pb->utility) return 1;
    return 0;
}

// Clean resources in context
static void context_free(HUSTContext *ctx) {
    if (ctx->singletons) {
        for (size_t i = 0; i < ctx->db->survivor_count; i++) {
            list_free(&ctx->singletons[i]);
        }
        free(ctx->singletons);
    }
    free(ctx->tail_marks);
    if (ctx->patterns) {
        for (size_t i = 0; i < ctx->pattern_count; i++) {
            free(ctx->patterns[i].tokens);
        }
        free(ctx->patterns);
    }
}

static long get_peak_rss_kb(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc))) {
        return (long)(pmc.PeakWorkingSetSize / 1024);
    }
    return 0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return usage.ru_maxrss;
    }
    return 0;
#endif
}

int main(int argc, char **argv) {
    char *input_path = NULL;
    char *output_path = NULL;
    char *patterns_path = NULL;
    size_t L = 64;
    size_t stride = 64;
    double theta = -1.0;
    double theta_ratio = 0.25;
    size_t sigma_0 = 2;
    double alpha = 1.0;
    int is_sequence_mode = 0;
    int char_level = 0;
    int run_benchmark = 0;
    int filter_noise = 1;
    size_t max_depth = 3;
    size_t max_patterns = 25000;
    
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "-input") == 0) && i + 1 < argc) {
            input_path = argv[++i];
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "-output") == 0) && i + 1 < argc) {
            output_path = argv[++i];
        } else if ((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "-patterns") == 0) && i + 1 < argc) {
            patterns_path = argv[++i];
        } else if ((strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "-L") == 0) && i + 1 < argc) {
            L = (size_t)atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "-d") == 0) && i + 1 < argc) {
            stride = (size_t)atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-u") == 0 || strcmp(argv[i], "-theta") == 0) && i + 1 < argc) {
            theta = atof(argv[++i]);
        } else if (strcmp(argv[i], "-theta_ratio") == 0 && i + 1 < argc) {
            theta_ratio = atof(argv[++i]);
        } else if ((strcmp(argv[i], "-s0") == 0 || strcmp(argv[i], "-minsup") == 0) && i + 1 < argc) {
            sigma_0 = (size_t)atoi(argv[++i]);
        } else if (strcmp(argv[i], "-alpha") == 0 && i + 1 < argc) {
            alpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            char *mode = argv[++i];
            if (strcmp(mode, "sequence") == 0) is_sequence_mode = 1;
        } else if (strcmp(argv[i], "--char") == 0) {
            char_level = 1;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            run_benchmark = 1;
        } else if (strcmp(argv[i], "--no-filter") == 0) {
            filter_noise = 0;
        } else if ((strcmp(argv[i], "-max_depth") == 0 || strcmp(argv[i], "-depth") == 0 || strcmp(argv[i], "--max-depth") == 0) && i + 1 < argc) {
            max_depth = (size_t)atoi(argv[++i]);
        } else if ((strcmp(argv[i], "-max_patterns") == 0 || strcmp(argv[i], "--max-patterns") == 0) && i + 1 < argc) {
            max_patterns = (size_t)atoi(argv[++i]);
        }
    }
    
    if (!input_path) {
        fprintf(stderr, "Usage: %s -input <file> [-output <file>] [-patterns <file>] [-L <win>] [-d <stride>] [-theta <val>] [-theta_ratio <val>] [-minsup <val>] [--mode <itemset|sequence>] [--char] [--benchmark]\n", argv[0]);
        return 1;
    }
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    size_t file_size = 0;
    unsigned char *mapped_data = read_input_file(input_path, &file_size);
    if (!mapped_data) {
        return 1;
    }
    
    HUSTDB db;
    db_init(&db);
    db.input_bytes = file_size;
    
    uint32_t *token_stream = NULL;
    size_t stream_len = 0;
    
    if (char_level) {
        if (tokenize_char_direct(&db, mapped_data, file_size, &token_stream, &stream_len) != 0) {
            fprintf(stderr, "Failed character tokenization\n");
            free(mapped_data);
            db_free(&db);
            return 1;
        }
    } else {
        if (tokenize_text_direct(&db, mapped_data, file_size, &token_stream, &stream_len) != 0) {
            fprintf(stderr, "Failed word tokenization\n");
            free(mapped_data);
            db_free(&db);
            return 1;
        }
    }
    
    free(mapped_data);
    
    HUSTContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.db = &db;
    ctx.theta = theta;
    ctx.theta_ratio = theta_ratio;
    ctx.sigma_0 = sigma_0;
    ctx.alpha = alpha;
    ctx.is_sequence_mode = is_sequence_mode;
    ctx.filter_noise = filter_noise;
    ctx.max_depth = max_depth;
    ctx.max_patterns = max_patterns;
    ctx.started = clock();
    
    if (patterns_path) {
        ctx.pattern_out = fopen(patterns_path, "w");
        if (!ctx.pattern_out) {
            perror("Failed to open pattern output file");
            free(token_stream);
            db_free(&db);
            return 1;
        }
    }
    
    // Compute statistics & weights
    if (compute_statistics(&ctx, token_stream, stream_len, L, stride) != 0) {
        fprintf(stderr, "Failed to compute statistics (perhaps file too small for L)\n");
        if (ctx.pattern_out) fclose(ctx.pattern_out);
        free(token_stream);
        db_free(&db);
        return 1;
    }
    
    // Prepare survivors
    if (prepare_survivors(&ctx) != 0) {
        fprintf(stderr, "Failed to prepare survivors\n");
        if (ctx.pattern_out) fclose(ctx.pattern_out);
        free(token_stream);
        db_free(&db);
        return 1;
    }
    
    // Project transactions
    if (project_transactions(&ctx) != 0) {
        fprintf(stderr, "Failed to project transactions\n");
        if (ctx.pattern_out) fclose(ctx.pattern_out);
        free(token_stream);
        db_free(&db);
        return 1;
    }
    
    // Build singletons
    if (build_singletons(&ctx) != 0) {
        fprintf(stderr, "Failed to build singletons\n");
        if (ctx.pattern_out) fclose(ctx.pattern_out);
        free(token_stream);
        db_free(&db);
        return 1;
    }
    
    // Run DFS-Merge
    for (size_t r = 0; r < db.survivor_count; r++) {
        uint32_t item = db.rank_to_item[r];
        uint32_t pattern = item;
        HUSTList *list = &ctx.singletons[r];
        if (list->count >= ctx.sigma_0) {
            dfs_hust(&ctx, &pattern, 1, list, r + 1);
        }
    }
    
    // Close pattern file if open
    if (ctx.pattern_out) {
        fclose(ctx.pattern_out);
    }
    
    // Perform greedy token merging
    // Only multi-token patterns (length >= 2) can be merged
    HUSTPattern *merge_patterns = malloc(ctx.pattern_count * sizeof(HUSTPattern));
    size_t merge_patterns_count = 0;
    for (size_t i = 0; i < ctx.pattern_count; i++) {
        if (ctx.patterns[i].len >= 2) {
            merge_patterns[merge_patterns_count++] = ctx.patterns[i];
        }
    }
    
    // Sort merge candidates by length descending, then utility descending
    qsort(merge_patterns, merge_patterns_count, sizeof(HUSTPattern), pattern_sort_cmp);
    
    // Allocate space for token text lookup map
    size_t max_tokens_map = db.token_count + merge_patterns_count + 1;
    char **token_text_map = calloc(max_tokens_map, sizeof(char *));
    for (size_t i = 0; i < db.token_count; i++) {
        token_text_map[i] = strdup(db.tokens[i].name);
    }
    
    // Greedy merging in the token stream
    // Assign new IDs for merged tokens
    uint32_t merged_vocab_start = (uint32_t)db.token_count;
    
    
    uint32_t next_merged_id = merged_vocab_start;
    
    for (size_t i = 0; i < merge_patterns_count; i++) {
        HUSTPattern *p = &merge_patterns[i];
        
        // Construct merged token name
        size_t name_len = 0;
        for (size_t k = 0; k < p->len; k++) {
            name_len += strlen(token_text_map[p->tokens[k]]) + 1;
        }
        char *merged_name = malloc(name_len + 1);
        merged_name[0] = '\0';
        for (size_t k = 0; k < p->len; k++) {
            if (k > 0) strcat(merged_name, "_");
            strcat(merged_name, token_text_map[p->tokens[k]]);
        }
        
        uint32_t m_id = next_merged_id++;
        token_text_map[m_id] = merged_name;
        
        // Scan stream and apply merge
        size_t write_pos = 0;
        size_t read_pos = 0;
        while (read_pos < stream_len) {
            int match = 0;
            if (read_pos + p->len <= stream_len) {
                match = 1;
                for (size_t k = 0; k < p->len; k++) {
                    if (token_stream[read_pos + k] != p->tokens[k]) {
                        match = 0;
                        break;
                    }
                }
            }
            if (match) {
                token_stream[write_pos++] = m_id;
                read_pos += p->len;
            } else {
                token_stream[write_pos++] = token_stream[read_pos++];
            }
        }
        stream_len = write_pos;
    }
    
    // Build final output transactions
    FILE *out = stdout;
    if (output_path) {
        out = fopen(output_path, "w");
        if (!out) {
            perror("Failed to open output file");
            out = stdout;
        }
    }
    
    size_t final_tx_count = 0;
    if (stream_len >= L) {
        final_tx_count = (stream_len - L) / stride + 1;
    }
    
    uint32_t *out_txn = malloc(L * sizeof(uint32_t));
    for (size_t q = 0; q < final_tx_count; q++) {
        size_t start = q * stride;
        size_t count = L;
        memcpy(out_txn, token_stream + start, count * sizeof(uint32_t));
        
        // In itemset mode, sort and unique
        if (!is_sequence_mode) {
            count = unique_sorted_ids(out_txn, count);
        }
        
        // Print transaction to output
        size_t print_count = 0;
        for (size_t i = 0; i < count; i++) {
            uint32_t t_id = out_txn[i];
            
            // Noise filtering: check if token was pruned
            if (ctx.filter_noise) {
                if (t_id < db.token_count && !db.tokens[t_id].survives) {
                    continue; // Skip pruned singleton
                }
            }
            
            fprintf(out, "%s%u", (print_count == 0) ? "" : " ", t_id);
            print_count++;
        }
        // Even if empty, output newline
        fputc('\n', out);
    }
    free(out_txn);
    
    if (output_path && out != stdout) {
        fclose(out);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_sec = (end_time.tv_sec - start_time.tv_sec) + 
                         (end_time.tv_nsec - start_time.tv_nsec) / 1000000000.0;
                         
    if (run_benchmark) {
        long peak_rss = get_peak_rss_kb();
        double mib_processed = (double)file_size / (1024.0 * 1024.0);
        double throughput = mib_processed / elapsed_sec;
        double reduction = (double)stream_len / (double)(file_size ? file_size : 1);
        
        printf("HUST-Tokenize Benchmark Results:\n");
        printf("Throughput: %.2f MiB/s\n", throughput);
        printf("Elapsed Time: %.4f s\n", elapsed_sec);
        printf("Peak RSS: %ld KB\n", peak_rss);
        printf("Initial Vocabulary: %zu\n", db.token_count);
        printf("Active Vocabulary (Survivors + Merged): %zu\n", db.survivor_count + merge_patterns_count);
        printf("Discovered Patterns: %zu\n", ctx.pattern_count);
        printf("Token Count Reduction Ratio: %.4f\n", reduction);
        printf("Visited Nodes: %zu\n", ctx.visited_nodes);
        printf("Joins: %zu\n", ctx.joins);
    }
    
    // Clean up
    for (size_t i = 0; i < max_tokens_map; i++) {
        free(token_text_map[i]);
    }
    free(token_text_map);
    free(merge_patterns);
    free(token_stream);
    context_free(&ctx);
    db_free(&db);
    
    return 0;
}
