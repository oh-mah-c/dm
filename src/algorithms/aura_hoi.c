#include "algorithms/aura_hoi.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"
#include "core/dm_threadpool.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Transaction Metadata for Length Sorting ─────────────────────────────── */
typedef struct {
    uint32_t orig_tid;
    uint32_t len;
} TransMeta;

static int cmp_trans_meta(const void *a, const void *b) {
    const TransMeta *ta = (const TransMeta *)a;
    const TransMeta *tb = (const TransMeta *)b;
    if (ta->len < tb->len) return -1;
    if (ta->len > tb->len) return 1;
    return 0;
}

/* ── TID Itemset representation (Equivalence Class Node) ─────────────────── */
typedef struct {
    uint32_t item;          /* active item index (for closed & tracking) */
    uint32_t length;        /* pattern size |X| */
    uint32_t *tids;         /* sorted transaction IDs */
    size_t num_tids;        /* support count */
    double recip_sum;       /* pre-calculated \sum_{q \in T(X)} 1 / |t_q| */
} AURATidNode;

/* ── Memory Pool for TID arrays (Avoid per-node malloc/free in DFS) ──────── */
typedef struct {
    uint32_t *pool;
    size_t capacity;
    size_t used;
    size_t high_water;
} AURAMemPool;

static int mempool_init(AURAMemPool *p, size_t cap_elements) {
    p->pool = (uint32_t *)malloc(cap_elements * sizeof(uint32_t));
    if (!p->pool) return -1;
    p->capacity = cap_elements;
    p->used = 0;
    p->high_water = 0;
    return 0;
}

static void mempool_free(AURAMemPool *p) {
    if (p->pool) free(p->pool);
    memset(p, 0, sizeof(*p));
}

static inline size_t mempool_mark(AURAMemPool *p) {
    return p->used;
}

static inline uint32_t *mempool_alloc(AURAMemPool *p, size_t count) {
    if (p->used + count > p->capacity) return NULL;
    uint32_t *ptr = p->pool + p->used;
    p->used += count;
    if (p->used > p->high_water) p->high_water = p->used;
    return ptr;
}

static inline void mempool_rewind(AURAMemPool *p, size_t mark) {
    if (mark <= p->used) p->used = mark;
}

/* ── Auditable Closed Ledger with 64-bit Hash Fingerprint & TID-List Verification */
typedef struct {
    uint64_t h1;
    uint64_t h2;
    size_t support;
    double occupancy;
    uint32_t *tids;
    uint32_t *items;
    size_t len;
} AURALedgerEntry;

typedef struct {
    AURALedgerEntry *data;
    size_t count;
    size_t cap;
    size_t *hash_table; /* 1-based index into data; 0 means empty slot */
    size_t table_cap;
} AURALedger;

static int ledger_init(AURALedger *l, size_t initial_cap) {
    l->count = 0;
    l->cap = initial_cap ? initial_cap : 256;
    l->data = (AURALedgerEntry *)malloc(l->cap * sizeof(AURALedgerEntry));
    if (!l->data) return -1;
    l->table_cap = initial_cap ? initial_cap * 2 : 512;
    l->hash_table = (size_t *)calloc(l->table_cap, sizeof(size_t));
    if (!l->hash_table) {
        free(l->data);
        return -1;
    }
    return 0;
}

static void ledger_free(AURALedger *l) {
    for (size_t i = 0; i < l->count; i++) {
        free(l->data[i].tids);
        free(l->data[i].items);
    }
    free(l->data);
    free(l->hash_table);
    memset(l, 0, sizeof(*l));
}

static inline void tid_hash(const uint32_t *tids, size_t n, uint64_t *h1, uint64_t *h2) {
    uint64_t a = 1469598103934665603ULL;
    uint64_t b = 1099511628211ULL ^ (uint64_t)n;
    for (size_t i = 0; i < n; i++) {
        uint64_t x = tids[i];
        a ^= x;
        a *= 1099511628211ULL;
        b ^= x + 0x9e3779b97f4a7c15ULL + (b << 6) + (b >> 2);
    }
    *h1 = a;
    *h2 = b;
}

static int ledger_has(AURALedger *l, const uint32_t *tids, size_t supp, uint64_t h1, uint64_t h2) {
    if (!l->table_cap) return 0;
    size_t mask = l->table_cap - 1;
    size_t idx = (h1 ^ (h2 * 0x9e3779b97f4a7c15ULL)) & mask;
    size_t step = ((h2 << 1) | 1ULL) & mask;
    if (!step) step = 1;

    for (size_t probe = 0; probe < l->table_cap; probe++) {
        size_t entry_idx = l->hash_table[idx];
        if (entry_idx == 0) return 0; /* not found */
        AURALedgerEntry *e = &l->data[entry_idx - 1];
        if (e->h1 == h1 && e->h2 == h2 && e->support == supp) {
            /* Full TID array comparison on fingerprint match */
            if (memcmp(e->tids, tids, supp * sizeof(uint32_t)) == 0) {
                return 1;
            }
        }
        idx = (idx + step) & mask;
    }
    return 0;
}

static int ledger_rehash(AURALedger *l) {
    size_t new_cap = l->table_cap * 2;
    size_t *new_table = (size_t *)calloc(new_cap, sizeof(size_t));
    if (!new_table) return -1;
    size_t mask = new_cap - 1;

    for (size_t i = 0; i < l->count; i++) {
        AURALedgerEntry *e = &l->data[i];
        size_t idx = (e->h1 ^ (e->h2 * 0x9e3779b97f4a7c15ULL)) & mask;
        size_t step = ((e->h2 << 1) | 1ULL) & mask;
        if (!step) step = 1;
        while (new_table[idx] != 0) {
            idx = (idx + step) & mask;
        }
        new_table[idx] = i + 1;
    }
    free(l->hash_table);
    l->hash_table = new_table;
    l->table_cap = new_cap;
    return 0;
}

static int ledger_add(AURALedger *l, const uint32_t *items, size_t len,
                      const uint32_t *tids, size_t supp, double occupancy) {
    uint64_t h1, h2;
    tid_hash(tids, supp, &h1, &h2);
    if (ledger_has(l, tids, supp, h1, h2)) {
        return 0; /* Duplicate support class */
    }

    if (l->count >= l->cap) {
        size_t nc = l->cap * 2;
        AURALedgerEntry *nd = (AURALedgerEntry *)realloc(l->data, nc * sizeof(*nd));
        if (!nd) return -1;
        l->data = nd;
        l->cap = nc;
    }

    if (l->count * 2 >= l->table_cap) {
        if (ledger_rehash(l) != 0) return -1;
    }

    size_t entry_index = l->count++;
    AURALedgerEntry *e = &l->data[entry_index];
    e->h1 = h1;
    e->h2 = h2;
    e->support = supp;
    e->occupancy = occupancy;
    e->len = len;
    e->tids = (uint32_t *)malloc(supp * sizeof(uint32_t));
    e->items = (uint32_t *)malloc(len * sizeof(uint32_t));
    if (!e->tids || !e->items) return -1;
    memcpy(e->tids, tids, supp * sizeof(uint32_t));
    memcpy(e->items, items, len * sizeof(uint32_t));

    size_t mask = l->table_cap - 1;
    size_t idx = (h1 ^ (h2 * 0x9e3779b97f4a7c15ULL)) & mask;
    size_t step = ((h2 << 1) | 1ULL) & mask;
    if (!step) step = 1;
    while (l->hash_table[idx] != 0) {
        idx = (idx + step) & mask;
    }
    l->hash_table[idx] = entry_index + 1;
    return 1;
}

/* ── Context for Mining ─────────────────────────────────────────────────── */
typedef struct {
    DM_Trans_Simple *trans;
    size_t ntrans;
    uint32_t max_id;
    
    /* Active candidate items (frequent 1-items) */
    uint32_t *active_items;
    size_t active_count;
    
    /* Support representations (pure TID-lists) */
    uint32_t **item_tids;     /* Sorted TID lists per active item */
    uint32_t *item_counts;    /* Support count per active item */
    double *item_recip_sums;  /* Reciprocal sum per active item */
    
    /* Transaction length structures */
    uint32_t *g_tsize;        /* Transaction length sorted by TID */
    double *recip_table;      /* 1.0 / len indexed by transaction length */
    uint32_t max_tsize;       /* Maximum transaction length */
    int uniform_length;       /* 1 if all transactions have identical length */
    uint32_t first_len;       /* Length when uniform_length == 1 */
    
    /* Thresholds */
    double min_occupancy;
    double threshold_value;   /* xi threshold */
    size_t min_support;
    int summed_occupancy_mode;
    int emit_raw_view;
    size_t max_patterns;
    double max_seconds;
    size_t top_k;
    clock_t start_clock;
    
    /* Statistics */
    size_t raw_hoi_count;
    size_t raw_total_output_items;
    size_t total_output_items;
    size_t visited_nodes;
    size_t pruned_support;
    size_t pruned_envelope;
    size_t pruned_backward;
    size_t closure_jumps;
    size_t ledger_duplicates;
    size_t topk_updates;
    int limited;
    
    /* Memory Pool & Pre-allocated Recursion Scratchpads */
    AURAMemPool pool;
    AURALedger ledger;
    uint32_t *tid_scratch;    /* size ntrans */
    
    /* Dynamic item prefixes for DFS paths */
    uint32_t *prefix_items;
} AURACtx;

static inline double elapsed_sec(const AURACtx *ctx) {
    return (double)(clock() - ctx->start_clock) / (double)CLOCKS_PER_SEC;
}

static inline int should_stop(AURACtx *ctx) {
    if (ctx->max_patterns && ctx->ledger.count >= ctx->max_patterns) {
        ctx->limited = 1;
        return 1;
    }
    if (ctx->max_seconds > 0.0 && elapsed_sec(ctx) >= ctx->max_seconds) {
        ctx->limited = 1;
        return 1;
    }
    return 0;
}

/* ── Simultaneous TID Intersection with Cumulative UBO & Score ───────────── */
/* Intersects P1->tids and P2->tids into out_tids, computing support, recip_sum, 
 * and testing UBO condition in a single linear pass over the common elements. */
static inline int intersect_tids_and_check(const AURACtx *ctx,
                                           const AURATidNode *P1, const AURATidNode *P2,
                                           uint32_t *out_tids, size_t *out_size,
                                           size_t new_k, size_t R_count,
                                           int *is_ho, double *out_recip_sum) {
    (void)R_count;
    const uint32_t *t1 = P1->tids;
    const uint32_t *t2 = P2->tids;
    size_t n1 = P1->num_tids;
    size_t n2 = P2->num_tids;
    
    size_t p1 = 0, p2 = 0, count = 0;
    
    /* Fast two-pointer intersection */
    while (p1 < n1 && p2 < n2) {
        if (t1[p1] < t2[p2]) {
            p1++;
        } else if (t1[p1] > t2[p2]) {
            p2++;
        } else {
            out_tids[count++] = t1[p1];
            p1++;
            p2++;
        }
    }
    
    *out_size = count;
    if (count < ctx->min_support) return 0;
    
    /* Uniform length fast-path (O(1)) */
    if (ctx->uniform_length) {
        double o_val = ((double)new_k * (double)count) / (double)ctx->first_len;
        *is_ho = (o_val >= ctx->threshold_value);
        *out_recip_sum = (double)count / (double)ctx->first_len;
        return 1;
    }
    
    /* Heterogeneous length: cumulative pass from largest transaction down */
    double current_sum = 0.0;
    double max_ubo = 0.0;
    const uint32_t *g_tsize = ctx->g_tsize;
    const double *recip_table = ctx->recip_table;
    uint32_t max_desc_len = (uint32_t)(new_k + R_count);
    
    for (int i = (int)count - 1; i >= 0; i--) {
        uint32_t tid = out_tids[i];
        uint32_t tsize = g_tsize[tid];
        current_sum += recip_table[tsize];
        
        if (i == 0 || g_tsize[out_tids[i - 1]] < tsize) {
            uint32_t eff_tsize = (tsize < max_desc_len) ? tsize : max_desc_len;
            double current_ubo = (double)eff_tsize * current_sum;
            if (current_ubo > max_ubo) {
                max_ubo = current_ubo;
            }
        }
    }
    
    if (max_ubo >= ctx->threshold_value) {
        *is_ho = (((double)new_k * current_sum) >= ctx->threshold_value);
        *out_recip_sum = current_sum;
        return 1;
    }
    return 0;
}

/* Forward declarations */
static void aura_dfs_tid_raw_engine(AURACtx *ctx, AURATidNode *classes, size_t class_size);
static void aura_dfs_tid_closed_engine(AURACtx *ctx, size_t depth, AURATidNode *classes, size_t class_size);

/* ── Pure TID-List Raw Fullset DFS Engine ────────────────────────────────── */
static void aura_process_root_item_raw(AURACtx *ctx, AURATidNode *classes, size_t class_size, size_t i) {
    if (should_stop(ctx)) return;
    AURATidNode *P1 = &classes[i];
    size_t remaining = class_size - i - 1;
    if (remaining == 0) return;
    
    /* ── Mathematically Safe Residual Upper Bound Pruning ───────────────
     * Theorem: For any descendant X = P1 \cup Z with Z \subseteq {classes[i+1..class_size-1]}:
     * |X| \le P1->length + remaining.
     * Furthermore, |X| \le |t_q| for any supporting transaction q \in T(X).
     * Since T(X) \subseteq T(P1), we have:
     * socc(X) \le \sum_{q \in T(P1)} min(P1->length + remaining, |t_q|) / |t_q|
     *         \le (P1->length + remaining) * P1->recip_sum.
     * If this upper bound < threshold_value, NO valid HOI descendant can exist! */
    double max_possible_k = (double)(P1->length + remaining);
    if (ctx->uniform_length) {
        if (max_possible_k > (double)ctx->first_len) max_possible_k = (double)ctx->first_len;
        double max_desc_score = (max_possible_k * (double)P1->num_tids) / (double)ctx->first_len;
        if (max_desc_score + 1e-12 < ctx->threshold_value) {
            ctx->pruned_envelope += remaining;
            return;
        }
    } else {
        double max_desc_score = max_possible_k * P1->recip_sum;
        if (max_desc_score + 1e-12 < ctx->threshold_value || (double)P1->num_tids + 1e-12 < ctx->threshold_value) {
            ctx->pruned_envelope += remaining;
            return;
        }
    }
    
    size_t mem_mark = mempool_mark(&ctx->pool);
    
    /* Allocate child headers on stack / pool */
    AURATidNode *children = (AURATidNode *)malloc(remaining * sizeof(AURATidNode));
    if (!children) return;
    size_t child_count = 0;
    
    for (size_t j = i + 1; j < class_size; j++) {
        AURATidNode *P2 = &classes[j];
        ctx->visited_nodes++;
        
        size_t new_size = 0;
        size_t new_k = P1->length + 1;
        size_t R_count = class_size - j - 1;
        int is_ho = 0;
        double recip_sum = 0.0;
        
        if (intersect_tids_and_check(ctx, P1, P2, ctx->tid_scratch, &new_size,
                                     new_k, R_count, &is_ho, &recip_sum)) {
            /* Allocate TID array from pool */
            uint32_t *tids_copy = mempool_alloc(&ctx->pool, new_size);
            int pool_alloc_ok = (tids_copy != NULL);
            if (!pool_alloc_ok) {
                tids_copy = (uint32_t *)malloc(new_size * sizeof(uint32_t));
            }
            memcpy(tids_copy, ctx->tid_scratch, new_size * sizeof(uint32_t));
            
            AURATidNode *ch = &children[child_count++];
            ch->item = P2->item;
            ch->length = (uint32_t)new_k;
            ch->num_tids = new_size;
            ch->recip_sum = recip_sum;
            ch->tids = tids_copy;
            
            if (is_ho) {
                ctx->raw_hoi_count++;
                ctx->raw_total_output_items += new_k;
            }
        } else {
            if (new_size < ctx->min_support) ctx->pruned_support++;
            else ctx->pruned_envelope++;
        }
    }
    
    if (child_count > 1) {
        /* Check if child class as a whole has enough tail length to reach threshold */
        double max_child_k = (double)(P1->length + child_count);
        int prune_subtree = 0;
        if (ctx->uniform_length) {
            if (max_child_k > (double)ctx->first_len) max_child_k = (double)ctx->first_len;
            double max_child_score = (max_child_k * (double)P1->num_tids) / (double)ctx->first_len;
            if (max_child_score + 1e-12 < ctx->threshold_value) prune_subtree = 1;
        } else {
            double max_child_score = max_child_k * P1->recip_sum;
            if (max_child_score + 1e-12 < ctx->threshold_value) prune_subtree = 1;
        }
        
        if (prune_subtree) {
            ctx->pruned_envelope += child_count;
        } else {
            aura_dfs_tid_raw_engine(ctx, children, child_count);
        }
    } else if (child_count == 1) {
        /* Single child has no siblings to join with (cannot form larger sets) */
        ctx->pruned_envelope += 1;
    }
    
    /* If any child had to use standard malloc, free it */
    if (ctx->pool.used + ctx->ntrans > ctx->pool.capacity) {
        for (size_t c = 0; c < child_count; c++) {
            if (children[c].tids < ctx->pool.pool ||
                children[c].tids >= ctx->pool.pool + ctx->pool.capacity) {
                free(children[c].tids);
            }
        }
    }
    
    free(children);
    mempool_rewind(&ctx->pool, mem_mark);
}

static void aura_dfs_tid_raw_engine(AURACtx *ctx, AURATidNode *classes, size_t class_size) {
    if (class_size < 2 || should_stop(ctx)) return;
    for (size_t i = 0; i < class_size; i++) {
        aura_process_root_item_raw(ctx, classes, class_size, i);
    }
}

/* ── Pure TID-List Closed Representative Ledger DFS Engine ───────────────── */
/* Utilizes:
 * 1. Closure Absorption: If |T(P \cup {e_j})| == |T(P)|, item absorbed without branching.
 * 2. Backward Check: Pruning branches whose prefix is subsumed by an earlier sibling.
 * 3. O(1) Double-hashed TID Ledger for verification and auditing.
 */
static int tid_is_subset(const uint32_t *sub, size_t n_sub, const uint32_t *sup, size_t n_sup) {
    if (n_sub > n_sup) return 0;
    size_t i = 0, j = 0;
    while (i < n_sub && j < n_sup) {
        if (sub[i] == sup[j]) {
            i++;
            j++;
        } else if (sub[i] > sup[j]) {
            j++;
        } else {
            return 0;
        }
    }
    return (i == n_sub);
}

static void aura_process_root_item_closed(AURACtx *ctx, size_t depth,
                                         AURATidNode *classes, size_t class_size, size_t i) {
    if (should_stop(ctx)) return;
    AURATidNode *P1 = &classes[i];
    
    /* Backward closure check: Check against previous siblings in this equivalence class */
    int backward = 0;
    for (size_t prev = 0; prev < i; prev++) {
        if (tid_is_subset(P1->tids, P1->num_tids, classes[prev].tids, classes[prev].num_tids)) {
            backward = 1;
            break;
        }
    }
    if (backward) {
        ctx->pruned_backward++;
        return;
    }
    
    /* Record item into current prefix path */
    ctx->prefix_items[depth] = ctx->active_items[P1->item];
    size_t current_len = depth + 1;
    
    /* Identify absorbed items (Closure Jump): items that appear in 100% of P1's transactions */
    size_t mem_mark = mempool_mark(&ctx->pool);
    size_t remaining = class_size - i - 1;
    AURATidNode *children = NULL;
    if (remaining > 0) {
        children = (AURATidNode *)malloc(remaining * sizeof(AURATidNode));
    }
    size_t child_count = 0;
    
    for (size_t j = i + 1; j < class_size; j++) {
        AURATidNode *P2 = &classes[j];
        
        /* Theorem 2: Pre-Intersection Sibling Pruning in O(1) */
        size_t R_count = class_size - j - 1;
        size_t max_desc_len = current_len + 1 + R_count;
        size_t min_supp = (P1->num_tids < P2->num_tids) ? P1->num_tids : P2->num_tids;
        if (min_supp < ctx->min_support) continue;
        if ((double)min_supp + 1e-12 < ctx->threshold_value) continue;
        
        double min_rsum = (P1->recip_sum < P2->recip_sum) ? P1->recip_sum : P2->recip_sum;
        if ((double)max_desc_len * min_rsum + 1e-12 < ctx->threshold_value) continue;
        
        ctx->visited_nodes++;
        
        size_t new_size = 0;
        size_t new_k = current_len + 1;
        int is_ho = 0;
        double recip_sum = 0.0;
        
        if (intersect_tids_and_check(ctx, P1, P2, ctx->tid_scratch, &new_size,
                                     new_k, R_count, &is_ho, &recip_sum)) {
            if (new_size == P1->num_tids) {
                /* Closure Absorption Jump! Item occurs in identical transaction set */
                ctx->prefix_items[current_len++] = ctx->active_items[P2->item];
                ctx->closure_jumps++;
            } else {
                /* Genuine branching child */
                uint32_t *tids_copy = mempool_alloc(&ctx->pool, new_size);
                if (!tids_copy) tids_copy = (uint32_t *)malloc(new_size * sizeof(uint32_t));
                memcpy(tids_copy, ctx->tid_scratch, new_size * sizeof(uint32_t));
                
                AURATidNode *ch = &children[child_count++];
                ch->item = P2->item;
                ch->length = (uint32_t)new_k;
                ch->num_tids = new_size;
                ch->recip_sum = recip_sum;
                ch->tids = tids_copy;
            }
        } else {
            if (new_size < ctx->min_support) ctx->pruned_support++;
            else ctx->pruned_envelope++;
        }
    }
    
    /* Evaluate closed representative pattern */
    double avg_occ = ((double)current_len * P1->recip_sum) / (double)P1->num_tids;
    double score = ctx->summed_occupancy_mode
        ? ((double)current_len * P1->recip_sum)
        : avg_occ;
    
    if (score + 1e-12 >= ctx->threshold_value) {
        ctx->raw_hoi_count++;
        if (ledger_add(&ctx->ledger, ctx->prefix_items, current_len, P1->tids, P1->num_tids, avg_occ)) {
            ctx->total_output_items += current_len;
        } else {
            ctx->ledger_duplicates++;
        }
    }
    
    /* Recurse into children equivalence classes with Safe Residual Upper Bound Pruning */
    if (child_count > 0) {
        double max_possible_k = (double)(current_len + child_count);
        int prune_children = 0;
        if (ctx->uniform_length) {
            if (max_possible_k > (double)ctx->first_len) max_possible_k = (double)ctx->first_len;
            double max_desc_score = (max_possible_k * (double)P1->num_tids) / (double)ctx->first_len;
            if (max_desc_score + 1e-12 < ctx->threshold_value) prune_children = 1;
        } else {
            double max_desc_score = max_possible_k * P1->recip_sum;
            if (max_desc_score + 1e-12 < ctx->threshold_value) prune_children = 1;
        }
        if (prune_children) {
            ctx->pruned_envelope += child_count;
        } else {
            aura_dfs_tid_closed_engine(ctx, current_len, children, child_count);
        }
    }
    
    /* Cleanup */
    if (ctx->pool.used + ctx->ntrans > ctx->pool.capacity) {
        for (size_t c = 0; c < child_count; c++) {
            if (children[c].tids < ctx->pool.pool ||
                children[c].tids >= ctx->pool.pool + ctx->pool.capacity) {
                free(children[c].tids);
            }
        }
    }
    if (children) free(children);
    mempool_rewind(&ctx->pool, mem_mark);
}

static void aura_dfs_tid_closed_engine(AURACtx *ctx, size_t depth,
                                       AURATidNode *classes, size_t class_size) {
    if (class_size == 0 || should_stop(ctx)) return;
    
    for (size_t i = 0; i < class_size; i++) {
        aura_process_root_item_closed(ctx, depth, classes, class_size, i);
    }
}

/* ── Multicore Parallel Worker Structures & Loop Functions ───────────────── */
typedef struct {
    AURACtx *main_ctx;
    AURATidNode *C1;
    size_t c1_count;
    size_t next_i;
    pthread_mutex_t task_mutex;
    pthread_mutex_t reduce_mutex;
    size_t pool_elements;
    size_t max_item_supp;
    size_t active_count;
} AURAParallelShared;

static void aura_parallel_worker_raw(void *arg) {
    AURAParallelShared *ps = (AURAParallelShared *)arg;
    AURACtx local_ctx;
    memcpy(&local_ctx, ps->main_ctx, sizeof(AURACtx));
    
    mempool_init(&local_ctx.pool, ps->pool_elements);
    local_ctx.tid_scratch = (uint32_t *)malloc((ps->max_item_supp + 1) * sizeof(uint32_t));
    local_ctx.prefix_items = (uint32_t *)malloc((ps->active_count + 1) * sizeof(uint32_t));
    
    local_ctx.raw_hoi_count = 0;
    local_ctx.raw_total_output_items = 0;
    local_ctx.visited_nodes = 0;
    local_ctx.pruned_support = 0;
    local_ctx.pruned_envelope = 0;
    local_ctx.pruned_backward = 0;
    local_ctx.closure_jumps = 0;
    
    while (1) {
        pthread_mutex_lock(&ps->task_mutex);
        size_t i = ps->next_i++;
        pthread_mutex_unlock(&ps->task_mutex);
        
        if (i >= ps->c1_count || should_stop(ps->main_ctx)) break;
        
        aura_process_root_item_raw(&local_ctx, ps->C1, ps->c1_count, i);
    }
    
    pthread_mutex_lock(&ps->reduce_mutex);
    ps->main_ctx->raw_hoi_count += local_ctx.raw_hoi_count;
    ps->main_ctx->raw_total_output_items += local_ctx.raw_total_output_items;
    ps->main_ctx->visited_nodes += local_ctx.visited_nodes;
    ps->main_ctx->pruned_support += local_ctx.pruned_support;
    ps->main_ctx->pruned_envelope += local_ctx.pruned_envelope;
    ps->main_ctx->pruned_backward += local_ctx.pruned_backward;
    ps->main_ctx->closure_jumps += local_ctx.closure_jumps;
    pthread_mutex_unlock(&ps->reduce_mutex);
    
    free(local_ctx.tid_scratch);
    free(local_ctx.prefix_items);
    mempool_free(&local_ctx.pool);
}

static void aura_parallel_worker_closed(void *arg) {
    AURAParallelShared *ps = (AURAParallelShared *)arg;
    AURACtx local_ctx;
    memcpy(&local_ctx, ps->main_ctx, sizeof(AURACtx));
    
    mempool_init(&local_ctx.pool, ps->pool_elements);
    local_ctx.tid_scratch = (uint32_t *)malloc((ps->max_item_supp + 1) * sizeof(uint32_t));
    local_ctx.prefix_items = (uint32_t *)malloc((ps->active_count + 1) * sizeof(uint32_t));
    ledger_init(&local_ctx.ledger, 256);
    
    local_ctx.raw_hoi_count = 0;
    local_ctx.total_output_items = 0;
    local_ctx.visited_nodes = 0;
    local_ctx.pruned_support = 0;
    local_ctx.pruned_envelope = 0;
    local_ctx.pruned_backward = 0;
    local_ctx.closure_jumps = 0;
    local_ctx.ledger_duplicates = 0;
    
    while (1) {
        pthread_mutex_lock(&ps->task_mutex);
        size_t i = ps->next_i++;
        pthread_mutex_unlock(&ps->task_mutex);
        
        if (i >= ps->c1_count || should_stop(ps->main_ctx)) break;
        
        aura_process_root_item_closed(&local_ctx, 0, ps->C1, ps->c1_count, i);
    }
    
    pthread_mutex_lock(&ps->reduce_mutex);
    ps->main_ctx->raw_hoi_count += local_ctx.raw_hoi_count;
    ps->main_ctx->visited_nodes += local_ctx.visited_nodes;
    ps->main_ctx->pruned_support += local_ctx.pruned_support;
    ps->main_ctx->pruned_envelope += local_ctx.pruned_envelope;
    ps->main_ctx->pruned_backward += local_ctx.pruned_backward;
    ps->main_ctx->closure_jumps += local_ctx.closure_jumps;
    ps->main_ctx->ledger_duplicates += local_ctx.ledger_duplicates;
    
    for (size_t k = 0; k < local_ctx.ledger.count; k++) {
        ledger_add(&ps->main_ctx->ledger, local_ctx.ledger.data[k].items,
                   local_ctx.ledger.data[k].len, local_ctx.ledger.data[k].tids,
                   local_ctx.ledger.data[k].support, local_ctx.ledger.data[k].occupancy);
    }
    ps->main_ctx->total_output_items += local_ctx.total_output_items;
    pthread_mutex_unlock(&ps->reduce_mutex);
    
    ledger_free(&local_ctx.ledger);
    free(local_ctx.tid_scratch);
    free(local_ctx.prefix_items);
    mempool_free(&local_ctx.pool);
}

/* ── Entry Point ────────────────────────────────────────────────────────── */
static DM_Status run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_TRANSACTIONAL) return DM_ERROR_INCOMPATIBLE;
    DM_AURA_HOI_Params *p = (DM_AURA_HOI_Params *)params;

    AURACtx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.trans = (DM_Trans_Simple *)ds->payload;
    ctx.ntrans = ds->count;
    ctx.max_id = ds->max_id;
    ctx.min_occupancy = p ? p->min_occupancy : 0.5;
    ctx.summed_occupancy_mode = p ? p->summed_occupancy_mode : 0;
    ctx.threshold_value = ctx.summed_occupancy_mode
        ? ((ctx.min_occupancy < 1.0) ? ctx.min_occupancy * (double)ctx.ntrans : ctx.min_occupancy)
        : ctx.min_occupancy;
    ctx.min_support = (p && p->min_support) ? p->min_support : (size_t)ceil(ctx.min_occupancy * (double)ctx.ntrans);
    if (ctx.min_support < 1) ctx.min_support = 1;
    ctx.max_patterns = p ? p->max_patterns : 0;
    ctx.max_seconds = p ? p->max_seconds : 0.0;
    ctx.emit_raw_view = p ? p->emit_raw_view : 0;
    ctx.top_k = p ? p->top_k : 0;
    ctx.start_clock = clock();

    /* ── Step 1: Preprocess Transaction Lengths & Length Equivalence ────── */
    TransMeta *meta = (TransMeta *)malloc(ctx.ntrans * sizeof(TransMeta));
    if (!meta) return DM_ERROR_MEMORY;

    ctx.uniform_length = 1;
    ctx.first_len = ctx.trans[0].count;

    for (size_t i = 0; i < ctx.ntrans; i++) {
        meta[i].orig_tid = (uint32_t)i;
        meta[i].len = (uint32_t)ctx.trans[i].count;
        if (ctx.trans[i].count != ctx.first_len) {
            ctx.uniform_length = 0;
        }
    }
    qsort(meta, ctx.ntrans, sizeof(TransMeta), cmp_trans_meta);

    ctx.max_tsize = 0;
    for (size_t i = 0; i < ctx.ntrans; i++) {
        if (meta[i].len > ctx.max_tsize) ctx.max_tsize = meta[i].len;
    }

    ctx.g_tsize = (uint32_t *)malloc(ctx.ntrans * sizeof(uint32_t));
    ctx.recip_table = (double *)malloc(((size_t)ctx.max_tsize + 1) * sizeof(double));
    if (!ctx.g_tsize || !ctx.recip_table) {
        free(meta);
        return DM_ERROR_MEMORY;
    }

    for (uint32_t l = 0; l <= ctx.max_tsize; l++) {
        ctx.recip_table[l] = l ? (1.0 / (double)l) : 0.0;
    }

    for (size_t i = 0; i < ctx.ntrans; i++) {
        ctx.g_tsize[i] = meta[i].len;
    }

    /* ── Step 2: 1-Item Support Counting ────────────────────────────────── */
    uint32_t *counts = (uint32_t *)calloc((size_t)ctx.max_id + 1, sizeof(uint32_t));
    if (!counts) {
        free(meta);
        return DM_ERROR_MEMORY;
    }

    for (size_t i = 0; i < ctx.ntrans; i++) {
        uint32_t orig = meta[i].orig_tid;
        for (size_t j = 0; j < ctx.trans[orig].count; j++) {
            counts[ctx.trans[orig].items[j]]++;
        }
    }

    for (uint32_t i = 0; i <= ctx.max_id; i++) {
        if (counts[i] >= ctx.min_support) ctx.active_count++;
    }

    if (ctx.active_count == 0) {
        free(meta);
        free(counts);
        free(ctx.g_tsize);
        free(ctx.recip_table);
        dm_bench_record_results(0, 0);
        return DM_SUCCESS;
    }

    ctx.active_items = (uint32_t *)malloc(ctx.active_count * sizeof(uint32_t));
    ctx.item_counts = (uint32_t *)malloc(ctx.active_count * sizeof(uint32_t));
    ctx.item_recip_sums = (double *)malloc(ctx.active_count * sizeof(double));
    uint32_t *id_to_active = (uint32_t *)malloc(((size_t)ctx.max_id + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i <= ctx.max_id; i++) id_to_active[i] = UINT32_MAX;

    size_t idx = 0;
    for (uint32_t i = 0; i <= ctx.max_id; i++) {
        if (counts[i] >= ctx.min_support) {
            id_to_active[i] = (uint32_t)idx;
            ctx.active_items[idx] = i;
            ctx.item_counts[idx] = counts[i];
            idx++;
        }
    }

    /* ── Step 3: Populate 1-Item TID Lists ───────────────────────────────── */
    ctx.item_tids = (uint32_t **)malloc(ctx.active_count * sizeof(uint32_t *));
    uint32_t *item_idx = (uint32_t *)calloc(ctx.active_count, sizeof(uint32_t));
    for (size_t i = 0; i < ctx.active_count; i++) {
        ctx.item_tids[i] = (uint32_t *)malloc(ctx.item_counts[i] * sizeof(uint32_t));
    }

    for (size_t i = 0; i < ctx.ntrans; i++) {
        uint32_t orig = meta[i].orig_tid;
        for (size_t j = 0; j < ctx.trans[orig].count; j++) {
            uint32_t item = ctx.trans[orig].items[j];
            uint32_t aid = id_to_active[item];
            if (aid != UINT32_MAX) {
                ctx.item_tids[aid][item_idx[aid]++] = (uint32_t)i;
            }
        }
    }

    /* Calculate reciprocal sum for each 1-item TID list */
    for (size_t i = 0; i < ctx.active_count; i++) {
        double rsum = 0.0;
        if (ctx.uniform_length) {
            rsum = (double)ctx.item_counts[i] / (double)ctx.first_len;
        } else {
            for (size_t t = 0; t < ctx.item_counts[i]; t++) {
                uint32_t tid = ctx.item_tids[i][t];
                rsum += ctx.recip_table[ctx.g_tsize[tid]];
            }
        }
        ctx.item_recip_sums[i] = rsum;
    }

    free(item_idx);
    free(id_to_active);
    free(counts);
    free(meta);

    /* Allocate scratchpad & memory pool based on maximum active item support */
    size_t max_item_supp = 0;
    for (size_t i = 0; i < ctx.active_count; i++) {
        if (ctx.item_counts[i] > max_item_supp) max_item_supp = ctx.item_counts[i];
    }
    ctx.tid_scratch = (uint32_t *)malloc((max_item_supp + 1) * sizeof(uint32_t));
    ctx.prefix_items = (uint32_t *)malloc((ctx.active_count + 1) * sizeof(uint32_t));
    
    /* Pool capacity: dynamic linear arena */
    size_t pool_elements = max_item_supp > 500000 ? max_item_supp * 2 : 1000000;
    mempool_init(&ctx.pool, pool_elements);

    if (!ctx.emit_raw_view) {
        ledger_init(&ctx.ledger, 256);
    }

    printf("[AURA-HOI] Pure TID-List Engine: transactions=%zu active_items=%zu minsup=%zu minocc=%.6f threshold=%.6f mode=%s view=%s uniform_len=%s\n",
           ctx.ntrans, ctx.active_count, ctx.min_support, ctx.min_occupancy, ctx.threshold_value,
           ctx.summed_occupancy_mode ? "summed-compatible" : "average",
           ctx.emit_raw_view ? "raw-fullset" : "closed-ledger",
           ctx.uniform_length ? "yes" : "no");

    /* ── Step 4: Build Equivalence Class C1 and Mine ─────────────────────── */
    AURATidNode *C1 = (AURATidNode *)malloc(ctx.active_count * sizeof(AURATidNode));
    size_t c1_count = 0;

    for (size_t i = 0; i < ctx.active_count; i++) {
        int is_ho = 0;
        double recip_sum = 0.0;
        size_t dummy_size = ctx.item_counts[i];
        
        /* Check 1-item status */
        if (ctx.uniform_length) {
            double o_val = (double)dummy_size / (double)ctx.first_len;
            is_ho = (o_val >= ctx.threshold_value);
            recip_sum = ctx.item_recip_sums[i];
        } else {
            /* compute 1-item UBO check */
            double current_sum = 0.0;
            double max_ubo = 0.0;
            for (int t = (int)dummy_size - 1; t >= 0; t--) {
                uint32_t tid = ctx.item_tids[i][t];
                uint32_t tsize = ctx.g_tsize[tid];
                current_sum += ctx.recip_table[tsize];
                if (t == 0 || ctx.g_tsize[ctx.item_tids[i][t - 1]] < tsize) {
                    double current_ubo = (double)tsize * current_sum;
                    if (current_ubo > max_ubo) max_ubo = current_ubo;
                }
            }
            if (max_ubo < ctx.threshold_value) continue;
            is_ho = (current_sum >= ctx.threshold_value);
            recip_sum = current_sum;
        }

        AURATidNode *node = &C1[c1_count++];
        node->item = (uint32_t)i;
        node->length = 1;
        node->num_tids = dummy_size;
        node->tids = ctx.item_tids[i];
        node->recip_sum = recip_sum;

        if (is_ho) {
            ctx.raw_hoi_count++;
            ctx.raw_total_output_items += 1;
            if (!ctx.emit_raw_view) {
                uint32_t single_item = ctx.active_items[i];
                double avg_occ = recip_sum / (double)dummy_size;
                ledger_add(&ctx.ledger, &single_item, 1, node->tids, dummy_size, avg_occ);
                ctx.total_output_items += 1;
            }
        }
    }

    int num_threads = (p && p->threads > 0) ? p->threads : dm_get_num_threads();
    if (num_threads < 1) num_threads = 1;
    if (num_threads > (int)c1_count && c1_count > 0) num_threads = (int)c1_count;

    if (num_threads > 1 && c1_count > 1) {
        AURAParallelShared ps;
        memset(&ps, 0, sizeof(ps));
        ps.main_ctx = &ctx;
        ps.C1 = C1;
        ps.c1_count = c1_count;
        ps.next_i = 0;
        pthread_mutex_init(&ps.task_mutex, NULL);
        pthread_mutex_init(&ps.reduce_mutex, NULL);
        ps.pool_elements = pool_elements;
        ps.max_item_supp = max_item_supp;
        ps.active_count = ctx.active_count;

        DM_ThreadPool *pool = dm_threadpool_create(num_threads);
        if (pool) {
            printf("[AURA-HOI] Multicore Mode Enabled: %d worker threads across %zu root branches\n", num_threads, c1_count);
            for (int t = 0; t < num_threads; t++) {
                dm_threadpool_submit(pool, ctx.emit_raw_view ? aura_parallel_worker_raw : aura_parallel_worker_closed, &ps);
            }
            dm_threadpool_wait(pool);
            dm_threadpool_destroy(pool);
        } else {
            if (ctx.emit_raw_view) aura_dfs_tid_raw_engine(&ctx, C1, c1_count);
            else aura_dfs_tid_closed_engine(&ctx, 0, C1, c1_count);
        }
        pthread_mutex_destroy(&ps.task_mutex);
        pthread_mutex_destroy(&ps.reduce_mutex);
    } else {
        if (ctx.emit_raw_view) {
            aura_dfs_tid_raw_engine(&ctx, C1, c1_count);
        } else {
            aura_dfs_tid_closed_engine(&ctx, 0, C1, c1_count);
        }
    }

    if (ctx.emit_raw_view) {
        printf("[AURA-HOI] Complete. Raw fullset HO itemsets found: %zu\n", ctx.raw_hoi_count);
        dm_bench_record_results(ctx.raw_hoi_count, ctx.raw_total_output_items);
    } else {
        printf("[AURA-HOI] Complete. Auditable closed HOI representatives found: %zu\n", ctx.ledger.count);
        dm_bench_record_results(ctx.ledger.count, ctx.total_output_items);
    }

    printf("[AURA-HOI] raw_accepts=%zu support_classes=%zu ledger_duplicates=%zu topk_updates=%zu\n",
           ctx.raw_hoi_count, ctx.ledger.count, ctx.ledger_duplicates, ctx.topk_updates);
    printf("[AURA-HOI] visited_nodes=%zu pruned_support=%zu pruned_backward=%zu pruned_envelope=%zu closure_jumps=%zu limited=%s\n",
           ctx.visited_nodes, ctx.pruned_support, ctx.pruned_backward, ctx.pruned_envelope,
           ctx.closure_jumps, ctx.limited ? "yes" : "no");

    /* ── Cleanup ────────────────────────────────────────────────────────── */
    free(C1);
    for (size_t i = 0; i < ctx.active_count; i++) {
        free(ctx.item_tids[i]);
    }
    free(ctx.item_tids);
    free(ctx.item_counts);
    free(ctx.item_recip_sums);
    free(ctx.active_items);
    free(ctx.g_tsize);
    free(ctx.recip_table);
    free(ctx.tid_scratch);
    free(ctx.prefix_items);
    mempool_free(&ctx.pool);

    if (!ctx.emit_raw_view) {
        ledger_free(&ctx.ledger);
    }

    return DM_SUCCESS;
}

DM_Algorithm aura_hoi_algo = {
    .id = "aura_hoi",
    .name = "AURA-HOI",
    .description = "Auditable representative high-occupancy itemset mining via Pure TID-List Equivalence Classes and Cumulative Bounding.",
    .supported_types = (1 << DM_TYPE_TRANSACTIONAL),
    .run = run
};

DM_REGISTER_ALGORITHM(aura_hoi_algo)
