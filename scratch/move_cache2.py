with open("include/core/dm_engine.h", "r") as f: text = f.read()

cache_struct = """
/* ── DM_WeightCache ─────────────────────────────────────────────────────── */
typedef struct {
    DM_Block **blocks;
    uint32_t *seeds;
    int count;
    int capacity;
} DM_WeightCache;

DM_WeightCache *dm_weight_cache_new(void);
DM_Block *dm_weight_cache_get(DM_WeightCache *cache, int ndim, const int64_t *shape, unsigned int seed, float scale);
void dm_weight_cache_free(DM_WeightCache *cache);
"""
text = text.replace('#endif /* DM_ENGINE_H */', cache_struct + '\n#endif /* DM_ENGINE_H */')
with open("include/core/dm_engine.h", "w") as f: f.write(text)

with open("src/core/dm_engine.c", "r") as f: text = f.read()
if "DM_WeightCache Implementation" not in text:
    cache_impl = """
/* ── DM_WeightCache Implementation ──────────────────────────────────────── */

DM_WeightCache *dm_weight_cache_new(void) {
    DM_WeightCache *c = malloc(sizeof(DM_WeightCache));
    if (!c) return NULL;
    c->blocks = NULL;
    c->seeds = NULL;
    c->count = 0;
    c->capacity = 0;
    return c;
}

DM_Block *dm_weight_cache_get(DM_WeightCache *cache, int ndim, const int64_t *shape, unsigned int seed, float scale) {
    for (int i = 0; i < cache->count; i++) {
        if (cache->seeds[i] == seed) return cache->blocks[i];
    }
    if (cache->count >= cache->capacity) {
        cache->capacity = cache->capacity ? cache->capacity * 2 : 16;
        cache->blocks = realloc(cache->blocks, cache->capacity * sizeof(DM_Block *));
        cache->seeds = realloc(cache->seeds, cache->capacity * sizeof(uint32_t));
    }
    DM_Block *b = malloc(sizeof(DM_Block));
    dm_block_create(b, DM_KIND_DENSE, DM_DTYPE_F32, DM_LAYOUT_ROW_MAJOR, DM_BACKEND_CPU, ndim, shape);
    
    // Generate weights
    uint32_t s = seed;
    size_t n = b->count;
    for (size_t i = 0; i < n; i++) {
        uint32_t x = s ? s : 2463534242u;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        s = x;
        ((float*)b->data)[i] = (((s >> 8) * (1.0f / 16777216.0f)) * 2.0f - 1.0f) * scale;
    }
    
    // Lower block once (Phase 5 persistent caching)
    DM_Block tf_b;
    dm_lower_block(b, &tf_b, DM_BACKEND_TENSORFLOW, DM_LOWER_VIEW);
    
    cache->blocks[cache->count] = b;
    cache->seeds[cache->count] = seed;
    cache->count++;
    return b;
}

void dm_weight_cache_free(DM_WeightCache *cache) {
    if (!cache) return;
    for (int i = 0; i < cache->count; i++) {
        dm_block_free(cache->blocks[i]);
        free(cache->blocks[i]);
    }
    free(cache->blocks);
    free(cache->seeds);
    free(cache);
}
"""
    text = text.replace('/* ── Initialization & TF Lifecycle', cache_impl + '\n/* ── Initialization & TF Lifecycle')
    with open("src/core/dm_engine.c", "w") as f: f.write(text)
