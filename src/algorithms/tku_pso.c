#include "algorithms/tku_pso.h"
#include "core/dm_dataset_types.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif

typedef struct {
    uint32_t id;
    double utility;
    double twu;
    double max_utility;
    double avg_utility;
    size_t support;
} PSOItem;

typedef struct {
    uint64_t *bits;
    double fitness;
} Particle;

typedef struct {
    uint32_t *items;
    size_t len;
    double utility;
} PSOPattern;

typedef struct {
    PSOPattern *data;
    size_t count;
    size_t cap;
    size_t k;
    double threshold;
    size_t threshold_raises;
} TopK;

typedef struct {
    uint64_t *keys;
    size_t count;
    size_t cap;
    size_t words;
} ExploredSet;

typedef struct {
    DM_Trans_Utility *trans;
    size_t transactions;
    uint32_t max_id;
    uint32_t *rank;
    PSOItem *items;
    size_t nitems;
    size_t words;
    TopK topk;
    ExploredSet explored;
    Particle *pop;
    Particle *pbest;
    Particle gbest;
    DM_TKU_PSO_Params params;
    DM_TKU_PSO_Stats *stats;
    double deviation;
    unsigned int rng;
    clock_t start_clock;
} Ctx;

static double elapsed_sec(Ctx *ctx) {
    return (double)(clock() - ctx->start_clock) / (double)CLOCKS_PER_SEC;
}

static int limited(Ctx *ctx) {
    if (ctx->params.max_seconds > 0.0 && elapsed_sec(ctx) >= ctx->params.max_seconds) {
        ctx->stats->limited = 1;
        return 1;
    }
    return 0;
}

static uint32_t rnd_u32(Ctx *ctx) {
    ctx->rng = ctx->rng * 1664525u + 1013904223u;
    return ctx->rng;
}

static double rnd01(Ctx *ctx) {
    return (double)rnd_u32(ctx) / 4294967295.0;
}

static int bit_get(const uint64_t *bits, size_t i) {
    return (bits[i / 64] >> (i % 64)) & 1ULL;
}

static void bit_set(uint64_t *bits, size_t i) {
    bits[i / 64] |= 1ULL << (i % 64);
}

static void bit_clear(uint64_t *bits, size_t i) {
    bits[i / 64] &= ~(1ULL << (i % 64));
}

static void bit_flip(uint64_t *bits, size_t i) {
    bits[i / 64] ^= 1ULL << (i % 64);
}

static size_t bit_count_words(const uint64_t *bits, size_t words) {
    size_t out = 0;
    for (size_t i = 0; i < words; i++) {
#ifdef _MSC_VER
        out += (size_t)__popcnt64(bits[i]);
#else
        out += (size_t)__builtin_popcountll(bits[i]);
#endif
    }
    return out;
}

static int cmp_pattern_desc(const void *a, const void *b) {
    const PSOPattern *x = (const PSOPattern *)a;
    const PSOPattern *y = (const PSOPattern *)b;
    if (x->utility < y->utility) return 1;
    if (x->utility > y->utility) return -1;
    if (x->len > y->len) return 1;
    if (x->len < y->len) return -1;
    return 0;
}

static int same_items(const PSOPattern *p, const uint32_t *items, size_t len) {
    if (p->len != len) return 0;
    for (size_t i = 0; i < len; i++) {
        if (p->items[i] != items[i]) return 0;
    }
    return 1;
}

static void topk_free(TopK *t) {
    for (size_t i = 0; i < t->count; i++) free(t->data[i].items);
    free(t->data);
    memset(t, 0, sizeof(*t));
}

static int topk_add(TopK *t, const uint32_t *items, size_t len, double utility) {
    if (t->k == 0) return 0;
    if (t->count >= t->k && utility <= t->threshold) return 0;
    for (size_t i = 0; i < t->count; i++) {
        if (same_items(&t->data[i], items, len)) return 0;
    }
    if (t->count == t->cap) {
        size_t nc = t->cap ? t->cap * 2 : 64;
        PSOPattern *nd = (PSOPattern *)realloc(t->data, nc * sizeof(*nd));
        if (!nd) return -1;
        t->data = nd;
        t->cap = nc;
    }
    t->data[t->count].items = (uint32_t *)malloc(len * sizeof(uint32_t));
    if (!t->data[t->count].items) return -1;
    memcpy(t->data[t->count].items, items, len * sizeof(uint32_t));
    t->data[t->count].len = len;
    t->data[t->count].utility = utility;
    t->count++;
    qsort(t->data, t->count, sizeof(*t->data), cmp_pattern_desc);
    if (t->count >= t->k) {
        double old = t->threshold;
        t->threshold = t->data[t->k - 1].utility;
        if (t->threshold > old) t->threshold_raises++;
        size_t keep = t->count;
        while (keep > t->k) {
            free(t->data[keep - 1].items);
            keep--;
        }
        t->count = keep;
    }
    return 0;
}

static void particle_free(Particle *p) {
    free(p->bits);
    memset(p, 0, sizeof(*p));
}

static int particle_alloc(Particle *p, size_t words) {
    memset(p, 0, sizeof(*p));
    p->bits = (uint64_t *)calloc(words, sizeof(uint64_t));
    return p->bits ? 0 : -1;
}

static int particle_copy(Particle *dst, const Particle *src, size_t words) {
    if (!dst->bits) {
        dst->bits = (uint64_t *)malloc(words * sizeof(uint64_t));
        if (!dst->bits) return -1;
    }
    memcpy(dst->bits, src->bits, words * sizeof(uint64_t));
    dst->fitness = src->fitness;
    return 0;
}

static int particle_empty(const Particle *p, size_t words) {
    for (size_t i = 0; i < words; i++) {
        if (p->bits[i]) return 0;
    }
    return 1;
}

static int explored_init(ExploredSet *s, size_t words, size_t cap) {
    s->words = words;
    s->cap = cap ? cap : 1024;
    s->keys = (uint64_t *)calloc(s->cap * words, sizeof(uint64_t));
    return s->keys ? 0 : -1;
}

static void explored_free(ExploredSet *s) {
    free(s->keys);
    memset(s, 0, sizeof(*s));
}

static int explored_contains(ExploredSet *s, const uint64_t *bits) {
    for (size_t i = 0; i < s->count; i++) {
        if (memcmp(&s->keys[i * s->words], bits, s->words * sizeof(uint64_t)) == 0) return 1;
    }
    return 0;
}

static int explored_add(ExploredSet *s, const uint64_t *bits) {
    if (explored_contains(s, bits)) return 0;
    if (s->count == s->cap) {
        size_t nc = s->cap * 2;
        uint64_t *nk = (uint64_t *)realloc(s->keys, nc * s->words * sizeof(uint64_t));
        if (!nk) return -1;
        s->keys = nk;
        s->cap = nc;
    }
    memcpy(&s->keys[s->count * s->words], bits, s->words * sizeof(uint64_t));
    s->count++;
    return 0;
}

static double kth_value_desc(double *vals, size_t n, size_t k) {
    if (k == 0 || n == 0) return 0.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = i + 1; j < n; j++) {
            if (vals[j] > vals[i]) {
                double tmp = vals[i];
                vals[i] = vals[j];
                vals[j] = tmp;
            }
        }
    }
    return vals[k <= n ? k - 1 : n - 1];
}

static int cmp_item_utility_desc(const void *a, const void *b) {
    const PSOItem *x = (const PSOItem *)a;
    const PSOItem *y = (const PSOItem *)b;
    if (x->utility < y->utility) return 1;
    if (x->utility > y->utility) return -1;
    if (x->id > y->id) return 1;
    if (x->id < y->id) return -1;
    return 0;
}

static double fitness(Ctx *ctx, Particle *p) {
    if (particle_empty(p, ctx->words)) return 0.0;
    double total = 0.0;
    for (size_t t = 0; t < ctx->transactions; t++) {
        DM_Trans_Utility *tr = &ctx->trans[t];
        double util = 0.0;
        int ok = 1;
        for (size_t r = 0; r < ctx->nitems; r++) {
            if (!bit_get(p->bits, r)) continue;
            uint32_t id = ctx->items[r].id;
            int found = 0;
            for (size_t i = 0; i < tr->count; i++) {
                if (tr->items[i].id == id) {
                    util += tr->items[i].utility;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                ok = 0;
                break;
            }
        }
        if (ok) total += util;
    }
    ctx->stats->evaluated_particles++;
    return total;
}

static int particle_to_items(Ctx *ctx, Particle *p, uint32_t **items, size_t *len) {
    *len = bit_count_words(p->bits, ctx->words);
    *items = (uint32_t *)malloc((*len ? *len : 1) * sizeof(uint32_t));
    if (!*items) return -1;
    size_t pos = 0;
    for (size_t i = 0; i < ctx->nitems; i++) {
        if (bit_get(p->bits, i)) (*items)[pos++] = ctx->items[i].id;
    }
    return 0;
}

static int add_particle_to_topk(Ctx *ctx, Particle *p) {
    uint32_t *items = NULL;
    size_t len = 0;
    if (particle_to_items(ctx, p, &items, &len) != 0) return -1;
    int rc = topk_add(&ctx->topk, items, len, p->fitness);
    free(items);
    return rc;
}

static int appears_in_transaction(Ctx *ctx, Particle *p) {
    if (particle_empty(p, ctx->words)) return 0;
    for (size_t t = 0; t < ctx->transactions; t++) {
        DM_Trans_Utility *tr = &ctx->trans[t];
        int ok = 1;
        for (size_t r = 0; r < ctx->nitems; r++) {
            if (!bit_get(p->bits, r)) continue;
            uint32_t id = ctx->items[r].id;
            int found = 0;
            for (size_t i = 0; i < tr->count; i++) {
                if (tr->items[i].id == id) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                ok = 0;
                break;
            }
        }
        if (ok) return 1;
    }
    return 0;
}

static size_t roulette_item(Ctx *ctx) {
    double sum = 0.0;
    for (size_t i = 0; i < ctx->nitems; i++) sum += ctx->items[i].twu > 0.0 ? ctx->items[i].twu : 1.0;
    double r = rnd01(ctx) * sum;
    for (size_t i = 0; i < ctx->nitems; i++) {
        r -= ctx->items[i].twu > 0.0 ? ctx->items[i].twu : 1.0;
        if (r <= 0.0) return i;
    }
    return ctx->nitems ? ctx->nitems - 1 : 0;
}

static void random_particle(Ctx *ctx, Particle *p) {
    memset(p->bits, 0, ctx->words * sizeof(uint64_t));
    if (ctx->nitems == 0) return;
    size_t target = 1 + (rnd_u32(ctx) % (ctx->nitems < 4 ? ctx->nitems : 4));
    for (size_t i = 0; i < target; i++) bit_set(p->bits, roulette_item(ctx));
}

static void pev_check(Ctx *ctx, Particle *p) {
    if (ctx->nitems == 0) return;
    size_t guard = ctx->nitems + 2;
    while (!appears_in_transaction(ctx, p) && guard-- > 0) {
        ctx->stats->pev_repairs++;
        size_t count = bit_count_words(p->bits, ctx->words);
        if (count <= 1) {
            memset(p->bits, 0, ctx->words * sizeof(uint64_t));
            bit_set(p->bits, roulette_item(ctx));
        } else {
            size_t target = rnd_u32(ctx) % count;
            for (size_t i = 0; i < ctx->nitems; i++) {
                if (!bit_get(p->bits, i)) continue;
                if (target-- == 0) {
                    bit_clear(p->bits, i);
                    break;
                }
            }
        }
    }
}

static double estimate(Ctx *ctx, Particle *p) {
    double sum = 0.0;
    size_t min_sup = 0;
    for (size_t i = 0; i < ctx->nitems; i++) {
        if (!bit_get(p->bits, i)) continue;
        sum += ctx->items[i].avg_utility + ctx->deviation;
        if (min_sup == 0 || ctx->items[i].support < min_sup) min_sup = ctx->items[i].support;
    }
    return (double)min_sup * sum;
}

static void flip_or_clear(Ctx *ctx, Particle *p, size_t item) {
    if (ctx->items[item].twu < ctx->topk.threshold) bit_clear(p->bits, item);
    else bit_flip(p->bits, item);
}

static void change_from_difference(Ctx *ctx, Particle *p, Particle *target, double r) {
    size_t diff_count = 0;
    for (size_t i = 0; i < ctx->nitems; i++) {
        if (bit_get(p->bits, i) != bit_get(target->bits, i)) diff_count++;
    }
    size_t b = (size_t)floor(r * (double)diff_count);
    while (b > 0 && diff_count > 0) {
        size_t pick = rnd_u32(ctx) % diff_count;
        for (size_t i = 0; i < ctx->nitems; i++) {
            if (bit_get(p->bits, i) == bit_get(target->bits, i)) continue;
            if (pick-- == 0) {
                flip_or_clear(ctx, p, i);
                b--;
                break;
            }
        }
        diff_count = 0;
        for (size_t i = 0; i < ctx->nitems; i++) {
            if (bit_get(p->bits, i) != bit_get(target->bits, i)) diff_count++;
        }
    }
}

static void update_particle(Ctx *ctx, Particle *p, Particle *pbest) {
    change_from_difference(ctx, p, pbest, rnd01(ctx));
    change_from_difference(ctx, p, &ctx->gbest, rnd01(ctx));
    if (explored_contains(&ctx->explored, p->bits) && ctx->nitems > 0) {
        size_t item = roulette_item(ctx);
        flip_or_clear(ctx, p, item);
    }
    pev_check(ctx, p);
}

static void reselect_gbest(Ctx *ctx) {
    if (ctx->topk.count == 0 || ctx->nitems == 0) return;
    double sum = 0.0;
    for (size_t i = 0; i < ctx->topk.count; i++) sum += ctx->topk.data[i].utility;
    double r = rnd01(ctx) * sum;
    size_t chosen = 0;
    for (; chosen < ctx->topk.count; chosen++) {
        r -= ctx->topk.data[chosen].utility;
        if (r <= 0.0) break;
    }
    if (chosen >= ctx->topk.count) chosen = ctx->topk.count - 1;
    memset(ctx->gbest.bits, 0, ctx->words * sizeof(uint64_t));
    for (size_t i = 0; i < ctx->topk.data[chosen].len; i++) {
        uint32_t id = ctx->topk.data[chosen].items[i];
        if (id <= ctx->max_id && ctx->rank[id] != UINT32_MAX) bit_set(ctx->gbest.bits, ctx->rank[id]);
    }
    ctx->gbest.fitness = ctx->topk.data[chosen].utility;
}

static int initialize(Ctx *ctx) {
    ctx->pop = (Particle *)calloc(ctx->params.population_size, sizeof(Particle));
    ctx->pbest = (Particle *)calloc(ctx->params.population_size, sizeof(Particle));
    if (!ctx->pop || !ctx->pbest) return -1;
    if (particle_alloc(&ctx->gbest, ctx->words) != 0) return -1;
    for (size_t i = 0; i < ctx->params.population_size; i++) {
        if (particle_alloc(&ctx->pop[i], ctx->words) != 0 || particle_alloc(&ctx->pbest[i], ctx->words) != 0) return -1;
        if (i < ctx->nitems) {
            bit_set(ctx->pop[i].bits, i);
            ctx->stats->initialized_singletons++;
        } else {
            random_particle(ctx, &ctx->pop[i]);
            pev_check(ctx, &ctx->pop[i]);
            ctx->stats->roulette_initialized++;
        }
        ctx->pop[i].fitness = fitness(ctx, &ctx->pop[i]);
        if (ctx->pop[i].fitness > ctx->topk.threshold && add_particle_to_topk(ctx, &ctx->pop[i]) != 0) return -1;
        if (particle_copy(&ctx->pbest[i], &ctx->pop[i], ctx->words) != 0) return -1;
        if (i == 0 || ctx->pop[i].fitness > ctx->gbest.fitness) {
            if (particle_copy(&ctx->gbest, &ctx->pop[i], ctx->words) != 0) return -1;
        }
        if (explored_add(&ctx->explored, ctx->pop[i].bits) != 0) return -1;
    }
    for (size_t i = ctx->params.population_size; i < ctx->nitems && ctx->topk.count < ctx->topk.k; i++) {
        Particle p;
        if (particle_alloc(&p, ctx->words) != 0) return -1;
        bit_set(p.bits, i);
        p.fitness = ctx->items[i].utility;
        if (add_particle_to_topk(ctx, &p) != 0) {
            particle_free(&p);
            return -1;
        }
        particle_free(&p);
    }
    return 0;
}

int tku_pso_mine_dataset(DM_Dataset *ds, const DM_TKU_PSO_Params *params, DM_TKU_PSO_Stats *stats) {
    if (!ds || ds->type != DM_TYPE_UTILITY || !params || !stats || params->k == 0) return -1;
    memset(stats, 0, sizeof(*stats));
    Ctx ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.trans = (DM_Trans_Utility *)ds->payload;
    ctx.transactions = ds->count;
    ctx.max_id = ds->max_id;
    ctx.params = *params;
    if (ctx.params.population_size == 0) ctx.params.population_size = 20;
    if (ctx.params.iterations == 0) ctx.params.iterations = 10000;
    if (ctx.params.seed == 0) ctx.params.seed = 42;
    ctx.rng = ctx.params.seed;
    ctx.stats = stats;
    ctx.start_clock = clock();

    double *twu = (double *)calloc(ds->max_id + 1, sizeof(double));
    double *util = (double *)calloc(ds->max_id + 1, sizeof(double));
    double *maxu = (double *)calloc(ds->max_id + 1, sizeof(double));
    size_t *sup = (size_t *)calloc(ds->max_id + 1, sizeof(size_t));
    ctx.rank = (uint32_t *)malloc((ds->max_id + 1) * sizeof(uint32_t));
    if (!twu || !util || !maxu || !sup || !ctx.rank) {
        free(twu); free(util); free(maxu); free(sup); free(ctx.rank);
        return -1;
    }
    for (uint32_t id = 0; id <= ds->max_id; id++) ctx.rank[id] = UINT32_MAX;
    for (size_t t = 0; t < ds->count; t++) {
        for (size_t i = 0; i < ctx.trans[t].count; i++) {
            uint32_t id = ctx.trans[t].items[i].id;
            double u = ctx.trans[t].items[i].utility;
            twu[id] += ctx.trans[t].total_utility;
            util[id] += u;
            if (sup[id] == 0 || u > maxu[id]) maxu[id] = u;
            sup[id]++;
        }
    }
    size_t distinct = 0;
    for (uint32_t id = 0; id <= ds->max_id; id++) {
        if (sup[id] > 0) distinct++;
    }
    double *single = (double *)malloc((distinct ? distinct : 1) * sizeof(double));
    if (!single) {
        free(twu); free(util); free(maxu); free(sup); free(ctx.rank);
        return -1;
    }
    size_t sp = 0;
    for (uint32_t id = 0; id <= ds->max_id; id++) {
        if (sup[id] > 0) single[sp++] = util[id];
    }
    ctx.topk.k = params->k;
    stats->cuv_threshold = kth_value_desc(single, distinct, params->k);
    ctx.items = (PSOItem *)malloc((distinct ? distinct : 1) * sizeof(*ctx.items));
    if (!ctx.items) {
        free(single); free(twu); free(util); free(maxu); free(sup); free(ctx.rank);
        return -1;
    }
    for (uint32_t id = 0; id <= ds->max_id; id++) {
        if (sup[id] == 0) continue;
        if (twu[id] < stats->cuv_threshold) {
            stats->pruned_twu_items++;
            continue;
        }
        PSOItem it;
        it.id = id;
        it.utility = util[id];
        it.twu = twu[id];
        it.max_utility = maxu[id];
        it.support = sup[id];
        it.avg_utility = sup[id] ? util[id] / (double)sup[id] : 0.0;
        ctx.items[ctx.nitems++] = it;
    }
    qsort(ctx.items, ctx.nitems, sizeof(*ctx.items), cmp_item_utility_desc);
    for (size_t i = 0; i < ctx.nitems; i++) ctx.rank[ctx.items[i].id] = (uint32_t)i;
    ctx.words = (ctx.nitems + 63) / 64;
    if (ctx.words == 0) ctx.words = 1;
    double dev = 0.0;
    for (size_t i = 0; i < ctx.nitems; i++) dev += fabs(ctx.items[i].max_utility - ctx.items[i].avg_utility);
    ctx.deviation = ctx.nitems ? dev / (double)ctx.nitems : 0.0;
    if (explored_init(&ctx.explored, ctx.words, ctx.params.population_size * ctx.params.iterations / 2 + 128) != 0) {
        free(ctx.items); free(single); free(twu); free(util); free(maxu); free(sup); free(ctx.rank);
        return -1;
    }
    if (initialize(&ctx) != 0) {
        explored_free(&ctx.explored); free(ctx.items); free(single); free(twu); free(util); free(maxu); free(sup); free(ctx.rank);
        return -1;
    }
    for (size_t iter = 0; iter < ctx.params.iterations && !limited(&ctx); iter++) {
        int gbest_updated = 0;
        size_t over = 0, under = 0;
        for (size_t j = 0; j < ctx.params.population_size && !limited(&ctx); j++) {
            update_particle(&ctx, &ctx.pop[j], &ctx.pbest[j]);
            if (explored_contains(&ctx.explored, ctx.pop[j].bits)) {
                stats->redundant_particles++;
                continue;
            }
            double est = estimate(&ctx, &ctx.pop[j]);
            if (!(est > ctx.topk.threshold || est > ctx.pbest[j].fitness)) {
                stats->skipped_estimation++;
                if (explored_add(&ctx.explored, ctx.pop[j].bits) != 0) goto fail;
                continue;
            }
            ctx.pop[j].fitness = fitness(&ctx, &ctx.pop[j]);
            if (est < ctx.pop[j].fitness) under++;
            else over++;
            if (ctx.pop[j].fitness > ctx.topk.threshold && add_particle_to_topk(&ctx, &ctx.pop[j]) != 0) goto fail;
            if (ctx.pop[j].fitness > ctx.pbest[j].fitness && particle_copy(&ctx.pbest[j], &ctx.pop[j], ctx.words) != 0) goto fail;
            if (ctx.pop[j].fitness > ctx.gbest.fitness) {
                if (particle_copy(&ctx.gbest, &ctx.pop[j], ctx.words) != 0) goto fail;
                gbest_updated = 1;
            }
            if (explored_add(&ctx.explored, ctx.pop[j].bits) != 0) goto fail;
        }
        if (!gbest_updated) reselect_gbest(&ctx);
        stats->overestimates += over;
        stats->underestimates += under;
        if (over + under > 0) ctx.deviation *= (double)(under + 2) / (double)(over + 2);
    }

    stats->transactions = ds->count;
    stats->distinct_items = distinct;
    stats->kept_items = ctx.nitems;
    stats->k = params->k;
    stats->population_size = ctx.params.population_size;
    stats->iterations = ctx.params.iterations;
    stats->final_threshold = ctx.topk.threshold;
    stats->threshold_raises = ctx.topk.threshold_raises;
    stats->output_count = ctx.topk.count;
    stats->deviation = ctx.deviation;
    stats->explored_particles = ctx.explored.count;
    for (size_t i = 0; i < ctx.topk.count; i++) {
        stats->total_output_items += ctx.topk.data[i].len;
        stats->avg_utility += ctx.topk.data[i].utility;
        if (i == 0) stats->best_utility = ctx.topk.data[i].utility;
    }
    if (ctx.topk.count > 0) {
        stats->avg_utility /= (double)ctx.topk.count;
        stats->avg_length = (double)stats->total_output_items / (double)ctx.topk.count;
    }
    stats->result_ram_bytes = ctx.topk.count * sizeof(PSOPattern) + stats->total_output_items * sizeof(uint32_t);
    stats->result_disk_est_bytes = ctx.topk.count * 32 + stats->total_output_items * 12;
    for (size_t i = 0; i < ctx.params.population_size; i++) {
        particle_free(&ctx.pop[i]);
        particle_free(&ctx.pbest[i]);
    }
    particle_free(&ctx.gbest);
    free(ctx.pop); free(ctx.pbest); topk_free(&ctx.topk); explored_free(&ctx.explored);
    free(ctx.items); free(single); free(twu); free(util); free(maxu); free(sup); free(ctx.rank);
    return 0;

fail:
    for (size_t i = 0; i < ctx.params.population_size; i++) {
        particle_free(&ctx.pop[i]);
        particle_free(&ctx.pbest[i]);
    }
    particle_free(&ctx.gbest);
    free(ctx.pop); free(ctx.pbest); topk_free(&ctx.topk); explored_free(&ctx.explored);
    free(ctx.items); free(single); free(twu); free(util); free(maxu); free(sup); free(ctx.rank);
    return -1;
}
