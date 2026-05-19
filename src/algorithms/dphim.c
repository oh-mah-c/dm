#include "algorithms/dphim.h"
#include "core/dm_benchmark.h"
#include "core/dm_dataset_types.h"
#include "core/dm_portability.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    uint32_t tid;
    double iutil;
    double rutil;
} DPHIMTuple;

typedef struct {
    uint32_t *items;
    size_t count;
    DPHIMTuple *tuples;
    size_t tuple_count;
    size_t tuple_capacity;
    double sum_iutil;
    double sum_rutil;
} DPHIMList;

typedef struct DPHIMTask {
    DPHIMList *p;
    DPHIMList *px;
    DPHIMList **suffix;
    size_t suffix_count;
    size_t depth;
    struct DPHIMTask *next;
} DPHIMTask;

typedef struct {
    DPHIMTask *head;
    size_t active_workers;
    int stop;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
} DPHIMQueue;

typedef struct {
    DPHIMQueue queue;
    double min_util;
    size_t hui_count;
    size_t total_items;
    size_t generated_tasks;
    pthread_mutex_t result_mutex;
} DPHIMContext;

typedef struct {
    uint32_t id;
    double twu;
} DPHIMItemTWU;

static uint32_t *g_rank = NULL;

static int cmp_item_twu(const void *a, const void *b) {
    const DPHIMItemTWU *ia = (const DPHIMItemTWU *)a;
    const DPHIMItemTWU *ib = (const DPHIMItemTWU *)b;
    if (ia->twu < ib->twu) return -1;
    if (ia->twu > ib->twu) return 1;
    return (ia->id < ib->id) ? -1 : ((ia->id > ib->id) ? 1 : 0);
}

static void list_tuple_add(DPHIMList *list, uint32_t tid, double iutil, double rutil) {
    if (list->tuple_count >= list->tuple_capacity) {
        list->tuple_capacity = list->tuple_capacity ? list->tuple_capacity * 2 : 8;
        list->tuples = realloc(list->tuples, list->tuple_capacity * sizeof(DPHIMTuple));
    }
    list->tuples[list->tuple_count].tid = tid;
    list->tuples[list->tuple_count].iutil = iutil;
    list->tuples[list->tuple_count].rutil = rutil;
    list->tuple_count++;
    list->sum_iutil += iutil;
    list->sum_rutil += rutil;
}

static DPHIMList *list_new_single(uint32_t item) {
    DPHIMList *list = calloc(1, sizeof(DPHIMList));
    if (!list) return NULL;
    list->items = malloc(sizeof(uint32_t));
    if (!list->items) {
        free(list);
        return NULL;
    }
    list->items[0] = item;
    list->count = 1;
    return list;
}

static void list_free(DPHIMList *list) {
    if (!list) return;
    free(list->items);
    free(list->tuples);
    free(list);
}

static DPHIMList *construct_list(DPHIMList *p, DPHIMList *px, DPHIMList *py) {
    DPHIMList *pxy = calloc(1, sizeof(DPHIMList));
    if (!pxy) return NULL;
    pxy->count = px->count + 1;
    pxy->items = malloc(pxy->count * sizeof(uint32_t));
    if (!pxy->items) {
        free(pxy);
        return NULL;
    }
    memcpy(pxy->items, px->items, px->count * sizeof(uint32_t));
    pxy->items[px->count] = py->items[py->count - 1];
    pxy->tuple_capacity = px->tuple_count < py->tuple_count ? px->tuple_count : py->tuple_count;
    pxy->tuples = pxy->tuple_capacity ? malloc(pxy->tuple_capacity * sizeof(DPHIMTuple)) : NULL;

    size_t ix = 0;
    size_t iy = 0;
    size_t ip = 0;
    while (ix < px->tuple_count && iy < py->tuple_count) {
        if (px->tuples[ix].tid == py->tuples[iy].tid) {
            uint32_t tid = px->tuples[ix].tid;
            double iutil = px->tuples[ix].iutil + py->tuples[iy].iutil;
            if (p) {
                while (ip < p->tuple_count && p->tuples[ip].tid < tid) ip++;
                if (ip < p->tuple_count && p->tuples[ip].tid == tid) {
                    iutil -= p->tuples[ip].iutil;
                }
            }
            list_tuple_add(pxy, tid, iutil, py->tuples[iy].rutil);
            ix++;
            iy++;
        } else if (px->tuples[ix].tid < py->tuples[iy].tid) {
            ix++;
        } else {
            iy++;
        }
    }
    return pxy;
}

static void queue_init(DPHIMQueue *q) {
    memset(q, 0, sizeof(*q));
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->cond, NULL);
}

static void queue_destroy(DPHIMQueue *q) {
    pthread_mutex_destroy(&q->mutex);
    pthread_cond_destroy(&q->cond);
}

static void task_free(DPHIMTask *task) {
    if (!task) return;
    (void)task->p;
    (void)task->px;
    free(task->suffix);
    free(task);
}

static void queue_push(DPHIMContext *ctx, DPHIMTask *task) {
    pthread_mutex_lock(&ctx->queue.mutex);
    task->next = ctx->queue.head;
    ctx->queue.head = task;
    ctx->generated_tasks++;
    pthread_cond_signal(&ctx->queue.cond);
    pthread_mutex_unlock(&ctx->queue.mutex);
}

static DPHIMTask *queue_pop(DPHIMContext *ctx) {
    DPHIMQueue *q = &ctx->queue;
    pthread_mutex_lock(&q->mutex);
    while (!q->head && !q->stop) {
        if (q->active_workers == 0) {
            q->stop = 1;
            pthread_cond_broadcast(&q->cond);
            break;
        }
        pthread_cond_wait(&q->cond, &q->mutex);
    }

    DPHIMTask *task = NULL;
    if (!q->stop && q->head) {
        task = q->head;
        q->head = task->next;
        task->next = NULL;
        q->active_workers++;
    }
    pthread_mutex_unlock(&q->mutex);
    return task;
}

static void queue_task_done(DPHIMContext *ctx) {
    DPHIMQueue *q = &ctx->queue;
    pthread_mutex_lock(&q->mutex);
    if (q->active_workers > 0) q->active_workers--;
    if (!q->head && q->active_workers == 0) {
        q->stop = 1;
        pthread_cond_broadcast(&q->cond);
    }
    pthread_mutex_unlock(&q->mutex);
}

static void record_hui(DPHIMContext *ctx, size_t len) {
    pthread_mutex_lock(&ctx->result_mutex);
    ctx->hui_count++;
    ctx->total_items += len;
    pthread_mutex_unlock(&ctx->result_mutex);
}

static void mine_local(DPHIMContext *ctx, DPHIMList *p, DPHIMList **extensions, size_t ext_count) {
    for (size_t i = 0; i < ext_count; i++) {
        DPHIMList *px = extensions[i];
        if (px->sum_iutil >= ctx->min_util) record_hui(ctx, px->count);
        if (px->sum_iutil + px->sum_rutil >= ctx->min_util) {
            size_t child_count = ext_count - i - 1;
            DPHIMList **children = child_count ? malloc(child_count * sizeof(DPHIMList *)) : NULL;
            size_t actual = 0;
            for (size_t j = i + 1; j < ext_count; j++) {
                DPHIMList *child = construct_list(p, px, extensions[j]);
                if (child && child->tuple_count > 0) children[actual++] = child;
                else list_free(child);
            }
            if (actual > 0) mine_local(ctx, px, children, actual);
            for (size_t j = 0; j < actual; j++) list_free(children[j]);
            free(children);
        }
    }
}

static void process_task(DPHIMContext *ctx, DPHIMTask *task) {
    DPHIMList *px = task->px;
    if (px->sum_iutil >= ctx->min_util) record_hui(ctx, px->count);
    if (px->sum_iutil + px->sum_rutil >= ctx->min_util && task->suffix_count > 0) {
        DPHIMList **children = malloc(task->suffix_count * sizeof(DPHIMList *));
        size_t child_count = 0;
        for (size_t i = 0; i < task->suffix_count; i++) {
            DPHIMList *child = construct_list(task->p, px, task->suffix[i]);
            if (child && child->tuple_count > 0) children[child_count++] = child;
            else list_free(child);
        }
        if (child_count > 0) mine_local(ctx, px, children, child_count);
        for (size_t i = 0; i < child_count; i++) list_free(children[i]);
        free(children);
    }
    task_free(task);
}

static void *worker_main(void *arg) {
    DPHIMContext *ctx = (DPHIMContext *)arg;
    for (;;) {
        DPHIMTask *task = queue_pop(ctx);
        if (!task) break;
        process_task(ctx, task);
        queue_task_done(ctx);
    }
    return NULL;
}

static int cmp_trans_item_rank(const void *a, const void *b) {
    const DM_Item *ia = (const DM_Item *)a;
    const DM_Item *ib = (const DM_Item *)b;
    uint32_t ra = g_rank[ia->id];
    uint32_t rb = g_rank[ib->id];
    if (ra < rb) return -1;
    if (ra > rb) return 1;
    return 0;
}

static DM_Status run(DM_Dataset *ds, void *params) {
    if (!ds || ds->type != DM_TYPE_UTILITY || !ds->payload) return DM_ERROR_INCOMPATIBLE;

    DM_DPHIM_Params *p = (DM_DPHIM_Params *)params;
    double min_util = p ? p->min_utility : 1000.0;
    int threads = p && p->threads > 0 ? p->threads : (int)dm_cpu_count();
    if (threads < 1) threads = 1;
    if (threads > 128) threads = 128;

    DM_Trans_Utility *data = (DM_Trans_Utility *)ds->payload;
    double *twu = calloc((size_t)ds->max_id + 1, sizeof(double));
    for (size_t i = 0; i < ds->count; i++) {
        for (size_t j = 0; j < data[i].count; j++) {
            twu[data[i].items[j].id] += data[i].total_utility;
        }
    }

    DPHIMItemTWU *items = malloc(((size_t)ds->max_id + 1) * sizeof(DPHIMItemTWU));
    size_t item_count = 0;
    for (uint32_t i = 0; i <= ds->max_id; i++) {
        if (twu[i] >= min_util) {
            items[item_count].id = i;
            items[item_count].twu = twu[i];
            item_count++;
        }
    }
    qsort(items, item_count, sizeof(DPHIMItemTWU), cmp_item_twu);

    g_rank = malloc(((size_t)ds->max_id + 1) * sizeof(uint32_t));
    memset(g_rank, 0xFF, ((size_t)ds->max_id + 1) * sizeof(uint32_t));
    DPHIMList **initial = malloc(item_count * sizeof(DPHIMList *));
    for (size_t i = 0; i < item_count; i++) {
        g_rank[items[i].id] = (uint32_t)i;
        initial[i] = list_new_single(items[i].id);
    }

    for (size_t i = 0; i < ds->count; i++) {
        DM_Item *filtered = malloc(data[i].count * sizeof(DM_Item));
        size_t fcount = 0;
        for (size_t j = 0; j < data[i].count; j++) {
            if (g_rank[data[i].items[j].id] != 0xFFFFFFFFu) {
                filtered[fcount++] = data[i].items[j];
            }
        }
        qsort(filtered, fcount, sizeof(DM_Item), cmp_trans_item_rank);

        double remaining = 0.0;
        for (size_t j = fcount; j-- > 0;) {
            uint32_t idx = g_rank[filtered[j].id];
            list_tuple_add(initial[idx], (uint32_t)i, filtered[j].utility, remaining);
            remaining += filtered[j].utility;
        }
        free(filtered);
    }

    DPHIMContext ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.min_util = min_util;
    queue_init(&ctx.queue);
    pthread_mutex_init(&ctx.result_mutex, NULL);

    for (size_t i = 0; i < item_count; i++) {
        DPHIMTask *task = calloc(1, sizeof(DPHIMTask));
        task->px = initial[i];
        task->suffix_count = item_count - i - 1;
        if (task->suffix_count > 0) {
            task->suffix = malloc(task->suffix_count * sizeof(DPHIMList *));
            for (size_t j = 0; j < task->suffix_count; j++) task->suffix[j] = initial[i + 1 + j];
        }
        queue_push(&ctx, task);
    }

    printf("[DPHIM] Starting dynamic parallel HUIM on %zu transactions, min_util=%.6g, threads=%d\n",
           ds->count, min_util, threads);

    pthread_t *workers = malloc((size_t)threads * sizeof(pthread_t));
    for (int i = 0; i < threads; i++) pthread_create(&workers[i], NULL, worker_main, &ctx);
    for (int i = 0; i < threads; i++) pthread_join(workers[i], NULL);

    printf("[DPHIM] Complete. HUIs=%zu, generated_tasks=%zu\n", ctx.hui_count, ctx.generated_tasks);
    dm_bench_record_results(ctx.hui_count, ctx.total_items);

    for (size_t i = 0; i < item_count; i++) list_free(initial[i]);
    free(workers);
    free(initial);
    free(items);
    free(twu);
    free(g_rank);
    g_rank = NULL;
    pthread_mutex_destroy(&ctx.result_mutex);
    queue_destroy(&ctx.queue);
    return DM_SUCCESS;
}

DM_Algorithm dphim_algo = {
    .id = "dphim",
    .name = "DPHIM",
    .description = "Dynamic Parallelization for High-utility Itemset Mining using utility-list subtasks.",
    .supported_types = (1 << DM_TYPE_UTILITY),
    .run = run
};

DM_REGISTER_ALGORITHM(dphim_algo)
