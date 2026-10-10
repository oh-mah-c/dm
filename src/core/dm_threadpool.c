#include "core/dm_threadpool.h"
#include "core/dm_portability.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_configured_threads = 0;

int dm_get_num_threads(void) {
    if (g_configured_threads > 0) {
        return g_configured_threads;
    }
    const char *env = getenv("DM_NUM_THREADS");
    if (!env) env = getenv("OMP_NUM_THREADS");
    if (env) {
        int val = atoi(env);
        if (val > 0) {
            if (val > 256) val = 256;
            return val;
        }
    }
    long count = dm_cpu_count();
    if (count < 1) count = 1;
    if (count > 256) count = 256;
    return (int)count;
}

void dm_set_num_threads(int num_threads) {
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 256) num_threads = 256;
    g_configured_threads = num_threads;
}

int dm_parse_thread_args(int *argc, char **argv) {
    if (!argc || !argv) return 0;
    int parsed = 0;
    int i = 1;
    while (i < *argc) {
        if (strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0) {
            if (i + 1 < *argc) {
                int val = atoi(argv[i + 1]);
                if (val > 0) {
                    dm_set_num_threads(val);
                    parsed = val;
                }
                /* Shift arguments to remove --threads <N> */
                for (int j = i; j + 2 <= *argc; j++) {
                    argv[j] = argv[j + 2];
                }
                *argc -= 2;
                continue;
            }
        } else if (strncmp(argv[i], "--threads=", 10) == 0) {
            int val = atoi(argv[i] + 10);
            if (val > 0) {
                dm_set_num_threads(val);
                parsed = val;
            }
            for (int j = i; j + 1 <= *argc; j++) {
                argv[j] = argv[j + 1];
            }
            *argc -= 1;
            continue;
        }
        i++;
    }
    return parsed;
}

static void *worker_thread_main(void *arg) {
    DM_ThreadPool *pool = (DM_ThreadPool *)arg;
    while (1) {
        pthread_mutex_lock(&pool->lock);
        while (pool->queue_head == NULL && !pool->stop) {
            pthread_cond_wait(&pool->notify_worker, &pool->lock);
        }
        if (pool->stop && pool->queue_head == NULL) {
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        DM_Task *task = pool->queue_head;
        if (task) {
            pool->queue_head = task->next;
            if (pool->queue_head == NULL) {
                pool->queue_tail = NULL;
            }
            pool->queue_size--;
            pool->active_tasks++;
        }
        pthread_mutex_unlock(&pool->lock);

        if (task) {
            if (task->fn) {
                task->fn(task->arg);
            }
            free(task);

            pthread_mutex_lock(&pool->lock);
            pool->active_tasks--;
            if (pool->queue_size == 0 && pool->active_tasks == 0) {
                pthread_cond_broadcast(&pool->notify_idle);
            }
            pthread_mutex_unlock(&pool->lock);
        }
    }
    return NULL;
}

DM_ThreadPool *dm_threadpool_create(int num_threads) {
    if (num_threads <= 0) {
        num_threads = dm_get_num_threads();
    }
    if (num_threads < 1) num_threads = 1;
    if (num_threads > 256) num_threads = 256;

    DM_ThreadPool *pool = (DM_ThreadPool *)calloc(1, sizeof(DM_ThreadPool));
    if (!pool) return NULL;

    pool->num_threads = num_threads;
    pool->threads = (pthread_t *)malloc((size_t)num_threads * sizeof(pthread_t));
    if (!pool->threads) {
        free(pool);
        return NULL;
    }

    if (pthread_mutex_init(&pool->lock, NULL) != 0) {
        free(pool->threads);
        free(pool);
        return NULL;
    }
    if (pthread_cond_init(&pool->notify_worker, NULL) != 0) {
        pthread_mutex_destroy(&pool->lock);
        free(pool->threads);
        free(pool);
        return NULL;
    }
    if (pthread_cond_init(&pool->notify_idle, NULL) != 0) {
        pthread_cond_destroy(&pool->notify_worker);
        pthread_mutex_destroy(&pool->lock);
        free(pool->threads);
        free(pool);
        return NULL;
    }

    int created = 0;
    for (int i = 0; i < num_threads; i++) {
        if (pthread_create(&pool->threads[i], NULL, worker_thread_main, pool) != 0) {
            pool->stop = 1;
            pthread_cond_broadcast(&pool->notify_worker);
            for (int j = 0; j < created; j++) {
                pthread_join(pool->threads[j], NULL);
            }
            pthread_cond_destroy(&pool->notify_idle);
            pthread_cond_destroy(&pool->notify_worker);
            pthread_mutex_destroy(&pool->lock);
            free(pool->threads);
            free(pool);
            return NULL;
        }
        created++;
    }

    return pool;
}

void dm_threadpool_destroy(DM_ThreadPool *pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->lock);
    pool->stop = 1;
    pthread_cond_broadcast(&pool->notify_worker);
    pthread_mutex_unlock(&pool->lock);

    for (int i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    /* Free any unexecuted tasks remaining in queue */
    DM_Task *curr = pool->queue_head;
    while (curr) {
        DM_Task *next = curr->next;
        free(curr);
        curr = next;
    }

    pthread_cond_destroy(&pool->notify_idle);
    pthread_cond_destroy(&pool->notify_worker);
    pthread_mutex_destroy(&pool->lock);

    free(pool->threads);
    free(pool);
}

int dm_threadpool_submit(DM_ThreadPool *pool, DM_TaskFn fn, void *arg) {
    if (!pool || !fn) return -1;

    DM_Task *task = (DM_Task *)malloc(sizeof(DM_Task));
    if (!task) return -1;
    task->fn = fn;
    task->arg = arg;
    task->next = NULL;

    pthread_mutex_lock(&pool->lock);
    if (pool->stop) {
        pthread_mutex_unlock(&pool->lock);
        free(task);
        return -1;
    }

    if (pool->queue_tail) {
        pool->queue_tail->next = task;
        pool->queue_tail = task;
    } else {
        pool->queue_head = task;
        pool->queue_tail = task;
    }
    pool->queue_size++;

    pthread_cond_signal(&pool->notify_worker);
    pthread_mutex_unlock(&pool->lock);
    return 0;
}

void dm_threadpool_wait(DM_ThreadPool *pool) {
    if (!pool) return;
    pthread_mutex_lock(&pool->lock);
    while (pool->queue_size > 0 || pool->active_tasks > 0) {
        pthread_cond_wait(&pool->notify_idle, &pool->lock);
    }
    pthread_mutex_unlock(&pool->lock);
}
