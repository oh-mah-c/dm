#ifndef DM_THREADPOOL_H
#define DM_THREADPOOL_H

#include <stddef.h>
#include <stdint.h>
#include "pthread.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*DM_TaskFn)(void *arg);

typedef struct DM_Task {
    DM_TaskFn fn;
    void *arg;
    struct DM_Task *next;
} DM_Task;

typedef struct DM_ThreadPool {
    pthread_t *threads;
    int num_threads;

    DM_Task *queue_head;
    DM_Task *queue_tail;
    size_t queue_size;

    size_t active_tasks;
    int stop;

    pthread_mutex_t lock;
    pthread_cond_t notify_worker;
    pthread_cond_t notify_idle;
} DM_ThreadPool;

/**
 * @brief RAII constructor: creates a worker thread pool.
 * @param num_threads Number of worker threads (if <= 0, defaults to dm_get_num_threads()).
 * @return Allocated thread pool, or NULL on error.
 */
DM_ThreadPool* dm_threadpool_create(int num_threads);

/**
 * @brief RAII destructor: gracefully stops all workers, waits for pending tasks,
 *        joins threads, and releases all resources.
 */
void dm_threadpool_destroy(DM_ThreadPool *pool);

/**
 * @brief Auto-cleanup helper for GNU C RAII scope:
 *        __attribute__((cleanup(dm_threadpool_auto_cleanup))) DM_ThreadPool *pool = ...
 */
static inline void dm_threadpool_auto_cleanup(DM_ThreadPool **p) {
    if (p && *p) {
        dm_threadpool_destroy(*p);
        *p = NULL;
    }
}

/**
 * @brief Submits a work task to the pool.
 * @return 0 on success, negative on error.
 */
int dm_threadpool_submit(DM_ThreadPool *pool, DM_TaskFn fn, void *arg);

/**
 * @brief Blocks until the task queue is completely empty AND all active workers are idle.
 */
void dm_threadpool_wait(DM_ThreadPool *pool);

/**
 * @brief Returns the configured thread count (from CLI, DM_NUM_THREADS env, or CPU count).
 */
int dm_get_num_threads(void);

/**
 * @brief Sets the global default thread count override.
 */
void dm_set_num_threads(int num_threads);

/**
 * @brief Parses and strips --threads <N> / -t <N> from argc/argv.
 * @return Parsed thread count (or 0 if not specified on CLI).
 */
int dm_parse_thread_args(int *argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif /* DM_THREADPOOL_H */
