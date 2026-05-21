/* Minimal pthread stub for MSVC/Windows — maps to Windows synchronization primitives */
#ifndef DM_PTHREAD_STUB_H
#define DM_PTHREAD_STUB_H

#ifdef _WIN32
#include <windows.h>
#include <process.h>

typedef HANDLE pthread_t;
typedef CRITICAL_SECTION pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;
typedef void pthread_mutexattr_t;
typedef void pthread_condattr_t;
typedef void pthread_attr_t;

static inline int pthread_mutex_init(pthread_mutex_t *m, const pthread_mutexattr_t *a) { (void)a; InitializeCriticalSection(m); return 0; }
static inline int pthread_mutex_destroy(pthread_mutex_t *m) { DeleteCriticalSection(m); return 0; }
static inline int pthread_mutex_lock(pthread_mutex_t *m) { EnterCriticalSection(m); return 0; }
static inline int pthread_mutex_unlock(pthread_mutex_t *m) { LeaveCriticalSection(m); return 0; }

static inline int pthread_cond_init(pthread_cond_t *c, const pthread_condattr_t *a) { (void)a; InitializeConditionVariable(c); return 0; }
static inline int pthread_cond_destroy(pthread_cond_t *c) { (void)c; return 0; }
static inline int pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) { SleepConditionVariableCS(c, m, INFINITE); return 0; }
static inline int pthread_cond_signal(pthread_cond_t *c) { WakeConditionVariable(c); return 0; }
static inline int pthread_cond_broadcast(pthread_cond_t *c) { WakeAllConditionVariable(c); return 0; }

typedef struct { void *(*fn)(void *); void *arg; } _PthreadStartArg;
static unsigned __stdcall _pthread_start(void *a) { _PthreadStartArg *p = (_PthreadStartArg *)a; p->fn(p->arg); free(p); return 0; }
static inline int pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*fn)(void *), void *arg) {
    (void)a;
    _PthreadStartArg *sa = (_PthreadStartArg *)malloc(sizeof(*sa));
    if (!sa) return 1;
    sa->fn = fn; sa->arg = arg;
    *t = (HANDLE)_beginthreadex(NULL, 0, _pthread_start, sa, 0, NULL);
    return *t ? 0 : 1;
}
static inline int pthread_join(pthread_t t, void **ret) { (void)ret; WaitForSingleObject(t, INFINITE); CloseHandle(t); return 0; }

#else
#error "This stub is for Windows only"
#endif /* _WIN32 */

#endif /* DM_PTHREAD_STUB_H */
