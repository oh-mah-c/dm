/* Minimal unistd stub for MSVC/Windows */
#ifndef DM_UNISTD_STUB_H
#define DM_UNISTD_STUB_H

#ifdef _WIN32
#include <io.h>
#include <process.h>
#include <direct.h>
#include <sys/types.h>
#include <basetsd.h>
typedef SSIZE_T ssize_t;
#define STDIN_FILENO  0
#define STDOUT_FILENO 1
#define STDERR_FILENO 2
static inline int usleep(unsigned int us) { Sleep((us + 999) / 1000); return 0; }
#endif /* _WIN32 */

#endif /* DM_UNISTD_STUB_H */
