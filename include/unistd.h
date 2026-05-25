/* Portability shim for unistd.h
 *
 * On Windows (MSVC/MinGW without a real unistd.h), provide the minimal
 * subset that dm needs.  On POSIX systems, just pull in the real header.
 */
#ifndef DM_UNISTD_SHIM_H
#define DM_UNISTD_SHIM_H

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
#else
/* On Linux/macOS the system unistd.h provides everything (sysconf, etc.).
 * Use the compiler's include_next to skip this shim and reach the real one. */
#include_next <unistd.h>
#endif /* _WIN32 */

#endif /* DM_UNISTD_SHIM_H */
