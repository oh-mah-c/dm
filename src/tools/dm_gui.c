/*
 * dm_gui: local web interface for dm.exe, in the spirit of the SPMF graphical interface
 * (choose an algorithm, an input file, an output file and parameters, run, read the log and
 * the output).
 *
 * The server listens on 127.0.0.1 only. Runs are executed with execv (no shell) and every path
 * given by the page must resolve inside one of the allowed directories (the root directory
 * and the --allow directories).
 *
 *   ./build/bin/dm_gui [--port 8765] [--root .] [--dm ./build/bin/dm.exe] [--static gui]
 *                      [--allow <dir>]...
 *   then open http://127.0.0.1:8765/
 *
 * HTTP API (used by gui/index.html):
 *   GET  /api/algorithms                  -> [{"id","name","desc"}] (parsed from "dm.exe" usage)
 *   GET  /api/ls?dir=<path>               -> {"dir", "parent", "entries":[{"name","dir","size"}]}
 *   GET  /api/file?path=<p>&max=<bytes>   -> text/plain (first bytes of the file)
 *   POST /api/run   (form: algo, input, type, args, env)   -> {"id"}
 *   GET  /api/job?id=<n>&offset=<k>       -> {"running","exit","elapsed","log","offset"}
 *   POST /api/stop?id=<n>                 -> {"ok"}
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <dirent.h>
#include <limits.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define MAX_ALLOW 16
#define MAX_JOBS 256
#define MAX_ARGS 32

static char g_root[PATH_MAX], g_dm[PATH_MAX], g_static[PATH_MAX], g_jobdir[PATH_MAX];
static char g_allow[MAX_ALLOW][PATH_MAX];
static int g_nallow;

typedef struct { pid_t pid; int running, exit_code; struct timespec t0, t1; char log[PATH_MAX]; } Job;
static Job g_jobs[MAX_JOBS];
static int g_njobs;

/* ---------- small string buffer ---------- */
typedef struct { char *s; size_t n, cap; } Buf;
static void bput(Buf *b, const char *p, size_t n) {
    if (b->n + n + 1 > b->cap) {
        b->cap = (b->n + n + 1) * 2;
        b->s = realloc(b->s, b->cap);
        if (!b->s) { perror("realloc"); exit(1); }
    }
    memcpy(b->s + b->n, p, n); b->n += n; b->s[b->n] = 0;
}
static void bstr(Buf *b, const char *s) { bput(b, s, strlen(s)); }
static void bprintf(Buf *b, const char *fmt, ...) {
    char tmp[1024]; va_list ap; va_start(ap, fmt);
    int n = vsnprintf(tmp, sizeof tmp, fmt, ap); va_end(ap);
    if (n > 0) bput(b, tmp, (size_t)n < sizeof tmp ? (size_t)n : sizeof tmp - 1);
}
static void bjson(Buf *b, const char *s, size_t n) {   /* JSON string literal */
    bstr(b, "\"");
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { char e[2] = { '\\', (char)c }; bput(b, e, 2); }
        else if (c == '\n') bstr(b, "\\n");
        else if (c == '\r') bstr(b, "\\r");
        else if (c == '\t') bstr(b, "\\t");
        else if (c < 0x20) bprintf(b, "\\u%04x", c);
        else bput(b, (const char *)&c, 1);
    }
    bstr(b, "\"");
}

/* ---------- HTTP ---------- */
static void send_all(int fd, const char *p, size_t n) {
    while (n > 0) { ssize_t w = send(fd, p, n, MSG_NOSIGNAL); if (w <= 0) return; p += w; n -= (size_t)w; }
}
static void reply(int fd, int code, const char *type, const char *body, size_t n) {
    char h[256];
    const char *msg = code == 200 ? "OK" : code == 404 ? "Not Found" : code == 403 ? "Forbidden" : "Bad Request";
    int hn = snprintf(h, sizeof h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
                      "Cache-Control: no-store\r\nConnection: close\r\n\r\n", code, msg, type, n);
    send_all(fd, h, (size_t)hn);
    send_all(fd, body, n);
}
static void reply_json(int fd, Buf *b) { reply(fd, 200, "application/json; charset=utf-8", b->s ? b->s : "{}", b->s ? b->n : 2); }
static void reply_err(int fd, int code, const char *msg) {
    Buf b = {0}; bstr(&b, "{\"error\":"); bjson(&b, msg, strlen(msg)); bstr(&b, "}");
    reply(fd, code, "application/json; charset=utf-8", b.s, b.n); free(b.s);
}

static int hexv(int c) { return isdigit(c) ? c - '0' : (tolower(c) >= 'a' && tolower(c) <= 'f') ? tolower(c) - 'a' + 10 : -1; }
static void url_decode(char *s) {
    char *o = s;
    for (; *s; s++) {
        if (*s == '+') *o++ = ' ';
        else if (*s == '%' && hexv(s[1]) >= 0 && hexv(s[2]) >= 0) { *o++ = (char)(hexv(s[1]) * 16 + hexv(s[2])); s += 2; }
        else *o++ = *s;
    }
    *o = 0;
}
/* value of key in "a=1&b=2" (decoded copy, caller frees) */
static char *form_get(const char *q, const char *key) {
    if (!q) return NULL;
    size_t kl = strlen(key);
    for (const char *p = q; *p; ) {
        const char *amp = strchr(p, '&'); size_t len = amp ? (size_t)(amp - p) : strlen(p);
        if (len > kl && strncmp(p, key, kl) == 0 && p[kl] == '=') {
            char *v = strndup(p + kl + 1, len - kl - 1); url_decode(v); return v;
        }
        if (!amp) break;
        p = amp + 1;
    }
    return NULL;
}

/* ---------- paths ---------- */
static int under(const char *path, const char *dir) {
    size_t n = strlen(dir);
    return strncmp(path, dir, n) == 0 && (path[n] == 0 || path[n] == '/' || (n == 1 && dir[0] == '/'));
}
/* resolves p (relative to root) and checks it lies in an allowed directory */
static int safe_path(const char *p, char *out) {
    char tmp[PATH_MAX];
    if (!p || !*p) return 0;
    if (p[0] == '/') snprintf(tmp, sizeof tmp, "%s", p); else snprintf(tmp, sizeof tmp, "%s/%s", g_root, p);
    if (!realpath(tmp, out)) return 0;
    for (int i = 0; i < g_nallow; i++) if (under(out, g_allow[i])) return 1;
    return 0;
}

/* ---------- API ---------- */
static void api_algorithms(int fd) {
    int pipefd[2];
    if (pipe(pipefd) != 0) { reply_err(fd, 400, "pipe failed"); return; }
    pid_t pid = fork();
    if (pid == 0) {
        dup2(pipefd[1], 1); dup2(pipefd[1], 2); close(pipefd[0]); close(pipefd[1]);
        if (chdir(g_root) != 0) _exit(127);
        execl(g_dm, g_dm, (char *)NULL); _exit(127);
    }
    close(pipefd[1]);
    Buf out = {0}; char tmp[4096]; ssize_t r;
    while ((r = read(pipefd[0], tmp, sizeof tmp)) > 0) bput(&out, tmp, (size_t)r);
    close(pipefd[0]); waitpid(pid, NULL, 0);
    Buf b = {0}; bstr(&b, "["); int first = 1;
    char *save = NULL, *id = NULL, *name = NULL;
    for (char *line = out.s ? strtok_r(out.s, "\n", &save) : NULL; line; line = strtok_r(NULL, "\n", &save)) {
        if (line[0] == '[') {
            char *close_b = strchr(line, ']');
            if (!close_b) continue;
            *close_b = 0; id = line + 1; name = close_b + 1; while (*name == ' ') name++;
        } else if (id && strncmp(line, "    Desc: ", 10) == 0) {
            if (!first) bstr(&b, ",");
            first = 0;
            bstr(&b, "{\"id\":"); bjson(&b, id, strlen(id));
            bstr(&b, ",\"name\":"); bjson(&b, name, strlen(name));
            bstr(&b, ",\"desc\":"); bjson(&b, line + 10, strlen(line + 10)); bstr(&b, "}");
            id = NULL;
        }
    }
    bstr(&b, "]");
    reply_json(fd, &b); free(b.s); free(out.s);
}

static int cmp_names(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static void api_ls(int fd, const char *q) {
    char *d = form_get(q, "dir"); char path[PATH_MAX];
    if (!safe_path(d && *d ? d : ".", path)) { free(d); reply_err(fd, 403, "directory not allowed"); return; }
    free(d);
    DIR *dir = opendir(path);
    if (!dir) { reply_err(fd, 404, "cannot open directory"); return; }
    char **names = NULL; size_t n = 0, cap = 0; struct dirent *e;
    while ((e = readdir(dir))) {
        if (e->d_name[0] == '.') continue;
        if (n == cap) { cap = cap ? 2 * cap : 64; names = realloc(names, cap * sizeof *names); }
        names[n++] = strdup(e->d_name);
    }
    closedir(dir);
    qsort(names, n, sizeof *names, cmp_names);
    Buf b = {0}; bstr(&b, "{\"dir\":"); bjson(&b, path, strlen(path));
    char parent[PATH_MAX]; snprintf(parent, sizeof parent, "%s/..", path);
    char rp[PATH_MAX]; int has_parent = 0;
    if (realpath(parent, rp)) for (int i = 0; i < g_nallow; i++) if (under(rp, g_allow[i])) has_parent = 1;
    bstr(&b, ",\"parent\":"); if (has_parent) bjson(&b, rp, strlen(rp)); else bstr(&b, "null");
    bstr(&b, ",\"entries\":[");
    for (size_t i = 0; i < n; i++) {
        char full[PATH_MAX]; struct stat st;
        snprintf(full, sizeof full, "%s/%s", path, names[i]);
        int isdir = stat(full, &st) == 0 && S_ISDIR(st.st_mode);
        if (i) bstr(&b, ",");
        bstr(&b, "{\"name\":"); bjson(&b, names[i], strlen(names[i]));
        bprintf(&b, ",\"dir\":%s,\"size\":%lld}", isdir ? "true" : "false", isdir ? 0LL : (long long)st.st_size);
        free(names[i]);
    }
    free(names);
    bstr(&b, "]}");
    reply_json(fd, &b); free(b.s);
}

static void api_file(int fd, const char *q) {
    char *p = form_get(q, "path"), *mx = form_get(q, "max"); char path[PATH_MAX];
    long max = mx ? atol(mx) : 2000000; free(mx);
    if (max <= 0 || max > 50000000) max = 2000000;
    if (!safe_path(p, path)) { free(p); reply_err(fd, 403, "file not allowed"); return; }
    free(p);
    FILE *f = fopen(path, "rb");
    if (!f) { reply_err(fd, 404, "cannot open file"); return; }
    char *buf = malloc((size_t)max); size_t n = fread(buf, 1, (size_t)max, f); fclose(f);
    reply(fd, 200, "text/plain; charset=utf-8", buf, n); free(buf);
}

static int valid_env_name(const char *s, size_t n) {
    if (n == 0 || n > 64) return 0;
    for (size_t i = 0; i < n; i++) if (!(isupper((unsigned char)s[i]) || isdigit((unsigned char)s[i]) || s[i] == '_')) return 0;
    return 1;
}

static void api_run(int fd, const char *body) {
    if (g_njobs >= MAX_JOBS) { reply_err(fd, 400, "too many jobs, restart dm_gui"); return; }
    char *algo = form_get(body, "algo"), *input = form_get(body, "input"), *type = form_get(body, "type");
    char *args = form_get(body, "args"), *env = form_get(body, "env");
    char inpath[PATH_MAX];
    char *argv[MAX_ARGS + 6]; int ac = 0;
    int ok = algo && *algo && strlen(algo) < 64;
    for (char *c = algo; ok && *c; c++) if (!(isalnum((unsigned char)*c) || *c == '_' || *c == '-')) ok = 0;
    if (!ok) { reply_err(fd, 400, "invalid algorithm id"); goto done; }
    if (!safe_path(input, inpath)) { reply_err(fd, 403, "input file not allowed or missing"); goto done; }
    argv[ac++] = g_dm; argv[ac++] = algo; argv[ac++] = inpath; argv[ac++] = (type && *type) ? type : "0";
    char *save = NULL;
    for (char *t = args ? strtok_r(args, " \t\r\n", &save) : NULL; t; t = strtok_r(NULL, " \t\r\n", &save)) {
        if (ac >= MAX_ARGS) { reply_err(fd, 400, "too many arguments"); goto done; }
        argv[ac++] = t;
    }
    argv[ac] = NULL;
    Job *j = &g_jobs[g_njobs];
    snprintf(j->log, sizeof j->log, "%s/job_%d.log", g_jobdir, g_njobs);
    int logfd = open(j->log, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (logfd < 0) { reply_err(fd, 400, "cannot create log file"); goto done; }
    clock_gettime(CLOCK_MONOTONIC, &j->t0);
    pid_t pid = fork();
    if (pid == 0) {
        dup2(logfd, 1); dup2(logfd, 2); close(logfd); close(fd);
        if (chdir(g_root) != 0) _exit(127);
        char *esave = NULL;
        for (char *l = env ? strtok_r(env, "\r\n", &esave) : NULL; l; l = strtok_r(NULL, "\r\n", &esave)) {
            char *eq = strchr(l, '=');
            if (!eq || !valid_env_name(l, (size_t)(eq - l))) continue;
            *eq = 0;
            /* output files must also stay inside the allowed directories */
            if (strstr(l, "_OUT")) {
                char op[PATH_MAX], dirpart[PATH_MAX], rp[PATH_MAX];
                snprintf(op, sizeof op, "%s", eq + 1);
                if (op[0] == 0) continue;
                char *slash = strrchr(op, '/');
                if (slash) { snprintf(dirpart, sizeof dirpart, "%.*s", (int)(slash - op), op); if (!dirpart[0]) strcpy(dirpart, "/"); }
                else strcpy(dirpart, ".");
                char full[PATH_MAX];
                if (dirpart[0] == '/') snprintf(full, sizeof full, "%s", dirpart); else snprintf(full, sizeof full, "%s/%s", g_root, dirpart);
                int allowed = 0;
                if (realpath(full, rp)) for (int i = 0; i < g_nallow; i++) if (under(rp, g_allow[i])) allowed = 1;
                if (!allowed) { fprintf(stderr, "[dm_gui] output path not allowed: %s\n", op); continue; }
            }
            setenv(l, eq + 1, 1);
        }
        printf("$");
        for (int i = 0; argv[i]; i++) printf(" %s", argv[i]);
        printf("\n"); fflush(stdout);
        execv(g_dm, argv);
        perror("execv"); _exit(127);
    }
    close(logfd);
    if (pid < 0) { reply_err(fd, 400, "fork failed"); goto done; }
    j->pid = pid; j->running = 1; j->exit_code = -1;
    Buf b = {0}; bprintf(&b, "{\"id\":%d}", g_njobs++);
    reply_json(fd, &b); free(b.s);
done:
    free(algo); free(input); free(type); free(args); free(env);
}

static void poll_job(Job *j) {
    if (!j->running) return;
    int st;
    if (waitpid(j->pid, &st, WNOHANG) == j->pid) {
        j->running = 0;
        j->exit_code = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + (WIFSIGNALED(st) ? WTERMSIG(st) : 0);
        clock_gettime(CLOCK_MONOTONIC, &j->t1);
    }
}

static Job *job_of(const char *q) {
    char *id = form_get(q, "id"); int k = id ? atoi(id) : -1; free(id);
    return (k >= 0 && k < g_njobs) ? &g_jobs[k] : NULL;
}

static void api_job(int fd, const char *q) {
    Job *j = job_of(q);
    if (!j) { reply_err(fd, 404, "no such job"); return; }
    poll_job(j);
    char *o = form_get(q, "offset"); long off = o ? atol(o) : 0; free(o);
    struct timespec now; clock_gettime(CLOCK_MONOTONIC, &now);
    struct timespec end = j->running ? now : j->t1;
    double el = (double)(end.tv_sec - j->t0.tv_sec) + (double)(end.tv_nsec - j->t0.tv_nsec) / 1e9;
    Buf b = {0}; bprintf(&b, "{\"running\":%s,\"exit\":%d,\"elapsed\":%.3f,\"log\":", j->running ? "true" : "false", j->exit_code, el);
    FILE *f = fopen(j->log, "rb"); long noff = off;
    if (f) {
        fseek(f, off, SEEK_SET);
        char *buf = malloc(1 << 20); size_t n = fread(buf, 1, 1 << 20, f); fclose(f);
        bjson(&b, buf, n); noff = off + (long)n; free(buf);
    } else bstr(&b, "\"\"");
    bprintf(&b, ",\"offset\":%ld}", noff);
    reply_json(fd, &b); free(b.s);
}

static void api_stop(int fd, const char *q) {
    Job *j = job_of(q);
    if (!j) { reply_err(fd, 404, "no such job"); return; }
    poll_job(j);
    if (j->running) kill(j->pid, SIGTERM);
    Buf b = {0}; bstr(&b, "{\"ok\":true}"); reply_json(fd, &b); free(b.s);
}

static void serve_static(int fd, const char *path) {
    if (strcmp(path, "/") == 0) path = "/index.html";
    if (strstr(path, "..")) { reply_err(fd, 403, "bad path"); return; }
    char full[PATH_MAX]; snprintf(full, sizeof full, "%s%s", g_static, path);
    FILE *f = fopen(full, "rb");
    if (!f) { reply_err(fd, 404, "not found"); return; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1); size_t r = fread(buf, 1, (size_t)n, f); fclose(f);
    const char *type = strstr(path, ".html") ? "text/html; charset=utf-8" : strstr(path, ".js") ? "text/javascript"
                     : strstr(path, ".css") ? "text/css" : "application/octet-stream";
    reply(fd, 200, type, buf, r); free(buf);
}

static void handle(int fd) {
    Buf req = {0}; char tmp[8192]; ssize_t r; char *hdr_end = NULL;
    while (!hdr_end && (r = recv(fd, tmp, sizeof tmp, 0)) > 0) {
        bput(&req, tmp, (size_t)r);
        hdr_end = strstr(req.s, "\r\n\r\n");
        if (req.n > (1 << 20)) break;
    }
    if (!hdr_end) { free(req.s); return; }
    size_t hlen = (size_t)(hdr_end - req.s) + 4;
    long clen = 0;
    char *cl = strcasestr(req.s, "\r\nContent-Length:");
    if (cl && cl < hdr_end) clen = atol(cl + 17);
    if (clen < 0 || clen > (1 << 20)) { reply_err(fd, 400, "body too large"); free(req.s); return; }
    while ((long)(req.n - hlen) < clen && (r = recv(fd, tmp, sizeof tmp, 0)) > 0) bput(&req, tmp, (size_t)r);
    /* only same-origin page requests: reject cross-site POSTs (Origin of another site) */
    char *origin = strcasestr(req.s, "\r\nOrigin:");
    char method[8] = {0}, target[4096] = {0};
    sscanf(req.s, "%7s %4095s", method, target);
    if (origin && origin < hdr_end && strcmp(method, "POST") == 0) {
        char ov[256] = {0}; sscanf(origin + 9, " %255s", ov);
        if (strncmp(ov, "http://127.0.0.1:", 17) != 0 && strncmp(ov, "http://localhost:", 17) != 0) {
            reply_err(fd, 403, "cross-origin request refused"); free(req.s); return;
        }
    }
    char *host = strcasestr(req.s, "\r\nHost:");
    char hv[256] = {0};
    if (host && host < hdr_end) sscanf(host + 7, " %255s", hv);
    if (strncmp(hv, "127.0.0.1:", 10) != 0 && strncmp(hv, "localhost:", 10) != 0) {
        reply_err(fd, 403, "Host must be 127.0.0.1 or localhost"); free(req.s); return;
    }
    char *query = strchr(target, '?');
    if (query) *query++ = 0;
    const char *body = req.s + hlen;
    if (strcmp(target, "/api/algorithms") == 0) api_algorithms(fd);
    else if (strcmp(target, "/api/ls") == 0) api_ls(fd, query);
    else if (strcmp(target, "/api/file") == 0) api_file(fd, query);
    else if (strcmp(target, "/api/run") == 0 && strcmp(method, "POST") == 0) api_run(fd, body);
    else if (strcmp(target, "/api/job") == 0) api_job(fd, query);
    else if (strcmp(target, "/api/stop") == 0 && strcmp(method, "POST") == 0) api_stop(fd, query);
    else if (strcmp(method, "GET") == 0) serve_static(fd, target);
    else reply_err(fd, 404, "unknown endpoint");
    free(req.s);
}

int main(int argc, char **argv) {
    int port = 8765;
    const char *root = ".", *dm = "./build/bin/dm.exe", *stat_dir = "gui";
    const char *extra[MAX_ALLOW]; int nextra = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--root") && i + 1 < argc) root = argv[++i];
        else if (!strcmp(argv[i], "--dm") && i + 1 < argc) dm = argv[++i];
        else if (!strcmp(argv[i], "--static") && i + 1 < argc) stat_dir = argv[++i];
        else if (!strcmp(argv[i], "--allow") && i + 1 < argc && nextra < MAX_ALLOW - 1) extra[nextra++] = argv[++i];
        else { fprintf(stderr, "usage: %s [--port N] [--root DIR] [--dm PATH] [--static DIR] [--allow DIR]...\n", argv[0]); return 2; }
    }
    if (!realpath(root, g_root)) { perror(root); return 1; }
    if (chdir(g_root) != 0) { perror("chdir"); return 1; }
    if (!realpath(dm, g_dm)) { fprintf(stderr, "dm.exe not found: %s\n", dm); return 1; }
    if (!realpath(stat_dir, g_static)) { fprintf(stderr, "static directory not found: %s\n", stat_dir); return 1; }
    strcpy(g_allow[g_nallow++], g_root);
    for (int i = 0; i < nextra; i++) if (realpath(extra[i], g_allow[g_nallow])) g_nallow++; else fprintf(stderr, "ignored --allow %s\n", extra[i]);
    snprintf(g_jobdir, sizeof g_jobdir, "/tmp/dm_gui_%d", (int)getpid());
    mkdir(g_jobdir, 0700);
    signal(SIGPIPE, SIG_IGN);

    int s = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_port = htons((uint16_t)port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(s, (struct sockaddr *)&a, sizeof a) != 0 || listen(s, 16) != 0) { perror("bind/listen"); return 1; }
    printf("dm_gui: http://127.0.0.1:%d/  (root %s, dm %s, logs %s)\n", port, g_root, g_dm, g_jobdir);
    for (int i = 1; i < g_nallow; i++) printf("dm_gui: also allowed %s\n", g_allow[i]);
    fflush(stdout);
    for (;;) {
        int c = accept(s, NULL, NULL);
        if (c < 0) { if (errno == EINTR) continue; perror("accept"); continue; }
        handle(c);
        close(c);
        for (int i = 0; i < g_njobs; i++) poll_job(&g_jobs[i]);
    }
}
