#include "tokenizer/maximal_munch.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int state;
    size_t pos;
} MMStackEntry;

typedef struct {
    size_t count;
    int start;
    int *trans;
    unsigned char *is_final;
    int *token_id;
    char **token_name;
} MMDFA;

typedef struct {
    size_t tokens;
    size_t errors;
    size_t transitions;
    size_t backtracks;
    size_t failed_hits;
    size_t failed_marks;
    size_t tab_states;
} MMStats;

static void dfa_free(MMDFA *dfa) {
    if (!dfa) return;
    if (dfa->token_name) {
        for (size_t i = 0; i < dfa->count; i++) free(dfa->token_name[i]);
    }
    free(dfa->trans);
    free(dfa->is_final);
    free(dfa->token_id);
    free(dfa->token_name);
    memset(dfa, 0, sizeof(*dfa));
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n + 1);
    return out;
}

static char *trim(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) *--end = '\0';
    return s;
}

static int parse_symbol(const char *text, int *lo, int *hi) {
    if (strcmp(text, "ANY") == 0) {
        *lo = 0;
        *hi = 255;
        return 0;
    }
    if (strncmp(text, "0x", 2) == 0 || strncmp(text, "0X", 2) == 0) {
        char *dash = strchr(text, '-');
        if (dash) {
            char left[32], right[32];
            size_t l = (size_t)(dash - text);
            if (l >= sizeof(left)) return -1;
            memcpy(left, text, l);
            left[l] = '\0';
            snprintf(right, sizeof(right), "%s", dash + 1);
            *lo = (int)strtol(left, NULL, 16);
            *hi = (int)strtol(right, NULL, 16);
        } else {
            *lo = *hi = (int)strtol(text, NULL, 16);
        }
        return (*lo >= 0 && *hi <= 255 && *lo <= *hi) ? 0 : -1;
    }
    if (strlen(text) == 3 && text[0] == '\'' && text[2] == '\'') {
        *lo = *hi = (unsigned char)text[1];
        return 0;
    }
    if (strlen(text) == 1) {
        *lo = *hi = (unsigned char)text[0];
        return 0;
    }
    return -1;
}

static int dfa_init(MMDFA *dfa, size_t states) {
    memset(dfa, 0, sizeof(*dfa));
    dfa->count = states;
    dfa->start = 0;
    dfa->trans = (int *)malloc(states * 256 * sizeof(int));
    dfa->is_final = (unsigned char *)calloc(states, 1);
    dfa->token_id = (int *)calloc(states, sizeof(int));
    dfa->token_name = (char **)calloc(states, sizeof(char *));
    if (!dfa->trans || !dfa->is_final || !dfa->token_id || !dfa->token_name) {
        dfa_free(dfa);
        return -1;
    }
    for (size_t i = 0; i < states * 256; i++) dfa->trans[i] = -1;
    for (size_t i = 0; i < states; i++) dfa->token_id[i] = -1;
    return 0;
}

static int dfa_set_trans(MMDFA *dfa, int from, int lo, int hi, int to) {
    if (from < 0 || to < 0 || (size_t)from >= dfa->count || (size_t)to >= dfa->count) return -1;
    for (int c = lo; c <= hi; c++) dfa->trans[(size_t)from * 256u + (size_t)c] = to;
    return 0;
}

static int load_dfa(const char *path, MMDFA *dfa) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        perror(path);
        return -1;
    }
    char line[4096];
    int initialized = 0;
    size_t lineno = 0;
    while (fgets(line, sizeof(line), fp)) {
        lineno++;
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';
        char *s = trim(line);
        if (!*s) continue;
        char *cmd = strtok(s, " \t\r\n");
        if (!cmd) continue;
        if (strcmp(cmd, "states") == 0) {
            char *n = strtok(NULL, " \t\r\n");
            if (!n || initialized || dfa_init(dfa, (size_t)strtoul(n, NULL, 10)) != 0) {
                fprintf(stderr, "invalid states line %zu\n", lineno);
                fclose(fp);
                return -1;
            }
            initialized = 1;
        } else if (strcmp(cmd, "start") == 0) {
            char *v = strtok(NULL, " \t\r\n");
            if (!initialized || !v) goto bad;
            dfa->start = atoi(v);
            if (dfa->start < 0 || (size_t)dfa->start >= dfa->count) goto bad;
        } else if (strcmp(cmd, "final") == 0) {
            char *st = strtok(NULL, " \t\r\n");
            char *id = strtok(NULL, " \t\r\n");
            char *name = strtok(NULL, " \t\r\n");
            if (!initialized || !st) goto bad;
            int q = atoi(st);
            if (q < 0 || (size_t)q >= dfa->count) goto bad;
            dfa->is_final[q] = 1;
            dfa->token_id[q] = id ? atoi(id) : q;
            free(dfa->token_name[q]);
            dfa->token_name[q] = xstrdup(name ? name : "TOKEN");
            if (!dfa->token_name[q]) goto bad;
        } else if (strcmp(cmd, "trans") == 0 || strcmp(cmd, "transition") == 0) {
            char *from = strtok(NULL, " \t\r\n");
            char *sym = strtok(NULL, " \t\r\n");
            char *to = strtok(NULL, " \t\r\n");
            int lo = 0, hi = 0;
            if (!initialized || !from || !sym || !to || parse_symbol(sym, &lo, &hi) != 0) goto bad;
            if (dfa_set_trans(dfa, atoi(from), lo, hi, atoi(to)) != 0) goto bad;
        } else {
            goto bad;
        }
        continue;
bad:
        fprintf(stderr, "invalid DFA spec at %s:%zu\n", path, lineno);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    if (!initialized) {
        fprintf(stderr, "DFA spec missing states line\n");
        return -1;
    }
    return 0;
}

static unsigned char *compute_tab_states(const MMDFA *dfa, size_t *tab_count) {
    size_t n = dfa->count;
    unsigned char *reachable = (unsigned char *)calloc(n, 1);
    unsigned char *bounded = (unsigned char *)calloc(n, 1);
    unsigned char *tab = (unsigned char *)calloc(n, 1);
    if (!reachable || !bounded || !tab) {
        free(reachable);
        free(bounded);
        free(tab);
        return NULL;
    }

    int changed = 1;
    while (changed) {
        changed = 0;
        for (size_t q = 0; q < n; q++) {
            unsigned char val = dfa->is_final[q];
            for (size_t p = 0; p < n && !val; p++) {
                if (!reachable[p]) continue;
                for (int c = 0; c < 256; c++) {
                    int r = dfa->trans[p * 256u + (size_t)c];
                    if (r == (int)q) {
                        val = 1;
                        break;
                    }
                }
            }
            if (val && !reachable[q]) {
                reachable[q] = 1;
                changed = 1;
            }
        }
    }

    for (size_t q = 0; q < n; q++) bounded[q] = dfa->is_final[q];
    changed = 1;
    while (changed) {
        changed = 0;
        for (size_t q = 0; q < n; q++) {
            if (bounded[q]) continue;
            unsigned char all = 1;
            for (int c = 0; c < 256; c++) {
                int r = dfa->trans[q * 256u + (size_t)c];
                if (r >= 0 && !bounded[r]) {
                    all = 0;
                    break;
                }
            }
            if (all) {
                bounded[q] = 1;
                changed = 1;
            }
        }
    }

    *tab_count = 0;
    for (size_t q = 0; q < n; q++) {
        if (reachable[q] && !bounded[q]) {
            tab[q] = 1;
            (*tab_count)++;
        }
    }
    free(reachable);
    free(bounded);
    return tab;
}

static int ensure_stack(MMStackEntry **stack, size_t *cap, size_t need) {
    if (need <= *cap) return 0;
    size_t next = *cap ? *cap * 2 : 128;
    while (next < need) next *= 2;
    MMStackEntry *tmp = (MMStackEntry *)realloc(*stack, next * sizeof(MMStackEntry));
    if (!tmp) return -1;
    *stack = tmp;
    *cap = next;
    return 0;
}

static int tokenize_reps(const MMDFA *dfa, const unsigned char *input, size_t len, int print_tokens, int print_lexeme, MMStats *stats) {
    memset(stats, 0, sizeof(*stats));
    size_t tab_count = 0;
    unsigned char *tab_states = compute_tab_states(dfa, &tab_count);
    if (!tab_states) return -1;
    stats->tab_states = tab_count;
    size_t *tab_index = NULL;
    unsigned char *failed = NULL;
    if (tab_count) {
        tab_index = (size_t *)calloc(dfa->count, sizeof(size_t));
        failed = (unsigned char *)calloc(tab_count * (len + 1), 1);
        if (!tab_index || !failed) {
            free(tab_index);
            free(tab_states);
            free(failed);
            return -1;
        }
        size_t idx = 0;
        for (size_t s = 0; s < dfa->count; s++) {
            if (tab_states[s]) tab_index[s] = idx++;
        }
    }
    MMStackEntry *stack = NULL;
    size_t sp = 0, cap = 0;
    size_t i = 0, token_start = 0;
    int q = dfa->start;

    if (ensure_stack(&stack, &cap, 1) != 0) goto oom;
    stack[sp++] = (MMStackEntry){-1, i};

    while (i < len) {
        while (i < len) {
            if (q >= 0 && tab_states[q] && failed[tab_index[q] * (len + 1) + i]) {
                stats->failed_hits++;
                break;
            }
            int next = dfa->trans[(size_t)q * 256u + input[i]];
            if (next < 0) break;
            q = next;
            i++;
            stats->transitions++;
            if (dfa->is_final[q] || tab_states[q]) {
                if (ensure_stack(&stack, &cap, sp + 1) != 0) goto oom;
                stack[sp++] = (MMStackEntry){q, i};
            }
        }

        MMStackEntry e = {-1, token_start};
        do {
            if (sp == 0) {
                fprintf(stderr, "internal stack underflow\n");
                goto fail;
            }
            e = stack[--sp];
            if (e.state >= 0 && tab_states[e.state]) {
                failed[tab_index[e.state] * (len + 1) + e.pos] = 1;
                stats->failed_marks++;
            }
            stats->backtracks++;
        } while (e.state >= 0 && !dfa->is_final[e.state]);

        if (e.state < 0) {
            if (print_tokens) {
                printf("ERROR\t%zu\t%zu", token_start, token_start + 1);
                if (print_lexeme) printf("\t%.*s", 1, (const char *)(input + token_start));
                putchar('\n');
            }
            stats->errors++;
            i = token_start + 1;
        } else {
            if (print_tokens) {
                const char *name = dfa->token_name[e.state] ? dfa->token_name[e.state] : "TOKEN";
                printf("%s\t%d\t%zu\t%zu", name, dfa->token_id[e.state], token_start, e.pos);
                if (print_lexeme) printf("\t%.*s", (int)(e.pos - token_start), (const char *)(input + token_start));
                putchar('\n');
            }
            stats->tokens++;
            i = e.pos;
        }
        token_start = i;
        q = dfa->start;
        sp = 0;
        if (ensure_stack(&stack, &cap, 1) != 0) goto oom;
        stack[sp++] = (MMStackEntry){-1, i};
    }

    free(stack);
    free(tab_index);
    free(failed);
    free(tab_states);
    return 0;
oom:
    fprintf(stderr, "out of memory\n");
fail:
    free(stack);
    free(tab_index);
    free(failed);
    free(tab_states);
    return -1;
}

static unsigned char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long n = ftell(fp);
    if (n < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    unsigned char *buf = (unsigned char *)malloc((size_t)n + 1);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)n, fp);
    fclose(fp);
    buf[got] = '\0';
    *len_out = got;
    return buf;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s maximal_munch --dfa spec.dfa --input text [--stats] [--no-lexeme]\n", prog);
    fprintf(stderr, "DFA format: states N | start Q | final Q TOKEN_ID NAME | trans FROM SYMBOL TO\n");
    fprintf(stderr, "SYMBOL: a, 'a', 0x61, 0x30-0x39, or ANY. Undefined transitions are scanner failure.\n");
}

int dm_maximal_munch_cli(int argc, char **argv) {
    const char *dfa_path = NULL;
    const char *input_path = NULL;
    int stats_only = 0;
    int print_lexeme = 1;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--dfa") == 0 && i + 1 < argc) {
            dfa_path = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--stats") == 0) {
            stats_only = 1;
        } else if (strcmp(argv[i], "--no-lexeme") == 0) {
            print_lexeme = 0;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!dfa_path || !input_path) {
        usage(argv[0]);
        return 2;
    }
    MMDFA dfa;
    if (load_dfa(dfa_path, &dfa) != 0) return 1;
    size_t len = 0;
    unsigned char *input = read_file(input_path, &len);
    if (!input) {
        dfa_free(&dfa);
        return 1;
    }
    MMStats st;
    int rc = tokenize_reps(&dfa, input, len, !stats_only, print_lexeme, &st);
    if (rc == 0 && stats_only) {
        printf("tokens\t%zu\n", st.tokens);
        printf("errors\t%zu\n", st.errors);
        printf("transitions\t%zu\n", st.transitions);
        printf("backtracks\t%zu\n", st.backtracks);
        printf("failed_hits\t%zu\n", st.failed_hits);
        printf("failed_marks\t%zu\n", st.failed_marks);
        printf("tab_states\t%zu\n", st.tab_states);
    }
    free(input);
    dfa_free(&dfa);
    return rc == 0 ? 0 : 1;
}
