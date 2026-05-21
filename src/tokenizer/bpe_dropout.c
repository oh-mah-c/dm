#include "tokenizer/bpe_dropout.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DM_BPE_EOW "</w>"
#define DM_BPE_DEFAULT_SEPARATOR "@@"

typedef struct {
    char *left;
    char *right;
    char *joined;
} DMBPEMerge;

typedef struct {
    DMBPEMerge *items;
    size_t count;
    size_t cap;
} DMBPEMergeTable;

typedef struct {
    char **items;
    size_t count;
    size_t cap;
} DMStringVec;

typedef struct {
    char *text;
    size_t count;
} DMSegCount;

typedef struct {
    DMSegCount *items;
    size_t count;
    size_t cap;
} DMSegCounter;

typedef struct {
    uint64_t state;
} DMRng;

static char *dm_xstrdup(const char *s) {
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n + 1);
    return out;
}

static char *dm_xstrndup(const char *s, size_t n) {
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static char *dm_concat2(const char *a, const char *b) {
    size_t na = strlen(a), nb = strlen(b);
    char *out = (char *)malloc(na + nb + 1);
    if (!out) return NULL;
    memcpy(out, a, na);
    memcpy(out + na, b, nb + 1);
    return out;
}

static void merge_table_free(DMBPEMergeTable *table) {
    if (!table) return;
    for (size_t i = 0; i < table->count; i++) {
        free(table->items[i].left);
        free(table->items[i].right);
        free(table->items[i].joined);
    }
    free(table->items);
    table->items = NULL;
    table->count = table->cap = 0;
}

static int merge_table_push(DMBPEMergeTable *table, const char *left, const char *right) {
    if (table->count == table->cap) {
        size_t next = table->cap ? table->cap * 2 : 128;
        DMBPEMerge *tmp = (DMBPEMerge *)realloc(table->items, next * sizeof(DMBPEMerge));
        if (!tmp) return -1;
        table->items = tmp;
        table->cap = next;
    }
    DMBPEMerge *m = &table->items[table->count];
    m->left = dm_xstrdup(left);
    m->right = dm_xstrdup(right);
    m->joined = (m->left && m->right) ? dm_concat2(left, right) : NULL;
    if (!m->left || !m->right || !m->joined) return -1;
    table->count++;
    return 0;
}

static char *trim_line(char *s) {
    while (*s && isspace((unsigned char)*s)) s++;
    char *end = s + strlen(s);
    while (end > s && isspace((unsigned char)end[-1])) *--end = '\0';
    return s;
}

static int load_merges(const char *path, DMBPEMergeTable *table) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return -1;
    }
    char line[8192];
    size_t lineno = 0;
    while (fgets(line, sizeof(line), fp)) {
        lineno++;
        char *s = trim_line(line);
        if (!*s || *s == '#') continue;
        char *left = strtok(s, " \t\r\n");
        char *right = strtok(NULL, " \t\r\n");
        char *extra = strtok(NULL, " \t\r\n");
        if (!left || !right || extra) {
            fprintf(stderr, "invalid BPE merge line at %s:%zu\n", path, lineno);
            fclose(fp);
            return -1;
        }
        if (merge_table_push(table, left, right) != 0) {
            fprintf(stderr, "out of memory while reading %s\n", path);
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}

static void strvec_free(DMStringVec *vec) {
    if (!vec) return;
    for (size_t i = 0; i < vec->count; i++) free(vec->items[i]);
    free(vec->items);
    vec->items = NULL;
    vec->count = vec->cap = 0;
}

static int strvec_push_owned(DMStringVec *vec, char *s) {
    if (vec->count == vec->cap) {
        size_t next = vec->cap ? vec->cap * 2 : 32;
        char **tmp = (char **)realloc(vec->items, next * sizeof(char *));
        if (!tmp) return -1;
        vec->items = tmp;
        vec->cap = next;
    }
    vec->items[vec->count++] = s;
    return 0;
}

static int strvec_push_copy(DMStringVec *vec, const char *s) {
    char *copy = dm_xstrdup(s);
    if (!copy) return -1;
    if (strvec_push_owned(vec, copy) != 0) {
        free(copy);
        return -1;
    }
    return 0;
}

static size_t utf8_char_len(unsigned char c) {
    if ((c & 0x80u) == 0) return 1;
    if ((c & 0xE0u) == 0xC0u) return 2;
    if ((c & 0xF0u) == 0xE0u) return 3;
    if ((c & 0xF8u) == 0xF0u) return 4;
    return 1;
}

static int word_to_symbols(const char *word, DMStringVec *symbols) {
    const unsigned char *p = (const unsigned char *)word;
    while (*p) {
        size_t n = utf8_char_len(*p);
        for (size_t i = 1; i < n; i++) {
            if ((p[i] & 0xC0u) != 0x80u) {
                n = 1;
                break;
            }
        }
        char *sym = dm_xstrndup((const char *)p, n);
        if (!sym || strvec_push_owned(symbols, sym) != 0) {
            free(sym);
            return -1;
        }
        p += n;
    }
    return strvec_push_copy(symbols, DM_BPE_EOW);
}

static int merge_rank(const DMBPEMergeTable *table, const char *left, const char *right) {
    for (size_t i = 0; i < table->count; i++) {
        if (strcmp(table->items[i].left, left) == 0 && strcmp(table->items[i].right, right) == 0) return (int)i;
    }
    return -1;
}

static void rng_seed(DMRng *rng, uint64_t seed) {
    rng->state = seed ? seed : 0x9E3779B97F4A7C15ull;
}

static uint64_t rng_next(DMRng *rng) {
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 0x2545F4914F6CDD1Dull;
}

static double rng_double(DMRng *rng) {
    return (double)(rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

static int segment_word(const DMBPEMergeTable *table, const char *word, double dropout, DMRng *rng, DMStringVec *pieces) {
    DMStringVec symbols = {0};
    if (word_to_symbols(word, &symbols) != 0) goto oom;
    while (symbols.count >= 2) {
        int best_rank = -1;
        size_t best_idx = 0;
        for (size_t i = 0; i + 1 < symbols.count; i++) {
            int rank = merge_rank(table, symbols.items[i], symbols.items[i + 1]);
            if (rank < 0) continue;
            if (dropout >= 1.0) continue;
            if (dropout > 0.0 && rng_double(rng) < dropout) continue;
            if (best_rank < 0 || rank < best_rank) {
                best_rank = rank;
                best_idx = i;
            }
        }
        if (best_rank < 0) break;
        char *joined = dm_concat2(symbols.items[best_idx], symbols.items[best_idx + 1]);
        if (!joined) goto oom;
        free(symbols.items[best_idx]);
        free(symbols.items[best_idx + 1]);
        symbols.items[best_idx] = joined;
        for (size_t j = best_idx + 1; j + 1 < symbols.count; j++) symbols.items[j] = symbols.items[j + 1];
        symbols.count--;
    }

    for (size_t i = 0; i < symbols.count; i++) {
        char *sym = symbols.items[i];
        size_t n = strlen(sym);
        if (strcmp(sym, DM_BPE_EOW) == 0) {
            free(sym);
            continue;
        }
        if (n >= strlen(DM_BPE_EOW) && strcmp(sym + n - strlen(DM_BPE_EOW), DM_BPE_EOW) == 0) {
            sym[n - strlen(DM_BPE_EOW)] = '\0';
            if (sym[0] == '\0') {
                free(sym);
                continue;
            }
        }
        if (strvec_push_owned(pieces, sym) != 0) {
            free(sym);
            goto oom;
        }
    }
    free(symbols.items);
    return 0;
oom:
    strvec_free(&symbols);
    return -1;
}

static char *join_marked_pieces(const DMStringVec *pieces, const char *separator) {
    size_t sep_len = strlen(separator);
    size_t total = 1;
    for (size_t i = 0; i < pieces->count; i++) {
        total += strlen(pieces->items[i]);
        if (i + 1 < pieces->count) total += sep_len;
        if (i) total += 1;
    }
    char *out = (char *)malloc(total);
    if (!out) return NULL;
    out[0] = '\0';
    for (size_t i = 0; i < pieces->count; i++) {
        if (i) strcat(out, " ");
        strcat(out, pieces->items[i]);
        if (i + 1 < pieces->count) strcat(out, separator);
    }
    return out;
}

static int write_segmented_line(const DMBPEMergeTable *table, const char *line, double dropout, const char *separator, DMRng *rng, FILE *out) {
    char *copy = dm_xstrdup(line);
    if (!copy) return -1;
    char *s = trim_line(copy);
    int first = 1;
    while (*s) {
        while (*s && isspace((unsigned char)*s)) s++;
        if (!*s) break;
        char *start = s;
        while (*s && !isspace((unsigned char)*s)) s++;
        char saved = *s;
        *s = '\0';
        DMStringVec pieces = {0};
        if (segment_word(table, start, dropout, rng, &pieces) != 0) {
            strvec_free(&pieces);
            free(copy);
            return -1;
        }
        for (size_t i = 0; i < pieces.count; i++) {
            if (!first) fputc(' ', out);
            fputs(pieces.items[i], out);
            if (i + 1 < pieces.count) fputs(separator, out);
            first = 0;
        }
        strvec_free(&pieces);
        if (!saved) break;
        *s++ = saved;
    }
    fputc('\n', out);
    free(copy);
    return 0;
}

static void counter_free(DMSegCounter *counter) {
    if (!counter) return;
    for (size_t i = 0; i < counter->count; i++) free(counter->items[i].text);
    free(counter->items);
    counter->items = NULL;
    counter->count = counter->cap = 0;
}

static int counter_add(DMSegCounter *counter, const char *text) {
    for (size_t i = 0; i < counter->count; i++) {
        if (strcmp(counter->items[i].text, text) == 0) {
            counter->items[i].count++;
            return 0;
        }
    }
    if (counter->count == counter->cap) {
        size_t next = counter->cap ? counter->cap * 2 : 32;
        DMSegCount *tmp = (DMSegCount *)realloc(counter->items, next * sizeof(DMSegCount));
        if (!tmp) return -1;
        counter->items = tmp;
        counter->cap = next;
    }
    counter->items[counter->count].text = dm_xstrdup(text);
    if (!counter->items[counter->count].text) return -1;
    counter->items[counter->count].count = 1;
    counter->count++;
    return 0;
}

static int cmp_seg_count(const void *a, const void *b) {
    const DMSegCount *x = (const DMSegCount *)a;
    const DMSegCount *y = (const DMSegCount *)b;
    if (x->count < y->count) return 1;
    if (x->count > y->count) return -1;
    return strcmp(x->text, y->text);
}

static void json_string(FILE *out, const char *s) {
    fputc('"', out);
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') {
            fputc('\\', out);
            fputc(c, out);
        } else if (c == '\n') {
            fputs("\\n", out);
        } else if (c == '\r') {
            fputs("\\r", out);
        } else if (c == '\t') {
            fputs("\\t", out);
        } else if (c < 32) {
            fprintf(out, "\\u%04x", c);
        } else {
            fputc(c, out);
        }
    }
    fputc('"', out);
}

static int collect_words_from_file(const char *path, DMStringVec *words) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return -1;
    }
    char line[16384];
    while (fgets(line, sizeof(line), fp)) {
        char *s = trim_line(line);
        while (*s) {
            while (*s && isspace((unsigned char)*s)) s++;
            if (!*s) break;
            char *start = s;
            while (*s && !isspace((unsigned char)*s)) s++;
            char saved = *s;
            *s = '\0';
            if (strvec_push_copy(words, start) != 0) {
                fclose(fp);
                return -1;
            }
            if (!saved) break;
            *s++ = saved;
        }
    }
    fclose(fp);
    return 0;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s bpe_dropout -c codes.bpe [-p dropout] [--seed N] [--separator S] segment [-i input] [-o output]\n", prog);
    fprintf(stderr, "       %s bpe_dropout -c codes.bpe [-p dropout] [--seed N] [--separator S] sample-word [-n samples] word...\n", prog);
    fprintf(stderr, "       %s bpe_dropout -c codes.bpe [-p dropout] [--seed N] [--separator S] stats [-i input] [-n samples] [word...]\n", prog);
}

int dm_bpe_dropout_cli(int argc, char **argv) {
    const char *codes = NULL;
    const char *separator = DM_BPE_DEFAULT_SEPARATOR;
    double dropout = 0.1;
    uint64_t seed = (uint64_t)time(NULL);
    int have_seed = 0;
    int start_arg = 1;
    if (argc >= 2 && (strcmp(argv[1], "bpe_dropout") == 0 || strcmp(argv[1], "bpede") == 0 || strcmp(argv[1], "dm_bpe_dropout") == 0)) start_arg = 2;
    int command_idx = -1;
    for (int i = start_arg; i < argc; i++) {
        if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--codes") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return 2;
            }
            codes = argv[i];
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--dropout") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return 2;
            }
            dropout = atof(argv[i]);
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return 2;
            }
            seed = (uint64_t)strtoull(argv[i], NULL, 10);
            have_seed = 1;
        } else if (strcmp(argv[i], "--separator") == 0) {
            if (++i >= argc) {
                usage(argv[0]);
                return 2;
            }
            separator = argv[i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            command_idx = i;
            break;
        }
    }
    if (!codes || command_idx < 0 || dropout < 0.0 || dropout > 1.0) {
        usage(argv[0]);
        return 2;
    }
    if (!have_seed) seed ^= (uintptr_t)&seed;
    DMRng rng;
    rng_seed(&rng, seed);
    DMBPEMergeTable table = {0};
    if (load_merges(codes, &table) != 0) {
        merge_table_free(&table);
        return 1;
    }
    const char *cmd = argv[command_idx];
    int rc = 0;
    if (strcmp(cmd, "segment") == 0) {
        const char *input = NULL;
        const char *output = NULL;
        for (int i = command_idx + 1; i < argc; i++) {
            if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) input = argv[++i];
            else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) output = argv[++i];
            else {
                usage(argv[0]);
                rc = 2;
                goto done;
            }
        }
        FILE *in = input ? fopen(input, "rb") : stdin;
        FILE *out = output ? fopen(output, "wb") : stdout;
        if (!in || !out) {
            if (input && !in) perror(input);
            if (output && !out) perror(output);
            if (in && input) fclose(in);
            if (out && output) fclose(out);
            rc = 1;
            goto done;
        }
        char line[16384];
        while (fgets(line, sizeof(line), in)) {
            if (write_segmented_line(&table, line, dropout, separator, &rng, out) != 0) {
                fprintf(stderr, "out of memory during segmentation\n");
                rc = 1;
                break;
            }
        }
        if (input) fclose(in);
        if (output) fclose(out);
    } else if (strcmp(cmd, "sample-word") == 0) {
        int samples = 5;
        DMStringVec words = {0};
        for (int i = command_idx + 1; i < argc; i++) {
            if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--samples") == 0) {
                if (i + 1 >= argc) {
                    strvec_free(&words);
                    usage(argv[0]);
                    rc = 2;
                    goto done;
                }
                samples = atoi(argv[++i]);
            } else if (strvec_push_copy(&words, argv[i]) != 0) {
                strvec_free(&words);
                rc = 1;
                goto done;
            }
        }
        if (samples < 0 || words.count == 0) {
            strvec_free(&words);
            usage(argv[0]);
            rc = 2;
            goto done;
        }
        for (size_t i = 0; i < words.count; i++) {
            for (int s = 0; s < samples; s++) {
                DMStringVec pieces = {0};
                if (segment_word(&table, words.items[i], dropout, &rng, &pieces) != 0) {
                    strvec_free(&pieces);
                    strvec_free(&words);
                    rc = 1;
                    goto done;
                }
                printf("{\"word\": ");
                json_string(stdout, words.items[i]);
                printf(", \"pieces\": [");
                for (size_t j = 0; j < pieces.count; j++) {
                    if (j) printf(", ");
                    char *marked = (j + 1 < pieces.count) ? dm_concat2(pieces.items[j], separator) : dm_xstrdup(pieces.items[j]);
                    if (!marked) {
                        strvec_free(&pieces);
                        strvec_free(&words);
                        rc = 1;
                        goto done;
                    }
                    json_string(stdout, marked);
                    free(marked);
                }
                printf("]}\n");
                strvec_free(&pieces);
            }
        }
        strvec_free(&words);
    } else if (strcmp(cmd, "stats") == 0) {
        const char *input = NULL;
        int samples = 20;
        DMStringVec words = {0};
        DMStringVec arg_words = {0};
        for (int i = command_idx + 1; i < argc; i++) {
            if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) {
                input = argv[++i];
            } else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--samples") == 0) && i + 1 < argc) {
                samples = atoi(argv[++i]);
            } else {
                if (strvec_push_copy(&arg_words, argv[i]) != 0) {
                    strvec_free(&arg_words);
                    rc = 1;
                    goto done;
                }
            }
        }
        if (input && collect_words_from_file(input, &words) != 0) {
            strvec_free(&arg_words);
            strvec_free(&words);
            rc = 1;
            goto done;
        }
        for (size_t i = 0; i < arg_words.count; i++) {
            if (strvec_push_copy(&words, arg_words.items[i]) != 0) {
                strvec_free(&arg_words);
                strvec_free(&words);
                rc = 1;
                goto done;
            }
        }
        strvec_free(&arg_words);
        if (samples < 0) {
            strvec_free(&words);
            usage(argv[0]);
            rc = 2;
            goto done;
        }
        DMSegCounter counter = {0};
        size_t total_pieces = 0, total_words = 0;
        for (int s = 0; s < samples; s++) {
            for (size_t i = 0; i < words.count; i++) {
                DMStringVec pieces = {0};
                if (segment_word(&table, words.items[i], dropout, &rng, &pieces) != 0) {
                    strvec_free(&pieces);
                    counter_free(&counter);
                    strvec_free(&words);
                    rc = 1;
                    goto done;
                }
                char *joined = join_marked_pieces(&pieces, separator);
                if (!joined || counter_add(&counter, joined) != 0) {
                    free(joined);
                    strvec_free(&pieces);
                    counter_free(&counter);
                    strvec_free(&words);
                    rc = 1;
                    goto done;
                }
                free(joined);
                total_pieces += pieces.count;
                total_words++;
                strvec_free(&pieces);
            }
        }
        qsort(counter.items, counter.count, sizeof(DMSegCount), cmp_seg_count);
        printf("{\n");
        printf("  \"avg_pieces_per_word\": %.12g,\n", total_words ? (double)total_pieces / (double)total_words : 0.0);
        printf("  \"distinct_segmentations\": %zu,\n", counter.count);
        printf("  \"samples\": %d,\n", samples);
        printf("  \"segmentations\": {\n");
        for (size_t i = 0; i < counter.count; i++) {
            printf("    ");
            json_string(stdout, counter.items[i].text);
            printf(": %zu%s\n", counter.items[i].count, i + 1 < counter.count ? "," : "");
        }
        printf("  },\n");
        printf("  \"word_tokens\": %zu\n", total_words);
        printf("}\n");
        counter_free(&counter);
        strvec_free(&words);
    } else {
        usage(argv[0]);
        rc = 2;
    }
done:
    merge_table_free(&table);
    return rc;
}
