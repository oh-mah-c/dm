#include "tokenizer/fast_wordpiece.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FWP_DEFAULT_UNK "[UNK]"
#define FWP_DEFAULT_SUFFIX "##"

typedef struct {
    char *label;
    int child;
} FWPChild;

typedef struct {
    FWPChild *children;
    size_t child_count;
    size_t child_cap;
    char *token;
    int token_id;
    int parent;
    char *ch;
    char *text;
    int fail;
    char **pops;
    size_t pop_count;
    size_t pop_cap;
} FWPNode;

typedef struct {
    FWPNode *nodes;
    size_t count;
    size_t cap;
    int root;
    int suffix_root;
    char **tokens;
    size_t token_count;
    size_t token_cap;
    char *unk_token;
    char *suffix_indicator;
} FWPTokenizer;

typedef struct {
    char **items;
    size_t count;
    size_t cap;
} StrVec;

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n + 1);
    return out;
}

static char *xstrndup(const char *s, size_t n) {
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

static char *concat2(const char *a, const char *b) {
    size_t na = strlen(a), nb = strlen(b);
    char *out = (char *)malloc(na + nb + 1);
    if (!out) return NULL;
    memcpy(out, a, na);
    memcpy(out + na, b, nb + 1);
    return out;
}

static size_t utf8_len(unsigned char c) {
    if ((c & 0x80u) == 0) return 1;
    if ((c & 0xE0u) == 0xC0u) return 2;
    if ((c & 0xF0u) == 0xE0u) return 3;
    if ((c & 0xF8u) == 0xF0u) return 4;
    return 1;
}

static void strvec_free(StrVec *v) {
    for (size_t i = 0; i < v->count; i++) free(v->items[i]);
    free(v->items);
    v->items = NULL;
    v->count = v->cap = 0;
}

static int strvec_push_owned(StrVec *v, char *s) {
    if (v->count == v->cap) {
        size_t next = v->cap ? v->cap * 2 : 16;
        char **tmp = (char **)realloc(v->items, next * sizeof(char *));
        if (!tmp) return -1;
        v->items = tmp;
        v->cap = next;
    }
    v->items[v->count++] = s;
    return 0;
}

static int strvec_push_copy(StrVec *v, const char *s) {
    char *copy = xstrdup(s);
    if (!copy) return -1;
    if (strvec_push_owned(v, copy) != 0) {
        free(copy);
        return -1;
    }
    return 0;
}

static int split_chars(const char *text, StrVec *out) {
    const unsigned char *p = (const unsigned char *)text;
    while (*p) {
        size_t n = utf8_len(*p);
        for (size_t i = 1; i < n; i++) {
            if ((p[i] & 0xC0u) != 0x80u) {
                n = 1;
                break;
            }
        }
        char *ch = xstrndup((const char *)p, n);
        if (!ch || strvec_push_owned(out, ch) != 0) {
            free(ch);
            return -1;
        }
        p += n;
    }
    return 0;
}

static int strvec_join_range(const StrVec *chars, size_t start, size_t end, char **out) {
    size_t total = 1;
    for (size_t i = start; i < end; i++) total += strlen(chars->items[i]);
    char *s = (char *)malloc(total);
    if (!s) return -1;
    s[0] = '\0';
    for (size_t i = start; i < end; i++) strcat(s, chars->items[i]);
    *out = s;
    return 0;
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

static void node_free(FWPNode *n) {
    for (size_t i = 0; i < n->child_count; i++) free(n->children[i].label);
    free(n->children);
    free(n->token);
    free(n->ch);
    free(n->text);
    for (size_t i = 0; i < n->pop_count; i++) free(n->pops[i]);
    free(n->pops);
}

static void tokenizer_free(FWPTokenizer *tok) {
    for (size_t i = 0; i < tok->count; i++) node_free(&tok->nodes[i]);
    free(tok->nodes);
    for (size_t i = 0; i < tok->token_count; i++) free(tok->tokens[i]);
    free(tok->tokens);
    free(tok->unk_token);
    free(tok->suffix_indicator);
    memset(tok, 0, sizeof(*tok));
}

static int tokenizer_add_node(FWPTokenizer *tok, int parent, const char *ch, int *idx) {
    if (tok->count == tok->cap) {
        size_t next = tok->cap ? tok->cap * 2 : 128;
        FWPNode *tmp = (FWPNode *)realloc(tok->nodes, next * sizeof(FWPNode));
        if (!tmp) return -1;
        tok->nodes = tmp;
        tok->cap = next;
    }
    FWPNode *n = &tok->nodes[tok->count];
    memset(n, 0, sizeof(*n));
    n->token_id = -1;
    n->parent = parent;
    n->fail = -1;
    n->ch = xstrdup(ch ? ch : "");
    n->text = parent >= 0 ? concat2(tok->nodes[parent].text, ch) : xstrdup("");
    if (!n->ch || !n->text) return -1;
    *idx = (int)tok->count++;
    return 0;
}

static int find_child(const FWPTokenizer *tok, int node, const char *label) {
    const FWPNode *n = &tok->nodes[node];
    for (size_t i = 0; i < n->child_count; i++) {
        if (strcmp(n->children[i].label, label) == 0) return n->children[i].child;
    }
    return -1;
}

static int add_child(FWPTokenizer *tok, int node, const char *label, int child) {
    FWPNode *n = &tok->nodes[node];
    if (n->child_count == n->child_cap) {
        size_t next = n->child_cap ? n->child_cap * 2 : 8;
        FWPChild *tmp = (FWPChild *)realloc(n->children, next * sizeof(FWPChild));
        if (!tmp) return -1;
        n->children = tmp;
        n->child_cap = next;
    }
    n->children[n->child_count].label = xstrdup(label);
    n->children[n->child_count].child = child;
    if (!n->children[n->child_count].label) return -1;
    n->child_count++;
    return 0;
}

static int ensure_path(FWPTokenizer *tok, const char *text) {
    StrVec chars = {0};
    if (split_chars(text, &chars) != 0) return -1;
    int node = tok->root;
    for (size_t i = 0; i < chars.count; i++) {
        int next = find_child(tok, node, chars.items[i]);
        if (next < 0) {
            if (tokenizer_add_node(tok, node, chars.items[i], &next) != 0 || add_child(tok, node, chars.items[i], next) != 0) {
                strvec_free(&chars);
                return -1;
            }
        }
        node = next;
    }
    strvec_free(&chars);
    return node;
}

static int vocab_has(const FWPTokenizer *tok, const char *token) {
    for (size_t i = 0; i < tok->token_count; i++) if (strcmp(tok->tokens[i], token) == 0) return 1;
    return 0;
}

static int vocab_id(const FWPTokenizer *tok, const char *token) {
    for (size_t i = 0; i < tok->token_count; i++) if (strcmp(tok->tokens[i], token) == 0) return (int)i;
    return -1;
}

static int vocab_push(FWPTokenizer *tok, const char *token) {
    if (vocab_has(tok, token)) return 0;
    if (tok->token_count == tok->token_cap) {
        size_t next = tok->token_cap ? tok->token_cap * 2 : 1024;
        char **tmp = (char **)realloc(tok->tokens, next * sizeof(char *));
        if (!tmp) return -1;
        tok->tokens = tmp;
        tok->token_cap = next;
    }
    tok->tokens[tok->token_count] = xstrdup(token);
    if (!tok->tokens[tok->token_count]) return -1;
    tok->token_count++;
    return 0;
}

static int insert_token(FWPTokenizer *tok, const char *token, int id) {
    int node = ensure_path(tok, token);
    if (node < 0) return -1;
    free(tok->nodes[node].token);
    tok->nodes[node].token = xstrdup(token);
    tok->nodes[node].token_id = id;
    return tok->nodes[node].token ? 0 : -1;
}

static int pops_push(FWPNode *n, const char *token) {
    if (n->pop_count == n->pop_cap) {
        size_t next = n->pop_cap ? n->pop_cap * 2 : 4;
        char **tmp = (char **)realloc(n->pops, next * sizeof(char *));
        if (!tmp) return -1;
        n->pops = tmp;
        n->pop_cap = next;
    }
    n->pops[n->pop_count] = xstrdup(token);
    if (!n->pops[n->pop_count]) return -1;
    n->pop_count++;
    return 0;
}

static int pops_extend(FWPNode *n, const FWPNode *src) {
    for (size_t i = 0; i < src->pop_count; i++) if (pops_push(n, src->pops[i]) != 0) return -1;
    return 0;
}

static int precompute_failure(FWPTokenizer *tok) {
    for (size_t i = 0; i < tok->count; i++) {
        tok->nodes[i].fail = -1;
        for (size_t j = 0; j < tok->nodes[i].pop_count; j++) free(tok->nodes[i].pops[j]);
        free(tok->nodes[i].pops);
        tok->nodes[i].pops = NULL;
        tok->nodes[i].pop_count = tok->nodes[i].pop_cap = 0;
    }
    int *queue = (int *)malloc((tok->count + 2) * sizeof(int));
    if (!queue) return -1;
    size_t head = 0, tail = 0;
    queue[tail++] = tok->root;
    if (tok->suffix_root != tok->root) queue[tail++] = tok->suffix_root;
    while (head < tail) {
        int u = queue[head++];
        FWPNode *un = &tok->nodes[u];
        for (size_t ci = 0; ci < un->child_count; ci++) {
            const char *ch = un->children[ci].label;
            int v = un->children[ci].child;
            if (v == tok->suffix_root) continue;
            FWPNode *vn = &tok->nodes[v];
            if (vn->token) {
                vn->fail = tok->suffix_root;
                if (pops_push(vn, vn->token) != 0) {
                    free(queue);
                    return -1;
                }
            } else {
                int z = un->fail;
                if (pops_extend(vn, un) != 0) {
                    free(queue);
                    return -1;
                }
                while (z >= 0 && find_child(tok, z, ch) < 0) {
                    if (pops_extend(vn, &tok->nodes[z]) != 0) {
                        free(queue);
                        return -1;
                    }
                    z = tok->nodes[z].fail;
                }
                if (z >= 0) vn->fail = find_child(tok, z, ch);
            }
            if (tail >= tok->count + 2) {
                free(queue);
                return -1;
            }
            queue[tail++] = v;
        }
    }
    free(queue);
    return 0;
}

static int load_tokenizer(const char *path, const char *unk, const char *suffix, FWPTokenizer *tok) {
    memset(tok, 0, sizeof(*tok));
    tok->unk_token = xstrdup(unk);
    tok->suffix_indicator = xstrdup(suffix);
    if (!tok->unk_token || !tok->suffix_indicator || tokenizer_add_node(tok, -1, "", &tok->root) != 0) return -1;
    tok->suffix_root = ensure_path(tok, suffix);
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        perror(path);
        return -1;
    }
    char line[8192];
    int saw_unk = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *s = line;
        while (*s && isspace((unsigned char)*s)) s++;
        if (!*s) continue;
        char *end = s;
        while (*end && !isspace((unsigned char)*end)) end++;
        *end = '\0';
        if (!*s) continue;
        if (strcmp(s, unk) == 0) saw_unk = 1;
        if (vocab_push(tok, s) != 0) {
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    if (!saw_unk) {
        if (tok->token_count == tok->token_cap) {
            size_t next = tok->token_cap ? tok->token_cap * 2 : 1024;
            char **tmp = (char **)realloc(tok->tokens, next * sizeof(char *));
            if (!tmp) return -1;
            tok->tokens = tmp;
            tok->token_cap = next;
        }
        for (size_t i = tok->token_count; i > 0; i--) tok->tokens[i] = tok->tokens[i - 1];
        tok->tokens[0] = xstrdup(unk);
        if (!tok->tokens[0]) return -1;
        tok->token_count++;
    }
    for (size_t i = 0; i < tok->token_count; i++) {
        if (insert_token(tok, tok->tokens[i], (int)i) != 0) return -1;
    }
    tok->suffix_root = ensure_path(tok, suffix);
    return precompute_failure(tok);
}

static int match_loop(const FWPTokenizer *tok, const StrVec *chars, size_t start, StrVec *tokens, int *u_out, size_t *i_out) {
    int u = tok->root;
    size_t i = start;
    while (i < chars->count) {
        const char *ch = chars->items[i];
        while (find_child(tok, u, ch) < 0) {
            if (tok->nodes[u].fail < 0) {
                *u_out = u;
                *i_out = i;
                return 0;
            }
            for (size_t p = 0; p < tok->nodes[u].pop_count; p++) {
                if (strvec_push_copy(tokens, tok->nodes[u].pops[p]) != 0) return -1;
            }
            u = tok->nodes[u].fail;
        }
        u = find_child(tok, u, ch);
        i++;
    }
    *u_out = u;
    *i_out = i;
    return 0;
}

static int original_wordpiece(const FWPTokenizer *tok, const char *word, StrVec *pieces) {
    StrVec chars = {0};
    if (split_chars(word, &chars) != 0) return -1;
    size_t start = 0;
    while (start < chars.count) {
        size_t end = chars.count;
        char *cur = NULL;
        while (start < end) {
            char *raw = NULL;
            if (strvec_join_range(&chars, start, end, &raw) != 0) {
                strvec_free(&chars);
                return -1;
            }
            char *piece = start > 0 ? concat2(tok->suffix_indicator, raw) : xstrdup(raw);
            free(raw);
            if (!piece) {
                strvec_free(&chars);
                return -1;
            }
            if (vocab_has(tok, piece)) {
                cur = piece;
                break;
            }
            free(piece);
            end--;
        }
        if (!cur) {
            strvec_free(&chars);
            return strvec_push_copy(pieces, tok->unk_token);
        }
        if (strvec_push_owned(pieces, cur) != 0) {
            free(cur);
            strvec_free(&chars);
            return -1;
        }
        start = end;
    }
    strvec_free(&chars);
    return 0;
}

static int tokenize_word(const FWPTokenizer *tok, const char *word, StrVec *out) {
    StrVec chars = {0};
    if (split_chars(word, &chars) != 0) return -1;
    size_t word_len = chars.count;
    if (strvec_push_copy(&chars, " ") != 0) {
        strvec_free(&chars);
        return -1;
    }
    int u = tok->root;
    size_t i = 0;
    if (match_loop(tok, &chars, 0, out, &u, &i) != 0) {
        strvec_free(&chars);
        return -1;
    }
    if (i < word_len || (u != tok->root && u != tok->suffix_root)) {
        strvec_free(out);
        memset(out, 0, sizeof(*out));
        strvec_free(&chars);
        return strvec_push_copy(out, tok->unk_token);
    }
    if (u == tok->suffix_root && out->count == 0) {
        strvec_free(out);
        memset(out, 0, sizeof(*out));
        strvec_free(&chars);
        return original_wordpiece(tok, tok->suffix_indicator, out);
    }
    strvec_free(&chars);
    return 0;
}

static int is_punctuation_ascii_or_utf8(const char *ch) {
    unsigned char c = (unsigned char)ch[0];
    if (ch[1] == '\0') {
        if ((33 <= c && c <= 47) || (58 <= c && c <= 64) || (91 <= c && c <= 96) || (123 <= c && c <= 126)) return 1;
    }
    return 0;
}

static int tokenize_text(const FWPTokenizer *tok, const char *text, StrVec *out) {
    StrVec chars = {0};
    if (split_chars(text, &chars) != 0) return -1;
    size_t i = 0;
    while (i < chars.count) {
        while (i < chars.count && strcmp(chars.items[i], " ") == 0) i++;
        if (i >= chars.count) break;
        if (is_punctuation_ascii_or_utf8(chars.items[i])) {
            if (vocab_has(tok, chars.items[i])) {
                if (strvec_push_copy(out, chars.items[i]) != 0) {
                    strvec_free(&chars);
                    return -1;
                }
            } else if (strvec_push_copy(out, tok->unk_token) != 0) {
                strvec_free(&chars);
                return -1;
            }
            i++;
            continue;
        }
        size_t start = i;
        while (i < chars.count && strcmp(chars.items[i], " ") != 0 && !is_punctuation_ascii_or_utf8(chars.items[i])) i++;
        char *word = NULL;
        if (strvec_join_range(&chars, start, i, &word) != 0) {
            strvec_free(&chars);
            return -1;
        }
        StrVec pieces = {0};
        if (tokenize_word(tok, word, &pieces) != 0) {
            free(word);
            strvec_free(&pieces);
            strvec_free(&chars);
            return -1;
        }
        free(word);
        for (size_t p = 0; p < pieces.count; p++) {
            char *owned = pieces.items[p];
            if (strvec_push_owned(out, owned) != 0) {
                pieces.items[p] = NULL;
                strvec_free(&pieces);
                strvec_free(&chars);
                return -1;
            }
            pieces.items[p] = NULL;
        }
        strvec_free(&pieces);
    }
    strvec_free(&chars);
    return 0;
}

static void print_tokens_json(const StrVec *tokens, int ids, const FWPTokenizer *tok) {
    printf("[");
    for (size_t i = 0; i < tokens->count; i++) {
        if (i) printf(", ");
        if (ids) {
            int id = vocab_id(tok, tokens->items[i]);
            if (id < 0) id = vocab_id(tok, tok->unk_token);
            if (id < 0) id = 0;
            printf("%d", id);
        } else {
            json_string(stdout, tokens->items[i]);
        }
    }
    printf("]\n");
}

static void write_tokens_line(FILE *out, const StrVec *tokens) {
    for (size_t i = 0; i < tokens->count; i++) {
        if (i) fputc(' ', out);
        fputs(tokens->items[i], out);
    }
    fputc('\n', out);
}

static void write_ids_line(FILE *out, const StrVec *tokens, const FWPTokenizer *tok) {
    fputc('[', out);
    for (size_t i = 0; i < tokens->count; i++) {
        if (i) fputs(", ", out);
        int id = vocab_id(tok, tokens->items[i]);
        if (id < 0) id = vocab_id(tok, tok->unk_token);
        if (id < 0) id = 0;
        fprintf(out, "%d", id);
    }
    fputs("]\n", out);
}

static void write_json_tokens_line(FILE *out, const StrVec *tokens) {
    fputc('[', out);
    for (size_t i = 0; i < tokens->count; i++) {
        if (i) fputs(", ", out);
        json_string(out, tokens->items[i]);
    }
    fputs("]\n", out);
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s fast_wordpiece [--unk-token T] [--suffix-indicator S] word -v vocab [--ids] [word...]\n", prog);
    fprintf(stderr, "       %s fast_wordpiece [--unk-token T] [--suffix-indicator S] encode -v vocab [-i input] [-o output] [--json-tokens|--ids]\n", prog);
    fprintf(stderr, "       %s fast_wordpiece [--unk-token T] [--suffix-indicator S] inspect -v vocab\n", prog);
}

int dm_fast_wordpiece_cli(int argc, char **argv) {
    int start = 1;
    if (argc >= 2 && (strcmp(argv[1], "fast_wordpiece") == 0 || strcmp(argv[1], "linmaxmatch") == 0 || strcmp(argv[1], "dm_fast_wordpiece") == 0)) start = 2;
    const char *unk = FWP_DEFAULT_UNK;
    const char *suffix = FWP_DEFAULT_SUFFIX;
    int command_idx = -1;
    for (int i = start; i < argc; i++) {
        if (strcmp(argv[i], "--unk-token") == 0 && i + 1 < argc) unk = argv[++i];
        else if (strcmp(argv[i], "--suffix-indicator") == 0 && i + 1 < argc) suffix = argv[++i];
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            command_idx = i;
            break;
        }
    }
    if (command_idx < 0) {
        usage(argv[0]);
        return 2;
    }
    const char *cmd = argv[command_idx];
    const char *vocab = NULL;
    int ids = 0, json_tokens = 0;
    const char *input = NULL, *output = NULL;
    int first_word = command_idx + 1;
    for (int i = command_idx + 1; i < argc; i++) {
        if ((strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--vocab") == 0) && i + 1 < argc) {
            vocab = argv[++i];
            if (first_word == i - 1) first_word = i + 1;
        } else if (strcmp(argv[i], "--ids") == 0) {
            ids = 1;
            if (first_word == i) first_word = i + 1;
        } else if (strcmp(argv[i], "--json-tokens") == 0) {
            json_tokens = 1;
            if (first_word == i) first_word = i + 1;
        } else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) {
            input = argv[++i];
            if (first_word == i - 1) first_word = i + 1;
        } else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) {
            output = argv[++i];
            if (first_word == i - 1) first_word = i + 1;
        } else {
            break;
        }
    }
    if (!vocab) {
        usage(argv[0]);
        return 2;
    }
    FWPTokenizer tok = {0};
    if (load_tokenizer(vocab, unk, suffix, &tok) != 0) {
        tokenizer_free(&tok);
        return 1;
    }
    int rc = 0;
    if (strcmp(cmd, "word") == 0) {
        if (first_word < argc) {
            for (int i = first_word; i < argc; i++) {
                if (argv[i][0] == '-' && (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--vocab") == 0 || strcmp(argv[i], "--ids") == 0)) continue;
                StrVec pieces = {0};
                if (tokenize_word(&tok, argv[i], &pieces) != 0) {
                    rc = 1;
                    strvec_free(&pieces);
                    break;
                }
                print_tokens_json(&pieces, ids, &tok);
                strvec_free(&pieces);
            }
        } else {
            char line[32768];
            while (fgets(line, sizeof(line), stdin)) {
                size_t n = strlen(line);
                while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
                StrVec pieces = {0};
                if (tokenize_word(&tok, line, &pieces) != 0) {
                    rc = 1;
                    strvec_free(&pieces);
                    break;
                }
                print_tokens_json(&pieces, ids, &tok);
                strvec_free(&pieces);
            }
        }
    } else if (strcmp(cmd, "encode") == 0) {
        FILE *in = input ? fopen(input, "rb") : stdin;
        FILE *out = output ? fopen(output, "wb") : stdout;
        if (!in || !out) {
            if (input && !in) perror(input);
            if (output && !out) perror(output);
            rc = 1;
        } else {
            char line[32768];
            while (fgets(line, sizeof(line), in)) {
                size_t n = strlen(line);
                while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = '\0';
                StrVec pieces = {0};
                if (tokenize_text(&tok, line, &pieces) != 0) {
                    rc = 1;
                    strvec_free(&pieces);
                    break;
                }
                if (ids) write_ids_line(out, &pieces, &tok);
                else if (json_tokens) write_json_tokens_line(out, &pieces);
                else write_tokens_line(out, &pieces);
                strvec_free(&pieces);
            }
        }
        if (input && in) fclose(in);
        if (output && out) fclose(out);
    } else if (strcmp(cmd, "inspect") == 0) {
        printf("{\n");
        printf("  \"nodes\": %zu,\n", tok.count);
        printf("  \"suffix_indicator\": ");
        json_string(stdout, tok.suffix_indicator);
        printf(",\n");
        printf("  \"suffix_root\": %d,\n", tok.suffix_root);
        printf("  \"unk_token\": ");
        json_string(stdout, tok.unk_token);
        printf(",\n");
        printf("  \"version\": \"dm-fast-wordpiece-emnlp-2021\",\n");
        printf("  \"vocabulary_size\": %zu\n", tok.token_count);
        printf("}\n");
    } else {
        usage(argv[0]);
        rc = 2;
    }
    tokenizer_free(&tok);
    return rc;
}
