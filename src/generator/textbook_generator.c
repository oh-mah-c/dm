#include "generator/textbook_generator.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

typedef struct {
    char **data;
    size_t len;
    size_t cap;
} TG_StrVec;

static const char *const DEFAULT_TOPICS[] = {
    "why plants need sunlight",
    "how rain forms",
    "sharing toys fairly",
    "basic addition with apples",
    "why magnets attract metal",
    "how to sort objects by size",
    "what makes a good question",
    "simple cause and effect",
    "how a seed becomes a plant",
    "using loops to repeat steps",
    "how maps help us find places",
    "why washing hands matters"
};

static const char *const STYLES[] = {
    "a concise textbook lesson",
    "a worked example with step-by-step reasoning",
    "a short Socratic dialogue between teacher and student",
    "a lesson followed by exercises and answers",
    "a concept explanation with common mistakes and corrections"
};

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s textbook prompts -o prompts.jsonl -n N [--topics topics.txt]\n"
        "       [--web-samples snippets.txt] [--seed N]\n"
        "  %s textbook filter -i generated.jsonl -o corpus.txt [--min-score F]\n"
        "  %s textbook score -i generated.jsonl\n"
        "  %s textbook mix --synthetic synth.txt --web web.txt --code code.txt -o corpus.txt\n"
        "       [--synthetic-ratio F] [--web-ratio F] [--code-ratio F] [--max-bytes N]\n"
        "  %s textbook config -o phi_textbook_generator.yaml\n\n"
        "Generated JSONL may contain text, content, output, completion, or story fields.\n",
        prog, prog, prog, prog, prog);
}

static char *xstrdup(const char *s) {
    size_t n;
    char *p;
    if (!s) return NULL;
    n = strlen(s);
    p = (char *)malloc(n + 1);
    if (p) memcpy(p, s, n + 1);
    return p;
}

static void trim(char *s) {
    char *e;
    if (!s) return;
    while (isspace((unsigned char)*s)) memmove(s, s + 1, strlen(s));
    e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = '\0';
}

static int vec_push(TG_StrVec *v, const char *s) {
    char **tmp;
    if (v->len == v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 32;
        tmp = (char **)realloc(v->data, nc * sizeof(*tmp));
        if (!tmp) return -1;
        v->data = tmp;
        v->cap = nc;
    }
    v->data[v->len] = xstrdup(s);
    if (!v->data[v->len]) return -1;
    v->len++;
    return 0;
}

static void vec_free(TG_StrVec *v) {
    size_t i;
    for (i = 0; i < v->len; i++) free(v->data[i]);
    free(v->data);
    memset(v, 0, sizeof(*v));
}

static int vec_load_file(TG_StrVec *v, const char *path) {
    FILE *fp;
    char line[4096];
    fp = fopen(path, "r");
    if (!fp) return -1;
    while (fgets(line, sizeof(line), fp)) {
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';
        trim(line);
        if (line[0] && vec_push(v, line) != 0) {
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}

static int vec_load_defaults(TG_StrVec *v) {
    size_t i;
    for (i = 0; i < sizeof(DEFAULT_TOPICS) / sizeof(DEFAULT_TOPICS[0]); i++) {
        if (vec_push(v, DEFAULT_TOPICS[i]) != 0) return -1;
    }
    return 0;
}

static uint32_t rng_next(uint32_t *state) {
    uint32_t x = *state ? *state : 2463534242u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static size_t rng_bounded(uint32_t *state, size_t n) {
    return n ? (size_t)(rng_next(state) % (uint32_t)n) : 0;
}

static void json_escape(FILE *out, const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    fputc('"', out);
    while (p && *p) {
        unsigned char c = *p++;
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
            fprintf(out, "\\u%04x", (unsigned)c);
        } else {
            fputc(c, out);
        }
    }
    fputc('"', out);
}

static char *json_get_string(const char *line, const char *key) {
    char needle[128];
    const char *p, *q;
    char *out;
    size_t cap = 256, len = 0;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = strstr(line, needle);
    if (!p) return NULL;
    p = strchr(p + strlen(needle), ':');
    if (!p) return NULL;
    p++;
    while (isspace((unsigned char)*p)) p++;
    if (*p != '"') return NULL;
    p++;
    out = (char *)malloc(cap);
    if (!out) return NULL;
    for (q = p; *q; q++) {
        char c = *q;
        if (c == '"') break;
        if (c == '\\') {
            q++;
            if (*q == 'n') c = '\n';
            else if (*q == 'r') c = '\r';
            else if (*q == 't') c = '\t';
            else if (*q) c = *q;
            else break;
        }
        if (len + 1 >= cap) {
            char *tmp;
            cap *= 2;
            tmp = (char *)realloc(out, cap);
            if (!tmp) { free(out); return NULL; }
            out = tmp;
        }
        out[len++] = c;
    }
    out[len] = '\0';
    return out;
}

static char *record_text(const char *line) {
    char *s = json_get_string(line, "text");
    if (!s) s = json_get_string(line, "content");
    if (!s) s = json_get_string(line, "output");
    if (!s) s = json_get_string(line, "completion");
    if (!s) s = json_get_string(line, "story");
    return s;
}

static int has_word_ci(const char *s, const char *w) {
    size_t wl = strlen(w);
    const char *p;
    for (p = s; p && *p; p++) {
        size_t i = 0;
        while (i < wl && p[i] && tolower((unsigned char)p[i]) == tolower((unsigned char)w[i])) i++;
        if (i == wl) return 1;
    }
    return 0;
}

static double textbook_score(const char *text) {
    size_t chars = 0, alpha = 0, digits = 0, punct = 0, words = 0;
    size_t sentences = 0, lines = 0;
    int in_word = 0;
    const unsigned char *p;
    double score = 0.0, avg_word_len;
    if (!text || !*text) return 0.0;
    for (p = (const unsigned char *)text; *p; p++) {
        unsigned char c = *p;
        chars++;
        if (isalpha(c)) alpha++;
        if (isdigit(c)) digits++;
        if (ispunct(c)) punct++;
        if (c == '\n') lines++;
        if (c == '.' || c == '?' || c == '!') sentences++;
        if (isalnum(c)) {
            if (!in_word) { words++; in_word = 1; }
        } else {
            in_word = 0;
        }
    }
    if (words < 30 || sentences < 2) return 0.0;
    avg_word_len = alpha ? (double)alpha / (double)words : 0.0;
    if ((double)alpha / (double)chars > 0.55) score += 0.18;
    if (avg_word_len >= 3.5 && avg_word_len <= 8.5) score += 0.14;
    if (sentences >= 4) score += 0.12;
    if (has_word_ci(text, "because") || has_word_ci(text, "therefore") || has_word_ci(text, "so ")) score += 0.14;
    if (has_word_ci(text, "example") || has_word_ci(text, "for instance")) score += 0.12;
    if (has_word_ci(text, "exercise") || has_word_ci(text, "answer") || has_word_ci(text, "question")) score += 0.12;
    if (has_word_ci(text, "step") || has_word_ci(text, "first") || has_word_ci(text, "then")) score += 0.10;
    if (has_word_ci(text, "mistake") || has_word_ci(text, "remember") || has_word_ci(text, "check")) score += 0.08;
    if (digits > 0) score += 0.04;
    if (punct > chars / 3) score -= 0.25;
    if (lines > 80) score -= 0.10;
    if (score < 0.0) score = 0.0;
    if (score > 1.0) score = 1.0;
    return score;
}

static int cmd_prompts(int argc, char **argv) {
    const char *out_path = NULL, *topics_path = NULL, *web_path = NULL;
    TG_StrVec topics = {0}, web = {0};
    size_t n = 0, i;
    uint32_t seed = 1;
    FILE *out;
    for (i = 2; i < (size_t)argc; i++) {
        if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < (size_t)argc) out_path = argv[++i];
        else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--count") == 0) && i + 1 < (size_t)argc) n = (size_t)strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--topics") == 0 && i + 1 < (size_t)argc) topics_path = argv[++i];
        else if (strcmp(argv[i], "--web-samples") == 0 && i + 1 < (size_t)argc) web_path = argv[++i];
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < (size_t)argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else return 2;
    }
    if (!out_path || n == 0) return 2;
    if (topics_path ? vec_load_file(&topics, topics_path) != 0 : vec_load_defaults(&topics) != 0) return 1;
    if (web_path && vec_load_file(&web, web_path) != 0) { vec_free(&topics); return 1; }
    out = fopen(out_path, "w");
    if (!out) { vec_free(&topics); vec_free(&web); return 1; }
    for (i = 0; i < n; i++) {
        const char *topic = topics.data[rng_bounded(&seed, topics.len)];
        const char *style = STYLES[rng_bounded(&seed, sizeof(STYLES) / sizeof(STYLES[0]))];
        const char *snippet = web.len ? web.data[rng_bounded(&seed, web.len)] : "";
        char prompt[8192];
        snprintf(prompt, sizeof(prompt),
                 "Write %s about \"%s\". Make it textbook-quality: clear, factual, self-contained, and useful for training a small language model. "
                 "Prefer common-sense reasoning, definitions, worked examples, and exercises with answers. Avoid web chatter, ads, personal data, unsafe content, and unexplained jargon. "
                 "If helpful, use this diversity seed from a web sample without copying it verbatim: %s",
                 style, topic, snippet);
        fprintf(out, "{\"id\":%zu,\"topic\":", i);
        json_escape(out, topic);
        fputs(",\"style\":", out); json_escape(out, style);
        fputs(",\"web_seed\":", out); json_escape(out, snippet);
        fputs(",\"prompt\":", out); json_escape(out, prompt);
        fputs("}\n", out);
    }
    fclose(out);
    vec_free(&topics); vec_free(&web);
    printf("textbook_prompts=%zu\noutput=%s\n", n, out_path);
    return 0;
}

static int cmd_filter_or_score(int argc, char **argv, int write_corpus) {
    const char *in_path = NULL, *out_path = NULL;
    double min_score = 0.55;
    FILE *in, *out = NULL;
    char *line = NULL;
    size_t cap = 0, total = 0, kept = 0;
    ssize_t nr;
    int i;
    for (i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) in_path = argv[++i];
        else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--min-score") == 0 && i + 1 < argc) min_score = atof(argv[++i]);
        else return 2;
    }
    if (!in_path || (write_corpus && !out_path)) return 2;
    in = fopen(in_path, "r");
    if (!in) return 1;
    if (write_corpus) {
        out = fopen(out_path, "w");
        if (!out) { fclose(in); return 1; }
    }
    while ((nr = getline(&line, &cap, in)) >= 0) {
        char *text = record_text(line);
        double s;
        (void)nr;
        if (!text) continue;
        total++;
        s = textbook_score(text);
        if (!write_corpus) {
            printf("record=%zu score=%.6f bytes=%zu\n", total, s, strlen(text));
        } else if (s >= min_score) {
            fprintf(out, "%s\n\n", text);
            kept++;
        }
        free(text);
    }
    free(line);
    fclose(in);
    if (out) fclose(out);
    printf("records=%zu\n", total);
    if (write_corpus) {
        printf("kept=%zu\nmin_score=%.6f\noutput=%s\n", kept, min_score, out_path);
    }
    return 0;
}

static int copy_quota(FILE *out, const char *path, size_t quota) {
    FILE *in;
    char buf[8192];
    size_t remaining = quota;
    in = fopen(path, "rb");
    if (!in) return -1;
    while (remaining > 0) {
        size_t want = remaining < sizeof(buf) ? remaining : sizeof(buf);
        size_t got = fread(buf, 1, want, in);
        if (!got) break;
        fwrite(buf, 1, got, out);
        remaining -= got;
    }
    fputs("\n\n", out);
    fclose(in);
    return 0;
}

static size_t file_size_or_zero(const char *path) {
    FILE *fp = fopen(path, "rb");
    long n;
    if (!fp) return 0;
    fseek(fp, 0, SEEK_END);
    n = ftell(fp);
    fclose(fp);
    return n > 0 ? (size_t)n : 0;
}

static int cmd_mix(int argc, char **argv) {
    const char *synthetic = NULL, *web = NULL, *code = NULL, *out_path = NULL;
    double rs = 0.40, rw = 0.40, rc = 0.20, sum;
    size_t max_bytes = 0, total_available, qs, qw, qc;
    FILE *out;
    int i;
    for (i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--synthetic") == 0 && i + 1 < argc) synthetic = argv[++i];
        else if (strcmp(argv[i], "--web") == 0 && i + 1 < argc) web = argv[++i];
        else if (strcmp(argv[i], "--code") == 0 && i + 1 < argc) code = argv[++i];
        else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--synthetic-ratio") == 0 && i + 1 < argc) rs = atof(argv[++i]);
        else if (strcmp(argv[i], "--web-ratio") == 0 && i + 1 < argc) rw = atof(argv[++i]);
        else if (strcmp(argv[i], "--code-ratio") == 0 && i + 1 < argc) rc = atof(argv[++i]);
        else if (strcmp(argv[i], "--max-bytes") == 0 && i + 1 < argc) max_bytes = (size_t)strtoull(argv[++i], NULL, 10);
        else return 2;
    }
    if (!synthetic || !web || !code || !out_path) return 2;
    sum = rs + rw + rc;
    if (sum <= 0.0) return 2;
    total_available = file_size_or_zero(synthetic) + file_size_or_zero(web) + file_size_or_zero(code);
    if (!max_bytes || max_bytes > total_available) max_bytes = total_available;
    qs = (size_t)((double)max_bytes * rs / sum);
    qw = (size_t)((double)max_bytes * rw / sum);
    qc = max_bytes > qs + qw ? max_bytes - qs - qw : 0;
    out = fopen(out_path, "wb");
    if (!out) return 1;
    if (copy_quota(out, synthetic, qs) != 0 || copy_quota(out, web, qw) != 0 || copy_quota(out, code, qc) != 0) {
        fclose(out);
        return 1;
    }
    fclose(out);
    printf("synthetic_bytes=%zu\nweb_bytes=%zu\ncode_bytes=%zu\noutput=%s\n", qs, qw, qc, out_path);
    return 0;
}

static int cmd_config(int argc, char **argv) {
    const char *out_path = NULL;
    FILE *out;
    int i;
    for (i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) out_path = argv[++i];
        else return 2;
    }
    if (!out_path) return 2;
    out = fopen(out_path, "w");
    if (!out) return 1;
    fputs("pipeline: phi_textbook_generator\n", out);
    fputs("paper: arXiv:2309.05463v1\n", out);
    fputs("principle: data_quality_over_raw_size\n", out);
    fputs("synthetic_textbook_data:\n", out);
    fputs("  topics: 20000_in_paper\n", out);
    fputs("  local_prompt_styles:\n", out);
    fputs("    - concise textbook lesson\n", out);
    fputs("    - worked example with step-by-step reasoning\n", out);
    fputs("    - Socratic teacher/student dialogue\n", out);
    fputs("    - lesson with exercises and answers\n", out);
    fputs("    - common mistakes and corrections\n", out);
    fputs("mixing:\n", out);
    fputs("  phi_1_5_training: {synthetic: 0.80, phi_1_data: 0.20}\n", out);
    fputs("  phi_1_5_web_variant: {synthetic: 0.40, filtered_web: 0.40, code: 0.20}\n", out);
    fclose(out);
    printf("output=%s\n", out_path);
    return 0;
}

int dm_textbook_generator_cli(int argc, char **argv) {
    const char *cmd;
    if (argc < 2) { usage(argv[0]); return 2; }
    cmd = argv[1];
    if (strcmp(cmd, "textbook") == 0 || strcmp(cmd, "dm_textbook_generator") == 0) {
        if (argc < 3) { usage(argv[0]); return 2; }
        cmd = argv[2];
        argv++;
        argc--;
    }
    if (strcmp(cmd, "prompts") == 0) return cmd_prompts(argc, argv);
    if (strcmp(cmd, "filter") == 0) return cmd_filter_or_score(argc, argv, 1);
    if (strcmp(cmd, "score") == 0) return cmd_filter_or_score(argc, argv, 0);
    if (strcmp(cmd, "mix") == 0) return cmd_mix(argc, argv);
    if (strcmp(cmd, "config") == 0) return cmd_config(argc, argv);
    usage(argv[0]);
    return 2;
}
