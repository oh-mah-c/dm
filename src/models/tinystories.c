#include "models/tinystories.h"

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
} TS_StrVec;

typedef struct {
    char *story;
    char *summary;
    char *features;
    char *sentence;
    char *words;
    char *beginning;
    char *completion;
    char *instructions;
} TS_Record;

static const char *const DEFAULT_VERBS[] = {
    "ask", "bring", "build", "clean", "climb", "come", "cry", "decorate",
    "draw", "eat", "find", "give", "go", "help", "hide", "jump", "learn",
    "look", "make", "open", "paint", "play", "run", "say", "see", "share",
    "sing", "sit", "sleep", "smile", "take", "thank", "walk", "want"
};

static const char *const DEFAULT_NOUNS[] = {
    "apple", "ball", "bed", "bird", "box", "bread", "butterfly", "car",
    "cat", "cheese", "dog", "door", "flower", "friend", "garden", "hat",
    "house", "mom", "park", "puppy", "room", "sandcastle", "soup", "spoon",
    "star", "sun", "thunder", "toy", "tree", "water", "wind"
};

static const char *const DEFAULT_ADJECTIVES[] = {
    "ancient", "angry", "big", "blue", "brave", "bright", "cold", "dark",
    "funny", "gentle", "good", "happy", "kind", "little", "loud", "new",
    "old", "pretty", "red", "sad", "scared", "shiny", "small", "sweet",
    "tall", "warm", "wide", "yummy"
};

static const char *const FEATURE_TEXT[] = {
    "the story should contain at least one dialogue",
    "the story has a bad ending",
    "the story has a moral value",
    "the story has a plot twist",
    "the story contains foreshadowing",
    "the story contains a conflict"
};

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s tinystories prompts -o prompts.jsonl -n N [--seed N]\n"
        "       [--verbs verbs.txt] [--nouns nouns.txt] [--adjectives adjectives.txt]\n"
        "       [--min-features N] [--max-features N]\n"
        "  %s tinystories instruct -i stories.jsonl -o instruct.jsonl [--seed N]\n"
        "  %s tinystories gpt-eval -i completions.jsonl -o eval_prompts.jsonl\n"
        "  %s tinystories verify -i stories.jsonl [--verbs verbs.txt] [--nouns nouns.txt] [--adjectives adjectives.txt]\n"
        "  %s tinystories config -o tinystories_config.json\n\n"
        "Input story JSONL fields are intentionally simple: story, summary, features,\n"
        "words, beginning, completion, instructions. The prompt generation follows\n"
        "TinyStories arXiv:2305.07759v2; actual GPT-3.5/GPT-4 calls are external.\n",
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

static void trim_in_place(char *s) {
    char *e;
    if (!s) return;
    while (isspace((unsigned char)*s)) memmove(s, s + 1, strlen(s));
    e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = '\0';
}

static int vec_push(TS_StrVec *v, const char *s) {
    char **nd;
    if (v->len == v->cap) {
        size_t nc = v->cap ? v->cap * 2 : 32;
        nd = (char **)realloc(v->data, nc * sizeof(*nd));
        if (!nd) return -1;
        v->data = nd;
        v->cap = nc;
    }
    v->data[v->len] = xstrdup(s);
    if (!v->data[v->len]) return -1;
    v->len++;
    return 0;
}

static void vec_free(TS_StrVec *v) {
    size_t i;
    if (!v) return;
    for (i = 0; i < v->len; i++) free(v->data[i]);
    free(v->data);
    memset(v, 0, sizeof(*v));
}

static int vec_load_default(TS_StrVec *v, const char *const *words, size_t n) {
    size_t i;
    for (i = 0; i < n; i++) if (vec_push(v, words[i]) != 0) return -1;
    return 0;
}

static int vec_load_file(TS_StrVec *v, const char *path) {
    FILE *fp;
    char line[1024];
    fp = fopen(path, "r");
    if (!fp) return -1;
    while (fgets(line, sizeof(line), fp)) {
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';
        trim_in_place(line);
        if (line[0] && vec_push(v, line) != 0) {
            fclose(fp);
            return -1;
        }
    }
    fclose(fp);
    return 0;
}

static uint32_t rng_next(uint32_t *state) {
    uint32_t x = *state;
    if (!x) x = 2463534242u;
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

static void append_text(char *buf, size_t cap, const char *text) {
    size_t used;
    if (!buf || !cap || !text) return;
    used = strlen(buf);
    if (used < cap) snprintf(buf + used, cap - used, "%s", text);
}

static void append_line_field(char *buf, size_t cap, const char *label, const char *value) {
    if (!value || !*value) return;
    append_text(buf, cap, label);
    append_text(buf, cap, value);
    append_text(buf, cap, "\n");
}

static void append_feature_list(char *buf, size_t cap, uint32_t mask) {
    int first = 1;
    size_t i;
    buf[0] = '\0';
    for (i = 0; i < sizeof(FEATURE_TEXT) / sizeof(FEATURE_TEXT[0]); i++) {
        if (mask & (1u << i)) {
            size_t used = strlen(buf);
            snprintf(buf + used, cap > used ? cap - used : 0, "%s%s",
                     first ? "" : ", ", FEATURE_TEXT[i]);
            first = 0;
        }
    }
}

static uint32_t sample_feature_mask(uint32_t *rng, int min_features, int max_features) {
    uint32_t available[6];
    uint32_t mask = 0;
    int i, k, nfeat;
    if (min_features < 0) min_features = 0;
    if (max_features > 6) max_features = 6;
    if (max_features < min_features) max_features = min_features;
    nfeat = min_features + (int)rng_bounded(rng, (size_t)(max_features - min_features + 1));
    for (i = 0; i < 6; i++) available[i] = (uint32_t)i;
    for (k = 0; k < nfeat; k++) {
        int j = k + (int)rng_bounded(rng, (size_t)(6 - k));
        uint32_t tmp = available[k];
        available[k] = available[j];
        available[j] = tmp;
        mask |= 1u << available[k];
    }
    return mask;
}

static int load_banks(TS_StrVec *verbs, TS_StrVec *nouns, TS_StrVec *adjs,
                      const char *verbs_path, const char *nouns_path, const char *adjs_path) {
    if (verbs_path) {
        if (vec_load_file(verbs, verbs_path) != 0) return -1;
    } else if (vec_load_default(verbs, DEFAULT_VERBS, sizeof(DEFAULT_VERBS) / sizeof(DEFAULT_VERBS[0])) != 0) return -1;
    if (nouns_path) {
        if (vec_load_file(nouns, nouns_path) != 0) return -1;
    } else if (vec_load_default(nouns, DEFAULT_NOUNS, sizeof(DEFAULT_NOUNS) / sizeof(DEFAULT_NOUNS[0])) != 0) return -1;
    if (adjs_path) {
        if (vec_load_file(adjs, adjs_path) != 0) return -1;
    } else if (vec_load_default(adjs, DEFAULT_ADJECTIVES, sizeof(DEFAULT_ADJECTIVES) / sizeof(DEFAULT_ADJECTIVES[0])) != 0) return -1;
    return (verbs->len && nouns->len && adjs->len) ? 0 : -1;
}

static int cmd_prompts(int argc, char **argv) {
    const char *out_path = NULL, *verbs_path = NULL, *nouns_path = NULL, *adjs_path = NULL;
    size_t n = 0, i;
    uint32_t seed = 1;
    int min_features = 1, max_features = 3;
    TS_StrVec verbs = {0}, nouns = {0}, adjs = {0};
    FILE *out;

    for (i = 2; i < (size_t)argc; i++) {
        if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < (size_t)argc) out_path = argv[++i];
        else if ((strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--count") == 0) && i + 1 < (size_t)argc) n = (size_t)strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < (size_t)argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--verbs") == 0 && i + 1 < (size_t)argc) verbs_path = argv[++i];
        else if (strcmp(argv[i], "--nouns") == 0 && i + 1 < (size_t)argc) nouns_path = argv[++i];
        else if (strcmp(argv[i], "--adjectives") == 0 && i + 1 < (size_t)argc) adjs_path = argv[++i];
        else if (strcmp(argv[i], "--min-features") == 0 && i + 1 < (size_t)argc) min_features = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-features") == 0 && i + 1 < (size_t)argc) max_features = atoi(argv[++i]);
        else return 2;
    }
    if (!out_path || n == 0 || load_banks(&verbs, &nouns, &adjs, verbs_path, nouns_path, adjs_path) != 0) {
        vec_free(&verbs); vec_free(&nouns); vec_free(&adjs);
        return 2;
    }
    out = fopen(out_path, "w");
    if (!out) {
        vec_free(&verbs); vec_free(&nouns); vec_free(&adjs);
        return 1;
    }
    for (i = 0; i < n; i++) {
        const char *verb = verbs.data[rng_bounded(&seed, verbs.len)];
        const char *noun = nouns.data[rng_bounded(&seed, nouns.len)];
        const char *adj = adjs.data[rng_bounded(&seed, adjs.len)];
        uint32_t mask = sample_feature_mask(&seed, min_features, max_features);
        char features[1024];
        char prompt[2048];
        append_feature_list(features, sizeof(features), mask);
        snprintf(prompt, sizeof(prompt),
                 "Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would likely understand. "
                 "The story should use the verb \"%s\", the noun \"%s\" and the adjective \"%s\". "
                 "The story should have the following features: %s. Remember to only use simple words!",
                 verb, noun, adj, features[0] ? features : "no extra feature constraints");
        fprintf(out, "{\"id\":%zu,\"verb\":", i);
        json_escape(out, verb);
        fputs(",\"noun\":", out); json_escape(out, noun);
        fputs(",\"adjective\":", out); json_escape(out, adj);
        fputs(",\"features\":", out); json_escape(out, features);
        fputs(",\"prompt\":", out); json_escape(out, prompt);
        fputs("}\n", out);
    }
    fclose(out);
    vec_free(&verbs); vec_free(&nouns); vec_free(&adjs);
    printf("tinystories_prompts=%zu\n", n);
    printf("output=%s\n", out_path);
    return 0;
}

static char *json_get_string(const char *line, const char *key) {
    char needle[128];
    const char *p, *q;
    char *out;
    size_t cap = 128, len = 0;
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

static void record_free(TS_Record *r) {
    free(r->story); free(r->summary); free(r->features); free(r->sentence);
    free(r->words); free(r->beginning); free(r->completion); free(r->instructions);
    memset(r, 0, sizeof(*r));
}

static TS_Record record_parse(const char *line) {
    TS_Record r;
    memset(&r, 0, sizeof(r));
    r.story = json_get_string(line, "story");
    r.summary = json_get_string(line, "summary");
    r.features = json_get_string(line, "features");
    r.sentence = json_get_string(line, "sentence");
    r.words = json_get_string(line, "words");
    r.beginning = json_get_string(line, "beginning");
    r.completion = json_get_string(line, "completion");
    r.instructions = json_get_string(line, "instructions");
    return r;
}

static char *extract_random_sentence(const char *story, uint32_t *seed) {
    const char *starts[256];
    size_t lens[256];
    size_t n = 0, i;
    const char *p = story, *s = story;
    if (!story) return xstrdup("");
    while (*p && n < 256) {
        if (*p == '.' || *p == '!' || *p == '?') {
            const char *e = p + 1;
            while (*s && isspace((unsigned char)*s)) s++;
            if ((size_t)(e - s) > 4) {
                starts[n] = s;
                lens[n] = (size_t)(e - s);
                n++;
            }
            s = e;
        }
        p++;
    }
    i = n > 1 ? 1 + rng_bounded(seed, n - 1) : 0;
    if (n == 0) return xstrdup("");
    {
        char *out = (char *)malloc(lens[i] + 1);
        if (!out) return NULL;
        memcpy(out, starts[i], lens[i]);
        out[lens[i]] = '\0';
        return out;
    }
}

static int cmd_instruct(int argc, char **argv) {
    const char *in_path = NULL, *out_path = NULL;
    uint32_t seed = 1;
    FILE *in, *out;
    char *line = NULL;
    size_t cap = 0, count = 0;
    ssize_t nr;
    int i;
    for (i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) in_path = argv[++i];
        else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) out_path = argv[++i];
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        else return 2;
    }
    if (!in_path || !out_path) return 2;
    in = fopen(in_path, "r");
    if (!in) return 1;
    out = fopen(out_path, "w");
    if (!out) { fclose(in); return 1; }
    while ((nr = getline(&line, &cap, in)) >= 0) {
        TS_Record r = record_parse(line);
        uint32_t mask = rng_next(&seed);
        char *sentence = r.sentence ? xstrdup(r.sentence) : extract_random_sentence(r.story, &seed);
        char instruction[8192];
        (void)nr;
        if (!r.story) { free(sentence); record_free(&r); continue; }
        if (!(mask & 15u)) mask |= 1u;
        instruction[0] = '\0';
        if (mask & 1u) append_line_field(instruction, sizeof(instruction), "Summary: ", r.summary);
        if (mask & 2u) append_line_field(instruction, sizeof(instruction), "Features: ", r.features);
        if (mask & 4u) append_line_field(instruction, sizeof(instruction), "Sentence: ", sentence);
        if (mask & 8u) append_line_field(instruction, sizeof(instruction), "Words: ", r.words);
        fprintf(out, "{\"id\":%zu,\"instruction\":", count);
        json_escape(out, instruction);
        fputs(",\"story\":", out); json_escape(out, r.story);
        fputs("}\n", out);
        count++;
        free(sentence);
        record_free(&r);
    }
    free(line);
    fclose(in); fclose(out);
    printf("tinystories_instruct=%zu\n", count);
    printf("output=%s\n", out_path);
    return 0;
}

static int contains_word_ci(const char *text, const char *word) {
    size_t wl;
    const char *p;
    if (!text || !word || !*word) return 0;
    wl = strlen(word);
    for (p = text; *p; p++) {
        size_t i = 0;
        while (i < wl && p[i] && tolower((unsigned char)p[i]) == tolower((unsigned char)word[i])) i++;
        if (i == wl) {
            int left = (p == text) || !isalpha((unsigned char)p[-1]);
            int right = !isalpha((unsigned char)p[i]);
            if (left && right) return 1;
        }
    }
    return 0;
}

static int cmd_verify(int argc, char **argv) {
    const char *in_path = NULL, *verbs_path = NULL, *nouns_path = NULL, *adjs_path = NULL;
    TS_StrVec verbs = {0}, nouns = {0}, adjs = {0};
    FILE *in;
    char *line = NULL;
    size_t cap = 0, stories = 0, word_hits = 0, word_checks = 0;
    ssize_t nr;
    int i;
    for (i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) in_path = argv[++i];
        else if (strcmp(argv[i], "--verbs") == 0 && i + 1 < argc) verbs_path = argv[++i];
        else if (strcmp(argv[i], "--nouns") == 0 && i + 1 < argc) nouns_path = argv[++i];
        else if (strcmp(argv[i], "--adjectives") == 0 && i + 1 < argc) adjs_path = argv[++i];
        else return 2;
    }
    if (!in_path || load_banks(&verbs, &nouns, &adjs, verbs_path, nouns_path, adjs_path) != 0) return 2;
    in = fopen(in_path, "r");
    if (!in) { vec_free(&verbs); vec_free(&nouns); vec_free(&adjs); return 1; }
    while ((nr = getline(&line, &cap, in)) >= 0) {
        TS_Record r = record_parse(line);
        char *verb = json_get_string(line, "verb");
        char *noun = json_get_string(line, "noun");
        char *adj = json_get_string(line, "adjective");
        (void)nr;
        if (r.story) {
            stories++;
            if (verb) { word_checks++; word_hits += contains_word_ci(r.story, verb) ? 1u : 0u; }
            if (noun) { word_checks++; word_hits += contains_word_ci(r.story, noun) ? 1u : 0u; }
            if (adj) { word_checks++; word_hits += contains_word_ci(r.story, adj) ? 1u : 0u; }
        }
        free(verb); free(noun); free(adj);
        record_free(&r);
    }
    free(line);
    fclose(in);
    printf("stories=%zu\n", stories);
    printf("required_word_hits=%zu\n", word_hits);
    printf("required_word_checks=%zu\n", word_checks);
    printf("required_word_hit_rate=%.6f\n", word_checks ? (double)word_hits / (double)word_checks : 0.0);
    vec_free(&verbs); vec_free(&nouns); vec_free(&adjs);
    return 0;
}

static int cmd_gpt_eval(int argc, char **argv) {
    const char *in_path = NULL, *out_path = NULL;
    FILE *in, *out;
    char *line = NULL;
    size_t cap = 0, count = 0;
    ssize_t nr;
    int i;
    for (i = 2; i < argc; i++) {
        if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) && i + 1 < argc) in_path = argv[++i];
        else if ((strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) && i + 1 < argc) out_path = argv[++i];
        else return 2;
    }
    if (!in_path || !out_path) return 2;
    in = fopen(in_path, "r");
    if (!in) return 1;
    out = fopen(out_path, "w");
    if (!out) { fclose(in); return 1; }
    while ((nr = getline(&line, &cap, in)) >= 0) {
        TS_Record r = record_parse(line);
        char prompt[8192];
        (void)nr;
        if (!r.beginning || !r.completion) { record_free(&r); continue; }
        if (r.instructions && *r.instructions) {
            snprintf(prompt, sizeof(prompt),
                     "The following exercise asks a student to write a short story from instructions.\\nInstructions:\\n%s\\n***\\nStudent story:\\n%s\\n\\n"
                     "Please provide your general assessment. Is it grammatically correct? Does it follow the instructions? Is the plot coherent?\\n"
                     "Now grade the student story in terms of Grammar, Creativity, Consistency with the instructions, and Plot. Use scores from 1/10 to 10/10 and provide an age group.",
                     r.instructions, r.completion);
        } else {
            snprintf(prompt, sizeof(prompt),
                     "In the following exercise, the student is given a beginning of a story. The student needs to complete it into a full story. "
                     "The exercise tests the student's language abilities and creativity. The symbol *** marks the separator between the prescribed beginning and the student's completion:\\n"
                     "%s***%s\\n\\n"
                     "Please provide your general assessment about the part written by the student (the one after the *** symbol). "
                     "Is it gramatically correct? Is it consistent with the beginning of the story? Pay special attention to whether the student manages to complete the sentence which is split in the middle by the separator ***.\\n"
                     "Now, grade the student's completion in terms of grammar, creativity, consistency with the story's beginning and whether the plot makes sense. "
                     "Moreover, please provide your best guess of what the age of the student might be, as reflected from the completion. "
                     "Choose from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E: 10-12. F: 13-16.",
                     r.beginning, r.completion);
        }
        fprintf(out, "{\"id\":%zu,\"gpt_eval_prompt\":", count);
        json_escape(out, prompt);
        fputs("}\n", out);
        count++;
        record_free(&r);
    }
    free(line);
    fclose(in); fclose(out);
    printf("gpt_eval_prompts=%zu\n", count);
    printf("output=%s\n", out_path);
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
    fputs("{\n", out);
    fputs("  \"paper\": \"TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv:2305.07759v2\",\n", out);
    fputs("  \"dataset\": {\n", out);
    fputs("    \"story_format\": \"short English stories, 3-5 paragraphs in the generation prompt\",\n", out);
    fputs("    \"vocabulary_constraint\": \"words a typical 3-4 year old child would understand\",\n", out);
    fputs("    \"word_sampling\": \"one verb, one noun, one adjective sampled per generation\",\n", out);
    fputs("    \"feature_sampling\": [\"dialogue\", \"bad ending\", \"moral value\", \"plot twist\", \"foreshadowing\", \"conflict\"]\n", out);
    fputs("  },\n", out);
    fputs("  \"model\": {\n", out);
    fputs("    \"architecture_family\": \"GPT-Neo style autoregressive transformer\",\n", out);
    fputs("    \"context_length\": 512,\n", out);
    fputs("    \"attention_window_size\": 256,\n", out);
    fputs("    \"tokenizer\": \"GPT-Neo tokenizer restricted to top 10000 most common tokens\",\n", out);
    fputs("    \"reported_parameter_scales\": [1000000, 2500000, 9000000, 28000000, 33000000]\n", out);
    fputs("  },\n", out);
    fputs("  \"evaluation\": {\n", out);
    fputs("    \"judge\": \"GPT-4\",\n", out);
    fputs("    \"story_completion_categories\": [\"grammar\", \"creativity\", \"consistency\", \"plot sense\", \"age group\"],\n", out);
    fputs("    \"instruct_categories\": [\"grammar\", \"creativity\", \"instruction consistency\", \"plot\"]\n", out);
    fputs("  }\n", out);
    fputs("}\n", out);
    fclose(out);
    printf("output=%s\n", out_path);
    return 0;
}

int dm_tinystories_cli(int argc, char **argv) {
    const char *cmd;
    if (argc < 2) { usage(argv[0]); return 2; }
    cmd = argv[1];
    if (strcmp(cmd, "tinystories") == 0 || strcmp(cmd, "dm_tinystories") == 0) {
        if (argc < 3) { usage(argv[0]); return 2; }
        cmd = argv[2];
        argv++;
        argc--;
    }
    if (strcmp(cmd, "prompts") == 0) return cmd_prompts(argc, argv);
    if (strcmp(cmd, "instruct") == 0) return cmd_instruct(argc, argv);
    if (strcmp(cmd, "gpt-eval") == 0) return cmd_gpt_eval(argc, argv);
    if (strcmp(cmd, "verify") == 0) return cmd_verify(argc, argv);
    if (strcmp(cmd, "config") == 0) return cmd_config(argc, argv);
    usage(argv[0]);
    return 2;
}
