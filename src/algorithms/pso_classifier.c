#include "algorithms/pso_classifier.h"
#include "core/dm_benchmark.h"

#include <ctype.h>
#include <dirent.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

typedef struct {
    char *path;
    char *label;
    size_t lines;
} ClassFile;

typedef struct {
    size_t n_records;
    size_t n_attrs;
    size_t n_classes;
    size_t words;
    uint64_t *bits;
    int *cls;
    char **labels;
} Dataset;

typedef struct {
    double *x;
    double *v;
    double *best;
    double q;
    double best_q;
} Particle;

typedef struct {
    double *values;
    int cls;
    int is_default;
} Rule;

typedef struct {
    Rule *rules;
    size_t count;
    size_t cap;
} RuleSet;

typedef struct {
    const Dataset *ds;
    unsigned char *active;
    unsigned char *is_train;
    size_t active_count;
    int target_class;
    const PSOClassifierParams *params;
} TrainView;

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *r = (char *)malloc(n);
    if (r) memcpy(r, s, n);
    return r;
}

static double now_sec(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static double rnd01(void) {
    return (double)rand() / ((double)RAND_MAX + 1.0);
}

static int ends_with(const char *s, const char *suffix) {
    size_t n = strlen(s), m = strlen(suffix);
    return n >= m && strcmp(s + n - m, suffix) == 0;
}

static int cmp_class_file(const void *a, const void *b) {
    const ClassFile *x = (const ClassFile *)a;
    const ClassFile *y = (const ClassFile *)b;
    return strcmp(x->label, y->label);
}

static void free_class_files(ClassFile *files, size_t n) {
    if (!files) return;
    for (size_t i = 0; i < n; i++) {
        free(files[i].path);
        free(files[i].label);
    }
    free(files);
}

static char *make_path(const char *folder, const char *name) {
    size_t a = strlen(folder), b = strlen(name);
    int slash = a > 0 && folder[a - 1] == '/';
    char *p = (char *)malloc(a + b + (slash ? 1 : 2));
    if (!p) return NULL;
    sprintf(p, "%s%s%s", folder, slash ? "" : "/", name);
    return p;
}

static char *label_from_name(const char *name) {
    char *label = xstrdup(name);
    if (!label) return NULL;
    char *p = strstr(label, "translated.txt");
    if (p) *p = '\0';
    size_t n = strlen(label);
    while (n > 0 && (label[n - 1] == '_' || label[n - 1] == '-' || label[n - 1] == '.')) {
        label[--n] = '\0';
    }
    return label;
}

static int collect_class_files(const char *folder, ClassFile **out, size_t *out_n) {
    DIR *dir = opendir(folder);
    if (!dir) return -1;
    ClassFile *files = NULL;
    size_t n = 0, cap = 0;
    struct dirent *ent;
    while ((ent = readdir(dir)) != NULL) {
        if (!ends_with(ent->d_name, "translated.txt")) continue;
        if (n == cap) {
            size_t nc = cap ? cap * 2 : 8;
            ClassFile *nf = (ClassFile *)realloc(files, nc * sizeof(*files));
            if (!nf) {
                closedir(dir);
                free_class_files(files, n);
                return -1;
            }
            files = nf;
            cap = nc;
        }
        files[n].path = make_path(folder, ent->d_name);
        files[n].label = label_from_name(ent->d_name);
        files[n].lines = 0;
        if (!files[n].path || !files[n].label) {
            closedir(dir);
            free_class_files(files, n + 1);
            return -1;
        }
        n++;
    }
    closedir(dir);
    qsort(files, n, sizeof(*files), cmp_class_file);
    *out = files;
    *out_n = n;
    return n > 1 ? 0 : -1;
}

static int parse_line_items(char *line, unsigned char *seen, size_t seen_n, uint32_t *max_item) {
    int has_item = 0;
    char *p = line;
    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p) break;
        if (v > 0) {
            has_item = 1;
            if ((uint32_t)v > *max_item) *max_item = (uint32_t)v;
            if (seen && (size_t)v <= seen_n) seen[v - 1] = 1;
        }
        p = end;
    }
    return has_item;
}

static int first_pass(ClassFile *files, size_t n, uint32_t *max_item, size_t *records) {
    char line[1048576];
    *max_item = 0;
    *records = 0;
    for (size_t c = 0; c < n; c++) {
        FILE *fp = fopen(files[c].path, "r");
        if (!fp) return -1;
        while (fgets(line, sizeof(line), fp)) {
            if (line[0] == '@' || line[0] == '\n' || line[0] == '\r') continue;
            uint32_t local_max = *max_item;
            if (parse_line_items(line, NULL, 0, &local_max)) {
                *max_item = local_max;
                files[c].lines++;
                (*records)++;
            }
        }
        fclose(fp);
    }
    return *records > 0 && *max_item > 0 ? 0 : -1;
}

static void dataset_free(Dataset *ds) {
    if (!ds) return;
    free(ds->bits);
    free(ds->cls);
    if (ds->labels) {
        for (size_t i = 0; i < ds->n_classes; i++) free(ds->labels[i]);
    }
    free(ds->labels);
    memset(ds, 0, sizeof(*ds));
}

static int load_dataset(const char *folder, const PSOClassifierParams *params, Dataset *ds) {
    memset(ds, 0, sizeof(*ds));
    ClassFile *files = NULL;
    size_t classes = 0, records = 0;
    uint32_t max_item = 0;
    if (collect_class_files(folder, &files, &classes) != 0) return -1;
    if (first_pass(files, classes, &max_item, &records) != 0) {
        free_class_files(files, classes);
        return -1;
    }
    if (params->max_records > 0 && records > params->max_records) records = params->max_records;
    ds->n_records = records;
    ds->n_attrs = max_item;
    ds->n_classes = classes;
    ds->words = (ds->n_attrs + 63) / 64;
    ds->bits = (uint64_t *)calloc(ds->n_records * ds->words, sizeof(uint64_t));
    ds->cls = (int *)malloc(ds->n_records * sizeof(int));
    ds->labels = (char **)calloc(classes, sizeof(char *));
    if (!ds->bits || !ds->cls || !ds->labels) {
        free_class_files(files, classes);
        dataset_free(ds);
        return -1;
    }
    for (size_t c = 0; c < classes; c++) {
        ds->labels[c] = xstrdup(files[c].label);
        if (!ds->labels[c]) {
            free_class_files(files, classes);
            dataset_free(ds);
            return -1;
        }
    }
    unsigned char *seen = (unsigned char *)calloc(ds->n_attrs, 1);
    char line[1048576];
    size_t row = 0;
    for (size_t c = 0; c < classes && row < ds->n_records; c++) {
        FILE *fp = fopen(files[c].path, "r");
        if (!fp) {
            free(seen);
            free_class_files(files, classes);
            dataset_free(ds);
            return -1;
        }
        while (row < ds->n_records && fgets(line, sizeof(line), fp)) {
            if (line[0] == '@' || line[0] == '\n' || line[0] == '\r') continue;
            memset(seen, 0, ds->n_attrs);
            uint32_t unused = 0;
            if (!parse_line_items(line, seen, ds->n_attrs, &unused)) continue;
            uint64_t *bits = ds->bits + row * ds->words;
            for (size_t a = 0; a < ds->n_attrs; a++) {
                if (seen[a]) bits[a / 64] |= (uint64_t)1 << (a % 64);
            }
            ds->cls[row] = (int)c;
            row++;
        }
        fclose(fp);
    }
    free(seen);
    free_class_files(files, classes);
    ds->n_records = row;
    return row > 0 ? 0 : -1;
}

static int bit_present(const Dataset *ds, size_t row, size_t attr) {
    const uint64_t *bits = ds->bits + row * ds->words;
    return (bits[attr / 64] >> (attr % 64)) & 1u;
}

static int rule_attr_matches(double v, int present, double t) {
    if (v >= t) return 1;
    if (v < 0.0 || v > 1.0) return 0;
    int rule_bin = (int)floor(v * 2.0);
    if (rule_bin < 0) rule_bin = 0;
    if (rule_bin > 1) rule_bin = 1;
    return rule_bin == present;
}

static int rule_covers(const Dataset *ds, const double *values, size_t row, double t) {
    for (size_t a = 0; a < ds->n_attrs; a++) {
        if (!rule_attr_matches(values[a], bit_present(ds, row, a), t)) return 0;
    }
    return 1;
}

static size_t rule_test_count(const double *values, size_t n_attrs, double t) {
    size_t k = 0;
    for (size_t a = 0; a < n_attrs; a++) {
        if (values[a] < t) k++;
    }
    return k;
}

static double evaluate_rule(const TrainView *view, const double *values) {
    const Dataset *ds = view->ds;
    for (size_t a = 0; a < ds->n_attrs; a++) {
        if (values[a] < 0.0 || values[a] > 1.0) return -1.0;
    }
    size_t tp = 0, tn = 0, fp = 0, fn = 0;
    for (size_t i = 0; i < ds->n_records; i++) {
        if (!view->is_train[i] || !view->active[i]) continue;
        int target = ds->cls[i] == view->target_class;
        int covers = rule_covers(ds, values, i, view->params->indifference_threshold);
        if (covers && target) tp++;
        else if (covers && !target) fp++;
        else if (!covers && !target) tn++;
        else fn++;
    }
    double sens = (tp + fn) ? (double)tp / (double)(tp + fn) : 1.0;
    double spec = (tn + fp) ? (double)tn / (double)(tn + fp) : 1.0;
    return sens * spec;
}

static int allocate_particles(Particle **out, size_t n, size_t dims) {
    Particle *p = (Particle *)calloc(n, sizeof(*p));
    if (!p) return -1;
    for (size_t i = 0; i < n; i++) {
        p[i].x = (double *)malloc(dims * sizeof(double));
        p[i].v = (double *)malloc(dims * sizeof(double));
        p[i].best = (double *)malloc(dims * sizeof(double));
        if (!p[i].x || !p[i].v || !p[i].best) {
            for (size_t j = 0; j <= i; j++) {
                free(p[j].x);
                free(p[j].v);
                free(p[j].best);
            }
            free(p);
            return -1;
        }
    }
    *out = p;
    return 0;
}

static void free_particles(Particle *p, size_t n) {
    if (!p) return;
    for (size_t i = 0; i < n; i++) {
        free(p[i].x);
        free(p[i].v);
        free(p[i].best);
    }
    free(p);
}

static double normalized_distance(const double *a, const double *b, size_t dims) {
    double sum = 0.0;
    for (size_t i = 0; i < dims; i++) {
        double d = a[i] - b[i];
        sum += d * d;
    }
    return dims ? sqrt(sum) / sqrt((double)dims) : 0.0;
}

static int swarm_converged(const Particle *p, size_t n, size_t dims, const double *best, double radius) {
    for (size_t i = 0; i < n; i++) {
        if (normalized_distance(p[i].x, best, dims) > radius) return 0;
    }
    return 1;
}

static int discover_rule(const TrainView *view, double *best_values, size_t *iterations) {
    const size_t n = view->params->particles ? view->params->particles : 25;
    const size_t dims = view->ds->n_attrs;
    Particle *p = NULL;
    if (allocate_particles(&p, n, dims) != 0) return -1;
    double *gbest = (double *)malloc(dims * sizeof(double));
    if (!gbest) {
        free_particles(p, n);
        return -1;
    }
    double gq = -2.0;
    for (size_t i = 0; i < n; i++) {
        for (size_t d = 0; d < dims; d++) {
            p[i].x[d] = rnd01();
            p[i].v[d] = rnd01() * 2.0 - 1.0;
            p[i].best[d] = p[i].x[d];
        }
        p[i].q = evaluate_rule(view, p[i].x);
        p[i].best_q = p[i].q;
        if (p[i].q > gq) {
            gq = p[i].q;
            memcpy(gbest, p[i].x, dims * sizeof(double));
        }
    }
    size_t iter = 0;
    size_t max_iter = view->params->max_iterations ? view->params->max_iterations : 1000;
    for (; iter < max_iter; iter++) {
        for (size_t i = 0; i < n; i++) {
            p[i].q = evaluate_rule(view, p[i].x);
            if (p[i].q > p[i].best_q) {
                p[i].best_q = p[i].q;
                memcpy(p[i].best, p[i].x, dims * sizeof(double));
            }
            if (p[i].q > gq) {
                gq = p[i].q;
                memcpy(gbest, p[i].x, dims * sizeof(double));
            }
        }
        if (swarm_converged(p, n, dims, gbest, view->params->convergence_radius)) break;
        for (size_t i = 0; i < n; i++) {
            for (size_t d = 0; d < dims; d++) {
                double u1 = rnd01() * view->params->acceleration_limit;
                double u2 = rnd01() * view->params->acceleration_limit;
                p[i].v[d] = view->params->constriction *
                    (p[i].v[d] + u1 * (p[i].best[d] - p[i].x[d]) + u2 * (gbest[d] - p[i].x[d]));
                p[i].x[d] += p[i].v[d];
                if (p[i].x[d] < -1.0) p[i].x[d] = -1.0;
                if (p[i].x[d] > 2.0) p[i].x[d] = 2.0;
            }
        }
    }
    memcpy(best_values, gbest, dims * sizeof(double));
    *iterations += iter + 1;
    free(gbest);
    free_particles(p, n);
    return 0;
}

static void prune_rule(const TrainView *view, double *values) {
    const double t = view->params->indifference_threshold;
    double base = evaluate_rule(view, values);
    for (size_t a = 0; a < view->ds->n_attrs; a++) {
        if (values[a] >= t) continue;
        double old = values[a];
        values[a] = 1.0;
        double q = evaluate_rule(view, values);
        if (q + 1e-12 < base) {
            values[a] = old;
        } else {
            base = q;
        }
    }
}

static int ruleset_add(RuleSet *rs, const double *values, size_t dims, int cls, int is_default) {
    if (rs->count == rs->cap) {
        size_t nc = rs->cap ? rs->cap * 2 : 16;
        Rule *nr = (Rule *)realloc(rs->rules, nc * sizeof(*nr));
        if (!nr) return -1;
        rs->rules = nr;
        rs->cap = nc;
    }
    Rule *r = &rs->rules[rs->count++];
    r->values = NULL;
    r->cls = cls;
    r->is_default = is_default;
    if (!is_default) {
        r->values = (double *)malloc(dims * sizeof(double));
        if (!r->values) return -1;
        memcpy(r->values, values, dims * sizeof(double));
    }
    return 0;
}

static void ruleset_free(RuleSet *rs) {
    if (!rs) return;
    for (size_t i = 0; i < rs->count; i++) free(rs->rules[i].values);
    free(rs->rules);
    memset(rs, 0, sizeof(*rs));
}

static int predominant_class(const Dataset *ds, const unsigned char *is_train, const unsigned char *active) {
    size_t *counts = (size_t *)calloc(ds->n_classes, sizeof(size_t));
    if (!counts) return 0;
    for (size_t i = 0; i < ds->n_records; i++) {
        if (is_train[i] && (!active || active[i])) counts[ds->cls[i]]++;
    }
    int best = 0;
    for (size_t c = 1; c < ds->n_classes; c++) {
        if (counts[c] > counts[best]) best = (int)c;
    }
    free(counts);
    return best;
}

static int is_subset_rule(const Rule *a, const Rule *b, size_t dims, double t) {
    if (a->is_default || b->is_default) return 0;
    for (size_t d = 0; d < dims; d++) {
        if (a->values[d] >= t) continue;
        if (b->values[d] >= t) return 0;
        if ((int)floor(a->values[d] * 2.0) != (int)floor(b->values[d] * 2.0)) return 0;
    }
    return 1;
}

static void remove_rule(RuleSet *rs, size_t idx) {
    free(rs->rules[idx].values);
    for (size_t i = idx + 1; i < rs->count; i++) rs->rules[i - 1] = rs->rules[i];
    rs->count--;
}

static size_t clean_ruleset(RuleSet *rs, size_t dims, double t) {
    size_t removed = 0;
    for (size_t i = 0; i < rs->count; i++) {
        for (size_t j = 0; j < i; j++) {
            if (is_subset_rule(&rs->rules[j], &rs->rules[i], dims, t)) {
                remove_rule(rs, i--);
                removed++;
                break;
            }
        }
    }
    while (rs->count >= 2) {
        Rule *last = &rs->rules[rs->count - 1];
        Rule *prev = &rs->rules[rs->count - 2];
        if (!last->is_default || prev->is_default || prev->cls != last->cls) break;
        remove_rule(rs, rs->count - 2);
        removed++;
    }
    return removed;
}

static int build_ruleset(const Dataset *ds, const unsigned char *is_train,
                         const PSOClassifierParams *params, RuleSet *rs,
                         PSOClassifierStats *stats) {
    memset(rs, 0, sizeof(*rs));
    unsigned char *active = (unsigned char *)calloc(ds->n_records, 1);
    double *values = (double *)malloc(ds->n_attrs * sizeof(double));
    if (!active || !values) {
        free(active);
        free(values);
        return -1;
    }
    size_t train_count = 0;
    for (size_t i = 0; i < ds->n_records; i++) {
        if (is_train[i]) {
            active[i] = 1;
            train_count++;
        }
    }
    size_t active_count = train_count;
    size_t stop_count = (size_t)ceil(params->uncovered_ratio * (double)train_count);
    if (stop_count < 1) stop_count = 1;
    while (active_count > stop_count) {
        TrainView view;
        view.ds = ds;
        view.active = active;
        view.is_train = (unsigned char *)is_train;
        view.active_count = active_count;
        view.target_class = predominant_class(ds, is_train, active);
        view.params = params;
        if (discover_rule(&view, values, &stats->total_iterations) != 0) {
            free(active);
            free(values);
            ruleset_free(rs);
            return -1;
        }
        prune_rule(&view, values);
        size_t removed_now = 0;
        for (size_t i = 0; i < ds->n_records; i++) {
            if (!is_train[i] || !active[i] || ds->cls[i] != view.target_class) continue;
            if (rule_covers(ds, values, i, params->indifference_threshold)) {
                active[i] = 0;
                removed_now++;
            }
        }
        if (removed_now == 0) {
            for (size_t a = 0; a < ds->n_attrs; a++) values[a] = 1.0;
            int cls = predominant_class(ds, is_train, active);
            if (ruleset_add(rs, values, ds->n_attrs, cls, 0) != 0) {
                free(active);
                free(values);
                ruleset_free(rs);
                return -1;
            }
            break;
        }
        if (ruleset_add(rs, values, ds->n_attrs, view.target_class, 0) != 0) {
            free(active);
            free(values);
            ruleset_free(rs);
            return -1;
        }
        active_count -= removed_now;
        stats->total_removed_instances += removed_now;
    }
    int def_cls = predominant_class(ds, is_train, active);
    if (ruleset_add(rs, NULL, ds->n_attrs, def_cls, 1) != 0) {
        free(active);
        free(values);
        ruleset_free(rs);
        return -1;
    }
    stats->total_cleaned_rules += clean_ruleset(rs, ds->n_attrs, params->indifference_threshold);
    free(active);
    free(values);
    return 0;
}

static int classify_record(const Dataset *ds, const RuleSet *rs, size_t row, double t) {
    for (size_t i = 0; i < rs->count; i++) {
        const Rule *r = &rs->rules[i];
        if (r->is_default || rule_covers(ds, r->values, row, t)) return r->cls;
    }
    return rs->count ? rs->rules[rs->count - 1].cls : 0;
}

PSOClassifierParams pso_classifier_default_params(void) {
    PSOClassifierParams p;
    p.particles = 25;
    p.max_iterations = 1000;
    p.indifference_threshold = 0.90;
    p.convergence_radius = 0.01;
    p.uncovered_ratio = 0.10;
    p.constriction = 0.73;
    p.acceleration_limit = 2.05;
    p.seed = 7;
    p.max_records = 0;
    return p;
}

int pso_classifier_run_spmf_folder(const char *folder,
                                   const PSOClassifierParams *params_in,
                                   PSOClassifierStats *stats) {
    if (!folder || !stats) return -1;
    PSOClassifierParams params = params_in ? *params_in : pso_classifier_default_params();
    memset(stats, 0, sizeof(*stats));
    srand(params.seed);
    Dataset ds;
    if (load_dataset(folder, &params, &ds) != 0) return -1;
    stats->records = ds.n_records;
    stats->attributes = ds.n_attrs;
    stats->classes = ds.n_classes;
    stats->folds = 10;
    stats->particles = params.particles;
    double fold_acc[10];
    memset(fold_acc, 0, sizeof(fold_acc));
    double train_start = now_sec();
    for (size_t fold = 0; fold < 10; fold++) {
        unsigned char *is_train = (unsigned char *)calloc(ds.n_records, 1);
        if (!is_train) {
            dataset_free(&ds);
            return -1;
        }
        size_t test_count = 0;
        for (size_t i = 0; i < ds.n_records; i++) {
            if (i % 10 == fold) test_count++;
            else is_train[i] = 1;
        }
        RuleSet rs;
        if (build_ruleset(&ds, is_train, &params, &rs, stats) != 0) {
            free(is_train);
            dataset_free(&ds);
            return -1;
        }
        stats->total_rules += rs.count;
        for (size_t r = 0; r < rs.count; r++) {
            if (!rs.rules[r].is_default) {
                stats->total_attribute_tests += rule_test_count(rs.rules[r].values, ds.n_attrs, params.indifference_threshold);
            }
        }
        double val_start = now_sec();
        size_t correct = 0;
        for (size_t i = 0; i < ds.n_records; i++) {
            if (i % 10 != fold) continue;
            int pred = classify_record(&ds, &rs, i, params.indifference_threshold);
            if (pred == ds.cls[i]) correct++;
        }
        stats->validation_sec += now_sec() - val_start;
        fold_acc[fold] = test_count ? (double)correct / (double)test_count : 0.0;
        ruleset_free(&rs);
        free(is_train);
    }
    stats->training_sec = now_sec() - train_start - stats->validation_sec;
    double sum = 0.0;
    for (size_t i = 0; i < 10; i++) sum += fold_acc[i];
    stats->accuracy_mean = sum / 10.0;
    double var = 0.0;
    for (size_t i = 0; i < 10; i++) {
        double d = fold_acc[i] - stats->accuracy_mean;
        var += d * d;
    }
    stats->accuracy_stddev = sqrt(var / 10.0);
    stats->rules_per_set = stats->total_rules / 10.0;
    stats->tests_per_rule = stats->total_rules ? (double)stats->total_attribute_tests / (double)stats->total_rules : 0.0;
    stats->iterations_per_rule = stats->total_rules ? (double)stats->total_iterations / (double)stats->total_rules : 0.0;
    stats->result_ram_bytes = stats->total_rules * (sizeof(Rule) + ds.n_attrs * sizeof(double));
    stats->result_disk_est_bytes = stats->total_rules * (32 + stats->tests_per_rule * 12);
    stats->peak_ram_mb = dm_bench_get_report().peak_memory_kb / 1024.0;
    dataset_free(&ds);
    return 0;
}
