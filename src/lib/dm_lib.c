/**
 * src/lib/dm_lib.c — Public API facade for libdm
 *
 * Implements every function declared in include/dm.h that is not already
 * provided by an internal translation unit under the same symbol name.
 *
 * Rules applied here:
 *  • If internal_name == public_name AND signatures are ABI-compatible →
 *    NO wrapper.  The internal .o is linked directly; the symbol is exported
 *    because we build the .so without -fvisibility=hidden.
 *  • If internal_name != public_name → thin wrapper declared below.
 *  • Opaque-handle objects (Algorithm, Tokenizer, Vision, LM, DataGen) are
 *    implemented here from scratch (they have no 1-to-1 internal equivalent).
 *
 * Internal headers are intentionally NOT included here to avoid typedef
 * conflicts (e.g. `typedef struct {...} DM_Algorithm` vs `typedef void*
 * DM_Algorithm`).  Instead we use `extern` declarations with compatible
 * plain-C types.
 *
 * Build:  gcc -shared -fPIC -DDM_BUILDING_LIB ... -o libdm.so
 * SPDX-License-Identifier: MIT
 */

#define DM_BUILDING_LIB
#include "dm.h"
#include "models/lm/bert.h"
#include "models/vae.h"
#include "models/gan.h"

typedef struct DM_Spec DM_Spec;
typedef struct DM_Ledger DM_Ledger;

DM_Spec* dm_spec_new(void);
void dm_spec_free(DM_Spec *spec);
DM_Ledger* dm_ledger_new(void);
void dm_ledger_free(DM_Ledger *ledger);
int dm_spec_parse(const char *path, DM_Spec *spec, DM_Ledger *ledger);
int dm_medm_repair_and_feedback(DM_Spec *spec, DM_Ledger *ledger, const char *output_base_path, const char *real_dataset_path, unsigned int seed);

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#if defined(_WIN32)
#  include <windows.h>
#  include <io.h>
#  define DUP(fd)        _dup(fd)
#  define DUP2(f, t)     _dup2(f, t)
#  define FILENO(f)      _fileno(f)
#  define CLOSE(fd)      _close(fd)
#else
#  include <unistd.h>
#  include <fcntl.h>
#  include <dlfcn.h>
#  define DUP(fd)        dup(fd)
#  define DUP2(f, t)     dup2(f, t)
#  define FILENO(f)      fileno(f)
#  define CLOSE(fd)      close(fd)
#endif

/* =========================================================================
 * Forward declarations — internal symbols used below
 * =========================================================================
 *
 * We declare these with `extern` using plain C types to avoid conflicting
 * with the dm.h typedefs.  At link time these resolve to the definitions in
 * their respective translation units.
 * ========================================================================= */

/* §2 Dataset */
typedef struct {
    int       type;      /* DM_DatasetType (int) */
    size_t    count;
    uint32_t  max_id;
    void     *payload;
    void    (*free_payload)(void *, size_t);
} _DmDataset;

extern _DmDataset *dm_dataset_load(const char *path, int type);
/* dm_dataset_free(_DmDataset*) is exported from dm_dataset.c with the public name */

/* §3 Algorithm */
typedef struct {
    const char *id;
    const char *name;
    const char *description;
    uint32_t    supported_types;
    int       (*run)(void *ds, void *params);
} _DmAlgo;

extern _DmAlgo *dm_get_algorithm(const char *id);

/* §4 Tokenizer CLI entry points */
extern int dm_bpe_cli           (int argc, char **argv);
extern int dm_bpe_dropout_cli   (int argc, char **argv);
extern int dm_unigram_cli       (int argc, char **argv);
extern int dm_sentencepiece_cli (int argc, char **argv);
extern int dm_gpe_cli           (int argc, char **argv);
extern int dm_parity_bpe_cli    (int argc, char **argv);
extern int dm_maximal_munch_cli (int argc, char **argv);
extern int dm_fast_wordpiece_cli(int argc, char **argv);
extern int dm_tokenizer_lab_cli (int argc, char **argv);
extern int dm_volt_cli          (int argc, char **argv);

/* §5-6 Model CLI entry points */
extern int dm_mobilenet_tiny_cli      (int argc, char **argv);
extern int dm_tinyvit_cli             (int argc, char **argv);
extern int dm_transformer_cli         (int argc, char **argv);
extern int dm_tiny_transformer_cli    (int argc, char **argv);
extern int dm_tinystories_cli         (int argc, char **argv);
extern int dm_bert_cli                (int argc, char **argv);
extern int dm_textbook_generator_cli  (int argc, char **argv);

/* §7 Image utilities (internal names) */
extern int dm_image_load_ppm_rgb_f32  (const char *path, DM_Tensor *out);
extern int dm_image_resize_nearest    (const DM_Tensor *in, DM_Tensor *out, int h, int w);
extern int dm_image_patchify          (const DM_Tensor *in, DM_Tensor *out, int ph, int pw);

/* §8 Tensor neural-network ops (internal names differ from dm_op_*) */
extern int  dm_conv2d_same          (const DM_Tensor *in, DM_Tensor *out,
                                     const float *w, const float *b,
                                     int out_c, int kernel, int stride);
extern int  dm_depthwise_conv2d_same(const DM_Tensor *in, DM_Tensor *out,
                                     const float *w, const float *b,
                                     int kernel, int stride);
extern int  dm_pointwise_conv2d     (const DM_Tensor *in, DM_Tensor *out,
                                     const float *w, const float *b, int out_c);
extern int  dm_linear               (const DM_Tensor *in, DM_Tensor *out,
                                     const float *w, const float *b, int out_c);
extern int  dm_global_avg_pool      (const DM_Tensor *in, DM_Tensor *out);
extern void dm_relu6                (DM_Tensor *t);
extern void dm_softmax              (DM_Tensor *t);

/* §9 Benchmark renamed functions */
extern void dm_bench_record_results (size_t num_itemsets, size_t total_items);
extern void dm_bench_print_report   (const char *algo_name, const char *dataset_name);

/* §10 BitSet renamed popcount */
extern size_t dm_bitset_count(void *bs);

/* §11 DataGen — medm is part of the algorithm framework; textbook has its own CLI */
/* medm synthetic: dm_datagen_run("medm", ...) uses the medm algorithm */
/* textbook: delegated to dm_textbook_generator_cli */

/* §12 Plugin */
typedef struct { const char *key; const char *value; } _DmPluginArg;
typedef struct {
    void           *flat;          /* DM_FlatDataset* (opaque) */
    void           *arena;         /* DM_Arena* */
    const char     *source_path;
    int             connector;     /* DM_ConnectorKind */
    const _DmPluginArg *args;
    size_t          arg_count;
} _DmPluginInput;
typedef struct { size_t num_patterns; double runtime_ms; } _DmPluginResult;
typedef struct _DmPlugin {
    const char *id; const char *name; const char *description;
    int (*run)(const _DmPluginInput *in, _DmPluginResult *out);
} _DmPlugin;

extern _DmPlugin *dm_plugin_get(const char *id);

/* §14 GPU renamed destroy */
extern void dm_gpu_destroy(void *ctx);

/* §17 Flat Dataset renamed functions */
extern DM_ConnectorOptions dm_connector_default_options(DM_ConnectorKind kind);
extern int dm_flat_load_mmap(const char *path,
                              const DM_ConnectorOptions *opts,
                              DM_Arena *arena,
                              DM_FlatDataset *out,
                              DM_ConnectorStats *stats);

/* §18 Experiment renamed functions */
extern double dm_timer_now_seconds(void);
extern double get_peak_ram_mb(void);
extern double dm_directory_size_mb(const char *path);
extern void   dm_csv_write_raw_header(const char *path);

/* Internal ExperimentRow (has extra MFHOIPatternQuality field) */
typedef struct {
    double avg_itemset_length;
    int    max_itemset_length;
    double avg_support, avg_relative_support;
    double avg_occupancy, median_occupancy, max_occupancy;
} _MFHOIQuality;

typedef struct {
    char   dataset[256];
    char   algorithm[64];
    double minsup_ratio;
    int    minsup_count;
    double minocc;
    int    run_id;
    char   status[32];
    double runtime_seconds;
    double peak_ram_mb;
    double temp_disk_mb;
    double output_disk_mb;
    int    num_generated_candidates;
    int    num_output_itemsets;
    int    num_fhoi;
    int    num_weak_mfhoi;
    int    num_strong_mfhoi;
    int    dominance_removed_count;
    _MFHOIQuality quality;           /* extra field not in public struct */
    double compression_vs_fi;
    double compression_vs_fhoi;
    double reduction_vs_fhoi_percent;
    double strong_extra_reduction_percent;
} _ExperimentRow;

extern void dm_csv_append_raw_row(const char *path, const _ExperimentRow *row);

/* =========================================================================
 * Globals defined in main.c that tokenizer_variants.c references.
 * Provide weak defaults so libdm.so doesn't have undefined symbols.
 * ========================================================================= */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((weak)) int output_json = 0;
#else
int output_json = 0;   /* fallback for non-GCC */
#endif

/* =========================================================================
 * § 1  Core / Version
 * ========================================================================= */

#define DM_VERSION_STR   "1.0.0"
#define DM_VERSION_MAJOR 1
#define DM_VERSION_MINOR 0
#define DM_VERSION_PATCH 0

DM_API const char *dm_version(void) { return DM_VERSION_STR; }

DM_API uint32_t dm_version_number(void) {
    return ((uint32_t)DM_VERSION_MAJOR << 16) |
           ((uint32_t)DM_VERSION_MINOR <<  8) |
            (uint32_t)DM_VERSION_PATCH;
}

DM_API DM_Status dm_init(void) {
    /* Algorithms self-register via __attribute__((constructor)).
       Reserve for future global setup (e.g. GPU warm-up). */
    return DM_OK;
}

DM_API const char *dm_strerror(DM_Status code) {
    switch (code) {
        case DM_OK:                return "OK";
        case DM_ERR_GENERIC:       return "Generic error";
        case DM_ERR_IO:            return "I/O error";
        case DM_ERR_MEMORY:        return "Out of memory";
        case DM_ERR_INVALID_PARAM: return "Invalid parameter";
        case DM_ERR_NOT_FOUND:     return "Not found";
        case DM_ERR_INCOMPATIBLE:  return "Incompatible";
        case DM_ERR_NOT_SUPPORTED: return "Not supported";
        default:                   return "Unknown error";
    }
}

/* =========================================================================
 * § 2  Dataset
 *
 * dm_dataset_free() is already provided by dm_dataset.c under the same
 * public symbol name — no wrapper needed.
 * ========================================================================= */

static int parse_dataset_type(const char *type) {
    /* DM_DatasetType values: TRANSACTIONAL=0, UTILITY=1, MATRIX=2,
       SEQUENCE_UTILITY=3, QUANTITY=4 */
    if (!type)                             return 0; /* TRANSACTIONAL */
    if (strstr(type, "util"))              return 1; /* UTILITY */
    if (strstr(type, "seq"))               return 3; /* SEQUENCE_UTILITY */
    if (strstr(type, "quant"))             return 4; /* QUANTITY */
    if (strstr(type, "mat"))               return 2; /* MATRIX */
    return 0;
}

DM_API DM_Dataset dm_dataset_open(const char *path, const char *type) {
    return (DM_Dataset)dm_dataset_load(path, parse_dataset_type(type));
}

DM_API size_t dm_dataset_count(DM_Dataset ds) {
    if (!ds) return 0;
    return ((_DmDataset *)ds)->count;
}

DM_API uint32_t dm_dataset_max_id(DM_Dataset ds) {
    if (!ds) return 0;
    return ((_DmDataset *)ds)->max_id;
}

/* dm_dataset_free — exported from dm_dataset.c */

/* =========================================================================
 * § 3  Algorithm
 * ========================================================================= */

typedef struct { char id[128]; } _AlgoHandle;

DM_API DM_Algorithm dm_algorithm_create(const char *id) {
    if (!id) return NULL;
    if (!dm_get_algorithm(id) && !dm_plugin_get(id)) return NULL;
    _AlgoHandle *h = (_AlgoHandle *)malloc(sizeof(*h));
    if (!h) return NULL;
    strncpy(h->id, id, sizeof(h->id) - 1);
    h->id[sizeof(h->id) - 1] = '\0';
    return (DM_Algorithm)h;
}

/* Generic first-field params (all algo param structs start with a double) */
typedef struct { double threshold; } _GenParams;

static DM_Status run_algo_redirect(_DmAlgo *algo, _DmDataset *ds,
                                    _GenParams *params, const char *output_path)
{
    if (!output_path || !output_path[0]) {
        int rc = algo->run(ds, params);
        return rc == 0 ? DM_OK : DM_ERR_GENERIC;
    }
    int saved = DUP(FILENO(stdout));
    if (saved < 0) return DM_ERR_IO;
    FILE *out = fopen(output_path, "w");
    if (!out) { CLOSE(saved); return DM_ERR_IO; }
    if (DUP2(FILENO(out), FILENO(stdout)) < 0) {
        fclose(out); CLOSE(saved); return DM_ERR_IO;
    }
    int rc = algo->run(ds, params);
    fflush(stdout);
    DUP2(saved, FILENO(stdout));
    CLOSE(saved);
    fclose(out);
    return rc == 0 ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_algorithm_run(DM_Algorithm  algo_handle,
                                   const char   *dataset_path,
                                   const char   *output_path,
                                   double        min_support,
                                   const char  **extra_args)
{
    if (!algo_handle || !dataset_path) return DM_ERR_INVALID_PARAM;
    const char *id = ((_AlgoHandle *)algo_handle)->id;

    /* Plugin path */
    _DmPlugin *plugin = dm_plugin_get(id);
    if (plugin) {
        /* Build a minimal plugin invocation using arena + flat dataset */
        DM_Arena   arena;
        DM_FlatDataset flat;
        DM_ConnectorStats stats;
        DM_ConnectorOptions opts = dm_connector_default_options(DM_CONNECTOR_SPMF);

        if (dm_arena_init(&arena, 256 * 1024 * 1024) != 0) return DM_ERR_MEMORY;
        if (dm_flat_load_mmap(dataset_path, &opts, &arena, &flat, &stats) != 0) {
            dm_arena_free(&arena); return DM_ERR_IO;
        }

        char minsup_buf[64];
        snprintf(minsup_buf, sizeof(minsup_buf), "%.17g", min_support);

        size_t n_extra = 0;
        if (extra_args) while (extra_args[n_extra]) n_extra++;

        size_t total_args = 2 + n_extra;
        _DmPluginArg *pa = (_DmPluginArg *)calloc(total_args, sizeof(*pa));
        if (!pa) { dm_arena_free(&arena); return DM_ERR_MEMORY; }
        pa[0].key = "minsup";  pa[0].value = minsup_buf;
        pa[1].key = "minutil"; pa[1].value = minsup_buf;
        for (size_t i = 0; i < n_extra && extra_args[i]; i++) {
            if (2 + i < total_args) {
                pa[2 + i].key   = extra_args[i];
                pa[2 + i].value = (i + 1 < n_extra) ? extra_args[i + 1] : "";
            }
        }

        _DmPluginInput in;
        memset(&in, 0, sizeof(in));
        in.flat        = &flat;
        in.arena       = &arena;
        in.source_path = dataset_path;
        in.connector   = (int)DM_CONNECTOR_SPMF;
        in.args        = pa;
        in.arg_count   = total_args;

        _DmPluginResult res = {0};

        int saved = -1; FILE *outf = NULL;
        if (output_path && output_path[0]) {
            saved = DUP(FILENO(stdout));
            outf  = fopen(output_path, "w");
            if (saved >= 0 && outf) DUP2(FILENO(outf), FILENO(stdout));
        }
        int prc = plugin->run(&in, &res);
        if (output_path && output_path[0]) {
            fflush(stdout);
            if (saved >= 0) { DUP2(saved, FILENO(stdout)); CLOSE(saved); }
            if (outf) fclose(outf);
        }
        free(pa);
        dm_arena_free(&arena);
        return prc == 0 ? DM_OK : DM_ERR_GENERIC;
    }

    /* Registry path */
    _DmAlgo *algo = dm_get_algorithm(id);
    if (!algo) return DM_ERR_NOT_FOUND;

    /* Determine dataset type from algo's supported_types bitmask */
    int dtype = 0; /* TRANSACTIONAL */
    if (algo->supported_types & (1 << 1)) dtype = 1; /* UTILITY */
    else if (algo->supported_types & (1 << 4)) dtype = 4; /* QUANTITY */
    else if (algo->supported_types & (1 << 3)) dtype = 3; /* SEQ_UTILITY */

    _DmDataset *ds = dm_dataset_load(dataset_path, dtype);
    if (!ds) return DM_ERR_IO;

    _GenParams params = { min_support };
    DM_Status  s      = run_algo_redirect(algo, ds, &params, output_path);
    dm_dataset_free((DM_Dataset)ds);
    return s;
}

DM_API DM_Status dm_algorithm_list(char *buf, int buf_size) {
    if (!buf || buf_size <= 0) return DM_ERR_INVALID_PARAM;
    static const char *KNOWN[] = {
        "apriori","apriori_tid","apriori_hybrid","apriori_inverse","apriori_rare",
        "eclat","fpgrowth","fpclose","fpmax","lcm","lcmver2","charm","close",
        "closet","closetplus","dci_closed","aclose","carpenter","mafia","max_miner",
        "genmax","pascal","prepost","prepostplus","cfi_stream","clostream","estdec",
        "defme","cori","rp_tree","tree_projection","negfin","fin","finplus",
        "regular_mine","cls_miner","slim","zart","sam","dic","ais","krimp",
        "opus_miner","dbv_miner","dfi_growth",
        "efim","efim_closed","fhm","fhim","fhn","fhmds","huiminer","hui_miner",
        "hui_list_ins","huci_miner","hup_miner","hupe_garm","ihup","haui_miner",
        "hauim_gmu","ehaupm","minfhm","fchm","foshu","thui","dphim","mlhui_miner",
        "ghui_miner","fcfia","ffiminer","memu","sum","tku_ce","tku_ce_plus","eihi",
        "thue","tshoun","mheinu","nafcp","feacp","closed_fhuim_kinana",
        "huim_abc","huim_aco","huim_bpso","huim_bpso_tree","huim_hc","huim_sa",
        "huim_afsa","huimsu","bio_huif_ba","bio_huif_ga","bio_huif_pso",
        "fhuqi_miner","vhuqi","uapriori","ubmffp","ufh","ulb_miner",
        "hupspm","uspan","prefixspan","spade","up_growth","up_hist",
        "hoimto","skyline_miner","skymine","ltm","sfui_uf","sfu_ce","medm_gen",
        "msapriori","tkq","tkhoim","tmku","topkphm",
        "fhoi","fhoi_miner","dfhoi","aura_hoi","hep","hiep","nam_hep","cloe_hoi",
        "mfhoi","mhoui","chuo_miner","sparc_hoi","strong_mfhoi","weak_mfhoi",
        "clhminer","talky_g","tipn_houi",
        "tku_pso","rminer",
        "chuimine_closed","chuimine_maximal","chui_miner","pso_classifier",
        NULL
    };
    int pos = 0;
    for (int i = 0; KNOWN[i]; i++) {
        int len = (int)strlen(KNOWN[i]);
        if (pos + len + 2 >= buf_size) break;
        memcpy(buf + pos, KNOWN[i], len);
        pos += len;
        buf[pos++] = '\n';
    }
    buf[pos] = '\0';
    return DM_OK;
}

DM_API void dm_algorithm_free(DM_Algorithm algo) { free(algo); }

/* =========================================================================
 * § 4  Tokenizer
 *
 * NLP tokenizers expose only a CLI function.  We implement the DM_Tokenizer
 * API using temp files for encode/decode.
 * ========================================================================= */

typedef struct {
    char type[32];
    char model_dir[512];
    int  trained;
    char **vocab;
    int  vocab_size;
} _TokHandle;

static int is_space_char(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static void free_vocab(_TokHandle *h) {
    if (h->vocab) {
        for (int i = 0; i < h->vocab_size; i++) {
            free(h->vocab[i]);
        }
        free(h->vocab);
        h->vocab = NULL;
    }
    h->vocab_size = 0;
}

static char *dm_xstrndup(const char *s, size_t n) {
    char *p = malloc(n + 1);
    if (p) {
        memcpy(p, s, n);
        p[n] = '\0';
    }
    return p;
}

static void load_vocab_for_handle(_TokHandle *h) {
    free_vocab(h);

    if (strcmp(h->type, "sentencepiece") == 0) {
        FILE *f = fopen(h->model_dir, "r");
        if (!f) return;
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        rewind(f);
        char *buf = malloc(size + 1);
        if (!buf) { fclose(f); return; }
        size_t read_bytes = fread(buf, 1, size, f);
        buf[read_bytes] = '\0';
        fclose(f);

        char *p = strstr(buf, "\"vocab\"");
        if (p) {
            p = strchr(p, '[');
            if (p) {
                p++;
                int cap = 1024;
                h->vocab = malloc(cap * sizeof(char *));
                h->vocab_size = 0;
                while (p && *p) {
                    char *end = strchr(p, ']');
                    char *q = strchr(p, '"');
                    if (!q || (end && end < q)) break;
                    q++;
                    char *e = strchr(q, '"');
                    if (!e) break;
                    
                    int len = e - q;
                    char *s = malloc(len + 1);
                    if (s) {
                        memcpy(s, q, len);
                        s[len] = '\0';
                    }

                    if (h->vocab_size >= cap) {
                        cap *= 2;
                        h->vocab = realloc(h->vocab, cap * sizeof(char *));
                    }
                    if (s) {
                        h->vocab[h->vocab_size++] = s;
                    }
                    p = e + 1;
                }
            }
        }
        free(buf);
    } else if (strcmp(h->type, "bpe") == 0) {
        int cap = 256 + 1024;
        h->vocab = malloc(cap * sizeof(char *));
        for (int i = 0; i < 256; i++) {
            h->vocab[i] = malloc(2);
            h->vocab[i][0] = (char)i;
            h->vocab[i][1] = '\0';
        }
        h->vocab_size = 256;

        FILE *f = fopen(h->model_dir, "r");
        if (f) {
            char line[1024];
            while (fgets(line, sizeof(line), f)) {
                size_t ln = strlen(line);
                while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0';
                char *s = line;
                while (*s && is_space_char(*s)) s++;
                if (!*s || *s == '#') continue;
                
                char *left = strtok(s, " \t");
                char *right = strtok(NULL, " \t");
                if (left && right) {
                    int len1 = strlen(left);
                    int len2 = strlen(right);
                    char *new_tok = malloc(len1 + len2 + 1);
                    if (new_tok) {
                        strcpy(new_tok, left);
                        strcat(new_tok, right);
                    }

                    if (h->vocab_size >= cap) {
                        cap *= 2;
                        h->vocab = realloc(h->vocab, cap * sizeof(char *));
                    }
                    if (new_tok) {
                        h->vocab[h->vocab_size++] = new_tok;
                    }
                }
            }
            fclose(f);
        }
    } else if (strcmp(h->type, "unigram") == 0) {
        FILE *f = fopen(h->model_dir, "r");
        if (f) {
            char line[4096];
            int cap = 1024;
            h->vocab = malloc(cap * sizeof(char *));
            h->vocab_size = 0;
            while (fgets(line, sizeof(line), f)) {
                size_t ln = strlen(line);
                while (ln && (line[ln - 1] == '\n' || line[ln - 1] == '\r')) line[--ln] = '\0';
                if (!line[0] || line[0] == '#') continue;
                
                char *tab = strrchr(line, '\t');
                if (tab) {
                    *tab = '\0';
                    char *new_piece = strdup(line);
                    if (h->vocab_size >= cap) {
                        cap *= 2;
                        h->vocab = realloc(h->vocab, cap * sizeof(char *));
                    }
                    if (new_piece) {
                        h->vocab[h->vocab_size++] = new_piece;
                    }
                }
            }
            fclose(f);
        }
    } else if (strcmp(h->type, "gpe") == 0 || strcmp(h->type, "parity_bpe") == 0) {
        int cap = 256 + 1024;
        h->vocab = malloc(cap * sizeof(char *));
        for (int i = 0; i < 256; i++) {
            h->vocab[i] = malloc(2);
            h->vocab[i][0] = (char)i;
            h->vocab[i][1] = '\0';
        }
        h->vocab_size = 256;

        FILE *f = fopen(h->model_dir, "r");
        if (f) {
            fseek(f, 0, SEEK_END);
            long size = ftell(f);
            rewind(f);
            char *buf = malloc(size + 1);
            if (buf) {
                size_t read_bytes = fread(buf, 1, size, f);
                buf[read_bytes] = '\0';
                
                char *p = strstr(buf, "\"merges\"");
                if (p) {
                    p = strchr(p, '[');
                    if (p) {
                        p++;
                        while (p && *p) {
                            char *end = strstr(p, "],\n");
                            char *pair = strchr(p, '[');
                            if (!pair || (end && end < pair)) break;
                            
                            char *q = strchr(pair, '"');
                            if (!q) break;
                            q++;
                            char *e = strchr(q, '"');
                            if (!e) break;
                            int len1 = e - q;
                            char *left = malloc(len1 + 1);
                            if (left) {
                                memcpy(left, q, len1);
                                left[len1] = '\0';
                            }
                            
                            q = strchr(e + 1, '"');
                            if (!q) { free(left); break; }
                            q++;
                            e = strchr(q, '"');
                            if (!e) { free(left); break; }
                            int len2 = e - q;
                            char *right = malloc(len2 + 1);
                            if (right) {
                                memcpy(right, q, len2);
                                right[len2] = '\0';
                            }
                            
                            char *new_tok = NULL;
                            if (left && right) {
                                new_tok = malloc(len1 + len2 + 1);
                                if (new_tok) {
                                    strcpy(new_tok, left);
                                    strcat(new_tok, right);
                                }
                            }
                            free(left);
                            free(right);
                            
                            if (h->vocab_size >= cap) {
                                cap *= 2;
                                h->vocab = realloc(h->vocab, cap * sizeof(char *));
                            }
                            if (new_tok) {
                                h->vocab[h->vocab_size++] = new_tok;
                            }
                            p = e + 1;
                        }
                    }
                }
                free(buf);
            }
            fclose(f);
        }
    }
}

static int find_vocab_id(_TokHandle *h, const char *piece) {
    if (strcmp(h->type, "bpe") == 0 || strcmp(h->type, "unigram") == 0) {
        int len = strlen(piece);
        if (len >= 2 && strcmp(piece + len - 2, "@@") == 0) {
            char clean[256];
            strncpy(clean, piece, len - 2);
            clean[len - 2] = '\0';
            for (int i = 0; i < h->vocab_size; i++) {
                if (strcmp(h->vocab[i], clean) == 0) return i;
            }
        } else {
            char with_eow[256];
            snprintf(with_eow, sizeof(with_eow), "%s</w>", piece);
            for (int i = 0; i < h->vocab_size; i++) {
                if (strcmp(h->vocab[i], with_eow) == 0) return i;
            }
            for (int i = 0; i < h->vocab_size; i++) {
                if (strcmp(h->vocab[i], piece) == 0) return i;
            }
        }
    } else {
        for (int i = 0; i < h->vocab_size; i++) {
            if (strcmp(h->vocab[i], piece) == 0) return i;
        }
    }
    return -1;
}

DM_API DM_Tokenizer dm_tokenizer_create(const char *type) {
    if (!type) return NULL;
    static const char *TYPES[] = {
        "bpe","bpe_dropout","unigram","sentencepiece","wordpiece",
        "gpe","parity_bpe","volt","maximal_munch","faro","tokenizer_lab", NULL
    };
    int ok = 0;
    for (int i = 0; TYPES[i]; i++) if (strcmp(type, TYPES[i]) == 0) { ok = 1; break; }
    if (!ok) return NULL;
    _TokHandle *h = (_TokHandle *)calloc(1, sizeof(*h));
    if (!h) return NULL;
    strncpy(h->type, type, sizeof(h->type) - 1);
    return (DM_Tokenizer)h;
}

static int call_tokenizer_cli(const char *type, int argc, char **argv) {
    if (strcmp(type, "bpe")           == 0) return dm_bpe_cli(argc, argv);
    if (strcmp(type, "bpe_dropout")   == 0) return dm_bpe_dropout_cli(argc, argv);
    if (strcmp(type, "unigram")       == 0) return dm_unigram_cli(argc, argv);
    if (strcmp(type, "sentencepiece") == 0) return dm_sentencepiece_cli(argc, argv);
    if (strcmp(type, "gpe")           == 0) return dm_gpe_cli(argc, argv);
    if (strcmp(type, "parity_bpe")    == 0) return dm_parity_bpe_cli(argc, argv);
    if (strcmp(type, "maximal_munch") == 0) return dm_maximal_munch_cli(argc, argv);
    if (strcmp(type, "tokenizer_lab") == 0) return dm_tokenizer_lab_cli(argc, argv);
    if (strcmp(type, "volt")          == 0) return dm_volt_cli(argc, argv);
    if (strcmp(type, "wordpiece") == 0 ||
        strcmp(type, "faro")      == 0)     return dm_fast_wordpiece_cli(argc, argv);
    return -1;
}

DM_API DM_Status dm_tokenizer_train(DM_Tokenizer tok,
                                     const char  *corpus_path,
                                     int          vocab_size,
                                     const char  *output_path)
{
    if (!tok || !corpus_path || !output_path) return DM_ERR_INVALID_PARAM;
    _TokHandle *h = (_TokHandle *)tok;
    char vs[32];
    snprintf(vs, sizeof(vs), "%d", vocab_size);

    int rc = 0;
    if (strcmp(h->type, "bpe") == 0) {
        char *argv[] = {
            "bpe", "learn-bpe",
            "-i", (char *)corpus_path,
            "-m", vs,
            "-o", (char *)output_path,
            NULL
        };
        rc = call_tokenizer_cli(h->type, 8, argv);
    } else if (strcmp(h->type, "sentencepiece") == 0) {
        char *argv[] = {
            "sentencepiece", "train",
            "--input", (char *)corpus_path,
            "--vocab-size", vs,
            "-o", (char *)output_path,
            NULL
        };
        rc = call_tokenizer_cli(h->type, 8, argv);
    } else if (strcmp(h->type, "parity_bpe") == 0) {
        char *argv[] = {
            "parity_bpe", "train",
            "--input-labeled", (char *)corpus_path,
            "--merges", vs,
            "-o", (char *)output_path,
            NULL
        };
        rc = call_tokenizer_cli(h->type, 8, argv);
    } else if (strcmp(h->type, "unigram") == 0 || strcmp(h->type, "gpe") == 0) {
        char *argv[] = {
            (char *)h->type, "train",
            "-i", (char *)corpus_path,
            "-o", (char *)output_path,
            "--vocab-size", vs,
            NULL
        };
        rc = call_tokenizer_cli(h->type, 8, argv);
    } else {
        return DM_ERR_GENERIC;
    }

    if (rc == 0) {
        strncpy(h->model_dir, output_path, sizeof(h->model_dir) - 1);
        h->trained = 1;
        load_vocab_for_handle(h);
    }
    return rc == 0 ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_tokenizer_load(DM_Tokenizer tok, const char *model_path) {
    if (!tok || !model_path) return DM_ERR_INVALID_PARAM;
    _TokHandle *h = (_TokHandle *)tok;
    strncpy(h->model_dir, model_path, sizeof(h->model_dir) - 1);
    h->trained = 1;
    load_vocab_for_handle(h);
    return DM_OK;
}

DM_API DM_Status dm_tokenizer_encode(DM_Tokenizer  tok,
                                      const char   *text,
                                      uint32_t     *out_ids,
                                      int          *in_out_len)
{
    if (!tok || !text || !out_ids || !in_out_len) return DM_ERR_INVALID_PARAM;
    _TokHandle *h = (_TokHandle *)tok;
    if (!h->trained) return DM_ERR_INVALID_PARAM;

    char tmp_in[256], tmp_out[256];
    snprintf(tmp_in,  sizeof(tmp_in),  "/tmp/dm_tok_in_%d.txt",  (int)getpid());
    snprintf(tmp_out, sizeof(tmp_out), "/tmp/dm_tok_out_%d.txt", (int)getpid());

    FILE *f = fopen(tmp_in, "w");
    if (!f) return DM_ERR_IO;
    fputs(text, f); fputc('\n', f); fclose(f);

    int rc = 0;
    if (strcmp(h->type, "sentencepiece") == 0) {
        char *argv[] = {
            "sentencepiece", "encode",
            "--model", h->model_dir,
            "--input", tmp_in,
            "--output", tmp_out,
            "--output-format", "id",
            NULL
        };
        rc = call_tokenizer_cli(h->type, 10, argv);
    } else if (strcmp(h->type, "bpe") == 0) {
        char *argv[] = {
            "bpe", "apply-bpe",
            "-c", h->model_dir,
            "-i", tmp_in,
            "-o", tmp_out,
            NULL
        };
        rc = call_tokenizer_cli(h->type, 8, argv);
    } else if (strcmp(h->type, "parity_bpe") == 0) {
        char *argv[] = {
            "parity_bpe", "encode",
            "-m", h->model_dir,
            "-i", tmp_in,
            "-o", tmp_out,
            NULL
        };
        rc = call_tokenizer_cli(h->type, 8, argv);
    } else if (strcmp(h->type, "unigram") == 0 || strcmp(h->type, "gpe") == 0) {
        char *argv[] = {
            (char *)h->type, "encode",
            "-m", h->model_dir,
            "-i", tmp_in,
            "-o", tmp_out,
            NULL
        };
        rc = call_tokenizer_cli(h->type, 8, argv);
    } else {
        remove(tmp_in);
        return DM_ERR_GENERIC;
    }

    remove(tmp_in);
    if (rc != 0) { remove(tmp_out); return DM_ERR_GENERIC; }

    f = fopen(tmp_out, "r");
    if (!f) { remove(tmp_out); return DM_ERR_IO; }

    int capacity = *in_out_len, count = 0;

    if (strcmp(h->type, "sentencepiece") == 0) {
        unsigned int val;
        while (fscanf(f, "%u", &val) == 1) {
            if (count < capacity) out_ids[count] = (uint32_t)val;
            count++;
        }
    } else {
        if (!h->vocab) load_vocab_for_handle(h);
        if (!h->vocab) { fclose(f); remove(tmp_out); return DM_ERR_GENERIC; }

        char piece[512];
        while (fscanf(f, "%511s", piece) == 1) {
            int id = find_vocab_id(h, piece);
            if (id != -1) {
                if (count < capacity) out_ids[count] = (uint32_t)id;
                count++;
            }
        }
    }

    fclose(f); remove(tmp_out);
    if (count > capacity) { *in_out_len = count; return DM_ERR_MEMORY; }
    *in_out_len = count;
    return DM_OK;
}

DM_API DM_Status dm_tokenizer_decode(DM_Tokenizer   tok,
                                      const uint32_t *ids,
                                      int             n_ids,
                                      char           *out_buf,
                                      int             buf_size)
{
    if (!tok || !ids || !out_buf || n_ids <= 0 || buf_size <= 0) return DM_ERR_INVALID_PARAM;
    _TokHandle *h = (_TokHandle *)tok;
    if (!h->trained) return DM_ERR_INVALID_PARAM;

    if (strcmp(h->type, "sentencepiece") == 0) {
        char tmp_in[256], tmp_out[256];
        snprintf(tmp_in,  sizeof(tmp_in),  "/tmp/dm_dec_in_%d.txt",  (int)getpid());
        snprintf(tmp_out, sizeof(tmp_out), "/tmp/dm_dec_out_%d.txt", (int)getpid());

        FILE *f = fopen(tmp_in, "w");
        if (!f) return DM_ERR_IO;
        for (int i = 0; i < n_ids; i++) fprintf(f, "%u%s", ids[i], i+1<n_ids?" ":"");
        fputc('\n', f); fclose(f);

        char *argv[] = {
            "sentencepiece", "decode",
            "--model", h->model_dir,
            "--input", tmp_in,
            "--output", tmp_out,
            "--input-format", "id",
            NULL
        };
        int rc = call_tokenizer_cli(h->type, 10, argv);
        remove(tmp_in);
        if (rc != 0) { remove(tmp_out); return DM_ERR_GENERIC; }

        f = fopen(tmp_out, "r");
        if (!f) { remove(tmp_out); return DM_ERR_IO; }
        size_t n = fread(out_buf, 1, (size_t)(buf_size - 1), f);
        out_buf[n] = '\0';
        if (n > 0 && out_buf[n-1] == '\n') out_buf[--n] = '\0';
        fclose(f); remove(tmp_out);
        return DM_OK;
    } else {
        if (!h->vocab) load_vocab_for_handle(h);
        if (!h->vocab) return DM_ERR_GENERIC;

        out_buf[0] = '\0';
        int current_len = 0;

        for (int i = 0; i < n_ids; i++) {
            uint32_t id = ids[i];
            if (id >= (uint32_t)h->vocab_size) return DM_ERR_INVALID_PARAM;
            const char *piece = h->vocab[id];

            char clean[256];
            strncpy(clean, piece, sizeof(clean) - 1);
            clean[sizeof(clean) - 1] = '\0';
            int len = strlen(clean);

            if (strcmp(h->type, "bpe") == 0 || strcmp(h->type, "unigram") == 0) {
                if (len >= 4 && strcmp(clean + len - 4, "</w>") == 0) {
                    clean[len - 4] = '\0';
                    len -= 4;
                }

                int has_suffix = 0;
                if (len >= 2 && strcmp(clean + len - 2, "@@") == 0) {
                    clean[len - 2] = '\0';
                    len -= 2;
                    has_suffix = 1;
                }

                int needed = len + (has_suffix ? 0 : 1);
                if (current_len + needed >= buf_size) {
                    return DM_ERR_MEMORY;
                }

                strcat(out_buf, clean);
                current_len += len;
                if (!has_suffix && i + 1 < n_ids) {
                    strcat(out_buf, " ");
                    current_len++;
                }
            } else {
                if (current_len + len >= buf_size) {
                    return DM_ERR_MEMORY;
                }
                strcat(out_buf, clean);
                current_len += len;
            }
        }
        return DM_OK;
    }
}

DM_API int dm_tokenizer_vocab_size(DM_Tokenizer tok) {
    if (!tok) return 0;
    _TokHandle *h = (_TokHandle *)tok;
    if (!h->trained) return 0;
    if (!h->vocab) load_vocab_for_handle(h);
    return h->vocab_size;
}

DM_API const char *dm_tokenizer_token_text(DM_Tokenizer tok, uint32_t id,
                                            uint32_t *out_len)
{
    if (!tok) return NULL;
    _TokHandle *h = (_TokHandle *)tok;
    if (!h->trained) return NULL;
    if (!h->vocab) load_vocab_for_handle(h);
    if (!h->vocab || id >= (uint32_t)h->vocab_size) {
        if (out_len) *out_len = 0;
        return NULL;
    }
    if (out_len) *out_len = (uint32_t)strlen(h->vocab[id]);
    return h->vocab[id];
}

DM_API void dm_tokenizer_free(DM_Tokenizer tok) {
    if (tok) {
        _TokHandle *h = (_TokHandle *)tok;
        free_vocab(h);
        free(h);
    }
}

DM_API DM_Status dm_tokenizer_volt_run(const char *corpus_path,
                                        int min_size, int max_size,
                                        int n_steps, const char *output_path)
{
    if (!corpus_path || !output_path) return DM_ERR_INVALID_PARAM;
    char min_s[16], max_s[16], steps_s[16];
    snprintf(min_s,   sizeof(min_s),   "%d", min_size);
    snprintf(max_s,   sizeof(max_s),   "%d", max_size);
    snprintf(steps_s, sizeof(steps_s), "%d", n_steps);
    char *argv[] = {
        "volt", "run",
        "-i", (char *)corpus_path,
        "-o", (char *)output_path,
        "--min-size", min_s, "--max-size", max_s, "--steps", steps_s,
        NULL
    };
    return dm_volt_cli(12, argv) == 0 ? DM_OK : DM_ERR_GENERIC;
}

/* =========================================================================
 * § 5  Vision (MobileNetV4-Tiny)
 * ========================================================================= */

typedef struct {
    char model_type[32];
    int  classes;
    unsigned int seed;
} _VisionHandle;

DM_API DM_Vision dm_vision_create(const char *model_type) {
    if (!model_type) return NULL;
    _VisionHandle *h = (_VisionHandle *)calloc(1, sizeof(*h));
    if (!h) return NULL;
    strncpy(h->model_type, model_type, sizeof(h->model_type) - 1);
    h->classes = 1000; h->seed = 42;
    return (DM_Vision)h;
}

DM_API DM_Status dm_vision_init(DM_Vision v, const char *saved_model_dir,
                                  int classes, int image_size,
                                  float width_mult, float learning_rate)
{
    if (!v) return DM_ERR_INVALID_PARAM;
    (void)saved_model_dir; (void)image_size; (void)width_mult; (void)learning_rate;
    ((_VisionHandle *)v)->classes = classes;
    return DM_OK;
}

DM_API DM_Status dm_vision_train(DM_Vision v, const char *manifest_path,
                                   int epochs, int batch_size, float lr)
{
    (void)v; (void)manifest_path; (void)epochs; (void)batch_size; (void)lr;
    return DM_ERR_NOT_SUPPORTED;
}

DM_API DM_Status dm_vision_eval(DM_Vision v, const char *manifest_path,
                                  float *out_loss, float *out_accuracy)
{
    (void)v; (void)manifest_path;
    if (out_loss)     *out_loss     = 0.0f;
    if (out_accuracy) *out_accuracy = 0.0f;
    return DM_ERR_NOT_SUPPORTED;
}

/* Forward-declare the internal mobilenet forward pass */
extern int dm_mobilenet_tiny_forward(DM_Tensor *in, DM_Tensor *logits,
                                      int classes, unsigned int seed);

DM_API DM_Status dm_vision_predict(DM_Vision v, const float *rgb,
                                     int h, int w,
                                     float *out_probs, int n_classes)
{
    if (!v || !rgb || !out_probs) return DM_ERR_INVALID_PARAM;
    _VisionHandle *vh = (_VisionHandle *)v;
    int cls = n_classes > 0 ? n_classes : vh->classes;

    DM_Tensor in, logits;
    if (dm_tensor_alloc(&in, 1, 3, h, w) != 0) return DM_ERR_MEMORY;
    if (dm_tensor_alloc(&logits, 1, cls, 1, 1) != 0) {
        dm_tensor_free(&in); return DM_ERR_MEMORY;
    }
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 3; c++)
                dm_tensor_set(&in, 0, c, y, x, rgb[(y * w + x) * 3 + c]);

    int rc = dm_mobilenet_tiny_forward(&in, &logits, cls, vh->seed);
    if (rc == 0) {
        dm_op_softmax(&logits);
        for (int i = 0; i < cls; i++) out_probs[i] = logits.data[i];
    }
    dm_tensor_free(&in); dm_tensor_free(&logits);
    return rc == 0 ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_vision_save(DM_Vision v) {
    (void)v; return DM_ERR_NOT_SUPPORTED;
}

DM_API void dm_vision_free(DM_Vision v) { free(v); }

DM_API DM_Status dm_vision_forward_raw(const float *rgb_nhwc,
                                         int n, int h, int w,
                                         float *out_logits, int classes)
{
    if (!rgb_nhwc || !out_logits) return DM_ERR_INVALID_PARAM;
    for (int b = 0; b < n; b++) {
        const float *src = rgb_nhwc + b * h * w * 3;
        float *dst = out_logits + b * classes;
        DM_Tensor in, logits;
        if (dm_tensor_alloc(&in, 1, 3, h, w) != 0) return DM_ERR_MEMORY;
        if (dm_tensor_alloc(&logits, 1, classes, 1, 1) != 0) {
            dm_tensor_free(&in); return DM_ERR_MEMORY;
        }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int c = 0; c < 3; c++)
                    dm_tensor_set(&in, 0, c, y, x, src[(y * w + x) * 3 + c]);
        int rc = dm_mobilenet_tiny_forward(&in, &logits, classes, 42);
        if (rc == 0) for (int i = 0; i < classes; i++) dst[i] = logits.data[i];
        dm_tensor_free(&in); dm_tensor_free(&logits);
        if (rc != 0) return DM_ERR_GENERIC;
    }
    return DM_OK;
}

/* =========================================================================
 * § 5b  TinyViT direct API (pure-C, no TF for inference)
 * ========================================================================= */

/*
 * We forward-declare the internal functions from tinyvit.c using their
 * internal TinyViTConfig struct instead of importing the full header (which
 * would conflict with typedef DM_TinyViTVariant defined here in dm.h).
 * We call them via the stable wrappers below.
 */

/* Internal TinyViTConfig struct mirror — must stay in sync with tinyvit.h */
typedef struct {
    int variant;
    int embed_dims[4];
    int depths[4];
    int window_sizes[3];
    int mbconv_expand;
    int mlp_ratio;
    int head_dim;
    int num_classes;
    int img_size;
} _TVCfg;

extern void   dm_tinyvit_config_init       (void *cfg, int variant, int classes, int img_size);
extern size_t dm_tinyvit_cfg_count(const void *cfg);
extern int    dm_tinyvit_cfg_forward  (const void *cfg, const float *weights,
                                             const float *input, int batch, float *logits);
extern int    dm_tinyvit_cfg_save     (const char *path, const void *cfg, const float *w);
extern int    dm_tinyvit_cfg_load     (const char *path, void *cfg, float **w_out);
extern int    dm_tinyvit_save_sparse_labels(const char *path, int n, int classes, int K,
                                             const void *labels);
extern float  dm_tinyvit_cfg_loss(const float *logits, const void *label, float T);

/* public dm.h wrappers */

DM_API size_t dm_tinyvit_weight_count(DM_TinyViTVariant variant, int classes, int img_size)
{
    _TVCfg cfg;
    dm_tinyvit_config_init(&cfg, (int)variant, classes, img_size);
    return dm_tinyvit_cfg_count(&cfg);
}

DM_API DM_Status dm_tinyvit_forward(DM_TinyViTVariant variant,
                                     const float *weights,
                                     const float *input_nhwc,
                                     int batch, int classes, int img_size,
                                     float *logits_out)
{
    if (!weights || !input_nhwc || !logits_out) return DM_ERR_INVALID_PARAM;
    _TVCfg cfg;
    dm_tinyvit_config_init(&cfg, (int)variant, classes, img_size);
    int rc = dm_tinyvit_cfg_forward(&cfg, weights, input_nhwc, batch, logits_out);
    return rc == 0 ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_tinyvit_load(const char *weight_path,
                                  DM_TinyViTVariant *variant_out,
                                  int *classes_out, int *img_size_out,
                                  float **weights_out)
{
    if (!weight_path || !weights_out) return DM_ERR_INVALID_PARAM;
    _TVCfg cfg;
    int rc = dm_tinyvit_cfg_load(weight_path, &cfg, weights_out);
    if (rc != 0) return DM_ERR_GENERIC;
    if (variant_out)   *variant_out  = (DM_TinyViTVariant)cfg.variant;
    if (classes_out)   *classes_out  = cfg.num_classes;
    if (img_size_out)  *img_size_out = cfg.img_size;
    return DM_OK;
}

DM_API void dm_tinyvit_free_weights(float *weights)
{
    free(weights);
}


DM_API DM_Status dm_tinyvit_save_labels(const char *out_path,
                                         int num_images, int num_classes, int topK,
                                         const uint32_t *indices, const float *values,
                                         const uint32_t *aug_seeds)
{
    if (!out_path || !indices || !values || !aug_seeds) return DM_ERR_INVALID_PARAM;
    /* Build TinyViTSparseLabel array for internal function */
    typedef struct { uint32_t *indices; float *values; int K; int C; uint32_t aug_seed; } _SL;
    _SL *labels = (_SL*)calloc(num_images, sizeof(_SL));
    if (!labels) return DM_ERR_MEMORY;
    for (int i = 0; i < num_images; i++) {
        labels[i].K        = topK;
        labels[i].C        = num_classes;
        labels[i].aug_seed = aug_seeds[i];
        labels[i].indices  = (uint32_t*)(indices + (size_t)i * topK);
        labels[i].values   = (float*)(values   + (size_t)i * topK);
    }
    int rc = dm_tinyvit_save_sparse_labels(out_path, num_images, num_classes, topK, labels);
    free(labels);
    return rc == 0 ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_tinyvit_distill_loss(const float *student_logits,
                                          const uint32_t *indices,
                                          const float *teacher_values,
                                          int K, int C, float temperature,
                                          float *loss_out)
{
    if (!student_logits || !indices || !teacher_values || !loss_out) return DM_ERR_INVALID_PARAM;
    typedef struct { uint32_t *indices; float *values; int K; int C; uint32_t aug_seed; } _SL;
    _SL lbl;
    lbl.indices  = (uint32_t*)indices;
    lbl.values   = (float*)teacher_values;
    lbl.K        = K; lbl.C = C; lbl.aug_seed = 0;
    *loss_out = dm_tinyvit_cfg_loss(student_logits, &lbl, temperature);
    return DM_OK;
}

/* =========================================================================
 * § 6  Language Model
 * ========================================================================= */

typedef struct {
    char type[32];
    char checkpoint_dir[512];
    int  loaded;
} _LMHandle;

DM_API DM_LM dm_lm_create(const char *model_type) {
    if (!model_type) return NULL;
    if (strcmp(model_type, "tiny_transformer") != 0 &&
        strcmp(model_type, "tinystories")      != 0 &&
        strcmp(model_type, "bert")             != 0)
        return NULL;
    _LMHandle *h = (_LMHandle *)calloc(1, sizeof(*h));
    if (!h) return NULL;
    strncpy(h->type, model_type, sizeof(h->type) - 1);
    return (DM_LM)h;
}

DM_API DM_Status dm_lm_train(DM_LM lm, const char *corpus_path,
                               const char *checkpoint_dir,
                               int epochs, int batch_size, float lr)
{
    if (!lm || !corpus_path || !checkpoint_dir) return DM_ERR_INVALID_PARAM;
    _LMHandle *h = (_LMHandle *)lm;
    char ep[16], bs[16], lrs[32];
    snprintf(ep, sizeof(ep), "%d", epochs);
    snprintf(bs, sizeof(bs), "%d", batch_size);
    snprintf(lrs, sizeof(lrs), "%.6g", (double)lr);
    char *argv[] = {
        (char *)h->type, "train",
        "--corpus",     (char *)corpus_path,
        "--checkpoint", (char *)checkpoint_dir,
        "--epochs", ep, "--batch-size", bs, "--lr", lrs,
        NULL
    };
    int rc;
    if (strcmp(h->type, "bert") == 0) {
        return DM_ERR_NOT_SUPPORTED;
    } else if (strcmp(h->type, "tiny_transformer") == 0) {
        rc = dm_tiny_transformer_cli(12, argv);
    } else {
        rc = dm_tinystories_cli(12, argv);
    }
    if (rc == 0) {
        strncpy(h->checkpoint_dir, checkpoint_dir,
                sizeof(h->checkpoint_dir) - 1);
        h->loaded = 1;
    }
    return rc == 0 ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_lm_load(DM_LM lm, const char *checkpoint_dir) {
    if (!lm || !checkpoint_dir) return DM_ERR_INVALID_PARAM;
    _LMHandle *h = (_LMHandle *)lm;
    strncpy(h->checkpoint_dir, checkpoint_dir, sizeof(h->checkpoint_dir) - 1);
    h->loaded = 1;
    return DM_OK;
}

DM_API DM_Status dm_lm_generate(DM_LM lm, const char *prompt,
                                  int max_tokens, char *out_buf, int buf_size)
{
    if (!lm || !prompt || !out_buf) return DM_ERR_INVALID_PARAM;
    _LMHandle *h = (_LMHandle *)lm;
    if (strcmp(h->type, "bert") == 0) {
        BertConfig cfg;
        float *weights = NULL;
        int *ids = NULL, seq = 0;
        if (!h->loaded || dm_bert_load(h->checkpoint_dir, &cfg, &weights) != 0)
            return DM_ERR_IO;
        {
            int cap = 16;
            const char *p = prompt;
            ids = (int *)malloc((size_t)cap * sizeof(int));
            if (!ids) { free(weights); return DM_ERR_MEMORY; }
            while (*p) {
                char *end = NULL;
                long v = strtol(p, &end, 10);
                if (p == end) { p++; continue; }
                if (v < 0 || v >= cfg.vocab_size) {
                    free(weights); free(ids); return DM_ERR_INVALID_PARAM;
                }
                if (seq == cap) {
                    cap *= 2;
                    int *tmp = (int *)realloc(ids, (size_t)cap * sizeof(int));
                    if (!tmp) { free(weights); free(ids); return DM_ERR_MEMORY; }
                    ids = tmp;
                }
                ids[seq++] = (int)v;
                p = end;
            }
        }
        if (seq <= 0 || seq > cfg.max_seq_len) {
            free(weights); free(ids); return DM_ERR_INVALID_PARAM;
        }
        float *hidden = (float *)malloc((size_t)seq * cfg.hidden_size * sizeof(float));
        float *cls = (float *)malloc((size_t)cfg.hidden_size * sizeof(float));
        int *seg = (int *)calloc((size_t)seq, sizeof(int));
        if (!hidden || !cls || !seg) {
            free(weights); free(ids); free(hidden); free(cls); free(seg);
            return DM_ERR_MEMORY;
        }
        if (dm_bert_forward(&cfg, weights, ids, seg, seq, hidden, cls) != 0) {
            free(weights); free(ids); free(hidden); free(cls); free(seg);
            return DM_ERR_GENERIC;
        }
        int written = 0;
        int limit = max_tokens > 0 && max_tokens < cfg.hidden_size ? max_tokens : cfg.hidden_size;
        for (int i = 0; i < limit && written < buf_size - 1; i++) {
            int n = snprintf(out_buf + written, (size_t)(buf_size - written),
                             "%s%.7g", i ? " " : "", (double)cls[i]);
            if (n < 0 || n >= buf_size - written) break;
            written += n;
        }
        out_buf[written < buf_size ? written : buf_size - 1] = '\0';
        free(weights); free(ids); free(hidden); free(cls); free(seg);
        return DM_OK;
    }
    char tmp_out[256], mt[16];
    snprintf(tmp_out, sizeof(tmp_out), "/tmp/dm_lm_out_%d.txt", (int)getpid());
    snprintf(mt, sizeof(mt), "%d", max_tokens);
    char *argv[] = {
        (char *)h->type, "generate",
        "--checkpoint", h->checkpoint_dir,
        "--prompt",     (char *)prompt,
        "--max-tokens", mt,
        "--output",     tmp_out,
        NULL
    };
    int rc = strcmp(h->type, "tiny_transformer") == 0
           ? dm_tiny_transformer_cli(10, argv)
           : dm_tinystories_cli(10, argv);
    if (rc != 0) return DM_ERR_GENERIC;
    FILE *f = fopen(tmp_out, "r");
    if (!f) return DM_ERR_IO;
    size_t n = fread(out_buf, 1, (size_t)(buf_size - 1), f);
    out_buf[n] = '\0';
    fclose(f); remove(tmp_out);
    return DM_OK;
}

DM_API void dm_lm_free(DM_LM lm) { free(lm); }

/* =========================================================================
 * § 6b  BERT
 * ========================================================================= */

DM_API size_t dm_bert_weight_count_raw(int variant, int vocab_size, int max_seq_len)
{
    BertConfig cfg;
    dm_bert_config_init(&cfg, (BertVariant)variant, vocab_size, max_seq_len);
    return dm_bert_weight_count(&cfg);
}

DM_API DM_Status dm_bert_load_raw(const char *path,
                                  int        *variant_out,
                                  int        *vocab_size_out,
                                  int        *max_seq_len_out,
                                  float     **weights_out)
{
    if (!path || !weights_out) return DM_ERR_INVALID_PARAM;
    BertConfig cfg;
    int rc = dm_bert_load(path, &cfg, weights_out);
    if (rc != 0) return DM_ERR_GENERIC;
    if (variant_out)     *variant_out     = (int)cfg.variant;
    if (vocab_size_out)  *vocab_size_out  = cfg.vocab_size;
    if (max_seq_len_out) *max_seq_len_out = cfg.max_seq_len;
    return DM_OK;
}

DM_API void dm_bert_free_weights(float *weights)
{
    free(weights);
}

DM_API DM_Status dm_bert_forward_raw(int          variant,
                                     int          vocab_size,
                                     int          max_seq_len,
                                     const float *weights,
                                     const int   *token_ids,
                                     const int   *segment_ids,
                                     int          seq,
                                     float       *hidden_out,
                                     float       *cls_out)
{
    if (!weights || !token_ids || !segment_ids || !hidden_out) return DM_ERR_INVALID_PARAM;
    BertConfig cfg;
    dm_bert_config_init(&cfg, (BertVariant)variant, vocab_size, max_seq_len);
    int rc = dm_bert_forward(&cfg, weights, token_ids, segment_ids, seq, hidden_out, cls_out);
    return rc == 0 ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_bert_forward_masked_raw(int          variant,
                                            int          vocab_size,
                                            int          max_seq_len,
                                            const float *weights,
                                            const int   *token_ids,
                                            const int   *segment_ids,
                                            const int   *attention_mask,
                                            int          seq,
                                            float       *hidden_out,
                                            float       *cls_out)
{
    if (!weights || !token_ids || !segment_ids || !hidden_out) return DM_ERR_INVALID_PARAM;
    BertConfig cfg;
    dm_bert_config_init(&cfg, (BertVariant)variant, vocab_size, max_seq_len);
    int rc = dm_bert_forward_masked(&cfg, weights, token_ids, segment_ids, attention_mask, seq, hidden_out, cls_out);
    return rc == 0 ? DM_OK : DM_ERR_GENERIC;
}

/* =========================================================================
 * § 7  Image utilities
 *
 * dm_image_patchify_raw: raw float-buffer version (avoids name clash with
 * the tensor-based internal dm_image_patchify declared above).
 * ========================================================================= */

DM_API DM_Status dm_image_load_resize(const char *path,
                                       int out_w, int out_h, float *out_buf)
{
    if (!path || !out_buf) return DM_ERR_INVALID_PARAM;
    DM_Tensor raw, resized;
    memset(&raw,     0, sizeof(raw));
    memset(&resized, 0, sizeof(resized));

    if (dm_image_load_ppm_rgb_f32(path, &raw) != 0) return DM_ERR_IO;
    if (dm_tensor_alloc(&resized, 1, 3, out_h, out_w) != 0) {
        dm_tensor_free(&raw); return DM_ERR_MEMORY;
    }
    if (dm_image_resize_nearest(&raw, &resized, out_h, out_w) != 0) {
        dm_tensor_free(&raw); dm_tensor_free(&resized); return DM_ERR_GENERIC;
    }
    for (int y = 0; y < out_h; y++)
        for (int x = 0; x < out_w; x++)
            for (int c = 0; c < 3; c++)
                out_buf[(y * out_w + x) * 3 + c] = dm_tensor_get(&resized, 0, c, y, x);

    dm_tensor_free(&raw); dm_tensor_free(&resized);
    return DM_OK;
}

DM_API DM_Status dm_image_patchify_raw(const float *in_nhwc,
                                        int n, int h, int w, int c,
                                        int patch_h, int patch_w,
                                        float *out_buf, int *out_patches)
{
    if (!in_nhwc || !out_buf) return DM_ERR_INVALID_PARAM;
    DM_Tensor in, out;
    if (dm_tensor_alloc(&in, n, c, h, w) != 0) return DM_ERR_MEMORY;

    for (int bi = 0; bi < n; bi++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                for (int ch = 0; ch < c; ch++)
                    dm_tensor_set(&in, bi, ch, y, x,
                                  in_nhwc[((bi * h + y) * w + x) * c + ch]);

    memset(&out, 0, sizeof(out));
    if (dm_image_patchify(&in, &out, patch_h, patch_w) != 0) {
        dm_tensor_free(&in); return DM_ERR_GENERIC;
    }
    size_t total = dm_tensor_count(&out);
    memcpy(out_buf, out.data, total * sizeof(float));
    if (out_patches) *out_patches = out.h;

    dm_tensor_free(&in); dm_tensor_free(&out);
    return DM_OK;
}

/* =========================================================================
 * § 8  Tensor neural-network ops
 *
 * dm_tensor_alloc/free/fill/get/set/count are exported by tensor.c directly.
 * Only the renamed dm_op_* wrappers are defined here.
 * ========================================================================= */

DM_API DM_Status dm_op_conv2d_same(const DM_Tensor *in, DM_Tensor *out,
                                    const float *w, const float *b,
                                    int out_c, int kernel, int stride)
{
    return dm_conv2d_same(in, out, w, b, out_c, kernel, stride) == 0
           ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_op_depthwise_conv(const DM_Tensor *in, DM_Tensor *out,
                                       const float *w, const float *b,
                                       int kernel, int stride)
{
    return dm_depthwise_conv2d_same(in, out, w, b, kernel, stride) == 0
           ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_op_pointwise_conv(const DM_Tensor *in, DM_Tensor *out,
                                       const float *w, const float *b, int out_c)
{
    return dm_pointwise_conv2d(in, out, w, b, out_c) == 0
           ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_op_linear(const DM_Tensor *in, DM_Tensor *out,
                               const float *w, const float *b, int out_c)
{
    return dm_linear(in, out, w, b, out_c) == 0 ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_op_global_avg_pool(const DM_Tensor *in, DM_Tensor *out) {
    return dm_global_avg_pool(in, out) == 0 ? DM_OK : DM_ERR_GENERIC;
}

DM_API void dm_op_relu6  (DM_Tensor *t) { dm_relu6(t);   }
DM_API void dm_op_softmax(DM_Tensor *t) { dm_softmax(t); }

DM_API void dm_op_relu(DM_Tensor *t) {
    dm_relu(t);
}

DM_API DM_Status dm_op_tensor_add(DM_Tensor *out, const DM_Tensor *in) {
    return dm_tensor_add(out, in) == 0 ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_op_max_pool2d_same(const DM_Tensor *in, DM_Tensor *out, int kernel, int stride) {
    return dm_max_pool2d_same(in, out, kernel, stride) == 0 ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_op_batch_norm(DM_Tensor *t, const float *gamma, const float *beta, const float *mean, const float *var, float eps) {
    return dm_batch_norm(t, gamma, beta, mean, var, eps) == 0 ? DM_OK : DM_ERR_GENERIC;
}

extern int dm_resnet18_forward(const DM_Tensor *input, DM_Tensor *logits, int classes, unsigned int seed);
extern int dm_resnet_basic_block(const DM_Tensor *in, DM_Tensor *out, int out_c, int stride, unsigned int seed);

DM_API DM_Status dm_op_resnet18_forward(const DM_Tensor *input, DM_Tensor *logits, int classes, unsigned int seed) {
    return dm_resnet18_forward(input, logits, classes, seed) == 0 ? DM_OK : DM_ERR_GENERIC;
}

DM_API DM_Status dm_op_resnet_basic_block(const DM_Tensor *in, DM_Tensor *out, int out_c, int stride, unsigned int seed) {
    return dm_resnet_basic_block(in, out, out_c, stride, seed) == 0 ? DM_OK : DM_ERR_GENERIC;
}

/* ViT FFI wrappers — forward-declare only the symbols we need to avoid
 * re-including vit.h (which would conflict with the DM_Tensor already
 * defined via dm.h at the top of this translation unit).              */
typedef int ViTVariant_int;
typedef struct { int variant; int img_size; int patch_size; int num_layers;
                 int d_model; int mlp_dim; int num_heads; int num_classes; } ViTConfig_fwd;

extern void dm_vit_config_init(void *cfg, ViTVariant_int v, int nc, int sz);
extern int  dm_vit_forward(const DM_Tensor *in, DM_Tensor *out, const void *cfg,
                            const float *w, unsigned int seed);
extern size_t dm_vit_param_count(const void *cfg);

DM_API DM_Status dm_op_vit_forward(const DM_Tensor *input, DM_Tensor *logits,
                                    int variant, int num_classes, unsigned int seed)
{
    if (variant < 0 || variant > 4) return DM_ERR_GENERIC;
    ViTConfig_fwd cfg;
    dm_vit_config_init(&cfg, variant, num_classes, input ? input->h : 224);
    return dm_vit_forward(input, logits, &cfg, NULL, seed) == 0
           ? DM_OK : DM_ERR_GENERIC;
}

DM_API size_t dm_op_vit_param_count(int variant, int img_size, int patch_size,
                                     int num_classes)
{
    if (variant < 0 || variant > 4) return 0;
    ViTConfig_fwd cfg;
    dm_vit_config_init(&cfg, variant, num_classes, img_size);
    if (patch_size > 0) cfg.patch_size = patch_size;
    return dm_vit_param_count(&cfg);
}

/* =========================================================================
 * § 9  Benchmark — renamed wrappers only
 *
 * dm_bench_reset/start/stop/get_report are exported directly from
 * dm_benchmark.c (same names, ABI-compatible struct layouts).
 * ========================================================================= */

DM_API void dm_bench_record(size_t num_patterns, size_t total_items) {
    dm_bench_record_results(num_patterns, total_items);
}

DM_API void dm_bench_print(const char *algo, const char *ds) {
    dm_bench_print_report(algo, ds);
}

/* =========================================================================
 * § 10  BitSet — renamed popcount only
 *
 * All other bitset functions are exported from dm_bitset.c under the same
 * names.
 * ========================================================================= */

DM_API size_t dm_bitset_popcount(DM_BitSet b) {
    return dm_bitset_count(b);
}

/* =========================================================================
 * § 11  DataGen
 * ========================================================================= */

typedef struct { char type[32]; } _DataGenHandle;

DM_API DM_DataGen dm_datagen_create(const char *type) {
    if (!type) return NULL;
    if (strcmp(type, "medm") != 0 && strcmp(type, "textbook") != 0) return NULL;
    _DataGenHandle *h = (_DataGenHandle *)calloc(1, sizeof(*h));
    if (!h) return NULL;
    strncpy(h->type, type, sizeof(h->type) - 1);
    return (DM_DataGen)h;
}

static void write_medm_spec_file(const char *spec_str, const char *temp_path) {
    FILE *f = fopen(temp_path, "w");
    if (!f) return;
    char *copy = strdup(spec_str);
    char *tok = strtok(copy, ",");
    while (tok) {
        char key[128], val[128];
        if (sscanf(tok, "%127[^=]=%127s", key, val) == 2) {
            if (strcmp(key, "nItems") == 0) strcpy(key, "item_count");
            else if (strcmp(key, "nTxn") == 0) strcpy(key, "size");
            else if (strcmp(key, "avgLen") == 0) strcpy(key, "avg_len");
            fprintf(f, "%s=%s\n", key, val);
        }
        tok = strtok(NULL, ",");
    }
    free(copy);
    fclose(f);
}

DM_API DM_Status dm_datagen_run(DM_DataGen gen, const char *spec,
                                  const char *output_path, unsigned seed)
{
    if (!gen || !spec || !output_path) return DM_ERR_INVALID_PARAM;
    _DataGenHandle *h = (_DataGenHandle *)gen;

    if (strcmp(h->type, "medm") == 0) {
        char temp_spec[256];
        temp_spec[0] = '\0';
        const char *spec_file = spec;
        FILE *test_f = fopen(spec, "r");
        if (test_f) {
            fclose(test_f);
        } else {
            snprintf(temp_spec, sizeof(temp_spec), "/tmp/dm_spec_%d.txt", (int)getpid());
            write_medm_spec_file(spec, temp_spec);
            spec_file = temp_spec;
        }

        DM_Spec *spec_ptr = dm_spec_new();
        DM_Ledger *ledger_ptr = dm_ledger_new();
        if (!spec_ptr || !ledger_ptr) {
            if (spec_ptr) dm_spec_free(spec_ptr);
            if (ledger_ptr) dm_ledger_free(ledger_ptr);
            if (temp_spec[0]) remove(temp_spec);
            return DM_ERR_MEMORY;
        }

        int rc = dm_spec_parse(spec_file, spec_ptr, ledger_ptr);
        if (temp_spec[0]) remove(temp_spec);

        if (rc != 0) {
            dm_spec_free(spec_ptr);
            dm_ledger_free(ledger_ptr);
            return DM_ERR_IO;
        }

        int status = dm_medm_repair_and_feedback(spec_ptr, ledger_ptr, output_path, NULL, seed);
        dm_spec_free(spec_ptr);
        dm_ledger_free(ledger_ptr);
        return status == 0 ? DM_OK : DM_ERR_GENERIC;
    }

    char seed_s[16];
    snprintf(seed_s, sizeof(seed_s), "%u", seed);
    char *argv[] = {
        (char *)h->type, "run",
        "--spec",   (char *)spec,
        "--output", (char *)output_path,
        "--seed",   seed_s,
        NULL
    };
    return dm_textbook_generator_cli(8, argv) == 0 ? DM_OK : DM_ERR_GENERIC;
}

DM_API void dm_datagen_free(DM_DataGen gen) { free(gen); }

/* =========================================================================
 * § 12  Plugin
 * ========================================================================= */

DM_API DM_Status dm_plugin_load(const char *so_path) {
    if (!so_path) return DM_ERR_INVALID_PARAM;
#if defined(_WIN32)
    return LoadLibraryA(so_path) ? DM_OK : DM_ERR_IO;
#elif defined(__unix__) || defined(__APPLE__)
    return dlopen(so_path, RTLD_NOW | RTLD_GLOBAL) ? DM_OK : DM_ERR_IO;
#else
    (void)so_path; return DM_ERR_NOT_SUPPORTED;
#endif
}

DM_API DM_Status dm_plugin_list(DM_PluginInfo *out_buf, int *in_out_count) {
    /* No public iterator for the plugin table currently */
    if (!out_buf || !in_out_count) return DM_ERR_INVALID_PARAM;
    *in_out_count = 0;
    return DM_OK;
}

DM_API DM_Status dm_plugin_run(const char *id, const char *dataset_path,
                                 const char *output_path, const char **args)
{
    if (!id || !dataset_path) return DM_ERR_INVALID_PARAM;
    _DmPlugin *p = dm_plugin_get(id);
    if (!p) return DM_ERR_NOT_FOUND;

    DM_Arena  arena;
    DM_FlatDataset flat;
    DM_ConnectorOptions opts = dm_connector_default_options(DM_CONNECTOR_SPMF);
    if (dm_arena_init(&arena, 128 * 1024 * 1024) != 0) return DM_ERR_MEMORY;
    if (dm_flat_load_mmap(dataset_path, &opts, &arena, &flat, NULL) != 0) {
        dm_arena_free(&arena); return DM_ERR_IO;
    }

    size_t n = 0;
    if (args) while (args[n]) n++;
    _DmPluginArg *pa = (_DmPluginArg *)calloc(n / 2 + 1, sizeof(*pa));
    if (!pa) { dm_arena_free(&arena); return DM_ERR_MEMORY; }
    for (size_t i = 0; i + 1 < n; i += 2) {
        pa[i/2].key   = args[i];
        pa[i/2].value = args[i+1];
    }

    _DmPluginInput in;
    memset(&in, 0, sizeof(in));
    in.flat = &flat; in.arena = &arena;
    in.source_path = dataset_path;
    in.connector   = (int)DM_CONNECTOR_SPMF;
    in.args        = pa; in.arg_count = n / 2;

    _DmPluginResult res = {0};

    int saved = -1; FILE *outf = NULL;
    if (output_path && output_path[0]) {
        saved = DUP(FILENO(stdout));
        outf  = fopen(output_path, "w");
        if (saved >= 0 && outf) DUP2(FILENO(outf), FILENO(stdout));
    }
    int rc = p->run(&in, &res);
    if (output_path && output_path[0]) {
        fflush(stdout);
        if (saved >= 0) { DUP2(saved, FILENO(stdout)); CLOSE(saved); }
        if (outf) fclose(outf);
    }
    free(pa); dm_arena_free(&arena);
    return rc == 0 ? DM_OK : DM_ERR_GENERIC;
}

/* =========================================================================
 * § 13  CLI passthrough
 * ========================================================================= */

DM_API int dm_cli_run(const char *command, int argc, char **argv) {
    if (!command) return -1;
    if (strcmp(command, "bpe")                == 0) return dm_bpe_cli(argc, argv);
    if (strcmp(command, "bpe_dropout")        == 0) return dm_bpe_dropout_cli(argc, argv);
    if (strcmp(command, "unigram")            == 0) return dm_unigram_cli(argc, argv);
    if (strcmp(command, "sentencepiece")      == 0) return dm_sentencepiece_cli(argc, argv);
    if (strcmp(command, "gpe")                == 0) return dm_gpe_cli(argc, argv);
    if (strcmp(command, "parity_bpe")         == 0) return dm_parity_bpe_cli(argc, argv);
    if (strcmp(command, "maximal_munch")      == 0) return dm_maximal_munch_cli(argc, argv);
    if (strcmp(command, "wordpiece")          == 0) return dm_fast_wordpiece_cli(argc, argv);
    if (strcmp(command, "tokenizer_lab")      == 0) return dm_tokenizer_lab_cli(argc, argv);
    if (strcmp(command, "volt")               == 0) return dm_volt_cli(argc, argv);
    if (strcmp(command, "mobilenet_tiny")     == 0) return dm_mobilenet_tiny_cli(argc, argv);
    if (strcmp(command, "tinyvit")            == 0) return dm_tinyvit_cli(argc, argv);
    if (strcmp(command, "transformer")        == 0) return dm_transformer_cli(argc, argv);
    if (strcmp(command, "tiny_transformer")   == 0) return dm_tiny_transformer_cli(argc, argv);
    if (strcmp(command, "tinystories")        == 0) return dm_tinystories_cli(argc, argv);
    if (strcmp(command, "bert")               == 0) return dm_bert_cli(argc, argv);
    if (strcmp(command, "textbook_generator") == 0) return dm_textbook_generator_cli(argc, argv);
    if (strcmp(command, "version")            == 0) { printf("%s\n", dm_version()); return 0; }
    fprintf(stderr, "dm: unknown command '%s'\n", command);
    return 1;
}

/* =========================================================================
 * § 14  GPU — only dm_gpu_free needs a rename; everything else is exported
 *        directly from dm_gpu.c under the same public names.
 * ========================================================================= */

DM_API void dm_gpu_free(DM_GpuCtx ctx) {
    dm_gpu_destroy(ctx);
}

/* =========================================================================
 * § 15  Arena — exported directly by dm_arena.c (same names, same layout)
 * § 16  MMap  — exported directly by dm_mmap.c  (same names, same layout)
 * § 17  Flat Dataset — dm_connector_name/parse_kind exported by dm_flat.c;
 *        dm_connector_default_opts and dm_flat_load are renamed here.
 * ========================================================================= */

DM_API DM_ConnectorOptions dm_connector_default_opts(DM_ConnectorKind k) {
    return dm_connector_default_options(k);
}

DM_API int dm_flat_load(const char *path, const DM_ConnectorOptions *opts,
                         DM_Arena *arena, DM_FlatDataset *out,
                         DM_ConnectorStats *stats)
{
    return dm_flat_load_mmap(path, opts, arena, out, stats);
}

/* =========================================================================
 * § 18  Experiment helpers
 *
 * dm_file_size_mb / dm_ensure_dir / dm_path_basename /
 * dm_experiment_generate_report / dm_dataset_stats_write — exported by
 * their respective .c files under the same public names.
 *
 * Renamed:  dm_timer_now → dm_timer_now_seconds
 *           dm_peak_ram_mb → get_peak_ram_mb
 *           dm_dir_size_mb → dm_directory_size_mb
 *           dm_experiment_write_header → dm_csv_write_raw_header
 *           dm_experiment_append_row   → dm_csv_append_raw_row  (+ struct conv)
 * ========================================================================= */

DM_API double dm_timer_now(void)             { return dm_timer_now_seconds(); }
DM_API double dm_peak_ram_mb(void)           { return get_peak_ram_mb(); }
DM_API double dm_dir_size_mb(const char *p)  { return dm_directory_size_mb(p); }

DM_API void dm_experiment_write_header(const char *csv_path) {
    dm_csv_write_raw_header(csv_path);
}

DM_API void dm_experiment_append_row(const char *csv_path,
                                      const DM_ExperimentRow *row)
{
    if (!csv_path || !row) return;
    _ExperimentRow ir;
    memset(&ir, 0, sizeof(ir));
    strncpy(ir.dataset,   row->dataset,   sizeof(ir.dataset)   - 1);
    strncpy(ir.algorithm, row->algorithm, sizeof(ir.algorithm) - 1);
    strncpy(ir.status,    row->status,    sizeof(ir.status)    - 1);
    ir.minsup_ratio                   = row->minsup_ratio;
    ir.minsup_count                   = row->minsup_count;
    ir.minocc                         = row->minocc;
    ir.run_id                         = row->run_id;
    ir.runtime_seconds                = row->runtime_seconds;
    ir.peak_ram_mb                    = row->peak_ram_mb;
    ir.temp_disk_mb                   = row->temp_disk_mb;
    ir.output_disk_mb                 = row->output_disk_mb;
    ir.num_generated_candidates       = row->num_generated_candidates;
    ir.num_output_itemsets            = row->num_output_itemsets;
    ir.num_fhoi                       = row->num_fhoi;
    ir.num_weak_mfhoi                 = row->num_weak_mfhoi;
    ir.num_strong_mfhoi               = row->num_strong_mfhoi;
    ir.dominance_removed_count        = row->dominance_removed_count;
    /* ir.quality is left zeroed — not present in public struct */
    ir.compression_vs_fi              = row->compression_vs_fi;
    ir.compression_vs_fhoi            = row->compression_vs_fhoi;
    ir.reduction_vs_fhoi_percent      = row->reduction_vs_fhoi_percent;
    ir.strong_extra_reduction_percent = row->strong_extra_reduction_percent;
    dm_csv_append_raw_row(csv_path, &ir);
}

/* ─────────────────────────────────────────────────────────────────────────
 * § 9  Variational Auto-Encoder (VAE)
 * ───────────────────────────────────────────────────────────────────────── */

DM_API void* dm_vae_create_raw(int input_dim, int hidden_dim, int latent_dim, float lr) {
    DM_VAE *vae = (DM_VAE*)malloc(sizeof(DM_VAE));
    if (!vae) return NULL;
    dm_vae_init(vae, input_dim, hidden_dim, latent_dim, lr);
    return vae;
}

DM_API void dm_vae_free_raw(void *vae) {
    if (!vae) return;
    dm_vae_free((DM_VAE*)vae);
    free(vae);
}

DM_API float dm_vae_train_step_raw(void *vae, const float *x_batch, int batch_size) {
    if (!vae || !x_batch) return 0.0f;
    DM_VAE *v = (DM_VAE*)vae;
    DM_Tensor x;
    dm_tensor_alloc(&x, batch_size, v->input_dim, 1, 1);
    for (int i = 0; i < batch_size * v->input_dim; i++) x.data[i] = x_batch[i];
    float loss = dm_vae_train_step(v, &x);
    dm_tensor_free(&x);
    return loss;
}

DM_API void dm_vae_encode_raw(void *vae, const float *x_batch, int batch_size, float *mean_out, float *logvar_out) {
    if (!vae || !x_batch || !mean_out || !logvar_out) return;
    DM_VAE *v = (DM_VAE*)vae;
    DM_Tensor x, mean, logvar;
    dm_tensor_alloc(&x, batch_size, v->input_dim, 1, 1);
    for (int i = 0; i < batch_size * v->input_dim; i++) x.data[i] = x_batch[i];
    dm_vae_encode(v, &x, &mean, &logvar);
    for (int i = 0; i < batch_size * v->latent_dim; i++) {
        mean_out[i] = mean.data[i];
        logvar_out[i] = logvar.data[i];
    }
    dm_tensor_free(&x);
    dm_tensor_free(&mean);
    dm_tensor_free(&logvar);
}

DM_API void dm_vae_decode_raw(void *vae, const float *z_batch, int batch_size, float *out) {
    if (!vae || !z_batch || !out) return;
    DM_VAE *v = (DM_VAE*)vae;
    DM_Tensor z, dec_out;
    dm_tensor_alloc(&z, batch_size, v->latent_dim, 1, 1);
    for (int i = 0; i < batch_size * v->latent_dim; i++) z.data[i] = z_batch[i];
    dm_vae_decode(v, &z, &dec_out);
    for (int i = 0; i < batch_size * v->input_dim; i++) {
        out[i] = dec_out.data[i];
    }
    dm_tensor_free(&z);
    dm_tensor_free(&dec_out);
}

/* ─────────────────────────────────────────────────────────────────────────
 * § 10  Generative Adversarial Network (GAN)
 * ───────────────────────────────────────────────────────────────────────── */

DM_API void* dm_gan_create_raw(int input_dim, int g_hidden, int noise_dim, int d_hidden, int maxout_k, float drop_prob, float lr, float momentum, int nesterov) {
    DM_GAN *gan = (DM_GAN*)malloc(sizeof(DM_GAN));
    if (!gan) return NULL;
    dm_gan_init(gan, input_dim, g_hidden, noise_dim, d_hidden, maxout_k, drop_prob, lr, momentum, nesterov);
    return gan;
}

DM_API void dm_gan_free_raw(void *gan) {
    if (!gan) return;
    dm_gan_free((DM_GAN*)gan);
    free(gan);
}

DM_API void dm_gan_generate_raw(void *gan, const float *z_batch, int batch_size, float *out) {
    if (!gan || !z_batch || !out) return;
    DM_GAN *g = (DM_GAN*)gan;
    DM_Tensor z, dec_out;
    dm_tensor_alloc(&z, batch_size, g->noise_dim, 1, 1);
    for (int i = 0; i < batch_size * g->noise_dim; i++) z.data[i] = z_batch[i];
    dm_tensor_alloc(&dec_out, batch_size, g->input_dim, 1, 1);
    
    dm_gan_generate(g, &z, &dec_out);
    for (int i = 0; i < batch_size * g->input_dim; i++) {
        out[i] = dec_out.data[i];
    }
    dm_tensor_free(&z);
    dm_tensor_free(&dec_out);
}

DM_API float dm_gan_train_d_step_raw(void *gan, const float *real_x_batch, const float *z_batch, int batch_size) {
    if (!gan || !real_x_batch || !z_batch) return 0.0f;
    DM_GAN *g = (DM_GAN*)gan;
    DM_Tensor x, z;
    dm_tensor_alloc(&x, batch_size, g->input_dim, 1, 1);
    dm_tensor_alloc(&z, batch_size, g->noise_dim, 1, 1);
    for (int i = 0; i < batch_size * g->input_dim; i++) x.data[i] = real_x_batch[i];
    for (int i = 0; i < batch_size * g->noise_dim; i++) z.data[i] = z_batch[i];
    
    float loss = dm_gan_train_d_step(g, &x, &z);
    
    dm_tensor_free(&x);
    dm_tensor_free(&z);
    return loss;
}

DM_API float dm_gan_train_g_step_raw(void *gan, const float *z_batch, int batch_size) {
    if (!gan || !z_batch) return 0.0f;
    DM_GAN *g = (DM_GAN*)gan;
    DM_Tensor z;
    dm_tensor_alloc(&z, batch_size, g->noise_dim, 1, 1);
    for (int i = 0; i < batch_size * g->noise_dim; i++) z.data[i] = z_batch[i];
    
    float loss = dm_gan_train_g_step(g, &z);
    
    dm_tensor_free(&z);
    return loss;
}

