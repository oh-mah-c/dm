/**
 * dm.h — DM Framework Public C API (Stable ABI v1)
 *
 * Single public header for libdm. All language bindings (Python, JS, Go, Java)
 * are built against this file only.
 *
 * Rules:
 *  - Opaque void* handles for all stateful objects
 *  - Only primitives, C-strings, and flat arrays cross the ABI
 *  - Every _create() has a matching _free()
 *  - All functions return DM_Status (0 = OK, negative = error)
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef DM_H
#define DM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Visibility ─────────────────────────────────────────────────────────── */
#if defined(_WIN32) || defined(__CYGWIN__)
#  ifdef DM_BUILDING_LIB
#    define DM_API __declspec(dllexport)
#  else
#    define DM_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#  define DM_API __attribute__((visibility("default")))
#else
#  define DM_API
#endif

/* ── Status codes ───────────────────────────────────────────────────────── */
typedef enum {
    DM_OK                = 0,
    DM_ERR_GENERIC       = -1,
    DM_ERR_IO            = -2,
    DM_ERR_MEMORY        = -3,
    DM_ERR_INVALID_PARAM = -4,
    DM_ERR_NOT_FOUND     = -5,
    DM_ERR_INCOMPATIBLE  = -6,
    DM_ERR_NOT_SUPPORTED = -7
} DM_Status;

/* ─────────────────────────────────────────────────────────────────────────
 * § 1  Core / Version
 * ───────────────────────────────────────────────────────────────────────── */
DM_API const char *dm_version(void);
DM_API uint32_t    dm_version_number(void);  /* major<<16|minor<<8|patch */
DM_API DM_Status   dm_init(void);
DM_API const char *dm_strerror(DM_Status code);

/* ─────────────────────────────────────────────────────────────────────────
 * § 2  Dataset
 * ───────────────────────────────────────────────────────────────────────── */

typedef void *DM_Dataset;

/** type: "transactional" | "utility" | "sequence" | "quantity" | "matrix" */
DM_API DM_Dataset dm_dataset_open  (const char *path, const char *type);
DM_API size_t     dm_dataset_count (DM_Dataset ds);
DM_API uint32_t   dm_dataset_max_id(DM_Dataset ds);
DM_API void       dm_dataset_free  (DM_Dataset ds);

/* ─────────────────────────────────────────────────────────────────────────
 * § 3  Algorithm — 132 registered data-mining algorithms
 *
 *  Supported algorithm IDs (pass to dm_algorithm_create):
 *
 *  Frequent Itemset Mining (closed/maximal/rare):
 *    "apriori"          "apriori_tid"       "apriori_hybrid"
 *    "apriori_inverse"  "apriori_rare"      "eclat"
 *    "fpgrowth"         "fpclose"           "fpmax"
 *    "lcm"              "lcmver2"           "charm"
 *    "close"            "closet"            "closetplus"
 *    "dci_closed"       "aclose"            "carpenter"
 *    "mafia"            "max_miner"         "genmax"
 *    "pascal"           "prepost"           "prepostplus"
 *    "cfi_stream"       "clostream"         "estdec"
 *    "defme"            "cori"              "rp_tree"
 *    "tree_projection"  "negfin"            "fin"
 *    "finplus"          "regular_mine"      "cls_miner"
 *    "slim"             "zart"              "sam"
 *    "dic"              "ais"               "krimp"
 *    "opus_miner"       "dbv_miner"         "dfi_growth"
 *
 *  High-Utility Itemset Mining (HUIM):
 *    "efim"             "efim_closed"       "fhm"
 *    "fhim"             "fhn"               "fhmds"
 *    "huiminer"         "hui_miner"         "hui_list_ins"
 *    "huci_miner"       "hup_miner"         "hupe_garm"
 *    "ihup"             "haui_miner"        "hauim_gmu"
 *    "ehaupm"           "minfhm"            "fchm"
 *    "foshu"            "thui"              "dphim"
 *    "mlhui_miner"      "ghui_miner"        "fcfia"
 *    "ffiminer"         "memu"              "sum"
 *    "tku_ce"           "tku_ce_plus"       "eihi"
 *    "thue"             "tshoun"            "mheinu"
 *    "nafcp"            "feacp"             "closed_fhuim_kinana"
 *
 *  High-Utility Itemset Mining — Bio-Inspired / Swarm:
 *    "huim_abc"         "huim_aco"          "huim_bpso"
 *    "huim_bpso_tree"   "huim_hc"           "huim_sa"
 *    "huim_afsa"        "huimsu"            "bio_huif_ba"
 *    "bio_huif_ga"      "bio_huif_pso"
 *
 *  High-Utility Itemset Mining — Quantity / Uncertain:
 *    "fhuqi_miner"      "vhuqi"             "uapriori"
 *    "ubmffp"           "ufh"               "ulb_miner"
 *
 *  High-Utility Sequential Pattern Mining:
 *    "hupspm"           "uspan"             "up_growth"
 *    "up_hist"
 *
 *  High-Occupancy / Skyline / Negative:
 *    "hoimto"           "skyline_miner"     "skymine"
 *    "ltm"              "sfui_uf"           "sfu_ce"
 *    "medm_gen"         "msapriori"         "tkq"
 *    "tkhoim"           "tmku"              "topkphm"
 *
 *  High-Order Interaction / Association (HOI):
 *    "fhoi"             "fhoi_miner"        "dfhoi"
 *    "aura_hoi"         "hep"               "hiep"
 *    "nam_hep"          "cloe_hoi"          "mfhoi"
 *    "mhoui"            "chuo_miner"        "sparc_hoi"
 *    "strong_mfhoi"     "weak_mfhoi"        "clhminer"
 *    "talky_g"          "tipn_houi"
 *
 *  Sequence / Stream Mining:
 *    "cfi_stream"       "clostream"         "estdec"
 *    "tku_pso"          "rminer"
 *
 *  Closed / Association Rule:
 *    "chuimine_closed"  "chuimine_maximal"  "chui_miner"
 *    "pso_classifier"
 * ───────────────────────────────────────────────────────────────────────── */

typedef void *DM_Algorithm;

DM_API DM_Algorithm dm_algorithm_create(const char *id);
DM_API DM_Status    dm_algorithm_run   (DM_Algorithm algo,
                                        const char  *dataset_path,
                                        const char  *output_path,
                                        double       min_support,
                                        const char **extra_args);  /* NULL-terminated or NULL */
DM_API DM_Status    dm_algorithm_list  (char *buf, int buf_size);  /* newline-separated IDs */
DM_API void         dm_algorithm_free  (DM_Algorithm algo);

/* ─────────────────────────────────────────────────────────────────────────
 * § 4  Tokenizer
 *
 *  type: "bpe" | "bpe_dropout" | "unigram" | "sentencepiece"
 *        "wordpiece" | "gpe" | "parity_bpe" | "volt" | "maximal_munch"
 *        "faro" | "tokenizer_lab"
 * ───────────────────────────────────────────────────────────────────────── */

typedef void *DM_Tokenizer;

DM_API DM_Tokenizer dm_tokenizer_create    (const char *type);
DM_API DM_Status    dm_tokenizer_train     (DM_Tokenizer tok,
                                            const char  *corpus_path,
                                            int          vocab_size,
                                            const char  *output_path);
DM_API DM_Status    dm_tokenizer_load      (DM_Tokenizer tok, const char *model_path);
DM_API DM_Status    dm_tokenizer_encode    (DM_Tokenizer  tok,
                                            const char   *text,
                                            uint32_t     *out_ids,
                                            int          *in_out_len);  /* in: capacity, out: count */
DM_API DM_Status    dm_tokenizer_decode    (DM_Tokenizer   tok,
                                            const uint32_t *ids,
                                            int             n_ids,
                                            char           *out_buf,
                                            int             buf_size);
DM_API int          dm_tokenizer_vocab_size(DM_Tokenizer tok);
DM_API const char  *dm_tokenizer_token_text(DM_Tokenizer tok, uint32_t id, uint32_t *out_len);
DM_API void         dm_tokenizer_free      (DM_Tokenizer tok);

/* VOLT-specific: Sinkhorn optimal-transport vocabulary learning */
DM_API DM_Status dm_tokenizer_volt_run(const char *corpus_path,
                                       int         min_size,
                                       int         max_size,
                                       int         n_steps,
                                       const char *output_path);

/* ─────────────────────────────────────────────────────────────────────────
 * § 5  Vision Model (MobileNetV4 Tiny — TF C API backend)
 * ───────────────────────────────────────────────────────────────────────── */

typedef void *DM_Vision;

/** model_type: currently "mobilenet_tiny" */
DM_API DM_Vision  dm_vision_create (const char *model_type);
DM_API DM_Status  dm_vision_init   (DM_Vision   v,
                                    const char *saved_model_dir,
                                    int         classes,
                                    int         image_size,
                                    float       width_mult,
                                    float       learning_rate);
DM_API DM_Status  dm_vision_train  (DM_Vision   v,
                                    const char *manifest_path,
                                    int         epochs,
                                    int         batch_size,
                                    float       lr);
DM_API DM_Status  dm_vision_eval   (DM_Vision   v,
                                    const char *manifest_path,
                                    float      *out_loss,
                                    float      *out_accuracy);
/**
 * Predict on a single image.
 * @param rgb      Row-major float32 [h*w*3], values in [0,1].
 * @param out_probs Caller-allocated [n_classes] floats.
 */
DM_API DM_Status  dm_vision_predict(DM_Vision    v,
                                    const float *rgb,
                                    int          h,
                                    int          w,
                                    float       *out_probs,
                                    int          n_classes);
DM_API DM_Status  dm_vision_save   (DM_Vision v);
DM_API void       dm_vision_free   (DM_Vision v);

/* Raw MobileNetV4-Tiny forward pass without TF (random weights, C-only) */
DM_API DM_Status  dm_vision_forward_raw(const float *rgb_nhwc,
                                        int          n,
                                        int          h,
                                        int          w,
                                        float       *out_logits,
                                        int          classes);

/* ─────────────────────────────────────────────────────────────────────────
 * § 6  Language Model (Tiny Transformer / TinyStories byte-LM)
 * ───────────────────────────────────────────────────────────────────────── */

typedef void *DM_LM;

/** model_type: "tiny_transformer" | "tinystories" */
DM_API DM_LM    dm_lm_create   (const char *model_type);
DM_API DM_Status dm_lm_train   (DM_LM lm,
                                 const char *corpus_path,
                                 const char *checkpoint_dir,
                                 int         epochs,
                                 int         batch_size,
                                 float       lr);
DM_API DM_Status dm_lm_generate(DM_LM        lm,
                                 const char  *prompt,
                                 int          max_tokens,
                                 char        *out_buf,
                                 int          buf_size);
DM_API DM_Status dm_lm_load    (DM_LM lm, const char *checkpoint_dir);
DM_API void      dm_lm_free    (DM_LM lm);

/* ─────────────────────────────────────────────────────────────────────────
 * § 7  Image utilities
 * ───────────────────────────────────────────────────────────────────────── */

/**
 * Load a PPM image, resize to (out_w × out_h) with nearest-neighbour,
 * write normalized float32 RGB [0,1] row-major into out_buf
 * (must hold out_h * out_w * 3 floats).
 */
DM_API DM_Status dm_image_load_resize(const char *path,
                                      int         out_w,
                                      int         out_h,
                                      float      *out_buf);

/**
 * Split an image tensor into non-overlapping patches.
 * @param in_nhwc   Input float32 [n,h,w,c] row-major.
 * @param out_buf   Output [n, n_patches, patch_h*patch_w*c].
 * @param patch_h   Patch height in pixels.
 * @param patch_w   Patch width in pixels.
 * @param out_patches Set by callee: number of patches per image.
 */
DM_API DM_Status dm_image_patchify(const float *in_nhwc,
                                   int          n,
                                   int          h,
                                   int          w,
                                   int          c,
                                   int          patch_h,
                                   int          patch_w,
                                   float       *out_buf,
                                   int         *out_patches);

/* ─────────────────────────────────────────────────────────────────────────
 * § 8  Tensor (primitive operations — C-only, no TF dependency)
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct {
    int    n, c, h, w;
    float *data;
} DM_Tensor_t;

DM_API DM_Status dm_tensor_alloc (DM_Tensor_t *t, int n, int c, int h, int w);
DM_API void      dm_tensor_free  (DM_Tensor_t *t);
DM_API void      dm_tensor_fill  (DM_Tensor_t *t, float value);
DM_API float     dm_tensor_get   (const DM_Tensor_t *t, int n, int c, int y, int x);
DM_API void      dm_tensor_set   (DM_Tensor_t *t, int n, int c, int y, int x, float v);
DM_API size_t    dm_tensor_count (const DM_Tensor_t *t);

/* Primitive neural ops */
DM_API DM_Status dm_op_conv2d_same   (const DM_Tensor_t *in, DM_Tensor_t *out,
                                      const float *w, const float *b,
                                      int out_c, int kernel, int stride);
DM_API DM_Status dm_op_depthwise_conv(const DM_Tensor_t *in, DM_Tensor_t *out,
                                      const float *w, const float *b,
                                      int kernel, int stride);
DM_API DM_Status dm_op_pointwise_conv(const DM_Tensor_t *in, DM_Tensor_t *out,
                                      const float *w, const float *b, int out_c);
DM_API DM_Status dm_op_linear        (const DM_Tensor_t *in, DM_Tensor_t *out,
                                      const float *w, const float *b, int out_c);
DM_API DM_Status dm_op_global_avg_pool(const DM_Tensor_t *in, DM_Tensor_t *out);
DM_API void      dm_op_relu6         (DM_Tensor_t *t);
DM_API void      dm_op_softmax       (DM_Tensor_t *t);

/* ─────────────────────────────────────────────────────────────────────────
 * § 9  Benchmark
 * ───────────────────────────────────────────────────────────────────────── */

typedef enum {
    DM_BENCH_PHASE_LOAD  = 0,
    DM_BENCH_PHASE_ALGO  = 1,
    DM_BENCH_PHASE_WRITE = 2,
    DM_BENCH_PHASE_TOTAL = 3
} DM_BenchPhase;

typedef struct {
    double  phase_times_ms[4];   /* Load, Algo, Write, Total */
    size_t  peak_memory_kb;
    double  user_cpu_ms;
    double  sys_cpu_ms;
    size_t  result_ram_bytes;
    size_t  result_disk_est_bytes;
    size_t  num_patterns;
    size_t  total_items;
    double  throughput_mb_s;
} DM_BenchReport;

DM_API void          dm_bench_reset         (void);
DM_API void          dm_bench_start         (DM_BenchPhase phase);
DM_API void          dm_bench_stop          (DM_BenchPhase phase);
DM_API void          dm_bench_record        (size_t num_patterns, size_t total_items);
DM_API DM_BenchReport dm_bench_get_report   (void);
DM_API void          dm_bench_print         (const char *algo_name, const char *dataset_name);

/* ─────────────────────────────────────────────────────────────────────────
 * § 10  BitSet
 * ───────────────────────────────────────────────────────────────────────── */

typedef void *DM_BitSet;

DM_API DM_BitSet dm_bitset_create  (size_t n_bits);
DM_API DM_BitSet dm_bitset_copy    (DM_BitSet src);
DM_API void      dm_bitset_free    (DM_BitSet bs);
DM_API void      dm_bitset_set     (DM_BitSet bs, size_t pos);
DM_API void      dm_bitset_clear   (DM_BitSet bs, size_t pos);
DM_API bool      dm_bitset_get     (DM_BitSet bs, size_t pos);
DM_API void      dm_bitset_and     (DM_BitSet dest, DM_BitSet src);
DM_API void      dm_bitset_or      (DM_BitSet dest, DM_BitSet src);
DM_API void      dm_bitset_not     (DM_BitSet bs);
DM_API void      dm_bitset_set_all (DM_BitSet bs);
DM_API size_t    dm_bitset_popcount(DM_BitSet bs);

/* ─────────────────────────────────────────────────────────────────────────
 * § 11  Synthetic Data Generator (MEDM / Textbook)
 * ───────────────────────────────────────────────────────────────────────── */

typedef void *DM_DataGen;

/**
 * type: "medm"     — synthetic transactional dataset (Ledger-spec driven)
 *       "textbook" — Phi/Textbooks-Are-All-You-Need corpus builder
 */
DM_API DM_DataGen dm_datagen_create  (const char *type);
DM_API DM_Status  dm_datagen_run     (DM_DataGen  gen,
                                      const char *spec_path_or_params,
                                      const char *output_path,
                                      unsigned    seed);
DM_API void       dm_datagen_free    (DM_DataGen gen);

/* ─────────────────────────────────────────────────────────────────────────
 * § 12  Plugin system
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct {
    const char *id;
    const char *name;
    const char *description;
} DM_PluginInfo;

/** Load an external .so plugin at runtime. */
DM_API DM_Status dm_plugin_load  (const char *so_path);
DM_API DM_Status dm_plugin_list  (DM_PluginInfo *out_buf, int *in_out_count);
DM_API DM_Status dm_plugin_run   (const char *id,
                                  const char *dataset_path,
                                  const char *output_path,
                                  const char **args);   /* NULL-terminated */

/* ─────────────────────────────────────────────────────────────────────────
 * § 13  CLI passthrough
 *
 * Execute any dm sub-command as if called from the terminal.
 * Useful for scripting languages that want simple shell-like access.
 *
 * Example: dm_cli_run("mobilenet_tiny", 5, argv_array)
 * ───────────────────────────────────────────────────────────────────────── */
DM_API int dm_cli_run(const char *command, int argc, char **argv);

#ifdef __cplusplus
}
#endif

#endif /* DM_H */
