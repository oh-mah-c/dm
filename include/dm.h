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
 * Sections:
 *   § 1   Core / Version
 *   § 2   Dataset
 *   § 3   Algorithm  (132 registered data-mining algorithms)
 *   § 4   Tokenizer
 *   § 5   Vision Model  (MobileNetV4 Tiny + TinyViT-5M/11M/21M)
 *   § 6   Language Model  (Transformer enc-dec + TinyTransformer + TinyStories)
 *   § 7   Image utilities
 *   § 8   Tensor  (primitive ops, C-only)
 *   § 9   Benchmark
 *   § 10  BitSet
 *   § 11  Synthetic Data Generator  (MEDM / Textbook)
 *   § 12  Plugin system
 *   § 13  CLI passthrough
 *   § 14  GPU Compute Acceleration  (Vulkan backend)
 *   § 15  Arena Allocator
 *   § 16  Memory-mapped I/O
 *   § 17  Flat Dataset / Connector
 *   § 18  Experiment helpers
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
DM_API uint32_t    dm_version_number(void);   /* major<<16 | minor<<8 | patch */
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
 * § 5  Vision Model
 *       • "mobilenet_tiny"   — MobileNetV4-Tiny (TF C API backend)
 *       • "tinyvit_5m"       — TinyViT-5M  (Wu et al. arXiv:2207.10666v1)
 *       • "tinyvit_11m"      — TinyViT-11M
 *       • "tinyvit_21m"      — TinyViT-21M  [DEFAULT]
 *
 * TinyViT architecture (Section 3.2):
 *   Patch Embed → Stage1:MBConv×2 → DS → Stage2:Transformer×2(W=7) →
 *   DS → Stage3:Transformer×6(W=14) → DS → Stage4:Transformer×2(W=7) →
 *   AvgPool+LN+Linear
 *   Shared: depths={2,2,6,2}, windows={7,14,7}, R=4, M=4, E=32
 *
 * Fast Pretraining Distillation (Section 3.1):
 *   Teacher logits are sparsified (top-K) and stored on disk.
 *   Student trains via dm_vision_distill_train() reusing stored labels.
 * ───────────────────────────────────────────────────────────────────────── */

typedef void *DM_Vision;

/** model_type: "mobilenet_tiny" | "tinyvit_5m" | "tinyvit_11m" | "tinyvit_21m" */
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

/* ── TinyViT direct API (pure-C, no TF dependency for inference) ────────── */

/** TinyViT variant selector */
typedef enum {
    DM_TINYVIT_5M  = 0,
    DM_TINYVIT_11M = 1,
    DM_TINYVIT_21M = 2
} DM_TinyViTVariant;

/**
 * Count total float parameters for a TinyViT variant.
 * @param variant   DM_TINYVIT_5M / 11M / 21M
 * @param classes   Number of output classes (e.g. 1000)
 * @param img_size  Input resolution (e.g. 224)
 */
DM_API size_t dm_tinyvit_weight_count(DM_TinyViTVariant variant,
                                       int               classes,
                                       int               img_size);

/**
 * Pure-C forward inference pass.
 * @param variant      Model size
 * @param weights      float[dm_tinyvit_weight_count(variant,classes,img_size)]
 * @param input_nhwc   float[batch × img_size × img_size × 3], values in [0,1]
 * @param batch        Batch size
 * @param classes      Number of output classes
 * @param img_size     Input resolution
 * @param logits_out   float[batch × classes]  (pre-softmax, caller-allocated)
 */
DM_API DM_Status  dm_tinyvit_forward(DM_TinyViTVariant variant,
                                      const float      *weights,
                                      const float      *input_nhwc,
                                      int               batch,
                                      int               classes,
                                      int               img_size,
                                      float            *logits_out);

/** Load weights from a .bin file written by 'dm tinyvit train'. */
DM_API DM_Status  dm_tinyvit_load(const char *weight_path,
                                   DM_TinyViTVariant *variant_out,
                                   int               *classes_out,
                                   int               *img_size_out,
                                   float            **weights_out);

/**
 * Fast Pretraining Distillation — save sparse teacher logits (Section 3.1).
 * @param out_path      Output .tspl file path
 * @param num_images    Number of images in the dataset
 * @param num_classes   Total number of classes C
 * @param topK          Top-K logits to store per image (K≪C)
 * @param indices       uint32[num_images × topK]  top-K class indices
 * @param values        float[num_images × topK]   top-K softmax values
 * @param aug_seeds     uint32[num_images]          PCG seeds (d_0)
 */
DM_API DM_Status  dm_tinyvit_save_labels(const char     *out_path,
                                          int             num_images,
                                          int             num_classes,
                                          int             topK,
                                          const uint32_t *indices,
                                          const float    *values,
                                          const uint32_t *aug_seeds);

/**
 * Compute sparse cross-entropy distillation loss (Eq. 1–2).
 * @param student_logits  float[C]  pre-softmax
 * @param indices         uint32[K] top-K teacher indices
 * @param teacher_values  float[K]  top-K teacher softmax values
 * @param K               sparsity
 * @param C               total classes
 * @param temperature     distillation temperature (1.0 per paper)
 * @param loss_out        output scalar loss
 */
DM_API DM_Status  dm_tinyvit_distill_loss(const float    *student_logits,
                                           const uint32_t *indices,
                                           const float    *teacher_values,
                                           int             K,
                                           int             C,
                                           float           temperature,
                                           float          *loss_out);

/* ─────────────────────────────────────────────────────────────────────────
 * § 6  Language Model
 *       • "transformer"       — Transformer encoder-decoder (Vaswani et al.
 *                               NeurIPS 2017 / arXiv:1706.03762)
 *                               Variants: base (65M) and big (213M)
 *       • "tiny_transformer"  — Compact byte-level causal LM
 *       • "tinystories"       — TinyStories byte-level causal LM
 * ───────────────────────────────────────────────────────────────────────── */

typedef void *DM_LM;

/** model_type: "transformer" | "tiny_transformer" | "tinystories" */
DM_API DM_LM     dm_lm_create   (const char *model_type);
DM_API DM_Status  dm_lm_train   (DM_LM lm,
                                  const char *corpus_path,
                                  const char *checkpoint_dir,
                                  int         epochs,
                                  int         batch_size,
                                  float       lr);
DM_API DM_Status  dm_lm_generate(DM_LM        lm,
                                  const char  *prompt,
                                  int          max_tokens,
                                  char        *out_buf,
                                  int          buf_size);
DM_API DM_Status  dm_lm_load    (DM_LM lm, const char *checkpoint_dir);
DM_API void       dm_lm_free    (DM_LM lm);

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
DM_API DM_Status dm_image_patchify_raw(const float *in_nhwc,
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
} DM_Tensor;

DM_API DM_Status dm_tensor_alloc (DM_Tensor *t, int n, int c, int h, int w);
DM_API void      dm_tensor_free  (DM_Tensor *t);
DM_API void      dm_tensor_fill  (DM_Tensor *t, float value);
DM_API float     dm_tensor_get   (const DM_Tensor *t, int n, int c, int y, int x);
DM_API void      dm_tensor_set   (DM_Tensor *t, int n, int c, int y, int x, float v);
DM_API size_t    dm_tensor_count (const DM_Tensor *t);

/* Primitive neural ops */
DM_API DM_Status dm_op_conv2d_same   (const DM_Tensor *in, DM_Tensor *out,
                                      const float *w, const float *b,
                                      int out_c, int kernel, int stride);
DM_API DM_Status dm_op_depthwise_conv(const DM_Tensor *in, DM_Tensor *out,
                                      const float *w, const float *b,
                                      int kernel, int stride);
DM_API DM_Status dm_op_pointwise_conv(const DM_Tensor *in, DM_Tensor *out,
                                      const float *w, const float *b, int out_c);
DM_API DM_Status dm_op_linear        (const DM_Tensor *in, DM_Tensor *out,
                                      const float *w, const float *b, int out_c);
DM_API DM_Status dm_op_global_avg_pool(const DM_Tensor *in, DM_Tensor *out);
DM_API void      dm_op_relu6         (DM_Tensor *t);
DM_API void      dm_op_softmax       (DM_Tensor *t);

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
    double  phase_times_ms[4];       /* Load, Algo, Write, Total */
    size_t  peak_memory_kb;          /* High-water mark in KB */
    double  user_cpu_ms;             /* User CPU time */
    double  sys_cpu_ms;              /* System CPU time */
    size_t  result_ram_bytes;        /* Exact RAM footprint of results */
    size_t  result_disk_est_bytes;   /* Estimated disk size (CSV/TXT) */
    size_t  num_patterns;            /* Total patterns / itemsets found */
    size_t  total_items;             /* Sum of lengths of all patterns */
    double  throughput_mb_s;         /* Data size / total time */
} DM_BenchReport;

DM_API void           dm_bench_reset        (void);
DM_API void           dm_bench_start        (DM_BenchPhase phase);
DM_API void           dm_bench_stop         (DM_BenchPhase phase);
DM_API void           dm_bench_record       (size_t num_patterns, size_t total_items);
DM_API DM_BenchReport dm_bench_get_report   (void);
DM_API void           dm_bench_print        (const char *algo_name, const char *dataset_name);

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

/** Load an external .so / .dll plugin at runtime. */
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

/* ─────────────────────────────────────────────────────────────────────────
 * § 14  GPU Compute Acceleration  (Vulkan 1.1 backend)
 *
 * Runtime-loaded — no link-time dependency on vulkan-1.lib.
 * Falls back to CPU automatically when a GPU is unavailable.
 *
 * Supported operations:
 *   dm_gpu_bpe_pair_count  — parallel pair-frequency histogram (BPE training)
 *   dm_gpu_sinkhorn        — Sinkhorn balanced OT iterations (VOLT)
 *   dm_gpu_unigram_em_step — forward-backward EM E-step (Unigram training)
 * ───────────────────────────────────────────────────────────────────────── */

/** Opaque GPU context — one per training run. NOT thread-safe. */
typedef void *DM_GpuCtx;

/** GPU error codes (returned by dm_gpu_* functions, not DM_Status). */
#define DM_GPU_OK                    0
#define DM_GPU_ERR_NO_DEVICE        -1   /* no Vulkan-capable GPU found       */
#define DM_GPU_ERR_INIT             -2   /* Vulkan init failed                */
#define DM_GPU_ERR_SHADER           -3   /* .spv file not found / invalid     */
#define DM_GPU_ERR_OOM              -4   /* GPU out of memory                 */
#define DM_GPU_ERR_VOCAB_TOO_LARGE  -5   /* vocab > DM_GPU_BPE_MAX_VOCAB      */
#define DM_GPU_ERR_NO_ATOMIC_FLOAT  -6   /* VK_EXT_shader_atomic_float absent */

/** Maximum vocabulary size for the dense BPE pair-count matrix (64 MB). */
#define DM_GPU_BPE_MAX_VOCAB 4096

/**
 * Create a Vulkan compute context.
 *   device_index  0 = first discrete GPU, -1 = driver pick.
 *   shader_dir    directory containing compiled .spv files, or NULL for
 *                 automatic search: shaders/ → shaders/vulkan/tokenizer/
 *                 → bin/shaders/ → exe-dir/shaders/
 * Returns NULL on failure; call dm_gpu_free() when done.
 */
DM_API DM_GpuCtx dm_gpu_create     (int device_index, const char *shader_dir);
DM_API void      dm_gpu_free       (DM_GpuCtx ctx);
DM_API int       dm_gpu_ready      (DM_GpuCtx ctx);   /* 1 = usable, 0 = not */
DM_API int       dm_gpu_device_name(DM_GpuCtx ctx, char *buf, size_t buf_len);

/**
 * Flat corpus for BPE pair counting.
 * All words concatenated into sym_ids[]; word w spans
 * [word_starts[w], word_starts[w] + word_lens[w]).
 */
typedef struct {
    const uint32_t *sym_ids;      /* flat symbol-ID array, length = total_syms */
    size_t          total_syms;
    const uint32_t *word_starts;  /* word_starts[w] = first index in sym_ids   */
    const uint32_t *word_lens;    /* word_lens[w]   = symbol count of word w   */
    const uint32_t *word_freqs;   /* word_freqs[w]  = corpus frequency         */
    size_t          n_words;
    uint32_t        vocab_size;   /* distinct symbol IDs, must be ≤ MAX_VOCAB  */
} DM_GpuBpeInput;

/**
 * Count adjacent-pair frequencies on the GPU (or CPU fallback).
 *   pair_counts[left * vocab_size + right] += word_freqs[w]
 * Caller allocates and zeroes pair_counts[vocab_size * vocab_size].
 */
DM_API int dm_gpu_bpe_pair_count(DM_GpuCtx ctx,
                                  const DM_GpuBpeInput *in,
                                  uint32_t             *pair_counts);

/** Sparse CSR matrix, used for the Sinkhorn kernel (§14 / VOLT). */
typedef struct {
    const uint32_t *row_ptr;   /* length = n_rows + 1   */
    const uint32_t *col_idx;   /* length = nnz          */
    const float    *vals;      /* length = nnz          */
    uint32_t        n_rows;
    uint32_t        n_cols;
    uint32_t        nnz;
} DM_GpuCSR;

/**
 * Run Sinkhorn iterations on GPU (or CPU fallback).
 * K and Kt must be the same matrix and its transpose in CSR form.
 * row_sums_out[n_tok] receives sum_j u[i]*K[i,j]*v[j] for each row.
 */
DM_API int dm_gpu_sinkhorn(DM_GpuCtx       ctx,
                            const DM_GpuCSR *K,
                            const DM_GpuCSR *Kt,
                            const float     *p_tok,
                            const float     *p_char,
                            int              max_iter,
                            float            tol,
                            float           *row_sums_out);

/** Unigram language model descriptor for the GPU EM step. */
typedef struct {
    const float   *log_probs;     /* log(prob) per piece, length = n_pieces    */
    uint32_t       n_pieces;
    uint32_t       max_piece_len; /* max codepoint-length of any piece         */
} DM_GpuUnigramModel;

/** Codepoint-ID corpus with pre-computed piece coverage in CSR form. */
typedef struct {
    const uint16_t *cp_ids;         /* flat codepoint-ID array, length = total_cps */
    size_t          total_cps;
    const uint32_t *word_starts;    /* start index in cp_ids for word w            */
    const uint32_t *word_lens;      /* codepoint length of word w                  */
    const uint32_t *word_freqs;     /* corpus frequency of word w                  */
    size_t          n_words;
    const uint32_t *piece_row_ptr;  /* length = total_cps + 1                      */
    const uint32_t *piece_col;      /* piece IDs valid at each position             */
    uint32_t        total_covered_arcs;
} DM_GpuUnigramCorpus;

/**
 * Run one EM E-step on GPU (or CPU fallback).
 * new_counts[n_pieces] and *expected_total are accumulated (not reset).
 */
DM_API int dm_gpu_unigram_em_step(DM_GpuCtx                  ctx,
                                   const DM_GpuUnigramModel  *model,
                                   const DM_GpuUnigramCorpus *corpus,
                                   float                     *new_counts,
                                   float                     *expected_total);

/**
 * Build a CSR matrix from COO (row, col, val) triples.
 * Free *row_ptr_out, *col_idx_out, *vals_out with free().
 */
DM_API void dm_gpu_build_csr(const uint32_t *coo_row,
                              const uint32_t *coo_col,
                              const float    *coo_val,
                              uint32_t        nnz,
                              uint32_t        n_rows,
                              uint32_t        n_cols,
                              uint32_t      **row_ptr_out,
                              uint32_t      **col_idx_out,
                              float         **vals_out);

/** Build the transpose of a CSR matrix. Free outputs with free(). */
DM_API void dm_gpu_csr_transpose(const DM_GpuCSR *K,
                                  uint32_t       **kt_row_ptr_out,
                                  uint32_t       **kt_col_idx_out,
                                  float          **kt_vals_out);

/* CPU fallbacks (called automatically; also available for direct use): */
DM_API void dm_gpu_bpe_pair_count_cpu(const DM_GpuBpeInput *in,
                                       uint32_t             *pair_counts);
DM_API void dm_gpu_sinkhorn_cpu      (const DM_GpuCSR *K,
                                       const DM_GpuCSR *Kt,
                                       const float     *p_tok,
                                       const float     *p_char,
                                       int              max_iter,
                                       float            tol,
                                       float           *row_sums_out);

/* ─────────────────────────────────────────────────────────────────────────
 * § 15  Arena Allocator
 *
 * Fast bump-pointer allocator.  Use when you need many small, short-lived
 * allocations that are freed together (algorithm work buffers, parsed
 * datasets, etc.).  The arena owns its memory unless created via
 * dm_arena_wrap() with an external buffer.
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct {
    unsigned char *base;       /* Backing store start     */
    size_t         capacity;   /* Total bytes available   */
    size_t         offset;     /* Next free byte          */
    int            owns_memory;/* 1 if we called malloc   */
} DM_Arena;

/**
 * Initialise a new arena backed by a malloc'd block of `capacity` bytes.
 * Returns 0 on success, -1 on allocation failure.
 */
DM_API int   dm_arena_init   (DM_Arena *arena, size_t capacity);

/** Wrap an externally-owned buffer.  dm_arena_free() becomes a no-op. */
DM_API void  dm_arena_wrap   (DM_Arena *arena, void *memory, size_t capacity);

/** Reset the bump pointer to zero (does NOT free backing memory). */
DM_API void  dm_arena_reset  (DM_Arena *arena);

/** Free backing memory if owns_memory, then zero-out the struct. */
DM_API void  dm_arena_free   (DM_Arena *arena);

/**
 * Allocate `size` bytes with `alignment` (must be power-of-two).
 * Returns NULL if the arena is exhausted.
 */
DM_API void *dm_arena_alloc  (DM_Arena *arena, size_t size, size_t alignment);

/** Duplicate at most `len` bytes of `src` into the arena (NUL-terminated). */
DM_API char *dm_arena_strndup(DM_Arena *arena, const char *src, size_t len);

/* ─────────────────────────────────────────────────────────────────────────
 * § 16  Memory-mapped I/O
 *
 * Thin cross-platform (mmap / MapViewOfFile) wrapper.  The mapped region
 * is read-only and alive until dm_mmap_close().
 * ───────────────────────────────────────────────────────────────────────── */

typedef struct {
    const unsigned char *data;   /* Pointer to mapped bytes    */
    size_t               size;   /* File size in bytes         */
    int                  fd;     /* File descriptor (POSIX)    */
    int                  mapped; /* 1 if mmap is active        */
} DM_MMap;

/** Open and mmap `path`.  Returns 0 on success. */
DM_API int  dm_mmap_open (const char *path, DM_MMap *map);

/** Unmap and close.  Safe to call on a zero-initialised struct. */
DM_API void dm_mmap_close(DM_MMap *map);

/* ─────────────────────────────────────────────────────────────────────────
 * § 17  Flat Dataset / Connector
 *
 * Low-level zero-copy dataset representation shared by all mining
 * algorithms.  The DM_FlatDataset owns no memory — it borrows from a
 * DM_MMap + DM_Arena pair.
 * ───────────────────────────────────────────────────────────────────────── */

typedef enum {
    DM_CONNECTOR_SPMF  = 0,   /* Space-separated integer items per line       */
    DM_CONNECTOR_TEXT  = 1,   /* Sliding-window tokeniser over UTF-8 text     */
    DM_CONNECTOR_GRAPH = 2    /* Adjacency list → per-node neighbourhood txns */
} DM_ConnectorKind;

typedef struct {
    uint32_t *items;           /* Flat array of all item IDs, row-major        */
    size_t    item_count;      /* Total number of items across all rows        */
    size_t   *row_offsets;     /* row_offsets[r] = start index of row r        */
    size_t    row_count;       /* Number of rows (transactions)                */
    uint32_t  max_item;        /* Maximum item ID present                      */
    DM_ConnectorKind source_kind;
    DM_Arena        *arena;          /* Arena that owns items[] and row_offsets[] */
    const unsigned char *borrowed_base;  /* Mmap base (borrowed, not owned)   */
    size_t               borrowed_bytes;
} DM_FlatDataset;

typedef struct {
    DM_ConnectorKind kind;
    size_t  text_window;       /* Token window size (TEXT connector)           */
    size_t  text_stride;       /* Window stride     (TEXT connector)           */
    int     text_sequence;     /* 1 = preserve order; 0 = sorted unique        */
    int     graph_undirected;  /* 1 = add reverse edges (GRAPH connector)      */
} DM_ConnectorOptions;

typedef struct {
    size_t   input_bytes;
    size_t   rows;
    size_t   items;
    size_t   distinct_estimate;
    uint32_t max_item;
    double   avg_row_len;
} DM_ConnectorStats;

DM_API const char          *dm_connector_name        (DM_ConnectorKind kind);
DM_API int                  dm_connector_parse_kind  (const char *name,
                                                       DM_ConnectorKind *out_kind);
DM_API DM_ConnectorOptions  dm_connector_default_opts(DM_ConnectorKind kind);

/**
 * Parse `path` into `out` using a memory-mapped zero-copy pass.
 * `arena` must outlive `out`.  `stats` may be NULL.
 */
DM_API int dm_flat_load(const char          *path,
                         const DM_ConnectorOptions *opts,
                         DM_Arena            *arena,
                         DM_FlatDataset      *out,
                         DM_ConnectorStats   *stats);

/* ─────────────────────────────────────────────────────────────────────────
 * § 18  Experiment helpers
 *
 * Lightweight CSV logger for multi-algorithm benchmark runs.
 * ───────────────────────────────────────────────────────────────────────── */

/**
 * One row in the experiment results CSV.
 * All string fields are NUL-terminated; numeric fields use 0 for "not set".
 */
typedef struct {
    char   dataset[256];
    char   algorithm[64];
    double minsup_ratio;
    int    minsup_count;
    double minocc;
    int    run_id;
    char   status[32];         /* "ok" | "timeout" | "oom" | "error" */
    double runtime_seconds;
    double peak_ram_mb;
    double temp_disk_mb;
    double output_disk_mb;
    int    num_generated_candidates;
    int    num_output_itemsets;
    /* HOI-specific counters (zero for non-HOI algorithms): */
    int    num_fhoi;
    int    num_weak_mfhoi;
    int    num_strong_mfhoi;
    int    dominance_removed_count;
    double compression_vs_fi;
    double compression_vs_fhoi;
    double reduction_vs_fhoi_percent;
    double strong_extra_reduction_percent;
} DM_ExperimentRow;

/** Return the current wall-clock time in seconds (monotonic clock). */
DM_API double dm_timer_now(void);

/** Return peak resident set size in MB (reads /proc/self/status on Linux). */
DM_API double dm_peak_ram_mb(void);

/** Return the size of a single file in MB (0 if not found). */
DM_API double dm_file_size_mb(const char *path);

/** Return the total size of all files under a directory in MB. */
DM_API double dm_dir_size_mb(const char *path);

/** Recursively create directories along `path` (like mkdir -p). */
DM_API int  dm_ensure_dir(const char *path);

/** Return the last path component (no allocation; points inside `path`). */
DM_API const char *dm_path_basename(const char *path);

/** Write dataset statistics for an array of file paths to a CSV. */
DM_API void dm_dataset_stats_write(const char  *csv_path,
                                    const char **paths,
                                    int          count);

/** Write the CSV header row (call once before dm_experiment_append_row). */
DM_API void dm_experiment_write_header(const char *csv_path);

/** Append one DM_ExperimentRow to an existing CSV. */
DM_API void dm_experiment_append_row(const char            *csv_path,
                                      const DM_ExperimentRow *row);

/** Scan results_root for CSV files and emit a Markdown summary report. */
DM_API void dm_experiment_generate_report(const char *results_root);

#ifdef __cplusplus
}
#endif

#endif /* DM_H */
