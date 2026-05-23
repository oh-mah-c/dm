/**
 * dm.hpp — C++ binding for libdm  (header-only, C++17)
 *
 * Provides RAII wrappers and an idiomatic C++ interface over the stable
 * C ABI defined in <dm.h>.  No source file is needed — just:
 *
 *   #include "bindings/cpp/dm.hpp"
 *
 * Compile with:
 *   g++ -std=c++17 -I include your_file.cpp -L . -ldm -o your_prog
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "dm.h"       /* stable C ABI — must be on the include path */

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <functional>
#include <cstring>

namespace dm {

/* ── Helpers ─────────────────────────────────────────────────────────────── */

/** Throw a std::runtime_error if status is not DM_OK. */
inline void check(DM_Status s, const char *ctx = "") {
    if (s != DM_OK)
        throw std::runtime_error(std::string(ctx) + ": " + dm_strerror(s));
}

/** Library version string. */
inline std::string version() { return dm_version(); }

/** Initialise the library (idempotent). */
inline void init() { check(dm_init(), "dm::init"); }

/* ─────────────────────────────────────────────────────────────────────────
 * Backend selection
 *
 * dm_engine picks the best available backend automatically.
 * Override with the DM_BACKEND env var or backend_set().
 * ───────────────────────────────────────────────────────────────────────── */

/** Detect available backends and select the best one (idempotent). */
inline void backend_init() { dm_backend_init(); }

/** Return the currently active backend. */
inline DM_Backend backend_get() { return dm_backend_get(); }

/**
 * Override the active backend.
 * Pass DM_BACKEND_AUTO to re-run auto-detection.
 */
inline void backend_set(DM_Backend b) { dm_backend_set(b); }

/** Return a full capability snapshot. */
inline DM_BackendInfo backend_query() { return dm_backend_query(); }

/** Human-readable name for a backend constant. */
inline std::string backend_name(DM_Backend b) { return dm_backend_name(b); }

/* ─────────────────────────────────────────────────────────────────────────
 * Dataset
 * ───────────────────────────────────────────────────────────────────────── */

class Dataset {
public:
    /**
     * @param type "transactional" | "utility" | "sequence" | "quantity" | "matrix"
     */
    explicit Dataset(std::string_view path, std::string_view type) {
        handle_ = dm_dataset_open(std::string(path).c_str(),
                                  std::string(type).c_str());
        if (!handle_) throw std::runtime_error("dm::Dataset: open failed");
    }
    ~Dataset() { if (handle_) dm_dataset_free(handle_); }

    Dataset(const Dataset &)             = delete;
    Dataset &operator=(const Dataset &)  = delete;
    Dataset(Dataset &&o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }

    size_t   count () const { return dm_dataset_count (handle_); }
    uint32_t max_id() const { return dm_dataset_max_id(handle_); }
    DM_Dataset raw()  const { return handle_; }

private:
    DM_Dataset handle_{nullptr};
};

/* ─────────────────────────────────────────────────────────────────────────
 * Algorithm
 * ───────────────────────────────────────────────────────────────────────── */

class Algorithm {
public:
    explicit Algorithm(std::string_view id) {
        handle_ = dm_algorithm_create(std::string(id).c_str());
        if (!handle_) throw std::runtime_error("dm::Algorithm: unknown id: " +
                                                std::string(id));
    }
    ~Algorithm() { if (handle_) dm_algorithm_free(handle_); }

    Algorithm(const Algorithm &)             = delete;
    Algorithm &operator=(const Algorithm &)  = delete;
    Algorithm(Algorithm &&o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }

    /**
     * Run the algorithm.
     * @param extra_args  Key=value pairs forwarded verbatim; may be empty.
     */
    void run(std::string_view dataset_path,
             std::string_view output_path,
             double           min_support,
             const std::vector<std::string> &extra_args = {})
    {
        /* Build a NULL-terminated array of C-string pointers. */
        std::vector<const char *> ptrs;
        ptrs.reserve(extra_args.size() + 1);
        for (auto &a : extra_args) ptrs.push_back(a.c_str());
        ptrs.push_back(nullptr);

        check(dm_algorithm_run(handle_,
                               std::string(dataset_path).c_str(),
                               std::string(output_path).c_str(),
                               min_support,
                               ptrs.data()),
              "dm::Algorithm::run");
    }

    /** List all registered algorithm IDs, newline-separated. */
    static std::string list() {
        std::string buf(16384, '\0');
        check(dm_algorithm_list(buf.data(), static_cast<int>(buf.size())),
              "dm::Algorithm::list");
        buf.resize(std::strlen(buf.c_str()));
        return buf;
    }

    DM_Algorithm raw() const { return handle_; }

private:
    DM_Algorithm handle_{nullptr};
};

/* ─────────────────────────────────────────────────────────────────────────
 * Tokenizer
 * ───────────────────────────────────────────────────────────────────────── */

class Tokenizer {
public:
    /**
     * @param type  "bpe" | "bpe_dropout" | "unigram" | "sentencepiece" |
     *              "wordpiece" | "gpe" | "parity_bpe" | "volt" |
     *              "maximal_munch" | "faro" | "tokenizer_lab"
     */
    explicit Tokenizer(std::string_view type) {
        handle_ = dm_tokenizer_create(std::string(type).c_str());
        if (!handle_) throw std::runtime_error("dm::Tokenizer: unknown type: " +
                                                std::string(type));
    }
    ~Tokenizer() { if (handle_) dm_tokenizer_free(handle_); }

    Tokenizer(const Tokenizer &)             = delete;
    Tokenizer &operator=(const Tokenizer &)  = delete;
    Tokenizer(Tokenizer &&o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }

    void train(std::string_view corpus_path,
               int              vocab_size,
               std::string_view output_path)
    {
        check(dm_tokenizer_train(handle_,
                                 std::string(corpus_path).c_str(),
                                 vocab_size,
                                 std::string(output_path).c_str()),
              "dm::Tokenizer::train");
    }

    void load(std::string_view model_path) {
        check(dm_tokenizer_load(handle_, std::string(model_path).c_str()),
              "dm::Tokenizer::load");
    }

    /** Encode `text` → token IDs. */
    std::vector<uint32_t> encode(std::string_view text) {
        /* First call with capacity 0 to get count (or use a generous buffer). */
        std::vector<uint32_t> ids(4096);
        int len = static_cast<int>(ids.size());
        DM_Status s = dm_tokenizer_encode(handle_,
                                          std::string(text).c_str(),
                                          ids.data(), &len);
        if (s == DM_ERR_MEMORY) {
            ids.resize(static_cast<size_t>(len));
            check(dm_tokenizer_encode(handle_,
                                      std::string(text).c_str(),
                                      ids.data(), &len),
                  "dm::Tokenizer::encode");
        } else {
            check(s, "dm::Tokenizer::encode");
        }
        ids.resize(static_cast<size_t>(len));
        return ids;
    }

    /** Decode token IDs → UTF-8 string. */
    std::string decode(const std::vector<uint32_t> &ids) {
        std::string buf(ids.size() * 8 + 64, '\0');
        check(dm_tokenizer_decode(handle_,
                                  ids.data(),
                                  static_cast<int>(ids.size()),
                                  buf.data(),
                                  static_cast<int>(buf.size())),
              "dm::Tokenizer::decode");
        buf.resize(std::strlen(buf.c_str()));
        return buf;
    }

    int         vocab_size()                              const { return dm_tokenizer_vocab_size(handle_); }
    const char *token_text(uint32_t id, uint32_t *len)   const { return dm_tokenizer_token_text(handle_, id, len); }

    /** VOLT vocabulary learning via Sinkhorn OT. */
    static void volt_run(std::string_view corpus_path,
                         int min_size, int max_size, int n_steps,
                         std::string_view output_path)
    {
        check(dm_tokenizer_volt_run(std::string(corpus_path).c_str(),
                                    min_size, max_size, n_steps,
                                    std::string(output_path).c_str()),
              "dm::Tokenizer::volt_run");
    }

    DM_Tokenizer raw() const { return handle_; }

private:
    DM_Tokenizer handle_{nullptr};
};

/* ─────────────────────────────────────────────────────────────────────────
 * Vision  (MobileNetV4 Tiny)
 * ───────────────────────────────────────────────────────────────────────── */

class Vision {
public:
    /** model_type: "mobilenet_tiny" */
    explicit Vision(std::string_view model_type = "mobilenet_tiny") {
        handle_ = dm_vision_create(std::string(model_type).c_str());
        if (!handle_) throw std::runtime_error("dm::Vision: create failed");
    }
    ~Vision() { if (handle_) dm_vision_free(handle_); }

    Vision(const Vision &)             = delete;
    Vision &operator=(const Vision &)  = delete;
    Vision(Vision &&o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }

    void init(std::string_view saved_model_dir,
              int classes, int image_size,
              float width_mult, float learning_rate)
    {
        check(dm_vision_init(handle_,
                             std::string(saved_model_dir).c_str(),
                             classes, image_size, width_mult, learning_rate),
              "dm::Vision::init");
    }

    void train(std::string_view manifest_path,
               int epochs, int batch_size, float lr)
    {
        check(dm_vision_train(handle_,
                              std::string(manifest_path).c_str(),
                              epochs, batch_size, lr),
              "dm::Vision::train");
    }

    struct EvalResult { float loss; float accuracy; };
    EvalResult eval(std::string_view manifest_path) {
        EvalResult r{};
        check(dm_vision_eval(handle_,
                             std::string(manifest_path).c_str(),
                             &r.loss, &r.accuracy),
              "dm::Vision::eval");
        return r;
    }

    /** Predict on row-major float32 RGB [h*w*3] in [0,1]. */
    std::vector<float> predict(const float *rgb, int h, int w, int n_classes) {
        std::vector<float> probs(static_cast<size_t>(n_classes));
        check(dm_vision_predict(handle_, rgb, h, w, probs.data(), n_classes),
              "dm::Vision::predict");
        return probs;
    }

    void save() { check(dm_vision_save(handle_), "dm::Vision::save"); }

    DM_Vision raw() const { return handle_; }

private:
    DM_Vision handle_{nullptr};
};

/* ─────────────────────────────────────────────────────────────────────────
 * Language Model
 * ───────────────────────────────────────────────────────────────────────── */

class LM {
public:
    /** model_type: "bert" | "tiny_transformer" | "tinystories" */
    explicit LM(std::string_view model_type) {
        handle_ = dm_lm_create(std::string(model_type).c_str());
        if (!handle_) throw std::runtime_error("dm::LM: create failed");
    }
    ~LM() { if (handle_) dm_lm_free(handle_); }

    LM(const LM &)            = delete;
    LM &operator=(const LM &) = delete;
    LM(LM &&o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }

    void train(std::string_view corpus_path,
               std::string_view checkpoint_dir,
               int epochs, int batch_size, float lr)
    {
        check(dm_lm_train(handle_,
                          std::string(corpus_path).c_str(),
                          std::string(checkpoint_dir).c_str(),
                          epochs, batch_size, lr),
              "dm::LM::train");
    }

    void load(std::string_view checkpoint_dir) {
        check(dm_lm_load(handle_, std::string(checkpoint_dir).c_str()),
              "dm::LM::load");
    }

    std::string generate(std::string_view prompt, int max_tokens) {
        std::string buf(max_tokens * 4 + 256, '\0');
        check(dm_lm_generate(handle_,
                             std::string(prompt).c_str(),
                             max_tokens, buf.data(),
                             static_cast<int>(buf.size())),
              "dm::LM::generate");
        buf.resize(std::strlen(buf.c_str()));
        return buf;
    }

    DM_LM raw() const { return handle_; }

private:
    DM_LM handle_{nullptr};
};

/* ─────────────────────────────────────────────────────────────────────────
 * Engine — Tensor and neural-op primitives
 *
 * dm_engine is the compute core: every op dispatches through the TensorFlow
 * Eager C API (TFE_*).  Users compose custom models from these primitives
 * the same way they would compose TensorFlow layers.
 *
 * Layout conventions:
 *   Tensors           — NCHW (n,c,h,w), row-major, contiguous float32
 *   Conv weights      — OIHW [out_c][in_c][ky][kx]
 *   Depthwise weights — [c][ky][kx]
 *   Linear weights    — [out][in]
 *   Sequence buffers  — row-major [seq_len × d_model]
 *
 * Quick example:
 *   dm::Tensor x(1, 3, 224, 224), y(1, 64, 112, 112);
 *   dm::op::conv2d_same(x, y, weights, bias, 64, 3, 2);
 *   dm::op::relu(y);
 * ───────────────────────────────────────────────────────────────────────── */

class Tensor {
public:
    Tensor() = default;
    Tensor(int n, int c, int h, int w) { alloc(n, c, h, w); }
    ~Tensor() { dm_tensor_free(&t_); }

    Tensor(const Tensor &)             = delete;
    Tensor &operator=(const Tensor &)  = delete;
    Tensor(Tensor &&o) noexcept : t_(o.t_) { o.t_ = {}; }

    void alloc(int n, int c, int h, int w) {
        check(dm_tensor_alloc(&t_, n, c, h, w), "dm::Tensor::alloc");
    }
    void         fill(float v)                              { dm_tensor_fill(&t_, v); }
    float        get(int n, int c, int y, int x) const     { return dm_tensor_get(&t_, n, c, y, x); }
    void         set(int n, int c, int y, int x, float v)  { dm_tensor_set(&t_, n, c, y, x, v); }
    size_t       count()   const { return dm_tensor_count(&t_); }
    float       *data()          { return t_.data; }
    const float *data()    const { return t_.data; }
    int          n() const { return t_.n; }
    int          c() const { return t_.c; }
    int          h() const { return t_.h; }
    int          w() const { return t_.w; }

    DM_Tensor       &raw()       { return t_; }
    const DM_Tensor &raw() const { return t_; }

private:
    DM_Tensor t_{};
};

/* ─────────────────────────────────────────────────────────────────────────
 * op — free functions matching the dm_op_* C API
 *
 * All functions throw std::runtime_error on failure (consistent with the
 * rest of dm.hpp).  Void ops (activations, optimisers) never throw.
 * ───────────────────────────────────────────────────────────────────────── */
namespace op {

/* ── Convolutions ────────────────────────────────────────────────────────── */

/** Standard conv2d, SAME padding.  w: OIHW [out_c][in_c][ky][kx]. */
inline void conv2d_same(const Tensor &in, Tensor &out,
                         const float *w, const float *b,
                         int out_c, int kernel, int stride)
{
    check(dm_op_conv2d_same(&in.raw(), &out.raw(), w, b, out_c, kernel, stride),
          "dm::op::conv2d_same");
}

/** Depthwise separable conv, SAME padding.  w: [c][ky][kx]. */
inline void depthwise_conv(const Tensor &in, Tensor &out,
                            const float *w, const float *b,
                            int kernel, int stride)
{
    check(dm_op_depthwise_conv(&in.raw(), &out.raw(), w, b, kernel, stride),
          "dm::op::depthwise_conv");
}

/** Pointwise (1×1) conv.  w: [out_c][in_c]. */
inline void pointwise_conv(const Tensor &in, Tensor &out,
                             const float *w, const float *b, int out_c)
{
    check(dm_op_pointwise_conv(&in.raw(), &out.raw(), w, b, out_c),
          "dm::op::pointwise_conv");
}

/* ── Linear / FC ─────────────────────────────────────────────────────────── */

/** Fully-connected.  in: [n,in_c,1,1] → out: [n,out_c,1,1].  w: [out_c×in_c]. */
inline void linear(const Tensor &in, Tensor &out,
                   const float *w, const float *b, int out_c)
{
    check(dm_op_linear(&in.raw(), &out.raw(), w, b, out_c), "dm::op::linear");
}

/* ── Pooling ─────────────────────────────────────────────────────────────── */

inline void global_avg_pool(const Tensor &in, Tensor &out) {
    check(dm_op_global_avg_pool(&in.raw(), &out.raw()), "dm::op::global_avg_pool");
}

inline void max_pool2d_same(const Tensor &in, Tensor &out, int kernel, int stride) {
    check(dm_op_max_pool2d_same(&in.raw(), &out.raw(), kernel, stride),
          "dm::op::max_pool2d_same");
}

/* ── Normalisation ───────────────────────────────────────────────────────── */

inline void batch_norm(Tensor &t,
                        const float *gamma, const float *beta,
                        const float *mean,  const float *var, float eps = 1e-5f)
{
    check(dm_op_batch_norm(&t.raw(), gamma, beta, mean, var, eps),
          "dm::op::batch_norm");
}

/**
 * Layer norm for sequence models.
 * x: row-major float32 [seq_len × d_model], mutated in-place.
 */
inline void layer_norm(float *x, int seq_len, int d_model,
                        const float *gamma, const float *beta, float eps = 1e-5f)
{
    check(dm_op_layer_norm(x, seq_len, d_model, gamma, beta, eps),
          "dm::op::layer_norm");
}

/* ── Elementwise ─────────────────────────────────────────────────────────── */

inline void add(Tensor &out, const Tensor &in) {
    check(dm_op_tensor_add(&out.raw(), &in.raw()), "dm::op::add");
}

/* ── Activations (in-place) ──────────────────────────────────────────────── */

inline void relu    (Tensor &t) { dm_op_relu(&t.raw());    }
inline void relu6   (Tensor &t) { dm_op_relu6(&t.raw());   }
inline void tanh_   (Tensor &t) { dm_op_tanh(&t.raw());    }   /* tanh_ avoids <cmath> clash */
inline void sigmoid (Tensor &t) { dm_op_sigmoid(&t.raw()); }

/** GELU on a raw float buffer (for sequence models). */
inline void gelu(float *x, int n) { dm_op_gelu(x, n); }

/* ── Softmax ─────────────────────────────────────────────────────────────── */

/** Softmax over the channel dim of an [n,c,1,1] tensor. */
inline void softmax(Tensor &t) { dm_op_softmax(&t.raw()); }

/** Softmax over rows of a raw [rows × cols] buffer (in-place). */
inline void softmax_rows(float *x, int rows, int cols) {
    dm_op_softmax_rows(x, rows, cols);
}

/* ── Matrix multiplication ───────────────────────────────────────────────── */

/** C = A × Bᵀ   A[M×K], B[N×K] → C[M×N]. */
inline void matmul_nt(const float *A, const float *B, float *C,
                       int M, int N, int K) {
    dm_op_matmul_nt(A, B, C, M, N, K);
}

/** C = A × B    A[M×K], B[K×N] → C[M×N]. */
inline void matmul_nn(const float *A, const float *B, float *C,
                       int M, int K, int N) {
    dm_op_matmul_nn(A, B, C, M, K, N);
}

/* ── Backward passes ─────────────────────────────────────────────────────── */

inline void linear_backward(const Tensor &in, const Tensor &grad_out,
                              Tensor &grad_in,
                              float *grad_w, float *grad_b,
                              const float *w, int out_c)
{
    check(dm_op_linear_backward(&in.raw(), &grad_out.raw(), &grad_in.raw(),
                                 grad_w, grad_b, w, out_c),
          "dm::op::linear_backward");
}

inline void relu_backward(const Tensor &in, const Tensor &grad_out,
                           Tensor &grad_in)
{
    dm_op_relu_backward(&in.raw(), &grad_out.raw(), &grad_in.raw());
}

inline void tanh_backward(const Tensor &out, const Tensor &grad_out,
                           Tensor &grad_in)
{
    dm_op_tanh_backward(&out.raw(), &grad_out.raw(), &grad_in.raw());
}

/* ── Maxout ──────────────────────────────────────────────────────────────── */

/**
 * Maxout pooling.  in: [n, k×c, 1, 1] → out: [n, c, 1, 1].
 * argmax: caller-allocated int[n×c].
 */
inline void maxout(const Tensor &in, Tensor &out, int k, int *argmax) {
    check(dm_op_maxout(&in.raw(), &out.raw(), k, argmax), "dm::op::maxout");
}
inline void maxout_backward(const Tensor &grad_out, Tensor &grad_in,
                              int k, const int *argmax)
{
    check(dm_op_maxout_backward(&grad_out.raw(), &grad_in.raw(), k, argmax),
          "dm::op::maxout_backward");
}

/* ── Dropout ─────────────────────────────────────────────────────────────── */

/** mask: caller-allocated int[n×c×h×w]. */
inline void dropout(const Tensor &in, Tensor &out, float drop_prob, int *mask) {
    dm_op_dropout(&in.raw(), &out.raw(), drop_prob, mask);
}
inline void dropout_backward(const Tensor &grad_out, Tensor &grad_in,
                               float drop_prob, const int *mask)
{
    dm_op_dropout_backward(&grad_out.raw(), &grad_in.raw(), drop_prob, mask);
}

/* ── Optimisers ──────────────────────────────────────────────────────────── */

/**
 * Adam parameter update.
 * param, grad, m, v: float[n].  t: current step (1-indexed).
 */
inline void adam_step(float *param, float *grad, float *m, float *v,
                       int n, float lr = 1e-3f,
                       float beta1 = 0.9f, float beta2 = 0.999f,
                       float eps = 1e-8f, float weight_decay = 0.f, int t = 1)
{
    dm_op_adam_step(param, grad, m, v, n, lr, beta1, beta2, eps, weight_decay, t);
}

/** Adagrad parameter update.  param, grad, g_sum: float[n]. */
inline void adagrad_step(float *param, float *grad, float *g_sum,
                          int n, float lr = 1e-2f,
                          float eps = 1e-8f, float weight_decay = 0.f)
{
    dm_op_adagrad_step(param, grad, g_sum, n, lr, eps, weight_decay);
}

/** SGD + momentum.  param, grad, velocity: float[n]. */
inline void sgd_momentum_step(float *param, float *grad, float *velocity,
                               int n, float lr = 1e-2f,
                               float momentum = 0.9f, float weight_decay = 0.f,
                               bool nesterov = false)
{
    dm_op_sgd_momentum_step(param, grad, velocity, n, lr, momentum,
                             weight_decay, nesterov ? 1 : 0);
}

} /* namespace op */

/* ─────────────────────────────────────────────────────────────────────────
 * Benchmark
 * ───────────────────────────────────────────────────────────────────────── */

struct BenchReport {
    double  phase_ms[4];         /* Load, Algo, Write, Total */
    size_t  peak_kb;
    double  user_ms, sys_ms;
    size_t  result_ram, result_disk;
    size_t  num_patterns, total_items;
    double  throughput_mb_s;
};

inline void        bench_reset()  { dm_bench_reset(); }
inline void        bench_start(DM_BenchPhase p) { dm_bench_start(p); }
inline void        bench_stop (DM_BenchPhase p) { dm_bench_stop(p);  }
inline void        bench_record(size_t n, size_t t) { dm_bench_record(n, t); }
inline BenchReport bench_report() {
    DM_BenchReport r = dm_bench_get_report();
    BenchReport out{};
    for (int i = 0; i < 4; ++i) out.phase_ms[i] = r.phase_times_ms[i];
    out.peak_kb      = r.peak_memory_kb;
    out.user_ms      = r.user_cpu_ms;
    out.sys_ms       = r.sys_cpu_ms;
    out.result_ram   = r.result_ram_bytes;
    out.result_disk  = r.result_disk_est_bytes;
    out.num_patterns = r.num_patterns;
    out.total_items  = r.total_items;
    out.throughput_mb_s = r.throughput_mb_s;
    return out;
}
inline void bench_print(std::string_view algo, std::string_view dataset) {
    dm_bench_print(std::string(algo).c_str(), std::string(dataset).c_str());
}

/* ─────────────────────────────────────────────────────────────────────────
 * BitSet
 * ───────────────────────────────────────────────────────────────────────── */

class BitSet {
public:
    explicit BitSet(size_t n_bits) {
        handle_ = dm_bitset_create(n_bits);
        if (!handle_) throw std::bad_alloc{};
    }
    /** Copy constructor. */
    BitSet(const BitSet &o) {
        handle_ = dm_bitset_copy(o.handle_);
        if (!handle_) throw std::bad_alloc{};
    }
    BitSet &operator=(const BitSet &) = delete;
    BitSet(BitSet &&o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }
    ~BitSet() { if (handle_) dm_bitset_free(handle_); }

    void   set    (size_t pos)             { dm_bitset_set    (handle_, pos); }
    void   clear  (size_t pos)             { dm_bitset_clear  (handle_, pos); }
    bool   get    (size_t pos)       const { return dm_bitset_get(handle_, pos); }
    void   set_all()                       { dm_bitset_set_all(handle_); }
    void   flip   ()                       { dm_bitset_not    (handle_); }
    size_t popcount()                const { return dm_bitset_popcount(handle_); }

    BitSet &operator&=(const BitSet &o) { dm_bitset_and(handle_, o.handle_); return *this; }
    BitSet &operator|=(const BitSet &o) { dm_bitset_or (handle_, o.handle_); return *this; }

    DM_BitSet raw() const { return handle_; }

private:
    DM_BitSet handle_{nullptr};
};

/* ─────────────────────────────────────────────────────────────────────────
 * DataGen
 * ───────────────────────────────────────────────────────────────────────── */

class DataGen {
public:
    /** type: "medm" | "textbook" */
    explicit DataGen(std::string_view type) {
        handle_ = dm_datagen_create(std::string(type).c_str());
        if (!handle_) throw std::runtime_error("dm::DataGen: unknown type: " +
                                                std::string(type));
    }
    ~DataGen() { if (handle_) dm_datagen_free(handle_); }

    DataGen(const DataGen &)             = delete;
    DataGen &operator=(const DataGen &)  = delete;
    DataGen(DataGen &&o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }

    void run(std::string_view spec, std::string_view output_path, unsigned seed = 0) {
        check(dm_datagen_run(handle_,
                             std::string(spec).c_str(),
                             std::string(output_path).c_str(),
                             seed),
              "dm::DataGen::run");
    }

    DM_DataGen raw() const { return handle_; }

private:
    DM_DataGen handle_{nullptr};
};

/* ─────────────────────────────────────────────────────────────────────────
 * GpuCtx
 * ───────────────────────────────────────────────────────────────────────── */

class GpuCtx {
public:
    /**
     * @param device_index  0 = first discrete GPU, -1 = driver pick.
     * @param shader_dir    Path to compiled .spv shaders (nullptr = auto-search).
     */
    explicit GpuCtx(int device_index = 0, const char *shader_dir = nullptr) {
        handle_ = dm_gpu_create(device_index, shader_dir);
        /* Soft failure: handle_ == nullptr means no GPU; check ready(). */
    }
    ~GpuCtx() { if (handle_) dm_gpu_free(handle_); }

    GpuCtx(const GpuCtx &)             = delete;
    GpuCtx &operator=(const GpuCtx &)  = delete;
    GpuCtx(GpuCtx &&o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }

    bool ready() const { return handle_ && dm_gpu_ready(handle_) != 0; }

    std::string device_name() const {
        if (!handle_) return "(no GPU)";
        char buf[256]{};
        dm_gpu_device_name(handle_, buf, sizeof(buf));
        return buf;
    }

    DM_GpuCtx raw() const { return handle_; }

private:
    DM_GpuCtx handle_{nullptr};
};

/* ─────────────────────────────────────────────────────────────────────────
 * Arena
 * ───────────────────────────────────────────────────────────────────────── */

class Arena {
public:
    explicit Arena(size_t capacity) {
        if (dm_arena_init(&arena_, capacity) != 0)
            throw std::bad_alloc{};
    }
    ~Arena() { dm_arena_free(&arena_); }

    Arena(const Arena &)             = delete;
    Arena &operator=(const Arena &)  = delete;

    void  reset()                                 { dm_arena_reset(&arena_); }
    void *alloc(size_t sz, size_t align = sizeof(void *)) {
        return dm_arena_alloc(&arena_, sz, align);
    }
    char *strndup(const char *s, size_t n)  { return dm_arena_strndup(&arena_, s, n); }

    DM_Arena       &raw()       { return arena_; }
    const DM_Arena &raw() const { return arena_; }

private:
    DM_Arena arena_{};
};

} /* namespace dm */
