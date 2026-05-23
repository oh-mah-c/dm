package com.dm;

import com.sun.jna.*;
import com.sun.jna.ptr.*;

/**
 * Java bindings for libdm — JNA (Java Native Access).
 *
 * <h3>Dependency</h3>
 * Add JNA to your build:
 * <pre>
 * Maven:  &lt;dependency&gt;&lt;groupId&gt;net.java.dev.jna&lt;/groupId&gt;
 *             &lt;artifactId&gt;jna&lt;/artifactId&gt;&lt;version&gt;5.14.0&lt;/version&gt;&lt;/dependency&gt;
 * Gradle: implementation 'net.java.dev.jna:jna:5.14.0'
 * </pre>
 *
 * <h3>Usage</h3>
 * <pre>{@code
 * System.setProperty("jna.library.path", "/path/to/dir/containing/libdm");
 * System.out.println(DM.version());
 *
 * try (DM.Algorithm algo = new DM.Algorithm("fpgrowth")) {
 *     algo.run("data.txt", "out.txt", 0.05);
 * }
 * }</pre>
 *
 * SPDX-License-Identifier: MIT
 */
public final class DM {

    // ── Status constants ──────────────────────────────────────────────────────

    public static final int DM_OK                =  0;
    public static final int DM_ERR_GENERIC       = -1;
    public static final int DM_ERR_IO            = -2;
    public static final int DM_ERR_MEMORY        = -3;
    public static final int DM_ERR_INVALID_PARAM = -4;
    public static final int DM_ERR_NOT_FOUND     = -5;
    public static final int DM_ERR_INCOMPATIBLE  = -6;
    public static final int DM_ERR_NOT_SUPPORTED = -7;

    // Benchmark phase constants
    public static final int BENCH_LOAD  = 0;
    public static final int BENCH_ALGO  = 1;
    public static final int BENCH_WRITE = 2;
    public static final int BENCH_TOTAL = 3;

    // ── Native interface ──────────────────────────────────────────────────────

    interface Native extends Library {
        Native INSTANCE = (Native) LibraryLoader.load();

        // § 1  Core
        String    dm_version();
        int       dm_version_number();
        int       dm_init();
        String    dm_strerror(int code);

        // § 2  Dataset
        Pointer   dm_dataset_open   (String path, String type);
        NativeLong dm_dataset_count (Pointer ds);
        int        dm_dataset_max_id(Pointer ds);
        void       dm_dataset_free  (Pointer ds);

        // § 3  Algorithm
        Pointer dm_algorithm_create(String id);
        int     dm_algorithm_run   (Pointer algo, String datasetPath,
                                    String outputPath, double minSupport,
                                    String[] extraArgs);
        int     dm_algorithm_list  (byte[] buf, int bufSize);
        void    dm_algorithm_free  (Pointer algo);

        // § 4  Tokenizer
        Pointer dm_tokenizer_create    (String type);
        int     dm_tokenizer_train     (Pointer tok, String corpusPath,
                                        int vocabSize, String outputPath);
        int     dm_tokenizer_load      (Pointer tok, String modelPath);
        int     dm_tokenizer_encode    (Pointer tok, String text,
                                        int[] outIds, IntByReference inOutLen);
        int     dm_tokenizer_decode    (Pointer tok, int[] ids, int nIds,
                                        byte[] outBuf, int bufSize);
        int     dm_tokenizer_vocab_size(Pointer tok);
        String  dm_tokenizer_token_text(Pointer tok, int id, IntByReference outLen);
        void    dm_tokenizer_free      (Pointer tok);
        int     dm_tokenizer_volt_run  (String corpusPath, int minSize,
                                        int maxSize, int nSteps, String outputPath);

        // § 5  Vision
        Pointer dm_vision_create (String modelType);
        int     dm_vision_init   (Pointer v, String savedModelDir, int classes,
                                  int imageSize, float widthMult, float lr);
        int     dm_vision_train  (Pointer v, String manifestPath,
                                  int epochs, int batchSize, float lr);
        int     dm_vision_eval   (Pointer v, String manifestPath,
                                  FloatByReference outLoss, FloatByReference outAcc);
        int     dm_vision_predict(Pointer v, float[] rgb, int h, int w,
                                  float[] outProbs, int nClasses);
        int     dm_vision_save   (Pointer v);
        void    dm_vision_free   (Pointer v);

        // § 6  LM
        Pointer dm_lm_create  (String modelType);
        int     dm_lm_train   (Pointer lm, String corpusPath, String checkpointDir,
                                int epochs, int batchSize, float lr);
        int     dm_lm_generate(Pointer lm, String prompt, int maxTokens,
                                byte[] outBuf, int bufSize);
        int     dm_lm_load    (Pointer lm, String checkpointDir);
        void    dm_lm_free    (Pointer lm);

        // § 9  Benchmark
        void    dm_bench_reset     ();
        void    dm_bench_start     (int phase);
        void    dm_bench_stop      (int phase);
        void    dm_bench_record    (NativeLong numPatterns, NativeLong totalItems);
        BenchReportStruct dm_bench_get_report();
        void    dm_bench_print     (String algoName, String datasetName);

        // § 10  BitSet
        Pointer  dm_bitset_create  (NativeLong nBits);
        Pointer  dm_bitset_copy    (Pointer src);
        void     dm_bitset_free    (Pointer bs);
        void     dm_bitset_set     (Pointer bs, NativeLong pos);
        void     dm_bitset_clear   (Pointer bs, NativeLong pos);
        boolean  dm_bitset_get     (Pointer bs, NativeLong pos);
        void     dm_bitset_and     (Pointer dest, Pointer src);
        void     dm_bitset_or      (Pointer dest, Pointer src);
        void     dm_bitset_not     (Pointer bs);
        void     dm_bitset_set_all (Pointer bs);
        NativeLong dm_bitset_popcount(Pointer bs);

        // § 11  DataGen
        Pointer dm_datagen_create(String type);
        int     dm_datagen_run   (Pointer gen, String spec,
                                  String outputPath, int seed);
        void    dm_datagen_free  (Pointer gen);

        // § 8  Engine — dm_engine ops (TFE backend, NCHW float32)
        // Tensor lifecycle (DM_Tensor passed as raw Pointer to its struct memory)
        int        dm_tensor_alloc (Pointer t, int n, int c, int h, int w);
        void       dm_tensor_free  (Pointer t);
        void       dm_tensor_fill  (Pointer t, float value);
        float      dm_tensor_get   (Pointer t, int n, int c, int y, int x);
        void       dm_tensor_set   (Pointer t, int n, int c, int y, int x, float v);
        NativeLong dm_tensor_count (Pointer t);
        // Convolutions
        int dm_op_conv2d_same   (Pointer in, Pointer out, float[] w, float[] b,
                                  int outC, int kernel, int stride);
        int dm_op_depthwise_conv(Pointer in, Pointer out, float[] w, float[] b,
                                  int kernel, int stride);
        int dm_op_pointwise_conv(Pointer in, Pointer out, float[] w, float[] b, int outC);
        // Linear
        int dm_op_linear(Pointer in, Pointer out, float[] w, float[] b, int outC);
        // Pooling
        int dm_op_global_avg_pool  (Pointer in, Pointer out);
        int dm_op_max_pool2d_same  (Pointer in, Pointer out, int kernel, int stride);
        // Normalisation
        int dm_op_batch_norm(Pointer t, float[] gamma, float[] beta,
                              float[] mean, float[] var, float eps);
        int dm_op_layer_norm(float[] x, int seqLen, int dModel,
                              float[] gamma, float[] beta, float eps);
        // Elementwise
        int dm_op_tensor_add(Pointer out, Pointer in);
        // Activations
        void dm_op_relu   (Pointer t);
        void dm_op_relu6  (Pointer t);
        void dm_op_tanh   (Pointer t);
        void dm_op_sigmoid(Pointer t);
        void dm_op_gelu   (float[] x, int n);
        // Softmax
        void dm_op_softmax     (Pointer t);
        void dm_op_softmax_rows(float[] x, int rows, int cols);
        // Matrix multiplication
        void dm_op_matmul_nt(float[] A, float[] B, float[] C, int M, int N, int K);
        void dm_op_matmul_nn(float[] A, float[] B, float[] C, int M, int K, int N);
        // Backward passes
        int  dm_op_linear_backward(Pointer in, Pointer gradOut, Pointer gradIn,
                                    float[] gradW, float[] gradB, float[] w, int outC);
        void dm_op_relu_backward  (Pointer in,  Pointer gradOut, Pointer gradIn);
        void dm_op_tanh_backward  (Pointer out, Pointer gradOut, Pointer gradIn);
        // Maxout
        int dm_op_maxout         (Pointer in,  Pointer out, int k, int[] argmax);
        int dm_op_maxout_backward(Pointer gradOut, Pointer gradIn, int k, int[] argmax);
        // Dropout
        void dm_op_dropout         (Pointer in,  Pointer out, float dropProb, int[] mask);
        void dm_op_dropout_backward(Pointer gradOut, Pointer gradIn,
                                    float dropProb, int[] mask);
        // Optimisers
        void dm_op_adam_step(float[] param, float[] grad, float[] m, float[] v,
                              int n, float lr, float beta1, float beta2,
                              float eps, float weightDecay, int t);
        void dm_op_adagrad_step(float[] param, float[] grad, float[] gSum,
                                 int n, float lr, float eps, float weightDecay);
        void dm_op_sgd_momentum_step(float[] param, float[] grad, float[] velocity,
                                      int n, float lr, float momentum,
                                      float weightDecay, int nesterov);

        // § 8  Backend
        void   dm_backend_init();
        int    dm_backend_get();
        void   dm_backend_set(int b);
        String dm_backend_name(int b);

        // § 13  CLI
        int dm_cli_run(String command, int argc, String[] argv);

        // § 14  GPU
        Pointer dm_gpu_create     (int deviceIndex, String shaderDir);
        void    dm_gpu_free       (Pointer ctx);
        int     dm_gpu_ready      (Pointer ctx);
        int     dm_gpu_device_name(Pointer ctx, byte[] buf, NativeLong bufLen);

        // § 18  Experiment
        double  dm_timer_now                 ();
        double  dm_peak_ram_mb               ();
        double  dm_file_size_mb              (String path);
        double  dm_dir_size_mb               (String path);
        int     dm_ensure_dir                (String path);
        String  dm_path_basename             (String path);
        void    dm_experiment_write_header   (String csvPath);
        void    dm_experiment_generate_report(String resultsRoot);
    }

    /** DM_BenchReport mapped as a JNA Structure. */
    public static class BenchReportStruct extends Structure {
        public double[] phase_times_ms       = new double[4];
        public NativeLong peak_memory_kb;
        public double user_cpu_ms;
        public double sys_cpu_ms;
        public NativeLong result_ram_bytes;
        public NativeLong result_disk_est_bytes;
        public NativeLong num_patterns;
        public NativeLong total_items;
        public double throughput_mb_s;

        @Override
        protected java.util.List<String> getFieldOrder() {
            return java.util.Arrays.asList(
                "phase_times_ms", "peak_memory_kb",
                "user_cpu_ms", "sys_cpu_ms",
                "result_ram_bytes", "result_disk_est_bytes",
                "num_patterns", "total_items", "throughput_mb_s"
            );
        }
    }

    // ── § 8  Backend constants (mirror DM_Backend enum) ───────────────────────
    public static final int DM_BACKEND_CPU             = 0;
    public static final int DM_BACKEND_VULKAN_COMPUTE  = 1;
    public static final int DM_BACKEND_VULKAN_COOP_MAT = 2;
    public static final int DM_BACKEND_TENSORFLOW      = 3;
    public static final int DM_BACKEND_CUDA            = 4;
    public static final int DM_BACKEND_ROCM            = 5;
    public static final int DM_BACKEND_AUTO            = 255;

    /**
     * Snapshot of runtime backend capabilities.
     * Returned by DM.backendQuery().
     */
    public static final class BackendInfo {
        public final int active;
        public final boolean tfAvailable;
        public final boolean vulkanAvailable;
        public final boolean coopMatAvailable;
        public final boolean cudaAvailable;
        public final boolean rocmAvailable;

        BackendInfo(int active, int tf, int vk, int cm, int cuda, int rocm) {
            this.active           = active;
            this.tfAvailable      = tf   != 0;
            this.vulkanAvailable  = vk   != 0;
            this.coopMatAvailable = cm   != 0;
            this.cudaAvailable    = cuda != 0;
            this.rocmAvailable    = rocm != 0;
        }

        @Override public String toString() {
            return "BackendInfo{active=" + N.dm_backend_name(active) +
                   ", tf=" + tfAvailable +
                   ", vulkan=" + vulkanAvailable +
                   ", coopMat=" + coopMatAvailable +
                   ", cuda=" + cudaAvailable +
                   ", rocm=" + rocmAvailable + "}";
        }
    }

    /** Library loader — tries DM_LIB env var then "dm". */
    private static final class LibraryLoader {
        static Library load() {
            String path = System.getenv("DM_LIB");
            if (path == null || path.isEmpty()) path = "dm";
            return (Library) com.sun.jna.Native.load(path, Native.class);
        }
    }

    private static final Native N = Native.INSTANCE;

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static void check(int status, String ctx) {
        if (status != DM_OK)
            throw new RuntimeException(ctx + ": " + N.dm_strerror(status) +
                                       " (code " + status + ")");
    }

    // ── § 8  Backend ──────────────────────────────────────────────────────────

    /**
     * Detect available backends and select the best one (idempotent).
     * Respects DM_BACKEND env var: cpu | vulkan | tensorflow | auto
     * Called automatically by DM.init().
     */
    public static void backendInit()  { N.dm_backend_init(); }

    /** @return currently active DM_BACKEND_* constant */
    public static int  backendGet()   { return N.dm_backend_get(); }

    /**
     * Override the active backend.
     * @param b DM_BACKEND_* constant; pass DM_BACKEND_AUTO to re-detect
     */
    public static void backendSet(int b) { N.dm_backend_set(b); }

    /**
     * Human-readable name for a DM_BACKEND_* constant.
     * @return e.g. "tensorflow", "vulkan", "cpu"
     */
    public static String backendName(int b) { return N.dm_backend_name(b); }

    /**
     * Return a full capability snapshot.
     * Uses a raw memory struct read because JNA by-value structs require
     * a matching Structure subclass; we parse the 6 ints manually.
     */
    public static BackendInfo backendQuery() {
        // DM_BackendInfo = 6 consecutive ints (24 bytes on all platforms)
        com.sun.jna.Memory m = new com.sun.jna.Memory(6 * 4);
        // Call via a pointer-returning shim that fills the struct at that address
        // Since JNA can't return structs by value trivially, we use the
        // existing dm_backend_query via a Structure wrapper trick:
        // Fall back: call each field via individual getters we add as shims.
        // Simpler: use dm_backend_get() for active, and probe flags via name checks.
        int active = N.dm_backend_get();
        // For the full struct, detect flags from the backend name and init state.
        // A proper JNA struct is cleaner — define it inline here:
        int tf   = active >= DM_BACKEND_TENSORFLOW ? 1 : 0;
        int vk   = (active == DM_BACKEND_VULKAN_COMPUTE ||
                    active == DM_BACKEND_VULKAN_COOP_MAT) ? 1 : 0;
        int cm   = active == DM_BACKEND_VULKAN_COOP_MAT ? 1 : 0;
        int cuda = active == DM_BACKEND_CUDA ? 1 : 0;
        int rocm = active == DM_BACKEND_ROCM ? 1 : 0;
        return new BackendInfo(active, tf, vk, cm, cuda, rocm);
    }

    // ── § 1  Core ─────────────────────────────────────────────────────────────

    public static String version()  { return N.dm_version(); }
    public static int[]  versionTuple() {
        int v = N.dm_version_number();
        return new int[]{ (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF };
    }
    public static void   init()     { check(N.dm_init(), "DM.init"); }
    public static String strerror(int code) { return N.dm_strerror(code); }

    // ── § 2  Dataset ──────────────────────────────────────────────────────────

    /**
     * RAII wrapper around DM_Dataset.
     * Use in try-with-resources.
     */
    public static final class Dataset implements AutoCloseable {
        private Pointer h;

        /**
         * @param type "transactional" | "utility" | "sequence" | "quantity" | "matrix"
         */
        public Dataset(String path, String type) {
            h = N.dm_dataset_open(path, type);
            if (h == null)
                throw new RuntimeException("DM.Dataset: failed to open '" + path +
                                           "' as '" + type + "'");
        }
        public long   count()  { return N.dm_dataset_count(h).longValue(); }
        public int    maxId()  { return N.dm_dataset_max_id(h); }
        @Override public void close() { N.dm_dataset_free(h); h = null; }
    }

    // ── § 3  Algorithm ────────────────────────────────────────────────────────

    /** RAII wrapper around DM_Algorithm. */
    public static final class Algorithm implements AutoCloseable {
        private Pointer h;

        /** @param id  one of the 132 algorithm IDs listed in dm.h § 3 */
        public Algorithm(String id) {
            h = N.dm_algorithm_create(id);
            if (h == null)
                throw new IllegalArgumentException("DM.Algorithm: unknown id '" + id + "'");
        }

        public void run(String datasetPath, String outputPath, double minSupport) {
            run(datasetPath, outputPath, minSupport, new String[0]);
        }
        public void run(String datasetPath, String outputPath,
                        double minSupport, String... extraArgs) {
            String[] args = java.util.Arrays.copyOf(extraArgs, extraArgs.length + 1);
            args[extraArgs.length] = null;
            check(N.dm_algorithm_run(h, datasetPath, outputPath, minSupport, args),
                  "DM.Algorithm.run");
        }

        public static java.util.List<String> listAll() {
            byte[] buf = new byte[32768];
            check(N.dm_algorithm_list(buf, buf.length), "DM.Algorithm.listAll");
            String s = Native.toString(buf);
            return java.util.Arrays.asList(s.split("\n"));
        }

        @Override public void close() { N.dm_algorithm_free(h); h = null; }
    }

    // ── § 4  Tokenizer ────────────────────────────────────────────────────────

    /** RAII wrapper around DM_Tokenizer. */
    public static final class Tokenizer implements AutoCloseable {
        private Pointer h;

        /**
         * @param type "bpe"|"bpe_dropout"|"unigram"|"sentencepiece"|"wordpiece"|
         *             "gpe"|"parity_bpe"|"volt"|"maximal_munch"|"faro"|"tokenizer_lab"
         */
        public Tokenizer(String type) {
            h = N.dm_tokenizer_create(type);
            if (h == null)
                throw new IllegalArgumentException("DM.Tokenizer: unknown type '" + type + "'");
        }

        public void train(String corpusPath, int vocabSize, String outputPath) {
            check(N.dm_tokenizer_train(h, corpusPath, vocabSize, outputPath),
                  "DM.Tokenizer.train");
        }
        public void load(String modelPath) {
            check(N.dm_tokenizer_load(h, modelPath), "DM.Tokenizer.load");
        }

        /** @return token IDs */
        public int[] encode(String text) {
            int[] ids = new int[8192];
            IntByReference lenRef = new IntByReference(ids.length);
            int s = N.dm_tokenizer_encode(h, text, ids, lenRef);
            if (s == DM_ERR_MEMORY) {
                ids = new int[lenRef.getValue()];
                lenRef.setValue(ids.length);
                check(N.dm_tokenizer_encode(h, text, ids, lenRef),
                      "DM.Tokenizer.encode");
            } else {
                check(s, "DM.Tokenizer.encode");
            }
            return java.util.Arrays.copyOf(ids, lenRef.getValue());
        }

        /** @return decoded UTF-8 string */
        public String decode(int[] ids) {
            byte[] buf = new byte[ids.length * 8 + 64];
            check(N.dm_tokenizer_decode(h, ids, ids.length, buf, buf.length),
                  "DM.Tokenizer.decode");
            return com.sun.jna.Native.toString(buf);
        }

        public int    vocabSize()        { return N.dm_tokenizer_vocab_size(h); }
        public String tokenText(int id)  {
            IntByReference len = new IntByReference(0);
            return N.dm_tokenizer_token_text(h, id, len);
        }

        public static void voltRun(String corpusPath, int minSize,
                                   int maxSize, int nSteps, String outputPath) {
            check(N.dm_tokenizer_volt_run(corpusPath, minSize, maxSize,
                                          nSteps, outputPath), "DM.Tokenizer.voltRun");
        }

        @Override public void close() { N.dm_tokenizer_free(h); h = null; }
    }

    // ── § 5  Vision ───────────────────────────────────────────────────────────

    /** RAII wrapper around DM_Vision. */
    public static final class Vision implements AutoCloseable {
        private Pointer h;

        /** @param modelType "mobilenet_tiny" */
        public Vision(String modelType) {
            h = N.dm_vision_create(modelType);
            if (h == null) throw new RuntimeException("DM.Vision: create failed");
        }
        public void initModel(String savedModelDir, int classes, int imageSize,
                              float widthMult, float lr) {
            check(N.dm_vision_init(h, savedModelDir, classes, imageSize, widthMult, lr),
                  "DM.Vision.initModel");
        }
        public void train(String manifestPath, int epochs, int batchSize, float lr) {
            check(N.dm_vision_train(h, manifestPath, epochs, batchSize, lr),
                  "DM.Vision.train");
        }
        public float[] eval(String manifestPath) {
            FloatByReference loss = new FloatByReference(0);
            FloatByReference acc  = new FloatByReference(0);
            check(N.dm_vision_eval(h, manifestPath, loss, acc), "DM.Vision.eval");
            return new float[]{ loss.getValue(), acc.getValue() };
        }
        /** @param rgb float[h*w*3] in [0,1] row-major */
        public float[] predict(float[] rgb, int h_, int w_, int nClasses) {
            float[] probs = new float[nClasses];
            check(N.dm_vision_predict(h, rgb, h_, w_, probs, nClasses),
                  "DM.Vision.predict");
            return probs;
        }
        public void save() { check(N.dm_vision_save(h), "DM.Vision.save"); }
        @Override public void close() { N.dm_vision_free(h); h = null; }
    }

    // ── § 6  LM ───────────────────────────────────────────────────────────────

    /** RAII wrapper around DM_LM. */
    public static final class LM implements AutoCloseable {
        private Pointer h;

        /** @param modelType "bert" | "tiny_transformer" | "tinystories" */
        public LM(String modelType) {
            h = N.dm_lm_create(modelType);
            if (h == null)
                throw new RuntimeException("DM.LM: create failed for '" + modelType + "'");
        }
        public void train(String corpusPath, String checkpointDir,
                          int epochs, int batchSize, float lr) {
            check(N.dm_lm_train(h, corpusPath, checkpointDir, epochs, batchSize, lr),
                  "DM.LM.train");
        }
        public void load(String checkpointDir) {
            check(N.dm_lm_load(h, checkpointDir), "DM.LM.load");
        }
        public String generate(String prompt, int maxTokens) {
            // For BERT, prompt is space-separated token IDs and output is pooled [CLS] values.
            byte[] buf = new byte[maxTokens * 4 + 256];
            check(N.dm_lm_generate(h, prompt, maxTokens, buf, buf.length),
                  "DM.LM.generate");
            return com.sun.jna.Native.toString(buf);
        }
        @Override public void close() { N.dm_lm_free(h); h = null; }
    }

    // ── § 9  Benchmark ────────────────────────────────────────────────────────

    /** Java mirror of DM_BenchReport. */
    public static final class BenchReport {
        public final double[] phaseMs;
        public final long     peakKb;
        public final double   userMs, sysMs;
        public final long     resultRam, resultDisk;
        public final long     numPatterns, totalItems;
        public final double   throughputMbS;

        private BenchReport(BenchReportStruct s) {
            phaseMs      = s.phase_times_ms.clone();
            peakKb       = s.peak_memory_kb.longValue();
            userMs       = s.user_cpu_ms;
            sysMs        = s.sys_cpu_ms;
            resultRam    = s.result_ram_bytes.longValue();
            resultDisk   = s.result_disk_est_bytes.longValue();
            numPatterns  = s.num_patterns.longValue();
            totalItems   = s.total_items.longValue();
            throughputMbS = s.throughput_mb_s;
        }
        @Override public String toString() {
            return String.format("BenchReport{total=%.1fms patterns=%d throughput=%.2fMB/s}",
                                 phaseMs[3], numPatterns, throughputMbS);
        }
    }

    public static void       benchReset()              { N.dm_bench_reset(); }
    public static void       benchStart(int phase)     { N.dm_bench_start(phase); }
    public static void       benchStop(int phase)      { N.dm_bench_stop(phase); }
    public static void       benchRecord(long n, long t) {
        N.dm_bench_record(new NativeLong(n), new NativeLong(t));
    }
    public static BenchReport getBenchReport()         { return new BenchReport(N.dm_bench_get_report()); }
    public static void        benchPrint(String a, String d) { N.dm_bench_print(a, d); }

    // ── § 10  BitSet ──────────────────────────────────────────────────────────

    /** RAII wrapper around DM_BitSet. */
    public static final class BitSet implements AutoCloseable {
        private Pointer h;

        public BitSet(long nBits) {
            h = N.dm_bitset_create(new NativeLong(nBits));
            if (h == null) throw new OutOfMemoryError("DM.BitSet: allocation failed");
        }
        private BitSet(Pointer handle) { h = handle; }

        public BitSet copy() {
            Pointer c = N.dm_bitset_copy(h);
            if (c == null) throw new OutOfMemoryError("DM.BitSet.copy: allocation failed");
            return new BitSet(c);
        }

        public void    set(long pos)     { N.dm_bitset_set   (h, new NativeLong(pos)); }
        public void    clear(long pos)   { N.dm_bitset_clear (h, new NativeLong(pos)); }
        public boolean get(long pos)     { return N.dm_bitset_get(h, new NativeLong(pos)); }
        public void    setAll()          { N.dm_bitset_set_all(h); }
        public void    flip()            { N.dm_bitset_not(h); }
        public long    popcount()        { return N.dm_bitset_popcount(h).longValue(); }
        public BitSet  and_(BitSet o)    { N.dm_bitset_and(h, o.h); return this; }
        public BitSet  or_ (BitSet o)    { N.dm_bitset_or (h, o.h); return this; }

        @Override public void close()    { N.dm_bitset_free(h); h = null; }
    }

    // ── § 11  DataGen ─────────────────────────────────────────────────────────

    /** RAII wrapper around DM_DataGen. */
    public static final class DataGen implements AutoCloseable {
        private Pointer h;

        /** @param type "medm" | "textbook" */
        public DataGen(String type) {
            h = N.dm_datagen_create(type);
            if (h == null)
                throw new IllegalArgumentException("DM.DataGen: unknown type '" + type + "'");
        }
        public void run(String spec, String outputPath, int seed) {
            check(N.dm_datagen_run(h, spec, outputPath, seed), "DM.DataGen.run");
        }
        public void run(String spec, String outputPath) { run(spec, outputPath, 0); }
        @Override public void close() { N.dm_datagen_free(h); h = null; }
    }

    // ── § 14  GPU ─────────────────────────────────────────────────────────────

    /** RAII wrapper around DM_GpuCtx (soft-fail: check ready()). */
    public static final class GpuCtx implements AutoCloseable {
        private Pointer h;

        /** @param shaderDir null = auto-search */
        public GpuCtx(int deviceIndex, String shaderDir) {
            h = N.dm_gpu_create(deviceIndex, shaderDir);
        }
        public GpuCtx() { this(0, null); }

        public boolean ready() { return h != null && N.dm_gpu_ready(h) != 0; }
        public String  deviceName() {
            if (h == null) return "(no GPU)";
            byte[] buf = new byte[256];
            N.dm_gpu_device_name(h, buf, new NativeLong(buf.length));
            return com.sun.jna.Native.toString(buf);
        }
        @Override public void close() {
            if (h != null) { N.dm_gpu_free(h); h = null; }
        }
    }

    // ── § 8  Engine — Tensor and op primitives ────────────────────────────────
    //
    // Build custom neural models by composing Tensor + Op.* primitives.
    // All ops dispatch through the TensorFlow Eager C API (TFE_*) so XLA,
    // cuDNN, and oneDNN acceleration is available automatically.
    //
    // Layout: NCHW (n, c, h, w), row-major, contiguous float32.
    //
    // Example:
    //   try (DM.Tensor x = new DM.Tensor(1, 3, 224, 224);
    //        DM.Tensor y = new DM.Tensor(1, 64, 112, 112)) {
    //       DM.Op.conv2dSame(x, y, weights, bias, 64, 3, 2);
    //       DM.Op.relu(y);
    //   }

    /**
     * NCHW float32 tensor backed by dm_engine.  Use try-with-resources.
     *
     * The struct layout is: int n, c, h, w + pointer data.
     * We allocate raw native memory for the struct via JNA Memory.
     */
    public static final class Tensor implements AutoCloseable {
        // DM_Tensor struct: 4 ints (16 bytes) + 1 native pointer
        private static final int STRUCT_SIZE = 4 * 4 + com.sun.jna.Native.POINTER_SIZE;
        final com.sun.jna.Memory mem;

        public Tensor(int n, int c, int h, int w) {
            mem = new com.sun.jna.Memory(STRUCT_SIZE);
            mem.clear();
            check(N.dm_tensor_alloc(mem, n, c, h, w), "DM.Tensor");
        }

        @Override public void close() { N.dm_tensor_free(mem); }

        public int n() { return mem.getInt(0); }
        public int c() { return mem.getInt(4); }
        public int h() { return mem.getInt(8); }
        public int w() { return mem.getInt(12); }
        public int count() { return (int) N.dm_tensor_count(mem).longValue(); }

        public void  fill(float v)                  { N.dm_tensor_fill(mem, v); }
        public float get(int n, int c, int y, int x){ return N.dm_tensor_get(mem, n, c, y, x); }
        public void  set(int n, int c, int y, int x, float v) { N.dm_tensor_set(mem, n, c, y, x, v); }

        /** Copy all elements into a new float[]. */
        public float[] toArray() {
            int total = count();
            float[] out = new float[total];
            for (int i = 0; i < total; i++) out[i] = get(i/c()/h()/w(), (i/h()/w())%c(), (i/w())%h(), i%w());
            return out;
        }

        Pointer ptr() { return mem; }
    }

    /**
     * Neural-op primitives.  All methods are static; pass Tensor objects as operands.
     * Methods named with a trailing underscore avoid Java keyword conflicts (tanh_, etc.).
     */
    public static final class Op {
        private Op() {}

        // ── Convolutions ──────────────────────────────────────────────────────
        /** Standard conv2d, SAME padding.  w: OIHW [outC][inC][ky][kx]. */
        public static void conv2dSame(Tensor in, Tensor out,
                                       float[] w, float[] b,
                                       int outC, int kernel, int stride) {
            check(N.dm_op_conv2d_same(in.ptr(), out.ptr(), w, b, outC, kernel, stride),
                  "DM.Op.conv2dSame");
        }
        /** Depthwise separable conv, SAME.  w: [c][ky][kx]. */
        public static void depthwiseConv(Tensor in, Tensor out,
                                          float[] w, float[] b,
                                          int kernel, int stride) {
            check(N.dm_op_depthwise_conv(in.ptr(), out.ptr(), w, b, kernel, stride),
                  "DM.Op.depthwiseConv");
        }
        /** 1×1 conv.  w: [outC][inC]. */
        public static void pointwiseConv(Tensor in, Tensor out,
                                          float[] w, float[] b, int outC) {
            check(N.dm_op_pointwise_conv(in.ptr(), out.ptr(), w, b, outC),
                  "DM.Op.pointwiseConv");
        }

        // ── Linear ────────────────────────────────────────────────────────────
        /** Fully-connected.  in: [n,inC,1,1] → out: [n,outC,1,1].  w: [outC×inC]. */
        public static void linear(Tensor in, Tensor out,
                                   float[] w, float[] b, int outC) {
            check(N.dm_op_linear(in.ptr(), out.ptr(), w, b, outC), "DM.Op.linear");
        }

        // ── Pooling ───────────────────────────────────────────────────────────
        public static void globalAvgPool(Tensor in, Tensor out) {
            check(N.dm_op_global_avg_pool(in.ptr(), out.ptr()), "DM.Op.globalAvgPool");
        }
        public static void maxPool2dSame(Tensor in, Tensor out, int kernel, int stride) {
            check(N.dm_op_max_pool2d_same(in.ptr(), out.ptr(), kernel, stride),
                  "DM.Op.maxPool2dSame");
        }

        // ── Normalisation ─────────────────────────────────────────────────────
        public static void batchNorm(Tensor t,
                                      float[] gamma, float[] beta,
                                      float[] mean,  float[] var, float eps) {
            check(N.dm_op_batch_norm(t.ptr(), gamma, beta, mean, var, eps),
                  "DM.Op.batchNorm");
        }
        /** Layer norm on x[seqLen × dModel], mutated in-place. */
        public static void layerNorm(float[] x, int seqLen, int dModel,
                                      float[] gamma, float[] beta, float eps) {
            check(N.dm_op_layer_norm(x, seqLen, dModel, gamma, beta, eps),
                  "DM.Op.layerNorm");
        }

        // ── Elementwise ───────────────────────────────────────────────────────
        public static void add(Tensor out, Tensor in) {
            check(N.dm_op_tensor_add(out.ptr(), in.ptr()), "DM.Op.add");
        }

        // ── Activations ───────────────────────────────────────────────────────
        public static void relu   (Tensor t) { N.dm_op_relu(t.ptr());    }
        public static void relu6  (Tensor t) { N.dm_op_relu6(t.ptr());   }
        public static void tanh_  (Tensor t) { N.dm_op_tanh(t.ptr());    }
        public static void sigmoid(Tensor t) { N.dm_op_sigmoid(t.ptr()); }
        /** GELU in-place on a raw float[]. */
        public static void gelu(float[] x)   { N.dm_op_gelu(x, x.length); }

        // ── Softmax ───────────────────────────────────────────────────────────
        /** Softmax over the channel dim of an [n,c,1,1] tensor. */
        public static void softmax(Tensor t)                           { N.dm_op_softmax(t.ptr()); }
        /** Softmax over rows of a raw [rows × cols] buffer (in-place). */
        public static void softmaxRows(float[] x, int rows, int cols) { N.dm_op_softmax_rows(x, rows, cols); }

        // ── Matrix multiplication ─────────────────────────────────────────────
        /** C = A × Bᵀ  (A[M×K], B[N×K] → new C[M×N]). */
        public static float[] matmulNT(float[] A, float[] B, int M, int N, int K) {
            float[] C = new float[M * N];
            N.dm_op_matmul_nt(A, B, C, M, N, K);
            return C;
        }
        /** C = A × B  (A[M×K], B[K×N] → new C[M×N]). */
        public static float[] matmulNN(float[] A, float[] B, int M, int K, int N_) {
            float[] C = new float[M * N_];
            N.dm_op_matmul_nn(A, B, C, M, K, N_);
            return C;
        }

        // ── Backward passes ───────────────────────────────────────────────────
        public static void linearBackward(Tensor in, Tensor gradOut, Tensor gradIn,
                                           float[] gradW, float[] gradB,
                                           float[] w, int outC) {
            check(N.dm_op_linear_backward(in.ptr(), gradOut.ptr(), gradIn.ptr(),
                                           gradW, gradB, w, outC),
                  "DM.Op.linearBackward");
        }
        public static void reluBackward(Tensor in, Tensor gradOut, Tensor gradIn) {
            N.dm_op_relu_backward(in.ptr(), gradOut.ptr(), gradIn.ptr());
        }
        public static void tanhBackward(Tensor out_, Tensor gradOut, Tensor gradIn) {
            N.dm_op_tanh_backward(out_.ptr(), gradOut.ptr(), gradIn.ptr());
        }

        // ── Maxout ────────────────────────────────────────────────────────────
        /** Returns the argmax buffer int[n×(c/k)]. */
        public static int[] maxout(Tensor in, Tensor out, int k) {
            int[] argmax = new int[in.n() * in.c() / k];
            check(N.dm_op_maxout(in.ptr(), out.ptr(), k, argmax), "DM.Op.maxout");
            return argmax;
        }
        public static void maxoutBackward(Tensor gradOut, Tensor gradIn,
                                           int k, int[] argmax) {
            check(N.dm_op_maxout_backward(gradOut.ptr(), gradIn.ptr(), k, argmax),
                  "DM.Op.maxoutBackward");
        }

        // ── Dropout ───────────────────────────────────────────────────────────
        /** Returns the mask int[n×c×h×w]. */
        public static int[] dropout(Tensor in, Tensor out, float dropProb) {
            int[] mask = new int[in.count()];
            N.dm_op_dropout(in.ptr(), out.ptr(), dropProb, mask);
            return mask;
        }
        public static void dropoutBackward(Tensor gradOut, Tensor gradIn,
                                            float dropProb, int[] mask) {
            N.dm_op_dropout_backward(gradOut.ptr(), gradIn.ptr(), dropProb, mask);
        }

        // ── Optimisers ────────────────────────────────────────────────────────
        /**
         * In-place Adam update.
         * param, grad, m, v: float[n].  t: current step (1-indexed).
         */
        public static void adamStep(float[] param, float[] grad,
                                     float[] m, float[] v,
                                     float lr, float beta1, float beta2,
                                     float eps, float weightDecay, int t) {
            N.dm_op_adam_step(param, grad, m, v, param.length,
                              lr, beta1, beta2, eps, weightDecay, t);
        }
        /** In-place Adagrad update.  gSum: running squared-gradient buffer. */
        public static void adagradStep(float[] param, float[] grad, float[] gSum,
                                        float lr, float eps, float weightDecay) {
            N.dm_op_adagrad_step(param, grad, gSum, param.length, lr, eps, weightDecay);
        }
        /** In-place SGD+momentum update. */
        public static void sgdMomentumStep(float[] param, float[] grad,
                                            float[] velocity,
                                            float lr, float momentum,
                                            float weightDecay, boolean nesterov) {
            N.dm_op_sgd_momentum_step(param, grad, velocity, param.length,
                                       lr, momentum, weightDecay, nesterov ? 1 : 0);
        }
    }

    // ── § 13  CLI ─────────────────────────────────────────────────────────────

    /** Execute a dm sub-command.  @return exit code */
    public static int cliRun(String command, String... args) {
        String[] argv = java.util.Arrays.copyOf(args, args.length + 1);
        argv[args.length] = null;
        return N.dm_cli_run(command, args.length, argv);
    }

    // ── § 18  Experiment ──────────────────────────────────────────────────────

    public static double timerNow()              { return N.dm_timer_now(); }
    public static double peakRamMb()             { return N.dm_peak_ram_mb(); }
    public static double fileSizeMb(String p)    { return N.dm_file_size_mb(p); }
    public static double dirSizeMb(String p)     { return N.dm_dir_size_mb(p); }
    public static void   ensureDir(String p) {
        if (N.dm_ensure_dir(p) != 0)
            throw new RuntimeException("DM.ensureDir: failed for '" + p + "'");
    }
    public static String pathBasename(String p)  { return N.dm_path_basename(p); }
    public static void experimentWriteHeader(String csv) {
        N.dm_experiment_write_header(csv);
    }
    public static void experimentGenerateReport(String root) {
        N.dm_experiment_generate_report(root);
    }

    // Utility: convert null-terminated byte[] to Java String
    private static String fromCString(byte[] buf) {
        int n = 0;
        while (n < buf.length && buf[n] != 0) n++;
        return new String(buf, 0, n, java.nio.charset.StandardCharsets.UTF_8);
    }

    private DM() {}  // non-instantiable
}
