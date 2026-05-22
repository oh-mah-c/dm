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
