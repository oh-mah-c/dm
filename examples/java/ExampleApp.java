import com.dm.DM;

/**
 * examples/java/ExampleApp.java — Java usage example for libdm
 *
 * Demonstrates:
 *   - DM.version()
 *   - DM.Algorithm  (FP-Growth, EFIM)
 *   - DM.Tokenizer  (BPE train, encode, decode)
 *   - DM.BitSet
 *   - DM.DataGen
 *   - DM.GpuCtx
 *   - DM.benchReset/start/stop + getBenchReport
 *   - DM.LM
 *
 * Compile:
 *   javac -cp jna-5.14.0.jar:../../bindings/java \
 *         ExampleApp.java
 *
 * Run:
 *   java -Djna.library.path=../.. \
 *        -cp .:jna-5.14.0.jar:../../bindings/java \
 *        ExampleApp
 */
public class ExampleApp {

    private static final String DATASETS =
        java.nio.file.Paths.get(
            ExampleApp.class.getResource("").toURI()
        ).getParent().getParent().getParent()
         .resolve("datasets").toString();

    static { } // force static initializer

    public static void main(String[] args) throws Exception {

        DM.init();
        int[] v = DM.versionTuple();
        System.out.printf("libdm %s  (%d.%d.%d)%n",
                          DM.version(), v[0], v[1], v[2]);

        // ── Algorithm: FP-Growth ─────────────────────────────────────────────
        section("Algorithm — FP-Growth");
        try (DM.Algorithm algo = new DM.Algorithm("fpgrowth")) {
            algo.run(
                DATASETS + "/itemsets/mushrooms.txt",
                "/tmp/fpgrowth_java_out.txt",
                0.05,
                "min_length=2"
            );
            System.out.println("FP-Growth complete → /tmp/fpgrowth_java_out.txt");
        } catch (Exception e) {
            System.err.println("FP-Growth: " + e.getMessage());
        }

        // ── Algorithm: EFIM ──────────────────────────────────────────────────
        section("Algorithm — EFIM (high-utility)");
        try (DM.Algorithm efim = new DM.Algorithm("efim")) {
            efim.run(
                DATASETS + "/utilities/foodmart.txt",
                "/tmp/efim_java_out.txt",
                50.0
            );
            System.out.println("EFIM complete → /tmp/efim_java_out.txt");
        } catch (Exception e) {
            System.err.println("EFIM: " + e.getMessage());
        }

        // ── Algorithm list ───────────────────────────────────────────────────
        section("Algorithm registry (first 5)");
        var list = DM.Algorithm.listAll();
        list.stream().limit(5).forEach(id -> System.out.println("  " + id));
        System.out.printf("  … (%d total)%n", list.size());

        // ── Tokenizer ────────────────────────────────────────────────────────
        section("Tokenizer — BPE train + encode/decode");
        try (DM.Tokenizer tok = new DM.Tokenizer("bpe")) {
            tok.train(
                DATASETS + "/tokenizer/real_corpus.txt",
                2000,
                "/tmp/bpe_java_model"
            );
            System.out.printf("Trained BPE, vocab_size=%d%n", tok.vocabSize());

            String text = "frequent itemset mining";
            int[] tokenIds = tok.encode(text);
            System.out.printf("Encoded '%s' → %d tokens%n", text, tokenIds.length);

            String decoded = tok.decode(tokenIds);
            System.out.printf("Decoded: '%s'%n", decoded);

            // Show first 3 token texts
            for (int i = 0; i < Math.min(3, tokenIds.length); i++) {
                System.out.printf("  token[%d] = '%s'%n",
                                  tokenIds[i], tok.tokenText(tokenIds[i]));
            }
        } catch (Exception e) {
            System.err.println("Tokenizer: " + e.getMessage());
        }

        // ── BitSet ───────────────────────────────────────────────────────────
        section("BitSet operations");
        try (DM.BitSet a = new DM.BitSet(128);
             DM.BitSet b = new DM.BitSet(128)) {
            for (long pos : new long[]{1, 7, 42, 100}) a.set(pos);
            for (long pos : new long[]{7, 42, 99})      b.set(pos);
            a.and_(b); // {7, 42}
            System.out.printf("popcount after AND = %d (expected 2: {7,42})%n",
                              a.popcount());
        }

        // ── DataGen ──────────────────────────────────────────────────────────
        section("DataGen — MEDM synthetic");
        try (DM.DataGen gen = new DM.DataGen("medm")) {
            gen.run("nItems=100,nTxn=1000,avgLen=10", "/tmp/syn_java.txt", 42);
            double sz = DM.fileSizeMb("/tmp/syn_java.txt");
            System.out.printf("Synthetic → /tmp/syn_java.txt (%.2f MB)%n", sz);
        } catch (Exception e) {
            System.err.println("DataGen: " + e.getMessage());
        }

        // ── GPU context ──────────────────────────────────────────────────────
        section("GPU context");
        try (DM.GpuCtx gpu = new DM.GpuCtx()) {
            System.out.printf("GPU ready: %b%n", gpu.ready());
            System.out.printf("GPU device: %s%n", gpu.deviceName());
        }

        // ── Benchmark ────────────────────────────────────────────────────────
        section("Benchmark — Apriori on T10I4D100K");
        DM.benchReset();
        DM.benchStart(DM.BENCH_TOTAL);
        try (DM.Algorithm algo = new DM.Algorithm("apriori")) {
            algo.run(
                DATASETS + "/itemsets/T10I4D100K.txt",
                "/tmp/apriori_java.txt",
                0.1
            );
        } catch (Exception e) {
            System.err.println("Apriori: " + e.getMessage());
        }
        DM.benchStop(DM.BENCH_TOTAL);
        DM.benchPrint("apriori", "T10I4D100K");
        var r = DM.getBenchReport();
        System.out.println(r);

        // ── Vision ───────────────────────────────────────────────────────────
        section("Vision — MobileNetV4-Tiny predict");
        try (DM.Vision vis = new DM.Vision("mobilenet_tiny")) {
            vis.initModel("/tmp/mobilenet_ckpt", 1000, 224, 1.0f, 0.001f);
            float[] dummyImg = new float[224 * 224 * 3];
            float[] probs = vis.predict(dummyImg, 224, 224, 1000);
            System.out.printf("Predicted probs length: %d%n", probs.length);
        } catch (Exception e) {
            System.out.println("(skipped — Vision init failed: " + e.getMessage() + ")");
        }

        // ── Language Model ───────────────────────────────────────────────────
        section("Language Model — TinyStories");
        try (DM.LM lm = new DM.LM("tinystories")) {
            lm.load("/tmp/tinystories_ckpt");
            String gen2 = lm.generate("Once upon a time", 64);
            System.out.println("Generated: " + gen2);
        } catch (Exception e) {
            System.out.println("(skipped — no checkpoint: " + e.getMessage() + ")");
        }

        // ── Experiment helpers ───────────────────────────────────────────────
        section("Experiment helpers");
        System.out.printf("timer_now   = %.3f s%n", DM.timerNow());
        System.out.printf("peak_ram_mb = %.1f MB%n", DM.peakRamMb());
    }

    private static void section(String title) {
        System.out.printf("%n%s%n  %s%n%s%n",
            "─".repeat(60), title, "─".repeat(60));
    }
}
