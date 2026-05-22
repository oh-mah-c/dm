// examples/go/main.go — Go usage example for dm
//
// Build:
//   CGO_CFLAGS="-I../../include" \
//   CGO_LDFLAGS="-L../../ -ldm" \
//   LD_LIBRARY_PATH=../../ \
//   go run main.go
//
// Or set a replace directive pointing at ../../bindings/go in your go.mod.

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	dm "github.com/pomaieco/dm/dm"
)

func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "ERROR:", err)
		os.Exit(1)
	}
}

func section(title string) {
	fmt.Printf("\n%s\n  %s\n%s\n", repeat('─', 60), title, repeat('─', 60))
}

func repeat(r rune, n int) string {
	buf := make([]rune, n)
	for i := range buf {
		buf[i] = r
	}
	return string(buf)
}

func repoRoot() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "../..")
}

func main() {
	must(dm.Init())
	maj, min, patch := dm.VersionNumber()
	fmt.Printf("libdm %s  (%d.%d.%d)\n", dm.Version(), maj, min, patch)

	root := repoRoot()

	// ── Algorithm: FP-Growth ─────────────────────────────────────────────────
	section("Algorithm — FP-Growth")
	{
		algo, err := dm.NewAlgorithm("fpgrowth")
		must(err)
		defer algo.Close()

		err = algo.Run(
			filepath.Join(root, "datasets/itemsets/mushrooms.txt"),
			"/tmp/fpgrowth_go_out.txt",
			0.05,
			[]string{"min_length=2"},
		)
		if err != nil {
			fmt.Println("FP-Growth:", err)
		} else {
			fmt.Println("FP-Growth complete → /tmp/fpgrowth_go_out.txt")
		}
	}

	// ── Algorithm: EFIM ──────────────────────────────────────────────────────
	section("Algorithm — EFIM (high-utility)")
	{
		efim, err := dm.NewAlgorithm("efim")
		must(err)
		defer efim.Close()

		err = efim.Run(
			filepath.Join(root, "datasets/utilities/foodmart.txt"),
			"/tmp/efim_go_out.txt",
			50.0, nil,
		)
		if err != nil {
			fmt.Println("EFIM:", err)
		} else {
			fmt.Println("EFIM complete")
		}
	}

	// ── Algorithm list ───────────────────────────────────────────────────────
	section("Algorithm registry (first 5)")
	{
		ids, err := dm.ListAlgorithms()
		must(err)
		for _, id := range ids[:min5(len(ids))] {
			fmt.Println(" ", id)
		}
		fmt.Printf("  … (%d total)\n", len(ids))
	}

	// ── Tokenizer ─────────────────────────────────────────────────────────────
	section("Tokenizer — BPE train + encode/decode")
	{
		tok, err := dm.NewTokenizer("bpe")
		must(err)
		defer tok.Close()

		err = tok.Train(
			filepath.Join(root, "datasets/tokenizer/real_corpus.txt"),
			2000,
			"/tmp/bpe_go_model",
		)
		if err != nil {
			fmt.Println("train:", err)
		} else {
			fmt.Printf("Trained BPE, vocab_size=%d\n", tok.VocabSize())

			text := "data mining with frequent itemsets"
			ids, err := tok.Encode(text)
			if err != nil {
				fmt.Println("encode:", err)
			} else {
				fmt.Printf("Encoded '%s' → %d tokens\n", text, len(ids))
				decoded, err := tok.Decode(ids)
				if err == nil {
					fmt.Printf("Decoded: '%s'\n", decoded)
				}
			}
		}
	}

	// ── BitSet ────────────────────────────────────────────────────────────────
	section("BitSet operations")
	{
		a, _ := dm.NewBitSet(128)
		b, _ := dm.NewBitSet(128)
		defer a.Close()
		defer b.Close()

		for _, pos := range []uint{1, 7, 42, 100} {
			a.Set(pos)
		}
		for _, pos := range []uint{7, 42, 99} {
			b.Set(pos)
		}
		a.And(b) // {7, 42}
		fmt.Printf("popcount after AND = %d (expected 2)\n", a.Popcount())
	}

	// ── DataGen ───────────────────────────────────────────────────────────────
	section("DataGen — MEDM synthetic")
	{
		gen, err := dm.NewDataGen("medm")
		must(err)
		defer gen.Close()

		err = gen.Run("nItems=100,nTxn=1000,avgLen=10", "/tmp/syn_go.txt", 42)
		if err != nil {
			fmt.Println("datagen:", err)
		} else {
			sz := dm.FileSizeMB("/tmp/syn_go.txt")
			fmt.Printf("Synthetic dataset → /tmp/syn_go.txt (%.2f MB)\n", sz)
		}
	}

	// ── GPU context ───────────────────────────────────────────────────────────
	section("GPU context")
	{
		gpu := dm.NewGpuCtx(0, "")
		defer gpu.Close()
		fmt.Printf("GPU ready: %v\n", gpu.Ready())
		fmt.Printf("GPU device: %s\n", gpu.DeviceName())
	}

	// ── Benchmark ─────────────────────────────────────────────────────────────
	section("Benchmark — Apriori")
	{
		dm.BenchReset()
		dm.BenchStart(dm.BenchPhaseTotal)

		algo, _ := dm.NewAlgorithm("apriori")
		_ = algo.Run(
			filepath.Join(root, "datasets/itemsets/T10I4D100K.txt"),
			"/tmp/apriori_go.txt",
			0.1, nil,
		)
		algo.Close()

		dm.BenchStop(dm.BenchPhaseTotal)
		dm.BenchPrint("apriori", "T10I4D100K")

		r := dm.GetBenchReport()
		fmt.Printf("total=%.1fms  patterns=%d  throughput=%.2fMB/s\n",
			r.PhaseMS[dm.BenchPhaseTotal], r.NumPatterns, r.ThroughputMBPerS)
	}

	// ── Experiment helpers ────────────────────────────────────────────────────
	section("Experiment helpers")
	fmt.Printf("timer_now   = %.3f s\n", dm.TimerNow())
	fmt.Printf("peak_ram_mb = %.1f MB\n", dm.PeakRAMMB())
}

func min5(n int) int {
	if n > 5 {
		return 5
	}
	return n
}
