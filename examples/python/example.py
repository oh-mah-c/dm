#!/usr/bin/env python3
"""
examples/python/example.py — Python usage example for dm

Demonstrates:
  - dm.version()
  - dm.Algorithm  (FP-Growth, EFIM)
  - dm.Tokenizer  (BPE train, encode, decode)
  - dm.BitSet
  - dm.DataGen
  - dm.GpuCtx
  - dm.bench_*  (benchmark API)
  - dm.LM        (language model generate)

Run:
    export DM_LIB=/path/to/libdm.so
    cd examples/python
    python example.py
"""

import sys
import os

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../bindings/python'))

import dm

DATASETS = os.path.join(os.path.dirname(__file__), '../../datasets')
TOKENIZER_DATASETS = os.path.join(DATASETS, 'tokenizer')

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


def main():
    dm.init()
    print(f"libdm {dm.version()}")

    # ── Algorithm: FP-Growth ────────────────────────────────────────────────
    section("Algorithm — FP-Growth (frequent itemset mining)")
    with dm.Algorithm("fpgrowth") as algo:
        algo.run(
            dataset_path=f"{DATASETS}/itemsets/mushrooms.txt",
            output_path="/tmp/fpgrowth_py_out.txt",
            min_support=0.05,
            extra_args=["min_length=2"],
        )
        print("FP-Growth complete → /tmp/fpgrowth_py_out.txt")

    # ── Algorithm: EFIM (high-utility) ──────────────────────────────────────
    section("Algorithm — EFIM (high-utility itemset mining)")
    with dm.Algorithm("efim") as efim:
        efim.run(
            dataset_path=f"{DATASETS}/utilities/foodmart.txt",
            output_path="/tmp/efim_py_out.txt",
            min_support=50.0,
        )
        print("EFIM complete → /tmp/efim_py_out.txt")

    # ── Algorithm list ───────────────────────────────────────────────────────
    section("Algorithm registry (first 10)")
    ids = dm.Algorithm.list_all()
    for i in ids[:10]:
        print(f"  {i}")
    print(f"  … ({len(ids)} total)")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    section("Tokenizer — BPE train + encode/decode")
    with dm.Tokenizer("bpe") as tok:
        corpus = f"{TOKENIZER_DATASETS}/real_corpus.txt"
        tok.train(corpus, vocab_size=2000, output_path="/tmp/bpe_py_model")
        print(f"Trained BPE, vocab_size={tok.vocab_size}")

        text = "data mining and machine learning"
        ids_list = tok.encode(text)
        print(f"Encoded '{text}' → {len(ids_list)} tokens: {ids_list[:8]}…")

        decoded = tok.decode(ids_list)
        print(f"Decoded: '{decoded}'")

        # Show first 5 token texts
        for tid in ids_list[:5]:
            print(f"  token[{tid}] = '{tok.token_text(tid)}'")

    # ── BitSet ───────────────────────────────────────────────────────────────
    section("BitSet operations")
    a = dm.BitSet(128)
    b = dm.BitSet(128)
    for pos in [1, 7, 42, 100]:
        a.set(pos)
    for pos in [7, 42, 99]:
        b.set(pos)
    a &= b   # intersection: {7, 42}
    print(f"popcount after AND = {a.popcount()} (expected 2: {{7, 42}})")
    for pos in [7, 42]:
        print(f"  bit[{pos}] = {a.get(pos)}")

    # ── DataGen ──────────────────────────────────────────────────────────────
    section("DataGen — MEDM synthetic transactions")
    with dm.DataGen("medm") as gen:
        gen.run(
            spec="nItems=100,nTxn=1000,avgLen=10",
            output_path="/tmp/syn_py.txt",
            seed=42,
        )
        print(f"Synthetic dataset → /tmp/syn_py.txt "
              f"({dm.file_size_mb('/tmp/syn_py.txt'):.2f} MB)")

    # ── GPU context ──────────────────────────────────────────────────────────
    section("GPU context")
    gpu = dm.GpuCtx(device_index=0)
    print(f"GPU ready: {gpu.ready}")
    print(f"GPU device: {gpu.device_name}")

    # ── Benchmark ────────────────────────────────────────────────────────────
    section("Benchmark — Apriori on T10I4D100K")
    dm.bench_reset()
    dm.bench_start(dm.BENCH_TOTAL)
    with dm.Algorithm("apriori") as a2:
        a2.run(
            dataset_path=f"{DATASETS}/itemsets/T10I4D100K.txt",
            output_path="/tmp/apriori_py.txt",
            min_support=0.1,
        )
    dm.bench_stop(dm.BENCH_TOTAL)
    dm.bench_print("apriori", "T10I4D100K")
    r = dm.bench_report()
    print(r)

    # ── Vision ───────────────────────────────────────────────────────────────
    section("Vision — MobileNetV4-Tiny predict")
    try:
        with dm.Vision("mobilenet_tiny") as vis:
            vis.init_model("/tmp/mobilenet_ckpt", classes=1000, image_size=224)
            dummy_img = [0.5] * (224 * 224 * 3)
            probs = vis.predict(dummy_img, 224, 224, 1000)
            print(f"Predicted probs length: {len(probs)}")
    except Exception as e:
        print(f"(skipped — Vision init failed: {e})")

    # ── Language model ───────────────────────────────────────────────────────
    section("Language Model — TinyStories generate")
    try:
        with dm.LM("tinystories") as lm:
            lm.load("/tmp/tinystories_ckpt")   # will fail if no checkpoint; that's ok
            text = lm.generate("Once upon a time", max_tokens=64)
            print(f"Generated: {text}")
    except Exception as e:
        print(f"(skipped — no checkpoint: {e})")

    # ── Experiment helpers ───────────────────────────────────────────────────
    section("Experiment helpers")
    print(f"timer_now()   = {dm.timer_now():.3f} s")
    print(f"peak_ram_mb() = {dm.peak_ram_mb():.1f} MB")


if __name__ == "__main__":
    main()
