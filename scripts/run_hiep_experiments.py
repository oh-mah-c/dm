#!/usr/bin/env python3
import random
import subprocess
import sys
from pathlib import Path


SIGNALS = [
    ("rare_api_timeout", "rare_retry_backoff", "rare_circuit_breaker"),
    ("rare_memory_leak", "rare_heap_snapshot", "rare_gc_pause"),
    ("rare_auth_refresh", "rare_token_rotation", "rare_oauth_scope"),
    ("rare_vector_index", "rare_embedding_drift", "rare_recall_drop"),
]


def write_zipf_corpus(path: Path, skew: float, tokens: int = 90000, seed: int = 42):
    random.seed(seed + int(skew * 1000))
    vocab = []
    vocab.extend([f"stop_{i:02d}" for i in range(36)])
    vocab.extend([f"tech_{i:03d}" for i in range(180)])
    vocab.extend([f"rare_domain_{i:03d}" for i in range(420)])
    weights = [1.0 / ((rank + 1) ** skew) for rank in range(len(vocab))]
    total = sum(weights)
    cdf = []
    acc = 0.0
    for w in weights:
        acc += w / total
        cdf.append(acc)

    def sample():
        x = random.random()
        lo, hi = 0, len(cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cdf[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return vocab[lo]

    out = []
    plant_every = max(tokens // 96, 1)
    signal_idx = 0
    i = 0
    while i < tokens:
        if i > 0 and i % plant_every == 0:
            sig = SIGNALS[signal_idx % len(SIGNALS)]
            out.extend(["filler_bridge", *sig, "junk_tail"])
            i += len(sig) + 2
            signal_idx += 1
        else:
            tok = sample()
            if random.random() < 0.015:
                tok = f"junk_{random.randrange(20):02d}"
            out.append(tok)
            i += 1
    path.write_text(" ".join(out), encoding="utf-8")


def run_case(out_file: Path, bin_path: Path, case_name: str, case_group: str, args, metadata=None):
    metadata = metadata or {}
    with out_file.open("a", encoding="utf-8") as f:
        label = "hiep " + " ".join(args)
        f.write(f"===== {label} =====\n")
        f.write(f"case_name={case_name}\n")
        f.write(f"case_group={case_group}\n")
        for k, v in metadata.items():
            f.write(f"{k}={v}\n")
        cmd = [str(bin_path), "hiep", *args]
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        f.write(proc.stdout)
        f.write(f"exit_code={proc.returncode}\n\n")
    if proc.returncode not in (0, 2):
        raise RuntimeError(f"HIEP case failed: {case_name} {args}\n{proc.stdout}")


def main():
    root = Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else Path(__file__).resolve().parents[1]
    out_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / "results" / "hiep_q1"
    bin_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root / "bin" / "dm.exe"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "patterns").mkdir(exist_ok=True)
    (out_dir / "synthetic").mkdir(exist_ok=True)

    real_cases = [
        ("retail", root / "datasets" / "itemsets" / "retail.txt", "0.015", "3", "12000"),
        ("accidents", root / "datasets" / "itemsets" / "accidents.txt", "0.03", "3", "3500"),
        ("chess", root / "datasets" / "itemsets" / "chess.txt", "0.08", "3", "3196"),
    ]
    for name, path, minsup, depth, maxtx in real_cases:
        out_file = out_dir / f"hiep_{name}.txt"
        out_file.write_text("", encoding="utf-8")
        for ratio in ("0.20", "0.35", "0.50", "0.70"):
            run_case(
                out_file,
                bin_path,
                name,
                "real_itemset",
                [
                    "--input", str(path),
                    "--input-type", "transactions",
                    "--mode", "itemset",
                    "--minsup", minsup,
                    "--theta-ratio", ratio,
                    "--max-depth", depth,
                    "--max-transactions", maxtx,
                    "--max-patterns", "25000",
                    "--max-seconds", "35",
                ],
            )

    text_file = out_dir / "hiep_databricks_dolly.txt"
    text_file.write_text("", encoding="utf-8")
    for ratio in ("0.20", "0.35", "0.50", "0.70"):
        run_case(
            text_file,
            bin_path,
            "databricks_dolly_15k",
            "real_text",
            [
                "--input", str(root / "datasets" / "prompt" / "databricks-dolly-15k.jsonl"),
                "--input-type", "text",
                "--mode", "itemset",
                "--window", "64",
                "--stride", "32",
                "--minsup", "0.01",
                "--theta-ratio", ratio,
                "--max-depth", "4",
                "--max-bytes", "5000000",
                "--max-patterns", "25000",
                "--max-seconds", "35",
            ],
        )

    synthetic_summary = out_dir / "hiep_synthetic_zipf.txt"
    synthetic_summary.write_text("", encoding="utf-8")
    for skew in (1.05, 1.15, 1.25, 1.35):
        corpus = out_dir / "synthetic" / f"zipf_s{str(skew).replace('.', 'p')}.txt"
        write_zipf_corpus(corpus, skew)
        for ratio in ("0.20", "0.35", "0.50", "0.70"):
            run_case(
                synthetic_summary,
                bin_path,
                f"zipf_s{skew}",
                "synthetic_zipf",
                [
                    "--input", str(corpus),
                    "--input-type", "text",
                    "--mode", "sequence",
                    "--window", "48",
                    "--stride", "24",
                    "--minsup", "3",
                    "--theta-ratio", ratio,
                    "--gamma", "0.12",
                    "--max-depth", "3",
                    "--max-patterns", "20000",
                    "--max-seconds", "35",
                ],
                {"zipf_skew": skew},
            )

    ablation_file = out_dir / "hiep_ablation.txt"
    ablation_file.write_text("", encoding="utf-8")
    corpus = out_dir / "synthetic" / "zipf_s1p15.txt"
    if not corpus.exists():
        write_zipf_corpus(corpus, 1.15)
    base_args = [
        "--input", str(corpus),
        "--input-type", "text",
        "--mode", "sequence",
        "--window", "48",
        "--stride", "24",
        "--minsup", "3",
        "--theta-ratio", "0.35",
        "--gamma", "0.12",
        "--max-depth", "3",
        "--max-patterns", "20000",
        "--max-seconds", "35",
    ]
    ablations = [
        ("full", []),
        ("no_tiub", ["--no-tiub"]),
        ("no_iwru", ["--no-iwru"]),
        ("uniform_weights", ["--uniform-weights"]),
        ("no_compactness", ["--no-compactness"]),
    ]
    for name, extra in ablations:
        run_case(ablation_file, bin_path, name, "ablation", [*base_args, *extra], {"ablation": name})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
