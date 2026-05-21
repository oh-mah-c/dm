#!/usr/bin/env python3
import csv
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin" / "dm.exe"
OUT = ROOT / "results" / "sparc_hoi_compare"

DATASETS = [
    ("mushrooms", ROOT / "datasets" / "itemsets" / "mushrooms.txt", [0.20, 0.30, 0.40]),
    ("foodmart", ROOT / "datasets" / "itemsets" / "foodmartFIM.txt", [0.10, 0.20, 0.30]),
    ("retail", ROOT / "datasets" / "itemsets" / "retail.txt", [0.02, 0.05, 0.10]),
]

ALGORITHMS = [
    ("sparc_hoi", "raw-average"),
    ("hep", "summed-xi"),
    ("dfhoi", "summed-xi"),
    ("cloe_hoi", "closed-average"),
]

COUNT_PATTERNS = [
    re.compile(r"Raw fullset HO itemsets found:\s*(\d+)"),
    re.compile(r"Support-closed HO itemsets found:\s*(\d+)"),
    re.compile(r"Total high occupancy itemsets found:\s*(\d+)"),
    re.compile(r"Frequent Itemsets:\s*(\d+)"),
]
TIME_RE = re.compile(r"Algorithm Core\s*:\s*([0-9.]+)\s*ms")
RAM_RE = re.compile(r"Peak RAM .*:\s*([0-9.]+)\s*MB")
ITEMS_RE = re.compile(r"Total Items\s*:\s*(\d+)")
LIMIT_RE = re.compile(r"limited=(yes|no)")


def extract(patterns, text, default=""):
    for pat in patterns if isinstance(patterns, list) else [patterns]:
        m = pat.search(text)
        if m:
            return m.group(1)
    return default


def run_one(algo, dataset_path, alpha, max_seconds=60):
    cmd = [str(BIN), algo, str(dataset_path), "0", f"{alpha:.6f}"]
    if algo in {"sparc_hoi", "cloe_hoi"}:
        cmd += ["0", str(max_seconds)]
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=max_seconds + 20,
    )
    wall = time.time() - started
    text = proc.stdout
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "wall_seconds": f"{wall:.6f}",
        "itemsets": extract(COUNT_PATTERNS, text, "0"),
        "total_items": extract(ITEMS_RE, text, "0"),
        "algo_ms": extract(TIME_RE, text, ""),
        "peak_ram_mb": extract(RAM_RE, text, ""),
        "limited": extract(LIMIT_RE, text, "no"),
        "stdout": text,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset_name, dataset_path, alphas in DATASETS:
        ddir = OUT / dataset_name
        ddir.mkdir(parents=True, exist_ok=True)
        for algo, semantics in ALGORITHMS:
            outfile = ddir / f"{algo}_{dataset_name}.txt"
            with outfile.open("w", encoding="utf-8") as fh:
                fh.write(f"algorithm: {algo}\n")
                fh.write(f"semantics: {semantics}\n")
                fh.write(f"dataset: {dataset_name}\n")
                fh.write(f"path: {dataset_path}\n\n")
                for alpha in alphas:
                    rec = run_one(algo, dataset_path, alpha)
                    fh.write("=" * 72 + "\n")
                    fh.write(f"alpha: {alpha:.6f}\n")
                    fh.write(f"command: {rec['cmd']}\n")
                    fh.write(f"returncode: {rec['returncode']}\n")
                    fh.write(f"wall_seconds: {rec['wall_seconds']}\n")
                    fh.write(f"itemsets: {rec['itemsets']}\n")
                    fh.write(f"total_items: {rec['total_items']}\n")
                    fh.write(f"algo_ms: {rec['algo_ms']}\n")
                    fh.write(f"peak_ram_mb: {rec['peak_ram_mb']}\n")
                    fh.write(f"limited: {rec['limited']}\n\n")
                    fh.write(rec["stdout"])
                    fh.write("\n")
                    row = {
                        "dataset": dataset_name,
                        "algorithm": algo,
                        "semantics": semantics,
                        "alpha": f"{alpha:.6f}",
                        "itemsets": rec["itemsets"],
                        "total_items": rec["total_items"],
                        "algo_ms": rec["algo_ms"],
                        "peak_ram_mb": rec["peak_ram_mb"],
                        "limited": rec["limited"],
                        "returncode": rec["returncode"],
                    }
                    rows.append(row)
    with (OUT / "sparc_hoi_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "dataset",
                "algorithm",
                "semantics",
                "alpha",
                "itemsets",
                "total_items",
                "algo_ms",
                "peak_ram_mb",
                "limited",
                "returncode",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT / 'sparc_hoi_summary.csv'}")


if __name__ == "__main__":
    main()
