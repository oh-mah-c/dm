"""Ground truth for the 10 settings of Table 2 of the HUE-Span paper with hue_mx (sound and complete,
oracle-checked), compared with the outputs of table2_audit.py (reference = SPMF HUE-Span port,
fixed = corrected utility with the original ERU/EEUCP/AWU pruning, traditional = UP-Span utility).

  gcc -O2 -o <bin> docs/core/HUES/hue_mx.c
  HUE_MX=<bin> python3 docs/core/HUES/table2_truth.py -> results/table2_truth.csv
"""
import csv
import os
import re
import subprocess
from pathlib import Path

BIN = os.environ["HUE_MX"]
DATA = os.environ.get("HUES_DATA", "/home/hutech/self/mri-baselines/spmf_datasets")
RES = Path("docs/core/HUES/results")
OUTDIR = RES / "table2_outputs"
TIMEOUT = int(os.environ.get("TIMEOUT", "3600"))
SETTINGS = [("retail", 0.01, 5, 1556, 1174), ("retail", 0.015, 5, 523, 422), ("retail", 0.02, 5, 179, 170),
            ("retail", 0.02, 6, 730, 439), ("retail", 0.02, 7, 2084, 1077),
            ("kosarak", 0.10, 5, 105, 73), ("kosarak", 0.15, 5, 27, 22), ("kosarak", 0.20, 5, 3, 2),
            ("kosarak", 0.20, 6, 21, 8), ("kosarak", 0.20, 7, 81, 28)]


def load(path):
    res = {}
    for line in open(path):
        if line.startswith("@") or not line.strip():
            continue
        body, util = line.split("#UTIL:")
        sets, cur = [], []
        for tok in body.split():
            if tok == "-1":
                sets.append(frozenset(cur)); cur = []
            else:
                cur.append(int(tok))
        res[tuple(sets)] = int(util)
    return res


def main():
    rows = []
    for ds, ratio, md, paper_star, paper_up in SETTINGS:
        path = os.path.join(DATA, f"{ds}_utility_spmf.txt")
        out = OUTDIR / f"{ds}_{ratio}_{md}_truth.txt"
        row = {"dataset": ds, "minUtil": ratio, "maxDur": md, "paper_HUE_star": paper_star, "paper_HUE_upspan": paper_up}
        try:
            p = subprocess.run(["/usr/bin/time", "-f", "RSS_KB=%M", BIN, path, str(ratio), str(md), str(out)],
                               capture_output=True, text=True, timeout=TIMEOUT, check=True)
        except subprocess.TimeoutExpired:
            row["truth_status"] = f"timeout>{TIMEOUT}s"
            rows.append(row)
            print(row, flush=True)
            continue
        m = re.search(r"HUEs=(\d+) evaluated=(\d+) expanded=(\d+) time_s=([\d.]+)", p.stdout)
        truth = load(out)
        ref = load(OUTDIR / f"{ds}_{ratio}_{md}_ref.txt")
        fixed = load(OUTDIR / f"{ds}_{ratio}_{md}_fixed.txt")
        trad = load(OUTDIR / f"{ds}_{ratio}_{md}_traditional.txt")
        row.update({
            "truth_status": "ok", "true_HUE": len(truth), "hue_mx_time_s": float(m.group(4)),
            "hue_mx_rss_MB": round(int(re.search(r"RSS_KB=(\d+)", p.stderr).group(1)) / 1024, 1),
            "hue_mx_evaluated": int(m.group(2)), "hue_mx_expanded": int(m.group(3)),
            "ref_HUE": len(ref), "ref_false": len(set(ref) - set(truth)), "ref_missed": len(set(truth) - set(ref)),
            "ref_wrong_util_true": sum(1 for e in ref if e in truth and truth[e] != ref[e]),
            "fixed_HUE": len(fixed), "fixed_missed": len(set(truth) - set(fixed)),
            "fixed_false_or_wrong": sum(1 for e in fixed if truth.get(e) != fixed[e]),
            "upspan_HUE": len(trad), "upspan_missed": len(set(truth) - set(trad)),
            "upspan_false": len(set(trad) - set(truth)),
        })
        rows.append(row)
        print(row, flush=True)
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(RES / "table2_truth.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
