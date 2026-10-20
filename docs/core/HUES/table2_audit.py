"""Batch A: re-run the 10 settings of Table 2 of the HUE-Span paper (ADMA 2019) with the reference
implementation (hue_span, parity-checked port of SPMF) and the corrected variant
(HUE_SPAN_FIXED=1), then check every reported episode with verify_episodes.py.

Datasets (SPMF website, publicdatasets/): retail_utility_spmf.txt, kosarak_utility_spmf.txt.
One run per setting, single thread; a run is stopped after TIMEOUT seconds.
Run from the repository root:
  python3 docs/core/HUES/table2_audit.py -> docs/core/HUES/results/table2_audit.csv
"""
import csv
import os
import re
import subprocess
import time
from pathlib import Path

BIN = "./build/bin/dm.exe"
DATA = os.environ.get("HUES_DATA", "/home/hutech/self/mri-baselines/spmf_datasets")
RES = Path("docs/core/HUES/results")
OUTDIR = RES / "table2_outputs"
TIMEOUT = int(os.environ.get("TIMEOUT", "1800"))
# (dataset, minUtil ratio, maxDur, #HUE* reported in Table 2 of the paper)
SETTINGS = [("retail", 0.01, 5, 1556), ("retail", 0.015, 5, 523), ("retail", 0.02, 5, 179),
            ("retail", 0.02, 6, 730), ("retail", 0.02, 7, 2084),
            ("kosarak", 0.10, 5, 105), ("kosarak", 0.15, 5, 27), ("kosarak", 0.20, 5, 3),
            ("kosarak", 0.20, 6, 21), ("kosarak", 0.20, 7, 81)]
FILES = {"retail": "retail_utility_spmf.txt", "kosarak": "kosarak_utility_spmf.txt"}


def run(path, ratio, max_dur, fixed, out):
    if out.exists():
        out.unlink()
    env = dict(os.environ, DM_HUE_OUT=str(out), HUE_SPAN_FIXED=str(fixed))
    t0 = time.monotonic()
    try:
        p = subprocess.run(["/usr/bin/time", "-f", "RSS_KB=%M", BIN, "hue_span", path, "0", str(ratio), str(max_dur),
                            "true"], capture_output=True, text=True, timeout=TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return None
    wall = time.monotonic() - t0
    hues = int(re.search(r"Episodes counts : (\d+)", p.stdout).group(1))
    cands = int(re.search(r"Candidate counts : (\d+)", p.stdout).group(1))
    rss = int(re.search(r"RSS_KB=(\d+)", p.stderr).group(1))
    return wall, hues, cands, rss


def verify(path, out, ratio, max_dur, csv_out):
    p = subprocess.run(["python3", "docs/core/HUES/verify_episodes.py", path, str(out), str(ratio), str(max_dur),
                        str(csv_out)], capture_output=True, text=True, check=True)
    m = re.search(r"wrong_utility=(\d+) false_hue=(\d+)", p.stdout)
    return int(m.group(1)), int(m.group(2))


def episodes(out):
    return {l.split(" #UTIL")[0] for l in open(out) if l.strip() and not l.startswith("@")}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, ratio, md, paper in SETTINGS:
        path = os.path.join(DATA, FILES[ds])
        row = {"dataset": ds, "minUtil": ratio, "maxDur": md, "paper_HUE_star": paper}
        outs = {}
        for fixed, tag in ((0, "ref"), (1, "fixed")):
            out = OUTDIR / f"{ds}_{ratio}_{md}_{tag}.txt"
            r = run(path, ratio, md, fixed, out)
            if r is None:
                row[f"{tag}_status"] = f"timeout>{TIMEOUT}s"
                print(f"{ds} {ratio} {md} {tag}: timeout", flush=True)
                continue
            wall, hues, cands, rss = r
            wrong, false = verify(path, out, ratio, md, OUTDIR / f"verify_{ds}_{ratio}_{md}_{tag}.csv")
            row.update({f"{tag}_status": "ok", f"{tag}_HUE": hues, f"{tag}_candidates": cands,
                        f"{tag}_wall_s": round(wall, 2), f"{tag}_rss_MB": round(rss / 1024, 1),
                        f"{tag}_wrong_utility": wrong, f"{tag}_false_HUE": false})
            outs[tag] = out
            print(f"{ds} {ratio} {md} {tag}: HUE={hues} wrong={wrong} false={false} wall={wall:.1f}s", flush=True)
        if len(outs) == 2:
            a, b = episodes(outs["ref"]), episodes(outs["fixed"])
            row["only_ref"], row["only_fixed"] = len(a - b), len(b - a)
        rows.append(row)
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(RES / "table2_audit.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
