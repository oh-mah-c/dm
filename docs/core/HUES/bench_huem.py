"""Runtime/memory benchmark for redefined HUEM (single thread, sequential runs, REPS repetitions).

Algorithms:
  hue_span          C port of the SPMF HUE-Span reference implementation (ERU + EEUCP, default flags)
  hue_span_fixed    same, corrected maximal utility (HUE_SPAN_FIXED=1) — still uses the original ERU
  up_span           HUE-Span code with traditional utility (UP-Span semantics)
  hue_mx            proposed: MERU + event pruning (sound and complete)
  hue_mx_noprune    ablation: no event deletion
  hue_mx_swu        ablation: SWU bound instead of MERU
  hue_exact_swu     naive sound miner (no incremental structures), small inputs only
Settings follow Fig. 3 of the HUE-Span paper (Retail: minUtil 1-3% at maxDur 5, maxDur 3-7 at 2%;
Kosarak: minUtil 10-30% at maxDur 5, maxDur 3-7 at 20%) plus a 3,000-line Retail prefix.
A run that exceeds TIMEOUT is recorded as a timeout and the remaining repetitions of that
algorithm/setting are skipped. Output: results/bench_huem.csv (one row per run).

Run from the repository root (build/bin/dm.exe, build/bin/hue_mx, build/bin/hue_exact):
  python3 docs/core/HUES/bench_huem.py
"""
import csv
import os
import re
import subprocess
import time
from pathlib import Path

DATA = os.environ.get("HUES_DATA", "/home/hutech/self/mri-baselines/spmf_datasets")
RES = Path("docs/core/HUES/results")
OUT = RES / os.environ.get("BENCH_OUT", "bench_huem.csv")
REPS = int(os.environ.get("REPS", "3"))
TIMEOUT = int(os.environ.get("TIMEOUT", "3600"))
SKIP_EXACT_ABOVE = 3000                     # hue_exact_swu only on the prefix


def settings():
    s = [("retail3k", 0.02, 5)]
    for r in (0.01, 0.015, 0.02, 0.025, 0.03):
        s.append(("retail", r, 5))
    for d in (3, 4, 6, 7):
        s.append(("retail", 0.02, d))
    for r in (0.10, 0.15, 0.20, 0.25, 0.30):
        s.append(("kosarak", r, 5))
    for d in (3, 4, 6, 7):
        s.append(("kosarak", 0.20, d))
    return s


def path_of(ds):
    if ds == "retail3k":
        p = RES / "retail_utility_spmf_first3000.txt"
        if not p.exists():
            with open(os.path.join(DATA, "retail_utility_spmf.txt")) as f, open(p, "w") as g:
                for i, line in enumerate(f):
                    if i >= 3000:
                        break
                    g.write(line)
        return str(p)
    return os.path.join(DATA, f"{ds}_utility_spmf.txt")


def command(alg, path, ratio, md):
    env = dict(os.environ)
    for k in ("DM_HUE_OUT", "HUE_SPAN_FIXED", "HUE_SPAN_ERU", "HUE_SPAN_EEUCP", "HUE_SPAN_PREFIX"):
        env.pop(k, None)
    if alg.startswith("hue_span") or alg == "up_span":
        cmd = ["./build/bin/dm.exe", "hue_span", path, "0", str(ratio), str(md), "true"]
        if alg == "up_span":
            cmd.append("true")
        if alg == "hue_span_fixed":
            env["HUE_SPAN_FIXED"] = "1"
        return cmd, env, r"Episodes counts : (\d+)"
    if alg == "hue_exact_swu":
        return ["./build/bin/hue_exact", path, str(ratio), str(md), "-", "swu", "noprune"], env, r"HUEs=(\d+)"
    extra = {"hue_mx": ["meru", "prune"], "hue_mx_noprune": ["meru", "noprune"], "hue_mx_swu": ["swu", "prune"]}[alg]
    return ["./build/bin/hue_mx", path, str(ratio), str(md), "-"] + extra, env, r"HUEs=(\d+)"


ALGS = ["hue_span", "hue_span_fixed", "up_span", "hue_mx", "hue_mx_noprune", "hue_mx_swu", "hue_exact_swu"]


def main():
    RES.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "minUtil", "maxDur", "algorithm", "rep", "status", "wall_s", "max_rss_MB", "HUEs"]
    new = not OUT.exists()
    f = open(OUT, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    for ds, ratio, md in settings():
        path = path_of(ds)
        for alg in ALGS:
            if alg == "hue_exact_swu" and ds != "retail3k":
                continue
            for rep in range(1, REPS + 1):
                cmd, env, pat = command(alg, path, ratio, md)
                row = {"dataset": ds, "minUtil": ratio, "maxDur": md, "algorithm": alg, "rep": rep}
                t0 = time.monotonic()
                try:
                    p = subprocess.run(["/usr/bin/time", "-f", "RSS_KB=%M"] + cmd, capture_output=True, text=True,
                                       timeout=TIMEOUT, env=env)
                    row["wall_s"] = round(time.monotonic() - t0, 3)
                    m = re.search(pat, p.stdout)
                    rss = re.search(r"RSS_KB=(\d+)", p.stderr)
                    row["status"] = "ok" if p.returncode == 0 and m else f"error{p.returncode}"
                    row["HUEs"] = int(m.group(1)) if m else ""
                    row["max_rss_MB"] = round(int(rss.group(1)) / 1024, 1) if rss else ""
                except subprocess.TimeoutExpired:
                    row.update({"status": f"timeout>{TIMEOUT}s", "wall_s": TIMEOUT})
                w.writerow(row)
                f.flush()
                print(row, flush=True)
                if row["status"] != "ok":
                    break
    f.close()


if __name__ == "__main__":
    main()
