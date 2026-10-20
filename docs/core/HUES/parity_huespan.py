"""Parity of the C port of HUE-Span (dm.exe hue_span) with the authors' Java implementation (SPMF).

For random complex event sequences in SPMF format, both programs are run with the same
minutil, maxDuration and flag configuration; the sets of output lines (episode + utility) and
the statistics printed by printStats() must be identical.

Requirements (paths can be overridden by environment variables):
  JAVA       a Java >= 25 runtime            (default: /home/hutech/self/mri-baselines/jdk25/jdk-25*/bin/java)
  SPMF_JAR   spmf.jar                        (default: /home/hutech/self/mri-baselines/spmf/spmf.jar)
  HARNESS    directory of HueSpanRun.class   (default: /home/hutech/self/mri-baselines/huespan_harness)
Run from the repository root:
  python3 docs/core/HUES/parity_huespan.py  -> docs/core/HUES/results/parity_huespan.csv
"""
import csv
import glob
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BIN = "./build/bin/dm.exe"
JAVA = os.environ.get("JAVA") or sorted(glob.glob("/home/hutech/self/mri-baselines/jdk25/jdk-25*/bin/java"))[-1]
SPMF_JAR = os.environ.get("SPMF_JAR", "/home/hutech/self/mri-baselines/spmf/spmf.jar")
HARNESS = os.environ.get("HARNESS", "/home/hutech/self/mri-baselines/huespan_harness")
OUT = Path("docs/core/HUES/results/parity_huespan.csv")
SEQUENCES = int(os.environ.get("SEQUENCES", "150"))
SEED = 2019
# (name, checkMaximumUtility, useTighterUpperBound, useCoocMatrix, pruningPrefix)
CONFIGS = [("default", 1, 1, 1, 1), ("traditional", 0, 1, 1, 1), ("ewu", 1, 0, 1, 1),
           ("no_eeucp", 1, 1, 0, 0), ("no_prefix", 1, 1, 1, 0), ("upspan_mode", 0, 0, 0, 0)]
STAT_KEYS = ["Episodes counts", "Candidate counts", "Combinated episode counts", "matrix pruning counts",
             "upper Bound pruning counts", "non maximal combinated episode counts atleast"]


def make_sequence(rng):
    n_events = rng.randint(2, 6)
    length = rng.randint(3, 25)
    lines = []
    for _ in range(length):
        if rng.random() < 0.08:
            lines.append("# comment line")
            continue
        k = rng.randint(1, n_events)
        evs = sorted(rng.sample(range(1, n_events + 1), k))
        if rng.random() < 0.5:
            rng.shuffle(evs)
        us = [rng.randint(1, 20) for _ in evs]
        lines.append(f"{' '.join(map(str, evs))}:{sum(us)}:{' '.join(map(str, us))}")
    return "\n".join(lines) + "\n"


def stats(text):
    res = {}
    for k in STAT_KEYS:
        m = re.search(re.escape(k) + r"\s*:\s*(-?\d+)", text)
        res[k] = int(m.group(1)) if m else None
    return res


def lines_of(path):
    if not os.path.exists(path):
        return []
    return sorted(l.rstrip("\n") for l in open(path) if l.strip() and not l.startswith("@"))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    cases = bad_out = bad_stats = skipped = 0
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        db, oj, oc = (os.path.join(tmp, x) for x in ("db.txt", "java.txt", "c.txt"))
        for s in range(SEQUENCES):
            open(db, "w").write(make_sequence(rng))
            for _ in range(3):
                ratio, dur = rng.choice([0.05, 0.1, 0.2, 0.3, 0.5]), rng.choice([1, 2, 3, 4])
                single = rng.choice([True, False])
                for name, cm, eru, cooc, pref in CONFIGS:
                    for p in (oj, oc):
                        if os.path.exists(p):
                            os.remove(p)
                    try:
                        j = subprocess.run([JAVA, "-cp", f"{SPMF_JAR}:{HARNESS}", "HueSpanRun", db, oj, str(ratio), str(dur),
                                        str(bool(cm)).lower(), str(bool(eru)).lower(), str(single).lower(),
                                        str(bool(cooc)).lower(), str(bool(pref)).lower()],
                                       capture_output=True, text=True, timeout=120)
                    except subprocess.TimeoutExpired:
                        skipped += 1
                        continue
                    env = dict(os.environ, DM_HUE_OUT=oc, HUE_SPAN_ERU=str(eru), HUE_SPAN_EEUCP=str(cooc),
                               HUE_SPAN_PREFIX=str(pref))
                    c = subprocess.run([BIN, "hue_span", db, "0", str(ratio), str(dur), str(single).lower(),
                                        str(not cm).lower()], capture_output=True, text=True, timeout=300, env=env)
                    cases += 1
                    same_out = lines_of(oj) == lines_of(oc)
                    sj, sc = stats(j.stdout), stats(c.stdout.replace("[HUE-Span] ", ""))
                    same_stats = sj == sc
                    bad_out += not same_out
                    bad_stats += not same_stats
                    if not (same_out and same_stats) and bad_out + bad_stats <= 6:
                        print(f"MISMATCH seq={s} ratio={ratio} dur={dur} single={single} cfg={name} "
                              f"out={same_out} stats={same_stats} java={sj} c={sc}", flush=True)
                        open(OUT.parent / f"hue_mismatch_{s}_{name}.txt", "w").write(open(db).read())
            if (s + 1) % 25 == 0:
                print(f"{s + 1} sequences, {cases} cases, output mismatches {bad_out}, stats mismatches {bad_stats}",
                      flush=True)
    rows.append({"sequences": SEQUENCES, "seed": SEED, "configs": len(CONFIGS), "cases": cases,
                 "output_mismatches": bad_out, "stats_mismatches": bad_stats, "skipped_java_timeout": skipped})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(rows[0])
    sys.exit(1 if bad_out or bad_stats else 0)


if __name__ == "__main__":
    main()
