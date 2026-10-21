"""Summarize results/bench_flchus.csv for the FLCHUS-X paper.

Writes into docs/core/FLCHUS/paper/gen/:
  - <dataset>_<sweep>.dat   pgfplots data: x value, median time of each algorithm (nan = time limit),
                            number of patterns
  - runtime_table.tex       median time [min-max] at the default setting of each dataset
  - ablation_table.tex      median time of FLCHUS-X and of each ablation at the default setting
  - sweeps_table.tex        speed-up ranges of FLCHUS-X per dataset and sweep
  - summary.txt             plain-text overview (also printed)
Medians are over the repetitions of a setting; a setting where any repetition exceeded the time limit
is reported as a timeout.

  python3 docs/core/FLCHUS/summarize_bench.py
"""
import csv
import math
import statistics as st
from collections import OrderedDict
from pathlib import Path

SRC = Path("docs/core/FLCHUS/results/bench_flchus.csv")
GEN = Path("docs/core/FLCHUS/paper/gen")
ALGS = ["FLCHUSPM", "FLCHUSPM-", "FLCHUS-X"]
ABL = ["X-noDESC", "X-noCOSTW", "X-noFIX", "X-noSEL"]
ORDER = ["Snake", "BMS", "SIGN", "LEVIATHAN", "BIBLE", "FIFA"]
DEFAULTS = {"Snake": (60, 5, 5), "BMS": (0.03, 0.05, 1), "SIGN": (15, 2, 3), "LEVIATHAN": (2, 1, 3),
            "BIBLE": (2, 1, 3), "FIFA": (8, 1, 3)}


def fnum(s):
    return float(s)


def load():
    data = OrderedDict()
    for r in csv.DictReader(open(SRC)):
        key = (r["dataset"], r["sweep"], fnum(r["ms"]), fnum(r["mu"]), fnum(r["mc"]))
        data.setdefault(key, {}).setdefault(r["algorithm"], []).append(r)
    return data


def stat(rows):
    """(median, min, max, patterns) or None if a repetition did not finish"""
    if not rows or any(x["status"] != "ok" for x in rows):
        return None
    t = [float(x["time_s"]) for x in rows]
    return st.median(t), min(t), max(t), int(rows[-1]["patterns"])


def fmt_t(s):
    if s is None:
        return "TL"
    m = s[0]
    return f"{m:.2f}" if m < 10 else f"{m:.1f}" if m < 100 else f"{m:.0f}"


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    data = load()
    lines = []
    # pgfplots data per dataset and sweep
    for ds in ORDER:
        for sweep, idx in (("ms", 2), ("mu", 3), ("mc", 4)):
            keys = [k for k in data if k[0] == ds and k[1] == sweep]
            if not keys:
                continue
            keys.sort(key=lambda k: k[idx])
            with open(GEN / f"{ds}_{sweep}.dat", "w") as f:
                f.write("x FL FLminus X patterns\n")
                for k in keys:
                    row = [stat(data[k].get(a)) for a in ALGS]
                    pat = next((s[3] for s in row if s), "nan")
                    f.write(f"{k[idx]} " + " ".join(f"{s[0]:.4f}" if s else "nan" for s in row) + f" {pat}\n")
    # default-setting runtime table (a default setting appears in each sweep; use the ms sweep)
    out = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
           r"Dataset & Setting & \#FLCHUS & \FL{} & \FL$^{-}$ & \FX{} & Gain \\", r"\midrule"]
    for ds in ORDER:
        ms, mu, mc = DEFAULTS[ds]
        k = (ds, "ms", float(ms), float(mu), float(mc))
        if k not in data:
            continue
        s = [stat(data[k].get(a)) for a in ALGS]
        pat = next((x[3] for x in s if x), 0)
        sp = (f"{s[1][0] / s[2][0]:.1f}$\\times$" if s[1] and s[2] else
              (f"$>${300 / s[2][0]:.0f}$\\times$" if s[2] else "--"))
        out.append(f"{ds} & ({ms}, {mu}, {mc}) & {pat:,} & {fmt_t(s[0])} & {fmt_t(s[1])} & {fmt_t(s[2])} & {sp} \\\\")
        lines.append(f"default {ds}: " + " ".join(f"{a}={fmt_t(x)}" for a, x in zip(ALGS, s)) + f" patterns={pat}")
    out += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "runtime_table.tex").write_text("\n".join(out) + "\n")
    # ablation table
    out = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
           r"Dataset & \FX{} & no \mdb{} & no cost width & no fixpoint & no selection \\", r"\midrule"]
    for ds in ORDER:
        ms, mu, mc = DEFAULTS[ds]
        kd = (ds, "ms", float(ms), float(mu), float(mc))
        ka = (ds, "ablation", float(ms), float(mu), float(mc))
        if ka not in data or kd not in data:
            continue
        full = stat(data[kd].get("FLCHUS-X"))
        cells = [fmt_t(full)]
        for a in ABL:
            s = stat(data[ka].get(a))
            cells.append(fmt_t(s) + (f" ({s[0] / full[0]:.1f}$\\times$)" if s and full else ""))
        out.append(f"{ds} & " + " & ".join(cells) + r" \\")
        lines.append(f"ablation {ds}: X={fmt_t(full)} " + " ".join(f"{a}={fmt_t(stat(data[ka].get(a)))}" for a in ABL))
    out += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "ablation_table.tex").write_text("\n".join(out) + "\n")
    # speed-up ranges per sweep
    out = [r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
           r"Dataset & Sweep & vs \FL{} & vs \FL$^{-}$ & Solved \\", r"\midrule"]
    for ds in ORDER:
        for sweep in ("ms", "mu", "mc"):
            keys = [k for k in data if k[0] == ds and k[1] == sweep]
            if not keys:
                continue
            r1, r2, solved = [], [], [0, 0, 0]
            for k in keys:
                s = [stat(data[k].get(a)) for a in ALGS]
                for i in range(3):
                    solved[i] += s[i] is not None
                if s[0] and s[2]:
                    r1.append(s[0][0] / s[2][0])
                if s[1] and s[2]:
                    r2.append(s[1][0] / s[2][0])
            rng = lambda r: f"{min(r):.1f}--{max(r):.1f}" if r else "--"
            out.append(f"{ds} & {sweep} & {rng(r1)} & {rng(r2)} & {solved[0]} / {solved[1]} / {solved[2]} of {len(keys)} \\\\")
            lines.append(f"sweep {ds} {sweep}: vsFL={rng(r1)} vsFLminus={rng(r2)} solved={solved} of {len(keys)}")
    out += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "sweeps_table.tex").write_text("\n".join(out) + "\n")
    (GEN / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
