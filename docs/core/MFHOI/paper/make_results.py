"""Build the macros, tables and figures of the experimental section.

Run from the repository root (needs numpy and matplotlib):
  python3 docs/core/MFHOI/paper/make_results.py
Inputs : docs/core/MFHOI/results/runs.csv (run_benchmarks.py),
         docs/core/MFHOI/results/oracle_sum.csv (oracle_test.py --sum)
Outputs: docs/core/MFHOI/paper/results_macros.tex, tables/*.tex, figures/*.pdf
"""
import csv
import math
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("docs/core/MFHOI")
PAPER = ROOT / "paper"
TAB = PAPER / "tables"
FIG = PAPER / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DATA = Path("datasets/itemsets")

ORDER = ["mushroom", "retail", "accidents", "kosarak", "T10I4D100K"]
FILES = {"mushroom": "mushrooms.txt", "retail": "retail.txt", "accidents": "accidents.txt",
         "kosarak": "kosarak.dat", "T10I4D100K": "T10I4D100K.txt"}
TYPE = {"mushroom": "real, dense", "retail": "real, sparse", "accidents": "real, dense",
        "kosarak": "real, sparse", "T10I4D100K": "synthetic"}
METHODS = ["mhoi", "hep", "dfhoi"]
LABEL = {"mhoi": "MHOI-Miner", "hep": "HEP+filter", "dfhoi": "DFHOI+filter"}
COLOR = {"mhoi": "#2e8b57", "hep": "#4c72b0", "dfhoi": "#dd8452"}
MARK = {"mhoi": "o", "hep": "^", "dfhoi": "s"}

rows = [r for r in csv.DictReader(open(ROOT / "results/runs.csv"))
        if r["experiment"] == "xi" and r["method"] in METHODS]
R = {(r["dataset"], float(r["xi_rel"]), r["method"]): r for r in rows}
# evaluated itemsets of MHOI-Miner: visited nodes plus the extensions skipped by rule P4,
# whose covers are scanned to compute the child bound (mhoi_support_computations.csv)
for r in csv.DictReader(open(ROOT / "results/mhoi_support_computations.csv")):
    k = (r["dataset"], float(r["xi_rel"]), "mhoi")
    if k in R and R[k]["status"] == "ok":
        assert int(R[k]["nodes"]) == int(r["nodes"])
        R[k]["visited"] = R[k]["nodes"]
        R[k]["nodes"] = r["support_computations"]
used = [d for d in ORDER if any(r["dataset"] == d for r in rows)]
macros = {"NumDatasets": ["zero", "one", "two", "three", "four", "five", "six"][len(used)]}


def ok(r):
    return r is not None and r["status"] == "ok"


def num(x):
    return f"{int(x):,}"


def ftime(t):
    t = float(t)
    return f"{t:.0f}" if t >= 100 else (f"{t:.1f}" if t >= 10 else (f"{t:.2f}" if t >= 0.1 else f"{t:.3f}"))


def ratio(x):
    return f"{x:,.0f}" if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")


def mb(kb):
    return f"{int(kb) / 1024:.0f}" if int(kb) >= 10240 else f"{int(kb) / 1024:.1f}"


def status_cell(r):
    if r is None:
        return "--"
    return {"timeout": "TL", "memory": "ML", "skipped": "--"}.get(r["status"], "F")


# ── datasets ───────────────────────────────────────────────────────────────
def dstats(name):
    n = tot = 0
    mn, mx = 10 ** 9, 0
    items = set()
    for line in open(DATA / FILES[name]):
        t = line.split()
        if not t:
            continue
        L = len(set(t))
        n += 1
        tot += L
        mn, mx = min(mn, L), max(mx, L)
        items.update(t)
    return n, len(items), tot / n, mn, mx


with open(TAB / "datasets.tex", "w") as f:
    for d in used:
        n, m, avg, mn, mx = dstats(d)
        L = f"{mn}" if mn == mx else f"{mn}--{mx}"
        f.write(f"\\texttt{{{d}}} & {TYPE[d]} & {num(n)} & {num(m)} & {avg:.1f} & {L} \\\\\n")

# ── settings, parity ───────────────────────────────────────────────────────
settings = sorted({(r["dataset"], float(r["xi_rel"])) for r in rows}, key=lambda s: (ORDER.index(s[0]), -s[1]))
checked = bad = 0
for d, x in settings:
    vals = {R[(d, x, m)]["mhoi"] for m in METHODS if ok(R.get((d, x, m)))}
    if sum(ok(R.get((d, x, m))) for m in METHODS) >= 2:
        checked += 1
        bad += len(vals) > 1
assert bad == 0, "methods disagree"
macros.update(NumSettings=str(len(settings)), ParityChecked=str(checked), ParityMismatches=str(bad))

# ── ratios baseline / MHOI-Miner on settings completed by both ─────────────
for base, tag in (("hep", "Hep"), ("dfhoi", "Dfhoi")):
    P = []
    for d, x in settings:
        rf, rb = R.get((d, x, "mhoi")), R.get((d, x, base))
        if ok(rf) and ok(rb):
            P.append((d, x, int(rb["nodes"]) / max(1, int(rf["nodes"])),
                      float(rb["time"]) / max(1e-6, float(rf["time"])),
                      int(rb["peak_rss_kb"]) / max(1, int(rf["peak_rss_kb"]))))
    failed = [(d, x) for d, x in settings if R.get((d, x, base)) is not None and not ok(R.get((d, x, base)))]
    macros[f"Failed{tag}"] = str(len(failed))
    macros[f"PairCount{tag}"] = str(len(P))
    if not P:
        continue
    nr, tr, mr = [p[2] for p in P], [p[3] for p in P], [p[4] for p in P]
    macros[f"MedianNodeReduction{tag}"] = ratio(statistics.median(nr))
    macros[f"MinNodeReduction{tag}"] = ratio(min(nr))
    macros[f"MaxNodeReduction{tag}"] = ratio(max(nr))
    macros[f"MedianTimeRatio{tag}"] = ratio(statistics.median(tr))
    macros[f"MinTimeRatio{tag}"] = ratio(min(tr))
    macros[f"MaxTimeRatio{tag}"] = ratio(max(tr))
    macros[f"GeoTimeRatio{tag}"] = ratio(float(np.exp(np.mean(np.log(tr)))))
    macros[f"MedianMemRatio{tag}"] = ratio(statistics.median(mr))
    macros[f"FasterCount{tag}"] = str(sum(t > 1 for t in tr))
    macros[f"MaxSlowdown{tag}"] = f"{1 / min(tr):.1f}"
# end-to-end times (loading and building the initial representation included)
for base, tag in (("hep", "Hep"), ("dfhoi", "Dfhoi")):
    E = []
    for d, x in settings:
        rf, rb = R.get((d, x, "mhoi")), R.get((d, x, base))
        if ok(rf) and ok(rb):
            tf = float(rf["time"]) + float(rf["time_load"])
            tb = float(rb["time"]) + float(rb["time_load"])
            E.append(tb / tf)
    if E:
        macros[f"MedianEtoERatio{tag}"] = ratio(statistics.median(E))
        macros[f"FasterEtoECount{tag}"] = str(sum(e > 1 for e in E))
        macros[f"MaxEtoESlowdown{tag}"] = f"{1 / min(E):.1f}"
# closure evaluations of MHOI-Miner relative to its visited nodes
ce = [int(R[(d, x, "mhoi")]["closure_evals"]) / int(R[(d, x, "mhoi")]["nodes"])
      for d, x in settings if ok(R.get((d, x, "mhoi"))) and int(R[(d, x, "mhoi")]["nodes"]) >= 10]
macros["MaxClosureEvalRatio"] = f"{max(ce):.2f}"
# set-level parity of the outputs (parity_sets.py)
ps = ROOT / "results/parity_sets.csv"
if ps.exists():
    P = list(csv.DictReader(open(ps)))
    macros["SetParityChecked"] = str(len(P))
    macros["SetParityIdentical"] = str(sum(int(r["identical"]) for r in P))
macros["FailedFull"] = str(sum(1 for d, x in settings if R.get((d, x, "mhoi")) is not None and not ok(R.get((d, x, "mhoi")))))

# ── full results table ─────────────────────────────────────────────────────
with open(TAB / "results.tex", "w") as f:
    for d in used:
        xs = [x for dd, x in settings if dd == d]
        for i, x in enumerate(xs):
            rs = [R.get((d, x, m)) for m in METHODS]
            out = next((r["mhoi"] for r in rs if ok(r)), None)
            hoi = next((r["hoi"] for r in rs[1:] if ok(r)), None)
            if i == 0:
                f.write(f"\\multicolumn{{12}}{{@{{}}l}}{{\\texttt{{{d}}}}} \\\\\n")
            cells = [num(r["nodes"]) if ok(r) else status_cell(r) for r in rs]
            cells += [ftime(r["time"]) if ok(r) else status_cell(r) for r in rs]
            cells += [mb(r["peak_rss_kb"]) if ok(r) else status_cell(r) for r in rs]
            f.write(f"{x * 100:g} & {num(out) if out else '--'} & {num(hoi) if hoi else '--'} & "
                    + " & ".join(cells) + " \\\\\n")
        f.write("\\midrule\n" if d != used[-1] else "")

# ── number of HOIs and MHOIs ────────────────────────────────────────────────
ratios = {}
for d, x in settings:
    rs = [R.get((d, x, m)) for m in METHODS]
    mh = next((int(r["mhoi"]) for r in rs if ok(r)), None)
    ho = next((int(r["hoi"]) for r in rs[1:] if ok(r)), None)
    if mh and ho:
        ratios.setdefault(d, []).append(ho / mh)
allr = [v for vs in ratios.values() for v in vs]
macros["MedianHoiMhoiRatio"] = ratio(statistics.median(allr))
for d, vs in ratios.items():
    tag = {"mushroom": "Mushroom", "retail": "Retail", "kosarak": "Kosarak", "T10I4D100K": "Tten", "accidents": "Accidents"}[d]
    macros[f"MinHoiMhoi{tag}"] = ratio(min(vs))
    macros[f"MaxHoiMhoi{tag}"] = ratio(max(vs))
fig, axes = plt.subplots(1, len(used), figsize=(2.6 * len(used), 2.3), squeeze=False)
for k, d in enumerate(used):
    ax = axes[0][k]
    xs = sorted(x for dd, x in settings if dd == d)
    hoi = [(x * 100, next((int(R[(d, x, m)]["hoi"]) for m in ("hep", "dfhoi") if ok(R.get((d, x, m)))), None)) for x in xs]
    mho = [(x * 100, next((int(R[(d, x, m)]["mhoi"]) for m in METHODS if ok(R.get((d, x, m)))), None)) for x in xs]
    hoi = [p for p in hoi if p[1]]
    mho = [p for p in mho if p[1]]
    if hoi:
        ax.plot([p[0] for p in hoi], [p[1] for p in hoi], marker="s", color="#4c72b0", label="HOIs", linewidth=1.4, markersize=3.5)
    if mho:
        ax.plot([p[0] for p in mho], [p[1] for p in mho], marker="o", color="#2e8b57", label="MHOIs", linewidth=1.4, markersize=3.5)
    ax.set_yscale("log")
    from matplotlib.ticker import LogLocator, NullFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_title(d, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
    ax.set_xlabel(r"$\xi$ (% of transactions)", fontsize=8)
    if k == 0:
        ax.set_ylabel("number of itemsets", fontsize=8)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(FIG / "xi_counts.pdf", bbox_inches="tight")
plt.close(fig)

# ── figure: one row per measurement, one column per dataset ───────────────
from matplotlib.ticker import LogLocator, NullFormatter

MEAS = (("nodes", "evaluated itemsets", 1), ("time", "runtime (s)", 1), ("peak_rss_kb", "peak memory (MiB)", 1 / 1024))
fig, axes = plt.subplots(len(MEAS), len(used), figsize=(2.6 * len(used), 2.15 * len(MEAS)), squeeze=False)
for row, (metric, ylab, scale) in enumerate(MEAS):
    for k, d in enumerate(used):
        ax = axes[row][k]
        xs = sorted(x for dd, x in settings if dd == d)
        for m in METHODS:
            pts = [(x * 100, float(R[(d, x, m)][metric]) * scale) for x in xs if ok(R.get((d, x, m)))]
            pts = [p for p in pts if p[1] > 0]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARK[m], color=COLOR[m],
                        label=LABEL[m], linewidth=1.4, markersize=3.5)
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)
        if row == 0:
            ax.set_title(d, fontsize=9)
        if row == len(MEAS) - 1:
            ax.set_xlabel(r"$\xi$ (% of transactions)", fontsize=8)
        if k == 0:
            ax.set_ylabel(ylab, fontsize=8)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(FIG / "xi_results.pdf", bbox_inches="tight")
plt.close(fig)

# ── ablation ───────────────────────────────────────────────────────────────
ABL = ["mhoi", "no_pre_o", "wubocc_o", "ubo_o", "no_hut_o", "no_sub_o", "full"]
arows = [r for r in csv.DictReader(open(ROOT / "results/runs.csv")) if r["experiment"] == "ablation"]
A = {(r["dataset"], r["method"]): r for r in arows}
if arows:
    with open(TAB / "ablation.tex", "w") as f:
        for d in used:
            rs = [A.get((d, m)) for m in ABL]
            if not any(rs):
                continue
            x = float(next(r for r in rs if r)["xi_rel"])
            f.write(f"\\texttt{{{d}}} & {x * 100:g} & " +
                    " & ".join(f"{num(r['nodes'])}" if ok(r) else status_cell(r) for r in rs) + " \\\\\n")
            f.write(" & & " + " & ".join(f"{ftime(r['time'])}~s" if ok(r) else "" for r in rs) + " \\\\\n")
    def aratio(m, base="mhoi", key="nodes"):
        v = [float(A[(d, m)][key]) / float(A[(d, base)][key]) for d in used
             if ok(A.get((d, m))) and ok(A.get((d, base))) and float(A[(d, base)][key]) > 0]
        return v
    for m, tag in (("no_pre_o", "NoPre"), ("wubocc_o", "W"), ("ubo_o", "Ubo"), ("full", "Closed"), ("no_hut_o", "NoHut"), ("no_sub_o", "NoSub")):
        v = aratio(m)
        if v:
            macros[f"AblNodes{tag}Min"] = ratio(min(v))
            macros[f"AblNodes{tag}Max"] = ratio(max(v))
        t = aratio(m, key="time")
        if t:
            macros[f"AblTime{tag}Min"] = ratio(min(t))
            macros[f"AblTime{tag}Max"] = ratio(max(t))
    # B against W and UBO, both without P4
    for m, tag in (("wubocc_o", "W"), ("ubo_o", "Ubo")):
        v = aratio(m, base="no_pre_o")
        if v:
            macros[f"AblNodes{tag}OverB"] = f"{ratio(min(v))}--{ratio(max(v))}"

# ── oracle ─────────────────────────────────────────────────────────────────
oc = ROOT / "results/oracle_sum.csv"
if oc.exists():
    O = list(csv.DictReader(open(oc)))
    macros["NumOracleCases"] = f"{sum(int(r['cases']) for r in O):,}"
    macros["NumOracleDatabases"] = f"{sum(int(r['databases']) for r in O):,}"
    macros["OracleMismatches"] = str(sum(int(r["mismatches"]) for r in O))

with open(PAPER / "results_macros.tex", "w") as f:
    f.write("% Generated by make_results.py -- do not edit by hand.\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
for k, v in sorted(macros.items()):
    print(k, "=", v)
