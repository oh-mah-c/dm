"""Build every number, table and figure of the experimental section from the raw results.

Run from the repository root (needs numpy, scipy, matplotlib):
  python3 docs/core/AURA-HOI/make_results.py
Inputs : docs/core/AURA-HOI/results/{main,ablation,scalability,pumsb,oracle}.csv
Outputs: docs/core/AURA-HOI/results_macros.tex, docs/core/AURA-HOI/tables/*.tex,
         docs/core/AURA-HOI/figures/*.pdf
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("docs/core/AURA-HOI")
RES = ROOT / "results"
TAB = ROOT / "tables"
FIG = ROOT / "figures"
DATA = Path("datasets/itemsets")
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

ORDER = ["mushrooms", "chess", "connect", "accidents", "retail", "T10I4D100K", "T20I6D100K", "kosarak", "foodmart"]
FILES = {"mushrooms": "mushrooms.txt", "chess": "chess.txt", "connect": "connect.txt", "accidents": "accidents.txt",
         "retail": "retail.txt", "T10I4D100K": "T10I4D100K.txt", "T20I6D100K": "t20i6d100k.txt",
         "kosarak": "kosarak.dat", "foodmart": "foodmartFIM.txt", "pumsb": "pumsb.txt"}
KIND = {"mushrooms": "dense", "chess": "dense", "connect": "dense", "accidents": "dense", "pumsb": "dense",
        "retail": "sparse", "T10I4D100K": "sparse", "T20I6D100K": "sparse", "kosarak": "sparse", "foodmart": "sparse"}
SOURCE = {"foodmart": "SPMF", "pumsb": "FIMI"}
ALGOS = ["hep", "dfhoi", "aura_hoi"]
LABEL = {"hep": "HEP", "dfhoi": "DFHOI", "aura_hoi": "AURA-HOI"}
COLOR = {"hep": "#e05c5c", "dfhoi": "#5c8de0", "aura_hoi": "#3dba6b"}
MARK = {"hep": "s", "dfhoi": "^", "aura_hoi": "o"}

macros = {"MemLimit": "20"}


def macro(name, value):
    macros[name] = value


def read(name):
    p = RES / f"{name}.csv"
    return list(csv.DictReader(open(p))) if p.exists() else []


def num(x):
    return f"{x:,}"


def fmt_time(t):
    if t >= 100:
        return f"{t:.0f}"
    if t >= 10:
        return f"{t:.1f}"
    if t >= 1:
        return f"{t:.2f}"
    return f"{t:.3f}"


def fmt_x(x):
    return f"{x:.0f}" if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")


# ── Dataset statistics ─────────────────────────────────────────────────────
def dataset_stats(name):
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
        mn = min(mn, L)
        mx = max(mx, L)
        items.update(t)
    return {"n": n, "m": len(items), "avg": tot / n, "min": mn, "max": mx, "density": 100 * tot / n / len(items)}


main = read("main")
alphas_of = defaultdict(list)
for r in main:
    if r["algorithm"] == "aura_hoi":
        alphas_of[r["dataset"]].append(float(r["alpha"]))
used = [d for d in ORDER if d in alphas_of]

rows = []
for d in used + ["pumsb"]:
    s = dataset_stats(d)
    a = sorted(alphas_of.get(d, [0.05, 0.07, 0.09]))
    rng = f"{a[0]:g}--{a[-1]:g}" if len(a) > 1 else f"{a[0]:g}"
    lens = f"{s['min']}" if s["min"] == s["max"] else f"{s['min']}--{s['max']}"
    rows.append(rf"\texttt{{{d}}} & {KIND[d]} & {num(s['n'])} & {num(s['m'])} & {s['avg']:.1f} & {lens} & "
                rf"{s['density']:.2f} & {len(a)} & {rng} \\")
with open(TAB / "datasets.tex", "w") as f:
    f.write("\n".join(rows) + "\n")
macro("NumDatasets", str(len(used)))

# ── Main comparison ────────────────────────────────────────────────────────
R = {(r["dataset"], float(r["alpha"]), r["algorithm"]): r for r in main}
configs = sorted({(r["dataset"], float(r["alpha"])) for r in main}, key=lambda c: (ORDER.index(c[0]), -c[1]))
macro("NumConfigs", str(len(configs)))


def ok(c, a):
    r = R.get((c[0], c[1], a))
    return r is not None and r["status"] == "ok"


def t(c, a):
    return float(R[(c[0], c[1], a)]["time"])


# output parity
parity_checked = parity_bad = 0
for c in configs:
    vals = {R[(c[0], c[1], a)]["items"] for a in ALGOS if ok(c, a)}
    if len([a for a in ALGOS if ok(c, a)]) >= 2:
        parity_checked += 1
        parity_bad += len(vals) > 1
macro("ParityChecked", str(parity_checked))
macro("ParityMismatches", str(parity_bad))
assert parity_bad == 0, "output mismatch between algorithms"

both_df = [c for c in configs if ok(c, "aura_hoi") and ok(c, "dfhoi")]
both_hep = [c for c in configs if ok(c, "aura_hoi") and ok(c, "hep")]
macro("NumBothFinished", str(len(both_df)))
macro("WinsDFHOI", str(sum(t(c, "aura_hoi") < t(c, "dfhoi") for c in both_df)))
macro("WinsHEP", str(sum(t(c, "aura_hoi") < t(c, "hep") for c in both_hep)))
macro("NumBothFinishedHEP", str(len(both_hep)))
macro("AuraOnlyFinished", str(sum(ok(c, "aura_hoi") and not ok(c, "dfhoi") for c in configs)))
macro("AuraFinished", str(sum(ok(c, "aura_hoi") for c in configs)))
macro("DfhoiFinished", str(sum(ok(c, "dfhoi") for c in configs)))
macro("HepFinished", str(sum(ok(c, "hep") for c in configs)))
macro("AuraFailed", str(sum(not ok(c, "aura_hoi") for c in configs)))
macro("NonEmptyConfigs", str(sum(ok(c, "aura_hoi") and int(R[(c[0], c[1], "aura_hoi")]["items"]) > 0 for c in configs)))

dense_sp = [t(c, "dfhoi") / t(c, "aura_hoi") for c in both_df if KIND[c[0]] == "dense"]
sparse_sp = [t(c, "dfhoi") / t(c, "aura_hoi") for c in both_df if KIND[c[0]] == "sparse"]
macro("MaxSpeedupDense", fmt_x(max(dense_sp)) if dense_sp else "--")
macro("GeoSpeedupDense", fmt_x(float(np.exp(np.mean(np.log(dense_sp))))) if dense_sp else "--")
macro("GeoSpeedupSparse", fmt_x(float(np.exp(np.mean(np.log(sparse_sp))))) if sparse_sp else "--")
macro("MinSpeedupSparse", fmt_x(min(sparse_sp)) if sparse_sp else "--")
macro("MaxSpeedupSparse", fmt_x(max(sparse_sp)) if sparse_sp else "--")


def wilcoxon(pairs):
    x = np.log([p[0] for p in pairs])
    y = np.log([p[1] for p in pairs])
    res = stats.wilcoxon(x, y, alternative="two-sided")
    return res.statistic, res.pvalue, len(pairs)


for name, base, sel in (("DFHOI", "dfhoi", both_df), ("HEP", "hep", both_hep)):
    if len(sel) >= 6:
        W, p, n_ = wilcoxon([(t(c, "aura_hoi"), t(c, base)) for c in sel])
        macro(f"Wilcoxon{name}W", f"{W:.0f}")
        macro(f"Wilcoxon{name}N", str(n_))
        macro(f"Wilcoxon{name}P", f"{p:.1e}".replace("e-0", r"\times 10^{-").replace("e-", r"\times 10^{-") + "}"
              if p < 1e-3 else f"{p:.3f}")


def cell(c, a):
    r = R.get((c[0], c[1], a))
    if r is None:
        return "--"
    if r["status"] == "ok":
        return fmt_time(float(r["time"]))
    return {"timeout": "TL", "skipped": "ML$^*$", "memory": "ML", "failed": "F"}.get(r["status"], "F")


# per-dataset summary
summ = []
for d in used:
    cs = [c for c in configs if c[0] == d]
    bd = [c for c in cs if c in both_df]
    sp = [t(c, "dfhoi") / t(c, "aura_hoi") for c in bd]
    red = []
    for c in bd:
        nd, na = int(R[(c[0], c[1], "dfhoi")]["nodes"]), int(R[(c[0], c[1], "aura_hoi")]["nodes"])
        if nd > 0:
            red.append(100.0 * (nd - na) / nd)
    fin = {a: sum(ok(c, a) for c in cs) for a in ALGOS}
    wins = sum(t(c, "aura_hoi") < t(c, "dfhoi") for c in bd)
    gm = fmt_x(float(np.exp(np.mean(np.log(sp))))) if sp else "--"
    mx = fmt_x(max(sp)) if sp else "--"
    md = f"{np.median(red):.1f}" if red else "--"
    summ.append(rf"\texttt{{{d}}} & {len(cs)} & {fin['hep']} & {fin['dfhoi']} & {fin['aura_hoi']} & "
                rf"{wins}/{len(bd)} & {gm} & {mx} & {md} \\")
with open(TAB / "summary.tex", "w") as f:
    f.write("\n".join(summ) + "\n")

# full per-configuration table
full = []
for d in used:
    cs = [c for c in configs if c[0] == d]
    for i, c in enumerate(cs):
        ra = R[(c[0], c[1], "aura_hoi")]
        items = num(int(ra["items"])) if ra["status"] == "ok" else "--"
        nd = R[(c[0], c[1], "dfhoi")]
        nodes_d = num(int(nd["nodes"])) if nd["status"] == "ok" else "--"
        nodes_a = num(int(ra["nodes"])) if ra["status"] == "ok" else "--"
        mem_a = f"{float(ra['mem']):.1f}" if ra["status"] == "ok" else "--"
        mem_d = f"{float(nd['mem']):.1f}" if nd["status"] == "ok" else "--"
        rh = R[(c[0], c[1], "hep")]
        mem_h = f"{float(rh['mem']):.0f}" if rh["status"] == "ok" else "--"
        head = rf"\texttt{{{d}}}" if i == 0 else ""
        times = [cell(c, a) for a in ALGOS]
        okt = [(t(c, a), k) for k, a in enumerate(ALGOS) if ok(c, a)]
        if okt:
            k = min(okt)[1]
            times[k] = rf"\textbf{{{times[k]}}}"
        full.append(rf"{head} & {c[1]:g} & {items} & {' & '.join(times)} & {nodes_d} & {nodes_a} & "
                    rf"{mem_h} & {mem_d} & {mem_a} \\")
    full.append(r"\midrule")
full.pop()
with open(TAB / "full.tex", "w") as f:
    f.write("\n".join(full) + "\n")

# memory summary
aura_mem = [float(R[(c[0], c[1], "aura_hoi")]["mem"]) for c in configs if ok(c, "aura_hoi")]
hep_mem = [float(R[(c[0], c[1], "hep")]["mem"]) for c in configs if ok(c, "hep")]
macro("AuraMemMax", f"{max(aura_mem):.1f}")
macro("AuraMemMedian", f"{np.median(aura_mem):.1f}")
macro("HepMemMax", f"{max(hep_mem):,.0f}")
macro("HepMemoryFailures", str(sum(R[(c[0], c[1], "hep")]["status"] == "memory" for c in configs)))

# ── Figures: runtime, visited nodes and memory per dataset ─────────────────
cols = 3
rows_n = math.ceil(len(used) / cols)
for metric, ylabel, fname, log in (("time", "Runtime (s)", "runtime", True),
                                    ("nodes", "Candidate intersections", "nodes", True),
                                    ("mem", "Peak extra memory (MB)", "memory", True)):
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, 3.0 * rows_n), squeeze=False)
    for k, d in enumerate(used):
        ax = axes[k // cols][k % cols]
        for a in ALGOS:
            pts = sorted((c[1], float(R[(c[0], c[1], a)][metric])) for c in configs
                         if c[0] == d and ok(c, a) and float(R[(c[0], c[1], a)][metric]) > 0)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=MARK[a], color=COLOR[a],
                        label=LABEL[a], linewidth=1.8, markersize=5)
        ax.set_title(d, fontsize=10)
        ax.set_xlabel(r"$\alpha$", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        if log:
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.tick_params(labelsize=8)
    for k in range(len(used), rows_n * cols):
        axes[k // cols][k % cols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / f"{fname}.pdf", bbox_inches="tight")
    plt.close(fig)

# ── Ablation ───────────────────────────────────────────────────────────────
abl = read("ablation")
if abl:
    A = {(r["dataset"], float(r["alpha"]), r["variant"]): r for r in abl}
    cfg = sorted({(r["dataset"], float(r["alpha"])) for r in abl}, key=lambda c: (ORDER.index(c[0]), -c[1]))
    variants = ["full", "no_tail_cap", "no_class_bound", "no_residual", "no_arena"]
    out = []
    for c in cfg:
        cells = []
        for v in variants:
            r = A.get((c[0], c[1], v))
            if r is None or r["status"] != "ok":
                cells.append("ML" if r is not None and r["status"] in ("memory", "skipped") else "F")
            else:
                cells.append(f"{fmt_time(float(r['time']))} ({num(int(r['nodes']))})")
        out.append(rf"\texttt{{{c[0]}}} & {c[1]:g} & " + " & ".join(cells) + r" \\")
    with open(TAB / "ablation.tex", "w") as f:
        f.write("\n".join(out) + "\n")

# ── Scalability ────────────────────────────────────────────────────────────
sc = read("scalability")
if sc:
    S = {(r["dataset"], float(r["fraction"]), r["algorithm"]): r for r in sc}
    dsets = [d for d in ORDER if any(k[0] == d for k in S)]
    fig, axes = plt.subplots(1, len(dsets), figsize=(5 * len(dsets), 3.4), squeeze=False)
    for k, d in enumerate(dsets):
        ax = axes[0][k]
        alpha = next(float(r["alpha"]) for r in sc if r["dataset"] == d)
        for a in ALGOS:
            pts = sorted((f, float(S[(d, f, a)]["time"])) for (dd, f, aa) in S
                         if dd == d and aa == a and S[(d, f, a)]["status"] == "ok")
            if pts:
                ax.plot([100 * p[0] for p in pts], [p[1] for p in pts], marker=MARK[a], color=COLOR[a],
                        label=LABEL[a], linewidth=1.8)
        ax.set_title(rf"{d} ($\alpha$ = {alpha:g})", fontsize=10)
        ax.set_xlabel("Transactions used (%)", fontsize=9)
        ax.set_ylabel("Runtime (s)", fontsize=9)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "scalability.pdf", bbox_inches="tight")
    plt.close(fig)

# ── pumsb ──────────────────────────────────────────────────────────────────
pu = read("pumsb")
if pu:
    finished = sum(r["status"] == "ok" for r in pu)
    macro("PumsbRuns", str(len(pu)))
    macro("PumsbFinished", str(finished))

# ── Oracle tests ───────────────────────────────────────────────────────────
orc = read("oracle")
if orc:
    macro("NumOracleCases", num(sum(int(r["cases"]) for r in orc)))
    macro("NumOracleDatabases", num(sum(int(r["databases"]) for r in orc)))
    macro("OracleMismatches", str(sum(int(r["mismatches"]) for r in orc)))

with open(ROOT / "results_macros.tex", "w") as f:
    f.write("% Generated by make_results.py from results/*.csv -- do not edit by hand.\n")
    for k, v in sorted(macros.items()):
        f.write(rf"\newcommand{{\{k}}}{{{v}}}" + "\n")
print(f"{len(macros)} macros written")
for k, v in sorted(macros.items()):
    print(f"  {k} = {v}")
