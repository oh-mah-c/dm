"""Build every number, table and figure of the experimental section from the raw results.

Run from the repository root (needs numpy, scipy, matplotlib):
  python3 docs/core/AURA-HOI/make_results.py
Inputs : docs/core/AURA-HOI/results/{main,ablation,scalability,oracle,parity}.csv
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

ORDER = ["mushroom", "accidents", "chess", "pumsb", "retail", "kosarak", "T10I4D100K"]
FILES = {"mushroom": "mushrooms.txt", "accidents": "accidents.txt", "retail": "retail.txt",
         "kosarak": "kosarak.dat", "T10I4D100K": "T10I4D100K.txt", "chess": "chess.txt", "pumsb": "pumsb.txt"}
KIND = {"mushroom": "dense", "accidents": "dense", "chess": "dense", "pumsb": "dense", "retail": "sparse", "kosarak": "sparse", "T10I4D100K": "sparse"}
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
for d in used:
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

def hois(c):
    return int(R[(c[0], c[1], "aura_hoi")]["items"])


def geo(xs):
    return fmt_x(float(np.exp(np.mean(np.log(xs))))) if xs else "--"


def pct(xs):  # geometric-mean speedup expressed as a runtime reduction in percent
    return f"{100 * (1 - float(np.exp(-np.mean(np.log(xs))))):.0f}" if xs else "--"


for name, base, sel in (("", "dfhoi", both_df), ("HEP", "hep", both_hep)):
    for kind in ("dense", "sparse"):
        sp = [t(c, base) / t(c, "aura_hoi") for c in sel if KIND[c[0]] == kind]
        spn = [t(c, base) / t(c, "aura_hoi") for c in sel if KIND[c[0]] == kind and hois(c) > 0]
        K = kind.capitalize()
        macro(f"GeoSpeedup{K}{name}", geo(sp))
        macro(f"MaxSpeedup{K}{name}", fmt_x(max(sp)) if sp else "--")
        macro(f"MinSpeedup{K}{name}", fmt_x(min(sp)) if sp else "--")
        macro(f"NumSpeedup{K}{name}", str(len(sp)))
        macro(f"GeoSpeedup{K}NonEmpty{name}", geo(spn))
        macro(f"MinSpeedup{K}NonEmpty{name}", fmt_x(min(spn)) if spn else "--")
        macro(f"MaxSpeedup{K}NonEmpty{name}", fmt_x(max(spn)) if spn else "--")
        spz = [t(c, base) / t(c, "aura_hoi") for c in sel if KIND[c[0]] == kind and hois(c) == 0]
        macro(f"MinSpeedup{K}Empty{name}", fmt_x(min(spz)) if spz else "--")
        macro(f"NumSpeedup{K}Empty{name}", str(len(spz)))
        macro(f"NumSpeedup{K}NonEmpty{name}", str(len(spn)))
        macro(f"TimeReduction{K}{name}", pct(sp))
best_sp = [min(t(c, "hep"), t(c, "dfhoi")) / t(c, "aura_hoi") for c in configs
           if KIND[c[0]] == "sparse" and ok(c, "aura_hoi") and ok(c, "hep") and ok(c, "dfhoi")]
macro("SparseVsBestMin", fmt_x(min(best_sp)))
macro("SparseVsBestMax", fmt_x(max(best_sp)))
macro("SparseSlowerThanBest", str(sum(x < 1 for x in best_sp)))
macro("SparseVsBestN", str(len(best_sp)))
macro("DfhoiTimeouts", str(sum(R[(c[0], c[1], "dfhoi")]["status"] == "timeout" for c in configs)))
macro("AuraMaxTimeEmptyDense", f"{max(t(c, 'aura_hoi') for c in configs if KIND[c[0]] == 'dense' and ok(c, 'aura_hoi') and hois(c) == 0):.2f}")
macro("ZeroHoiConfigs", str(sum(ok(c, "aura_hoi") and hois(c) == 0 for c in configs)))

# spread of the repetitions: (max - min) / median over every repeated run
spread = [(float(r["time_max"]) - float(r["time_min"])) / float(r["time"])
          for r in main if r["status"] == "ok" and int(r["reps"]) > 1]
macro("RepSpreadMedian", f"{100 * np.median(spread):.1f}")
macro("RepSpreadMax", f"{100 * max(spread):.0f}")
macro("NumRepeatedRuns", str(len(spread)))
macro("NumSingleRuns", str(sum(r["status"] == "ok" and int(r["reps"]) == 1 for r in main)))

# reduction of candidate intersections relative to DFHOI, per dataset kind
for kind in ("dense", "sparse"):
    red = [100 * (1 - int(R[(c[0], c[1], "aura_hoi")]["nodes"]) / int(R[(c[0], c[1], "dfhoi")]["nodes"]))
           for c in both_df if KIND[c[0]] == kind]
    macro(f"NodeReductionMin{kind.capitalize()}", f"{min(red):.1f}")
    macro(f"NodeReductionMax{kind.capitalize()}", f"{max(red):.1f}")


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
    if r["status"] == "skipped":  # marked after the failure of the same algorithm at a larger threshold
        prev = [R[(c[0], cc[1], a)]["status"] for cc in configs if cc[0] == c[0] and cc[1] > c[1]]
        return "TL$^*$" if "timeout" in prev else "ML$^*$"
    return {"timeout": "TL", "memory": "ML", "failed": "F"}.get(r["status"], "F")


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
mem_ratio = [float(R[(c[0], c[1], "aura_hoi")]["mem"]) / float(R[(c[0], c[1], "dfhoi")]["mem"]) for c in both_df]
macro("MemRatioMin", f"{min(mem_ratio):.2f}")
macro("MemRatioMax", f"{max(mem_ratio):.2f}")
macro("MemLowerThanDfhoi", str(sum(x < 1 for x in mem_ratio)))
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

    for d in dsets:
        a = S[(d, 1.0, "aura_hoi")]
        b = S[(d, 0.2, "aura_hoi")]
        macro(f"ScalRatio{d.replace('T10I4D100K', 'Tten')}", f"{float(a['time']) / float(b['time']):.1f}")

# ── Oracle tests ───────────────────────────────────────────────────────────
orc = read("oracle")
if orc:
    macro("NumOracleCases", num(sum(int(r["cases"]) for r in orc)))
    macro("NumOracleTies", num(sum(int(r["cases_with_exact_tie"]) for r in orc)))
    macro("NumOracleDatabases", num(sum(int(r["databases"]) for r in orc)))
    macro("NumOraclePrograms", orc[0]["programs"])
    macro("OracleMismatches", str(sum(int(r["mismatches"]) for r in orc)))

# ── Independent brute-force check of accidents (brute_check.py) ────────────
bp = RES / "brute_accidents.txt"
if bp.exists():
    runs = {}
    for line in open(bp):
        if line.startswith("n="):
            cur = line.split()[1].split("=")[1]
        elif line.startswith("frequent itemsets="):
            runs[cur] = dict(kv.split("=") for kv in line.replace("frequent ", "").replace("max occ/n", "maxocc").split())
    if "0.2" in runs:
        macro("AccidentsMaxOcc", f"{float(runs['0.2']['maxocc']):.4f}")
        macro("AccidentsBruteFrequent", num(int(runs["0.2"]["itemsets"])))
        macro("AccidentsBruteHois", runs["0.2"]["HOIs"])

# ── Set-level agreement on the benchmarks ──────────────────────────────────
par = read("parity")
if par:
    pm = [r for r in par if r["experiment"] == "main"]
    macro("SetParityMain", str(len(pm)))
    macro("SetParityAblation", str(len(par) - len(pm)))
    macro("SetParityDisagree", str(sum(1 - int(r["agree"]) for r in par)))
    assert all(int(r["agree"]) for r in par), "set-level disagreement"

with open(ROOT / "results_macros.tex", "w") as f:
    f.write("% Generated by make_results.py from results/*.csv -- do not edit by hand.\n")
    for k, v in sorted(macros.items()):
        f.write(rf"\newcommand{{\{k}}}{{{v}}}" + "\n")
print(f"{len(macros)} macros written")
for k, v in sorted(macros.items()):
    print(f"  {k} = {v}")
