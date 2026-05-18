#!/usr/bin/env python3
import csv
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {
    "vifp_plaintext": "VIFP Plain",
    "vifp_smpc": "VIFP-SMPC",
    "vifp_fhe": "VIFP-FHE",
    "fpgrowth": "FP-Growth",
    "eclat": "Eclat",
    "apriori": "Apriori",
    "fpmax": "FPmax",
}

ORDER = ["vifp_plaintext", "vifp_smpc", "vifp_fhe", "fpgrowth", "eclat", "apriori", "fpmax"]

COLORS = {
    "vifp_plaintext": "#4C78A8",
    "vifp_smpc": "#72B7B2",
    "vifp_fhe": "#F58518",
    "fpgrowth": "#54A24B",
    "eclat": "#B279A2",
    "apriori": "#E45756",
    "fpmax": "#9D755D",
}


def num(v):
    if v is None:
        return None
    v = str(v).strip()
    if not v or v.upper() == "NA":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def median(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return statistics.median(xs) if xs else None


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None and not math.isnan(x))
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def load_rows(result_dir):
    rows = []
    with open(os.path.join(result_dir, "run_summary.csv"), newline="") as f:
        for row in csv.DictReader(f):
            for k in [
                "minsup_ratio",
                "minsup_count",
                "runtime_sec",
                "peak_ram_mb",
                "output_count",
                "occurrence_records",
                "hpa_records",
                "projected_states",
                "cpb_records",
                "predecessor_fetches",
                "histogram_updates",
                "secure_comparisons",
                "stable_partitions",
                "oblivious_sorts",
                "estimated_comm_bytes",
                "estimated_ciphertext_bytes",
                "estimated_bootstraps",
                "result_disk_est_bytes",
            ]:
                row[k + "_n"] = num(row.get(k))
            rows.append(row)
    return rows


def algos(rows):
    present = {r["algorithm"] for r in rows}
    return [a for a in ORDER if a in present] + sorted(present - set(ORDER))


def datasets(rows):
    return sorted({r["dataset"] for r in rows})


def ok(rows):
    return [r for r in rows if r.get("status") == "OK"]


def label(a):
    return LABELS.get(a, a)


def style(ax):
    ax.grid(axis="y", color="#D8DEE5", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", rotation=30)


def save(fig, out_dir, name, captions, caption):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name + ".png"), dpi=220)
    fig.savefig(os.path.join(out_dir, name + ".pdf"))
    plt.close(fig)
    captions.append((name, caption))


def bar(rows, field, title, ylabel, out_dir, name, captions, symlog=False, percentile=None):
    rows = ok(rows)
    xs = algos(rows)
    vals = []
    for a in xs:
        data = [r[field] for r in rows if r["algorithm"] == a]
        vals.append(pct(data, percentile) if percentile is not None else median(data))
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    ax.bar(range(len(xs)), [v or 0 for v in vals], color=[COLORS.get(a, "#6B7280") for a in xs], edgecolor="#30343B", linewidth=0.35)
    ax.set_xticks(range(len(xs)), [label(a) for a in xs], ha="right")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    if symlog:
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
    style(ax)
    save(fig, out_dir, name, captions, title)


def grouped(rows, field, title, ylabel, out_dir, name, captions, only=None, symlog=False):
    rows = ok(rows)
    ds = datasets(rows)
    xs = only or algos(rows)
    width = 0.78 / max(1, len(xs))
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    base = list(range(len(ds)))
    for i, a in enumerate(xs):
        vals = [median([r[field] for r in rows if r["dataset"] == d and r["algorithm"] == a]) or 0 for d in ds]
        pos = [b - 0.39 + width / 2 + i * width for b in base]
        ax.bar(pos, vals, width=width, label=label(a), color=COLORS.get(a, "#6B7280"), edgecolor="#30343B", linewidth=0.25)
    ax.set_xticks(base, ds, ha="right")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    if symlog:
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
    style(ax)
    ax.legend(frameon=False, ncol=3)
    save(fig, out_dir, name, captions, title)


def status(rows, out_dir, captions):
    xs = algos(rows)
    stats = ["OK", "LIMITED", "TIMEOUT", "FAILED"]
    colors = {"OK": "#54A24B", "LIMITED": "#F58518", "TIMEOUT": "#E45756", "FAILED": "#B279A2"}
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    bottoms = [0] * len(xs)
    for st in stats:
        vals = [sum(1 for r in rows if r["algorithm"] == a and r["status"] == st) for a in xs]
        if any(vals):
            ax.bar(range(len(xs)), vals, bottom=bottoms, color=colors[st], label=st, edgecolor="#30343B", linewidth=0.25)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(range(len(xs)), [label(a) for a in xs], ha="right")
    ax.set_title("Run Status Distribution", loc="left", fontsize=14, fontweight="bold")
    ax.set_ylabel("Runs")
    style(ax)
    ax.legend(frameon=False, ncol=4)
    save(fig, out_dir, "fig_status_distribution", captions, "Run status distribution.")


def threshold_lines(rows, out_dir, captions):
    for d in datasets(rows):
        fig, ax = plt.subplots(figsize=(9.4, 5.2))
        for a in [x for x in algos(rows) if x.startswith("vifp_") or x in {"fpgrowth", "eclat"}]:
            pts = [(r["minsup_ratio_n"], r["runtime_sec_n"]) for r in ok(rows) if r["dataset"] == d and r["algorithm"] == a]
            pts = sorted((x, y) for x, y in pts if x is not None and y is not None)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=label(a), color=COLORS.get(a))
        ax.invert_xaxis()
        ax.set_title(f"{d}: Runtime vs Minimum Support", loc="left", fontsize=14, fontweight="bold")
        ax.set_xlabel("Minimum support ratio")
        ax.set_ylabel("Runtime (seconds)")
        ax.set_yscale("symlog", linthresh=0.001)
        style(ax)
        ax.legend(frameon=False, ncol=2)
        save(fig, out_dir, f"fig_runtime_threshold_{d}", captions, f"{d} runtime versus support threshold.")


def crypto_proxy(rows, out_dir, captions):
    vifp = [r for r in ok(rows) if r["algorithm"].startswith("vifp_")]
    grouped(vifp, "secure_comparisons_n", "VIFP Secure Comparison Proxy", "Median threshold tests", out_dir, "fig_secure_comparisons", captions, only=["vifp_plaintext", "vifp_smpc", "vifp_fhe"], symlog=True)
    grouped(vifp, "estimated_comm_bytes_n", "VIFP Communication Proxy", "Median estimated bytes", out_dir, "fig_estimated_communication", captions, only=["vifp_plaintext", "vifp_smpc"], symlog=True)
    grouped(vifp, "estimated_ciphertext_bytes_n", "VIFP FHE Ciphertext Footprint Proxy", "Median estimated bytes", out_dir, "fig_estimated_ciphertext", captions, only=["vifp_fhe"], symlog=True)
    grouped(vifp, "estimated_bootstraps_n", "VIFP FHE Bootstrap Proxy", "Median bootstraps", out_dir, "fig_estimated_bootstraps", captions, only=["vifp_fhe"], symlog=True)


def structure_charts(rows, out_dir, captions):
    vifp = [r for r in ok(rows) if r["algorithm"].startswith("vifp_")]
    grouped(vifp, "occurrence_records_n", "Occurrence Tape Size", "Median OTT records", out_dir, "fig_occurrence_records", captions, only=["vifp_plaintext"], symlog=True)
    grouped(vifp, "projected_states_n", "Projected Interval States", "Median states", out_dir, "fig_projected_states", captions, only=["vifp_plaintext"], symlog=True)
    grouped(vifp, "cpb_records_n", "Conditional Pattern Base Work", "Median CPB records scanned", out_dir, "fig_cpb_records", captions, only=["vifp_plaintext"], symlog=True)


def write_index(out_dir, captions):
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write("# VIFP Visualization Pack\n\n")
        f.write("Generated from `run_summary.csv`. PNG and PDF versions are emitted for each figure.\n\n")
        for name, caption in captions:
            f.write(f"## {name}\n\n{caption}\n\n- PNG: `{name}.png`\n- PDF: `{name}.pdf`\n\n")


def write_summary(rows, out_dir):
    with open(os.path.join(out_dir, "plot_dataset_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "algorithm", "ok_runs", "timeouts", "limited", "median_runtime_sec", "median_ram_mb", "median_output_count"])
        for d in datasets(rows):
            for a in algos(rows):
                subset = [r for r in rows if r["dataset"] == d and r["algorithm"] == a]
                oks = [r for r in subset if r["status"] == "OK"]
                w.writerow([
                    d,
                    a,
                    len(oks),
                    sum(1 for r in subset if r["status"] == "TIMEOUT"),
                    sum(1 for r in subset if r["status"] == "LIMITED"),
                    median([r["runtime_sec_n"] for r in oks]),
                    median([r["peak_ram_mb_n"] for r in oks]),
                    median([r["output_count_n"] for r in oks]),
                ])


def main():
    result_dir = sys.argv[1] if len(sys.argv) > 1 else "results/vifp_compare_full"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(result_dir, "charts")
    os.makedirs(out_dir, exist_ok=True)
    rows = load_rows(result_dir)
    captions = []
    bar(rows, "runtime_sec_n", "Median Runtime by Algorithm", "Median runtime (seconds)", out_dir, "fig_median_runtime_by_algorithm", captions, symlog=True)
    bar(rows, "output_count_n", "Median Frequent Itemsets by Algorithm", "Median output itemsets", out_dir, "fig_median_itemsets_by_algorithm", captions, symlog=True)
    bar(rows, "peak_ram_mb_n", "Median Peak RAM by Algorithm", "Median peak RAM (MB)", out_dir, "fig_median_ram_by_algorithm", captions, symlog=True)
    bar(rows, "runtime_sec_n", "90th-Percentile Runtime by Algorithm", "P90 runtime (seconds)", out_dir, "fig_p90_runtime_by_algorithm", captions, symlog=True, percentile=0.9)
    grouped(rows, "runtime_sec_n", "Dataset-Level Runtime Comparison", "Median runtime (seconds)", out_dir, "fig_dataset_runtime_comparison", captions, symlog=True)
    grouped(rows, "output_count_n", "Dataset-Level Frequent Itemset Count", "Median output itemsets", out_dir, "fig_dataset_itemset_count_comparison", captions, symlog=True)
    grouped(rows, "peak_ram_mb_n", "Dataset-Level RAM Comparison", "Median peak RAM (MB)", out_dir, "fig_dataset_ram_comparison", captions, symlog=True)
    status(rows, out_dir, captions)
    threshold_lines(rows, out_dir, captions)
    structure_charts(rows, out_dir, captions)
    crypto_proxy(rows, out_dir, captions)
    write_summary(rows, out_dir)
    write_index(out_dir, captions)
    print(f"Generated {len(captions)} figures in {out_dir}")


if __name__ == "__main__":
    main()
