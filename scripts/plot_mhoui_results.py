#!/usr/bin/env python3
import csv
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALGO_LABELS = {
    "houi": "HOUI",
    "weak_mhoui": "Weak MHOUI",
    "strong_mhoui": "Strong MHOUI",
    "direct_strong_mhoui": "Direct Strong MHOUI",
    "twophase": "Two-Phase",
    "fhm": "FHM",
    "efim": "EFIM",
    "huiminer": "HUI-Miner",
    "upgrowth": "UP-Growth",
    "closed_fhuim_kinana": "Closed-FHUIM",
    "chui_miner": "CHUI-Miner",
}

ALGO_ORDER = [
    "houi",
    "weak_mhoui",
    "strong_mhoui",
    "direct_strong_mhoui",
    "closed_fhuim_kinana",
    "chui_miner",
    "efim",
    "fhm",
    "huiminer",
    "twophase",
    "upgrowth",
]

DATASET_LABELS = {
    "foodmart": "Foodmart",
    "liquor_11": "Liquor",
    "fruithut_utility": "Fruithut",
    "chainstore": "Chainstore",
}

COLORS = {
    "houi": "#4C78A8",
    "weak_mhoui": "#72B7B2",
    "strong_mhoui": "#F58518",
    "direct_strong_mhoui": "#E45756",
    "closed_fhuim_kinana": "#54A24B",
    "chui_miner": "#B279A2",
    "efim": "#9D755D",
    "fhm": "#BAB0AC",
    "huiminer": "#EECA3B",
    "twophase": "#FF9DA6",
    "upgrowth": "#8CD17D",
}


def fnum(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.upper() == "NA":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def inum(value):
    number = fnum(value)
    if number is None:
        return None
    return int(round(number))


def median(values):
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None
    return statistics.median(clean)


def mean(values):
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def pct(values, p):
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)


def load_rows(result_dir):
    path = os.path.join(result_dir, "run_summary.csv")
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row["runtime_sec_n"] = fnum(row.get("runtime_sec"))
            row["peak_ram_mb_n"] = fnum(row.get("peak_ram_mb"))
            row["output_count_n"] = fnum(row.get("output_count"))
            row["avg_len_n"] = fnum(row.get("avg_len"))
            row["avg_occ_n"] = fnum(row.get("avg_occupancy"))
            row["minsup_n"] = fnum(row.get("minsup_ratio"))
            row["minocc_n"] = fnum(row.get("minocc"))
            row["minutil_ratio_n"] = fnum(row.get("minutil_ratio"))
            row["visited_nodes_n"] = fnum(row.get("visited_nodes"))
            row["candidates_n"] = fnum(row.get("candidates"))
            for key in ["pruned_support", "pruned_twu", "pruned_uub", "pruned_oub1", "pruned_oub2", "pruned_dom"]:
                row[key + "_n"] = fnum(row.get(key))
            rows.append(row)
    disk_bytes = parse_report_disk_bytes(result_dir)
    for row in rows:
        rid = row["run_id"]
        if rid in disk_bytes:
            row["disk_bytes_n"] = float(disk_bytes[rid])
        else:
            count = row["output_count_n"]
            avg_len = row["avg_len_n"]
            if count is not None and avg_len is not None:
                row["disk_bytes_n"] = max(0.0, count * avg_len * 5.0 + count * 8.0)
            elif count is not None:
                row["disk_bytes_n"] = max(0.0, count * 13.0)
            else:
                row["disk_bytes_n"] = None
    return rows


def parse_report_disk_bytes(result_dir):
    disk = {}
    run_id = None
    rid_re = re.compile(r"^run_id=(.+)")
    disk_re = re.compile(r"\((\d+)\s+Bytes\)")
    for root, _, files in os.walk(result_dir):
        for name in files:
            if not name.endswith(".txt"):
                continue
            with open(os.path.join(root, name), errors="replace") as f:
                for line in f:
                    m = rid_re.match(line.strip())
                    if m:
                        run_id = m.group(1)
                        continue
                    if "Est. Disk" in line and run_id:
                        m = disk_re.search(line)
                        if m:
                            disk[run_id] = int(m.group(1))
    return disk


def label_algo(algo):
    return ALGO_LABELS.get(algo, algo)


def label_dataset(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def ordered_algos(rows):
    present = {r["algorithm"] for r in rows}
    ordered = [a for a in ALGO_ORDER if a in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def ordered_datasets(rows):
    present = sorted({r["dataset"] for r in rows})
    preferred = ["foodmart", "liquor_11", "fruithut_utility", "chainstore"]
    ordered = [d for d in preferred if d in present]
    ordered.extend(d for d in present if d not in ordered)
    return ordered


def ok_rows(rows):
    return [r for r in rows if r.get("status") == "OK"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def style_axes(ax):
    ax.grid(axis="y", color="#D0D7DE", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelrotation=32)


def savefig(fig, out_dir, name, captions, caption):
    ensure_dir(out_dir)
    png = os.path.join(out_dir, f"{name}.png")
    pdf = os.path.join(out_dir, f"{name}.pdf")
    fig.tight_layout()
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    captions.append((name, caption, png, pdf))


def bar_chart(rows, metric, title, ylabel, out_dir, name, captions, use_symlog=False):
    algos = ordered_algos(rows)
    values = [median([r[metric] for r in rows if r["algorithm"] == algo and r["status"] == "OK"]) for algo in algos]
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    xs = list(range(len(algos)))
    draw_values = [0 if v is None else v for v in values]
    ax.bar(xs, draw_values, color=[COLORS.get(a, "#6B7280") for a in algos], edgecolor="#2F3437", linewidth=0.4)
    ax.set_xticks(xs, [label_algo(a) for a in algos], ha="right")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    if use_symlog:
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
    style_axes(ax)
    for x, v in zip(xs, values):
        if v is not None and v > 0:
            ax.text(x, v, short_number(v), ha="center", va="bottom", fontsize=8)
    savefig(fig, out_dir, name, captions, title)


def grouped_dataset_bars(rows, metric, title, ylabel, out_dir, name, captions, algos=None, use_symlog=False):
    datasets = ordered_datasets(rows)
    if algos is None:
        algos = ordered_algos(rows)
    width = 0.76 / max(1, len(algos))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    xs = list(range(len(datasets)))
    for j, algo in enumerate(algos):
        vals = []
        for ds in datasets:
            vals.append(median([r[metric] for r in rows if r["dataset"] == ds and r["algorithm"] == algo and r["status"] == "OK"]))
        offsets = [x - 0.38 + width / 2 + j * width for x in xs]
        ax.bar(offsets, [0 if v is None else v for v in vals], width=width, label=label_algo(algo), color=COLORS.get(algo, "#6B7280"), edgecolor="#2F3437", linewidth=0.25)
    ax.set_xticks(xs, [label_dataset(d) for d in datasets])
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    if use_symlog:
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
    style_axes(ax)
    ax.legend(ncol=2, fontsize=8, frameon=False)
    savefig(fig, out_dir, name, captions, title)


def status_chart(rows, out_dir, captions):
    algos = ordered_algos(rows)
    statuses = ["OK", "TIMEOUT", "FAILED", "LIMITED"]
    colors = {"OK": "#54A24B", "TIMEOUT": "#E45756", "FAILED": "#B279A2", "LIMITED": "#F58518"}
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    bottoms = [0] * len(algos)
    counts_by_status = {s: [] for s in statuses}
    for algo in algos:
        c = Counter(r["status"] for r in rows if r["algorithm"] == algo)
        for s in statuses:
            counts_by_status[s].append(c.get(s, 0))
    xs = list(range(len(algos)))
    for s in statuses:
        vals = counts_by_status[s]
        if not any(vals):
            continue
        ax.bar(xs, vals, bottom=bottoms, label=s, color=colors[s], edgecolor="#2F3437", linewidth=0.25)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(xs, [label_algo(a) for a in algos], ha="right")
    ax.set_title("Run Status Distribution", loc="left", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of threshold configurations")
    style_axes(ax)
    ax.legend(frameon=False, ncol=4)
    savefig(fig, out_dir, "fig_status_distribution", captions, "Run status distribution by algorithm.")


def runtime_output_tradeoff(rows, out_dir, captions):
    ok = ok_rows(rows)
    algos = ordered_algos(ok)
    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    label_offsets = {
        "houi": (-50, -18),
        "weak_mhoui": (-50, -2),
        "strong_mhoui": (12, 12),
        "direct_strong_mhoui": (12, -8),
        "upgrowth": (12, -22),
        "closed_fhuim_kinana": (10, 0),
        "chui_miner": (10, 0),
        "efim": (8, 0),
        "fhm": (8, 0),
        "huiminer": (8, 0),
        "twophase": (8, 0),
    }
    for algo in algos:
        runtime = median([r["runtime_sec_n"] for r in ok if r["algorithm"] == algo])
        output = median([r["output_count_n"] for r in ok if r["algorithm"] == algo])
        ram = median([r["peak_ram_mb_n"] for r in ok if r["algorithm"] == algo]) or 1.0
        if runtime is None or output is None:
            continue
        draw_y = output + 1.0
        jitter = 1.0 + (algos.index(algo) - len(algos) / 2.0) * 0.015
        draw_x = runtime * max(0.85, jitter)
        size = 30 + min(300, math.sqrt(max(ram, 0.0)) * 25)
        ax.scatter(draw_x, draw_y, s=size, color=COLORS.get(algo, "#6B7280"), alpha=0.85, edgecolor="#2F3437", linewidth=0.5)
        offset = label_offsets.get(algo, (8, 0))
        ha = "right" if offset[0] < 0 else "left"
        ax.annotate(
            label_algo(algo),
            xy=(draw_x, draw_y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            va="center",
            ha=ha,
            arrowprops={"arrowstyle": "-", "color": "#6B7280", "lw": 0.45, "shrinkA": 3, "shrinkB": 3} if algo in {"houi", "weak_mhoui", "strong_mhoui", "direct_strong_mhoui", "upgrowth"} else None,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.margins(x=0.15, y=0.18)
    ax.set_xlabel("Median runtime on completed runs (seconds, log scale)")
    ax.set_ylabel("Median output itemsets + 1 (log scale)")
    ax.set_title("Runtime / Output Trade-off", loc="left", fontsize=14, fontweight="bold")
    style_axes(ax)
    savefig(fig, out_dir, "fig_runtime_output_tradeoff", captions, "Runtime-output trade-off; marker size reflects median RAM.")


def mhoui_ablation(rows, out_dir, captions):
    algos = ["houi", "weak_mhoui", "strong_mhoui", "direct_strong_mhoui"]
    grouped_dataset_bars(rows, "output_count_n", "MHOUI Family Output Ablation", "Median output itemsets", out_dir, "fig_mhoui_family_output_ablation", captions, algos=algos, use_symlog=True)
    grouped_dataset_bars(rows, "runtime_sec_n", "MHOUI Family Runtime Ablation", "Median runtime (seconds)", out_dir, "fig_mhoui_family_runtime_ablation", captions, algos=algos, use_symlog=True)


def compression_chart(rows, out_dir, captions):
    datasets = ordered_datasets(rows)
    vals = []
    labels = []
    for ds in datasets:
        ratios = []
        keys = defaultdict(dict)
        for r in rows:
            if r["dataset"] != ds or r["status"] != "OK":
                continue
            if r["algorithm"] not in ("houi", "strong_mhoui"):
                continue
            key = (r["minsup_ratio"], r["minocc"], r["minutil_ratio"])
            keys[key][r["algorithm"]] = r["output_count_n"]
        for pair in keys.values():
            h = pair.get("houi")
            s = pair.get("strong_mhoui")
            if h is not None and s is not None and s > 0:
                ratios.append(h / s)
            elif h is not None and s == 0 and h == 0:
                ratios.append(1.0)
        vals.append(median(ratios))
        labels.append(label_dataset(ds))
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.bar(range(len(labels)), [0 if v is None else v for v in vals], color="#F58518", edgecolor="#2F3437", linewidth=0.4)
    ax.axhline(1.0, color="#6B7280", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Median HOUI / Strong MHOUI output ratio")
    ax.set_title("Compression Achieved by Strong MHOUI", loc="left", fontsize=14, fontweight="bold")
    style_axes(ax)
    for x, v in enumerate(vals):
        if v is not None:
            ax.text(x, v, f"{v:.2f}x", ha="center", va="bottom", fontsize=9)
    savefig(fig, out_dir, "fig_strong_mhoui_compression", captions, "Median compression from HOUI to Strong MHOUI.")


def pruning_breakdown(rows, out_dir, captions):
    mrows = [r for r in rows if r["algorithm"] in ("houi", "weak_mhoui", "strong_mhoui", "direct_strong_mhoui") and r["status"] == "OK"]
    algos = ["houi", "weak_mhoui", "strong_mhoui", "direct_strong_mhoui"]
    prune_keys = [
        ("pruned_support_n", "Support", "#4C78A8"),
        ("pruned_twu_n", "TWU", "#F58518"),
        ("pruned_uub_n", "UUB", "#54A24B"),
        ("pruned_oub1_n", "OUB1", "#B279A2"),
        ("pruned_oub2_n", "OUB2", "#E45756"),
        ("pruned_dom_n", "Dominance", "#72B7B2"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = range(len(algos))
    bottoms = [0.0] * len(algos)
    for key, label, color in prune_keys:
        vals = [median([r[key] for r in mrows if r["algorithm"] == algo]) or 0.0 for algo in algos]
        ax.bar(xs, vals, bottom=bottoms, label=label, color=color, edgecolor="#2F3437", linewidth=0.25)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(list(xs), [label_algo(a) for a in algos], ha="right")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Median pruned nodes (symlog)")
    ax.set_title("MHOUI Pruning Breakdown", loc="left", fontsize=14, fontweight="bold")
    style_axes(ax)
    ax.legend(frameon=False, ncol=3)
    savefig(fig, out_dir, "fig_mhoui_pruning_breakdown", captions, "Median pruning contribution by safe bound.")


def timeout_rate_chart(rows, out_dir, captions):
    algos = ordered_algos(rows)
    vals = []
    for algo in algos:
        subset = [r for r in rows if r["algorithm"] == algo]
        vals.append(100.0 * sum(1 for r in subset if r["status"] == "TIMEOUT") / len(subset) if subset else 0.0)
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    ax.bar(range(len(algos)), vals, color=[COLORS.get(a, "#6B7280") for a in algos], edgecolor="#2F3437", linewidth=0.4)
    ax.set_xticks(range(len(algos)), [label_algo(a) for a in algos], ha="right")
    ax.set_ylabel("Timeout rate (%)")
    ax.set_ylim(0, max(100, max(vals) * 1.1 if vals else 100))
    ax.set_title("Robustness Under Time Budget", loc="left", fontsize=14, fontweight="bold")
    style_axes(ax)
    savefig(fig, out_dir, "fig_timeout_rate", captions, "Timeout rate under the configured benchmark budget.")


def nonempty_rate_chart(rows, out_dir, captions):
    algos = ordered_algos(rows)
    vals = []
    for algo in algos:
        subset = [r for r in rows if r["algorithm"] == algo and r["status"] == "OK"]
        vals.append(100.0 * sum(1 for r in subset if (r["output_count_n"] or 0.0) > 0) / len(subset) if subset else 0.0)
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    ax.bar(range(len(algos)), vals, color=[COLORS.get(a, "#6B7280") for a in algos], edgecolor="#2F3437", linewidth=0.4)
    ax.set_xticks(range(len(algos)), [label_algo(a) for a in algos], ha="right")
    ax.set_ylabel("Completed runs with non-empty output (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Non-empty Output Rate", loc="left", fontsize=14, fontweight="bold")
    style_axes(ax)
    savefig(fig, out_dir, "fig_nonempty_output_rate", captions, "Share of completed configurations that returned at least one pattern.")


def percentile_bar_chart(rows, metric, percentile, title, ylabel, out_dir, name, captions, use_symlog=False):
    algos = ordered_algos(rows)
    values = [pct([r[metric] for r in rows if r["algorithm"] == algo and r["status"] == "OK"], percentile) for algo in algos]
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    xs = list(range(len(algos)))
    ax.bar(xs, [0 if v is None else v for v in values], color=[COLORS.get(a, "#6B7280") for a in algos], edgecolor="#2F3437", linewidth=0.4)
    ax.set_xticks(xs, [label_algo(a) for a in algos], ha="right")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    if use_symlog:
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
    style_axes(ax)
    for x, v in zip(xs, values):
        if v is not None and v > 0:
            ax.text(x, v, short_number(v), ha="center", va="bottom", fontsize=8)
    savefig(fig, out_dir, name, captions, title)


def threshold_heatmaps(rows, out_dir, captions):
    for ds in ordered_datasets(rows):
        subset = [r for r in rows if r["dataset"] == ds and r["algorithm"] == "strong_mhoui" and r["status"] == "OK"]
        if not subset:
            continue
        for metric, title, fname, fmt in [
            ("output_count_n", f"{label_dataset(ds)} Strong MHOUI Output Sensitivity", f"fig_heatmap_output_strong_mhoui_{ds}", "{:.0f}"),
            ("runtime_sec_n", f"{label_dataset(ds)} Strong MHOUI Runtime Sensitivity", f"fig_heatmap_runtime_strong_mhoui_{ds}", "{:.3f}"),
        ]:
            occs = sorted({r["minocc_n"] for r in subset if r["minocc_n"] is not None})
            utils = sorted({r["minutil_ratio_n"] for r in subset if r["minutil_ratio_n"] is not None}, reverse=True)
            cell_values = {}
            for occ in occs:
                for ur in utils:
                    vals = [r[metric] for r in subset if r["minocc_n"] == occ and r["minutil_ratio_n"] == ur]
                    cell_values[(occ, ur)] = median(vals)
            matrix = [[cell_values.get((occ, ur), 0.0) or 0.0 for occ in occs] for ur in utils]
            fig, ax = plt.subplots(figsize=(7.4, 5.4))
            im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
            ax.set_xticks(range(len(occs)), [f"{x:g}" for x in occs])
            ax.set_yticks(range(len(utils)), [f"{x:g}" for x in utils])
            ax.set_xlabel("minocc")
            ax.set_ylabel("minutil ratio")
            ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
            for y, row in enumerate(matrix):
                for x, value in enumerate(row):
                    ax.text(x, y, fmt.format(value), ha="center", va="center", fontsize=8, color="#111827")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            savefig(fig, out_dir, fname, captions, title)


def pareto_proxy(rows, out_dir, captions):
    mrows = [r for r in rows if r["algorithm"] in ("houi", "strong_mhoui") and r["status"] == "OK" and r["output_count_n"] is not None]
    for ds in ordered_datasets(mrows):
        subset = [r for r in mrows if r["dataset"] == ds]
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(7.8, 5.8))
        for algo in ("houi", "strong_mhoui"):
            points = [r for r in subset if r["algorithm"] == algo]
            ax.scatter(
                [r["avg_occ_n"] or 0.0 for r in points],
                [r["output_count_n"] or 0.0 for r in points],
                s=42,
                alpha=0.78,
                label=label_algo(algo),
                color=COLORS.get(algo),
                edgecolor="#2F3437",
                linewidth=0.4,
            )
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Average occupancy of returned patterns")
        ax.set_ylabel("Output itemsets (symlog)")
        ax.set_title(f"{label_dataset(ds)} Occupancy / Output Profile", loc="left", fontsize=13, fontweight="bold")
        style_axes(ax)
        ax.legend(frameon=False)
        savefig(fig, out_dir, f"fig_occupancy_output_profile_{ds}", captions, f"{label_dataset(ds)} occupancy-output profile for HOUI vs Strong MHOUI.")


def per_dataset_summary(rows, out_dir, captions):
    for ds in ordered_datasets(rows):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        safe_ds = ds
        grouped_dataset_bars(ds_rows, "runtime_sec_n", f"{label_dataset(ds)} Runtime by Algorithm", "Median runtime (seconds)", out_dir, f"fig_{safe_ds}_runtime_by_algorithm", captions, use_symlog=True)
        grouped_dataset_bars(ds_rows, "output_count_n", f"{label_dataset(ds)} Output Count by Algorithm", "Median output itemsets", out_dir, f"fig_{safe_ds}_output_by_algorithm", captions, use_symlog=True)
        grouped_dataset_bars(ds_rows, "peak_ram_mb_n", f"{label_dataset(ds)} Peak RAM by Algorithm", "Median peak RAM (MB)", out_dir, f"fig_{safe_ds}_ram_by_algorithm", captions, use_symlog=True)


def short_number(v):
    if v is None:
        return ""
    av = abs(v)
    if av >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if av >= 1_000:
        return f"{v/1_000:.1f}K"
    if av >= 10:
        return f"{v:.0f}"
    if av >= 1:
        return f"{v:.2g}"
    return f"{v:.2g}"


def write_index(out_dir, captions):
    path = os.path.join(out_dir, "README.md")
    with open(path, "w") as f:
        f.write("# MHOUI Visualization Pack\n\n")
        f.write("Generated from `results/mhoui_compare_full/run_summary.csv`. PNG and PDF versions are emitted for each figure.\n\n")
        for name, caption, png, pdf in captions:
            f.write(f"## {name}\n\n")
            f.write(f"{caption}\n\n")
            f.write(f"- PNG: `{os.path.basename(png)}`\n")
            f.write(f"- PDF: `{os.path.basename(pdf)}`\n\n")


def write_plot_data(rows, out_dir):
    path = os.path.join(out_dir, "plot_dataset_summary.csv")
    algos = ordered_algos(rows)
    datasets = ordered_datasets(rows)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "algorithm", "ok_runs", "timeouts", "median_runtime_sec", "median_peak_ram_mb", "median_output_count", "median_est_disk_bytes"])
        for ds in datasets:
            for algo in algos:
                subset = [r for r in rows if r["dataset"] == ds and r["algorithm"] == algo]
                ok = [r for r in subset if r["status"] == "OK"]
                w.writerow([
                    ds,
                    algo,
                    len(ok),
                    sum(1 for r in subset if r["status"] == "TIMEOUT"),
                    median([r["runtime_sec_n"] for r in ok]),
                    median([r["peak_ram_mb_n"] for r in ok]),
                    median([r["output_count_n"] for r in ok]),
                    median([r["disk_bytes_n"] for r in ok]),
                ])


def main():
    result_dir = sys.argv[1] if len(sys.argv) > 1 else "results/mhoui_compare_full"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(result_dir, "charts")
    ensure_dir(out_dir)
    rows = load_rows(result_dir)
    captions = []

    bar_chart(rows, "runtime_sec_n", "Median Runtime by Algorithm", "Median runtime on completed runs (seconds)", out_dir, "fig_median_runtime_by_algorithm", captions, use_symlog=True)
    bar_chart(rows, "output_count_n", "Median Number of Itemsets Found", "Median output itemsets", out_dir, "fig_median_itemsets_by_algorithm", captions, use_symlog=True)
    bar_chart(rows, "peak_ram_mb_n", "Median Peak RAM by Algorithm", "Median peak RAM (MB)", out_dir, "fig_median_ram_by_algorithm", captions, use_symlog=True)
    bar_chart(rows, "disk_bytes_n", "Median Output Footprint by Algorithm", "Median output footprint (bytes; recorded or estimated)", out_dir, "fig_median_disk_usage_by_algorithm", captions, use_symlog=True)

    grouped_dataset_bars(rows, "runtime_sec_n", "Dataset-Level Runtime Comparison", "Median runtime (seconds)", out_dir, "fig_dataset_runtime_comparison", captions, use_symlog=True)
    grouped_dataset_bars(rows, "output_count_n", "Dataset-Level Itemset Count Comparison", "Median output itemsets", out_dir, "fig_dataset_itemset_count_comparison", captions, use_symlog=True)
    grouped_dataset_bars(rows, "peak_ram_mb_n", "Dataset-Level RAM Comparison", "Median peak RAM (MB)", out_dir, "fig_dataset_ram_comparison", captions, use_symlog=True)
    grouped_dataset_bars(rows, "disk_bytes_n", "Dataset-Level Output Footprint Comparison", "Median output footprint (bytes)", out_dir, "fig_dataset_disk_usage_comparison", captions, use_symlog=True)

    status_chart(rows, out_dir, captions)
    timeout_rate_chart(rows, out_dir, captions)
    nonempty_rate_chart(rows, out_dir, captions)
    percentile_bar_chart(rows, "runtime_sec_n", 0.9, "90th-Percentile Runtime by Algorithm", "90th-percentile runtime on completed runs (seconds)", out_dir, "fig_p90_runtime_by_algorithm", captions, use_symlog=True)
    percentile_bar_chart(rows, "output_count_n", 0.9, "90th-Percentile Itemsets Found", "90th-percentile output itemsets", out_dir, "fig_p90_itemsets_by_algorithm", captions, use_symlog=True)
    runtime_output_tradeoff(rows, out_dir, captions)
    mhoui_ablation(rows, out_dir, captions)
    compression_chart(rows, out_dir, captions)
    pruning_breakdown(rows, out_dir, captions)
    threshold_heatmaps(rows, out_dir, captions)
    pareto_proxy(rows, out_dir, captions)
    per_dataset_summary(rows, out_dir, captions)

    write_plot_data(rows, out_dir)
    write_index(out_dir, captions)
    print(f"Generated {len(captions)} figures in {out_dir}")


if __name__ == "__main__":
    main()
